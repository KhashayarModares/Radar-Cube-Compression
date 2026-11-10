"""Orthonormal block DCT codec for complex IQ; the radar receiver is unchanged.

Compression transforms each raw slow/fast-time block independently. The same
real DCT-II matrix acts on I and Q and on every antenna, with no antenna-axis
transform. Its transpose supplies the inverse. ``spectrum`` is retained only as
a compatibility name for the assembled block-DCT coefficient canvas: its
indices are NOT range/Doppler coordinates. Shared coefficient support and the
existing per-tile IQ quantizer/accounted syntax are retained.

The codec uses CUDA matrix products in production; no compression FFT or CPU
numerical fallback is used. File decoding and metadata orchestration remain
outside this module. Lossy support/quantization can still alter phase and target
preservation; invertibility of the unquantized transform is not such a guarantee.
"""
from __future__ import annotations

import math
import torch

from .arm_e_adaptive_gpu_physics import EPS, _align8

TRANSFORM_ID = 'block-dct2-ortho-separate-IQ-v1'


class BlockDCTCodecMixin:
    """Place before an existing radar-physics class in the MRO.

    The receiver, label mapper and physical CFAR metrics remain inherited.
    Required instance fields are ``device``, ``shape``, ``block_slow``,
    ``block_fast`` and the existing ``tensor`` helper. Matrices are cached lazily
    on that device. A default chunk of sixteen equal-size blocks bounds the
    temporary real-component tensor independently of total cube size.
    """
    dct_chunk_blocks = 16
    codec_transform = TRANSFORM_ID

    def _cosine_matrix(self, length):
        cache = getattr(self, '_dct_matrix_cache', None)
        if cache is None:
            cache = {}; self._dct_matrix_cache = cache
        key = (int(length), str(self.device))
        if key not in cache:
            position = torch.arange(length,device=self.device,dtype=torch.float64)
            matrix = torch.cos(math.pi/length*position[:,None]*(position[None,:]+.5))
            matrix *= math.sqrt(2./length)
            matrix[0] /= math.sqrt(2.)
            cache[key] = matrix
        return cache[key]

    def _real_dct2(self, value, *, inverse=False):
        slow, fast = self._cosine_matrix(value.shape[-2]), self._cosine_matrix(value.shape[-1])
        if inverse:
            return slow.T @ value @ fast
        return slow @ value @ fast.T

    def _block_slices(self):
        return [(slice(None),slice(m,min(m+self.block_slow,self.shape[1])),
                 slice(n,min(n+self.block_fast,self.shape[2])))
                for m in range(0,self.shape[1],self.block_slow)
                for n in range(0,self.shape[2],self.block_fast)]

    @torch.no_grad()
    def dct_blocks(self, cube, *, inverse=False):
        """Forward/inverse real-linear block transform, returned complex128.

        The transform is complex-linear because both components use identical
        real matrices. Full unquantized inverse/forward composition is identity
        up to roundoff and preserves inter-antenna phase/coherence.
        """
        if (not isinstance(cube,torch.Tensor) or cube.device!=self.device
                or tuple(cube.shape)!=self.shape or not torch.is_complex(cube)
                or not bool(torch.isfinite(cube).all())):
            raise ValueError('DCT requires a finite complex cube on the declared device with the declared shape.')
        chunk = self.dct_chunk_blocks
        if isinstance(chunk,bool) or not isinstance(chunk,int) or chunk<1:
            raise ValueError('dct_chunk_blocks must be a positive integer.')
        result = torch.empty(self.shape,device=self.device,dtype=torch.complex128)
        groups = {}
        for slc in self._block_slices():
            shape = (slc[1].stop-slc[1].start,slc[2].stop-slc[2].start)
            groups.setdefault(shape,[]).append(slc)
        for slices in groups.values():
            for first in range(0,len(slices),chunk):
                current = slices[first:first+chunk]
                blocks = torch.stack([cube[slc] for slc in current]).to(torch.complex128)
                # [block, antenna, I/Q, slow, fast]; no mixing of I/Q or antennas.
                components = torch.view_as_real(blocks).movedim(-1,-3)
                transformed = self._real_dct2(components,inverse=inverse)
                joined = torch.complex(transformed[:,:,0],transformed[:,:,1])
                for index,slc in enumerate(current):result[slc]=joined[index]
        return result

    @torch.no_grad()
    def prepare(self, cube):
        coefficients = self.dct_blocks(cube)
        energy = cube.to(torch.complex128).abs().square().sum()
        tiles=[]; slices=self._block_slices()
        for slc in slices:
            z=coefficients[slc]
            power=z.abs().square().sum(dim=0).reshape(-1)
            rank=torch.argsort(power,descending=True,stable=True)
            e=power.sum(); probability=power/e.clamp_min(EPS)
            log_probability=probability.clamp_min(torch.finfo(torch.float64).tiny).log()
            tile=dict(Z=z,rank_desc=rank,support_cells=power.numel(),tile_energy=e,
                block_mean_power=e/z.numel(),power=power,block_shape=tuple(z.shape[1:]),
                spectral_entropy_normalized=-(probability*log_probability).sum()/math.log(max(2,power.numel())),
                spectral_peak_to_mean_log=torch.log1p(power.max()/power.mean().clamp_min(EPS)))
            for fraction,name in ((.01,'top1'),(.05,'top5'),(.10,'top10')):
                k=max(1,math.ceil(fraction*len(power)))
                tile['spectral_'+name+'_fraction']=power[rank[:k]].sum()/e.clamp_min(EPS)
            tiles.append(tile)
        return dict(cube=cube,spectrum=coefficients,coefficients=coefficients,
            codec_transform=TRANSFORM_ID,tile_domain='raw-IQ blocks; local DCT coefficient indices',
            tiles=tiles,prepared_blocks=tiles,slices=slices,tile_slices=slices,
            cube_energy=energy,whole_cube_energy=energy,cube_shape=self.shape,
            cube_mean_power=energy/cube.numel(),cube_header_bits=self.tensor(768.))

    def _check_dct_source(self, source):
        if source.get('codec_transform')!=TRANSFORM_ID:
            raise ValueError('DCT inverse requires its declared block-DCT source; an FFT source is incompatible.')
        z=source['spectrum']
        if (z.device!=self.device or z.dtype!=torch.complex128 or tuple(z.shape)!=self.shape
                or not torch.is_complex(z) or not bool(torch.isfinite(z).all())):
            raise ValueError('DCT coefficients have an incompatible shape, device, dtype or value.')

    @torch.no_grad()
    def inverse_coefficients(self, source, coefficients=None):
        """Unquantized inverse diagnostic; not a rate-zero or lossless codec point."""
        self._check_dct_source(source)
        return self.dct_blocks(source['spectrum'] if coefficients is None else coefficients,inverse=True)

    @torch.no_grad()
    def quantize_coefficients(self, source, rho, q):
        """Prune/quantize once; expose exact block distortion before the inverse.

        This is the same quantization path as ``encode``. Block squared errors
        are available for TRAIN query placement without a receiver evaluation;
        they are not a claim of block-to-range/Doppler target attribution.
        """
        self._check_dct_source(source)
        rho,q=self.tensor(rho),self.tensor(q)
        if rho.shape!=(len(source['tiles']),) or q.shape!=rho.shape:
            raise ValueError('Supply exactly one rho and q per DCT tile.')
        if (not bool(torch.isfinite(rho).all() & torch.isfinite(q).all())
                or not bool(((rho>0)&(rho<=1)&(q>=1)&(q<=30)&(q==q.round())).all())):
            raise ValueError('Codec actions require 0<rho<=1 and integer 1<=q<=30.')
        out=torch.zeros_like(source['spectrum']);bit_rows=[];kept_rows=[];scales=[];errors=[]
        for i,(tile,slc) in enumerate(zip(source['tiles'],source['slices'])):
            z,cells=tile['Z'],tile['support_cells']
            kept=torch.ceil(rho[i]*cells).to(torch.int64).clamp(1,cells)
            ranked=torch.arange(cells,device=self.device)<kept
            mask=torch.zeros(cells,dtype=torch.bool,device=self.device)
            mask[tile['rank_desc']]=ranked;mask=mask.reshape(z.shape[1:])
            selected=z*mask
            peak=torch.maximum(selected.real.abs().max(),selected.imag.abs().max())
            binary=(selected.real.abs().sum()+selected.imag.abs().sum())/(2*self.shape[0]*kept)
            scale=torch.where(q[i]==1,binary,peak)
            levels=(2.**(q[i]-1)-1).clamp_min(1)
            denominator=torch.where(scale>0,scale,torch.ones_like(scale))
            real=torch.round(selected.real/denominator*levels)*scale/levels
            imag=torch.round(selected.imag/denominator*levels)*scale/levels
            real=torch.where(q[i]==1,torch.where(selected.real>=0,scale,-scale),real)
            imag=torch.where(q[i]==1,torch.where(selected.imag>=0,scale,-scale),imag)
            out[slc]=torch.complex(real,imag)*mask
            errors.append((z-out[slc]).abs().square().sum())
            width=max(1,(cells-1).bit_length())
            support=torch.minimum(_align8(self.tensor(cells,torch.int64)),_align8(kept*width))
            payload=_align8(2*q[i].to(torch.int64)*self.shape[0]*kept)
            bit_rows.append(torch.stack((payload,support,payload+support+128)))
            kept_rows.append(kept);scales.append(scale)
        bits=torch.stack(bit_rows);total=bits[:,2].sum()+768
        rate=total.to(torch.float64)/source['cube'].numel()
        errors=torch.stack(errors)
        return out,dict(bpcs=rate,realized_bpcs=rate,realized_bits=total,
            payload_bits=bits[:,0].sum(),support_bits=bits[:,1].sum(),
            header_bits=self.tensor(768+128*len(bits)),tile_bits=bits[:,2],
            kept=torch.stack(kept_rows),rho=rho,q=q,scale=torch.stack(scales),codec_transform=TRANSFORM_ID,
            tile_squared_error=errors,spectral_nmse=errors.sum()/source['cube_energy'].clamp_min(EPS))

    @torch.no_grad()
    def tile_squared_errors(self, source, rho, q):
        """Exact Parseval block distortion via the production quantizer."""
        return self.quantize_coefficients(source,rho,q)[1]['tile_squared_error']

    @torch.no_grad()
    def encode(self, source, rho, q):
        coefficients,meta=self.quantize_coefficients(source,rho,q)
        cube_hat=self.inverse_coefficients(source,coefficients).to(source['cube'].dtype)
        nmse=(source['cube'].to(torch.complex128)-cube_hat).abs().square().sum()/source['cube_energy'].clamp_min(EPS)
        meta.update(nmse=nmse,cube_nmse=nmse)
        return cube_hat,meta
