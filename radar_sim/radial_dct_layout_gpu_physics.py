"""A separately identified global-DCT layout for the physical ablation.

The orthonormal real DCT acts on each complete I/Q slow/fast plane; coefficient
allocation still uses the ordinary 32-by-64 tiles. This keeps the allocation
count comparable with global FFT and does not transform the antenna axis.
No legacy codec or receiver implementation is modified.
"""
from __future__ import annotations
import torch
from .radial_dct_gpu_physics import BlockDCTCodecMixin

GLOBAL_DCT_ID='global-dct2-ortho-separate-IQ-coefficient-tiles-v1'


class GlobalDCTCodecMixin(BlockDCTCodecMixin):
    """Place before the unchanged radar physics class in the MRO."""
    codec_transform=GLOBAL_DCT_ID

    @torch.no_grad()
    def dct_blocks(self,cube,*,inverse=False):
        if (not isinstance(cube,torch.Tensor) or cube.device!=self.device
                or tuple(cube.shape)!=self.shape or not torch.is_complex(cube)
                or not bool(torch.isfinite(cube).all())):
            raise ValueError('Global DCT requires an aligned finite device-resident complex cube.')
        # [antenna,I/Q,slow,fast]; the leading dimensions are independent batches.
        value=torch.stack((cube.real.double(),cube.imag.double()),dim=1)
        transformed=self._real_dct2(value,inverse=inverse)
        return torch.complex(transformed[:,0],transformed[:,1])

    @torch.no_grad()
    def prepare(self,cube):
        # Parent builds coefficient tiles from the globally transformed canvas.
        # Override all layout markers because its default is raw block DCT.
        source=super().prepare(cube)
        source.update(codec_transform=GLOBAL_DCT_ID,
            tile_domain='global-DCT coefficient tiles; no direct RD/RA coordinate mapping',
            transform_scope='complete slow/fast plane independently for each antenna and I/Q component',
            allocation_tile_shape=[self.block_slow,self.block_fast])
        return source

    def _check_dct_source(self,source):
        if source.get('codec_transform')!=GLOBAL_DCT_ID:
            raise ValueError('Global DCT requires its own source; raw-block DCT and FFT sources are incompatible.')
        z=source['spectrum']
        if (not isinstance(z,torch.Tensor) or z.device!=self.device
                or z.dtype!=torch.complex128 or tuple(z.shape)!=self.shape
                or not bool(torch.isfinite(z).all())):
            raise ValueError('Global DCT coefficient shape/device/dtype is incompatible.')

    @torch.no_grad()
    def quantize_coefficients(self,source,rho,q):
        coefficients,metadata=super().quantize_coefficients(source,rho,q)
        metadata['codec_transform']=GLOBAL_DCT_ID
        return coefficients,metadata
