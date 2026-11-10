"""V12.40 tile codec under a hard 4.6-bpcs cap: same (rho, q) actions, refined signal path.

  * q sets a noise-referenced step, step = sigma_t*2^(6-q), so q means the same SNR loss in every tile. The peak-scaled
    step of V12.39 rounded 92% of the integers to zero at q=4 on real cubes.
  * Subtractive dither makes the error uniform and independent of the signal.
  * Pruned cells are refilled with noise of the transmitted tile power. They hold the CFAR background: without the fill,
    pruning at (.25, 8) added about 2467 unmatched peaks per cube on real cubes, and with it none.
  * The rate is the adaptive arithmetic code length of the integers. Fixed-length accounting charged 4.07 bpcs for
    integers that cost 2.37 bpcs.

Receiver, CFAR thresholds and label gates are unchanged, so original and reconstruction stay comparable.
"""
from __future__ import annotations

import math

import torch

from . import radial_v13_detection_codec as v13
from .radial_adaptive_gpu_physics import RadialGPUPhysics, _distances

CAP_BPCS = 4.6
Q_UNITY = 6            # q at which the step equals the tile noise level
TILE_SIDE_BITS = 32    # noise level, rho and q of one tile
FILL_BITS = 16         # power of the pruned cells
CUBE_HEADER_BITS = 768
DOPPLER_TILES, RANGE_TILES = 8, 8
SPREAD = DOPPLER_TILES  # a range-angle event is shared by every Doppler tile of its range band


def beta_of(q):
    return 2.**(Q_UNITY-q)


def snr_loss_db(q):
    return 10*torch.log10(1+beta_of(q)**2/12)


class CapPhysics(RadialGPUPhysics):
    @torch.no_grad()
    def prepare(self, cube, seed=0):
        source = super().prepare(cube)
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        shape = source['spectrum'].shape
        source['dither'] = torch.rand((2, *shape), generator=generator, device=self.device, dtype=torch.float64)-.5
        source['fill'] = torch.complex(torch.randn(shape, generator=generator, device=self.device, dtype=torch.float64),
                                       torch.randn(shape, generator=generator, device=self.device, dtype=torch.float64))/math.sqrt(2)
        source['sigma'] = torch.stack([(t['Z'].abs().square().flatten().median()/(2*math.log(2))).sqrt() for t in source['tiles']])
        return source

    @torch.no_grad()
    def encode(self, source, rho, q):
        rho, q = self.tensor(rho), self.tensor(q)
        if rho.shape != (len(source['tiles']),) or q.shape != rho.shape:
            raise ValueError('Supply exactly one rho and q per tile.')
        if not bool(((rho > 0) & (rho <= 1) & (q >= 1) & (q <= 12) & (q == q.round())).all()):
            raise ValueError('Codec actions require 0<rho<=1 and integer 1<=q<=12.')
        out = torch.zeros_like(source['spectrum'])
        tile_bits, tile_errors = [], []
        for i, (tile, slc) in enumerate(zip(source['tiles'], source['slices'])):
            z, cells = tile['Z'], tile['support_cells']
            kept = int(torch.ceil(rho[i]*cells).clamp(1, cells))
            ranked = torch.arange(cells, device=self.device) < kept
            mask = torch.zeros(cells, dtype=torch.bool, device=self.device)
            mask[tile['rank_desc']] = ranked
            mask = mask.reshape(z.shape[1:])
            step = source['sigma'][i].clamp_min(1e-12)*2.**(Q_UNITY-q[i])
            dither = source['dither'][(slice(None), *slc)]*step
            integers = torch.round((torch.stack((z.real, z.imag))+dither)/step)
            decoded = (integers*step-dither)*mask
            rebuilt = torch.complex(decoded[0], decoded[1])
            bits = self.tensor(TILE_SIDE_BITS)
            if kept < cells:
                hidden = ~mask
                power = (z.abs().square()*hidden).sum()/(hidden.sum()*z.shape[0])
                rebuilt = rebuilt+source['fill'][slc]*power.sqrt()*hidden
                bits = bits+FILL_BITS+min(cells, kept*max(1, (cells-1).bit_length()))
            out[slc] = rebuilt
            classes = v13.magnitude_class(integers[:, mask.expand_as(z)].to(torch.int64).flatten())
            counts = torch.bincount(classes, minlength=v13.CLASSES).to(torch.float64)
            tile_bits.append(bits+v13.kt_bits(counts[None])[0]+classes.sum())
            tile_errors.append((z-rebuilt).abs().square().sum())
        cube_hat = torch.fft.ifft2(torch.fft.ifftshift(out, dim=(1, 2)), dim=(1, 2), norm='ortho').to(source['cube'].dtype)
        bits = torch.stack(tile_bits)
        total = bits.sum()+CUBE_HEADER_BITS
        errors = torch.stack(tile_errors)
        return cube_hat, dict(bpcs=total/source['cube'].numel(), realized_bits=total, tile_bits=bits, tile_error=errors,
                              nmse=errors.sum()/source['cube_energy'].clamp_min(1e-12), rho=rho, q=q)

    def legacy(self, source, rho, q):
        """The peak-scaled, unfilled, fixed-length codec of V12.39, kept as a baseline."""
        return RadialGPUPhysics.encode(self, source, rho, q)


def tile_of(rows, columns):
    """Native RD bins to the shifted codec tile that owns them."""
    doppler = ((columns+128) % 256)//32
    rng = ((rows+256) % 512)//64
    return doppler*RANGE_TILES+rng


@torch.no_grad()
def tile_terms(physics, original, method, metric, labels):
    """Attribute each decision to the tile that controls it: an RD cell to its tile, an RA event to its range band."""
    device, low = physics.device, physics.thresholds[0]
    n = DOPPLER_TILES*RANGE_TILES
    miss, loss, excess = (torch.zeros(n, dtype=torch.float64, device=device) for _ in range(3))
    bands = torch.arange(n, device=device) % RANGE_TILES

    def add(target, rows, columns, values, kind):
        if not len(rows):
            return
        if kind == 'RD':
            target.scatter_add_(0, tile_of(rows, columns), values)
        else:
            per_band = torch.zeros(RANGE_TILES, dtype=torch.float64, device=device)
            per_band.scatter_add_(0, ((rows+256) % 512)//64, values)
            target.add_(per_band[bands]/SPREAD)

    for kind in ('RD', 'RA'):
        bins = physics.label_bins(labels, original, kind)
        o, c = metric[kind+'_original_best_scores'], metric[kind+'_reconstructed_best_scores']
        supported = o > low
        add(miss, bins[:, 0], bins[:, 1], (supported & (c <= low)).to(torch.float64), kind)
        add(loss, bins[:, 0], bins[:, 1], (10*torch.log10(o.clamp_min(1e-3)/c.clamp_min(1e-3))).clamp(0, 20)*supported, kind)
        period = 256 if kind == 'RD' else None
        counts = []
        for detector in (original['detectors'][kind], method['detectors'][kind]):
            points = torch.nonzero(detector['base_peaks'] & (detector['score'] > low), as_tuple=False)
            if len(bins) and len(points):
                points = points[~(_distances(points, bins, period) <= 2).all(2).any(1)]
            size = n if kind == 'RD' else RANGE_TILES
            index = tile_of(points[:, 0], points[:, 1]) if kind == 'RD' else ((points[:, 0]+256) % 512)//64
            counts.append(torch.bincount(index, minlength=size).to(torch.float64))
        extra = (counts[1]-counts[0]).clamp_min(0)
        excess.add_(extra if kind == 'RD' else extra[bands]/SPREAD)
    return dict(miss=miss, score_loss_db=loss, excess=excess)
