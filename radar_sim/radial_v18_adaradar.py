"""AdaRadar static compression front end, as in the official code, for the V18 baseline.

AdaRadar (Park, Chun & Seok, CVPR 2026, arXiv 2603.17979; code
github.com/jp4327/adaradar, utils/dct.py ``dct_based_compression``) compresses
the tensor FFTRadNet takes as input: the post-FFT range-Doppler spectrum as 32
real channels (16 Rx x real/imaginary), each normalized by the dataset's
per-channel mean and std (FFTRadNet dataset loader), 512 range x 256 Doppler.
Per channel:
  1. split into BL x BL tiles (BL = 64) and apply the orthonormal 2-D DCT-II;
  2. pruning: keep_ratio = 1 / comp_ratio, k = floor(keep_ratio * BL^2); the
     k-th largest |coefficient| of each tile is the threshold and smaller
     coefficients are zeroed (ties at the threshold are kept);
  3. symmetric per-tile quantization: levels = 2^(qbit-1) - 1,
     step = max|coefficient| / levels, round and clip to +-levels;
  4. decoder: dequantize, inverse DCT, merge tiles (and undo the normalization).
The online feedback controller that adapts comp_ratio frame by frame is in
the V18 runner (AdaptiveRate). Not reproduced: the retraining of FFTRadNet on
decompressed tensors; the official FFTRadNet stays frozen, the paper's
"without fine-tuning" setting (Suppl. S4).

Rate. The official code reports compression ratio = elements / nonzero
coefficients after quantization, i.e. qbit bits per nonzero value and nothing
else (``total_nonzero_only``). A decoder also needs WHERE the nonzero values
are: per tile the count n (13 bits for 0..4096) and one subset of n out of
BL^2 positions, log2 C(BL^2, n) bits with enumerative coding (Cover 1973),
plus the per-tile float32 step. ``total`` is this complete rate.
"""
from __future__ import annotations

import math

import torch

BLOCK = 64          # BL, the official default and the paper's RADIal setting
SCALE_BITS = 32     # one float32 step per tile
COUNT_BITS = 13     # nonzero count per tile, 0..4096
HEADER_BITS = 64


def dct_matrix(size, device, dtype=torch.float64):
    """Orthonormal DCT-II matrix D (scipy norm='ortho'): coefficients = D @ x."""
    n = torch.arange(size, device=device, dtype=dtype)
    matrix = torch.cos(math.pi*(2*n[None]+1)*n[:, None]/(2*size))*math.sqrt(2/size)
    matrix[0] /= math.sqrt(2)
    return matrix


def kept_per_block(prune_ratio, block=BLOCK):
    k = math.floor(block*block/prune_ratio)
    if not 0 < k <= block*block:
        raise ValueError('comp_ratio leaves no coefficient or more than a tile.')
    return k


def subset_bits(total, kept):
    """ceil(log2 C(total, kept)) per element of ``kept`` (a tensor): enumerative code length of positions."""
    kept = torch.as_tensor(kept, dtype=torch.float64)
    nats = math.lgamma(total+1)-torch.lgamma(kept+1)-torch.lgamma(total-kept+1)
    return torch.ceil(nats/math.log(2)-1e-9).clamp_min(0)


def _blocks(channels, block):
    c, h, w = channels.shape
    return channels.reshape(c, h//block, block, w//block, block).permute(0, 1, 3, 2, 4)


def _unblocks(blocks, shape):
    c, h, w = shape
    return blocks.permute(0, 1, 3, 2, 4).reshape(c, h, w)


@torch.no_grad()
def compress(spectrum, prune_ratio, bits, mean=None, std=None, block=BLOCK):
    """Encode and decode a [range, Doppler, Rx] complex spectrum; returns (reconstruction, rate dict).
    ``mean``/``std`` (32 values) are FFTRadNet's input statistics; None codes the raw spectrum."""
    rng, doppler, rx = spectrum.shape
    if rng % block or doppler % block:
        raise ValueError('The spectrum does not tile into BL x BL blocks.')
    x = torch.cat((spectrum.real, spectrum.imag), -1).to(torch.float64)
    mean = torch.zeros(2*rx, dtype=torch.float64, device=x.device) if mean is None else torch.as_tensor(mean, dtype=torch.float64, device=x.device)
    std = torch.ones(2*rx, dtype=torch.float64, device=x.device) if std is None else torch.as_tensor(std, dtype=torch.float64, device=x.device)
    channels = ((x-mean)/std).permute(2, 0, 1)  # [2C, H, W], the network input
    d = dct_matrix(block, spectrum.device)
    blocks = _blocks(channels, block)
    z = (d@blocks@d.T).reshape(*blocks.shape[:3], block*block)
    k = kept_per_block(prune_ratio, block)
    threshold = z.abs().topk(k, dim=-1).values[..., -1:]
    pruned = torch.where(z.abs() >= threshold, z, torch.zeros_like(z))
    levels = 2**(bits-1)-1
    step = pruned.abs().amax(-1, keepdim=True)/levels
    step = torch.where(step > 0, step, torch.ones_like(step))
    restored = torch.round(pruned/step).clamp(-levels, levels)*step
    nonzero = (restored != 0).sum(-1)
    channels_hat = _unblocks(d.T@restored.reshape(blocks.shape)@d, channels.shape)
    x_hat = channels_hat.permute(1, 2, 0)*std+mean
    reconstruction = torch.complex(x_hat[..., :rx], x_hat[..., rx:])
    tiles = nonzero.numel()
    values = int(nonzero.sum())*bits
    index = int(subset_bits(block*block, nonzero).sum())+tiles*COUNT_BITS
    scales = tiles*SCALE_BITS
    return reconstruction, dict(
        total=values+index+scales+HEADER_BITS, total_nonzero_only=values, value_bits=values, index_bits=index,
        scale_bits=scales, kept_fraction=k/(block*block), nonzero_fraction=float(nonzero.sum())/channels.numel(),
        official_compression_ratio=channels.numel()/max(int(nonzero.sum()), 1),
        accounting='AdaRadar static (BL=64): nonzero values + their positions (count + enumerative subset) + float32 tile steps')
