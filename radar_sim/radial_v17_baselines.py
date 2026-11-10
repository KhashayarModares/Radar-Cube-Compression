"""Standard raw-ADC compression baselines for the V17 publication run.

Both operate on the raw ADC cube, i.e. on what a radar front end would transmit,
and both are fixed-rate, like the codecs they represent.

* BAQ, block adaptive quantization (Kwok & Johnson 1989, Magellan SAR; the
  standard raw-data compressor of spaceborne SAR). Each block of I or Q samples
  is normalized by its RMS and quantized with the n-bit Lloyd-Max quantizer for a
  unit Gaussian (Max 1960; Lloyd 1982). Rate: n bits per real sample plus one
  8-bit RMS code per block.
* Uniform bit-depth reduction: an n-bit uniform mid-tread quantizer (2^n - 1
  levels) scaled to the cube's peak |I|, |Q|, i.e. the ADC itself with n instead
  of 16 bits and ideal gain control (the most favourable plain variant; cf.
  Widrow & Kollar 2008). Rate: n bits per real sample plus a 32-bit scale.
"""
from __future__ import annotations

import math

import torch

BAQ_BLOCK = (16, 64)       # chirps x fast-time samples per block (1024 samples), per Rx and per I/Q
BAQ_SCALE_BITS = 8         # log2 RMS in 1/8 octave steps
HEADER_BITS = 64


def lloyd_max_gaussian(bits, iterations=500):
    """Thresholds (2^n - 1) and levels (2^n) of the MSE-optimal quantizer for N(0, 1)."""
    n = 2**bits
    levels = torch.linspace(-2.5, 2.5, n, dtype=torch.float64)
    phi = lambda x: torch.exp(-x.square()/2)/math.sqrt(2*math.pi)
    cdf = lambda x: .5*(1+torch.erf(x/math.sqrt(2)))
    for _ in range(iterations):
        thresholds = (levels[1:]+levels[:-1])/2
        edges = torch.cat((torch.tensor([-math.inf], dtype=torch.float64), thresholds,
                           torch.tensor([math.inf], dtype=torch.float64)))
        a, b = edges[:-1], edges[1:]
        density = torch.where(torch.isfinite(a), phi(a), torch.zeros_like(a))-torch.where(torch.isfinite(b), phi(b), torch.zeros_like(b))
        levels = density/(cdf(b)-cdf(a))      # conditional mean of each cell (Lloyd condition)
    return (levels[1:]+levels[:-1])/2, levels


def _blocks(real, shape):
    rx, chirps, samples = shape
    bc, bs = BAQ_BLOCK
    return real.reshape(rx, chirps//bc, bc, samples//bs, bs, 2).permute(0, 1, 3, 5, 2, 4)


def _unblocks(blocks, shape):
    rx, chirps, samples = shape
    return blocks.permute(0, 1, 4, 2, 5, 3).reshape(rx, chirps, samples, 2)


@torch.no_grad()
def baq(cube, bits):
    """Encode and decode one cube with n-bit BAQ; returns (decoded cube, rate dict)."""
    shape = tuple(cube.shape)
    real = torch.view_as_real(cube.to(torch.complex128))
    blocks = _blocks(real, shape)
    rms = blocks.square().mean((-1, -2)).sqrt()
    code = torch.round(torch.log2(rms.clamp_min(1e-12))*8).clamp(-128, 127)
    sigma = torch.exp2(code/8)[..., None, None]
    thresholds, levels = (t.to(cube.device) for t in lloyd_max_gaussian(bits))
    index = torch.bucketize((blocks/sigma).contiguous(), thresholds)
    decoded = torch.view_as_complex(_unblocks(levels[index]*sigma, shape).contiguous()).to(cube.dtype)
    total = real.numel()*bits+code.numel()*BAQ_SCALE_BITS+HEADER_BITS
    return decoded, dict(total=int(total), accounting=f'BAQ {bits}-bit fixed rate (Kwok & Johnson 1989)')


@torch.no_grad()
def uniform(cube, bits):
    """Encode and decode one cube with the n-bit peak-scaled mid-tread quantizer."""
    real = torch.view_as_real(cube.to(torch.complex128))
    top = 2**(bits-1)-1
    step = real.abs().max().clamp_min(1e-12)/top
    decoded = torch.view_as_complex((torch.round(real/step).clamp(-top, top)*step).contiguous()).to(cube.dtype)
    total = real.numel()*bits+32+HEADER_BITS
    return decoded, dict(total=int(total), accounting=f'uniform {bits}-bit bit-depth reduction, fixed rate')
