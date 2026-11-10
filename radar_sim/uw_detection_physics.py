"""The proposed codec's receiver layer for the UW raw-ADC automotive dataset.

Radar: TI AWR1843, 2 Tx (TDM) x 4 Rx = 8 virtual channels, 255 chirp loops of 128
complex samples, 21.0017 MHz/us, 4 Msps, 77 GHz, 30 frames/s. One frame is stored as
adcData [samples, chirps, rx, tx]; the codec's cube is [virtual channel, chirp, sample].

The codec (radial_v13_detection_codec) is unchanged. This module sets its tile geometry
and is the receiver it verifies against: the RADIal receiver (mean removal, range and
Doppler windows, amplitude sum over channels, frozen 11x11/3x3 CA-CFAR, 5x5 NMS) with
the Hamming windows of RAMP-CNN's own preprocessing.

The CFAR is RADIal's cyclic wrapper. Doppler is unshifted here, so zero Doppler lies on the
map edge, and a plain CA-CFAR would mark those cells invalid and miss every static object.

Differences from RADIal, all geometric: 255 chirps do not divide by 16, so tiles are
15 chirps wide, which keeps about the RADIal number of cells per tile.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

from . import radial_v13_detection_codec as codec
from .radial_adaptive_gpu_physics import cyclic_cfar

N_CHANNELS, N_CHIRPS, N_SAMPLES = 8, 255, 128
SAMPLE_RATE, SLOPE = 4e6, 21.0017e12
LIGHT = 299792458.
RANGE_BIN_M = LIGHT*SAMPLE_RATE/(2*SLOPE*N_SAMPLES)
RAW_BPCS = 32
TILE_DOPPLER, TILE_RANGE, VAR_BLOCK = 15, 32, (5, 4)
CLASS_NAMES = {0: 'person', 2: 'car', 3: 'motorbike', 5: 'bus', 7: 'truck', 80: 'cyclist'}


def configure_codec(tile_doppler=TILE_DOPPLER, tile_range=TILE_RANGE):
    """Tile geometry only; no codec equation changes."""
    codec.TILE_DOPPLER, codec.TILE_RANGE, codec.VAR_BLOCK = tile_doppler, tile_range, VAR_BLOCK
    codec.encode_ecdq.__wrapped__.__kwdefaults__['block'] = VAR_BLOCK  # a default bound when the function was defined


def load_frame(path):
    """One .mat frame -> numpy [channel, chirp, sample]: Tx 0 Rx 0-3, then Tx 1 Rx 0-3."""
    adc = sio.loadmat(path, variable_names=['adcData'])['adcData']
    if adc.shape != (N_SAMPLES, N_CHIRPS, 4, 2):
        raise ValueError(f'{path}: wrong frame shape {adc.shape}.')
    return np.ascontiguousarray(adc.reshape(N_SAMPLES, N_CHIRPS, N_CHANNELS, order='F').transpose(2, 1, 0))


def read_labels(path):
    """Rows of (uid, class id, px, py, width, length) in metres. Some sequences write
    every field as a float (``4.00,0.00,...``), so ids go through float."""
    rows = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        fields = line.strip().split(',')
        if len(fields) == 6:
            uid, cls, *rest = map(float, fields)
            rows.append((int(uid), int(cls), *rest))
    return rows


class UWPhysics:
    def __init__(self, device, engineering=False):
        self.device = torch.device(device)
        if self.device.type != 'cuda' and not (engineering and self.device.type == 'cpu'):
            raise ValueError('Production requires CUDA; CPU is engineering-only.')
        self.shape = (N_CHANNELS, N_CHIRPS, N_SAMPLES)
        real = dict(dtype=torch.float64, device=self.device)
        self.range_window = torch.hamming_window(N_SAMPLES, periodic=False, **real)[:, None, None]
        self.doppler_window = torch.hamming_window(N_CHIRPS, periodic=False, **real)[None, :, None]

    @torch.no_grad()
    def receive(self, cube):
        return codec.rd_receive(cube, self.range_window, self.doppler_window, cyclic_cfar)

    @torch.no_grad()
    def encoder_view(self, received):
        """The whole map is coded (C): unscored edge cells stay in the support."""
        detector = received['detector']
        return dict(received, detector=dict(detector, valid=torch.ones_like(detector['valid'])))

    def calibrate(self, maps, seed, per_map=1.):
        scores = codec.h0_scores(maps, self.shape, self.range_window, self.doppler_window, cyclic_cfar, self.device, seed)
        return codec.threshold_for_rate(scores, maps, per_map)


def alpha_db(alpha):
    return 10*math.log10(alpha)
