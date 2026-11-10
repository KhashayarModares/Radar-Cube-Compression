"""CFAR detections of the UW receiver against the dataset's labels.

A label is a position in range and angle; a CFAR detection is a range-Doppler cell. Each
detection takes its angle from an FFT over the 8 virtual channels of that cell, on the
axis of slice3d.py and without a TDM Doppler phase term, which did not bring detections
closer to labels. A detection supports a label inside RANGE_GATE and ANGLE_GATE bins.

The gates and the speed threshold were set on original maps and label files, never on a
decoded map: 95% of the closest detections lie within -4 to 5 range bins, and the angle
error has its knee near 4 bins. Speeds are 0 for parked objects, near 1.3 m/s for walkers.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from . import radial_v13_detection_codec as codec
from . import uw_detection_physics as uw
from . import uw_ramp_data as data

RANGE_GATE, ANGLE_GATE = 4, 6
MOVING_MPS, SPEED_SPAN, FRAME_RATE = .5, 15, 30.  # speed over +-1/2 s of a label's own track
ROBUST_MARGIN_DB = 1.


def label_rows(sequence):
    """{frame: [[range bin, angle bin, class, speed m/s]]} inside the field of view.
    Speed is relative to the radar and NaN for a track of one frame."""
    parsed, tracks = {}, {}
    for frame in sequence.frames:
        path = sequence.label(frame)
        parsed[int(frame)] = uw.read_labels(path) if path.is_file() else []
        for uid, _, px, py, _, _ in parsed[int(frame)]:
            tracks.setdefault(uid, []).append((int(frame), px, py))
    speed = {}
    for uid, track in tracks.items():
        for i, (frame, _, _) in enumerate(track):
            a, b = track[max(i-SPEED_SPAN, 0)], track[min(i+SPEED_SPAN, len(track)-1)]
            speed[uid, frame] = math.hypot(b[1]-a[1], b[2]-a[2])*FRAME_RATE/(b[0]-a[0]) if b[0] > a[0] else math.nan
    rows = {}
    for frame, labels in parsed.items():
        rows[frame] = [[*found[0], speed[row[0], frame]] for row in labels for found in [data.objects_of([row])] if found]
    return rows


def motion(speed):
    return 'unknown' if math.isnan(speed) else 'moving' if speed >= MOVING_MPS else 'static'


def frame_motion(labels):
    """A map with any moving label is moving; static needs at least one label."""
    states = {motion(row[3]) for row in labels}
    return 'moving' if 'moving' in states else 'static' if 'static' in states else 'no_labels'


def angle_bins(spectrum, points):
    """Strongest angle bin of the 128-point axis for [range, Doppler] detections."""
    if not len(points):  # cuFFT refuses an empty batch
        return torch.zeros(0, dtype=torch.int64, device=spectrum.device)
    window =torch.hamming_window(uw.N_CHANNELS, periodic=False, dtype=torch.float64, device=spectrum.device)
    cells = spectrum[points[:, 0], points[:, 1]]
    return torch.fft.fftshift(torch.fft.fft(cells*window, data.N_FFT, dim=1), dim=1).abs().argmax(1)


def peaks(received, alpha):
    detector = received['detector']
    points = codec.detections(detector, alpha)
    stable = codec.stable_peaks(detector, alpha, 10**(ROBUST_MARGIN_DB/10))[points[:, 0], points[:, 1]]
    return dict(points=points, angles=angle_bins(received['spectrum'], points), stable=stable)


def gated(bins, found):
    """[labels, detections] True where the detection supports the label."""
    return (((bins[:, None, 0]-found['points'][None, :, 0]).abs() <= RANGE_GATE)
            & ((bins[:, None, 1]-found['angles'][None]).abs() <= ANGLE_GATE))


def compare(original, method, bins):
    """Per-label booleans and the detections no label explains, on both maps."""
    a, b = gated(bins, original), gated(bins, method)
    return dict(detected=a.any(1), robust=(a & original['stable'][None]).any(1), kept=b.any(1),
                unmatched_original=int((~a.any(0)).sum()), unmatched_decoded=int((~b.any(0)).sum()),
                matched_original=a.any(0), matched_decoded=b.any(0))


def label_tensor(labels, device):
    return torch.as_tensor(np.array([row[:2] for row in labels], np.int64).reshape(-1, 2), device=device)
