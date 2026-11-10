"""RADIal data/coordinate adapter for the unchanged adaptive coherent codec.

Numerical production work stays on CUDA. Host exceptions are raw byte reads,
calibration/reference metadata loading and annotation-table selection. There is
no MATLAB ULA, invented chirp timing, CPU point cloud or Doppler-to-m/s conversion.
RADIal's native DDMA RD coordinates are unshifted cyclic bins 0..255.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .arm_e_adaptive_gpu_physics import (
    GPUPhysics, EPS, THRESHOLDS, COMMON_SPAN_DB, ca_cfar, pooled_ap,
)
from . import arm_e_adaptive_gpu_learning as learning
from .radial_compression_dataset import (
    SHAPE, CHIP_BYTES, RANGE_EXTENT_M, RadialCubeConfig, load_reference_classes,
    portable_path,
)


def _device(device, engineering):
    result = torch.device(device)
    if result.type != 'cuda' and not (engineering and result.type == 'cpu'):
        raise ValueError('RADIal production requires CUDA; CPU is engineering-only.')
    if result.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no RADIal CPU fallback.')
    return result


def load_cube_gpu(row, device='cuda:0', *, engineering=False):
    """Read four ADC byte buffers, decode I/Q and official RX order on device.

    int16 is interpreted on the little-endian host used by LMS/WSL; every sample
    converts exactly to float32. The native Fortran reshape is expressed as a
    Torch reshape/permutation, not a CPU complex-array reconstruction.
    """
    import sys
    if sys.byteorder != 'little':
        raise RuntimeError('This raw int16 boundary requires a little-endian host.')
    device = _device(device, engineering)
    chips = {}
    for ch in range(4):
        offset = int(row[f'chip{ch}_offset'])
        if offset < 0:
            raise ValueError('Negative RADIal ADC offset.')
        path = portable_path(row[f'chip{ch}_path'])
        with path.open('rb') as stream:
            stream.seek(offset)
            raw = bytearray(stream.read(CHIP_BYTES))
        if len(raw) != CHIP_BYTES:
            raise IOError(f'Incomplete chip {ch} ADC frame in {path}.')
        adc = torch.frombuffer(raw, dtype=torch.int16).to(device=device)
        iq = torch.complex(adc[0::2].to(torch.float32), adc[1::2].to(torch.float32))
        chips[ch] = iq.reshape(256, 4, 512).permute(1, 0, 2)
    cube = torch.cat([chips[ch] for ch in (3, 0, 1, 2)], dim=0).contiguous()
    if tuple(cube.shape) != SHAPE or not bool(torch.isfinite(cube).all()):
        raise ValueError('Malformed decoded RADIal cube.')
    return cube


def cyclic_cfar(power):
    """Exact existing RADIal wrapper: wrap six columns, native CA/NMS, crop."""
    padded = F.pad(power[None, None], (6, 6, 0, 0), mode='circular')[0, 0]
    return {name: value[:, 6:-6] for name, value in ca_cfar(padded).items()}


def _distances(points, bins, period=None):
    delta = (points[:, None]-bins[None]).abs()
    if period is not None:
        delta[:, :, 1] = torch.minimum(delta[:, :, 1], period-delta[:, :, 1])
    return delta


def ranked_metrics(detector, bins, thresholds, period=None):
    """Same one-to-one AP/F1 protocol; RD gate distances wrap cyclic columns."""
    points = torch.nonzero(detector['base_peaks'], as_tuple=False)
    scores = detector['score'][points[:, 0], points[:, 1]]
    order = torch.argsort(scores, descending=True, stable=True)
    points, scores = points[order], scores[order]
    n = len(points)
    tp = torch.zeros(n, dtype=torch.float64, device=scores.device)
    if len(bins) and n:
        delta = _distances(points, bins, period)
        inside = (delta <= 2).all(dim=2)
        distances = (delta.to(torch.float64)/2).square().sum(dim=2).masked_fill(~inside, torch.inf)
        target = distances.argmin(dim=1)
        supported = inside.any(dim=1)
        position = torch.arange(n, device=scores.device)
        first = torch.full((len(bins),), n, dtype=torch.int64, device=scores.device)
        first.scatter_reduce_(0, target[supported], position[supported], reduce='amin', include_self=True)
        tp = (supported & (position == first[target])).to(torch.float64)
    active = scores[:, None] > thresholds[None]
    tp_at = (tp[:, None]*active).sum(0)
    fp_at = ((1.-tp)[:, None]*active).sum(0)
    fn_at = len(bins)-tp_at
    return dict(AP=pooled_ap(scores, tp, len(bins)), precision=tp_at/(tp_at+fp_at).clamp_min(1),
        recall=tp_at/max(len(bins), 1), F1=2*tp_at/(2*tp_at+fp_at+fn_at).clamp_min(1),
        tp=tp_at, fp=fp_at, fn=fn_at, ranked_scores=scores, ranked_tp=tp,
        annotation_count=torch.tensor(len(bins), device=scores.device), peak_points=points)


class RadialGPUPhysics(GPUPhysics):
    """Inherit the exact coherent codec; replace dataset-specific processing only."""
    def __init__(self, reference_root, device='cuda:0', engineering=False, range_chunk=8):
        self.device = _device(device, engineering)
        if isinstance(range_chunk, bool) or int(range_chunk) != range_chunk or range_chunk < 1:
            raise ValueError('RADIal range_chunk must be a positive integer.')
        self.range_chunk = int(range_chunk)
        self.shape, self.cfg = SHAPE, RadialCubeConfig()
        self.block_slow, self.block_fast = 32, 64
        root = portable_path(reference_root)
        cls, _ = load_reference_classes(root/'SignalProcessing/rpl.py')
        # The supplied class loads calibration and constructs its declared
        # Hamming/DDMA tables. It performs no ADC processing at initialization.
        with redirect_stdout(io.StringIO()):
            reference = cls(str(root/'SignalProcessing/CalibrationTable.npy'), method='RA', device='cpu', lib='CuPy')
        if reference.CalibMat.shape != (751, 192):
            raise ValueError('Expected official RADIal 751-by-192 azimuth calibration.')
        self.range_window = self.tensor(reference.range_fft_coef)
        self.doppler_window = self.tensor(reference.doppler_fft_coef)
        self.calibration = self.tensor(reference.CalibMat, torch.complex128)
        self.virtual_window = self.tensor(reference.window, torch.complex128)
        slots = torch.cat((torch.tensor([0], device=self.device), torch.arange(5, 16, device=self.device)))*16
        self.tx_indices = (torch.arange(256, device=self.device)[:, None]+slots[None]) % 256
        self.ranges = torch.arange(512, device=self.device, dtype=torch.float64)*RANGE_EXTENT_M/512
        self.ranges_full = self.ranges
        self.vel = torch.arange(256, device=self.device, dtype=torch.float64)
        self.angles = self.tensor(reference.AoA_mat['Azimuth_table'])
        if tuple(self.angles.shape) != (751,) or not bool(torch.isfinite(self.calibration).all() & torch.isfinite(self.virtual_window).all()):
            raise ValueError('Malformed RADIal calibration metadata.')
        self.keep = torch.ones(512, dtype=torch.bool, device=self.device)
        self.thresholds = self.tensor(THRESHOLDS)

    @torch.no_grad()
    def ra_from_spectrum(self, spectrum):
        """Official calibrated 12-TX/16-RX DDMA hypotheses, bounded range chunks."""
        if spectrum.ndim != 3 or tuple(spectrum.shape[1:]) != (256, 16) or spectrum.device != self.device:
            raise ValueError('Expected device-resident [range,256,16] spectrum.')
        result = torch.empty((len(spectrum), 751), dtype=torch.float64, device=self.device)
        for start in range(0, len(spectrum), self.range_chunk):
            chunk = spectrum[start:start+self.range_chunk]
            mimo = chunk[:, self.tx_indices, :].reshape(-1, 192)
            response = (self.calibration@(mimo*self.virtual_window).T).abs()
            result[start:start+len(chunk)] = response.reshape(751, len(chunk), 256).sum(dim=2).T
        return result

    @torch.no_grad()
    def maps(self, cube):
        if not isinstance(cube, torch.Tensor) or cube.device != self.device or tuple(cube.shape) != SHAPE or not torch.is_complex(cube):
            raise ValueError('Expected device-resident RADIal IQ cube [16,256,512].')
        adc = cube.permute(2, 1, 0).to(torch.complex128)
        adc = adc-adc.mean(dim=(0, 1), keepdim=True)
        spectrum = torch.fft.fft(adc*self.range_window, n=512, dim=0)
        spectrum = torch.fft.fft(spectrum*self.doppler_window, n=256, dim=1)
        amplitude_rd = spectrum.abs().sum(dim=2)
        amplitude_ra = self.ra_from_spectrum(spectrum)
        tiny = torch.finfo(torch.float64).tiny
        rd, ra = 20*torch.log10(amplitude_rd.clamp_min(tiny)), 20*torch.log10(amplitude_ra.clamp_min(tiny))
        return dict(RD=rd, RA=ra, ranges=self.ranges, vel=self.vel, angle_axis_deg=self.angles,
            detectors=dict(RD=cyclic_cfar(10.**(rd/10.)), RA=ca_cfar(10.**(ra/10.))),
            doppler_units='cyclic_bin', power_convention='official amplitude squared')

    def label_bins(self, labels, post, kind):
        return self.label_bins_with_indices(labels, post, kind)[0]

    def label_bins_with_indices(self, labels, post, kind):
        """Return map bins and their input ``labels.iloc`` positions on device.

        Explicit provenance survives both annotation-evaluable filtering and
        physical-axis exclusion; DataFrame index labels are never treated as
        positional indices or silently renumbered after a filter.
        """
        field = 'doppler_bin' if kind == 'RD' else 'azimuth_deg'
        raw = self.tensor([(float(lab.get('range_m', math.nan)), float(lab.get(field, math.nan)))
                           for _, lab in labels.iterrows()]).reshape(-1, 2)
        rr, ss = post['ranges'], post['vel'] if kind == 'RD' else post['angle_axis_deg']
        valid = torch.isfinite(raw).all(1) & (raw[:, 0] >= rr.min()) & (raw[:, 0] <= rr.max()) & (raw[:, 1] >= ss.min()) & (raw[:, 1] <= ss.max())
        if 'annotation_evaluable' in labels:
            valid &= self.tensor(labels.annotation_evaluable.astype(bool).tolist(), torch.bool)
        positions = torch.arange(len(labels), device=self.device, dtype=torch.int64)[valid]
        raw = raw[valid]
        bins = torch.stack(((raw[:, 0, None]-rr).abs().argmin(1), (raw[:, 1, None]-ss).abs().argmin(1)), dim=1)
        return bins, positions

    @torch.no_grad()
    def evaluate(self, original, method, labels, detection_metrics=True):
        """Unchanged losses/counts; only RD association uses cyclic Doppler bins."""
        result = {}
        for kind, high in (('RD', 49152.), ('RA', 128.)):
            d0, d1 = original['detectors'][kind], method['detectors'][kind]
            bins = self.label_bins(labels, original, kind)
            period = 256 if kind == 'RD' else None
            best, outside = [], []
            for detector in (d0, d1):
                points = torch.nonzero(detector['base_peaks'], as_tuple=False)
                scores = detector['score'][points[:, 0], points[:, 1]]
                if len(bins) and len(points):
                    inside = (_distances(points, bins, period) <= 2).all(dim=2).T
                    best.append(torch.where(inside, scores[None], 0.).max(1).values)
                    outside.append(scores[~inside.any(0)])
                else:
                    best.append(self.tensor(torch.zeros(len(bins), device=self.device)))
                    outside.append(scores)
            original_best, candidate_best = best
            eligible = original_best[:, None] > self.thresholds
            preserved = eligible & (candidate_best[:, None] > self.thresholds)
            supported, retained = eligible.sum(0), preserved.sum(0)
            unmatched0 = (outside[0][:, None] > self.thresholds).sum(0)
            unmatched1 = (outside[1][:, None] > self.thresholds).sum(0)
            low_eligible = original_best > self.thresholds[0]
            margin = ((torch.minimum(original_best, self.thresholds[0]*1.25)-candidate_best).clamp_min(0)/self.thresholds[0]*low_eligible).sum()
            db0 = torch.clamp(10*torch.log10(original_best.clamp_min(1.25)), 10*math.log10(1.25), 10*math.log10(high))
            db1 = torch.clamp(10*torch.log10(candidate_best.clamp_min(1.25)), 10*math.log10(1.25), 10*math.log10(high))
            regret = ((db0-db1).clamp_min(0)*low_eligible).sum()/COMMON_SPAN_DB
            span = math.log(high/1.25)
            curves = [((scores.clamp_min(1.25).log()-math.log(1.25)).clamp(0, span)/span).sum() for scores in outside]
            p0, p1 = d0['power'], d1['power']
            scale = p0.mean()+EPS
            l0, l1 = torch.log1p(p0/scale), torch.log1p(p1/scale)
            values = dict(low_miss=(supported-retained)[0], six_miss=(supported-retained).sum(),
                low_supported=supported[0], six_supported=supported.sum(), low_preserved=retained[0], six_preserved=retained.sum(),
                unmatched=unmatched1.sum(), original_unmatched=unmatched0.sum(), unmatched_excess=(unmatched1.sum()-unmatched0.sum()).clamp_min(0),
                margin_deficit=margin, curve_regret=regret, unmatched_curve=curves[1], original_unmatched_curve=curves[0],
                unmatched_curve_excess=(curves[1]-curves[0]).clamp_min(0), map_error=(l1-l0).square().mean()/(l0.square().mean()+EPS),
                supported_by_threshold=supported, preserved_by_threshold=retained, missed_by_threshold=supported-retained,
                unmatched_by_threshold=unmatched1, original_unmatched_by_threshold=unmatched0,
                original_best_scores=original_best, reconstructed_best_scores=candidate_best)
            if detection_metrics:
                values.update(ranked_metrics(d1, bins, self.thresholds, period))
            result.update({kind+'_'+name: value for name, value in values.items()})
        for key in ('low_miss', 'six_miss', 'low_supported', 'six_supported', 'low_preserved', 'six_preserved',
                    'unmatched', 'original_unmatched', 'margin_deficit', 'curve_regret', 'map_error',
                    'unmatched_curve', 'original_unmatched_curve', 'unmatched_curve_excess', 'unmatched_excess'):
            result[key] = result['RD_'+key]+result['RA_'+key]
        return result


@torch.no_grad()
def static_features(physics, source, post):
    """Same 18 descriptors, with native DDMA bins attributed to shifted tiles.

    This is an index permutation only: detector scores/masks are not recomputed
    in a fictitious MATLAB coordinate system. RA remains a nominal range slab.
    """
    shifted = {}
    for kind, detector in post['detectors'].items():
        dims = (0, 1) if kind == 'RD' else (0,)
        shifted[kind] = {name: torch.fft.fftshift(value, dim=dims) for name, value in detector.items()}
    proxy = SimpleNamespace(device=physics.device, shape=physics.shape, keep=physics.keep, tensor=physics.tensor)
    return learning.static_features(proxy, source, dict(detectors=shifted))
