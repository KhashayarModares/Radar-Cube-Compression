"""UW raw-ADC adapter for the V12.40 tile codec, tuned reward terms and CFAR scoring.

The noise-referenced codec, the 4.6-bpcs cap, the thresholds and the receiver chain are V12.40's.
Everything below is geometry or a consequence of what the labels contain:

  * The cube is 8 x 255 x 128, so 64 tiles are 32 chirps by 16 samples; the last Doppler row
    holds 31 chirps because 255 does not divide by 8.
  * The RA map comes from a 128-point FFT over the 8 virtual channels, in place of RADIal's
    calibrated 751-point azimuth table.
  * UW labels give range and angle but no Doppler. On the RA map a label is used directly,
    inside the gate of radar_sim/uw_cfar_labels.py. On the RD map it is anchored to the
    strongest original peak inside that gate, and a label with no such peak has no RD position.
  * Labels beyond +-60 degrees are ignored: they lie outside the radar's useful field of view.

Reward terms act at the detector's own threshold alpha: a label counts only with a robust peak,
because weaker labels flip by chance, and lost robust detections and new robust peaks are charged
to their tile. Metrics that cannot exist here are absent: no mAP or F1 on this dataset.
"""
from __future__ import annotations

import torch

from . import radial_cap_codec as cap
from . import uw_cfar_labels as labelling
from . import uw_detection_physics as uw
from . import uw_ramp_data as data
from .radial_adaptive_gpu_physics import THRESHOLDS, _device, _distances, ca_cfar, cyclic_cfar

v13 = cap.v13
DOPPLER_TILES, RANGE_TILES = 8, 8
TILE_DOPPLER, TILE_RANGE = 32, 16
RD_GATE = (2, 2)                                        # V12.40, around the anchored peak
RA_GATE = (labelling.RANGE_GATE, labelling.ANGLE_GATE)  # the label itself, a few cells from the peak
SPREAD = DOPPLER_TILES                                  # an RA event is shared by every Doppler tile of its band
MAX_LABEL_ANGLE = 60.
ROBUST_MARGIN = 10**(labelling.ROBUST_MARGIN_DB/10)


def visible(labels):
    """Labels inside the useful field of view."""
    return [row for row in labels if abs(data.ANGLE_GRID[row[1]]) <= MAX_LABEL_ANGLE]


class UWPPOPhysics(cap.CapPhysics):
    def __init__(self, device='cuda:0', engineering=False):
        self.device = _device(device, engineering)
        self.shape = (uw.N_CHANNELS, uw.N_CHIRPS, uw.N_SAMPLES)
        self.block_slow, self.block_fast = TILE_DOPPLER, TILE_RANGE
        real = dict(dtype=torch.float64, device=self.device)
        self.range_window = torch.hamming_window(uw.N_SAMPLES, periodic=False, **real)[:, None, None]
        self.doppler_window = torch.hamming_window(uw.N_CHIRPS, periodic=False, **real)[None, :, None]
        self.angle_window = torch.hamming_window(uw.N_CHANNELS, periodic=False, **real)[None, None, :]
        self.ranges = self.ranges_full = self.tensor(data.RANGE_GRID)
        self.vel = torch.arange(uw.N_CHIRPS, **real)
        self.angles = self.tensor(data.ANGLE_GRID)
        self.keep = torch.ones(uw.N_SAMPLES, dtype=torch.bool, device=self.device)
        self.thresholds = self.tensor(THRESHOLDS)

    @torch.no_grad()
    def maps(self, cube):
        if not isinstance(cube, torch.Tensor) or cube.device != self.device or tuple(cube.shape) != self.shape or not torch.is_complex(cube):
            raise ValueError('Expected device-resident UW IQ cube [8,255,128].')
        adc = cube.permute(2, 1, 0).to(torch.complex128)
        adc = adc-adc.mean(dim=(0, 1), keepdim=True)
        spectrum = torch.fft.fft(adc*self.range_window, n=uw.N_SAMPLES, dim=0)
        spectrum = torch.fft.fft(spectrum*self.doppler_window, n=uw.N_CHIRPS, dim=1)
        amplitude_ra = torch.fft.fftshift(torch.fft.fft(spectrum*self.angle_window, data.N_FFT, dim=2), dim=2).abs().sum(dim=1)
        tiny = torch.finfo(torch.float64).tiny
        rd, ra = 20*torch.log10(spectrum.abs().sum(dim=2).clamp_min(tiny)), 20*torch.log10(amplitude_ra.clamp_min(tiny))
        return dict(RD=rd, RA=ra, spectrum=spectrum, ranges=self.ranges, vel=self.vel, angle_axis_deg=self.angles,
                    detectors=dict(RD=cyclic_cfar(10.**(rd/10.)), RA=ca_cfar(10.**(ra/10.))))

    @torch.no_grad()
    def label_positions(self, original, bins):
        """RA label bins and the RD peaks that anchor them, from the original map only."""
        detector = original['detectors']['RD']
        points = torch.nonzero(detector['base_peaks'] & (detector['score'] > self.thresholds[0]), as_tuple=False)
        found = dict(points=points, angles=labelling.angle_bins(original['spectrum'], points))
        inside = labelling.gated(bins, found)
        anchored = inside.any(1)
        if not bool(anchored.any()):
            return dict(RA=bins, RD=torch.zeros((0, 2), dtype=torch.int64, device=self.device))
        scores = detector['score'][points[:, 0], points[:, 1]]
        best = torch.where(inside, scores[None], 0.).argmax(1)
        return dict(RA=bins, RD=points[best[anchored]])

    def label_bins(self, labels, post, kind):
        return labels[kind]

    @torch.no_grad()
    def evaluate(self, original, method, labels, detection_metrics=False):
        """V12.40 counts with UW gates, cyclic Doppler and label positions from ``label_positions``."""
        result = {}
        for kind, gate, period in (('RD', RD_GATE, uw.N_CHIRPS), ('RA', RA_GATE, None)):
            bins = labels[kind]
            best, outside = [], []
            for detector in (original['detectors'][kind], method['detectors'][kind]):
                points = torch.nonzero(detector['base_peaks'], as_tuple=False)
                scores = detector['score'][points[:, 0], points[:, 1]]
                if len(bins) and len(points):
                    delta = _distances(points, bins, period)
                    inside = ((delta[:, :, 0] <= gate[0]) & (delta[:, :, 1] <= gate[1])).T
                    best.append(torch.where(inside, scores[None], 0.).max(1).values)
                    outside.append(scores[~inside.any(0)])
                else:
                    best.append(torch.zeros(len(bins), dtype=torch.float64, device=self.device))
                    outside.append(scores)
            eligible = best[0][:, None] > self.thresholds
            preserved = eligible & (best[1][:, None] > self.thresholds)
            supported, retained = eligible.sum(0), preserved.sum(0)
            unmatched0, unmatched1 = [(values[:, None] > self.thresholds).sum(0).sum() for values in outside]
            values = dict(low_miss=(supported-retained)[0], six_miss=(supported-retained).sum(), low_supported=supported[0], six_supported=supported.sum(),
                          unmatched=unmatched1, original_unmatched=unmatched0, unmatched_excess=(unmatched1-unmatched0).clamp_min(0))
            result.update({f'{kind}_{name}': value for name, value in values.items()})
            result.update({f'{kind}_original_best_scores': best[0], f'{kind}_reconstructed_best_scores': best[1]})
        for key in ('low_miss', 'six_miss', 'low_supported', 'six_supported', 'unmatched', 'original_unmatched', 'unmatched_excess'):
            result[key] = result['RD_'+key]+result['RA_'+key]
        return result


def tile_of(rows, columns):
    """Native RD bins to the shifted codec tile that owns them."""
    doppler = ((columns+uw.N_CHIRPS//2) % uw.N_CHIRPS)//TILE_DOPPLER
    return doppler*RANGE_TILES+range_band(rows)


def range_band(rows):
    return ((rows+uw.N_SAMPLES//2) % uw.N_SAMPLES)//TILE_RANGE


def _tile_counts(mask, device):
    rows, columns = torch.nonzero(mask, as_tuple=True)
    return torch.bincount(tile_of(rows, columns), minlength=DOPPLER_TILES*RANGE_TILES).to(torch.float64) if len(rows) else torch.zeros(DOPPLER_TILES*RANGE_TILES, dtype=torch.float64, device=device)


@torch.no_grad()
def tile_terms(physics, original, method, metric, labels, alpha):
    """Attribute each decision to the tile that controls it: an RD cell to its tile, an RA event to its range band."""
    device = physics.device
    n = DOPPLER_TILES*RANGE_TILES
    floor = alpha*ROBUST_MARGIN
    miss, loss, excess = (torch.zeros(n, dtype=torch.float64, device=device) for _ in range(3))
    bands = torch.arange(n, device=device) % RANGE_TILES

    def add(target, rows, columns, values, kind):
        if not len(rows):
            return
        if kind == 'RD':
            target.scatter_add_(0, tile_of(rows, columns), values)
        else:
            per_band = torch.zeros(RANGE_TILES, dtype=torch.float64, device=device)
            per_band.scatter_add_(0, range_band(rows), values)
            target.add_(per_band[bands]/SPREAD)

    for kind, gate, period in (('RD', RD_GATE, uw.N_CHIRPS), ('RA', RA_GATE, None)):
        bins = labels[kind]
        o, c = metric[kind+'_original_best_scores'], metric[kind+'_reconstructed_best_scores']
        supported = o > floor
        add(miss, bins[:, 0], bins[:, 1], (supported & (c <= alpha)).to(torch.float64), kind)
        add(loss, bins[:, 0], bins[:, 1], (10*torch.log10(o.clamp_min(1e-3)/c.clamp_min(1e-3))).clamp(0, 20)*supported, kind)
        counts = []
        for detector in (original['detectors'][kind], method['detectors'][kind]):
            points = torch.nonzero(detector['base_peaks'] & (detector['score'] > alpha), as_tuple=False)
            if len(bins) and len(points):
                delta = _distances(points, bins, period)
                points = points[~((delta[:, :, 0] <= gate[0]) & (delta[:, :, 1] <= gate[1])).any(1)]
            index = tile_of(points[:, 0], points[:, 1]) if kind == 'RD' else range_band(points[:, 0])
            counts.append(torch.bincount(index, minlength=n if kind == 'RD' else RANGE_TILES).to(torch.float64))
        extra = (counts[1]-counts[0]).clamp_min(0)
        excess.add_(extra if kind == 'RD' else extra[bands]/SPREAD)
    d0, d1 = original['detectors']['RD'], method['detectors']['RD']
    s0, s1 = v13.peak_scores(d0), v13.peak_scores(d1)
    near0, near1 = v13.neighbourhood_max(s0, v13.NMS_RADIUS, v13.NMS_RADIUS), v13.neighbourhood_max(s1, v13.NMS_RADIUS, v13.NMS_RADIUS)
    robust_lost = _tile_counts(v13.stable_peaks(d0, alpha, ROBUST_MARGIN) & (near1 <= alpha), device)
    robust_new = _tile_counts(v13.stable_peaks(d1, alpha, ROBUST_MARGIN) & (near0 <= alpha/ROBUST_MARGIN), device)
    return dict(miss=miss, score_loss_db=loss, excess=excess, robust_lost=robust_lost, robust_new=robust_new)
