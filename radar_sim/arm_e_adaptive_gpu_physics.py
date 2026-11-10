"""Device-resident coherent codec and the unchanged MATLAB CFAR protocol.

Production requires CUDA. ``engineering=True, device='cpu'`` is an explicit
parity-test mode, never a fallback. File decoding and report serialization are
outside this module. No numerical NumPy/SciPy path is used here.

The one-bit quantizer is NEW: retained real and imaginary values use two signed
levels (-a,+a), where a is their mean absolute magnitude. This is the exact
least-squares amplitude for those fixed signs; using the peak as a would amplify
weak retained coefficients. For q>=2 the historical symmetric rounded quantizer
is unchanged. Bit counts describe the declared byte-aligned syntax, not a saved
bitstream. Original-relative any-in-gate preservation and one-to-one AP are
deliberately distinct metrics.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

EPS = 1e-12
THRESHOLDS = (1.25, 2., 4., 8., 16., 32.)
COMMON_SPAN_DB = 10. * math.log10(128. / 1.25)


def _align8(value):
    return 8 * torch.div(value + 7, 8, rounding_mode='floor')


def collapse_plateaus(power, candidates):
    """Four-connected components; first row-major maximum, as scipy.label.

    Adjacent 5x5 local maxima necessarily have equal power. Label propagation
    stays on the device; only a convergence predicate synchronizes the host.
    """
    ids = torch.arange(power.numel(), device=power.device).reshape(power.shape) + 1
    sentinel = power.numel() + 1
    labels = torch.where(candidates, ids, sentinel)
    for _ in range(sum(power.shape)):
        padded = F.pad(labels, (1, 1, 1, 1), value=sentinel)
        nearby = torch.minimum(torch.minimum(padded[:-2, 1:-1], padded[2:, 1:-1]),
                               torch.minimum(padded[1:-1, :-2], padded[1:-1, 2:]))
        updated = torch.where(candidates, torch.minimum(labels, nearby), sentinel)
        if torch.equal(updated, labels):
            return candidates & (updated == ids)
        labels = updated
    raise RuntimeError('Plateau component propagation did not converge.')


def ca_cfar(power):
    """Frozen complete 11x11/3x3 CA-CFAR window and fixed 5x5 power NMS."""
    if power.ndim != 2 or power.dtype != torch.float64 or torch.is_complex(power):
        raise ValueError('CA-CFAR expects a float64 two-dimensional power map.')
    power = power.clamp_min(0)
    kernel = torch.ones((1, 1, 11, 11), dtype=power.dtype, device=power.device)
    kernel[:, :, 4:7, 4:7] = 0
    # No cuDNN mixed-precision or nondeterministic algorithm substitution.
    with torch.backends.cudnn.flags(deterministic=True, benchmark=False, allow_tf32=False):
        noise = F.conv2d(power[None, None], kernel, padding=5)[0, 0] / 112.
    valid = torch.zeros_like(power, dtype=torch.bool)
    if power.shape[0] >= 11 and power.shape[1] >= 11:
        valid[5:-5, 5:-5] = True
    maximum = F.max_pool2d(F.pad(power[None, None], (2, 2, 2, 2), mode='replicate'), 5, stride=1)[0, 0]
    peaks = collapse_plateaus(power, valid & (power > 0) & (power >= maximum))
    return dict(power=power, noise=noise, score=power/(noise+EPS), valid=valid,
                base_peaks=peaks, peaks=peaks)


def ranked_detection_metrics(detector, bins, thresholds):
    """One-to-one confidence ranking with nearest-gate duplicates counted FP.

    Return TP/FP scores for exact later population pooling (do not average
    per-cube AP to claim population AP). Equal-confidence cutoffs are grouped.
    """
    points = torch.nonzero(detector['base_peaks'], as_tuple=False)
    scores = detector['score'][points[:, 0], points[:, 1]]
    order = torch.argsort(scores, descending=True, stable=True)
    points, scores = points[order], scores[order]
    n = len(points)
    tp = torch.zeros(n, dtype=torch.float64, device=scores.device)
    if len(bins) and n:
        delta = (points[:, None, :] - bins[None, :, :]).abs()
        inside = (delta <= 2).all(dim=2)
        distance = (delta.to(torch.float64)/2).square().sum(dim=2)
        distance = distance.masked_fill(~inside, torch.inf)
        target = distance.argmin(dim=1)
        supported = inside.any(dim=1)
        position = torch.arange(n, device=scores.device)
        first = torch.full((len(bins),), n, dtype=torch.int64, device=scores.device)
        first.scatter_reduce_(0, target[supported], position[supported], reduce='amin', include_self=True)
        tp = (supported & (position == first[target])).to(torch.float64)
    fp = 1.-tp
    active = scores[:, None] > thresholds[None, :]
    tp_at = (tp[:, None]*active).sum(dim=0)
    fp_at = (fp[:, None]*active).sum(dim=0)
    fn_at = len(bins)-tp_at
    precision = tp_at/(tp_at+fp_at).clamp_min(1)
    recall = tp_at/max(len(bins), 1)
    f1 = 2*tp_at/(2*tp_at+fp_at+fn_at).clamp_min(1)
    ap = pooled_ap(scores, tp, len(bins))
    return dict(AP=ap, precision=precision, recall=recall, F1=f1,
                tp=tp_at, fp=fp_at, fn=fn_at, ranked_scores=scores, ranked_tp=tp,
                annotation_count=torch.tensor(len(bins), device=scores.device), peak_points=points)


def pooled_ap(scores, tp, annotation_count):
    """All-points precision-envelope AP; inputs stay on their current device."""
    if tp.shape != scores.shape or not bool(((tp == 0) | (tp == 1)).all()):
        raise ValueError('Ranked AP requires one binary TP indicator per score.')
    count = torch.as_tensor(annotation_count, device=scores.device, dtype=torch.float64)
    if not len(scores):
        return torch.where(count > 0, count.new_zeros(()), count.new_full((), torch.nan))
    order = torch.argsort(scores, descending=True, stable=True)
    scores, tp = scores[order], tp[order]
    # Binary counts have an exact integer prefix scan. CUDA float64 cumsum is
    # not deterministic on some supported Torch versions; do not relax strict
    # determinism or move ranked detections to CPU to work around that.
    cumulative = tp.to(torch.int64).cumsum(0).to(torch.float64)
    precision = cumulative/torch.arange(1, len(tp)+1, device=tp.device, dtype=torch.float64)
    recall = cumulative/count.clamp_min(1)
    end = torch.cat((scores[:-1] != scores[1:], torch.ones(1, dtype=torch.bool, device=scores.device)))
    precision, recall = precision[end], recall[end]
    recall = torch.cat((recall.new_zeros(1), recall, recall.new_ones(1)))
    precision = torch.cat((precision.new_zeros(1), precision, precision.new_zeros(1)))
    envelope = torch.cummax(precision.flip(0), dim=0).values.flip(0)
    ap = ((recall[1:]-recall[:-1])*envelope[1:]).sum()
    return torch.where(count > 0, ap, ap.new_full((), torch.nan))


@torch.no_grad()
def tile_action_descriptors(source, actions):
    """TRAIN audit only: exact action costs/Parseval errors without inverses.

    The historical fixed-menu action descriptors can be examined for correlation
    even though the delta controller has no fixed physical action head. This
    routine does not select or remove actions and is not used for deployment.
    """
    spectrum = source['spectrum']
    action = torch.as_tensor(actions, dtype=torch.float64, device=spectrum.device)
    if action.ndim != 2 or action.shape[1] != 2 or not len(action):
        raise ValueError('Audit actions must be a nonempty N-by-2 rho/q matrix.')
    rho, q = action[:, 0], action[:, 1]
    if not bool((torch.isfinite(action).all() & ((rho > 0) & (rho <= 1) & (q >= 1) & (q <= 30) & (q == q.round())).all())):
        raise ValueError('Invalid audit physical actions.')
    costs, errors = [], []
    for tile in source['tiles']:
        z = tile['Z'].reshape(spectrum.shape[0], -1)
        cells = z.shape[1]
        kept = torch.ceil(rho*cells).to(torch.int64).clamp(1, cells)
        rank_mask = torch.arange(cells, device=z.device)[None] < kept[:, None]
        mask = torch.zeros_like(rank_mask)
        mask[:, tile['rank_desc']] = rank_mask
        selected = z[None]*mask[:, None]
        peak = torch.maximum(selected.real.abs().amax(dim=(1, 2)), selected.imag.abs().amax(dim=(1, 2)))
        binary_amplitude = (selected.real.abs().sum(dim=(1, 2))+selected.imag.abs().sum(dim=(1, 2)))/(2*z.shape[0]*kept)
        scale = torch.where(q == 1, binary_amplitude, peak)
        denominator = torch.where(scale > 0, scale, torch.ones_like(scale))[:, None, None]
        scale = scale[:, None, None]
        levels = (2.**(q-1)-1).clamp_min(1)[:, None, None]
        real = torch.round(selected.real/denominator*levels)*scale/levels
        imag = torch.round(selected.imag/denominator*levels)*scale/levels
        real = torch.where((q == 1)[:, None, None], torch.where(selected.real >= 0, scale, -scale), real)
        imag = torch.where((q == 1)[:, None, None], torch.where(selected.imag >= 0, scale, -scale), imag)
        reconstructed = torch.complex(real, imag)*mask[:, None]
        errors.append((z[None]-reconstructed).abs().square().sum(dim=(1, 2))/source['cube_energy'].clamp_min(EPS))
        support = torch.minimum(_align8(torch.tensor(cells, device=z.device)), _align8(kept*max(1, (cells-1).bit_length())))
        costs.append(_align8(2*q.to(torch.int64)*z.shape[0]*kept)+support+128)
    return dict(action_bits_by_block=torch.stack(costs),
                action_distortion_by_block=torch.stack(errors))


class GPUPhysics:
    """One-cube-at-a-time signal computation; all numeric buffers are tensors."""
    def __init__(self, cfg, device='cuda:0', engineering=False, block_slow=32,
                 block_fast=64, theta_res=121, range_min=1., range_max=150.):
        self.device = torch.device(device)
        if self.device.type != 'cuda' and not (engineering and self.device.type == 'cpu'):
            raise ValueError('Production requires CUDA; CPU is engineering-only.')
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; no CPU fallback is permitted.')
        self.cfg = cfg
        self.shape = (int(cfg.num_rx), int(cfg.M), int(cfg.N))
        self.block_slow, self.block_fast = int(block_slow), int(block_fast)
        if min(*self.shape, self.block_slow, self.block_fast, int(theta_res)) < 1:
            raise ValueError('All cube, tile and angle dimensions must be positive.')
        self.thresholds = self.tensor(THRESHOLDS)
        self.theta = torch.linspace(-math.pi/3, math.pi/3, int(theta_res), dtype=torch.float64, device=self.device)
        antenna = torch.arange(cfg.num_rx, dtype=torch.float64, device=self.device)
        self.beam = torch.exp(1j*math.pi*torch.sin(self.theta[:, None])*antenna[None, :])/math.sqrt(cfg.num_rx)
        self.fast = torch.hann_window(cfg.N, periodic=False, dtype=torch.float64, device=self.device)[None, None, :]
        self.slow = torch.hann_window(cfg.M, periodic=False, dtype=torch.float64, device=self.device)[None, :, None]
        self.ranges_full = cfg.c0*torch.fft.fftshift(torch.fft.fftfreq(cfg.N, d=1/cfg.fs, dtype=torch.float64, device=self.device))/(2*cfg.S)
        self.vel = cfg.lam/2*torch.fft.fftshift(torch.fft.fftfreq(cfg.M, d=cfg.Tchirp, dtype=torch.float64, device=self.device))
        self.keep = (self.ranges_full >= float(range_min)) & (self.ranges_full <= float(range_max))
        if not bool(self.keep.any()):
            raise ValueError('No range bins lie within the physical ROI.')
        self.ranges = self.ranges_full[self.keep]
        self.angles = torch.rad2deg(self.theta)

    def tensor(self, value, dtype=torch.float64):
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    @torch.no_grad()
    def prepare(self, cube):
        if not isinstance(cube, torch.Tensor) or cube.device != self.device:
            raise ValueError('Transfer decoded IQ to the declared device before prepare().')
        if tuple(cube.shape) != self.shape or not torch.is_complex(cube) or not bool(torch.isfinite(cube).all()):
            raise ValueError('Expected a finite complex [Rx, chirp, fast-time] cube matching cfg.')
        spectrum = torch.fft.fftshift(torch.fft.fft2(cube.to(torch.complex128), dim=(1, 2), norm='ortho'), dim=(1, 2))
        energy = spectrum.abs().square().sum()
        tiles, slices = [], []
        for m in range(0, self.shape[1], self.block_slow):
            for n in range(0, self.shape[2], self.block_fast):
                slc = (slice(None), slice(m, m+self.block_slow), slice(n, n+self.block_fast))
                z = spectrum[slc]
                power = z.abs().square().sum(dim=0).reshape(-1)
                rank = torch.argsort(power, descending=True, stable=True)
                e = power.sum()
                probability = power/e.clamp_min(EPS)
                p = probability.clamp_min(torch.finfo(torch.float64).tiny)
                tile = dict(Z=z, rank_desc=rank, support_cells=power.numel(), tile_energy=e,
                            block_mean_power=e/z.numel(), power=power,
                            spectral_entropy_normalized=-(probability*p.log()).sum()/math.log(max(2, power.numel())),
                            spectral_peak_to_mean_log=torch.log1p(power.max()/power.mean().clamp_min(EPS)))
                for fraction, name in ((.01, 'top1'), (.05, 'top5'), (.10, 'top10')):
                    k = max(1, math.ceil(fraction*len(power)))
                    tile['spectral_'+name+'_fraction'] = power[rank[:k]].sum()/e.clamp_min(EPS)
                tiles.append(tile)
                slices.append(slc)
        return dict(cube=cube, spectrum=spectrum, tiles=tiles, prepared_blocks=tiles,
                    slices=slices, tile_slices=slices, cube_energy=energy,
                    whole_cube_energy=energy, cube_shape=self.shape,
                    cube_mean_power=energy/cube.numel(), cube_header_bits=self.tensor(768.))

    @torch.no_grad()
    def encode(self, source, rho, q):
        rho, q = self.tensor(rho), self.tensor(q)
        if rho.shape != (len(source['tiles']),) or q.shape != rho.shape:
            raise ValueError('Supply exactly one rho and q per tile.')
        if not bool(torch.isfinite(rho).all() & torch.isfinite(q).all()) or not bool(((rho > 0) & (rho <= 1) & (q >= 1) & (q <= 30) & (q == q.round())).all()):
            raise ValueError('Codec actions require 0<rho<=1 and integer 1<=q<=30.')
        out = torch.zeros_like(source['spectrum'])
        bit_rows, kept_rows, scales = [], [], []
        for i, (tile, slc) in enumerate(zip(source['tiles'], source['slices'])):
            z, cells = tile['Z'], tile['support_cells']
            kept = torch.ceil(rho[i]*cells).to(torch.int64).clamp(1, cells)
            ranked = torch.arange(cells, device=self.device) < kept
            mask = torch.zeros(cells, dtype=torch.bool, device=self.device)
            mask[tile['rank_desc']] = ranked
            mask = mask.reshape(z.shape[1:])
            selected = z*mask
            peak = torch.maximum(selected.real.abs().max(), selected.imag.abs().max())
            binary_amplitude = (selected.real.abs().sum()+selected.imag.abs().sum())/(2*self.shape[0]*kept)
            scale = torch.where(q[i] == 1, binary_amplitude, peak)
            levels = (2.**(q[i]-1)-1).clamp_min(1)
            denominator = torch.where(scale > 0, scale, torch.ones_like(scale))
            real = torch.round(selected.real/denominator*levels)*scale/levels
            imag = torch.round(selected.imag/denominator*levels)*scale/levels
            # MSE-optimal common amplitude for fixed one-bit signs. For n real
            # retained components, SSE(a)=sum(x^2)-2*a*sum(abs(x))+n*a^2.
            # Its minimizer a=mean(abs(x)) cannot have higher SSE than a=0.
            # Exactly-zero tiles reconstruct zero. No NMSE rejection cap is used.
            real = torch.where(q[i] == 1, torch.where(selected.real >= 0, scale, -scale), real)
            imag = torch.where(q[i] == 1, torch.where(selected.imag >= 0, scale, -scale), imag)
            out[slc] = torch.complex(real, imag)*mask
            index_width = max(1, (cells-1).bit_length())
            support = torch.minimum(_align8(self.tensor(cells, torch.int64)), _align8(kept*index_width))
            payload = _align8(2*q[i].to(torch.int64)*self.shape[0]*kept)
            bit_rows.append(torch.stack((payload, support, payload+support+128)))
            kept_rows.append(kept)
            scales.append(scale)
        cube_hat = torch.fft.ifft2(torch.fft.ifftshift(out, dim=(1, 2)), dim=(1, 2), norm='ortho').to(source['cube'].dtype)
        bits = torch.stack(bit_rows)
        total = bits[:, 2].sum()+768
        nmse = (source['cube'].to(torch.complex128)-cube_hat).abs().square().sum()/source['cube_energy'].clamp_min(EPS)
        accounted_bpcs = total.to(torch.float64)/source['cube'].numel()
        return cube_hat, dict(bpcs=accounted_bpcs, realized_bpcs=accounted_bpcs,
                             realized_bits=total, nmse=nmse, cube_nmse=nmse,
                             payload_bits=bits[:, 0].sum(), support_bits=bits[:, 1].sum(),
                             header_bits=self.tensor(768+128*len(bits)), tile_bits=bits[:, 2],
                             kept=torch.stack(kept_rows), rho=rho, q=q, scale=torch.stack(scales),
                             spectral_nmse=(source['spectrum']-out).abs().square().sum()/source['cube_energy'].clamp_min(EPS))

    @torch.no_grad()
    def full_fft(self, cube, q=14):
        """Historical independent FFT14: full 3-D FFT, one scale, 128-bit header.

        This is intentionally not encode(rho=1,q=14): that would silently
        replace the comparator with the coherent tiled codec and extra headers.
        """
        if not isinstance(cube, torch.Tensor) or cube.device != self.device or tuple(cube.shape) != self.shape or not torch.is_complex(cube):
            raise ValueError('full_fft() requires the device-resident IQ cube.')
        if isinstance(q, bool) or int(q) != q or not 2 <= int(q) <= 30:
            raise ValueError('Historical full-FFT comparator requires integer q>=2.')
        z = torch.fft.fftn(cube.to(torch.complex128), dim=(0, 1, 2))
        scale = torch.maximum(z.real.abs().max(), z.imag.abs().max())
        denominator = torch.where(scale > 0, scale, torch.ones_like(scale))
        levels = 2**(int(q)-1)-1
        quantized = torch.complex(torch.round(z.real/denominator*levels),
                                  torch.round(z.imag/denominator*levels))*scale/levels
        result = torch.fft.ifftn(quantized, dim=(0, 1, 2)).to(cube.dtype)
        nmse = (cube.to(torch.complex128)-result).abs().square().sum()/cube.abs().square().sum().clamp_min(EPS)
        bits = self.tensor(2*int(q)*cube.numel()+128.)
        return result, dict(bpcs=bits/cube.numel(), realized_bpcs=bits/cube.numel(),
                            realized_bits=bits, nmse=nmse, cube_nmse=nmse,
                            payload_bits=bits-128, support_bits=self.tensor(0.), header_bits=self.tensor(128.))

    @torch.no_grad()
    def maps(self, cube):
        if not isinstance(cube, torch.Tensor) or cube.device != self.device or tuple(cube.shape) != self.shape:
            raise ValueError('maps() requires a device-resident IQ cube matching cfg.')
        raw = cube.to(torch.complex128)
        ranged = torch.fft.fftshift(torch.fft.fft(raw*self.slow*self.fast, dim=2), dim=2)
        filtered = torch.zeros_like(ranged)
        filtered[:, 1:] = ranged[:, 1:]-ranged[:, :-1]
        spectrum = torch.fft.fftshift(torch.fft.fft(filtered, dim=1), dim=1)
        rd_power = spectrum.abs().square().sum(dim=0)
        ra_power = torch.zeros((len(self.theta), self.shape[2]), dtype=torch.float64, device=self.device)
        # Batch beamforming only (bounded memory), retaining increasing-Doppler
        # accumulation order from the native processor and the first LMS study.
        for first in range(0, self.shape[1], 8):
            chunk = spectrum[:, first:first+8, :].permute(1, 0, 2)
            powers = (self.beam@chunk).abs().square()
            for offset in range(len(powers)):
                ra_power.add_(powers[offset])
        rd = (10*torch.log10(rd_power+EPS)).T[self.keep]
        ra = (10*torch.log10(ra_power+EPS)).T[self.keep]
        detectors = {name: ca_cfar(10.**(data/10.)) for name, data in (('RD', rd), ('RA', ra))}
        return dict(RD=rd, RA=ra, ranges=self.ranges, vel=self.vel,
                    angle_axis_deg=self.angles, detectors=detectors)

    def label_bins(self, labels, post, kind):
        """Only metadata crosses to CUDA; finite physical gates are not clamped."""
        second_name = 'radial_velocity_mps' if kind == 'RD' else 'azimuth_deg'
        rows = [(float(row.get('range_m', math.nan)), float(row.get(second_name, math.nan)))
                for _, row in labels.iterrows()]
        values = self.tensor(rows).reshape(-1, 2)
        second = post['vel'] if kind == 'RD' else post['angle_axis_deg']
        rr = post['ranges']
        valid = torch.isfinite(values).all(dim=1) & (values[:, 0] >= rr.min()) & (values[:, 0] <= rr.max()) & (values[:, 1] >= second.min()) & (values[:, 1] <= second.max())
        values = values[valid]
        return torch.stack(((values[:, 0, None]-rr).abs().argmin(dim=1),
                            (values[:, 1, None]-second).abs().argmin(dim=1)), dim=1)

    @torch.no_grad()
    def evaluate(self, original, method, labels, detection_metrics=True):
        result = {}
        for kind, high in (('RD', 49152.), ('RA', 128.)):
            d0, d1 = original['detectors'][kind], method['detectors'][kind]
            bins = self.label_bins(labels, original, kind)
            best, outside = [], []
            for detector in (d0, d1):
                points = torch.nonzero(detector['base_peaks'], as_tuple=False)
                scores = detector['score'][points[:, 0], points[:, 1]]
                if len(bins) and len(points):
                    inside = ((bins[:, None]-points[None]).abs() <= 2).all(dim=2)
                    best.append(torch.where(inside, scores[None], 0.).max(dim=1).values)
                    outside.append(scores[~inside.any(dim=0)])
                else:
                    best.append(self.tensor(torch.zeros(len(bins), device=self.device)))
                    outside.append(scores)
            original_best, candidate_best = best
            eligible = original_best[:, None] > self.thresholds
            preserved = eligible & (candidate_best[:, None] > self.thresholds)
            supported, retained = eligible.sum(dim=0), preserved.sum(dim=0)
            unmatched0 = (outside[0][:, None] > self.thresholds).sum(dim=0)
            unmatched1 = (outside[1][:, None] > self.thresholds).sum(dim=0)
            low_eligible = original_best > self.thresholds[0]
            margin = ((torch.minimum(original_best, self.thresholds[0]*1.25)-candidate_best).clamp_min(0)/self.thresholds[0]*low_eligible).sum()
            db0 = torch.clamp(10*torch.log10(original_best.clamp_min(1.25)), 10*math.log10(1.25), 10*math.log10(high))
            db1 = torch.clamp(10*torch.log10(candidate_best.clamp_min(1.25)), 10*math.log10(1.25), 10*math.log10(high))
            regret = ((db0-db1).clamp_min(0)*low_eligible).sum()/COMMON_SPAN_DB
            log_span = math.log(high/1.25)
            curves = [((scores.clamp_min(1.25).log()-math.log(1.25)).clamp(0, log_span)/log_span).sum()
                      for scores in outside]
            p0, p1 = d0['power'], d1['power']
            scale = p0.mean()+EPS
            log0, log1 = torch.log1p(p0/scale), torch.log1p(p1/scale)
            values = dict(low_miss=(supported-retained)[0], six_miss=(supported-retained).sum(),
                          low_supported=supported[0], six_supported=supported.sum(),
                          low_preserved=retained[0], six_preserved=retained.sum(),
                          unmatched=unmatched1.sum(), original_unmatched=unmatched0.sum(),
                          unmatched_excess=(unmatched1.sum()-unmatched0.sum()).clamp_min(0),
                          margin_deficit=margin, curve_regret=regret,
                          unmatched_curve=curves[1], original_unmatched_curve=curves[0],
                          unmatched_curve_excess=(curves[1]-curves[0]).clamp_min(0),
                          map_error=(log1-log0).square().mean()/(log0.square().mean()+EPS),
                          supported_by_threshold=supported, preserved_by_threshold=retained,
                          missed_by_threshold=supported-retained,
                          unmatched_by_threshold=unmatched1, original_unmatched_by_threshold=unmatched0,
                          original_best_scores=original_best, reconstructed_best_scores=candidate_best)
            if detection_metrics:
                values.update(ranked_detection_metrics(d1, bins, self.thresholds))
            result.update({kind+'_'+name: value for name, value in values.items()})
        for key in ('low_miss', 'six_miss', 'low_supported', 'six_supported', 'low_preserved',
                    'six_preserved', 'unmatched', 'original_unmatched', 'margin_deficit', 'curve_regret', 'map_error',
                    'unmatched_curve', 'original_unmatched_curve', 'unmatched_curve_excess'):
            result[key] = result['RD_'+key]+result['RA_'+key]
        # Historical sum_positive semantics: preserve each map's signed excess.
        result['unmatched_excess'] = result['RD_unmatched_excess']+result['RA_unmatched_excess']
        return result
