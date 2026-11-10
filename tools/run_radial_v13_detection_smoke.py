"""V13.1 LMS smoke study: detection-preserving RADIal cube coding, no learning.

Detector: the unchanged RADIal RD receiver and cyclic CA-CFAR/NMS. Its
threshold alpha is calibrated on noise-only cubes passed through that exact
receiver (Neyman-Pearson: 1 and 0.1 false NMS peaks per map). Preservation is
set agreement with the ORIGINAL cube's detections (lost/new), plus score change,
background shift and label recall. RA-map CFAR is not an objective; angle
consistency of preserved detections is reported as a diagnostic.

V13.3 adds per-label margins (robust vs near-threshold labels), a perturbation
control that measures the receiver's own decision fragility, and a closed loop
that also repairs new near-threshold peaks. V13.2 codecs: ECDQ
with subtractive dither over the whole map, and a closed-loop
(analysis-by-synthesis) ECDQ: faithful coding on the detection support,
synthesized background elsewhere, and the encoder decodes its own stream and
grows/refines the CA-CFAR window of every robust disagreement. References:
lossless int16 ADC entropy coding, the V12.36 legacy codec, V13 noise_b2.
Rates are adaptive arithmetic code lengths incl. all side information.

Populations: 20 TRAIN + 60 VALIDATION cubes from the frozen recording roles.
Nothing is fitted on them; TEST stays locked until the design is frozen.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from radar_sim import arm_e_adaptive_gpu_run as common
from radar_sim import radial_adaptive_gpu_data as data
from radar_sim import radial_v13_detection_codec as codec
from radar_sim.radial_adaptive_gpu_physics import RadialGPUPhysics, cyclic_cfar, load_cube_gpu
from radar_sim.radial_dct_layout_gpu_physics import GlobalDCTCodecMixin


def sibling(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


partition = sibling('radial_frozen_partition')

REVISION = '13.3.0-closed-loop-ECDQ-robust-labels-perturbation-control'
SEED = 20261029
COUNTS = dict(train=20, validation=60)
H0_MAPS = 200
FALSE_PEAKS_PER_MAP = dict(a1=1., a01=.1)
MARGIN_DB = 1.
SUPPORT_RADIUS = 2
LEGACY = (.25, 8)
LABEL_RANGE_GATES = (2, 5, 10, 20)
# A car spans ~20 range bins at 0.2 m; the label is the box position. Label recall
# is scored within +-10 range bins and the native +-2 Doppler bins.
LABEL_GATE = (10, 2)
# Per-cube encoder-side verification tolerance (no labels): robust lost at most
# this fraction of the robust original detections and at most one robust new.
VERIFY_LOST_FRACTION, VERIFY_NEW = .005, 1
METHODS = (
    ('lossless_adc', dict(kind='lossless')), ('legacy_dct_r025_q8', dict(kind='legacy_dct')),
    # Control, not a codec: independent white noise 20 dB below the tile noise
    # level (0.04 dB background rise) shows which decisions are inherently fragile.
    ('perturb_m20db', dict(kind='perturb', level_db=-20.)),
    ('noise_b2', dict(kind='noise', beta=2.)),
    ('ecdq_b1', dict(kind='ecdq', beta=1.)), ('ecdq_b2', dict(kind='ecdq', beta=2.)),
    ('closed_g0_b2', dict(kind='closed', gamma=0., beta=2.)),
    # Also repair every strict lost detection, then also every strict new one; no labels used.
    ('closed_strict_g0_b2', dict(kind='closed', gamma=0., beta=2., strict=True)),
    ('closed_strict2_g0_b2', dict(kind='closed', gamma=0., beta=2., strict=True, strict_new=True)),
)
VERIFY_CANDIDATES = tuple(name for name, spec in METHODS if spec['kind'] in ('ecdq', 'closed'))
CONFIG = dict(counts=COUNTS, H0_maps=H0_MAPS, false_peaks_per_map=FALSE_PEAKS_PER_MAP,
    margin_db=MARGIN_DB, match_radius=codec.NMS_RADIUS, support_radius=SUPPORT_RADIUS,
    legacy=list(LEGACY), tile=[codec.TILE_DOPPLER, codec.TILE_RANGE], methods=[m for m, _ in METHODS]+['verified'],
    method_specs={m: s for m, s in METHODS}, label_range_gates=list(LABEL_RANGE_GATES), label_gate=list(LABEL_GATE),
    refinement_betas=list(codec.REFINE_BETAS), closed_loop_iterations=6,
    verification=f'cheapest of {list(VERIFY_CANDIDATES)} with robust lost <= {VERIFY_LOST_FRACTION} of robust '
                 f'original detections and robust new <= {VERIFY_NEW} at alpha_1; original vs decoded only; no labels',
    robust_detection='score > alpha + margin and NMS power margin >= margin; others reported as ambiguous',
    support='cells with original CFAR score >= alpha_1 - gamma dB (and unscored edge rows), dilated +-2; '
            'closed loop grows/refines the +-5 CA-CFAR window of each robust disagreement',
    rate='adaptive arithmetic (KT) code length incl. side information; Exp-Golomb length reported as bpcs_eg',
    codec_domain='unshifted ortho FFT2 over (chirp, fast time) per Rx')


class LegacyDCT(GlobalDCTCodecMixin, RadialGPUPhysics):
    pass


def select_rows(frame, count):
    """Recording-balanced, time-spread metadata selection; no detector outcomes."""
    if len(frame) < count or frame.record_name.duplicated().any():
        raise ValueError(f'Need {count} distinct eligible cubes; only {len(frame)} available.')
    groups = [g.sort_values(['frame_in_scene', 'record_name']).reset_index(drop=True)
              for _, g in frame.groupby('scene_name', sort=True)]
    quotas = [0]*len(groups)
    for _ in range(count):
        i = min((j for j, g in enumerate(groups) if quotas[j] < len(g)), key=lambda j: (quotas[j], j))
        quotas[i] += 1
    pieces = [g.iloc[[(2*i+1)*len(g)//(2*q) for i in range(q)]] for g, q in zip(groups, quotas) if q]
    result = pd.concat(pieces, ignore_index=True)
    if len(result) != count or result.record_name.duplicated().any():
        raise AssertionError('Metadata sampling repeated or dropped a cube.')
    return result


def source_hashes():
    files = [Path(__file__), Path(codec.__file__), Path(partition.__file__),
             common.ROOT/'radar_sim/radial_adaptive_gpu_physics.py', common.ROOT/'radar_sim/radial_dct_layout_gpu_physics.py']
    return {str(p.resolve().relative_to(common.ROOT)): common.sha(p) for p in files}


class Runtime:
    def __init__(self, dataset, assignment, device):
        self.dataset, self.device = dataset, torch.device(device)
        frames = partition.assign_frozen_partitions(dataset, assignment, bounded=True)
        self.rows = {role: select_rows(frames[i], COUNTS[role]).assign(role=role)
                     for i, role in enumerate(('train', 'validation'))}
        for role, frame in self.rows.items():
            partition.assert_frozen_role_access(dataset, frame.drop(columns='role'), role)
        if set(self.rows['train'].scene_name) & set(self.rows['validation'].scene_name):
            raise AssertionError('A recording crosses smoke partitions.')
        self.physics = RadialGPUPhysics(dataset.reference_root, device=self.device)
        self.legacy = LegacyDCT(dataset.reference_root, device=self.device)
        self.labels = data.scoped_labels(dataset, pd.concat(self.rows.values(), ignore_index=True))

    def roster(self):
        return pd.concat(self.rows.values(), ignore_index=True)

    def cube(self, row):
        frame = pd.DataFrame([row.drop(labels='role').to_dict()])
        partition.assert_frozen_role_access(self.dataset, frame, row.role)
        cube = load_cube_gpu(row, self.device)
        if not cube.is_cuda:
            raise RuntimeError('ADC decoding left CUDA.')
        return cube

    def receive(self, cube):
        return codec.rd_receive(cube, self.physics.range_window, self.physics.doppler_window, cyclic_cfar)

    def label_bins(self, record_name):
        labels = self.labels.loc[self.labels.record_name.eq(record_name)].reset_index(drop=True)
        post = dict(ranges=self.physics.ranges, vel=self.physics.vel)
        return self.physics.label_bins(labels, post, 'RD'), len(labels)

    def angles(self, spectrum, points):
        p = self.physics
        mimo = spectrum[points[:, 0][:, None], p.tx_indices[points[:, 1]], :].reshape(len(points), -1)
        response = (p.calibration@(mimo*p.virtual_window).T).abs()
        return p.angles[response.argmax(0)]


def context(runtime, out, assignment):
    return dict(revision=REVISION, source_root=str(common.ROOT.resolve()), output=str(out),
        dataset_root=str(runtime.dataset.root), reference_root=str(runtime.dataset.reference_root),
        device=str(runtime.device), gpu=torch.cuda.get_device_name(runtime.device), torch=torch.__version__,
        cuda_runtime=torch.version.cuda, split_audit=str(assignment.resolve()),
        split_audit_sha256=common.sha(assignment), source_hashes=source_hashes(),
        dataset_identity=data.dataset_identity(runtime.dataset),
        populations={role: frame.record_name.tolist() for role, frame in runtime.rows.items()},
        configuration=CONFIG, seed=SEED, learning=False, test_evaluated=False,
        objective='Preserve the original receiver RD detection set at calibrated alpha; minimise measured bits')


def calibrate(runtime):
    """Neyman-Pearson alpha for the exact receiver on noise-only cubes."""
    p = runtime.physics
    scores = codec.h0_scores(H0_MAPS, p.shape, p.range_window, p.doppler_window, cyclic_cfar, runtime.device, SEED)
    alphas = {name: codec.threshold_for_rate(scores, H0_MAPS, rate) for name, rate in FALSE_PEAKS_PER_MAP.items()}
    grid = torch.arange(0., 20.01, .1, dtype=torch.float64, device=scores.device)
    ordered = torch.sort(scores).values
    survival = (len(ordered)-torch.searchsorted(ordered, 10**(grid/10), right=True)).double()/H0_MAPS
    margin = 10**(MARGIN_DB/10)
    result = dict(alphas=alphas, alphas_db={k: 10*math.log10(v) for k, v in alphas.items()},
        H0_maps=H0_MAPS, H0_peaks_per_map=len(scores)/H0_MAPS,
        H0_robust_new_per_map={k: float((scores > v*margin).sum())/H0_MAPS for k, v in alphas.items()},
        legacy_factors_false_peaks_per_map={str(f): float((scores > f).sum())/H0_MAPS for f in p.thresholds.tolist()},
        noise_model='white complex Gaussian ADC, unit variance; CA-CFAR score is scale invariant',
        caveat='Real receiver noise may be coloured in range; real-map peak counts are reported separately')
    curve = pd.DataFrame(dict(threshold_db=grid.cpu().numpy(), H0_peaks_per_map=survival.cpu().numpy()))
    return result, curve


def legacy_encode(physics, cube):
    source = physics.prepare(cube)
    n = len(source['tiles'])
    decoded, meta = physics.encode(source, physics.tensor([LEGACY[0]]*n), physics.tensor([float(LEGACY[1])]*n))
    return decoded, int(meta['realized_bits'])


def run_method(runtime, spec, cube, original, alpha, seed):
    p = runtime.physics
    kind = spec['kind']
    if kind == 'fft14':
        decoded, meta = p.full_fft(cube, 14)
        return decoded, dict(total=int(meta['realized_bits']), accounting='legacy fixed-length')
    if kind == 'legacy_dct':
        decoded, bits = legacy_encode(runtime.legacy, cube)
        return decoded, dict(total=bits, accounting='legacy fixed-length (V12.36)')
    if kind == 'lossless':
        adc = torch.view_as_real(cube).to(torch.int64)
        if not torch.equal(adc.to(torch.float32), torch.view_as_real(cube)):
            raise ValueError('Lossless reference expects integer ADC samples.')
        bits = int(codec.arithmetic_bits(codec.to_tiles(adc, tuple(cube.shape))).sum())+64
        return cube, dict(total=bits, accounting='lossless int16 ADC; per-tile adaptive arithmetic code length')
    if kind == 'perturb':
        shape = tuple(cube.shape)
        tiles = codec.to_tiles(torch.view_as_real(codec.forward(cube)), shape)
        sigma = codec.sigma_value(codec.sigma_code(tiles))[:, None]*10**(spec['level_db']/20)
        noisy = tiles+codec._gaussian(tiles.shape, seed, cube.device)*sigma
        decoded = codec.inverse(torch.view_as_complex(codec.from_tiles(noisy, shape).contiguous())).to(cube.dtype)
        return decoded, dict(total=math.nan, accounting='control: original plus independent noise, not a codec')
    if kind == 'noise':
        stream = codec.encode(cube, beta_bg=spec['beta'], fill=False, seed=seed)
        bits = codec.stream_bits(stream)
        bits['accounting'] = 'V13 dead-zone quantizer; adaptive arithmetic code length plus side information'
        return codec.decode(stream, tuple(cube.shape), cube.dtype), bits
    if kind == 'closed':
        stream, decoded, history = codec.encode_closed_loop(cube, runtime.receive, original['detector'], alpha,
                                                            gamma_db=spec['gamma'], beta=spec['beta'], seed=seed,
                                                            strict_lost=spec.get('strict', False),
                                                            strict_new=spec.get('strict_new', False))
        bits = codec.ecdq_bits(stream)
        bits.update(accounting='closed-loop ECDQ; adaptive arithmetic code length plus all side information',
                    closed_loop_history=json.dumps(history), closed_loop_iterations=len(history)-1)
        return decoded, bits
    stream = codec.encode_ecdq(cube, beta=spec['beta'], seed=seed)
    bits = codec.ecdq_bits(stream)
    bits['accounting'] = 'ECDQ; adaptive arithmetic code length plus all side information'
    bits['predicted_background_rise_db'] = 10*math.log10(1+spec['beta']**2/12)
    return codec.decode_ecdq(stream, cube.dtype), bits


def score(runtime, row, name, cube, original, decoded, bits, alphas, bins, label_count, label_log=None, received=None):
    received = runtime.receive(decoded) if received is None else received
    error = (cube.to(torch.complex128)-decoded.to(torch.complex128)).abs().square().sum()
    energy = cube.to(torch.complex128).abs().square().sum()
    rd_error = (received['RD']-original['RD'])[original['detector']['valid']]
    record = dict(role=row.role, record_name=row.record_name, scene_name=row.scene_name, method=name,
        bits=bits['total'], bpcs=bits['total']/cube.numel(), accounting=bits['accounting'],
        bpcs_eg=bits['total_eg']/cube.numel() if bits.get('total_eg') else math.nan,
        payload_bits=bits.get('payload'), side_bits=bits.get('side'), header_bits=bits.get('header'),
        support_fraction=bits.get('support_fraction', 1.), refine_fraction=bits.get('refine_fraction', 0.),
        closed_loop_history=bits.get('closed_loop_history'), closed_loop_iterations=bits.get('closed_loop_iterations'),
        predicted_background_rise_db=bits.get('predicted_background_rise_db', math.nan),
        nmse=float(error/energy), rd_rmse_db=float(rd_error.square().mean().sqrt()))
    for key, alpha in alphas.items():
        metrics = codec.detection_metrics(original['detector'], received['detector'], alpha, MARGIN_DB)
        record.update({f'{key}_{k}': v for k, v in metrics.items()})
    record.update(codec.label_metrics(original['detector'], received['detector'], bins, alphas['a1'], *LABEL_GATE))
    record['labels_annotated'] = label_count
    if label_log is not None and len(bins):
        g0, g1 = codec.label_scores(original['detector'], received['detector'], bins, *LABEL_GATE)
        status = {k: v.tolist() for k, v in codec.label_status(original['detector'], received['detector'], bins,
                                                               alphas['a1'], *LABEL_GATE, MARGIN_DB).items()}
        a_db = 10*math.log10(alphas['a1'])
        for i, (s0, s1) in enumerate(zip(g0.tolist(), g1.tolist())):
            d0, d1 = 10*math.log10(max(s0, 1e-300)), 10*math.log10(max(s1, 1e-300))
            label_log.append(dict(role=row.role, record_name=row.record_name, method=name, label=i,
                range_bin=int(bins[i, 0]), doppler_bin=int(bins[i, 1]), original_score_db=d0, method_score_db=d1,
                original_margin_db=d0-a_db, detected_original=status['detected'][i],
                detected_method=status['detected_method'][i], preserved=status['preserved'][i],
                robust=status['robust'][i], robust_preserved=status['robust_preserved'][i]))
    kept = codec.detections(original['detector'], alphas['a1'])
    if len(kept):
        change = (runtime.angles(original['spectrum'], kept)-runtime.angles(received['spectrum'], kept)).abs()
        record.update(angle_change_deg_median=float(change.median()), angle_change_deg_p95=float(torch.quantile(change, .95)),
                      angle_change_over_1deg_fraction=float((change > 1).double().mean()))
    return record


def label_gate_recall(detector, bins, alpha):
    """Original-receiver label recall vs range gate (Doppler +-2). A labelled
    car spans ~20 range bins at 0.2 m, and the label is the box position, not
    necessarily the strongest scattering cell; this shows whether the +-2 gate
    or the detector explains the low label recall."""
    if not len(bins):
        return []
    scores = codec.peak_scores(detector)
    rows = []
    for gate in LABEL_RANGE_GATES:
        best = codec.neighbourhood_max(scores, gate, 2)[bins[:, 0], bins[:, 1]]
        rows.append(dict(range_gate_bins=gate, labels=len(bins), detected=int((best > alpha).sum()),
                         best_score_db_median=float(10*torch.log10(best.clamp_min(1e-300)).median())))
    return rows


def verify_pass(record):
    return (record['a1_lost_robust'] <= VERIFY_LOST_FRACTION*record['a1_original_robust']
            and record['a1_new_robust'] <= VERIFY_NEW)


def verified(records):
    """Cheapest candidate passing encoder-side verification (original vs
    decoded only, no labels); the encoder can run this because it holds the
    original cube and the receiver. Falls back to ecdq_b1 if none passes."""
    by_name = {r['method']: r for r in records}
    trials = [dict(method=name, bpcs=by_name[name]['bpcs'], passed=verify_pass(by_name[name]),
                   lost_robust=by_name[name]['a1_lost_robust'], new_robust=by_name[name]['a1_new_robust'])
              for name in VERIFY_CANDIDATES]
    passing = sorted((t['bpcs'], t['method']) for t in trials if t['passed'])
    choice = passing[0][1] if passing else 'ecdq_b1'
    return dict(by_name[choice], method='verified', verified_choice=choice, verified_fallback=not passing), trials


def summarize(frame):
    rows = []
    for (role, method), g in frame.groupby(['role', 'method'], sort=False):
        row = dict(role=role, method=method, cubes=len(g), bpcs_mean=g.bpcs.mean(), bpcs_max=g.bpcs.max(),
                   bpcs_eg_mean=g.bpcs_eg.mean(), compression_ratio=32/g.bpcs.mean(),
                   support_fraction_mean=g.support_fraction.mean(),
                   predicted_background_rise_db=g.predicted_background_rise_db.mean(),
                   nmse_mean=g.nmse.mean(), rd_rmse_db_mean=g.rd_rmse_db.mean())
        for key in FALSE_PEAKS_PER_MAP:
            total, stable = g[f'{key}_original'].sum(), g[f'{key}_original_robust'].sum()
            row.update({f'{key}_original': int(total), f'{key}_original_robust': int(stable),
                f'{key}_lost_fraction': g[f'{key}_lost'].sum()/max(total, 1),
                f'{key}_lost_robust_fraction': g[f'{key}_lost_robust'].sum()/max(stable, 1),
                f'{key}_new_per_map': g[f'{key}_new'].mean(), f'{key}_new_robust_per_map': g[f'{key}_new_robust'].mean(),
                f'{key}_score_change_db_mean': g[f'{key}_score_change_db_mean'].mean(),
                f'{key}_score_change_db_p05_worst': g[f'{key}_score_change_db_p05'].min(),
                f'{key}_background_shift_db_median': g[f'{key}_background_shift_db_median'].median()})
        row.update(labels=int(g.labels.sum()), labels_detected_original=int(g.labels_detected_original.sum()),
                   labels_robust=int(g.labels_robust.sum()), labels_lost=int(g.labels_lost.sum()),
                   labels_lost_robust=int(g.labels_lost_robust.sum()), labels_new=int(g.labels_new.sum()))
        if 'angle_change_deg_median' in g:
            row['angle_change_deg_median'] = g.angle_change_deg_median.median()
        rows.append(row)
    return pd.DataFrame(rows)


def recording_bootstrap(frame, draws=2000):
    """Recording-level percentile CIs for validation means (clusters, not cubes)."""
    rng = np.random.default_rng(SEED)
    rows = []
    for method, g in frame.groupby('method', sort=False):
        groups = [x for _, x in g.groupby('scene_name')]
        stats = []
        for _ in range(draws):
            sample = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
            stats.append((sample.bpcs.mean(), sample.a1_lost_robust.sum()/max(sample.a1_original_robust.sum(), 1),
                          sample.a1_new_robust.mean()))
        stats = np.asarray(stats)
        low, high = np.percentile(stats, 2.5, axis=0), np.percentile(stats, 97.5, axis=0)
        # Controls have no bitrate (NaN); JSON receipts forbid NaN, so write null.
        ci = lambda k: [None if not np.isfinite(low[k]) else float(low[k]), None if not np.isfinite(high[k]) else float(high[k])]
        rows.append(dict(method=method, recordings=len(groups), bpcs_ci=ci(0),
                         a1_lost_robust_fraction_ci=ci(1), a1_new_robust_per_map_ci=ci(2)))
    return rows


def gates(summary, calibration):
    """Pre-registered smoke criteria on VALIDATION; legacy rows are references."""
    val = summary.loc[summary.role.eq('validation')].set_index('method')
    legacy = float(val.loc['legacy_dct_r025_q8', 'bpcs_mean'])
    lossless = float(val.loc['lossless_adc', 'bpcs_mean'])
    floor = calibration['H0_robust_new_per_map']['a1']+1
    rows = {}
    for method, row in val.iterrows():
        if method.startswith(('legacy', 'fft14', 'lossless', 'perturb')):
            continue
        result = dict(bpcs_mean=float(row.bpcs_mean), ratio_vs_lossless=lossless/float(row.bpcs_mean),
            robust_lost_le_1pct=bool(row.a1_lost_robust_fraction <= .01),
            robust_new_le_H0_floor_plus_1=bool(row.a1_new_robust_per_map <= floor),
            no_robust_label_lost=bool(row.labels_lost_robust == 0),
            bitrate_below_legacy=bool(row.bpcs_mean < legacy),
            strict_labels_lost=int(row.labels_lost), strict_lost_fraction=float(row.a1_lost_fraction))
        result['passed'] = all(v for k, v in result.items()
                               if k not in ('bpcs_mean', 'ratio_vs_lossless', 'strict_labels_lost', 'strict_lost_fraction'))
        rows[method] = result
    passing = sorted((v['bpcs_mean'], k) for k, v in rows.items() if v['passed'])
    control = val.loc['perturb_m20db']
    fragility = dict(strict_lost_fraction=float(control.a1_lost_fraction), robust_lost_fraction=float(control.a1_lost_robust_fraction),
                     strict_new_per_map=float(control.a1_new_per_map), labels_lost=int(control.labels_lost),
                     labels_lost_robust=int(control.labels_lost_robust),
                     meaning='decision changes caused by independent noise 20 dB below the receiver noise')
    return dict(legacy_dct_bpcs=legacy, lossless_adc_bpcs=lossless, H0_robust_new_floor_per_map=floor,
                receiver_fragility_control=fragility, methods=rows,
                scientific_gate_passed=bool(passing), best_passing_method=passing[0][1] if passing else None,
                best_passing_bpcs=passing[0][0] if passing else None)


def make_figures(out, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    val = summary.loc[summary.role.eq('validation')]
    folder = out/'figures'
    folder.mkdir()
    with plt.rc_context({'font.family': 'serif', 'font.size': 8}):
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), layout='constrained')
        for ax, key, label in ((axes[0], 'a1_lost_robust_fraction', 'Robust lost fraction (alpha_1)'),
                               (axes[1], 'a1_new_robust_per_map', 'Robust new peaks per map (alpha_1)')):
            ax.scatter(val.bpcs_mean, val[key].clip(lower=1e-4), s=14, color='#0072B2')
            for _, r in val.iterrows():
                ax.annotate(r.method, (r.bpcs_mean, max(r[key], 1e-4)), fontsize=6, xytext=(3, 2), textcoords='offset points')
            ax.set(xscale='log', yscale='log', xlabel='Bits per complex sample', ylabel=label)
            ax.grid(alpha=.25, which='both')
        fig.savefig(folder/'rate_vs_detection_agreement.pdf')
        fig.savefig(folder/'rate_vs_detection_agreement.png', dpi=250)
        plt.close(fig)


def run_check(runtime, out, expected):
    path = Path(str(out)+'.CHECK.json')
    if path.exists():
        old = common.read_json(path)
        if old.get('passed') is True and old.get('context') == expected:
            print('Existing identity-bound V13 check: PASS.', flush=True)
            return
        raise ValueError('Existing check identity differs. Preserve it; use a fresh deployment.')
    p = runtime.physics
    rows, checks = runtime.rows['train'].iloc[:2], []
    scores = codec.h0_scores(4, p.shape, p.range_window, p.doppler_window, cyclic_cfar, runtime.device, SEED)
    alpha = codec.threshold_for_rate(scores, 4, 1.)
    for _, row in rows.iterrows():
        cube = runtime.cube(row)
        maps, mine = p.maps(cube), runtime.receive(cube)
        torch.testing.assert_close(mine['RD'], maps['RD'], rtol=0, atol=1e-9)
        if not torch.equal(mine['detector']['base_peaks'], maps['detectors']['RD']['base_peaks']):
            raise AssertionError('V13 RD receiver differs from the native RADIal receiver.')
        roundtrip = (codec.inverse(codec.forward(cube))-cube.to(torch.complex128)).abs().max()/cube.abs().max()
        if float(roundtrip) > 1e-12:
            raise AssertionError('Orthonormal FFT2 codec domain does not invert.')
        fine = codec.decode_ecdq(codec.encode_ecdq(cube, beta=.02, seed=SEED), cube.dtype)
        received = runtime.receive(fine)
        valid = mine['detector']['valid']
        change = (received['RD']-mine['RD'])[valid].abs()
        map_change, map_change_p99 = float(change.max()), float(torch.quantile(change, .99))
        agreement = codec.detection_metrics(mine['detector'], received['detector'], alpha)
        adjacent = codec.detection_metrics(mine['detector'], received['detector'], alpha, match=1)
        flips = codec.disagreements(mine['detector'], received['detector'], alpha)
        print('Near-lossless:', json.dumps(dict(max_RD_change_db=map_change, p99_RD_change_db=map_change_p99,
                                                nms_radius_match=agreement, one_bin_match=adjacent)), flush=True)
        print('Near-lossless disagreements:', json.dumps(flips), flush=True)
        # Engineering gate only: the codec must leave the map, the detection scores
        # and the CFAR background essentially unchanged. Set agreement is a measured
        # outcome (the receiver's NMS/threshold decisions are discontinuous), so it
        # is recorded here and evaluated in the run, not asserted.
        if (map_change_p99 > .1 or agreement['score_change_db_p05'] < -.1
                or abs(agreement['background_shift_db_median']) > .01):
            raise AssertionError('Near-lossless V13 codec changed the RD map, detection scores or CFAR background.')
        support = codec.detection_support(mine['detector'], alpha, 2., SUPPORT_RADIUS)
        stream = codec.encode_ecdq(cube, beta=1.5, support=support, gamma_db=2., seed=SEED)
        first, second = codec.decode_ecdq(stream), codec.decode_ecdq(stream)
        if not torch.equal(first, second):
            raise AssertionError('Decoder is not deterministic.')
        keep = codec.cells_to_tiles(support, p.shape)
        arithmetic = codec.arithmetic_bits(stream['index'], keep)
        eg, k_run, k_mag = codec.payload_bits(stream['index'], keep)
        busiest = int(keep.sum(1).argmax())
        coded = []
        for t in sorted({busiest, int(keep.sum(1).argmin()), len(eg)//2}):
            symbols = stream['index'][t][keep[t]].tolist()
            if not symbols:
                continue
            written = codec.write_tile(symbols, int(k_run[t]), int(k_mag[t]))
            if len(written) != int(eg[t]) or codec.read_tile(written, len(symbols)) != symbols:
                raise AssertionError('Exp-Golomb length/decodability check failed.')
            real = codec.write_arithmetic_tile(symbols)
            if codec.read_arithmetic_tile(real, len(symbols)) != symbols or len(real) > int(arithmetic[t])+8:
                raise AssertionError('Arithmetic coder length/decodability check failed.')
            coded.append(dict(tile=t, symbols=len(symbols), arithmetic_real_bits=len(real),
                              arithmetic_ideal_bits=int(arithmetic[t]), exp_golomb_bits=int(eg[t])))
        print('Arithmetic coder:', json.dumps(coded), flush=True)
        checks.append(dict(record_name=row.record_name, rd_max_abs_db_error=float((mine['RD']-maps['RD']).abs().max()),
            fft2_roundtrip=float(roundtrip), near_lossless_nmse=float((fine-cube).abs().square().sum()/cube.abs().square().sum()),
            near_lossless_max_RD_change_db=map_change, near_lossless_p99_RD_change_db=map_change_p99, near_lossless_strict=agreement, near_lossless_one_bin_match=adjacent,
            near_lossless_disagreements=flips, support_g2_fraction=float(support.double().mean()),
            support_g2_b1p5_bpcs=codec.ecdq_bits(stream)['total']/cube.numel(), coded_tiles=coded))
        print('V13 receiver/codec/bitstream preflight PASS:', row.record_name, flush=True)
    labels = [dict(record_name=r.record_name, role=r.role, annotations=runtime.label_bins(r.record_name)[1],
                   mapped_to_RD=len(runtime.label_bins(r.record_name)[0])) for _, r in runtime.roster().iterrows()]
    common.csv(Path(str(out)+'.LABEL_AUDIT.csv'), labels)
    common.write_json(path, dict(completed=True, passed=True, context=expected, checks=checks,
        quick_H0_alpha=alpha, training_started=False, test_evaluated=False))
    print('V13 CUDA check complete. Scientific success is decided only by the run.', flush=True)


def run(runtime, out, expected, assignment):
    checked = common.read_json(Path(str(out)+'.CHECK.json'))
    if checked.get('context') != expected or checked.get('passed') is not True:
        raise ValueError('Run the matching CUDA check first.')
    if out.exists():
        raise FileExistsError('Preserve partial/completed results; never overwrite this study.')
    with common.exclusive(out):
        out.mkdir(parents=True)
        common.write_json(out/'manifest.json', expected)
        runtime.roster().to_csv(out/'POPULATIONS.csv', index=False)
        start = time.time()
        calibration, curve = calibrate(runtime)
        common.write_json(out/'H0_CALIBRATION.json', calibration)
        curve.to_csv(out/'H0_SURVIVAL.csv', index=False)
        alphas = calibration['alphas']
        print('H0 calibration:', json.dumps(calibration['alphas_db']), f'{time.time()-start:.0f}s', flush=True)
        records, ladders, peaks, label_gates, label_log = [], [], [], [], []
        for position, (_, row) in enumerate(runtime.roster().iterrows()):
            cube = runtime.cube(row)
            original = runtime.receive(cube)
            bins, label_count = runtime.label_bins(row.record_name)
            seed = SEED+position
            peaks.append(dict(role=row.role, record_name=row.record_name,
                ambiguous_a1=len(codec.detections(original['detector'], alphas['a1']))
                    -int(codec.stable_peaks(original['detector'], alphas['a1'], 10**(MARGIN_DB/10)).sum()),
                **{f'support_g{g:g}_fraction': float(codec.detection_support(original['detector'], alphas['a1'], g,
                                                                             SUPPORT_RADIUS).double().mean()) for g in (1., 2., 3.)},
                **{f'{k}_original_peaks': len(codec.detections(original['detector'], a)) for k, a in alphas.items()},
                **{f'legacy_{f}_peaks': len(codec.detections(original['detector'], f)) for f in runtime.physics.thresholds.tolist()}))
            label_gates.extend(dict(role=row.role, record_name=row.record_name, **gate)
                               for gate in label_gate_recall(original['detector'], bins, alphas['a1']))
            cube_records = []
            for name, spec in METHODS:
                decoded, bits = run_method(runtime, spec, cube, original, alphas['a1'], seed)
                cube_records.append(score(runtime, row, name, cube, original, decoded, bits, alphas, bins, label_count, label_log))
                del decoded
            record, trials = verified(cube_records)
            records.extend(cube_records+[record])
            ladders.extend(dict(trial, role=row.role, record_name=row.record_name) for trial in trials)
            print(f'[{position+1}/{len(runtime.roster())}] {row.role} {row.record_name}: '
                  f'peaks(a1)={peaks[-1]["a1_original_peaks"]} support(g2)={peaks[-1]["support_g2_fraction"]:.2f} '
                  f'verified={record["verified_choice"]} bpcs={record["bpcs"]:.3f} {time.time()-start:.0f}s', flush=True)
            del cube, original
        frame = pd.DataFrame(records)
        frame.to_csv(out/'RESULTS_BY_CUBE.csv', index=False)
        pd.DataFrame(ladders).to_csv(out/'VERIFIED_LADDER.csv', index=False)
        pd.DataFrame(peaks).to_csv(out/'REAL_MAP_PEAKS.csv', index=False)
        pd.DataFrame(label_log).to_csv(out/'LABELS.csv', index=False)
        gates_frame = pd.DataFrame(label_gates)
        gates_frame.to_csv(out/'LABEL_GATE_DIAGNOSTIC.csv', index=False)
        if len(gates_frame):
            gates_frame.groupby(['role', 'range_gate_bins'])[['labels', 'detected']].sum().reset_index().to_csv(
                out/'LABEL_GATE_SUMMARY.csv', index=False)
        summary = summarize(frame)
        summary.to_csv(out/'SUMMARY.csv', index=False)
        common.write_json(out/'VALIDATION_RECORDING_CI.json',
                          recording_bootstrap(frame.loc[frame.role.eq('validation')]))
        result = gates(summary, calibration)
        common.write_json(out/'GATES.json', result)
        make_figures(out, summary)
        runtime.dataset.assert_unchanged()
        common.write_json(out/'SMOKE_COMPLETED.json', dict(completed=True, passed=True, context=expected,
            seed=SEED, test_evaluated=False, test_used_for_selection=False, full_training_started=False,
            scientific_gate_passed=result['scientific_gate_passed'], best_passing_method=result['best_passing_method'],
            alphas_db=calibration['alphas_db'], runtime_seconds=time.time()-start))
    print('V13 smoke COMPLETE; inspect GATES.json and SUMMARY.csv:', out, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('check', 'run'), required=True)
    for key in ('dataset-root', 'reference-root', 'split-audit', 'out'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    if torch.device(args.device).type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('CUDA is required. There is no CPU fallback.')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dataset = data.prepare_dataset(args.dataset_root, args.reference_root)
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    runtime = Runtime(dataset, args.split_audit, args.device)
    # JSON round trip: the check receipt stores lists, so tuples would never compare equal.
    expected = json.loads(json.dumps(context(runtime, out, args.split_audit)))
    print(json.dumps({k: expected[k] for k in ('revision', 'gpu', 'configuration')}, indent=2), flush=True)
    if args.stage == 'check':
        run_check(runtime, out, expected)
    else:
        run(runtime, out, expected, args.split_audit)


if __name__ == '__main__':
    main()
