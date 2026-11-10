"""V20: the proposed codec at fixed rates from 64x to 16x, rate-matched to AdaRadar
(1.45 bpcs), tuned on TRAIN/VALIDATION, with FFTRadNet fine-tuned per codec.

1. FIXED-RATE OPERATION of the support-coded variant (A: CA-CFAR support coded,
   background synthesized; V17). Per cube the encoder takes the finest
   configuration whose complete rate (closed-loop refinements, min(learned, KT)
   code, all side information, flag bit) fits the budget. Configurations, finest
   first (local probe on 4 VALIDATION cubes: with the full closed loop the rate
   is lowest near beta = 4, ~1.05 bpcs; coarser steps need more repair):
     a. beta = 2^(k/8) from 0.5 to 4 with the full closed loop (6 rounds);
     b. beta 4 with 5, 4, ..., 0 verification rounds;
     c. beta 4 to 32 without verification.
   Below the floor of (a) not every detection can be kept; the sweep shows where
   labels start to go. Budgets 0.5, 0.667, 1, 1.45, 2 bpcs = 64x, 48x, 32x,
   22.1x, 16x of the raw 32 bpcs; R* = 1.45 is AdaRadar's complete rate at its
   paper point (V18 TEST: 1.454).

2. TUNING, TRAIN and VALIDATION only.
   a. The learned entropy model is fine-tuned from the V17 checkpoint on TRAIN
      streams drawn over the whole configuration ladder; epoch chosen on
      VALIDATION (epoch 0 is the V17 model). Saved bits buy a finer step.
   b. The support margin gamma in {0, 3, 6, 10} dB, at R*, on every 4th
      VALIDATION cube. Rule fixed before the run: among the gammas losing <= 1%
      robust CA-CFAR detections and no robust label, the highest FFTRadNet F1
      (ties: smaller gamma); if none qualifies, the fewest robust losses.

3. FFTRADNET FINE-TUNING (radar_sim/radial_v20_fftradnet_finetune.py) on every
   3rd TRAIN cube, epoch chosen on every 3rd VALIDATION cube, three arms:
     proposed  the proposed codec at R*;
     adaradar  AdaRadar 4-bit with a random pruning ratio in [1, 40] per sample
               (its 4-bit training command: Train_w_compress.py --cr_random);
     original  the uncompressed input, i.e. what fine-tuning alone gives.

4. TEST, unlocked once: the proposed codec at the five budgets; C at beta* 1.5
   (anchor with the V17 entropy model: reproduces V17/V19); AdaRadar static at
   r = 50, 25, 12.57, 8 and adaptive. The official FFTRadNet sees every method;
   each fine-tuned network sees the original and the R* codecs; AdaRadar's
   adaptive control is also run with its fine-tuned network. Every TEST stream
   at R* is written to a file and decoded from it.

PRE-REGISTERED (TEST): P1 the proposed codec at R* loses <= 1% robust CA-CFAR
detections and no robust label; P2 its FFTRadNet F1 (official network) is at
least AdaRadar's at r = 12.57; P3 the same with each codec's fine-tuned network.
The label-loss onset along the sweep is reported, not tested.

Frozen V17 system otherwise: split, receivers, metrics, statistics.
STAGES: count (metadata-only inventory), check (CUDA pre-flight), run.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from radar_sim import radial_fftradnet_eval as fftradnet
from radar_sim import radial_v18_adaradar as adaradar
from radar_sim import radial_v20_fftradnet_finetune as finetune


def sibling(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V18 = sibling('run_radial_v18_followup')  # AdaRadar (static and adaptive), spectrum-domain receiver
V17 = V18.V17  # the one frozen V17 system V18 also uses: runtime, receivers, scoring, files, perception, statistics
V13, V14, common, data, codec, entropy = V17.V13, V17.V14, V17.common, V17.data, V17.codec, V17.entropy

REVISION = '20.0.1-fixed-rate-sweep-rate-matched-tuned-finetuned'
PACKAGE = Path(__file__).resolve().parent/'v20_rate_matched_gpu'
V17.PACKAGE = V18.PACKAGE = PACKAGE  # FFTRadNet files, the frozen split and the V17 model are copied into the V20 package
FROZEN_MODEL = V18.FROZEN_MODEL
SEED = V17.SEED

STEPS_PER_OCTAVE, LOOP_ROUNDS = 8, 6
K_FINEST, K_LOOP, K_COARSEST = -8, 16, 40  # beta 0.5, 4, 32
LADDER = tuple([(k, LOOP_ROUNDS) for k in range(K_FINEST, K_LOOP+1)]
               + [(K_LOOP, n) for n in range(LOOP_ROUNDS-1, -1, -1)]
               + [(k, 0) for k in range(K_LOOP+1, K_COARSEST+1)])
START = LADDER.index((0, LOOP_ROUNDS))  # beta 1: first guess of a recording's search


def beta_of(k):
    return 2.**(k/STEPS_PER_OCTAVE)


R_STAR = 1.45
BUDGETS = {'fixed_0p50': .5, 'fixed_0p67': 2/3, 'fixed_1p00': 1., 'fixed_1p45': R_STAR, 'fixed_2p00': 2.}
HEADLINE, ADARADAR_HEADLINE = 'fixed_1p45', 'adaradar_r12p57_s4'
GAMMAS = (0., 3., 6., 10.)
C_ANCHOR = (('closed_full_b1p5', dict(kind='closed', gamma=300., beta=1.5)),)
ADARADAR = (*((f'adaradar_r{str(r).replace(".", "p")}_s4', dict(kind='adaradar', prune_ratio=r, bits=4)) for r in (50, 25, 12.57, 8)),
            ('adaradar_adaptive_s4', dict(kind='adaradar_adaptive', **V18.ADAPTIVE_SETTINGS)),
            ('adaradar_adaptive_s4_ft', dict(kind='adaradar_adaptive_ft', **V18.ADAPTIVE_SETTINGS)))
ENTROPY_TUNING = dict(train_every_nth_cube=4, selection_every_nth_validation_cube=6, samples_per_stream=4096,
                      epochs=8, lr=5e-4, batch=32768, start='V17 checkpoint')
GAMMA_SELECTION = dict(every_nth_validation_cube=4, robust_lost_limit=V17.GATE['robust_lost_fraction'],
                       rule='among gammas with robust CA-CFAR lost <= 1% and no robust label lost: highest FFTRadNet F1 '
                            '(ties: smaller gamma); if none qualifies: fewest robust losses')
FINETUNE = dict(train_every_nth_cube=3, validation_every_nth_cube=3, epochs=5, lr=1e-5, micro_batch=finetune.MICRO_BATCH,
                accumulate=finetune.ACCUMULATE, adaradar_ratio_range=(1., 40.), adaradar_bits=4,
                free_space_target='official network on the ORIGINAL cube, sigmoid >= 0.5',
                detection_target='official ra_encoder of every evaluable label (easy and difficult)')
ARMS = ('proposed', 'adaradar', 'original')
# Networks that see each TEST input besides the official one; stored as '<method>@ft_<arm>'.
FINETUNED_EVALUATION = {'original': ARMS, HEADLINE: ('proposed', 'original'), ADARADAR_HEADLINE: ('adaradar', 'original'),
                        'adaradar_adaptive_s4_ft': ('adaradar',)}
CACHE_FREE_GB = 25  # float16 network inputs of the fine-tuning subsets (~18 GB) plus a reserve
V17.BITSTREAM_POINTS = (HEADLINE,)
PREDICTIONS = dict(P1='fixed_1p45 on TEST: robust CA-CFAR lost <= 1% and no robust label lost',
                   P2='FFTRadNet F1 (official network): fixed_1p45 >= adaradar_r12p57_s4',
                   P3='FFTRadNet F1 (fine-tuned networks): fixed_1p45@ft_proposed >= adaradar_r12p57_s4@ft_adaradar')
CONFIG = dict(ladder=[dict(beta=beta_of(k), loop_rounds=n) for k, n in LADDER], budgets_bpcs=BUDGETS, r_star=R_STAR,
              gammas=list(GAMMAS), c_anchor=dict(C_ANCHOR), adaradar=dict(ADARADAR), entropy_tuning=ENTROPY_TUNING,
              gamma_selection=GAMMA_SELECTION, fftradnet_finetune=FINETUNE, arms=list(ARMS),
              finetuned_evaluation={k: list(v) for k, v in FINETUNED_EVALUATION.items()}, predictions=PREDICTIONS,
              frozen_model=FROZEN_MODEL, bitstream_points=list(V17.BITSTREAM_POINTS), H0_maps=V13.H0_MAPS,
              margin_db=V13.MARGIN_DB, label_gate=list(V13.LABEL_GATE), sync_tolerance_us=data.TOLERANCE_US,
              test_policy='TEST unlocked once, after the entropy model, gamma and the fine-tuned networks are frozen')


# ---------------------------------------------------------------------------
# Fixed-rate operation
# ---------------------------------------------------------------------------
def first_feasible(feasible, start, count):
    """Smallest index with feasible(index), assuming feasibility only switches on along the ladder:
    galloping from ``start``, then bisection. None if even the last index does not fit."""
    if feasible(start):
        hi, lo, step = start, start-1, 1
        while lo >= 0 and feasible(lo):
            hi, step = lo, 2*step
            lo = hi-step
        lo = max(lo, -1)
    else:
        lo, hi, step = start, start+1, 1
        while hi < count and not feasible(hi):
            lo, step = hi, 2*step
            hi = lo+step
        if hi >= count:
            hi = count-1
            if not feasible(hi):
                return None
    while hi-lo > 1:
        mid = (lo+hi)//2
        if feasible(mid):
            hi = mid
        else:
            lo = mid
    return hi


class FixedRate:
    """Per-cube rate control over LADDER; each search starts from the previous cube's choice in the recording."""

    def __init__(self, model):
        self.model, self.start = model, {}

    def encode(self, runtime, row, cube, original, alpha, budget, gamma, key):
        numel, tried = cube.numel(), {}

        def feasible(i):
            if i not in tried:
                k, rounds = LADDER[i]
                stream, decoded, history = codec.encode_closed_loop(
                    cube, runtime.receive, original['detector'], alpha, gamma_db=gamma, beta=beta_of(k),
                    seed=runtime.seed(row), iterations=rounds)
                bits = codec.ecdq_bits(stream)
                learned = entropy.learned_payload_bits(self.model, entropy.compact(stream))
                total = V14.rate_fields(bits, learned, numel)['bits']
                tried[i] = dict(stream=stream, decoded=decoded, history=history, bits=bits, learned=learned,
                                fits=total <= budget*numel)
            return tried[i]['fits']

        state = (key, row.scene_name)
        index = first_feasible(feasible, self.start.get(state, START), len(LADDER))
        within = index is not None
        index = len(LADDER)-1 if index is None else index
        feasible(index)
        self.start[state] = index
        chosen, (k, rounds) = tried[index], LADDER[index]
        bits = dict(chosen['bits'], accounting='fixed-rate A: finest ladder configuration within the budget',
                    closed_loop_history=json.dumps(chosen['history']), closed_loop_iterations=len(chosen['history'])-1)
        control = dict(budget_bpcs=budget, gamma=gamma, ladder_index=index, beta=beta_of(k), loop_rounds=rounds,
                       loop_relaxed=rounds < LOOP_ROUNDS, within_budget=within, attempts=len(tried))
        return chosen['stream'], chosen['decoded'], bits, chosen['learned'], control


class Runtime(V18.Runtime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptive_ft = V18.AdaptiveRate(V18.ADAPTIVE_SETTINGS)  # AdaRadar's control driven by its fine-tuned network
        self.fixed_rate = None


# ---------------------------------------------------------------------------
# Perception with fine-tuned networks
# ---------------------------------------------------------------------------
class Network:
    """One FFTRadNet on one spectrum: (decoded rows, free-space map), as V17.Perception.run."""

    def __init__(self, net, config):
        self.net, self.config = net, config

    def run(self, spectrum):
        detection, segmentation = fftradnet.predict(self.net, fftradnet.network_input(spectrum, self.config)[None])
        return fftradnet.decode(detection[0], self.config), segmentation[0].cpu().numpy()


class Perception(V17.Perception):
    """V17 perception (official network on every method) plus the fine-tuned networks of FINETUNED_EVALUATION."""

    def __init__(self, runtime, finetuned=None):
        super().__init__(runtime)
        self.finetuned = {arm: Network(net, self.config) for arm, net in (finetuned or {}).items()}

    def store(self, key, prediction, segmentation):
        if self.frame['labelled']:
            self.frame['counts'][key] = fftradnet.frame_counts(prediction, self.frame['labels'])
        self.frame['agreement'][key] = fftradnet.agreement(self.prediction, prediction)
        self.frame['freespace'][key] = fftradnet.freespace_iou(self.segmentation, segmentation)

    def add(self, name, received):
        self.store(name, *self.run(received['spectrum']))
        self.add_finetuned(name, received)

    def add_finetuned(self, name, received):
        for arm in FINETUNED_EVALUATION.get(name, ()):
            if arm in self.finetuned:
                self.store(f'{name}@ft_{arm}', *self.finetuned[arm].run(received['spectrum']))


def labels_of(runtime, row, easy):
    """[N, 2] (range m, azimuth deg) of a frame; zero rows for an explicit negative; None if not annotated."""
    labels = runtime.labels.loc[runtime.labels.record_name.eq(row.record_name)]
    if easy:
        labels = labels.loc[labels.Difficult.eq(0)]
    if labels.empty:
        return np.zeros((0, 2)) if bool(getattr(row, 'explicit_negative', False)) else None
    return labels[['range_m', 'azimuth_deg']].to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# One TEST cube
# ---------------------------------------------------------------------------
def evaluate_cube(runtime, row, methods, alphas, models, perception, label_log, control_log, bitstreams=None):
    """Every method on one cube, scored against the receivers' output on the ORIGINAL cube (V17/V18 scoring)."""
    device = runtime.device
    cube = runtime.cube(row)
    original = runtime.receive(cube)
    original_os = V17.receivers.cyclic_os_cfar(original['detector']['power'])
    bins, label_count = runtime.label_bins(row.record_name)
    perception.start(row, original)
    perception.add_finetuned('original', original)
    energy = original['spectrum'].abs().square().sum()
    stats = perception.config['dataset']['statistics']
    records = []
    for name, spec in methods:
        kind, stream, learned, control = spec['kind'], None, None, None
        if kind.startswith('adaradar'):
            if kind == 'adaradar':
                (spectrum, bits), ms = V17.timed(device, adaradar.compress, original['spectrum'], spec['prune_ratio'],
                                                 spec['bits'], stats['input_mean'], stats['input_std'])
                ratio = spec['prune_ratio']
            else:
                controller, network = ((runtime.adaptive, perception) if kind == 'adaradar_adaptive'
                                       else (runtime.adaptive_ft, perception.finetuned['adaradar']))
                (spectrum, bits, ratio), ms = V17.timed(device, controller.code, row, original['spectrum'], stats, network)
            received, decoded = runtime.receive_spectrum(spectrum), cube
            times = dict(codec_ms=ms, encode_ms=math.nan, decode_ms=math.nan, adaradar_comp_ratio=ratio,
                         bpcs_nonzero_only=bits['total_nonzero_only']/cube.numel(), index_bpcs=bits['index_bits']/cube.numel(),
                         kept_fraction=bits['kept_fraction'], nonzero_fraction=bits['nonzero_fraction'])
        elif kind == 'fixed':
            (stream, decoded, bits, learned, control), encode_ms = V17.timed(
                device, runtime.fixed_rate.encode, runtime, row, cube, original, alphas['ca']['a1'], spec['budget'],
                spec['gamma'], name)
            again, decode_ms = V17.timed(device, codec.decode_ecdq, stream, cube.dtype)
            if not torch.equal(again, decoded):
                raise AssertionError('The decoder differs from the decode the rate control verified.')
            received = runtime.receive(decoded)
            times = dict(encode_ms=encode_ms, decode_ms=decode_ms, codec_ms=encode_ms+decode_ms)
            control_log.append(dict(record_name=row.record_name, scene_name=row.scene_name, method=name, **control))
        else:
            decoded, bits, stream, times = V17.encode_decode(runtime, spec, cube, original, alphas['ca']['a1'], runtime.seed(row))
            received = runtime.receive(decoded)
        record = V13.score(runtime, row, name, cube, original, decoded, bits, alphas['ca'], bins, label_count, label_log, received)
        record.update(times, spectrum_nmse=float((received['spectrum']-original['spectrum']).abs().square().sum()/energy))
        if kind.startswith('adaradar'):
            record['nmse'] = math.nan  # not an ADC-domain codec; its distortion is spectrum_nmse
        method_os = V17.receivers.cyclic_os_cfar(received['detector']['power'])
        for key, alpha in alphas['os'].items():
            metrics = codec.detection_metrics(original_os, method_os, alpha, V13.MARGIN_DB)
            record.update({f'os_{key}_{k}': v for k, v in metrics.items()})
        if kind == 'closed':
            learned, record['entropy_model_ms'] = V17.timed(device, entropy.learned_payload_bits, models['v17'], entropy.compact(stream))
            record.update(V14.rate_fields(bits, learned, cube.numel()))
        elif kind == 'fixed':
            record.update(V14.rate_fields(bits, learned, cube.numel()), entropy_model_ms=math.nan,
                          **{f'rc_{k}': v for k, v in control.items()})
            if bitstreams is not None and name in V17.BITSTREAM_POINTS:
                bitstreams.submit(row, name, stream, decoded, record)
        perception.add(name, received)
        records.append(record)
        del decoded, received, stream
    return records


# ---------------------------------------------------------------------------
# 2a. Entropy model on the ladder
# ---------------------------------------------------------------------------
def ladder_symbols(runtime, rows, alpha, per_stream, generator, rng, log, label):
    """One stream per cube and gamma at a ladder configuration drawn uniformly; random coded components."""
    features, targets = [], []
    for i, (_, row) in enumerate(rows.iterrows()):
        cube = runtime.cube(row)
        original = runtime.receive(cube)
        for gamma in GAMMAS:
            k, rounds = LADDER[int(rng.integers(len(LADDER)))]
            stream, _, _ = codec.encode_closed_loop(cube, runtime.receive, original['detector'], alpha, gamma_db=gamma,
                                                    beta=beta_of(k), seed=runtime.seed(row), iterations=rounds)
            f, t = entropy.sample(entropy.compact(stream), per_stream, generator)
            features.append(f)
            targets.append(t)
        if (i+1) % 50 == 0 or i+1 == len(rows):
            log(f'{label} {i+1}/{len(rows)}')
    return torch.cat(features), torch.cat(targets)


def tune_entropy_model(runtime, alpha, log):
    p = ENTROPY_TUNING
    generator = torch.Generator(device=runtime.device).manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    rows_t, targets_t = ladder_symbols(runtime, runtime.roster('train').iloc[::p['train_every_nth_cube']], alpha,
                                       p['samples_per_stream'], generator, rng, log, 'entropy model TRAIN')
    selection = runtime.roster('validation').iloc[::p['selection_every_nth_validation_cube']]
    rows_v, targets_v = ladder_symbols(runtime, selection, alpha, p['samples_per_stream'], generator, rng, log,
                                       'entropy model VALIDATION selection')
    torch.manual_seed(SEED)
    model = V18.load_frozen_model(runtime.device)
    history, best = entropy.train(model, rows_t, targets_t, rows_v, targets_v, epochs=p['epochs'], batch=p['batch'],
                                  lr=p['lr'], seed=SEED, log=lambda r: log('entropy epoch', json.dumps(r)))
    return model, history, best, len(selection)


# ---------------------------------------------------------------------------
# 2b. Support margin gamma on VALIDATION
# ---------------------------------------------------------------------------
def select_gamma(runtime, alphas, model, perception, log):
    rows = runtime.roster('validation').iloc[::GAMMA_SELECTION['every_nth_validation_cube']]
    control = FixedRate(model)
    records, counts = [], {g: np.zeros((len(fftradnet.THRESHOLDS), 3), dtype=np.int64) for g in ('original', *GAMMAS)}
    for i, (_, row) in enumerate(rows.iterrows()):
        cube = runtime.cube(row)
        original = runtime.receive(cube)
        bins, label_count = runtime.label_bins(row.record_name)
        labels = labels_of(runtime, row, easy=True)
        if labels is not None:
            counts['original'] += fftradnet.frame_counts(perception.run(original['spectrum'])[0], labels)
        for gamma in GAMMAS:
            stream, decoded, bits, learned, ctl = control.encode(runtime, row, cube, original, alphas['ca']['a1'], R_STAR,
                                                                 gamma, f'gamma_{gamma:g}')
            received = runtime.receive(decoded)
            record = V13.score(runtime, row, f'gamma_{gamma:g}', cube, original, decoded, bits, alphas['ca'], bins,
                               label_count, None, received)
            record.update(V14.rate_fields(bits, learned, cube.numel()), **ctl)
            if labels is not None:
                counts[gamma] += fftradnet.frame_counts(perception.run(received['spectrum'])[0], labels)
            records.append(record)
        if (i+1) % 25 == 0 or i+1 == len(rows):
            log(f'gamma selection {i+1}/{len(rows)}')
    frame = pd.DataFrame(records)
    base = fftradnet.metrics_from_counts(counts['original'])
    candidates = {}
    for gamma, g in frame.groupby('gamma', sort=True):
        metric = fftradnet.metrics_from_counts(counts[gamma])
        lost = float(g.a1_lost_robust.sum()/max(g.a1_original_robust.sum(), 1))
        candidates[f'{gamma:g}'] = dict(gamma=float(gamma), bpcs=float(g.bpcs.mean()), bpcs_max=float(g.bpcs.max()),
                                        robust_lost_fraction=lost, labels_lost_robust=int(g.labels_lost_robust.sum()),
                                        labels_lost_strict=int(g.labels_lost.sum()), F1=metric['F1'], mAP=metric['mAP'],
                                        mAR=metric['mAR'], dF1=metric['F1']-base['F1'], beta_median=float(g.beta.median()),
                                        loop_relaxed_fraction=float(g.loop_relaxed.mean()),
                                        within_budget_fraction=float(g.within_budget.mean()),
                                        qualifies=lost <= GAMMA_SELECTION['robust_lost_limit'] and int(g.labels_lost_robust.sum()) == 0)
    qualified = [c for c in candidates.values() if c['qualifies']]
    chosen = (max(qualified, key=lambda c: (c['F1'], -c['gamma'])) if qualified
              else min(candidates.values(), key=lambda c: (c['robust_lost_fraction'], c['gamma'])))
    return dict(rule=GAMMA_SELECTION['rule'], r_star=R_STAR, cubes=len(rows), original_F1=base['F1'], candidates=candidates,
                gamma_selected=chosen['gamma'], none_qualified=not qualified, selected_on='VALIDATION only',
                test_unlocked_before_selection=False), frame


# ---------------------------------------------------------------------------
# 3. FFTRadNet fine-tuning
# ---------------------------------------------------------------------------
INPUT_SHAPE = (32, 512, 256)


def to_input_array(path, count):
    return np.lib.format.open_memmap(path, mode='w+', dtype=np.float16, shape=(count, *INPUT_SHAPE))


def store(array, i, x):
    value = x.to(torch.float16)
    if not bool(torch.isfinite(value).all()):
        raise ValueError('A network input does not fit float16.')
    array[i] = value.cpu().numpy()


def build_training_set(runtime, alphas, perception, gamma, folder, log):
    """Every n-th TRAIN cube: original and proposed (R*) network inputs, detection targets, free-space teacher."""
    rows = runtime.roster('train').iloc[::FINETUNE['train_every_nth_cube']]
    rows = rows.loc[[labels_of(runtime, row, easy=False) is not None for _, row in rows.iterrows()]]
    arrays = {arm: to_input_array(folder/f'train_{arm}.npy', len(rows)) for arm in ('original', 'proposed')}
    detection = np.zeros((len(rows), 3, 128, 224), dtype=np.float32)
    freespace = np.zeros((len(rows), 256, 224), dtype=bool)
    for i, (_, row) in enumerate(rows.iterrows()):
        cube = runtime.cube(row)
        original = runtime.receive(cube)
        x = fftradnet.network_input(original['spectrum'], perception.config)
        store(arrays['original'], i, x)
        freespace[i] = perception.run(original['spectrum'])[1][0] >= .5
        detection[i] = finetune.encode_targets(labels_of(runtime, row, easy=False), perception.config)
        _, decoded, _, _, _ = runtime.fixed_rate.encode(runtime, row, cube, original, alphas['ca']['a1'], R_STAR, gamma, 'finetune')
        store(arrays['proposed'], i, fftradnet.network_input(runtime.receive(decoded)['spectrum'], perception.config))
        if (i+1) % 50 == 0 or i+1 == len(rows):
            log(f'fine-tuning TRAIN inputs {i+1}/{len(rows)}')
    for array in arrays.values():
        array.flush()
    return rows, {arm: np.load(folder/f'train_{arm}.npy', mmap_mode='r') for arm in arrays}, detection, freespace


def build_selection_set(runtime, alphas, perception, gamma, folder, log):
    """Every n-th VALIDATION cube with easy labels, each arm's codec at R* (AdaRadar: r = 12.57)."""
    rows = runtime.roster('validation').iloc[::FINETUNE['validation_every_nth_cube']]
    rows = rows.loc[[labels_of(runtime, row, easy=True) is not None for _, row in rows.iterrows()]]
    arrays = {arm: to_input_array(folder/f'validation_{arm}.npy', len(rows)) for arm in ARMS}
    stats = perception.config['dataset']['statistics']
    labels = []
    for i, (_, row) in enumerate(rows.iterrows()):
        cube = runtime.cube(row)
        original = runtime.receive(cube)
        store(arrays['original'], i, fftradnet.network_input(original['spectrum'], perception.config))
        coded, _ = adaradar.compress(original['spectrum'], 12.57, FINETUNE['adaradar_bits'], stats['input_mean'], stats['input_std'])
        store(arrays['adaradar'], i, fftradnet.network_input(coded, perception.config))
        _, decoded, _, _, _ = runtime.fixed_rate.encode(runtime, row, cube, original, alphas['ca']['a1'], R_STAR, gamma, 'finetune')
        store(arrays['proposed'], i, fftradnet.network_input(runtime.receive(decoded)['spectrum'], perception.config))
        labels.append(labels_of(runtime, row, easy=True))
        if (i+1) % 50 == 0 or i+1 == len(rows):
            log(f'fine-tuning VALIDATION inputs {i+1}/{len(rows)}')
    for array in arrays.values():
        array.flush()
    return rows, {arm: np.load(folder/f'validation_{arm}.npy', mmap_mode='r') for arm in ARMS}, labels


class Samples:
    """Micro-batches of (network input, detection target, free-space target) in a given order."""

    def __init__(self, inputs, detection, freespace, device, transform=None):
        self.inputs, self.detection, self.freespace, self.device, self.transform = inputs, detection, freespace, device, transform
        self.count = len(inputs)

    def __call__(self, order):
        for start in range(0, len(order), finetune.MICRO_BATCH):
            pick = order[start:start+finetune.MICRO_BATCH]
            x = torch.from_numpy(np.stack([self.inputs[i] for i in pick])).to(self.device).float()
            if self.transform is not None:
                x = self.transform(x)
            yield (x, torch.from_numpy(self.detection[pick]).to(self.device),
                   torch.from_numpy(self.freespace[pick]).to(self.device))


def adaradar_transform(config, seed):
    """AdaRadar 4-bit at a ratio drawn uniformly in [1, 40] per sample (its --cr_random training)."""
    stats = config['dataset']['statistics']
    generator = torch.Generator().manual_seed(seed)
    low, high = FINETUNE['adaradar_ratio_range']

    def transform(x):
        mean = torch.as_tensor(stats['input_mean'], dtype=torch.float64, device=x.device)
        std = torch.as_tensor(stats['input_std'], dtype=torch.float64, device=x.device)
        out = []
        for sample in x:
            channels = sample.double().permute(1, 2, 0)*std+mean
            spectrum = torch.complex(channels[..., :16], channels[..., 16:])
            ratio = low+(high-low)*float(torch.rand((), generator=generator, dtype=torch.float64))
            coded, _ = adaradar.compress(spectrum, ratio, FINETUNE['adaradar_bits'], stats['input_mean'], stats['input_std'])
            out.append(fftradnet.network_input(coded, config))
        return torch.stack(out)
    return transform


def finetune_networks(runtime, alphas, perception, gamma, out, log):
    """Three fine-tuned copies of the official FFTRadNet; returns ({arm: net}, history, receipt)."""
    folder = out/'_finetune_cache'
    folder.mkdir()
    device = runtime.device
    train_rows, train, detection, freespace = build_training_set(runtime, alphas, perception, gamma, folder, log)
    val_rows, validation, val_labels = build_selection_set(runtime, alphas, perception, gamma, folder, log)
    nets, history, receipt = {}, [], {}
    for n, arm in enumerate(ARMS):
        source = train['proposed' if arm == 'proposed' else 'original']
        transform = adaradar_transform(perception.config, SEED+1) if arm == 'adaradar' else None
        samples = Samples(source, detection, freespace, device, transform)
        frames = lambda arm=arm: ((torch.from_numpy(np.array(validation[arm][i])).to(device), val_labels[i])
                                  for i in range(len(val_labels)))
        net, config = fftradnet.load(PACKAGE/V17.FFTRADNET_FILES['model'], PACKAGE/V17.FFTRADNET_FILES['config'],
                                     PACKAGE/V17.FFTRADNET_FILES['weights'], device)
        torch.manual_seed(SEED+n)
        rows, best = finetune.finetune(net, samples, lambda m: finetune.validation_f1(m, frames, config),
                                       epochs=FINETUNE['epochs'], lr=FINETUNE['lr'], seed=SEED+n,
                                       log=lambda r, arm=arm: log(f'FFTRadNet {arm}', json.dumps(r)))
        history.extend(dict(arm=arm, **r) for r in rows)
        path = out/f'fftradnet_ft_{arm}.pt'
        torch.save(net.state_dict(), path)
        receipt[arm] = dict(checkpoint=path.name, sha256=common.sha(path), selected_epoch=int(best),
                            validation_F1_official=rows[0]['val_F1'], validation_F1_selected=rows[best]['val_F1'])
        nets[arm] = net
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    del train, validation, source, samples, frames
    shutil.rmtree(folder)
    receipt.update(train_cubes=len(train_rows), validation_cubes=len(val_rows))
    return nets, history, receipt


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def sweep_table(summary, perception, agreement, control):
    """TEST: the proposed codec at every budget, C at beta 1.5 and AdaRadar on one rate axis."""
    test = summary.loc[summary.role.eq('test')].set_index('method')
    seen = perception.loc[perception.role.eq('test') & perception.subset.eq('all')].set_index('method')
    agree = agreement.loc[agreement.role.eq('test')].set_index('method')
    rc = control.groupby('method')
    rows = []
    for method, r in test.iterrows():
        s, a = seen.loc[method], agree.loc[method]
        row = dict(method=method, family='proposed' if method.startswith(('fixed', 'closed')) else 'adaradar',
                   budget_bpcs=BUDGETS.get(method), bpcs=r.bpcs_mean, ratio_vs_raw=32/r.bpcs_mean,
                   ca_robust_lost=r.a1_lost_robust_fraction, ca_strict_lost=r.a1_lost_fraction,
                   ca_robust_new_per_map=r.a1_new_robust_per_map, os_robust_lost=r.os_a1_lost_robust_fraction,
                   ca01_robust_lost=r.a01_lost_robust_fraction, labels_detected_original=r.labels_detected_original,
                   labels_lost_strict=r.labels_lost, labels_lost_robust=r.labels_lost_robust, rd_rmse_db=r.rd_rmse_db_mean,
                   mAP=s.mAP, mAR=s.mAR, F1=s.F1, dF1=s.dF1, dF1_ci_low=s.get('dF1_ci_low'), dF1_ci_high=s.get('dF1_ci_high'),
                   objects_kept_fraction=a.objects_kept_fraction, freespace_iou=a.freespace_iou_mean)
        if method in BUDGETS:
            g = rc.get_group(method)
            row.update(bpcs_max=float(r.bpcs_max), beta_median=float(g.beta.median()), loop_relaxed_fraction=float(g.loop_relaxed.mean()),
                       within_budget_fraction=float(g.within_budget.mean()), attempts_mean=float(g.attempts.mean()))
        rows.append(row)
    return pd.DataFrame(rows).sort_values('bpcs')


def label_onset(sweep):
    """Highest budget at which the proposed codec loses a label (strict and robust)."""
    fixed = sweep.loc[sweep.method.isin(list(BUDGETS))].sort_values('budget_bpcs', ascending=False)
    points = [dict(method=r.method, budget_bpcs=r.budget_bpcs, ratio=32/r.budget_bpcs, bpcs=r.bpcs,
                   labels_lost_strict=int(r.labels_lost_strict), labels_lost_robust=int(r.labels_lost_robust),
                   labels_detected_original=int(r.labels_detected_original), loop_relaxed_fraction=r.loop_relaxed_fraction,
                   ca_robust_lost=r.ca_robust_lost) for r in fixed.itertuples()]
    first = lambda key: next((p['budget_bpcs'] for p in points if p[key] > 0), None)
    return dict(points=points, highest_budget_losing_a_label_strict=first('labels_lost_strict'),
                highest_budget_losing_a_label_robust=first('labels_lost_robust'),
                note='Budgets are per-cube maxima; the loop is relaxed only where its rate floor exceeds the budget.')


def paired_differences(frame, perception, draws=V17.BOOTSTRAP_DRAWS):
    """Recording-level paired bootstrap at R*: proposed minus AdaRadar (CFAR, labels, FFTRadNet F1/mAR)."""
    rng = np.random.default_rng(SEED)
    pairs = {'official network': (HEADLINE, ADARADAR_HEADLINE),
             'official network, adaptive AdaRadar': (HEADLINE, 'adaradar_adaptive_s4'),
             'fine-tuned networks': (f'{HEADLINE}@ft_proposed', f'{ADARADAR_HEADLINE}@ft_adaradar'),
             'fine-tuned networks, adaptive AdaRadar': (f'{HEADLINE}@ft_proposed', 'adaradar_adaptive_s4_ft@ft_adaradar')}
    test = frame.loc[frame.role.eq('test')]
    recordings = sorted(test.scene_name.unique())
    pick = rng.integers(0, len(recordings), (draws, len(recordings)))
    sums = lambda m, c: test.loc[test.method.eq(m)].groupby('scene_name')[c].sum().reindex(recordings).to_numpy(float)
    labelled = [f for f in perception.frames if f['role'] == 'test' and f['labelled']]
    scenes = sorted({f['scene_name'] for f in labelled})
    pick_p = rng.integers(0, len(scenes), (draws, len(scenes)))
    counts = lambda key: np.stack([np.stack([f['counts'][key] for f in labelled if f['scene_name'] == s]).sum(0) for s in scenes])
    out = {}
    for name, (a, b) in pairs.items():
        stats = {}
        for stat, (num, den) in dict(ca_robust_lost=('a1_lost_robust', 'a1_original_robust'),
                                     os_robust_lost=('os_a1_lost_robust', 'os_a1_original_robust'),
                                     labels_lost=('labels_lost', 'labels_detected_original')).items():
            ma, mb = a.split('@')[0], b.split('@')[0]
            ratio = lambda m, idx: sums(m, num)[idx].sum(-1)/np.maximum(sums(m, den)[idx].sum(-1), 1)
            full = np.arange(len(recordings))
            values = ratio(ma, pick)-ratio(mb, pick)
            stats[stat] = dict(proposed=float(ratio(ma, full)), adaradar=float(ratio(mb, full)),
                               difference=float(ratio(ma, full)-ratio(mb, full)),
                               ci=[float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))])
        f1_a, recall_a = V17.f1_and_recall(counts(a)[pick_p].sum(1))
        f1_b, recall_b = V17.f1_and_recall(counts(b)[pick_p].sum(1))
        point_a, point_b = V17.f1_and_recall(counts(a).sum(0)), V17.f1_and_recall(counts(b).sum(0))
        stats['F1'] = dict(proposed=float(point_a[0]), adaradar=float(point_b[0]), difference=float(point_a[0]-point_b[0]),
                           ci=[float(np.percentile(f1_a-f1_b, 2.5)), float(np.percentile(f1_a-f1_b, 97.5))])
        stats['mAR'] = dict(proposed=float(point_a[1]), adaradar=float(point_b[1]), difference=float(point_a[1]-point_b[1]),
                            ci=[float(np.percentile(recall_a-recall_b, 2.5)), float(np.percentile(recall_a-recall_b, 97.5))])
        out[name] = dict(proposed=a, adaradar=b, **stats)
    return dict(r_star=R_STAR, recordings=len(recordings), draws=draws, comparisons=out)


def rate_matched_table(summary, perception, agreement):
    """TEST at R*: each codec with the official and the fine-tuned networks."""
    test = summary.loc[summary.role.eq('test')].set_index('method')
    seen = perception.loc[perception.role.eq('test') & perception.subset.eq('all')].set_index('method')
    agree = agreement.loc[agreement.role.eq('test')].set_index('method')
    keys = ['original', *(f'original@ft_{a}' for a in ARMS), HEADLINE, f'{HEADLINE}@ft_proposed', f'{HEADLINE}@ft_original',
            ADARADAR_HEADLINE, f'{ADARADAR_HEADLINE}@ft_adaradar', f'{ADARADAR_HEADLINE}@ft_original',
            'adaradar_adaptive_s4', 'adaradar_adaptive_s4_ft@ft_adaradar']
    rows = []
    for key in keys:
        method, _, net = key.partition('@')
        s = seen.loc[key]
        r = test.loc[method] if method in test.index else None
        a = agree.loc[key] if key in agree.index else None
        rows.append(dict(input=method, network=net or 'official', bpcs=32. if r is None else r.bpcs_mean,
                         mAP=s.mAP, mAR=s.mAR, F1=s.F1, dF1_vs_official_on_original=s.dF1, dF1_ci_low=s.get('dF1_ci_low'),
                         dF1_ci_high=s.get('dF1_ci_high'), objects_kept_fraction=None if a is None else a.objects_kept_fraction,
                         freespace_iou=None if a is None else a.freespace_iou_mean,
                         ca_robust_lost=None if r is None else r.a1_lost_robust_fraction,
                         labels_lost_strict=None if r is None else r.labels_lost))
    return pd.DataFrame(rows)


def predictions(summary, perception):
    test = summary.loc[summary.role.eq('test')].set_index('method').loc[HEADLINE]
    f1 = perception.loc[perception.role.eq('test') & perception.subset.eq('all')].set_index('method').F1
    return dict(P1=bool(test.a1_lost_robust_fraction <= V17.GATE['robust_lost_fraction'] and test.labels_lost_robust == 0),
                P2=bool(f1[HEADLINE] >= f1[ADARADAR_HEADLINE]),
                P3=bool(f1[f'{HEADLINE}@ft_proposed'] >= f1[f'{ADARADAR_HEADLINE}@ft_adaradar']))


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def context(runtime, out, assignment, scan):
    hashes = V17.source_hashes()
    for path in (Path(__file__), Path(V18.__file__), Path(adaradar.__file__), Path(finetune.__file__), PACKAGE/FROZEN_MODEL['file']):
        hashes[str(path.resolve().relative_to(common.ROOT))] = common.sha(path)
    return dict(revision=REVISION, source_root=str(common.ROOT.resolve()), output=str(out),
                dataset_root=str(runtime.dataset.root), reference_root=str(runtime.dataset.reference_root),
                device=str(runtime.device), torch=torch.__version__, cuda_runtime=torch.version.cuda,
                gpu=torch.cuda.get_device_name(runtime.device) if runtime.device.type == 'cuda' else 'cpu-engineering',
                split=str(Path(assignment).resolve()), split_sha256=common.sha(assignment),
                split_provenance_sha256=common.sha(V17.provenance_path(assignment)), inventory=scan, source_hashes=hashes,
                dataset_identity=data.dataset_identity(runtime.dataset),
                populations={role: frame.record_name.tolist() for role, frame in runtime.rows.items()},
                configuration=CONFIG, seed=SEED, bitstream_workers=V17.BITSTREAM_WORKERS)


def run_check(runtime, out, expected):
    """V20 checks on one TRAIN cube (rate control at every budget, file at R*, one FFTRadNet training step,
    disk for the fine-tuning inputs), then V18's AdaRadar checks and the V17 CUDA pre-flight (writes CHECK.json)."""
    if V18.V17 is not V17 or {V17.PACKAGE, V18.PACKAGE} != {PACKAGE}:
        raise AssertionError('Every runner must read the V20 package (FFTRadNet files, split, V17 model).')
    if not Path(str(out)+'.CHECK.json').exists():
        device = runtime.device
        model = V18.load_frozen_model(device)
        row = runtime.roster('train').iloc[len(runtime.roster('train'))//2]
        cube = runtime.cube(row)
        original = runtime.receive(cube)
        alpha = 10**(4.18/10)
        control, rates = FixedRate(model), {}
        for name, budget in BUDGETS.items():
            (stream, decoded, bits, learned, ctl), ms = V17.timed(device, control.encode, runtime, row, cube, original, alpha,
                                                                  budget, 0., name)
            rate = V14.rate_fields(bits, learned, cube.numel())['bpcs']
            if not ctl['within_budget'] or rate > budget+1e-12 or not bool(torch.isfinite(torch.view_as_real(decoded)).all()):
                raise AssertionError(f'Rate control failed at {budget} bpcs ({rate:.4f}).')
            rates[name] = dict(bpcs=rate, encode_ms=ms, **ctl)
            if name == HEADLINE:
                path = Path(f'{out}.CHECK_{name}.rdv17')
                files = V17.bitstream.roundtrip(stream, path, cube.dtype)
                path.unlink()
                if not files['bit_exact'] or files['file_over_ideal'] > .05:
                    raise AssertionError('R* file decode is not bit-exact or exceeds its ideal length by >5%.')
        print('Rate control on one TRAIN cube:', json.dumps(rates), flush=True)
        net, config = fftradnet.load(PACKAGE/V17.FFTRADNET_FILES['model'], PACKAGE/V17.FFTRADNET_FILES['config'],
                                     PACKAGE/V17.FFTRADNET_FILES['weights'], device)
        x = fftradnet.network_input(original['spectrum'], config)[None]
        seg = Network(net, config).run(original['spectrum'])[1][0] >= .5
        labels = labels_of(runtime, row, easy=False)
        detection = torch.from_numpy(finetune.encode_targets(np.zeros((0, 2)) if labels is None else labels, config))
        batch = finetune.MICRO_BATCH
        x = adaradar_transform(config, SEED)(x.repeat(batch, 1, 1, 1))
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        finetune.set_trainable(net)
        (value, parts), step_ms = V17.timed(device, lambda: finetune.loss(net, x, detection[None].repeat(batch, 1, 1, 1).to(device),
                                                                          torch.from_numpy(seg)[None].repeat(batch, 1, 1).to(device)))
        _, backward_ms = V17.timed(device, value.backward)
        peak = torch.cuda.max_memory_allocated(device)/1e9 if device.type == 'cuda' else 0.
        total = torch.cuda.get_device_properties(device).total_memory/1e9 if device.type == 'cuda' else 0.
        if not math.isfinite(float(value)) or (device.type == 'cuda' and peak > .85*total):
            raise AssertionError(f'FFTRadNet training step: loss {float(value)}, peak {peak:.2f} of {total:.2f} GB.')
        free = shutil.disk_usage(out.parent).free/1e9
        if free < CACHE_FREE_GB:
            raise OSError(f'{free:.0f} GB free next to the output; the fine-tuning inputs need {CACHE_FREE_GB} GB.')
        preflight = dict(rates=rates, fftradnet_step=dict(loss=float(value), **parts, micro_batch=batch,
                                                          forward_ms=step_ms, backward_ms=backward_ms, peak_gb=peak, device_gb=total),
                         disk_free_gb=free)
        V17.write_report(Path(str(out)+'.PREFLIGHT_V20.json'), preflight)
        print('V20 pre-flight PASS:', json.dumps(preflight['fftradnet_step']), f'disk free {free:.0f} GB', flush=True)
        del net, x
    points = V17.BITSTREAM_POINTS
    V17.BITSTREAM_POINTS = ('closed_full_b2',)  # the V17 check codes A, B and C at beta 2
    try:
        V18.run_check(runtime, out, expected)
    finally:
        V17.BITSTREAM_POINTS = points


def run(runtime, out, expected, log_every=25):
    checked = common.read_json(Path(str(out)+'.CHECK.json'))
    if checked.get('context') != expected or checked.get('passed') is not True:
        raise ValueError('Run the matching CUDA check first.')
    if out.exists():
        raise FileExistsError('Preserve partial/completed results; never overwrite this study.')
    with common.exclusive(out):
        out.mkdir(parents=True)
        common.write_json(out/'manifest.json', expected)
        pd.concat(runtime.rows.values(), ignore_index=True).to_csv(out/'POPULATIONS.csv', index=False)
        start = time.time()
        log = lambda *a: print(*a, f'{time.time()-start:.0f}s', flush=True)
        device = runtime.device
        models = dict(v17=V18.load_frozen_model(device))
        calibration, curve = V17.calibrate(runtime)
        V17.write_report(out/'H0_CALIBRATION.json', calibration)
        curve.to_csv(out/'H0_SURVIVAL.csv', index=False)
        alphas = dict(ca=calibration['alphas'], os=calibration['os_cfar']['alphas'])
        log('alpha dB CA:', json.dumps(calibration['alphas_db']), 'OS:', json.dumps(calibration['os_cfar']['alphas_db']))

        # 2a. Entropy model on the configuration ladder (TRAIN; epoch on VALIDATION).
        model, history, best, selection_cubes = tune_entropy_model(runtime, alphas['ca']['a1'], log)
        torch.save(model.state_dict(), out/'entropy_model_v20.pt')
        pd.DataFrame(history).to_csv(out/'ENTROPY_MODEL_TRAINING.csv', index=False)
        models['v20'] = model
        runtime.fixed_rate = FixedRate(model)
        log(f'Entropy model frozen at epoch {best} (0 = V17).')

        # 2b. Support margin on VALIDATION.
        perception = V17.Perception(runtime)
        selection, frame_gamma = select_gamma(runtime, alphas, model, perception, log)
        frame_gamma.to_csv(out/'GAMMA_SELECTION_VALIDATION.csv', index=False)
        V17.write_report(out/'GAMMA_SELECTION.json', selection)
        gamma = selection['gamma_selected']
        log('gamma* =', gamma, json.dumps({k: [round(c['F1'], 4), round(c['robust_lost_fraction'], 4), c['qualifies']]
                                            for k, c in selection['candidates'].items()}))

        # 3. FFTRadNet fine-tuning, three arms.
        runtime.fixed_rate = FixedRate(model)
        nets, ft_history, ft_receipt = finetune_networks(runtime, alphas, perception, gamma, out, log)
        pd.DataFrame(ft_history).to_csv(out/'FFTRADNET_FINETUNING.csv', index=False)

        # Freeze everything that was learned or selected, then unlock TEST once.
        receipt = out/'FROZEN_MODEL.json'
        V17.write_report(receipt, dict(completed=True, checkpoint='entropy_model_v20.pt',
                                       checkpoint_sha256=common.sha(out/'entropy_model_v20.pt'), selected_epoch=int(best),
                                       entropy_selection_cubes=selection_cubes, anchor_checkpoint_sha256=FROZEN_MODEL['sha256'],
                                       gamma_selected=gamma, fftradnet_finetuned=ft_receipt, test_used_for_selection=False))
        methods = (tuple((name, dict(kind='fixed', budget=budget, gamma=gamma)) for name, budget in BUDGETS.items())
                   + C_ANCHOR + ADARADAR)
        runtime.fixed_rate = FixedRate(model)
        perception = Perception(runtime, nets)
        records, label_log, control_log, bitstreams = [], [], [], None
        try:
            runtime.authorize_test(receipt)
            bitstreams = V17.Bitstreams(out, V17.BITSTREAM_WORKERS)
            log(f'TEST unlocked; {V17.BITSTREAM_WORKERS} bitstream workers; methods:', [m for m, _ in methods])
            roster = runtime.roster('test')
            for i, (_, row) in enumerate(roster.iterrows()):
                records.extend(evaluate_cube(runtime, row, methods, alphas, models, perception, label_log, control_log, bitstreams))
                if (i+1) % log_every == 0 or i+1 == len(roster):
                    log(f'TEST {i+1}/{len(roster)}')
            frame_bits = bitstreams.finish()
        except BaseException:
            if bitstreams is not None:
                bitstreams.close()
            raise
        frozen = common.read_json(receipt)
        if (common.sha(out/'entropy_model_v20.pt') != frozen['checkpoint_sha256']
                or any(common.sha(out/v['checkpoint']) != v['sha256'] for v in frozen['fftradnet_finetuned'].values() if isinstance(v, dict))):
            raise ValueError('A frozen model changed during TEST.')
        log('Evaluation complete; writing reports.')

        frame, summary = V18.report_tables(records)  # bpcs_max: the largest per-cube rate (V13 summary)
        frame.to_csv(out/'RESULTS_BY_CUBE.csv', index=False)
        pd.DataFrame(label_log).to_csv(out/'LABELS.csv', index=False)
        control = pd.DataFrame(control_log)
        control.to_csv(out/'RATE_CONTROL.csv', index=False)
        frame_bits.to_csv(out/'BITSTREAM.csv', index=False)
        bits_summary = V17.bitstream_summary(frame_bits)
        V17.write_report(out/'BITSTREAM_SUMMARY.json', bits_summary)
        summary.to_csv(out/'SUMMARY.csv', index=False)
        V17.write_report(out/'RECORDING_CI.json', V17.recording_bootstrap(frame))
        seen, agreement = perception.report(out)
        sweep = sweep_table(summary, seen, agreement, control)
        sweep.to_csv(out/'RATE_SWEEP_TEST.csv', index=False)
        V17.write_report(out/'LABEL_LOSS_ONSET.json', label_onset(sweep))
        rate_matched_table(summary, seen, agreement).to_csv(out/'RATE_MATCHED_TEST.csv', index=False)
        V17.write_report(out/'RATE_MATCHED_DIFFERENCES.json', paired_differences(frame, perception))
        passed = predictions(summary, seen)
        V17.write_report(out/'PREDICTIONS.json', dict(predictions=PREDICTIONS, passed=passed))
        runtime.dataset.assert_unchanged()
        V17.write_report(out/'SMOKE_COMPLETED.json', dict(
            completed=True, passed=True, context=expected, seed=SEED, test_evaluated=True, test_used_for_selection=False,
            entropy_model_trained=True, entropy_selected_epoch=int(best), gamma_selected=gamma,
            fftradnet_selected_epochs={arm: ft_receipt[arm]['selected_epoch'] for arm in ARMS},
            alphas_db=calibration['alphas_db'], os_alphas_db=calibration['os_cfar']['alphas_db'],
            cubes={r: len(runtime.roster(r)) for r in ('train', 'validation', 'test')},
            pre_registered_points_passed=passed, bitstreams_all_exact=bits_summary['all_decoded_from_file_exactly'],
            runtime_seconds=time.time()-start))
    print(REVISION.split('-')[0], 'run COMPLETE; start with RATE_SWEEP_TEST.csv, RATE_MATCHED_TEST.csv, LABEL_LOSS_ONSET.json:',
          out, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', choices=('count', 'check', 'run'), required=True)
    for key in ('dataset-root', 'reference-root', 'split-audit', 'out'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    dataset = data.prepare_dataset(args.dataset_root, args.reference_root)
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    table, scan = V17.inventory(dataset, args.split_audit)
    with pd.option_context('display.width', 200, 'display.max_rows', 200):
        print(table.sort_values(['role', 'recording']).to_string(index=False))
    print(json.dumps(scan, indent=1), flush=True)
    if args.stage == 'count':
        if not Path(str(out)+'.COUNT.csv').exists():
            table.to_csv(Path(str(out)+'.COUNT.csv'), index=False)
            V17.write_report(Path(str(out)+'.COUNT.json'), scan)
        return
    if not scan['matches_frozen_split']:
        raise ValueError('The dataset folder differs from the frozen V17 split; no rebalancing. See the inventory above.')
    if torch.device(args.device).type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('CUDA is required. There is no CPU fallback.')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    runtime = Runtime(dataset, args.split_audit, args.device)
    expected = json.loads(json.dumps(context(runtime, out, args.split_audit, scan)))
    print(json.dumps({k: expected[k] for k in ('revision', 'gpu', 'bitstream_workers')}, indent=1), flush=True)
    if args.stage == 'check':
        run_check(runtime, out, expected)
    else:
        run(runtime, out, expected)


if __name__ == '__main__':
    main()
