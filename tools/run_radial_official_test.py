"""One frozen evaluation on RADIal's official test recordings, plus AdaRadar's published network on our 662 test cubes. Nothing is trained or tuned.

FRAMES. RADIal's four official test recordings: PPO never trained on them, and neither FFTRadNet network did. One of them,
12.03.47, was in our VALIDATION role and helped select the checkpoint on CFAR scores; the other three were never used. Our 662
test cubes come from recordings used to train both networks.

ON THE OFFICIAL RECORDINGS. PPO with its frozen policy and prices, three fixed settings of its codec, AdaRadar at its six static
ratios and with its feedback controller, BAQ at 1, 2 and 3 bits, and Learned ECDQ: the whole-map dithered codec with the V20
entropy model, no pruning and no decoding at the encoder.

ON OUR 662 CUBES. AdaRadar only, because PPO and the classical methods are already scored there.

NETWORKS. The official FFTRadNet scores every method. AdaRadar's published 4-bit network, retrained by its authors on compressed
data, also scores AdaRadar, PPO and the original. AdaRadar's feedback runs twice, once driven by each network.

RATE OF ADARADAR. Three counts of the same reconstruction: values only, its own count; complete, with positions and steps; and
coded, with the adaptive code of our tiles for values and an adaptive code for positions.

GATE. The run first reproduces saved PPO and AdaRadar rows on the first of our 662 cubes, and stops if they differ.

python -B tools/run_radial_official_test.py --stage count|check|run --dataset-root ... --reference-root ... --split-audit ... --out ...
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
from radar_sim import radial_fftradnet_eval as fftradnet
from radar_sim import radial_v14_entropy_model as entropy
from radar_sim import radial_v17_baselines as baselines
from radar_sim import radial_v18_adaradar as adaradar


def sibling(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


T = sibling('run_radial_ppo_tuned')  # the frozen case loader, the scoring receiver and the frozen roles
sweep, learning, cap, V20, V18, V17, V13 = T.sweep, T.learning, T.cap, T.V20, T.V18, T.V17, T.V13
V14, common, data, codec = V17.V14, T.common, T.data, T.codec
PACKAGE = Path(__file__).resolve().parent/'radial_official_test_gpu'
sweep.PACKAGE = V20.PACKAGE = V18.PACKAGE = V17.PACKAGE = T.PACKAGE = PACKAGE
REVISION = 'official.1.0.0-frozen-ppo-adaradar-published-weights-official-test-recordings'
SEED = T.SEED
OFFICIAL = ('RECORD@2020-11-22_12.03.47', 'RECORD@2020-11-22_12.25.47', 'RECORD@2020-11-22_12.45.05', 'RECORD@2020-11-22_12.54.38')
NEW_RECORDINGS = OFFICIAL[1:]
FIXED = ((.75, 5), (.75, 6), (.5, 5))
RATIOS, ADARADAR_BITS = (50., 25., 12.57, 8., 4., 2.), 4
BAQ_BITS = (1, 2, 3)
ECDQ_BUDGETS, ECDQ_BETAS = (.4585, .8208, 1., 1.4541, 2., 2.1064), (2., 1.5, 1.)
ECDQ_LADDER = tuple(2.**(k/8) for k in range(-8, 57))  # the V21 ladder: beta .5 to 128, eight steps per octave
ECDQ_START = ECDQ_LADDER.index(1.)
WEIGHTS = 'adaradar_4bit_state_dict.pt'
OWN = 'adaradar'     # AdaRadar's published network; its scores are stored as '<method>@adaradar_weights'
REPLAY_CUBES = 12
PARTIAL_EVERY = 200
PUBLISHED = dict(adaptive=dict(ratio=12.57, precision=96.25, recall=94.04, F1=95.13), baseline=dict(precision=97.24, recall=95.93, F1=96.58), official_dF1_r12=-2.08)
PREDICTIONS = dict(P1='AdaRadar with feedback and its own weights on the official recordings has F1 within 2 points of its published 95.13',
                   P2='AdaRadar at ratio 12.57 with official weights on the official recordings loses F1 within 2 points of its published no-fine-tuning curve, -2.08',
                   P3='PPO keeps more robust CFAR detections than AdaRadar at each AdaRadar rate above 1 bpcs on the official recordings')
CONFIG = dict(official_recordings=OFFICIAL, new_recordings=NEW_RECORDINGS, fixed=FIXED, ratios=RATIOS, adaradar_bits=ADARADAR_BITS, baq_bits=BAQ_BITS, ecdq_budgets=ECDQ_BUDGETS, ecdq_betas=ECDQ_BETAS,
              ecdq_ladder=[ECDQ_LADDER[0], ECDQ_LADDER[-1], len(ECDQ_LADDER)], adaptive=V18.ADAPTIVE_SETTINGS, replay_cubes=REPLAY_CUBES, published=PUBLISHED, predictions=PREDICTIONS, sync_tolerance_us=data.TOLERANCE_US)


class Perception(V20.Perception):
    """The official network on every method, and AdaRadar's published network on the methods named here."""

    def __init__(self, runtime, net, own_scored):
        super().__init__(runtime, finetuned={OWN: net})
        self.own_scored = own_scored

    def add_finetuned(self, name, received):
        if name in self.own_scored:
            self.store(f'{name}@adaradar_weights', *self.finetuned[OWN].run(received['spectrum']))


def inventory(dataset, assignment):
    """Metadata-only scan: the 36 frozen recordings must be unchanged, and the official recordings must be usable."""
    roles = pd.read_csv(assignment).set_index('recording').role
    known = pd.read_csv(PACKAGE/'RECORDING_COUNTS.csv').set_index('recording').eligible_cubes
    table = dataset.recording_audit[['recording', 'synchronized_frames', 'complete_frames', 'usable_label_indexed_frames']].rename(columns={'usable_label_indexed_frames': 'usable_cubes'})
    table = table.assign(role=table.recording.map(roles).fillna('UNASSIGNED'), frozen_cubes=table.recording.map(known).fillna(-1).astype(int))
    changed = table.loc[table.frozen_cubes.ge(0) & table.usable_cubes.ne(table.frozen_cubes)].recording.tolist()
    usable = table.set_index('recording').usable_cubes
    official = {name: int(usable.get(name, 0)) for name in OFFICIAL}
    scan = dict(recordings=int(table.usable_cubes.gt(0).sum()), total_usable_cubes=int(table.usable_cubes.sum()), cubes_by_role=table.loc[table.usable_cubes.gt(0)].groupby('role').usable_cubes.sum().astype(int).to_dict(),
                official_cubes=official, official_total=sum(official.values()), changed_recordings=changed, missing_recordings=sorted(set(known.index)-set(table.recording)),
                unassigned_recordings=table.loc[table.role.eq('UNASSIGNED') & table.usable_cubes.gt(0)].recording.tolist(), synchronization_tolerance_us=data.TOLERANCE_US, ADC_bodies_read=False)
    scan['matches_frozen_split'] = not changed and not scan['missing_recordings'] and not scan['unassigned_recordings'] and all(official.values())
    return table, scan


def rosters(dataset, assignment):
    """All cubes of the test role of this evaluation; one cube fills each unused role."""
    train, validation, test = V17.partition.assign_frozen_partitions(dataset, assignment, bounded=False)
    return dict(train=train.iloc[:1].assign(role='train'), validation=validation.iloc[:1].assign(role='validation'), test=test.assign(role='test'))


def load_entropy_model(device):
    model = entropy.ContextModel().to(device)
    model.load_state_dict(torch.load(PACKAGE/'entropy_model_v20.pt', map_location=device, weights_only=True))
    return model.eval()


def load_policy(runtime):
    frozen = common.read_json(PACKAGE/'FROZEN_POLICY.json')
    receipt = common.read_json(PACKAGE/'FROZEN_MODEL.json')
    if common.sha(PACKAGE/receipt['checkpoint']) != frozen['checkpoint_sha256']:
        raise ValueError('The frozen policy changed.')
    fitted = torch.load(PACKAGE/'feature_normalizer.pt', map_location=runtime.device, weights_only=False)
    actor = learning.DeltaPolicy(len(fitted['names']), runtime.device)
    actor.load_state_dict(torch.load(PACKAGE/receipt['checkpoint'], map_location=runtime.device, weights_only=False)['actor'])
    return actor.eval(), fitted, frozen


def frozen_alphas():
    calibration = common.read_json(PACKAGE/'H0_CALIBRATION.json')
    return dict(ca=calibration['alphas'], os=calibration['os_cfar']['alphas'])


@torch.no_grad()
def recoded_rate(spectrum, ratio, stats):
    """AdaRadar's integers priced by an adaptive code: magnitude classes per spatial tile, and positions by the best of three contexts."""
    rng, doppler, rx = spectrum.shape
    x = torch.cat((spectrum.real, spectrum.imag), -1).to(torch.float64)
    mean, std = (torch.as_tensor(stats[key], dtype=torch.float64, device=x.device) for key in ('input_mean', 'input_std'))
    channels = ((x-mean)/std).permute(2, 0, 1)
    d = adaradar.dct_matrix(adaradar.BLOCK, spectrum.device)
    blocks = adaradar._blocks(channels, adaradar.BLOCK)
    z = (d@blocks@d.T).reshape(*blocks.shape[:3], adaradar.BLOCK**2)
    threshold = z.abs().topk(adaradar.kept_per_block(ratio), dim=-1).values[..., -1:]
    pruned = torch.where(z.abs() >= threshold, z, torch.zeros_like(z))
    levels = 2**(ADARADAR_BITS-1)-1
    step = pruned.abs().amax(-1, keepdim=True)/levels
    step = torch.where(step > 0, step, torch.ones_like(step))
    integers = torch.round(pruned/step).clamp(-levels, levels).to(torch.int64)
    mask = integers != 0
    numel, tiles = rng*doppler*rx, mask.sum(-1).numel()
    values = 0.
    for h in range(integers.shape[1]):
        for w in range(integers.shape[2]):
            classes = codec.magnitude_class(integers[:, h, w][mask[:, h, w]])
            values += float(codec.kt_bits(torch.bincount(classes, minlength=codec.CLASSES).to(torch.float64)[None])[0]+classes.sum())

    def binary(ones, total):
        ones = ones.to(torch.float64)
        return float(codec.kt_bits(torch.stack((ones, total-ones), -1)).sum())

    positions = min(binary(mask.sum(0).flatten(), mask.shape[0]), binary(mask.flatten(0, 2).sum(0), tiles), binary(mask.sum((1, 2)).flatten(), mask.shape[1]*mask.shape[2]))
    return (values+positions+tiles*adaradar.SCALE_BITS+adaradar.HEADER_BITS)/numel


class Evaluator:
    """The fixed plan of one run; every method on one cube is scored as the earlier runs scored it."""

    def __init__(self, runtime, alphas, policy, net):
        self.runtime, self.alphas, self.net = runtime, alphas, net
        self.actor, self.fitted, self.frozen = policy
        self.model = load_entropy_model(runtime.device)
        self.adaptive = {name: V18.AdaptiveRate(V18.ADAPTIVE_SETTINGS) for name in ('official', 'own')}
        self.starts = {}
        self.stats = common.read_json(PACKAGE/V17.FFTRADNET_FILES['config'])['dataset']['statistics']

    def begin(self, row, perception, records, label_log):
        """Load one cube, start its perception record, and return the case and the function that scores one method."""
        runtime, alphas = self.runtime, self.alphas
        case = T.load_case(runtime, row)
        cube, original, bins, label_count = case['cube'], case['original'], case['bins'], case['label_count']
        original_os = V17.receivers.cyclic_os_cfar(original['detector']['power'])
        energy = original['spectrum'].abs().square().sum()
        perception.start(row, original)
        perception.add_finetuned('original', original)

        def finish(name, decoded, accounted, received, **notes):
            record = V13.score(runtime, row, name, cube, original, decoded, accounted, alphas['ca'], bins, label_count, label_log, received)
            method_os = V17.receivers.cyclic_os_cfar(received['detector']['power'])
            for key, alpha in alphas['os'].items():
                metrics = codec.detection_metrics(original_os, method_os, alpha, V13.MARGIN_DB)
                record.update({f'os_{key}_{k}': v for k, v in metrics.items()})
            record.update(bpcs_kt=record['bpcs'], bpcs_learned=math.nan, learned_chosen=0., codec_ms=math.nan, encode_ms=math.nan, decode_ms=math.nan, entropy_model_ms=math.nan,
                          spectrum_nmse=float((received['spectrum']-original['spectrum']).abs().square().sum()/energy))
            record.update(notes)
            perception.add(name, received)
            records.append(record)
            return record
        return case, finish

    def ppo(self, case, finish):
        """The frozen policy at AdaRadar's six budgets, and three fixed settings of the same codec."""
        physics, features = self.runtime.codec, case['features']
        allocations = {}
        points = [(f'ppo_{sweep.tag(budget)}', self.frozen['prices'][f'{budget:g}'], None, budget) for budget in T.BUDGETS]
        points += [(f'fixed_r{round(100*rho)}_q{q}', None, (rho, q), cap.CAP_BPCS) for rho, q in FIXED]
        for name, price, fixed, budget in points:
            if fixed is None:
                if price not in allocations:
                    allocations[price] = learning.rollout(self.actor, self.fitted, features, price, deterministic=True)
                rho, q = allocations[price]['rho'], allocations[price]['q']
            else:
                rho, q = features.new_full((len(features),), fixed[0]), features.new_full((len(features),), float(fixed[1]))
            decoded, meta, q, steps = sweep.project(physics, case['source'], case['order'], rho, q, budget)
            bits = dict(total=int(math.ceil(float(meta['realized_bits']))), accounting=T.ACCOUNTING, closed_loop_iterations=0)
            finish(name, decoded, bits, self.runtime.receive(decoded), price=price, budget_bpcs=budget, within_budget=float(meta['bpcs']) <= budget, projection_steps=steps, mean_rho=float(rho.mean()), mean_q=float(q.to(torch.float64).mean()))

    def adaradar_row(self, case, finish, name, spectrum, bits, ratio):
        record = finish(name, case['cube'], bits, self.runtime.receive_spectrum(spectrum), adaradar_comp_ratio=ratio, bpcs_nonzero_only=bits['total_nonzero_only']/case['cube'].numel(), index_bpcs=bits['index_bits']/case['cube'].numel(),
                        nonzero_fraction=bits['nonzero_fraction'], bpcs_coded=recoded_rate(case['original']['spectrum'], ratio, self.stats))
        record['nmse'] = math.nan  # not an ADC-domain codec; its distortion is spectrum_nmse

    def adaradar(self, row, case, finish, perception):
        spectrum = case['original']['spectrum']
        for ratio in RATIOS:
            decoded, bits = adaradar.compress(spectrum, ratio, ADARADAR_BITS, self.stats['input_mean'], self.stats['input_std'])
            self.adaradar_row(case, finish, f'adaradar_r{sweep.tag(ratio).rstrip("0").rstrip("p")}_s{ADARADAR_BITS}', decoded, bits, ratio)
        for name, driver in (('official', perception), ('own', perception.finetuned[OWN])):
            decoded, bits, ratio = self.adaptive[name].code(row, spectrum, self.stats, driver)
            self.adaradar_row(case, finish, f'adaradar_adaptive_{name}_s{ADARADAR_BITS}', decoded, bits, ratio)

    def baq(self, case, finish):
        for bits_per_component in BAQ_BITS:
            decoded, bits = baselines.baq(case['cube'], bits_per_component)
            finish(f'baq_{bits_per_component}bit', decoded, bits, self.runtime.receive(decoded))

    def ecdq_at(self, cube, beta, seed):
        stream = codec.encode_ecdq(cube, beta=beta, seed=seed)
        bits = dict(codec.ecdq_bits(stream), accounting='Learned ECDQ, open loop: the shorter of the per-tile adaptive code and the learned model, plus all side information', closed_loop_iterations=0)
        return stream, bits, V14.rate_fields(bits, entropy.learned_payload_bits(self.model, entropy.compact(stream)), cube.numel())

    def ecdq(self, row, case, finish):
        cube, seed = case['cube'], self.runtime.seed(row)
        for budget in ECDQ_BUDGETS:
            tried = {}

            def fits(index):
                if index not in tried:
                    tried[index] = self.ecdq_at(cube, ECDQ_LADDER[index], seed)
                return tried[index][2]['bits'] <= budget*cube.numel()

            key = (row.scene_name, budget)
            index = V20.first_feasible(fits, self.starts.get(key, ECDQ_START), len(ECDQ_LADDER))
            within = index is not None
            index = len(ECDQ_LADDER)-1 if index is None else index
            fits(index)
            self.starts[key] = index
            stream, bits, rate = tried[index]
            decoded = codec.decode_ecdq(stream, cube.dtype)
            finish(f'ecdq_{sweep.tag(budget)}', decoded, bits, self.runtime.receive(decoded), budget_bpcs=budget, beta=ECDQ_LADDER[index], within_budget=within, **rate)
        for beta in ECDQ_BETAS:
            stream, bits, rate = self.ecdq_at(cube, beta, seed)
            decoded = codec.decode_ecdq(stream, cube.dtype)
            finish(f'ecdq_b{beta:g}'.replace('.', 'p'), decoded, bits, self.runtime.receive(decoded), beta=beta, **rate)

    def official_cube(self, row, perception, records, label_log):
        case, finish = self.begin(row, perception, records, label_log)
        self.ppo(case, finish)
        self.adaradar(row, case, finish, perception)
        self.baq(case, finish)
        self.ecdq(row, case, finish)

    def ours_cube(self, row, perception, records, label_log):
        case, finish = self.begin(row, perception, records, label_log)
        self.adaradar(row, case, finish, perception)


def own_scored():
    """The methods AdaRadar's network sees: the original, every AdaRadar point and PPO at the six budgets."""
    names = {'original', f'adaradar_adaptive_official_s{ADARADAR_BITS}', f'adaradar_adaptive_own_s{ADARADAR_BITS}'}
    names |= {f'adaradar_r{sweep.tag(ratio).rstrip("0").rstrip("p")}_s{ADARADAR_BITS}' for ratio in RATIOS}
    return names | {f'ppo_{sweep.tag(budget)}' for budget in T.BUDGETS}


def replay(runtime, evaluator, rows, out):
    """Saved PPO and AdaRadar rows must come back exactly on the first of our cubes before the new frames open."""
    reference = pd.read_csv(PACKAGE/'REFERENCE_ROWS.csv').set_index(['record_name', 'method'])
    alpha = evaluator.alphas['ca']['a1']
    checks = []
    for _, row in rows.head(REPLAY_CUBES).iterrows():
        case = T.load_case(runtime, row)
        original, stats = case['original'], evaluator.stats
        allocation = learning.rollout(evaluator.actor, evaluator.fitted, case['features'], evaluator.frozen['prices']['1.4541'], deterministic=True)
        decoded, meta, *_ = sweep.project(runtime.codec, case['source'], case['order'], allocation['rho'], allocation['q'], 1.4541)
        spectrum, bits = adaradar.compress(original['spectrum'], 12.57, ADARADAR_BITS, stats['input_mean'], stats['input_std'])
        for method, received, bpcs in (('ppo_1p45', runtime.receive(decoded), float(meta['bpcs'])), ('adaradar_r12p57_s4', runtime.receive_spectrum(spectrum), bits['total']/case['cube'].numel())):
            metric = codec.detection_metrics(original['detector'], received['detector'], alpha, V13.MARGIN_DB)
            if (row.record_name, method) in reference.index:
                saved = reference.loc[(row.record_name, method)]
                checks.append(dict(record_name=row.record_name, method=method, saved_bpcs=float(saved.bpcs), bpcs=bpcs, saved_lost=int(saved.a1_lost_robust), lost=metric['lost_robust'], saved_new=int(saved.a1_new_robust), new=metric['new_robust']))
    table = pd.DataFrame(checks)
    agrees = bool(len(table)) and bool(((table.saved_bpcs-table.bpcs).abs() < 1e-6).all() and table.saved_lost.eq(table.lost).all() and table.saved_new.eq(table.new).all())
    common.write_json(out/'REPLAY.json', dict(checks=len(table), agrees=agrees, compared=not runtime.engineering, rows=json.loads(table.to_json(orient='records')) if len(table) else []))
    print(f'Replay of saved rows: {len(table)} checks, agrees={agrees}', flush=True)
    if not runtime.engineering and not agrees:
        raise ValueError('The saved rows did not come back; the new frames stay closed.')


def phase(runtime, evaluator, rows, name, score, net, out, log):
    """One population: every cube, then the tables the earlier runs wrote, in its own folder."""
    folder = out/name
    folder.mkdir()
    perception = Perception(runtime, net, own_scored())
    records, label_log = [], []
    for i, (_, row) in enumerate(rows.iterrows()):
        score(row, perception, records, label_log)
        if (i+1) % 25 == 0 or i+1 == len(rows):
            log(f'{name} {i+1}/{len(rows)}')
        if (i+1) % PARTIAL_EVERY == 0:
            pd.DataFrame(records).to_csv(folder/'RESULTS_BY_CUBE.partial.csv', index=False)
    frame = pd.DataFrame(records)
    frame['learned_chosen'] = frame.learned_chosen.astype(float)
    frame.to_csv(folder/'RESULTS_BY_CUBE.csv', index=False)
    (folder/'RESULTS_BY_CUBE.partial.csv').unlink(missing_ok=True)
    pd.DataFrame(label_log).to_csv(folder/'LABELS.csv', index=False)
    summary = V17.summarize(frame)
    summary.to_csv(folder/'SUMMARY.csv', index=False)
    V17.write_report(folder/'RECORDING_CI.json', V17.recording_bootstrap(frame))
    observed, _ = perception.report(folder)
    return frame, observed


def assess(official, observed, out):
    seen = observed.loc[observed.subset.eq('all')].set_index('method')
    robust = lambda g: 100*(1-g.a1_lost_robust.sum()/max(g.a1_original_robust.sum(), 1))
    own = seen.loc[f'adaradar_adaptive_own_s{ADARADAR_BITS}@adaradar_weights']
    fixed_ratio = seen.loc[f'adaradar_r12p57_s{ADARADAR_BITS}']
    pairs = {1.4541: 'adaradar_r12p57_s4', 2.1064: 'adaradar_r8_s4', 3.6327: 'adaradar_r4_s4', 4.7299: 'adaradar_r2_s4'}
    lead = {f'{budget:g}': float(robust(official.loc[official.method.eq(f'ppo_{sweep.tag(budget)}')])-robust(official.loc[official.method.eq(method)])) for budget, method in pairs.items()}
    passed = dict(P1=bool(abs(100*own.F1-PUBLISHED['adaptive']['F1']) <= 2), P2=bool(abs(100*(fixed_ratio.F1-seen.loc['original'].F1)-PUBLISHED['official_dF1_r12']) <= 2), P3=all(value > 0 for value in lead.values()))
    detail = dict(adaradar_own_F1=100*float(own.F1), adaradar_own_precision=100*float(own.mAP), adaradar_own_recall=100*float(own.mAR), adaradar_official_dF1_r12=100*float(fixed_ratio.F1-seen.loc['original'].F1), robust_lead=lead)
    V17.write_report(out/'PREDICTIONS.json', dict(predictions=PREDICTIONS, passed=passed, detail=detail, published=PUBLISHED))
    return passed


def run_check(runtime, out, expected, assignment):
    """CUDA pre-flight on one training cube: every method and both networks. It never opens a test frame."""
    path = Path(str(out)+'.CHECK.json')
    if path.exists():
        old = common.read_json(path)
        if old.get('passed') is True and old.get('context') == expected:
            print('Existing identity-bound check: PASS.', flush=True)
            return
        raise ValueError('Existing check identity differs. Preserve it; use a fresh deployment.')
    if data.TOLERANCE_US != 20000 or runtime.dataset.tolerance_us != 20000:
        raise AssertionError('Label synchronization must be the official 20 ms.')
    alphas = frozen_alphas()
    runtime.alpha = alphas['ca']['a1']
    net, _ = fftradnet.load(PACKAGE/V17.FFTRADNET_FILES['model'], PACKAGE/V17.FFTRADNET_FILES['config'], PACKAGE/WEIGHTS, runtime.device)
    evaluator = Evaluator(runtime, alphas, load_policy(runtime), net)
    perception, records, labels = Perception(runtime, net, own_scored()), [], []
    if runtime.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(runtime.device)
    row = runtime.roster('train').iloc[0]
    begun = time.time()
    with torch.no_grad():
        evaluator.official_cube(row, perception, records, labels)
    seconds = time.time()-begun
    begun = time.time()
    with torch.no_grad():
        evaluator.ours_cube(row, perception, records, labels)
    ours_seconds = time.time()-begun
    rates = {r['method']: round(r['bpcs'], 3) for r in records}
    roster = runtime.roster('test')
    official = roster.scene_name.isin(OFFICIAL)
    plan = dict(seconds_per_official_cube=seconds, seconds_per_our_cube=ours_seconds, official_cubes=int(official.sum()), our_cubes=int((~official).sum()), methods=len(rates), estimated_hours=(seconds*official.sum()+ours_seconds*(~official).sum())/3600)
    memory = torch.cuda.max_memory_allocated(runtime.device)/1e9 if runtime.device.type == 'cuda' else 0.
    result = dict(completed=True, passed=True, context=expected, record_name=row.record_name, plan=plan, rates=rates, gpu_peak_memory_gb=memory, training_started=False, test_evaluated=False)
    V17.write_report(path, result)
    print(json.dumps({k: result[k] for k in ('plan', 'gpu_peak_memory_gb', 'rates')}, indent=1), flush=True)
    print('Official-test CUDA check complete. Nothing kept, no test frame opened.', flush=True)


def run(runtime, out, expected, assignment):
    checked = common.read_json(Path(str(out)+'.CHECK.json'))
    if checked.get('context') != expected or checked.get('passed') is not True:
        raise ValueError('Run the matching CUDA check first.')
    if out.exists():
        raise FileExistsError('Preserve partial or completed results; never overwrite this study.')
    with common.exclusive(out):
        (out/'tables').mkdir(parents=True)
        common.write_json(out/'manifest.json', expected)
        start = time.time()
        log = lambda *a: print(*a, f'{time.time()-start:.0f}s', flush=True)
        alphas = frozen_alphas()
        runtime.alpha = alphas['ca']['a1']
        runtime.authorize_test(PACKAGE/'FROZEN_MODEL.json')
        net, _ = fftradnet.load(PACKAGE/V17.FFTRADNET_FILES['model'], PACKAGE/V17.FFTRADNET_FILES['config'], PACKAGE/WEIGHTS, runtime.device)
        evaluator = Evaluator(runtime, alphas, load_policy(runtime), net)
        roster = runtime.roster('test')
        official, ours = roster.loc[roster.scene_name.isin(OFFICIAL)], roster.loc[~roster.scene_name.isin(OFFICIAL)]
        log(f'{len(official)} official cubes, {len(ours)} of our cubes')
        with torch.no_grad():
            replay(runtime, evaluator, ours, out)
            frame, observed = phase(runtime, evaluator, official, 'official', evaluator.official_cube, net, out, log)
            phase(runtime, evaluator, ours, 'ours', evaluator.ours_cube, net, out, log)
        passed = assess(frame, observed, out)
        runtime.dataset.assert_unchanged()
        V17.write_report(out/'SMOKE_COMPLETED.json', dict(completed=True, passed=True, context=expected, seed=SEED, test_evaluated=True, test_used_for_selection=False, entropy_model_trained=False, alphas_db=common.read_json(PACKAGE/'H0_CALIBRATION.json')['alphas_db'],
                                                          official_cubes=len(official), our_cubes=len(ours), pre_registered_points_passed=passed, runtime_seconds=time.time()-start))
    print('Official-test run COMPLETE; start with PREDICTIONS.json, official/PERCEPTION_SUMMARY.csv and official/SUMMARY.csv:', out, flush=True)


def context(runtime, out, assignment, scan):
    hashes = V17.source_hashes()
    for path in (Path(__file__), Path(T.__file__), Path(sweep.__file__), Path(V20.__file__), Path(V18.__file__), Path(cap.__file__), Path(learning.__file__), Path(adaradar.__file__), Path(baselines.__file__), *(PACKAGE/name for name in (WEIGHTS, 'entropy_model_v20.pt', 'arm_0_update_0336.pt', 'feature_normalizer.pt', 'FROZEN_POLICY.json', 'H0_CALIBRATION.json', 'REFERENCE_ROWS.csv'))):
        hashes[str(path.resolve().relative_to(common.ROOT)).replace('\\', '/')] = common.sha(path)
    return dict(revision=REVISION, source_root=str(common.ROOT.resolve()), output=str(out), dataset_root=str(runtime.dataset.root), reference_root=str(runtime.dataset.reference_root), device=str(runtime.device), torch=torch.__version__,
                cuda_runtime=torch.version.cuda, gpu=torch.cuda.get_device_name(runtime.device) if runtime.device.type == 'cuda' else 'cpu-engineering', split=str(Path(assignment).resolve()), split_sha256=common.sha(assignment),
                inventory=scan, source_hashes=hashes, dataset_identity=data.dataset_identity(runtime.dataset), test_cubes=len(runtime.rows['test']), configuration=json.loads(json.dumps(CONFIG, default=str)), seed=SEED)


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
    table, scan = inventory(dataset, args.split_audit)
    with pd.option_context('display.width', 200, 'display.max_rows', 200):
        print(table.sort_values(['role', 'recording']).to_string(index=False))
    print(json.dumps(scan, indent=1), flush=True)
    if args.stage == 'count':
        if not Path(str(out)+'.COUNT.csv').exists():
            table.to_csv(Path(str(out)+'.COUNT.csv'), index=False)
            V17.write_report(Path(str(out)+'.COUNT.json'), scan)
        return
    if not scan['matches_frozen_split']:
        raise ValueError('The dataset folder does not hold the frozen recordings and all four official test recordings. See the inventory above.')
    if torch.device(args.device).type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('CUDA is required. There is no CPU fallback.')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    runtime = sweep.build_runtime(dataset, args.split_audit, args.device, rows=rosters(dataset, args.split_audit))
    expected = json.loads(json.dumps(context(runtime, out, args.split_audit, scan)))
    print(json.dumps({k: expected[k] for k in ('revision', 'gpu', 'test_cubes')}, indent=1), flush=True)
    if args.stage == 'check':
        run_check(runtime, out, expected, args.split_audit)
    else:
        run(runtime, out, expected, args.split_audit)


if __name__ == '__main__':
    main()
