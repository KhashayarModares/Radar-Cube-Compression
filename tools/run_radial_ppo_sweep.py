"""The frozen V12.40 PPO policy at every rate AdaRadar reports, on AdaRadar's own TEST cubes.

WHY. V12.40 tested PPO on 20 cubes that share no cube with the 662 TEST cubes of V17-V21, at
four rates between 2.1 and 3.6 bpcs. AdaRadar's saved results cannot be drawn against those
points. This run scores the policy on those 662 cubes with the unchanged V17 receiver,
labels, FFTRadNet and statistics, so its curve shares axes and cubes with C and AdaRadar.
AdaRadar is not rerun.

NOTHING IS TRAINED OR SELECTED. The policy (V12.40 update 40), its feature normalizer and the
tile codec are frozen. Budgets are per-cube maxima, as in V21: AdaRadar's six complete rates
0.4585, 0.8208, 1.4541, 2.1064, 3.6327, 4.7299 bpcs.

PRICE AT EACH BUDGET, from V12.40 validation only: the highest rate price whose uncapped mean
rate still reaches the budget (price 0: 3.81, 0.05: 3.44, 0.1: 3.04, 0.2: 2.39 bpcs), else
price 0. The four uncapped prices also run, under their training cap of 4.6 bpcs.

BELOW 2.1 BPCS the policy was never trained: its rates stop there, so these points are its
allocation pushed down by the cap projection of V12.40, which lowers q on the finest tiles.
They show what the projection does, not what the policy learned.

SEEN CUBES. 25 TRAIN and 4 VALIDATION cubes of V12.40 are among the 662, and 6 of the 7 TEST
recordings held V12.40 TRAIN cubes. All 662 are scored, because AdaRadar's saved FFTRadNet
metrics exist only for all of them. seen_by_ppo marks those 29, and PERCEPTION_UNSEEN.csv
repeats FFTRadNet without them. Recordings cannot be excluded: only one would remain.

RATE. Adaptive code length of the integers plus all side information, as V12.40. No file coder
exists for this tile codec, so unlike C the rates are not confirmed by real files.

PRE-REGISTERED (TEST): P1 at each AdaRadar rate PPO loses fewer labels and fewer robust CFAR
detections than AdaRadar; P2 every cube reaches its budget.

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
import zlib

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from radar_sim import radial_cap_codec as cap
from radar_sim import radial_cap_learning as learning
from radar_sim import radial_fftradnet_eval as fftradnet
from radar_sim.radial_adaptive_gpu_physics import static_features


def sibling(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V20 = sibling('run_radial_v20_rate_matched')  # runtime and perception of V17-V21
V17, V18 = V20.V17, V20.V18
V13, V14, common, data, codec = V17.V13, V17.V14, V17.common, V17.data, V17.codec

REVISION = '1.0.0-frozen-ppo-over-adaradar-rates'
PACKAGE = Path(__file__).resolve().parent/'ppo_sweep_gpu'
V17.PACKAGE = V18.PACKAGE = V20.PACKAGE = PACKAGE  # FFTRadNet files and the frozen split
SEED = V17.SEED
PPO_SEED = 20261022  # the seed of V12.40, so a cube gets the same dither and fill as there
POLICY = dict(file='ppo_policy_update_0040.pt', sha256='b30cd8f7a2f0dc4db9591d051c1d0b7377c0aea5526b98e4a6e2418a6995966b', source='V12.40 run 20261030T234421680Z')
NORMALIZER = 'feature_normalizer.pt'
ROSTERS = ('roster_train.csv', 'roster_validation.csv')
FREE_CAP, MAX_STEPS = cap.CAP_BPCS, 60
ADARADAR_RATES = (.4585, .8208, 1.4541, 2.1064, 3.6327, 4.7299)
tag = lambda bpcs: f'{bpcs:.2f}'.replace('.', 'p')
# Per price, budgets from high to low, so q carries down and the projection does less work.
PLAN = ((0., (('ppo_4p73', 4.7299), ('ppo_free_0', FREE_CAP), ('ppo_3p63', 3.6327))),
        (.05, (('ppo_free_0p05', FREE_CAP),)),
        (.1, (('ppo_free_0p1', FREE_CAP),)),
        (.2, (('ppo_free_0p2', FREE_CAP), ('ppo_2p11', 2.1064), ('ppo_1p45', 1.4541), ('ppo_0p82', .8208), ('ppo_0p46', .4585))))
ACCOUNTING = 'frozen PPO allocation, tile codec of V12.40: adaptive code length plus all side information'
# (AdaRadar complete rate): (labels lost, robust lost fraction, F1), TEST, frozen official network (V18/V20).
ADARADAR_REFERENCE = {.4585: (37, .5301, .6298), .8208: (21, .4004, .7323), 1.4541: (15, .2568, .7991),
                      2.1064: (11, .1646, .8272), 3.6327: (5, .0559, .8471), 4.7299: (2, .0280, .8546)}
PREDICTIONS = dict(P1='at each AdaRadar rate PPO loses fewer labels and fewer robust CFAR detections than AdaRadar', P2='every cube reaches its budget')
CONFIG = dict(plan=PLAN, adaradar_rates=ADARADAR_RATES, adaradar_reference=ADARADAR_REFERENCE, predictions=PREDICTIONS, policy=POLICY, H0_maps=V13.H0_MAPS,
              margin_db=V13.MARGIN_DB, label_gate=list(V13.LABEL_GATE), sync_tolerance_us=data.TOLERANCE_US, ppo_seed=PPO_SEED,
              test_policy='TEST unlocked once; nothing is trained or selected; AdaRadar is not rerun')


class Policy:
    def __init__(self, device):
        path = PACKAGE/POLICY['file']
        if common.sha(path) != POLICY['sha256']:
            raise ValueError('The frozen PPO checkpoint changed.')
        self.fitted = torch.load(PACKAGE/NORMALIZER, map_location=device, weights_only=False)
        self.actor = learning.DeltaPolicy(len(self.fitted['names']), device)
        self.actor.load_state_dict(torch.load(path, map_location=device, weights_only=False)['actor'])
        self.actor.eval()

    @torch.no_grad()
    def allocate(self, features, price):
        allocation = learning.rollout(self.actor, self.fitted, features, price, deterministic=True)
        return allocation['rho'], allocation['q']


def seen_cubes():
    return set().union(*(set(pd.read_csv(PACKAGE/name).record_name) for name in ROSTERS))


@torch.no_grad()
def project(physics, source, order, rho, q, budget):
    """The V12.40 cap projection at any budget: the finest tiles give up precision first,
    those with the fewest weak peaks before others. If q bottoms out the cube stays over budget."""
    decoded, meta = physics.encode(source, rho, q)
    steps = 0
    while float(meta['bpcs']) > budget and steps < MAX_STEPS:
        ranked = order[torch.argsort(-q[order], stable=True)]
        movable = ranked[q[ranked] > 1]
        if not len(movable):
            break
        share = min(1., max(.1, 3*(float(meta['bpcs'])-budget)/float(meta['bpcs'])))
        q = q.clone()
        q[movable[:max(1, math.ceil(share*len(q)))]] -= 1
        decoded, meta = physics.encode(source, rho, q)
        steps += 1
    return decoded, meta, q, steps


def evaluate_cube(runtime, policy, row, alphas, perception, label_log, seen):
    """All points on one cube, scored against the receivers' output on the ORIGINAL cube (V17 scoring)."""
    device, physics = runtime.device, runtime.codec
    cube = runtime.cube(row)
    numel = cube.numel()
    original = runtime.receive(cube)
    original_os = V17.receivers.cyclic_os_cfar(original['detector']['power'])
    bins, label_count = runtime.label_bins(row.record_name)
    perception.start(row, original)
    energy = original['spectrum'].abs().square().sum()
    source = physics.prepare(cube, seed=PPO_SEED+zlib.crc32(row.record_name.encode()) % 2**31)
    features = static_features(physics, source, physics.maps(cube))
    order = torch.argsort(features[:, learning.base.STATIC_NAMES.index('weak_peak_frac')], stable=True)
    records = []
    for price, points in PLAN:
        rho, q = policy.allocate(features, price)
        for name, budget in points:
            (decoded, meta, q, steps), encode_ms = V17.timed(device, project, physics, source, order, rho, q, budget)
            bits = dict(total=int(math.ceil(float(meta['realized_bits']))), accounting=ACCOUNTING, closed_loop_iterations=0)
            received = runtime.receive(decoded)
            record = V13.score(runtime, row, name, cube, original, decoded, bits, alphas['ca'], bins, label_count, label_log, received)
            method_os = V17.receivers.cyclic_os_cfar(received['detector']['power'])
            for key, alpha in alphas['os'].items():
                metrics = codec.detection_metrics(original_os, method_os, alpha, V13.MARGIN_DB)
                record.update({f'os_{key}_{k}': v for k, v in metrics.items()})
            record.update(price=price, budget_bpcs=budget, within_budget=float(meta['bpcs']) <= budget, projection_steps=steps, mean_rho=float(rho.mean()), mean_q=float(q.mean()),
                          seen_by_ppo=row.record_name in seen, spectrum_nmse=float((received['spectrum']-original['spectrum']).abs().square().sum()/energy), bpcs_kt=record['bpcs'],
                          bpcs_learned=math.nan, learned_chosen=0., codec_ms=encode_ms, encode_ms=encode_ms, decode_ms=math.nan, entropy_model_ms=math.nan)
            perception.add(name, received)
            records.append(record)
            del decoded, received
    return records


def unseen_perception(perception, records, seen, out):
    """FFTRadNet metrics without the cubes V12.40 used for training or checkpoint selection."""
    keep = [name not in seen for name in records]
    labelled = [f for f, k in zip(perception.frames, keep) if k and f['labelled']]
    rows = []
    for method in labelled[0]['counts'] if labelled else ():
        metric = fftradnet.metrics_from_counts(np.stack([f['counts'][method] for f in labelled]).sum(0))
        rows.append(dict(method=method, frames=len(labelled), **metric))
    frame = pd.DataFrame(rows)
    frame.to_csv(out/'PERCEPTION_UNSEEN.csv', index=False)
    return frame


def sweep_table(summary, perception, agreement, frame):
    test = summary.loc[summary.role.eq('test')].set_index('method')
    observed = perception.loc[perception.role.eq('test') & perception.subset.eq('all')].set_index('method')
    agree = agreement.loc[agreement.role.eq('test')].set_index('method')
    rows = []
    for method, g in frame.groupby('method', sort=False):
        r, s, a = test.loc[method], observed.loc[method], agree.loc[method]
        unseen = g.loc[~g.seen_by_ppo]
        rows.append(dict(method=method, price=g.price.iloc[0], budget_bpcs=g.budget_bpcs.iloc[0], bpcs=r.bpcs_mean, bpcs_max=r.bpcs_max, within_budget_fraction=g.within_budget.mean(),
                         mean_rho=g.mean_rho.mean(), mean_q=g.mean_q.mean(), projection_steps_mean=g.projection_steps.mean(), labels_detected_original=r.labels_detected_original,
                         labels_lost_strict=r.labels_lost, labels_lost_robust=r.labels_lost_robust, ca_robust_lost=r.a1_lost_robust_fraction, ca_strict_lost=r.a1_lost_fraction,
                         ca_robust_new_per_map=r.a1_new_robust_per_map, rd_rmse_db=r.rd_rmse_db_mean, nmse=r.nmse_mean, mAP=s.mAP, mAR=s.mAR, F1=s.F1, dF1=s.dF1, dF1_ci_low=s.get('dF1_ci_low'),
                         dF1_ci_high=s.get('dF1_ci_high'), objects_kept_fraction=a.objects_kept_fraction, unseen_cubes=len(unseen), unseen_labels_lost=int(unseen.labels_lost.sum()),
                         unseen_ca_robust_lost=unseen.a1_lost_robust.sum()/max(unseen.a1_original_robust.sum(), 1)))
    return pd.DataFrame(rows).sort_values('bpcs')


def predictions(sweep):
    table = sweep.set_index('method')
    detail = {}
    for rate, (labels, lost, _) in ADARADAR_REFERENCE.items():
        name = f'ppo_{tag(rate)}'
        if name in table.index:
            detail[f'{rate:.4f}'] = bool(table.loc[name].labels_lost_strict < labels and table.loc[name].ca_robust_lost < lost)
    budgets = {k: float(table.loc[k].within_budget_fraction) for k in table.index}
    return dict(P1=all(detail.values()), P2=all(v == 1 for v in budgets.values())), dict(P1=detail, P2=budgets)


def context(runtime, out, assignment, scan):
    hashes = V17.source_hashes()
    for path in (Path(__file__), Path(V20.__file__), Path(V18.__file__), Path(cap.__file__), Path(learning.__file__), PACKAGE/POLICY['file'], PACKAGE/NORMALIZER):
        hashes[str(path.resolve().relative_to(common.ROOT))] = common.sha(path)
    return dict(revision=REVISION, source_root=str(common.ROOT.resolve()), output=str(out), dataset_root=str(runtime.dataset.root), reference_root=str(runtime.dataset.reference_root),
                device=str(runtime.device), torch=torch.__version__, cuda_runtime=torch.version.cuda,
                gpu=torch.cuda.get_device_name(runtime.device) if runtime.device.type == 'cuda' else 'cpu-engineering',
                split=str(Path(assignment).resolve()), split_sha256=common.sha(assignment), split_provenance_sha256=common.sha(V17.provenance_path(assignment)), inventory=scan,
                source_hashes=hashes, dataset_identity=data.dataset_identity(runtime.dataset), populations={role: frame.record_name.tolist() for role, frame in runtime.rows.items() if role != 'train'},
                configuration=json.loads(json.dumps(CONFIG, default=str)), seed=SEED)


def run_check(runtime, out, expected):
    """CUDA pre-flight on one TRAIN cube: the policy, every point, the V17 scoring, FFTRadNet."""
    path = Path(str(out)+'.CHECK.json')
    if path.exists():
        old = common.read_json(path)
        if old.get('passed') is True and old.get('context') == expected:
            print('Existing identity-bound check: PASS.', flush=True)
            return
        raise ValueError('Existing check identity differs. Preserve it; use a fresh deployment.')
    if data.TOLERANCE_US != 20000 or runtime.dataset.tolerance_us != 20000:
        raise AssertionError('Label synchronization must be the official 20 ms.')
    device, p = runtime.device, runtime.physics
    alpha = codec.threshold_for_rate(codec.h0_scores(4, p.shape, p.range_window, p.doppler_window, V17.cyclic_cfar, device, SEED), 4, 1.)
    os_alpha = codec.threshold_for_rate(codec.h0_scores(4, p.shape, p.range_window, p.doppler_window, V17.receivers.cyclic_os_cfar, device, SEED), 4, 1.)
    alphas = dict(ca=dict(a1=alpha, a01=alpha), os=dict(a1=os_alpha, a01=os_alpha))
    row = runtime.roster('train').iloc[len(runtime.roster('train'))//2]
    perception, policy = V20.Perception(runtime), Policy(device)
    begun = time.time()
    records = evaluate_cube(runtime, policy, row, alphas, perception, [], seen_cubes())
    seconds = time.time()-begun
    rates = {r['method']: dict(bpcs=r['bpcs'], budget=r['budget_bpcs'], within=r['within_budget'], steps=r['projection_steps'], q=r['mean_q']) for r in records}
    print('Rates on one TRAIN cube:', json.dumps(rates), flush=True)
    if not all(v['within'] for v in rates.values()):
        raise AssertionError('A budget was not reached.')
    roster = len(runtime.roster('test'))
    result = dict(completed=True, passed=True, context=expected, record_name=row.record_name, rates=rates, seconds_per_cube=seconds, estimated_run_hours=seconds*roster/3600,
                  gpu_peak_memory_gb=torch.cuda.max_memory_allocated(device)/1e9 if device.type == 'cuda' else 0., training_started=False, test_evaluated=False)
    V17.write_report(path, result)
    print(json.dumps({k: result[k] for k in ('seconds_per_cube', 'estimated_run_hours', 'gpu_peak_memory_gb')}), flush=True)
    print('PPO sweep CUDA check complete.', flush=True)


def run(runtime, out, expected, log_every=25):
    checked = common.read_json(Path(str(out)+'.CHECK.json'))
    if checked.get('context') != expected or checked.get('passed') is not True:
        raise ValueError('Run the matching CUDA check first.')
    if out.exists():
        raise FileExistsError('Preserve partial/completed results; never overwrite this study.')
    with common.exclusive(out):
        out.mkdir(parents=True)
        common.write_json(out/'manifest.json', expected)
        runtime.rows['test'].to_csv(out/'POPULATIONS.csv', index=False)
        start = time.time()
        log = lambda *a: print(*a, f'{time.time()-start:.0f}s', flush=True)
        device = runtime.device
        policy = Policy(device)
        seen = seen_cubes()
        shutil.copyfile(PACKAGE/POLICY['file'], out/POLICY['file'])
        calibration, curve = V17.calibrate(runtime)
        V17.write_report(out/'H0_CALIBRATION.json', calibration)
        curve.to_csv(out/'H0_SURVIVAL.csv', index=False)
        alphas = dict(ca=calibration['alphas'], os=calibration['os_cfar']['alphas'])
        log('alpha dB CA:', json.dumps(calibration['alphas_db']), 'OS:', json.dumps(calibration['os_cfar']['alphas_db']))
        receipt = out/'FROZEN_MODEL.json'
        V17.write_report(receipt, dict(completed=True, checkpoint=POLICY['file'], checkpoint_sha256=common.sha(out/POLICY['file']), source=POLICY['source'], trained_in_this_run=False, test_used_for_selection=False))
        perception = V20.Perception(runtime)
        records, label_log = [], []
        runtime.authorize_test(receipt)
        roster = runtime.roster('test')
        log(f'TEST unlocked; {len(roster)} cubes; {sum(r["record_name"] in seen for r in roster.to_dict("records"))} seen by PPO')
        for i, (_, row) in enumerate(roster.iterrows()):
            records.extend(evaluate_cube(runtime, policy, row, alphas, perception, label_log, seen))
            if (i+1) % log_every == 0 or i+1 == len(roster):
                log(f'TEST {i+1}/{len(roster)}')
        log('Evaluation complete; writing reports.')
        frame = pd.DataFrame(records)
        frame.to_csv(out/'RESULTS_BY_CUBE.csv', index=False)
        pd.DataFrame(label_log).to_csv(out/'LABELS.csv', index=False)
        summary = V17.summarize(frame)
        summary.to_csv(out/'SUMMARY.csv', index=False)
        V17.write_report(out/'RECORDING_CI.json', V17.recording_bootstrap(frame))
        observed, agreement = perception.report(out)
        unseen_perception(perception, roster.record_name.tolist(), seen, out)
        sweep = sweep_table(summary, observed, agreement, frame)
        sweep.to_csv(out/'RATE_SWEEP_TEST.csv', index=False)
        passed, detail = predictions(sweep)
        V17.write_report(out/'PREDICTIONS.json', dict(predictions=PREDICTIONS, passed=passed, detail=detail))
        runtime.dataset.assert_unchanged()
        V17.write_report(out/'SMOKE_COMPLETED.json', dict(completed=True, passed=True, context=expected, seed=SEED, test_evaluated=True, test_used_for_selection=False, entropy_model_trained=False,
                                                          alphas_db=calibration['alphas_db'], cubes=len(roster), pre_registered_points_passed=passed, runtime_seconds=time.time()-start))
    print('PPO sweep run COMPLETE; start with RATE_SWEEP_TEST.csv and PREDICTIONS.json:', out, flush=True)


def build_runtime(dataset, assignment, device, rows=None):
    runtime = V20.Runtime(dataset, assignment, device, rows=rows)
    runtime.codec = cap.CapPhysics(dataset.reference_root, device=runtime.device, engineering=runtime.engineering)
    return runtime


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
    runtime = build_runtime(dataset, args.split_audit, args.device)
    expected = json.loads(json.dumps(context(runtime, out, args.split_audit, scan)))
    print(json.dumps({k: expected[k] for k in ('revision', 'gpu')}, indent=1), flush=True)
    if args.stage == 'check':
        run_check(runtime, out, expected)
    else:
        run(runtime, out, expected)


if __name__ == '__main__':
    main()
