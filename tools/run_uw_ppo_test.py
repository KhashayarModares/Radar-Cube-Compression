"""One TEST pass of the PPO policy chosen on VALIDATION. Nothing is trained or tuned here.

The study (tools/run_uw_ppo.py) froze arm 0 update 90 by a rule that ignored false peaks. The policy here was
chosen from that study's VALIDATION table by a stricter rule: zero labels lost, at most 0.2 robust false peaks
per map, then the lowest rate. That gives arm 1, update 360, price 0.1. This run first replays VALIDATION and
stops unless it reproduces the table, then evaluates every frame of the 4 TEST sequences once. The earlier
policy and the baselines run on the same frames, with the same dither and fill, and must reproduce the study.

Two checks are added, because any decoded peak inside a label's gate keeps that label:
  * same-peak labels: a label stays only if a decoded peak lies within the NMS radius of an original peak supporting it;
  * empty gates: how often a decoded peak falls in a random label-sized gate that holds no original peak.

The classical references of the paper's RADIal comparison run on the same frames, fixed before TEST opens. Neither sees a
detector or a label:
  * BAQ at 1, 2 and 3 bits, on blocks of 15 chirps because 255 chirps do not divide by RADIal's 16;
  * Learned ECDQ: every cell coded, no pruning, no noise filling, and the encoder never decodes its own stream. The rate is
    the shorter of the per-tile adaptive code and the learned entropy model, which is the RADIal V20 checkpoint, unchanged.
    Each budget takes the finest step within it on every frame; the two fixed steps give the rates above the budgets.

python -B tools/run_uw_ppo_test.py --stage check|run --dataset-root ... --out ...
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import zlib

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('run_uw_ppo', ROOT/'tools/run_uw_ppo.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)
from radar_sim import radial_v14_entropy_model as entropy
from radar_sim import radial_v17_baselines as baselines

learning, labelling, uw, cap = study.learning, study.labelling, study.uw, study.cap
physics_module, data = study.physics_module, study.data
codec = cap.v13

REVISION = 'uw.ppo.test.2.0.0-arm-1-update-360-price-0p1-with-baq-and-learned-ecdq'
PACKAGE = ROOT/'tools/uw_ppo_test_gpu'
FROZEN = PACKAGE/'frozen'
SELECTED = dict(name='PPO selected price=0.1', arm=1, update=360, price=.1)
EARLIER = dict(name='PPO price=0.2', arm=0, update=90, price=.2)
FIXED = (('Uniform noise rho=1,q=5', 1., 5), ('Uniform noise rho=.5,q=6', .5, 6), ('Fixed noise rho=.25,q=5', .25, 5))
LEGACY = study.LEGACY
SELECTION_RULE = 'VALIDATION rows with zero labels lost and at most 0.2 robust false peaks per map, then the lowest rate.'
FALSE_LIMIT = .5
EMPTY_GATES, RANGE_SPAN = 400, (8, 120)
PARTIAL_EVERY = 500
BAQ_BITS, BAQ_BLOCK = (1, 2, 3), (15, 64)
baselines.BAQ_BLOCK = BAQ_BLOCK
ECDQ_BUDGETS, ECDQ_BETAS = (.82, 1.25, 2., 3.), (1.5, 1.)
ECDQ_LADDER = tuple(2.**(k/8) for k in range(-8, 57))  # the V21 ladder: beta .5 to 128, eight steps per octave
ECDQ_START = ECDQ_LADDER.index(1.)
ENTROPY_MODEL = dict(file='entropy_model_v20.pt', sha256='a9ba8d8a1d83894e4ca855b6b5ffcc05a7ce34ca09cd7f060619e8c0db1e5c65')
read_json, write_json, sha = study.read_json, study.write_json, study.sha


def frozen_hashes():
    return {path.name: sha(path) for path in sorted(FROZEN.iterdir())}


def load_entropy_model(device):
    path = FROZEN/ENTROPY_MODEL['file']
    if sha(path) != ENTROPY_MODEL['sha256']:
        raise ValueError('The V20 entropy-model checkpoint changed.')
    model = entropy.ContextModel().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.eval()


def ecdq_at(runtime, cube, beta, seed):
    """One whole-map stream and its rate: the shorter of the per-tile adaptive code and the learned model, plus one flag bit."""
    stream = codec.encode_ecdq(cube, beta=beta, seed=seed)
    bits = codec.ecdq_bits(stream)
    learned = entropy.learned_payload_bits(runtime.entropy_model, entropy.compact(stream))+bits['side']+bits['header']
    return stream, (min(bits['total'], learned)+1)/cube.numel()


def ladder_search(fits, start, count):
    """The finest ladder index that fits, walking from ``start``; the coarsest index if none does."""
    index = start
    if fits(index):
        while index > 0 and fits(index-1):
            index -= 1
        return index
    while index+1 < count:
        index += 1
        if fits(index):
            return index
    return index


def ecdq_within(runtime, cube, budget, seed, start):
    """The finest step within the budget, from the rate alone; each search starts at the previous frame's choice."""
    tried = {}

    def fits(index):
        if index not in tried:
            tried[index] = ecdq_at(runtime, cube, ECDQ_LADDER[index], seed)
        return tried[index][1] <= budget

    index = ladder_search(fits, start, len(ECDQ_LADDER))
    return index, tried[index]


def classical(runtime, source, sequence, name, starts):
    """BAQ and Learned ECDQ on one frame: method, decoded cube, rate and notes."""
    cube = source['cube']
    for bits in BAQ_BITS:
        decoded, rate = baselines.baq(cube, bits)
        yield f'BAQ {bits}-bit', decoded, rate['total']/cube.numel(), {}
    seed = study.SEED+zlib.crc32(name.encode()) % 2**31
    for budget in ECDQ_BUDGETS:
        index, (stream, bpcs) = ecdq_within(runtime, cube, budget, seed, starts.get((sequence, budget), ECDQ_START))
        starts[sequence, budget] = index
        yield f'Learned ECDQ budget={budget:g}', codec.decode_ecdq(stream, cube.dtype), bpcs, dict(beta=ECDQ_LADDER[index], budget=budget, within_budget=bpcs <= budget)
    for beta in ECDQ_BETAS:
        stream, bpcs = ecdq_at(runtime, cube, beta, seed)
        yield f'Learned ECDQ beta={beta:g}', codec.decode_ecdq(stream, cube.dtype), bpcs, dict(beta=beta)


def plain_record(metric, bpcs, nmse):
    """The run's score columns for a method that has no policy."""
    return dict(low_miss=int(metric['low_miss']), six_miss=int(metric['six_miss']), low_supported=int(metric['low_supported']), six_supported=int(metric['six_supported']),
                unmatched_excess=float(metric['unmatched_excess']), bpcs=float(bpcs), nmse=float(nmse))


def load_policy(runtime, fitted, setting):
    variant = learning.state_variant(fitted, study.ARMS[setting['arm']]['state'])
    saved = torch.load(FROZEN/f'arm_{setting["arm"]}_update_{setting["update"]:04d}.pt', map_location=runtime.device, weights_only=False)
    if saved['arm'] != setting['arm'] or saved['update'] != setting['update'] or saved['feature_names'] != variant['names']:
        raise ValueError('The checkpoint does not match its arm, update or state.')
    actor = learning.DeltaPolicy(len(variant['names']), runtime.device)
    actor.load_state_dict(saved['actor'])
    return dict(setting, actor=actor, fitted=variant)


def same_peak(bins, original, decoded):
    """Labels whose supporting original peak has no decoded peak within the NMS radius, ignoring decoded peaks elsewhere in the gate."""
    support = labelling.gated(bins, original)
    points, found = original['points'], decoded['points']
    near = torch.zeros(len(points), dtype=torch.bool, device=points.device)
    if len(points) and len(found):
        delta = (points[:, None]-found[None]).abs()
        delta[..., 1] = torch.minimum(delta[..., 1], uw.N_CHIRPS-delta[..., 1])
        near = (delta <= cap.v13.NMS_RADIUS).all(-1).any(1)
    kept = (support & near[None]).any(1)
    detected, robust = support.any(1), (support & original['stable'][None]).any(1)
    return int((detected & ~kept).sum()), int((robust & ~kept).sum())


def empty_gates(name, original, device):
    """Random label-sized gates inside the field of view that hold no original peak; the same ones for every method of a frame."""
    rng = np.random.default_rng(study.SEED+zlib.crc32(name.encode()) % 2**31)
    angles = np.flatnonzero(np.abs(data.ANGLE_GRID) <= physics_module.MAX_LABEL_ANGLE)
    gates = torch.as_tensor(np.stack([rng.integers(*RANGE_SPAN, EMPTY_GATES), rng.choice(angles, EMPTY_GATES)], 1), device=device)
    return gates[~labelling.gated(gates, original).any(1)]


def audit(runtime, reference, decoded, bins, empty):
    found = labelling.peaks(runtime.receiver.receive(decoded), runtime.alpha)
    lost, lost_robust = same_peak(bins, reference['found'], found)
    return dict(same_peak_lost=lost, same_peak_lost_robust=lost_robust, empty_gates=len(empty), empty_hits=int(labelling.gated(empty, found).any(1).sum()))


@torch.no_grad()
def replay(runtime, policy, expected, out):
    """Reproduce the VALIDATION row that chose this policy before TEST opens."""
    rows = []
    for row in runtime.rows[1].itertuples():
        source, original, labs, features = runtime.case(row)
        reference = study.reference_of(runtime, source)
        allocation = learning.rollout(policy['actor'], policy['fitted'], features, policy['price'], deterministic=True)
        _, record, decoded, *_ = runtime.outcome(source, original, labs, features, allocation['rho'], allocation['q'], policy['price'])
        rows.append(dict(record, **study.alpha_scores(runtime, reference, decoded, labs['RA'])))
    frame = pd.DataFrame(rows)
    found = dict(bpcs=frame.bpcs.mean(), nmse=frame.nmse.mean(), labels=int(frame.labels_detected.sum()), labels_lost=int(frame.labels_lost.sum()), robust=int(frame.robust.sum()),
                 robust_lost=int(frame.robust_lost.sum()), robust_false_per_map=frame.robust_new.mean())
    write_json(out/'VALIDATION_REPLAY.json', dict(found=found, expected=expected, compared=not runtime.engineering))
    print('VALIDATION replay:', json.dumps(found), flush=True)
    if not runtime.engineering and any(abs(found[key]-value) > 1e-6 for key, value in expected.items()):
        raise ValueError('VALIDATION does not reproduce the study; TEST stays closed.')


@torch.no_grad()
def report(runtime, policies, out):
    runtime.test_open = True
    records, allocations, starts = [], [], {}
    start = time.time()
    frames = runtime.rows[2]
    for index, row in enumerate(frames.itertuples(), 1):
        source, original, labs, features = runtime.case(row, test=True)
        n, bins = len(features), labs['RA']
        reference = study.reference_of(runtime, source)
        empty = empty_gates(row.record_name, reference['found'], runtime.device)
        tags = dict(record_name=row.record_name, sequence=row.sequence, frame=row.frame, scenario=row.scenario, motion=row.motion)

        def add(name, decoded, record, rho, q):
            records.append(dict(tags, method=name, **record, **study.alpha_scores(runtime, reference, decoded, bins), **audit(runtime, reference, decoded, bins, empty),
                                rho_mean=float(rho.mean()), rho_std=float(rho.std()), q_mean=float(q.mean()), q_std=float(q.std())))

        for policy in policies:
            allocation = learning.rollout(policy['actor'], policy['fitted'], features, policy['price'], deterministic=True)
            _, record, decoded, rho, q = runtime.outcome(source, original, labs, features, allocation['rho'], allocation['q'], policy['price'])
            add(policy['name'], decoded, record, rho, q)
            if policy['name'] == SELECTED['name']:
                allocations.extend(dict(record_name=row.record_name, tile=i, rho=float(rho[i]), q=int(q[i])) for i in range(n))
        for name, rho, q in FIXED:
            _, record, decoded, rho, q = runtime.outcome(source, original, labs, features, features.new_full((n,), rho), features.new_full((n,), float(q)), EARLIER['price'])
            add(name, decoded, record, rho, q)
        rho, q = features.new_full((n,), .25), features.new_full((n,), 8.)
        decoded, meta = runtime.physics.legacy(source, rho, q)
        metric = runtime.physics.evaluate(original, runtime.physics.maps(decoded), labs)
        add(LEGACY, decoded, plain_record(metric, meta['bpcs'], meta['nmse']), rho, q)
        rho, q = features.new_ones(n), features.new_zeros(n)
        for name, decoded, bpcs, notes in classical(runtime, source, row.sequence, row.record_name, starts):
            metric = runtime.physics.evaluate(original, runtime.physics.maps(decoded), labs)
            nmse = (decoded.to(torch.complex128)-source['cube'].to(torch.complex128)).abs().square().sum()/source['cube'].abs().square().sum()
            add(name, decoded, dict(plain_record(metric, bpcs, nmse), **notes), rho, q)
        if index % 100 == 0 or index == len(frames):
            print(f'test reports {index}/{len(frames)} {time.time()-start:.0f}s', flush=True)
        if index % PARTIAL_EVERY == 0:
            pd.DataFrame(records).to_csv(out/'tables/method_by_cube.partial.csv', index=False)
    frame = pd.DataFrame(records)
    frame.to_csv(out/'tables/method_by_cube.csv', index=False)
    (out/'tables/method_by_cube.partial.csv').unlink(missing_ok=True)
    study.csv(out/'tables/allocations.csv', allocations)
    summary = summarize(frame, ['method'])
    summary.to_csv(out/'SUMMARY.csv', index=False)
    summarize(frame, ['motion', 'method']).to_csv(out/'SUMMARY_BY_MOTION.csv', index=False)
    summarize(frame, ['sequence', 'method']).to_csv(out/'SUMMARY_BY_SEQUENCE.csv', index=False)
    with pd.option_context('display.width', 260, 'display.max_columns', 30):
        print(summary[['method', 'bpcs', 'compression_ratio', 'labels_lost', 'same_peak_lost', 'labels_detected', 'labels_preserved_pct', 'labels_same_peak_pct', 'robust_kept_pct', 'robust_false_per_map', 'detections_new_per_map', 'unmatched_excess', 'nmse', 'low_miss', 'empty_gate_pct']].round(3).to_string(index=False), flush=True)
    return summary


def summarize(frame, keys):
    table = study.summarize(frame, keys)
    extra = frame.groupby(keys, sort=False).agg(same_peak_lost=('same_peak_lost', 'sum'), same_peak_lost_robust=('same_peak_lost_robust', 'sum'), empty_gates=('empty_gates', 'sum'), empty_hits=('empty_hits', 'sum'),
                                                rho_mean=('rho_mean', 'mean'), rho_std=('rho_std', 'mean'), q_mean=('q_mean', 'mean'), q_std=('q_std', 'mean')).reset_index()
    table = table.merge(extra, on=keys, validate='one_to_one')
    table['labels_same_peak_pct'] = 100*(1-table.same_peak_lost/table.labels_detected.clip(lower=1))
    table['empty_gate_pct'] = 100*table.empty_hits/table.empty_gates.clip(lower=1)
    return table


def assess(runtime, out, summary, expected):
    table = summary.set_index('method')
    policy, reference = table.loc[SELECTED['name']], table.loc[study.REFERENCE]
    gates = dict(labels_at_least_as_uniform=int(policy.labels_lost) <= int(reference.labels_lost), same_peak_labels_at_least_as_uniform=int(policy.same_peak_lost) <= int(reference.same_peak_lost),
                 within_cap_on_every_cube=float(policy.bpcs_max) <= cap.CAP_BPCS+1e-9, rate_lower_than_uniform=float(policy.bpcs) < float(reference.bpcs),
                 robust_false_at_most_0p5=float(policy.robust_false_per_map) <= FALSE_LIMIT, robust_kept_at_least_99=float(policy.robust_kept_pct) >= 99.)
    reproduced = {} if runtime.engineering else {name: all(abs(float(table.loc[name, key])-value) <= 1e-6 for key, value in row.items()) for name, row in expected.items()}
    result = dict(gates=gates, passed=all(gates.values()), reproduces_study=reproduced, policy_bpcs=float(policy.bpcs), reference=study.REFERENCE, reference_bpcs=float(reference.bpcs), selection_rule=SELECTION_RULE, test_used_for_selection=False, scope='Single seed')
    write_json(out/'SCIENTIFIC_GATES.json', result)
    return result


def context(runtime, out):
    return dict(study.context(runtime, out), revision=REVISION, test_runner_sha256=sha(Path(__file__)), frozen_sha256=frozen_hashes(), selected={k: SELECTED[k] for k in ('arm', 'update', 'price')}, selection_rule=SELECTION_RULE,
                classical=dict(baq_bits=BAQ_BITS, baq_block=BAQ_BLOCK, ecdq_budgets=ECDQ_BUDGETS, ecdq_betas=ECDQ_BETAS, ecdq_ladder=[ECDQ_LADDER[0], ECDQ_LADDER[-1], len(ECDQ_LADDER)], entropy_model=ENTROPY_MODEL))


def prepare(runtime):
    runtime.alpha = read_json(FROZEN/'H0_CALIBRATION.json')['alpha']
    runtime.unmatched_scale = 1.
    uw.configure_codec()
    runtime.entropy_model = load_entropy_model(runtime.device)
    fitted = torch.load(FROZEN/'feature_normalizer.pt', map_location=runtime.device, weights_only=False)
    return [load_policy(runtime, fitted, setting) for setting in (SELECTED, EARLIER)]


@torch.no_grad()
def run_check(runtime, out, expected):
    path = Path(str(out)+'.CHECK.json')
    if path.is_file():
        receipt = read_json(path)
        if receipt.get('context') != expected or receipt.get('passed') is not True:
            raise ValueError('Existing preflight has a different identity; use a fresh output.')
        return
    policies = prepare(runtime)
    row = next(runtime.rows[1].itertuples())
    source, original, labs, features = runtime.case(row)
    reference = study.reference_of(runtime, source)
    empty = empty_gates(row.record_name, reference['found'], runtime.device)
    begun = time.time()
    n, rates = len(features), {}
    for policy in policies:
        allocation = learning.rollout(policy['actor'], policy['fitted'], features, policy['price'], deterministic=True)
        _, record, decoded, *_ = runtime.outcome(source, original, labs, features, allocation['rho'], allocation['q'], policy['price'])
        rates[policy['name']] = round(record['bpcs'], 3)
        study.alpha_scores(runtime, reference, decoded, labs['RA'])
        audit(runtime, reference, decoded, labs['RA'], empty)
    for name, rho, q in FIXED:
        _, record, decoded, *_ = runtime.outcome(source, original, labs, features, features.new_full((n,), rho), features.new_full((n,), float(q)), EARLIER['price'])
        rates[name] = round(record['bpcs'], 3)
    same = audit(runtime, reference, source['cube'], labs['RA'], empty)
    if same['same_peak_lost'] or same['empty_hits']:
        raise AssertionError('The original cube fails its own label checks.')
    seconds = (time.time()-begun)*(len(policies)+len(FIXED)+1)/(len(policies)+.5*len(FIXED))
    begun = time.time()
    for name, decoded, bpcs, _ in classical(runtime, source, row.sequence, row.record_name, {}):
        runtime.physics.evaluate(original, runtime.physics.maps(decoded), labs)
        study.alpha_scores(runtime, reference, decoded, labs['RA'])
        audit(runtime, reference, decoded, labs['RA'], empty)
        rates[name] = round(bpcs, 3)
    seconds += time.time()-begun
    plan = dict(seconds_per_cube=seconds, validation_cubes=len(runtime.rows[1]), test_cubes=len(runtime.rows[2]), training_updates=0, estimated_hours=seconds*len(runtime.rows[2])/3600+seconds*len(runtime.rows[1])/3600/5)
    write_json(path, dict(completed=True, passed=True, context=expected, run_started=False, plan=plan, uniform_response=rates))
    print('Rates:', json.dumps(rates), flush=True)
    print('Time plan:', json.dumps(plan), flush=True)
    print('UW PPO TEST check complete. Nothing trained, TEST not opened.', flush=True)


def run_all(runtime, out, expected):
    check = read_json(Path(str(out)+'.CHECK.json'))
    if check.get('context') != expected or check.get('passed') is not True:
        raise ValueError('Run requires a passing preflight for these exact identities.')
    if out.exists():
        raise FileExistsError('Preserve partial or completed results; use a fresh output.')
    (out/'tables').mkdir(parents=True)
    write_json(out/'manifest.json', expected)
    runtime.rows[2].to_csv(out/'tables/cube_index_test.csv', index=False)
    frozen = read_json(FROZEN/'EXPECTED.json')
    policies = prepare(runtime)
    replay(runtime, policies[0], frozen['validation'], out)
    write_json(out/'SELECTION_FROZEN.json', dict(variant=REVISION, **{k: SELECTED[k] for k in ('arm', 'update', 'price')}, learning_rate=study.ARMS[SELECTED['arm']]['learning_rate'], state=study.ARMS[SELECTED['arm']]['state'],
                                                 selection_rule=SELECTION_RULE, source_run=frozen['source_run'], test_used_for_selection=False))
    print('FROZEN before TEST:', json.dumps({k: SELECTED[k] for k in ('arm', 'update', 'price')}), flush=True)
    summary = report(runtime, policies, out)
    scientific = assess(runtime, out, summary, frozen['test'])
    write_json(out/'RUN_COMPLETED.json', dict(completed=True, passed=True, context=expected, test_used_for_selection=False, training_updates=0, scientific_gate_passed=scientific['passed'], reproduces_study=scientific['reproduces_study']))
    print('UW PPO TEST COMPLETE; start with SUMMARY.csv, SCIENTIFIC_GATES.json:', out, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', choices=('check', 'run'), required=True)
    for name in ('dataset-root', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('CUDA is mandatory; do not run this study on CPU.')
    torch.cuda.set_device(device)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = args.dataset_root.resolve()
    sequences, tables, rows = study.rosters(root)
    if sum(len(group) for group in sequences.values()) != study.SEQUENCES or sum(len(s.frames) for group in sequences.values() for s in group) != study.ALL_FRAMES:
        raise ValueError('The dataset differs from the sequences checked locally.')
    runtime = study.Runtime(root, sequences, tables, rows, device)
    print(json.dumps(dict(gpu=torch.cuda.get_device_name(device), validation=len(rows[1]), test=len(rows[2]), selected={k: SELECTED[k] for k in ('arm', 'update', 'price')})), flush=True)
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    expected = json.loads(json.dumps(context(runtime, out)))
    if args.stage == 'check':
        run_check(runtime, out, expected)
    else:
        run_all(runtime, out, expected)


if __name__ == '__main__':
    main()
