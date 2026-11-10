"""PPO allocation on the UW raw-ADC automotive dataset, tuned on TRAIN and VALIDATION.

The V12.40 codec, cap, actor, critic, state, commands, start and update are unchanged and come from
tools/run_radial_cap_gpu.py and radar_sim/radial_cap_learning.py. Only what is listed here differs: the
training reward (radar_sim/uw_ppo_learning.py), the optimizer's learning rate, and labels limited to
+-60 degrees. At run time the policy sees the V12.40 state and nothing else.

TRAIN holds 14 sequences, VALIDATION 4 and TEST 4, split by whole sequence. Half of the TRAIN cubes are
evenly spaced labelled frames, half are frames where a fixed quantizer lost a robust detection or a label
(hard_frames_train.csv, frame ids of TRAIN sequences only). Three arms are trained: two learning rates with
the V12.40 state, and one with the same state without its CFAR-derived fields, which decides whether those
fields earn their place. As in V12.40, the arm and checkpoint are chosen on VALIDATION at price 0.2, by fewest
labels lost and then lowest rate, and TEST is opened only after that choice is frozen.

python -B tools/run_uw_ppo.py --stage check|run --dataset-root ... --out ...
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
import zlib

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('run_radial_cap_gpu', ROOT/'tools/run_radial_cap_gpu.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
cap, v13 = base.cap, base.v13
from radar_sim import uw_cfar_labels as labelling
from radar_sim import uw_detection_physics as uw
from radar_sim import uw_ppo_learning as learning
from radar_sim import uw_ppo_physics as physics_module
from radar_sim import uw_ramp_data as data
from radar_sim.radial_adaptive_gpu_physics import static_features

REVISION = 'uw.ppo.3.0.0-v12-40-model-tuned-reward-and-optimizer'
PACKAGE = ROOT/'tools/uw_ppo_gpu'
SEED = base.SEED
ROLES = ('train', 'validation', 'test')
EVEN_CUBES, HARD_CUBES, VALIDATION_CUBES = 8, 8, 15
TEST_STRIDE = 1
PASSES, BATCH, ROLLOUTS, EPOCHS, AUDIT_CUBES = 10, 5, 2, 5, 16
ARMS = (dict(learning_rate=1e-4, state='v12_40'), dict(learning_rate=3e-4, state='v12_40'), dict(learning_rate=3e-4, state='no_cfar'))
CHECKPOINT_SHARE = .1
H0_MAPS = 200
SEQUENCES, ALL_FRAMES = 22, 19754
UNIFORM, REFERENCE = base.UNIFORM, base.REFERENCE
LEGACY = 'Peak-scaled rho=.25,q=8'
read_json, write_json, csv, sha = base.read_json, base.write_json, base.csv, base.sha


def configuration(counts, passes=None):
    passes = passes or PASSES
    updates = passes*math.ceil(counts[0]/BATCH)
    steps = max(1, round(updates*CHECKPOINT_SHARE))
    return dict(train=int(counts[0]), validation=int(counts[1]), test=int(counts[2]), passes=passes, batch=BATCH, rollouts=ROLLOUTS, epochs=EPOCHS, audit_cubes=min(AUDIT_CUBES, int(counts[0])), updates=updates,
                checkpoints=sorted({0, updates, *range(steps, updates, steps)}), arms=list(ARMS))


def even(items, count):
    """``count`` evenly spaced items, the same ones on every run."""
    if len(items) <= count:
        return list(items)
    return [items[i] for i in np.linspace(0, len(items)-1, count).round().astype(int)]


def rosters(root, tables=None):
    """Frozen by rule: nothing here reads a radar frame or a VALIDATION or TEST outcome."""
    sequences = data.sequences(root)
    tables = tables if tables is not None else {}
    hard = pd.read_csv(PACKAGE/'hard_frames_train.csv')
    if not set(hard.sequence) <= set(data.SPLIT['train']):
        raise ValueError('The hard-frame list may name TRAIN sequences only.')
    rows = []
    for role in ROLES:
        records = []
        for s in sequences[role]:
            table = tables.setdefault(s.name, {f: physics_module.visible(r) for f, r in labelling.label_rows(s).items()})
            everything = [int(f) for f in s.frames]
            labelled = [f for f in everything if table[f]]
            if role == 'train':
                frames = even(labelled or everything, EVEN_CUBES)
                frames = sorted({*frames, *even([f for f in hard.loc[hard.sequence.eq(s.name)].frame.tolist() if f not in frames], HARD_CUBES)})
            elif role == 'validation':
                frames = even(labelled or everything, VALIDATION_CUBES)
            else:
                frames = everything[::TEST_STRIDE]
            records.extend(dict(record_name=f'{s.name}_{f:06d}', sequence=s.name, frame=f, split_role=role, scenario=data.scenario(s.name), motion=labelling.frame_motion(table[f])) for f in frames)
        rows.append(pd.DataFrame(records))
    return sequences, tables, rows


class Runtime(base.Runtime):
    def __init__(self, root, sequences, tables, rows, device, passes=None, engineering=False):
        self.device, self.engineering = torch.device(device), engineering
        if not engineering and (self.device.type != 'cuda' or not torch.cuda.is_available()):
            raise RuntimeError('CUDA required; no CPU numerical fallback.')
        self.root, self.tables, self.rows = Path(root), tables, rows
        self.lookup = {s.name: s for group in sequences.values() for s in group}
        self.config = configuration(tuple(map(len, rows)), passes)
        self.physics = physics_module.UWPPOPhysics(self.device, engineering=engineering)
        self.receiver = uw.UWPhysics(self.device, engineering=engineering)
        self.unmatched_scale, self.alpha = 1., None
        self.test_open = False
        self.test_records = set(rows[2].record_name)

    def cube(self, row):
        return torch.from_numpy(uw.load_frame(self.lookup[row.sequence].radar(row.frame))).to(self.device).to(torch.complex64)

    def case(self, row, *, test=False):
        if row.record_name in self.test_records and not self.test_open:
            raise PermissionError('Freeze validation selection before TEST access.')
        if bool(test) != (row.record_name in self.test_records):
            raise PermissionError('Requested label scope does not match the cube role.')
        cube = self.cube(row)
        source = self.physics.prepare(cube, seed=SEED+zlib.crc32(row.record_name.encode()) % 2**31)
        original = self.physics.maps(cube)
        bins = labelling.label_tensor(self.tables[row.sequence][int(row.frame)], self.device)
        return source, original, self.physics.label_positions(original, bins), static_features(self.physics, source, original)

    def outcome(self, source, original, labs, features, rho, q, price):
        decoded, meta, q, steps = self.project(source, features, rho, q)
        post = self.physics.maps(decoded)
        metric = self.physics.evaluate(original, post, labs)
        tile = physics_module.tile_terms(self.physics, original, post, metric, labs, self.alpha)
        meta['numel'] = source['cube'].numel()
        reward, terms = learning.reward_components(tile, meta, price, self.unmatched_scale, source['cube_energy'])
        record = dict(low_miss=int(metric['low_miss']), six_miss=int(metric['six_miss']), low_supported=int(metric['low_supported']), six_supported=int(metric['six_supported']),
                      RD_low_miss=int(metric['RD_low_miss']), RA_low_miss=int(metric['RA_low_miss']), RD_low_supported=int(metric['RD_low_supported']), RA_low_supported=int(metric['RA_low_supported']),
                      robust_label_miss=float(tile['miss'].sum()), robust_lost_tiles=float(tile['robust_lost'].sum()), robust_new_tiles=float(tile['robust_new'].sum()),
                      unmatched_excess=float(metric['unmatched_excess']), unmatched=float(metric['unmatched']), original_unmatched=float(metric['original_unmatched']),
                      bpcs=float(meta['bpcs']), nmse=float(meta['nmse']), projection_steps=steps, mean_rho=float(rho.mean()), mean_q=float(q.mean()),
                      mean_tile_reward=float(reward.mean()), **terms)
        return reward, record, decoded, rho, q


def alpha_scores(runtime, reference, decoded, bins):
    """The labels and detections at one false peak per noise map, with the robust margin."""
    received = runtime.receiver.receive(decoded)
    compared = labelling.compare(reference['found'], labelling.peaks(received, runtime.alpha), bins)
    detected, robust, kept = compared['detected'], compared['robust'], compared['kept']
    metric = v13.detection_metrics(reference['received']['detector'], received['detector'], runtime.alpha)
    return dict(labels=len(bins), labels_detected=int(detected.sum()), labels_lost=int((detected & ~kept).sum()), labels_robust=int(robust.sum()), labels_lost_robust=int((robust & ~kept).sum()),
                detections=metric['original'], detections_lost=metric['lost'], detections_new=metric['new'], robust=metric['original_robust'], robust_lost=metric['lost_robust'], robust_new=metric['new_robust'],
                unmatched_original_alpha=compared['unmatched_original'], unmatched_decoded_alpha=compared['unmatched_decoded'])


def reference_of(runtime, source):
    received = runtime.receiver.receive(source['cube'])
    return dict(received=received, found=labelling.peaks(received, runtime.alpha))


def context(runtime, out):
    files = [Path(__file__), ROOT/'tools/run_radial_cap_gpu.py', PACKAGE/'hard_frames_train.csv', *sorted((ROOT/'radar_sim').glob('*.py'))]
    return dict(revision=REVISION, source_root=str(ROOT), output=str(out), dataset_root=str(runtime.root), device=str(runtime.device), torch=torch.__version__,
                gpu=torch.cuda.get_device_name(runtime.device) if runtime.device.type == 'cuda' else 'cpu-engineering', seed=SEED, population_counts=list(map(len, runtime.rows)), configuration=runtime.config,
                bitrate_cap_bpcs=cap.CAP_BPCS, initial_allocation=[learning.INITIAL_RHO, learning.INITIAL_Q], rate_prices=list(learning.PRICES), selection_rate_price=learning.DEFAULT_PRICE,
                max_label_angle_deg=physics_module.MAX_LABEL_ANGLE, gates=dict(rd=physics_module.RD_GATE, ra=physics_module.RA_GATE), source_hashes={str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in files},
                populations={role: frame.record_name.tolist() for role, frame in zip(ROLES, runtime.rows)})


@torch.no_grad()
def prepare_features(runtime, out):
    panel = runtime.rows[0].iloc[even(list(range(len(runtime.rows[0]))), runtime.config['audit_cubes'])]
    generator = torch.Generator(device=runtime.device).manual_seed(SEED+41)
    matrix, counts = [], []
    for row in panel.itertuples():
        source, original, labs, features = runtime.case(row)
        detector = original['detectors']['RD']
        counts.append(float((detector['base_peaks'] & (detector['score'] > runtime.alpha)).sum()))
        n = len(features)
        for step in range(learning.ROUNDS):
            rho = torch.randint(1, 101, (n,), generator=generator, device=runtime.device).to(torch.float64)/100.
            q = torch.randint(1, 13, (n,), generator=generator, device=runtime.device).to(torch.float64)
            matrix.append(learning.raw_states(features, rho, q, learning.ROUNDS-step, learning.PRICES[step % len(learning.PRICES)]))
    runtime.unmatched_scale = max(1., float(torch.tensor(counts).median())/n)
    fitted = learning.fit_features(torch.cat(matrix))
    fitted['unmatched_scale'] = runtime.unmatched_scale
    csv(out/'tables/feature_selection.csv', fitted['rows'])
    torch.save(fitted, out/'feature_normalizer.pt')
    write_json(out/'OBSERVATION_CONTRACT.json', dict(raw_dimension=len(learning.NAMES), selected_dimension=len(fitted['names']), selected=fitted['names'], source_partition='TRAIN only'))
    write_json(out/'REWARD_CONTRACT.json', dict(unmatched_scale_per_tile=runtime.unmatched_scale, label_miss=8., global_miss='8/64 per tile', score_loss='per 3 dB of robust label score', robust_lost=learning.ROBUST_LOST_WEIGHT,
                                                robust_new=learning.ROBUST_NEW_WEIGHT, rate='5*price*tile_bits/(cap share)', nmse=.5, hard_bitrate_cap=cap.CAP_BPCS))
    return fitted


def selection_key(row):
    """V12.40's order: fewest misses, then lowest rate, at the selection price."""
    return row['labels_lost'], round(row['bpcs'], 4), row['robust_lost'], row['nmse']


@torch.no_grad()
def checkpoint(runtime, actor, fitted, arm, update, cases):
    rows = []
    for price in learning.PRICES:
        totals = dict(bpcs=0., labels=0, labels_lost=0, robust=0, robust_lost=0, robust_new=0, nmse=0.)
        for source, original, labs, features, reference in cases:
            allocation = learning.rollout(actor, fitted, features, price, deterministic=True)
            _, record, decoded, *_ = runtime.outcome(source, original, labs, features, allocation['rho'], allocation['q'], price)
            scores = alpha_scores(runtime, reference, decoded, labs['RA'])
            totals['bpcs'] += record['bpcs']
            totals['nmse'] += record['nmse']
            totals['labels'] += scores['labels_detected']
            for key in ('labels_lost', 'robust', 'robust_lost', 'robust_new'):
                totals[key] += scores[key]
        count = len(cases)
        row = dict(arm=arm, update=update, price=price, bpcs=totals['bpcs']/count, nmse=totals['nmse']/count, labels=totals['labels'], labels_lost=totals['labels_lost'], robust=totals['robust'], robust_lost=totals['robust_lost'],
                   labels_kept_pct=100*(1-totals['labels_lost']/max(totals['labels'], 1)), robust_kept_pct=100*(1-totals['robust_lost']/max(totals['robust'], 1)), robust_false_per_map=totals['robust_new']/count)
        rows.append(row)
        print(f'Validation arm {arm} update {update} price {price:g}: R={row["bpcs"]:.3f} labels {row["labels_kept_pct"]:.2f}% robust {row["robust_kept_pct"]:.2f}% false/map {row["robust_false_per_map"]:.3f}', flush=True)
    return rows


@torch.no_grad()
def query_edit(runtime, actor, fitted, source, original, labs, features, greedy, update):
    """Measure every command on one tile of the greedy state, with the rest of the policy recomputed."""
    step = (update-1) % learning.ROUNDS
    state = greedy['history'][step]
    tile = (37*(update-1)) % len(features)
    r, bits, mask = learning.alternatives(state['rho_before'][tile:tile+1], state['q_before'][tile:tile+1])
    feasible = mask[0]
    low, cost = features.new_full((learning.N_ACTIONS,), torch.inf), features.new_full((learning.N_ACTIONS,), torch.inf)
    following = greedy['history'][step+1] if step+1 < learning.ROUNDS else None
    records = []
    for action in torch.nonzero(feasible).flatten().tolist():
        rho = (following['rho_before'] if following is not None else greedy['rho']).clone()
        q = (following['q_before'] if following is not None else greedy['q']).clone()
        rho[tile], q[tile] = r[0, action], bits[0, action]
        suffix = learning.rollout(actor, fitted, features, greedy['price'], deterministic=True, rho=rho, q=q, start_step=step+1)
        _, record, *_ = runtime.outcome(source, original, labs, features, suffix['rho'], suffix['q'], greedy['price'])
        low[action], cost[action] = record['robust_label_miss'], -record['mean_tile_reward']
        records.append(dict(tile=tile, edit_round=step+1, action=action, robust_label_miss=record['robust_label_miss'], bpcs=record['bpcs'], secondary_cost=-record['mean_tile_reward']))
    chosen = learning.compatible_mask(low, cost, feasible)
    for record in records:
        record['teacher_selected'] = bool(chosen[record['action']])
    return dict(state=state['states'][tile].detach(), mask=feasible, compatible=chosen, probabilities_before=state['probabilities'][tile].detach()), records


def train_arm(runtime, fitted, out, arm, rate, cases):
    cfg = runtime.config
    torch.manual_seed(SEED)
    if runtime.device.type == 'cuda':
        torch.cuda.manual_seed_all(SEED)
    actor = learning.DeltaPolicy(len(fitted['names']), runtime.device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=rate, eps=1e-5, capturable=runtime.device.type == 'cuda')
    generator = torch.Generator(device=runtime.device).manual_seed(SEED+1)
    records, validation, queries_out, chosen = [], [], [], []

    def snapshot(update):
        state = torch.get_rng_state()
        rows = checkpoint(runtime, actor, fitted, arm, update, cases)
        torch.set_rng_state(state)
        path = out/'checkpoints'/f'arm_{arm}_update_{update:04d}.pt'
        path.parent.mkdir(exist_ok=True)
        torch.save(dict(actor=actor.state_dict(), arm=arm, update=update, learning_rate=rate, feature_names=fitted['names']), path)
        validation.extend(rows)
        chosen.extend((selection_key(row), arm, update, row['price'], path) for row in rows if row['price'] == learning.DEFAULT_PRICE)

    start = time.time()
    snapshot(0)
    update = 0
    for pass_index in range(cfg['passes']):
        order = torch.randperm(len(runtime.rows[0]), generator=generator, device=runtime.device).cpu().tolist()
        for first in range(0, len(order), cfg['batch']):
            update += 1
            trajectories, queries, logs = [], [], []
            for offset, position in enumerate(order[first:first+cfg['batch']]):
                row = runtime.rows[0].iloc[position]
                source, original, labs, features = runtime.case(row)
                for index in range(cfg['rollouts']):
                    price = learning.PRICES[(update+offset+index) % len(learning.PRICES)]
                    allocation = learning.rollout(actor, fitted, features, price)
                    reward, record, *_ = runtime.outcome(source, original, labs, features, allocation['rho'], allocation['q'], price)
                    trajectories.append((allocation, reward))
                    logs.append(dict(record, arm=arm, record_name=row.record_name, update=update, rate_price=price, pass_index=pass_index+1))
                    if offset == 0 and index == 0:
                        greedy = learning.rollout(actor, fitted, features, price, deterministic=True)
                        packet, measured = query_edit(runtime, actor, fitted, source, original, labs, features, greedy, update)
                        queries.append(packet)
                        queries_out.extend(dict(m, arm=arm, record_name=row.record_name, update=update) for m in measured)
            stats = learning.update(actor, optimizer, trajectories, queries, seed=SEED+update, epochs=cfg['epochs'])
            for log in logs:
                log.update({key: float(value) for key, value in stats.items()})
            records.extend(logs)
            mean = pd.DataFrame(logs)
            if update % 10 == 0 or update == 1:
                print(f'Arm {arm} update {update}/{cfg["updates"]}: robust label miss={mean.robust_label_miss.mean():.3f} robust lost={mean.robust_lost_tiles.mean():.2f} false={mean.robust_new_tiles.mean():.2f} R={mean.bpcs.mean():.3f} '
                      f'rho={mean.mean_rho.mean():.3f} q={mean.mean_q.mean():.2f} reward={mean.mean_tile_reward.mean():.4f} KL={float(stats["approx_kl"]):.5f} entropy={float(stats["entropy"]):.3f} {time.time()-start:.0f}s', flush=True)
            if update in cfg['checkpoints']:
                snapshot(update)
    return records, validation, queries_out, chosen


def train(runtime, fitted, out):
    cases = []
    for row in runtime.rows[1].itertuples():
        case = runtime.case(row)
        cases.append((*case, reference_of(runtime, case[0])))
    records, validation, queries, chosen = [], [], [], []
    for arm, setting in enumerate(ARMS):
        print(f'Training arm {arm}: learning rate {setting["learning_rate"]:g}, state {setting["state"]}', flush=True)
        for target, values in zip((records, validation, queries, chosen), train_arm(runtime, learning.state_variant(fitted, setting['state']), out, arm, setting['learning_rate'], cases)):
            target.extend(values)
    csv(out/'tables/training.csv', records)
    csv(out/'tables/validation.csv', validation)
    csv(out/'tables/conditional_queries.csv', queries)
    table = pd.DataFrame(validation)
    table = table.loc[table.price.eq(learning.DEFAULT_PRICE)].sort_values(['arm', 'labels_lost', 'bpcs', 'robust_lost']).groupby('arm', sort=False).head(1)
    table.assign(state=[ARMS[a]['state'] for a in table.arm], learning_rate=[ARMS[a]['learning_rate'] for a in table.arm]).to_csv(out/'STATE_COMPARISON.csv', index=False)
    key, arm, update, price, path = min(chosen, key=lambda item: (item[0], item[1], item[2], item[3]))
    write_json(out/'FROZEN_POLICY.json', dict(variant=REVISION, arm=arm, learning_rate=ARMS[arm]['learning_rate'], state=ARMS[arm]['state'], update=update, selected_price=price, checkpoint=str(path.relative_to(out)).replace('\\', '/'), checkpoint_sha256=sha(path),
                                              selection_key=list(key), test_used_for_selection=False))
    write_json(out/'TRAIN_COMPLETED.json', dict(completed=True, updates=runtime.config['updates'], arms=len(ARMS), test_evaluated=False))
    print(f'FROZEN: arm {arm} state {ARMS[arm]["state"]} update {update} price {price:g} key {key}', flush=True)


@torch.no_grad()
def final_reports(runtime, out, fitted):
    frozen = read_json(out/'FROZEN_POLICY.json')
    path = out/frozen['checkpoint']
    if sha(path) != frozen['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint changed.')
    fitted = learning.state_variant(fitted, frozen['state'])
    actor = learning.DeltaPolicy(len(fitted['names']), runtime.device)
    actor.load_state_dict(torch.load(path, map_location=runtime.device, weights_only=False)['actor'])
    runtime.test_open = True
    records, allocations = [], []
    start = time.time()
    for split, rows in zip(ROLES[1:], runtime.rows[1:]):
        for index, row in enumerate(rows.itertuples(), 1):
            source, original, labs, features = runtime.case(row, test=split == 'test')
            n = len(features)
            bins = labs['RA']
            reference = reference_of(runtime, source)
            tags = dict(split=split, record_name=row.record_name, sequence=row.sequence, frame=row.frame, scenario=row.scenario, motion=row.motion)

            def add(name, decoded, record):
                records.append(dict(tags, method=name, **record, **alpha_scores(runtime, reference, decoded, bins)))

            for price in learning.PRICES:
                allocation = learning.rollout(actor, fitted, features, price, deterministic=True)
                _, record, decoded, rho, q = runtime.outcome(source, original, labs, features, allocation['rho'], allocation['q'], price)
                add(f'PPO price={price:g}', decoded, record)
                if price == frozen['selected_price']:
                    allocations.extend(dict(split=split, record_name=row.record_name, tile=i, rho=float(rho[i]), q=int(q[i])) for i in range(n))
            for name, rho, q in UNIFORM:
                _, record, decoded, *_ = runtime.outcome(source, original, labs, features, features.new_full((n,), rho), features.new_full((n,), float(q)), frozen['selected_price'])
                add(name, decoded, record)
            decoded, meta = runtime.physics.legacy(source, features.new_full((n,), .25), features.new_full((n,), 8.))
            metric = runtime.physics.evaluate(original, runtime.physics.maps(decoded), labs)
            add(LEGACY, decoded, dict(low_miss=int(metric['low_miss']), six_miss=int(metric['six_miss']), low_supported=int(metric['low_supported']), six_supported=int(metric['six_supported']),
                                      RD_low_miss=int(metric['RD_low_miss']), RA_low_miss=int(metric['RA_low_miss']), RD_low_supported=int(metric['RD_low_supported']), RA_low_supported=int(metric['RA_low_supported']),
                                      unmatched_excess=float(metric['unmatched_excess']), bpcs=float(meta['bpcs']), nmse=float(meta['nmse'])))
            if index % 100 == 0 or index == len(rows):
                print(f'{split} reports {index}/{len(rows)} {time.time()-start:.0f}s', flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(out/'tables/method_by_cube.csv', index=False)
    csv(out/'tables/allocations.csv', allocations)
    summary = summarize(frame, ['split', 'method'])
    summary.to_csv(out/'SUMMARY.csv', index=False)
    summarize(frame, ['split', 'motion', 'method']).to_csv(out/'SUMMARY_BY_MOTION.csv', index=False)
    summarize(frame, ['split', 'sequence', 'method']).to_csv(out/'SUMMARY_BY_SEQUENCE.csv', index=False)
    with pd.option_context('display.width', 240, 'display.max_columns', 30):
        print(summary[['split', 'method', 'bpcs', 'compression_ratio', 'labels_preserved_pct', 'labels_lost', 'labels_detected', 'robust_kept_pct', 'robust_false_per_map', 'detections_new_per_map', 'nmse']].round(3).to_string(index=False), flush=True)
    return summary, frozen


def summarize(frame, keys):
    table = frame.groupby(keys, sort=False).agg(cubes=('bpcs', 'size'), bpcs=('bpcs', 'mean'), bpcs_max=('bpcs', 'max'), low_miss=('low_miss', 'sum'), low_supported=('low_supported', 'sum'), six_miss=('six_miss', 'sum'), six_supported=('six_supported', 'sum'),
                                                unmatched_excess=('unmatched_excess', 'mean'), nmse=('nmse', 'mean'), labels=('labels', 'sum'), labels_detected=('labels_detected', 'sum'), labels_lost=('labels_lost', 'sum'), labels_robust=('labels_robust', 'sum'),
                                                labels_lost_robust=('labels_lost_robust', 'sum'), detections=('detections', 'sum'), detections_lost=('detections_lost', 'sum'), robust=('robust', 'sum'), robust_lost=('robust_lost', 'sum'),
                                                robust_false_per_map=('robust_new', 'mean'), detections_new_per_map=('detections_new', 'mean')).reset_index()
    table['compression_ratio'] = uw.RAW_BPCS/table.bpcs
    table['labels_preserved_pct'] = 100*(1-table.labels_lost/table.labels_detected.clip(lower=1))
    table['robust_kept_pct'] = 100*(1-table.robust_lost/table.robust.clip(lower=1))
    return table


def assess(out, summary, frozen):
    table = summary.set_index(['split', 'method'])
    policy, reference = (table.loc[('test', name)] for name in (f'PPO price={frozen["selected_price"]:g}', REFERENCE))
    gates = dict(labels_at_least_as_uniform=int(policy.labels_lost) <= int(reference.labels_lost), within_cap_on_every_cube=float(policy.bpcs_max) <= cap.CAP_BPCS+1e-9, rate_lower_than_uniform=float(policy.bpcs) < float(reference.bpcs),
                 robust_kept_at_least_99=float(policy.robust_kept_pct) >= 99.)
    result = dict(gates=gates, passed=all(gates.values()), selected_price=frozen['selected_price'], policy_bpcs=float(policy.bpcs), reference=REFERENCE, reference_bpcs=float(reference.bpcs), test_used_for_selection=False, scope='Single seed')
    write_json(out/'SCIENTIFIC_GATES.json', result)
    return result


@torch.no_grad()
def run_check(runtime, out, expected):
    path = Path(str(out)+'.CHECK.json')
    if path.is_file():
        receipt = read_json(path)
        if receipt.get('context') != expected or receipt.get('passed') is not True:
            raise ValueError('Existing preflight has a different identity; use a fresh output.')
        return
    runtime.alpha = runtime.receiver.calibrate(20, SEED)
    row = next(runtime.rows[0].itertuples())
    source, original, labs, features = runtime.case(row)
    if len(source['tiles']) != physics_module.DOPPLER_TILES*physics_module.RANGE_TILES:
        raise AssertionError('Expected 64 tiles.')
    identity = runtime.physics.evaluate(original, original, labs)
    if int(identity['low_miss']) or int(identity['six_miss']):
        raise AssertionError('Original fails its own support identity test.')
    n = len(features)
    decoded, meta = runtime.physics.encode(source, features.new_ones(n), features.new_full((n,), 12.))
    if float(meta['nmse']) > 1e-3:
        raise AssertionError('Fine quantization did not reconstruct the cube.')
    base.assert_device(runtime, (source, original, features, decoded, meta, identity))
    response, begun = [], time.time()
    for name, rho, q in UNIFORM:
        reward, record, *_ = runtime.outcome(source, original, labs, features, features.new_full((n,), rho), features.new_full((n,), float(q)), .1)
        if reward.shape != (n,) or record['bpcs'] > cap.CAP_BPCS:
            raise AssertionError('Per-tile reward or the rate cap failed.')
        response.append(dict(record, name=name))
    seconds = (time.time()-begun)/len(response)
    csv(Path(str(out)+'.ACTION_RESPONSE.csv'), response)
    _, projected, _, steps = runtime.project(source, features, features.new_ones(n), features.new_full((n,), 12.))
    if not steps or float(projected['bpcs']) > cap.CAP_BPCS:
        raise AssertionError('The cap projection did not engage.')
    fitted = dict(indices=torch.arange(len(learning.NAMES), device=runtime.device), mean=features.new_zeros(len(learning.NAMES)), std=features.new_ones(len(learning.NAMES)), names=list(learning.NAMES))
    actor = learning.DeltaPolicy(len(learning.NAMES), runtime.device)
    allocation = learning.rollout(actor, fitted, features, .1)
    for entry in allocation['history']:
        torch.testing.assert_close(actor(entry['states'], entry['mask'])[0].log_prob(entry['actions']), entry['log_prob'], rtol=1e-12, atol=1e-12)
    cfg = runtime.config
    evaluations = len(ARMS)*(cfg['updates']*(BATCH*ROLLOUTS+21)+len(cfg['checkpoints'])*len(runtime.rows[1])*len(learning.PRICES)*1.5)+(len(runtime.rows[1])+len(runtime.rows[2]))*(len(learning.PRICES)+len(UNIFORM)+1)*1.5
    plan = dict(seconds_per_physical_evaluation=seconds, updates_per_arm=cfg['updates'], arms=len(ARMS), report_cubes=len(runtime.rows[1])+len(runtime.rows[2]), estimated_hours=seconds*evaluations/3600)
    rates = {r['name']: round(r['bpcs'], 3) for r in response}
    write_json(path, dict(completed=True, passed=True, context=expected, run_started=False, plan=plan, uniform_response=rates, cap_projection_steps=steps))
    print('Rates:', json.dumps(rates), flush=True)
    print('Time plan:', json.dumps(plan), flush=True)
    print('UW PPO check complete. No training started.', flush=True)


def run_all(runtime, out, expected, h0_maps=H0_MAPS):
    check = read_json(Path(str(out)+'.CHECK.json'))
    if check.get('context') != expected or check.get('passed') is not True:
        raise ValueError('Run requires a passing preflight for these exact identities.')
    if out.exists():
        raise FileExistsError('Preserve partial or completed results; use a fresh output.')
    (out/'tables').mkdir(parents=True)
    write_json(out/'manifest.json', expected)
    for role, frame in zip(ROLES, runtime.rows):
        frame.to_csv(out/'tables'/f'cube_index_{role}.csv', index=False)
    runtime.alpha = runtime.receiver.calibrate(h0_maps, SEED)
    write_json(out/'H0_CALIBRATION.json', dict(alpha=runtime.alpha, alpha_db=uw.alpha_db(runtime.alpha), maps=h0_maps, false_peaks_per_map=1.))
    fitted = prepare_features(runtime, out)
    train(runtime, fitted, out)
    summary, frozen = final_reports(runtime, out, fitted)
    scientific = assess(out, summary, frozen)
    write_json(out/'RUN_COMPLETED.json', dict(completed=True, passed=True, context=expected, test_used_for_selection=False, training_cubes=len(runtime.rows[0]), updates=runtime.config['updates'], selected_price=frozen['selected_price'], scientific_gate_passed=scientific['passed']))
    print('UW PPO COMPLETE; start with SUMMARY.csv, SUMMARY_BY_MOTION.csv, SCIENTIFIC_GATES.json:', out, flush=True)


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
    sequences, tables, rows = rosters(root)
    if sum(len(group) for group in sequences.values()) != SEQUENCES or sum(len(s.frames) for group in sequences.values() for s in group) != ALL_FRAMES:
        raise ValueError('The dataset differs from the sequences checked locally.')
    runtime = Runtime(root, sequences, tables, rows, device)
    print(json.dumps(dict(gpu=torch.cuda.get_device_name(device), cubes=dict(zip(ROLES, map(len, rows))), updates_per_arm=runtime.config['updates'], arms=ARMS)), flush=True)
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    expected = json.loads(json.dumps(context(runtime, out)))
    if args.stage == 'check':
        run_check(runtime, out, expected)
    else:
        run_all(runtime, out, expected)


if __name__ == '__main__':
    main()
