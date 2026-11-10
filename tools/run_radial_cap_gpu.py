"""V12.40: PPO allocation on RADIal at 20-ms synchronization under a hard 4.6-bpcs cap, scored by CFAR and FFTRadNet.

Same frozen 100/20/20 roster and seed as V12.38/V12.39. TEST is read once, after the checkpoint is frozen on validation.
Numerics, training, codec, reconstruction, CFAR and FFTRadNet run on CUDA; only file I/O and reports stay on the host.

python -B tools/run_radial_cap_gpu.py --stage check|run --dataset-root ... --reference-root ... --out ...
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time
import zlib

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from radar_sim import arm_e_adaptive_gpu_run as common
from radar_sim import radial_cap_codec as cap
from radar_sim import radial_cap_learning as learning
from radar_sim import radial_fftradnet_eval as fftradnet
from radar_sim import radial_sync20_gpu_data as data
from radar_sim import radial_v13_detection_codec as v13
from radar_sim.radial_adaptive_gpu_physics import cyclic_cfar, load_cube_gpu, static_features

REVISION = '12.40.0-RADIal-noise-referenced-cap-4p6-per-tile-reward'
PACKAGE = ROOT/'tools/v12_40_radial_cap_gpu'
SEED = common.SEED
read_json, write_json, csv, sha = common.read_json, common.write_json, common.csv, common.sha
WEAK = learning.base.STATIC_NAMES.index('weak_peak_frac')
UNIFORM = (('Uniform noise rho=1,q=5', 1., 5), ('Uniform noise rho=1,q=6', 1., 6), ('Uniform noise rho=.5,q=6', .5, 6))
REFERENCE = 'Uniform noise rho=1,q=5'
PERCEPTION_DROP_LIMIT = .02


def configuration(counts, passes):
    if len(counts) != 3 or any(int(n) != n or n < 1 for n in counts) or int(passes) != passes or passes < 1:
        raise ValueError('Three positive population counts and positive passes required.')
    updates = int(passes)*math.ceil(counts[0]/5)
    return dict(train=int(counts[0]), validation=int(counts[1]), test=int(counts[2]), passes=int(passes), batch=5, rollouts=2, epochs=5,
                audit_cubes=min(16, int(counts[0])), updates=updates,
                checkpoints=sorted({0, updates, *[max(1, round(updates*f)) for f in (.2, .4, .6, .8)]}))


def freeze_rosters(dataset, rows):
    scenes = [set(frame.scene_name) for frame in rows]
    if any(scenes[i] & scenes[j] for i in range(3) for j in range(i)):
        raise AssertionError('A frozen recording crosses roles.')
    dataset._adaptive_partitions = rows
    dataset._adaptive_test_records = set(rows[2].record_name)
    dataset._adaptive_authorized_test_records = set()
    dataset._splits = dict(zip(data.ROLES, rows))
    dataset.partition_audit = dict(recording_disjoint=True, used_cube_counts=[len(r) for r in rows],
                                   recordings={role: sorted(group) for role, group in zip(data.ROLES, scenes)},
                                   roster_source='Frozen V12.38 20-ms smoke roster', historical_test_previously_examined=True,
                                   publication_independent_test_claim=False)
    dataset.smoke_partition_audit = dataset.partition_audit
    return rows


def choose_rosters(dataset, counts):
    if tuple(counts) != (100, 20, 20):
        raise ValueError('This run keeps the frozen 100/20/20 roster.')
    available = dataset.index.set_index('record_name', drop=False)
    rows = []
    for role, count in zip(data.ROLES, counts):
        old = pd.read_csv(PACKAGE/f'roster_{role}.csv')
        missing = set(old.record_name)-set(available.index)
        if len(old) != count or old.record_name.duplicated().any() or missing:
            raise ValueError(f'Invalid or unavailable frozen roster {role}.')
        selected = available.loc[old.record_name].reset_index(drop=True)
        for before, after in zip(old.itertuples(), selected.itertuples()):
            fields = ['scene_name', 'frame_in_scene', 'global_frame', 'timestamp_us']
            fields += [f'chip{ch}_{key}' for ch in range(4) for key in ('ordinal', 'sample', 'offset')]
            if any(str(getattr(before, key)) != str(getattr(after, key)) for key in fields):
                raise AssertionError('Frozen 20-ms frame or chip identity changed.')
        selected['split_role'] = role
        rows.append(selected)
    return freeze_rosters(dataset, rows)


class Perception:
    """Official FFTRadNet on the reconstructed spectrum, on the same device as everything else."""

    def __init__(self, physics, device):
        self.physics = physics
        self.net, self.config = fftradnet.load(PACKAGE/'FFTRadNet.py', PACKAGE/'config_FFTRadNet_192_56.json', PACKAGE/'fftradnet_state_dict.pt', device)

    @torch.no_grad()
    def predict(self, cube):
        received = v13.rd_receive(cube, self.physics.range_window, self.physics.doppler_window, cyclic_cfar)
        detection, _ = fftradnet.predict(self.net, fftradnet.network_input(received['spectrum'], self.config)[None])
        return fftradnet.decode(detection[0], self.config)


class Runtime:
    def __init__(self, dataset, rows, device, passes, engineering=False):
        self.device, self.engineering = torch.device(device), engineering
        if not engineering and (self.device.type != 'cuda' or not torch.cuda.is_available()):
            raise RuntimeError('CUDA required; no CPU numerical fallback.')
        if dataset.tolerance_us != 20000:
            raise ValueError('This experiment requires 20-ms synchronization.')
        self.dataset, self.rows = dataset, rows
        self.config = configuration(tuple(map(len, rows)), passes)
        self.physics = cap.CapPhysics(dataset.reference_root, device=self.device, engineering=engineering)
        self.labels = data.scoped_labels(dataset, pd.concat(rows[:2], ignore_index=True))
        self.test_labels = None
        self.unmatched_scale = 1.
        self.allowed = set(pd.concat(rows).record_name)

    def case(self, row, *, test=False):
        if row.record_name not in self.allowed:
            raise PermissionError('Cube is outside the frozen roster.')
        data.assert_partition_access(self.dataset, pd.DataFrame([row.to_dict()]))
        if bool(test) != (row.record_name in set(self.rows[2].record_name)):
            raise PermissionError('Requested label scope does not match the cube role.')
        labels = self.test_labels if test else self.labels
        if labels is None:
            raise PermissionError('Freeze validation selection before TEST access.')
        cube = load_cube_gpu(row, self.device, engineering=self.engineering)
        source = self.physics.prepare(cube, seed=SEED+zlib.crc32(row.record_name.encode()) % 2**31)
        original = self.physics.maps(cube)
        return source, original, labels.loc[labels.record_name.eq(row.record_name)].reset_index(drop=True), static_features(self.physics, source, original)

    def project(self, source, features, rho, q):
        """Hard cap, part of the encoder: the finest tiles give up precision first, those with the fewest weak peaks before others."""
        order = torch.argsort(features[:, WEAK], stable=True)
        decoded, meta = self.physics.encode(source, rho, q)
        steps = 0
        while float(meta['bpcs']) > cap.CAP_BPCS:
            ranked = order[torch.argsort(-q[order], stable=True)]
            movable = ranked[q[ranked] > 1]
            if not len(movable) or steps == 40:
                raise AssertionError('The rate cap could not be reached.')
            share = min(1., max(.1, 3*(float(meta['bpcs'])-cap.CAP_BPCS)/float(meta['bpcs'])))
            q = q.clone()
            q[movable[:max(1, math.ceil(share*len(q)))]] -= 1
            decoded, meta = self.physics.encode(source, rho, q)
            steps += 1
        return decoded, meta, q, steps

    def outcome(self, source, original, labs, features, rho, q, price, *, detection=False):
        decoded, meta, q, steps = self.project(source, features, rho, q)
        post = self.physics.maps(decoded)
        metric = self.physics.evaluate(original, post, labs, detection_metrics=detection)
        tile = cap.tile_terms(self.physics, original, post, metric, labs)
        meta['numel'] = source['cube'].numel()
        reward, terms = learning.reward_components(tile, meta, metric, price, self.unmatched_scale, source['cube_energy'])
        record = dict(low_miss=int(metric['low_miss']), six_miss=int(metric['six_miss']), low_supported=int(metric['low_supported']),
                      unmatched_excess=float(metric['unmatched_excess']), unmatched=float(metric['unmatched']), original_unmatched=float(metric['original_unmatched']),
                      bpcs=float(meta['bpcs']), nmse=float(meta['nmse']), projection_steps=steps, mean_rho=float(rho.mean()), mean_q=float(q.mean()),
                      mean_tile_reward=float(reward.mean()), **terms)
        return reward, record, decoded, rho, q


def context(runtime, out):
    return dict(revision=REVISION, source_root=str(ROOT), output=str(out), dataset_root=str(runtime.dataset.root), reference_root=str(runtime.dataset.reference_root),
                device=str(runtime.device), gpu=torch.cuda.get_device_name(runtime.device) if runtime.device.type == 'cuda' else 'cpu-engineering', torch=torch.__version__,
                synchronization_tolerance_us=20000, seed=SEED, passes=runtime.config['passes'], population_counts=list(map(len, runtime.rows)), configuration=runtime.config,
                source_hashes={str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), *sorted((ROOT/'radar_sim').glob('*.py'))]},
                dataset_identity=data.dataset_identity(runtime.dataset), populations={role: frame.record_name.tolist() for role, frame in zip(data.ROLES, runtime.rows)},
                bitrate_cap_bpcs=cap.CAP_BPCS, initial_allocation=[learning.INITIAL_RHO, learning.INITIAL_Q], rate_prices=list(learning.PRICES), selection_rate_price=learning.DEFAULT_PRICE,
                checkpoint_order='Fewest LOW misses, then lowest rate, unmatched excess and NMSE', receiver_changed=False, independent_final_test_claim=False,
                roster_hashes={role: sha(PACKAGE/f'roster_{role}.csv') for role in data.ROLES}, fftradnet_weights_sha256=sha(PACKAGE/'fftradnet_state_dict.pt'))


def assert_device(runtime, value):
    if runtime.engineering:
        return
    if isinstance(value, torch.Tensor):
        if not value.is_cuda:
            raise AssertionError('A numerical tensor left CUDA.')
    elif isinstance(value, dict):
        for child in value.values():
            assert_device(runtime, child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            assert_device(runtime, child)


def label_audit(dataset, rows):
    """Check the annotations against the exact source CSV rows; never shift indices."""
    data.assert_partition_access(dataset, rows)
    labels = data.scoped_labels(dataset, rows)
    if labels.source_csv_row.duplicated().any():
        raise AssertionError('Scoped annotation source rows are duplicated.')
    raw = pd.read_csv(dataset.label_path)
    identities = rows.set_index('record_name')
    records = []
    for lab in labels.itertuples():
        source, row = raw.iloc[int(lab.source_csv_row)], identities.loc[lab.record_name]
        if str(source['dataset']) != str(row.scene_name) or int(source['index']) != int(row.frame_in_scene) or int(source['numSample']) != int(row.global_frame):
            raise AssertionError('Annotation source CSV/frame identity mismatch.')
        records.append(dict(record_name=lab.record_name, object_id=lab.object_id, source_csv_row=int(lab.source_csv_row), frame_index=int(row.frame_in_scene),
                            range_m=float(lab.range_m), azimuth_deg=float(lab.azimuth_deg), cyclic_doppler_bin=float(lab.doppler_bin)))
    return records


@torch.no_grad()
def prepare_features(runtime, out):
    panel = common.historical.diverse_rows(runtime.rows[0], runtime.config['audit_cubes'])
    generator = torch.Generator(device=runtime.device).manual_seed(SEED+41)
    matrix, counts = [], []
    for _, row in panel.iterrows():
        source, original, labs, features = runtime.case(row)
        counts.append(float(runtime.physics.evaluate(original, original, labs, detection_metrics=False)['original_unmatched']))
        n = len(features)
        for step in range(learning.ROUNDS):
            rho = torch.randint(1, 101, (n,), generator=generator, device=runtime.device).to(torch.float64)/100.
            q = torch.randint(1, 13, (n,), generator=generator, device=runtime.device).to(torch.float64)
            matrix.append(learning.raw_states(features, rho, q, learning.ROUNDS-step, learning.PRICES[step % len(learning.PRICES)]))
        print('TRAIN feature audit:', row.record_name, flush=True)
    runtime.unmatched_scale = max(1., float(torch.tensor(counts).median())/len(features))
    fitted = learning.fit_features(torch.cat(matrix))
    fitted['unmatched_scale'] = runtime.unmatched_scale
    csv(out/'tables/feature_selection.csv', fitted['rows'])
    csv(out/'tables/feature_audit_roster.csv', panel)
    torch.save(fitted, out/'feature_normalizer.pt')
    write_json(out/'OBSERVATION_CONTRACT.json', dict(raw_dimension=len(learning.NAMES), selected_dimension=len(fitted['names']), selected=fitted['names'], source_partition='TRAIN only'))
    write_json(out/'REWARD_CONTRACT.json', dict(unmatched_scale_per_tile=runtime.unmatched_scale, label_miss=8., score_loss='per 3 dB of label CFAR score', global_miss='8/64 per tile',
                                                rate='5*price*tile_bits/(cap share)', nmse=.5, hard_bitrate_cap=cap.CAP_BPCS))
    return fitted


@torch.no_grad()
def evaluate(runtime, actor, fitted, rows, price, update):
    records = []
    for _, row in rows.iterrows():
        source, original, labs, features = runtime.case(row)
        allocation = learning.rollout(actor, fitted, features, price, deterministic=True)
        _, record, *_ = runtime.outcome(source, original, labs, features, allocation['rho'], allocation['q'], price)
        records.append(dict(record, record_name=row.record_name, update=update))
    frame = pd.DataFrame(records)
    print(f'Validation {update}: LOW={int(frame.low_miss.sum())} SIX={int(frame.six_miss.sum())} U+={frame.unmatched_excess.mean():.1f} '
          f'R={frame.bpcs.mean():.3f} max={frame.bpcs.max():.3f} NMSE={frame.nmse.mean():.4f}', flush=True)
    return records


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
        low[action], cost[action] = record['low_miss'], -record['mean_tile_reward']
        records.append(dict(tile=tile, edit_round=step+1, action=action, low_miss=record['low_miss'], bpcs=record['bpcs'], secondary_cost=-record['mean_tile_reward']))
    chosen = learning.compatible_mask(low, cost, feasible)
    for record in records:
        record['teacher_selected'] = bool(chosen[record['action']])
    return dict(state=state['states'][tile].detach(), mask=feasible, compatible=chosen, probabilities_before=state['probabilities'][tile].detach()), records


def train(runtime, fitted, out):
    cfg = runtime.config
    torch.manual_seed(SEED)
    if runtime.device.type == 'cuda':
        torch.cuda.manual_seed_all(SEED)
    actor = learning.DeltaPolicy(len(fitted['names']), runtime.device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=learning.LEARNING_RATE, eps=1e-5, capturable=runtime.device.type == 'cuda')
    generator = torch.Generator(device=runtime.device).manual_seed(SEED+1)
    validation, train_records, query_records, selections = [], [], [], []

    def checkpoint(update):
        rng = torch.get_rng_state()
        rows = evaluate(runtime, actor, fitted, runtime.rows[1], learning.DEFAULT_PRICE, update)
        torch.set_rng_state(rng)
        validation.extend(rows)
        frame = pd.DataFrame(rows)
        key = [int(frame.low_miss.sum()), round(float(frame.bpcs.mean()), 4), float(frame.unmatched_excess.mean()), float(frame.nmse.mean())]
        path = out/'checkpoints'/f'update_{update:04d}.pt'
        path.parent.mkdir(exist_ok=True)
        torch.save(dict(actor=actor.state_dict(), update=update, selection_key=key, feature_names=fitted['names']), path)
        selections.append((key, update, path))

    start = time.time()
    checkpoint(0)
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
                    logs.append(dict(record, record_name=row.record_name, update=update, rate_price=price, pass_index=pass_index+1))
                    if offset == 0 and index == 0:
                        greedy = learning.rollout(actor, fitted, features, price, deterministic=True)
                        packet, measured = query_edit(runtime, actor, fitted, source, original, labs, features, greedy, update)
                        queries.append(packet)
                        query_records.extend(dict(m, record_name=row.record_name, update=update) for m in measured)
            stats = learning.update(actor, optimizer, trajectories, queries, seed=SEED+update, epochs=cfg['epochs'])
            for log in logs:
                log.update({key: float(value) for key, value in stats.items()})
            train_records.extend(logs)
            mean = pd.DataFrame(logs)
            print(f'Update {update}/{cfg["updates"]}: LOW={mean.low_miss.mean():.3f} U+={mean.unmatched_excess.mean():.1f} R={mean.bpcs.mean():.3f} rho={mean.mean_rho.mean():.3f} '
                  f'q={mean.mean_q.mean():.2f} reward={mean.mean_tile_reward.mean():.4f} KL={float(stats["approx_kl"]):.5f} entropy={float(stats["entropy"]):.3f} {time.time()-start:.0f}s', flush=True)
            if update in cfg['checkpoints']:
                checkpoint(update)
    csv(out/'tables/training.csv', train_records)
    csv(out/'tables/validation.csv', validation)
    csv(out/'tables/conditional_queries.csv', query_records)
    key, selected, path = min(selections, key=lambda item: (item[0], item[1]))
    write_json(out/'FROZEN_POLICY.json', dict(arm='E', variant=REVISION, update=selected, checkpoint=str(path.relative_to(out)), checkpoint_sha256=sha(path),
                                              selection_key=key, selection_rate_price=learning.DEFAULT_PRICE, test_used_for_selection=False))
    write_json(out/'TRAIN_COMPLETED.json', dict(completed=True, updates=update, test_evaluated=False))


@torch.no_grad()
def final_reports(runtime, out, fitted):
    frozen = read_json(out/'FROZEN_POLICY.json')
    path = out/frozen['checkpoint']
    if sha(path) != frozen['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint changed.')
    actor = learning.DeltaPolicy(len(fitted['names']), runtime.device)
    actor.load_state_dict(torch.load(path, map_location=runtime.device, weights_only=False)['actor'])
    perception = Perception(runtime.physics, runtime.device)
    runtime.test_labels = data.scoped_labels(runtime.dataset, runtime.rows[2], frozen=out/'FROZEN_POLICY.json')
    csv(out/'tables/label_CSV_join_test.csv', label_audit(runtime.dataset, runtime.rows[2]))
    records, allocations, summaries = [], [], []
    for split, rows in (('validation', runtime.rows[1]), ('test', runtime.rows[2])):
        names = ['Original', *(f'PPO price={p:g}' for p in learning.PRICES), *(u[0] for u in UNIFORM), 'Legacy peak-scaled rho=.25,q=8']
        counts = {name: 0 for name in names}
        agreement = {name: [0, 0] for name in names}
        labelled = 0
        for index, (_, row) in enumerate(rows.iterrows(), 1):
            source, original, labs, features = runtime.case(row, test=split == 'test')
            n = len(features)
            cube = source['cube']
            reference = perception.predict(cube)
            frame_labels = labs.loc[labs.Difficult.eq(0)][['range_m', 'azimuth_deg']].to_numpy(dtype=float) if 'Difficult' in labs else labs[['range_m', 'azimuth_deg']].to_numpy(dtype=float)
            usable = len(frame_labels) > 0 or bool(getattr(row, 'explicit_negative', False))
            labelled += usable

            def score(name, decoded, record):
                prediction = perception.predict(decoded)
                if usable:
                    counts[name] = counts[name]+fftradnet.frame_counts(prediction, frame_labels)
                agree = fftradnet.agreement(reference, prediction)
                agreement[name][0] += agree['objects_kept']
                agreement[name][1] += agree['objects_original']
                records.append(dict(record, split=split, method=name, record_name=row.record_name, objects_kept=agree['objects_kept'], objects_original=agree['objects_original']))

            metric = runtime.physics.evaluate(original, original, labs, detection_metrics=False)
            score('Original', cube, dict(low_miss=0, six_miss=0, low_supported=int(metric['low_supported']), unmatched_excess=0., bpcs=32., nmse=0.))
            for price in learning.PRICES:
                allocation = learning.rollout(actor, fitted, features, price, deterministic=True)
                _, record, decoded, rho, q = runtime.outcome(source, original, labs, features, allocation['rho'], allocation['q'], price)
                score(f'PPO price={price:g}', decoded, record)
                if price == learning.DEFAULT_PRICE:
                    allocations.extend(dict(split=split, record_name=row.record_name, tile=i, rho=float(rho[i]), q=int(q[i])) for i in range(n))
            for name, rho, q in UNIFORM:
                _, record, decoded, *_ = runtime.outcome(source, original, labs, features, features.new_full((n,), rho), features.new_full((n,), float(q)), learning.DEFAULT_PRICE)
                score(name, decoded, record)
            decoded, meta = runtime.physics.legacy(source, features.new_full((n,), .25), features.new_full((n,), 8.))
            metric = runtime.physics.evaluate(original, runtime.physics.maps(decoded), labs, detection_metrics=False)
            score(names[-1], decoded, dict(low_miss=int(metric['low_miss']), six_miss=int(metric['six_miss']), low_supported=int(metric['low_supported']),
                                           unmatched_excess=float(metric['unmatched_excess']), bpcs=float(meta['bpcs']), nmse=float(meta['nmse'])))
            print(f'{split} reports {index}/{len(rows)}', flush=True)
        for name in names:
            summaries.append(dict(split=split, method=name, labelled_frames=int(labelled), objects_kept=agreement[name][0], objects_original=agreement[name][1],
                                  **fftradnet.metrics_from_counts(counts[name])))
    csv(out/'tables/method_by_cube.csv', records)
    csv(out/'tables/allocations.csv', allocations)
    frame = pd.DataFrame(records)
    grouped = frame.groupby(['split', 'method'], sort=False).agg(cubes=('bpcs', 'size'), bpcs=('bpcs', 'mean'), bpcs_max=('bpcs', 'max'), low_miss=('low_miss', 'sum'), six_miss=('six_miss', 'sum'),
                                                                 low_supported=('low_supported', 'sum'), unmatched_excess=('unmatched_excess', 'mean'), nmse=('nmse', 'mean')).reset_index()
    summary = grouped.merge(pd.DataFrame(summaries), on=['split', 'method'])
    summary.to_csv(out/'SUMMARY.csv', index=False)
    with pd.option_context('display.width', 220, 'display.max_columns', 20):
        print(summary[['split', 'method', 'bpcs', 'bpcs_max', 'low_miss', 'six_miss', 'unmatched_excess', 'nmse', 'mAP', 'F1']].round(4).to_string(index=False), flush=True)
    return summary


def assess(out, summary):
    table = summary.set_index(['split', 'method'])
    policy, reference, original = (table.loc[('test', name)] for name in (f'PPO price={learning.DEFAULT_PRICE:g}', REFERENCE, 'Original'))
    gates = dict(zero_low_misses=int(policy.low_miss) == 0, within_cap_on_every_cube=float(policy.bpcs_max) <= cap.CAP_BPCS+1e-9,
                 rate_lower_than_uniform=float(policy.bpcs) < float(reference.bpcs), unmatched_excess_not_worse=float(policy.unmatched_excess) <= float(reference.unmatched_excess),
                 f1_within_two_points_of_original=float(policy.F1) >= float(original.F1)-PERCEPTION_DROP_LIMIT)
    result = dict(gates=gates, passed=all(gates.values()), reference=REFERENCE, nmse_reported_not_gated=dict(policy=float(policy.nmse), reference=float(reference.nmse)),
                  test_used_for_selection=False, scope='Single seed on 20 previously examined TEST cubes; no population generalization claim')
    write_json(out/'SMOKE_SCIENTIFIC_GATES.json', result)
    return result


@torch.no_grad()
def run_check(runtime, out, expected):
    check = Path(str(out)+'.CHECK.json')
    if check.is_file():
        receipt = read_json(check)
        if receipt.get('context') != expected or receipt.get('passed') is not True:
            raise ValueError('Existing preflight has a different identity; use a fresh output.')
        return
    if not bool(runtime.dataset.sync20_pairing_audit.all_four_chip_identities_verified.all()):
        raise AssertionError('Not every eligible frame passed the 20-ms four-chip check.')
    labels = label_audit(runtime.dataset, pd.concat(runtime.rows[:2], ignore_index=True))
    annotated = runtime.rows[0].loc[runtime.rows[0].record_name.isin(runtime.labels.record_name)]
    row = annotated.iloc[0]
    source, original, labs, features = runtime.case(row)
    identity = runtime.physics.evaluate(original, original, labs, detection_metrics=False)
    if int(identity['low_miss']) or int(identity['six_miss']):
        raise AssertionError('Original fails its own support identity test.')
    n = len(features)
    decoded, meta = runtime.physics.encode(source, features.new_ones(n), features.new_full((n,), 12.))
    if float(meta['nmse']) > 1e-3:
        raise AssertionError('Fine quantization did not reconstruct the cube.')
    assert_device(runtime, (source, original, features, decoded, meta, identity))
    response, begun = [], time.time()
    for rho, q in ((1., 5), (1., 6), (.5, 6), (.25, 6), (.25, 5)):
        reward, record, *_ = runtime.outcome(source, original, labs, features, features.new_full((n,), rho), features.new_full((n,), float(q)), .1)
        if reward.shape != (n,) or record['bpcs'] > cap.CAP_BPCS:
            raise AssertionError('Per-tile reward or the rate cap failed.')
        response.append(dict(record, rho=rho, q=q))
    seconds = (time.time()-begun)/len(response)
    print(f'One physical evaluation: {seconds:.2f} s', flush=True)
    csv(Path(str(out)+'.ACTION_RESPONSE.csv'), response)
    forced, _, projected, steps = runtime.project(source, features, features.new_ones(n), features.new_full((n,), 12.))
    if not steps or float(_['bpcs']) > cap.CAP_BPCS:
        raise AssertionError('The cap projection did not engage.')
    fitted = dict(indices=torch.arange(len(learning.NAMES), device=runtime.device), mean=features.new_zeros(len(learning.NAMES)), std=features.new_ones(len(learning.NAMES)), names=list(learning.NAMES))
    actor = learning.DeltaPolicy(len(learning.NAMES), runtime.device)
    allocation = learning.rollout(actor, fitted, features, .1)
    for entry in allocation['history']:
        torch.testing.assert_close(actor(entry['states'], entry['mask'])[0].log_prob(entry['actions']), entry['log_prob'], rtol=1e-12, atol=1e-12)
    perception = Perception(runtime.physics, runtime.device)
    agreement = fftradnet.agreement(perception.predict(source['cube']), perception.predict(decoded))
    write_json(Path(str(out)+'.CHECK.json'), dict(completed=True, passed=True, context=expected, training_started=False, test_evaluated=False, source_CSV_rows_verified=len(labels),
                                                  seconds_per_physical_evaluation=seconds, cap_projection_steps=steps, fftradnet_objects_kept=agreement['objects_kept'], fftradnet_objects_original=agreement['objects_original']))
    print('V12.40 / RADIal 20 ms preflight: PASS. No training started.', flush=True)


def run_all(runtime, out, expected):
    if (out/'SMOKE_COMPLETED.json').is_file() or (out.exists() and any(out.iterdir())):
        raise FileExistsError('Preserve partial or completed results; use a fresh output.')
    (out/'tables').mkdir(parents=True, exist_ok=True)
    write_json(out/'manifest.json', expected)
    for role, frame in zip(data.ROLES, runtime.rows):
        csv(out/'tables'/f'cube_index_{role}.csv', frame)
    write_json(out/'SPLIT_AUDIT.json', runtime.dataset.smoke_partition_audit)
    fitted = prepare_features(runtime, out)
    train(runtime, fitted, out)
    summary = final_reports(runtime, out, fitted)
    scientific = assess(out, summary)
    write_json(out/'SMOKE_COMPLETED.json', dict(completed=True, passed=True, context=expected, test_evaluated=True, test_used_for_selection=False, training_cubes=len(runtime.rows[0]),
                                                updates=runtime.config['updates'], synchronization_tolerance_us=20000, scientific_gate_passed=scientific['passed']))
    print('V12.40 / RADIal 20 ms COMPLETE:', out, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('check', 'run'), required=True)
    for name in ('out', 'dataset-root', 'reference-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    for name, default in (('train-cubes', 100), ('validation-cubes', 20), ('test-cubes', 20), ('passes', 10)):
        parser.add_argument('--'+name, type=int, default=default)
    args = parser.parse_args(argv)
    counts = (args.train_cubes, args.validation_cubes, args.test_cubes)
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('CUDA is mandatory; do not run this study on CPU.')
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print(f'STAGE: {args.stage} | GPU: {torch.cuda.get_device_name(device)} | synchronization: 20000 us | cap: {cap.CAP_BPCS} bpcs', flush=True)
    dataset = data.prepare_dataset(args.dataset_root, args.reference_root)
    rows = choose_rosters(dataset, counts)
    runtime = Runtime(dataset, rows, device, args.passes)
    out = args.out.resolve()
    expected = context(runtime, out)
    if args.stage == 'check':
        with common.exclusive(out):
            run_check(runtime, out, expected)
        return
    check = read_json(Path(str(out)+'.CHECK.json'))
    if check.get('context') != expected or check.get('passed') is not True:
        raise ValueError('Run requires a passing preflight for these exact identities.')
    with common.exclusive(out):
        run_all(runtime, out, expected)


if __name__ == '__main__':
    main()
