"""PPO on all of RADIal, tuned to beat AdaRadar above 1 bpcs: training, selection on VALIDATION, one pass of TEST.

WHAT STAYS. V12.40's actor, critic, state, 21 commands, twelve rounds, start, and the noise-referenced dithered codec with
noise filling. There is no feedback: the encoder sees the original cube and nothing returns from the decoder.

WHAT IS TUNED (radar_sim/radial_tuned_learning.py). Global command bounds, rate prices, the learning rate of two arms and
the training reward. The reward uses the receiver that scores the results, so training and scoring share one definition.

DATA. The frozen V17 roles: 2074 TRAIN, 708 VALIDATION and 662 TEST cubes, the cubes AdaRadar was scored on. One pass over
every TRAIN cube. Scoring is V17's: CA-CFAR at one false peak per noise map, labels, FFTRadNet with its official weights.
AdaRadar is not rerun; its saved per-cube VALIDATION results are in the package.

SELECTION, ON VALIDATION ONLY. Each checkpoint runs on SCREEN_CUBES evenly spaced VALIDATION cubes at every price. A
checkpoint scores one win per metric and budget above 1 bpcs where it beats AdaRadar on those same cubes: labels, robust
detections, robust false peaks, RD error. The most wins are frozen, then the largest robust gain, then the earlier arm and
update. Update 0 is not a trained policy and cannot win. A price is then fixed for each of AdaRadar's six rates on the
same cubes. Every cube stays within its budget. TEST opens after that choice and runs once.

python -B tools/run_radial_ppo_tuned.py --stage count|check|run --dataset-root ... --reference-root ... --split-audit ... --out ...
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
from radar_sim import radial_tuned_learning as learning
from radar_sim.radial_adaptive_gpu_physics import static_features


def sibling(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sweep = sibling('run_radial_ppo_sweep')  # the V17 receiver, labels, FFTRadNet, statistics and the budget projection
V20, V18, V17, V13 = sweep.V20, sweep.V18, sweep.V17, sweep.V13
common, data, codec = sweep.common, sweep.data, sweep.codec
REVISION = 'tuned.1.0.0-v12-40-model-radial-reward-bounds-prices'
PACKAGE = Path(__file__).resolve().parent/'radial_ppo_tuned_gpu'
sweep.PACKAGE = V20.PACKAGE = V18.PACKAGE = V17.PACKAGE = PACKAGE  # FFTRadNet files and the frozen split
SEED, PPO_SEED = V17.SEED, sweep.PPO_SEED
ARMS = (dict(learning_rate=1e-4), dict(learning_rate=3e-4))
PASSES, BATCH, ROLLOUTS, EPOCHS, AUDIT_CUBES, CHECKPOINT_SHARE, SCREEN_CUBES = 1, 5, 2, 5, 16, .1, 60
PRICE_GRID = (0., .0125, .025, .05, .075, .1, .15, .2, .25, .3)
BUDGETS = sweep.ADARADAR_RATES
MAIN_BUDGETS = tuple(rate for rate in BUDGETS if rate > 1)
FIXED = ((1., 6), (1., 5), (.75, 6), (.75, 5), (.5, 6), (.5, 5), (.5, 4), (.35, 5))
ADARADAR_METHOD = dict(zip(BUDGETS, ('adaradar_r50_s4', 'adaradar_r25_s4', 'adaradar_r12p57_s4', 'adaradar_r8_s4', 'adaradar_r4_s4', 'adaradar_r2_s4')))
ADARADAR_VALIDATION = 'adaradar_validation.csv'
# Saved TEST means of AdaRadar on the 662 cubes (V18, V20): labels and robust kept %, robust false peaks per map, RD dB, mAP, mAR, F1 %.
ADARADAR_TEST = {.4585: dict(bpcs=.4585, labels=95.19, robust=46.99, false=11.29, rd=4.86, mAP=87.32, mAR=49.25, F1=62.98),
                 .8208: dict(bpcs=.8208, labels=97.27, robust=59.96, false=13.93, rd=3.84, mAP=89.40, mAR=62.02, F1=73.23),
                 1.4541: dict(bpcs=1.4541, labels=98.05, robust=74.32, false=13.34, rd=3.02, mAP=90.98, mAR=71.24, F1=79.91),
                 2.1064: dict(bpcs=2.1064, labels=98.57, robust=83.54, false=10.96, rd=2.53, mAP=91.59, mAR=75.41, F1=82.72),
                 3.6327: dict(bpcs=3.6327, labels=99.35, robust=94.41, false=4.52, rd=1.80, mAP=92.42, mAR=78.19, F1=84.71),
                 4.7299: dict(bpcs=4.7299, labels=99.74, robust=97.20, false=1.26, rd=1.70, mAP=93.47, mAR=78.71, F1=85.46)}
ACCOUNTING = 'tuned PPO allocation, tile codec of V12.40: adaptive code length plus all side information'
SELECTION_RULE = ('Most wins over AdaRadar on the VALIDATION screen, at 1.45, 2.11, 3.63 and 4.73 bpcs, on labels, robust detections, robust false peaks and RD error; '
                  'then the largest robust gain; then the earlier arm and update. A label tie counts as a win. Update 0 cannot win.')
CONFIG = dict(arms=ARMS, passes=PASSES, batch=BATCH, rollouts=ROLLOUTS, epochs=EPOCHS, screen_cubes=SCREEN_CUBES, prices=learning.PRICES, price_grid=PRICE_GRID, budgets=BUDGETS, fixed=FIXED, rho_range=learning.RHO_RANGE,
              q_range=learning.Q_RANGE, reward=dict(label_miss=learning.LABEL_MISS_WEIGHT, robust_lost=learning.ROBUST_LOST_WEIGHT, robust_new=learning.ROBUST_NEW_WEIGHT, weak_new=learning.WEAK_NEW_WEIGHT, rate=learning.RATE_WEIGHT,
              nmse=learning.NMSE_WEIGHT), selection_rule=SELECTION_RULE, H0_maps=V13.H0_MAPS, margin_db=V13.MARGIN_DB, label_gate=list(V13.LABEL_GATE), sync_tolerance_us=data.TOLERANCE_US, ppo_seed=PPO_SEED,
              test_policy='TEST opens once, after the policy and prices are frozen on VALIDATION; AdaRadar is not rerun')
WEAK = learning.v12.base.STATIC_NAMES.index('weak_peak_frac')


def evenly(frame, count):
    """``count`` evenly spaced rows, the same ones on every run."""
    return frame if len(frame) <= count else frame.iloc[np.linspace(0, len(frame)-1, count).round().astype(int)]


def schedule(cubes, passes=None):
    passes = passes or PASSES
    updates = passes*math.ceil(cubes/BATCH)
    step = max(1, round(updates*CHECKPOINT_SHARE))
    return dict(updates=updates, checkpoints=sorted({0, updates, *range(step, updates, step)}))


def load_case(runtime, row):
    physics, cube = runtime.codec, runtime.cube(row)
    bins, label_count = runtime.label_bins(row.record_name)
    source = physics.prepare(cube, seed=PPO_SEED+zlib.crc32(row.record_name.encode()) % 2**31)
    features = static_features(physics, source, physics.maps(cube))
    return dict(row=row, cube=cube, original=runtime.receive(cube), bins=bins, label_count=label_count, source=source, features=features, order=torch.argsort(features[:, WEAK], stable=True))


@torch.no_grad()
def outcome(runtime, case, rho, q, price, alpha, budget=cap.CAP_BPCS):
    """One allocation through the codec, the budget projection and the scoring receiver."""
    decoded, meta, q, steps = sweep.project(runtime.codec, case['source'], case['order'], rho, q, budget)
    received = runtime.receive(decoded)
    detector = case['original']['detector']
    tile = learning.tile_outcomes(detector, received['detector'], case['bins'], alpha, V13.LABEL_GATE)
    meta['numel'] = case['source']['cube'].numel()
    reward, terms = learning.reward_components(tile, meta, price, runtime.weak_scale, case['source']['cube_energy'])
    rd = (received['RD']-case['original']['RD'])[detector['valid']]
    labels = codec.label_metrics(detector, received['detector'], case['bins'], alpha, *V13.LABEL_GATE)
    margin = 10**(learning.MARGIN_DB/10)
    record = dict(bpcs=float(meta['bpcs']), nmse=float(meta['nmse']), rd_rmse_db=float(rd.square().mean().sqrt()), labels_detected=labels['labels_detected_original'], label_miss=float(tile['miss'].sum()),
                  robust=int(codec.stable_peaks(detector, alpha, margin).sum()), robust_lost=float(tile['robust_lost'].sum()), robust_new=float(tile['robust_new'].sum()), new=float(tile['new'].sum()),
                  projection_steps=steps, mean_rho=float(rho.mean()), rho_std=float(rho.std()), mean_q=float(q.mean()), q_std=float(q.to(torch.float64).std()), mean_tile_reward=float(reward.mean()), within_budget=float(meta['bpcs']) <= budget, **terms)
    return reward, record, decoded, received, tile


@torch.no_grad()
def prepare_features(runtime, out):
    rows = evenly(runtime.roster('train'), AUDIT_CUBES)
    generator = torch.Generator(device=runtime.device).manual_seed(SEED+41)
    matrix, counts = [], []
    for _, row in rows.iterrows():
        case = load_case(runtime, row)
        n = len(case['features'])
        counts.append(len(codec.detections(case['original']['detector'], runtime.alpha))/n)
        for step in range(learning.ROUNDS):
            rho = torch.randint(1, 101, (n,), generator=generator, device=runtime.device).to(torch.float64)/100.
            q = torch.randint(1, 13, (n,), generator=generator, device=runtime.device).to(torch.float64)
            matrix.append(learning.raw_states(case['features'], rho, q, learning.ROUNDS-step, learning.PRICES[step % len(learning.PRICES)]))
    runtime.weak_scale = max(1., float(np.median(counts)))
    fitted = learning.fit_features(torch.cat(matrix))
    fitted['weak_scale'] = runtime.weak_scale
    common.csv(out/'tables/feature_selection.csv', fitted['rows'])
    torch.save(fitted, out/'feature_normalizer.pt')
    common.write_json(out/'OBSERVATION_CONTRACT.json', dict(raw_dimension=len(learning.NAMES), selected_dimension=len(fitted['names']), selected=fitted['names'], source_partition='TRAIN only'))
    common.write_json(out/'REWARD_CONTRACT.json', dict(weak_scale_per_tile=runtime.weak_scale, **CONFIG['reward'], rate_formula='RATE_WEIGHT*price*tile_bits/(cap share)', hard_bitrate_cap=cap.CAP_BPCS))
    return fitted


@torch.no_grad()
def query_edit(runtime, actor, fitted, case, greedy, update):
    """Measure every command on one tile of the greedy state, with the rest of the policy recomputed."""
    step = (update-1) % learning.ROUNDS
    state = greedy['history'][step]
    features = case['features']
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
        _, record, *_ = outcome(runtime, case, suffix['rho'], suffix['q'], greedy['price'], runtime.alpha)
        low[action], cost[action] = record['label_miss'], -record['mean_tile_reward']
        records.append(dict(tile=tile, edit_round=step+1, action=action, label_miss=record['label_miss'], bpcs=record['bpcs'], secondary_cost=-record['mean_tile_reward']))
    chosen = learning.compatible_mask(low, cost, feasible)
    for record in records:
        record['teacher_selected'] = bool(chosen[record['action']])
    return dict(state=state['states'][tile].detach(), mask=feasible, compatible=chosen, probabilities_before=state['probabilities'][tile].detach()), records


@torch.no_grad()
def screen(runtime, actor, fitted, rows, tag, prices=learning.PRICES, budgets=(cap.CAP_BPCS,)):
    """Every price on every screen cube. With several budgets, each projects the same allocation."""
    records = []
    for _, row in rows.iterrows():
        case = load_case(runtime, row)
        for price in prices:
            allocation = learning.rollout(actor, fitted, case['features'], price, deterministic=True)
            for budget in budgets:
                _, record, *_ = outcome(runtime, case, allocation['rho'], allocation['q'], price, runtime.alpha, budget)
                records.append(dict(record, **tag, price=price, budget=budget, record_name=row.record_name))
    return records


def train_arm(runtime, fitted, out, arm, setting, rows):
    plan = schedule(len(runtime.roster('train')))
    torch.manual_seed(SEED)
    if runtime.device.type == 'cuda':
        torch.cuda.manual_seed_all(SEED)
    actor = learning.DeltaPolicy(len(fitted['names']), runtime.device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=setting['learning_rate'], eps=1e-5, capturable=runtime.device.type == 'cuda')
    generator = torch.Generator(device=runtime.device).manual_seed(SEED+1)
    train_rows, records, queries_out = runtime.roster('train'), [], []
    start = time.time()

    def snapshot(update):
        state = torch.get_rng_state()
        path = out/'checkpoints'/f'arm_{arm}_update_{update:04d}.pt'
        path.parent.mkdir(exist_ok=True)
        torch.save(dict(actor=actor.state_dict(), arm=arm, update=update, learning_rate=setting['learning_rate'], feature_names=fitted['names']), path)
        rows_out = screen(runtime, actor, fitted, rows, dict(arm=arm, update=update))
        torch.set_rng_state(state)
        table = pd.DataFrame(rows_out)
        for price, g in table.groupby('price'):
            print(f'Screen arm {arm} update {update} price {price:g}: R={g.bpcs.mean():.3f} labels {100*(1-g.label_miss.sum()/max(g.labels_detected.sum(), 1)):.2f}% robust {100*(1-g.robust_lost.sum()/max(g.robust.sum(), 1)):.2f}% '
                  f'false/map {g.robust_new.mean():.2f} RD {g.rd_rmse_db.mean():.2f} dB rho {g.mean_rho.mean():.3f}+-{g.rho_std.mean():.3f} q {g.mean_q.mean():.2f}+-{g.q_std.mean():.2f}', flush=True)
        return rows_out

    screened = snapshot(0)
    update = 0
    for pass_index in range(PASSES):
        order = torch.randperm(len(train_rows), generator=generator, device=runtime.device).cpu().tolist()
        for first in range(0, len(order), BATCH):
            update += 1
            trajectories, queries, logs = [], [], []
            for offset, position in enumerate(order[first:first+BATCH]):
                row = train_rows.iloc[position]
                case = load_case(runtime, row)
                for index in range(ROLLOUTS):
                    price = learning.PRICES[(update+offset+index) % len(learning.PRICES)]
                    allocation = learning.rollout(actor, fitted, case['features'], price)
                    reward, record, *_ = outcome(runtime, case, allocation['rho'], allocation['q'], price, runtime.alpha)
                    trajectories.append((allocation, reward))
                    logs.append(dict(record, arm=arm, record_name=row.record_name, update=update, rate_price=price, pass_index=pass_index+1))
                    if offset == 0 and index == 0:
                        greedy = learning.rollout(actor, fitted, case['features'], price, deterministic=True)
                        packet, measured = query_edit(runtime, actor, fitted, case, greedy, update)
                        queries.append(packet)
                        queries_out.extend(dict(m, arm=arm, record_name=row.record_name, update=update) for m in measured)
            stats = learning.update(actor, optimizer, trajectories, queries, seed=SEED+update, epochs=EPOCHS)
            for log in logs:
                log.update({key: float(value) for key, value in stats.items()})
            records.extend(logs)
            mean = pd.DataFrame(logs)
            if update % 10 == 0 or update == 1:
                print(f'Arm {arm} update {update}/{plan["updates"]}: labels missed={mean.label_miss.mean():.2f} robust lost={mean.robust_lost.mean():.1f} false={mean.robust_new.mean():.2f} R={mean.bpcs.mean():.3f} rho={mean.mean_rho.mean():.3f} '
                      f'q={mean.mean_q.mean():.2f} reward={mean.mean_tile_reward.mean():.4f} KL={float(stats["approx_kl"]):.5f} entropy={float(stats["entropy"]):.3f} {time.time()-start:.0f}s', flush=True)
            if update in plan['checkpoints']:
                screened.extend(snapshot(update))
                pd.DataFrame(screened).to_csv(out/'tables'/f'screen_arm_{arm}.partial.csv', index=False)
    return records, queries_out, screened


def point(rows):
    """One operating point on the screen cubes."""
    return dict(bpcs=rows.bpcs.mean(), labels=100*(1-rows.label_miss.sum()/max(rows.labels_detected.sum(), 1)), robust=100*(1-rows.robust_lost.sum()/max(rows.robust.sum(), 1)), false=rows.robust_new.mean(), rd=rows.rd_rmse_db.mean(), nmse=rows.nmse.mean())


def adaradar_points(cubes):
    """AdaRadar on the same VALIDATION cubes, from its saved per-cube results."""
    table = pd.read_csv(PACKAGE/ADARADAR_VALIDATION)
    table = table.loc[table.record_name.isin(cubes)]
    points = {}
    for budget, method in ADARADAR_METHOD.items():
        g = table.loc[table.method.eq(method)]
        if len(g):
            points[budget] = dict(bpcs=g.bpcs.mean(), labels=100*(1-g.labels_lost.sum()/max(g.labels_detected_original.sum(), 1)), robust=100*(1-g.a1_lost_robust.sum()/max(g.a1_original_robust.sum(), 1)),
                                  false=g.a1_new_robust.mean(), rd=g.rd_rmse_db.mean())
    return points


def wins(ours, reference):
    return dict(labels=ours['labels'] >= reference['labels'], robust=ours['robust'] > reference['robust'], false=ours['false'] < reference['false'], rd=ours['rd'] < reference['rd'])


def choose_checkpoint(table, references):
    """The pre-registered rule; each checkpoint uses, at each budget, the price that spends the most bits within it."""
    rows = []
    for (arm, update), by_update in table.loc[table['update'] > 0].groupby(['arm', 'update'], sort=True):
        points = {price: point(g) for price, g in by_update.groupby('price')}
        count, gain, detail = 0, 0., {}
        for budget in MAIN_BUDGETS:
            within = [p for p in points.values() if p['bpcs'] <= budget]
            if within and budget in references:
                best = max(within, key=lambda p: p['bpcs'])
                won = wins(best, references[budget])
                count += sum(won.values())
                gain += best['robust']-references[budget]['robust']
                detail[f'{budget:g}'] = dict(best, won=won)
        rows.append(dict(arm=int(arm), update=int(update), wins=count, robust_gain=gain, detail=json.dumps(detail, default=float)))
    ranking = pd.DataFrame(rows).sort_values(['wins', 'robust_gain', 'arm', 'update'], ascending=[False, False, True, True]).reset_index(drop=True)
    return ranking


def choose_prices(table, references):
    """For each budget, the price with the most wins on the screen, then the most robust detections kept.
    Prices that leave a cube over its budget are used only if every price does."""
    chosen, rows = {}, []
    for budget, g in table.groupby('budget'):
        scored = []
        for price, h in g.groupby('price'):
            p = point(h)
            won = wins(p, references[budget]) if budget in references else {}
            scored.append((bool(h.within_budget.all()), sum(won.values()), p['robust'], -price))
            rows.append(dict(budget=budget, price=price, **p, wins=sum(won.values()), within_budget=bool(h.within_budget.all())))
        chosen[float(budget)] = -max(scored)[3]
    return chosen, rows


def plan_of(frozen):
    points = [(f'ppo_{sweep.tag(budget)}', frozen['prices'][f'{budget:g}'], None, budget) for budget in BUDGETS]
    points += [(f'ppo_free_{str(price).replace(".", "p")}', price, None, cap.CAP_BPCS) for price in learning.PRICES]
    points += [(f'fixed_r{round(100*rho)}_q{q}', None, (rho, q), cap.CAP_BPCS) for rho, q in FIXED]
    return points


def score_cube(runtime, actor, fitted, row, plan, alphas, perception, label_log):
    """Every point on one cube, against the receivers' output on the ORIGINAL cube (V17 scoring)."""
    device, physics = runtime.device, runtime.codec
    case = load_case(runtime, row)
    cube, original, bins, label_count = case['cube'], case['original'], case['bins'], case['label_count']
    original_os = V17.receivers.cyclic_os_cfar(original['detector']['power'])
    perception.start(row, original)
    energy = original['spectrum'].abs().square().sum()
    allocations, records = {}, []
    n = len(case['features'])
    for name, price, fixed, budget in plan:
        if fixed is None:
            if price not in allocations:
                allocations[price] = learning.rollout(actor, fitted, case['features'], price, deterministic=True)
            rho, q = allocations[price]['rho'], allocations[price]['q']
        else:
            rho, q = case['features'].new_full((n,), fixed[0]), case['features'].new_full((n,), float(fixed[1]))
        (decoded, meta, q, steps), encode_ms = V17.timed(device, sweep.project, physics, case['source'], case['order'], rho, q, budget)
        bits = dict(total=int(math.ceil(float(meta['realized_bits']))), accounting=ACCOUNTING, closed_loop_iterations=0)
        received = runtime.receive(decoded)
        record = V13.score(runtime, row, name, cube, original, decoded, bits, alphas['ca'], bins, label_count, label_log, received)
        method_os = V17.receivers.cyclic_os_cfar(received['detector']['power'])
        for key, alpha in alphas['os'].items():
            metrics = codec.detection_metrics(original_os, method_os, alpha, V13.MARGIN_DB)
            record.update({f'os_{key}_{k}': v for k, v in metrics.items()})
        record.update(price=price, budget_bpcs=budget, within_budget=float(meta['bpcs']) <= budget, projection_steps=steps, mean_rho=float(rho.mean()), rho_std=float(rho.std()), mean_q=float(q.mean()), q_std=float(q.std()),
                      spectrum_nmse=float((received['spectrum']-original['spectrum']).abs().square().sum()/energy), bpcs_kt=record['bpcs'], bpcs_learned=math.nan, learned_chosen=0., codec_ms=encode_ms, encode_ms=encode_ms,
                      decode_ms=math.nan, entropy_model_ms=math.nan)
        perception.add(name, received)
        records.append(record)
        del decoded, received
    return records


def sweep_rows(summary, perception, agreement, frame, role):
    table = summary.loc[summary.role.eq(role)].set_index('method')
    observed = perception.loc[perception.role.eq(role) & perception.subset.eq('all')].set_index('method')
    agree = agreement.loc[agreement.role.eq(role)].set_index('method')
    rows = []
    for method, g in frame.loc[frame.role.eq(role)].groupby('method', sort=False):
        r, s, a = table.loc[method], observed.loc[method], agree.loc[method]
        rows.append(dict(role=role, method=method, price=g.price.iloc[0], budget_bpcs=g.budget_bpcs.iloc[0], bpcs=r.bpcs_mean, bpcs_max=r.bpcs_max, within_budget_fraction=g.within_budget.mean(), mean_rho=g.mean_rho.mean(), rho_std=g.rho_std.mean(),
                         mean_q=g.mean_q.mean(), q_std=g.q_std.mean(), projection_steps_mean=g.projection_steps.mean(), labels_kept_pct=100*(1-g.labels_lost.sum()/max(g.labels_detected_original.sum(), 1)),
                         robust_kept_pct=100*(1-g.a1_lost_robust.sum()/max(g.a1_original_robust.sum(), 1)), robust_false_per_map=g.a1_new_robust.mean(), rd_rmse_db=g.rd_rmse_db.mean(), nmse=g.nmse.mean(), mAP=100*s.mAP, mAR=100*s.mAR, F1=100*s.F1,
                         dF1=100*s.dF1, dF1_ci_low=100*s.get('dF1_ci_low', math.nan), dF1_ci_high=100*s.get('dF1_ci_high', math.nan), objects_kept_fraction=a.objects_kept_fraction))
    return pd.DataFrame(rows).sort_values('bpcs')


def scorecard(table):
    """Each budget against AdaRadar's saved TEST means. Reported as measured; nothing here feeds back into any choice."""
    table = table.set_index('method')
    rows = []
    for budget, reference in ADARADAR_TEST.items():
        r = table.loc[f'ppo_{sweep.tag(budget)}']
        ours = dict(bpcs=r.bpcs, labels=r.labels_kept_pct, robust=r.robust_kept_pct, false=r.robust_false_per_map, rd=r.rd_rmse_db, mAP=r.mAP, mAR=r.mAR, F1=r.F1)
        won = {key: (ours[key] < reference[key] if key in ('false', 'rd', 'bpcs') else ours[key] > reference[key]) for key in ours}
        rows.append(dict(budget=budget, **{f'ppo_{k}': v for k, v in ours.items()}, **{f'adaradar_{k}': v for k, v in reference.items()}, **{f'win_{k}': v for k, v in won.items()}))
    return pd.DataFrame(rows)


def context(runtime, out, assignment, scan):
    hashes = V17.source_hashes()
    for path in (Path(__file__), Path(sweep.__file__), Path(V20.__file__), Path(V18.__file__), Path(cap.__file__), Path(learning.__file__), PACKAGE/ADARADAR_VALIDATION):
        hashes[str(path.resolve().relative_to(common.ROOT)).replace('\\', '/')] = common.sha(path)
    return dict(revision=REVISION, source_root=str(common.ROOT.resolve()), output=str(out), dataset_root=str(runtime.dataset.root), reference_root=str(runtime.dataset.reference_root), device=str(runtime.device), torch=torch.__version__,
                cuda_runtime=torch.version.cuda, gpu=torch.cuda.get_device_name(runtime.device) if runtime.device.type == 'cuda' else 'cpu-engineering', split=str(Path(assignment).resolve()), split_sha256=common.sha(assignment),
                split_provenance_sha256=common.sha(V17.provenance_path(assignment)), inventory=scan, source_hashes=hashes, dataset_identity=data.dataset_identity(runtime.dataset),
                populations={role: frame.record_name.tolist() for role, frame in runtime.rows.items() if role != 'train'}, train_cubes=len(runtime.rows['train']), schedule=schedule(len(runtime.rows['train'])),
                configuration=json.loads(json.dumps(CONFIG, default=str)), seed=SEED)


def run_check(runtime, out, expected):
    """CUDA pre-flight on real cubes. It trains nothing that is kept and never opens TEST."""
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
    runtime.alpha = codec.threshold_for_rate(codec.h0_scores(4, p.shape, p.range_window, p.doppler_window, V17.cyclic_cfar, device, SEED), 4, 1.)
    os_alpha = codec.threshold_for_rate(codec.h0_scores(4, p.shape, p.range_window, p.doppler_window, V17.receivers.cyclic_os_cfar, device, SEED), 4, 1.)
    alphas = dict(ca=dict(a1=runtime.alpha, a01=runtime.alpha), os=dict(a1=os_alpha, a01=os_alpha))
    runtime.weak_scale = 1.
    row = runtime.roster('train').iloc[len(runtime.roster('train'))//2]
    case = load_case(runtime, row)
    n = len(case['features'])
    reward, record, decoded, received, tile = outcome(runtime, case, case['features'].new_full((n,), .5), case['features'].new_full((n,), 5.), .1, runtime.alpha)
    metric = codec.detection_metrics(case['original']['detector'], received['detector'], runtime.alpha, learning.MARGIN_DB)
    if (int(tile['robust_lost'].sum()), int(tile['robust_new'].sum()), int(tile['new'].sum()), record['robust']) != (metric['lost_robust'], metric['new_robust'], metric['new'], metric['original_robust']):
        raise AssertionError('The per-tile reward counts differ from the scoring receiver.')
    if reward.shape != (n,) or not record['within_budget']:
        raise AssertionError('Per-tile reward or the cap failed.')
    same = learning.tile_outcomes(case['original']['detector'], case['original']['detector'], case['bins'], runtime.alpha, V13.LABEL_GATE)
    if any(float(same[key].sum()) for key in ('miss', 'robust_lost', 'robust_new', 'new')):
        raise AssertionError('The original cube fails its own reward identity.')
    r, bits, mask = learning.alternatives(case['features'].new_full((n,), learning.RHO_RANGE[0]), case['features'].new_full((n,), learning.Q_RANGE[0]))
    if float(r.min()) < learning.RHO_RANGE[0] or float(bits.min()) < learning.Q_RANGE[0] or not bool(mask.any(1).all()):
        raise AssertionError('The command bounds failed.')
    fitted = dict(indices=torch.arange(len(learning.NAMES), device=device), mean=case['features'].new_zeros(len(learning.NAMES)), std=case['features'].new_ones(len(learning.NAMES)), names=list(learning.NAMES))
    actor = learning.DeltaPolicy(len(learning.NAMES), device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=ARMS[1]['learning_rate'], eps=1e-5, capturable=device.type == 'cuda')
    begun = time.time()
    trajectories = []
    for price in learning.PRICES[:ROLLOUTS]:
        allocation = learning.rollout(actor, fitted, case['features'], price)
        for entry in allocation['history']:
            torch.testing.assert_close(actor(entry['states'], entry['mask'])[0].log_prob(entry['actions']), entry['log_prob'], rtol=1e-12, atol=1e-12)
        trajectories.append((allocation, outcome(runtime, case, allocation['rho'], allocation['q'], price, runtime.alpha)[0]))
    per_rollout = (time.time()-begun)/ROLLOUTS
    greedy = learning.rollout(actor, fitted, case['features'], learning.PRICES[1], deterministic=True)
    packet, measured = query_edit(runtime, actor, fitted, case, greedy, 1)
    learning.update(actor, optimizer, trajectories, [packet], seed=SEED, epochs=EPOCHS)
    seconds = (time.time()-begun)/(ROLLOUTS+len(measured))
    references = adaradar_points(runtime.roster('validation').record_name)
    if set(references) != set(BUDGETS)-{.4585}:
        raise AssertionError('The saved AdaRadar VALIDATION results are incomplete.')
    perception = V20.Perception(runtime)
    frozen = dict(prices={f'{budget:g}': learning.PRICES[-1] for budget in BUDGETS})
    begun = time.time()
    records = score_cube(runtime, actor, fitted, row, plan_of(frozen), alphas, perception, [])
    per_cube = time.time()-begun
    missed = {r['method']: (round(r['bpcs'], 4), r['budget_bpcs'], round(r['mean_rho'], 3), round(r['mean_q'], 2), r['projection_steps']) for r in records if not r['within_budget'] and r['budget_bpcs'] > 1}
    if missed:  # this actor is untrained, and below 1 bpcs its retention can exceed what q=1 reaches
        raise AssertionError(f'A budget was not reached: {missed}')
    cubes, counts = schedule(len(runtime.rows['train'])), {role: len(runtime.roster(role)) for role in ('validation', 'test')}
    case_seconds = per_rollout
    training = len(ARMS)*cubes['updates']*(BATCH*ROLLOUTS+21)*seconds/3600
    screening = len(ARMS)*len(cubes['checkpoints'])*min(SCREEN_CUBES, counts['validation'])*len(learning.PRICES)*case_seconds/3600
    tuning = min(SCREEN_CUBES, counts['validation'])*len(PRICE_GRID)*len(BUDGETS)*case_seconds/3600
    reports = (counts['validation']+counts['test'])*per_cube/3600
    plan = dict(seconds_per_evaluation=seconds, seconds_per_report_cube=per_cube, updates_per_arm=cubes['updates'], arms=len(ARMS), training_hours=training, screening_hours=screening, price_hours=tuning, report_hours=reports, estimated_run_hours=training+screening+tuning+reports)
    result = dict(completed=True, passed=True, context=expected, record_name=row.record_name, plan=plan, rates={r['method']: round(r['bpcs'], 3) for r in records}, gpu_peak_memory_gb=torch.cuda.max_memory_allocated(device)/1e9 if device.type == 'cuda' else 0.,
                  training_started=False, test_evaluated=False)
    V17.write_report(path, result)
    print(json.dumps({k: result[k] for k in ('plan', 'gpu_peak_memory_gb', 'rates')}), flush=True)
    print('Tuned PPO CUDA check complete. Nothing kept, TEST not opened.', flush=True)


def report(runtime, frozen, fitted, alphas, out, log):
    path = out/frozen['checkpoint']
    if common.sha(path) != frozen['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint changed.')
    actor = learning.DeltaPolicy(len(fitted['names']), runtime.device)
    actor.load_state_dict(torch.load(path, map_location=runtime.device, weights_only=False)['actor'])
    actor.eval()
    perception = V20.Perception(runtime)
    plan = plan_of(frozen)
    records, label_log = [], []
    for role in ('validation', 'test'):
        if role == 'test':
            runtime.authorize_test(out/'FROZEN_MODEL.json')
        rows = runtime.roster(role)
        log(f'{role} reports: {len(rows)} cubes, {len(plan)} points')
        for i, (_, row) in enumerate(rows.iterrows()):
            records.extend(score_cube(runtime, actor, fitted, row, plan, alphas, perception, label_log))
            if (i+1) % 25 == 0 or i+1 == len(rows):
                log(f'{role} {i+1}/{len(rows)}')
    frame = pd.DataFrame(records)
    frame.to_csv(out/'RESULTS_BY_CUBE.csv', index=False)
    pd.DataFrame(label_log).to_csv(out/'LABELS.csv', index=False)
    summary = V17.summarize(frame)
    summary.to_csv(out/'SUMMARY.csv', index=False)
    V17.write_report(out/'RECORDING_CI.json', V17.recording_bootstrap(frame))
    observed, agreement = perception.report(out)
    tables = {role: sweep_rows(summary, observed, agreement, frame, role) for role in ('validation', 'test')}
    tables['validation'].to_csv(out/'RATE_SWEEP_VALIDATION.csv', index=False)
    tables['test'].to_csv(out/'RATE_SWEEP_TEST.csv', index=False)
    card = scorecard(tables['test'])
    card.to_csv(out/'ADARADAR_SCORECARD.csv', index=False)
    return tables, card


def run(runtime, out, expected):
    checked = common.read_json(Path(str(out)+'.CHECK.json'))
    if checked.get('context') != expected or checked.get('passed') is not True:
        raise ValueError('Run the matching CUDA check first.')
    if out.exists():
        raise FileExistsError('Preserve partial/completed results; never overwrite this study.')
    with common.exclusive(out):
        (out/'tables').mkdir(parents=True)
        common.write_json(out/'manifest.json', expected)
        start = time.time()
        log = lambda *a: print(*a, f'{time.time()-start:.0f}s', flush=True)
        calibration, curve = V17.calibrate(runtime)
        V17.write_report(out/'H0_CALIBRATION.json', calibration)
        curve.to_csv(out/'H0_SURVIVAL.csv', index=False)
        alphas = dict(ca=calibration['alphas'], os=calibration['os_cfar']['alphas'])
        runtime.alpha = alphas['ca']['a1']
        log('alpha dB CA:', json.dumps(calibration['alphas_db']), 'OS:', json.dumps(calibration['os_cfar']['alphas_db']))
        fitted = prepare_features(runtime, out)
        rows = evenly(runtime.roster('validation'), SCREEN_CUBES)
        records, queries, screened = [], [], []
        for arm, setting in enumerate(ARMS):
            log(f'Training arm {arm}: learning rate {setting["learning_rate"]:g}')
            for target, values in zip((records, queries, screened), train_arm(runtime, fitted, out, arm, setting, rows)):
                target.extend(values)
        common.csv(out/'tables/training.csv', records)
        common.csv(out/'tables/conditional_queries.csv', queries)
        table = pd.DataFrame(screened)
        table.to_csv(out/'tables/screen.csv', index=False)
        references = adaradar_points(rows.record_name)
        ranking = choose_checkpoint(table, references)
        ranking.to_csv(out/'CHECKPOINT_RANKING.csv', index=False)
        best = ranking.iloc[0]
        arm, update = int(best.arm), int(best['update'])
        checkpoint = out/'checkpoints'/f'arm_{arm}_update_{update:04d}.pt'
        log(f'Selected arm {arm} update {update}: {int(best.wins)} wins, robust gain {best.robust_gain:.2f}')
        actor = learning.DeltaPolicy(len(fitted['names']), runtime.device)
        actor.load_state_dict(torch.load(checkpoint, map_location=runtime.device, weights_only=False)['actor'])
        actor.eval()
        tuned = pd.DataFrame(screen(runtime, actor, fitted, rows, dict(arm=arm, update=update), PRICE_GRID, BUDGETS))
        tuned.to_csv(out/'tables/price_screen.csv', index=False)
        prices, price_rows = choose_prices(tuned, references)
        common.csv(out/'tables/price_selection.csv', price_rows)
        frozen = dict(variant=REVISION, arm=arm, update=update, learning_rate=ARMS[arm]['learning_rate'], prices={f'{budget:g}': prices[budget] for budget in BUDGETS}, checkpoint=str(checkpoint.relative_to(out)).replace('\\', '/'),
                      checkpoint_sha256=common.sha(checkpoint), selection_rule=SELECTION_RULE, wins=int(best.wins), test_used_for_selection=False)
        common.write_json(out/'FROZEN_POLICY.json', frozen)
        V17.write_report(out/'FROZEN_MODEL.json', dict(completed=True, checkpoint=frozen['checkpoint'], checkpoint_sha256=frozen['checkpoint_sha256'], source=REVISION, trained_in_this_run=True, test_used_for_selection=False))
        log('FROZEN:', json.dumps({k: frozen[k] for k in ('arm', 'update', 'learning_rate', 'prices', 'wins')}))
        tables, card = report(runtime, frozen, fitted, alphas, out, log)
        main_card = card.loc[card.budget.isin(MAIN_BUDGETS)]
        keys = ('labels', 'robust', 'false', 'rd', 'mAP', 'mAR', 'F1')
        passed = {f'{row.budget:g}': {key: bool(getattr(row, f'win_{key}')) for key in keys} for row in main_card.itertuples()}
        V17.write_report(out/'PREDICTIONS.json', dict(predictions=dict(P1='above 1 bpcs the frozen policy beats AdaRadar on every score at each of its rates'), passed=dict(P1=all(all(v.values()) for v in passed.values())), detail=dict(P1=passed)))
        runtime.dataset.assert_unchanged()
        V17.write_report(out/'SMOKE_COMPLETED.json', dict(completed=True, passed=True, context=expected, seed=SEED, test_evaluated=True, test_used_for_selection=False, entropy_model_trained=False, alphas_db=calibration['alphas_db'],
                                                          cubes=len(runtime.roster('test')), pre_registered_points_passed=dict(P1=all(all(v.values()) for v in passed.values())), runtime_seconds=time.time()-start))
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print(tables['test'][['method', 'budget_bpcs', 'bpcs', 'labels_kept_pct', 'robust_kept_pct', 'robust_false_per_map', 'rd_rmse_db', 'mAP', 'mAR', 'F1', 'mean_rho', 'rho_std', 'mean_q', 'q_std']].round(3).to_string(index=False), flush=True)
        print(card.round(3).T.to_string(), flush=True)
    print('Tuned PPO run COMPLETE; start with ADARADAR_SCORECARD.csv, RATE_SWEEP_TEST.csv and FROZEN_POLICY.json:', out, flush=True)


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
    runtime = sweep.build_runtime(dataset, args.split_audit, args.device)
    expected = json.loads(json.dumps(context(runtime, out, args.split_audit, scan)))
    print(json.dumps({k: expected[k] for k in ('revision', 'gpu', 'train_cubes', 'schedule')}, indent=1), flush=True)
    if args.stage == 'check':
        run_check(runtime, out, expected)
    else:
        run(runtime, out, expected)


if __name__ == '__main__':
    main()
