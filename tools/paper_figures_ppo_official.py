"""Rate sweep of the PPO paper over all RADIal test cubes: the 1,386 cubes of the four official test recordings and our 662 cubes.

Official cubes come from the official-test run, and so does AdaRadar on our 662. PPO, BAQ and Learned ECDQ on our 662 come from the
earlier verified runs, which that run replayed. CFAR scores are pooled over cubes. FFTRadNet scores use the official recordings only,
because both networks trained on the other seven recordings and per-frame counts were not saved.

Learned ECDQ above 3 bpcs is charged the adaptive code on all cubes, because the earlier runs have no learned rate there.

Reads result CSVs only; nothing here touches the manuscript.

python -B tools/paper_figures_ppo_official.py [--out outputs/paper_figures_ppo_official]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_figures_ppo_paper as paper
import paper_figures_v20 as base
from paper_figures_v20 import BLACK, BLUE

OFFICIAL_RUN = paper.REMOTE/'official_test_20261101T122803763Z_verified_20261101T191517655Z'
UNTOUCHED = ('RECORD@2020-11-22_12.25.47', 'RECORD@2020-11-22_12.45.05', 'RECORD@2020-11-22_12.54.38')
BUDGETS = ('0p46', '0p82', '1p45', '2p11', '3p63', '4p73')
RATIOS = ('r50', 'r25', 'r12p57', 'r8', 'r4', 'r2')
SERIES = [('PPO', f'ppo_{tag}', 'ppo', f'ppo_{tag}') for tag in BUDGETS]
SERIES += [('Fixed', f'fixed_{name}', 'ppo', f'fixed_{name}') for name in ('r50_q5', 'r75_q5', 'r75_q6')]
SERIES += [('AdaRadar', f'adaradar_{ratio}_s4', None, None) for ratio in RATIOS]
SERIES += [('AdaRadar_feedback', 'adaradar_adaptive_official_s4', None, None), ('AdaRadar_feedback_own', 'adaradar_adaptive_own_s4', None, None)]
SERIES += [('Learned_ECDQ', f'ecdq_{tag}', 'v21', f'c_{tag}') for tag in ('0p46', '0p82', '1p00', '1p45', '2p00', '2p11')]
SERIES += [('Learned_ECDQ', 'ecdq_b2', 'v17', 'ecdq_b2'), ('Learned_ECDQ', 'ecdq_b1p5', 'v19', 'ecdq_b1p5'), ('Learned_ECDQ', 'ecdq_b1', 'v17', 'ecdq_b1')]
SERIES += [('BAQ', f'baq_{bits}bit', 'v17', f'baq_{bits}bit') for bits in (1, 2, 3)]
ADAPTIVE_RATE = ('ecdq_b2', 'ecdq_b1p5', 'ecdq_b1')
CURVE = dict(PPO=[f'ppo_{tag}' for tag in BUDGETS if tag != '2p11'])  # 1.45 and 2.11 are one operating point
CODED_STYLE = dict(color=BLUE, ls=(0, (1, 1.2)), marker='s', ms=3, zorder=6, label='AdaRadar, coded')
DRAWS, SEED = 10000, 20261101


def load(arm1=None):
    """One frame per population with the same method names, on one rate count. With ``arm1`` the budget points of PPO come from that run on every cube."""
    official = pd.read_csv(OFFICIAL_RUN/'official/RESULTS_BY_CUBE.csv', low_memory=False)
    ours = pd.read_csv(OFFICIAL_RUN/'ours/RESULTS_BY_CUBE.csv', low_memory=False)
    parts = [ours]
    for run in ('ppo', 'v17', 'v19', 'v21'):
        rename = {early: name for _, name, source, early in SERIES if source == run}
        frame = pd.read_csv(paper.REMOTE/paper.RUNS[run]/'RESULTS_BY_CUBE.csv', low_memory=False)
        parts.append(frame.loc[frame.role.eq('test') & frame.method.isin(rename)].assign(method=lambda g: g.method.map(rename)))
    ours = pd.concat(parts, ignore_index=True)
    for frame in (official, ours):
        high = frame.method.isin(ADAPTIVE_RATE)
        frame.loc[high, 'bpcs'] = frame.loc[high, 'bpcs_kt']
    if arm1 is not None:
        rows = pd.read_csv(arm1/'RESULTS_BY_CUBE.csv', low_memory=False)
        scenes = set(official.scene_name)
        official = pd.concat((official.loc[~official.method.str.startswith('ppo_')], rows.loc[rows.scene_name.isin(scenes)]), ignore_index=True)
        ours = pd.concat((ours.loc[~ours.method.str.startswith('ppo_')], rows.loc[~rows.scene_name.isin(scenes)]), ignore_index=True)
    return official, ours


def score(g):
    """Means over all cubes of one method; counts are pooled over cubes."""
    share = lambda lost, total: 100*(1-g[lost].sum()/g[total].sum())
    os_cfar = g.os_a1_original_robust.notna().any() if 'os_a1_original_robust' in g else False   # the Arm 1 run has no OS-CFAR
    return pd.Series(dict(cubes=g.record_name.nunique(), recordings=g.scene_name.nunique(), bpcs=g.bpcs.mean(), coded=g.bpcs_coded.mean() if 'bpcs_coded' in g else np.nan,
                          labels=share('labels_lost', 'labels_detected_original'), labels_lost=g.labels_lost.sum(), labels_detected=g.labels_detected_original.sum(),
                          robust=share('a1_lost_robust', 'a1_original_robust'), robust_original=g.a1_original_robust.sum(), false=g.a1_new_robust.mean(), rd=g.rd_rmse_db.mean(),
                          robust_os=share('os_a1_lost_robust', 'os_a1_original_robust') if os_cfar else np.nan, false_os=g.os_a1_new_robust.mean() if os_cfar else np.nan,
                          robust_a01=share('a01_lost_robust', 'a01_original_robust'), false_a01=g.a01_new_robust.mean(), score_change=g.a1_score_change_db_mean.mean()), dtype=float)


def sweep(frames):
    rows = []
    for population, frame in frames.items():
        for system, name, *_ in SERIES:
            g = frame.loc[frame.method.eq(name)]
            if len(g):
                rows.append(pd.concat((pd.Series(dict(population=population, system=system, method=name)), score(g))))
    table = pd.DataFrame(rows)
    sizes = table.groupby(['population', 'method']).cubes.first().unstack('population')
    if not (sizes.dropna().eq(dict(all=2048, official=1386, ours=662, untouched=1206)).all().all()):
        raise AssertionError('Every point must cover its whole population.')
    return table


def paired(frame):
    """PPO minus AdaRadar at each budget: paired by cube, recordings resampled with the same draws for every pair."""
    rng = np.random.default_rng(SEED)
    columns = ['labels_lost', 'labels_detected_original', 'a1_lost_robust', 'a1_original_robust', 'a1_new_robust']
    per = {name: frame.loc[frame.method.eq(name)].groupby('scene_name')[columns].sum() for name in frame.method.unique()}
    scenes = per['ppo_1p45'].index
    cubes = frame.loc[frame.method.eq('ppo_1p45')].groupby('scene_name').size().reindex(scenes).to_numpy(float)
    draws = rng.integers(0, len(scenes), (DRAWS, len(scenes)))
    stats = dict(labels=lambda s, d: 100*(1-s.labels_lost.to_numpy(float)[d].sum(-1)/s.labels_detected_original.to_numpy(float)[d].sum(-1)),
                 robust=lambda s, d: 100*(1-s.a1_lost_robust.to_numpy(float)[d].sum(-1)/s.a1_original_robust.to_numpy(float)[d].sum(-1)),
                 false=lambda s, d: s.a1_new_robust.to_numpy(float)[d].sum(-1)/cubes[d].sum(-1))
    rows = []
    for tag, ratio in zip(BUDGETS, RATIOS):
        a, b = per[f'ppo_{tag}'].reindex(scenes), per[f'adaradar_{ratio}_s4'].reindex(scenes)
        row = dict(budget=tag, adaradar=ratio, recordings=len(scenes))
        each = np.arange(len(scenes))[:, None]
        for key, stat in stats.items():
            difference = stat(a, draws)-stat(b, draws)
            whole = np.arange(len(scenes))
            single = stat(a, each)-stat(b, each)
            better = (single < 0) if key == 'false' else (single > 0)
            row.update({key: float(stat(a, whole)-stat(b, whole)), f'{key}_lo': float(np.percentile(difference, 2.5)), f'{key}_hi': float(np.percentile(difference, 97.5)),
                        f'{key}_recordings_better': int(better.sum()), f'{key}_recordings_equal': int((single == 0).sum())})
        rows.append(row)
    return pd.DataFrame(rows)


def perception():
    """FFTRadNet on the official recordings: official weights and AdaRadar's published weights, each with its change from its own uncompressed score."""
    seen = pd.read_csv(OFFICIAL_RUN/'official/PERCEPTION_SUMMARY.csv')
    seen = seen.loc[seen.subset.eq('all')].set_index('method')
    rows = []
    for system, name, *_ in [('Original', 'original'), *SERIES]:
        for key, method, reference in (('official', name, 'original'), ('adaradar', f'{name}@adaradar_weights', 'original@adaradar_weights')):
            if method in seen.index:
                s, o = seen.loc[method], seen.loc[reference]
                rows.append(dict(system=system, method=name, weights=key, mAP=100*s.mAP, mAR=100*s.mAR, F1=100*s.F1, dF1=100*(s.F1-o.F1)))
    return pd.DataFrame(rows)


def draw(table, path, coded):
    base.style()
    rows = table.loc[table.population.eq('all')]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.15), constrained_layout=True)
    for ax, (key, label, limits) in zip(axes, paper.PANELS):
        ax.axhline(0 if key == 'false' else 100, color=BLACK, ls=(0, (1, 1.5)), lw=.8, label='Original')
        for system, style in paper.STYLES.items():
            line = rows.loc[rows.system.eq(system)]
            line = line.loc[line.method.isin(CURVE.get(system, line.method))].sort_values('bpcs')
            ax.plot(line.bpcs, line[key], lw=1.2, **style)
            if system == 'AdaRadar' and coded:
                ax.plot(line.coded, line[key], lw=1., **CODED_STYLE)
        base.axes_setup(ax, label, (.3 if coded else .4, 7), paper.XTICKS)
        for mark in ax.get_lines():
            if mark.get_xdata()[0] == base.R_STAR and mark.get_xdata()[-1] == base.R_STAR:
                mark.remove()
        if key == 'false':
            ax.set_yscale('symlog', linthresh=.1)
            ax.set_yticks([0, .1, 1, 10, 100], ['0', '0.1', '1', '10', '100'])
            ax.set_yticks([], minor=True)
            ax.set_ylim(-.02, 400)
        else:
            ax.set_ylim(*limits)
    base.tags(axes)
    axes[2].texts[-1].set_position((.86, .97))
    axes[0].legend(loc='lower right', **base.LEGEND)
    fig.savefig(f'{path}.pdf')
    fig.savefig(f'{path}.png', dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1]/'outputs/paper_figures_ppo_official')
    parser.add_argument('--arm1', type=Path, help='Verified Arm 1 run: its points replace PPO in the tables and figures.')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    official, ours = load(args.arm1)
    every = pd.concat((official, ours), ignore_index=True)
    frames = dict(all=every, official=official, ours=ours, untouched=official.loc[official.scene_name.isin(UNTOUCHED)])
    table = sweep(frames)
    pairs = pd.concat([paired(frame).assign(population=name) for name, frame in frames.items() if name in ('all', 'official')], ignore_index=True)
    seen = perception()
    table.to_csv(args.out/'RATE_SWEEP_ALL_TESTS.csv', index=False)
    pairs.to_csv(args.out/'PAIRED_PPO_MINUS_ADARADAR.csv', index=False)
    seen.to_csv(args.out/'FFTRADNET_OFFICIAL_RECORDINGS.csv', index=False)
    draw(table, args.out/'fig_rate_sweep_all_tests', False)
    draw(table, args.out/'fig_rate_sweep_all_tests_coded', True)
    with pd.option_context('display.width', 230, 'display.max_columns', 40, 'display.max_rows', 200):
        show = ['population', 'method', 'cubes', 'bpcs', 'coded', 'labels', 'labels_lost', 'robust', 'false', 'rd', 'robust_os', 'false_os', 'robust_a01', 'false_a01', 'score_change']
        print(table.loc[table.population.isin(('all', 'ours')), show].round(3).to_string(index=False))
        print(table.groupby('population')[['labels_detected', 'robust_original']].first())
        print(pairs.round(3).to_string(index=False))
        print(seen.round(2).to_string(index=False))
    print('written to', args.out)


if __name__ == '__main__':
    main()
