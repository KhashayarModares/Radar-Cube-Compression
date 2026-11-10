"""Figures and tables of the UW PPO run. Reads the verified result folder only.

  fig_uw_ppo_tradeoff       labels, robust detections and robust false alarms against rate, on TEST
  fig_uw_ppo_training       rate, misses and reward against update, on TRAIN
  fig_uw_ppo_validation     VALIDATION rate and labels at every checkpoint, with the frozen choice
  fig_uw_ppo_allocation     where the frozen policy spends bits on TEST
  table_uw_ppo.csv/.tex     one row per method, table_uw_ppo_motion.csv by static and moving maps

python -B tools/paper_figures_uw_ppo.py --run <verified folder> [--out outputs/paper_figures_uw_ppo]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_figures_v20 as base
from paper_figures_v20 import BLACK, BLUE, RED, ROOT

RAW_BPCS = 32
GROUPS = (('PPO', RED, '-', 'o', 'PPO'), ('Uniform', BLUE, 'none', 's', 'Uniform'))
PANELS = (('labels_preserved_pct', 'Labels preserved (%)', 100.), ('robust_kept_pct', 'Robust CFAR detections kept (%)', 100.), ('robust_false_per_map', 'Robust false alarms per map', 0.))
XTICKS = [.5, .75, 1, 1.5, 2, 3, 4]
ARM_STYLES = ((0, RED, '-', 'V12.40 state, rate 1e-4'), (1, BLUE, '--', 'V12.40 state, rate 3e-4'), (2, BLACK, ':', 'No CFAR state, rate 3e-4'))


def tradeoff(summary, selected, path):
    test = summary.loc[summary.split.eq('test')]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5), constrained_layout=True)
    for ax, (key, label, reference) in zip(axes, PANELS):
        ax.axhline(reference, color=BLACK, ls=(0, (1, 1.5)), lw=.9, label='Original')
        for prefix, color, line, marker, name in GROUPS:
            rows = test.loc[test.method.str.startswith(prefix)].sort_values('bpcs')
            ax.plot(rows.bpcs, rows[key], color=color, ls=line, marker=marker, mfc='white' if prefix != 'PPO' else color, zorder=7 if prefix == 'PPO' else 5, label=name)
        chosen = test.loc[test.method.eq(selected)]
        ax.plot(chosen.bpcs, chosen[key], ls='none', marker='o', ms=9, mfc='none', mec=RED, mew=1.1, zorder=8, label='PPO, frozen on validation')
        shown = test.loc[~test.method.str.startswith('Peak')]
        base.axes_setup(ax, label, (.9*shown.bpcs.min(), 1.15*shown.bpcs.max()), XTICKS)
        for line in [l for l in ax.get_lines() if l.get_linestyle() == '--' and len(l.get_xdata()) == 2 and l.get_xdata()[0] == l.get_xdata()[1]]:
            line.remove()  # the shared helper marks the AdaRadar rate, which has no meaning on this dataset
        low, high = base.limits(np.append(shown[key].to_numpy(float), reference))
        ax.set_ylim(low, high+.08*(high-low))
    base.tags(axes)
    axes[0].legend(loc='lower right', **base.LEGEND)
    save(fig, path)


def training(frame, frozen, path):
    keys = (('bpcs', 'Rate (bpcs)'), ('robust_lost_tiles', 'Robust detections lost per cube'), ('mean_tile_reward', 'Mean tile reward'))
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.3), constrained_layout=True)
    for ax, (key, label) in zip(axes, keys):
        for arm, color, style, name in ARM_STYLES:
            rows = frame.loc[frame.arm.eq(arm)].groupby('update')[key].mean()
            if len(rows):
                ax.plot(rows.index, rows.rolling(10, min_periods=1).mean(), color=color, ls=style, lw=1.1, marker=None, label=name)
        ax.set_xlabel('Update')
        ax.set_ylabel(label)
        ax.grid(True, lw=.4, alpha=.35)
    base.tags(axes)
    axes[0].legend(loc='best', **base.LEGEND)
    save(fig, path)


def validation(frame, frozen, path):
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.3), constrained_layout=True)
    rows = frame.loc[frame.price.eq(frozen['selected_price'])]
    for ax, (key, label) in zip(axes, (('bpcs', 'Validation rate (bpcs)'), ('labels_kept_pct', 'Validation labels preserved (%)'))):
        for arm, color, style, name in ARM_STYLES:
            part = rows.loc[rows.arm.eq(arm)]
            if len(part):
                ax.plot(part['update'], part[key], color=color, ls=style, marker='o', ms=3, label=name)
        chosen = rows.loc[rows.arm.eq(frozen['arm']) & rows['update'].eq(frozen['update'])]
        ax.plot(chosen['update'], chosen[key], ls='none', marker='o', ms=9, mfc='none', mec=BLACK, mew=1.1, label='Frozen')
        ax.set_xlabel('Update')
        ax.set_ylabel(label)
        ax.grid(True, lw=.4, alpha=.35)
    axes[0].legend(loc='best', **base.LEGEND)
    save(fig, path)


def allocation(frame, path):
    test = frame.loc[frame.split.eq('test')]
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.1), constrained_layout=True)
    for ax, key, bins, label in ((axes[0], 'rho', np.linspace(0, 1, 21), 'Retained share'), (axes[1], 'q', np.arange(.5, 13), 'Precision q')):
        ax.hist(test[key], bins=bins, weights=np.full(len(test), 100/len(test)), color=RED, lw=0)
        ax.set_xlabel(label)
        ax.set_ylabel('Share of tiles (%)')
    save(fig, path)


def save(fig, path):
    fig.savefig(f'{path}.pdf')
    fig.savefig(f'{path}.png', dpi=300)
    plt.close(fig)


def table(summary, selected, out):
    keep = ['method', 'cubes', 'bpcs', 'bpcs_max', 'compression_ratio', 'labels_preserved_pct', 'labels_lost', 'labels_detected', 'robust_kept_pct', 'robust_false_per_map', 'detections_new_per_map', 'low_miss', 'low_supported', 'nmse']
    rows = summary.loc[summary.split.eq('test')][keep].sort_values('bpcs')
    rows.to_csv(out/'table_uw_ppo.csv', index=False)
    lines = ['\\begin{tabular}{@{}lrrrrrr@{}}', '\\toprule', 'Method & $R$ & CR & Labels (\\%) & Robust (\\%) & FA/map & NMSE\\\\', '\\midrule']
    for r in rows.itertuples():
        name = r.method.replace('rho=', '$\\rho$=').replace('_', ' ')
        cells = f'{name} & {r.bpcs:.2f} & {r.compression_ratio:.1f} & {r.labels_preserved_pct:.2f} & {r.robust_kept_pct:.2f} & {r.robust_false_per_map:.3f} & {r.nmse:.3f}'
        lines.append(('\\textbf{'+cells.replace(' & ', '} & \\textbf{')+'}' if r.method == selected else cells)+'\\\\')
    (out/'table_uw_ppo.tex').write_text('\n'.join([*lines, '\\bottomrule', '\\end{tabular}', '']), encoding='utf-8')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=ROOT.parent/'paper_figures_uw_ppo')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    base.style()
    frozen = json.loads((args.run/'FROZEN_POLICY.json').read_text())
    selected = f'PPO price={frozen["selected_price"]:g}'
    summary = pd.read_csv(args.run/'SUMMARY.csv')
    tradeoff(summary, selected, args.out/'fig_uw_ppo_tradeoff')
    training(pd.read_csv(args.run/'tables/training.csv'), frozen, args.out/'fig_uw_ppo_training')
    validation(pd.read_csv(args.run/'tables/validation.csv'), frozen, args.out/'fig_uw_ppo_validation')
    allocation(pd.read_csv(args.run/'tables/allocations.csv'), args.out/'fig_uw_ppo_allocation')
    rows = table(summary, selected, args.out)
    motion = pd.read_csv(args.run/'SUMMARY_BY_MOTION.csv')
    motion.loc[motion.split.eq('test') & motion.method.isin([selected, 'Uniform noise rho=1,q=5'])].to_csv(args.out/'table_uw_ppo_motion.csv', index=False)
    with pd.option_context('display.width', 240, 'display.max_columns', 30):
        print(rows.round(3).to_string(index=False))
    print(json.dumps({k: frozen[k] for k in ('arm', 'state', 'learning_rate', 'update', 'selected_price')}))
    print(pd.read_csv(args.run/'STATE_COMPARISON.csv')[['state', 'learning_rate', 'update', 'bpcs', 'labels_kept_pct', 'robust_kept_pct', 'robust_false_per_map']].round({'bpcs': 3, 'labels_kept_pct': 2, 'robust_kept_pct': 2, 'robust_false_per_map': 3}).to_string(index=False))
    print('figures written to', args.out)


if __name__ == '__main__':
    main()
