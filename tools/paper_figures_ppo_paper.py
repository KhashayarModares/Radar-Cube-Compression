"""Rate-sweep figure and table data of the PPO paper: PPO, AdaRadar, Learned ECDQ and BAQ on the same 662 RADIal TEST cubes.

Rates are complete rates. AdaRadar comes from V18 and V20, BAQ from V17, and Learned ECDQ is the whole-map dithered codec with its
learned entropy model and no detector at the encoder: the V21 rows up to 2.11 bpcs, which ran without repair rounds, and the
open-loop V17 and V19 rows above. Reads result CSVs only.

python -B tools/paper_figures_ppo_paper.py [--out outputs/paper_figures_ppo_paper]
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
import paper_figures_v20 as base
from paper_figures_v20 import BLACK, BLUE, RED

GREEN = '#1a7f37'
REMOTE = Path(__file__).resolve().parents[1]/'results'/'runs'
RUNS = dict(ppo='ppo_tuned_20261031T222106858Z_verified_20261101T073239520Z', v17='v17_pub_20261030T001402535Z_verified_20261030T071541560Z',
            v18='v18_fu_20261030T075802880Z_verified_20261030T092823868Z', v19='v19_fin_20261030T094623709Z_verified_20261030T104143728Z',
            v20='v20_rm_20261030T131713629Z_verified_20261030T154146779Z', v21='v21_sw_20261030T165446096Z_verified_20261030T203953211Z')
SERIES = dict(PPO=(('ppo', ('ppo_0p46', 'ppo_0p82', 'ppo_1p45', 'ppo_3p63', 'ppo_4p73')),),
              AdaRadar=(('v20', ('adaradar_r50_s4',)), ('v18', ('adaradar_r25_s4', 'adaradar_r12p57_s4', 'adaradar_r8_s4', 'adaradar_r4_s4', 'adaradar_r2_s4'))),
              AdaRadar_feedback=(('v18', ('adaradar_adaptive_s4',)),),
              Learned_ECDQ=(('v21', ('c_0p46', 'c_0p82', 'c_1p00', 'c_1p45', 'c_2p00', 'c_2p11')), ('v17', ('ecdq_b2', 'ecdq_b1')), ('v19', ('ecdq_b1p5',))),
              BAQ=(('v17', ('baq_1bit', 'baq_2bit', 'baq_3bit')),),
              Zero_filled=(('v17', ('legacy_dct_r025_q8',)),))
PANELS = (('labels', 'Labels preserved (%)', (90, 100.5)), ('robust', 'Robust CFAR detections kept (%)', (0, 104)), ('false', 'Robust false peaks per map', None))
PERCEPTION = (('mAP', 'FFTRadNet mAP (%)', (86.5, 94.6)), ('mAR', 'FFTRadNet mAR (%)', (4, 84)), ('F1', 'FFTRadNet F1 (%)', (17, 90)))
ORIGINAL = dict(mAP=93.1018, mAR=79.0593, F1=85.5079)
XTICKS = [.5, 1, 2, 3, 4, 6]
STYLES = dict(PPO=dict(color=RED, ls='-', marker='o', ms=4, zorder=7, label='PPO'),
              AdaRadar=dict(color=BLUE, ls='--', marker='s', mfc='white', ms=3.6, zorder=6, label='AdaRadar'),
              Learned_ECDQ=dict(color=GREEN, ls='-.', marker='D', mfc='white', ms=3.2, zorder=5, label='Learned ECDQ'),
              BAQ=dict(color=BLACK, ls=':', marker='^', mfc='white', ms=4, zorder=4, label='BAQ'))


def points(run, methods):
    """Means over all TEST cubes: counts pooled over cubes, false peaks and RD error per map, FFTRadNet over labelled frames."""
    folder = REMOTE/RUNS[run]
    cubes = pd.read_csv(folder/'RESULTS_BY_CUBE.csv', low_memory=False)
    cubes = cubes.loc[cubes.role.eq('test') & cubes.method.isin(methods)]
    seen = pd.read_csv(folder/'PERCEPTION_SUMMARY.csv')
    seen = seen.loc[seen.role.eq('test') & seen.subset.eq('all')].set_index('method')
    rows = []
    for method, g in cubes.groupby('method', sort=False):
        s = seen.loc[method]
        rows.append(dict(method=method, cubes=len(g), bpcs=g.bpcs.mean(), ratio=32/g.bpcs.mean(), labels_lost=int(g.labels_lost.sum()), labels=100*(1-g.labels_lost.sum()/g.labels_detected_original.sum()),
                         robust=100*(1-g.a1_lost_robust.sum()/g.a1_original_robust.sum()), false=g.a1_new_robust.mean(), rd=g.rd_rmse_db.mean(), mAP=100*s.mAP, mAR=100*s.mAR, F1=100*s.F1,
                         F1_lo=100*(s.F1-s.dF1+s.dF1_ci_low), F1_hi=100*(s.F1-s.dF1+s.dF1_ci_high), mAR_lo=100*(s.mAR-s.dmAR+s.dmAR_ci_low), mAR_hi=100*(s.mAR-s.dmAR+s.dmAR_ci_high)))
    return pd.DataFrame(rows)


def table():
    parts = [points(run, methods).assign(system=system) for system, sources in SERIES.items() for run, methods in sources]
    frame = pd.concat(parts, ignore_index=True)
    if not frame.cubes.eq(662).all():
        raise AssertionError('Every point must cover all 662 TEST cubes.')
    return frame.sort_values(['system', 'bpcs']).reset_index(drop=True)


def figure(frame, path):
    base.style()
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.15), constrained_layout=True)
    for ax, (key, label, limits) in zip(axes, PANELS):
        ax.axhline(0 if key == 'false' else 100, color=BLACK, ls=(0, (1, 1.5)), lw=.8, label='Original')
        for system, style in STYLES.items():
            rows = frame.loc[frame.system.eq(system)].sort_values('bpcs')
            ax.plot(rows.bpcs, rows[key], lw=1.2, **style)
        base.axes_setup(ax, label, (.4, 7), XTICKS)
        for line in ax.get_lines():
            if line.get_xdata()[0] == base.R_STAR and line.get_xdata()[-1] == base.R_STAR:
                line.remove()
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


def perception_figure(frame, path, ci):
    """FFTRadNet with its official weights. The recording bootstrap covers mAR and F1; mAP has no interval."""
    base.style()
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5), constrained_layout=True)
    for ax, (key, label, limits) in zip(axes, PERCEPTION):
        ax.axhline(ORIGINAL[key], color=BLACK, ls=(0, (1, 1.5)), lw=.8, label='Original')
        for system, style in STYLES.items():
            rows = frame.loc[frame.system.eq(system)].sort_values('bpcs')
            if ci and f'{key}_lo' in rows:
                error = np.vstack([(rows[key]-rows[f'{key}_lo']).clip(lower=0), (rows[f'{key}_hi']-rows[key]).clip(lower=0)])
                ax.errorbar(rows.bpcs, rows[key], yerr=error, fmt='none', ecolor=style['color'], elinewidth=.6, capsize=1.4, alpha=.65, zorder=style['zorder']-1)
            ax.plot(rows.bpcs, rows[key], lw=1.2, **style)
        base.axes_setup(ax, label, (.4, 7), XTICKS)
        for line in ax.get_lines():
            if line.get_xdata()[0] == base.R_STAR and line.get_xdata()[-1] == base.R_STAR:
                line.remove()
        ax.set_ylim(*limits)
    base.tags(axes)
    axes[1].legend(loc='lower right', **base.LEGEND)
    fig.savefig(f'{path}.pdf')
    fig.savefig(f'{path}.png', dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1]/'outputs/paper_figures_ppo_paper')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frame = table()
    frame.to_csv(args.out/'RATE_SWEEP_662_CUBES.csv', index=False)
    figure(frame, args.out/'fig_rate_sweep_preservation')
    perception_figure(frame, args.out/'fig_rate_sweep_fftradnet', False)
    perception_figure(frame, args.out/'fig_rate_sweep_fftradnet_ci', True)
    with pd.option_context('display.width', 200, 'display.max_columns', 20, 'display.max_rows', 60):
        print(frame.drop(columns='cubes').round(2).to_string(index=False))
    print('figure written to', args.out)


if __name__ == '__main__':
    main()
