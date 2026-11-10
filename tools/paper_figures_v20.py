"""Paper rate-sweep figures from the verified TEST results of V17, V18 and V20 (662 cubes, 7 recordings).

Every point is the mean over ALL TEST cubes:
  * labels preserved = 1 - (labels lost) / (labels the original receiver detects), pooled over cubes;
  * robust CFAR detections kept = 1 - (robust detections lost) / (robust original detections), pooled;
  * RD-map error = mean per-cube RMSE (dB);
  * FFTRadNet mAP / mAR / F1 = official metrics over all labelled TEST frames.
The *_ci versions add 95% recording-level bootstrap intervals (2000 draws; frames of one recording are
correlated, so recordings are resampled). For mAR and F1 the interval is that of the paired change vs
the original cube, drawn around the point; mAP has no interval in the run outputs.

Proposed: the fixed-rate support-coded codec at 64x, 48x, 32x, 22.1x, 16x (V20) and the whole-map codec C at
beta 3, 2.5, 2, 1.5 (V17); beta* = 1.5 is the operating point. AdaRadar: complete rate (values + positions +
scales), its own reported rate (values only), and its adaptive control. Reads result CSVs only.

python -B tools/paper_figures_v20.py [--out outputs/paper_figures_v20]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]/'results'/'runs'
import release_io  # noqa: F401,E402  (reads .csv.gz)
RUNS = dict(v17=ROOT/'v17_pub_20261030T001402535Z_verified_20261030T071541560Z',
            v18=ROOT/'v18_fu_20261030T075802880Z_verified_20261030T092823868Z',
            v20=ROOT/'v20_rm_20261030T131713629Z_verified_20261030T154146779Z')
FIXED = ('fixed_0p50', 'fixed_0p67', 'fixed_1p00', 'fixed_1p45', 'fixed_2p00')
C_CURVE = ('closed_full_b3', 'closed_full_b2p5', 'closed_full_b2', 'closed_full_b1p5')
ADARADAR = ('adaradar_r25_s4', 'adaradar_r12p57_s4', 'adaradar_r8_s4', 'adaradar_r4_s4', 'adaradar_r2_s4')
SOURCES = {**{m: 'v20' for m in FIXED+('adaradar_r50_s4',)}, **{m: 'v17' for m in C_CURVE+('baq_1bit', 'baq_2bit', 'baq_3bit')},
           **{m: 'v18' for m in ADARADAR+('adaradar_adaptive_s4',)}}
DRAWS, SEED = 2000, 20261030
RED, BLUE, BLACK, GREY = '#C00000', '#0033B5', 'black', '0.55'
RATIOS = [5, 8, 16, 32, 64, 100]
R_STAR = 1.45


def cube_table(method):
    frame = pd.read_csv(RUNS[SOURCES[method]]/'RESULTS_BY_CUBE.csv')
    frame = frame.loc[frame.role.eq('test') & frame.method.eq(method)]
    if len(frame) != 662:
        raise ValueError(f'{method}: {len(frame)} TEST cubes, expected 662.')
    return frame


def receiver_points(rng):
    """Mean over all TEST cubes and recording-bootstrap 95% intervals, one row per method."""
    rows, draws = [], None
    for method in SOURCES:
        g = cube_table(method)
        sums = g.groupby('scene_name')[['labels_lost', 'labels_detected_original', 'a1_lost_robust', 'a1_original_robust',
                                        'rd_rmse_db', 'bpcs']].sum()
        cubes = g.groupby('scene_name').size().reindex(sums.index).to_numpy(float)
        if draws is None:
            draws = rng.integers(0, len(sums), (DRAWS, len(sums)))
        pick = lambda column: sums[column].to_numpy(float)[draws].sum(1)
        stats = dict(labels=100*(1-pick('labels_lost')/pick('labels_detected_original')),
                     cfar=100*(1-pick('a1_lost_robust')/pick('a1_original_robust')), rd=pick('rd_rmse_db')/cubes[draws].sum(1))
        point = dict(labels=100*(1-g.labels_lost.sum()/g.labels_detected_original.sum()),
                     cfar=100*(1-g.a1_lost_robust.sum()/g.a1_original_robust.sum()), rd=g.rd_rmse_db.mean())
        row = dict(method=method, bpcs=g.bpcs.mean(), bpcs_nonzero_only=g.bpcs_nonzero_only.mean() if 'bpcs_nonzero_only' in g else np.nan,
                   labels_lost=int(g.labels_lost.sum()), labels_detected=int(g.labels_detected_original.sum()),
                   recordings=len(sums), cubes=len(g))
        for key, value in point.items():
            row.update({key: value, f'{key}_lo': np.percentile(stats[key], 2.5), f'{key}_hi': np.percentile(stats[key], 97.5)})
        rows.append(row)
    return pd.DataFrame(rows).set_index('method')


def perception_points():
    rows, original = [], None
    for method in ('original', *SOURCES):
        run = RUNS['v17' if method == 'original' else SOURCES[method]]
        p = pd.read_csv(run/'PERCEPTION_SUMMARY.csv')
        r = p.loc[p.role.eq('test') & p.subset.eq('all') & p.method.eq(method)].iloc[0]
        rows.append(dict(method=method, mAP=100*r.mAP, mAR=100*r.mAR, F1=100*r.F1,
                         F1_lo=100*(r.F1-r.dF1+r.get('dF1_ci_low', np.nan)), F1_hi=100*(r.F1-r.dF1+r.get('dF1_ci_high', np.nan)),
                         mAR_lo=100*(r.mAR-r.dmAR+r.get('dmAR_ci_low', np.nan)), mAR_hi=100*(r.mAR-r.dmAR+r.get('dmAR_ci_high', np.nan))))
    table = pd.DataFrame(rows).set_index('method')
    for method in SOURCES:  # every run scores the same frames with the same network
        run = RUNS[SOURCES[method]]
        p = pd.read_csv(run/'PERCEPTION_SUMMARY.csv')
        o = p.loc[p.role.eq('test') & p.subset.eq('all') & p.method.eq('original')].iloc[0]
        if abs(100*o.F1-table.loc['original', 'F1']) > 1e-9:
            raise ValueError(f'{run.name}: the original F1 differs; the runs are not comparable.')
    return table


def style():
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 8,
                         'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 6, 'xtick.labelsize': 7,
                         'ytick.labelsize': 7, 'axes.linewidth': .6, 'lines.linewidth': 1.2, 'lines.markersize': 4,
                         'mathtext.fontset': 'stix', 'pdf.fonttype': 42, 'ps.fonttype': 42})


def axes_setup(ax, ylabel, xlim, xticks):
    ax.set_xscale('log')
    ax.set_xlim(*xlim)
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xlabel('Bit rate (bits per complex sample)')
    ax.set_ylabel(ylabel)
    ax.grid(True, which='major', lw=.4, alpha=.35)
    ax.tick_params(direction='in', which='both', top=False)
    ax.axvline(R_STAR, color=GREY, lw=.6, ls=(0, (4, 3)), zorder=0)
    twin = ax.twiny()
    twin.set_xscale('log')
    twin.set_xlim(ax.get_xlim())
    twin.xaxis.set_major_locator(FixedLocator([32/r for r in RATIOS if xlim[0] <= 32/r <= xlim[1]]))
    twin.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{32/v:g}×'))
    twin.xaxis.set_minor_locator(FixedLocator([]))
    twin.set_xlabel('Compression ratio vs raw ADC (32 bpcs)', labelpad=3)
    twin.tick_params(direction='in', labelsize=7)


def curves(ax, table, key, original, ci, reported=None):
    """Proposed (red: fixed-rate support-coded and C, two variants, two lines), AdaRadar (blue), BAQ (black).
    With ``ci``, 95% intervals on the two compared methods, the proposed codec and AdaRadar's complete rate."""
    def draw(rows, interval=False, **kw):
        rows = rows.sort_values('bpcs')
        y = rows[key]
        if ci and interval and f'{key}_lo' in rows and rows[f'{key}_lo'].notna().all():
            err = np.vstack([(y-rows[f'{key}_lo']).clip(lower=0), (rows[f'{key}_hi']-y).clip(lower=0)])
            ax.errorbar(rows.bpcs, y, yerr=err, fmt='none', ecolor=kw.get('color', BLACK), elinewidth=.7, capsize=1.6,
                        alpha=.7, zorder=4)
        ax.plot(rows.bpcs, y, **kw)
    ax.axhline(original, color=BLACK, ls=(0, (1, 1.5)), lw=.9, label='Original')
    draw(table.loc[list(FIXED)], True, color=RED, ls='-', marker='o', mfc='white', zorder=6, label='Proposed, fixed rate')
    draw(table.loc[list(C_CURVE)], True, color=RED, ls='-', marker='o', zorder=6, label='Proposed, C $\\beta$ 3$\\to$1.5')
    star = table.loc[['closed_full_b1p5']]
    ax.plot(star.bpcs, star[key], ls='none', marker='o', ms=9, mfc='none', mec=RED, mew=.9, zorder=7,
            label='Proposed, $\\beta^*$ = 1.5')
    draw(table.loc[['adaradar_r50_s4', *ADARADAR]], True, color=BLUE, ls='--', marker='s', mfc='white',
         label='AdaRadar, complete rate')
    if reported is not None:
        ada = table.loc[['adaradar_r50_s4', *ADARADAR]]
        ax.plot(reported.loc[ada.index], ada[key], color=BLUE, ls=':', marker='s', ms=3, alpha=.6, label='AdaRadar, reported rate')
    draw(table.loc[['adaradar_adaptive_s4']], color=BLUE, ls='none', marker='D', ms=4.5, label='AdaRadar adaptive')
    draw(table.loc[['baq_1bit', 'baq_2bit', 'baq_3bit']], color=BLACK, ls='-.', marker='^', mfc='white', label='BAQ 1/2/3 bit')
    ax.plot([], [], color=GREY, lw=.6, ls=(0, (4, 3)), label='1.45 bpcs (AdaRadar point)')


def tags(axes):
    for ax, tag in zip(axes, 'abc'):
        ax.text(.02, .97, f'({tag})', transform=ax.transAxes, va='top', fontsize=8, bbox=dict(fc='white', ec='none', pad=.5, alpha=.8))


LEGEND = dict(frameon=True, framealpha=.92, edgecolor='0.7', borderpad=.3, handlelength=2.2, labelspacing=.25)


def split_legend(axes, proposed, baselines):
    """Original and proposed in one panel, the baselines in another, each in an empty lower-right corner."""
    handles, labels = axes[0].get_legend_handles_labels()
    ours = [i for i, label in enumerate(labels) if label.startswith(('Original', 'Proposed'))]
    rest = [i for i in range(len(labels)) if i not in ours]
    for ax, pick in ((axes[proposed], ours), (axes[baselines], rest)):
        ax.legend([handles[i] for i in pick], [labels[i] for i in pick], loc='lower right', **LEGEND)


def limits(values, pad=.06, top=None):
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    span = high-low
    return low-pad*span, (top if top is not None else high+pad*span)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=ROOT.parent/'paper_figures_v20')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    receiver = receiver_points(np.random.default_rng(SEED))
    perception = perception_points()
    table = receiver.join(perception.drop(index='original'))
    table.to_csv(args.out/'RATE_SWEEP_FIGURE_DATA.csv')
    style()
    lo_rate = .9*table.bpcs.min()
    for ci in (False, True):
        suffix = '_ci' if ci else ''
        # Receiver preservation and distortion.
        fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5), constrained_layout=True)
        for ax, key, label, top in zip(axes, ('labels', 'cfar', 'rd'),
                                       ('Labels preserved (%)', 'Robust CFAR detections kept (%)', 'RD-map error (dB RMSE)'),
                                       (100.4, 102., None)):
            curves(ax, table, key, 0. if key == 'rd' else 100., ci)
            axes_setup(ax, label, (lo_rate, 7.5), [.5, .75, 1, 1.5, 2, 3, 4, 6])
            values = pd.concat([table[key], table[f'{key}_lo'], table[f'{key}_hi']]) if ci else table[key]
            ax.set_ylim(*limits(values, top=top))
        tags(axes)
        split_legend(axes, 0, 1)
        fig.savefig(args.out/f'fig_rate_sweep_preservation{suffix}.pdf')
        fig.savefig(args.out/f'fig_rate_sweep_preservation{suffix}.png', dpi=300)
        plt.close(fig)
        # FFTRadNet perception.
        reported = table.bpcs_nonzero_only
        fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5), constrained_layout=True)
        for ax, key, label in zip(axes, ('mAP', 'mAR', 'F1'), ('FFTRadNet mAP (%)', 'FFTRadNet mAR (%)', 'FFTRadNet F1 (%)')):
            curves(ax, table, key, perception.loc['original', key], ci, reported)
            axes_setup(ax, label, (.9*reported.min(), 7.5), [.2, .3, .5, 1, 2, 4, 6])
            values = [table[key], [perception.loc['original', key]]]
            if ci and f'{key}_lo' in table:
                values += [table[f'{key}_lo'], table[f'{key}_hi']]
            ax.set_ylim(*limits(np.concatenate([np.asarray(v, float) for v in values])))
        tags(axes)
        split_legend(axes, 1, 2)
        fig.savefig(args.out/f'fig_rate_sweep_fftradnet{suffix}.pdf')
        fig.savefig(args.out/f'fig_rate_sweep_fftradnet{suffix}.png', dpi=300)
        plt.close(fig)
    with pd.option_context('display.width', 200):
        print(table[['bpcs', 'labels', 'labels_lo', 'labels_hi', 'cfar', 'cfar_lo', 'cfar_hi', 'rd', 'mAP', 'mAR', 'F1']].round(2).to_string())
    print('figures written to', args.out)


if __name__ == '__main__':
    main()
