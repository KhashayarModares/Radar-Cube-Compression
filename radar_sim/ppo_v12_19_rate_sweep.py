"""Compression-only validation diagnostics; no active-pipeline import or PPO.

Uniform existing-library allocations and existing full-support FFT quantizers
are separate codec families, not new PPO actions or calibrated RDO policies.
Rates include each codec's declared syntax overhead but are not serialized
bitstream measurements. No policy is selected or repaired by this module.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

FFT_BITS = (15, 14, 12, 10, 8, 6, 4)
COUNT_COLUMNS = (
    'num_labels_supported_original', 'num_labels_preserved_from_original',
    'num_labels_missed_from_original', 'num_label_unmatched_peaks_original',
    'num_label_unmatched_peaks_method',
)


def summarize_operating_points(sweeps, quality):
    """Ratios use pooled integer counts, never averages of label percentages."""
    group_keys = ['policy_id', 'family', 'map_type']
    combined = sweeps.groupby(
        ['record_name', 'policy_id', 'family', 'threshold_factor'], as_index=False
    )[list(COUNT_COLUMNS)].sum()
    combined['map_type'] = 'RD+RA'
    measured = pd.concat([sweeps, combined], ignore_index=True)
    rows = []
    low = float(sweeps.threshold_factor.min())
    for key, group in measured.groupby(group_keys, sort=False):
        policy_id, family, kind = key
        qrows = quality.loc[quality.policy_id == policy_id]
        if not qrows.record_name.is_unique:
            raise AssertionError('Each fixed policy needs exactly one quality row per cube.')
        source_bits = float(qrows.source_bits.sum())
        rate = float(qrows.realized_bits.sum()/qrows.complex_samples.sum())
        for condition, frame in (
            ('lowest', group.loc[group.threshold_factor == low]), ('six_average', group)
        ):
            supported = int(frame.num_labels_supported_original.sum())
            preserved = int(frame.num_labels_preserved_from_original.sum())
            misses = int(frame.num_labels_missed_from_original.sum())
            if supported-preserved != misses or not 0 <= preserved <= supported:
                raise AssertionError('Original-supported object counting is inconsistent.')
            rows.append(dict(
                policy_id=policy_id, family=family, map_type=kind, condition=condition,
                action_index=int(qrows.action_index.iloc[0]), rho=float(qrows.rho.iloc[0]),
                q=int(qrows.q.iloc[0]), validation_cubes=int(qrows.record_name.nunique()),
                measured_cube_threshold_pairs=len(frame),
                threshold_count=int(frame.threshold_factor.nunique()),
                supported_original_instances=supported, preserved_instances=preserved,
                missed_instances=misses,
                preservation_percent=100.0*preserved/supported if supported else np.nan,
                mean_supported_labels_per_cube_threshold=float(frame.num_labels_supported_original.mean()),
                mean_preserved_labels_per_cube_threshold=float(frame.num_labels_preserved_from_original.mean()),
                mean_missed_labels_per_cube_threshold=float(frame.num_labels_missed_from_original.mean()),
                unmatched_per_cube_threshold=float(frame.num_label_unmatched_peaks_method.mean()),
                original_unmatched_per_cube_threshold=float(frame.num_label_unmatched_peaks_original.mean()),
                unmatched_excess_per_cube_threshold=float((
                    frame.num_label_unmatched_peaks_method-frame.num_label_unmatched_peaks_original).mean()),
                positive_unmatched_excess_per_cube_threshold=float(np.maximum(
                    frame.num_label_unmatched_peaks_method-frame.num_label_unmatched_peaks_original, 0).mean()),
                accounted_bpcs=rate,
                worst_accounted_bpcs=float(qrows.accounted_bpcs.max())
                    if 'accounted_bpcs' in qrows else float((qrows.realized_bits/qrows.complex_samples).max()),
                compression_rate_reduction_percent=100.0*(1-qrows.realized_bits.sum()/source_bits),
                mean_MiB_saved=float(qrows.MiB_saved.mean()),
                mean_cube_nmse=float(qrows.cube_nmse.mean()),
                worst_cube_nmse=float(qrows.cube_nmse.max()),
                mean_RD_log_power_mse=float(qrows.RD_log_power_mse.mean()),
                mean_RA_log_power_mse=float(qrows.RA_log_power_mse.mean()),
                above_PPO_4p8_cap=bool(np.any(qrows.realized_bits/qrows.complex_samples > 4.8+1e-8)),
                PPO_cap_enforced=False, actual_bitstream_saved=False,
                diagnostic_operating_point_only=True,
            ))
    return pd.DataFrame(rows)


def plot_rate_sweep(ex, summary, directory):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    paths = []
    colors = {'RD': '#1675ba', 'RA': '#d66323'}
    for metric, ylabel, filename in (
        ('preservation_percent', 'Original-supported object instances preserved [%]',
         'compression_sweep_true_label_preservation_v12_19.png'),
        ('unmatched_per_cube_threshold', 'Peaks outside all GT gates / cube / threshold',
         'compression_sweep_unmatched_peaks_v12_19.png'),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        # Reserve explicit header/footer space. A shared external legend must
        # not cover low-count FFT points or overlap the panel titles.
        fig.subplots_adjust(left=.07, right=.985, top=.78, bottom=.24, wspace=.25)
        for ax, condition, title in zip(axes, ('lowest', 'six_average'), (
            'Lowest CFAR threshold: factor 1.25',
            'Six-threshold mean: factors 1.25, 2, 4, 8, 16, 32',
        )):
            for kind, color in colors.items():
                frame = summary.loc[(summary.map_type == kind) & (summary.condition == condition)]
                for family, marker, style in (
                    ('FFT_full_support', 'o', '--'), ('Uniform_library', 's', 'None'),
                    ('Original', '*', 'None'),
                ):
                    points = frame.loc[frame.family == family].sort_values(
                        ['compression_rate_reduction_percent', 'policy_id'])
                    ax.plot(points.compression_rate_reduction_percent, points[metric],
                            color=color, marker=marker, linestyle=style, linewidth=1.1,
                            markersize=10 if family == 'Original' else 5,
                            markerfacecolor='none' if family == 'Uniform_library' else color)
                    if family == 'Uniform_library':
                        # Two physical reference actions, not selected/deployed policies.
                        for point in points.loc[points.action_index.isin([11, 13])].itertuples():
                            ax.annotate(f'A{point.action_index}',
                                (point.compression_rate_reduction_percent, getattr(point, metric)),
                                xytext=(4, 5 if kind == 'RD' else -12),
                                textcoords='offset points', fontsize=7, color=color)
            ax.axvline(85., color='0.5', linestyle=':', linewidth=1)
            ax.set(title=title, xlabel='Accounted encoded-rate reduction vs 32-bpc source [%]',
                   ylabel=ylabel, xlim=(-1, 100))
            ax.grid(alpha=.22)
            ax.set_ylim(bottom=0)
            if metric == 'preservation_percent':
                ax.set_ylim(0, 103)
        handles = [Line2D([], [], color=color, label=kind) for kind, color in colors.items()]
        handles += [
            Line2D([], [], color='black', marker='*', linestyle='None', label='Original, 0% reduction'),
            Line2D([], [], color='black', marker='o', linestyle='--', label='Full-support FFT quantization'),
            Line2D([], [], color='black', marker='s', markerfacecolor='none', linestyle='None',
                   label='16 unchanged uniform library actions'),
            Line2D([], [], color='0.5', linestyle=':', label='4.8-bpcs cap = 85% rate reduction'),
        ]
        fig.legend(handles=handles, fontsize=8, loc='upper center',
                   bbox_to_anchor=(.5, .94), ncol=3, frameon=False)
        fig.suptitle('V12.19 compression-only validation sweep — no PPO training or action selection', y=.985)
        fig.text(.5, .04, 'Measured operating points only; dashed connections are guides, not interpolated performance.\n'
                      'Six-threshold preservation = sum preserved / sum original-supported object–threshold instances.\n'
                      'Outside-GT peaks are not confirmed physical false alarms; 100% reduction is not evaluated.',
                 ha='center', va='bottom', fontsize=8)
        path = directory/filename
        ex.save_publication_figure(fig, path)
        plt.close(fig)
        paths.append(path)
    return paths


def run_compression_rate_sweep(ex, rows, labels, output_dir):
    """Evaluate original plus 23 fixed diagnostic points on validation only."""
    output = Path(output_dir)
    if output.exists():
        raise ValueError('Compression-sweep output must be a NEW directory.')
    if rows.empty or not rows.record_name.is_unique or set(rows.split) != {'validation'}:
        raise ValueError('Supply unique saved validation rows only; never held-out test rows.')
    thresholds = tuple(map(float, ex._nearest_training_thresholds_v11()))
    if thresholds != (1.25, 2.0, 4.0, 8.0, 16.0, 32.0) or len(ex.ACTION_LIBRARY_V11) != 16:
        raise AssertionError('Require the unchanged six-threshold detector protocol and 16 actions.')
    output.mkdir(parents=True)
    rows.to_csv(output/'protocol_validation.csv', index=False, mode='x')
    protocol = dict(validation_records=rows.record_name.tolist(), thresholds=list(thresholds),
        original_source_bpcs=32, compression_x='100*(1-accounted_bits/(32*complex_samples))',
        full_support_fft_q=list(FFT_BITS), uniform_actions=[list(x) for x in ex.ACTION_LIBRARY_V11],
        full_support_fft_axes=[0, 1, 2], uniform_codec_axes=[1, 2],
        families_are_distinct_codecs=True, main_PPO_action_library_changed=False,
        uniform_points_are_calibrated_RDO=False, PPO_training=False, optimizer_updates=0,
        PPO_cap_enforced=False, diagnostic_points_can_exceed_PPO_cap=True,
        heldout_test_loaded=False, heldout_test_evaluated=False, action_selection=False,
        deployment_refinement=False, actual_bitstream_saved=False,
        rate_accounting='declared syntax, including values/support/headers; not measured file sizes',
        object_counting='original-supported object counted once; any valid in-gate fixed-NMS peak',
        six_preservation='sum preserved / sum original-supported object-threshold instances',
        unmatched='outside all evaluable GT gates; not confirmed physical false alarms',
        interpolation_used=False, monotonicity_enforced=False, hundred_percent_endpoint_evaluated=False)
    with (output/'protocol.json').open('x', encoding='utf-8') as stream:
        json.dump(protocol, stream, indent=2, allow_nan=False)
    quality, sweep_frames, processors = [], [], {}
    for cube_number, (_, row) in enumerate(rows.iterrows(), 1):
        print(f'Compression sweep cube {cube_number}/{len(rows)}: {row.record_name}', flush=True)
        cube = ex.load_cube_v6_1000(row)
        labs = ex.labels_for_cube_v6_1000(labels, row)
        processor = ex._postprocessor_for_cube_v11(cube, processors)
        original = ex.postprocess_cube(cube, processor)
        caches, originals = {}, {}
        for kind, gate in (('RD', ex.CFAR_LABEL_GATE_RD_V8), ('RA', ex.CFAR_LABEL_GATE_RA_V8)):
            data, ranges, second = ex._map_triplet_v12_6(original, kind)
            originals[kind] = data
            caches[kind] = ex.precompute_original_cfar_count_sweep_v11_2(
                data, labs, ranges, second, kind, thresholds, gate)
        context = ex.prepare_policy_source_v12_12(cube)

        def record_point(policy_id, family, decoded, meta, action=-1, rho=1.0, q=0):
            post = original if family == 'Original' else ex.postprocess_cube(decoded, processor)
            bits = float(meta['realized_bits'])
            source_bits = 32*cube.size
            item = dict(record_name=str(row.record_name), scene_name=str(row.scene_name),
                policy_id=policy_id, family=family, action_index=action, rho=rho, q=q,
                complex_samples=cube.size, source_bits=source_bits, realized_bits=bits,
                accounted_bpcs=bits/cube.size,
                compression_rate_reduction_percent=100.0*(1-bits/source_bits),
                compression_ratio=source_bits/bits, MiB_saved=(source_bits-bits)/(8*1024**2),
                cube_nmse=float(meta['cube_nmse']), above_PPO_4p8_cap=bool(bits/cube.size > 4.8+1e-8),
                actual_bitstream_saved=False, PPO_cap_enforced=False)
            for kind in ('RD', 'RA'):
                data, _, _ = ex._map_triplet_v12_6(post, kind)
                frame = ex.evaluate_cfar_count_sweep_method_v11_2(caches[kind], data)
                frame['record_name'], frame['scene_name'] = str(row.record_name), str(row.scene_name)
                frame['policy_id'], frame['family'] = policy_id, family
                frame['action_index'], frame['rho'], frame['q'] = action, rho, q
                sweep_frames.append(frame)
                item[kind+'_log_power_mse'] = ex._normalized_log_power_mse_v11(originals[kind], data)
            quality.append(item)

        record_point('Original', 'Original', cube, dict(realized_bits=32*cube.size, cube_nmse=0.0))
        # Existing full-support quantization, including the receiver FFT, is a
        # different family from the coherent shared-antenna-support tile codec.
        for q in FFT_BITS:
            name = f'FFT_full_q{q:02d}'
            decoded, _, meta = ex.compress_cube_fft_full_support_multi(cube, {name: q})[name]
            record_point(name, 'FFT_full_support', decoded, meta, q=q)
            del decoded
        for index, (rho, q) in enumerate(ex.ACTION_LIBRARY_V11):
            sequence = np.full(len(context['tile_slices']), index, dtype=np.int64)
            encoded = ex.coherent_codec_v14.encode_sequence(cube, sequence, source=context)
            record_point(f'Uniform_A{index:02d}_rho{rho:.2f}_q{q:02d}', 'Uniform_library',
                         encoded['cube_hat'], encoded['meta'], action=index, rho=float(rho), q=int(q))
            del encoded
        del cube, context, original
        gc.collect()
    quality = pd.DataFrame(quality)
    sweeps = pd.concat(sweep_frames, ignore_index=True)
    summary = summarize_operating_points(sweeps, quality)
    for name, frame in (('per_cube_quality', quality), ('per_threshold_counts', sweeps),
                        ('operating_points', summary)):
        frame.to_csv(output/f'compression_sweep_{name}_v12_19.csv', index=False, mode='x')
    figures = plot_rate_sweep(ex, summary, output)
    with (output/'status.json').open('x', encoding='utf-8') as stream:
        json.dump(dict(complete=True, validation_cubes=len(rows), operating_points_per_cube=24,
                       figures=[str(x) for x in figures], optimizer_updates=0,
                       heldout_test_evaluated=False), stream, indent=2)
    return dict(quality=quality, sweeps=sweeps, summary=summary, figures=figures)
