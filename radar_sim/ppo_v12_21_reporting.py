"""Additional V12.21 reports; no active pipeline import, training or encoding.

The inherited ANY-in-gate original-relative preservation metrics are unchanged.
This module adds a separate annotation-detection protocol: confidence-ranked,
one-to-one object matching, duplicate detections counted as annotation FPs, and
all-points interpolated AP. Geometry uses existing radar gates, NOT VOC box IoU.
AP integration/duplicate convention: official VOC2012 devkit, sections3.4.1/4.4:
https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/htmldoc/index.html
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from radar_sim.ppo_v12_14_margin import cyclic_column_period


VOC_REFERENCE = 'https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/htmldoc/index.html'
DEFAULT_THRESHOLDS = (1.25, 2., 4., 8., 16., 32.)


def bits_saved_percent(bpcs, source_bpcs=32.):
    """No clipping: rates above the int16-IQ reference produce negative savings."""
    rate = np.asarray(bpcs, dtype=float)
    if not np.isfinite(rate).all() or np.any(rate < 0) or source_bpcs <= 0:
        raise ValueError('Rates must be finite/nonnegative and the source rate positive.')
    return 100.*(1.-rate/float(source_bpcs))


def detection_metric_config():
    return dict(
        bitrate_cap=None, source_bpcs=32, bits_saved_x='100*(1-accounted_bpcs/32); negative values retained',
        candidate_pool='ALL positive finite-score fixed power-NMS peaks in complete CA-CFAR windows; no lowest-threshold truncation',
        confidence='linear CA-CFAR peak power/background estimate, not a calibrated probability',
        object_population='all finite in-axis annotation objects; NOT only original-supported objects',
        matching='descending confidence; nearest normalized geometric gate; one detection per object; duplicates FP',
        matching_ties='score ties use record name then row/column; PR points group equal confidence',
        ap='area under all-unique-score precision envelope; no 11-point or 101-point approximation',
        plotted_thresholds='independent reporting grid; never truncates ranked AP candidates',
        f1='2*TP/(2*TP+FP+FN) with pooled object counts at the displayed strict score>CFAR factor',
        original_preservation='unchanged any-in-gate support relative to original map; distinct from AP/recall',
        annotation_caveat='outside-gate peaks and duplicate FPs are annotation-based, not confirmed physical false alarms',
        rate_caveat='analytically codec-accounted bits, not measured serialized wireless payload bytes',
        interpolation_reference=VOC_REFERENCE)


def _coordinates(values, name):
    points = np.asarray(values, dtype=float)
    if points.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if (points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all()
            or np.any(points < 0) or np.any(points != np.rint(points))):
        raise ValueError(name+' must contain finite nonnegative integer row/column pairs.')
    return points.astype(np.int64)


def ranked_object_metrics(frames, thresholds=DEFAULT_THRESHOLDS):
    """Pool same-method/map frames, retaining cube identity during matching.

    Each frame: record_name, label_bins, peak_points, scores, gate; optional
    ``cyclic_period`` declares native cyclic Doppler column coordinates.
    A detection
    chooses its nearest gated GT independently of previous assignments. If that
    object is already matched, the detection is a duplicate FP (not reassigned
    to a farther annotation). This is explicitly NOT many-to-many preservation.
    """
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.ndim != 1 or not len(thresholds) or not np.isfinite(thresholds).all() or np.any(thresholds < 0):
        raise ValueError('Supply finite nonnegative CFAR factors.')
    frames = list(frames)
    if not frames or len({str(f['record_name']) for f in frames}) != len(frames):
        raise ValueError('Supply nonempty frames with unique cube identities.')
    detections, total_labels, prepared = [], 0, []
    for frame in frames:
        labels = _coordinates(frame['label_bins'], 'label_bins')
        peaks = _coordinates(frame['peak_points'], 'peak_points')
        period = cyclic_column_period(frame['label_bins'], period=frame.get('cyclic_period'))
        if period is not None and (np.any(labels[:, 1] >= period) or np.any(peaks[:, 1] >= period)):
            raise ValueError('Cyclic label/peak columns must lie inside their declared period.')
        scores = np.asarray(frame['scores'], dtype=float)
        gate = np.asarray(frame['gate'], dtype=float)
        if (scores.shape != (len(peaks),) or not np.isfinite(scores).all() or np.any(scores <= 0)
                or gate.shape != (2,) or not np.isfinite(gate).all() or np.any(gate < 1)
                or np.any(gate != np.rint(gate))):
            raise ValueError('Positive finite scores, aligned peaks and positive integer gates required.')
        if len(np.unique(peaks, axis=0)) != len(peaks):
            raise ValueError('Fixed-NMS candidate positions must be unique within a cube.')
        used = set()
        name = str(frame['record_name'])
        order = sorted(range(len(peaks)), key=lambda i:(-scores[i], *peaks[i]))
        for index in order:
            delta = np.abs(labels-peaks[index])
            if period is not None:
                delta[:, 1] = np.minimum(delta[:, 1], period-delta[:, 1])
            inside = np.all(delta <= gate, axis=1)
            target, duplicate, outside = -1, False, not bool(inside.any())
            if not outside:
                candidates = np.flatnonzero(inside)
                distances = np.sum((delta[candidates]/gate)**2, axis=1)
                target = int(candidates[int(np.argmin(distances))])
                duplicate = target in used
                used.add(target)
            tp = int(not outside and not duplicate)
            detections.append(dict(record_name=name, row=int(peaks[index, 0]), col=int(peaks[index, 1]),
                confidence=float(scores[index]), matched_label_index=target,
                tp=tp, fp=1-tp, duplicate_fp=int(duplicate), outside_gate_unmatched=int(outside)))
        prepared.append((labels, peaks, scores, gate, period))
        total_labels += len(labels)
    columns = ['record_name', 'row', 'col', 'confidence', 'matched_label_index',
               'tp', 'fp', 'duplicate_fp', 'outside_gate_unmatched']
    ranked = pd.DataFrame(detections, columns=columns).sort_values(
        ['confidence', 'record_name', 'row', 'col'], ascending=[False, True, True, True], kind='stable').reset_index(drop=True)
    ranked['rank'] = np.arange(1, len(ranked)+1)
    for key in ('tp', 'fp'):
        ranked['cumulative_'+key] = ranked[key].cumsum()
    ranked['precision'] = ranked.cumulative_tp/np.maximum(ranked['rank'], 1)
    ranked['recall'] = ranked.cumulative_tp/total_labels if total_labels else np.nan
    # Equal scores form one attainable confidence cutoff, not an artificial order.
    pr = ranked.loc[~ranked.confidence.duplicated(keep='last'),
                    ['confidence', 'cumulative_tp', 'cumulative_fp', 'precision', 'recall']].copy()
    if total_labels:
        recall = np.r_[0., pr.recall.to_numpy(float), 1.]
        precision = np.r_[0., pr.precision.to_numpy(float), 0.]
        envelope = np.maximum.accumulate(precision[::-1])[::-1]
        ap = float(np.sum(np.diff(recall)*envelope[1:]))
    else:
        ap = np.nan
    operating = []
    for threshold in sorted(set(thresholds.tolist())):
        active = ranked.loc[ranked.confidence > threshold]
        tp, fp = int(active.tp.sum()), int(active.fp.sum())
        fn = total_labels-tp
        # Kept separate: overlapping GT gates can share a preservation peak.
        any_count = 0
        for labels, peaks, scores, gate, period in prepared:
            candidates = peaks[scores > threshold]
            if len(candidates):
                delta = np.abs(labels[:, None]-candidates[None, :])
                if period is not None:
                    delta[:, :, 1] = np.minimum(delta[:, :, 1], period-delta[:, :, 1])
                any_count += int(np.any(np.all(delta <= gate, axis=2), axis=1).sum())
        operating.append(dict(threshold_factor=threshold, annotation_labels=total_labels,
            detections=len(active), tp=tp, fp=fp, fn=fn,
            precision=tp/(tp+fp) if tp+fp else 0., recall=tp/total_labels if total_labels else np.nan,
            f1=2*tp/(2*tp+fp+fn) if total_labels else np.nan,
            any_peak_annotation_support=any_count, duplicate_fp=int(active.duplicate_fp.sum()),
            outside_gate_unmatched=int(active.outside_gate_unmatched.sum())))
    return dict(average_precision=ap, annotation_labels=total_labels, cubes=len(frames),
                ranked=ranked, pr=pr, operating_points=pd.DataFrame(operating))


def collect_detection_frame(ex, post, labels, map_type, record_name, method='PPO'):
    """Reuse the actual publication detector; no detector/axis/gate replacement."""
    kind = str(map_type).upper()
    if kind not in ('RD', 'RA'):
        raise ValueError('map_type must be RD or RA.')
    data, ranges, second = ex._map_triplet_v12_6(post, kind)
    bins, source_indices, excluded = ex._evaluable_label_bins_v12(labels, ranges, second, kind)
    power = ex.as_linear_power(data, assume_db=True)
    detector = ex.precompute_nested_cfar_peak_masks_v11_2(power, [0.])[0.]
    points = np.argwhere(detector['peaks'] & detector['valid'])
    score = np.asarray(detector['cfar_score'])[tuple(points.T)]
    if not np.isfinite(score).all() or np.any(score <= 0):
        raise ValueError('Publication detector returned invalid ranked confidence.')
    gate = ex.CFAR_LABEL_GATE_RD_V8 if kind == 'RD' else ex.CFAR_LABEL_GATE_RA_V8
    period = cyclic_column_period(bins, data.shape[1])
    return dict(record_name=str(record_name), method=str(method), map_type=kind,
                label_bins=np.asarray(bins, dtype=int).reshape(-1, 2),
                cyclic_period=period,
                source_label_indices=list(map(int, source_indices)), excluded_out_of_axis=int(excluded),
                peak_points=points, scores=score, gate=tuple(gate))


def export_detection_reports(ex, frames, output_dir, split='validation', thresholds_by_map=None):
    """Add AP, genuine PR and F1 figures without replacing inherited reports.

    Call collect_detection_frame during the inherited postprocessing loop;
    this exporter never loads a cube or recomputes radar postprocessing.
    """
    import matplotlib.pyplot as plt
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    frames = list(frames)
    if not frames:
        raise ValueError('Detection report requires collected frames.')
    thresholds_by_map = thresholds_by_map or {
        'RD':getattr(ex, 'V12_7_RD_PRESERVATION_THRESHOLD_FACTORS', DEFAULT_THRESHOLDS),
        'RA':getattr(ex, 'V12_7_BASE_CFAR_THRESHOLD_FACTORS', DEFAULT_THRESHOLDS)}
    methods = list(dict.fromkeys(frame['method'] for frame in frames))
    summaries, ranks, curves, operating = [], [], [], []
    for kind in ('RD', 'RA'):
        reference = {f['record_name']:f for f in frames if f['map_type'] == kind and f['method'] == methods[0]}
        for method in methods:
            selected = [f for f in frames if f['map_type'] == kind and f['method'] == method]
            if {f['record_name'] for f in selected} != set(reference):
                raise ValueError('Detection methods must use the identical cube population.')
            for item in selected:
                ref = reference[item['record_name']]
                if (not np.array_equal(item['label_bins'], ref['label_bins'])
                        or tuple(item['gate']) != tuple(ref['gate'])
                        or item.get('cyclic_period') != ref.get('cyclic_period')):
                    raise ValueError('Annotation bins and gates must not change across compression methods.')
            result = ranked_object_metrics(selected, thresholds_by_map[kind])
            summaries.append(dict(map_type=kind, method=method, average_precision=result['average_precision'],
                annotation_labels=result['annotation_labels'], cubes=result['cubes'],
                excluded_out_of_axis=sum(f.get('excluded_out_of_axis', 0) for f in selected)))
            for target, key in ((ranks, 'ranked'), (curves, 'pr'), (operating, 'operating_points')):
                table = result[key].copy()
                table['method'], table['map_type'] = method, kind
                target.append(table)
    summary, ranked, pr, points = (pd.DataFrame(summaries), pd.concat(ranks, ignore_index=True),
                                   pd.concat(curves, ignore_index=True), pd.concat(operating, ignore_index=True))
    targets = {key:directory/f'{split}_{key}_v12_21.csv' for key in
               ('annotation_AP', 'ranked_detections', 'confidence_PR', 'annotation_F1')}
    figure_targets = [directory/f'{split}_{key}_v12_21.png' for key in
                      ('annotation_AP', 'confidence_PR', 'annotation_F1')]
    contract_path = directory/f'{split}_detection_metric_contract_v12_21.json'
    if any(path.exists() for path in [*targets.values(), *figure_targets, contract_path]):
        raise ValueError('New V12.21 reports must not overwrite existing artifacts.')
    for key, table in zip(targets, (summary, ranked, pr, points)):
        ex.safe_write_csv_v11(table, targets[key])
    colors = {method:getattr(ex,'V12_7_COLORS',{}).get(method,plt.get_cmap('tab10')(i % 10))
              for i,method in enumerate(methods)}
    for metric, path in zip(('AP', 'PR', 'F1'), figure_targets):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
        fig.subplots_adjust(left=.08, right=.98, top=.82, bottom=.25, wspace=.24)
        for ax, kind in zip(axes, ('RD', 'RA')):
            for i, method in enumerate(methods):
                if metric == 'AP':
                    value = float(summary.loc[(summary.map_type == kind) & (summary.method == method), 'average_precision'].iloc[0])
                    ax.bar(i, 100*value, color=colors[method], label=method)
                    ax.text(i, 100*value+1 if np.isfinite(value) else 1,
                            f'{100*value:.2f}%' if np.isfinite(value) else 'undefined', ha='center', fontsize=8)
                else:
                    data = (pr if metric == 'PR' else points)
                    data = data.loc[(data.map_type == kind) & (data.method == method)]
                    x, y = (data.recall, data.precision) if metric == 'PR' else (10*np.log10(np.maximum(data.threshold_factor, 1e-12)), data.f1)
                    ax.plot(x, y, label=method, color=colors[method], lw=1.4, drawstyle='steps-post' if metric == 'PR' else 'default')
            ax.set_title(kind+' — annotation-based object detection')
            ax.set_ylim(0, 105 if metric == 'AP' else 1.03)
            ax.grid(alpha=.2)
            if metric == 'AP':
                ax.set(xticks=range(len(methods)), xticklabels=methods, ylabel='All-points interpolated AP [%]')
                ax.tick_params(axis='x', rotation=15)
            else:
                ax.set(xlabel='Annotation recall' if metric == 'PR' else 'CFAR threshold [dB]',
                       ylabel='Precision' if metric == 'PR' else 'Annotation-based F1')
                if metric == 'PR':
                    ax.set_xlim(0, 1)
                ax.legend(fontsize=8, frameon=False)
        fig.suptitle({'AP':'Confidence-ranked AP; not original-relative preservation',
                      'PR':'Genuine confidence-ranked precision–recall; no forced recall endpoint',
                      'F1':'One-to-one object detection across the unchanged CFAR thresholds'}[metric], fontsize=13)
        fig.text(.08, .075, 'All in-axis annotation objects form the denominator. Highest-confidence in-gate detection matches once; duplicates count as annotation FP.\n'
                 'Any-peak original-relative preservation is a DIFFERENT, unchanged metric. Outside-gate peaks are not confirmed physical false alarms.\n'
                 'AP integrates the full positive-score fixed-NMS candidate pool, not just the plotted CFAR grid; GT completeness limits interpretation.', fontsize=8.5)
        ex.save_publication_figure(fig, path)
        plt.close(fig)
    contract = detection_metric_config()
    contract['plot_thresholds_by_map'] = {kind:list(map(float, values)) for kind,values in thresholds_by_map.items()}
    contract_path.write_text(json.dumps(contract, indent=2)+'\n', encoding='utf-8')
    return dict(summary=summary, ranked=ranked, pr=pr, operating_points=points,
                figure_paths=figure_targets, table_paths=list(targets.values()), contract=contract)


def plot_percentage_rate_reports(ex, sweep_summary, directory):
    """Measured inherited operating points only; no cap, pruning or interpolation."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    frame = sweep_summary.copy()
    required = {'map_type','condition','policy_id','family','accounted_bpcs',
                'supported_original_instances','preserved_instances','unmatched_per_cube_threshold'}
    if required-set(frame):
        raise ValueError('Missing inherited rate-sweep columns: '+str(sorted(required-set(frame))))
    frame['bits_saved_percent'] = bits_saved_percent(frame.accounted_bpcs)
    if (frame.preserved_instances < 0).any() or (frame.preserved_instances > frame.supported_original_instances).any():
        raise ValueError('Original-supported pooled preservation counts are inconsistent.')
    frame['preservation_percent'] = np.divide(100.*frame.preserved_instances.to_numpy(float),
        frame.supported_original_instances.to_numpy(float), out=np.full(len(frame), np.nan), where=frame.supported_original_instances.to_numpy() > 0)
    for kind in ('RD','RA'):
        for condition in ('lowest','six_average'):
            original = frame.loc[(frame.map_type == kind) & (frame.condition == condition) & (frame.family == 'Original')]
            if len(original) != 1 or not np.isclose(float(original.accounted_bpcs.iloc[0]),32.):
                raise ValueError('Require one actual 32-bpc uncompressed reference per map/condition.')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory/f'{metric}_vs_bits_saved_v12_21.png' for metric in ('label_preservation','unmatched_peaks')]
    if any(path.exists() for path in paths):
        raise ValueError('Use new V12.21 rate report filenames/directory.')
    colors = {'RD':'#0072B2','RA':'#D55E00'}
    families = list(dict.fromkeys(frame.family))
    markers = {family:('*' if family == 'Original' else ('o','s','^','D','v')[i % 5]) for i,family in enumerate(families)}
    for metric, ylabel, path in zip(('preservation_percent','unmatched_per_cube_threshold'),
            ('Original-supported object instances preserved [%]', 'Peaks outside all annotation gates / cube / threshold'), paths):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.subplots_adjust(left=.08, right=.985, top=.76, bottom=.24, wspace=.25)
        for ax, condition, title in zip(axes, ('lowest','six_average'), ('Lowest CFAR threshold', 'Six-threshold pooled summary')):
            subset = frame.loc[(frame.condition == condition) & frame.map_type.isin(colors)]
            for kind, color in colors.items():
                for family in families:
                    rows = subset.loc[(subset.map_type == kind) & (subset.family == family)]
                    highlight=family=='Trained_PPO'
                    ax.scatter(rows.bits_saved_percent, rows[metric], color=color,
                               marker=markers[family], s=130 if highlight else 100 if family == 'Original' else 35,
                               edgecolors='black' if highlight else color,
                               linewidths=1.1 if highlight else .4,alpha=1. if highlight else .8,zorder=6 if highlight else 3)
                    if highlight:
                        for point in rows.itertuples():
                            unit='%' if metric=='preservation_percent' else ''
                            ax.text(.03,.16 if kind=='RD' else .08,
                                f'Trained PPO {kind}: {getattr(point,metric):.1f}{unit}',
                                transform=ax.transAxes,color=color,fontsize=9,
                                bbox=dict(facecolor='white',edgecolor='none',alpha=.9))
            ax.set(title=title, xlabel='Bits saved versus 32-bpc original [%]  → stronger compression', ylabel=ylabel,
                   xlim=(min(-1., float(frame.bits_saved_percent.min())-1.),100.))
            ax.axvline(0, color='.5', lw=.8)
            ax.grid(alpha=.2)
            ax.set_ylim(0, 103 if metric == 'preservation_percent' else None)
            secondary = ax.secondary_xaxis('top', functions=(lambda x:32*(1-x/100), lambda r:100*(1-r/32)))
            secondary.set_xlabel('Remaining accounted bits / complex sample', fontsize=9)
            secondary.set_xticks([32,24,16,8,0])
        handles = [Line2D([],[],color=color,label=kind) for kind,color in colors.items()]
        names={'Original':'Original, no compression','FFT_full_support':'Full-support FFT quantization',
               'Uniform_library':'Uniform codec actions','Trained_PPO':'Trained PPO (one operating point)'}
        handles += [Line2D([],[],color='.2',marker=markers[family],ls='None',label=names.get(family,family)) for family in families]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5,.94), ncol=3, frameon=False, fontsize=8)
        fig.suptitle('Rate–perception operating points: 0% = uncompressed; no bitrate cap', fontsize=13)
        fig.text(.08,.06,'Markers are measured operating points, not an interpolated optimum or an action-selection gate. No unmeasured 100%-saving point is added.\n'
                 'Six-threshold preservation pools preserved/original-supported object–threshold instances. Negative savings mean encoded rate exceeds the source.\n'
                 'Unmatched peaks are annotation-relative, not confirmed physical false alarms; rates include accounted syntax, not measured wireless payloads.', fontsize=8.5)
        ex.save_publication_figure(fig, path)
        plt.close(fig)
    return paths
