"""Bounded, validation-only detector diagnostics; never a deployment repair.

The caller supplies the active pipeline so this module neither imports it nor
runs its initialization/preflights. All scores, fixed NMS peaks and GT bins
come from that pipeline's unchanged detector. Single-tile interventions keep
every other action fixed and never select an allocation for deployment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from radar_sim.ppo_v12_14_margin import cyclic_column_period, label_gate_region


def _noise_at(detector, row, col, epsilon):
    score = float(detector['cfar_score'][row, col])
    power = float(detector['power'][row, col])
    # The existing detector computes score = power / (noise + EPS).
    return max(power / score - float(epsilon), 0.0) if score > 0 else np.nan


def _best_peak(detector, region, epsilon):
    peaks = np.asarray(detector['peaks'], bool) & np.asarray(detector['valid'], bool)
    points = np.argwhere(peaks[region])
    if not len(points):
        return dict(score=0.0, power=np.nan, noise=np.nan, row=-1, col=-1)
    points[:, 0] += region[0].start
    if isinstance(region[1], slice):
        points[:, 1] += region[1].start
    else:
        points[:, 1] = np.asarray(region[1])[points[:, 1]]
    values = detector['cfar_score'][points[:, 0], points[:, 1]]
    row, col = map(int, points[int(np.argmax(values))])
    return dict(score=float(detector['cfar_score'][row, col]),
                power=float(detector['power'][row, col]),
                noise=_noise_at(detector, row, col, epsilon), row=row, col=col)


def gate_peak_diagnostics(original, method, bins, gate, low_threshold,
                          high_threshold, epsilon=1e-12):
    """Per-object diagnostic measurements with the existing any-in-gate rule.

    Base peaks must be produced at threshold 0 by the existing fixed-NMS
    detector. Original-supported objects are included even when preserved.
    Power/noise changes at the ORIGINAL peak use identical map coordinates;
    method gate-best measurements may refer to a different physical peak.
    Missing peaks have score 0, position -1, and undefined power/noise (NaN).
    """
    low, high = float(low_threshold), float(high_threshold)
    if not (0 < low < high):
        raise ValueError('Diagnostic CFAR bounds must satisfy 0 < low < high.')
    shape = np.asarray(original['cfar_score']).shape
    if len(shape) != 2 or shape != np.asarray(method['cfar_score']).shape:
        raise ValueError('Original/method diagnostic maps must have identical 2-D shape.')
    if not np.array_equal(original['valid'], method['valid']):
        raise ValueError('Original/method complete-window validity differs.')
    period = cyclic_column_period(bins, shape[1])
    radius_r, radius_c = map(int, gate)
    if min(radius_r, radius_c) < 0:
        raise ValueError('Gate radii must be nonnegative.')
    result = []
    for label_index, (row, col) in enumerate(bins):
        row, col = int(row), int(col)
        if not (0 <= row < shape[0] and 0 <= col < shape[1]):
            raise ValueError('Diagnostic label bin lies outside map.')
        region = label_gate_region(shape, row, col, radius_r, radius_c, period)
        first, second = (_best_peak(d, region, epsilon) for d in (original, method))
        if first['score'] <= low:
            continue
        r0, c0 = first['row'], first['col']
        candidate_power = float(method['power'][r0, c0])
        candidate_noise = _noise_at(method, r0, c0, epsilon)
        power_delta = 10*np.log10(max(candidate_power, epsilon) / max(first['power'], epsilon))
        # +EPS denominators match the actual detector, including noise-floor cases.
        noise_delta = 10*np.log10((candidate_noise+epsilon) / (first['noise']+epsilon))
        lost_db = max(10*np.log10(np.clip(first['score'], low, high)) -
                      10*np.log10(np.clip(second['score'], low, high)), 0.0)
        result.append(dict(label_index=label_index, label_row=row, label_col=col,
            **{'original_'+key: value for key, value in first.items()},
            **{'method_'+key: value for key, value in second.items()},
            low_threshold=low, high_threshold=high,
            original_detectable=True, missed_low=bool(second['score'] <= low),
            lost_original_support_db=float(lost_db),
            method_power_at_original_peak=candidate_power,
            method_noise_at_original_peak=candidate_noise,
            method_score_at_original_peak=float(method['cfar_score'][r0, c0]),
            method_original_peak_is_fixed_nms=bool(method['peaks'][r0, c0]),
            power_change_db_at_original_peak=float(power_delta),
            noise_change_db_at_original_peak=float(noise_delta),
            score_change_db_at_original_peak=float(power_delta-noise_delta),
            gate_best_peak_changed=bool(first['row'] != second['row'] or first['col'] != second['col']),
            diagnosis='no_method_fixed_nms_peak_in_gate' if second['row'] < 0 else
                      'method_peak_below_low_threshold' if second['score'] <= low else
                      'higher_threshold_support_loss' if lost_db > 1e-10 else 'preserved_support',
            counting_rule='one original-supported object; any in-gate fixed-NMS peak'))
    return pd.DataFrame(result)


def tile_error_decomposition(codec, prepared, action, cube_energy):
    """Exact disjoint spectral SSE, not a causal decomposition of CFAR errors."""
    rho, q = action
    decoded, _ = codec.materialize_action(prepared, rho, q)
    support = codec.support_mask(prepared, rho)
    source = np.asarray(prepared['Z'])
    removed = float(np.sum(np.abs(source[:, ~support])**2, dtype=np.float64))
    quantized = float(np.sum(np.abs(source[:, support]-decoded[:, support])**2, dtype=np.float64))
    total = float(np.sum(np.abs(source-decoded)**2, dtype=np.float64))
    if not np.isclose(removed+quantized, total, rtol=1e-10, atol=1e-10):
        raise AssertionError('Removed and retained-quantization SSE must sum to tile SSE.')
    denominator = max(float(cube_energy), 1e-12)
    return dict(removed_coefficient_sse=removed, retained_quantization_sse=quantized,
        total_spectral_sse=total, removed_nmse_contribution=removed/denominator,
        quantization_nmse_contribution=quantized/denominator,
        total_nmse_contribution=total/denominator,
        retained_cells_all_receivers_quantized_to_zero=int(np.sum(np.all(decoded[:, support] == 0, axis=0))),
        decomposition_scope='exact disjoint spectral SSE; not detector-error attribution')


def intervention_action_indices(actions, base_index, maximum=2):
    """Predeclared diagnostic contrasts, not task-based action preselection.

    Prefer a one-coordinate precision increase and a one-coordinate support
    increase. Where unavailable, use nearest declared alternatives and report
    that both coordinates changed. No new codec action is introduced.
    """
    rho, q = actions[int(base_index)]
    candidates = [i for i in range(len(actions)) if i != int(base_index)]
    precision = sorted((i for i in candidates if actions[i][1] > q),
                       key=lambda i: (abs(actions[i][0]-rho), actions[i][1]-q, i))
    retention = sorted((i for i in candidates if actions[i][0] > rho),
                       key=lambda i: (abs(actions[i][1]-q), actions[i][0]-rho, i))
    selected = []
    for sequence in (precision, retention):
        for index in sequence:
            if index not in selected:
                selected.append(index)
                break
    return selected[:int(maximum)]


def _label_rows(ex, post0, post1, labels_cube, record_name, scene_name):
    records = []
    low = float(min(ex._nearest_training_thresholds_v11()))
    for kind, gate, thresholds in (
        ('RD', ex.CFAR_LABEL_GATE_RD_V8, ex.V12_7_RD_PRESERVATION_THRESHOLD_FACTORS),
        ('RA', ex.CFAR_LABEL_GATE_RA_V8, ex.V12_7_BASE_CFAR_THRESHOLD_FACTORS)):
        original_map, ranges, second_axis = ex._map_triplet_v12_6(post0, kind)
        method_map, ranges1, second1 = ex._map_triplet_v12_6(post1, kind)
        if not (np.array_equal(ranges, ranges1) and np.array_equal(second_axis, second1)):
            raise AssertionError('Diagnostic original/decoded physical axes disagree.')
        bins, source_indices, _ = ex._evaluable_label_bins_v12(labels_cube, ranges, second_axis, kind)
        original = ex.precompute_nested_cfar_peak_masks_v11_2(ex.as_linear_power(original_map), [0.0])[0.0]
        method = ex.precompute_nested_cfar_peak_masks_v11_2(ex.as_linear_power(method_map), [0.0])[0.0]
        frame = gate_peak_diagnostics(original, method, bins, gate, low, max(thresholds), ex.EPS)
        for item in frame.to_dict('records'):
            source_index = source_indices[int(item['label_index'])]
            label = labels_cube.iloc[source_index]
            item.update(record_name=record_name, scene_name=scene_name, map_type=kind,
                        object_id=str(label.get('object_id', label.get('actor_id', source_index))),
                        source_label_index=int(source_index), range_m=float(label.get('range_m', np.nan)),
                        radial_velocity_mps=float(label.get('radial_velocity_mps', np.nan)),
                        doppler_bin=float(label.get('doppler_bin', np.nan)),
                        azimuth_deg=float(label.get('azimuth_deg', np.nan)))
            for prefix in ('original', 'method'):
                rr, cc = int(item[prefix+'_row']), int(item[prefix+'_col'])
                item[prefix+'_peak_range_m'] = float(ranges[rr]) if rr >= 0 else np.nan
                item[prefix+'_peak_second_axis'] = float(second_axis[cc]) if cc >= 0 else np.nan
            records.append(item)
    return pd.DataFrame(records)


def _suspect_tiles(source, post_original, label_frame, decompositions, maximum):
    selected = []
    if len(label_frame):
        failed = label_frame.loc[label_frame.lost_original_support_db > 1e-10].sort_values(
            ['missed_low', 'lost_original_support_db'], ascending=False)
        for label in failed.to_dict('records'):
            # Windowing/clutter filtering spreads spectral influence. A tile
            # containing this GT bin is a diagnostic candidate, not a proof
            # that it alone caused the missed detector peak.
            if not (np.isfinite(label['range_m']) and np.isfinite(label['radial_velocity_mps'])):
                continue
            rr = int(np.argmin(abs(np.asarray(post_original['ranges'])-label['range_m'])))
            vv = int(np.argmin(abs(np.asarray(post_original['vel'])-label['radial_velocity_mps'])))
            for index, slc in enumerate(source['tile_slices']):
                if slc[1].start <= vv < slc[1].stop and slc[2].start <= rr < slc[2].stop:
                    if index not in [value[0] for value in selected]:
                        selected.append((index, 'tile_contains_regressed_GT_RD_bin; windowed detector may couple tiles'))
                    break
            if len(selected) >= maximum:
                return selected
    for index in sorted(range(len(decompositions)), key=lambda i: decompositions[i]['total_spectral_sse'], reverse=True):
        if index not in [value[0] for value in selected]:
            selected.append((index, 'largest_remaining_spectral_SSE; diagnostic control'))
        if len(selected) >= maximum:
            break
    return selected


def diagnose_allocations(ex, rows, labels, policy, normalizer, output_dir,
                         max_cubes=2, max_tiles=2, split_name='validation'):
    """Replay at most 2 cubes x 2 tiles x 2 alternatives, with no policy repair.

    Pass ONLY fixed validation rows, never held-out test rows. The caller owns
    split provenance. Base allocations are deterministic greedy PPO. Infeasible
    substitutions are logged and skipped without modifying any other action.
    Returns/writes label, tile-error, intervention and intervention-label CSVs.
    """
    if split_name != 'validation':
        raise ValueError('V12.19 diagnostics are validation-only, not test tuning.')
    if not (1 <= int(max_cubes) <= 2 and 1 <= int(max_tiles) <= 2):
        raise ValueError('Bounded diagnostics allow 1-2 cubes and 1-2 tiles.')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processors, label_frames, error_records, interventions, intervention_labels = {}, [], [], [], []
    codec = ex.coherent_codec_v14
    library = tuple(ex.ACTION_LIBRARY_V11)
    for _, row in rows.head(int(max_cubes)).iterrows():
        cube = ex.load_cube_v6_1000(row)
        processor = ex._postprocessor_for_cube_v11(cube, processors)
        labels_cube = ex.labels_for_cube_v6_1000(labels, row)
        source = ex.prepare_policy_source_v12_12(cube)
        baseline = ex.encode_policy_v11(cube, policy, normalizer, False, source_context=source)
        sequence = baseline['actions'].copy()
        post0 = ex.postprocess_cube(cube, processor)
        post1 = ex.postprocess_cube(baseline['cube_hat'], processor)
        record_name, scene_name = str(row['record_name']), str(row['scene_name'])
        label_frame = _label_rows(ex, post0, post1, labels_cube, record_name, scene_name)
        label_frames.append(label_frame)
        _, base_detail = ex.radar_task_components_v11(post0, post1, labels_cube)
        decompositions = []
        for tile_index, (prepared, action_index) in enumerate(zip(source['prepared_blocks'], sequence)):
            detail = tile_error_decomposition(codec, prepared, library[int(action_index)], source['whole_cube_energy'])
            decompositions.append(detail)
            error_records.append(dict(record_name=record_name, scene_name=scene_name, tile_index=tile_index,
                action_index=int(action_index), rho=library[int(action_index)][0], q=library[int(action_index)][1], **detail))
        for tile_index, reason in _suspect_tiles(source, post0, label_frame, decompositions, int(max_tiles)):
            old_index = int(sequence[tile_index])
            for new_index in intervention_action_indices(library, old_index):
                delta_bits = float(source['action_bits_by_block'][tile_index, new_index]-
                                   source['action_bits_by_block'][tile_index, old_index])
                total_bits = float(baseline['meta']['realized_bits'])+delta_bits
                cap = float(ex.RL_RATE_TARGET_BPCS_V11)*cube.size
                feasible = total_bits <= cap+1e-6
                rho0, q0 = library[old_index]
                rho1, q1 = library[new_index]
                identifier = f'{record_name}:tile{tile_index}:action{new_index}'
                record = dict(intervention_id=identifier, record_name=record_name, scene_name=scene_name,
                    split=split_name, tile_index=tile_index, selection_reason=reason,
                    old_action_index=old_index, new_action_index=int(new_index),
                    old_rho=rho0, new_rho=rho1, old_q=q0, new_q=q1,
                    both_coordinates_changed=bool(rho0 != rho1 and q0 != q1),
                    bit_delta=delta_bits, baseline_bpcs=float(baseline['meta']['realized_bpcs']),
                    baseline_cube_nmse=float(baseline['meta']['cube_nmse']),
                    candidate_bpcs=total_bits/cube.size, budget_bpcs=float(ex.RL_RATE_TARGET_BPCS_V11),
                    cap_feasible=bool(feasible), deployment_refinement=False,
                    all_other_actions_fixed=True, actual_codec_bitstream_saved=False)
                if feasible:
                    changed = sequence.copy()
                    changed[tile_index] = new_index
                    if np.count_nonzero(changed != sequence) != 1:
                        raise AssertionError('A diagnostic intervention must change exactly one tile.')
                    replay = codec.encode_sequence(cube, changed, source=source)
                    if not np.isclose(replay['meta']['realized_bits'], total_bits, rtol=0, atol=1e-6):
                        raise AssertionError('Replay analytical bits differ from the fixed-allocation delta.')
                    post = ex.postprocess_cube(replay['cube_hat'], processor)
                    _, detail = ex.radar_task_components_v11(post0, post, labels_cube)
                    for metric in ('low_threshold_labels_missed_total', 'true_labels_missed_total',
                                   'new_false_detections_total', 'cfar_curve_regret'):
                        record['baseline_'+metric] = float(base_detail[metric])
                        record['candidate_'+metric] = float(detail[metric])
                        record['delta_'+metric] = float(detail[metric]-base_detail[metric])
                    record['candidate_cube_nmse'] = float(replay['meta']['cube_nmse'])
                    for key, value in tile_error_decomposition(codec, source['prepared_blocks'][tile_index],
                                                              library[new_index], source['whole_cube_energy']).items():
                        record['candidate_tile_'+key] = value
                    altered_labels = _label_rows(ex, post0, post, labels_cube, record_name, scene_name)
                    altered_labels['intervention_id'] = identifier
                    intervention_labels.append(altered_labels)
                interventions.append(record)
    frames = dict(
        labels=pd.concat(label_frames, ignore_index=True) if label_frames else pd.DataFrame(),
        tile_errors=pd.DataFrame(error_records), interventions=pd.DataFrame(interventions),
        intervention_labels=pd.concat(intervention_labels, ignore_index=True) if intervention_labels else pd.DataFrame())
    for name, frame in frames.items():
        ex.safe_write_csv_v11(frame, output_dir/f'ppo_validation_{name}_diagnostics_v12_19.csv')
    return frames
