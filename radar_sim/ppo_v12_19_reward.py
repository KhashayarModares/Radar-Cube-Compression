"""Pure original-relative V12.19 reward accounting, independent of radar I/O.

Only the preservation normalization and one common preservation multiplier differ
from the stationary V12.18 ``sum_positive`` objective. Positive ``cost_*`` fields
and their signed ``reward_*`` counterparts make independent bookkeeping possible.
The helper requires measured whole-cube NMSE; absent distortion is never zero.
No compressor comparator, labels, action masking, or deployment logic lives here.
"""

from __future__ import annotations

import math
from typing import Mapping


DEFAULT_COMMON_SPAN_DB = 10.0 * math.log10(128.0 / 1.25)
LOSS_COMPONENT_NAMES = (
    'cost_low_miss', 'cost_sweep_miss', 'cost_margin', 'cost_curve',
    'cost_unmatched', 'cost_map', 'cost_cube_nmse', 'cost_rate',
)


def _nonnegative(name, value):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be finite and nonnegative.') from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f'{name} must be finite and nonnegative.')
    return number


def balanced_curve_regret(original_detail: Mapping, *,
                          reference_span_db=DEFAULT_COMMON_SPAN_DB,
                          mode='balanced'):
    """Return original-supported object-dB divided by ONE fixed common span.

    Lost support is summed over objects and both maps, not divided by a map's
    plotting interval or by its annotation count. Changing only a plotting upper
    limit therefore cannot dilute an identical loss of object-dB. ``legacy``
    returns the recorded V12.18 sum of map-specific normalized curve regrets for
    the unmodified-objective control. Neither mode changes detector/count logic.
    """
    span = _nonnegative('reference_span_db', reference_span_db)
    if span <= 0.0:
        raise ValueError('reference_span_db must be positive.')
    if mode == 'legacy':
        return _nonnegative('cfar_curve_regret', original_detail['cfar_curve_regret'])
    if mode != 'balanced':
        raise ValueError('curve mode must be balanced or legacy.')
    lost = sum(_nonnegative(f'{map_type}_cfar_lost_support_db_sum',
                           original_detail[f'{map_type}_cfar_lost_support_db_sum'])
               for map_type in ('RD', 'RA'))
    return lost / span


common_balanced_regret = balanced_curve_regret


def preservation_weighted_reward_terms(
        components: Mapping, detail: Mapping, bpcs, *, preservation_multiplier,
        curve_mode='balanced', common_span_db=DEFAULT_COMMON_SPAN_DB,
        rate_cap=4.8, low_weight=5000., unmatched_weight=2000.,
        sweep_weight=500., map_weight=500., margin_weight=2500.,
        curve_weight=2500., nmse_weight=5000.,
        false_allowance_per_threshold=2., training_threshold_count=6):
    """Reproduce V12.18 sum-positive costs, changing preservation terms only.

    The multiplier scales four terms: lowest-threshold misses, six-threshold
    misses, low-threshold margin deficit, and continuous preservation regret.
    Unmatched, map, NMSE and over-cap costs do not scale. There is no rate-saving
    bonus and no dependence on the RDO/FFT comparator. V12.18 compatibility fields
    are returned alongside an explicit additive loss decomposition.
    """
    k = _nonnegative('preservation_multiplier', preservation_multiplier)
    if k <= 0.0:
        raise ValueError('preservation_multiplier must be positive.')
    weights = {name: _nonnegative(name, value) for name, value in (
        ('low_weight', low_weight), ('unmatched_weight', unmatched_weight),
        ('sweep_weight', sweep_weight), ('map_weight', map_weight),
        ('margin_weight', margin_weight), ('curve_weight', curve_weight),
        ('nmse_weight', nmse_weight),
    )}
    cap = _nonnegative('rate_cap', rate_cap)
    if cap <= 0.0:
        raise ValueError('rate_cap must be positive.')
    rate = _nonnegative('bpcs', bpcs)
    allowed = _nonnegative('false_allowance_per_threshold', false_allowance_per_threshold)
    n_thresholds = _nonnegative('training_threshold_count',
        detail.get('training_cfar_threshold_count', training_threshold_count))
    if int(n_thresholds) != n_thresholds or n_thresholds < 1:
        raise ValueError('training_cfar_threshold_count must be a positive integer.')
    n_thresholds = int(n_thresholds)
    low = _nonnegative('low_threshold_labels_missed_total', detail['low_threshold_labels_missed_total'])
    miss = _nonnegative('true_labels_missed_total', detail['true_labels_missed_total'])
    unmatched = _nonnegative('new_false_detections_total', detail['new_false_detections_total'])
    original = _nonnegative('original_unmatched_peaks_total', detail['original_unmatched_peaks_total'])
    low_unmatched = _nonnegative('low_unmatched_peaks_method_total',
        detail.get('low_unmatched_peaks_method_total', unmatched))
    low_original = _nonnegative('low_unmatched_peaks_original_total',
        detail.get('low_unmatched_peaks_original_total', original))
    supported = _nonnegative('low_threshold_labels_supported_total',
        detail['low_threshold_labels_supported_total'])
    rd_distortion = _nonnegative('RD_log_power_mse', components['RD_log_power_mse'])
    ra_distortion = _nonnegative('RA_log_power_mse', components['RA_log_power_mse'])
    # Deliberately no .get(..., 0): a missing measurement caused the V12.18
    # comparator/reporting inconsistency that this version must detect early.
    nmse = _nonnegative('cube_nmse', detail['cube_nmse'])
    margin = _nonnegative('low_cfar_margin_deficit', detail.get('low_cfar_margin_deficit', 0.))
    legacy_regret = _nonnegative('cfar_curve_regret', detail['cfar_curve_regret'])
    regret = balanced_curve_regret(detail, reference_span_db=common_span_db, mode=curve_mode)
    curve_original = _nonnegative('unmatched_curve_original_total',
        detail.get('unmatched_curve_original_total', 0.))
    curve_method = _nonnegative('unmatched_curve_method_total',
        detail.get('unmatched_curve_method_total', 0.))
    allowance = allowed * n_thresholds
    sweep_cost = (unmatched - original - allowance) / max(original + allowance, 1.)
    low_allowance = 2. * allowed
    low_cost = (low_unmatched - low_original - low_allowance) / max(low_original + low_allowance, 1.)
    curve_costs = []
    for map_type in ('RD', 'RA'):
        method_value = _nonnegative(map_type + '_unmatched_curve_method',
            detail.get(map_type + '_unmatched_curve_method', curve_method / 2.))
        original_value = _nonnegative(map_type + '_unmatched_curve_original',
            detail.get(map_type + '_unmatched_curve_original', curve_original / 2.))
        curve_costs.append((method_value-original_value-allowed) / max(original_value+allowed, 1.))
    unmatched_cost = (max(sweep_cost, 0.) + max(low_cost, 0.)
                      + sum(max(value, 0.) for value in curve_costs))
    rate_excess = max(rate-cap, 0.)
    map_distortion = rd_distortion + ra_distortion
    result = dict(
        cost_low_miss=k*weights['low_weight']*low,
        cost_sweep_miss=k*weights['sweep_weight']*miss,
        cost_margin=k*weights['margin_weight']*margin,
        cost_curve=k*weights['curve_weight']*regret,
        cost_unmatched=weights['unmatched_weight']*unmatched_cost,
        cost_map=weights['map_weight']*map_distortion,
        cost_cube_nmse=weights['nmse_weight']*nmse,
        cost_rate=10000.*rate_excess,
    )
    if any(not math.isfinite(value) for value in result.values()):
        raise ValueError('Reward cost overflow: reduce invalidly large weights or measurements.')
    total_loss = math.fsum(result[name] for name in LOSS_COMPONENT_NAMES)
    task_loss = math.fsum(result[name] for name in LOSS_COMPONENT_NAMES if name != 'cost_cube_nmse')
    feasible = low == 0 and max(sweep_cost, low_cost, *curve_costs) <= 0 and rate_excess <= 1e-6
    result.update(
        total_loss=total_loss, total_reward=-total_loss,
        terminal_label_first_reward=-total_loss, reward_task_only=-task_loss,
        reward_cube_distortion=-result['cost_cube_nmse'], fixed_monitor_return=-total_loss,
        reward_low_miss_penalty=-result['cost_low_miss'],
        reward_sweep_gain=-result['cost_sweep_miss'],
        reward_unmatched_penalty=-result['cost_unmatched'], reward_rate_saving=0.,
        reward_margin_penalty=-result['cost_margin'],
        reward_cfar_curve_penalty=-result['cost_curve'],
        reward_map_penalty=-result['cost_map'], reward_rate_penalty=-result['cost_rate'],
        preservation_multiplier=k, preservation_curve_mode=curve_mode,
        preservation_common_span_db=float(common_span_db),
        cfar_curve_regret=regret, legacy_cfar_curve_regret=legacy_regret,
        balanced_cfar_curve_regret=(balanced_curve_regret(detail, reference_span_db=common_span_db)
            if all(f'{mt}_cfar_lost_support_db_sum' in detail for mt in ('RD', 'RA')) else float('nan')),
        low_cfar_margin_deficit=margin,
        constraint_low_cost=low, constraint_unmatched_cost=unmatched_cost,
        unmatched_curve_original_total=curve_original, unmatched_curve_method_total=curve_method,
        constraint_unmatched_curve_cost=max(curve_costs),
        RD_constraint_unmatched_curve_cost=curve_costs[0],
        RA_constraint_unmatched_curve_cost=curve_costs[1],
        constraint_unmatched_sweep_cost=sweep_cost, constraint_unmatched_low_cost=low_cost,
        lambda_low=k*weights['low_weight'], lambda_unmatched=weights['unmatched_weight'],
        cube_nmse=nmse, safety_feasible=feasible,
        safety_failure_reason='FEASIBLE' if feasible else 'ORIGINAL_RELATIVE_VIOLATION',
        low_threshold_labels_missed_candidate=low, low_threshold_labels_missed_reference=0.,
        low_threshold_labels_supported_candidate_basis=supported,
        low_label_gain=-low, aggregate_labels_missed_candidate=miss,
        aggregate_labels_missed_reference=0., aggregate_label_gain=-miss,
        aggregate_label_regression_count=miss, method_unmatched_false_peaks=unmatched,
        original_unmatched_false_peaks=original, false_peak_excess=max(unmatched-original, 0.),
        false_peak_allowance=allowance, false_peak_hard_violation=max(unmatched-original-allowance, 0.),
        false_peak_violation_count=max(unmatched-original-allowance, 0.),
        false_penalty=result['cost_unmatched'], map_error_excess=map_distortion,
        map_penalty=result['cost_map'], map_error_is_direct_original_distortion=True,
        rate_limit_bpcs=cap, rate_hard_violation=rate_excess,
        rate_saving_bpcs=max(cap-rate, 0.), rate_penalty_for_reward=result['cost_rate'],
        secondary_penalty=math.fsum(result[name] for name in
            ('cost_unmatched', 'cost_map', 'cost_margin', 'cost_curve', 'cost_rate')),
        alignment_reward_shaping=0., alignment_span_gain_db=0., weighted_label_deficit_gain=0.,
        training_uses_compressor_comparison=False, unmatched_objective_mode='sum_positive',
        metric_protocol='unchanged absolute object counts; direct original-map distortion',
    )
    return result
