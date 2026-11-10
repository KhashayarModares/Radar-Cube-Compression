"""Bounded original-relative perception objective; no rate or NMSE incentive.

One extra integer lowest-threshold miss costs 4. All other costs together are
bounded above by 2.2 for ONE allocation. This is not a guarantee about expected
policy risk, finite-training convergence or unseen objects.
"""
from __future__ import annotations

import math

from radar_sim.ppo_v12_19_reward import balanced_curve_regret, LOSS_COMPONENT_NAMES


def _number(detail, key):
    value = float(detail[key])
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{key} must be measured, finite and nonnegative.')
    return value


def perception_terms(components, detail, bpcs):
    low = _number(detail, 'low_threshold_labels_missed_total')
    missed = _number(detail, 'true_labels_missed_total')
    supported = _number(detail, 'true_labels_supported_total')
    low_supported = _number(detail, 'low_threshold_labels_supported_total')
    if low != int(low) or missed != int(missed) or low > low_supported or missed > supported:
        raise ValueError('Object/map miss counts must be valid integers with matching support.')
    nmse = _number(detail, 'cube_nmse')  # required even though not rewarded
    if not math.isfinite(float(bpcs)) or float(bpcs) <= 0:
        raise ValueError('Accounted bitrate must be finite and positive.')
    unmatched = _number(detail, 'new_false_detections_total')
    original = _number(detail, 'original_unmatched_peaks_total')
    low_u = _number(detail, 'low_unmatched_peaks_method_total')
    low_u0 = _number(detail, 'low_unmatched_peaks_original_total')
    margin = _number(detail, 'low_cfar_margin_deficit')
    curve = float(balanced_curve_regret(detail))
    sweep_fraction = missed / supported if supported else 0.
    unmatched_bounded = .5 * unmatched / (unmatched + original + 1.)
    unmatched_bounded += .5 * low_u / (low_u + low_u0 + 1.)
    costs = dict(cost_low_miss=4. * low, cost_sweep_miss=sweep_fraction,
        cost_margin=.1 * margin / (1. + margin),
        cost_curve=.1 * curve / (1. + curve), cost_unmatched=unmatched_bounded,
        cost_map=0., cost_cube_nmse=0., cost_rate=0.)
    loss = math.fsum(costs.values())
    if not 0 <= loss - costs['cost_low_miss'] < 2.2 + 1e-12:
        raise AssertionError('The declared per-allocation preservation priority was violated.')
    result = dict(**costs, total_loss=loss, total_reward=-loss,
        terminal_label_first_reward=-loss, episode_return=-loss,
        reward_task_only=-loss, reward_cube_distortion=0.,
        reward_components_sum=-math.fsum(costs[k] for k in LOSS_COMPONENT_NAMES),
        reward_components_error=0., reward_rate_saving=0.,
        reward_low_miss_penalty=-costs['cost_low_miss'],
        reward_sweep_gain=-costs['cost_sweep_miss'],
        reward_margin_penalty=-costs['cost_margin'],
        reward_cfar_curve_penalty=-costs['cost_curve'],
        reward_unmatched_penalty=-costs['cost_unmatched'],
        reward_map_penalty=0., reward_rate_penalty=0.,
        fixed_monitor_return=-loss, task_loss=loss, total_objective=loss,
        cube_nmse=nmse, rate_excess=0., rate_hard_violation=0.,
        rate_limit_bpcs=None, rate_budget_enforced_by_action_mask=False,
        low_label_gain=-low, aggregate_label_gain=-missed,
        low_threshold_labels_missed_candidate=low,
        aggregate_labels_missed_candidate=missed,
        false_peak_excess=max(unmatched-original, 0.),
        false_penalty=unmatched_bounded, constraint_low_cost=low,
        constraint_unmatched_cost=unmatched_bounded,
        low_cfar_margin_deficit=margin, cfar_curve_regret=curve,
        preservation_multiplier=1., lambda_low=4., lambda_unmatched=1.,
        map_error_excess=float(components['RD_log_power_mse'])+float(components['RA_log_power_mse']),
        map_penalty=0., secondary_penalty=loss-4.*low,
        safety_feasible=bool(low == 0),
        safety_failure_reason='LOWEST_THRESHOLD_MISS' if low else 'NO_LOWEST_THRESHOLD_MISS',
        training_uses_compressor_comparison=False,
        unmatched_objective_mode='bounded_original_scaled_raw_counts',
        perception_priority_per_allocation=True, expected_zero_miss_guarantee=False)
    return result


def checkpoint_key(frame):
    """Original-relative validation only; no reward/rate/NMSE/RDO selection."""
    if frame.empty:
        raise ValueError('A nonempty fixed validation population is required.')
    excess = (frame.new_false_detections_total-frame.original_unmatched_peaks_total).clip(lower=0)
    return (int(frame.low_threshold_labels_missed_total.sum()),
            float(excess.mean()),
            int(frame.true_labels_missed_total.sum()),
            float(frame.cfar_curve_regret.mean()))


class PerceptionObjective:
    def total(self, components, detail, bpcs):
        return perception_terms(components, detail, bpcs)

    def as_dict(self):
        return dict(version='12.21', name='bounded_perception_first',
            lowest_miss_weight=4., sweep_fraction_weight=1., unmatched_weight=1.,
            bounded_margin_weight=.1, bounded_continuous_regret_weight=.1,
            nmse_weight=0., map_weight=0., rate_weight=0., bitrate_cap=None,
            secondary_cost_upper_bound=2.2, comparator_used=False,
            checkpoint_order=['lowest_miss_sum', 'mean_positive_unmatched_excess', 'sweep_miss_sum', 'mean_curve_regret'],
            guarantee='Per-allocation priority only, not finite-training/unseen-data safety.')
