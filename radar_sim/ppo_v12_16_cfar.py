"""Exact log-threshold area of lost original label preservation.

This is an offline task-distortion measure derived from the project's existing
fixed-NMS, any-peak-in-GT-gate count protocol. It does not change the detector,
labels, gates, peak masks, strict CFAR comparison, or reported integer counts.
The integration identity below is our mathematical derivation, not a claim
that this particular loss was published or guarantees a PPO improvement.

For gate i let s0 and s1 be its best valid ORIGINAL and METHOD fixed-NMS peak
scores. The original supports the label at threshold a exactly when a < s0;
the method supports it when a < s1. Therefore its lost-preservation indicator
is 1{s1 <= a < s0}. Integrating on log(a) over [lo, hi] gives

  max(clip(log(s0), log(lo), log(hi))
      - clip(log(max(s1, lo)), log(lo), log(hi)), 0).

Division by log(hi)-log(lo) bounds each object's regret between zero and one.
The sum retains absolute-object semantics: a surviving in-gate peak preserves
an object, and one peak may support multiple overlapping gates as before.
Endpoints have measure zero, so the identity is consistent with strict >
threshold detection even at equality. No GT is needed by a deployed encoder.

The optional outside-GT quantity integrates the unchanged unmatched peak
count. Each valid fixed-NMS peak outside the UNION of ALL evaluable GT gates
contributes its log-threshold lifetime, clipped to this same interval. The
normalized sum is the mean unmatched count over a uniform continuous dB
sweep, not a probability and not a count of confirmed physical false alarms.
"""

from __future__ import annotations

import numpy as np

from radar_sim.ppo_v12_14_margin import (
    cyclic_column_period, label_gate_region, lowest_cfar_margin_loss_v12_14,
)


def cfar_preservation_regret_v12_16(
    original_score,
    method_score,
    original_base_peaks,
    method_base_peaks,
    valid_map,
    label_bins,
    gate,
    low_threshold=1.25,
    high_threshold=49152.0,
    margin_factor=1.25,
    include_unmatched_curve=True,
):
    """Return continuous curve regret using the unchanged detector convention.

    All maps/masks must come from the SAME CA-CFAR/fixed-NMS implementation as
    the reported counts. The validity mask encodes its complete-window rule.
    Gates clip at physical edges, except explicitly declared cyclic Doppler
    columns (``label_bins.period``). Invalid cells and large
    nonpeak scores cannot support a label. Only original-detectable labels at
    the lowest threshold are eligible; weaker labels contribute no regret.

    ``high_threshold`` is map-specific: 49152 for the existing extended RD
    preservation report and 128 for its ordinary RA report. Regret is an exact
    CONTINUOUS area, not a relabeling of the finite-grid report's missed counts.
    ``regret_db_sum`` is the lost dB-span summed over eligible objects; dividing
    by ``threshold_span_db`` gives ``normalized_regret_sum``.

    The existing lowest-margin function validates inputs once and supplies the
    same eligibility, method gate-best scores and strict lowest miss count.
    Its margin outputs are returned for the unchanged auxiliary low-CFAR term;
    the continuous regret itself is independent of ``margin_factor``.

    With ``include_unmatched_curve=True`` (default), the result additionally
    contains original/method mean outside-GT peak counts over this continuous
    log-threshold interval and their signed excess. Every evaluable GT gate is
    excluded, including gates without an original-detectable peak. Passing
    False omits those additional quantities without changing regret outputs.
    """
    try:
        high = float(high_threshold)
        low = float(low_threshold)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('CFAR integration bounds must be finite positive scalars.') from exc
    if not np.isfinite(high) or not np.isfinite(low) or low <= 0 or high <= low:
        raise ValueError('CFAR integration requires finite 0 < low_threshold < high_threshold.')
    log_low, log_high = float(np.log(low)), float(np.log(high))
    log_span = log_high - log_low
    if not np.isfinite(log_span) or log_span <= 0:
        raise ValueError('The CFAR integration interval must have positive finite log width.')
    if not isinstance(include_unmatched_curve, (bool, np.bool_)):
        raise ValueError('include_unmatched_curve must be a boolean.')

    # Reuse, rather than duplicate, the detector-compatible input validation,
    # eligibility and any-in-gate method support implemented in V12.14.
    margin = lowest_cfar_margin_loss_v12_14(
        original_score, method_score, original_base_peaks, method_base_peaks,
        valid_map, label_bins, gate, low, margin_factor)
    source = np.asarray(original_score, dtype=np.float64)
    source_peaks = np.asarray(original_base_peaks, dtype=bool) & np.asarray(valid_map, dtype=bool)
    eligible = margin['eligible_label_indices']
    period = cyclic_column_period(label_bins, source.shape[1])
    bins = np.asarray(label_bins, dtype=np.int64)
    row_radius, col_radius = np.asarray(gate, dtype=np.int64)
    original_best = np.empty(len(eligible), dtype=np.float64)
    for j, label_index in enumerate(eligible):
        row, col = bins[int(label_index)]
        region = label_gate_region(source.shape, row, col, row_radius, col_radius, period)
        values = source[region][source_peaks[region]]
        # Eligibility from the existing validator ensures a real valid peak.
        original_best[j] = float(values.max())
    method_best = np.asarray(margin['method_best_scores'], dtype=np.float64)
    clipped_original_log = np.clip(np.log(original_best), log_low, log_high)
    clipped_method_log = np.clip(np.log(np.maximum(method_best, low)), log_low, log_high)
    lost_log_span = np.clip(clipped_original_log-clipped_method_log, 0.0, log_span)
    per_label_regret = lost_log_span / log_span
    db_per_natural_log = 10.0 / np.log(10.0)
    per_label_regret_db = lost_log_span * db_per_natural_log
    result = {
        'num_eligible': int(len(eligible)),
        'eligible_label_indices': eligible.copy(),
        'original_best_scores': original_best,
        'method_best_scores': method_best.copy(),
        'per_label_normalized_regret': per_label_regret,
        'normalized_regret_sum': float(per_label_regret.sum()),
        'normalized_regret_mean': float(per_label_regret.mean()) if per_label_regret.size else 0.0,
        'per_label_regret_db': per_label_regret_db,
        'regret_db_sum': float(per_label_regret_db.sum()),
        'lost_log_threshold_area_sum': float(lost_log_span.sum()),
        'threshold_span_db': float(log_span * db_per_natural_log),
        'low_threshold': low,
        'high_threshold': high,
        'num_low_misses': int(margin['num_low_misses']),
        'low_margin_sum': float(margin['margin_sum']),
        'low_margin_mean': float(margin['margin_mean']),
        'metric_protocol': 'continuous_log_threshold_area_of_original_supported_label_loss',
        'reported_integer_counts_changed': False,
        'unmatched_curve_computed': bool(include_unmatched_curve),
    }
    if include_unmatched_curve:
        # The existing unmatched metric excludes any peak inside ANY label
        # gate, not merely a gate eligible for preservation at the low factor.
        gate_union = np.zeros_like(source_peaks, dtype=bool)
        for row, col in bins.reshape(-1, 2):
            region = label_gate_region(source.shape, row, col, row_radius, col_radius, period)
            gate_union[region] = True
        candidate = np.asarray(method_score, dtype=np.float64)
        candidate_peaks = np.asarray(method_base_peaks, dtype=bool) & np.asarray(valid_map, dtype=bool)

        def curve_mean(score, peaks):
            outside_scores = score[peaks & ~gate_union]
            clipped_log = np.clip(np.log(np.maximum(outside_scores, low)), log_low, log_high)
            lifetimes = np.clip(clipped_log-log_low, 0.0, log_span) / log_span
            return float(lifetimes.sum())

        original_unmatched_mean = curve_mean(source, source_peaks)
        method_unmatched_mean = curve_mean(candidate, candidate_peaks)
        result.update(
            original_unmatched_curve_mean=original_unmatched_mean,
            method_unmatched_curve_mean=method_unmatched_mean,
            unmatched_curve_signed_excess=method_unmatched_mean-original_unmatched_mean,
            unmatched_curve_semantics='mean valid fixed-NMS peak count outside every evaluable GT gate; not confirmed false alarms')
    return result
