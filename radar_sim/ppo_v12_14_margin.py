"""Offline, same-detector lowest-CFAR robustness surrogate for V12.14.

The publication metric is still a count: an ORIGINAL-detectable label is
preserved if any METHOD fixed-NMS peak in its existing GT gate has CA-CFAR
score strictly greater than the threshold. This auxiliary loss distinguishes
allocations on that count's plateaus; it does not replace or redefine counts.

For an eligible gate i, t_i = min(s_i(original), factor * threshold) and
L_i = max(t_i - s_i(method), 0) / threshold. The capped original score avoids
requiring a weak original peak to achieve a margin it never had. The factor
is an explicit experimental robustness hyperparameter, not a paper guarantee.
This is an offline task-distortion term, NOT potential-based reward shaping.
Ground-truth bins are never needed by the deployment encoder.
"""

from __future__ import annotations

import numpy as np


def cyclic_column_period(label_bins=None, width=None, *, period=None):
    """Validate an opt-in cyclic column coordinate, leaving ordinary maps alone.

    RADIal annotations carry their native Doppler FFT-bin period. No cyclic
    behavior is inferred from map dimensions, so existing MATLAB/RA callers
    retain clipped physical-coordinate gates.
    """
    value = getattr(label_bins, 'period', None) if period is None else period
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError('Cyclic Doppler period must be a positive integer.')
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Cyclic Doppler period must be a positive integer.') from exc
    if not np.isfinite(numeric) or numeric <= 0 or numeric != np.floor(numeric):
        raise ValueError('Cyclic Doppler period must be a positive integer.')
    result = int(numeric)
    if width is not None and result != int(width):
        raise ValueError('Cyclic Doppler period must equal the full map column width.')
    return result


def label_gate_region(shape, row, col, row_radius, col_radius, period=None):
    """Index one unchanged rectangular gate, optionally wrapping its columns."""
    rows = slice(max(0, int(row-row_radius)), min(shape[0], int(row+row_radius+1)))
    if period is None:
        columns = slice(max(0, int(col-col_radius)), min(shape[1], int(col+col_radius+1)))
    else:
        delta = np.abs(np.arange(shape[1], dtype=np.int64)-int(col))
        columns = np.flatnonzero(np.minimum(delta, int(period)-delta) <= col_radius)
    return rows, columns


def lowest_cfar_margin_loss_v12_14(
    original_score,
    method_score,
    original_base_peaks,
    method_base_peaks,
    valid_map,
    label_bins,
    gate,
    threshold,
    margin_factor=1.25,
):
    """Return a dense loss and the unchanged strict lowest-CFAR miss count.

    Scores, fixed-NMS base peaks and validity MUST come from the same detector
    used for reported counts (before thresholding the fixed base peaks).
    ``valid_map`` must encode the existing complete-training-window rule.
    A high-score nonpeak cannot provide label support or reduce this loss.
    Gates are clipped at physical edges unless ``label_bins.period`` explicitly
    declares native cyclic Doppler columns. Row coordinates never wrap.
    A peak may support multiple labels exactly as in the any-in-gate metric.
    """
    if np.iscomplexobj(original_score) or np.iscomplexobj(method_score):
        raise ValueError("CA-CFAR scores must be real, nonnegative power ratios.")
    source = np.asarray(original_score, dtype=np.float64)
    candidate = np.asarray(method_score, dtype=np.float64)
    if source.ndim != 2 or 0 in source.shape or candidate.shape != source.shape:
        raise ValueError("CA-CFAR score maps must be nonempty matching 2-D arrays.")
    if not np.isfinite(source).all() or not np.isfinite(candidate).all():
        raise ValueError("CA-CFAR score maps must contain only finite scores.")
    if np.any(source < 0) or np.any(candidate < 0):
        raise ValueError("CA-CFAR scores must be nonnegative.")

    masks = []
    for name, value in (
        ("original_base_peaks", original_base_peaks),
        ("method_base_peaks", method_base_peaks),
        ("valid_map", valid_map),
    ):
        mask = np.asarray(value)
        if mask.shape != source.shape or mask.dtype.kind != "b":
            raise ValueError(f"{name} must be a matching boolean map.")
        masks.append(mask)
    source_peaks, candidate_peaks, valid = masks
    source_peaks = source_peaks & valid
    candidate_peaks = candidate_peaks & valid

    threshold = float(threshold)
    margin_factor = float(margin_factor)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("The lowest-CFAR threshold must be finite and positive.")
    if not np.isfinite(margin_factor) or margin_factor < 1:
        raise ValueError("The experimental margin factor must be finite and >= 1.")
    cap = threshold * margin_factor
    if not np.isfinite(cap):
        raise ValueError("The threshold times margin factor must remain finite.")

    period = cyclic_column_period(label_bins, source.shape[1])
    bins = np.asarray(label_bins)
    if bins.size == 0:
        bins = np.empty((0, 2), dtype=np.int64)
    if bins.ndim != 2 or bins.shape[1] != 2:
        raise ValueError("GT bins must be an N-by-2 array of physical map indices.")
    if bins.dtype.kind not in "iuf" or not np.isfinite(bins).all():
        raise ValueError("GT bins must be finite numeric indices.")
    if np.any(bins != np.floor(bins)):
        raise ValueError("GT bin coordinates must be integers.")
    bins = bins.astype(np.int64)
    if np.any(bins < 0) or np.any(bins[:, 0] >= source.shape[0]) or np.any(bins[:, 1] >= source.shape[1]):
        raise ValueError("GT bins must lie inside the physical RA/RD map.")
    gate_array = np.asarray(gate)
    if gate_array.shape != (2,) or gate_array.dtype.kind not in "iuf":
        raise ValueError("The existing GT gate must contain two nonnegative integers.")
    if not np.isfinite(gate_array).all() or np.any(gate_array < 0) or np.any(gate_array != np.floor(gate_array)):
        raise ValueError("The existing GT gate must contain two nonnegative integers.")
    row_radius, col_radius = gate_array.astype(np.int64)

    eligible_indices, targets, candidate_best, observed_misses = [], [], [], []
    for label_index, (row, col) in enumerate(bins):
        region = label_gate_region(source.shape, row, col, row_radius, col_radius, period)
        original_values = source[region][source_peaks[region]]
        original_best = float(original_values.max()) if original_values.size else 0.0
        if original_best <= threshold:
            continue
        method_values = candidate[region][candidate_peaks[region]]
        method_best = float(method_values.max()) if method_values.size else 0.0
        eligible_indices.append(label_index)
        targets.append(min(original_best, cap))
        candidate_best.append(method_best)
        observed_misses.append(method_best <= threshold)

    targets = np.asarray(targets, dtype=np.float64)
    candidate_best = np.asarray(candidate_best, dtype=np.float64)
    deficits = np.maximum(targets - candidate_best, 0.0) / threshold
    return {
        "margin_sum": float(deficits.sum()),
        "margin_mean": float(deficits.mean()) if deficits.size else 0.0,
        "num_eligible": int(deficits.size),
        "num_low_misses": int(np.count_nonzero(observed_misses)),
        "eligible_label_indices": np.asarray(eligible_indices, dtype=np.int64),
        "target_scores": targets,
        "method_best_scores": candidate_best,
        "per_label_margin": deficits,
        "margin_factor": margin_factor,
    }
