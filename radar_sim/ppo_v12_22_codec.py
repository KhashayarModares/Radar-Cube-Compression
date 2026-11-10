"""V12.22 immutable action menus and label-free observation/codec adapter.

The physical codec is the unchanged V12.14 coherent global-RD transform. This
module never changes its historical action library, detector, labels or gates.
Rates describe the declared analytical syntax, not a serialized bitstream.
The four last state slots replace obsolete budget fields; missing original-map
coverage is mapped to normalized zero, not mistaken for a measured CFAR value.
"""
from __future__ import annotations

from collections.abc import Sequence
import time

import numpy as np
import pandas as pd

from . import ppo_v12_14_codec as physical


ORIGINAL_ACTIONS = tuple(physical.DEFAULT_ACTIONS) + ((0.4, 4), (0.4, 5))
ORIGINAL16 = ORIGINAL_ACTIONS
PROPOSED_ACTIONS = (
    (0.20, 8), (1.00, 12), (0.60, 10), (0.70, 12),
    (0.90, 12), (0.05, 8), (0.10, 8), (0.05, 10),
    (0.05, 2), (0.30, 12), (0.80, 10), (0.05, 4),
    (0.10, 12), (0.05, 6), (0.30, 10), (0.50, 12),
    (0.08, 5), (0.25, 7), (0.12, 9), (0.10, 10), (0.40, 12),
)
CFAR_NAMES = ('weak_peak_frac', 'cfar_margin', 'bg_spread', 'peak_avail')
BASE_NAMES = (
    'doppler_tile_center', 'range_frequency_tile_center', 'running_rate_fraction',
    'previous_keep_fraction', 'block_log_energy_ratio',
    'spectral_top10_energy_fraction', 'cube_mean_spectral_concentration',
    'cube_block_log_energy_std', 'cube_spectral_concentration_std',
    'spectral_top1_energy_fraction', 'spectral_top5_energy_fraction',
    'spectral_entropy_normalized', 'spectral_peak_to_mean_log',
    'previous_q_div_16', 'block_energy_fraction', 'future_energy_fraction',
)


def _actions(actions: Sequence) -> tuple:
    result = tuple(physical._validated_action(action) for action in actions)
    if not result or len(set(result)) != len(result):
        raise ValueError('A nonempty immutable menu of distinct physical actions is required.')
    return result


def state_names(actions=PROPOSED_ACTIONS) -> tuple[str, ...]:
    """The native 16+2K state descriptors followed by four CFAR/zero slots."""
    actions = _actions(actions)
    descriptors = tuple(f'action_{i}_{name}' for i in range(len(actions))
                        for name in ('rate_fraction', 'log1p_scaled_parseval_error'))
    return BASE_NAMES + descriptors + CFAR_NAMES


def prepare_source(ex, cube, actions=PROPOSED_ACTIONS):
    """Prepare every supplied action without consulting or mutating ex globals."""
    actions = _actions(actions)
    source = ex.coherent_codec_v14.prepare_source(cube, actions)
    fractions = np.asarray([p['tile_energy'] for p in source['prepared_blocks']], float)
    fractions /= max(float(source['whole_cube_energy']), float(ex.EPS))
    suffix = np.cumsum(fractions[::-1])[::-1]
    for i, prepared in enumerate(source['prepared_blocks']):
        prepared['block_energy_fraction'] = float(fractions[i])
        prepared['future_energy_fraction'] = float(suffix[i + 1]) if i + 1 < len(suffix) else 0.
        prepared['action_distortion'] = prepared['action_distortion_contributions']
    return source


def _control_state(ex, slc, shape, running, previous_index, prepared, source):
    """Native V12.21 physical state, generalized only in its explicit menu size."""
    actions = source['actions']
    previous = (0., 0) if previous_index is None else actions[int(previous_index)]
    eps = float(ex.EPS)
    reference = float(ex.RL_RATE_TARGET_BPCS_V11)
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError('The inherited positive state-reference bitrate is required; it is not a cap.')
    values = [
        .5 * (slc[1].start + slc[1].stop) / shape[1],
        .5 * (slc[2].start + slc[2].stop) / shape[2],
        running / max(int(np.prod(shape)) * reference, eps), previous[0],
        float(np.clip(np.log((prepared['block_mean_power'] + eps) /
                            (source['cube_mean_power'] + eps)), -12, 12)),
        prepared['spectral_top10_fraction'], source['cube_mean_spectral_concentration'],
        source['cube_block_log_energy_std'], source['cube_spectral_concentration_std'],
        prepared['spectral_top1_fraction'], prepared['spectral_top5_fraction'],
        prepared['spectral_entropy_normalized'], prepared['spectral_peak_to_mean_log'],
        previous[1] / 16., prepared['block_energy_fraction'], prepared['future_energy_fraction'],
    ]
    for bits, distortion in zip(prepared['action_bits'], prepared['action_distortion']):
        values.extend([bits / max(prepared['total_coeffs'] * reference, eps),
                       float(np.log1p(1000. * distortion))])
    return np.asarray(values + [0.] * 4, np.float32)


def _coverage(covered, n):
    if covered is None:
        return np.zeros(n, bool)
    value = np.asarray(covered)
    if value.dtype != np.bool_ or value.shape not in ((n,), ()):
        raise ValueError('Coverage must be an explicit boolean scalar or one value per observation.')
    return np.full(n, bool(value), bool) if value.ndim == 0 else value.copy()


class Normalizer:
    """TRAIN-fitted normalization with explicit original-map feature coverage.

    Missing ``covered`` means unknown coverage and yields four normalized zeros.
    Valid coverage with no detected peaks is NOT missing: its measured values
    (including the native -60 dB no-candidate margin) are normalized normally.
    """
    def __init__(self, mean, std, clip=5., cfar_enabled=False):
        self.mean = np.asarray(mean, np.float32).copy()
        self.std = np.asarray(std, np.float32).copy()
        if (self.mean.ndim != 1 or self.mean.shape != self.std.shape or
                len(self.mean) < 22 or (len(self.mean) - 20) % 2 or
                not np.isfinite(np.r_[self.mean, self.std]).all() or np.any(self.std <= 0)):
            raise ValueError('Finite 20+2K mean/std vectors with strictly positive scales are required.')
        if clip is not None and (not np.isfinite(clip) or float(clip) <= 0):
            raise ValueError('Normalization clip must be positive or None.')
        self.clip = None if clip is None else float(clip)
        self.cfar_enabled = bool(cfar_enabled)

    def _standardize(self, raw, covered):
        raw = np.asarray(raw, np.float32)
        if raw.ndim not in (1, 2) or raw.shape[-1] != len(self.mean) or not np.isfinite(raw).all():
            raise ValueError('Finite raw observations with this normalizer dimension are required.')
        one = raw.ndim == 1
        rows = raw[None] if one else raw
        mask = _coverage(covered, len(rows))
        output = (rows - self.mean) / self.std
        if not self.cfar_enabled:
            mask[:] = False
        output[~mask, -4:] = 0.
        return output, mask, one

    def transform(self, raw, covered=None):
        output, _, one = self._standardize(raw, covered)
        if self.clip is not None:
            output = np.clip(output, -self.clip, self.clip)
        output = output.astype(np.float32)
        return output[0] if one else output

    def clipping_stats(self, raw, covered=None):
        before, mask, _ = self._standardize(raw, covered)
        clipped = np.zeros_like(before, bool) if self.clip is None else np.abs(before) > self.clip
        denominator = np.full(before.shape[1], len(before), np.int64)
        denominator[-4:] = int(mask.sum())
        counts = clipped.sum(axis=0).astype(np.int64)
        fraction = np.divide(counts, denominator, out=np.zeros(len(counts), float), where=denominator > 0)
        return dict(rows=len(before), covered_rows=int(mask.sum()),
                    clipped_count=counts.tolist(), eligible_count=denominator.tolist(),
                    clipped_fraction=fraction.tolist())

    def as_dict(self):
        k = (len(self.mean) - 20) // 2
        return dict(mean=self.mean.tolist(), std=self.std.tolist(), clip=self.clip,
                    cfar_enabled=self.cfar_enabled, state_dimension=len(self.mean),
                    action_count=k, feature_names=list(CFAR_NAMES),
                    fitted_from='TRAIN label-free source observations only',
                    cfar_fit_uses_covered_rows_only=True,
                    missing_coverage_normalized_zero=True,
                    rate_reference_is_not_cap=True)

    @classmethod
    def from_dict(cls, values):
        if tuple(values.get('feature_names', CFAR_NAMES)) != CFAR_NAMES:
            raise ValueError('Saved CFAR feature definitions differ.')
        result = cls(values['mean'], values['std'], values.get('clip', 5.),
                     values.get('cfar_enabled', False))
        if int(values.get('state_dimension', len(result.mean))) != len(result.mean):
            raise ValueError('Saved state dimension differs from mean/std vectors.')
        return result


def fit_normalizer(raw_states, feature_values=None, coverage=None, *,
                   cfar_enabled=False, clip=5., split='TRAIN'):
    """Fit native descriptor statistics and separately fit covered TRAIN CFAR rows.

    ``feature_values`` has shape [n_feature_rows,4]. Its row count need not equal
    the native exploration-state count; the two sufficient-statistic pools are
    label-free TRAIN observations. Passing the same pools to B/C yields equal
    vectors, while only ``cfar_enabled`` differs.
    """
    if split != 'TRAIN':
        raise ValueError('Observation normalization may only be fitted on TRAIN.')
    raw = np.asarray(raw_states, np.float32)
    if raw.ndim != 2 or not len(raw) or not np.isfinite(raw).all():
        raise ValueError('A finite nonempty TRAIN state matrix is required.')
    mean = raw.mean(axis=0)
    std = raw.std(axis=0)
    std[std < 1e-5] = 1.
    mean[-4:], std[-4:] = 0., 1.
    if feature_values is not None:
        features = np.asarray(feature_values, np.float32)
        if features.ndim != 2 or features.shape[1] != 4 or not np.isfinite(features).all():
            raise ValueError('Finite TRAIN CFAR features must have four columns.')
        mask = _coverage(coverage, len(features))
        if mask.any():
            mean[-4:] = features[mask].mean(axis=0)
            scales = features[mask].std(axis=0)
            scales[scales < 1e-5] = 1.
            std[-4:] = scales
        elif cfar_enabled:
            raise ValueError('Cannot fit enabled CFAR channels without any covered TRAIN observation.')
    elif cfar_enabled:
        raise ValueError('Enabled CFAR normalization requires covered TRAIN feature values.')
    return Normalizer(mean, std, clip=clip, cfar_enabled=cfar_enabled)


def feature_bundle(ex, original, source):
    """Original-only P3 feature definitions with corrected coverage bookkeeping.

    No GT, reconstructed map, action outcomes, or future frame is consumed.
    Timing excludes the required original RD/RA postprocessing, as P3 does.
    """
    from .ppo_priority3_cfar_features import compute_cfar_features
    start = time.perf_counter()
    values, rows, contract = compute_cfar_features(ex, original, source)
    covered = np.asarray([int(r['RD_valid_CUTs']) + int(r['RA_valid_CUTs']) > 0 for r in rows], bool)
    contract = dict(contract)
    contract.update(state_dimension=20 + 2 * len(source['actions']),
                    slots=list(range(16 + 2 * len(source['actions']), 20 + 2 * len(source['actions']))),
                    native_first48_unchanged=None,
                    native_first_16_plus_2K_unchanged=True,
                    normalization='covered TRAIN rows only; missing coverage normalized zero',
                    covered_tiles=int(covered.sum()), uncovered_tiles=int((~covered).sum()),
                    feature_extraction_seconds=float(time.perf_counter() - start))
    return dict(values=np.asarray(values, np.float32), covered=covered,
                diagnostics=rows, contract=contract,
                feature_extraction_seconds=contract['feature_extraction_seconds'])


def encode(ex, cube, policy, normalizer, stochastic, source, features=None):
    """Uncapped raw PPO reconstruction; every immutable-menu action is allowed."""
    cube = np.asarray(cube)
    actions = _actions(source['actions'])
    k, n = len(actions), len(source['slices'])
    if tuple(cube.shape) != tuple(source['cube_shape']):
        raise ValueError('Prepared source shape differs from the raw cube.')
    allowed = np.asarray(policy.allowed_action_mask, bool)
    if allowed.shape != (k,) or not allowed.all():
        raise ValueError('Every action must remain available; no cap or action preselection.')
    if len(normalizer.mean) != 20 + 2 * k:
        raise ValueError('Normalizer and physical action library differ in dimension.')
    table, covered = np.zeros((n, 4), np.float32), np.zeros(n, bool)
    if features is not None:
        if isinstance(features, dict):
            table = np.asarray(features['values'], np.float32)
            covered = _coverage(features.get('covered'), n)
        else:
            raise ValueError('Features require explicit values and coverage, not an ambiguous bare matrix.')
        if table.shape != (n, 4) or not np.isfinite(table).all():
            raise ValueError('Finite four-feature values are required in exact tile order.')
    spectrum = np.zeros_like(source['spectrum'])
    states, raw_states, chosen, logp, values, logs, errors, probabilities = ([] for _ in range(8))
    running, previous = float(source['cube_header_bits']), None
    for index, (slc, prepared) in enumerate(zip(source['slices'], source['prepared_blocks'])):
        raw = _control_state(ex, slc, cube.shape, running, previous, prepared, source)
        raw[-4:] = table[index] if covered[index] else 0.
        state = normalizer.transform(raw[None], covered=covered[index:index + 1])[0]
        p, value, _ = policy.forward(state[None])
        p = np.asarray(p[0], np.float64)
        if p.shape != (k,) or not np.isfinite(p).all() or np.any(p < 0) or p.sum() <= 0:
            raise FloatingPointError('Invalid categorical actor probabilities.')
        p = p / p.sum()
        action = int(policy.rng.choice(k, p=p)) if stochastic else int(p.argmax())
        rho, q = actions[action]
        tile, bits = ex.coherent_codec_v14.materialize_action(prepared, rho, q)
        spectrum[slc] = tile
        running += float(bits['realized_bits'])
        states.append(state); raw_states.append(raw); chosen.append(action)
        logp.append(float(np.log(p[action] + ex.EPS))); values.append(float(value[0]))
        probabilities.append(p)
        errors.append(float(prepared['action_distortion'][action]))
        logs.append(dict({**bits, 'block_idx': index, 'action_index': action, 'rho': rho, 'q': q},
            block_log_energy_ratio=float(raw[4]), spectral_top10_energy_fraction=float(raw[5]),
            cube_mean_spectral_concentration=float(raw[6]), cube_block_log_energy_std=float(raw[7]),
            cube_spectral_concentration_std=float(raw[8]), feasible_action_count=k,
            compact_action_count=k, masked_action_count=0,
            transform_distortion_contribution=errors[-1], cfar_covered=bool(covered[index]),
            **{name: float(v) for name, v in zip(CFAR_NAMES, raw[-4:])}))
        previous = action
    decoded = ex.coherent_codec_v14.reconstruct(source, spectrum)
    raw_states = np.asarray(raw_states, np.float32)
    states = np.asarray(states, np.float32)
    if states.shape != (n, 20 + 2 * k) or not np.isfinite(states).all():
        raise FloatingPointError('Nonfinite or malformed sequential observations.')
    nmse = float(np.sum(errors))
    decoded_error = sum(float(np.sum(np.abs(np.asarray(cube[rx], np.complex128) - decoded[rx]) ** 2,
                                     dtype=np.float64)) for rx in range(cube.shape[0]))
    decoded_nmse = decoded_error / max(float(source['whole_cube_energy']), float(ex.EPS))
    if not np.isclose(nmse, decoded_nmse, rtol=1e-4, atol=1e-8):
        raise AssertionError('Shared-support Parseval distortion differs from decoded-cube NMSE.')
    meta = dict(method='PPO', codec='global_RD_shared_antenna_support_v14',
        realized_bits=running, realized_bpcs=running / cube.size, cube_nmse=nmse,
        decoded_cube_nmse=float(decoded_nmse), rel_error=float(np.sqrt(nmse)),
        compression_ratio_vs_v43_int16_iq_32bpc=32. * cube.size / running,
        memory_ratio_proxy_64b_complex=64. * cube.size / running,
        actual_codec_bitstream_saved=False, codec_size_is_analytical=True,
        rate_budget_bits=None, cube_header_bits=float(source['cube_header_bits']),
        rate_budget_enforced_by_action_mask=False, compact_policy_action_count=k,
        raw_cube_metrics_used_for_training=True, refinement_used=False,
        cfar_features_enabled=bool(normalizer.cfar_enabled),
        original_map_covered_tiles=int(covered.sum()),
        rate_reference_bpcs=float(ex.RL_RATE_TARGET_BPCS_V11), rate_reference_is_not_cap=True,
        shared_antenna_support=True, receiver_axis_transformed=False,
        phase_exact_after_quantization=False)
    return dict(cube_hat=decoded, states_raw=raw_states, states=states,
        actions=np.asarray(chosen, np.int64), action_masks=np.ones((n, k), bool),
        old_log_probs=np.asarray(logp, np.float32), old_values=np.asarray(values, np.float32),
        rewards=np.zeros(n, np.float64), block_log=pd.DataFrame(logs), meta=meta,
        distortion_contributions=np.asarray(errors, np.float64),
        action_probabilities=np.asarray(probabilities, np.float64),
        cfar_features=table.copy(), cfar_covered=covered.copy(),
        normalizer_clipping=normalizer.clipping_stats(raw_states, covered=covered))


# The facade replays a prefix, forces exactly one action, and samples a fresh
# behaviour-policy suffix. It is diagnostic/conditional credit, not refinement.
from .ppo_v12_21_codec import ForcedAction
