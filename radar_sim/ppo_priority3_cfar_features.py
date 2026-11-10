"""Label-free pre-action CFAR descriptors for the unchanged global-RD codec.

Information model: the encoder observes the complete ORIGINAL raw cube before
compression. This is already the V12.21 model (global FFT, cube statistics and
future-energy fractions), not a streaming/causal-sample encoder. The caller may
compute original RD/RA maps once, before choosing any compression action. No GT,
compressed cube, candidate action outcome or future decoded state is consumed.

Codec tiles index [receiver, shifted Doppler bin, shifted range-frequency bin].
Native detector maps are [range, Doppler] and [range, azimuth] after ROI cropping.
We match their physical axes back to the ORIGINAL complete FFT grids, never
reinterpret Doppler as azimuth. RD descriptors use the nominal tile rectangle;
RA descriptors use the corresponding range slab over ALL retained angles,
because native RA integrates over Doppler. Hann windows and nonperiodic MTI mix
spectral bins: this nominal-bin context is NOT an exact action-damage footprint.
RA range-slab context deliberately repeats across Doppler tiles.
"""
from __future__ import annotations

import copy
import time

import numpy as np


FEATURE_NAMES = ('weak_peak_frac', 'cfar_margin', 'bg_spread', 'peak_avail')
OBSOLETE_SLOT_NAMES = ('remaining_budget_fraction', 'minimum_remaining_cost_fraction',
                       'maximum_remaining_cost_fraction', 'allowance_per_remaining_coefficient')
FEATURE_SLOTS = (48, 49, 50, 51)
FEATURE_LOWER = np.asarray([0., -60., 0., 0.])
FEATURE_UPPER = np.asarray([1., 60., 120., 1.])
WEAK_BAND_DB = 3.
MARGIN_QUANTILE = .10
SOURCE_TRANSFORM = 'global_shifted_unitary_RD_fft2_no_receiver_fft'


def feature_contract():
    """Fixed definitions/bounds; these engineering choices are not DEV-tuned."""
    return dict(feature_names=list(FEATURE_NAMES), slots=list(FEATURE_SLOTS), state_dimension=52,
        replacement_slots=list(OBSOLETE_SLOT_NAMES), native_first_48_unchanged=True,
        information_model='complete ORIGINAL raw cube available before first compression action',
        streaming_claim=False, GT_used=False, post_action_outcomes_used=False,
        original_RD_RA_computation_required=True, detector_modified=False,
        CFAR_guard=[1, 1], CFAR_training=[4, 4], fixed_power_NMS_radius=[2, 2],
        peak_candidates='native positive-power, complete-window fixed-NMS maxima at threshold zero; includes subthreshold candidates',
        threshold='minimum unchanged native training CFAR factor (currently 1.25)',
        score='unchanged native score P/(B+native_EPS); threshold uses strict score > factor',
        definitions={
            'weak_peak_frac': dict(units='fraction', bounds=[0., 1.],
                formula='N(0 <= 10log10(native_score/threshold) <= 3 dB) / max(1,N_candidates)',
                subthreshold_candidates_in_denominator=True, weak_band_db=WEAK_BAND_DB),
            'cfar_margin': dict(units='dB', bounds=[-60., 60.],
                formula='clipped 10th percentile of 10log10(native_score/threshold) over all candidates',
                empty_candidate_value=-60., quantile=MARGIN_QUANTILE),
            'bg_spread': dict(units='dB', bounds=[0., 120.],
                formula='clipped IQR of 10log10(max(P/native_score-native_EPS,native_EPS)) at candidate positions',
                background_samples='candidate positions, not all CUTs', empty_candidate_value=0.),
            'peak_avail': dict(units='fraction', bounds=[0., 1.],
                formula='N_candidates / N_complete_window_valid_CUTs', empty_coverage_value=0.)},
        aggregation='equal mean of RD/RA weak fraction, background spread and candidate density over maps with valid CUTs; minimum per-map q10 margin',
        no_map_coverage_value=[0., 0., 0., 0.],
        RD_tile_attribution='physical FFT-axis matched nominal range/Doppler rectangle after native ROI crop',
        RA_tile_attribution='same nominal range slab, ALL retained azimuth bins, all Doppler contributions',
        RA_context_reused_across_Doppler_tiles=True,
        exact_action_influence_claim=False,
        limitations=['window/MTI leakage couples tiles',
                    'RA integrates Doppler and cannot have an exclusive Doppler-tile footprint',
                    'pooled four descriptors can hide separate RD/RA conditions; per-map diagnostics retained',
                    'peak/background descriptors describe Original observations, not guaranteed compression vulnerability',
                    'feature extraction timing excludes the required Original RD/RA postprocessing pass'],
        normalization='TRAIN-only CFAR mean/std; preserve native mapping for first 48; unchanged clip',
        initial_policy='zero new-slot actor/critic input weights, compensate old constant-zero normalized contribution in biases')


def _axis_indices(full, subset, name):
    full, subset = np.asarray(full, float), np.asarray(subset, float)
    if (full.ndim != 1 or subset.ndim != 1 or not len(full) or not len(subset)
            or not np.isfinite(full).all() or not np.isfinite(subset).all()
            or np.any(np.diff(full) <= 0) or np.any(np.diff(subset) <= 0)):
        raise ValueError('Finite strictly increasing complete/subset axes required: ' + name)
    position = np.searchsorted(full, subset)
    left = np.clip(position - 1, 0, len(full)-1)
    right = np.clip(position, 0, len(full)-1)
    chosen = np.where(np.abs(full[left]-subset) <= np.abs(full[right]-subset), left, right)
    if (not np.allclose(full[chosen], subset, rtol=1e-10, atol=1e-8)
            or len(np.unique(chosen)) != len(chosen)):
        raise ValueError('Detector axis does not match the complete codec FFT grid: ' + name)
    return chosen.astype(np.int64)


def _interval(value, extent, name):
    if not isinstance(value, slice) or value.step not in (None, 1):
        raise ValueError('Unit-stride spectral tile slice required: ' + name)
    start = 0 if value.start is None else int(value.start)
    stop = int(extent) if value.stop is None else int(value.stop)
    if start < 0 or stop <= start or stop > extent:
        raise ValueError('Spectral tile lies outside its FFT axis: ' + name)
    return start, stop


def _native_map(ex, original_post, kind):
    values, ranges, secondary = ex._map_triplet_v12_6(original_post, kind)
    values = np.asarray(values)
    if values.shape != (len(ranges), len(secondary)) or not np.isfinite(values).all():
        raise ValueError('Native ROI map must be finite [range, secondary-axis]: ' + kind)
    power = np.asarray(ex.as_linear_power(values, assume_db=True), np.float64)
    if power.shape != values.shape or not np.isfinite(power).all() or np.any(power < 0.):
        raise ValueError('Nonfinite/negative native linear-power map: ' + kind)
    factors = np.asarray(ex._nearest_training_thresholds_v11(), float)
    if factors.ndim != 1 or not len(factors) or not np.isfinite(factors).all() or np.any(factors <= 0.):
        raise ValueError('Finite positive native detector threshold factors required')
    threshold = float(factors.min())
    # Native publication CA-CFAR and fixed NMS are delegated unchanged.
    masks = ex.precompute_nested_cfar_peak_masks_v11_2(power, [0., threshold])
    zero = masks[0.]
    score, valid, candidates = (np.asarray(zero[k]) for k in ('cfar_score', 'valid', 'peaks'))
    if (score.shape != power.shape or valid.shape != power.shape or candidates.shape != power.shape
            or valid.dtype != np.bool_ or candidates.dtype != np.bool_
            or not np.isfinite(score).all() or np.any(score < 0)
            or np.any(candidates & (~valid | (power <= 0) | (score <= 0)))):
        raise ValueError('Malformed native CA-CFAR candidate fields: ' + kind)
    if not np.array_equal(np.asarray(masks[threshold]['peaks'], bool), candidates & (score > threshold)):
        raise AssertionError('Native low-threshold peaks differ from fixed candidate masking')
    return dict(kind=kind, power=power, score=score, valid=valid, candidates=candidates,
                ranges=np.asarray(ranges, float), secondary=np.asarray(secondary, float), threshold=threshold)


def _summarize(context, rows, columns, eps):
    selected = np.ix_(np.asarray(rows, bool), np.asarray(columns, bool))
    valid = context['valid'][selected]
    candidate = context['candidates'][selected]
    nvalid, npeaks = int(valid.sum()), int(candidate.sum())
    info = dict(valid_CUTs=nvalid, candidate_peaks=npeaks, subthreshold_candidates=0,
                low_threshold_surviving_candidates=0, weak_candidates=0,
                nominal_map_cells=int(valid.size), margin_q10_unclipped_db=None,
                background_IQR_unclipped_db=None)
    if not nvalid:
        return np.zeros(4, np.float64), info
    if not npeaks:
        return np.asarray([0., -60., 0., 0.]), info
    score = context['score'][selected][candidate]
    peak_power = context['power'][selected][candidate]
    margin = 10. * (np.log10(score) - np.log10(context['threshold']))
    # Native score is P/(B+EPS). Recover B only where P/score is identifiable:
    # positive NMS candidate positions, not zero-power/noncandidate CUTs.
    background = np.maximum(peak_power / score - eps, 0.)
    background_db = 10.*np.log10(np.maximum(background, eps))
    q10 = float(np.quantile(margin, MARGIN_QUANTILE))
    iqr = float(np.quantile(background_db, .75)-np.quantile(background_db, .25))
    weak = (margin >= 0.) & (margin <= WEAK_BAND_DB)
    info.update(subthreshold_candidates=int((score <= context['threshold']).sum()),
        low_threshold_surviving_candidates=int((score > context['threshold']).sum()),
        weak_candidates=int(weak.sum()), margin_q10_unclipped_db=q10,
        background_IQR_unclipped_db=iqr)
    value = np.asarray([float(weak.mean()), np.clip(q10, -60., 60.),
                        np.clip(iqr, 0., 120.), npeaks/nvalid], np.float64)
    if not np.isfinite(value).all() or np.any(value < FEATURE_LOWER) or np.any(value > FEATURE_UPPER):
        raise FloatingPointError('CFAR feature value outside its finite declared bounds')
    return value, info


def compute_cfar_features(ex, original_post, source):
    """Return ``(tile_features[n_tiles,4], diagnostic_rows, contract)``.

    All values are fixed for this Original cube, independent of the chosen
    action/prefix. The rest of the native observation still carries allocation
    history. Feature rows follow ``source['slices']`` exactly.
    """
    start = time.perf_counter()
    shape = tuple(int(v) for v in source['cube_shape'])
    if len(shape) != 3 or min(shape) < 1 or source.get('transform') != SOURCE_TRANSFORM:
        raise ValueError('Features require the native global shifted RD FFT2 codec, not local-time tiles')
    ranges = np.asarray(original_post['ranges'], float)
    velocity = np.asarray(original_post['vel'], float)
    angles = np.asarray(original_post['angle_axis_deg'], float)
    if (ranges.shape != (shape[2],) or velocity.shape != (shape[1],)
            or np.shape(original_post['RD']) != (shape[1], shape[2])
            or np.shape(original_post['RA']) != (len(angles), shape[2])):
        raise ValueError('Original full-map dimensions do not match codec FFT axes')
    eps = float(ex.EPS)
    if not np.isfinite(eps) or eps <= 0.:
        raise ValueError('Finite positive native detector EPS required')
    maps = {kind: _native_map(ex, original_post, kind) for kind in ('RD', 'RA')}
    for kind, context in maps.items():
        context['range_indices'] = _axis_indices(ranges, context['ranges'], kind+' range')
        if kind == 'RD':
            context['secondary_indices'] = _axis_indices(velocity, context['secondary'], 'RD Doppler')
    values, diagnostics = [], []
    slices = source['slices']
    if not len(slices):
        raise ValueError('At least one spectral tile is required')
    for index, slc in enumerate(slices):
        if len(slc) != 3 or _interval(slc[0], shape[0], 'receiver') != (0, shape[0]):
            raise ValueError('All receiver channels must share each original spectral tile')
        m0, m1 = _interval(slc[1], shape[1], 'Doppler')
        n0, n1 = _interval(slc[2], shape[2], 'range-frequency')
        permap, valid_maps = {}, []
        row = dict(tile_index=index, doppler_start=m0, doppler_stop=m1,
                   range_frequency_start=n0, range_frequency_stop=n1,
                   pre_action_original_only=True, prefix_dependent=False,
                   exact_action_footprint=False, RA_Doppler_integrated=True)
        for kind, context in maps.items():
            rmask = (context['range_indices'] >= n0) & (context['range_indices'] < n1)
            if kind == 'RD':
                smask = (context['secondary_indices'] >= m0) & (context['secondary_indices'] < m1)
            else:
                smask = np.ones(len(context['secondary']), bool)
            feature, detail = _summarize(context, rmask, smask, eps)
            permap[kind] = feature
            if detail['valid_CUTs']:
                valid_maps.append(kind)
            row.update({kind+'_'+name: float(v) for name, v in zip(FEATURE_NAMES, feature)})
            row.update({kind+'_'+key: value for key, value in detail.items()})
        if valid_maps:
            stack = np.stack([permap[kind] for kind in valid_maps])
            combined = stack.mean(axis=0)
            combined[1] = stack[:, 1].min()
        else:
            combined = np.zeros(4, np.float64)
        values.append(combined)
        row.update(dict(zip(FEATURE_NAMES, map(float, combined))))
        row['contributing_maps'] = ','.join(valid_maps)
        diagnostics.append(row)
    matrix = np.asarray(values, np.float32)
    if matrix.shape != (len(slices), 4) or not np.isfinite(matrix).all():
        raise AssertionError('Malformed tile CFAR feature matrix')
    contract = feature_contract()
    contract.update(feature_extraction_seconds=float(time.perf_counter()-start),
        original_postprocessing_included_in_timing=False, native_CFar_NMS_calls=2,
        tile_count=len(slices), complete_cube_shape=list(shape),
        native_threshold_factors={kind: context['threshold'] for kind, context in maps.items()},
        maps_with_no_valid_CUTs={kind: not bool(context['valid'].any()) for kind, context in maps.items()})
    return matrix, diagnostics, contract


class FeatureNormalizer:
    """Replace slots48:52; copy the native first48 mapping exactly."""
    def __init__(self, old_normalizer, feature_mean, feature_std, enabled=(True, True, True, True)):
        old_mean = np.asarray(old_normalizer.mean, np.float32)
        old_std = np.asarray(old_normalizer.std, np.float32)
        mean, std = np.asarray(feature_mean, np.float32), np.asarray(feature_std, np.float32)
        active = np.asarray(enabled)
        if (old_mean.shape != (52,) or old_std.shape != (52,) or mean.shape != (4,) or std.shape != (4,)
                or active.shape != (4,) or active.dtype != np.bool_
                or not np.isfinite(np.r_[old_mean, old_std, mean, std]).all()
                or np.any(old_std <= 0) or np.any(std <= 0)):
            raise ValueError('Finite native52 and four-feature normalization vectors required')
        self.mean = np.r_[old_mean[:48], mean].astype(np.float32)
        self.std = np.r_[old_std[:48], std].astype(np.float32)
        self.clip = old_normalizer.clip
        self.enabled = active.copy()
        self.old_mean, self.old_std = old_mean.copy(), old_std.copy()

    @classmethod
    def from_dict(cls, old_normalizer, values):
        """Restore new channels only, rejecting changes to the native mapping."""
        mean, std = np.asarray(values['mean'], np.float32), np.asarray(values['std'], np.float32)
        if (mean.shape != (52,) or std.shape != (52,)
                or tuple(values.get('CFAR_features', ())) != FEATURE_NAMES
                or values.get('clip') != old_normalizer.clip
                or not np.array_equal(mean[:48], np.asarray(old_normalizer.mean, np.float32)[:48])
                or not np.array_equal(std[:48], np.asarray(old_normalizer.std, np.float32)[:48])):
            raise ValueError('Saved feature normalizer differs from the native first48 mapping/contract')
        return cls(old_normalizer, mean[48:], std[48:], values['CFAR_enabled'])

    def transform(self, states):
        values = np.asarray(states, np.float32)
        if values.ndim < 1 or values.shape[-1] != 52 or not np.isfinite(values).all():
            raise ValueError('Finite raw52 observations with injected CFAR slots required')
        output = (values-self.mean)/self.std
        if self.clip is not None:
            output = np.clip(output, -self.clip, self.clip)
        output[..., 48:] = np.where(self.enabled, output[..., 48:], 0.)
        return output.astype(np.float32)

    def as_dict(self):
        return dict(mean=self.mean.tolist(), std=self.std.tolist(), clip=self.clip,
            CFAR_features=list(FEATURE_NAMES), CFAR_enabled=self.enabled.tolist(),
            original_first48_mapping_unchanged=True, fitted_from='TRAIN original pre-action maps only',
            disabled_channels_are_zero_after_normalization=True,
            historical_mean=self.old_mean.tolist(), historical_std=self.old_std.tolist())


def fit_feature_normalizer(old_normalizer, train_feature_rows, enabled=(True, True, True, True), *, split='TRAIN'):
    """Fit four new channels only; never refit the existing first48 features."""
    if split != 'TRAIN':
        raise ValueError('CFAR normalization may be fitted on TRAIN observations only')
    values = np.asarray(train_feature_rows, np.float32)
    if (values.ndim != 2 or values.shape[1] != 4 or not len(values) or not np.isfinite(values).all()
            or np.any(values < FEATURE_LOWER) or np.any(values > FEATURE_UPPER)):
        raise ValueError('Finite TRAIN CFAR feature rows within declared bounds required')
    mean, std = values.mean(axis=0), values.std(axis=0)
    std[std < 1e-5] = 1.
    return FeatureNormalizer(old_normalizer, mean, std, enabled)


def assess_obsolete_slots(old_normalizer, raw_states=None, state_names=None, policy=None):
    """Assess actual supplied historical states/weights, not an assumed zero."""
    if state_names is not None and tuple(state_names[-4:]) != OBSOLETE_SLOT_NAMES:
        raise ValueError('The four historical slots do not have the native obsolete budget names')
    zeros = np.zeros((1, 52), np.float32)
    constant = np.asarray(old_normalizer.transform(zeros))[0, 48:]
    result = dict(slots=list(FEATURE_SLOTS), old_raw_constant=[0., 0., 0., 0.],
        old_normalized_constant=constant.tolist(), state_names_checked=state_names is not None,
        raw_states_checked=False, obsolete_slots_proved_constant_in_supplied_states=False)
    if raw_states is not None:
        states = np.asarray(raw_states)
        if states.ndim != 2 or states.shape[1] != 52 or not len(states) or not np.isfinite(states).all():
            raise ValueError('Finite historical raw52 state matrix required')
        result.update(raw_states_checked=True, checked_rows=len(states),
            obsolete_slots_proved_constant_in_supplied_states=bool(np.all(states[:, 48:] == 0.)))
        if not result['obsolete_slots_proved_constant_in_supplied_states']:
            raise ValueError('Cannot replace nonzero historical budget slots under the uncapped information model')
    if policy is not None:
        result['old_actor_slot_weight_norm'] = float(np.linalg.norm(policy.params['W1'][48:]))
        result['old_critic_slot_weight_norm'] = float(np.linalg.norm(policy.params['Wc1'][48:]))
    return result


def identity_initialize(policy, old_normalizer):
    """Clone native52 actor/critic and remove only constant obsolete inputs.

    For old normalized constant c and old slot weights W, b'=b+c@W, W'=0.
    Thus the NEW feature values initially cannot change the old policy. With
    historical mean0/std1 budget slots, biases are unchanged exactly. A real
    observation parity check is still required by the experiment preflight.
    """
    result = copy.deepcopy(policy)
    info = assess_obsolete_slots(old_normalizer, policy=policy)
    constant = np.asarray(info['old_normalized_constant'], np.float32)
    for weight, bias in [('W1', 'b1'), ('Wc1', 'bc1')]:
        old_weight = np.asarray(policy.params[weight])
        if old_weight.ndim != 2 or old_weight.shape[0] != 52:
            raise ValueError('Original52-input actor and independent critic required')
        result.params[bias][...] = policy.params[bias] + constant @ old_weight[48:]
        result.params[weight][48:] = 0.
    info.update(new_feature_input_weights_zero=True, actor_and_critic_identity_by_construction=True,
        empirical_output_equivalence_verified=False, architecture_changed=False,
        real_state_parity_still_required=True)
    return result, info


def verify_initial_equivalence(reference, augmented, old_normalizer, new_normalizer, raw_states, feature_values):
    """Numerically verify actual supplied observations without any fitting."""
    raw = np.asarray(raw_states, np.float32)
    features = np.asarray(feature_values, np.float32)
    if (raw.ndim != 2 or raw.shape[1] != 52 or features.shape != (len(raw), 4)
            or not len(raw) or np.any(raw[:, 48:] != 0.)):
        raise ValueError('Aligned raw historical states with zero obsolete slots and CFAR values required')
    enriched = raw.copy(); enriched[:, 48:] = features
    before, after = old_normalizer.transform(raw), new_normalizer.transform(enriched)
    p0, v0, h0 = reference.forward(before)
    p1, v1, h1 = augmented.forward(after)
    z0 = h0 @ reference.params['Wa'] + reference.params['ba']
    z1 = h1 @ augmented.params['Wa'] + augmented.params['ba']
    error = dict(probability=float(np.max(np.abs(np.asarray(p0, float)-p1))),
                 critic_value=float(np.max(np.abs(np.asarray(v0, float)-v1))),
                 actor_logits=float(np.max(np.abs(np.asarray(z0, float)-z1))))
    unchanged = bool(np.array_equal(before[:, :48], after[:, :48]))
    passed = unchanged and all(np.isfinite(value) and value <= 2e-6 for value in error.values())
    return dict(passed=passed, rows=len(raw), first48_normalized_bitwise_equal=unchanged,
                maximum_absolute_errors=error, tolerance=2e-6,
                scope='supplied real observations if caller supplies real states; no physical performance claim')
