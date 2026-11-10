"""V12.22 isolated actor auxiliaries on fresh TRAIN conditional contexts.

The V12.21 PPO update, ordinary same-cube LOO credit, fresh conditional
replacement, primary-only critic targets, Adam schedule and global KL stop are
delegated unchanged. One optional auxiliary contributes a separate mean over
the queried states at every native minibatch. Queries are not extra rollouts,
and their gradients are not attenuated by queried/primary-row frequency.

These are explicitly modified actor objectives, not unmodified PPO or a
preservation guarantee. No files, radar data, detector or deployment refinement
are accessed here. All conditional outcomes must come from fresh TRAIN branches.
"""
from __future__ import annotations

import numpy as np

from radar_sim.ppo_v12_21_learning import update_ppo_v12_21


ACTOR_NAMES = ('W1', 'b1', 'Wa', 'ba')
AUXILIARIES = ('native', 'margin', 'compatible', 'risk')
MARGIN = .05
MARGIN_COEFFICIENT = .30


class NoninformativeAuxiliaryCalibration(ValueError):
    """The fixed TRAIN calibration cannot identify one or more coefficients."""

    def __init__(self, report):
        self.report = report
        super().__init__('Noninformative TRAIN auxiliary calibration: '
                         + ', '.join(report['noninformative_auxiliaries']))


def _actor_names(policy):
    names = tuple(name for name in ACTOR_NAMES if name in policy.params)
    if not names:
        raise ValueError('The policy has no recognized actor parameter arrays.')
    return names


def _norm(gradients, names):
    return float(np.sqrt(sum(np.sum(np.asarray(gradients[name], float)**2)
                             for name in names)))


def _cosine(left, right, names):
    denominator = _norm(left, names)*_norm(right, names)
    if denominator <= 0:
        return None
    numerator = sum(float(np.sum(np.asarray(left[name], float)
                                  * np.asarray(right[name], float))) for name in names)
    return float(np.clip(numerator/denominator, -1., 1.))


def _check_gradients(policy, gradients, *, actor_only=False):
    if (set(gradients) != set(policy.params) or any(
            np.shape(value) != np.shape(policy.params[name])
            or not np.isfinite(value).all() for name, value in gradients.items())):
        raise ValueError('Network gradients must be finite and match policy arrays.')
    if actor_only and any(np.any(np.asarray(value) != 0)
                          for name, value in gradients.items() if name not in _actor_names(policy)):
        raise AssertionError('An actor-only auxiliary produced a critic gradient.')


def _packet_queries(states, packets):
    states = np.asarray(states)
    packets = list(packets)
    if (states.ndim != 2 or not len(states) or not np.isfinite(states).all()
            or not packets):
        raise ValueError('Auxiliaries need finite primary states and nonempty fresh queries.')
    indices, masks = [], []
    for packet in packets:
        index = packet['primary_row']
        if (isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer))
                or not 0 <= index < len(states) or int(index) in indices):
            raise ValueError('Queries need distinct valid primary-row indices.')
        old = np.asarray(packet['old_probabilities'], float)
        if (old.ndim != 1 or not len(old) or not np.isfinite(old).all()
                or np.any(old < 0) or not np.isclose(old.sum(), 1.)):
            raise ValueError('Queries need finite normalized old categorical probabilities.')
        indices.append(int(index))
        masks.append(old > 0.)
    try:
        masks = np.stack(masks)
    except ValueError as exc:
        raise ValueError('All query action dimensions must match.') from exc
    return states[np.asarray(indices)], masks, packets


def auxiliary_loss_gradient(probabilities, masks, packets, auxiliary):
    """Mean queried-state loss and unaveraged per-row logit derivatives.

    ``gradients_fn`` averages its input rows, so these row derivatives must NOT
    be divided by the query count again. For an independent logit finite-
    difference test of the scalar mean, divide this returned array by M.

    D: hinge .05 against one lexicographic LOW/SIX/U/rate target (stable ID ties).
    E: negative log probability mass of all LOW/SIX lexicographic minimizers.
    F: expected absolute lowest-threshold misses, sum_a pi(a)*LOW(a).
    """
    if auxiliary not in AUXILIARIES:
        raise ValueError('Unknown actor auxiliary: '+str(auxiliary))
    p, mask, packets = np.asarray(probabilities, float), np.asarray(masks), list(packets)
    if (p.ndim != 2 or min(p.shape) < 1 or mask.shape != p.shape or mask.dtype != np.bool_
            or not mask.any(1).all() or len(packets) != len(p)
            or not np.isfinite(p).all() or np.any(p < 0)
            or np.any(p[~mask] != 0) or not np.allclose(p.sum(1), 1.)
            or np.any(p[mask] <= 0)):
        raise ValueError('Aligned positive feasible categorical probabilities and query masks required.')
    # Normalize roundoff consistently with the native categorical surrogate.
    p = p/p.sum(1, keepdims=True)
    derivative, losses, rows = np.zeros_like(p), [], []
    for row, packet in enumerate(packets):
        metrics = {}
        for name in ('lowest_misses', 'six_misses', 'unmatched', 'bpcs'):
            value = np.asarray(packet[name], float)
            if (value.shape != (p.shape[1],) or not np.isfinite(value[mask[row]]).all()
                    or np.any(value[mask[row]] < 0)):
                raise ValueError('Query outcomes must be finite, nonnegative and action aligned: '+name)
            if name in ('lowest_misses', 'six_misses') and not np.all(
                    value[mask[row]] == np.floor(value[mask[row]])):
                raise ValueError('Preservation outcomes must retain absolute integer counts.')
            metrics[name] = value
        feasible = np.flatnonzero(mask[row])
        target = int(min(feasible, key=lambda action: (
            metrics['lowest_misses'][action], metrics['six_misses'][action],
            metrics['unmatched'][action], metrics['bpcs'][action], int(action))))
        low_min = metrics['lowest_misses'][feasible].min()
        compatible = mask[row] & (metrics['lowest_misses'] == low_min)
        six_min = metrics['six_misses'][compatible].min()
        compatible &= metrics['six_misses'] == six_min
        mass = float(p[row, compatible].sum())
        if mass <= 0:
            raise FloatingPointError('Compatible-action probability mass underflowed.')
        expected_low = float(p[row, feasible] @ metrics['lowest_misses'][feasible])
        loss, rival, gap = 0., None, None
        if auxiliary == 'margin' and len(feasible) > 1:
            others = feasible[feasible != target]
            rival = int(others[np.argmax(p[row, others])])
            gap = float(np.log(p[row, target])-np.log(p[row, rival]))
            loss = max(0., MARGIN-gap)
            if loss > 0:
                derivative[row, target], derivative[row, rival] = -1., 1.
        elif auxiliary == 'compatible' and not compatible[feasible].all():
            loss = -float(np.log(mass))
            derivative[row] = p[row]
            derivative[row, compatible] -= p[row, compatible]/mass
        elif auxiliary == 'risk':
            loss = expected_low
            derivative[row, feasible] = p[row, feasible]*(metrics['lowest_misses'][feasible]-expected_low)
        losses.append(loss)
        greedy = int(np.argmax(p[row]))
        rows.append(dict(primary_row=int(packet['primary_row']), auxiliary=auxiliary,
            auxiliary_loss=float(loss), target_action=target, target_probability=float(p[row, target]),
            compatible_probability=mass, compatible_action_count=int(compatible.sum()),
            compatible_actions=np.flatnonzero(compatible).tolist(),
            greedy_action=greedy, greedy_is_target=bool(greedy == target),
            greedy_is_compatible=bool(compatible[greedy]), rival_action=rival, logit_margin=gap,
            expected_lowest_misses=expected_low,
            greedy_lowest_misses=float(metrics['lowest_misses'][greedy]),
            greedy_six_misses=float(metrics['six_misses'][greedy]),
            lowest_misses_min=float(low_min), six_misses_min_given_low=float(six_min)))
    return float(np.mean(losses)), derivative, rows


def _auxiliary_gradients(policy, states, masks, packets, auxiliary, gradients_fn, mask_fn=None):
    probability = np.asarray(policy.forward(states)[0], float)
    if mask_fn is None:
        probability = np.where(masks, probability, 0.)
        probability /= probability.sum(1, keepdims=True)
    else:
        probability = np.asarray(mask_fn(probability, masks), float)
    loss, row_derivative, rows = auxiliary_loss_gradient(probability, masks, packets, auxiliary)
    gradients = gradients_fn(policy, states, row_derivative, None)
    _check_gradients(policy, gradients, actor_only=True)
    return loss, gradients, rows, probability


def _observational_query_probability(policy, states, masks, packets, mask_fn):
    """Pure forward diagnostics, with no gradient callback or optimizer access."""
    probability = np.asarray(mask_fn(policy.forward(states)[0], masks), float)
    _, _, rows = auxiliary_loss_gradient(probability, masks, packets, 'native')
    return probability, rows


def _query_change_report(old, probability, masks, packets, before_rows, after_rows):
    kl = np.sum(np.where(masks, old*np.log(np.maximum(old, np.finfo(float).tiny)
        /np.maximum(probability, np.finfo(float).tiny)), 0.), axis=1)
    action_rows = [dict(primary_row=int(packet['primary_row']), action=action,
                       feasible=bool(masks[index, action]),
                       old_probability=float(old[index, action]),
                       probability_after=float(probability[index, action]))
                  for index, packet in enumerate(packets)
                  for action in range(probability.shape[1])]
    return dict(query_diagnostics_available=True,
        query_local_kl_mean=max(float(kl.mean()), 0.), query_local_kl_max=max(float(kl.max()), 0.),
        target_probability_before=float(np.mean([row['target_probability'] for row in before_rows])),
        target_probability_after=float(np.mean([row['target_probability'] for row in after_rows])),
        compatible_probability_before=float(np.mean([row['compatible_probability'] for row in before_rows])),
        compatible_probability_after=float(np.mean([row['compatible_probability'] for row in after_rows])),
        auxiliary_context_rows=[dict(row, stage='before') for row in before_rows]
            + [dict(row, stage='after') for row in after_rows],
        conditional_action_probabilities_after=action_rows)


def calibrate_auxiliaries(policy, states, packets, gradients_fn):
    """Freeze E/F scales using initial-policy TRAIN per-context gradient RMS.

    G_j = sqrt(mean_context ||grad_actor L_j(context)||_2**2).
    lambda_D=.30; lambda_E/F=.30*G_D/G_E/F. This is a fixed calibration,
    not online normalization, validation tuning or a terminal reward change.
    A zero/nonfinite G stops calibration with an attached report; it never
    quietly becomes an active zero-coefficient experiment.
    """
    x, masks, packets = _packet_queries(states, packets)
    expected = np.stack([np.asarray(packet['old_probabilities'], float) for packet in packets])
    observed = np.where(masks, np.asarray(policy.forward(x)[0], float), 0.)
    observed /= observed.sum(1, keepdims=True)
    if not np.allclose(observed, expected, rtol=1e-5, atol=2e-7):
        raise AssertionError('Calibration outcomes must refer to the initial current TRAIN policy.')
    rows, squares = [], {name: [] for name in AUXILIARIES[1:]}
    for index, packet in enumerate(packets):
        for auxiliary in AUXILIARIES[1:]:
            _, gradients, details, _ = _auxiliary_gradients(policy, x[index:index+1],
                masks[index:index+1], [packet], auxiliary, gradients_fn)
            norm = _norm(gradients, _actor_names(policy))
            squares[auxiliary].append(norm**2)
            rows.append(dict(details[0], actor_parameter_gradient_norm=norm))
    rms = {name: float(np.sqrt(np.mean(value))) for name, value in squares.items()}
    bad = [name for name, value in rms.items() if not np.isfinite(value) or value <= 1e-12]
    coefficients = dict(native=0., margin=MARGIN_COEFFICIENT)
    coefficients.update({name: (MARGIN_COEFFICIENT*rms['margin']/rms[name]
                               if name not in bad and 'margin' not in bad else None)
                         for name in ('compatible', 'risk')})
    report = dict(coefficients=coefficients, gradient_rms=rms, rows=rows,
        informative=not bool(bad), noninformative_auxiliaries=bad, contexts=len(packets),
        source_partition='TRAIN', policy_stage='initial_before_any_optimizer_updates',
        normalization='sqrt(mean_context squared actor-parameter gradient L2 norm)',
        frozen_after_calibration=True)
    if bad:
        raise NoninformativeAuxiliaryCalibration(report)
    return report


def update_ppo_v12_22(policy, optimizer, trajectories, conditional_contexts, *,
                     auxiliary='native', auxiliary_coefficient=0., epochs, clip,
                     entropy_coef, target_kl, value_coef, reward_scale, gradients_fn,
                     mask_fn, seed, eps, batch_size=256):
    """Unchanged native PPO plus at most one explicit, actor-only auxiliary.

    Each native SGD minibatch contributes its native mean plus lambda times
    the full queried-context mean. The existing V12.21 epoch-level global KL
    stop is the sole update guard; query-local KL is measured, not clamped.
    Native mode (or zero lambda) delegates exactly, including optimizer state.
    """
    if (auxiliary not in AUXILIARIES or not np.isfinite(auxiliary_coefficient)
            or auxiliary_coefficient < 0 or (auxiliary == 'native' and auxiliary_coefficient != 0)):
        raise ValueError('Use one known auxiliary and a finite nonnegative coefficient; native lambda=0.')
    trajectories, packets = list(trajectories), list(conditional_contexts)
    settings = dict(epochs=epochs, clip=clip, entropy_coef=entropy_coef, target_kl=target_kl,
        value_coef=value_coef, reward_scale=reward_scale, mask_fn=mask_fn, seed=seed,
        eps=eps, batch_size=batch_size)
    # The exact control path has no extra gradient, parameter or RNG changes.
    # Additional forward-only query diagnostics are observational, not losses.
    if auxiliary == 'native' or auxiliary_coefficient == 0:
        observation = None
        metrics = {'lowest_misses', 'six_misses', 'unmatched', 'bpcs'}
        if packets and all(metrics.issubset(packet) for packet in packets):
            states = np.concatenate([np.asarray(item['states'], np.float32) for item in trajectories])
            query_states, query_masks, _ = _packet_queries(states, packets)
            observed_old, before_rows = _observational_query_probability(
                policy, query_states, query_masks, packets, mask_fn)
            observation = (query_states, query_masks, observed_old, before_rows)
        result = update_ppo_v12_21(policy, optimizer, trajectories, packets,
                                  gradients_fn=gradients_fn, **settings)
        result.update(auxiliary=auxiliary, auxiliary_coefficient=float(auxiliary_coefficient),
            auxiliary_loss=0., weighted_auxiliary_loss=0., auxiliary_enabled=False,
            native_ppo_loss=result['ppo_loss'], total_loss=result['ppo_loss'],
            native_actor_loss=result['actor_loss'], total_actor_loss=result['actor_loss'],
            auxiliary_gradient_norm=0., weighted_auxiliary_gradient_norm=0.,
            auxiliary_gradient_rows=[], auxiliary_context_rows=[],
            query_diagnostics_available=False)
        if observation is not None:
            query_states, query_masks, observed_old, before_rows = observation
            after_probability, after_rows = _observational_query_probability(
                policy, query_states, query_masks, packets, mask_fn)
            result.update(_query_change_report(observed_old, after_probability,
                query_masks, packets, before_rows, after_rows))
        return result
    states = np.concatenate([np.asarray(item['states'], np.float32) for item in trajectories])
    x, masks, packets = _packet_queries(states, packets)
    # Validate outcomes and initial queried-state probability before any mutation.
    initial_loss, _, before_rows, old = _auxiliary_gradients(
        policy, x, masks, packets, auxiliary, gradients_fn, mask_fn)
    expected = np.stack([np.asarray(packet['old_probabilities'], float) for packet in packets])
    if not np.allclose(old, expected, rtol=1e-5, atol=2e-7):
        raise AssertionError('Auxiliary queries are stale/off-policy.')
    actor_names, gradient_rows = _actor_names(policy), []

    def augmented_gradients(current_policy, minibatch_states, logits_gradient, value_gradient=None):
        native = gradients_fn(current_policy, minibatch_states, logits_gradient, value_gradient)
        _check_gradients(current_policy, native)
        loss, additional, _, probability = _auxiliary_gradients(
            current_policy, x, masks, packets, auxiliary, gradients_fn, mask_fn)
        kl = np.sum(np.where(masks, old*np.log(np.maximum(old, np.finfo(float).tiny)
            /np.maximum(probability, np.finfo(float).tiny)), 0.), axis=1)
        raw_norm = _norm(additional, actor_names)
        gradient_rows.append(dict(optimizer_step_before=int(optimizer.t),
            auxiliary=auxiliary, auxiliary_coefficient=float(auxiliary_coefficient),
            queried_contexts=len(x), primary_minibatch_rows=len(minibatch_states),
            auxiliary_loss=loss, weighted_auxiliary_loss=float(auxiliary_coefficient*loss),
            native_actor_gradient_norm=_norm(native, actor_names),
            raw_auxiliary_gradient_norm=raw_norm,
            weighted_auxiliary_gradient_norm=float(auxiliary_coefficient*raw_norm),
            native_auxiliary_gradient_cosine=_cosine(native, additional, actor_names),
            query_local_kl_mean=max(float(kl.mean()), 0.),
            query_local_kl_max=max(float(kl.max()), 0.)))
        return {name: (np.asarray(value)+auxiliary_coefficient*np.asarray(additional[name]))
                .astype(np.asarray(value).dtype, copy=False) for name, value in native.items()}

    result = update_ppo_v12_21(policy, optimizer, trajectories, packets,
                              gradients_fn=augmented_gradients, **settings)
    loss, final_gradient, after_rows, probability = _auxiliary_gradients(
        policy, x, masks, packets, auxiliary, gradients_fn, mask_fn)
    conflicts = [row['native_auxiliary_gradient_cosine'] for row in gradient_rows
                 if row['native_auxiliary_gradient_cosine'] is not None]
    result.update(auxiliary=auxiliary, auxiliary_enabled=True,
        auxiliary_coefficient=float(auxiliary_coefficient), auxiliary_loss=loss,
        auxiliary_loss_before=initial_loss, weighted_auxiliary_loss=float(auxiliary_coefficient*loss),
        native_ppo_loss=result['ppo_loss'],
        native_actor_loss=result['actor_loss'],
        total_loss=result['ppo_loss']+auxiliary_coefficient*loss,
        total_actor_loss=result['actor_loss']+auxiliary_coefficient*loss,
        auxiliary_gradient_norm=float(np.mean([row['raw_auxiliary_gradient_norm'] for row in gradient_rows])),
        weighted_auxiliary_gradient_norm=float(np.mean([row['weighted_auxiliary_gradient_norm'] for row in gradient_rows])),
        auxiliary_gradient_norm_after=_norm(final_gradient, actor_names),
        native_auxiliary_gradient_cosine=float(np.mean(conflicts)) if conflicts else None,
        auxiliary_gradient_rows=gradient_rows,
        auxiliary_normalization='separate equal-context query mean; no primary-frequency factor',
        auxiliary_critic_gradient_zero=True)
    result.update(_query_change_report(old, probability, masks, packets, before_rows, after_rows))
    result['ppo_loss'] = result['total_loss']
    result['actor_loss'] = result['total_actor_loss']
    return result
