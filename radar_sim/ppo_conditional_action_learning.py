"""Pure finite-context actor diagnostic, NOT sequential PPO or deployment.

Q and outcome tables are supplied externally from original-relative radar
returns. This module neither constructs observations nor reads GT/data/files.
The all-action arm exactly integrates the frozen old categorical policy;
the sampled arm uses four independent old-policy samples per context.
"""
from __future__ import annotations

import numpy as np


def _mask(value, shape):
    mask = np.asarray(value)
    if (len(shape) != 2 or min(shape) < 1 or mask.shape != shape
            or mask.dtype != np.bool_ or not mask.any(axis=1).all()):
        raise ValueError('Every context needs an aligned Boolean, nonempty feasibility mask.')
    return mask


def _probability(value, mask):
    p = np.asarray(value, np.float64)
    if p.shape != mask.shape or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError('Policy probabilities must be finite, nonnegative and aligned.')
    p = np.where(mask, p, 0.)
    if np.any(p.sum(axis=1) <= 0):
        raise ValueError('A context has no feasible probability mass.')
    p /= p.sum(axis=1, keepdims=True)
    if np.any(p[mask] <= 0):
        raise ValueError('Feasible probabilities must be positive; softmax underflow is not silently repaired.')
    return p


def conditional_advantages(q_values, old_probability, mask, reward_scale=5000.):
    """Frozen A=(Q-E_old Q)/5000, then one old-policy-weighted choice-state RMS."""
    q = np.asarray(q_values, np.float64)
    mask = _mask(mask, q.shape)
    p = _probability(old_probability, mask)
    if not np.isfinite(reward_scale) or reward_scale <= 0 or not np.isfinite(q[mask]).all():
        raise ValueError('Finite feasible Q values and a positive fixed reward scale are required.')
    q = np.where(mask, q/reward_scale, 0.)
    raw = np.where(mask, q-np.sum(p*q, axis=1, keepdims=True), 0.)
    choice = mask.sum(axis=1) > 1
    rms = np.sqrt(np.mean(np.sum(p[choice]*raw[choice]**2, axis=1))) if choice.any() else 0.
    return np.where(choice[:, None], raw/max(float(rms), 1e-6), 0.), float(rms)


def sampled_action_weights(old_probability, samples_per_context, rng):
    if (isinstance(samples_per_context, (bool, np.bool_))
            or not isinstance(samples_per_context, (int, np.integer)) or samples_per_context < 1):
        raise ValueError('Samples per context must be a positive integer.')
    p = np.asarray(old_probability, np.float64)
    if p.ndim != 2 or not len(p) or not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1.):
        raise ValueError('Sampling needs normalized finite categorical probabilities.')
    return np.asarray([np.bincount(rng.choice(p.shape[1], samples_per_context, p=row),
        minlength=p.shape[1])/samples_per_context for row in p], np.float64)


def categorical_surrogate(probability, old_probability, advantages, mask, *,
                          weights=None, clip=.2, entropy_coef=.002):
    """Loss and exact logits gradient; external gradients_fn averages contexts.

Weights are old probabilities for exact integration or frozen empirical action
frequencies for the sampled arm. They never change inside a five-epoch block.
"""
    mask = _mask(mask, np.shape(probability))
    p, old = _probability(probability, mask), _probability(old_probability, mask)
    advantage = np.asarray(advantages, np.float64)
    weight = old if weights is None else np.asarray(weights, np.float64)
    if (advantage.shape != p.shape or not np.isfinite(advantage).all()
            or weight.shape != p.shape or not np.isfinite(weight).all() or np.any(weight < 0)
            or np.any(weight[~mask] != 0) or not np.allclose(weight.sum(1), 1.)
            or not np.isfinite([clip, entropy_coef]).all() or not 0 < clip < 1 or entropy_coef < 0):
        raise ValueError('Malformed frozen advantages, action weights or PPO coefficients.')
    ratio = np.divide(p, old, out=np.zeros_like(p), where=mask)
    surrogate = np.minimum(ratio*advantage, np.clip(ratio, 1-clip, 1+clip)*advantage)
    active = mask & ~(((advantage >= 0) & (ratio > 1+clip))
                      | ((advantage < 0) & (ratio < 1-clip)))
    logp = np.zeros_like(p)
    logold = np.zeros_like(old)
    logp[mask], logold[mask] = np.log(p[mask]), np.log(old[mask])
    entropy = -np.sum(p*logp, axis=1)
    logp_derivative = -weight*ratio*advantage*active
    gradient = logp_derivative-p*logp_derivative.sum(axis=1, keepdims=True)
    gradient += entropy_coef*p*(logp+entropy[:, None])
    gradient[~mask] = 0.
    choice = mask.sum(axis=1) > 1
    kl = np.maximum(np.sum(old*(logold-logp), axis=1), 0.)
    clipped = np.abs(ratio-1.) > clip
    loss = -float(np.mean(np.sum(weight*surrogate, axis=1)))-entropy_coef*float(entropy.mean())
    diagnostics = dict(loss=loss, exact_kl=float(kl[choice].mean()) if choice.any() else 0.,
        entropy=float(entropy.mean()), clip_fraction=float(np.mean(np.sum(weight[choice]*clipped[choice], axis=1)))
        if choice.any() else 0.)
    return loss, gradient, diagnostics


def fit_conditional_actor(policy, optimizer, states, masks, q_values, metric_tables,
                          gradients_fn, *, mode='full', seed=0, max_steps=200,
                          refresh_epochs=5, samples_per_context=4, clip=.2,
                          entropy_coef=.002, target_kl=.02, reward_scale=5000.,
                          actor_names=('W1', 'b1', 'Wa', 'ba')):
    """Return every raw endpoint for a frozen conditional-Q bandit capacity test.

policy.forward/params, optimizer.step and gradients_fn are supplied by caller.
    Critic parameter arrays are unused and must remain bitwise unchanged;
    optimizer moments are not audited. Refreshing old policy,
advantages and sample draws describes repeated bandit updates, not a replayed
sequential PPO trajectory, preservation guarantee or trained deployment policy.
"""
    x, q = np.asarray(states, np.float32), np.asarray(q_values, np.float64)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all() or q.ndim != 2 or len(q) != len(x):
        raise ValueError('Nonempty finite context states and an aligned context/action Q table are required.')
    mask = _mask(masks, q.shape)
    if not np.isfinite(q[mask]).all() or mode not in ('full', 'sampled'):
        raise ValueError('Use finite feasible Q values and full or sampled integration.')
    if (isinstance(max_steps, bool) or not isinstance(max_steps, (int, np.integer)) or max_steps < 1
            or not isinstance(refresh_epochs, (int, np.integer)) or refresh_epochs < 1
            or not np.isfinite(target_kl) or target_kl <= 0):
        raise ValueError('Positive finite step/refresh budgets and KL threshold are required.')
    metrics = {name: np.asarray(value, np.float64) for name, value in metric_tables.items()}
    required = {'low_misses', 'unmatched', 'bpcs', 'cube_nmse'}
    if not required.issubset(metrics) or any(value.shape != q.shape or not np.isfinite(value[mask]).all()
                                          or np.any(value[mask] < 0) for value in metrics.values()):
        raise ValueError('Supply aligned, finite nonnegative feasible low/U/rate/NMSE tables.')
    actor = set(actor_names) & set(policy.params)
    if not actor:
        raise ValueError('No named actor parameters were found.')
    critic = {key: value.copy() for key, value in policy.params.items() if key not in actor}
    clean_q = np.where(mask, q, 0.)
    best = np.argmax(np.where(mask, q, -np.inf), axis=1)
    rng, history, updates, exposure_rows = np.random.default_rng(seed), [], [], []

    def probabilities():
        return _probability(policy.forward(x)[0], mask)

    def record(step, p, old, rms):
        greedy = p.argmax(axis=1)
        entropy = -np.sum(np.where(p > 0, p*np.log(np.maximum(p, np.finfo(float).tiny)), 0.), axis=1)
        kl = np.sum(np.where(mask, old*np.log(np.maximum(old, np.finfo(float).tiny)
                    /np.maximum(p, np.finfo(float).tiny)), 0.), axis=1)
        for context in range(len(x)):
            expected = float(p[context] @ clean_q[context])
            row = dict(step=int(step), context=context, mode=mode, seed=int(seed),
                expected_table_reward=expected, greedy_action=int(greedy[context]),
                best_action=int(best[context]), best_probability=float(p[context, best[context]]),
                greedy_table_reward=float(q[context, greedy[context]]),
                greedy_best_reward_gap=float(q[context, best[context]]-q[context, greedy[context]]),
                expected_reward_regret=float(q[context, best[context]]-expected),
                entropy=float(entropy[context]), exact_kl_to_refresh=max(float(kl[context]), 0.),
                advantage_rms_before_normalization=float(rms))
            for name, table in metrics.items():
                row['greedy_table_'+name] = float(table[context, greedy[context]])
                row['expected_table_'+name] = float(p[context] @ np.where(mask[context], table[context], 0.))
            history.append(row)

    p, steps, blocks = probabilities(), 0, 0
    record(0, p, p, 0.)
    while steps < max_steps:
        blocks += 1
        old = probabilities()
        advantage, rms = conditional_advantages(q, old, mask, reward_scale)
        weights = old if mode == 'full' else sampled_action_weights(old, samples_per_context, rng)
        for context in range(len(x)):
            for action in range(q.shape[1]):
                exposure_rows.append(dict(block=blocks, context=context, action=action,
                    mode=mode, seed=int(seed), optimizer_step_before_refresh=steps,
                    feasible=bool(mask[context, action]),
                    old_probability=float(old[context, action]),
                    action_weight=float(weights[context, action]),
                    actual_sample_count=(int(round(weights[context, action]*samples_per_context))
                                         if mode == 'sampled' else None),
                    integration_kind=('sampled_categorical' if mode == 'sampled'
                                      else 'exact_policy_expectation'),
                    q_value=float(q[context, action]) if mask[context, action] else None,
                    normalized_advantage=float(advantage[context, action]),
                    best_action=bool(action == best[context])))
        for epoch in range(refresh_epochs):
            p = probabilities()
            _, logits_gradient, info = categorical_surrogate(p, old, advantage, mask,
                weights=weights, clip=clip, entropy_coef=entropy_coef)
            if epoch and info['exact_kl'] > target_kl:
                break
            gradient = gradients_fn(policy, x, logits_gradient)
            if set(gradient) != set(policy.params):
                raise ValueError('External gradient function returned different parameter keys.')
            gradient = {key: np.asarray(value).copy() if key in actor else np.zeros_like(value)
                        for key, value in gradient.items()}
            if any(not np.isfinite(value).all() or value.shape != policy.params[key].shape
                   for key, value in gradient.items()):
                raise ValueError('External gradients must be finite and match parameter shapes.')
            optimizer.step(gradient)
            if any(not np.array_equal(policy.params[key], value) for key, value in critic.items()):
                raise AssertionError('Actor-only diagnostic changed a critic parameter.')
            steps += 1
            updates.append(dict(step=steps, optimizer_step_before=steps-1,
                optimizer_step_after=steps, block=blocks, epoch=epoch,
                diagnostics_timing='pre_update', **info))
            record(steps, probabilities(), old, rms)
            if steps >= max_steps:
                break
    return dict(history=history, updates=updates, exposure_rows=exposure_rows,
        final_probabilities=probabilities(),
        steps_completed=steps, refresh_blocks=blocks, critic_unchanged=True,
        critic_parameters_unchanged=True, critic_optimizer_state_checked=False,
        diagnostic_only=True, sequential_PPO=False, deployment_policy=False)
