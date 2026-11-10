"""Fresh on-policy conditional-action credit at selected primary prefixes.

No detector, codec, GT, reward design, prefix selection or file I/O is here.
The caller must sample each branch from the exact current primary history,
force only its current feasible action, then draw a fresh behavior-policy
suffix with recomputed states/masks. Old fixed-continuation tables are NOT
Q^pi. Arrays cannot certify this provenance or selection before current action.

Each selected context REPLACES one sampled actor row by its old-policy action
expectation; it is never appended as16 equally weighted states. Ordinary rows
retain the original same-cube LOO residual credit. Critic MC targets use only
primary rollouts. Finite noisy branches, global RMS and clipped PPO do not
guarantee improvement, target preservation or an unbiased complete algorithm.
"""
from __future__ import annotations

import numpy as np

from radar_sim.ppo_advantage_diagnostics import advantage_snapshot
from radar_sim.ppo_conditional_action_learning import categorical_surrogate


def actor_credit_table(trajectories, old_probability, conditional_contexts, reward_scale):
    """Return frozen action weights/credits with equal weight per primary state."""
    trajectories = list(trajectories)
    snapshot = advantage_snapshot(trajectories, reward_scale)
    raw = snapshot['advantage_raw'].copy()
    actions = np.concatenate([np.asarray(item['actions']) for item in trajectories])
    masks = np.concatenate([np.asarray(item['action_masks']) for item in trajectories])
    probability = np.asarray(old_probability, np.float64)
    if (actions.shape != raw.shape or not np.issubdtype(actions.dtype, np.integer)
            or masks.dtype != np.bool_ or probability.shape != masks.shape
            or not np.isfinite(probability).all() or np.any(probability < 0)
            or np.any(probability[~masks] != 0) or not np.allclose(probability.sum(1), 1.)
            or np.any(actions < 0) or np.any(actions >= masks.shape[1])
            or not masks[np.arange(len(actions)), actions].all()):
        raise ValueError('Malformed primary actions/masks or behavior probabilities.')
    probability = probability/probability.sum(axis=1, keepdims=True)
    if np.any(probability[masks] <= 0):
        raise ValueError('Every feasible behavior action needs positive probability.')
    weights, credits = np.zeros_like(probability), np.zeros_like(probability)
    weights[np.arange(len(actions)), actions] = 1.
    credits[np.arange(len(actions)), actions] = raw
    selected, q_tables = set(), {}
    for packet in conditional_contexts:
        index = packet['primary_row']
        if (isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer))
                or not 0 <= index < len(actions) or index in selected):
            raise ValueError('Conditional contexts need distinct valid primary-row indices.')
        index = int(index)
        q = np.asarray(packet['q_values_unscaled'], np.float64)
        old = np.asarray(packet['old_probabilities'], np.float64)
        if (q.shape != (masks.shape[1],) or not np.isfinite(q[masks[index]]).all()
                or old.shape != q.shape or not np.isfinite(old).all()
                or not np.allclose(old, probability[index], rtol=1e-5, atol=2e-7)):
            raise ValueError('Conditional values/masks must match the fresh behavior prefix.')
        q = np.where(masks[index], q, 0.)
        scaled = q/reward_scale
        credits[index] = np.where(masks[index], scaled-probability[index] @ scaled, 0.)
        weights[index] = probability[index]
        selected.add(index)
        q_tables[index] = q
    choice = masks.sum(axis=1) > 1
    credits[~choice] = 0.
    per_state_second_moment = np.sum(weights*credits**2, axis=1)
    rms = float(np.sqrt(per_state_second_moment[choice].mean())) if choice.any() else 0.
    normalized = credits/max(rms, 1e-6)
    rows, action_rows, offset = [], [], 0
    for trajectory in trajectories:
        for tile, action in enumerate(trajectory['actions']):
            index = offset+tile
            rows.append(dict(primary_row=index, record_name=str(trajectory.get('record_name', '')),
                rollout_index=int(trajectory['rollout_index']), tile_index=tile, sampled_action=int(action),
                conditional=index in selected, choice=bool(choice[index]),
                original_raw_credit=float(raw[index]), current_sampled_raw_credit=float(credits[index, action]),
                current_sampled_normalized_credit=float(normalized[index, action]),
                sampled_action_weight=float(weights[index, action]),
                state_expected_squared_credit=float(per_state_second_moment[index])))
        offset += len(trajectory['actions'])
    for index in sorted(selected):
        for action in range(masks.shape[1]):
            action_rows.append(dict(primary_row=index, action=action, feasible=bool(masks[index, action]),
                old_probability=float(probability[index, action]), actor_weight=float(weights[index, action]),
                q_value_unscaled=float(q_tables[index][action]) if masks[index, action] else None,
                raw_credit=float(credits[index, action]), normalized_credit=float(normalized[index, action])))
    return dict(weights=weights, raw_credits=credits, advantages=normalized, choice=choice,
        rms=rms, primary_raw_rms=snapshot['mc_advantage_rms'], targets=snapshot['returns_scaled'],
        old_values=snapshot['old_values'], credit_rows=rows, conditional_action_rows=action_rows,
        selected=selected, conditional_context_count=len(selected),
        conditional_choice_rows=sum(bool(choice[index]) for index in selected))


def update_ppo_v12_21(policy, optimizer, trajectories, conditional_contexts, *,
                     epochs, clip, entropy_coef, target_kl, value_coef, reward_scale,
                     gradients_fn, mask_fn, seed, eps, batch_size=256):
    """One fresh behavior-policy PPO batch; no stale-Q replay or bandit refresh.

The external network-gradient callback averages its minibatch. Critic targets
and value gradients retain all primary rows; no branch is a critic trajectory.
Conditional tables and one common RMS stay frozen for the limited PPO epochs.
"""
    settings = np.asarray([clip, entropy_coef, target_kl, value_coef, reward_scale, eps], float)
    if (not np.isfinite(settings).all() or not 0 < clip < 1 or entropy_coef < 0
            or target_kl <= 0 or value_coef < 0 or reward_scale <= 0 or eps <= 0
            or isinstance(epochs, bool) or not isinstance(epochs, (int, np.integer)) or epochs < 1
            or isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1):
        raise ValueError('Invalid on-policy PPO settings.')
    trajectories = list(trajectories)
    if not trajectories:
        raise ValueError('At least two independent same-cube primary rollouts are required.')
    x = np.concatenate([np.asarray(item['states'], np.float32) for item in trajectories])
    masks = np.concatenate([np.asarray(item['action_masks']) for item in trajectories])
    actions = np.concatenate([np.asarray(item['actions']) for item in trajectories])
    old_log = np.concatenate([np.asarray(item['old_log_probs'], float) for item in trajectories])
    old_value = np.concatenate([np.asarray(item['old_values'], float) for item in trajectories])
    if (x.ndim != 2 or not len(x) or not np.isfinite(x).all() or masks.ndim != 2
            or len(masks) != len(x) or masks.dtype != np.bool_ or not masks.any(1).all()
            or actions.shape != (len(x),) or not np.issubdtype(actions.dtype, np.integer)
            or old_log.shape != actions.shape or old_value.shape != actions.shape
            or not np.isfinite(old_log).all() or not np.isfinite(old_value).all()
            or np.any(actions < 0) or np.any(actions >= masks.shape[1])
            or not masks[np.arange(len(x)), actions].all()):
        raise ValueError('Malformed primary rollout states, actions, masks or frozen predictions.')
    old, current_value, _ = policy.forward(x)
    old = np.asarray(mask_fn(old, masks), np.float64)
    if (not np.allclose(np.log(old[np.arange(len(x)), actions]+eps), old_log, atol=2e-4, rtol=0)
            or not np.allclose(np.asarray(current_value), old_value, atol=2e-4, rtol=1e-5)):
        raise AssertionError('Stale/off-policy primary rollouts or critic predictions.')
    credit = actor_credit_table(trajectories, old, list(conditional_contexts), reward_scale)
    old = old/old.sum(1, keepdims=True)
    choice, target = credit['choice'], credit['targets']
    actor_scale = len(x)/max(int(choice.sum()), 1)
    before = int(optimizer.t)
    rng = np.random.default_rng(int(seed)+len(x)+before)
    actor_names = tuple(key for key in ('W1', 'b1', 'Wa', 'ba') if key in policy.params)
    gradient_norms, actor_norms, critic_norms = [], [], []
    completed, early_stop, kl = 0, False, 0.
    for _ in range(epochs):
        order = rng.permutation(len(x))
        for start in range(0, len(x), batch_size):
            index = order[start:start+batch_size]
            probability, value, _ = policy.forward(x[index])
            probability = np.asarray(mask_fn(probability, masks[index]), np.float64)
            _, logits_gradient, _ = categorical_surrogate(probability, old[index],
                credit['advantages'][index], masks[index], weights=credit['weights'][index],
                clip=clip, entropy_coef=entropy_coef)
            logits_gradient[~choice[index]] = 0.
            gradients = gradients_fn(policy, x[index], (actor_scale*logits_gradient).astype(np.float32),
                (value_coef*(np.asarray(value)-target[index])).astype(np.float32))
            if set(gradients) != set(policy.params) or any(
                    value.shape != policy.params[key].shape or not np.isfinite(value).all()
                    for key, value in gradients.items()):
                raise ValueError('Network gradients must be finite and match parameter arrays.')
            norm = lambda keys: float(np.sqrt(sum(np.sum(np.asarray(gradients[key], float)**2) for key in keys)))
            gradient_norms.append(norm(gradients))
            actor_norms.append(norm(actor_names))
            critic_norms.append(norm(key for key in gradients if key not in actor_names))
            optimizer.step(gradients)
        completed += 1
        probability = np.asarray(mask_fn(policy.forward(x)[0], masks), float)
        _, _, info = categorical_surrogate(probability, old, credit['advantages'], masks,
            weights=credit['weights'], clip=clip, entropy_coef=entropy_coef)
        kl = info['exact_kl']
        if kl > target_kl:
            early_stop = True
            break
    probability, value, _ = policy.forward(x)
    probability = np.asarray(mask_fn(probability, masks), float)
    loss, _, info = categorical_surrogate(probability, old, credit['advantages'], masks,
        weights=credit['weights'], clip=clip, entropy_coef=entropy_coef)
    entropy = info['entropy']*actor_scale
    actor_loss = (loss+entropy_coef*info['entropy'])*actor_scale
    value_loss, target_variance = float(np.mean((value-target)**2)), float(np.var(target))
    groups = {}
    for item in trajectories:
        groups.setdefault(item['rollout_group'], []).append(float(np.sum(item['rewards'])))
    returns = list(groups.values())
    ev = lambda prediction: float(1.-np.var(target-prediction)/target_variance) if target_variance > 1e-12 else float('nan')
    return dict(ppo_loss=actor_loss+.5*value_coef*value_loss-entropy_coef*entropy,
        actor_loss=actor_loss, value_loss=value_loss, policy_entropy=entropy,
        approximate_kl=kl, approx_kl=kl, exact_categorical_kl_choice=kl,
        clip_fraction=info['clip_fraction'], choice_fraction=float(choice.mean()),
        forced_action_fraction=float((~choice).mean()), mean_feasible_actions=float(masks.sum(1).mean()),
        critic_explained_variance_before=ev(old_value), critic_explained_variance_after=ev(value),
        gradient_norm_before_clip=float(np.mean(gradient_norms)), ppo_epochs_completed=completed,
        ppo_early_stop=int(early_stop), ppo_steps=len(x), ppo_reward_scale=reward_scale,
        ppo_value_coef=value_coef, optimizer_step_before=before, optimizer_step_after=int(optimizer.t),
        actor_gradient_norm=float(np.mean(actor_norms)), critic_gradient_norm=float(np.mean(critic_norms)),
        mc_advantage_rms=credit['rms'], paired_advantage_rms=credit['rms'],
        primary_raw_advantage_rms=credit['primary_raw_rms'],
        within_cube_return_std=float(np.mean([np.std(group) for group in returns])),
        between_cube_mean_return_std=float(np.std([np.mean(group) for group in returns])),
        zero_contrast_group_fraction=float(np.mean([np.ptp(group) <= 1e-8 for group in returns])),
        rollout_groups_per_update=len(groups), rollouts_per_cube=float(np.mean([len(group) for group in returns])),
        conditional_context_count=credit['conditional_context_count'],
        conditional_choice_rows=credit['conditional_choice_rows'],
        conditional_state_fraction=credit['conditional_context_count']/len(x),
        actor_weighting='one primary state, sampled or exact conditional action expectation',
        advantage_mode='loo_residual_with_fresh_conditional_action_replacement',
        value_mse_before=float(np.mean((target-old_value)**2)), value_mse_after=value_loss,
        credit_rows=credit['credit_rows'], conditional_action_rows=credit['conditional_action_rows'])
