"""Stationary-objective, masked Monte Carlo PPO for the separate V12.18 run.

The frozen behavior critic supplies the state-dependent baseline; an optional
independent same-cube leave-one-out residual supplies an actor-only control
variate. Full undiscounted rewards supply the unchanged critic target; no
bootstrap, imitation, action search,
or detector/label access occurs here. The caller provides its independent actor
and critic network, optimizer, derivatives, and categorical masking function.
"""

from __future__ import annotations

import numpy as np


def remaining_budget_features(source, cube_size, index, running, budget, allowed):
    """Four label-free predecision features, including the current tile.

    Return remaining bits / cap, minimum and maximum future action costs / cap,
    and affordable bits per remaining complex coefficient / configured bpcs.
    Costs respect the static action library, not a new spending requirement.
    Ratios are clipped for finite network observations; feasibility continues
    to be enforced by the encoder's unchanged upper-budget action mask.
    """
    costs = np.asarray(source['action_bits_by_block'], np.float64)
    allowed = np.asarray(allowed, bool)
    if costs.ndim != 2 or allowed.shape != (costs.shape[1],) or not allowed.any():
        raise ValueError('A nonempty action library and matching mask are required.')
    if not np.isfinite(costs).all() or np.any(costs < 0):
        raise ValueError('Action costs must be finite and nonnegative.')
    if int(index) != index or not 0 <= int(index) <= len(costs):
        raise ValueError('The predecision tile index is out of range.')
    values = np.asarray([cube_size, running, budget], np.float64)
    if not np.isfinite(values).all() or cube_size <= 0 or budget <= 0 or running < 0:
        raise ValueError('Cube size, running bits, and budget must be valid.')
    index = int(index)
    if index == len(costs):
        return np.asarray([np.clip((budget-running)/budget, 0., 1.), 0., 0., 0.], np.float32)
    if 'slices' in source:
        shape = tuple(source['cube_shape'])
        slices = source['slices']
        if len(slices) != len(costs):
            raise ValueError('Tile slices and action costs disagree.')
        remaining_coefficients = sum(
            np.prod([len(range(*slc.indices(int(size)))) for slc, size in zip(tile, shape)])
            for tile in slices[index:])
    else:
        blocks = source['prepared_blocks']
        if len(blocks) != len(costs):
            raise ValueError('Prepared tiles and action costs disagree.')
        remaining_coefficients = sum(float(block['total_coeffs']) for block in blocks[index:])
    if not np.isfinite(remaining_coefficients) or remaining_coefficients <= 0:
        raise ValueError('Remaining coefficient count must be finite and positive.')
    residual = max(float(budget-running), 0.)
    future = costs[index:, allowed]
    feature = np.asarray([
        residual/budget,
        future.min(axis=1).sum()/budget,
        future.max(axis=1).sum()/budget,
        (residual/remaining_coefficients)/(budget/cube_size),
    ], np.float64)
    return np.clip(feature, 0., [1., 32., 32., 32.]).astype(np.float32)


def update_mc_ppo(policy, optimizer, trajectories, *, epochs, clip, entropy_coef,
                  target_kl, value_coef, reward_scale, gradients_fn, mask_fn,
                  seed, eps, advantage_mode='loo_residual'):
    """One on-policy clipped update with an optional independent control variate.

    ``old_values`` must be frozen behavior-critic predictions in scaled-return
    units. ``critic`` uses A = G - V_old. ``loo_residual`` subtracts the mean
    G_j - V_old(s_j) of OTHER independent rollouts of the same cube at the same
    tile index. It requires explicit ``rollout_group`` and distinct integer
    ``rollout_index`` identifiers, at least two trajectories per group, and
    identical step counts. Independent sampling is the caller's responsibility;
    identifiers cannot certify independence. The control variate is detached,
    excludes the current rollout, and never changes the critic MC target.

    Advantages are divided once by their global RMS over choice states;
    there is no per-cube standardization. Forced actions
    contribute zero actor/entropy gradient but still train the critic. The
    gradients callback takes (policy, states, grad_logits, grad_values), averaging
    over its batch, as the existing independent V12.16 network callback does.
    """
    if advantage_mode not in ('critic', 'loo_residual'):
        raise ValueError("advantage_mode must be 'critic' or 'loo_residual'.")
    hyperparameters = np.asarray([clip, entropy_coef, target_kl, value_coef, reward_scale, eps], float)
    if (not np.isfinite(hyperparameters).all() or int(epochs) != epochs or epochs < 1
            or not 0 < clip < 1 or entropy_coef < 0 or target_kl <= 0
            or value_coef < 0 or reward_scale <= 0 or eps <= 0):
        raise ValueError('Invalid PPO update hyperparameters.')
    trajectories = list(trajectories)
    if not trajectories:
        raise ValueError('A PPO update needs at least one trajectory.')
    if advantage_mode == 'loo_residual' and len({id(t) for t in trajectories}) != len(trajectories):
        raise ValueError('Leave-one-out groups cannot repeat the same trajectory object.')
    fields = {key: [] for key in ('states', 'actions', 'action_masks', 'old_log_probs', 'old_values')}
    targets, group_returns = [], {}
    group_spans, group_rollout_ids = {}, {}
    step_offset = 0
    for number, trajectory in enumerate(trajectories):
        rewards = np.asarray(trajectory['rewards'], np.float64)
        states = np.asarray(trajectory['states'], np.float32)
        if rewards.ndim != 1 or not len(rewards) or states.ndim != 2 or len(states) != len(rewards):
            raise ValueError('Trajectory states/rewards have inconsistent shapes.')
        if not np.isfinite(rewards).all() or not np.isfinite(states).all():
            raise ValueError('Trajectory states and rewards must be finite.')
        fields['states'].append(states)
        for key in ('actions', 'action_masks', 'old_log_probs', 'old_values'):
            field = np.asarray(trajectory[key])
            if len(field) != len(rewards) or not np.isfinite(field).all():
                raise ValueError('Malformed or nonfinite trajectory field: ' + key)
            fields[key].append(field)
        targets.append(np.cumsum(rewards[::-1], dtype=np.float64)[::-1]/reward_scale)
        key = trajectory.get('rollout_group', ('trajectory', number))
        if advantage_mode == 'loo_residual':
            if 'rollout_group' not in trajectory or 'rollout_index' not in trajectory:
                raise ValueError('Leave-one-out trajectories need rollout_group and rollout_index.')
            rollout_id = trajectory['rollout_index']
            if (isinstance(rollout_id, (bool, np.bool_))
                    or not isinstance(rollout_id, (int, np.integer)) or rollout_id < 0):
                raise ValueError('rollout_index must be a nonnegative integer.')
            seen = group_rollout_ids.setdefault(key, set())
            if rollout_id in seen:
                raise ValueError('Duplicate rollout_index within a leave-one-out group.')
            seen.add(rollout_id)
        group_returns.setdefault(key, []).append(float(rewards.sum()))
        group_spans.setdefault(key, []).append((step_offset, step_offset+len(rewards)))
        step_offset += len(rewards)
    x = np.concatenate(fields['states']).astype(np.float32)
    raw_actions = np.concatenate(fields['actions'])
    action = raw_actions.astype(np.int64)
    mask = np.concatenate(fields['action_masks']).astype(bool)
    old_log = np.concatenate(fields['old_log_probs']).astype(np.float64)
    old_value = np.concatenate(fields['old_values']).astype(np.float64)
    target = np.concatenate(targets).astype(np.float64)
    if (action.shape != (len(x),) or old_log.shape != action.shape
            or old_value.shape != action.shape or not np.array_equal(action, raw_actions)
            or mask.ndim != 2 or len(mask) != len(x) or not mask.any(axis=1).all()
            or np.any(action < 0) or np.any(action >= mask.shape[1])):
        raise ValueError('Malformed actions, masks, values, or log probabilities.')
    if not np.all(mask[np.arange(len(action)), action]):
        raise AssertionError('Rollout contains an invalid action.')
    old_p, current_value, _ = policy.forward(x)
    old_p = np.asarray(mask_fn(old_p, mask), np.float64)
    if old_p.shape != mask.shape or not np.isfinite(old_p).all():
        raise ValueError('Masked policy probabilities must match the action masks.')
    if not np.allclose(np.log(old_p[np.arange(len(action)), action]+eps), old_log,
                       atol=2e-4, rtol=0):
        raise AssertionError('Stale/off-policy rollouts passed to PPO.')
    if not np.allclose(np.asarray(current_value), old_value, atol=2e-4, rtol=1e-5):
        raise AssertionError('Stale behavior critic values passed to PPO.')
    choice = mask.sum(axis=1) > 1
    residual = target-old_value
    residual_rms = float(np.sqrt(np.mean(residual[choice]**2))) if choice.any() else 0.
    control_variate = np.zeros_like(residual)
    if advantage_mode == 'loo_residual':
        for spans in group_spans.values():
            if len(spans) < 2:
                raise ValueError('Every leave-one-out group needs at least two independent rollouts.')
            if len({end-start for start, end in spans}) != 1:
                raise ValueError('Leave-one-out rollouts must have identical step counts.')
            members = np.stack([residual[start:end] for start, end in spans])
            other_means = (members.sum(axis=0)[None, :]-members)/(len(spans)-1)
            for (start, end), baseline in zip(spans, other_means):
                control_variate[start:end] = baseline
    advantage = residual-control_variate
    advantage_rms = float(np.sqrt(np.mean(advantage[choice]**2))) if choice.any() else 0.
    advantage = advantage/max(advantage_rms, 1e-6)
    advantage[~choice] = 0.
    before = int(optimizer.t)
    rng = np.random.default_rng(int(seed)+len(x)+before)
    gradient_norms, actor_norms, critic_norms = [], [], []
    completed, early_stop = 0, False
    choice_kl, kl_rows = 0., np.zeros(len(x), float)
    for _ in range(int(epochs)):
        order = rng.permutation(len(x))
        for start in range(0, len(x), 256):
            idx = order[start:start+256]
            probability, value, _ = policy.forward(x[idx])
            probability = np.asarray(mask_fn(probability, mask[idx]), np.float64)
            ratio = np.exp(np.log(probability[np.arange(len(idx)), action[idx]]+eps)-old_log[idx])
            active = np.where(advantage[idx] >= 0., ratio <= 1.+clip, ratio >= 1.-clip)
            coefficient = np.where(active, ratio*advantage[idx], 0.)
            grad_logits = probability.copy()
            grad_logits[np.arange(len(idx)), action[idx]] -= 1.
            grad_logits *= coefficient[:, None]
            entropy = -(probability*np.log(probability+eps)).sum(axis=1)
            grad_logits += entropy_coef*probability*(np.log(probability+eps)+entropy[:, None])
            grad_logits[~choice[idx]] = 0.
            grad_logits *= len(x)/max(int(choice.sum()), 1)
            gradients = gradients_fn(policy, x[idx], grad_logits.astype(np.float32),
                                     (value_coef*(np.asarray(value)-target[idx])).astype(np.float32))
            if not all(np.isfinite(gradient).all() for gradient in gradients.values()):
                raise FloatingPointError('PPO produced a nonfinite network gradient.')
            norm = lambda names: float(np.sqrt(sum(np.sum(np.asarray(gradients[k], float)**2) for k in names)))
            gradient_norms.append(norm(gradients))
            actor_norms.append(norm(optimizer.actor_names))
            critic_norms.append(norm(optimizer.critic_names))
            optimizer.step(gradients)
        completed += 1
        new_p = np.asarray(mask_fn(policy.forward(x)[0], mask), np.float64)
        kl_rows = np.sum(old_p*(np.log(old_p+eps)-np.log(new_p+eps)), axis=1)
        choice_kl = max(float(kl_rows[choice].mean()), 0.) if choice.any() else 0.
        if choice_kl > target_kl:
            early_stop = True
            break
    probability, value, _ = policy.forward(x)
    probability = np.asarray(mask_fn(probability, mask), np.float64)
    ratio = np.exp(np.log(probability[np.arange(len(x)), action]+eps)-old_log)
    entropy = -(probability*np.log(probability+eps)).sum(axis=1)
    actor_loss = -float(np.minimum(ratio[choice]*advantage[choice],
                        np.clip(ratio[choice], 1.-clip, 1.+clip)*advantage[choice]).mean()) if choice.any() else 0.
    value_loss = float(np.mean((np.asarray(value)-target)**2))
    selected_entropy = float(entropy[choice].mean()) if choice.any() else 0.
    target_variance = float(np.var(target))
    groups = list(group_returns.values())
    cube_nmse = [float(t['meta']['cube_nmse']) for t in trajectories if 'cube_nmse' in t.get('meta', {})]
    return dict(
        ppo_loss=actor_loss+.5*value_coef*value_loss-entropy_coef*selected_entropy,
        actor_loss=actor_loss, value_loss=value_loss, policy_entropy=selected_entropy,
        approximate_kl=choice_kl, approx_kl=choice_kl, exact_categorical_kl_choice=choice_kl,
        exact_categorical_kl_all=max(float(kl_rows.mean()), 0.),
        clip_fraction=float(np.mean(np.abs(ratio[choice]-1.) > clip)) if choice.any() else 0.,
        choice_fraction=float(choice.mean()), forced_action_fraction=float((~choice).mean()),
        mean_feasible_actions=float(mask.sum(axis=1).mean()),
        critic_explained_variance_before=float(1.-np.var(target-old_value)/target_variance) if target_variance > 1e-12 else float('nan'),
        critic_explained_variance_after=float(1.-np.var(target-value)/target_variance) if target_variance > 1e-12 else float('nan'),
        gradient_norm_before_clip=float(np.mean(gradient_norms)), ppo_epochs_completed=completed,
        ppo_early_stop=int(early_stop), ppo_steps=len(x), ppo_reward_scale=float(reward_scale),
        ppo_value_coef=float(value_coef), gae_lambda=1., gae_used=False,
        optimizer_step_before=before, optimizer_step_after=int(optimizer.t),
        actor_gradient_norm=float(np.mean(actor_norms)), critic_gradient_norm=float(np.mean(critic_norms)),
        critic_target_variance=target_variance, mean_cube_nmse=float(np.mean(cube_nmse)) if cube_nmse else float('nan'),
        advantage_estimator=('undiscounted Monte Carlo minus frozen state value and independent '
            'same-cube leave-one-out residual; no bootstrap' if advantage_mode == 'loo_residual'
            else 'undiscounted Monte Carlo minus frozen behavior state value; no bootstrap'),
        mc_advantage_rms=advantage_rms, paired_advantage_rms=advantage_rms,
        mc_residual_rms_before_control_variate=residual_rms,
        mc_advantage_rms_after_control_variate=advantage_rms,
        control_variate_rms=float(np.sqrt(np.mean(control_variate[choice]**2))) if choice.any() else 0.,
        advantage_mode=advantage_mode,
        control_variate_groups=len(group_spans) if advantage_mode == 'loo_residual' else 0,
        value_mse_before=float(np.mean((target-old_value)**2)), value_mse_after=value_loss,
        within_cube_return_std=float(np.mean([np.std(g) for g in groups])),
        between_cube_mean_return_std=float(np.std([np.mean(g) for g in groups])),
        zero_contrast_group_fraction=float(np.mean([np.ptp(g) <= 1e-8 for g in groups])),
        rollout_groups_per_update=len(groups), rollouts_per_cube=float(np.mean([len(g) for g in groups])),
    )

