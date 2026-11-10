"""Read-only initial actor-credit snapshot of the original loo_residual updater.

No actor/critic update or production change occurs. Optional component_rewards
maps each component name to a list of reward vectors in trajectory order; the
components must sum to the complete reward at every step. Their return-only
LOO terms plus the separately exported critic LOO term reconstruct actor credit.
"""
from __future__ import annotations

import numpy as np


def advantage_snapshot(trajectories, reward_scale=5000., component_rewards=None):
    """Reproduce original MC/frozen-value/LOO/global-choice-RMS credit exactly.

    Returned arrays flatten trajectories in supplied order, then tile order.
    ``advantage_raw`` is before forced-state zeroing, as in the actual updater;
    ``advantage_normalized`` is the actor-used array, zero at forced states.
    Supplied trajectories must be independent on-policy same-cube samples. IDs
    can detect duplicate members but cannot establish sampling independence.
    """
    if not np.isfinite(reward_scale) or reward_scale <= 0.:
        raise ValueError('reward_scale must be finite and positive.')
    trajectories = list(trajectories)
    if not trajectories or len({id(t) for t in trajectories}) != len(trajectories):
        raise ValueError('Provide distinct nonempty trajectory objects.')
    returns, values, choices, groups, ids, spans = [], [], [], {}, {}, []
    offset = 0
    for trajectory in trajectories:
        reward = np.asarray(trajectory['rewards'], np.float64)
        value = np.asarray(trajectory['old_values'], np.float64)
        mask = np.asarray(trajectory['action_masks'])
        if (reward.ndim != 1 or not len(reward) or value.shape != reward.shape
                or not np.isfinite(reward).all() or not np.isfinite(value).all()
                or mask.ndim != 2 or len(mask) != len(reward)
                or not np.isin(mask, [False, True]).all() or not mask.any(axis=1).all()):
            raise ValueError('Malformed rewards, frozen values, or action masks.')
        key, member = trajectory['rollout_group'], trajectory['rollout_index']
        if (isinstance(member, (bool, np.bool_)) or not isinstance(member, (int, np.integer))
                or member < 0 or member in ids.setdefault(key, set())):
            raise ValueError('Each group needs distinct nonnegative integer rollout IDs.')
        ids[key].add(member)
        span = (offset, offset+len(reward))
        groups.setdefault(key, []).append(span)
        spans.append(span)
        offset += len(reward)
        returns.append(np.cumsum(reward[::-1], dtype=np.float64)[::-1]/reward_scale)
        values.append(value)
        choices.append(mask.sum(axis=1) > 1)
    for members in groups.values():
        if len(members) < 2 or len({b-a for a, b in members}) != 1:
            raise ValueError('Each LOO group needs >=2 equal-length trajectories.')

    def others(array):
        output = np.zeros_like(array)
        for members in groups.values():
            stacked = np.stack([array[a:b] for a, b in members])
            means = (stacked.sum(axis=0)[None, :]-stacked)/(len(members)-1)
            for (a, b), mean in zip(members, means):
                output[a:b] = mean
        return output

    target, old_value, choice = np.concatenate(returns), np.concatenate(values), np.concatenate(choices)
    residual = target-old_value
    control = others(residual)
    raw = residual-control
    rms = lambda array: float(np.sqrt(np.mean(array[choice]**2))) if choice.any() else 0.
    divisor = max(rms(raw), 1e-6)
    normalize = lambda array: np.where(choice, array/divisor, 0.)
    critic = -old_value-others(-old_value)
    result = dict(returns_scaled=target, old_values=old_value, residual=residual,
        control_variate=control, advantage_raw=raw, advantage_normalized=normalize(raw),
        choice=choice, return_loo_raw=target-others(target), critic_loo_raw=critic,
        critic_loo_normalized=normalize(critic), mc_advantage_rms=rms(raw),
        control_variate_rms=rms(control), mc_residual_rms_before_control_variate=rms(residual))
    if component_rewards is not None:
        reconstructed = np.zeros_like(target)
        component_returns, component_loo = {}, {}
        for name, rewards in component_rewards.items():
            if len(rewards) != len(trajectories):
                raise ValueError('Component vectors must match trajectory order/count.')
            targets = []
            for sequence, (a, b) in zip(rewards, spans):
                sequence = np.asarray(sequence, np.float64)
                if sequence.shape != (b-a,) or not np.isfinite(sequence).all():
                    raise ValueError('Malformed component reward vector.')
                targets.append(np.cumsum(sequence[::-1], dtype=np.float64)[::-1]/reward_scale)
            component_returns[name] = np.concatenate(targets)
            component_loo[name] = component_returns[name]-others(component_returns[name])
            reconstructed += component_returns[name]
        if not np.allclose(reconstructed, target, rtol=1e-12, atol=1e-8):
            raise AssertionError('Reward components do not sum to complete MC returns.')
        result.update(component_returns_scaled=component_returns, component_loo_raw=component_loo,
            component_loo_normalized={name: normalize(array) for name, array in component_loo.items()})
    return result
