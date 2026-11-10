"""Pure replay contracts for independent same-prefix auxiliary suffixes.

No codec, radar pipeline, fitting or optimization is imported here. PrefixPolicy
is an ephemeral facade for one unchanged encoder call, not a deployed policy.
Only earlier actions are replayed. The current and every later action use the
frozen policy and a distinct auxiliary RNG with states/masks rebuilt by encoder.
"""
from __future__ import annotations

import numpy as np


class PrefixPolicy:
    def __init__(self, frozen_policy, saved_actions, prefix, *, seed):
        actions = np.asarray(saved_actions)
        allowed = np.asarray(frozen_policy.allowed_action_mask, bool)
        if (actions.ndim != 1 or not len(actions)
                or not np.issubdtype(actions.dtype, np.integer)
                or int(prefix) != prefix or not 0 <= int(prefix) < len(actions)
                or allowed.ndim != 1 or not allowed.all()
                or np.any(actions < 0) or np.any(actions >= len(allowed))):
            raise ValueError('Require a valid saved allocation, prefix and full action library.')
        self.frozen_policy = frozen_policy
        self.saved_actions, self.prefix = actions.copy(), int(prefix)
        self.allowed_action_mask = allowed.copy()
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.first_suffix_probabilities = None

    def forward(self, states):
        if self.steps >= len(self.saved_actions):
            raise RuntimeError('One PrefixPolicy facade may be used for only one encoder call.')
        states = np.asarray(states)
        if states.ndim != 2 or len(states) != 1:
            raise ValueError('The sequential encoder must request exactly one state per call.')
        probability, value, hidden = self.frozen_policy.forward(states)
        probability = np.asarray(probability).copy()
        if probability.shape != (1, len(self.allowed_action_mask)):
            raise ValueError('Frozen actor output does not match the action library.')
        if self.steps < self.prefix:
            probability.fill(0.)
            probability[0, int(self.saved_actions[self.steps])] = 1.
        elif self.steps == self.prefix:
            self.first_suffix_probabilities = probability[0].copy()
        self.steps += 1
        return probability, value, hidden


def rewards_from_scaled_returns(returns_scaled, reward_scale):
    value = np.asarray(returns_scaled, np.float64)
    if (value.ndim != 1 or not len(value) or not np.isfinite(value).all()
            or not np.isfinite(reward_scale) or reward_scale <= 0.):
        raise ValueError('Finite nonempty scaled returns and positive reward scale required.')
    return np.diff(np.r_[value, 0.])*(-float(reward_scale))


def suffix_return_from_total(total_reward, dense_rewards, prefix):
    dense = np.asarray(dense_rewards, np.float64)
    if (dense.ndim != 1 or not len(dense) or not np.isfinite(dense).all()
            or not np.isfinite(total_reward) or int(prefix) != prefix
            or not 0 <= int(prefix) < len(dense)):
        raise ValueError('Require a finite reward, dense vector and valid predecision prefix.')
    return float(total_reward)-float(dense[:int(prefix)].sum())


def trajectory_from_snapshot(snapshot, record, number, *, steps=81,
                             reward_scale=5000., nmse_weight=5000., eps=1e-12):
    """Reconstruct an archived primary sample; never create auxiliary PPO rows."""
    if int(number) != number or number < 0 or int(steps) != steps or steps < 1:
        raise ValueError('Nonnegative trajectory index and positive step count required.')
    span = slice(int(number)*int(steps), (int(number)+1)*int(steps))
    required = ('states', 'masks', 'actions', 'old_probabilities', 'old_values', 'returns_scaled',
                'component_returns_scaled__cost_cube_nmse')
    if any(len(snapshot[name][span]) != steps for name in required):
        raise ValueError('Saved replay does not contain a complete trajectory.')
    states = np.asarray(snapshot['states'][span], np.float32).copy()
    masks = np.asarray(snapshot['masks'][span], bool).copy()
    actions = np.asarray(snapshot['actions'][span], np.int64).copy()
    probabilities = np.asarray(snapshot['old_probabilities'][span], np.float64)
    if (states.ndim != 2 or masks.ndim != 2 or probabilities.shape != masks.shape
            or len(states) != steps or not np.isfinite(states).all()
            or not np.isfinite(probabilities).all() or np.any(probabilities < 0.)
            or np.any(actions < 0) or np.any(actions >= masks.shape[1])
            or not np.all(masks[np.arange(steps), actions])):
        raise ValueError('Malformed saved states, masks, actions or probabilities.')
    selected = probabilities[np.arange(steps), actions]
    if np.any(selected <= 0.):
        raise ValueError('Primary sampled actions require positive behavior probability.')
    rewards = rewards_from_scaled_returns(snapshot['returns_scaled'][span], reward_scale)
    dense = rewards_from_scaled_returns(snapshot['component_returns_scaled__cost_cube_nmse'][span], reward_scale)
    if not np.isfinite(nmse_weight) or nmse_weight <= 0.:
        raise ValueError('Positive source NMSE weight required.')
    if not np.isclose(rewards.sum(), float(record['episode_return']), rtol=1e-12, atol=1e-7):
        raise AssertionError('Reconstructed reward does not match the archived primary sample.')
    return dict(states=states, actions=actions, action_masks=masks,
        old_log_probs=np.log(selected+eps).astype(np.float32),
        old_values=np.asarray(snapshot['old_values'][span], np.float32).copy(),
        rewards=rewards, distortion_contributions=-dense/float(nmse_weight),
        rollout_group=str(record['rollout_group']), rollout_index=int(record['rollout_index']),
        record_name=str(record['record_name']),
        meta=dict(cube_nmse=float(record['cube_nmse']), realized_bpcs=float(record['realized_bpcs'])))


def assert_prefix_matches(primary, auxiliary, prefix, action_costs):
    """Check replayed prefix actions and predecision states/masks through t."""
    t = int(prefix)
    if t != prefix or not 0 <= t < len(primary['actions']):
        raise ValueError('Prefix outside the primary allocation.')
    if not np.array_equal(primary['actions'][:t], auxiliary['actions'][:t]):
        raise AssertionError('An earlier replayed action changed.')
    if not np.array_equal(primary['states'][:t+1], auxiliary['states'][:t+1]):
        raise AssertionError('Same-prefix encoder states changed before the fresh action.')
    if not np.array_equal(primary['action_masks'][:t+1], auxiliary['action_masks'][:t+1]):
        raise AssertionError('Same-prefix budget masks changed before the fresh action.')
    costs = np.asarray(action_costs, np.float64)
    if costs.ndim != 2 or len(costs) != len(primary['actions']):
        raise ValueError('Cost matrix must match the full allocation.')
    expected = costs[np.arange(t), primary['actions'][:t]]
    actual = np.asarray(auxiliary['block_log'].realized_bits, np.float64)[:t]
    if not np.array_equal(expected, actual):
        raise AssertionError('Prefix analytical codec costs changed.')

