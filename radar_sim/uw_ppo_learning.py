"""PPO controller for UW: V12.40's actor, critic, state, 21 commands, twelve rounds and start, unchanged.

Two things are tuned. The commands act inside narrower global bounds, the same for every tile. The training
reward counts a label only with a robust peak, because weaker labels flip by chance, charges lost robust
detections and new robust peaks to their tile, and prices rate twice as hard. The reward exists only during
training: at run time the policy sees exactly the V12.40 state. No work runs on import.
"""
from __future__ import annotations

import torch

from . import radial_cap_codec as cap
from . import radial_cap_learning as v12

ROUNDS, PRICES, N_ACTIONS, DEFAULT_PRICE = v12.ROUNDS, v12.PRICES, v12.N_ACTIONS, v12.DEFAULT_PRICE
NAMES, INITIAL_RHO, INITIAL_Q = v12.NAMES, v12.INITIAL_RHO, v12.INITIAL_Q
raw_states, fit_features, compatible_mask = v12.raw_states, v12.fit_features, v12.compatible_mask
DeltaPolicy, update, normalize = v12.DeltaPolicy, v12.update, v12.normalize
ROBUST_LOST_WEIGHT, ROBUST_NEW_WEIGHT = 1., 1.
RATE_WEIGHT = 10.
RHO_RANGE, Q_RANGE = (.25, 1.), (4., 7.)   # the same 21 commands, inside narrower global bounds
CFAR_STATE = ('weak_peak_frac', 'cfar_margin', 'bg_spread', 'peak_avail', 'cfar_coverage', 'predicted_margin_db')


def state_variant(fitted, name):
    """The V12.40 state, or the same state without its CFAR-derived fields."""
    if name == 'v12_40':
        return fitted
    keep = [i for i, item in enumerate(fitted['names']) if item not in CFAR_STATE]
    index = torch.tensor(keep, dtype=torch.int64, device=fitted['indices'].device)
    return dict(fitted, indices=fitted['indices'][index], mean=fitted['mean'][index], std=fitted['std'][index], names=[fitted['names'][i] for i in keep])


def alternatives(rho, q):
    """V12.40's commands, clipped to the bounds; a clipped duplicate is masked as in V12.40."""
    r, bits, _ = v12.alternatives(rho, q)
    r, bits = r.clamp(*RHO_RANGE), bits.clamp(*Q_RANGE)
    duplicate = (r[:, :, None] == r[:, None, :]) & (bits[:, :, None] == bits[:, None, :])
    earlier = torch.tril(torch.ones((N_ACTIONS, N_ACTIONS), dtype=torch.bool, device=rho.device), diagonal=-1)
    return r, bits, ~(duplicate & earlier).any(2)


@torch.no_grad()
def rollout(policy, fitted, features, price, *, deterministic=False, rho=None, q=None, start_step=0):
    n = len(features)
    rho = features.new_full((n,), INITIAL_RHO) if rho is None else rho.clone()
    q = features.new_full((n,), INITIAL_Q) if q is None else q.clone()
    rows = torch.arange(n, device=features.device)
    history = []
    for step in range(start_step, ROUNDS):
        states = normalize(raw_states(features, rho, q, ROUNDS-step, price), fitted)
        r, bits, mask = alternatives(rho, q)
        distribution, value = policy(states, mask)
        action = distribution.probs.argmax(1) if deterministic else distribution.sample()
        history.append(dict(states=states, mask=mask, actions=action, log_prob=distribution.log_prob(action), value=value,
                            rho_before=rho, q_before=q, probabilities=distribution.probs))
        rho, q = r[rows, action], bits[rows, action]
    return dict(rho=rho, q=q, history=history, price=float(price))


def reward_components(tile, meta, price, unmatched_scale, cube_energy):
    """Per-tile return: robust labels first, then robust detections and false peaks, then rate."""
    bits, errors = meta['tile_bits'].to(torch.float64), meta['tile_error'].to(torch.float64)
    n = len(bits)
    share = cap.CAP_BPCS*float(meta['numel'])/n
    terms = dict(label_miss=-8.*tile['miss'], global_miss=-8.*tile['miss'].sum().expand(n)/n, score_loss=-tile['score_loss_db']/3.,
                 robust_lost=-ROBUST_LOST_WEIGHT*tile['robust_lost'], robust_new=-ROBUST_NEW_WEIGHT*tile['robust_new'], unmatched=-tile['excess']/max(unmatched_scale, 1.),
                 rate=-RATE_WEIGHT*float(price)*bits/share, nmse=-.5*errors*n/cube_energy.clamp_min(1e-12))
    reward = sum(terms.values())
    if not bool(torch.isfinite(reward).all()):
        raise FloatingPointError('Nonfinite physical reward component.')
    return reward, {f'{key}_penalty': float(value.sum()) for key, value in terms.items()}
