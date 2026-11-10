"""PPO controller tuned on RADIal: V12.40's actor, critic, state, 21 commands, twelve rounds, start and codec, unchanged.

Three things are tuned. The commands act inside global bounds, the same for every tile. The rate prices span the rates
to be reached. The training reward uses the receiver that scores the results: each tile answers for the labels, the robust
detections and the new peaks in it, at the detector's own threshold alpha. The reward exists only during training: at
run time the policy sees exactly the V12.40 state, and nothing returns from the decoder. No work runs on import.
"""
from __future__ import annotations

import torch

from . import radial_cap_codec as cap
from . import radial_cap_learning as v12

v13 = cap.v13
ROUNDS, N_ACTIONS, NAMES, INITIAL_RHO, INITIAL_Q = v12.ROUNDS, v12.N_ACTIONS, v12.NAMES, v12.INITIAL_RHO, v12.INITIAL_Q
raw_states, fit_features, compatible_mask = v12.raw_states, v12.fit_features, v12.compatible_mask
DeltaPolicy, update, normalize = v12.DeltaPolicy, v12.update, v12.normalize
PRICES = (0., .025, .05, .1, .2, .3)
RHO_RANGE, Q_RANGE = (.25, 1.), (4., 7.)   # the same 21 commands, inside global bounds
LABEL_MISS_WEIGHT, ROBUST_LOST_WEIGHT, ROBUST_NEW_WEIGHT, WEAK_NEW_WEIGHT = 8., 1., 2., 1.
RATE_WEIGHT, NMSE_WEIGHT = 10., .5
MARGIN_DB = 1.


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


@torch.no_grad()
def tile_outcomes(original, method, bins, alpha, gate):
    """Per-tile counts of the scoring receiver: the definitions of v13.detection_metrics and v13.label_status at alpha."""
    n = cap.DOPPLER_TILES*cap.RANGE_TILES
    margin = 10**(MARGIN_DB/10)
    device = original['score'].device
    zeros = torch.zeros(n, dtype=torch.float64, device=device)

    def count(mask):
        rows, columns = torch.nonzero(mask, as_tuple=True)
        return torch.bincount(cap.tile_of(rows, columns), minlength=n).to(torch.float64) if len(rows) else zeros.clone()

    s0, s1 = v13.peak_scores(original), v13.peak_scores(method)
    near0 = v13.neighbourhood_max(s0, v13.NMS_RADIUS, v13.NMS_RADIUS)
    near1 = v13.neighbourhood_max(s1, v13.NMS_RADIUS, v13.NMS_RADIUS)
    robust_lost = count(v13.stable_peaks(original, alpha, margin) & (near1 <= alpha))
    robust_new = count(v13.stable_peaks(method, alpha, margin) & (near0 <= alpha/margin))
    new = count((s1 > alpha) & (near0 <= alpha))
    miss, score_loss = zeros.clone(), zeros.clone()
    if len(bins):
        status = v13.label_status(original, method, bins, alpha, *gate, MARGIN_DB)
        g0, g1 = v13.label_scores(original, method, bins, *gate)
        tiles = cap.tile_of(bins[:, 0], bins[:, 1])
        miss.scatter_add_(0, tiles, (status['detected'] & ~status['preserved']).to(torch.float64))
        loss = (10*torch.log10(g0.clamp_min(1e-3)/g1.clamp_min(1e-3))).clamp(0, 20)*status['detected']
        score_loss.scatter_add_(0, tiles, loss)
    return dict(miss=miss, score_loss_db=score_loss, robust_lost=robust_lost, robust_new=robust_new, new=new, weak_new=(new-robust_new).clamp_min(0))


def reward_components(tile, meta, price, weak_scale, cube_energy):
    """Per-tile return: labels and robust detections first, then new peaks, then rate."""
    bits, errors = meta['tile_bits'].to(torch.float64), meta['tile_error'].to(torch.float64)
    n = len(bits)
    share = cap.CAP_BPCS*float(meta['numel'])/n
    terms = dict(label_miss=-LABEL_MISS_WEIGHT*tile['miss'], global_miss=-LABEL_MISS_WEIGHT*tile['miss'].sum().expand(n)/n, score_loss=-tile['score_loss_db']/3.,
                 robust_lost=-ROBUST_LOST_WEIGHT*tile['robust_lost'], robust_new=-ROBUST_NEW_WEIGHT*tile['robust_new'], weak_new=-WEAK_NEW_WEIGHT*tile['weak_new']/max(weak_scale, 1.),
                 rate=-RATE_WEIGHT*float(price)*bits/share, nmse=-NMSE_WEIGHT*errors*n/cube_energy.clamp_min(1e-12))
    reward = sum(terms.values())
    if not bool(torch.isfinite(reward).all()):
        raise FloatingPointError('Nonfinite physical reward component.')
    return reward, {f'{key}_penalty': float(value.sum()) for key, value in terms.items()}
