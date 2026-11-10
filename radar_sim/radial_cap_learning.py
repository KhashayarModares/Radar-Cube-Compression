"""V12.40 controller: the same PPO, actor, critic, 21 commands and twelve rounds, with a per-tile reward.

V12.38 gave all 768 decisions of a cube one shared return, and only 5% of the rollouts saw a label miss, so the policy moved
by a KL of 0.0005 per update. Here each tile earns the outcome it controls, and a label loses reward as soon as its CFAR score
falls, before it is missed. Coding starts at full support, which already keeps the labels, and the policy learns to save
bits from there. No work runs on import.
"""
from __future__ import annotations

import torch

from . import arm_e_adaptive_gpu_learning as arm
from . import radial_cap_codec as cap
from . import radial_preserve_gpu_learning as base

ROUNDS, PRICES, N_ACTIONS = base.ROUNDS, base.PRICES, base.N_ACTIONS
DEFAULT_PRICE = .2     # rate saving is a first priority
INITIAL_RHO, INITIAL_Q = 1., 5.
LEARNING_RATE = 1e-3
EXTRA_NAMES = ('predicted_margin_db', 'predicted_snr_loss_db')
NAMES = base.NAMES+EXTRA_NAMES
commands, alternatives, compatible_mask = base.commands, base.alternatives, base.compatible_mask
DeltaPolicy, update, normalize = base.DeltaPolicy, base.update, base.normalize
MARGIN = base.STATIC_NAMES.index('cfar_margin')


def raw_states(features, rho, q, remaining, price):
    """The original state plus the SNR loss of the current step and the CFAR margin left after it."""
    loss = cap.snr_loss_db(q)
    return torch.cat((base.raw_states(features, rho, q, remaining, price), (features[:, MARGIN]-loss)[:, None], loss[:, None]), dim=1)


def fit_features(matrix, threshold=.98):
    if matrix.ndim != 2 or matrix.shape[1] != len(NAMES) or not bool(torch.isfinite(matrix).all()):
        raise ValueError('Finite TRAIN-only observation matrix required.')
    pearson, spearman = base.correlation(matrix), base.correlation(matrix, rank=True)
    mean, std = matrix.mean(0), matrix.std(0, unbiased=False)
    protected = (set(arm.PROTECTED)-{'cfar_coverage'}) | set(EXTRA_NAMES)
    kept, rows = [], []
    for i, name in enumerate(NAMES):
        retain, reason = True, 'retained: no strong TRAIN redundancy'
        if name == 'future_energy_fraction':
            retain, reason = False, 'removed: canonical-order suffix energy is not control state in parallel editing'
        elif float(std[i]) < 1e-8:
            retain, reason = False, 'constant on this TRAIN audit'
        elif name in protected:
            reason = 'protected local allocation, position or detector context'
        else:
            for j in kept:
                if abs(float(pearson[i, j])) >= threshold and abs(float(spearman[i, j])) >= threshold:
                    retain, reason = False, 'TRAIN Pearson and Spearman >= .98 with '+NAMES[j]
                    break
        if retain:
            kept.append(i)
        rows.append(dict(feature=name, keep=retain, reason=reason))
    coverage = matrix[:, base.STATIC_NAMES.index('cfar_coverage')] > 0
    for name in ('weak_peak_frac', 'cfar_margin', 'bg_spread', 'peak_avail'):
        j = NAMES.index(name)
        mean[j] = matrix[coverage, j].mean() if bool(coverage.any()) else 0.
        std[j] = matrix[coverage, j].std(unbiased=False) if bool(coverage.any()) else 1.
    indices = torch.tensor(kept, dtype=torch.int64, device=matrix.device)
    return dict(indices=indices, mean=mean[indices], std=std[indices].clamp_min(1e-6), pearson=pearson, spearman=spearman,
                rows=rows, names=[NAMES[i] for i in kept])


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


def reward_components(tile, meta, metric, price, unmatched_scale, cube_energy):
    """Per-tile return. Labels first, then rate; a rate term of 1 is a tile spending its whole share of the cap at price .2."""
    bits, errors = meta['tile_bits'].to(torch.float64), meta['tile_error'].to(torch.float64)
    n = len(bits)
    share = cap.CAP_BPCS*float(meta['numel'])/n
    terms = dict(label_miss=-8.*tile['miss'], global_miss=-8.*metric['low_miss'].to(torch.float64).expand(n)/n,
                 score_loss=-tile['score_loss_db']/3., unmatched=-tile['excess']/max(unmatched_scale, 1.),
                 rate=-5.*float(price)*bits/share, nmse=-.5*errors*n/cube_energy.clamp_min(1e-12))
    reward = sum(terms.values())
    if not bool(torch.isfinite(reward).all()):
        raise FloatingPointError('Nonfinite physical reward component.')
    return reward, {f'{key}_penalty': float(value.sum()) for key, value in terms.items()}
