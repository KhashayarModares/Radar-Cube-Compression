"""Small preservation-first PPO refinement; native codec/receiver are unchanged.

Coarse/fine relative commands reach the full 0.01..1 retention lattice without
expanding the 12-round horizon. Conditional targets first minimize measured LOW
misses and then measured rate/peak/fidelity cost. No work runs on import.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.distributions import Categorical

from . import arm_e_adaptive_gpu_learning as base

ROUNDS = 12
PRICES = base.PRICES
DEFAULT_PRICE = .1
STATIC_NAMES, DYNAMIC_NAMES, NAMES = base.STATIC_NAMES, base.DYNAMIC_NAMES, base.NAMES
RHO_EDITS = (-20, -5, -1, 0, 1, 5, 20)  # hundredths; exact integer lattice
N_ACTIONS = len(RHO_EDITS)*3
HOLD = RHO_EDITS.index(0)*3+1
raw_states, normalize, correlation = base.raw_states, base.normalize, base.correlation


def commands(device):
    return torch.tensor([(r/100., q) for r in RHO_EDITS for q in (-1.,0.,1.)],
                        dtype=torch.float64, device=device)


def alternatives(rho, q):
    delta = commands(rho.device)
    hundredths = (rho*100).round().to(torch.int64)
    edits = (delta[:,0]*100).round().to(torch.int64)
    r = (hundredths[:,None]+edits[None]).clamp(1,100).to(torch.float64)/100.
    bits = (q[:,None]+delta[None,:,1]).clamp(1,12)
    duplicate = (r[:,:,None] == r[:,None,:]) & (bits[:,:,None] == bits[:,None,:])
    earlier = torch.tril(torch.ones((N_ACTIONS,N_ACTIONS),dtype=torch.bool,device=rho.device),diagonal=-1)
    mask = ~(duplicate & earlier).any(2)
    return r,bits,mask


def reward_components(metrics, bpcs, rate_price):
    """Unsaturated secondary costs, with an explicit LOW-weighted PPO surrogate.

    Strict LOW priority is implemented in conditional selection/checkpoint
    ordering, not claimed as a mathematical guarantee of this weighted reward.
    Scale is frozen from original-map TRAIN unmatched counts, never TEST.
    """
    bpcs = bpcs.to(torch.float64)
    def value(name, default=0.):
        return torch.as_tensor(metrics.get(name,default),device=bpcs.device,dtype=torch.float64)
    price = torch.as_tensor(rate_price,device=bpcs.device,dtype=torch.float64)
    scale = value('unmatched_scale',1.)
    if not bool(torch.isfinite(scale).all() & (scale > 0).all()
                & torch.isfinite(price).all() & (price >= 0).all() & (price <= .2).all()):
        raise ValueError('Finite positive TRAIN scale and trained rate price required.')
    terms = dict(lowest_miss_penalty=-4.*value('low_miss'),
        cfar_margin_penalty=-.1*value('margin_deficit'),
        unmatched_penalty=-.5*value('unmatched_excess')/scale,
        rate_penalty=-price*bpcs/4.,
        nmse_penalty=-.05*value('nmse'))
    terms['secondary_cost'] = -(terms['cfar_margin_penalty']+terms['unmatched_penalty']
                                +terms['rate_penalty']+terms['nmse_penalty'])
    terms['reward'] = terms['lowest_miss_penalty']-terms['secondary_cost']
    if not all(bool(torch.isfinite(v).all()) for v in terms.values()):
        raise FloatingPointError('Nonfinite physical reward component.')
    return terms


def compatible_mask(low, cost, feasible):
    """Preservation first; secondary measurements break otherwise silent ties."""
    if not bool(feasible.any()) or not bool(torch.isfinite(low[feasible]).all()
                                           & torch.isfinite(cost[feasible]).all()):
        raise ValueError('At least one finite physically measured command required.')
    first = feasible & (low == low.masked_fill(~feasible,torch.inf).min())
    minimum = cost.masked_fill(~first,torch.inf).min()
    return first & torch.isclose(cost,minimum,rtol=0.,atol=1e-9)


def fit_features(matrix, threshold=.98):
    if matrix.ndim != 2 or matrix.shape[1] != len(NAMES) or not bool(torch.isfinite(matrix).all()):
        raise ValueError('Finite TRAIN-only observation matrix required.')
    pearson, spearman = correlation(matrix),correlation(matrix,rank=True)
    mean, std = matrix.mean(0),matrix.std(0,unbiased=False)
    # Suffix energy does not describe remaining bits in parallel tile editing.
    excluded = {'future_energy_fraction'}
    protected = set(base.PROTECTED)-{'cfar_coverage'}
    kept, rows = [],[]
    for i,name in enumerate(NAMES):
        retain, reason = True,'retained: no strong TRAIN redundancy'
        if name in excluded:
            retain,reason = False,'removed: canonical-order suffix energy is not control state in parallel editing'
        elif float(std[i]) < 1e-8:
            retain,reason = False,'constant on this TRAIN audit; raw coverage still masks missing CFAR descriptors'
        elif name in protected:
            reason = 'protected local allocation, position or detector context'
        else:
            for j in kept:
                if abs(float(pearson[i,j])) >= threshold and abs(float(spearman[i,j])) >= threshold:
                    retain,reason = False,'TRAIN Pearson and Spearman >= .98 with '+NAMES[j]
                    break
        if retain:
            kept.append(i)
        rows.append(dict(feature=name,keep=retain,reason=reason))
    coverage = matrix[:,STATIC_NAMES.index('cfar_coverage')] > 0
    for name in ('weak_peak_frac','cfar_margin','bg_spread','peak_avail'):
        j=NAMES.index(name)
        mean[j]=matrix[coverage,j].mean() if bool(coverage.any()) else 0.
        std[j]=matrix[coverage,j].std(unbiased=False) if bool(coverage.any()) else 1.
    indices=torch.tensor(kept,dtype=torch.int64,device=matrix.device)
    return dict(indices=indices,mean=mean[indices],std=std[indices].clamp_min(1e-6),
        pearson=pearson,spearman=spearman,rows=rows,names=[NAMES[i] for i in kept])


class DeltaPolicy(nn.Module):
    def __init__(self, dim, device, hidden=32):
        super().__init__()
        self.actor=nn.Sequential(nn.Linear(dim,hidden),nn.Tanh(),nn.Linear(hidden,N_ACTIONS))
        self.critic=nn.Sequential(nn.Linear(dim,hidden),nn.Tanh(),nn.Linear(hidden,1))
        self.to(device=device,dtype=torch.float64)
        for layer in self.modules():
            if isinstance(layer,nn.Linear):
                nn.init.orthogonal_(layer.weight,math.sqrt(2.)); nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor[-1].weight,.01)
        nn.init.orthogonal_(self.critic[-1].weight,1.)
        # A small action prior makes initial greedy inference hold (.25,8).
        # Exploration remains near uniform; no comparator/annotation imitation.
        with torch.no_grad():
            self.actor[-1].bias[HOLD]=2*.01*math.sqrt(hidden)+.01

    def forward(self, states, mask):
        return Categorical(logits=self.actor(states).masked_fill(~mask,-torch.inf)),self.critic(states).flatten()


@torch.no_grad()
def rollout(policy, fitted, features, price, *, deterministic=False, rho=None, q=None, start_step=0):
    if not 0 <= start_step <= ROUNDS:
        raise ValueError('Edit round outside declared horizon.')
    n=len(features)
    rho=features.new_full((n,),.25) if rho is None else rho.clone()
    q=features.new_full((n,),8.) if q is None else q.clone()
    rows=torch.arange(n,device=features.device)
    history=[]
    for step in range(start_step,ROUNDS):
        states=normalize(raw_states(features,rho,q,ROUNDS-step,price),fitted)
        r,bits,mask=alternatives(rho,q)
        distribution,value=policy(states,mask)
        action=distribution.probs.argmax(1) if deterministic else distribution.sample()
        history.append(dict(states=states,mask=mask,actions=action,log_prob=distribution.log_prob(action),
            value=value,rho_before=rho,q_before=q,probabilities=distribution.probs))
        rho,q=r[rows,action],bits[rows,action]
    return dict(rho=rho,q=q,history=history,price=float(price))


def update(policy, optimizer, trajectories, queries, *, seed, epochs=5, auxiliary=.3):
    before_actor=[p.detach().clone() for p in policy.actor.parameters()]
    before_critic=[p.detach().clone() for p in policy.critic.parameters()]
    result=base.update(policy,optimizer,trajectories,queries,seed=seed,epochs=epochs,auxiliary=auxiliary)
    with torch.no_grad():
        result['actor_parameter_change']=torch.stack([(p-a).square().sum()
            for p,a in zip(policy.actor.parameters(),before_actor)]).sum().sqrt()
        result['critic_parameter_change']=torch.stack([(p-a).square().sum()
            for p,a in zip(policy.critic.parameters(),before_critic)]).sum().sqrt()
    return result
