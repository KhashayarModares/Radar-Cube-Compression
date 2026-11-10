"""Uncapped adapter for the unchanged coherent codec and 16 physical actions.

The four obsolete remaining-budget state slots are constant zero. All other
observations retain their original definitions and finite reference units.
No cap, cost-based action masking, refinement, teacher or fallback is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def encode_uncapped(ex, cube, policy, normalizer, stochastic, source_context=None):
    cube = np.asarray(cube)
    source = ex.prepare_policy_source_v12_12(cube) if source_context is None else source_context
    if tuple(cube.shape) != tuple(source['cube_shape']):
        raise ValueError('Prepared source shape differs from the raw cube.')
    allowed = np.asarray(policy.allowed_action_mask, bool)
    if allowed.shape != (len(ex.ACTION_LIBRARY_V11),) or not allowed.all():
        raise ValueError('Uncapped PPO must expose every action; no action preselection.')
    spectrum = np.zeros_like(source['spectrum'])
    states, raw_states, actions, logp, values, logs, errors = ([] for _ in range(7))
    running, previous = float(source['cube_header_bits']), None
    for index, (slc, prepared) in enumerate(zip(source['slices'], source['prepared_blocks'])):
        raw = ex._codec_control_state_v11(slc, cube.shape, running, cube.size, previous,
            prepared, source['cube_mean_power'], source['cube_mean_spectral_concentration'],
            source['cube_block_log_energy_std'], source['cube_spectral_concentration_std'])
        raw = np.r_[raw, np.zeros(4)].astype(np.float32)
        state = normalizer.transform(raw[None])[0]
        p, value, _ = policy.forward(state[None])
        p = np.asarray(p[0], np.float64)
        if not np.isfinite(p).all() or np.any(p < 0) or p.sum() <= 0:
            raise FloatingPointError('Invalid categorical actor probabilities.')
        p = p / p.sum()
        action = int(policy.rng.choice(len(p), p=p)) if stochastic else int(p.argmax())
        rho, q = ex.ACTION_LIBRARY_V11[action]
        tile, bits = ex.coherent_codec_v14.materialize_action(prepared, rho, q)
        spectrum[slc] = tile
        running += float(bits['realized_bits'])
        states.append(state); raw_states.append(raw); actions.append(action)
        logp.append(float(np.log(p[action]+ex.EPS))); values.append(float(value[0]))
        errors.append(float(prepared['action_distortion'][action]))
        logs.append(dict({**bits, 'block_idx':index, 'action_index':action, 'rho':rho, 'q':q},
            block_log_energy_ratio=float(raw[4]), spectral_top10_energy_fraction=float(raw[5]),
            cube_mean_spectral_concentration=float(raw[6]), cube_block_log_energy_std=float(raw[7]),
            cube_spectral_concentration_std=float(raw[8]), feasible_action_count=len(p),
            compact_action_count=len(p), masked_action_count=0,
            transform_distortion_contribution=errors[-1], remaining_budget_fraction=0.,
            minimum_remaining_cost_fraction=0., maximum_remaining_cost_fraction=0.,
            allowance_per_remaining_coefficient=0.))
        previous = action
    decoded = ex.coherent_codec_v14.reconstruct(source, spectrum)
    if not np.isfinite(np.asarray(states)).all():
        raise FloatingPointError('Uncapped state construction produced infinity/NaN.')
    nmse = float(np.sum(errors))
    meta = dict(method='PPO', codec='global_RD_shared_antenna_support_v14',
        realized_bits=running, realized_bpcs=running/cube.size, cube_nmse=nmse,
        rel_error=float(np.sqrt(nmse)), compression_ratio_vs_v43_int16_iq_32bpc=32.*cube.size/running,
        memory_ratio_proxy_64b_complex=64.*cube.size/running, actual_codec_bitstream_saved=False,
        codec_size_is_analytical=True, rate_budget_bits=None,
        cube_header_bits=float(source['cube_header_bits']), rate_budget_enforced_by_action_mask=False,
        compact_policy_action_count=len(allowed), raw_cube_metrics_used_for_training=True,
        refinement_used=False, obsolete_budget_state_slots='four constant zeros')
    return dict(cube_hat=decoded, states_raw=np.asarray(raw_states, np.float32),
        states=np.asarray(states, np.float32), actions=np.asarray(actions, np.int64),
        action_masks=np.ones((len(actions), len(allowed)), bool),
        old_log_probs=np.asarray(logp, np.float32), old_values=np.asarray(values, np.float32),
        rewards=np.zeros(len(actions), np.float64), block_log=pd.DataFrame(logs),
        meta=meta, distortion_contributions=np.asarray(errors, np.float64))


class ForcedAction:
    """TRAIN-only frozen-policy suffix: replay past, force current, resample future."""
    def __init__(self, policy, actions, prefix, action, seed):
        self.policy, self.actions = policy, np.asarray(actions, np.int64)
        self.prefix, self.action, self.index = int(prefix), int(action), 0
        self.allowed_action_mask = np.asarray(policy.allowed_action_mask, bool).copy()
        self.rng = np.random.default_rng(seed)

    def forward(self, states):
        if len(states) != 1 or self.index >= len(self.actions):
            raise ValueError('Use one forced facade for a single sequential encoding.')
        p, value, hidden = self.policy.forward(states)
        p = np.asarray(p).copy()
        if self.index <= self.prefix:
            action = int(self.actions[self.index]) if self.index < self.prefix else self.action
            p.fill(0.); p[0, action] = 1.
        self.index += 1
        return p, value, hidden
