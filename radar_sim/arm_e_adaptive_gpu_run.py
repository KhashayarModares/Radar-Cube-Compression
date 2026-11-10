"""Separately versioned adaptive Arm-E-derived GPU MATLAB experiment.

CUDA numerical path; host file decoding, provenance/label validation, control
flow and plotting are explicit exceptions. Never launches work on import.
600 TRAIN / 200 validation / 200 frozen TEST, historical whole-episode split.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import pandas as pd
import torch

from . import arm_e_adaptive_gpu_learning as learning
from .arm_e_adaptive_gpu_physics import GPUPhysics, THRESHOLDS, pooled_ap, tile_action_descriptors
from . import arm_e_full_v12_22_run as historical
from . import matlab_v44_dataset as native

ROOT = Path(__file__).resolve().parents[1]
REVISION = '12.24.1-adaptive-E-CUDA-preservation-first-MSE-onebit'
SEED = 20261022


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def configuration(profile):
    if profile == 'full':
        return dict(train=600, validation=200, test=200, passes=2, batch=5,
                    rollouts=2, epochs=5, updates=240, checkpoints=[0, 60, 120, 180, 240], audit_cubes=16)
    if profile == 'tiny':
        return dict(train=2, validation=2, test=2, passes=1, batch=2,
                    rollouts=1, epochs=1, updates=1, checkpoints=[0, 1], audit_cubes=2)
    raise ValueError('Unknown profile.')


def radar_config(dataset):
    spec = read_json(Path(dataset)/'metadata/radar_spec.json')
    return SimpleNamespace(N=int(spec['NumFastTimeSamples']), M=int(spec['NumChirps']),
        num_rx=int(spec['NumReceiveElements']), fs=float(spec['SampleRateHz']),
        S=float(spec['SweepSlopeHzPerS']), Tchirp=float(spec['SweepTimeS']),
        fc=float(spec['CenterFrequencyHz']), lam=float(spec['WavelengthM']), c0=299792458., snr_db=14.)


def context(dataset, device, profile):
    if torch.device(device).type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('CUDA is required. No production CPU fallback.')
    data = Path(dataset).resolve()
    native.validated_manifest(data)  # metadata/file inventory, not TEST bodies
    splits = historical.planned_populations(data, profile)
    return dict(revision=REVISION, dataset=str(data), device=str(torch.device(device)),
        gpu=torch.cuda.get_device_name(device), torch=torch.__version__, seed=SEED,
        profile=profile, configuration=configuration(profile),
        source_hashes={p.name: sha(p) for p in sorted((ROOT/'radar_sim').glob('*.py'))},
        split_hashes={p.name: sha(p) for p in historical.INPUTS},
        baseline_hash=sha(historical.RDO_REFERENCE),
        metadata_hashes={p.name: sha(p) for p in sorted((data/'metadata').glob('*.json'))},
        populations=[frame.record_name.astype(str).tolist() for frame in splits],
        bounds=dict(rho=[.01, .5], q=[1, 12], rho_step=.05, q_step=1, rounds=learning.ROUNDS),
        rate_prices=list(learning.PRICES), selection_rate_price=learning.DEFAULT_PRICE,
        NMSE_constraint=False, bitrate_cap=False, actor_codec_detector_and_metrics='torch CUDA',
        CPU_exceptions=['filesystem/MAT decompression', 'metadata and label provenance checks',
                        'Python orchestration', 'serialization', 'matplotlib report rendering'],
        pretrained_checkpoint_loaded=False, receiver_support_shared=True,
        actual_bitstream_saved=False, historical_test_previously_used_in_research=True,
        claim='new Arm-E-derived controller, not an isolated action-set ablation or guaranteed zero-miss solution')


def load_cube_gpu(path, expected, device):
    """Read integer MAT payloads on host, decode I/Q and scale on declared GPU.

    Provenance is metadata; physics and compact ADC reconstruction are tensors.
    Classic MAT decompression cannot run on CUDA through scipy.io.loadmat.
    """
    import h5py
    import numpy as np  # file representations only; no CPU signal arithmetic
    metadata = {}
    scalar_fields = ('sceneID', 'frameIndex', 'simulationTime')
    nested = dict(episodeInfo=('SceneID', 'SceneTemplateID', 'EpisodeIndex', 'EpisodeSeed', 'TrafficTimeScale'),
        radarSpec=('ExpectedCubeCount', 'NumTrafficEpisodes', 'NumSceneTemplates', 'NumReceiveElements', 'NumChirps', 'NumFastTimeSamples'))
    if h5py.is_hdf5(path):
        with h5py.File(path, 'r') as handle:
            def value(node):
                raw = node[()]
                return handle[np.asarray(raw).item()][()] if h5py.check_dtype(ref=np.asarray(raw).dtype) else raw
            i, q, scale = [handle[key][()] for key in ('adcI', 'adcQ', 'adcScale')]
            metadata.update({key: value(handle[key]) for key in scalar_fields})
            for name, keys in nested.items():
                group = handle[name]
                if isinstance(group, h5py.Dataset):
                    group = handle[np.asarray(group[()]).item()]
                metadata[name] = {key: value(group[key]) for key in keys}
        axes = (1, 0, 2)
    else:
        import warnings
        from scipy.io import loadmat
        from scipy.io.matlab import MatReadWarning
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Duplicate variable name "None" in stream.*', category=MatReadWarning)
            raw = loadmat(path, variable_names=['adcI', 'adcQ', 'adcScale', *scalar_fields, *nested], struct_as_record=False)
        i, q, scale = [raw[key] for key in ('adcI', 'adcQ', 'adcScale')]
        metadata.update({key: raw[key] for key in scalar_fields})
        for name, keys in nested.items():
            group = np.asarray(raw[name]).item()
            metadata[name] = {key: getattr(group, key) for key in keys}
        axes = (1, 2, 0)
    if i.ndim != 3 or i.shape != q.shape or any(a.dtype.kind != 'i' or a.dtype.itemsize != 2 for a in (i, q)):
        raise ValueError('Expected matching int16 compact I/Q arrays.')
    scale = np.asarray(scale)
    if scale.size != 1 or scale.dtype.kind != 'f' or scale.dtype.itemsize != 4:
        raise ValueError('Expected a single-precision compact scale.')
    def scalar(record, key):
        return float(np.asarray(record[key]).item())
    if (scalar(metadata, 'sceneID') != int(expected['episode_id'])
            or scalar(metadata, 'frameIndex') != int(expected['frame_in_scene'])+1
            or abs(scalar(metadata, 'simulationTime')-float(expected['simulation_time_s'])) > 1e-8):
        raise ValueError('Embedded scene/frame/time does not match frozen split metadata.')
    for key, name in (('SceneID', 'episode_id'), ('SceneTemplateID', 'template_id'),
                      ('EpisodeIndex', 'episode_index'), ('EpisodeSeed', 'episode_seed'), ('TrafficTimeScale', 'traffic_time_scale')):
        if abs(scalar(metadata['episodeInfo'], key)-float(expected[name])) > 1e-12:
            raise ValueError('Embedded episode provenance mismatch: '+key)
    expected_spec = dict(ExpectedCubeCount=1000, NumTrafficEpisodes=25, NumSceneTemplates=7,
                         NumReceiveElements=24, NumChirps=96, NumFastTimeSamples=1670)
    if any(scalar(metadata['radarSpec'], key) != val for key, val in expected_spec.items()):
        raise ValueError('Embedded radar dimensions/dataset provenance mismatch.')
    # The only host payload operation is making file buffers contiguous.
    real = torch.as_tensor(np.ascontiguousarray(i), device=device).to(torch.float32)
    imag = torch.as_tensor(np.ascontiguousarray(q), device=device).to(torch.float32)
    factor = torch.as_tensor(scale.item(), device=device, dtype=torch.float32)
    if not bool(torch.isfinite(factor) & (factor > 0)):
        raise ValueError('Nonfinite/nonpositive ADC scale.')
    cube = (torch.complex(real, imag)*factor).permute(*axes).contiguous()
    if tuple(cube.shape) != native.EXPECTED_SHAPE or not bool(torch.isfinite(cube).all()):
        raise ValueError('Malformed decoded cube.')
    return cube


def scalar_record(metrics, meta, row, *, price, update=0):
    record = {key: value.detach().item() for key, value in metrics.items()
              if isinstance(value, torch.Tensor) and value.numel() == 1}
    record.update(record_name=str(row.record_name), episode_id=int(row.episode_id),
                  scene_name=str(row.scene_name), update=int(update), rate_price=float(price),
                  bpcs=float(meta['bpcs']), nmse=float(meta['nmse']))
    components = learning.reward_components(metrics, meta['bpcs'], price)
    record.update({key: float(value) for key, value in components.items()})
    expected = sum(record[key] for key in ('lowest_miss_penalty', 'six_miss_penalty',
        'cfar_margin_penalty', 'unmatched_penalty', 'zero_miss_rate_bonus'))
    if abs(expected-record['reward']) > 1e-10:
        raise AssertionError('Independent exported reward-component sum failed.')
    return record


def csv(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError('Preserve existing table: '+str(path))
    pd.DataFrame(records).to_csv(path, index=False)


class Runtime:
    def __init__(self, dataset, device, profile):
        self.dataset, self.device, self.profile = Path(dataset), torch.device(device), profile
        self.config = configuration(profile)
        self.rows = historical.planned_populations(dataset, profile)
        self.cfg = radar_config(dataset)
        self.physics = GPUPhysics(self.cfg, device=device)
        table = pd.read_csv(historical.RDO_REFERENCE)
        selected = table.loc[table.selected_fixed_action.astype(str).str.lower().isin(('true', '1'))]
        if len(selected) != 1:
            raise ValueError('Historical independent RDO summary must select exactly one action.')
        self.fixed_rdo = (float(selected.iloc[0].rho), int(selected.iloc[0].q))
        permitted = pd.concat(self.rows[:2], ignore_index=True)
        self.labels = historical.scoped_labels(dataset, self.cfg, permitted, stage='TRAIN_VALIDATION')
        self.labels = self.labels.merge(permitted[['global_frame', 'record_name']], on='global_frame', validate='many_to_one')
        self.test_labels = None

    def case(self, row, *, test=False):
        cube = load_cube_gpu(row.cube_path, row.to_dict(), self.device)
        source = self.physics.prepare(cube)
        post = self.physics.maps(cube)
        labels = self.test_labels if test else self.labels
        # Existing label extraction is an exact record-name join, not GT inference.
        labs = labels.loc[labels.record_name == row.record_name]
        features = learning.static_features(self.physics, source, post)
        return source, post, labs, features

    def outcome(self, source, original, labs, rho, q, *, detection=False):
        decoded, meta = self.physics.encode(source, rho, q)
        post = self.physics.maps(decoded)
        metrics = self.physics.evaluate(original, post, labs, detection_metrics=detection)
        return metrics, meta


def prepare_features(runtime, out):
    panels = historical.diverse_rows(runtime.rows[0], runtime.config['audit_cubes'])
    matrix, legacy_matrix = [], []
    from .ppo_v12_22_codec import PROPOSED_ACTIONS, state_names
    legacy_names = state_names(PROPOSED_ACTIONS)
    print('Fitting and pruning label-free TRAIN observations only...', flush=True)
    for _, row in panels.iterrows():
        source, post, labs, static = runtime.case(row)
        del labs
        # Offline original-menu descriptors are audited, not fed to the actor.
        descriptors = tile_action_descriptors(source, PROPOSED_ACTIONS)
        reference = 4.8  # historical descriptor scaling ONLY, not a new cap
        costs = descriptors['action_bits_by_block']
        errors = descriptors['action_distortion_by_block']
        sizes = static.new_tensor([tile['Z'].numel() for tile in source['tiles']])
        legacy_actions = torch.stack((costs/(sizes[:, None]*reference), torch.log1p(1000.*errors)), dim=2).flatten(1)
        cube_stats = torch.stack((static[:, 3].mean(), static[:, 2].std(unbiased=False), static[:, 3].std(unbiased=False)))
        # Independent label-free control sweeps prevent fake rank correlations
        # between rho, q and time caused solely by a diagonal sampling design.
        for step in range(learning.ROUNDS):
            r = .01+.49*step/(learning.ROUNDS-1)
            q = 1.+((step*5+3) % 12)
            remaining = learning.ROUNDS-((step*7+2) % 12)
            matrix.append(learning.raw_states(static, static.new_full((len(static),), r),
                static.new_full((len(static),), q), remaining, learning.PRICES[(step*3) % len(learning.PRICES)]))
            before = static.new_zeros((len(static), 16))
            before[:, :2], before[:, 3] = static[:, :2], r
            before[:, 2] = (costs[:, 0].cumsum(0)-costs[:, 0]+768)/(source['cube'].numel()*reference)
            before[:, 4:6] = static[:, 2:4]
            before[:, 6:9] = cube_stats
            before[:, 9:13] = static[:, [4, 5, 6, 7]]
            before[:, 13], before[:, 14:16] = q/16., static[:, 8:10]
            legacy_matrix.append(torch.cat((before, legacy_actions, static[:, 10:14]), dim=1))
        print('  TRAIN feature audit:', row.record_name, flush=True)
    fitted = learning.fit_features(torch.cat(matrix))
    tables = out/'tables'
    tables.mkdir(parents=True, exist_ok=True)
    for name in ('pearson', 'spearman'):
        pd.DataFrame(fitted[name].cpu().numpy(), index=learning.NAMES, columns=learning.NAMES).to_csv(tables/f'feature_correlation_{name}.csv')
        legacy = learning.correlation(torch.cat(legacy_matrix), rank=name == 'spearman')
        pd.DataFrame(legacy.cpu().numpy(), index=legacy_names, columns=legacy_names).to_csv(tables/f'legacy_feature_correlation_{name}.csv')
    csv(tables/'feature_selection.csv', fitted['rows'])
    csv(tables/'feature_audit_roster.csv', panels.to_dict('records'))
    # Explain every original input, including structurally replaced fields.
    old_rows = []
    retained_names = set(fitted['names'])
    for name in legacy_names:
        if name.startswith('action_'):
            reason = 'structurally removed: no fixed absolute-action head; physical delta command is the policy action'
        elif name == 'running_rate_fraction':
            reason = 'replaced: explicit trained rate-price input, no bitrate cap or arbitrary running allowance'
        elif name in ('previous_keep_fraction', 'previous_q_div_16'):
            reason = 'replaced: current tile rho/q plus whole-allocation mean rho/q'
        elif name in retained_names:
            reason = 'retained (subject to exported TRAIN correlation audit)'
        elif name in learning.NAMES:
            reason = next(item['reason'] for item in fitted['rows'] if item['feature'] == name)
        else:
            reason = 'removed cube-wide aggregate for simpler local controller; correlation exported, harmlessness NOT established'
        old_rows.append(dict(feature=name, keep=name in retained_names, reason=reason))
    csv(tables/'legacy_feature_selection.csv', old_rows)
    torch.save(fitted, out/'feature_normalizer.pt')
    write_json(out/'OBSERVATION_CONTRACT.json', dict(raw_dimension=len(learning.NAMES), selected_dimension=len(fitted['names']),
        selected=fitted['names'], protected=sorted(learning.PROTECTED), correlation_threshold=.98,
        fixed_menu_descriptors_removed=42, previous_62_feature_comparison='structural removal of action-index-specific inputs, not a claim they had no predictive utility',
        legacy_correlation_scope='recomputed label-free TRAIN descriptors with declared historical scaling and control sweeps; not imported old trajectories or causal feature importance',
        legacy_global_aggregate_slots_removed=[name for name in
            ('cube_mean_spectral_concentration', 'cube_block_log_energy_std', 'cube_spectral_concentration_std')
            if name not in retained_names],
        simplification_caveat='Structural removals are a new-model design choice, not empirically established harmless ablations; all legacy correlations exported for scrutiny.',
        source_partition='TRAIN only', causal_feature_importance_claim=False,
        same_feature_used_for_all_rate_prices=True, missing_CFAR_coverage='explicit indicator and zero raw descriptors'))
    return fitted


@torch.no_grad()
def evaluate(runtime, actor, fitted, rows, price, update, *, test=False):
    records = []
    for _, row in rows.iterrows():
        source, original, labs, features = runtime.case(row, test=test)
        allocation = learning.rollout(actor, fitted, features, price, deterministic=True)
        metrics, meta = runtime.outcome(source, original, labs, allocation['rho'], allocation['q'])
        records.append(scalar_record(metrics, meta, row, price=price, update=update))
    frame = pd.DataFrame(records)
    print(f'Adaptive E greedy {"TEST" if test else "VALIDATION"} update {update}: '
          f'LOW={int(frame.low_miss.sum())} SIX={int(frame.six_miss.sum())} '
          f'U+={frame.unmatched_excess.mean():.2f} bpcs={frame.bpcs.mean():.3f}', flush=True)
    return records


@torch.no_grad()
def query_final_edit(runtime, source, original, labs, features, allocation, update):
    last = allocation['history'][-1]
    covered = torch.nonzero(features[:, learning.STATIC_NAMES.index('cfar_coverage')] > 0, as_tuple=False).flatten()
    eligible = covered if len(covered) else torch.arange(len(features), device=features.device)
    tile = int(eligible[update % len(eligible)])
    r, q, mask = learning.alternatives(last['rho_before'][tile:tile+1], last['q_before'][tile:tile+1])
    losses = source['cube_energy'].new_full((9,), torch.inf)
    six = losses.clone()
    records = []
    for action in torch.nonzero(mask[0], as_tuple=False).flatten().tolist():
        rho, bits = allocation['rho'].clone(), allocation['q'].clone()
        rho[tile], bits[tile] = r[0, action], q[0, action]
        metric, meta = runtime.outcome(source, original, labs, rho, bits)
        losses[action], six[action] = metric['low_miss'], metric['six_miss']
        records.append(dict(tile=tile, action=action, rho=float(rho[tile]), q=int(bits[tile]),
            low_miss=int(metric['low_miss']), six_miss=int(metric['six_miss']),
            unmatched_excess=float(metric['unmatched_excess']), bpcs=float(meta['bpcs'])))
    compatible = learning.compatible_mask(losses, six, mask[0])
    packet = dict(state=last['states'][tile].detach(), mask=mask[0], compatible=compatible,
                  probabilities_before=last['probabilities'][tile].detach())
    return packet, records


def train(runtime, fitted, out):
    cfg = runtime.config
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    actor = learning.DeltaPolicy(len(fitted['names']), runtime.device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4, eps=1e-5)
    generator = torch.Generator(device=runtime.device).manual_seed(SEED+1)
    validation, train_records, query_records, selections = [], [], [], []

    def checkpoint(update):
        # Greedy validation must not advance the stochastic TRAIN RNG.
        with torch.random.fork_rng(devices=[runtime.device.index or 0]):
            rows = evaluate(runtime, actor, fitted, runtime.rows[1], learning.DEFAULT_PRICE, update)
        validation.extend(rows)
        frame = pd.DataFrame(rows)
        key = [int(frame.low_miss.sum()), int(frame.six_miss.sum()), float(frame.unmatched_excess.mean()), float(frame.bpcs.mean())]
        path = out/'checkpoints'/f'update_{update:04d}.pt'
        path.parent.mkdir(exist_ok=True)
        torch.save(dict(actor=actor.state_dict(), optimizer=optimizer.state_dict(), update=update,
            rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(runtime.device),
            selection_key=key, feature_names=fitted['names']), path)
        selections.append((key, update, path))
        csv(out/'tables'/f'validation_update_{update:04d}.csv', rows)

    checkpoint(0)
    update = 0
    for pass_index in range(cfg['passes']):
        order = torch.randperm(len(runtime.rows[0]), generator=generator, device=runtime.device).cpu().tolist()
        for start in range(0, len(order), cfg['batch']):
            update += 1
            trajectories, queries, logs = [], [], []
            for offset, position in enumerate(order[start:start+cfg['batch']]):
                row = runtime.rows[0].iloc[position]
                source, original, labs, features = runtime.case(row)
                for rollout_index in range(cfg['rollouts']):
                    price = learning.PRICES[(update+offset+rollout_index) % len(learning.PRICES)]
                    allocation = learning.rollout(actor, fitted, features, price)
                    metric, meta = runtime.outcome(source, original, labs, allocation['rho'], allocation['q'])
                    terms = learning.reward_components(metric, meta['bpcs'], price)
                    trajectories.append((allocation, terms['reward']))
                    logs.append(scalar_record(metric, meta, row, price=price, update=update))
                    if offset == 0 and rollout_index == 0:
                        # Query the deployment-greedy TRAIN state as well as
                        # sampling fresh PPO trajectories. Never put this greedy
                        # branch in the on-policy PPO likelihood/critic batch.
                        greedy = learning.rollout(actor, fitted, features, price, deterministic=True)
                        packet, measurements = query_final_edit(runtime, source, original, labs, features, greedy, update)
                        queries.append(packet)
                        query_records.extend(dict(record_name=row.record_name, update=update,
                            context='fresh_greedy_TRAIN_final_edit', **m) for m in measurements)
            stats = learning.update(actor, optimizer, trajectories, queries, seed=SEED+update, epochs=cfg['epochs'])
            for log in logs:
                log.update({key: float(value) for key, value in stats.items()})
                log['pass_index'] = pass_index+1
            train_records.extend(logs)
            csv(out/'tables'/f'train_update_{update:04d}.csv', logs)
            mean = pd.DataFrame(logs)
            print(f'Adaptive E update {update}/{cfg["updates"]} stochastic: LOW={mean.low_miss.mean():.3f} '
                f'SIX={mean.six_miss.mean():.3f} U+={mean.unmatched_excess.mean():.2f} '
                f'reward={mean.reward.mean():.4f} bpcs={mean.bpcs.mean():.3f} KL={float(stats["approx_kl"]):.5g}', flush=True)
            if update in cfg['checkpoints']:
                checkpoint(update)
    if update != cfg['updates']:
        raise AssertionError('Optimizer budget differs from declared configuration.')
    csv(out/'tables/training.csv', train_records)
    csv(out/'tables/validation.csv', validation)
    csv(out/'tables/conditional_final_edit.csv', query_records)
    key, selected_update, selected_path = min(selections, key=lambda item: (item[0], item[1]))
    # Freeze using validation only, before reading any TEST label/cube bodies.
    write_json(out/'FROZEN_POLICY.json', dict(arm='E', variant=REVISION, update=selected_update,
        checkpoint=str(selected_path.relative_to(out)), checkpoint_sha256=sha(selected_path),
        selection_key=key, selection_rate_price=learning.DEFAULT_PRICE, test_used_for_selection=False,
        lowest_validation_misses_zero=key[0] == 0))
    write_json(out/'TRAIN_COMPLETED.json', dict(completed=True, updates=update, test_evaluated=False))


def final_reports(runtime, out, fitted, *, include_test):
    frozen = read_json(out/'FROZEN_POLICY.json')
    path = out/frozen['checkpoint']
    if sha(path) != frozen['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint changed.')
    actor = learning.DeltaPolicy(len(fitted['names']), runtime.device)
    actor.load_state_dict(torch.load(path, map_location=runtime.device, weights_only=False)['actor'])
    splits = [('validation', runtime.rows[1])]
    if include_test:
        runtime.test_labels = historical.scoped_labels(runtime.dataset, runtime.cfg, runtime.rows[2],
            stage='FINAL_TEST', frozen=out/'FROZEN_POLICY.json')
        runtime.test_labels = runtime.test_labels.merge(runtime.rows[2][['global_frame', 'record_name']],
                                                       on='global_frame', validate='many_to_one')
        splits.append(('test', runtime.rows[2]))
    rate_rows, allocations, population, command_rows = [], [], [], []
    for split, rows in splits:
        # Predeclared sweeps, not selected using TEST outcomes.
        methods = [('PPO adaptive E', f'price={price:g}', price, None) for price in learning.PRICES]
        methods += [('Original', 'uncompressed', .1, None), ('FFT14', 'full-support14', .1, (1., 14)),
                    ('CalibratedRDO', 'frozen-historical-action', .1, runtime.fixed_rdo),
                    ('Uniform codec', 'rho=.01,q=1', .1, (.01, 1)),
                    ('Uniform codec', 'rho=.05,q=4', .1, (.05, 4)),
                    ('Uniform codec', 'rho=.25,q=8', .1, (.25, 8)),
                    ('Uniform codec', 'rho=.5,q=12', .1, (.5, 12))]
        ranked = {(method, point): {kind: ([], [], 0) for kind in ('RD', 'RA')}
                  for method, point, _, _ in methods}
        command_counts = {point: torch.zeros((learning.ROUNDS, 9), dtype=torch.int64, device=runtime.device)
                          for method, point, _, _ in methods if method == 'PPO adaptive E'}
        print(f'Starting matched {split} reports: {len(rows)} cubes, {len(methods)} operating points.', flush=True)
        # Load/transform one cube once, then evaluate every method on that exact
        # cached source. No repeated MAT decompression and no population cache.
        for cube_index, (_, row) in enumerate(rows.iterrows(), 1):
            source, original, labs, features = runtime.case(row, test=split == 'test')
            n = len(features)
            for method, point, price, uniform in methods:
                if method == 'Original':
                    metrics = runtime.physics.evaluate(original, original, labs)
                    meta = dict(bpcs=features.new_tensor(32.), nmse=features.new_zeros(()))
                else:
                    if method == 'FFT14':
                        decoded, meta = runtime.physics.full_fft(source['cube'])
                        metrics = runtime.physics.evaluate(original, runtime.physics.maps(decoded), labs)
                        rho, bits = None, None
                    elif uniform is None:
                        allocation = learning.rollout(actor, fitted, features, price, deterministic=True)
                        rho, bits = allocation['rho'], allocation['q']
                        for step, history in enumerate(allocation['history']):
                            command_counts[point][step] += torch.bincount(history['actions'], minlength=9)
                    else:
                        rho, bits = features.new_full((n,), uniform[0]), features.new_full((n,), float(uniform[1]))
                    if method != 'FFT14':
                        metrics, meta = runtime.outcome(source, original, labs, rho, bits, detection=True)
                    if method == 'PPO adaptive E':
                        allocations.extend(dict(split=split, operating_point=point, record_name=row.record_name,
                            tile=i, rho=float(rho[i]), q=int(bits[i])) for i in range(n))
                for kind in ('RD', 'RA'):
                    scores, tp, total = ranked[(method, point)][kind]
                    scores.append(metrics[kind+'_ranked_scores']); tp.append(metrics[kind+'_ranked_tp'])
                    ranked[(method, point)][kind] = scores, tp, total+int(metrics[kind+'_annotation_count'])
                    for j, threshold in enumerate(THRESHOLDS):
                        rate_rows.append(dict(split=split, method=method, operating_point=point, record_name=row.record_name,
                            episode_id=int(row.episode_id), bpcs=float(meta['bpcs']), nmse=float(meta['nmse']),
                            map_kind=kind, threshold=threshold,
                            original_supported=int(metrics[kind+'_supported_by_threshold'][j]),
                            preserved=int(metrics[kind+'_preserved_by_threshold'][j]),
                            missed=int(metrics[kind+'_missed_by_threshold'][j]),
                            unmatched=int(metrics[kind+'_unmatched_by_threshold'][j]),
                            AP=float(metrics[kind+'_AP']), F1=float(metrics[kind+'_F1'][j]),
                            recall=float(metrics[kind+'_recall'][j]), precision=float(metrics[kind+'_precision'][j]),
                            num_labels=int(metrics[kind+'_annotation_count'])))
            if cube_index % 20 == 0 or cube_index == len(rows):
                print(f'Adaptive E matched {split} reports: {cube_index}/{len(rows)} (all operating points)', flush=True)
        for method, point, price, uniform in methods:
            for kind, (scores, tp, total) in ranked[(method, point)].items():
                ap = float(pooled_ap(torch.cat(scores), torch.cat(tp), total))
                population.append(dict(split=split, method=method, operating_point=point, map_kind=kind,
                                       AP=ap, num_labels=total, n_cubes=len(rows)))
            if method == 'PPO adaptive E':
                deltas = learning.commands(runtime.device)
                command_rows.extend(dict(split=split, operating_point=point, step=step+1,
                    delta_rho=float(deltas[action, 0]), delta_q=int(deltas[action, 1]),
                    count=int(command_counts[point][step, action]))
                    for step in range(learning.ROUNDS) for action in range(9))
    csv(out/'tables/rate_sweep.csv', rate_rows)
    csv(out/'tables/allocations.csv', allocations)
    csv(out/'tables/population_AP.csv', population)
    csv(out/'tables/step_commands.csv', command_rows)
    from .arm_e_adaptive_gpu_reports import report
    report(out)
    primary = pd.DataFrame(rate_rows)
    primary = primary[(primary.method == 'PPO adaptive E') & (primary.operating_point == 'price=0.1') & (primary.threshold == THRESHOLDS[0])]
    write_json(out/'ADAPTIVE_ARM_E_COMPLETED.json', dict(completed=True, test_evaluated=include_test,
        profile=runtime.profile, frozen_update=frozen['update'],
        validation_low_misses=int(primary.loc[primary.split == 'validation', 'missed'].sum()),
        test_low_misses=int(primary.loc[primary.split == 'test', 'missed'].sum()) if include_test else None,
        no_zero_miss_guarantee=True, scientific_improvement_not_assumed=True))


def run_check(out, expected, dataset, device, profile):
    """Real TRAIN compact-decode/backend parity, never a learning claim."""
    path = Path(str(out)+'.CHECK.json')
    if path.exists():
        old = read_json(path)
        if old.get('context') != expected or not old.get('passed'):
            raise ValueError('Existing check belongs to changed source/data/device. Use fresh output.')
        print('Existing source/data/device-bound check: PASS', flush=True)
        return
    runtime = Runtime(dataset, device, profile)
    row = runtime.rows[0].iloc[0]
    print('Checking compact GPU decoding and codec on TRAIN:', row.record_name, flush=True)
    cube = load_cube_gpu(row.cube_path, row.to_dict(), device)
    reference = native.load_compact_cube(row.cube_path, expected_metadata=row.to_dict())
    torch.testing.assert_close(cube, torch.as_tensor(reference, device=device), rtol=0, atol=0)
    source = runtime.physics.prepare(cube)
    rho = cube.real.new_full((len(source['tiles']),), .25)
    q = cube.real.new_full(rho.shape, 8.)
    decoded, meta = runtime.physics.encode(source, rho, q)
    torch.testing.assert_close(meta['nmse'], meta['spectral_nmse'], rtol=1e-5, atol=1e-7)
    # Native CPU reference is used ONLY by preflight, not production training.
    from . import ppo_v12_14_codec as old_codec
    native_encoded = old_codec.uniform_encode(reference.astype('complex128'), (.25, 8))
    torch.testing.assert_close(decoded.to(torch.complex128), torch.as_tensor(native_encoded['cube_hat'], device=device), rtol=2e-5, atol=1e-5)
    if float(meta['realized_bits']) != native_encoded['meta']['realized_bits']:
        raise AssertionError('GPU/native accounted bit counts differ.')
    original = runtime.physics.maps(cube)
    metrics = runtime.physics.evaluate(original, original, runtime.labels.loc[runtime.labels.record_name == row.record_name])
    if int(metrics['low_miss']) != 0 or int(metrics['six_miss']) != 0:
        raise AssertionError('Original-vs-original preservation identity failed.')
    # CPU references are confined to this check stage. They are never selected
    # as a fallback by train/report. The tiny engineering parity tests separately
    # verify these reference equations against the historical native processor.
    cpu = GPUPhysics(runtime.cfg, device='cpu', engineering=True)
    cpu_cube = cube.cpu()
    cpu_source = cpu.prepare(cpu_cube)
    parity_rows = []
    variants = [('Original', cube, cpu_cube)]
    for rho_value, q_value in ((.25, 8), (.01, 1)):
        gpu_decoded, gpu_meta = runtime.physics.encode(source,
            cube.real.new_full((len(source['tiles']),), rho_value), cube.real.new_full((len(source['tiles']),), float(q_value)))
        cpu_decoded, cpu_meta = cpu.encode(cpu_source,
            cpu_cube.real.new_full((len(cpu_source['tiles']),), rho_value), cpu_cube.real.new_full((len(cpu_source['tiles']),), float(q_value)))
        torch.testing.assert_close(gpu_decoded.cpu(), cpu_decoded, rtol=1e-5, atol=1e-6)
        if float(gpu_meta['realized_bits']) != float(cpu_meta['realized_bits']):
            raise AssertionError('CUDA/CPU codec byte accounting differs.')
        variants.append((f'rho={rho_value},q={q_value}', gpu_decoded, cpu_decoded))
    gpu_fft, gpu_meta = runtime.physics.full_fft(cube)
    cpu_fft, cpu_meta = cpu.full_fft(cpu_cube)
    torch.testing.assert_close(gpu_fft.cpu(), cpu_fft, rtol=1e-5, atol=1e-6)
    if float(gpu_meta['realized_bits']) != float(cpu_meta['realized_bits']):
        raise AssertionError('CUDA/CPU FFT14 accounting differs.')
    variants.append(('FFT14', gpu_fft, cpu_fft))
    for name, gpu_variant, cpu_variant in variants:
        posts = (runtime.physics.maps(gpu_variant), cpu.maps(cpu_variant))
        for kind in ('RD', 'RA'):
            torch.testing.assert_close(posts[0][kind].cpu(), posts[1][kind], rtol=1e-8, atol=1e-7)
            det_gpu, det_cpu = posts[0]['detectors'][kind], posts[1]['detectors'][kind]
            for key in ('base_peaks', 'valid'):
                if not torch.equal(det_gpu[key].cpu(), det_cpu[key]):
                    raise AssertionError(f'CUDA/CPU {kind} {key} differs for {name}; do not train.')
            for threshold in THRESHOLDS:
                gp = det_gpu['base_peaks'] & (det_gpu['score'] > threshold)
                cp = det_cpu['base_peaks'] & (det_cpu['score'] > threshold)
                if not torch.equal(gp.cpu(), cp):
                    raise AssertionError(f'CUDA/CPU {kind} threshold masks differ for {name}; do not train.')
            parity_rows.append(dict(variant=name, map_kind=kind,
                maximum_map_error_db=float((posts[0][kind].cpu()-posts[1][kind]).abs().max()),
                all_six_peak_masks_equal=True))
        print('  Real TRAIN CPU/CUDA parity:', name, 'PASS', flush=True)
    parity_csv = Path(str(out)+'.PARITY.csv')
    csv(parity_csv, parity_rows)
    # Every trainable parameter and every model/codec computation is CUDA.
    actor = learning.DeltaPolicy(len(learning.NAMES), runtime.device)
    if not all(parameter.is_cuda for parameter in actor.parameters()):
        raise AssertionError('Actor/critic parameter escaped CUDA.')
    loss = sum(parameter.square().sum() for parameter in actor.parameters())
    loss.backward()
    if not all(parameter.grad is not None and parameter.grad.is_cuda for parameter in actor.parameters()):
        raise AssertionError('CUDA gradient residency failed.')
    write_json(path, dict(completed=True, passed=True, context=expected,
        compact_decode_exact=True, codec_native_parity=True, bits_exact=True,
        original_identity=True, actor_critic_gradient_device='CUDA',
        parity_csv=str(parity_csv), parity_csv_sha256=sha(parity_csv),
        training_started=False, test_evaluated=False))
    print('Adaptive E real TRAIN CUDA check: PASS (not proof of learning).', flush=True)


@contextmanager
def exclusive(out):
    lock = Path(str(out)+'.lock')
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        yield
    finally:
        lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('check', 'smoke', 'train', 'report', 'all'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--profile', choices=('full', 'tiny'), default='full')
    args = parser.parse_args(argv)
    torch.set_num_threads(1)  # CPU reference/file glue only; signal kernels stay CUDA.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    out = args.out.resolve()
    print('Auditing source, dataset metadata and frozen episode splits. Mounted disks can take several minutes. '
          'No TEST cube or truth bodies are read by this audit.', flush=True)
    expected = context(args.dataset_root, args.device, args.profile)
    print(json.dumps({key: expected[key] for key in ('revision', 'gpu', 'configuration', 'CPU_exceptions')}, indent=2), flush=True)
    if args.stage == 'check':
        run_check(out, expected, args.dataset_root, args.device, args.profile)
        return
    check = read_json(Path(str(out)+'.CHECK.json'))
    if not check.get('passed') or check.get('context') != expected:
        raise RuntimeError('A matching successful check stage is required.')
    profile = args.profile
    if args.stage == 'smoke':
        out = Path(str(out)+'.smoke')
        profile = 'tiny'
    with exclusive(out):
        if args.stage in ('train', 'all', 'smoke'):
            if out.exists():
                raise FileExistsError('Output already exists; preserve it. No silent overwrite/resume.')
            out.mkdir(parents=True)
            write_json(out/'manifest.json', expected)
            write_json(out/'EXECUTION_CONFIG.json', dict(profile=profile, configuration=configuration(profile),
                smoke_isolated_from_full=args.stage == 'smoke', TEST_evaluation_requested=args.stage != 'smoke'))
            runtime = Runtime(args.dataset_root, args.device, profile)
            for name, frame in zip(('train', 'validation', 'test'), runtime.rows):
                csv(out/'tables'/f'cube_index_{name}.csv', frame.to_dict('records'))
            fitted = prepare_features(runtime, out)
            train(runtime, fitted, out)
        else:
            if read_json(out/'manifest.json') != expected:
                raise ValueError('Source/data/device configuration changed since training.')
            runtime = Runtime(args.dataset_root, args.device, profile)
            fitted = torch.load(out/'feature_normalizer.pt', map_location=args.device, weights_only=False)
        if args.stage in ('all', 'report', 'smoke'):
            final_reports(runtime, out, fitted, include_test=args.stage != 'smoke')
    print('Adaptive Arm E stage complete:', out, flush=True)


if __name__ == '__main__':
    main()
