"""Separate full-data V12.22 Arm E: 600 TRAIN / 200 VAL / 200 frozen TEST.

Only execution scope, storage and backend adaptation differ from the original
Arm E. The categorical policy, codec, conditional credit, calibrated compatible
auxiliary, objective and detector are reused unchanged. No work on import.
Use --target-gpu lms for the existing Quadro RTX 4000 workspace, or l4 for
the separate L4 deployment. The target is recorded in the immutable context.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
from types import FunctionType

import numpy as np
import pandas as pd

from . import lms_gpu_v12_22_run as gpu
from . import ppo_v12_22_run as cpu
from . import ppo_v12_22_codec as codec
from .ppo_v12_22_storage import (
    actor_hash, digest, load_tree, read_json, receipt, save_tree, sha,
    verify_receipt, write_csv, write_json,
)

ROOT = Path(__file__).resolve().parents[1]
REVISION = '12.22.3-arm-E-full600-frozen-test-CUDA'
SEED = 20261022
SPLIT_SEED = 20260825
TABLES = ROOT/'outputs/v12_19_v44_preservation_balanced_ppo_full_train600_val200_test200/tables'
INPUTS = tuple(TABLES/f'cube_index_{name}_v12.csv' for name in ('train', 'validation', 'test'))
RDO_REFERENCE = TABLES/'task_rdo_action_summary_v12.csv'
EPISODES = dict(TRAIN=[1, 3, 4, 5, 7, 8, 10, 12, 14, 15, 17, 20, 21, 22, 24],
                VALIDATION=[2, 9, 13, 18, 23], TEST=[6, 11, 16, 19, 25])


def configuration(profile='full'):
    if profile not in ('full', 'tiny'):
        raise ValueError('Use full or tiny profile.')
    value = dict(cpu.profile(profile))
    if profile == 'full':
        value.update(train=600, validation=200, test=200, updates=240,
                     checkpoints=[0, 60, 120, 180, 240])
    else:
        value['test'] = 2
    return value


def source_hashes():
    result = dict(gpu.source_hashes())
    for name in ('arm_e_full_v12_22_run.py', 'arm_e_full_v12_22_reports.py'):
        result['radar_sim/'+name] = sha(ROOT/'radar_sim'/name)
    return result


def storage_check(path, minimum_gib=8.):
    if not np.isfinite(minimum_gib) or minimum_gib < 0:
        raise ValueError('Minimum free GiB must be finite and nonnegative.')
    ancestor = Path(path).resolve()
    while not ancestor.exists():
        if ancestor == ancestor.parent:
            raise FileNotFoundError('No existing output parent.')
        ancestor = ancestor.parent
    free = shutil.disk_usage(ancestor).free/1024**3
    if free < minimum_gib:
        raise OSError(f'Only {free:.2f} GiB free; need {minimum_gib:.2f} GiB headroom. '
                      'No files were deleted. Preserve this run and free approved space.')
    return dict(filesystem_path=str(ancestor), free_gib=free, minimum_free_gib=minimum_gib)


def planned_populations(dataset, profile='full'):
    """Restore all frames of the historical episodes, never change membership."""
    from . import matlab_v44_dataset as v44
    inventory = v44.validated_manifest(dataset)
    frames = [cpu._relocate(pd.read_csv(path), inventory) for path in INPUTS]
    union = set()
    for name, frame, count in zip(EPISODES, frames, (600, 200, 200)):
        if (len(frame) != count or not frame.record_name.is_unique
                or sorted(map(int, frame.episode_id.unique())) != EPISODES[name]
                or not frame.groupby('episode_id').size().eq(40).all()
                or any(not np.array_equal(np.sort(part.frame_in_scene), np.arange(40))
                       for _, part in frame.groupby('episode_id'))):
            raise ValueError('Historical full population changed: '+name)
        if union & set(frame.record_name):
            raise ValueError('TRAIN/VAL/TEST cube overlap.')
        union.update(frame.record_name)
    if union != set(inventory.record_name):
        raise ValueError('Full split does not cover the complete audited dataset.')
    if profile == 'tiny':
        frames = [diverse_rows(frame, 2) for frame in frames]
    elif profile != 'full':
        raise ValueError('Unknown profile.')
    return frames


def diverse_rows(frame, count):
    """Round-robin over episodes, not a consecutive-frame smoke panel."""
    groups = [part.sort_values('frame_in_scene') for _, part in
              frame.groupby('episode_id', sort=True)]
    order = [part.iloc[[i]] for i in range(max(map(len, groups))) for part in groups if i < len(part)]
    return pd.concat(order, ignore_index=True).iloc[:min(count, len(frame))].copy()


def scoped_labels(dataset, cfg, rows, *, stage, frozen=None):
    """Bind native per-frame label code to this runner's explicit permission set."""
    from . import ppo_priority2_scoped_labels_v2 as native
    if stage not in ('TRAIN_VALIDATION', 'PARITY_TRAIN', 'FINAL_TEST'):
        raise ValueError('Unknown label permission stage.')
    allowed_episodes = set(EPISODES['TEST'] if stage == 'FINAL_TEST' else
                           EPISODES['TRAIN'] if stage == 'PARITY_TRAIN' else
                           EPISODES['TRAIN']+EPISODES['VALIDATION'])
    if (rows.empty or not rows.record_name.is_unique
            or not set(map(int, rows.episode_id)) <= allowed_episodes):
        raise ValueError('Out-of-scope label request.')
    if stage == 'FINAL_TEST' and (frozen is None or not Path(frozen).is_file()
                                 or read_json(frozen).get('arm') != 'E'):
        raise RuntimeError('Freeze validation selection before reading TEST labels.')
    permitted = rows.set_index('record_name')
    namespace = {**vars(native), '_permitted_records': lambda: permitted}
    loader = FunctionType(native.load_record_labels.__code__, namespace,
                          name='load_arm_e_explicit_labels')
    labels = loader(dataset, cfg, rows.record_name.astype(str).tolist())
    labels.attrs['scoped_loading'].update(stage=stage,
        TEST_truth_and_measurement_bodies_read=stage == 'FINAL_TEST',
        native_per_frame_code_unchanged=True)
    return labels


def validate_target_gpu(info, target_gpu):
    """Deployment identity only: never change the scientific configuration."""
    names = {'l4': 'NVIDIA L4', 'lms': 'Quadro RTX 4000'}
    if target_gpu not in names:
        raise ValueError('Unknown GPU deployment target: '+str(target_gpu))
    detected = str(info['gpu'])
    # Some driver releases prefix the Quadro name with NVIDIA.
    matches = (detected == names[target_gpu] or
               (target_gpu == 'lms' and detected == 'NVIDIA '+names[target_gpu]))
    if not matches:
        raise RuntimeError(f'GPU target {target_gpu!r} requires {names[target_gpu]}; '
                           f'detected {detected}. No training was started.')


def context(dataset, device, seed, profile, minimum_gib, target_gpu='l4'):
    info = gpu.gpu_information(device)
    validate_target_gpu(info, target_gpu)
    return dict(revision=REVISION, dataset_root=str(Path(dataset).resolve()),
        target_gpu=target_gpu,
        source_hashes=source_hashes(), package_versions=cpu.package_versions(),
        gpu_environment=info, seed=int(seed), profile=profile,
        configuration=configuration(profile), minimum_free_gib=float(minimum_gib),
        input_hashes={str(path.relative_to(ROOT)): sha(path) for path in (*INPUTS, RDO_REFERENCE)},
        dataset_metadata_hashes={name: sha(Path(dataset)/'metadata'/name) for name in gpu.METADATA})


def parity_path(out):
    return Path(str(Path(out).resolve())+'.PARITY.json')


def verify_parity(out, expected):
    path = parity_path(out)
    value = read_json(path)
    if (not value.get('completed') or not value.get('passed')
            or digest(value.get('context')) != digest(expected)):
        raise RuntimeError('Run parity for this exact source/data/GPU/output context first.')
    verify_receipt(path)
    return value


def run_parity(out, expected, *, resume=False):
    """Two TRAIN cubes, originals and three compressed variants, no optimizer."""
    path = parity_path(out)
    if path.exists():
        if not resume:
            raise FileExistsError('Parity already exists; use --resume to verify it.')
        return verify_parity(out, expected)
    from .ppo_v12_21_pipeline import load_primitives
    from .lms_gpu_v12_22_processing import TorchPostProcessing
    from . import matlab_v44_dataset as v44
    data = Path(expected['dataset_root'])
    frames = planned_populations(data, expected['profile'])
    rows = diverse_rows(frames[0], 2)
    ex = load_primitives(Path(str(out)+'.PARITY_tables')/'primitive_reports')
    labels = scoped_labels(data, ex.RADAR_CFG, rows, stage='PARITY_TRAIN')
    results = []
    processors = {}
    for _, row in rows.iterrows():
        cube = v44.load_compact_cube(row.cube_path, expected_metadata=row.to_dict())
        labs = ex.labels_for_cube_v6_1000(labels, row)
        native = ex._postprocessor_for_cube_v11(cube, processors)
        accelerated = TorchPostProcessing(121, ex.cfg_for_cube_shape(cube, ex.RADAR_CFG),
                                         device=expected['gpu_environment']['device'])
        originals = [ex.postprocess_cube(cube, processor) for processor in (native, accelerated)]
        variants = [('Original', cube)]
        for action in ((.12, 6), (1., 12)):
            decoded, _, _ = ex.compress_uniform_action_v11(cube, action)
            variants.append((str(action), decoded))
        variants.append(('FFT14', ex.compress_cube_fft_full_support_multi(cube, {'FFT14':14})['FFT14'][0]))
        for name, decoded in variants:
            storage_check(path.parent, expected['minimum_free_gib'])
            posts = originals if name == 'Original' else [ex.postprocess_cube(decoded, p) for p in (native, accelerated)]
            result = gpu._compare_posts(ex, *originals, *posts, labs)
            results.append(dict(record_name=row.record_name, variant=name, **result))
            print(f'Arm E full CUDA parity TRAIN {row.record_name}: {name} PASS', flush=True)
        source = codec.prepare_source(ex, cube, codec.PROPOSED_ACTIONS)
        features = [codec.feature_bundle(ex, post, source) for post in originals]
        np.testing.assert_array_equal(features[0]['covered'], features[1]['covered'])
        np.testing.assert_allclose(features[0]['values'], features[1]['values'], rtol=1e-6, atol=1e-6)
        del cube, variants, source, features, originals, posts
        gc.collect()
    table = Path(str(out)+'.PARITY_tables')/'CPU_CUDA_PARITY.csv'
    write_csv(table, results)
    receipt(path, [table], passed=True, context=expected, training_started=False,
            test_evaluated=False, cases=len(results), full_GPU_training_claim=False)
    return verify_parity(out, expected)


def make_driver(expected, parity):
    """Private execution adapter; no public module globals are monkey-patched."""
    driver = gpu.make_driver(expected['gpu_environment']['device'], parity)
    parent = driver.Runtime
    driver.REVISION = REVISION
    driver.source_hashes = source_hashes
    driver.profile = configuration

    class Runtime(parent):
        def __init__(self, root, *, verify=True):
            self.root = Path(root).resolve()
            self.manifest = read_json(self.root/'manifest.json')
            if (self.manifest.get('revision') != REVISION
                    or self.manifest.get('test_evaluation') is not False
                    or self.manifest.get('arms') != ['E']):
                raise ValueError('Not this isolated Arm E experiment.')
            if verify:
                if digest(self.manifest['context']) != digest(expected):
                    raise ValueError('Source/data/GPU/configuration changed; use a new run.')
                verify_parity(self.root.parent, expected)
            os.environ['MATLAB_RADAR_DATASET_ROOT'] = self.manifest['dataset_root']
            os.environ['RDO_RL_SEED'] = str(SPLIT_SEED)
            from .ppo_v12_21_pipeline import load_primitives
            self.ex = load_primitives(self.root/'primitive_reports')
            if (tuple(map(tuple, self.ex.ACTION_LIBRARY_V11)) != codec.ORIGINAL_ACTIONS
                    or tuple(self.ex._nearest_training_thresholds_v11()) != (1.25, 2., 4., 8., 16., 32.)):
                raise ValueError('Native detector/action contract changed.')
            self.config = self.manifest['configuration']
            self.train_rows, self.validation_rows, self.train_probe, self.dev_probe = [
                pd.read_csv(self.root/'populations'/f'{name}.csv')
                for name in ('TRAIN', 'VALIDATION', 'TRAIN_PROBE', 'DEV_PROBE')]
            allowed = pd.concat([self.train_rows, self.validation_rows], ignore_index=True)
            if (not allowed.record_name.is_unique
                    or set(self.train_rows.episode_id) & set(self.validation_rows.episode_id)
                    or set(allowed.record_name) != set(self.manifest['approved_records'])):
                raise ValueError('Training permission/episode split changed.')
            self.labels = scoped_labels(self.manifest['dataset_root'], self.ex.RADAR_CFG,
                                        allowed, stage='TRAIN_VALIDATION')
            self.processors = {}

        def post(self, decoded, row, kind, **metadata):
            storage_check(self.root, expected['minimum_free_gib'])
            return super().post(decoded, row, kind, **metadata)

        def original(self, row, cube):
            # Do not persist large full-resolution RD/RA arrays for every cube.
            return self.post(cube, row, 'original_processing')

        def baseline(self, row, cube, original, method):
            return baseline(self, row, cube, original, method)

    driver.Runtime = Runtime
    return driver


def baseline(runtime, row, cube, original, method):
    """Same independent frozen comparator, without a disk-heavy map cache."""
    ex = runtime.ex
    if method == 'CalibratedRDO':
        decoded, _, meta = ex.compress_uniform_action_v11(cube, tuple(runtime.manifest['fixed_rdo_action']))
    elif method == 'FFT14':
        decoded, _, meta = ex.compress_cube_fft_full_support_multi(cube, {'FFT14':14})['FFT14']
    else:
        raise ValueError('Unknown independent baseline.')
    post = runtime.post(decoded, row, 'independent_baseline', method=method)
    quality = ex._cube_quality_row_v12_6(row.record_name, row.scene_name, method,
                                       cube, decoded, meta, original, post)
    return dict(post=post, meta=meta, quality=quality)


def prepare(driver, out, expected, *, resume=False):
    root = Path(out)/'training'
    complete = root/'PREPARE_COMPLETED.json'
    if complete.is_file():
        if not resume:
            raise FileExistsError('Existing prepared experiment; use --resume.')
        verify_receipt(complete)
        return driver.Runtime(root)
    if root.exists() and not resume:
        raise FileExistsError('Partial experiment; use --resume, never overwrite.')
    root.mkdir(parents=True, exist_ok=True)
    frames = planned_populations(expected['dataset_root'], expected['profile'])
    cfg = configuration(expected['profile'])
    frozen = pd.read_csv(RDO_REFERENCE)
    chosen = frozen.loc[frozen.selected_fixed_action.astype(str).str.lower().eq('true')]
    if len(chosen) != 1:
        raise ValueError('Frozen RDO reference must identify exactly one action.')
    fixed = [float(chosen.iloc[0].rho), int(chosen.iloc[0].q)]
    if not np.allclose(fixed, codec.ORIGINAL_ACTIONS[int(chosen.iloc[0].action_index)]):
        raise ValueError('Frozen independent comparator index/tuple mismatch.')
    value = dict(revision=REVISION, context=expected, profile=expected['profile'], configuration=cfg,
        arms=['E'], seeds=[expected['seed']], split_seed=SPLIT_SEED,
        dataset_root=expected['dataset_root'], source_hashes=source_hashes(),
        input_hashes=expected['input_hashes'], package_versions=expected['package_versions'],
        dataset_metadata_hashes=expected['dataset_metadata_hashes'],
        approved_records=pd.concat(frames[:2]).record_name.tolist(),
        fixed_rdo_action=fixed, test_evaluation=False, rate_cap=None,
        no_imitation=True, no_refinement=True, rate_reward_weight=0., nmse_reward_weight=0.,
        objective='UNCHANGED V12.21 PerceptionObjective',
        provisional_nmse_failure=.05, preferred_nmse=.01,
        seed_selection='Prior validation preservation-first LOW/SIX/U+/rate ordering; not TEST.',
        single_seed_confirmation_not_multiseed_evidence=True,
        generalization_scope='held_out_traffic_episodes_of_seen_templates',
        historical_test_previously_used_in_research=True, raw_map_disk_cache=False)
    manifest = root/'manifest.json'
    if manifest.is_file() and digest(read_json(manifest)) != digest(value):
        raise ValueError('Partial preparation identity changed.')
    write_json(manifest, value)
    populations = [('TRAIN', frames[0]), ('VALIDATION', frames[1]), ('TEST', frames[2]),
                   ('TRAIN_PROBE', diverse_rows(frames[0], cfg['train_probes'])),
                   ('DEV_PROBE', diverse_rows(frames[1], cfg['dev_probes']))]
    for name, frame in populations:
        write_csv(root/'populations'/f'{name}.csv', frame)
    write_csv(root/'frozen_RDO_reference.csv', frozen)
    runtime = driver.Runtime(root)
    driver.prepare_models(runtime)  # Native TRAIN normalizer and coefficient calibration.
    artifacts = [manifest, root/'frozen_RDO_reference.csv', *sorted((root/'populations').glob('*.csv'))]
    artifacts += [path for path in sorted((root/'preparation').rglob('*')) if path.is_file()]
    receipt(complete, artifacts, stage='prepare', test_evaluated=False,
            old_menu_comparison_not_repeated=True, no_action_preselection=True)
    return runtime


def frozen_selection(root, seed):
    job = Path(root)/'training/jobs'/f'E_seed{seed}'
    verify_receipt(job/'TRAIN_COMPLETED.json')
    status = read_json(job/'job_status.json')
    if not status.get('completed') or status.get('arm') != 'E' or status.get('seed') != seed:
        raise RuntimeError('Complete Arm E validation-only training first.')
    return dict(arm='E', seed=int(seed), selected_update=int(status['selected_update']),
        selected_initialization=bool(status['selected_initialization']),
        model_sha256=sha(job/'models/selected.npz'), status_sha256=sha(job/'job_status.json'),
        normalizer_sha256=sha(job/'models/normalizer.json'),
        selected_using='validation_only', test_used_for_selection=False)


class EvaluationRuntime:
    """Explicitly scoped frozen evaluation, never an optimizer runtime."""
    features = cpu.Runtime.features
    measure = cpu.Runtime.measure
    baseline = baseline

    def __init__(self, training, rows, labels, root, split):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ex, self.rows, self.labels = training.ex, rows, labels
        self.manifest = {**training.manifest, 'approved_records':rows.record_name.tolist(),
                         'test_evaluation':split == 'test', 'evaluation_split':split}
        self.processors = training.processors
        self._training = training

    def load_cube(self, row):
        if str(row.record_name) not in self.manifest['approved_records']:
            raise ValueError('Unapproved frozen evaluation cube.')
        return self.ex.load_cube_v6_1000(row)

    def post(self, cube, row, kind, **metadata):
        if str(row.record_name) not in self.manifest['approved_records']:
            raise ValueError('Out-of-scope frozen evaluation.')
        # Call native CUDA post without its TRAIN-only cube loader.
        return self._training.post(cube, row, kind, **metadata,
                                   evaluation_split=self.manifest['evaluation_split'])

    def original(self, row, cube):
        return self.post(cube, row, 'original_processing')


def evaluate(driver, runtime, out, split, *, save_reconstructed=0):
    from .arm_e_full_v12_22_reports import export_population
    selected = frozen_selection(out, runtime.manifest['seeds'][0])
    freeze = Path(out)/'SELECTION_FROZEN_BEFORE_TEST.json'
    if freeze.is_file() and read_json(freeze) != selected:
        raise ValueError('Selected model changed after selection was frozen.')
    if not freeze.is_file():
        write_json(freeze, selected)  # Commit before any TEST body access.
    actor, norm, actions, cfar, job = driver.load_job(runtime, 'E', selected['seed'], True)
    rows = pd.read_csv(runtime.root/'populations'/('TEST.csv' if split == 'test' else 'VALIDATION.csv'))
    labels = scoped_labels(runtime.manifest['dataset_root'], runtime.ex.RADAR_CFG,
                            rows, stage='FINAL_TEST', frozen=freeze) if split == 'test' else runtime.labels
    evaluation = EvaluationRuntime(runtime, rows, labels, Path(out)/('final_'+split), split)
    before = actor_hash(actor)
    result = export_population(evaluation, actor, norm, actions, cfar, job, split,
                               save_reconstructed=save_reconstructed)
    if before != actor_hash(actor):
        raise AssertionError('Frozen evaluation changed the actor.')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('check','parity','smoke','train','test','all'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--target-gpu', choices=('l4', 'lms'), default='l4',
                        help='Execution target only; LMS uses the existing Quadro RTX 4000.')
    parser.add_argument('--seed', type=int, default=SEED, choices=(SEED,))
    parser.add_argument('--profile', choices=('full','tiny'), default='full')
    parser.add_argument('--min-free-gib', type=float, default=8.)
    parser.add_argument('--save-reconstructed', type=int, default=0, choices=(0,1,2))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if args.stage == 'smoke':
        out = Path(str(out)+'.smoke')
        args.profile = 'tiny'
    storage = storage_check(out, args.min_free_gib)
    from . import matlab_v44_dataset as v44
    dataset = v44.find_dataset_root(ROOT, args.dataset_root)
    os.environ['MATLAB_RADAR_DATASET_ROOT'] = str(dataset)
    frames = planned_populations(dataset, args.profile)
    expected = context(dataset, args.device, args.seed, args.profile, args.min_free_gib,
                       target_gpu=args.target_gpu)
    from .ppo_priority2_scoped_labels_v2 import verify_native_label_loop_parity
    report = dict(revision=REVISION, stage=args.stage, output=str(out), arm='E', seed=args.seed,
        populations={name:len(frame) for name, frame in zip(EPISODES, frames)},
        configuration=configuration(args.profile), storage=storage,
        gpu=expected['gpu_environment'], native_labels=verify_native_label_loop_parity(),
        training_started=False, test_evaluated=False,
        full_actor_codec_GPU_acceleration=False, no_dataset_copy=True)
    print(json.dumps(report, indent=2), flush=True)
    if args.stage == 'check':
        return report
    with cpu._exclusive_run(out, args.resume):
        if args.stage in ('parity', 'smoke'):
            run_parity(out, expected, resume=args.resume)
            if args.stage == 'parity':
                return
        verify_parity(out, expected)
        out.mkdir(parents=True, exist_ok=True)
        driver = make_driver(expected, parity_path(out))
        runtime = prepare(driver, out, expected, resume=args.resume)
        try:
            if args.stage in ('smoke','train','all'):
                driver.train_job(runtime, 'E', args.seed, resume=args.resume)
                evaluate(driver, runtime, out, 'validation', save_reconstructed=args.save_reconstructed)
                driver.rate_reports(runtime)  # Fixed validation diagnostic, never action pruning.
            if args.stage in ('test','all'):
                evaluate(driver, runtime, out, 'test', save_reconstructed=args.save_reconstructed)
                receipt(out/'ARM_E_FULL_COMPLETED.json',
                    [out/'training/jobs'/f'E_seed{args.seed}'/'TRAIN_COMPLETED.json',
                     out/'SELECTION_FROZEN_BEFORE_TEST.json',
                     out/'final_validation/REPORT_COMPLETED.json', out/'final_test/REPORT_COMPLETED.json'],
                    arm='E', seed=args.seed, train=configuration(args.profile)['train'],
                    validation=configuration(args.profile)['validation'], test=configuration(args.profile)['test'],
                    test_evaluated=True, scientific_superiority_not_assumed=True)
        finally:
            driver.export_execution_cost(runtime)
    print('ARM E STAGE COMPLETE: '+str(out), flush=True)


if __name__ == '__main__':
    main()
