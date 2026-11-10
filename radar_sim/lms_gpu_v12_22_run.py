"""Separate V12.22 LMS CUDA signal-processing runner; no work on import.

The scientific driver is loaded unchanged into a private namespace. Only its
runtime's radar processor is replaced. Codec, policy, Adam, rewards, action
menus, labels, baselines and output families retain their CPU implementation.
No shared module is monkey-patched and no CPU V12.22 file is edited.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from . import ppo_v12_22_run as cpu
from .ppo_v12_22_storage import digest, read_json, sha, write_csv, write_json

ROOT = cpu.ROOT
REVISION = '12.22.0-lms-cuda-signal-processing'
METADATA = ('manifest.csv', 'episode_plan.csv', 'generation_complete.json',
            'audit_report.json', 'radar_spec.json', 'label_schema.json')
INPUTS = (cpu.HISTORY/'tables/cube_index_train_v12_21.csv',
          cpu.HISTORY/'tables/cube_index_validation_v12_21.csv',
          cpu.TRAIN_PANEL, cpu.DEV_PANEL, cpu.RDO_REFERENCE)


def source_hashes():
    result = dict(cpu.source_hashes())
    for path in sorted((ROOT/'radar_sim').glob('lms_gpu_v12_22_*.py')):
        result[str(path.relative_to(ROOT))] = sha(path)
    return result


def gpu_information(device='cuda:0'):
    import torch
    selected = torch.device(device)
    if selected.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('This separate runner requires CUDA. There is no silent CPU fallback.')
    index = torch.cuda.current_device() if selected.index is None else selected.index
    if not 0 <= index < torch.cuda.device_count():
        raise ValueError('Requested CUDA device index does not exist.')
    selected = torch.device('cuda', index)
    with torch.inference_mode():
        probe = torch.ones((8, 8), dtype=torch.float64, device=selected)
        if not bool(torch.isfinite(probe @ probe).all().item()):
            raise RuntimeError('CUDA float64 arithmetic check failed.')
    torch.cuda.synchronize(selected)
    return dict(device=str(selected), torch=torch.__version__, cuda_runtime=torch.version.cuda,
                gpu=torch.cuda.get_device_name(selected),
                capability=list(torch.cuda.get_device_capability(selected)),
                GPU_signal_processing=True, CPU_codec_and_PPO_unchanged=True,
                training_started=False)


def context_signature(device):
    from . import matlab_v44_dataset as v44
    dataset = v44.find_dataset_root(ROOT)
    return dict(revision=REVISION, source_hashes=source_hashes(),
                package_versions=cpu.package_versions(), gpu_environment=gpu_information(device),
                dataset_root=str(dataset),
                dataset_metadata_hashes={name:sha(dataset/'metadata'/name) for name in METADATA},
                input_hashes={str(path.relative_to(ROOT)):sha(path) for path in INPUTS})


def parity_path(out, explicit=None):
    return Path(explicit).resolve() if explicit is not None else Path(out).resolve().with_name(Path(out).name+'.PARITY.json')


def verify_parity(path, device):
    if not Path(path).is_file():
        raise RuntimeError('Run --stage parity with this same output path before preparation or training.')
    value = read_json(path)
    if not value.get('passed') or not value.get('completed') or value.get('test_evaluated') is not False:
        raise RuntimeError('GPU parity did not pass. Preserve its report; do not launch training.')
    if digest(value['context']) != digest(context_signature(device)):
        raise RuntimeError('GPU/data/source environment changed since parity. Use a fresh run and rerun parity.')
    for artifact, checksum in value.get('artifacts', {}).items():
        if sha(Path(path).parent/artifact) != checksum:
            raise ValueError('A frozen parity artifact changed: '+artifact)
    return value


def _gpu_processor(ex, cube, cache, device):
    """Cache only the CUDA processor; preserve the legacy module's identity.

    Inherited plotting functions read their defining module's globals. A copied
    namespace would break the scoped report directories, so runtime processing
    is adapted directly without copying or replacing that module.
    """
    from .lms_gpu_v12_22_processing import TorchPostProcessing
    shape = tuple(map(int, cube.shape))
    if shape not in cache:
        cache[shape] = TorchPostProcessing(121, ex.cfg_for_cube_shape(cube, ex.RADAR_CFG), device=device)
    return cache[shape]


def make_driver(device, parity_report=None):
    """Reuse every scientific block without modifying CPU module globals.

    The private Runtime/manifest factories are backend adapters only. Canonical
    manifest decoration is used in BOTH writing and digesting, so interrupted
    preparation resumes consistently rather than comparing unlike manifests.
    """
    path = ROOT/'radar_sim/ppo_v12_22_run.py'
    spec = importlib.util.spec_from_file_location('radar_sim._lms_v1222_driver', path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    parent_runtime, native_versions = driver.Runtime, driver.package_versions
    info = gpu_information(device)
    driver.REVISION = REVISION
    driver.source_hashes = source_hashes

    def versions():
        return {**native_versions(), 'torch':info['torch'], 'cuda_runtime':info['cuda_runtime'],
                'gpu_model':info['gpu'], 'cuda_device':info['device']}

    def decorate(value):
        if isinstance(value, dict) and value.get('revision') == REVISION and 'approved_records' in value:
            return {**value, 'cpu_backend':False, 'backend':'CUDA radar signal processing; CPU codec/PPO',
                    'gpu_environment':info, 'gpu_parity_report':str(parity_report),
                    'gpu_parity_sha256':sha(parity_report) if parity_report is not None else None,
                    'scientific_protocol_unchanged':True}
        return value

    driver.package_versions = versions
    driver.digest = lambda value: digest(decorate(value))
    driver.write_json = lambda path, value: write_json(path, decorate(value))

    class GPURuntime(parent_runtime):
        def __init__(self, root, *, verify=True):
            super().__init__(root, verify=verify)
            expected = self.manifest.get('gpu_parity_sha256')
            if parity_report is None or expected != sha(parity_report):
                raise RuntimeError('This GPU experiment requires its original successful parity report.')

        def post(self, decoded, row, kind, **metadata):
            processor = _gpu_processor(self.ex, decoded, self.processors, device)
            before = copy.deepcopy(processor.stats)
            started = time.perf_counter()
            result = self.ex.postprocess_cube(decoded, processor)
            event = dict(kind=kind, record_name=str(row.record_name),
                         seconds=time.perf_counter()-started, **metadata,
                         backend='torch_cuda_signal_processing', device=str(processor.device),
                         gpu_calls=int(processor.stats['gpu_calls']-before['gpu_calls']),
                         gpu_math_seconds=float(processor.stats['fft_beamforming_seconds']-before['fft_beamforming_seconds']))
            if event['gpu_calls'] != 1:
                raise RuntimeError('The CUDA processor did not execute; refusing an incorrectly labelled GPU run.')
            with (self.root/'physical_evaluations.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(event, allow_nan=False)+'\n')
                stream.flush()
            return result

    driver.Runtime = GPURuntime
    return driver


def _compare_posts(ex, original_cpu, original_gpu, post_cpu, post_gpu, labels):
    result = {}
    for axis in ('ranges', 'vel', 'angle_axis_deg'):
        if not np.array_equal(post_cpu[axis], post_gpu[axis]):
            raise AssertionError('CPU/CUDA physical axes differ: '+axis)
    if not np.array_equal(post_cpu['detections'], post_gpu['detections']):
        raise AssertionError('Native GO-CFAR detection decisions changed on the parity case.')
    if len(post_cpu['clusters']) != len(post_gpu['clusters']):
        raise AssertionError('Native DBSCAN cluster count changed.')
    for left, right in zip(post_cpu['clusters'], post_gpu['clusters']):
        if not np.array_equal(left['index'], right['index']):
            raise AssertionError('Native cluster peak position changed.')
    np.testing.assert_allclose(post_cpu['point_cloud'], post_gpu['point_cloud'], rtol=1e-8, atol=1e-8)
    for kind in ('RD', 'RA'):
        left, ranges, second = ex._map_triplet_v12_6(post_cpu, kind)
        right, _, _ = ex._map_triplet_v12_6(post_gpu, kind)
        error = float(np.max(np.abs(left-right)))
        result[kind+'_max_absolute_map_error_db'] = error
        if error > 1e-6:
            raise AssertionError(f'{kind} CPU/CUDA map error {error} dB exceeds 1e-6 dB.')
        gate = ex.CFAR_LABEL_GATE_RD_V8 if kind == 'RD' else ex.CFAR_LABEL_GATE_RA_V8
        factors = ex.V12_7_RD_PRESERVATION_THRESHOLD_FACTORS if kind == 'RD' else ex.V12_7_BASE_CFAR_THRESHOLD_FACTORS
        summaries = []
        for original, method in ((original_cpu, left), (original_gpu, right)):
            reference, _, _ = ex._map_triplet_v12_6(original, kind)
            cache = ex.precompute_original_cfar_count_sweep_v11_2(reference, labels, ranges, second, kind, factors, gate)
            summaries.append(ex.evaluate_cfar_count_sweep_method_v11_2(cache, method))
        pd.testing.assert_frame_equal(summaries[0], summaries[1], check_exact=True)
        for power in (left, right):
            if not np.isfinite(power).all():
                raise AssertionError('Nonfinite CUDA postprocessing map.')
        masks = [ex.precompute_nested_cfar_peak_masks_v11_2(ex.as_linear_power(value), [1.25])[1.25]
                 for value in (left, right)]
        np.testing.assert_array_equal(masks[0]['peaks'], masks[1]['peaks'])
        np.testing.assert_allclose(masks[0]['cfar_score'], masks[1]['cfar_score'], rtol=1e-8, atol=1e-8)
        result[kind+'_integer_sweep_and_peak_positions_equal'] = True
    return result


def run_parity(out, device, report=None, *, resume=False):
    """Two fixed TRAIN cubes x four variants, no training and no TEST reads."""
    path = parity_path(out, report)
    if path.exists():
        if not resume:
            raise FileExistsError('Parity report already exists; use --resume or a fresh run path.')
        return verify_parity(path, device)
    context = context_signature(device)
    from . import matlab_v44_dataset as v44
    from .ppo_v12_21_pipeline import load_primitives
    from .ppo_priority2_scoped_labels_v2 import load_record_labels
    from .lms_gpu_v12_22_processing import TorchPostProcessing
    dataset = Path(context['dataset_root'])
    inventory = v44.validated_manifest(dataset)
    rows = cpu._relocate(pd.read_csv(INPUTS[0]), inventory).set_index('record_name')
    names = pd.read_csv(cpu.TRAIN_PANEL).record_name.head(2).tolist()
    rows = rows.loc[names].reset_index()
    tables = path.with_name(path.stem+'_tables')
    ex = load_primitives(tables/'primitive_reports')
    labels = load_record_labels(dataset, ex.RADAR_CFG, names)
    processors = {}
    outcomes = []
    try:
        for _, row in rows.iterrows():
            cube = ex.load_cube_v6_1000(row)
            labs = ex.labels_for_cube_v6_1000(labels, row)
            cfg = ex.cfg_for_cube_shape(cube, ex.RADAR_CFG)
            native = ex._postprocessor_for_cube_v11(cube, processors)
            accelerated = TorchPostProcessing(121, cfg, device=device)
            originals = []
            timings = []
            for processor in (native, accelerated):
                started = time.perf_counter()
                originals.append(ex.postprocess_cube(cube, processor))
                timings.append(time.perf_counter()-started)
            variants = [('Original', cube, None)]
            for action in ((.12, 6), (1., 12)):
                decoded, _, meta = ex.compress_uniform_action_v11(cube, action)
                variants.append((f'Uniform_rho{action[0]}_q{action[1]}', decoded, meta))
            fft, _, meta = ex.compress_cube_fft_full_support_multi(cube, {'FFT14':14})['FFT14']
            variants.append(('FFT14', fft, meta))
            for name, decoded, meta in variants:
                print(f'V12.22 CUDA parity TRAIN {row.record_name}: {name}', flush=True)
                if name == 'Original':
                    reference, candidate = originals
                    cpu_seconds, gpu_seconds = timings
                else:
                    started = time.perf_counter(); reference = ex.postprocess_cube(decoded, native)
                    cpu_seconds = time.perf_counter()-started
                    started = time.perf_counter(); candidate = ex.postprocess_cube(decoded, accelerated)
                    gpu_seconds = time.perf_counter()-started
                metrics = _compare_posts(ex, *originals, reference, candidate, labs)
                outcomes.append(dict(record_name=row.record_name, variant=name,
                    source_split='TRAIN', passed=True, cpu_postprocessing_seconds=cpu_seconds,
                    cuda_postprocessing_seconds=gpu_seconds, decoded_IQ_identical_input=True,
                    accounted_bpcs=32. if meta is None else meta['realized_bpcs'], **metrics))
            # Label-free CFAR feature coverage and definitions are also identical.
            from . import ppo_v12_22_codec as codec
            source = codec.prepare_source(ex, cube, codec.PROPOSED_ACTIONS)
            features = [codec.feature_bundle(ex, original, source) for original in originals]
            np.testing.assert_array_equal(features[0]['covered'], features[1]['covered'])
            np.testing.assert_allclose(features[0]['values'], features[1]['values'], rtol=1e-6, atol=1e-6)
        table = tables/'CPU_CUDA_PARITY.csv'
        write_csv(table, outcomes)
        result = dict(completed=True, passed=True, context=context, test_evaluated=False,
            training_started=False, cases=len(outcomes), fixed_TRAIN_records=names,
            preservation_and_unmatched_counts_equal=True, bitwise_cross_platform_guarantee=False,
            benchmark_scope='Postprocessing only; not a full training speedup claim.',
            artifacts={str(table.relative_to(path.parent)):sha(table)})
        write_json(path, result)
        print('CUDA parity PASS: '+str(path), flush=True)
        return result
    except Exception as exc:
        write_csv(tables/'PARTIAL_CPU_CUDA_PARITY.csv', outcomes)
        write_json(path, dict(completed=False, passed=False, context=context,
            test_evaluated=False, training_started=False, error=f'{type(exc).__name__}: {exc}'))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('check','parity','prepare','train','diagnostics','report','all'))
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--parity-report', type=Path)
    parser.add_argument('--profile', choices=('full','tiny'), default='full')
    parser.add_argument('--arms', nargs='+', choices=tuple(cpu.ARMS), default=list(cpu.ARMS))
    parser.add_argument('--seeds', nargs='+', type=int, default=list(cpu.DEFAULT_SEEDS))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--acknowledge-scope', action='store_true')
    parser.add_argument('--interaction-arm', choices=('D','E','F'))
    args = parser.parse_args(argv)
    info = gpu_information(args.device)
    if args.stage == 'check':
        from .ppo_priority2_scoped_labels_v2 import verify_native_label_loop_parity
        required = {str(path.relative_to(ROOT)):path.is_file() for path in INPUTS}
        if not all(required.values()):
            raise FileNotFoundError('Upload the five frozen reference CSVs: '+str(required))
        result = dict(revision=REVISION, gpu_environment=info, configuration=cpu.profile(args.profile),
            reference_files=required, native_labels=verify_native_label_loop_parity(),
            source_hashes=source_hashes(), training_started=False, test_evaluation=False,
            backend='CUDA radar processing + unchanged CPU codec/PPO; no automatic fallback')
        print(json.dumps(result, indent=2), flush=True)
        return result
    path = parity_path(args.out, args.parity_report)
    if args.stage == 'parity':
        with cpu._exclusive_run(args.out, args.resume):
            return run_parity(args.out, info['device'], path, resume=args.resume)
    verify_parity(path, info['device'])
    driver = make_driver(info['device'], path)
    print('V12.22 LMS CUDA radar processor active; native CPU codec/PPO unchanged.', flush=True)
    return driver.run_stage(args.stage, args.out, args.profile, args.arms, args.seeds,
        resume=args.resume, acknowledge_scope=args.acknowledge_scope, interaction_arm=args.interaction_arm)


if __name__ == '__main__':
    main()
