"""Local CPU pre-flight of the official-test run on REAL RADIal cubes. Not a result.

Run it from the extracted upload bundle, so a file missing from the bundle fails here and not on LMS. The official recordings
are on LMS only, so local VALIDATION cubes stand in for them and no test frame is read. Saved rows cannot be compared on CPU,
whose dither differs from the GPU's. FFTs refuse empty input, as cuFFT does on LMS and the CPU does not.

python -B tools/radial_official_test_gpu/local_real_check.py --dataset-root ... --reference-root ... --out ...
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from unittest import mock

import pandas as pd
import torch

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC))
spec = importlib.util.spec_from_file_location('official_runner', SRC/'tools/run_radial_official_test.py')
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
V17, V13 = R.V17, R.V13


def strict(function):
    def call(value, *args, **kwargs):
        if not value.numel():
            raise RuntimeError('Empty FFT: cuFFT would refuse this.')
        return function(value, *args, **kwargs)
    return call


def test_split():
    """The evaluation split is the frozen split, with the four official recordings in the test role."""
    frozen = pd.read_csv(R.PACKAGE/'GLOBAL_SPLIT_V17.csv').set_index('recording').role
    split = pd.read_csv(R.PACKAGE/'GLOBAL_SPLIT_OFFICIAL.csv').set_index('recording').role
    assert split.loc[list(R.OFFICIAL)].eq('test').all() and set(split.index)-set(frozen.index) == set(R.NEW_RECORDINGS)
    same = [name for name in frozen.index if name not in R.OFFICIAL]
    assert split.loc[same].equals(frozen.loc[same])


def test_rate_counts(spectrum):
    """The recoded rate of a real spectrum is below AdaRadar's complete rate, and its own count is below both."""
    stats = R.common.read_json(R.PACKAGE/V17.FFTRADNET_FILES['config'])['dataset']['statistics']
    for ratio in R.RATIOS:
        _, bits = R.adaradar.compress(spectrum, ratio, 4, stats['input_mean'], stats['input_std'])
        coded = R.recoded_rate(spectrum, ratio, stats)
        assert bits['total_nonzero_only']/spectrum.numel() < coded < bits['total']/spectrum.numel(), ratio


def test_weights():
    """Both networks load into the same official code and differ in every layer."""
    official = torch.load(R.PACKAGE/'fftradnet_state_dict.pt', map_location='cpu', weights_only=True)
    own = torch.load(R.PACKAGE/R.WEIGHTS, map_location='cpu', weights_only=True)
    assert set(official) == set(own)
    floats = [key for key in own if torch.is_floating_point(own[key])]
    assert floats and not any(torch.equal(own[key], official[key]) for key in floats)
    provenance = R.common.read_json(R.PACKAGE/'ADARADAR_WEIGHTS.provenance.json')
    import hashlib
    h = hashlib.sha256()  # tensor contents; file bytes depend on the torch version that saved them
    for key in sorted(own):
        t = own[key].detach().cpu().contiguous()
        h.update(key.encode()); h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode()); h.update(t.numpy().tobytes())
    assert h.hexdigest() == provenance['state_dict_tensor_sha256']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    test_split()
    test_weights()
    print('split and weights PASS', flush=True)
    assignment = R.PACKAGE/'GLOBAL_SPLIT_V17.csv'
    dataset = V17.data.prepare_dataset(args.dataset_root, args.reference_root)
    frames = V17.partition.assign_frozen_partitions(dataset, assignment, bounded=True)
    train, validation = frames[0], frames[1]
    labels = dataset.labels.loc[dataset.labels.annotation_evaluable & dataset.labels.Difficult.eq(0)]
    labelled = validation.loc[validation.record_name.isin(set(labels.record_name))]  # so that both networks are scored on real objects
    counts = labelled.groupby('scene_name').size().sort_values(ascending=False)
    scenes = counts.index.tolist()
    pick = lambda names, count: V13.select_rows(labelled.loc[labelled.scene_name.isin(names)], count)
    R.OFFICIAL, R.REPLAY_CUBES = tuple(scenes[:1]), 1
    rows = dict(train=V13.select_rows(train.loc[train.scene_name.eq(sorted(train.scene_name.unique())[0])], 1), validation=pick(scenes[2:3], 1), test=pd.concat([pick(scenes[:1], 6), pick(scenes[1:2], 2)], ignore_index=True))
    runtime = R.sweep.build_runtime(dataset, assignment, 'cpu', rows={role: frame.assign(role=role) for role, frame in rows.items()})
    with torch.no_grad():
        test_rate_counts(runtime.receive(runtime.cube(runtime.roster('train').iloc[0]))['spectrum'])
    print('rate counts PASS', flush=True)
    out = args.out.resolve()
    scan = dict(local_preflight=True)
    expected = json.loads(json.dumps(R.context(runtime, out, assignment, scan)))
    names = ('fft', 'ifft', 'fft2', 'ifft2', 'fftn', 'ifftn')
    with mock.patch.multiple(torch.fft, **{name: strict(getattr(torch.fft, name)) for name in names}):
        R.run_check(runtime, out, expected, assignment)
        R.run(runtime, out, expected, assignment)
    summary = pd.read_csv(out/'official/SUMMARY.csv')
    seen = pd.read_csv(out/'official/PERCEPTION_SUMMARY.csv')
    cube = pd.read_csv(out/'official/RESULTS_BY_CUBE.csv', low_memory=False)
    with pd.option_context('display.width', 220, 'display.max_columns', 30, 'display.max_rows', 60):
        print(cube.groupby('method', sort=False)[['bpcs', 'bpcs_kt', 'bpcs_nonzero_only', 'bpcs_coded', 'a1_lost_robust', 'a1_new_robust', 'rd_rmse_db', 'adaradar_comp_ratio', 'beta']].mean().round(3).to_string())
        print(seen.loc[seen.subset.eq('all'), ['method', 'frames', 'labels', 'mAP', 'mAR', 'F1']].round(3).to_string(index=False))
    expected_methods = {'original@adaradar_weights', 'adaradar_r12p57_s4@adaradar_weights', 'adaradar_adaptive_own_s4@adaradar_weights', 'ppo_1p45@adaradar_weights', 'baq_2bit', 'ecdq_b1p5', 'ecdq_1p45', 'fixed_r75_q5'}
    assert expected_methods <= set(seen.method), expected_methods-set(seen.method)
    scores = seen.loc[seen.subset.eq('all')].set_index('method')
    assert scores.loc['original'].labels > 0 and scores.loc['original'].F1 > .3 and scores.loc['original@adaradar_weights'].F1 > .3, 'A network failed on the original cube.'
    assert scores.loc['adaradar_r12p57_s4@adaradar_weights'].F1 > .3, 'AdaRadar\'s network failed on its own compression.'
    assert cube.loc[cube.method.str.startswith('ecdq_') & cube.budget_bpcs.notna(), 'within_budget'].astype(bool).all()
    assert len(summary) and (out/'ours/PERCEPTION_SUMMARY.csv').is_file() and (out/'PREDICTIONS.json').is_file() and (out/'SMOKE_COMPLETED.json').is_file()
    print((out/'PREDICTIONS.json').read_text()[:900])
    print('LOCAL PRE-FLIGHT PASS', flush=True)


if __name__ == '__main__':
    main()
