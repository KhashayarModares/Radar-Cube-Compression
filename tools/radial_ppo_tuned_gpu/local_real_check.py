"""Local CPU pre-flight of the complete tuned PPO run on REAL RADIal cubes. Not a result.

Tiny counts, one short pass per arm. Pseudo roles come from the local TRAIN and VALIDATION recordings; the real TEST partition
is never touched. Run it from the extracted upload bundle, so a file missing from the bundle fails here and not on LMS.
FFTs refuse empty input, as cuFFT does on LMS and the CPU does not.

python -B tools/radial_ppo_tuned_gpu/local_real_check.py --dataset-root ... --reference-root ... --out ...
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
spec = importlib.util.spec_from_file_location('ppo_tuned_runner', SRC/'tools/run_radial_ppo_tuned.py')
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
V17, V13 = R.V17, R.V13


def strict(function):
    def call(value, *args, **kwargs):
        if not value.numel():
            raise RuntimeError('Empty FFT: cuFFT would refuse this.')
        return function(value, *args, **kwargs)
    return call


def test_selection():
    """The rule picks the checkpoint that beats AdaRadar most often, and update 0 cannot win."""
    rows = []
    for arm, update, bpcs, kept in ((0, 0, 1.2, 99.), (0, 40, 1.2, 80.), (1, 40, 1.2, 90.), (1, 80, 1.2, 60.)):
        rows.append(dict(arm=arm, update=update, price=.1, bpcs=bpcs, label_miss=0., labels_detected=10, robust=100, robust_lost=100.-kept, robust_new=1., rd_rmse_db=2., nmse=.1))
    references = {1.4541: dict(labels=99., robust=70., false=5., rd=3.)}
    ranking = R.choose_checkpoint(pd.DataFrame(rows), references)
    assert tuple(ranking.iloc[0][['arm', 'update']]) == (1, 40) and 0 not in set(ranking['update'])
    table = pd.DataFrame([dict(budget=1.4541, price=p, bpcs=1.4, label_miss=0., labels_detected=10, robust=100, robust_lost=k, robust_new=1., rd_rmse_db=2., nmse=.1, within_budget=within) for p, k, within in ((.05, 20., True), (.1, 10., True), (.2, 40., True), (.3, 1., False))])
    assert R.choose_prices(table, references)[0][1.4541] == .1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    test_selection()
    print('selection PASS', flush=True)
    V13.H0_MAPS = 4
    V17.BOOTSTRAP_DRAWS = 200
    R.SCREEN_CUBES, R.PRICE_GRID, R.AUDIT_CUBES, R.CHECKPOINT_SHARE = 2, (0., .2), 2, .5
    assignment = R.PACKAGE/'GLOBAL_SPLIT_V17.csv'
    dataset = V17.data.prepare_dataset(args.dataset_root, args.reference_root)
    _, scan = V17.inventory(dataset, assignment)
    frames = V17.partition.assign_frozen_partitions(dataset, assignment, bounded=True)
    train, validation = frames[0], frames[1]
    scenes = sorted(validation.scene_name.unique())
    pick = lambda frame, names, count: V13.select_rows(frame.loc[frame.scene_name.isin(names)], count)
    rows = dict(train=pick(train, sorted(train.scene_name.unique())[:1], 6), validation=pick(validation, scenes[:1], 2), test=pick(validation, scenes[2:4], 2))
    runtime = R.sweep.build_runtime(dataset, assignment, 'cpu', rows={role: frame.assign(role=role) for role, frame in rows.items()})
    out = args.out.resolve()
    expected = json.loads(json.dumps(R.context(runtime, out, assignment, scan)))
    names = ('fft', 'ifft', 'fft2', 'ifft2', 'fftn', 'ifftn')
    with mock.patch.multiple(torch.fft, **{name: strict(getattr(torch.fft, name)) for name in names}):
        R.run_check(runtime, out, expected)
        R.run(runtime, out, expected)
    with pd.option_context('display.width', 250, 'display.max_columns', 40):
        print(pd.read_csv(out/'CHECKPOINT_RANKING.csv').drop(columns='detail').round(3).to_string(index=False))
        print(pd.read_csv(out/'RATE_SWEEP_TEST.csv')[['method', 'budget_bpcs', 'bpcs', 'bpcs_max', 'within_budget_fraction', 'labels_kept_pct', 'robust_kept_pct', 'robust_false_per_map', 'rd_rmse_db', 'mAP', 'F1']].round(3).to_string(index=False))
    print((out/'FROZEN_POLICY.json').read_text()[:700])
    print((out/'PREDICTIONS.json').read_text()[:600])
    print(json.dumps({k: v for k, v in json.loads((out/'SMOKE_COMPLETED.json').read_text()).items() if k != 'context'}, indent=1))
    print(sorted(p.name for p in out.iterdir()))


if __name__ == '__main__':
    main()
