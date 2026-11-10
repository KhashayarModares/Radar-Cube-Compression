"""Local CPU pre-flight of the UW PPO TEST pass on REAL frames. Not a result.

Tiny counts on VALIDATION frames standing in for TEST, so no TEST frame is read. Run it from the extracted
upload bundle, so a file missing from the bundle fails here and not on LMS. The VALIDATION replay cannot
match on CPU, whose dither differs from the GPU's, so it is not compared here. FFTs refuse empty input, as
cuFFT does on LMS and the CPU does not.

python -B tools/uw_ppo_test_gpu/local_real_check.py --dataset-root ... --out ...
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
spec = importlib.util.spec_from_file_location('uw_ppo_test_runner', SRC/'tools/run_uw_ppo_test.py')
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
S = R.study


def strict(function):
    def call(value, *args, **kwargs):
        if not value.numel():
            raise RuntimeError('Empty FFT: cuFFT would refuse this.')
        return function(value, *args, **kwargs)
    return call


def test_same_peak():
    """A peak elsewhere in the gate keeps the label under the run's count, not under the same-peak count."""
    bins = torch.tensor([[50, 60]])
    original = dict(points=torch.tensor([[50, 100]]), angles=torch.tensor([60]), stable=torch.tensor([True]))
    moved = dict(points=torch.tensor([[53, 200]]), angles=torch.tensor([62]), stable=torch.tensor([True]))
    assert bool(R.labelling.gated(bins, moved).any()) and R.same_peak(bins, original, moved) == (1, 1)
    assert R.same_peak(bins, original, dict(points=torch.tensor([[51, 101]]), angles=torch.tensor([60]), stable=torch.tensor([True]))) == (0, 0)
    assert R.same_peak(bins, original, dict(points=torch.zeros((0, 2), dtype=torch.int64), angles=torch.zeros(0, dtype=torch.int64), stable=torch.zeros(0, dtype=torch.bool))) == (1, 1)
    wrapped = dict(points=torch.tensor([[50, 254]]), angles=torch.tensor([60]), stable=torch.tensor([True]))
    assert R.same_peak(bins, dict(original, points=torch.tensor([[50, 1]])), wrapped) == (0, 0)


def test_ladder():
    """The search finds the finest step within the budget from any start, and the coarsest when none fits."""
    rates = [5., 4., 3., 2., 1.]
    assert all(R.ladder_search(lambda i: rates[i] <= 2.5, start, len(rates)) == 3 for start in range(len(rates)))
    assert R.ladder_search(lambda i: False, 1, len(rates)) == len(rates)-1
    assert R.ECDQ_LADDER[R.ECDQ_START] == 1. and R.uw.N_CHIRPS % R.BAQ_BLOCK[0] == 0 and R.uw.N_SAMPLES % R.BAQ_BLOCK[1] == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    root = args.dataset_root.resolve()
    test_same_peak()
    test_ladder()
    print('same-peak and ladder PASS', flush=True)
    assert R.frozen_hashes()['arm_1_update_0360.pt'] == S.sha(R.FROZEN/'arm_1_update_0360.pt')
    expected_run = R.read_json(R.FROZEN/'EXPECTED.json')
    assert expected_run['validation']['labels_lost'] == 0 and expected_run['validation']['robust_false_per_map'] <= .2
    sequences, tables, rows = S.rosters(root)
    assert [len(frame) for frame in rows] == [224, 60, 3592]
    tiny = [rows[0].iloc[[0]].reset_index(drop=True), rows[1].iloc[[0, 20]].reset_index(drop=True), rows[1].iloc[[32, 48]].assign(split_role='test').reset_index(drop=True)]
    runtime = S.Runtime(root, sequences, tables, tiny, 'cpu', 1, engineering=True)
    names = ('fft', 'ifft', 'fft2', 'ifft2', 'fftn', 'ifftn')
    with mock.patch.multiple(torch.fft, **{name: strict(getattr(torch.fft, name)) for name in names}):
        identity = json.loads(json.dumps(R.context(runtime, out)))
        R.run_check(runtime, out, identity)
        runtime.alpha = None
        R.run_all(runtime, out, identity)
    with pd.option_context('display.width', 250, 'display.max_columns', 60):
        print(pd.read_csv(out/'SUMMARY.csv').round(3).to_string(index=False))
    summary = pd.read_csv(out/'SUMMARY.csv').set_index('method')
    for budget in R.ECDQ_BUDGETS:
        assert summary.loc[f'Learned ECDQ budget={budget:g}', 'bpcs_max'] <= budget
    assert all(f'BAQ {bits}-bit' in summary.index for bits in R.BAQ_BITS) and all(f'Learned ECDQ beta={beta:g}' in summary.index for beta in R.ECDQ_BETAS)
    print('classical references PASS', flush=True)
    print(json.dumps(json.loads((out/'SCIENTIFIC_GATES.json').read_text()), indent=1))
    print(sorted(p.name for p in out.iterdir()))


if __name__ == '__main__':
    main()
