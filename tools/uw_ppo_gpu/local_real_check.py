"""Local CPU pre-flight of the complete UW PPO run on REAL frames. Not a result.

Tiny counts and one update per arm, on sequences outside the TEST split. Run it from the extracted
upload bundle, so a file missing from the bundle fails here and not on LMS. FFTs refuse empty input,
as cuFFT does on LMS and the CPU does not.

python -B tools/uw_ppo_gpu/local_real_check.py --dataset-root ... --out ...
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
spec = importlib.util.spec_from_file_location('uw_ppo_runner', SRC/'tools/run_uw_ppo.py')
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
P, L = R.physics_module, R.learning
FULL = {'train': (200, 224), 'validation': (60, 60), 'test': (3592, 3592)}


def strict(function):
    def call(value, *args, **kwargs):
        if not value.numel():
            raise RuntimeError('Empty FFT: cuFFT would refuse this.')
        return function(value, *args, **kwargs)
    return call


def test_geometry(runtime):
    """tile_of agrees with the codec's own shifted slices on every native cell."""
    row = next(runtime.rows[0].itertuples())
    source = runtime.physics.prepare(runtime.cube(row))
    assert len(source['tiles']) == 64
    owner = torch.full((255, 128), -1, dtype=torch.int64)
    for i, (_, doppler, rng) in enumerate(source['slices']):
        owner[doppler, rng] = i
    assert (owner >= 0).all()
    rows, columns = torch.arange(128)[None].expand(255, 128), torch.arange(255)[:, None].expand(255, 128)
    assert torch.equal(P.tile_of(rows, columns), torch.fft.ifftshift(owner, dim=(0, 1)))


def test_receiver(runtime):
    """A static scene must show detections at zero Doppler, on the edge of the unshifted map."""
    row = next(runtime.rows[0].itertuples())
    found = R.labelling.peaks(runtime.receiver.receive(runtime.cube(row)), runtime.alpha)['points'][:, 1]
    assert int(((found <= 4) | (found >= R.uw.N_CHIRPS-5)).sum()) > 0
    source, original, labs, features = runtime.case(row)
    assert features.shape == (64, len(L.v12.base.STATIC_NAMES)) and bool(torch.isfinite(features).all())
    same = runtime.physics.evaluate(original, original, labs)
    tile = P.tile_terms(runtime.physics, original, original, same, labs, runtime.alpha)
    assert int(same['low_miss']) == 0 and float(tile['miss'].sum()) == 0 and float(tile['robust_lost'].sum()) == 0 and float(tile['robust_new'].sum()) == 0 and float(tile['excess'].sum()) == 0
    empty = dict(labs, RD=labs['RD'][:0], RA=labs['RA'][:0])
    terms = P.tile_terms(runtime.physics, original, original, runtime.physics.evaluate(original, original, empty), empty, runtime.alpha)
    assert float(terms['miss'].sum()) == 0
    assert len(runtime.physics.label_positions(original, labs['RA'][:0])['RD']) == 0
    assert L.INITIAL_RHO == 1 and L.INITIAL_Q == 5 and R.cap.Q_UNITY == 6 and L.DeltaPolicy is L.v12.DeltaPolicy and L.raw_states is L.v12.raw_states and L.N_ACTIONS == 21
    fitted = dict(indices=torch.arange(len(L.NAMES)), mean=torch.zeros(len(L.NAMES), dtype=torch.float64), std=torch.ones(len(L.NAMES), dtype=torch.float64), names=list(L.NAMES))
    plain = L.state_variant(fitted, 'no_cfar')
    assert L.state_variant(fitted, 'v12_40') is fitted and not set(plain['names']) & set(L.CFAR_STATE) and len(plain['names']) == len(L.NAMES)-len(L.CFAR_STATE)
    r, bits, mask = L.alternatives(features.new_full((64,), .25), features.new_full((64,), 4.))
    assert r.min() >= .25 and bits.min() >= 4 and bool(mask.any(1).all())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    root = args.dataset_root.resolve()
    sequences, tables, rows = R.rosters(root)
    for role, frame in zip(R.ROLES, rows):
        low, high = FULL[role]
        assert low <= len(frame) <= high, (role, len(frame))
    assert '2019_05_09_bm1s007' in set(rows[0].sequence)
    assert not (set(rows[0].sequence) & set(rows[1].sequence)) and not (set(rows[0].sequence) & set(rows[2].sequence)) and not (set(rows[1].sequence) & set(rows[2].sequence))
    assert all(abs(R.data.ANGLE_GRID[label[1]]) <= 60 for table in tables.values() for labels in table.values() for label in labels)
    assert not any(tables['2019_05_29_pcms005'].values())
    print('cubes', [len(frame) for frame in rows], flush=True)
    tiny = [rows[0].iloc[[0, 18, 50, 66, 90]].reset_index(drop=True), rows[1].iloc[[0, 20]].reset_index(drop=True), rows[1].iloc[[32, 48]].assign(split_role='test').reset_index(drop=True)]
    R.ARMS = (R.ARMS[1], R.ARMS[2])
    runtime = R.Runtime(root, sequences, tables, tiny, 'cpu', 1, engineering=True)
    runtime.alpha = runtime.receiver.calibrate(4, R.SEED)
    names = ('fft', 'ifft', 'fft2', 'ifft2', 'fftn', 'ifftn')
    with mock.patch.multiple(torch.fft, **{name: strict(getattr(torch.fft, name)) for name in names}):
        test_geometry(runtime)
        test_receiver(runtime)
        print('pieces PASS', flush=True)
        runtime.alpha = None
        expected = json.loads(json.dumps(R.context(runtime, out)))
        R.run_check(runtime, out, expected)
        R.run_all(runtime, out, expected, h0_maps=4)
    with pd.option_context('display.width', 250, 'display.max_columns', 60):
        print(pd.read_csv(out/'tables/validation.csv').round(3).to_string(index=False))
    print(json.dumps(json.loads((out/'SCIENTIFIC_GATES.json').read_text()), indent=1))
    print(json.dumps(json.loads((out/'FROZEN_POLICY.json').read_text()), indent=1))
    print(sorted(p.name for p in out.iterdir()))


if __name__ == '__main__':
    main()
