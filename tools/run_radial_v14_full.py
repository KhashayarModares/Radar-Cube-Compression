"""V14 full run: detection-preserving RADIal cube compression with a learned entropy model.

System (fixed by theory and the V13 smoke studies, no tuning on TEST):
  * receiver: unchanged RADIal RD chain, cyclic CA-CFAR + 5x5 NMS, alpha from
    noise-only cubes (Neyman-Pearson, 1 false peak per map);
  * quantizer: subtractive-dither uniform quantization (ECDQ) of the per-Rx
    FFT2 coefficients, step beta x tile noise sigma (Zamir & Feder 1992);
  * closed loop (analysis-by-synthesis, Atal & Remde 1982): faithful coding on
    the detection support, synthesized background elsewhere (rate-distortion-
    perception, Blau & Michaeli 2019); the encoder decodes its own stream and
    grows/refines the CA-CFAR window of each disagreement. Two operating points:
      A closed_g0_b2         repairs robust disagreements   (detector-faithful)
      B closed_strict2_g0_b2 repairs every lost/new decision (decision-transparent)
  * learned conditional entropy model (the trained component): a 7k-parameter
    MLP over a causal cross-Rx context, trained on TRAIN by minimizing the code
    length; it changes bits only, never the decoded cube (Balle et al. 2017/2018,
    Minnen et al. 2018, Minnen & Singh 2020). Per cube the encoder sends the
    shorter of the learned and the per-tile KT code (one flag bit).

Populations: 400 TRAIN / 150 VALIDATION / 150 TEST cubes, recording-disjoint
frozen roles. VALIDATION selects the model epoch; TEST is unlocked only after
the model checkpoint is frozen and hashed, and is evaluated once.
CUDA for ADC decoding, transforms, codec, receiver, features and training.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from radar_sim import radial_v13_detection_codec as codec
from radar_sim import radial_v14_entropy_model as entropy
from radar_sim.radial_adaptive_gpu_physics import RadialGPUPhysics, cyclic_cfar, load_cube_gpu


def sibling(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V13 = sibling('run_radial_v13_detection_smoke')  # receiver, codec runs, metrics, reports
common, data, partition = V13.common, V13.data, V13.partition

REVISION = '14.0.0-closed-loop-ECDQ-learned-entropy-model'
SEED = 20261030
COUNTS = dict(train=400, validation=150, test=150)
SAMPLES_PER_CUBE = 16384
TRAINING = dict(epochs=15, batch=32768, lr=2e-3, hidden=entropy.HIDDEN, features=entropy.FEATURES,
                samples_per_cube_and_operating_point=SAMPLES_PER_CUBE)
OPERATING_POINTS = (
    ('closed_g0_b2', dict(kind='closed', gamma=0., beta=2.)),
    ('closed_strict2_g0_b2', dict(kind='closed', gamma=0., beta=2., strict=True, strict_new=True)),
)
REFERENCES = (
    ('lossless_adc', dict(kind='lossless')), ('legacy_dct_r025_q8', dict(kind='legacy_dct')),
    ('perturb_m20db', dict(kind='perturb', level_db=-20.)), ('ecdq_b1', dict(kind='ecdq', beta=1.)),
)
CONFIG = dict(counts=COUNTS, training=TRAINING, operating_points={k: v for k, v in OPERATING_POINTS},
              references={k: v for k, v in REFERENCES}, H0_maps=V13.H0_MAPS, margin_db=V13.MARGIN_DB,
              label_gate=list(V13.LABEL_GATE), refinement_betas=list(codec.REFINE_BETAS),
              label_rule='label preserved if an original detection in its gate is matched (+-2) by a decoded detection; '
                         'robust if supported by a stable peak (1 dB clear of alpha and of its NMS rival)',
              rate='min(learned model, per-tile KT) arithmetic code length + 1 flag bit + all side information',
              test_policy='TEST unlocked only after the entropy-model checkpoint is frozen and hashed')


class Runtime(V13.Runtime):
    """Three frozen roles; TEST cube bodies/labels stay locked until authorize_test()."""

    def __init__(self, dataset, assignment, device, counts=COUNTS, rows=None):
        self.dataset, self.device = dataset, torch.device(device)
        if rows is None:
            frames = partition.assign_frozen_partitions(dataset, assignment, bounded=False)
            rows = {role: V13.select_rows(frames[i], counts[role]).assign(role=role)
                    for i, role in enumerate(('train', 'validation', 'test'))}
        self.rows = rows
        scenes = [set(frame.scene_name) for frame in rows.values()]
        if any(scenes[i] & scenes[j] for i in range(len(scenes)) for j in range(i)):
            raise AssertionError('A recording crosses roles.')
        for role in ('train', 'validation'):
            frame = rows[role].drop(columns='role')
            partition.assert_frozen_role_access(dataset, frame, frame.split_role.iloc[0])
        engineering = self.device.type == 'cpu'
        self.physics = RadialGPUPhysics(dataset.reference_root, device=self.device, engineering=engineering)
        self.legacy = V13.LegacyDCT(dataset.reference_root, device=self.device, engineering=engineering)
        self.labels = data.scoped_labels(dataset, pd.concat([rows['train'], rows['validation']], ignore_index=True))
        self.engineering = engineering

    def authorize_test(self, receipt):
        test = self.rows['test']
        data.assert_partition_access(self.dataset, test.drop(columns='role'), frozen=receipt)
        self.labels = pd.concat([self.labels, data.scoped_labels(self.dataset, test.drop(columns='role'), frozen=receipt)],
                                ignore_index=True)

    def cube(self, row):
        frame = pd.DataFrame([row.drop(labels='role').to_dict()])
        partition.assert_frozen_role_access(self.dataset, frame, row.split_role)
        return load_cube_gpu(row, self.device, engineering=self.engineering)

    def roster(self, role):
        return self.rows[role]


def source_hashes():
    files = [Path(__file__), Path(V13.__file__), Path(codec.__file__), Path(entropy.__file__), Path(partition.__file__),
             common.ROOT/'radar_sim/radial_adaptive_gpu_physics.py']
    return {str(p.resolve().relative_to(common.ROOT)): common.sha(p) for p in files}


def context(runtime, out, assignment):
    return dict(revision=REVISION, source_root=str(common.ROOT.resolve()), output=str(out),
        dataset_root=str(runtime.dataset.root), reference_root=str(runtime.dataset.reference_root),
        device=str(runtime.device), torch=torch.__version__,
        gpu=torch.cuda.get_device_name(runtime.device) if runtime.device.type == 'cuda' else 'cpu-engineering',
        cuda_runtime=torch.version.cuda, split_audit=str(assignment.resolve()),
        split_audit_sha256=common.sha(assignment), source_hashes=source_hashes(),
        dataset_identity=data.dataset_identity(runtime.dataset),
        populations={role: frame.record_name.tolist() for role, frame in runtime.rows.items()},
        configuration=CONFIG, seed=SEED)


def run_operating_point(runtime, spec, cube, original, alpha, seed):
    stream, decoded, history = codec.encode_closed_loop(cube, runtime.receive, original['detector'], alpha,
        gamma_db=spec['gamma'], beta=spec['beta'], seed=seed, strict_lost=spec.get('strict', False),
        strict_new=spec.get('strict_new', False))
    bits = codec.ecdq_bits(stream)
    bits.update(accounting='closed-loop ECDQ', closed_loop_history=json.dumps(history),
                closed_loop_iterations=len(history)-1)
    return decoded, bits, stream


def rate_fields(bits, learned_payload, numel):
    """Per-tile KT vs learned model; the encoder sends the shorter (+1 flag bit)."""
    kt = bits['total']
    fields = dict(bpcs_kt=(kt+1)/numel)
    if learned_payload is not None:
        learned = learned_payload+bits['side']+bits['header']
        fields.update(bpcs_learned=(learned+1)/numel, bits=min(kt, learned)+1, bpcs=(min(kt, learned)+1)/numel,
                      learned_chosen=learned < kt)
    else:
        fields.update(bpcs_learned=math.nan, bits=kt+1, bpcs=(kt+1)/numel, learned_chosen=False)
    return fields


def evaluate_cube(runtime, row, position, alphas, methods, label_log, model=None, keep=None, sampler=None,
                  on_decoded=None):
    cube = runtime.cube(row)
    original = runtime.receive(cube)
    bins, label_count = runtime.label_bins(row.record_name)
    seed = SEED+position
    records = []
    if on_decoded is not None:
        on_decoded(row, 'original', cube, original)
    for name, spec in methods:
        if spec['kind'] == 'closed':
            decoded, bits, stream = run_operating_point(runtime, spec, cube, original, alphas['a1'], seed)
            summary = entropy.compact(stream)
            if sampler is not None:
                sampler(name, summary)
            if keep is not None:
                keep[(row.record_name, name)] = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in summary.items()}
            learned = entropy.learned_payload_bits(model, summary) if model is not None else None
        else:
            decoded, bits = V13.run_method(runtime, spec, cube, original, alphas['a1'], seed)
            learned = None
        record = V13.score(runtime, row, name, cube, original, decoded, bits, alphas, bins, label_count, label_log)
        if spec['kind'] == 'closed':
            record.update(rate_fields(bits, learned, cube.numel()))
        records.append(record)
        if on_decoded is not None:
            on_decoded(row, name, decoded, original)
        del decoded
    return records, original


def summarize(frame):
    summary = V13.summarize(frame)
    extra = frame.groupby(['role', 'method'], sort=False)[['bpcs_kt', 'bpcs_learned']].mean().reset_index() \
        if 'bpcs_kt' in frame else None
    if extra is not None:
        summary = summary.merge(extra, on=['role', 'method'], how='left')
    return summary


def gates(summary, calibration, role):
    """Pre-registered criteria (fixed before TEST) for each operating point."""
    rows = summary.loc[summary.role.eq(role)].set_index('method')
    legacy, lossless = float(rows.loc['legacy_dct_r025_q8', 'bpcs_mean']), float(rows.loc['lossless_adc', 'bpcs_mean'])
    floor = calibration['H0_robust_new_per_map']['a1']+1
    control = rows.loc['perturb_m20db']
    out = {}
    for name, _ in OPERATING_POINTS:
        r = rows.loc[name]
        result = dict(bpcs_mean=float(r.bpcs_mean), bpcs_kt_mean=float(r.bpcs_kt), bpcs_learned_mean=float(r.bpcs_learned),
            ratio_vs_lossless=lossless/float(r.bpcs_mean), ratio_vs_raw_32bit=32/float(r.bpcs_mean),
            learned_rate_saving=1-float(r.bpcs_mean)/float(r.bpcs_kt),
            robust_lost_le_1pct=bool(r.a1_lost_robust_fraction <= .01),
            robust_new_le_H0_floor_plus_1=bool(r.a1_new_robust_per_map <= floor),
            no_robust_label_lost=bool(r.labels_lost_robust == 0),
            bitrate_below_legacy=bool(r.bpcs_mean < legacy),
            strict_lost_fraction=float(r.a1_lost_fraction), strict_new_per_map=float(r.a1_new_per_map),
            strict_labels_lost=int(r.labels_lost), labels_detected_original=int(r.labels_detected_original))
        result['passed'] = all(result[k] for k in ('robust_lost_le_1pct', 'robust_new_le_H0_floor_plus_1',
                                                   'no_robust_label_lost', 'bitrate_below_legacy'))
        out[name] = result
    return dict(role=role, legacy_dct_bpcs=legacy, lossless_adc_bpcs=lossless, H0_robust_new_floor_per_map=floor,
                receiver_fragility_control=dict(strict_lost_fraction=float(control.a1_lost_fraction),
                    strict_new_per_map=float(control.a1_new_per_map), labels_lost=int(control.labels_lost)),
                operating_points=out, scientific_gate_passed=any(v['passed'] for v in out.values()))


def make_figures(out, summary, history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = out/'figures'
    folder.mkdir(exist_ok=True)
    with plt.rc_context({'font.family': 'serif', 'font.size': 8}):
        fig, axes = plt.subplots(1, 3, figsize=(10, 2.8), layout='constrained')
        for role, marker in (('validation', 'o'), ('test', 's')):
            rows = summary.loc[summary.role.eq(role) & summary.bpcs_mean.notna()]
            for ax, key, label in ((axes[0], 'a1_lost_robust_fraction', 'Robust lost fraction'),
                                   (axes[1], 'a1_lost_fraction', 'Strict lost fraction')):
                ax.scatter(rows.bpcs_mean, rows[key].clip(lower=1e-5), marker=marker, s=16, label=role)
                for _, r in rows.iterrows():
                    ax.annotate(r.method, (r.bpcs_mean, max(r[key], 1e-5)), fontsize=5, xytext=(2, 2), textcoords='offset points')
                ax.set(xscale='log', yscale='log', xlabel='Bits per complex sample', ylabel=label)
        frame = pd.DataFrame(history)
        axes[2].plot(frame.epoch, frame.train_bits_per_symbol, label='TRAIN')
        axes[2].plot(frame.epoch, frame.val_bits_per_symbol, label='VALIDATION')
        axes[2].set(xlabel='Epoch', ylabel='Class code length (bits/symbol)')
        for ax in axes:
            ax.grid(alpha=.25, which='both')
            ax.legend(fontsize=6)
        fig.savefig(folder/'rate_fidelity_and_training.pdf')
        fig.savefig(folder/'rate_fidelity_and_training.png', dpi=250)
        plt.close(fig)


def run_check(runtime, out, expected):
    path = Path(str(out)+'.CHECK.json')
    if path.exists():
        old = common.read_json(path)
        if old.get('passed') is True and old.get('context') == expected:
            print('Existing identity-bound V14 check: PASS.', flush=True)
            return
        raise ValueError('Existing check identity differs. Preserve it; use a fresh deployment.')
    p = runtime.physics
    scores = codec.h0_scores(4, p.shape, p.range_window, p.doppler_window, cyclic_cfar, runtime.device, SEED)
    alpha = codec.threshold_for_rate(scores, 4, 1.)
    checks = []
    model = entropy.ContextModel().to(runtime.device)
    for _, row in runtime.roster('train').iloc[:2].iterrows():
        cube = runtime.cube(row)
        maps, mine = p.maps(cube), runtime.receive(cube)
        torch.testing.assert_close(mine['RD'], maps['RD'], rtol=0, atol=1e-9)
        if not torch.equal(mine['detector']['base_peaks'], maps['detectors']['RD']['base_peaks']):
            raise AssertionError('V14 RD receiver differs from the native RADIal receiver.')
        roundtrip = float((codec.inverse(codec.forward(cube))-cube.to(torch.complex128)).abs().max()/cube.abs().max())
        if roundtrip > 1e-12:
            raise AssertionError('FFT2 codec domain does not invert.')
        decoded, bits, stream = run_operating_point(runtime, dict(OPERATING_POINTS)['closed_g0_b2'], cube, mine, alpha, SEED)
        if not torch.equal(codec.decode_ecdq(stream, cube.dtype), decoded):
            raise AssertionError('Closed-loop stream does not reproduce the verified decode.')
        summary = entropy.compact(stream)
        for plane in (0, 7, 15):
            entropy.assert_causal(summary, plane)
        model_check = entropy.coder_check(model, summary)
        if model_check['real_bits'] > model_check['ideal_bits']*1.01+64:
            raise AssertionError('Learned-model arithmetic coder exceeds its ideal length.')
        keep = codec.cells_to_tiles(stream['support'], p.shape)
        tile = int(keep.sum(1).argmax())
        symbols = stream['index'][tile][keep[tile]].tolist()
        written = codec.write_arithmetic_tile(symbols)
        if codec.read_arithmetic_tile(written, len(symbols)) != symbols:
            raise AssertionError('KT arithmetic coder does not decode.')
        checks.append(dict(record_name=row.record_name, fft2_roundtrip=roundtrip, closed_loop_bpcs_kt=bits['total']/cube.numel(),
                           kt_tile_real_bits=len(written), learned_coder=model_check,
                           gpu_peak_memory_gb=torch.cuda.max_memory_allocated(runtime.device)/1e9 if runtime.device.type == 'cuda' else 0.))
        print('V14 receiver/codec/entropy-model preflight PASS:', row.record_name, json.dumps(model_check), flush=True)
    common.write_json(path, dict(completed=True, passed=True, context=expected, checks=checks,
        available=dict(train=len(runtime.rows['train']), validation=len(runtime.rows['validation']), test=len(runtime.rows['test'])),
        training_started=False, test_evaluated=False))
    print('V14 CUDA check complete.', flush=True)


def run(runtime, out, expected, log_every=10, perception=None):
    """``perception`` (optional) receives every VALIDATION/TEST decode and writes its own reports."""
    checked = common.read_json(Path(str(out)+'.CHECK.json'))
    if checked.get('context') != expected or checked.get('passed') is not True:
        raise ValueError('Run the matching CUDA check first.')
    if out.exists():
        raise FileExistsError('Preserve partial/completed results; never overwrite this study.')
    with common.exclusive(out):
        out.mkdir(parents=True)
        common.write_json(out/'manifest.json', expected)
        pd.concat(runtime.rows.values(), ignore_index=True).to_csv(out/'POPULATIONS.csv', index=False)
        start = time.time()
        log = lambda *a: print(*a, f'{time.time()-start:.0f}s', flush=True)
        calibration, curve = V13.calibrate(runtime)
        common.write_json(out/'H0_CALIBRATION.json', calibration)
        curve.to_csv(out/'H0_SURVIVAL.csv', index=False)
        alphas = calibration['alphas']
        log('H0 alpha dB:', json.dumps(calibration['alphas_db']))
        generator = torch.Generator(device=runtime.device).manual_seed(SEED)
        records, label_log, samples = [], [], {'train': ([], []), 'validation': ([], [])}

        # 1. TRAIN: operating points only; collect entropy-model training samples.
        for position, (_, row) in enumerate(runtime.roster('train').iterrows()):
            def sampler(name, summary):
                f, t = entropy.sample(summary, SAMPLES_PER_CUBE, generator)
                samples['train'][0].append(f); samples['train'][1].append(t)
            cube_records, _ = evaluate_cube(runtime, row, position, alphas, OPERATING_POINTS, None, sampler=sampler)
            records.extend(cube_records)
            if (position+1) % log_every == 0:
                log(f'TRAIN {position+1}/{len(runtime.roster("train"))}')

        # 2. VALIDATION: all methods; keep compact symbols for the learned rate after training.
        kept = {}
        for position, (_, row) in enumerate(runtime.roster('validation').iterrows(), start=len(runtime.roster('train'))):
            def sampler(name, summary):
                f, t = entropy.sample(summary, SAMPLES_PER_CUBE//4, generator)
                samples['validation'][0].append(f); samples['validation'][1].append(t)
            cube_records, _ = evaluate_cube(runtime, row, position, alphas, REFERENCES+OPERATING_POINTS, label_log,
                                            keep=kept, sampler=sampler, on_decoded=perception and perception.hook('validation'))
            records.extend(cube_records)
            if (position+1) % log_every == 0:
                log(f'VALIDATION {position+1-len(runtime.roster("train"))}/{len(runtime.roster("validation"))}')

        # 3. Train the learned entropy model (rate only), select epoch on VALIDATION.
        torch.manual_seed(SEED)
        model = entropy.ContextModel().to(runtime.device)
        rows_t, targets_t = torch.cat(samples['train'][0]), torch.cat(samples['train'][1])
        rows_v, targets_v = torch.cat(samples['validation'][0]), torch.cat(samples['validation'][1])
        log(f'Entropy model: {sum(p.numel() for p in model.parameters())} parameters, '
            f'{len(rows_t)} TRAIN / {len(rows_v)} VALIDATION symbols')
        history, best_epoch = entropy.train(model, rows_t, targets_t, rows_v, targets_v, epochs=TRAINING['epochs'],
            batch=TRAINING['batch'], lr=TRAINING['lr'], seed=SEED, log=lambda r: log('epoch', json.dumps(r)))
        del rows_t, targets_t, rows_v, targets_v, samples
        pd.DataFrame(history).to_csv(out/'ENTROPY_MODEL_TRAINING.csv', index=False)
        torch.save(model.state_dict(), out/'entropy_model.pt')

        # 4. Learned rate for VALIDATION operating points.
        numel = int(np.prod(runtime.physics.shape))
        for record in records:
            key = (record['record_name'], record['method'])
            if key in kept:
                summary = {k: v.to(runtime.device) if isinstance(v, torch.Tensor) else v for k, v in kept[key].items()}
                payload = entropy.learned_payload_bits(model, summary)
                kt_total = round(record['bpcs_kt']*numel)-1
                record.update(rate_fields(dict(total=kt_total, side=record['side_bits'], header=record['header_bits']),
                                          payload, numel))
        del kept

        # 5. Freeze the model, then unlock TEST once.
        receipt = out/'FROZEN_MODEL.json'
        common.write_json(receipt, dict(completed=True, checkpoint='entropy_model.pt',
            checkpoint_sha256=common.sha(out/'entropy_model.pt'), selected_epoch=int(best_epoch),
            selection='lowest VALIDATION bits/symbol', test_used_for_selection=False))
        runtime.authorize_test(receipt)
        log(f'Model frozen at epoch {best_epoch}; TEST unlocked.')
        base = len(runtime.roster('train'))+len(runtime.roster('validation'))
        for position, (_, row) in enumerate(runtime.roster('test').iterrows(), start=base):
            cube_records, _ = evaluate_cube(runtime, row, position, alphas, REFERENCES+OPERATING_POINTS, label_log, model=model,
                                            on_decoded=perception and perception.hook('test'))
            records.extend(cube_records)
            if (position+1-base) % log_every == 0:
                log(f'TEST {position+1-base}/{len(runtime.roster("test"))}')
        if common.sha(out/'entropy_model.pt') != common.read_json(receipt)['checkpoint_sha256']:
            raise ValueError('Entropy model changed after freezing.')

        # 6. Reports.
        frame = pd.DataFrame(records)
        frame.to_csv(out/'RESULTS_BY_CUBE.csv', index=False)
        pd.DataFrame(label_log).to_csv(out/'LABELS.csv', index=False)
        summary = summarize(frame)
        summary.to_csv(out/'SUMMARY.csv', index=False)
        for role in ('validation', 'test'):
            common.write_json(out/f'{role.upper()}_RECORDING_CI.json', V13.recording_bootstrap(frame.loc[frame.role.eq(role)]))
            common.write_json(out/f'{role.upper()}_GATES.json', gates(summary, calibration, role))
        make_figures(out, summary, history)
        if perception is not None:
            perception.report(out)
        runtime.dataset.assert_unchanged()
        result = common.read_json(out/'TEST_GATES.json')
        common.write_json(out/'SMOKE_COMPLETED.json', dict(completed=True, passed=True, context=expected, seed=SEED,
            test_evaluated=True, test_used_for_selection=False, full_training_started=False,
            scientific_gate_passed=result['scientific_gate_passed'], selected_epoch=int(best_epoch), entropy_model_trained=True,
            alphas_db=calibration['alphas_db'], runtime_seconds=time.time()-start,
            perception_gate_passed=common.read_json(out/'PERCEPTION_GATES.json')['passed'] if perception is not None else None))
    print(REVISION.split('-')[0], 'full run COMPLETE; inspect TEST_GATES.json and SUMMARY.csv:', out, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('check', 'run'), required=True)
    for key in ('dataset-root', 'reference-root', 'split-audit', 'out'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    if torch.device(args.device).type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('CUDA is required. There is no CPU fallback.')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dataset = data.prepare_dataset(args.dataset_root, args.reference_root)
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    runtime = Runtime(dataset, args.split_audit, args.device)
    expected = json.loads(json.dumps(context(runtime, out, args.split_audit)))
    print(json.dumps({k: expected[k] for k in ('revision', 'gpu', 'configuration')}, indent=2), flush=True)
    if args.stage == 'check':
        run_check(runtime, out, expected)
    else:
        run(runtime, out, expected)


if __name__ == '__main__':
    main()
