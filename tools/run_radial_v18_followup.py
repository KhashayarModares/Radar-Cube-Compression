"""V18 follow-up to the V17 publication run: two additions, nothing else changes.

1. A faithful AdaRadar baseline (radar_sim/radial_v18_adaradar.py): AdaRadar's
   static coder (Park, Chun & Seok, CVPR 2026), following the official code
   (github.com/jp4327/adaradar, utils/dct.py), applied where AdaRadar applies it,
   to the normalized post-FFT range-Doppler tensor FFTRadNet takes as input: 64 x 64
   block DCT, top-k pruning, 4-bit scaled quantization, at pruning ratios
   r = 2, 4, 8, 12.57 (the paper's RADIal operating point) and 25, and with
   AdaRadar's online rate control using the official 4-bit RADIal command
   (r0 = 12, per-recording reset, objective 'balance'/'add'). The receivers
   run directly on the reconstructed tensor. Rates are reported both as AdaRadar
   counts them (kept values only) and complete (plus the positions of the kept
   coefficients and the block scales). FFTRadNet stays frozen, as in the paper's
   "without fine-tuning" evaluation (Suppl. S4); its retraining of FFTRadNet on
   decompressed tensors (the paper's Table 1 checkpoints) is not reproduced.

2. A VALIDATION-only choice of the rate knob beta of operating point C. Rule,
   fixed before this run: beta* is the largest beta in {1.5, 2, 2.5, 3} whose
   VALIDATION results pass all pre-registered V17 criteria (CA-CFAR in the loop;
   OS-CFAR and alpha_0.1 generalization; FFTRadNet F1 and mAR drop <= 0.02);
   if none passes, beta* = 1.5. The choice is written to BETA_SELECTION.json
   before TEST is unlocked. V17 already measured C on TEST at every beta; V18
   re-evaluates C at beta 2 (and at beta* if different) on TEST only as an
   exact reproducibility anchor.

Everything else is the frozen V17 system: the same split, receivers, metrics,
statistics and the entropy model frozen in V17 (checkpoint copied into the
package and hash-checked; nothing is trained). TEST is unlocked once, with that
checkpoint as the frozen receipt.

STAGES: count (metadata-only inventory), check (CUDA pre-flight), run.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from radar_sim import radial_v18_adaradar as adaradar
from radar_sim.radial_adaptive_gpu_physics import cyclic_cfar


def sibling(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V17 = sibling('run_radial_v17_publication')  # frozen V17 system: runtime, receivers, scoring, perception, statistics
V13, common, data, partition, codec, entropy = V17.V13, V17.common, V17.data, V17.partition, V17.codec, V17.entropy

REVISION = '18.0.0-adaradar-rd-baseline-validation-beta-selection'
PACKAGE = Path(__file__).resolve().parent/'v18_followup_gpu'
V17.PACKAGE = PACKAGE  # FFTRadNet files and the frozen split are copied into the V18 package
FROZEN_MODEL = dict(file='v17_entropy_model.pt', sha256='2653cfc6536cda41fff5f1cc17ac5194e81dcad16330e907fa1389c7d1e538c9',
                    source='V17 publication run v17_pub_20261030T001402535Z, selected epoch 15')
REFERENCES = tuple((name, spec) for name, spec in V17.REFERENCES if name in ('lossless_adc', 'legacy_dct_r025_q8', 'perturb_m20db'))
BETAS = {'closed_full_b1p5': 1.5, 'closed_full_b2': 2., 'closed_full_b2p5': 2.5, 'closed_full_b3': 3.}
C_POINTS = tuple((name, dict(kind='closed', gamma=300., beta=beta)) for name, beta in BETAS.items())
ADARADAR = tuple((f'adaradar_r{str(r).replace(".", "p")}_s4', dict(kind='adaradar', prune_ratio=r, bits=4))
                 for r in (2, 4, 8, 12.57, 25))
# AdaRadar's online rate control with the official README's 4-bit RADIal command: Evaluation.py --comp_ratio 12
# --qbit 4 --enable_feedback --init_cr_per_scene --loss_type balance --objective add --lr 1 --lambda_val 1
# --grad_clip 1 --conf_thd 0.8 (epsilon, min/max ratio: Evaluation.py defaults).
ADAPTIVE_SETTINGS = dict(comp_ratio=12., bits=4, epsilon=.05, lr=1., lambda_val=1., grad_clip=1., conf_thd=.8,
                         min_comp_ratio=1., max_comp_ratio=100.)
ADARADAR = ADARADAR+(('adaradar_adaptive_s4', dict(kind='adaradar_adaptive', **ADAPTIVE_SETTINGS)),)
VALIDATION_METHODS = REFERENCES+C_POINTS+ADARADAR
BETA_RULE = ('beta* = largest beta in {1.5, 2, 2.5, 3} whose VALIDATION results pass all pre-registered V17 criteria '
             '(CA-CFAR in the loop; OS-CFAR and alpha_0.1 generalization; FFTRadNet F1 and mAR drop <= 0.02); '
             'if none passes, beta* = 1.5')
CONFIG = dict(adaradar=dict(ADARADAR), adaradar_block=adaradar.BLOCK, references=dict(REFERENCES), c_points=dict(C_POINTS),
              validation_methods=[m for m, _ in VALIDATION_METHODS], beta_rule=BETA_RULE, frozen_model=FROZEN_MODEL,
              gates=V17.GATE, H0_maps=V13.H0_MAPS, margin_db=V13.MARGIN_DB, label_gate=list(V13.LABEL_GATE),
              sync_tolerance_us=data.TOLERANCE_US,
              test_policy='TEST unlocked once after beta* is written; C on TEST only as a reproducibility anchor of V17')


class AdaptiveRate:
    """AdaRadar's zeroth-order rate control (paper Algorithm 1; official run_FullEvaluation_SGD).

    The ratio starts at comp_ratio for every recording (init_cr_per_scene). Each
    frame is coded at r + eps, which is transmitted and scored, and at r - eps,
    used only for the gradient. FFTRadNet's highest box confidence on both gives
    the finite difference (zeroed when both are below conf_thd); the rate term
    lambda / r^2 is added (loss 'balance', objective 'add'); the step is clipped
    to +-grad_clip and r <- r + lr * step, clipped to [min, max]. Without any box
    in either prediction, r is not updated.
    """

    def __init__(self, settings):
        self.settings, self.scene, self.ratio = settings, None, None

    def code(self, row, spectrum, stats, perception):
        p = self.settings
        if row.scene_name != self.scene:
            self.scene, self.ratio = row.scene_name, p['comp_ratio']
        clip = lambda r: float(np.clip(r, p['min_comp_ratio'], p['max_comp_ratio']))
        plus_r, minus_r = clip(self.ratio+p['epsilon']), clip(self.ratio-p['epsilon'])
        plus, bits = adaradar.compress(spectrum, plus_r, p['bits'], stats['input_mean'], stats['input_std'])
        minus, _ = adaradar.compress(spectrum, minus_r, p['bits'], stats['input_mean'], stats['input_std'])
        (pred_p, _), (pred_m, _) = perception.run(plus), perception.run(minus)
        if len(pred_p) and len(pred_m):
            conf_p, conf_m = float(pred_p[:, 2].max()), float(pred_m[:, 2].max())
            denominator = plus_r-minus_r
            if abs(denominator) < 1/adaradar.BLOCK**2 or (conf_p < p['conf_thd'] and conf_m < p['conf_thd']):
                gradient = 0.
            else:
                gradient = (conf_p-conf_m)/denominator
            gradient += p['lambda_val']/((plus_r+minus_r)/2)**2
            self.ratio += p['lr']*float(np.clip(gradient, -p['grad_clip'], p['grad_clip']))
        self.ratio = clip(self.ratio)
        return plus, bits, plus_r


class Runtime(V17.Runtime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptive = AdaptiveRate(ADAPTIVE_SETTINGS)

    def receive_spectrum(self, spectrum):
        """The receiver's steps after its FFTs (same as rd_receive), applied to a given spectrum."""
        amplitude = spectrum.abs().sum(dim=2)
        rd = 20*torch.log10(amplitude.clamp_min(torch.finfo(torch.float64).tiny))
        return dict(RD=rd, spectrum=spectrum, detector=cyclic_cfar(10.**(rd/10.)))


def load_frozen_model(device):
    path = PACKAGE/FROZEN_MODEL['file']
    if common.sha(path) != FROZEN_MODEL['sha256']:
        raise ValueError('The frozen V17 entropy-model checkpoint changed.')
    model = entropy.ContextModel().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.eval()


def evaluate_cube(runtime, row, methods, alphas, model, perception, label_log):
    """As V17.evaluate_cube, plus AdaRadar (coded in the spectrum domain) and the spectrum NMSE of every method."""
    cube = runtime.cube(row)
    original = runtime.receive(cube)
    original_os = V17.receivers.cyclic_os_cfar(original['detector']['power'])
    bins, label_count = runtime.label_bins(row.record_name)
    perception.start(row, original)
    energy = original['spectrum'].abs().square().sum()
    records = []
    for name, spec in methods:
        if spec['kind'] in ('adaradar', 'adaradar_adaptive'):
            stats = perception.config['dataset']['statistics']  # AdaRadar codes the normalized network input
            if spec['kind'] == 'adaradar':
                (spectrum, bits), ms = V17.timed(runtime.device, adaradar.compress, original['spectrum'], spec['prune_ratio'],
                                                 spec['bits'], stats['input_mean'], stats['input_std'])
                ratio = spec['prune_ratio']
            else:
                (spectrum, bits, ratio), ms = V17.timed(runtime.device, runtime.adaptive.code, row, original['spectrum'],
                                                        stats, perception)
            received, decoded, stream = runtime.receive_spectrum(spectrum), cube, None
            times = dict(codec_ms=ms, encode_ms=math.nan, decode_ms=math.nan, adaradar_comp_ratio=ratio,
                         bpcs_nonzero_only=bits['total_nonzero_only']/cube.numel(), index_bpcs=bits['index_bits']/cube.numel(),
                         kept_fraction=bits['kept_fraction'], nonzero_fraction=bits['nonzero_fraction'])
        else:
            decoded, bits, stream, times = V17.encode_decode(runtime, spec, cube, original, alphas['ca']['a1'], runtime.seed(row))
            received = runtime.receive(decoded)
        record = V13.score(runtime, row, name, cube, original, decoded, bits, alphas['ca'], bins, label_count, label_log, received)
        record.update(times, spectrum_nmse=float((received['spectrum']-original['spectrum']).abs().square().sum()/energy))
        if spec['kind'] in ('adaradar', 'adaradar_adaptive'):
            record['nmse'] = math.nan  # not an ADC-domain codec; its distortion is spectrum_nmse
        method_os = V17.receivers.cyclic_os_cfar(received['detector']['power'])
        for key, alpha in alphas['os'].items():
            metrics = codec.detection_metrics(original_os, method_os, alpha, V13.MARGIN_DB)
            record.update({f'os_{key}_{k}': v for k, v in metrics.items()})
        if spec['kind'] == 'closed':
            learned, record['entropy_model_ms'] = V17.timed(runtime.device, entropy.learned_payload_bits, model, entropy.compact(stream))
            record.update(V17.V14.rate_fields(bits, learned, cube.numel()))
        if name not in V17.PERCEPTION_SKIPPED:
            perception.add(name, received)
        records.append(record)
        del decoded, received, stream
    return records


def report_tables(records):
    frame = pd.DataFrame(records)
    frame['learned_chosen'] = frame.learned_chosen.astype(float)
    summary = V17.summarize(frame)
    extra = frame.groupby(['role', 'method'], sort=False)[['spectrum_nmse', 'bpcs_nonzero_only', 'index_bpcs',
                                                          'nonzero_fraction', 'adaradar_comp_ratio']].mean().reset_index()
    # The official code reports the harmonic mean of per-frame elements / nonzero = 1 / mean nonzero fraction.
    extra['official_compression_ratio'] = 1/extra.nonzero_fraction
    return frame, summary.merge(extra, on=['role', 'method'])


def select_beta(gates):
    """The pre-declared rule (BETA_RULE) applied to VALIDATION gate verdicts."""
    rows = {name: dict(beta=beta, passed=gates['operating_points'][name]['passed'],
                       ca_cfar_passed=gates['operating_points'][name]['ca_cfar']['passed'],
                       generalization_passed=gates['operating_points'][name]['generalization']['passed'],
                       perception_passed=gates['operating_points'][name]['perception']['passed'],
                       os_cfar_robust_lost=gates['operating_points'][name]['generalization']['os_cfar_robust_lost_fraction'],
                       alpha_01_robust_lost=gates['operating_points'][name]['generalization']['alpha_01_robust_lost_fraction'],
                       dF1=gates['operating_points'][name]['perception']['dF1'], bpcs=gates['operating_points'][name]['bpcs'])
            for name, beta in BETAS.items()}
    passing = [row['beta'] for row in rows.values() if row['passed']]
    chosen = max(passing) if passing else min(BETAS.values())
    return dict(rule=BETA_RULE, candidates=rows, beta_selected=chosen, none_passed=not passing,
                method_selected=next(name for name, beta in BETAS.items() if beta == chosen),
                selected_on='VALIDATION only', test_unlocked_before_selection=False,
                note='V17 measured C on TEST at every beta before this rule was applied; those TEST numbers were not used here.')


def adaradar_table(summary, perception, agreement):
    """TEST: AdaRadar at every pruning ratio next to C, with both rate accountings."""
    test = summary.loc[summary.role.eq('test')].set_index('method')
    seen = perception.loc[perception.role.eq('test') & perception.subset.eq('all')].set_index('method')
    agree = agreement.loc[agreement.role.eq('test')].set_index('method')
    rows = []
    for method, r in test.iterrows():
        if method in ('lossless_adc', 'perturb_m20db'):
            continue
        s, a = seen.loc[method], agree.loc[method]
        rows.append(dict(method=method, bpcs=r.bpcs_mean, bpcs_nonzero_only=r.bpcs_nonzero_only, index_bpcs=r.index_bpcs,
                         official_compression_ratio=r.official_compression_ratio, mean_pruning_ratio=r.adaradar_comp_ratio,
                         official_bits_per_element=r.bpcs_nonzero_only/2,
                         ratio_vs_raw=32/r.bpcs_mean, spectrum_nmse_db=10*math.log10(max(r.spectrum_nmse, 1e-30)),
                         rd_rmse_db=r.rd_rmse_db_mean, ca_robust_lost=r.a1_lost_robust_fraction,
                         ca_strict_lost=r.a1_lost_fraction, ca_robust_new_per_map=r.a1_new_robust_per_map,
                         os_robust_lost=r.os_a1_lost_robust_fraction, ca01_robust_lost=r.a01_lost_robust_fraction,
                         background_shift_db=r.a1_background_shift_db_median, labels_lost_strict=r.labels_lost,
                         labels_lost_robust=r.labels_lost_robust, mAP=s.mAP, mAR=s.mAR, F1=s.F1, dF1=s.dF1,
                         dF1_ci_low=s.get('dF1_ci_low'), dF1_ci_high=s.get('dF1_ci_high'), dmAR=s.dmAR,
                         objects_kept_fraction=a.objects_kept_fraction, freespace_iou=a.freespace_iou_mean,
                         codec_ms_median=r.codec_ms_median))
    return pd.DataFrame(rows).sort_values('bpcs')


def context(runtime, out, assignment, scan):
    hashes = V17.source_hashes()
    for path in (Path(__file__), Path(adaradar.__file__), PACKAGE/FROZEN_MODEL['file']):
        hashes[str(path.resolve().relative_to(common.ROOT))] = common.sha(path)
    return dict(revision=REVISION, source_root=str(common.ROOT.resolve()), output=str(out),
                dataset_root=str(runtime.dataset.root), reference_root=str(runtime.dataset.reference_root),
                device=str(runtime.device), torch=torch.__version__, cuda_runtime=torch.version.cuda,
                gpu=torch.cuda.get_device_name(runtime.device) if runtime.device.type == 'cuda' else 'cpu-engineering',
                split=str(Path(assignment).resolve()), split_sha256=common.sha(assignment),
                split_provenance_sha256=common.sha(V17.provenance_path(assignment)), inventory=scan, source_hashes=hashes,
                dataset_identity=data.dataset_identity(runtime.dataset),
                populations={role: frame.record_name.tolist() for role, frame in runtime.rows.items() if role != 'train'},
                configuration=CONFIG, seed=V17.SEED)


def run_check(runtime, out, expected):
    """AdaRadar and model checks on one TRAIN cube, then the complete V17 CUDA pre-flight (writes CHECK.json)."""
    if Path(str(out)+'.CHECK.json').exists():
        return V17.run_check(runtime, out, expected)
    model = load_frozen_model(runtime.device)
    row = runtime.roster('train').iloc[len(runtime.roster('train'))//2]
    original = runtime.receive(runtime.cube(row))
    spectrum = original['spectrum']
    d = adaradar.dct_matrix(adaradar.BLOCK, runtime.device)
    if float((d@d.T-torch.eye(adaradar.BLOCK, dtype=d.dtype, device=d.device)).abs().max()) > 1e-12:
        raise AssertionError('The DCT matrix is not orthonormal.')
    stats = common.read_json(PACKAGE/V17.FFTRADNET_FILES['config'])['dataset']['statistics']
    nearly, _ = adaradar.compress(spectrum, 1., 24, stats['input_mean'], stats['input_std'])  # no pruning, 24 bits: near-transparent
    error = float((nearly-spectrum).abs().square().sum()/spectrum.abs().square().sum())
    if error > 1e-9:
        raise AssertionError(f'AdaRadar without pruning is not transparent (NMSE {error:.2e}).')
    same = runtime.receive_spectrum(spectrum)
    if not torch.equal(same['detector']['base_peaks'], original['detector']['base_peaks']):
        raise AssertionError('receive_spectrum differs from the receiver.')
    rates = {}
    for name, spec in ((n, s) for n, s in ADARADAR if s['kind'] == 'adaradar'):
        _, bits = adaradar.compress(spectrum, spec['prune_ratio'], spec['bits'], stats['input_mean'], stats['input_std'])
        rates[name] = dict(bpcs=bits['total']/spectrum.numel(), bpcs_nonzero_only=bits['total_nonzero_only']/spectrum.numel(),
                           official_compression_ratio=bits['official_compression_ratio'])
    print('V18 AdaRadar pre-flight PASS:', json.dumps(dict(no_pruning_nmse=error, rates=rates)), flush=True)
    stream, _, _ = codec.encode_closed_loop(runtime.cube(row), runtime.receive, original['detector'], 10**(4.18/10),
                                            gamma_db=300., beta=2., seed=V17.SEED)
    learned, kt = entropy.learned_payload_bits(model, entropy.compact(stream)), codec.ecdq_bits(stream)['payload']
    if learned >= kt:
        raise AssertionError('The frozen V17 entropy model does not beat the KT code on a C stream.')
    print(f'Frozen V17 entropy model: learned payload {learned} bits vs KT {kt} bits ({1-learned/kt:.1%} saving).', flush=True)
    V17.run_check(runtime, out, expected)


def run(runtime, out, expected, log_every=25):
    checked = common.read_json(Path(str(out)+'.CHECK.json'))
    if checked.get('context') != expected or checked.get('passed') is not True:
        raise ValueError('Run the matching CUDA check first.')
    if out.exists():
        raise FileExistsError('Preserve partial/completed results; never overwrite this study.')
    with common.exclusive(out):
        out.mkdir(parents=True)
        common.write_json(out/'manifest.json', expected)
        pd.concat([runtime.rows[r] for r in ('validation', 'test')], ignore_index=True).to_csv(out/'POPULATIONS.csv', index=False)
        start = time.time()
        log = lambda *a: print(*a, f'{time.time()-start:.0f}s', flush=True)
        model = load_frozen_model(runtime.device)
        shutil.copyfile(PACKAGE/FROZEN_MODEL['file'], out/'entropy_model.pt')
        receipt = out/'FROZEN_MODEL.json'
        V17.write_report(receipt, dict(completed=True, checkpoint='entropy_model.pt', checkpoint_sha256=common.sha(out/'entropy_model.pt'),
                                       source=FROZEN_MODEL['source'], trained_in_this_run=False, test_used_for_selection=False))
        calibration, curve = V17.calibrate(runtime)
        V17.write_report(out/'H0_CALIBRATION.json', calibration)
        curve.to_csv(out/'H0_SURVIVAL.csv', index=False)
        alphas = dict(ca=calibration['alphas'], os=calibration['os_cfar']['alphas'])
        log('alpha dB CA:', json.dumps(calibration['alphas_db']), 'OS:', json.dumps(calibration['os_cfar']['alphas_db']))
        perception = V17.Perception(runtime)
        records, label_log = [], []

        # 1. VALIDATION, then the beta choice, written before TEST is unlocked.
        roster = runtime.roster('validation')
        for i, (_, row) in enumerate(roster.iterrows()):
            records.extend(evaluate_cube(runtime, row, VALIDATION_METHODS, alphas, model, perception, label_log))
            if (i+1) % log_every == 0 or i+1 == len(roster):
                log(f'VALIDATION {i+1}/{len(roster)}')
        frame, summary = report_tables(records)
        cis = V17.recording_bootstrap(frame)
        seen, _ = perception.report(out)
        selection = select_beta(V17.gates(summary, seen, calibration, 'validation', cis))
        V17.write_report(out/'BETA_SELECTION.json', selection)
        log('beta* =', selection['beta_selected'], 'from VALIDATION:', json.dumps({k: v['passed'] for k, v in selection['candidates'].items()}))

        # 2. TEST once: AdaRadar, the references, and C at beta 2 (and beta*) as an anchor to V17.
        anchors = tuple((name, spec) for name, spec in C_POINTS if name in ('closed_full_b2', selection['method_selected']))
        test_methods = REFERENCES+anchors+ADARADAR
        runtime.authorize_test(receipt)
        log('TEST unlocked:', [m for m, _ in test_methods])
        roster = runtime.roster('test')
        for i, (_, row) in enumerate(roster.iterrows()):
            records.extend(evaluate_cube(runtime, row, test_methods, alphas, model, perception, label_log))
            if (i+1) % log_every == 0 or i+1 == len(roster):
                log(f'TEST {i+1}/{len(roster)}')
        if common.sha(out/'entropy_model.pt') != FROZEN_MODEL['sha256']:
            raise ValueError('The entropy model changed during the run.')

        # 3. Reports.
        frame, summary = report_tables(records)
        frame.to_csv(out/'RESULTS_BY_CUBE.csv', index=False)
        pd.DataFrame(label_log).to_csv(out/'LABELS.csv', index=False)
        summary.to_csv(out/'SUMMARY.csv', index=False)
        cis = V17.recording_bootstrap(frame)
        V17.write_report(out/'RECORDING_CI.json', cis)
        seen, agreement = perception.report(out)
        for role in ('validation', 'test'):
            V17.write_report(out/f'{role.upper()}_GATES.json', V17.gates(summary, seen, calibration, role, cis))
        table = adaradar_table(summary, seen, agreement)
        table.to_csv(out/'ADARADAR_VS_C_TEST.csv', index=False)
        runtime.dataset.assert_unchanged()
        test_gates = common.read_json(out/'TEST_GATES.json')
        V17.write_report(out/'SMOKE_COMPLETED.json', dict(
            completed=True, passed=True, context=expected, seed=V17.SEED, test_evaluated=True, test_used_for_selection=False,
            entropy_model_trained=False, frozen_model_sha256=FROZEN_MODEL['sha256'], beta_selected=selection['beta_selected'],
            beta_none_passed=selection['none_passed'], alphas_db=calibration['alphas_db'],
            os_alphas_db=calibration['os_cfar']['alphas_db'], cubes={r: len(runtime.roster(r)) for r in ('validation', 'test')},
            pre_registered_points_passed={m: v['passed'] for m, v in test_gates['operating_points'].items()},
            runtime_seconds=time.time()-start))
    print(REVISION.split('-')[0], 'follow-up run COMPLETE; start with BETA_SELECTION.json and ADARADAR_VS_C_TEST.csv:', out, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', choices=('count', 'check', 'run'), required=True)
    for key in ('dataset-root', 'reference-root', 'split-audit', 'out'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    dataset = data.prepare_dataset(args.dataset_root, args.reference_root)
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    table, scan = V17.inventory(dataset, args.split_audit)
    with pd.option_context('display.width', 200, 'display.max_rows', 200):
        print(table.sort_values(['role', 'recording']).to_string(index=False))
    print(json.dumps(scan, indent=1), flush=True)
    if args.stage == 'count':
        if not Path(str(out)+'.COUNT.csv').exists():
            table.to_csv(Path(str(out)+'.COUNT.csv'), index=False)
            V17.write_report(Path(str(out)+'.COUNT.json'), scan)
        return
    if not scan['matches_frozen_split']:
        raise ValueError('The dataset folder differs from the frozen V17 split; no rebalancing. See the inventory above.')
    if torch.device(args.device).type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('CUDA is required. There is no CPU fallback.')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    runtime = Runtime(dataset, args.split_audit, args.device)
    expected = json.loads(json.dumps(context(runtime, out, args.split_audit, scan)))
    print(json.dumps({k: expected[k] for k in ('revision', 'gpu')}, indent=1), flush=True)
    if args.stage == 'check':
        run_check(runtime, out, expected)
    else:
        run(runtime, out, expected)


if __name__ == '__main__':
    main()
