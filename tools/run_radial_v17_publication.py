"""V17 publication run: detection-preserving compression of raw RADIal ADC cubes.

WHAT IS COMPRESSED. The raw RADIal ADC cube: 16 Rx x 256 chirps x 512 samples,
int16 I/Q, i.e. 32 bits per complex sample (bpcs). WHAT MUST SURVIVE. The
detections of the unchanged RADIal receiver (range FFT, Doppler FFT, cyclic
11x11 CA-CFAR, 5x5 NMS) and the output of a learned perception network.

THE CODEC (frozen in V14-V16; nothing is tuned in V17)
  * transform: orthonormal 2-D FFT per Rx, so coding error stays inside the
    range-Doppler cell (and window main lobe) where it is made;
  * quantizer: subtractive-dither uniform quantization (ECDQ; Schuchman 1964,
    Zamir & Feder 1992) with step beta x tile noise level. The error is white,
    signal independent, of variance step^2/12: the CFAR background rises by a
    predictable 10 log10(1 + beta^2/12) dB (1.25 dB at beta = 2);
  * closed loop (analysis by synthesis; Atal & Remde 1982): the encoder decodes
    its own stream, runs the receiver, and refines the CA-CFAR window of every
    detection that would change;
  * learned conditional entropy model (the only trained part, ~7k parameters;
    Balle et al. 2018, Minnen & Singh 2020): changes how many bits the same
    integers cost, never the decoded cube. Rate = min(learned, per-tile KT) + 1.
  Operating points (beta = 2):
    A closed_g0_b2          CFAR support coded, background synthesized (Blau & Michaeli 2019)
    B closed_strict2_g0_b2  A + every strict lost/new CFAR decision repaired
    C closed_full_b2        whole map coded, no synthesis (keeps what learned receivers use)
  Rate curve: C with beta 1.5, 2.5, 3 (TEST reporting only).

BASELINES (radar_sim/radial_v17_baselines.py): block adaptive quantization
(BAQ 1/2/3 bit; Kwok & Johnson 1989, Lloyd-Max levels), plain ADC bit-depth
reduction (uniform 2/3/4 bit), an AdaRadar-style static block-DCT codec (keep
the largest 8/14/20 % of each 32x64 block, 8-bit; AdaRadar, CVPR 2026, has no
public code, so its feedback controller is not reproduced), the V12 global-DCT
codec and open-loop ECDQ (= C without the closed loop).

RECEIVERS AND METRICS
  * CA-CFAR at alpha_1 (1 false peak per map on noise-only cubes, Neyman-Pearson)
    is the receiver inside the closed loop. Outside the loop (generalization):
    CA-CFAR at alpha_0.1 and OS-CFAR (Rohling 1983, k = 84 of 112) at its own
    alpha_1 / alpha_0.1;
  * detection-set agreement with the ORIGINAL cube: strict lost/new (match
    within the +-2 NMS radius) and robust lost/new (decisions 1 dB clear of the
    threshold and of the NMS rival); labels: preserved if an original detection
    in the label gate is matched;
  * perception: the official, frozen FFTRadNet (Rebut et al., CVPR 2022):
    F1/mAR change vs the original cube, objects kept/new, free-space IoU;
  * statistics: recording-level cluster bootstrap (frames of one recording are
    correlated), 2000 draws, 95 % percentile intervals.

PROTOCOL. GLOBAL_SPLIT_V17.csv: recording-disjoint 60/20/20 split of ALL
available cubes (make_split.py; TEST holds only recordings that never informed a
V12-V16 decision). TRAIN fits the entropy model; every 3rd VALIDATION cube
selects its epoch; the checkpoint is then frozen and hashed; VALIDATION is
evaluated once; only then is TEST unlocked, and TEST is evaluated once.
Every TEST stream of A, B and C is written to a real file and decoded from that
file alone. CUDA for ADC decoding, transforms, codec, receivers and networks.

STAGES: count (metadata-only inventory of the dataset folder), check (CUDA
pre-flight on TRAIN cubes), run (the study).
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import importlib.util
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from radar_sim import radial_fftradnet_eval as fftradnet
from radar_sim import radial_v17_baselines as baselines
from radar_sim import radial_v17_bitstream as bitstream
from radar_sim import radial_v17_receivers as receivers
from radar_sim.radial_adaptive_gpu_physics import RadialGPUPhysics, cyclic_cfar
from radar_sim.radial_dct_gpu_physics import BlockDCTCodecMixin


def sibling(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V14 = sibling('run_radial_v14_full')  # frozen-role runtime, closed-loop operating point, rate bookkeeping
V13 = V14.V13                         # receiver, reference codecs, per-cube scoring, H0 calibration
common, data, partition, codec, entropy = V13.common, V13.data, V13.partition, V13.codec, V14.entropy

REVISION = '17.0.0-publication-run-60-20-20'
SEED = 20261030
PACKAGE = Path(__file__).resolve().parent/'v17_publication_gpu'
FFTRADNET_FILES = dict(model='FFTRadNet.py', config='config_FFTRadNet_192_56.json', weights='fftradnet_state_dict.pt')
# FFTRadNet's own held-out sequences (FFTRadNet/dataset/dataloader.py); it was trained on all others.
FFTRADNET_HELD_OUT = frozenset({'RECORD@2020-11-22_12.49.56', 'RECORD@2020-11-22_12.11.49', 'RECORD@2020-11-22_12.28.47',
                                'RECORD@2020-11-21_14.25.06', 'RECORD@2020-11-22_12.45.05', 'RECORD@2020-11-22_12.25.47',
                                'RECORD@2020-11-22_12.03.47', 'RECORD@2020-11-22_12.54.38'})
TRAINING = dict(epochs=15, batch=32768, lr=2e-3, train_samples_per_stream=4096,
                selection_samples_per_stream=2048, selection_every_nth_validation_cube=3)
# Pre-registered pass criteria (unchanged since V14/V15; generalization added before V17 TEST).
GATE = dict(robust_lost_fraction=.01, perception_drop=.02)
BOOTSTRAP_DRAWS = 2000
BITSTREAM_WORKERS = max(1, min(16, (os.cpu_count() or 2)-2))

OPERATING_POINTS = (
    ('closed_g0_b2', dict(kind='closed', gamma=0., beta=2.)),                                      # A
    ('closed_strict2_g0_b2', dict(kind='closed', gamma=0., beta=2., strict=True, strict_new=True)),  # B
    ('closed_full_b2', dict(kind='closed', gamma=300., beta=2.)),                                  # C (gamma 300 dB: whole map)
)
RATE_CURVE = tuple((f'closed_full_b{tag}', dict(kind='closed', gamma=300., beta=beta))
                   for tag, beta in (('1p5', 1.5), ('2p5', 2.5), ('3', 3.)))
REFERENCES = (
    ('lossless_adc', dict(kind='lossless')),
    ('legacy_dct_r025_q8', dict(kind='legacy_dct')),
    # Control, not a codec: independent noise 20 dB below the receiver noise shows inherently fragile decisions.
    ('perturb_m20db', dict(kind='perturb', level_db=-20.)),
    ('ecdq_b1', dict(kind='ecdq', beta=1.)),
    ('ecdq_b2', dict(kind='ecdq', beta=2.)),  # C without the closed loop
)
BASELINES = (*((f'baq_{n}bit', dict(kind='baq', bits=n)) for n in (1, 2, 3)),
             *((f'uniform_{n}bit', dict(kind='uniform', bits=n)) for n in (2, 3, 4)),
             *((f'blockdct_r{round(rho*100):03d}_q8', dict(kind='blockdct', rho=rho, q=8)) for rho in (.08, .14, .20)))
VALIDATION_METHODS = REFERENCES+OPERATING_POINTS
TEST_METHODS = REFERENCES+BASELINES+OPERATING_POINTS+RATE_CURVE
SHORT = {'closed_g0_b2': 'A', 'closed_strict2_g0_b2': 'B', 'closed_full_b2': 'C', 'closed_full_b1p5': 'C β1.5',
         'closed_full_b2p5': 'C β2.5', 'closed_full_b3': 'C β3', 'legacy_dct_r025_q8': 'V12 DCT',
         'ecdq_b1': 'ECDQ β1 open loop', 'ecdq_b2': 'ECDQ β2 open loop'}
NAMES = {'closed_g0_b2': 'A', 'closed_strict2_g0_b2': 'B', 'closed_full_b2': 'C (beta 2)', 'closed_full_b1p5': 'C (beta 1.5)',
         'closed_full_b2p5': 'C (beta 2.5)', 'closed_full_b3': 'C (beta 3)'}
PERCEPTION_SKIPPED = frozenset({'lossless_adc'})  # decodes to the original cube
BITSTREAM_POINTS = tuple(name for name, _ in OPERATING_POINTS)
CONFIG = dict(
    operating_points=dict(OPERATING_POINTS), rate_curve=dict(RATE_CURVE), references=dict(REFERENCES), baselines=dict(BASELINES),
    validation_methods=[m for m, _ in VALIDATION_METHODS], test_methods=[m for m, _ in TEST_METHODS],
    training=TRAINING, gates=GATE, H0_maps=V13.H0_MAPS, false_peaks_per_map=V13.FALSE_PEAKS_PER_MAP,
    margin_db=V13.MARGIN_DB, label_gate=list(V13.LABEL_GATE), refinement_betas=list(codec.REFINE_BETAS),
    os_cfar=dict(window='11x11, 3x3 guard, 112 training cells', order=receivers.ORDER, reference='Rohling 1983'),
    baq=dict(block=list(baselines.BAQ_BLOCK), scale_bits=baselines.BAQ_SCALE_BITS, levels='Lloyd-Max for N(0,1)'),
    perception=dict(network='official FFTRadNet_RA_192_56 (epoch 78), frozen', drop_gate=GATE['perception_drop'],
                    metric='official: thresholds 0.1-0.9, NMS IoU 0.05, 5-100 m, IoU 0.5, easy labels',
                    held_out_recordings=sorted(FFTRADNET_HELD_OUT)),
    bitstream=dict(points=list(BITSTREAM_POINTS), role='test', coder='per-tile adaptive KT arithmetic coder (WNC 1987)'),
    rate='closed loop: min(learned, per-tile KT) + 1 flag bit + all side information; others: their own syntax',
    test_policy='TEST unlocked once, after the entropy-model checkpoint is frozen and hashed',
    sync_tolerance_us=data.TOLERANCE_US)


class BlockDCT(BlockDCTCodecMixin, RadialGPUPhysics):
    """AdaRadar-style static codec: 32x64 raw-ADC block DCT, keep the largest rho of each block, q-bit values."""


class Runtime(V14.Runtime):
    """Frozen V17 roles, all cubes of each role; TEST stays locked until ``authorize_test``."""

    def __init__(self, dataset, assignment, device, rows=None):
        if rows is None:
            frames = partition.assign_frozen_partitions(dataset, assignment, bounded=False)
            rows = {role: frame.assign(role=role) for role, frame in zip(partition.ROLES, frames)}
        super().__init__(dataset, assignment, device, rows=rows)
        self.block_dct = BlockDCT(dataset.reference_root, device=self.device, engineering=self.engineering)
        order = pd.concat([rows[role] for role in partition.ROLES], ignore_index=True).record_name
        self.positions = {name: i for i, name in enumerate(order)}

    def seed(self, row):
        """One dither/synthesis seed per cube, identical whenever the cube is encoded."""
        return SEED+self.positions[row.record_name]


def write_report(path, value):
    """JSON receipt without NaN (written as null) and with numpy scalars as Python numbers."""
    def clean(item):
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(v) for v in item]
        if isinstance(item, np.generic):
            item = item.item()
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item
    common.write_json(path, clean(value))


def timed(device, function, *args, **kwargs):
    """Run ``function``; wall time in ms with the GPU queue drained before and after."""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = function(*args, **kwargs)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return result, 1e3*(time.perf_counter()-start)


# ---------------------------------------------------------------------------
# Dataset inventory (count stage) and the frozen split
# ---------------------------------------------------------------------------
def provenance_path(assignment):
    return Path(assignment).with_name(Path(assignment).stem+'.provenance.json')


def inventory(dataset, assignment):
    """Metadata-only scan: usable labelled cubes per recording and per V17 role (no ADC body is read)."""
    mapping, _ = partition.load_global_assignment(assignment)
    provenance = common.read_json(provenance_path(assignment))
    if provenance['csv_sha256'] != common.sha(assignment):
        raise ValueError('GLOBAL_SPLIT_V17.csv differs from the frozen provenance.')
    table = dataset.recording_audit[['recording', 'synchronized_frames', 'complete_frames',
                                     'usable_label_indexed_frames']].rename(columns={'usable_label_indexed_frames': 'usable_cubes'})
    table = table.assign(role=table.recording.map(mapping).fillna('UNASSIGNED'),
                         frozen_cubes=table.recording.map(provenance['cubes_by_recording']).fillna(-1).astype(int))
    mismatch = table.loc[table.usable_cubes.ne(table.frozen_cubes) & (table.usable_cubes.gt(0) | table.frozen_cubes.gt(0))]
    missing = sorted(set(provenance['cubes_by_recording'])-set(table.recording))
    by_role = table.loc[table.usable_cubes.gt(0)].groupby('role').usable_cubes.sum().astype(int).to_dict()
    scan = dict(recordings=int(table.usable_cubes.gt(0).sum()), total_usable_cubes=int(table.usable_cubes.sum()),
                cubes_by_role=by_role, frozen_cubes_by_role=provenance['cubes_by_role'],
                fractions_by_role={k: v/max(table.usable_cubes.sum(), 1) for k, v in by_role.items()},
                mismatched_recordings=mismatch.recording.tolist(), missing_recordings=missing,
                matches_frozen_split=mismatch.empty and not missing and 'UNASSIGNED' not in by_role,
                synchronization_tolerance_us=data.TOLERANCE_US, ADC_bodies_read=False)
    return table, scan


# ---------------------------------------------------------------------------
# One method on one cube
# ---------------------------------------------------------------------------
def encode_decode(runtime, spec, cube, original, alpha, seed):
    """Returns (decoded cube, bit accounting, ECDQ stream or None, GPU times in ms)."""
    kind, device = spec['kind'], runtime.device
    if kind == 'closed':
        (stream, decoded, history), encode_ms = timed(
            device, codec.encode_closed_loop, cube, runtime.receive, original['detector'], alpha, gamma_db=spec['gamma'],
            beta=spec['beta'], seed=seed, strict_lost=spec.get('strict', False), strict_new=spec.get('strict_new', False))
        again, decode_ms = timed(device, codec.decode_ecdq, stream, cube.dtype)
        if not torch.equal(again, decoded):
            raise AssertionError('The decoder differs from the decode the closed loop verified.')
        bits = codec.ecdq_bits(stream)
        bits.update(accounting='closed-loop ECDQ; arithmetic code length plus all side information',
                    closed_loop_history=json.dumps(history), closed_loop_iterations=len(history)-1)
        return decoded, bits, stream, dict(encode_ms=encode_ms, decode_ms=decode_ms, codec_ms=encode_ms+decode_ms)
    if kind == 'ecdq':
        stream, encode_ms = timed(device, codec.encode_ecdq, cube, beta=spec['beta'], seed=seed)
        decoded, decode_ms = timed(device, codec.decode_ecdq, stream, cube.dtype)
        bits = codec.ecdq_bits(stream)
        bits.update(accounting='open-loop ECDQ; arithmetic code length plus all side information',
                    predicted_background_rise_db=10*math.log10(1+spec['beta']**2/12))
        return decoded, bits, stream, dict(encode_ms=encode_ms, decode_ms=decode_ms, codec_ms=encode_ms+decode_ms)
    if kind in ('baq', 'uniform'):
        (decoded, bits), ms = timed(device, getattr(baselines, kind), cube, spec['bits'])
    elif kind == 'blockdct':
        def block_dct():
            dct = runtime.block_dct
            source = dct.prepare(cube)
            tiles = len(source['tiles'])
            decoded, meta = dct.encode(source, dct.tensor([spec['rho']]*tiles), dct.tensor([float(spec['q'])]*tiles))
            return decoded.to(cube.dtype), dict(total=int(meta['realized_bits']), accounting=(
                'AdaRadar-style static block DCT: largest rho per 32x64 block, 8-bit values, fixed-length syntax'))
        (decoded, bits), ms = timed(device, block_dct)
    else:
        (decoded, bits), ms = timed(device, V13.run_method, runtime, spec, cube, original, alpha, seed)
    return decoded, bits, None, dict(encode_ms=math.nan, decode_ms=math.nan, codec_ms=ms)


def evaluate_cube(runtime, row, methods, alphas, model, perception, label_log, bitstreams=None):
    """Every method on one cube; scored against the receivers' output on the ORIGINAL cube."""
    cube = runtime.cube(row)
    original = runtime.receive(cube)
    original_os = receivers.cyclic_os_cfar(original['detector']['power'])
    bins, label_count = runtime.label_bins(row.record_name)
    perception.start(row, original)
    records = []
    for name, spec in methods:
        decoded, bits, stream, times = encode_decode(runtime, spec, cube, original, alphas['ca']['a1'], runtime.seed(row))
        received = runtime.receive(decoded)
        record = V13.score(runtime, row, name, cube, original, decoded, bits, alphas['ca'], bins, label_count,
                           label_log, received)
        record.update(times)
        method_os = receivers.cyclic_os_cfar(received['detector']['power'])
        for key, alpha in alphas['os'].items():
            metrics = codec.detection_metrics(original_os, method_os, alpha, V13.MARGIN_DB)
            record.update({f'os_{key}_{k}': v for k, v in metrics.items()})
        if spec['kind'] == 'closed':
            learned, record['entropy_model_ms'] = timed(runtime.device, entropy.learned_payload_bits, model,
                                                         entropy.compact(stream))
            record.update(V14.rate_fields(bits, learned, cube.numel()))
            if bitstreams is not None and name in BITSTREAM_POINTS:
                bitstreams.submit(row, name, stream, decoded, record)
        elif stream is not None:
            record['bpcs_kt'] = record['bpcs']
        if name not in PERCEPTION_SKIPPED:
            perception.add(name, received)
        records.append(record)
        del decoded, received, stream
    return records


# ---------------------------------------------------------------------------
# Bitstreams: every TEST stream of A, B, C as a real file
# ---------------------------------------------------------------------------
class Bitstreams:
    """Writes each stream to a file and decodes it from the file bytes alone.

    The first cube of every TEST recording (per operating point) is checked in
    this process: the cube is decoded on the GPU from the file and compared bit
    for bit with the encoder's decode; these files are kept as evidence. All
    other streams go to CPU worker processes running the reference arithmetic
    coder, which rebuild the stream from the file and compare every field the
    decoder reads (the decoder is deterministic, so equal fields give the same
    cube); those files are deleted after the check to save disk.
    """

    def __init__(self, out, workers):
        self.folder = out/'bitstreams'
        self.folder.mkdir()
        self.pool = concurrent.futures.ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'),
                                                           initializer=bitstream.worker_init)
        self.pending, self.rows, self.verified, self.limit = collections.deque(), [], set(), 2*workers

    def submit(self, row, name, stream, decoded, record):
        path = self.folder/f'{row.record_name}_{name}.rdv17'
        base = dict(record_name=row.record_name, scene_name=row.scene_name, method=name,
                    bpcs_kt=record['bpcs_kt'], bpcs_reported=record['bpcs'])
        if (row.scene_name, name) not in self.verified:
            self.verified.add((row.scene_name, name))
            result = bitstream.roundtrip(stream, path, decoded.dtype)
            self.rows.append(dict(base, verification='GPU decode from file, bit-exact', **result))
            return
        while len(self.pending) >= self.limit:
            self._collect()
        transport = {k: (v.cpu().to(torch.int32) if k == 'index' else v.cpu()) if isinstance(v, torch.Tensor) else v
                     for k, v in stream.items()}
        self.pending.append((base, self.pool.submit(bitstream.file_check, transport, path, False)))

    def _collect(self):
        base, future = self.pending.popleft()
        self.rows.append(dict(base, verification='stream rebuilt from file, all fields identical', **future.result()))

    def finish(self):
        while self.pending:
            self._collect()
        self.close()
        return pd.DataFrame(self.rows)

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)


def bitstream_summary(frame):
    column = lambda name: frame[name].fillna(True).astype(bool) if name in frame else pd.Series(True, index=frame.index)
    exact = column('bit_exact') & column('stream_identical')
    return dict(streams=len(frame), all_decoded_from_file_exactly=bool(exact.all()), failures=int((~exact).sum()),
                gpu_bit_exact_files=int(frame['bit_exact'].notna().sum()) if 'bit_exact' in frame else 0,
                by_method={m: dict(cubes=len(g), file_bpcs_mean=float(g.file_bpcs.mean()),
                                   ideal_kt_bpcs_mean=float(g.bpcs_kt.mean()), reported_bpcs_mean=float(g.bpcs_reported.mean()),
                                   file_over_kt=float(g.file_bpcs.sum()/g.bpcs_kt.sum()-1),
                                   cpu_entropy_encode_s_median=float(g.cpu_entropy_encode_s.median()),
                                   cpu_entropy_decode_s_median=float(g.cpu_entropy_decode_s.median()))
                           for m, g in frame.groupby('method', sort=False)},
                note='Files implement the per-tile KT coder; the learned-model coder is verified by coder_check.')


# ---------------------------------------------------------------------------
# Perception: frozen official FFTRadNet
# ---------------------------------------------------------------------------
def f1_and_recall(total):
    """Official mAP/mAR/F1 (fftradnet.metrics_from_counts), vectorized over leading axes of [..., 9, 3]."""
    tp, fp, fn = total[..., 0], total[..., 1], total[..., 2]
    precision = np.where(tp > 0, tp/np.maximum(tp+fp, 1), 0.).mean(-1)
    recall = np.where(tp > 0, tp/np.maximum(tp+fn, 1), 0.).mean(-1)
    f1 = np.where(precision+recall > 0, 2*precision*recall/np.maximum(precision+recall, 1e-300), 0.)
    return f1, recall


class Perception:
    """FFTRadNet on the original and every decoded cube of each VALIDATION/TEST frame.

    Per method and frame only small numbers are kept: official counts vs the easy
    labels, objects kept/new vs the network's output on the original cube, and
    free-space IoU vs the original free-space mask. Caveat: the official
    radar_FFT input came from a non-public Valeo library and window, so absolute
    scores are below the published ones; the comparison is paired.
    """

    def __init__(self, runtime):
        self.runtime = runtime
        self.net, self.config = fftradnet.load(PACKAGE/FFTRADNET_FILES['model'], PACKAGE/FFTRADNET_FILES['config'],
                                               PACKAGE/FFTRADNET_FILES['weights'], runtime.device)
        self.frames = []

    def run(self, spectrum):
        detection, segmentation = fftradnet.predict(self.net, fftradnet.network_input(spectrum, self.config)[None])
        return fftradnet.decode(detection[0], self.config), segmentation[0].cpu().numpy()

    def easy_labels(self, row):
        labels = self.runtime.labels.loc[self.runtime.labels.record_name.eq(row.record_name)]
        if labels.empty:
            return np.zeros((0, 2)) if bool(getattr(row, 'explicit_negative', False)) else None
        easy = labels.loc[labels.Difficult.eq(0)]
        return None if easy.empty else easy[['range_m', 'azimuth_deg']].to_numpy(dtype=float)

    def start(self, row, original):
        prediction, self.segmentation = self.run(original['spectrum'])
        labels = self.easy_labels(row)
        self.prediction = prediction
        self.frame = dict(role=row.role, scene_name=row.scene_name, labelled=labels is not None, labels=labels,
                          counts=dict(original=fftradnet.frame_counts(prediction, labels) if labels is not None else None),
                          agreement={}, freespace={})
        self.frames.append(self.frame)

    def add(self, name, received):
        prediction, segmentation = self.run(received['spectrum'])
        if self.frame['labelled']:
            self.frame['counts'][name] = fftradnet.frame_counts(prediction, self.frame['labels'])
        self.frame['agreement'][name] = fftradnet.agreement(self.prediction, prediction)
        self.frame['freespace'][name] = fftradnet.freespace_iou(self.segmentation, segmentation)

    def report(self, out):
        rows, agreement_rows = [], []
        rng = np.random.default_rng(SEED)
        for role in ('validation', 'test'):
            frames = [f for f in self.frames if f['role'] == role]
            if not frames:
                continue
            methods = list(frames[0]['freespace'])
            labelled = [f for f in frames if f['labelled']]
            scenes = np.array([f['scene_name'] for f in labelled])
            for subset, mask in (('all', np.ones(len(labelled), bool)),
                                 ('fftradnet_held_out', np.isin(scenes, list(FFTRADNET_HELD_OUT))),
                                 ('fftradnet_training', ~np.isin(scenes, list(FFTRADNET_HELD_OUT)))):
                if not mask.any():
                    continue
                total = {m: np.stack([f['counts'][m] for f, k in zip(labelled, mask) if k]).sum(0) for m in ['original']+methods}
                base = fftradnet.metrics_from_counts(total['original'])
                for m in ['original']+methods:
                    metric = fftradnet.metrics_from_counts(total[m])
                    rows.append(dict(role=role, subset=subset, method=m, frames=int(mask.sum()),
                                     labels=int(sum(len(f['labels']) for f, k in zip(labelled, mask) if k)), **metric,
                                     dF1=metric['F1']-base['F1'], dmAR=metric['mAR']-base['mAR'], dmAP=metric['mAP']-base['mAP']))
            # Recording-level bootstrap of the change vs the original cube (same draws for every method).
            recordings = sorted(set(scenes))
            per_recording = {m: np.stack([np.stack([f['counts'][m] for f in labelled if f['scene_name'] == s]).sum(0)
                                          for s in recordings]) for m in ['original']+methods}
            draws = rng.integers(0, len(recordings), (BOOTSTRAP_DRAWS, len(recordings)))
            f1_0, recall_0 = f1_and_recall(per_recording['original'][draws].sum(1))
            for m in methods:
                f1, recall = f1_and_recall(per_recording[m][draws].sum(1))
                row = next(r for r in rows if r['role'] == role and r['subset'] == 'all' and r['method'] == m)
                row.update(dF1_ci_low=float(np.percentile(f1-f1_0, 2.5)), dF1_ci_high=float(np.percentile(f1-f1_0, 97.5)),
                           dmAR_ci_low=float(np.percentile(recall-recall_0, 2.5)),
                           dmAR_ci_high=float(np.percentile(recall-recall_0, 97.5)), recordings=len(recordings))
            for m in methods:
                total = collections.Counter()
                for f in frames:
                    total.update(f['agreement'][m])
                agreement_rows.append(dict(role=role, method=m, frames=len(frames), **total,
                    objects_kept_fraction=total['objects_kept']/max(total['objects_original'], 1),
                    objects_new_per_frame=total['objects_new']/len(frames),
                    freespace_iou_mean=float(np.mean([f['freespace'][m] for f in frames])),
                    freespace_iou_p10=float(np.percentile([f['freespace'][m] for f in frames], 10))))
        table, agreement = pd.DataFrame(rows), pd.DataFrame(agreement_rows)
        table.to_csv(out/'PERCEPTION_SUMMARY.csv', index=False)
        agreement.to_csv(out/'PERCEPTION_AGREEMENT.csv', index=False)
        return table, agreement


# ---------------------------------------------------------------------------
# Calibration, entropy model, summaries, gates, figures
# ---------------------------------------------------------------------------
def calibrate(runtime):
    """Neyman-Pearson thresholds of both CFAR receivers on the same noise-only cubes."""
    calibration, curve = V13.calibrate(runtime)
    p = runtime.physics
    scores = codec.h0_scores(V13.H0_MAPS, p.shape, p.range_window, p.doppler_window, receivers.cyclic_os_cfar,
                             runtime.device, V13.SEED)
    alphas = {k: codec.threshold_for_rate(scores, V13.H0_MAPS, rate) for k, rate in V13.FALSE_PEAKS_PER_MAP.items()}
    margin = 10**(V13.MARGIN_DB/10)
    calibration['os_cfar'] = dict(alphas=alphas, alphas_db={k: 10*math.log10(v) for k, v in alphas.items()},
                                  H0_peaks_per_map=len(scores)/V13.H0_MAPS,
                                  H0_robust_new_per_map={k: float((scores > v*margin).sum())/V13.H0_MAPS
                                                         for k, v in alphas.items()})
    return calibration, curve


def closed_loop_symbols(runtime, rows, alpha, per_stream, generator, log, label, rate_rows=None):
    """A/B/C streams of ``rows``; random coded components as (float16 features, int8 classes)."""
    features, targets = [], []
    numel = int(np.prod(runtime.physics.shape))
    for i, (_, row) in enumerate(rows.iterrows()):
        cube = runtime.cube(row)
        original = runtime.receive(cube)
        for name, spec in OPERATING_POINTS:
            _, bits, stream = V14.run_operating_point(runtime, spec, cube, original, alpha, runtime.seed(row))
            f, t = entropy.sample(entropy.compact(stream), per_stream, generator)
            features.append(f)
            targets.append(t)
            if rate_rows is not None:
                rate_rows.append(dict(role=row.role, record_name=row.record_name, scene_name=row.scene_name, method=name,
                                      bpcs_kt=(bits['total']+1)/numel, support_fraction=bits['support_fraction'],
                                      refine_fraction=bits['refine_fraction'], closed_loop_iterations=bits['closed_loop_iterations']))
        if (i+1) % 100 == 0 or i+1 == len(rows):
            log(f'{label} {i+1}/{len(rows)}')
    return torch.cat(features), torch.cat(targets)


def summarize(frame):
    """Per role and method: V13 summary (CA-CFAR at alpha_1/alpha_0.1, labels, distortion) + OS-CFAR, rate, time."""
    summary = V13.summarize(frame)
    rows = []
    for (role, method), g in frame.groupby(['role', 'method'], sort=False):
        row = dict(role=role, method=method, bpcs_kt=g.bpcs_kt.mean(), bpcs_learned=g.bpcs_learned.mean(),
                   learned_chosen_fraction=g.learned_chosen.mean(), codec_ms_median=g.codec_ms.median(),
                   encode_ms_median=g.encode_ms.median(), decode_ms_median=g.decode_ms.median(),
                   entropy_model_ms_median=g.entropy_model_ms.median(), closed_loop_iterations_mean=g.closed_loop_iterations.mean())
        for key in V13.FALSE_PEAKS_PER_MAP:
            k = f'os_{key}'
            total, stable = g[f'{k}_original'].sum(), g[f'{k}_original_robust'].sum()
            row.update({f'{k}_original': int(total), f'{k}_lost_fraction': g[f'{k}_lost'].sum()/max(total, 1),
                        f'{k}_lost_robust_fraction': g[f'{k}_lost_robust'].sum()/max(stable, 1),
                        f'{k}_new_per_map': g[f'{k}_new'].mean(), f'{k}_new_robust_per_map': g[f'{k}_new_robust'].mean()})
        rows.append(row)
    return summary.merge(pd.DataFrame(rows), on=['role', 'method'])


# statistic: (numerator column, denominator column or None = per cube)
CI_STATISTICS = dict(bpcs=('bpcs', None), a1_lost_robust_fraction=('a1_lost_robust', 'a1_original_robust'),
                     a1_lost_fraction=('a1_lost', 'a1_original'), a1_new_robust_per_map=('a1_new_robust', None),
                     a1_new_per_map=('a1_new', None), a01_lost_robust_fraction=('a01_lost_robust', 'a01_original_robust'),
                     os_a1_lost_robust_fraction=('os_a1_lost_robust', 'os_a1_original_robust'),
                     os_a1_new_robust_per_map=('os_a1_new_robust', None),
                     labels_lost_fraction=('labels_lost', 'labels_detected_original'))


def recording_bootstrap(frame):
    """95 % percentile intervals, resampling recordings (clusters) with replacement; same draws for all methods."""
    rng = np.random.default_rng(SEED)
    out = []
    for role, by_role in frame.groupby('role', sort=False):
        recordings = sorted(by_role.scene_name.unique())
        draws = rng.integers(0, len(recordings), (BOOTSTRAP_DRAWS, len(recordings)))
        for method, g in by_role.groupby('method', sort=False):
            columns = sorted({c for pair in CI_STATISTICS.values() for c in pair if c})
            sums = g.groupby('scene_name')[columns].sum(min_count=1).reindex(recordings)
            cubes = g.groupby('scene_name').size().reindex(recordings).to_numpy(float)
            row = dict(role=role, method=method, recordings=len(recordings), cubes=len(g))
            for stat, (num, den) in CI_STATISTICS.items():
                top = sums[num].to_numpy(float)
                if not np.isfinite(top).all():
                    row[stat+'_ci'] = None
                    continue
                bottom = cubes if den is None else sums[den].to_numpy(float)
                value = top[draws].sum(1)/np.maximum(bottom[draws].sum(1), 1)
                row[stat+'_ci'] = [float(np.percentile(value, 2.5)), float(np.percentile(value, 97.5))]
            out.append(row)
    return out


def gates(summary, perception, calibration, role, cis):
    """Pre-registered criteria per operating point: CFAR (in the loop), generalization, perception."""
    rows = summary.loc[summary.role.eq(role)].set_index('method')
    seen = perception.loc[perception.role.eq(role) & perception.subset.eq('all')].set_index('method')
    ci = {r['method']: r for r in cis if r['role'] == role}
    floor_ca = calibration['H0_robust_new_per_map']['a1']+1
    floor_os = calibration['os_cfar']['H0_robust_new_per_map']['a1']+1
    limit = GATE['robust_lost_fraction']
    out = {}
    for name in [m for m, _ in OPERATING_POINTS+RATE_CURVE if m in rows.index]:
        r, p = rows.loc[name], seen.loc[name]
        cfar = dict(robust_lost_le_1pct=bool(r.a1_lost_robust_fraction <= limit),
                    robust_new_le_H0_floor_plus_1=bool(r.a1_new_robust_per_map <= floor_ca),
                    no_robust_label_lost=bool(r.labels_lost_robust == 0),
                    bitrate_below_legacy=bool(r.bpcs_mean < rows.loc['legacy_dct_r025_q8', 'bpcs_mean']))
        general = dict(os_cfar_robust_lost_le_1pct=bool(r.os_a1_lost_robust_fraction <= limit),
                       os_cfar_robust_new_le_H0_floor_plus_1=bool(r.os_a1_new_robust_per_map <= floor_os),
                       alpha_01_robust_lost_le_1pct=bool(r.a01_lost_robust_fraction <= limit))
        seen_ok = dict(F1_drop_le_gate=bool(p.dF1 >= -GATE['perception_drop']),
                       mAR_drop_le_gate=bool(p.dmAR >= -GATE['perception_drop']))
        out[name] = dict(
            name=NAMES.get(name, name), pre_registered=name in dict(OPERATING_POINTS),
            bpcs=float(r.bpcs_mean), bpcs_ci=ci[name]['bpcs_ci'], bpcs_kt=float(r.bpcs_kt),
            ratio_vs_raw_32bit=32/float(r.bpcs_mean), ratio_vs_lossless=float(rows.loc['lossless_adc', 'bpcs_mean'])/float(r.bpcs_mean),
            learned_rate_saving=1-float(r.bpcs_mean)/float(r.bpcs_kt),
            ca_cfar=dict(cfar, robust_lost_fraction=float(r.a1_lost_robust_fraction), robust_lost_ci=ci[name]['a1_lost_robust_fraction_ci'],
                         strict_lost_fraction=float(r.a1_lost_fraction), strict_new_per_map=float(r.a1_new_per_map),
                         robust_new_per_map=float(r.a1_new_robust_per_map), labels_lost_strict=int(r.labels_lost),
                         labels_detected_original=int(r.labels_detected_original), passed=all(cfar.values())),
            generalization=dict(general, os_cfar_robust_lost_fraction=float(r.os_a1_lost_robust_fraction),
                                alpha_01_robust_lost_fraction=float(r.a01_lost_robust_fraction), passed=all(general.values())),
            perception=dict(seen_ok, F1=float(p.F1), dF1=float(p.dF1), dF1_ci=[p.get('dF1_ci_low'), p.get('dF1_ci_high')],
                            dmAR=float(p.dmAR), passed=all(seen_ok.values())),
            timing_ms=dict(encode_median=float(r.encode_ms_median), decode_median=float(r.decode_ms_median),
                           entropy_model_median=float(r.entropy_model_ms_median)))
        out[name]['passed'] = out[name]['ca_cfar']['passed'] and out[name]['generalization']['passed'] and out[name]['perception']['passed']
    control = rows.loc['perturb_m20db']
    return dict(role=role, legacy_dct_bpcs=float(rows.loc['legacy_dct_r025_q8', 'bpcs_mean']),
                lossless_adc_bpcs=float(rows.loc['lossless_adc', 'bpcs_mean']),
                H0_robust_new_floor_per_map=dict(ca_cfar=floor_ca, os_cfar=floor_os),
                receiver_fragility_control=dict(strict_lost_fraction=float(control.a1_lost_fraction),
                                                robust_lost_fraction=float(control.a1_lost_robust_fraction),
                                                strict_new_per_map=float(control.a1_new_per_map),
                                                dF1=float(seen.loc['perturb_m20db'].dF1),
                                                meaning='decision changes caused by independent noise 20 dB below the receiver noise'),
                operating_points=out,
                pre_registered_points_passed={m: out[m]['passed'] for m, _ in OPERATING_POINTS if m in out})


def family(method):
    if method in NAMES:
        return 'proposed'
    return next((f for f in ('baq', 'uniform', 'blockdct') if method.startswith(f)), 'reference')


def rate_comparison(summary, perception, agreement):
    """TEST table of every method on one rate axis, and which baselines each operating point beats."""
    test = summary.loc[summary.role.eq('test')].set_index('method')
    seen = perception.loc[perception.role.eq('test') & perception.subset.eq('all')].set_index('method')
    agree = agreement.loc[agreement.role.eq('test')].set_index('method')
    rows = []
    for method, r in test.iterrows():
        s = seen.loc[method] if method in seen.index else None
        a = agree.loc[method] if method in agree.index else None
        rows.append(dict(method=method, name=NAMES.get(method, method), family=family(method), bpcs=r.bpcs_mean,
                         bpcs_kt=r.bpcs_kt, nmse=r.nmse_mean, ca_robust_lost=r.a1_lost_robust_fraction,
                         ca_strict_lost=r.a1_lost_fraction, ca_robust_new_per_map=r.a1_new_robust_per_map,
                         ca_strict_new_per_map=r.a1_new_per_map, ca01_robust_lost=r.a01_lost_robust_fraction,
                         os_robust_lost=r.os_a1_lost_robust_fraction, os_robust_new_per_map=r.os_a1_new_robust_per_map,
                         labels_lost_robust=r.labels_lost_robust, labels_lost_strict=r.labels_lost,
                         dF1=None if s is None else s.dF1, dF1_ci_low=None if s is None else s.get('dF1_ci_low'),
                         dF1_ci_high=None if s is None else s.get('dF1_ci_high'), dmAR=None if s is None else s.dmAR,
                         objects_kept_fraction=None if a is None else a.objects_kept_fraction,
                         objects_new_per_frame=None if a is None else a.objects_new_per_frame,
                         freespace_iou=None if a is None else a.freespace_iou_mean,
                         codec_ms_median=r.codec_ms_median, encode_ms_median=r.encode_ms_median,
                         decode_ms_median=r.decode_ms_median))
    table = pd.DataFrame(rows).sort_values('bpcs', na_position='last')
    lossy = table.loc[table.family.isin(['baq', 'uniform', 'blockdct']) | table.method.eq('legacy_dct_r025_q8')]
    verdict = {}
    for name in [m for m, _ in OPERATING_POINTS+RATE_CURVE]:
        mine = table.set_index('method').loc[name]
        rivals = lossy.loc[lossy.bpcs >= mine.bpcs]
        better = lambda k, low=True: bool(((mine[k] <= rivals[k]) if low else (mine[k] >= rivals[k])).all())
        verdict[name] = dict(bpcs=float(mine.bpcs), baselines_at_equal_or_higher_rate=rivals.method.tolist(),
                             lower_ca_robust_lost_than_all=better('ca_robust_lost'), lower_os_robust_lost_than_all=better('os_robust_lost'),
                             fewer_ca_robust_new_than_all=better('ca_robust_new_per_map'),
                             higher_dF1_than_all=better('dF1', low=False), higher_freespace_iou_than_all=better('freespace_iou', low=False))
    return table, verdict


def make_figures(out, comparison, history):
    """Rate curves on TEST (one rate axis, four quality panels) and the entropy-model training curve."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = out/'figures'
    folder.mkdir(exist_ok=True)
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter
    # First three slots of the all-pairs CVD-validated reference palette + neutral inks; marker shape is a
    # second identity channel. Zero losses are drawn at FLOOR on the log axes (stated in the axis label).
    style = {'proposed': ('#2a78d6', 'o', 'Proposed A, B, C (line: C at β = 3, 2.5, 2, 1.5)'),
             'baq': ('#eb6834', 's', 'BAQ 1/2/3 bit'), 'blockdct': ('#1baf7a', 'D', 'Block DCT + pruning, 8 bit (AdaRadar-style)'),
             'uniform': ('#52514e', '^', 'ADC bit-depth reduction 2/3/4 bit'),
             'reference': ('#9b9a95', 'x', 'V12 DCT codec; open-loop ECDQ β = 1, 2')}
    floor = 1e-4
    labelled = ('closed_g0_b2', 'closed_strict2_g0_b2', 'closed_full_b2', 'closed_full_b1p5', 'closed_full_b3')
    lossy = comparison.loc[comparison.bpcs.notna() & ~comparison.method.eq('lossless_adc')].copy()
    control = comparison.set_index('method').loc['perturb_m20db']
    panels = (('ca_robust_lost', 'CA-CFAR robust lost fraction\n(in the loop; 0 drawn at $10^{-4}$)', True),
              ('os_robust_lost', 'OS-CFAR robust lost fraction\n(outside the loop; 0 drawn at $10^{-4}$)', True),
              ('dF1', 'FFTRadNet F1 change vs original', False), ('freespace_iou', 'Free-space IoU vs original', False))
    with plt.rc_context({'font.family': 'serif', 'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.6), layout='constrained')
        for panel, (ax, (key, label, log)) in enumerate(zip(axes.ravel(), panels)):
            value = lambda frame: frame[key].astype(float).clip(lower=floor) if log else frame[key].astype(float)
            for fam, (color, marker, legend) in style.items():
                rows = lossy.loc[lossy.family.eq(fam)].sort_values('bpcs')
                line = rows.loc[rows.method.str.startswith('closed_full')] if fam == 'proposed' else rows
                if fam != 'reference':
                    ax.plot(line.bpcs, value(line), color=color, lw=1.5, zorder=3)
                filled = dict(edgecolors='white', linewidths=.6) if marker != 'x' else {}
                ax.scatter(rows.bpcs, value(rows), color=color, marker=marker, s=22, label=legend, zorder=4, **filled)
                if key == 'dF1' and fam == 'proposed':
                    ax.errorbar(rows.bpcs, rows.dF1, yerr=[(rows.dF1-rows.dF1_ci_low).clip(lower=0),
                                                           (rows.dF1_ci_high-rows.dF1).clip(lower=0)],
                                fmt='none', ecolor=color, elinewidth=.8, capsize=2)
                named = rows.method.isin(labelled) if fam == 'proposed' else (rows.method.notna() if fam == 'reference' and panel == 0
                                                                              else rows.method.isna())
                for _, r in rows.loc[named].iterrows():
                    ax.annotate(SHORT.get(r.method, r.method), (r.bpcs, max(float(r[key]), floor) if log else float(r[key])),
                                fontsize=6, color='#52514e', xytext=(3, 3), textcoords='offset points')
            reference = control[key]
            if reference is not None and np.isfinite(float(reference)):
                ax.axhline(max(float(reference), floor) if log else float(reference), color='#9b9a95', lw=.8, ls=':')
            ax.set(xscale='log', xlabel='Bits per complex sample (raw ADC: 32)', ylabel=label)
            ax.xaxis.set_major_locator(FixedLocator([1, 1.5, 2, 3, 4, 6, 8]))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
            ax.xaxis.set_minor_formatter(NullFormatter())
            if log:
                ax.set_yscale('log')
            ax.grid(alpha=.25, which='major', lw=.5)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        handles.append(Line2D([], [], color='#9b9a95', lw=.8, ls=':'))
        labels.append('Control: original + noise 20 dB below receiver noise')
        fig.legend(handles, labels, loc='outside upper center', ncol=2, frameon=False, fontsize=7)
        fig.savefig(folder/'rate_curves_test.pdf')
        fig.savefig(folder/'rate_curves_test.png', dpi=250)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(3.4, 2.4), layout='constrained')
        frame = pd.DataFrame(history)
        ax.plot(frame.epoch, frame.train_bits_per_symbol, color='#2a78d6', lw=1.5, label='TRAIN')
        ax.plot(frame.epoch, frame.val_bits_per_symbol, color='#eb6834', lw=1.5, label='VALIDATION (selection)')
        ax.set(xlabel='Epoch', ylabel='Class code length (bits/symbol)')
        ax.grid(alpha=.25, lw=.5)
        ax.legend(frameon=False, fontsize=7)
        fig.savefig(folder/'entropy_model_training.pdf')
        fig.savefig(folder/'entropy_model_training.png', dpi=250)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def source_hashes():
    files = [Path(__file__), Path(V14.__file__), Path(V13.__file__), Path(codec.__file__), Path(entropy.__file__),
             Path(partition.__file__), Path(data.__file__), Path(baselines.__file__), Path(receivers.__file__),
             Path(bitstream.__file__), Path(fftradnet.__file__), common.ROOT/'radar_sim/radial_adaptive_gpu_physics.py',
             common.ROOT/'radar_sim/radial_dct_gpu_physics.py', common.ROOT/'radar_sim/radial_dct_layout_gpu_physics.py',
             *(PACKAGE/name for name in FFTRADNET_FILES.values())]
    return {str(p.resolve().relative_to(common.ROOT)): common.sha(p) for p in files}


def context(runtime, out, assignment, scan):
    return dict(revision=REVISION, source_root=str(common.ROOT.resolve()), output=str(out),
                dataset_root=str(runtime.dataset.root), reference_root=str(runtime.dataset.reference_root),
                device=str(runtime.device), torch=torch.__version__, cuda_runtime=torch.version.cuda,
                gpu=torch.cuda.get_device_name(runtime.device) if runtime.device.type == 'cuda' else 'cpu-engineering',
                split=str(Path(assignment).resolve()), split_sha256=common.sha(assignment),
                split_provenance_sha256=common.sha(provenance_path(assignment)), inventory=scan,
                source_hashes=source_hashes(), dataset_identity=data.dataset_identity(runtime.dataset),
                populations={role: frame.record_name.tolist() for role, frame in runtime.rows.items()},
                configuration=CONFIG, seed=SEED, bitstream_workers=BITSTREAM_WORKERS)


def run_check(runtime, out, expected):
    """CUDA pre-flight on one TRAIN cube: receivers, codecs, rates, coders, file round trip, FFTRadNet."""
    path = Path(str(out)+'.CHECK.json')
    if path.exists():
        old = common.read_json(path)
        if old.get('passed') is True and old.get('context') == expected:
            print('Existing identity-bound V17 check: PASS.', flush=True)
            return
        raise ValueError('Existing check identity differs. Preserve it; use a fresh deployment.')
    if data.TOLERANCE_US != 20000 or runtime.dataset.tolerance_us != 20000:
        raise AssertionError('Label synchronization must be the official 20 ms.')
    p, device = runtime.physics, runtime.device
    alpha = codec.threshold_for_rate(codec.h0_scores(4, p.shape, p.range_window, p.doppler_window, cyclic_cfar, device, SEED), 4, 1.)
    row = runtime.roster('train').iloc[len(runtime.roster('train'))//2]
    cube = runtime.cube(row)
    numel = cube.numel()
    original, native = runtime.receive(cube), p.maps(cube)
    torch.testing.assert_close(original['RD'], native['RD'], rtol=0, atol=1e-9)
    if not torch.equal(original['detector']['base_peaks'], native['detectors']['RD']['base_peaks']):
        raise AssertionError('The RD receiver differs from the native RADIal receiver.')
    if not torch.equal(receivers.cyclic_os_cfar(original['detector']['power'])['base_peaks'], original['detector']['base_peaks']):
        raise AssertionError('OS-CFAR must share the CA-CFAR NMS peaks.')
    if float((codec.inverse(codec.forward(cube))-cube.to(torch.complex128)).abs().max()/cube.abs().max()) > 1e-12:
        raise AssertionError('FFT2 codec domain does not invert.')
    rates, streams = {}, {}
    for name, spec in BASELINES+REFERENCES+OPERATING_POINTS:
        decoded, bits, stream, times = encode_decode(runtime, spec, cube, original, alpha, SEED)
        if not bool(torch.isfinite(torch.view_as_real(decoded)).all()):
            raise AssertionError(f'{name} decoded a non-finite cube.')
        rates[name] = dict(bpcs=bits['total']/numel if bits['total'] == bits['total'] else None, **times)
        if spec['kind'] == 'closed':
            streams[name] = stream
    for n in (1, 2, 3):
        if abs(rates[f'baq_{n}bit']['bpcs']-2*n) > .05:
            raise AssertionError('BAQ rate differs from 2n bits per complex sample plus side information.')
    summary = entropy.compact(streams['closed_full_b2'])
    for plane in (0, 7, 15):
        entropy.assert_causal(summary, plane)
    coder = entropy.coder_check(entropy.ContextModel().to(device), summary)
    if coder['real_bits'] > coder['ideal_bits']*1.01+64:
        raise AssertionError('Learned-model arithmetic coder exceeds its ideal length.')
    files = {}
    for name in BITSTREAM_POINTS:
        files[name] = bitstream.roundtrip(streams[name], Path(f'{out}.CHECK_{name}.rdv17'), cube.dtype)
        Path(f'{out}.CHECK_{name}.rdv17').unlink()
        if not files[name]['bit_exact'] or files[name]['file_over_ideal'] > .05:
            raise AssertionError(f'{name}: file decode is not bit-exact or the file exceeds its ideal length by >5%.')
    stream = streams['closed_g0_b2']
    transport = {k: (v.cpu().to(torch.int32) if k == 'index' else v.cpu()) if isinstance(v, torch.Tensor) else v
                 for k, v in stream.items()}
    with concurrent.futures.ProcessPoolExecutor(1, mp_context=multiprocessing.get_context('spawn'),
                                                initializer=bitstream.worker_init) as pool:
        worker = pool.submit(bitstream.file_check, transport, Path(f'{out}.CHECK_worker.rdv17'), False).result()
    if not worker['stream_identical']:
        raise AssertionError('CPU worker did not rebuild the stream from its file.')
    perception = Perception(runtime)
    prediction, segmentation = perception.run(original['spectrum'])
    if segmentation.shape != (1, 256, 224):
        raise AssertionError('Unexpected FFTRadNet segmentation shape.')
    result = dict(completed=True, passed=True, context=expected, record_name=row.record_name, rates_and_times=rates,
                  learned_coder=coder, files=files, worker_file_check=worker, bitstream_workers=BITSTREAM_WORKERS,
                  fftradnet_raw_cells=len(prediction),
                  gpu_peak_memory_gb=torch.cuda.max_memory_allocated(device)/1e9 if device.type == 'cuda' else 0.,
                  training_started=False, test_evaluated=False)
    write_report(path, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'context'}, indent=1, default=str), flush=True)
    print('V17 CUDA check complete.', flush=True)


def run(runtime, out, expected, log_every=25):
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

        # 1. Neyman-Pearson thresholds of both receivers on noise-only cubes.
        calibration, curve = calibrate(runtime)
        write_report(out/'H0_CALIBRATION.json', calibration)
        curve.to_csv(out/'H0_SURVIVAL.csv', index=False)
        alphas = dict(ca=calibration['alphas'], os=calibration['os_cfar']['alphas'])
        log('alpha dB CA:', json.dumps(calibration['alphas_db']), 'OS:', json.dumps(calibration['os_cfar']['alphas_db']))

        # 2. Entropy-model data: all TRAIN cubes; epoch selection on every n-th VALIDATION cube.
        generator = torch.Generator(device=runtime.device).manual_seed(SEED)
        train_rates = []
        rows_t, targets_t = closed_loop_symbols(runtime, runtime.roster('train'), alphas['ca']['a1'],
                                                TRAINING['train_samples_per_stream'], generator, log, 'TRAIN', train_rates)
        pd.DataFrame(train_rates).to_csv(out/'TRAIN_RATES.csv', index=False)
        selection = runtime.roster('validation').iloc[::TRAINING['selection_every_nth_validation_cube']]
        rows_v, targets_v = closed_loop_symbols(runtime, selection, alphas['ca']['a1'],
                                                TRAINING['selection_samples_per_stream'], generator, log, 'VALIDATION selection')

        # 3. Train, select the epoch, freeze and hash the checkpoint.
        torch.manual_seed(SEED)
        model = entropy.ContextModel().to(runtime.device)
        log(f'Entropy model: {sum(p.numel() for p in model.parameters())} parameters, '
            f'{len(rows_t)} TRAIN / {len(rows_v)} VALIDATION symbols')
        history, best_epoch = entropy.train(model, rows_t, targets_t, rows_v, targets_v, epochs=TRAINING['epochs'],
                                            batch=TRAINING['batch'], lr=TRAINING['lr'], seed=SEED,
                                            log=lambda r: log('epoch', json.dumps(r)))
        del rows_t, targets_t, rows_v, targets_v
        pd.DataFrame(history).to_csv(out/'ENTROPY_MODEL_TRAINING.csv', index=False)
        torch.save(model.state_dict(), out/'entropy_model.pt')
        receipt = out/'FROZEN_MODEL.json'
        common.write_json(receipt, dict(completed=True, checkpoint='entropy_model.pt', checkpoint_sha256=common.sha(out/'entropy_model.pt'),
                                        selected_epoch=int(best_epoch), selection='lowest bits/symbol on every 3rd VALIDATION cube',
                                        validation_selection_cubes=len(selection), test_used_for_selection=False))
        log(f'Entropy model frozen at epoch {best_epoch}.')

        # 4. VALIDATION once with the frozen system, 5. TEST once.
        perception = Perception(runtime)
        records, label_log = [], []
        bitstreams = None
        try:
            for role, methods in (('validation', VALIDATION_METHODS), ('test', TEST_METHODS)):
                if role == 'test':
                    runtime.authorize_test(receipt)
                    bitstreams = Bitstreams(out, BITSTREAM_WORKERS)
                    log(f'TEST unlocked; {BITSTREAM_WORKERS} bitstream worker processes.')
                roster = runtime.roster(role)
                for i, (_, row) in enumerate(roster.iterrows()):
                    records.extend(evaluate_cube(runtime, row, methods, alphas, model, perception, label_log, bitstreams))
                    if (i+1) % log_every == 0 or i+1 == len(roster):
                        log(f'{role.upper()} {i+1}/{len(roster)}')
            frame_bits = bitstreams.finish()
        except BaseException:
            if bitstreams is not None:
                bitstreams.close()
            raise
        if common.sha(out/'entropy_model.pt') != common.read_json(receipt)['checkpoint_sha256']:
            raise ValueError('Entropy model changed after freezing.')
        log('Evaluation complete; writing reports.')

        # 6. Reports.
        frame = pd.DataFrame(records)
        frame['learned_chosen'] = frame.learned_chosen.astype(float)
        frame.to_csv(out/'RESULTS_BY_CUBE.csv', index=False)
        pd.DataFrame(label_log).to_csv(out/'LABELS.csv', index=False)
        frame_bits.to_csv(out/'BITSTREAM.csv', index=False)
        bits_summary = bitstream_summary(frame_bits)
        write_report(out/'BITSTREAM_SUMMARY.json', bits_summary)
        summary = summarize(frame)
        summary.to_csv(out/'SUMMARY.csv', index=False)
        cis = recording_bootstrap(frame)
        write_report(out/'RECORDING_CI.json', cis)
        perception_table, agreement = perception.report(out)
        for role in ('validation', 'test'):
            write_report(out/f'{role.upper()}_GATES.json', gates(summary, perception_table, calibration, role, cis))
        comparison, verdict = rate_comparison(summary, perception_table, agreement)
        comparison.to_csv(out/'RATE_COMPARISON_TEST.csv', index=False)
        write_report(out/'RATE_COMPARISON_TEST.json', verdict)
        make_figures(out, comparison, history)
        runtime.dataset.assert_unchanged()
        test_gates = common.read_json(out/'TEST_GATES.json')
        write_report(out/'SMOKE_COMPLETED.json', dict(
            completed=True, passed=True, context=expected, seed=SEED, test_evaluated=True, test_used_for_selection=False,
            selected_epoch=int(best_epoch), entropy_model_trained=True, alphas_db=calibration['alphas_db'],
            os_alphas_db=calibration['os_cfar']['alphas_db'], cubes={r: len(runtime.roster(r)) for r in partition.ROLES},
            pre_registered_points_passed=test_gates['pre_registered_points_passed'],
            bitstreams_all_exact=bits_summary['all_decoded_from_file_exactly'], runtime_seconds=time.time()-start))
    print(REVISION.split('-')[0], 'publication run COMPLETE; start with TEST_GATES.json and RATE_COMPARISON_TEST.csv:', out, flush=True)


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
    table, scan = inventory(dataset, args.split_audit)
    with pd.option_context('display.width', 200, 'display.max_rows', 200):
        print(table.sort_values(['role', 'recording']).to_string(index=False))
    print(json.dumps(scan, indent=1), flush=True)
    if args.stage == 'count':
        if not Path(str(out)+'.COUNT.csv').exists():
            table.to_csv(Path(str(out)+'.COUNT.csv'), index=False)
            write_report(Path(str(out)+'.COUNT.json'), scan)
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
    print(json.dumps({k: expected[k] for k in ('revision', 'gpu', 'bitstream_workers')}, indent=1), flush=True)
    if args.stage == 'check':
        run_check(runtime, out, expected)
    else:
        run(runtime, out, expected)


if __name__ == '__main__':
    main()
