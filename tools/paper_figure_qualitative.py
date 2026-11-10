"""RADIal-style qualitative figure on one real cube: original vs the proposed codec vs AdaRadar (r = 12.57, 4 bit).

The cube is AdaRadar's own example frame (its public data/fft_000018.npy = RADIal sample 18 =
RECORD@2020-11-21_13.44.44 frame 26, a VALIDATION recording, available locally). Two versions:
  operating     proposed = C at beta* = 1.5 (V17 entropy model), the paper's operating point;
  rate-matched  proposed = the fixed-rate codec at AdaRadar's rate, 1.45 bpcs (V20: gamma* = 0, V20 entropy model).
CPU, frozen models; the camera image is for reference only. Reads the local RADIal subset.

python -B tools/paper_figure_qualitative.py [--version operating rate-matched] [--out outputs/paper_figures_v20]
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
from PIL import Image
import torch

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
spec = importlib.util.spec_from_file_location('v20_runner', SRC/'tools/run_radial_v20_rate_matched.py')
V20 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V20)
codec, entropy, data, partition = V20.codec, V20.entropy, V20.data, V20.V17.partition
from radar_sim import radial_fftradnet_eval as F
from radar_sim import radial_v18_adaradar as adaradar
from radar_sim.radial_adaptive_gpu_physics import RadialGPUPhysics, cyclic_cfar, load_cube_gpu

PKG = SRC/'tools/v20_rate_matched_gpu'
RUN = SRC/'results/runs/v20_rm_20261030T131713629Z_verified_20261030T154146779Z'
NAME = 'RECORD@2020-11-21_13.44.44_frame_000026'
SEED = 20261030
MARGIN = 10**(1/10)
cpu = torch.device('cpu')


def load_model(path, sha=None):
    if sha is not None and V20.common.sha(path) != sha:
        raise ValueError(f'{path.name} differs from the frozen checkpoint.')
    model = entropy.ContextModel()
    model.load_state_dict(torch.load(path, map_location=cpu, weights_only=True))
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--version', nargs='+', choices=('operating', 'rate-matched', 'no-feedback'), default=['operating', 'rate-matched'])
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=SRC/'outputs/paper_figures_v20')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    alpha = json.loads((RUN/'H0_CALIBRATION.json').read_text())['alphas']['a1']  # alpha_1 of the CA-CFAR receiver
    frozen = json.loads((RUN/'FROZEN_MODEL.json').read_text())
    dataset = data.prepare_dataset(args.dataset_root, args.reference_root)
    frames = partition.assign_frozen_partitions(dataset, PKG/'GLOBAL_SPLIT_V17.csv', bounded=True)
    row = frames[1].set_index('record_name').loc[NAME].copy()
    row['record_name'] = NAME
    labels = dataset.labels[dataset.labels.annotation_evaluable & dataset.labels.Difficult.eq(0)]
    frame_labels = labels[labels.record_name.eq(NAME)]

    p = RadialGPUPhysics(dataset.reference_root, device=cpu, engineering=True)
    receive = lambda c: codec.rd_receive(c, p.range_window, p.doppler_window, cyclic_cfar)

    def receive_spectrum(spectrum):
        rd = 20*torch.log10(spectrum.abs().sum(2).clamp_min(torch.finfo(torch.float64).tiny))
        return dict(RD=rd, spectrum=spectrum, detector=cyclic_cfar(10.**(rd/10.)))

    cube = load_cube_gpu(row, cpu, engineering=True)
    numel = cube.numel()
    original = receive(cube)
    net, config = F.load(PKG/'FFTRadNet.py', PKG/'config_FFTRadNet_192_56.json', PKG/'fftradnet_state_dict.pt', cpu)
    stats = config['dataset']['statistics']
    spectrum_ada, rate_ada = adaradar.compress(original['spectrum'], 12.57, 4, stats['input_mean'], stats['input_std'])
    ada = dict(title='AdaRadar (r = 12.57, 4 bit)', bpcs=rate_ada['total']/numel, reported=rate_ada['total_nonzero_only']/numel,
               received=receive_spectrum(spectrum_ada))

    def proposed(version):
        if version == 'operating':
            model = load_model(PKG/'v17_entropy_model.pt', V20.FROZEN_MODEL['sha256'])
            stream, decoded, _ = codec.encode_closed_loop(cube, receive, original['detector'], alpha, gamma_db=300., beta=1.5, seed=SEED)
            bits = codec.ecdq_bits(stream)
            learned = entropy.learned_payload_bits(model, entropy.compact(stream))
            title = 'Proposed, $\\beta^*$ = 1.5'
        elif version == 'no-feedback':
            model = load_model(PKG/'v17_entropy_model.pt', V20.FROZEN_MODEL['sha256'])
            stream = codec.encode_ecdq(cube, beta=1.5, seed=SEED)
            decoded = codec.decode_ecdq(stream, cube.dtype)
            bits = codec.ecdq_bits(stream)
            learned = entropy.learned_payload_bits(model, entropy.compact(stream))
            title = 'Proposed, $\\beta^*$ = 1.5'
        else:
            model = load_model(SRC/'weights'/'entropy_model_v20.pt', frozen['checkpoint_sha256'])
            runtime = SimpleNamespace(receive=receive, seed=lambda _: SEED)
            stream, decoded, bits, learned, control = V20.FixedRate(model).encode(
                runtime, row, cube, original, alpha, V20.R_STAR, frozen['gamma_selected'], 'figure')
            print('rate control:', json.dumps(control))
            title = 'Proposed, fixed rate'
        bpcs = V20.V14.rate_fields(bits, learned, numel)['bpcs']
        return dict(title=title, bpcs=bpcs, reported=None, received=receive(decoded))

    def cfar_changes(method):
        s0, s1 = codec.peak_scores(original['detector']), codec.peak_scores(method['detector'])
        near0, near1 = codec.neighbourhood_max(s0, 2, 2), codec.neighbourhood_max(s1, 2, 2)
        stable0 = codec.stable_peaks(original['detector'], alpha, MARGIN)
        lost = stable0 & (near1 <= alpha)
        new = codec.stable_peaks(method['detector'], alpha, MARGIN) & (near0 <= alpha/MARGIN)
        return int(stable0.sum()), torch.nonzero(lost).numpy(), torch.nonzero(new).numpy()

    def perceive(received):
        detection, segmentation = F.predict(net, F.network_input(received['spectrum'], config)[None])
        prediction = F.decode(detection[0], config)
        boxes, _ = F.official_objects(prediction, .5)
        return prediction, boxes, segmentation[0, 0].cpu().numpy()

    # Camera frame nearest in time to the radar frame; label boxes drawn in pixel coordinates.
    camera = dataset.events[row.scene_name]['camera']
    ev = camera['events']
    k = int((ev.timestamp-row.timestamp_us).abs().idxmin())
    with open(camera['filename'], 'rb') as stream_file:
        stream_file.seek(int(ev.offset[k]))
        image = Image.open(io.BytesIO(stream_file.read(int(ev.datasize[k])))).convert('RGB')
    gt = F.boxes(frame_labels.range_m.to_numpy(float), frame_labels.azimuth_deg.to_numpy(float))
    bins = p.label_bins(frame_labels.reset_index(drop=True), dict(ranges=p.ranges, vel=p.vel), 'RD').numpy()
    pred0, boxes0, seg0 = perceive(original)

    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 8,
                         'axes.titlesize': 8, 'axes.linewidth': .5, 'mathtext.fontset': 'stix', 'pdf.fonttype': 42})
    for version in args.version:
        panels = [dict(title='Original', bpcs=32., reported=None, received=original), proposed(version), ada]
        for r in panels:
            r['robust'], r['lost'], r['new'] = cfar_changes(r['received'])
            prediction, r['boxes'], r['seg'] = perceive(r['received'])
            agree = F.agreement(pred0, prediction)
            r.update(kept=agree['objects_kept'], total=agree['objects_original'], fs=F.freespace_iou(seg0[None], r['seg'][None]))
            print(version, r['title'], 'bpcs %.3f' % r['bpcs'], 'robust %d lost %d new %d' % (r['robust'], len(r['lost']), len(r['new'])),
                  'objects kept %d/%d' % (r['kept'], r['total']), 'free space %.2f' % r['fs'])
        draw(panels, image, frame_labels, gt, bins, args.out/f'fig_qualitative_radial_{version.replace("-", "_")}')


def draw(results, image, frame_labels, gt, bins, stem):
    fig = plt.figure(figsize=(7.16, 4.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 4, width_ratios=[1.35, 1, 1, 1], height_ratios=[1, 1.05])
    RMAX, XLIM, YLIM = 50., (-15, 15), (0, 50)
    rows_rd = int(RMAX/0.2)
    rd_all = [r['received']['RD'][:rows_rd].numpy() for r in results]
    vmin, vmax = np.percentile(rd_all[0], 5), np.percentile(rd_all[0], 99.9)

    def bev(ax, title):
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.set_aspect('equal')
        ax.set_xlabel('Lateral, right positive (m)')
        ax.set_title(title, pad=3)
        ax.tick_params(direction='in', length=2)
        # RADIal radar x is positive to the left (FFTRadNet projects -x into the camera); plot -x so left/right match the camera.
        for b in gt:
            ax.add_patch(Rectangle((-b[2], b[1]), b[2]-b[0], b[3]-b[1], fill=False, ec='black', lw=.9, ls='--'))

    ax = fig.add_subplot(grid[0, 0])
    ax.imshow(image)
    for _, lab in frame_labels.iterrows():
        ax.add_patch(Rectangle((lab.x1_pix, lab.y1_pix), lab.x2_pix-lab.x1_pix, lab.y2_pix-lab.y1_pix, fill=False, ec='#00C000', lw=1))
    ax.set_axis_off()
    ax.set_title('Camera (reference only) and labels', pad=3)
    key = [Line2D([], [], ls='none', marker='o', mfc='none', mec='0.35', ms=5, label='label (RD map)'),
           Line2D([], [], ls='none', marker='x', color='red', ms=4, label='robust detection lost'),
           Line2D([], [], ls='none', marker='+', color='orange', ms=5, label='false alarm added'),
           Patch(fc='none', ec='black', ls='--', label='label box'),
           Patch(fc='red', ec='red', label='FFTRadNet box'),
           Patch(fc='#7CCB7C', ec='none', label='predicted free space')]
    ax.legend(handles=key, loc='upper center', bbox_to_anchor=(.5, -.04), ncol=2, fontsize=7, frameon=True,
              framealpha=.95, edgecolor='0.7', borderpad=.4, labelspacing=.35, handlelength=1.4, columnspacing=1.2)
    ax = fig.add_subplot(grid[1, 0])
    bev(ax, 'Ground-truth objects')
    ax.set_ylabel('Range (m)')

    rr = np.arange(257)*0.4
    aa = np.radians((np.arange(225)-112)*0.8)
    R, A = np.meshgrid(rr, aa, indexing='ij')
    X, Y = R*np.sin(A), R*np.cos(A)
    for i, r in enumerate(results):
        ax = fig.add_subplot(grid[0, i+1])
        ax.imshow(rd_all[i], origin='lower', aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax, extent=(0, 256, 0, RMAX))
        ax.scatter(bins[:, 1], bins[:, 0]*0.2, s=26, facecolors='none', edgecolors='white', linewidths=.8)
        for points, marker, color, size in ((r['lost'], 'x', 'red', 9), (r['new'], '+', 'orange', 9)):
            points = points[points[:, 0] < rows_rd] if len(points) else points
            if len(points):
                ax.scatter(points[:, 1], points[:, 0]*0.2, s=size, marker=marker, c=color, linewidths=.8)
        rate = (f"{r['bpcs']:.2f} bpcs ({32/r['bpcs']:.1f}\u00d7)" if r['reported'] is None else
                f"{r['bpcs']:.2f} bpcs ({32/r['bpcs']:.0f}\u00d7); reported {r['reported']:.2f}")
        ax.set_title(f"{r['title']}\n{rate}" if i else f"{r['title']}\n32 bpcs (1\u00d7)", pad=3)
        ax.set_xlabel('Doppler bin')
        if i == 0:
            ax.set_ylabel('Range (m)')
        else:
            ax.set_yticklabels([])
        ax.tick_params(direction='in', length=2)
        ax.text(.03, .03, f"CFAR robust kept {r['robust']-len(r['lost'])}/{r['robust']}\nfalse alarms {len(r['new'])}",
                transform=ax.transAxes, fontsize=7, color='white', va='bottom', bbox=dict(fc='black', ec='none', alpha=.6, pad=1.5))
        ax = fig.add_subplot(grid[1, i+1])
        ax.pcolormesh(-X, Y, (r['seg'] >= .5).astype(float), cmap=matplotlib.colors.ListedColormap(['white', '#7CCB7C']),
                      shading='flat', rasterized=True)
        for b in r['boxes']:
            ax.add_patch(Rectangle((-b[2], b[1]), b[2]-b[0], b[3]-b[1], fill=True, fc='red', ec='red', lw=.6, alpha=.9))
        bev(ax, f"FFTRadNet: {r['kept']}/{r['total']} objects kept")
        if i:
            ax.set_yticklabels([])
            ax.text(.97, .03, f"free-space IoU {r['fs']:.2f}", transform=ax.transAxes, ha='right', va='bottom', fontsize=7)
        else:
            ax.set_ylabel('Range (m)')
    fig.savefig(stem.with_suffix('.pdf'), dpi=300)
    fig.savefig(stem.with_suffix('.png'), dpi=300)
    plt.close(fig)
    print('saved', stem)


if __name__ == '__main__':
    main()
