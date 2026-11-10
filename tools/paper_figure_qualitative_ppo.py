"""RADIal-style qualitative figure on one real VALIDATION cube: original, the tuned PPO policy, AdaRadar at the matching rate.

The cube is AdaRadar's own example frame, RECORD@2020-11-21_13.44.44 frame 26, which is on this machine. It runs on CPU with the
frozen policy and FFTRadNet's official weights. Reads the local RADIal subset.

Each method is one column: its RD map above its bird's-eye view, both with range to the right so that an object lines up across
them. The RD panels keep all of Doppler, centred on bin 0 because it is cyclic. A narrower window around the labels would hide
more of PPO's marks than of AdaRadar's: at +-64 bins and 5-40 m it shows 5 of 38 and 54 of 188 lost detections, so it is not used.
The map runs to 103 m and the counts in the column titles cover all of it.

python -B tools/paper_figure_qualitative_ppo.py --run <verified folder> [--budget 1.4541] [--out outputs/paper_figures_ppo_tuned]
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
from pathlib import Path
import sys
import zlib

import matplotlib
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
from PIL import Image
import torch

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
spec = importlib.util.spec_from_file_location('qualitative', SRC/'tools/paper_figure_qualitative.py')
Q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(Q)
spec = importlib.util.spec_from_file_location('tuned', SRC/'tools/run_radial_ppo_tuned.py')
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
learning, cap, data, partition, codec, F, adaradar = T.learning, T.cap, T.data, T.V17.partition, T.codec, Q.F, Q.adaradar
from radar_sim.radial_adaptive_gpu_physics import static_features

PACKAGE = SRC/'tools/radial_ppo_tuned_gpu'
RATES = {.4585: 50., .8208: 25., 1.4541: 12.57, 2.1064: 8., 3.6327: 4., 4.7299: 2.}
MARGIN = 10**(1/10)
RANGE, DOPPLER, LATERAL = (0., 50.), (-128, 128), (-10., 10.)  # the extent of FFTRadNet's map, all of Doppler
BIN_M = .2  # metres per range bin of the RD map
cpu = torch.device('cpu')


def compute(args):
    frozen = json.loads((args.run/'FROZEN_POLICY.json').read_text())
    alpha = json.loads((args.run/'H0_CALIBRATION.json').read_text())['alphas']['a1']
    fitted = torch.load(args.run/'feature_normalizer.pt', map_location=cpu, weights_only=True)
    actor = learning.DeltaPolicy(len(fitted['names']), cpu)
    actor.load_state_dict(torch.load(args.run/frozen['checkpoint'], map_location=cpu, weights_only=True)['actor'])
    actor.eval()
    dataset = data.prepare_dataset(args.dataset_root, args.reference_root)
    frames = partition.assign_frozen_partitions(dataset, PACKAGE/'GLOBAL_SPLIT_V17.csv', bounded=True)
    row = frames[1].set_index('record_name').loc[Q.NAME].copy()
    row['record_name'] = Q.NAME
    labels = dataset.labels[dataset.labels.annotation_evaluable & dataset.labels.Difficult.eq(0)]
    frame_labels = labels[labels.record_name.eq(Q.NAME)]
    physics = cap.CapPhysics(dataset.reference_root, device=cpu, engineering=True)
    receive = lambda c: codec.rd_receive(c, physics.range_window, physics.doppler_window, Q.cyclic_cfar)

    def receive_spectrum(spectrum):
        rd = 20*torch.log10(spectrum.abs().sum(2).clamp_min(torch.finfo(torch.float64).tiny))
        return dict(RD=rd, spectrum=spectrum, detector=Q.cyclic_cfar(10.**(rd/10.)))

    cube = Q.load_cube_gpu(row, cpu, engineering=True)
    numel = cube.numel()
    original = receive(cube)
    net, config = F.load(PACKAGE/'FFTRadNet.py', PACKAGE/'config_FFTRadNet_192_56.json', PACKAGE/'fftradnet_state_dict.pt', cpu)
    stats = config['dataset']['statistics']
    source = physics.prepare(cube, seed=T.PPO_SEED+zlib.crc32(Q.NAME.encode()) % 2**31)
    features = static_features(physics, source, physics.maps(cube))
    order = torch.argsort(features[:, T.WEAK], stable=True)
    allocation = learning.rollout(actor, fitted, features, frozen['prices'][f'{args.budget:g}'], deterministic=True)
    decoded, meta, _, _ = T.sweep.project(physics, source, order, allocation['rho'], allocation['q'], args.budget)
    spectrum_ada, rate_ada = adaradar.compress(original['spectrum'], RATES[args.budget], 4, stats['input_mean'], stats['input_std'])
    panels = [dict(title='Original', bpcs=32., reported=None, received=original), dict(title='PPO', bpcs=float(meta['bpcs']), reported=None, received=receive(decoded)),
              dict(title='AdaRadar', bpcs=rate_ada['total']/numel, reported=None, received=receive_spectrum(spectrum_ada))]

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

    camera = dataset.events[row.scene_name]['camera']
    ev = camera['events']
    k = int((ev.timestamp-row.timestamp_us).abs().idxmin())
    with open(camera['filename'], 'rb') as stream_file:
        stream_file.seek(int(ev.offset[k]))
        image = Image.open(io.BytesIO(stream_file.read(int(ev.datasize[k])))).convert('RGB')
    gt = F.boxes(frame_labels.range_m.to_numpy(float), frame_labels.azimuth_deg.to_numpy(float))
    bins = physics.label_bins(frame_labels.reset_index(drop=True), dict(ranges=physics.ranges, vel=physics.vel), 'RD').numpy()
    pred0, _, seg0 = perceive(original)
    Q.plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 8, 'axes.titlesize': 8, 'axes.linewidth': .5, 'mathtext.fontset': 'stix', 'pdf.fonttype': 42})
    for r in panels:
        r['robust'], r['lost'], r['new'] = cfar_changes(r['received'])
        prediction, r['boxes'], r['seg'] = perceive(r['received'])
        agree = F.agreement(pred0, prediction)
        r.update(kept=agree['objects_kept'], total=agree['objects_original'], fs=F.freespace_iou(seg0[None], r['seg'][None]))
        print(r['title'], 'bpcs %.3f' % r['bpcs'], 'robust %d lost %d new %d' % (r['robust'], len(r['lost']), len(r['new'])), 'objects kept %d/%d' % (r['kept'], r['total']), 'free space %.2f' % r['fs'])
        r['RD'] = r.pop('received')['RD'].numpy()
    return panels, np.asarray(image), frame_labels, gt, bins


def centred(points):
    """Native [range bin, Doppler bin] points as (range m, Doppler bin about 0), inside the windows."""
    if not len(points):
        return np.zeros(0), np.zeros(0)
    rng, doppler = points[:, 0]*BIN_M, (points[:, 1]+128) % 256-128
    inside = (rng >= RANGE[0]) & (rng <= RANGE[1]) & (doppler >= DOPPLER[0]) & (doppler < DOPPLER[1])
    return rng[inside], doppler[inside]


def draw(results, image, frame_labels, gt, bins, stem):
    plt = Q.plt
    fig = plt.figure(figsize=(7.16, 2.05), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=.015, h_pad=.01, wspace=.015, hspace=.005)
    grid = fig.add_gridspec(2, 4, width_ratios=[.92, 1, 1, 1], height_ratios=[1.06, 1])
    window = (int(RANGE[0]/BIN_M), int(RANGE[1]/BIN_M)), (DOPPLER[0]+128, DOPPLER[1]+128)
    maps = [np.fft.fftshift(r['RD'], axes=1)[window[0][0]:window[0][1], window[1][0]:window[1][1]].T for r in results]
    whole = results[0]['RD'][:250]
    vmin, vmax = np.percentile(whole, 5), np.percentile(whole, 99.9)
    rr = np.arange(257)*.4
    aa = np.radians((np.arange(225)-112)*.8)
    R, A = np.meshgrid(rr, aa, indexing='ij')
    forward, right = R*np.cos(A), -R*np.sin(A)

    ax = fig.add_subplot(grid[0, 0])
    x0, x1 = max(0, int(frame_labels.x1_pix.min())-60), min(image.shape[1], int(frame_labels.x2_pix.max())+60)
    y0, y1 = max(0, int(frame_labels.y1_pix.min())-60), min(image.shape[0], int(frame_labels.y2_pix.max())+60)
    ax.imshow(image[y0:y1, x0:x1], extent=(x0, x1, y1, y0))
    for _, lab in frame_labels.iterrows():
        ax.add_patch(Rectangle((lab.x1_pix, lab.y1_pix), lab.x2_pix-lab.x1_pix, lab.y2_pix-lab.y1_pix, fill=False, ec='#00C000', lw=.9))
    ax.set_axis_off()
    ax.set_title('Camera and labels\n(reference only)', pad=2)
    ax = fig.add_subplot(grid[1, 0])
    ax.set_axis_off()
    key = [Line2D([], [], ls='none', marker='o', mfc='none', mec='0.35', ms=5, label='label'),
           Line2D([], [], ls='none', marker='x', color='red', ms=4, label='lost robust detection'),
           Line2D([], [], ls='none', marker='+', color='darkorange', ms=5, label='added false peak'),
           Patch(fc='none', ec='black', ls='--', label='label box'),
           Patch(fc='red', ec='red', label='FFTRadNet box'),
           Patch(fc='#7CCB7C', ec='none', label='predicted free space')]
    ax.legend(handles=key, loc='center', ncol=1, fontsize=7, frameon=True, framealpha=1, edgecolor='.7', borderpad=.35, labelspacing=.22, handlelength=1.3, handletextpad=.5)

    for i, r in enumerate(results):
        top = fig.add_subplot(grid[0, i+1])
        top.imshow(maps[i], origin='lower', aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax, extent=(*RANGE, *DOPPLER), interpolation='nearest')
        rng, doppler = centred(bins)
        top.scatter(rng, doppler, s=30, facecolors='none', edgecolors='white', linewidths=.9, zorder=3)
        for points, marker, color, size in ((r['lost'], 'x', 'red', 11), (r['new'], '+', 'orange', 15)):
            rng, doppler = centred(points)
            top.scatter(rng, doppler, s=size, marker=marker, c=color, linewidths=.8, zorder=4)
        kept = r['robust']-len(r['lost'])
        name = f"{r['title']}, 32 bpcs" if i == 0 else f"{r['title']}, {r['bpcs']:.2f} bpcs ({32/r['bpcs']:.1f}\u00d7)"
        top.set_title(f"{name}\n{kept}/{r['robust']} robust kept, {len(r['new'])} false peaks", pad=2)
        top.set_yticks((-100, 0, 100))
        top.tick_params(direction='in', length=2, pad=1.5, labelbottom=False)
        bottom = fig.add_subplot(grid[1, i+1], sharex=top)
        bottom.pcolormesh(forward, right, (r['seg'] >= .5).astype(float), cmap=matplotlib.colors.ListedColormap(['white', '#7CCB7C']), shading='flat', rasterized=True)
        for b in gt:
            bottom.add_patch(Rectangle((b[1], -b[2]), b[3]-b[1], b[2]-b[0], fill=False, ec='black', lw=.9, ls='--', zorder=4))
        for b in r['boxes']:
            bottom.add_patch(Rectangle((b[1], -b[2]), b[3]-b[1], b[2]-b[0], fill=True, fc='red', ec='red', lw=.5, alpha=.85, zorder=3))
        bottom.set_xlim(*RANGE)
        bottom.set_ylim(LATERAL[1], LATERAL[0])  # the map turned a quarter clockwise: forward to the right, the driver's left up
        bottom.set_xlabel('Range (m)', labelpad=1)
        bottom.tick_params(direction='in', length=2, pad=1.5)
        note = f"{r['kept']}/{r['total']} objects" if i == 0 else f"{r['kept']}/{r['total']} objects, free-space IoU {r['fs']:.2f}"
        bottom.text(.985, .05, note, transform=bottom.transAxes, ha='right', va='bottom', fontsize=7, bbox=dict(fc='white', ec='none', alpha=.8, pad=.8), zorder=5)
        if i == 0:
            top.set_ylabel('Doppler bin', labelpad=1)
            bottom.set_ylabel('Lateral (m)', labelpad=1)
        else:
            top.set_yticklabels([])
            bottom.set_yticklabels([])
    fig.savefig(stem.with_suffix('.pdf'), dpi=300)
    fig.savefig(stem.with_suffix('.png'), dpi=300)
    plt.close(fig)
    print('saved', stem)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--budget', type=float, default=1.4541, choices=sorted(RATES))
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=SRC/'outputs/paper_figures_ppo_tuned')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    draw(*compute(args), args.out/f'fig_qualitative_radial_ppo_{str(args.budget).replace(".", "p")}')
    print('figure written to', args.out)


if __name__ == '__main__':
    main()
