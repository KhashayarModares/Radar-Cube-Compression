"""V20: fine-tuning the official FFTRadNet on decoded inputs.

AdaRadar (Train_w_compress.py) trains FFTRadNet on its compressed tensors. V20
fine-tunes the official checkpoint instead of training from scratch, with the
official recipe (FFTRadNet/1-Train.py) everywhere it applies:

  * detection target: ``ra_encoder.encode`` (ported below) of every evaluable
    label of the frame, easy and difficult (the official training uses
    difficult=True);
  * free-space target: RADIal's free-space masks belong to the ready-to-use
    dataset, not to the raw recordings this study codes. The target is the
    official network's own mask (sigmoid >= 0.5) on the ORIGINAL cube, so the
    decoded input is taught to give the free space the uncompressed input gives;
  * loss: official pixor_loss (focal loss, gamma 2, summed; smooth L1 on the
    positive cells, divided by their count) and BCE-with-logits free space
    times the batch size; weights [1, 100, 100];
  * optimizer: Adam, as official, at lr 1e-5 (a tenth of the official
    from-scratch rate). BatchNorm statistics stay at the official values and
    two micro-batches of 2 are accumulated into the official batch of 4 (an
    8 GB GPU holds ~2 samples of activations).

The epoch is chosen by the official F1 on VALIDATION frames coded with the same
codec; epoch 0 is the official checkpoint, so a fine-tune that does not help
is never selected.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from . import radial_fftradnet_eval as fftradnet

MICRO_BATCH, ACCUMULATE = 2, 2
LOSS_WEIGHTS = (1., 100., 100.)
FOCAL_GAMMA = 2.


def encode_targets(labels, config):
    """Official ra_encoder.encode (geometry size 3) of [N, 2] (range m, azimuth deg) -> [3, 128, 224] float32.

    Ported line by line, including its edge handling: a box whose 3x3 footprint would cross the last azimuth
    column gets no target, as in the official encoder."""
    g, s = config['dataset']['geometry'], config['dataset']['statistics']
    out_dim = (3, g['ranges'][0]//4, g['ranges'][1]//4)
    target = np.zeros(out_dim, dtype=np.float64)
    size = g['size']
    half = (size-1)//2
    r_lin = np.linspace(g['resolution'][0]*half, -g['resolution'][0]*half, size)*4
    a_lin = np.linspace(g['resolution'][1]*half, -g['resolution'][1]*half, size)*4
    px_a, px_r = np.meshgrid(a_lin, r_lin)
    for rng, azimuth in labels:
        if rng == -1:
            continue
        range_bin = int(np.clip(rng/g['resolution'][0]/4, 0, out_dim[1]))
        range_mod = rng-range_bin*g['resolution'][0]*4
        angle_bin = int(np.clip(np.floor(azimuth/g['resolution'][1]/4+out_dim[2]/2), 0, out_dim[2]))
        angle_mod = azimuth-(angle_bin-out_dim[2]/2)*g['resolution'][1]*4
        rows = slice(range_bin-half, range_bin+half+1)
        if half <= angle_bin < out_dim[2]-half:
            cols, pr, pa = slice(angle_bin-half, angle_bin+half+1), px_r, px_a
        elif angle_bin < half:
            cols, pr, pa = slice(0, angle_bin+half+1), px_r[:, half-angle_bin:], px_a[:, half-angle_bin:]
        elif angle_bin >= out_dim[2]:
            end = half+(out_dim[2]-angle_bin)
            cols, pr, pa = slice(angle_bin-half, None), px_r[:, :end], px_a[:, :end]
        else:
            continue
        target[0, rows, cols] = 1
        target[1, rows, cols] = ((pr+range_mod)-s['reg_mean'][0])/s['reg_std'][0]
        target[2, rows, cols] = ((pa+angle_mod)-s['reg_mean'][1])/s['reg_std'][1]
    return target.astype(np.float32)


def pixor_loss(prediction, target):
    """Official pixor_loss: focal classification loss (sum) and smooth-L1 regression on positive cells."""
    p, t = prediction[:, 0].flatten(), target[:, 0].flatten()
    pt = torch.where(t == 1., p, 1-p)
    classification = (-(1-pt)**FOCAL_GAMMA*torch.log(pt+1e-6)).sum()
    mask = target[:, :1]
    regression = F.smooth_l1_loss(prediction[:, 1:]*mask, target[:, 1:], reduction='sum')
    positives = mask.sum()
    return classification, regression/positives if positives > 0 else regression


def loss(net, inputs, detection_target, freespace_target):
    out = net(inputs)
    classification, regression = pixor_loss(out['Detection'], detection_target)
    freespace = F.binary_cross_entropy_with_logits(out['Segmentation'].flatten().double(),
                                                   freespace_target.flatten().double())*len(inputs)
    w = LOSS_WEIGHTS
    return w[0]*classification+w[1]*regression+w[2]*freespace, dict(classification=float(classification.detach()),
                                                                        regression=float(regression.detach()),
                                                                        freespace=float(freespace.detach()))


def set_trainable(net):
    """Weights train; BatchNorm keeps the official running statistics (eval mode), so micro-batches add up exactly."""
    net.eval()
    for p in net.parameters():
        p.requires_grad_(True)
    return net


def finetune(net, samples, evaluate, *, epochs, lr, seed, log):
    """``samples(order)`` yields (inputs, detection target, free-space target) micro-batches in ``order``;
    ``evaluate(net)`` returns VALIDATION metrics with 'F1'. Keeps the best epoch (0 = official weights)."""
    count = samples.count
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    history, best, best_state = [], -math.inf, None
    for epoch in range(epochs+1):
        terms = dict(loss=0., classification=0., regression=0., freespace=0.)
        if epoch:
            set_trainable(net)
            order = torch.randperm(count, generator=generator).tolist()
            optimizer.zero_grad(set_to_none=True)
            for step, (inputs, detection, freespace) in enumerate(samples(order), 1):
                value, parts = loss(net, inputs, detection, freespace)
                value.backward()
                terms['loss'] += float(value)
                for k, v in parts.items():
                    terms[k] += v
                if step % ACCUMULATE == 0 or step*MICRO_BATCH >= count:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            terms = {k: v/count for k, v in terms.items()}
        net.eval()
        metrics = evaluate(net)
        row = dict(epoch=epoch, **{f'train_{k}_per_sample': v for k, v in terms.items()}, **{f'val_{k}': v for k, v in metrics.items()})
        history.append(row)
        log(row)
        if metrics['F1'] > best:
            best, best_state = metrics['F1'], {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return history, max(history, key=lambda r: (r['val_F1'], -r['epoch']))['epoch']


@torch.no_grad()
def validation_f1(net, frames, config):
    """Official metrics of ``net`` on (network input, easy labels) pairs."""
    total = np.zeros((len(fftradnet.THRESHOLDS), 3), dtype=np.int64)
    for inputs, labels in frames():
        detection, _ = fftradnet.predict(net, inputs[None].float())
        total += fftradnet.frame_counts(fftradnet.decode(detection[0], config), labels)
    return fftradnet.metrics_from_counts(total)
