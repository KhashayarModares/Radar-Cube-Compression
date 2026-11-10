"""FFTRadNet perception check for decoded RADIal cubes (Rebut et al., CVPR 2022).

The official, unmodified ``FFTRadNet.py`` and its config are executed as shipped
(an unused ``torchvision`` import is stubbed only when torchvision is absent),
with weights-only tensors extracted from the official checkpoint. The network
input is the official per-Rx range-Doppler spectrum (``rpl.py`` method 'RD':
mean removal, range/Doppler windows, unnormalized FFTs), stacked as real and
imaginary channels and normalized with the config statistics.

Metrics port the official ``GetFullMetrics``: confidence thresholds 0.1..0.9,
official box NMS (IoU 0.05), predictions and labels within 5-100 m, IoU 0.5,
easy labels only; mAP, mAR and F1 = mAP*mAR/((mAP+mAR)/2). The official boxes
are axis-aligned 1.8 m x 4 m rectangles, so their polygon IoU equals the
axis-aligned IoU computed here (no shapely dependency).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import types

import numpy as np
import torch

BOX_WIDTH, BOX_LENGTH = 1.8, 4.
THRESHOLDS = np.arange(.1, .96, .1)
RANGE_MIN, RANGE_MAX, IOU_THRESHOLD, NMS_IOU, DECODE_THRESHOLD = 5, 100, .5, .05, .05


def _torchvision_stub():
    try:
        import torchvision  # noqa: F401
        return
    except ImportError:
        pass
    root, transforms, inner = (types.ModuleType(n) for n in ('torchvision', 'torchvision.transforms',
                                                              'torchvision.transforms.transforms'))
    inner.Sequence = object
    root.transforms, transforms.transforms = transforms, inner
    sys.modules.update({'torchvision': root, 'torchvision.transforms': transforms,
                        'torchvision.transforms.transforms': inner})


def load(model_py, config_json, weights, device):
    config = json.loads(Path(config_json).read_text(encoding='utf-8'))
    _torchvision_stub()
    module = types.ModuleType('fftradnet_official')
    exec(compile(Path(model_py).read_text(encoding='utf-8'), str(model_py), 'exec'), module.__dict__)
    m = config['model']
    net = module.FFTRadNet(blocks=m['backbone_block'], mimo_layer=m['MIMO_output'], channels=m['channels'],
                           regression_layer=2, detection_head=m['DetectionHead'], segmentation_head=m['SegmentationHead'])
    net.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    return net.to(device).eval(), config


def network_input(spectrum, config):
    """[512, 256, 16] complex RD spectrum -> [32, 512, 256] normalized float32."""
    stats = config['dataset']['statistics']
    x = torch.cat((spectrum.real, spectrum.imag), -1).to(torch.float64)
    mean = torch.as_tensor(stats['input_mean'], dtype=torch.float64, device=x.device)
    std = torch.as_tensor(stats['input_std'], dtype=torch.float64, device=x.device)
    return ((x-mean)/std).to(torch.float32).permute(2, 0, 1)


@torch.no_grad()
def predict(net, inputs):
    out = net(inputs)
    return out['Detection'], torch.sigmoid(out['Segmentation'])


def decode(detection, config, threshold=DECODE_THRESHOLD):
    """Official ra_encoder.decode: [3, 128, 224] map -> rows [R, A, C]."""
    g, s = config['dataset']['geometry'], config['dataset']['statistics']
    cls = detection[0]
    r_bin, a_bin = torch.nonzero(cls >= threshold, as_tuple=True)
    width = g['ranges'][1]//4
    R = r_bin*4*g['resolution'][0]+detection[1, r_bin, a_bin]*s['reg_std'][0]+s['reg_mean'][0]
    A = (a_bin-width/2)*4*g['resolution'][1]+detection[2, r_bin, a_bin]*s['reg_std'][1]+s['reg_mean'][1]
    return torch.stack((R, A, cls[r_bin, a_bin]), 1).double().cpu().numpy()


def boxes(R, A):
    x, y = np.sin(np.radians(A))*R, np.cos(np.radians(A))*R
    return np.stack((x-BOX_WIDTH/2, y, x+BOX_WIDTH/2, y+BOX_LENGTH), 1)


def iou(box, others):
    w = np.clip(np.minimum(box[2], others[:, 2])-np.maximum(box[0], others[:, 0]), 0, None)
    h = np.clip(np.minimum(box[3], others[:, 3])-np.maximum(box[1], others[:, 1]), 0, None)
    inter = w*h
    area = lambda b: (b[..., 2]-b[..., 0])*(b[..., 3]-b[..., 1])
    return inter/(area(box)+area(others)-inter)


def official_objects(prediction, threshold):
    """process_predictions_FFT + range filter: confidence, official NMS, 5-100 m."""
    if not len(prediction):
        return np.zeros((0, 4)), np.zeros(0)
    b, c = boxes(prediction[:, 0], prediction[:, 1]), prediction[:, 2]
    keep = c > threshold
    b, c = b[keep], c[keep]
    order = np.argsort(c)[::-1]
    b, c = b[order], c[order]
    i = 0
    while i < len(b):
        ious = np.zeros(len(b))
        ious[i+1:] = iou(b[i], b[i+1:])
        mask = ious < NMS_IOU
        b, c = b[mask], c[mask]
        i += 1
    forward = b[:, 1]
    in_range = (forward >= RANGE_MIN) & (forward <= RANGE_MAX)
    return b[in_range], c[in_range]


def frame_counts(prediction, label):
    """[thresholds, 3] TP/FP/FN of one frame, exactly as GetFullMetrics."""
    out = np.zeros((len(THRESHOLDS), 3), dtype=np.int64)
    label = label[(label[:, 0] >= RANGE_MIN) & (label[:, 0] <= RANGE_MAX)] if len(label) else label
    gt = boxes(label[:, 0], label[:, 1]) if len(label) else np.zeros((0, 4))
    for t, threshold in enumerate(THRESHOLDS):
        pred, _ = official_objects(prediction, threshold)
        if len(gt) and len(pred):
            used = np.zeros(len(gt), dtype=bool)
            for box in pred:
                hit = iou(box, gt) >= IOU_THRESHOLD
                if hit.any():
                    out[t, 0] += 1
                    used |= hit
                else:
                    out[t, 1] += 1
            out[t, 2] += int((~used).sum())
        elif not len(gt):
            out[t, 1] += len(pred)
        else:
            out[t, 2] += len(gt)
    return out


def metrics_from_counts(total):
    """Official mAP/mAR/F1 from summed [thresholds, 3] counts."""
    tp, fp, fn = total[:, 0], total[:, 1], total[:, 2]
    precision = np.where(tp > 0, tp/np.maximum(tp+fp, 1), 0.)
    recall = np.where(tp > 0, tp/np.maximum(tp+fn, 1), 0.)
    p, r = float(precision.mean()), float(recall.mean())
    return dict(mAP=p, mAR=r, F1=(p*r)/((p+r)/2) if p+r else 0.)


def full_metrics(predictions, labels):
    total = sum(frame_counts(p, l) for p, l in zip(predictions, labels))
    return dict(metrics_from_counts(total), frames=len(predictions),
                labels=int(sum(len(l[(l[:, 0] >= RANGE_MIN) & (l[:, 0] <= RANGE_MAX)]) if len(l) else 0 for l in labels)))


def agreement(original, method, threshold=.5):
    """Label-free: original FFTRadNet objects kept / new after decoding (IoU >= 0.5)."""
    a, _ = official_objects(original, threshold)
    b, _ = official_objects(method, threshold)
    kept = sum(bool(len(b)) and bool((iou(x, b) >= IOU_THRESHOLD).any()) for x in a)
    new = sum(not (len(a) and (iou(y, a) >= IOU_THRESHOLD).any()) for y in b)
    return dict(objects_original=len(a), objects_method=len(b), objects_kept=int(kept), objects_new=int(new))


def freespace_iou(original, method, rows=124):
    """Label-free: free-space mask agreement within 0-50 m (official 124 rows)."""
    a, b = original[0, :rows] >= .5, method[0, :rows] >= .5
    union = (a | b).sum()
    return float((a & b).sum()/union) if union else 1.
