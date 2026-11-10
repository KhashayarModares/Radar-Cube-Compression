"""V14 learned conditional entropy model for closed-loop ECDQ symbols.

The quantizer and the closed loop fix WHAT is transmitted (and therefore every
detection and label outcome). This model only changes HOW MANY bits the same
integers cost: it predicts the magnitude class c = bit_length(|q|) of each coded
real component from a causal context, and the code length is the cross-entropy
-log2 p(c) (Shannon), achieved by arithmetic coding within ~2 bits (Witten,
Neal & Cleary 1987). Mantissa and sign bits stay raw (c bits per symbol).

Context (decodable plane by plane, channel-wise autoregressive as in Minnen &
Singh 2020; learned context model as in Minnen, Balle & Toderici 2018): the
classes of the previous one/two Rx planes around the same range-Doppler cell,
the real-part class of the same cell for the imaginary part, and side
information the decoder already has (support, refinement step, Rx index).
Trained by minimizing the rate term of the rate-distortion Lagrangian (Balle,
Laparra & Simoncelli 2017); the distortion term is fixed by the quantizer.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import radial_v13_detection_codec as codec

FEATURES = 12
HIDDEN = 64
EPSILON = 1e-4          # uniform mixture: bounded worst-case cost log2(32/eps)
FREQUENCY_BITS = 16     # integer model precision for the arithmetic coder


class ContextModel(nn.Module):
    """Two hidden layers of 64 units; ~7k parameters."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(FEATURES, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                                 nn.Linear(HIDDEN, codec.CLASSES))

    def forward(self, features):
        p = torch.softmax(self.net(features), -1)
        return (1-EPSILON)*p+EPSILON/codec.CLASSES


def cell_beta(stream):
    beta = torch.full(stream['support'].shape, stream['beta'], dtype=torch.float32, device=stream['support'].device)
    if stream['levels'] is not None:
        for level, value in enumerate(codec.REFINE_BETAS, 1):
            beta[stream['levels'] == level] = value
    return beta


def compact(stream):
    """What the decoder knows besides the classes: CPU-storable int8 summary."""
    shape = stream['shape']
    classes = codec.magnitude_class(codec.from_tiles(stream['index'], shape)).to(torch.int8)
    return dict(classes=classes, support=stream['support'], beta=cell_beta(stream))


@torch.no_grad()
def features(classes, support, beta):
    """[rx, d, r, 2, FEATURES] causal features (float32) for every component."""
    c = classes.to(torch.float32)/8
    rx = c.shape[0]
    zeros = torch.zeros_like(c[:1])
    prev1 = torch.cat((zeros, c[:-1]))
    prev2 = torch.cat((zeros, zeros, c[:-2]))
    p1max = prev1.max(-1).values
    near_max = F.max_pool2d(p1max[:, None], 3, 1, 1)[:, 0]
    near_mean = F.avg_pool2d(p1max[:, None], 3, 1, 1)[:, 0]
    wide_mean = F.avg_pool2d(p1max[:, None], 9, 1, 4)[:, 0]
    sup = support.to(torch.float32)
    sup_mean = F.avg_pool2d(sup[None, None], 3, 1, 1)[0, 0].expand_as(p1max)
    plane = torch.arange(rx, device=c.device, dtype=torch.float32)[:, None, None].expand_as(p1max)
    log_beta = (torch.log2(beta)/4).expand_as(p1max)
    out = []
    for comp in (0, 1):
        same_real = c[..., 0] if comp == 1 else torch.zeros_like(p1max)
        out.append(torch.stack((prev1[..., 0], prev1[..., 1], near_max, near_mean, prev2.max(-1).values,
                                same_real, torch.full_like(p1max, float(comp)), plane/15, (plane == 0).float(),
                                log_beta, sup_mean, wide_mean), -1))
    return torch.stack(out, 3)


def coded(support, shape):
    return support[None, :, :, None].expand(shape[0], -1, -1, 2)


@torch.no_grad()
def sample(summary, count, generator):
    """Random coded components: (features, class targets) for training."""
    f = features(summary['classes'], summary['support'], summary['beta'])
    mask = coded(summary['support'], summary['classes'].shape)
    rows, targets = f[mask], summary['classes'][mask].to(torch.int64)
    pick = torch.randperm(len(rows), generator=generator, device=rows.device)[:count]
    return rows[pick].to(torch.float16), targets[pick].to(torch.int8)


def quantized_probabilities(model, rows):
    """Integer frequencies (>=1, total <= 2^16) exactly as the coder uses them."""
    p = model(rows.to(torch.float32))
    freq = torch.floor(p*((1 << FREQUENCY_BITS)-codec.CLASSES)).to(torch.int64)+1
    return freq, freq.sum(-1)


@torch.no_grad()
def learned_payload_bits(model, summary, chunk=1 << 20):
    """Ideal arithmetic code length with the quantized model + raw bits + flush."""
    f = features(summary['classes'], summary['support'], summary['beta'])
    mask = coded(summary['support'], summary['classes'].shape)
    rows, targets = f[mask], summary['classes'][mask].to(torch.int64)
    total = 0.
    for start in range(0, len(rows), chunk):
        freq, norm = quantized_probabilities(model, rows[start:start+chunk])
        chosen = freq.gather(1, targets[start:start+chunk, None])[:, 0]
        total += float((torch.log2(norm.double())-torch.log2(chosen.double())).sum())
    return math.ceil(total)+codec.ARITHMETIC_TERMINATION_BITS+int(targets.sum())


def train(model, train_rows, train_targets, val_rows, val_targets, *, epochs, batch, lr, seed, log):
    """Minimize cross-entropy in bits; keep the epoch with the lowest VAL bits/symbol."""
    generator = torch.Generator(device=train_rows.device).manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, history = math.inf, None, []

    def bits_per_symbol(rows, targets):
        model.eval()
        with torch.no_grad():
            total = sum(float(-torch.log2(model(rows[i:i+batch].float()).gather(1, targets[i:i+batch, None].long())).sum())
                        for i in range(0, len(rows), batch))
        return total/len(rows)

    for epoch in range(epochs+1):
        if epoch:
            model.train()
            order = torch.randperm(len(train_rows), generator=generator, device=train_rows.device)
            for i in range(0, len(order), batch):
                pick = order[i:i+batch]
                p = model(train_rows[pick].float()).gather(1, train_targets[pick, None].long())
                loss = -torch.log2(p).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        row = dict(epoch=epoch, train_bits_per_symbol=bits_per_symbol(train_rows, train_targets),
                   val_bits_per_symbol=bits_per_symbol(val_rows, val_targets))
        history.append(row)
        log(row)
        if row['val_bits_per_symbol'] < best:
            best = row['val_bits_per_symbol']
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return history, min(history, key=lambda r: r['val_bits_per_symbol'])['epoch']


@torch.no_grad()
def assert_causal(summary, rx_plane):
    """Features of plane rx use only planes < rx (and, for imag, the same real)."""
    full = features(summary['classes'], summary['support'], summary['beta'])[rx_plane]
    masked = summary['classes'].clone()
    masked[rx_plane+1:] = 0
    masked[rx_plane, ..., 1] = 0
    partial = features(masked, summary['support'], summary['beta'])[rx_plane]
    if not torch.equal(full, partial):
        raise AssertionError('Entropy-model context is not causal; the stream would not be decodable.')


# CPU reference coder with per-symbol model frequencies (Witten-Neal-Cleary).
def write_model_stream(classes, frequencies):
    low, high, pending, bits = 0, codec._FULL-1, 0, []

    def emit(bit):
        nonlocal pending
        bits.append(bit)
        bits.extend([1-bit]*pending)
        pending = 0

    for c, freq in zip(classes, frequencies):
        total, below = sum(freq), sum(freq[:c])
        span = high-low+1
        high = low+span*(below+freq[c])//total-1
        low = low+span*below//total
        while True:
            if high < codec._HALF:
                emit(0)
            elif low >= codec._HALF:
                emit(1); low -= codec._HALF; high -= codec._HALF
            elif low >= codec._QUARTER and high < 3*codec._QUARTER:
                pending += 1; low -= codec._QUARTER; high -= codec._QUARTER
            else:
                break
            low, high = 2*low, 2*high+1
    pending += 1
    emit(0 if low < codec._QUARTER else 1)
    return bits


def read_model_stream(bits, frequencies):
    low, high, value, position = 0, codec._FULL-1, 0, 0

    def take():
        nonlocal position
        position += 1
        return bits[position-1] if position-1 < len(bits) else 0

    for _ in range(32):
        value = (value << 1) | take()
    out = []
    for freq in frequencies:
        total = sum(freq)
        span = high-low+1
        target = ((value-low+1)*total-1)//span
        below, c = 0, 0
        while below+freq[c] <= target:
            below += freq[c]
            c += 1
        high = low+span*(below+freq[c])//total-1
        low = low+span*below//total
        while True:
            if high < codec._HALF:
                pass
            elif low >= codec._HALF:
                low -= codec._HALF; high -= codec._HALF; value -= codec._HALF
            elif low >= codec._QUARTER and high < 3*codec._QUARTER:
                low -= codec._QUARTER; high -= codec._QUARTER; value -= codec._QUARTER
            else:
                break
            low, high, value = 2*low, 2*high+1, 2*value+take()
        out.append(c)
    return out


@torch.no_grad()
def coder_check(model, summary, count=4096):
    """Real arithmetic coding of the first coded components with the model."""
    f = features(summary['classes'], summary['support'], summary['beta'])
    mask = coded(summary['support'], summary['classes'].shape)
    rows, targets = f[mask][:count], summary['classes'][mask][:count].to(torch.int64)
    freq, norm = quantized_probabilities(model, rows)
    ideal = float((torch.log2(norm.double())-torch.log2(freq.gather(1, targets[:, None])[:, 0].double())).sum())
    tables, symbols = freq.cpu().tolist(), targets.cpu().tolist()
    written = write_model_stream(symbols, tables)
    if read_model_stream(written, tables) != symbols:
        raise AssertionError('Learned-model arithmetic stream does not decode.')
    return dict(symbols=len(symbols), real_bits=len(written), ideal_bits=ideal)
