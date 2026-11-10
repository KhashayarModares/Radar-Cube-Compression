"""V13 detection-preserving RADIal cube codec and RD detection-set metrics.

Codec domain: the unshifted orthonormal 2-D FFT over (chirp, fast time) of each
Rx channel. Index [rx, d, r] is the receiver's range bin r and cyclic Doppler
bin d before its range/Doppler windows, so coding error stays within the
window main lobe of the cell it was made in.

Nothing is ranked or pruned. Every real component is uniformly quantized with
a step tied to the tile noise level (step = beta * sigma_tile):
  * the added error per component is about step^2/12, so the CFAR background
    rises by a bounded 10*log10(1 + beta^2/12) dB without noise filling;
  * components below step/2 become zero; the decoder replaces zeroed
    background components with Gaussian noise of the transmitted energy
    (perceptual noise substitution), so the CA-CFAR background is preserved;
  * cells around the encoder's own detections use a finer step (ROI coding);
  * nonzero indices use a centroid reconstruction offset theta per tile.
Indices are coded with a decodable zero-run/Exp-Golomb code. ``payload_bits``
returns its exact length; ``write_tile``/``read_tile`` are the CPU reference
bit writer/reader used by the check stage to prove that length and decodability.

The decoder (``decode``) sees only the transmitted stream, never the cube.

V13.1 adds the codec family that the V13 smoke result motivated
(``encode_ecdq``/``decode_ecdq``):
  * subtractive dithered uniform quantization (ECDQ): the error is uniform,
    independent of the data and of variance step^2/12 (Schuchman 1964; Gray &
    Stockham 1993), and entropy-coded it is within 0.254 bit per real sample of
    the rate-distortion function (Zamir & Feder 1992). No dead zone removes
    noise, so the CFAR background cannot collapse; the predicted background
    rise is 10*log10(1 + beta^2/12) dB;
  * an optional detection support S (cells whose ORIGINAL CFAR score is within
    gamma dB of alpha, dilated by the window main lobe). Outside S nothing is
    quantized: the decoder synthesizes Gaussian noise with the transmitted
    local variance. This is the perfect-perception end of the
    rate-distortion-perception trade-off (Blau & Michaeli 2019): distortion
    where the detector needs fidelity, matched statistics where it only needs
    the CA-CFAR background estimate;
  * adaptive arithmetic coding: per-tile Krichevsky-Trofimov models of the
    magnitude class bit_length(|q|), with mantissa and sign bits sent raw.
    ``arithmetic_bits`` is the exact ideal adaptive code length;
    ``write_arithmetic_tile``/``read_arithmetic_tile`` are a 32-bit
    Witten-Neal-Cleary coder used to show it is achieved.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

NMS_RADIUS = 2  # the receiver's fixed 5x5 power NMS
TILE_DOPPLER = 16
TILE_RANGE = 32
K_MAX = 24
SIGMA_BITS, THETA_BITS, FILL_BITS, K_BITS = 16, 9, 8, 5
CUBE_HEADER_BITS = 32+8+8+2+1+8+16  # seed, beta_bg, beta_det, radius, fill, layout, point count
POINT_BITS = 9+8                    # range bin (512), Doppler bin (256)
FILL_RATIO_MAX = .25                # zeroed |x| < step/2, so E[x^2]/step^2 < 1/4


def forward(cube):
    return torch.fft.fft2(cube.to(torch.complex128), dim=(1, 2), norm='ortho')


def inverse(spectrum):
    return torch.fft.ifft2(spectrum, dim=(1, 2), norm='ortho')


def layout(shape):
    rx, doppler, rng = shape
    if doppler % TILE_DOPPLER or rng % TILE_RANGE:
        raise ValueError('Cube does not tile exactly.')
    return rx, doppler//TILE_DOPPLER, rng//TILE_RANGE


def to_tiles(real, shape):
    """[rx, d, r, 2] real components -> [tile, rx*td*tr*2], rows contiguous per tile."""
    rx, nd, nr = layout(shape)
    return (real.reshape(rx, nd, TILE_DOPPLER, nr, TILE_RANGE, 2)
            .permute(1, 3, 0, 2, 4, 5).reshape(nd*nr, -1))


def from_tiles(tiles, shape):
    rx, nd, nr = layout(shape)
    return (tiles.reshape(nd, nr, rx, TILE_DOPPLER, TILE_RANGE, 2)
            .permute(2, 0, 3, 1, 4, 5).reshape(rx, shape[1], shape[2], 2))


def cells_to_tiles(cells, shape):
    """[d, r] cell map -> broadcast onto the tile component layout."""
    rx, nd, nr = layout(shape)
    value = cells.reshape(nd, TILE_DOPPLER, nr, TILE_RANGE).permute(0, 2, 1, 3)
    value = value.reshape(nd*nr, 1, TILE_DOPPLER, TILE_RANGE, 1)
    return value.expand(-1, rx, -1, -1, 2).reshape(nd*nr, -1)


def sigma_code(tiles):
    """Robust per-component noise level, coded as 16-bit log2 with 1/256 octave."""
    power = tiles.reshape(len(tiles), -1, 2).square().sum(-1)
    # |z|^2 of a complex Gaussian is exponential; its median is 2*sigma^2*ln 2.
    sigma = (power.median(dim=1).values/(2*math.log(2))).sqrt()
    tiny = torch.finfo(torch.float64).tiny
    return torch.round(torch.log2(sigma.clamp_min(tiny))*256).clamp(-32768, 32767).to(torch.int64)


def sigma_value(code):
    return torch.exp2(code.to(torch.float64)/256)


def protection_mask(points, radius, shape):
    """[d, r] cells within +-radius (cyclic Doppler) of the given (range, Doppler) points."""
    _, doppler, rng = shape
    mask = torch.zeros((doppler, rng), dtype=torch.float64, device=points.device)
    if len(points) and radius >= 0:
        mask[points[:, 1], points[:, 0]] = 1.
        if radius:
            padded = F.pad(mask[None, None], (0, 0, radius, radius), mode='circular')
            padded = F.pad(padded, (radius, radius, 0, 0))
            mask = F.max_pool2d(padded, 2*radius+1, stride=1)[0, 0]
    return mask > 0


def _steps(stream, shape):
    sigma = sigma_value(stream['sigma_code'])
    protect = protection_mask(stream['points'], stream['radius'], shape)
    beta = torch.full(protect.shape, float(stream['beta_bg']), dtype=torch.float64, device=protect.device)
    beta[protect] = float(stream['beta_det'])
    return sigma[:, None]*cells_to_tiles(beta, shape), cells_to_tiles(~protect, shape)


@torch.no_grad()
def encode(cube, *, beta_bg, beta_det=None, points=None, radius=2, fill=True, seed=0):
    """Return the transmitted stream (integers and codes only)."""
    shape = tuple(cube.shape)
    device = cube.device
    if points is None or beta_det is None:
        points, beta_det = torch.zeros((0, 2), dtype=torch.int64, device=device), beta_bg
    tiles = to_tiles(torch.view_as_real(forward(cube)), shape)
    stream = dict(sigma_code=sigma_code(tiles), points=points.to(torch.int64), radius=int(radius),
                  beta_bg=float(beta_bg), beta_det=float(beta_det), fill=bool(fill), seed=int(seed))
    step, background = _steps(stream, shape)
    scaled = tiles/step
    index = torch.round(scaled).to(torch.int64)
    nonzero = index != 0
    gap = ((index.abs()-scaled.abs())*nonzero).sum(1)/nonzero.sum(1).clamp_min(1)
    stream['theta_code'] = torch.round(gap.clamp(0, .5)*512).to(torch.int64)
    zeroed = ~nonzero & background
    ratio = (scaled.square()*zeroed).sum(1)/zeroed.sum(1).clamp_min(1)
    code = torch.round(ratio.clamp(0, FILL_RATIO_MAX)/FILL_RATIO_MAX*255).to(torch.int64)
    stream['fill_code'] = code if fill else torch.zeros_like(code)
    stream['index'] = index
    return stream


@torch.no_grad()
def decode(stream, shape, dtype=torch.complex64):
    index = stream['index']
    step, background = _steps(stream, shape)
    theta = stream['theta_code'].to(torch.float64)[:, None]/512
    value = torch.sign(index)*(index.abs()-theta)*step
    value = torch.where(index != 0, value, torch.zeros_like(value))
    if stream['fill']:
        generator = torch.Generator(device=index.device).manual_seed(stream['seed'])
        noise = torch.randn(index.shape, generator=generator, device=index.device, dtype=torch.float64)
        energy = stream['fill_code'].to(torch.float64)[:, None]/255*FILL_RATIO_MAX
        value = torch.where((index == 0) & background, noise*energy.sqrt()*step, value)
    spectrum = torch.view_as_complex(from_tiles(value, shape).contiguous())
    return inverse(spectrum).to(dtype)


def floor_log2(value):
    _, exponent = torch.frexp(value.to(torch.float64))
    return exponent.to(torch.int64)-1


def exp_golomb_length(value, k):
    """Exact length of the order-k Exp-Golomb code of nonnegative integers."""
    return 2*floor_log2(value+(1 << k))-k+1


@torch.no_grad()
def payload_bits(index, keep=None):
    """Per-tile exact code length and chosen (k_run, k_mag).

    Tile syntax: EG0(nonzero count), k_run (5 bits), k_mag (5 bits), then for
    each nonzero in row order: EG_krun(zero run before it), EG_kmag(|i|-1),
    sign bit. Trailing zeros are implied by the count and the known tile size.
    With ``keep`` only kept components are coded (the decoder knows the mask),
    runs count kept positions, and tiles without kept components cost nothing.
    """
    nonzero = index != 0
    if keep is not None:
        nonzero &= keep
    counts = nonzero.sum(1)
    position = torch.nonzero(nonzero)
    tile, column = position[:, 0], position[:, 1]
    magnitude = index[tile, column].abs()-1
    if keep is not None:
        column = (keep.cumsum(1)-1)[tile, column]
    previous = torch.full_like(column, -1)
    if len(column):
        previous[1:] = column[:-1]
        first = torch.ones_like(tile, dtype=torch.bool)
        first[1:] = tile[1:] != tile[:-1]
        previous = torch.where(first, -1, previous)
    run = column-previous-1
    ends = counts.cumsum(0)
    starts = ends-counts

    def per_tile(lengths):
        total = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
        return total[ends]-total[starts]

    run_bits = torch.stack([per_tile(exp_golomb_length(run, k)) for k in range(K_MAX)], 1)
    mag_bits = torch.stack([per_tile(exp_golomb_length(magnitude, k)) for k in range(K_MAX)], 1)
    best_run, k_run = run_bits.min(1)
    best_mag, k_mag = mag_bits.min(1)
    bits = exp_golomb_length(counts, 0)+2*K_BITS+best_run+best_mag+counts
    if keep is not None:
        bits = torch.where(keep.any(1), bits, torch.zeros_like(bits))
    return bits, k_run, k_mag


@torch.no_grad()
def stream_bits(stream):
    tile_payload, _, _ = payload_bits(stream['index'])
    arithmetic = arithmetic_bits(stream['index'])
    side = len(tile_payload)*(SIGMA_BITS+THETA_BITS+(FILL_BITS if stream['fill'] else 0))
    header = CUBE_HEADER_BITS+POINT_BITS*len(stream['points'])
    return dict(total=int(arithmetic.sum())+side+header, total_eg=int(tile_payload.sum())+side+header,
                payload=int(arithmetic.sum()), side=side, header=header, protected_points=len(stream['points']))


# V13.1: ECDQ, detection support with synthesized background, arithmetic coding.
CLASSES = 32                    # magnitude classes bit_length(|q|) = 0..31
ARITHMETIC_TERMINATION_BITS = 2  # Witten-Neal-Cleary flush
VAR_BLOCK = (8, 4)              # Doppler x range cells per synthesis-variance value
VAR_BITS, GAIN_BITS = 12, 8     # log2 variance in 1/16 octave; per-Rx log2 gain in 1/64 octave
ECDQ_HEADER_BITS = 32+8+8+1+8   # seed, beta, gamma, synthesis flag, layout


def magnitude_class(index):
    magnitude = index.abs()
    return torch.where(magnitude > 0, floor_log2(magnitude.clamp_min(1))+1, torch.zeros_like(magnitude))


def kt_bits(counts):
    """Krichevsky-Trofimov sequential (adaptive) code length; depends on counts only."""
    counts = counts.to(torch.float64)
    letters = counts.shape[-1]
    nats = (torch.lgamma(counts.sum(-1)+letters/2)-math.lgamma(letters/2)
            -(torch.lgamma(counts+.5)-math.lgamma(.5)).sum(-1))
    return nats/math.log(2)


@torch.no_grad()
def arithmetic_bits(index, keep=None):
    """Per-tile adaptive arithmetic code length of the kept indices.

    Classes c = bit_length(|q|) are arithmetic-coded with a per-tile KT model;
    each nonzero then sends c-1 mantissa bits and one sign bit raw, i.e. c raw
    bits in total. Tiles without kept components cost nothing.
    """
    if keep is None:
        keep = torch.ones_like(index, dtype=torch.bool)
    classes = magnitude_class(index)
    if bool((classes[keep] >= CLASSES).any()):
        raise ValueError('Quantization index exceeds the 31-bit magnitude classes.')
    counts = torch.stack([((classes == c) & keep).sum(1) for c in range(CLASSES)], 1)
    raw = (classes*keep).sum(1).to(torch.float64)
    bits = torch.ceil(kt_bits(counts))+ARITHMETIC_TERMINATION_BITS+raw
    return torch.where(keep.any(1), bits, torch.zeros_like(bits)).to(torch.int64)


def _uniform(shape, seed, device):
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.rand(shape, generator=generator, device=device, dtype=torch.float64)-.5


def _gaussian(shape, seed, device):
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(shape, generator=generator, device=device, dtype=torch.float64)


@torch.no_grad()
def detection_support(detector, alpha, gamma_db, radius=2):
    """Codec-domain [d, r] cells the detector could need: original CFAR score
    within gamma dB of alpha (or unscored edge rows), dilated by the window main
    lobe. Computed by the encoder from the original cube only."""
    cells = (detector['score'] >= alpha*10**(-gamma_db/10)) | ~detector['valid']
    if radius:
        cells = neighbourhood_max(cells.to(torch.float64), radius, radius) > 0
    return cells.T.contiguous()


def _synthesis_codes(real, background, block):
    """Per-block mean |z|^2 over background cells and per-Rx relative gain."""
    rx, doppler, rng = real.shape[:3]
    bd, br = block
    power = real.square().sum(-1)
    mask = background.to(power.dtype)
    per_rx = (power*mask).sum((1, 2))/mask.sum().clamp_min(1)
    gain = per_rx/per_rx.mean().clamp_min(torch.finfo(torch.float64).tiny)
    block_power = (power.mean(0)*mask).reshape(doppler//bd, bd, rng//br, br).sum((1, 3))
    block_count = mask.reshape(doppler//bd, bd, rng//br, br).sum((1, 3))
    variance = block_power/block_count.clamp_min(1)
    tiny = torch.finfo(torch.float64).tiny
    return dict(variance_code=torch.round(torch.log2(variance.clamp_min(tiny))*16).clamp(-2048, 2047).to(torch.int64),
                gain_code=torch.round(torch.log2(gain.clamp_min(tiny))*64).clamp(-128, 127).to(torch.int64))


REFINE_BETAS = (.5, .1)  # successive-refinement steps (x sigma) for refinement levels 1, 2


def _ecdq_steps(stream):
    """Per-component step: beta*sigma_tile; REFINE_BETAS[k-1]*sigma_tile at refinement level k."""
    beta = torch.full(stream['support'].shape, stream['beta'], dtype=torch.float64, device=stream['support'].device)
    if stream['levels'] is not None:
        for level, value in enumerate(REFINE_BETAS, 1):
            beta[stream['levels'] == level] = value
    return sigma_value(stream['sigma_code'])[:, None]*cells_to_tiles(beta, stream['shape'])


@torch.no_grad()
def encode_ecdq(cube, *, beta, support=None, gamma_db=None, seed=0, block=VAR_BLOCK, levels=None):
    """Subtractive-dither quantization of the support; statistics elsewhere.
    ``levels`` (int map, nonzero only inside the support) selects finer steps."""
    shape = tuple(cube.shape)
    real = torch.view_as_real(forward(cube))
    tiles = to_tiles(real, shape)
    code = sigma_code(tiles)
    if support is None:
        support = torch.ones(shape[1:], dtype=torch.bool, device=cube.device)
    if levels is not None:
        levels = torch.where(support, levels, torch.zeros_like(levels))
    keep = cells_to_tiles(support, shape)
    stream = dict(mode='ecdq', shape=shape, sigma_code=code, beta=float(beta), seed=int(seed), block=tuple(block),
                  support=support, gamma_db=gamma_db, synthesis=not bool(support.all()),
                  levels=levels)
    step = _ecdq_steps(stream)
    dither = _uniform(tiles.shape, seed, cube.device)*step
    stream['index'] = torch.where(keep, torch.round((tiles+dither)/step), torch.zeros_like(tiles)).to(torch.int64)
    if stream['synthesis']:
        stream.update(_synthesis_codes(real, ~support, stream['block']))
    return stream


@torch.no_grad()
def decode_ecdq(stream, dtype=torch.complex64):
    shape, index = stream['shape'], stream['index']
    keep = cells_to_tiles(stream['support'], shape)
    step = _ecdq_steps(stream)
    dither = _uniform(index.shape, stream['seed'], index.device)*step
    real = from_tiles(torch.where(keep, index*step-dither, torch.zeros_like(step.expand_as(dither))), shape)
    if stream['synthesis']:
        bd, br = stream['block']
        variance = torch.exp2(stream['variance_code'].to(torch.float64)/16)
        variance = variance.repeat_interleave(bd, 0).repeat_interleave(br, 1)
        gain = torch.exp2(stream['gain_code'].to(torch.float64)/64)
        std = (variance[None]*gain[:, None, None]/2).sqrt()
        noise = _gaussian(real.shape, stream['seed']+1, index.device)*std[..., None]
        real = torch.where(stream['support'][None, :, :, None], real, noise)
    return inverse(torch.view_as_complex(real.contiguous())).to(dtype)


@torch.no_grad()
def ecdq_bits(stream):
    shape, index = stream['shape'], stream['index']
    keep = cells_to_tiles(stream['support'], shape)
    arithmetic = arithmetic_bits(index, keep)
    eg, _, _ = payload_bits(index, keep)
    side = int(keep.any(1).sum())*SIGMA_BITS
    if stream['synthesis']:
        support = stream['support']
        n = support.numel()
        ones = int(support.sum())
        side += math.ceil(float(kt_bits(torch.tensor([[n-ones, ones]]))[0]))+ARITHMETIC_TERMINATION_BITS
        side += stream['variance_code'].numel()*VAR_BITS+stream['gain_code'].numel()*GAIN_BITS
    if stream['levels'] is not None:
        counts = torch.stack([(stream['levels'] == k).sum() for k in range(len(REFINE_BETAS)+1)]).cpu()
        side += math.ceil(float(kt_bits(counts[None])[0]))+ARITHMETIC_TERMINATION_BITS
    return dict(total=int(arithmetic.sum())+side+ECDQ_HEADER_BITS, total_eg=int(eg.sum())+side+ECDQ_HEADER_BITS,
                payload=int(arithmetic.sum()), side=side, header=ECDQ_HEADER_BITS,
                support_fraction=float(stream['support'].double().mean()),
                refine_fraction=0. if stream['levels'] is None else float((stream['levels'] > 0).double().mean()))


@torch.no_grad()
def disagreement_cells(original, method, alpha, margin_db=1., match=NMS_RADIUS, strict_lost=False, strict_new=False):
    """RD [range, Doppler] cells of robust lost and robust new detections; with
    ``strict_lost`` every original detection above alpha that vanishes counts."""
    margin = 10**(margin_db/10)
    s0, s1 = peak_scores(original), peak_scores(method)
    near0, near1 = neighbourhood_max(s0, match, match), neighbourhood_max(s1, match, match)
    kept = (s0 > alpha) if strict_lost else stable_peaks(original, alpha, margin)
    lost = kept & (near1 <= alpha)
    new = ((s1 > alpha) & (near0 <= alpha)) if strict_new else (stable_peaks(method, alpha, margin) & (near0 <= alpha/margin))
    return lost | new


CFAR_HALF_WINDOW = 5  # the receiver's 11x11 CA-CFAR window


@torch.no_grad()
def encode_closed_loop(cube, receive, original, alpha, *, gamma_db, beta, seed, iterations=6, margin_db=1.,
                       strict_lost=False, strict_new=False):
    """Analysis-by-synthesis: the decoder is deterministic, so the encoder
    decodes its own stream, runs the receiver, and moves the full CA-CFAR
    window of every robust lost/new detection into the faithfully coded
    support; repeat until no robust disagreement is left or the iteration
    budget is spent (cf. analysis-by-synthesis coding, Atal & Remde 1982).
    A window already inside the support moves one refinement level finer
    instead (successive refinement): there the loss came from the quantization
    noise of its current step."""
    support = detection_support(original, alpha, gamma_db)
    levels = torch.zeros(support.shape, dtype=torch.int64, device=support.device)
    history = []
    for step in range(iterations+1):
        stream = encode_ecdq(cube, beta=beta, support=support, gamma_db=gamma_db, seed=seed,
                             levels=levels if bool(levels.any()) else None)
        decoded = decode_ecdq(stream, cube.dtype)
        cells = disagreement_cells(original, receive(decoded)['detector'], alpha, margin_db,
                                   strict_lost=strict_lost, strict_new=strict_new)
        history.append(int(cells.sum()))
        if not bool(cells.any()) or step == iterations:
            break
        grown = (neighbourhood_max(cells.to(torch.float64), CFAR_HALF_WINDOW, CFAR_HALF_WINDOW) > 0).T
        levels = torch.where(grown & support, (levels+1).clamp_max(len(REFINE_BETAS)), levels)
        support = support | grown
    return stream, decoded, history


# CPU reference arithmetic coder (Witten, Neal & Cleary 1987), 32-bit.
_FULL = 1 << 32
_HALF, _QUARTER = _FULL >> 1, _FULL >> 2


def write_arithmetic_tile(symbols, with_section=False):
    """Classes with an adaptive KT model, then raw mantissa and sign bits.
    ``with_section`` also returns the arithmetic-section length (bits)."""
    bits = _encode_classes([abs(value).bit_length() for value in symbols])
    section = len(bits)
    for value in symbols:
        if value:
            bits.extend(int(b) for b in format(abs(value), 'b')[1:])
            bits.append(1 if value < 0 else 0)
    return (bits, section) if with_section else bits


def _encode_classes(classes):
    # freq[c] = 2*count[c]+1 and total = sum(freq): the KT estimator in integers.
    freq, total, low, high, pending, bits = [1]*CLASSES, CLASSES, 0, _FULL-1, 0, []

    def emit(bit):
        nonlocal pending
        bits.append(bit)
        bits.extend([1-bit]*pending)
        pending = 0

    for c in classes:
        below = sum(freq[:c])
        span = high-low+1
        high = low+span*(below+freq[c])//total-1
        low = low+span*below//total
        while True:
            if high < _HALF:
                emit(0)
            elif low >= _HALF:
                emit(1); low -= _HALF; high -= _HALF
            elif low >= _QUARTER and high < 3*_QUARTER:
                pending += 1; low -= _QUARTER; high -= _QUARTER
            else:
                break
            low, high = 2*low, 2*high+1
        freq[c] += 2
        total += 2
    pending += 1
    emit(0 if low < _QUARTER else 1)
    return bits


def read_arithmetic_tile(bits, length, section=None):
    """Inverse of write_arithmetic_tile; ``section`` (if stored) skips re-encoding."""
    freq, total, low, high, position = [1]*CLASSES, CLASSES, 0, _FULL-1, 0

    def take():
        nonlocal position
        position += 1
        return bits[position-1] if position-1 < len(bits) else 0

    value = 0
    for _ in range(32):
        value = (value << 1) | take()
    classes = []
    for _ in range(length):
        span = high-low+1
        target = ((value-low+1)*total-1)//span
        below, c = 0, 0
        while below+freq[c] <= target:
            below += freq[c]
            c += 1
        high = low+span*(below+freq[c])//total-1
        low = low+span*below//total
        while True:
            if high < _HALF:
                pass
            elif low >= _HALF:
                low -= _HALF; high -= _HALF; value -= _HALF
            elif low >= _QUARTER and high < 3*_QUARTER:
                low -= _QUARTER; high -= _QUARTER; value -= _QUARTER
            else:
                break
            low, high, value = 2*low, 2*high+1, 2*value+take()
        freq[c] += 2
        total += 2
        classes.append(c)
    # The arithmetic section length is a deterministic function of the decoded
    # classes; without a stored length the decoder finds it by re-encoding them.
    position = len(_encode_classes(classes)) if section is None else section
    symbols = []
    for c in classes:
        if not c:
            symbols.append(0)
            continue
        magnitude = int(''.join(['1']+[str(b) for b in bits[position:position+c-1]]), 2)
        position += c-1
        symbols.append(-magnitude if bits[position] else magnitude)
        position += 1
    if position != len(bits):
        raise ValueError('Trailing bits in arithmetic tile stream.')
    return symbols


# CPU reference bitstream, used only to verify payload_bits and decodability.
def _eg_write(value, k):
    value += 1 << k
    width = value.bit_length()
    return '0'*(width-1-k)+format(value, 'b')


def _eg_read(bits, at, k):
    zeros = 0
    while bits[at+zeros] == '0':
        zeros += 1
    width = zeros+1+k
    value = int(bits[at+zeros:at+zeros+width], 2)
    return value-(1 << k), at+zeros+width


def write_tile(symbols, k_run, k_mag):
    nonzero = [(j, v) for j, v in enumerate(symbols) if v]
    parts = [_eg_write(len(nonzero), 0), format(k_run, '05b'), format(k_mag, '05b')]
    previous = -1
    for j, v in nonzero:
        parts += [_eg_write(j-previous-1, k_run), _eg_write(abs(v)-1, k_mag), '1' if v < 0 else '0']
        previous = j
    return ''.join(parts)


def read_tile(bits, length):
    count, at = _eg_read(bits, 0, 0)
    k_run, k_mag = int(bits[at:at+5], 2), int(bits[at+5:at+10], 2)
    at += 10
    symbols = [0]*length
    position = -1
    for _ in range(count):
        run, at = _eg_read(bits, at, k_run)
        magnitude, at = _eg_read(bits, at, k_mag)
        position += run+1
        symbols[position] = -(magnitude+1) if bits[at] == '1' else magnitude+1
        at += 1
    if at != len(bits):
        raise ValueError('Trailing bits in tile stream.')
    return symbols


# Receiver (RD only) and detection-set metrics.
@torch.no_grad()
def rd_receive(cube, range_window, doppler_window, cfar):
    """Same operations as RadialGPUPhysics.maps() for RD; returns spectrum too."""
    adc = cube.permute(2, 1, 0).to(torch.complex128)
    adc = adc-adc.mean(dim=(0, 1), keepdim=True)
    spectrum = torch.fft.fft(adc*range_window, n=adc.shape[0], dim=0)
    spectrum = torch.fft.fft(spectrum*doppler_window, n=adc.shape[1], dim=1)
    amplitude = spectrum.abs().sum(dim=2)
    rd = 20*torch.log10(amplitude.clamp_min(torch.finfo(torch.float64).tiny))
    return dict(RD=rd, spectrum=spectrum, detector=cfar(10.**(rd/10.)))


def neighbourhood_max(value, radius_range, radius_doppler):
    """Max over +-radius; [range, Doppler] maps, Doppler cyclic, range zero-padded."""
    padded = F.pad(value[None, None], (radius_doppler, radius_doppler, 0, 0), mode='circular')
    padded = F.pad(padded, (0, 0, radius_range, radius_range))
    return F.max_pool2d(padded, (2*radius_range+1, 2*radius_doppler+1), stride=1)[0, 0]


def peak_scores(detector):
    return torch.where(detector['base_peaks'] & detector['valid'], detector['score'], 0.)


def detections(detector, alpha):
    """(range, Doppler) positions of NMS peaks whose CFAR score exceeds alpha."""
    return torch.nonzero(peak_scores(detector) > alpha)


def nms_margin(detector):
    """Power of each cell over the largest other power in its 5x5 NMS window."""
    power = detector['power']
    rival = torch.zeros_like(power)
    for dr in range(-NMS_RADIUS, NMS_RADIUS+1):
        for dd in range(-NMS_RADIUS, NMS_RADIUS+1):
            if dr or dd:
                rival = torch.maximum(rival, torch.roll(power, (dr, dd), dims=(0, 1)))
    return power/rival.clamp_min(1e-300)


def stable_peaks(detector, alpha, margin):
    """Peaks whose decision is not within the margin of the CFAR threshold or of
    the NMS power comparison. The receiver thresholds the CFAR score but picks
    NMS winners by power, so a near-tie can hand the peak to a neighbour whose
    differently windowed score lies below alpha; such decisions are ambiguous
    for any perturbation, including a second noise realisation."""
    return (peak_scores(detector) > alpha*margin) & (nms_margin(detector) >= margin)


@torch.no_grad()
def detection_metrics(original, method, alpha, margin_db=1., match=NMS_RADIUS):
    """Set agreement of NMS peaks above alpha within +-match bins.

    The NMS keeps one maximum per 5x5 window, so a detection's position is only
    defined up to +-2 bins: on a flat-topped extended target an arbitrarily
    small perturbation can move the winning cell two bins. Matching at +-1
    would count that as one lost plus one new detection. Robust counts consider
    only ``stable_peaks``; the rest are reported as ambiguous.
    """
    margin = 10**(margin_db/10)
    s0, s1 = peak_scores(original), peak_scores(method)
    near0, near1 = neighbourhood_max(s0, match, match), neighbourhood_max(s1, match, match)
    d0, d1 = s0 > alpha, s1 > alpha
    stable0, stable1 = stable_peaks(original, alpha, margin), stable_peaks(method, alpha, margin)
    # Same-cell CFAR score change; a neighbourhood max would bias it upwards.
    change = 10*torch.log10(method['score'][d0].clamp_min(1e-300)/s0[d0])
    valid = original['valid']
    background = 10*torch.log10(method['noise'][valid].clamp_min(1e-300)/original['noise'][valid].clamp_min(1e-300))
    count = int(d0.sum())
    return dict(original=count, reconstructed=int(d1.sum()), original_robust=int(stable0.sum()),
        original_ambiguous=count-int(stable0.sum()),
        lost=int((d0 & (near1 <= alpha)).sum()), new=int((d1 & (near0 <= alpha)).sum()),
        lost_robust=int((stable0 & (near1 <= alpha)).sum()),
        new_robust=int((stable1 & (near0 <= alpha/margin)).sum()),
        score_change_db_mean=float(change.mean()) if count else math.nan,
        score_change_db_p05=float(torch.quantile(change, .05)) if count else math.nan,
        background_shift_db_median=float(background.median()))


@torch.no_grad()
def disagreements(original, method, alpha, limit=20, match=NMS_RADIUS):
    """Lost and new strict detections with the quantities that decide them."""
    s0, s1 = peak_scores(original), peak_scores(method)
    near0, near1 = neighbourhood_max(s0, match, match), neighbourhood_max(s1, match, match)
    any1 = neighbourhood_max(torch.where(method['valid'], method['score'], 0.), match, match)
    any0 = neighbourhood_max(torch.where(original['valid'], original['score'], 0.), match, match)
    margin0, margin1 = nms_margin(original), nms_margin(method)
    db = lambda x: 10*math.log10(max(float(x), 1e-300))
    rows = []
    for kind, mask, own, other, other_any, own_margin in (
            ('lost', (s0 > alpha) & (near1 <= alpha), s0, near1, any1, margin0),
            ('new', (s1 > alpha) & (near0 <= alpha), s1, near0, any0, margin1)):
        for r, d in torch.nonzero(mask)[:limit].tolist():
            rows.append(dict(kind=kind, range_bin=r, doppler_bin=d, score_db=db(own[r, d]),
                alpha_db=db(alpha), other_best_peak_score_db=db(other[r, d]) if other[r, d] > 0 else None,
                other_best_cell_score_db=db(other_any[r, d]), nms_margin_db=db(own_margin[r, d]),
                same_cell_score_change_db=db(method['score'][r, d])-db(original['score'][r, d])))
    return rows


@torch.no_grad()
def label_scores(original, method, bins, range_gate=2, doppler_gate=2):
    """Best NMS-peak CFAR score inside each label gate, original and method."""
    g0 = neighbourhood_max(peak_scores(original), range_gate, doppler_gate)[bins[:, 0], bins[:, 1]]
    g1 = neighbourhood_max(peak_scores(method), range_gate, doppler_gate)[bins[:, 0], bins[:, 1]]
    return g0, g1


@torch.no_grad()
def label_status(original, method, bins, alpha, range_gate=2, doppler_gate=2, margin_db=1., match=NMS_RADIUS):
    """Per-label booleans, consistent with detection matching.

    A label is detected if an original detection (score > alpha) lies in its
    gate; it is preserved if at least one of those detections is matched by a
    decoded detection within +-match bins, even if the match moved out of the
    gate. It is robust if a stable peak (1 dB clear of alpha and of its NMS
    rival) supports it.
    """
    margin = 10**(margin_db/10)
    s0, s1 = peak_scores(original), peak_scores(method)
    detected0 = (s0 > alpha).to(torch.float64)
    matched0 = ((s0 > alpha) & (neighbourhood_max(s1, match, match) > alpha)).to(torch.float64)
    stable0 = stable_peaks(original, alpha, margin)
    stable_matched0 = (stable0 & (neighbourhood_max(s1, match, match) > alpha)).to(torch.float64)
    at = lambda m: neighbourhood_max(m, range_gate, doppler_gate)[bins[:, 0], bins[:, 1]] > 0
    return dict(detected=at(detected0), preserved=at(matched0), robust=at(stable0.to(torch.float64)),
                robust_preserved=at(stable_matched0), detected_method=at((s1 > alpha).to(torch.float64)))


@torch.no_grad()
def label_metrics(original, method, bins, alpha, range_gate=2, doppler_gate=2, margin_db=1.):
    """Label counts from ``label_status`` (gate +-range_gate / +-doppler_gate)."""
    if not len(bins):
        return dict(labels=0, labels_detected_original=0, labels_robust=0, labels_lost=0, labels_lost_robust=0, labels_new=0)
    g0, _ = label_scores(original, method, bins, range_gate, doppler_gate)
    st = label_status(original, method, bins, alpha, range_gate, doppler_gate, margin_db)
    return dict(labels=len(bins), labels_detected_original=int(st['detected'].sum()), labels_robust=int(st['robust'].sum()),
                labels_lost=int((st['detected'] & ~st['preserved']).sum()),
                labels_lost_robust=int((st['robust'] & ~st['robust_preserved']).sum()),
                labels_new=int((~st['detected'] & st['detected_method']).sum()),
                label_original_best_score_db_median=float(10*torch.log10(g0.clamp_min(1e-300)).median()))


@torch.no_grad()
def h0_scores(count, shape, range_window, doppler_window, cfar, device, seed):
    """NMS peak scores of the exact receiver on white complex Gaussian cubes."""
    generator = torch.Generator(device=device).manual_seed(seed)
    scores = []
    for _ in range(count):
        noise = torch.randn((*shape, 2), generator=generator, device=device, dtype=torch.float32)
        cube = torch.view_as_complex(noise)
        value = peak_scores(rd_receive(cube, range_window, doppler_window, cfar)['detector'])
        scores.append(value[value > 0])
    return torch.cat(scores)


def threshold_for_rate(scores, maps, per_map):
    """Smallest alpha with about per_map NMS peaks above it per noise-only map."""
    ordered = torch.sort(scores, descending=True).values
    k = max(1, int(round(per_map*maps)))
    return float(ordered[min(k, len(ordered)-1)])
