"""Coherent global Range-Doppler transform codec for the V12.14 ablation.

The input and output are raw IQ cubes in [receiver, chirp, fast-time] order.
A single orthonormal full-cube transform on the two time axes precedes tiling;
the receiver axis stays in sensor space. A tile's support is ranked by power
summed over receivers, and a retained Range-Doppler cell retains every receiver
coefficient. This preserves the *support* of antenna vectors, not their phase
exactly: finite precision scalar I/Q quantization can still perturb phase.

This intentionally changes the V12.13 local-time-block codec. It is a controlled
representation ablation, not evidence or a guarantee of better perception.
The byte-aligned value/support/header counts describe a declared binary syntax;
this module does NOT serialize a bitstream or claim a measured on-disk size.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np

EPS = 1e-12
BLOCK_HEADER_BYTES = 16
CUBE_HEADER_BYTES = 96
DEFAULT_ACTIONS = (
    (0.04, 4), (0.06, 4), (0.08, 5), (0.10, 5),
    (0.12, 6), (0.16, 6), (0.20, 6), (0.25, 7),
    (0.32, 7), (0.40, 8), (0.55, 8), (0.70, 9),
    (0.12, 9), (0.16, 10),
)


def _validated_action(action: Sequence[float]) -> tuple[float, int]:
    if len(action) != 2:
        raise ValueError('An action must be (retained_cell_fraction, I/Q_bits).')
    rho, raw_q = float(action[0]), float(action[1])
    if not np.isfinite(rho) or not 0.0 < rho <= 1.0:
        raise ValueError('The retained cell fraction must be finite and in (0, 1].')
    if not np.isfinite(raw_q) or raw_q != int(raw_q) or not 2 <= int(raw_q) <= 30:
        raise ValueError('Quantizer width must be an integer from 2 to 30 bits.')
    return rho, int(raw_q)


def iter_spectral_tiles(shape: Sequence[int], block_slow: int = 32,
                        block_fast: int = 64) -> Iterable[tuple[slice, slice, slice]]:
    """Tile shifted global Doppler/range bins, including thin trailing tiles."""
    nrx, slow, fast = map(int, shape)
    if min(nrx, slow, fast, int(block_slow), int(block_fast)) < 1:
        raise ValueError('Cube dimensions and tile dimensions must be positive.')
    for m0 in range(0, slow, int(block_slow)):
        for n0 in range(0, fast, int(block_fast)):
            yield (slice(0, nrx), slice(m0, min(m0 + int(block_slow), slow)),
                   slice(n0, min(n0 + int(block_fast), fast)))


def support_mask(prepared: dict[str, Any], rho: float) -> np.ndarray:
    """Deterministic top-energy *2D* support; no action or GT preselection.

    A mask refers to cells in the global shifted Range-Doppler grid, not to
    receiver coefficients. The same mask is used for all antenna channels.
    Stable ties use the ascending flattened cell index. Different rho masks
    are therefore nested, unlike truncated neighborhood expansion.
    """
    rho, _ = _validated_action((rho, 2))
    cells = int(prepared['support_cells'])
    kept = min(cells, max(1, int(math.ceil(rho * cells))))
    mask = np.zeros(cells, dtype=bool)
    mask[np.asarray(prepared['rank_desc'], dtype=np.int64)[:kept]] = True
    return mask.reshape(tuple(prepared['shape'])[1:])


def exact_tile_bits(num_rx: int, support_cells: int, kept_cells: int,
                    q: int) -> dict[str, Any]:
    """Exact analytical bytes for the declared shared-support tile syntax.

    Values: signed q-bit I and Q, receiver-major, for every retained cell.
    Support: whichever is smaller, a packed 2D binary mask or packed fixed-width
    flattened cell indices. Both sections are independently byte-aligned.
    Header: 16 bytes (float64 common scale, uint32 kept count, uint8 q, uint8
    support mode, uint16 flags). Tile position/shape is implicit in scan order.
    Even full support uses the ordinary support syntax; no hidden exception.
    """
    _, q = _validated_action((1.0, q))
    num_rx, support_cells, kept_cells = map(int, (num_rx, support_cells, kept_cells))
    if num_rx < 1 or support_cells < 1 or not 0 <= kept_cells <= support_cells:
        raise ValueError('Invalid tile dimensions or retained support count.')
    align8 = lambda n: 8 * ((int(n) + 7) // 8)
    payload_raw = 2 * q * num_rx * kept_cells
    mask_bits = align8(support_cells)
    index_width = max(1, (support_cells - 1).bit_length())
    index_bits = align8(kept_cells * index_width)
    support_bits = min(mask_bits, index_bits)
    payload_bits = align8(payload_raw)
    header_bits = 8 * BLOCK_HEADER_BYTES
    total = payload_bits + support_bits + header_bits
    return dict(payload_raw_bits=float(payload_raw), payload_coded_bits=float(payload_bits),
                support_bits=float(support_bits), header_bits=float(header_bits),
                support_mode_mask_bits=float(mask_bits), support_mode_index_bits=float(index_bits),
                support_mode='mask' if mask_bits <= index_bits else 'indices',
                support_index_width=index_width, shared_support_cells=support_cells,
                kept_support_cells=kept_cells, kept_complex_coefficients=num_rx * kept_cells,
                total_coeffs=num_rx * support_cells, q=q, realized_bits=float(total),
                packed_bits=float(total), realized_bpcs=float(total / (num_rx * support_cells)),
                packed_bpcs=float(total / (num_rx * support_cells)),
                packed_rate_is_exact_for_declared_syntax=True,
                actual_codec_bitstream_saved=False, codec_size_is_analytical=True)


def _quantize_complex(values: np.ndarray, q: int) -> tuple[np.ndarray, float]:
    """Symmetric scalar I/Q quantizer with one scale shared by the whole tile."""
    _, q = _validated_action((1.0, q))
    values = np.asarray(values)
    scale = float(max(np.max(np.abs(values.real), initial=0.0),
                      np.max(np.abs(values.imag), initial=0.0)))
    if scale == 0.0:
        return np.zeros_like(values), scale
    levels = 2 ** (q - 1) - 1
    iq = np.rint(values.real / scale * levels) + 1j * np.rint(values.imag / scale * levels)
    return (iq * (scale / levels)).astype(values.dtype, copy=False), scale


def materialize_action(prepared: dict[str, Any], rho: float,
                       q: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the quantized GLOBAL-spectrum tile and its analytical bit count."""
    rho, q = _validated_action((rho, q))
    mask = support_mask(prepared, rho)
    spectrum = np.asarray(prepared['Z'])
    values_hat, scale = _quantize_complex(spectrum[:, mask], q)
    tile_hat = np.zeros_like(spectrum)
    tile_hat[:, mask] = values_hat
    bits = exact_tile_bits(spectrum.shape[0], mask.size, int(mask.sum()), q)
    bits.update(rho=rho, scale=scale,
                distortion_energy=float(np.sum(np.abs(spectrum - tile_hat) ** 2,
                                               dtype=np.float64)))
    return tile_hat, bits


def _prepare_tile(tile: np.ndarray, cube_energy: float) -> dict[str, Any]:
    power = np.sum(np.abs(tile) ** 2, axis=0, dtype=np.float64).reshape(-1)
    rank = np.argsort(-power, kind='stable')
    energy = float(power.sum())
    denominator = max(energy, EPS)
    probability = power / denominator
    positive = probability > 0.0
    entropy = -float(np.sum(probability[positive] * np.log(probability[positive])))

    def top_fraction(fraction: float) -> float:
        k = max(1, min(power.size, int(math.ceil(fraction * power.size))))
        return float(power[rank[:k]].sum() / denominator)

    return dict(Z=tile, rank_desc=rank, shape=tuple(tile.shape),
                total_coeffs=int(tile.size), support_cells=int(power.size),
                block_mean_power=float(energy / tile.size), tile_energy=energy,
                whole_cube_energy=cube_energy,
                spectral_top1_fraction=top_fraction(0.01),
                spectral_top5_fraction=top_fraction(0.05),
                spectral_top10_fraction=top_fraction(0.10),
                spectral_entropy_normalized=float(entropy / max(math.log(max(power.size, 2)), EPS)),
                spectral_peak_to_mean_log=float(np.log1p(float(power.max(initial=0.0)) /
                                                        max(float(power.mean()), EPS))),
                mask_cache={})


def prepare_source(cube: np.ndarray, actions: Sequence[Sequence[float]] = DEFAULT_ACTIONS,
                   block_slow: int = 32, block_fast: int = 64) -> dict[str, Any]:
    """Cache global FFT, features, and EXACT per-action Parseval distortions.

    Distortion contributions use the whole cube's energy as denominator. Their
    sum equals global normalized squared error before the output dtype cast.
    The supplied complete action library is evaluated; no pruning is performed.
    No labels, CFAR maps, validation cubes, or learned policy are consulted.

    The 96-byte cube header declares magic/version, transform normalization and
    axis/shift flags, endian/component precision, dimensions, tile dimensions,
    tile count, and optional radar axis calibration metadata/reserved fields.
    Block scales/counts/q/support modes are charged in the separate tile headers.
    """
    cube = np.asarray(cube)
    if cube.ndim != 3 or not np.issubdtype(cube.dtype, np.complexfloating):
        raise ValueError('A cube must be a 3D complex IQ array [Rx, chirp, fast-time].')
    if min(cube.shape) < 1 or not np.isfinite(cube).all():
        raise ValueError('Cube dimensions must be nonzero and samples finite.')
    actions = tuple(_validated_action(a) for a in actions)
    if not actions:
        raise ValueError('The action library must not be empty.')
    spectrum = np.fft.fftshift(np.fft.fft2(cube, axes=(1, 2), norm='ortho'), axes=(1, 2))
    cube_energy = float(np.sum(np.abs(spectrum) ** 2, dtype=np.float64))
    slices = list(iter_spectral_tiles(cube.shape, block_slow, block_fast))
    prepared = [_prepare_tile(spectrum[slc], cube_energy) for slc in slices]
    costs = np.empty((len(slices), len(actions)), dtype=np.float64)
    distortion = np.empty_like(costs)
    for b, tile in enumerate(prepared):
        for a, (rho, q) in enumerate(actions):
            _, bits = materialize_action(tile, rho, q)
            costs[b, a] = bits['realized_bits']
            distortion[b, a] = bits['distortion_energy'] / max(cube_energy, EPS)
        tile['action_bits'] = costs[b].copy()
        tile['action_distortion_contributions'] = distortion[b].copy()
    concentration = np.asarray([p['spectral_top10_fraction'] for p in prepared], dtype=float)
    mean_power = float(cube_energy / cube.size)
    return dict(cube_shape=tuple(cube.shape), output_dtype=cube.dtype, spectrum=spectrum,
                transform='global_shifted_unitary_RD_fft2_no_receiver_fft',
                tile_slices=slices, slices=slices, prepared_blocks=prepared, actions=actions,
                whole_cube_energy=cube_energy, cube_mean_power=mean_power,
                cube_mean_spectral_concentration=float(concentration.mean()),
                cube_block_log_energy_std=float(np.std([
                    np.log((p['block_mean_power'] + EPS) / (mean_power + EPS)) for p in prepared])),
                cube_spectral_concentration_std=float(concentration.std()),
                action_bits_by_block=costs, action_distortion_by_block=distortion,
                cube_header_bits=float(8 * CUBE_HEADER_BYTES),
                block_slow=int(block_slow), block_fast=int(block_fast),
                actual_codec_bitstream_saved=False, codec_size_is_analytical=True)


def reconstruct(source: dict[str, Any], spectral_hat: np.ndarray) -> np.ndarray:
    """One global inverse RD transform; there is no local inverse/reassembly."""
    spectral_hat = np.asarray(spectral_hat)
    if tuple(spectral_hat.shape) != tuple(source['cube_shape']) or not np.isfinite(spectral_hat).all():
        raise ValueError('Reconstruction spectrum must be finite and match the source shape.')
    unshifted = np.fft.ifftshift(spectral_hat, axes=(1, 2))
    return np.fft.ifft2(unshifted, axes=(1, 2), norm='ortho').astype(source['output_dtype'], copy=False)


def encode_sequence(cube: np.ndarray, sequence: Sequence[int],
                    actions: Sequence[Sequence[float]] = DEFAULT_ACTIONS,
                    block_slow: int = 32, block_fast: int = 64,
                    source: dict[str, Any] | None = None) -> dict[str, Any]:
    """Encode a supplied action allocation; never select candidates/refine it."""
    cube = np.asarray(cube)
    source = prepare_source(cube, actions, block_slow, block_fast) if source is None else source
    if tuple(cube.shape) != tuple(source['cube_shape']):
        raise ValueError('Cached source shape does not match cube.')
    sequence = np.asarray(sequence)
    if sequence.ndim != 1 or len(sequence) != len(source['tile_slices']):
        raise ValueError('The sequence must contain one integer action per spectral tile.')
    if not np.issubdtype(sequence.dtype, np.integer):
        raise ValueError('Action indices must be integers.')
    if np.any(sequence < 0) or np.any(sequence >= len(source['actions'])):
        raise ValueError('An action index lies outside the supplied library.')
    spectral_hat = np.zeros_like(source['spectrum'])
    records = []
    total_bits = float(source['cube_header_bits'])
    for b, (slc, tile, action_index) in enumerate(zip(source['tile_slices'],
                                                   source['prepared_blocks'], sequence)):
        rho, q = source['actions'][int(action_index)]
        tile_hat, bits = materialize_action(tile, rho, q)
        spectral_hat[slc] = tile_hat
        total_bits += float(bits['realized_bits'])
        records.append(dict(block_idx=b, action_index=int(action_index), **bits))
    cube_hat = reconstruct(source, spectral_hat)
    denominator = max(float(source['whole_cube_energy']), EPS)
    nmse = float(np.sum(np.abs(cube.astype(np.complex128) - cube_hat) ** 2,
                        dtype=np.float64) / denominator)
    meta = dict(realized_bits=total_bits, realized_bpcs=total_bits / cube.size,
                cube_header_bits=float(source['cube_header_bits']),
                payload_bits=float(sum(r['payload_coded_bits'] for r in records)),
                support_bits=float(sum(r['support_bits'] for r in records)),
                header_bits=float(source['cube_header_bits'] + sum(r['header_bits'] for r in records)),
                cube_nmse=nmse, rel_error=math.sqrt(max(nmse, 0.0)),
                spectral_nmse=float(np.sum(np.abs(source['spectrum'] - spectral_hat) ** 2,
                                           dtype=np.float64) / denominator),
                compression_ratio_vs_v43_int16_iq_32bpc=32.0 / (total_bits / cube.size),
                memory_ratio_proxy_64b_complex=64.0 / (total_bits / cube.size),
                transform=source['transform'], receiver_axis_transformed=False,
                shared_antenna_support=True, phase_exact_after_quantization=False,
                raw_ppo_only=True, deterministic_refinement=False,
                actual_codec_bitstream_saved=False, codec_size_is_analytical=True,
                rate_accounting='byte-aligned values + shared 2D support + 16-byte tile headers + 96-byte cube header')
    return dict(cube_hat=cube_hat, spectral_hat=spectral_hat,
                block_log=records, blocks=records, meta=meta)


def uniform_encode(cube: np.ndarray, action: Sequence[float],
                   block_slow: int = 32, block_fast: int = 64) -> dict[str, Any]:
    """Deterministic uniform baseline using the identical global-transform codec."""
    source = prepare_source(cube, (action,), block_slow, block_fast)
    result = encode_sequence(cube, np.zeros(len(source['tile_slices']), dtype=np.int64),
                             source=source)
    result['meta']['method'] = 'uniform_global_RD_shared_support'
    return result
