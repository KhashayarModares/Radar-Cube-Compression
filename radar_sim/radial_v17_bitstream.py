"""A real file for one closed-loop ECDQ cube, decoded from the file alone.

The rates reported in the study are ideal adaptive code lengths (per-tile
Krichevsky-Trofimov models, Krichevsky & Trofimov 1981). This module shows that
an actual arithmetic coder (Witten, Neal & Cleary 1987) achieves them: the
encoder writes everything the decoder needs into one file; the decoder rebuilds
the stream from the file bytes only and runs the unchanged ``decode_ecdq``. The
round trip must reproduce the encoder's decoded cube bit for bit.

File layout (little endian):
  header   magic, version, Rx, Doppler, range, beta (float64), seed, variance
           block, flags (bit 0 synthesized background, bit 1 refinement levels)
  support  [synthesis only] DEFLATE(packbits(support map))       (RFC 1951)
  levels   [refinement only] DEFLATE(uint8 refinement-level map)
  sigma    16-bit noise code of every active tile (tile with a coded component)
  stats    [synthesis only] 12-bit block variance codes, 8-bit per-Rx gain codes
  table    per active tile: u32 arithmetic-section bits, u32 total bits
  payload  the tile bitstreams back to back (classes arithmetic coded, mantissa
           and sign bits raw), zero-padded to a whole byte
The support and level maps use DEFLATE instead of the KT code assumed by the
rate accounting, and the table costs 64 bits per tile, so the file is slightly
larger than the ideal length; the difference is reported, not hidden.
"""
from __future__ import annotations

import struct
import time
import zlib

import numpy as np
import torch

from . import radial_v13_detection_codec as codec

MAGIC, VERSION = b'RDV17', 1
_HEADER = struct.Struct('<5sBBHHdIBBB')


def _with_length(data):
    return struct.pack('<I', len(data))+data


def _pack_unsigned(values, width):
    bits = (values[:, None] >> np.arange(width-1, -1, -1)) & 1
    return np.packbits(bits.astype(np.uint8).ravel()).tobytes()


def _unpack_unsigned(data, width, count):
    bits = np.unpackbits(np.frombuffer(data, np.uint8))[:width*count].reshape(count, width).astype(np.int64)
    return (bits << np.arange(width-1, -1, -1)).sum(1)


def _active_tiles(support, shape):
    keep = codec.cells_to_tiles(support, shape).cpu()
    return keep, torch.nonzero(keep.any(1))[:, 0].tolist()


def encode_file(stream):
    """Serialize an ECDQ stream; returns (file bytes, byte count of each part)."""
    rx, doppler, rng = stream['shape']
    keep, active = _active_tiles(stream['support'], stream['shape'])
    levels = stream['levels']
    flags = int(stream['synthesis']) | int(levels is not None) << 1
    parts = dict(header=_HEADER.pack(MAGIC, VERSION, rx, doppler, rng, float(stream['beta']), int(stream['seed']),
                                     *stream['block'], flags))
    if stream['synthesis']:
        parts['support'] = _with_length(zlib.compress(np.packbits(stream['support'].cpu().numpy().ravel()).tobytes(), 9))
    if levels is not None:
        parts['levels'] = _with_length(zlib.compress(levels.cpu().numpy().astype(np.uint8).tobytes(), 9))
    parts['sigma'] = stream['sigma_code'].cpu()[active].numpy().astype('<i2').tobytes()
    if stream['synthesis']:
        parts['stats'] = (_pack_unsigned(stream['variance_code'].cpu().numpy().ravel()+2048, codec.VAR_BITS)
                          + stream['gain_code'].cpu().numpy().astype(np.int8).tobytes())
    index, table, payload = stream['index'].cpu(), [], []
    for tile in active:
        bits, section = codec.write_arithmetic_tile(index[tile][keep[tile]].tolist(), with_section=True)
        table.append((section, len(bits)))
        payload.extend(bits)
    parts['table'] = np.asarray(table, dtype='<u4').tobytes()
    parts['payload'] = np.packbits(np.asarray(payload, dtype=np.uint8)).tobytes()
    return b''.join(parts.values()), {name: len(value) for name, value in parts.items()}


def decode_file(data, device):
    """Rebuild the ECDQ stream from the file bytes only."""
    magic, version, rx, doppler, rng, beta, seed, bd, br, flags = _HEADER.unpack_from(data)
    if magic != MAGIC or version != VERSION:
        raise ValueError('Not a V17 ECDQ file.')
    at = _HEADER.size

    def take(count):
        nonlocal at
        at += count
        return data[at-count:at]

    def deflated():
        return zlib.decompress(take(struct.unpack('<I', take(4))[0]))

    shape, synthesis, has_levels = (rx, doppler, rng), bool(flags & 1), bool(flags & 2)
    support = np.ones((doppler, rng), dtype=bool)
    if synthesis:
        support = np.unpackbits(np.frombuffer(deflated(), np.uint8))[:doppler*rng].reshape(doppler, rng).astype(bool)
    support = torch.from_numpy(support).to(device)
    levels = None
    if has_levels:
        levels = torch.from_numpy(np.frombuffer(deflated(), np.uint8).reshape(doppler, rng).astype(np.int64)).to(device)
    keep, active = _active_tiles(support, shape)
    sigma = torch.zeros(len(keep), dtype=torch.int64)
    sigma[active] = torch.from_numpy(np.frombuffer(take(2*len(active)), '<i2').astype(np.int64))
    stream = dict(mode='ecdq', shape=shape, sigma_code=sigma.to(device), beta=beta, seed=seed, block=(bd, br),
                  support=support, gamma_db=None, synthesis=synthesis, levels=levels)
    if synthesis:
        count = (doppler//bd)*(rng//br)
        variance = _unpack_unsigned(take(-(-codec.VAR_BITS*count//8)), codec.VAR_BITS, count)-2048
        stream['variance_code'] = torch.from_numpy(variance.reshape(doppler//bd, rng//br)).to(device)
        stream['gain_code'] = torch.from_numpy(np.frombuffer(take(rx), np.int8).astype(np.int64)).to(device)
    table = np.frombuffer(take(8*len(active)), '<u4').reshape(-1, 2).tolist()
    payload = np.unpackbits(np.frombuffer(data[at:], np.uint8)).tolist()
    if not 0 <= len(payload)-sum(total for _, total in table) < 8:
        raise ValueError('Payload length does not match the tile table.')
    index, position = torch.zeros(keep.shape, dtype=torch.int64), 0
    for tile, (section, total) in zip(active, table):
        symbols = codec.read_arithmetic_tile(payload[position:position+total], int(keep[tile].sum()), section)
        index[tile][keep[tile]] = torch.tensor(symbols, dtype=torch.int64)
        position += total
    stream['index'] = index.to(device)
    return stream


def roundtrip(stream, path, dtype=torch.complex64):
    """Write ``stream`` to ``path``, decode the cube from the file, compare bit for bit."""
    device = stream['support'].device
    start = time.perf_counter()
    data, parts = encode_file(stream)
    path.write_bytes(data)
    write_s = time.perf_counter()-start
    start = time.perf_counter()
    restored = decode_file(path.read_bytes(), device)
    read_s = time.perf_counter()-start
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    decoded = codec.decode_ecdq(restored, dtype)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    decode_ms = 1e3*(time.perf_counter()-start)
    numel = int(np.prod(stream['shape']))
    ideal = codec.ecdq_bits(stream)
    return dict(file=path.name, file_bytes=len(data), file_bpcs=8*len(data)/numel, ideal_kt_bpcs=ideal['total']/numel,
                file_over_ideal=8*len(data)/ideal['total']-1, payload_file_bits=8*parts['payload'],
                payload_ideal_bits=ideal['payload'], **{f'{name}_bytes': size for name, size in parts.items()},
                bit_exact=bool(torch.equal(decoded, codec.decode_ecdq(stream, dtype))),
                cpu_entropy_encode_s=write_s, cpu_entropy_decode_s=read_s, gpu_reconstruct_from_file_ms=decode_ms)


def same_stream(a, b):
    """True if two streams agree in every field ``decode_ecdq`` reads; decode_ecdq is a
    deterministic function of these fields, so equal fields give the identical cube."""
    keep = codec.cells_to_tiles(a['support'].cpu(), a['shape']).any(1)
    tensors = lambda s: [s['support'].cpu(), s['index'].cpu().long(), s['sigma_code'].cpu()[keep]] + (
        [s['variance_code'].cpu(), s['gain_code'].cpu()] if s['synthesis'] else [])
    if (tuple(a['shape']), float(a['beta']), int(a['seed']), tuple(a['block']), bool(a['synthesis'])) != \
            (tuple(b['shape']), float(b['beta']), int(b['seed']), tuple(b['block']), bool(b['synthesis'])):
        return False
    if (a['levels'] is None) != (b['levels'] is None) or (
            a['levels'] is not None and not torch.equal(a['levels'].cpu(), b['levels'].cpu())):
        return False
    return all(torch.equal(x, y) for x, y in zip(tensors(a), tensors(b)))


def worker_init():
    torch.set_num_threads(1)


def file_check(stream, path, keep_file=False):
    """CPU worker: write the file, rebuild the stream from the file bytes alone and
    compare it with the encoder's stream (``same_stream``)."""
    start = time.perf_counter()
    data, parts = encode_file(stream)
    path.write_bytes(data)
    write_s = time.perf_counter()-start
    start = time.perf_counter()
    restored = decode_file(path.read_bytes(), torch.device('cpu'))
    read_s = time.perf_counter()-start
    if not keep_file:
        path.unlink()
    numel = int(np.prod(stream['shape']))
    return dict(file=path.name, file_kept=keep_file, file_bytes=len(data), file_bpcs=8*len(data)/numel,
                payload_file_bits=8*parts['payload'], **{f'{name}_bytes': size for name, size in parts.items()},
                stream_identical=same_stream(stream, restored), cpu_entropy_encode_s=write_s, cpu_entropy_decode_s=read_s)
