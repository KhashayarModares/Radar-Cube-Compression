"""Atomic, pickle-free experiment storage and explicit resumable optimizer state."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np
import pandas as pd


def serial(value):
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    if isinstance(value, (Path, np.generic)):
        return serial(value.item()) if isinstance(value, np.generic) else str(value)
    if isinstance(value, np.ndarray):
        return serial(value.tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def digest(value):
    return hashlib.sha256(json.dumps(serial(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            result.update(chunk)
    return result.hexdigest()


def _replace(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('wb') as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json(path, value):
    payload = (json.dumps(serial(value), indent=2, allow_nan=False) + '\n').encode()
    _replace(path, lambda stream: stream.write(payload))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_csv(path, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    _replace(path, lambda stream: stream.write(frame.to_csv(index=False, lineterminator='\n').encode()))


def save_tree(path, tree):
    """NPZ arrays plus a JSON structure; never unpickle untrusted objects."""
    arrays = {}

    def pack(value):
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise TypeError('Object arrays are prohibited in checkpoints.')
            key = 'array_' + str(len(arrays))
            arrays[key] = value
            return {'__array__': key}
        if isinstance(value, np.generic):
            return pack(value.item())
        if isinstance(value, dict):
            return {'__dict__': [[pack(k), pack(v)] for k, v in value.items()]}
        if isinstance(value, tuple):
            return {'__tuple__': [pack(v) for v in value]}
        if isinstance(value, list):
            return [pack(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value):
            return {'__float__': str(value)}
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError('Unsupported checkpoint type: ' + str(type(value)))

    arrays['structure'] = np.frombuffer(json.dumps(pack(tree), allow_nan=False).encode(), dtype=np.uint8)
    _replace(path, lambda stream: np.savez_compressed(stream, **arrays))


def load_tree(path):
    with np.load(path, allow_pickle=False) as archive:
        def unpack(value):
            if isinstance(value, list):
                return [unpack(v) for v in value]
            if isinstance(value, dict):
                if '__array__' in value:
                    return archive[value['__array__']].copy()
                if '__dict__' in value:
                    return {unpack(k): unpack(v) for k, v in value['__dict__']}
                if '__tuple__' in value:
                    return tuple(unpack(v) for v in value['__tuple__'])
                if '__float__' in value:
                    return float(value['__float__'])
                raise ValueError('Unknown checkpoint structure.')
            return value
        return unpack(json.loads(archive['structure'].tobytes().decode()))


def optimizer_state(optimizer):
    return {name: dict(t=part.t, learning_rate=part.learning_rate,
                      m={k: v.copy() for k, v in part.m.items()},
                      v={k: v.copy() for k, v in part.v.items()})
            for name, part in (('actor', optimizer.actor), ('critic', optimizer.critic))}


def restore_optimizer(optimizer, saved):
    """Validate the entire split-Adam state before changing either optimizer.

    A rejected/corrupted critic checkpoint must not partially restore actor
    moments or its cursor. Counters are exact nonnegative integers; second
    moments cannot be negative. Parameter identity, shape, dtype and the fixed
    learning rate are part of the resumability contract.
    """
    if not isinstance(saved, dict) or set(saved) != {'actor', 'critic'}:
        raise ValueError('Expected complete actor and critic optimizer state.')
    validated = {}
    for name in ('actor', 'critic'):
        part, state = getattr(optimizer, name), saved[name]
        if not isinstance(state, dict) or set(state) != {'t', 'learning_rate', 'm', 'v'}:
            raise ValueError('Incomplete optimizer state: '+name)
        cursor = state['t']
        if (isinstance(cursor, (bool, np.bool_)) or not isinstance(cursor, (int, np.integer))
                or cursor < 0):
            raise ValueError('Optimizer cursor must be a nonnegative integer: '+name)
        if not isinstance(state['m'], dict) or not isinstance(state['v'], dict):
            raise ValueError('Optimizer moments must be parameter dictionaries: '+name)
        if set(state['m']) != set(part.m) or set(state['v']) != set(part.v):
            raise ValueError('Optimizer parameter identities changed.')
        try:
            rate = float(state['learning_rate'])
        except (TypeError, ValueError) as exc:
            raise ValueError('Invalid optimizer learning rate: '+name) from exc
        if not np.isfinite(rate) or rate != part.learning_rate:
            raise ValueError('Optimizer learning rate changed on resume.')
        validated[name] = dict(t=int(cursor), m={}, v={})
        for kind in ('m', 'v'):
            for key, value in state[kind].items():
                destination = getattr(part, kind)[key]
                if (not isinstance(value, np.ndarray) or value.shape != destination.shape
                        or value.dtype != destination.dtype or not np.isfinite(value).all()
                        or (kind == 'v' and np.any(value < 0))):
                    raise ValueError('Invalid optimizer moment: ' + key)
                validated[name][kind][key] = value.copy()
    if validated['actor']['t'] != validated['critic']['t']:
        raise ValueError('Actor and critic optimizer cursors differ.')
    # Commit only after both halves and all arrays passed validation.
    for name in ('actor', 'critic'):
        part, state = getattr(optimizer, name), validated[name]
        part.t = state['t']
        for kind in ('m', 'v'):
            for key, value in state[kind].items():
                getattr(part, kind)[key][...] = value


def actor_hash(policy):
    return digest({k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
                   for k, v in sorted(policy.params.items())})


def receipt(path, artifacts, **metadata):
    """Written LAST; a receipt authenticates only fully committed artifacts."""
    path = Path(path)
    files = {os.path.relpath(Path(p), path.parent): sha(p) for p in artifacts}
    write_json(path, dict(completed=True, artifacts=files, **metadata))


def verify_receipt(path):
    path = Path(path)
    values = read_json(path)
    if not values.get('completed'):
        raise ValueError('Incomplete receipt: ' + str(path))
    for name, expected in values['artifacts'].items():
        if sha(path.parent / name) != expected:
            raise ValueError('Committed artifact changed: ' + str(path.parent / name))
    return values
