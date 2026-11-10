"""Place the third-party detector files that this repository does not redistribute.

The experiments score decoded RADIal cubes with FFTRadNet (Rebut et al., CVPR 2022). Its model code, its config and its
trained weights come from the authors' repository (https://github.com/valeoai/RADIal), which has no license file, so they
are not committed here. The official-test run also scores AdaRadar's retrained 4-bit FFTRadNet (https://github.com/jp4327/adaradar).
This script verifies the files you downloaded, converts each checkpoint into the weights-only state dict the code loads, and
copies everything into every package folder listed in scripts/third_party_slots.json.

Usage (from the repository root):
    python scripts/setup_third_party.py --radial-repo <clone of valeoai/RADIal> \
        --fftradnet-checkpoint <FFTRadNet_RA_192_56_epoch78_loss_172.8239_AP_0.9813.pth> \
        [--adaradar-checkpoint <FFTRadNet_pretrained_4bit.pth>]

Checkpoints are opened with torch.load(weights_only=True) and a short allowlist of numpy types; no pickled code is run.
The tensors written are checked against a hash of the exact tensors used in the paper runs.
"""
from __future__ import annotations

import argparse
import codecs
import collections
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SLOTS = ROOT/'scripts'/'third_party_slots.json'

EXPECTED = {
    # files of the RADIal repository, as used in every run
    'FFTRadNet.py': 'be8dcaee5924d3122cbeea73981baaf6881292993ae5e3289d1a2687d4d4839c',
    'config_FFTRadNet_192_56.json': 'a710a9af2b6c948655f07d94d97cd9e164fc1b02466139021384890b854f3089',
    # checkpoint files as downloaded from the authors' links
    'fftradnet_checkpoint': 'd9545e7383b29193243ef08b245f633b64b9c24d7097ccc96a3d8a98f76dcdd9',
    'adaradar_checkpoint': '080ee74cdfb6ee0906daf845320cc671c51c1e14b510bb83a5249f8d5e1ee256',
    # hash over (name, dtype, shape, bytes) of every tensor, sorted by name: what the code actually loads
    'fftradnet_state_dict.pt': 'bcc0c06628ee4d8628e73904214935a2f12a079812512d77f7228c5000f9a153',
    'adaradar_4bit_state_dict.pt': '91871524c98e0d6db9ec2c1ebb8346eed1696cbf978c8a7c51e128ef2c45aabe',
}


def file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(state: dict) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        t = state[key].detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def safe_load(path: Path) -> dict:
    """weights_only load; the checkpoints hold numpy scalars from their training history, so those types are allowed."""
    scalar = np._core.multiarray.scalar if hasattr(np, '_core') else np.core.multiarray.scalar
    allowed = [scalar, (scalar, 'numpy.core.multiarray.scalar'), np.dtype, codecs.encode, collections.OrderedDict,
               type(np.dtype('float64')), type(np.dtype('float32')), type(np.dtype('int64'))]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location='cpu', weights_only=True)


def check(name: str, actual: str, force: bool):
    if actual == EXPECTED[name]:
        print(f'  ok   {name}')
        return
    message = f'  hash of {name} differs from the one used in the paper: {actual}'
    if not force:
        raise SystemExit(message + '\n  Use the exact file from the authors\' link, or pass --force to continue anyway.')
    print(message + ' (continuing because of --force)')


def official_model_keys(model_py: Path, config_json: Path) -> list[str]:
    """Parameter names of the official FFTRadNet class, in definition order."""
    try:
        import torchvision  # noqa: F401
    except ImportError:  # FFTRadNet.py imports torchvision without using it
        root, transforms, inner = (types.ModuleType(n) for n in ('torchvision', 'torchvision.transforms',
                                                                  'torchvision.transforms.transforms'))
        inner.Sequence = object
        root.transforms, transforms.transforms = transforms, inner
        sys.modules.update({'torchvision': root, 'torchvision.transforms': transforms,
                            'torchvision.transforms.transforms': inner})
    module = types.ModuleType('fftradnet_official')
    exec(compile(model_py.read_text(encoding='utf-8'), str(model_py), 'exec'), module.__dict__)
    m = json.loads(config_json.read_text(encoding='utf-8'))['model']
    net = module.FFTRadNet(blocks=m['backbone_block'], mimo_layer=m['MIMO_output'], channels=m['channels'],
                           regression_layer=2, detection_head=m['DetectionHead'], segmentation_head=m['SegmentationHead'])
    return list(net.state_dict().keys())


# The released checkpoint stores its tensors under older module names than the released FFTRadNet.py.
RENAME = (('backbone.preproc.', 'FPN.pre_enc.'), ('backbone.', 'FPN.'), ('RAmap_header.', 'RA_decoder.'))


def fftradnet_state(checkpoint: Path, keys: list[str]) -> dict:
    """Weights-only state dict of the official checkpoint, renamed to the module names of the official class.
    Tensors are not changed; the renamed names must equal the class's parameter names exactly."""
    def rename(key):
        for old, new in RENAME:
            if key.startswith(old):
                return new + key[len(old):]
        return key
    state = collections.OrderedDict((rename(k), v) for k, v in safe_load(checkpoint)['net_state_dict'].items())
    if set(state) != set(keys):
        raise SystemExit(f'  renamed checkpoint does not match the FFTRadNet class: {sorted(set(state) ^ set(keys))[:5]}')
    return state


def place(name: str, payload_path: Path, slots: list[dict]) -> int:
    count = 0
    for slot in slots:
        if slot['file'] != name:
            continue
        dst = ROOT/slot['destination']
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(payload_path, dst)
        count += 1
    return count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--radial-repo', type=Path, required=True, help='clone of https://github.com/valeoai/RADIal')
    parser.add_argument('--fftradnet-checkpoint', type=Path, required=True,
                        help='FFTRadNet_RA_192_56_epoch78_loss_172.8239_AP_0.9813.pth from the RADIal README link')
    parser.add_argument('--adaradar-checkpoint', type=Path,
                        help='FFTRadNet_pretrained_4bit.pth from the AdaRadar README link (official-test run only)')
    parser.add_argument('--force', action='store_true', help='continue when a downloaded file has a different hash')
    args = parser.parse_args(argv)

    slots = json.loads(SLOTS.read_text(encoding='utf-8'))
    model_py = args.radial_repo/'FFTRadNet'/'model'/'FFTRadNet.py'
    config_json = args.radial_repo/'FFTRadNet'/'config'/'config_FFTRadNet_192_56.json'
    print('Checking downloaded files')
    check('FFTRadNet.py', file_sha(model_py), args.force)
    check('config_FFTRadNet_192_56.json', file_sha(config_json), args.force)
    check('fftradnet_checkpoint', file_sha(args.fftradnet_checkpoint), args.force)
    if args.adaradar_checkpoint:
        check('adaradar_checkpoint', file_sha(args.adaradar_checkpoint), args.force)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        print('Converting checkpoints to weights-only state dicts')
        state = fftradnet_state(args.fftradnet_checkpoint, official_model_keys(model_py, config_json))
        check('fftradnet_state_dict.pt', tensor_sha(state), args.force)
        torch.save(state, tmp/'fftradnet_state_dict.pt')
        if args.adaradar_checkpoint:
            ada = safe_load(args.adaradar_checkpoint)['net_state_dict']
            check('adaradar_4bit_state_dict.pt', tensor_sha(ada), args.force)
            torch.save(ada, tmp/'adaradar_4bit_state_dict.pt')

        print('Placing files')
        placed = dict(model=place('FFTRadNet.py', model_py, slots), config=place('config_FFTRadNet_192_56.json', config_json, slots),
                      fftradnet=place('fftradnet_state_dict.pt', tmp/'fftradnet_state_dict.pt', slots))
        if args.adaradar_checkpoint:
            placed['adaradar'] = place('adaradar_4bit_state_dict.pt', tmp/'adaradar_4bit_state_dict.pt', slots)
    print('  ' + ', '.join(f'{k}: {v} folders' for k, v in placed.items()))
    missing = [s['destination'] for s in slots if not (ROOT/s['destination']).exists()]
    if missing:
        print('Not placed (the AdaRadar checkpoint is needed only by the official-test run):')
        for m in missing:
            print('  ' + m)
    print('Done.')


if __name__ == '__main__':
    main()
