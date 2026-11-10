"""Load the trained PPO policies and print what they contain.

Run from the repository root:   python -B examples/load_trained_models.py
Needs only PyTorch and this repository; no dataset.
"""
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from radar_sim.radial_preserve_gpu_learning import DeltaPolicy  # noqa: E402
from radar_sim.radial_v14_entropy_model import ContextModel  # noqa: E402

cpu = torch.device('cpu')
for dataset, checkpoint in (('radial', 'arm_0_update_0336.pt'), ('uwcr', 'arm_1_update_0360.pt')):
    ckpt = torch.load(ROOT/'weights'/dataset/checkpoint, map_location=cpu, weights_only=True)
    normalizer = torch.load(ROOT/'weights'/dataset/'feature_normalizer.pt', map_location=cpu, weights_only=True)
    policy = DeltaPolicy(len(ckpt['feature_names']), cpu)
    policy.load_state_dict(ckpt['actor'])
    policy.eval()
    assert ckpt['feature_names'] == normalizer['names']
    # A standardized feature vector of zeros is the average training tile.
    x = torch.zeros(1, len(ckpt['feature_names']), dtype=torch.float64)
    with torch.no_grad():
        probabilities = torch.softmax(policy.actor(x), -1)
    print(f'{dataset}: {checkpoint}, arm {ckpt["arm"]}, update {ckpt["update"]}, learning rate {ckpt["learning_rate"]}')
    print(f'  {len(ckpt["feature_names"])} features: {", ".join(ckpt["feature_names"])}')
    print(f'  actor {sum(p.numel() for p in policy.actor.parameters())} parameters, '
          f'critic {sum(p.numel() for p in policy.critic.parameters())} parameters, 21 actions')
    print(f'  most likely action at the average tile: {int(probabilities.argmax())} (p = {float(probabilities.max()):.2f})')

model = ContextModel()
model.load_state_dict(torch.load(ROOT/'weights'/'radial'/'entropy_model_v20.pt', map_location=cpu, weights_only=True))
print(f'Learned ECDQ context model (baseline): {sum(p.numel() for p in model.parameters())} parameters')
