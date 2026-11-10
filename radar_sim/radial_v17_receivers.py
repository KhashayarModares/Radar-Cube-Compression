"""A second CFAR receiver, used only for evaluation (never inside the closed loop).

Ordered-statistic CFAR (Rohling 1983): the noise level of a cell is the k-th
smallest power of its training cells instead of their mean, which makes it robust
to interfering targets and clutter edges. Same RD power map, 11x11 window with a
3x3 guard (112 training cells), cyclic Doppler padding and 5x5 power NMS as the
RADIal CA-CFAR; k = 84 = 3/4 of 112, the order Rohling recommends. Its threshold
is calibrated separately on noise-only cubes, like the CA-CFAR's.

The codec's closed loop verifies against CA-CFAR only, so agreement under OS-CFAR
shows whether the guarantee generalizes to a receiver it was not tuned to.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .arm_e_adaptive_gpu_physics import EPS, ca_cfar

ORDER = 84
_MASK = torch.ones(11, 11, dtype=torch.bool)
_MASK[4:7, 4:7] = False


def os_cfar(power, order=ORDER):
    detector = ca_cfar(power)      # identical validity mask and NMS peaks
    windows = F.unfold(F.pad(power[None, None], (5, 5, 5, 5)), 11)[0]
    training = windows[_MASK.reshape(-1).to(power.device)]
    noise = training.kthvalue(order, dim=0).values.reshape(power.shape)
    return dict(detector, noise=noise, score=power/(noise+EPS))


def cyclic_os_cfar(power):
    """RADIal wrapper: wrap six Doppler columns, run OS-CFAR, crop (as cyclic_cfar)."""
    padded = F.pad(power[None, None], (6, 6, 0, 0), mode='circular')[0, 0]
    return {name: value[:, 6:-6] for name, value in os_cfar(padded).items()}
