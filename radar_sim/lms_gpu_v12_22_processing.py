"""Explicit Torch FFT/Bartlett backend for the separate V12.22 LMS variant.

Range/Doppler FFTs and noncoherent Bartlett map construction move to the
requested device. Optional GO-CFAR convolution reproduces the exact historical
flattened masked-window split, including its unusual threshold-cell count.
The inherited CPU GO-CFAR remains available as an explicit reference. DBSCAN
clustering and point-cloud routines remain inherited CPU functions. Publication
CA-CFAR and label evaluation are outside this processor and are not changed.

NumPy builds the exact historical windows, steering matrix and physical axes.
Complex128/float64 math avoids an unannounced precision change. CUDA/NumPy FFT
rounding need not be bitwise identical: a separate physical parity preflight is
required before training. No speedup is promised, particularly for FP64-limited
GPUs or while the unchanged CPU GO-CFAR remains dominant.
"""
from __future__ import annotations

import time

import numpy as np

from .processing import PostProcessing as NativePostProcessing


def _go_kernels(torch, device):
    """Native 33x21 flattened masked-window halves; NOT textbook range halves."""
    tr, td, gr, gd = 12, 8, 4, 2
    mask = np.ones((2 * (tr + gr) + 1, 2 * (td + gd) + 1), bool)
    mask[tr:tr + 2 * gr + 1, td:td + 2 * gd + 1] = False
    selected = np.flatnonzero(mask.ravel())
    mid = len(selected) // 2
    weights = np.zeros((2, mask.size), np.float64)
    weights[0, selected[:mid]] = 1. / mid
    weights[1, selected[mid:]] = 1. / (len(selected) - mid)
    return torch.as_tensor(weights.reshape(2, 1, *mask.shape),
                           dtype=torch.float64, device=device)


def go_cfar_2d_torch(power, kernels=None):
    """Exact native GO-CFAR formula with two float64 cross-correlations.

    The native mask contains 648 samples split into two flattened halves of
    324. Its alpha still uses N_half=12*21=252. That historical discrepancy is
    intentionally preserved, not corrected while changing execution backends.
    Border cells remain false. Strict CUT>threshold is unchanged. Floating
    summation differences require an exact boolean-mask parity preflight on
    representative real inputs before a production CUDA run.
    """
    import torch
    import torch.nn.functional as functional
    if (not isinstance(power, torch.Tensor) or power.ndim != 2
            or power.dtype != torch.float64 or torch.is_complex(power)):
        raise ValueError('Native GO-CFAR convolution requires a float64 2-D power tensor.')
    result = torch.zeros_like(power, dtype=torch.bool)
    if power.shape[0] < 33 or power.shape[1] < 21:
        return result
    kernels = _go_kernels(torch, power.device) if kernels is None else kernels
    if kernels.shape != (2, 1, 33, 21) or kernels.device != power.device or kernels.dtype != torch.float64:
        raise ValueError('The frozen native GO-CFAR kernels do not match the power tensor.')
    # Fix cuDNN algorithm selection for replay; restore process flags afterward.
    # This does not change the historical mask, averaging or threshold formula.
    with torch.backends.cudnn.flags(deterministic=True, benchmark=False, allow_tf32=False):
        background = functional.conv2d(power[None, None], kernels)[0]
    n_half = 12 * (2 * 8 + 2 * 2 + 1)
    alpha = n_half * ((1e-5) ** (-1 / n_half) - 1)
    threshold = alpha * torch.maximum(background[0], background[1])
    result[16:power.shape[0] - 16, 10:power.shape[1] - 10] = (
        power[16:power.shape[0] - 16, 10:power.shape[1] - 10] > threshold)
    return result


class TorchPostProcessing(NativePostProcessing):
    """Drop-in ``compute_RD_RA`` eight-tuple, with an explicit device contract.

    ``device='cpu', strict_cuda=False`` exists only for engineering parity tests.
    The default requires CUDA; an unavailable GPU never silently falls back.
    ``stats`` counts completed calls and synchronized wall time, not a benchmark.
    """
    def __init__(self, theta_res, cfg, *, device='cuda:0', strict_cuda=True,
                 doppler_batch=8, gpu_go_cfar=True):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError('The separate LMS backend requires PyTorch; the CPU V12.22 files remain usable.') from exc
        super().__init__(theta_res=theta_res, cfg=cfg)
        self._torch = torch
        self.device = torch.device(device)
        self.is_cuda = self.device.type == 'cuda'
        if strict_cuda and not self.is_cuda:
            raise ValueError('strict_cuda=True requires an explicit CUDA device, not CPU fallback.')
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError('Only explicit CPU engineering tests and CUDA execution are supported.')
        if self.is_cuda and not torch.cuda.is_available():
            raise RuntimeError('CUDA is required but unavailable. Do not label this run GPU accelerated.')
        if isinstance(doppler_batch, bool) or int(doppler_batch) != doppler_batch or int(doppler_batch) < 1:
            raise ValueError('doppler_batch must be a positive integer.')
        self.doppler_batch = int(doppler_batch)
        self.gpu_go_cfar = bool(gpu_go_cfar)
        self._theta = np.linspace(-np.pi / 3, np.pi / 3, self.theta_res)
        self._steering = self.ula_steering_matrix(theta=self._theta, d=self.lam / 2, sign=-1)
        fast = np.hanning(self.N)[None, None, :]
        slow = np.hanning(self.M)[None, :, None]
        self._fast = torch.as_tensor(fast, dtype=torch.float64, device=self.device)
        self._slow = torch.as_tensor(slow, dtype=torch.float64, device=self.device)
        self._beam = torch.as_tensor(np.ascontiguousarray(self._steering.conj().T),
                                     dtype=torch.complex128, device=self.device)
        self._go_weights = _go_kernels(torch, self.device) if self.gpu_go_cfar else None
        self._ranges = self.c0 * np.fft.fftshift(np.fft.fftfreq(self.N, d=1 / self.fs)) / (2 * self.S)
        self._velocity = self.lam / 2 * np.fft.fftshift(np.fft.fftfreq(self.M, d=self.Tchirp))
        self.stats = dict(calls=0, gpu_calls=0, device=str(self.device),
                          fft_beamforming_seconds=0., total_seconds=0.,
                          GO_CFAR_equations_unchanged=True,
                          GO_CFAR_deterministic_convolution=True,
                          GO_CFAR_backend='torch_convolution' if self.gpu_go_cfar else 'native_CPU',
                          CPU_native_clustering_and_point_cloud_unchanged=True,
                          dtype='complex128/float64')

    def _synchronize(self):
        if self.is_cuda:
            self._torch.cuda.synchronize(self.device)

    def compute_RD_RA(self, x):
        """Same FFT shift, two-tap MTI, Bartlett sign and Doppler integration."""
        x = np.asarray(x)
        if (x.shape != (self.num_rx, self.M, self.N) or not np.iscomplexobj(x)
                or not np.isfinite(x).all()):
            raise ValueError('Expected finite complex IQ in native [Rx, chirp, fast-time] order.')
        torch = self._torch
        self._synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            # NumPy's double Hann arrays promote complex64 input to complex128.
            raw = torch.as_tensor(np.ascontiguousarray(x), dtype=torch.complex128, device=self.device)
            windowed = raw * self._slow * self._fast
            ranged = torch.fft.fftshift(torch.fft.fft(windowed, dim=2), dim=2)
            clutter_removed = torch.zeros_like(ranged)
            clutter_removed[:, 1:, :] = ranged[:, 1:, :] - ranged[:, :-1, :]
            spectrum = torch.fft.fftshift(torch.fft.fft(clutter_removed, dim=1), dim=1)
            rd_power = torch.sum(torch.abs(spectrum) ** 2, dim=0)
            rd = 10 * torch.log10(rd_power + 1e-12)
            ra_power = torch.zeros((self.theta_res, self.N), dtype=torch.float64, device=self.device)
            for first in range(0, self.M, self.doppler_batch):
                # Batch only matrix multiplication. Accumulate each Doppler row
                # in native increasing order rather than introducing a new sum
                # reduction over Doppler or coherent cross-Doppler integration.
                chunk = spectrum[:, first:first + self.doppler_batch, :].permute(1, 0, 2)
                beams = torch.matmul(self._beam, chunk)
                powers = torch.abs(beams) ** 2
                for offset in range(len(powers)):
                    ra_power.add_(powers[offset])
            ra = 10 * torch.log10(ra_power + 1e-12)
            detections_tensor = go_cfar_2d_torch(rd_power, self._go_weights) if self.gpu_go_cfar else None
            # Unchanged CPU detector/point-cloud routines receive native arrays.
            rd_numpy = rd.cpu().numpy()
            ra_numpy = ra.cpu().numpy()
            power_numpy = rd_power.cpu().numpy()
            spectrum_numpy = spectrum.cpu().numpy()
            detections = detections_tensor.cpu().numpy() if detections_tensor is not None else None
        self._synchronize()
        math_finished = time.perf_counter()
        ranges, velocity = self._ranges.copy(), self._velocity.copy()
        if detections is None:
            detections = self.go_cfar_2d(power_numpy)
        clusters = self.cluster_detections(detections, power_numpy, ranges, velocity)
        point_cloud, point_list = self.detections_to_point_cloud(
            clusters, spectrum_numpy, self._theta, self._steering, ranges, velocity, power_numpy)
        self.stats['calls'] += 1
        self.stats['gpu_calls'] += int(self.is_cuda)
        self.stats['fft_beamforming_seconds'] += math_finished - started
        self.stats['total_seconds'] += time.perf_counter() - started
        return rd_numpy, ra_numpy, ranges, velocity, detections, clusters, point_cloud, point_list


GPUPostProcessing = TorchPostProcessing
