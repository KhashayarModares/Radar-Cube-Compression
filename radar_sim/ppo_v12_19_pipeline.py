from __future__ import annotations
import os
import re
import json
import math
import shutil
import warnings
import time
import gc
import io
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    from IPython.display import display
except Exception:
    display = print
try:
    from scipy.ndimage import convolve as scipy_convolve
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except Exception:
    scipy_convolve = None
    cKDTree = None
    SCIPY_AVAILABLE = False
np.set_printoptions(precision=4, suppress=True)
warnings.filterwarnings('default')
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
for p in [PROJECT_ROOT, PROJECT_ROOT / 'src', PROJECT_ROOT / 'radar_sim']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
CFAR_THRESHOLD_FACTORS = np.geomspace(0.1, 80.0, 32)
EPS = 1e-12
METHOD_ALIASES = {'Current_RDO_v3_RLActionSet': 'Current_RDO learned', 'Current_RDO_v3_fixed_action_set': 'Current_RDO fixed', 'FSLQ_10bit': 'FFT quant. 10b', 'FSLQ_12bit': 'FFT quant. 12b', 'RawLinear_12bit': 'Raw linear 12b'}
from radar_sim import matlab_v44_dataset as v44_dataset
from radar_sim.ppo_v12_18_learning import update_mc_ppo, remaining_budget_features
V12_18_VALIDATION_PROBE_CUBES = int(os.environ.get('RDO_V1219_VALIDATION_PROBE_CUBES', '200' if os.environ.get('RDO_RL_RUN_MODE', 'full') == 'full' else '25'))
if V12_18_VALIDATION_PROBE_CUBES < 1:
    raise ValueError('A positive fixed validation probe count is required.')
V12_18_ADVANTAGE_MODE = os.environ.get('RDO_V1219_ADVANTAGE_MODE', 'loo_residual').strip().lower()
V12_18_NORMALIZER_MODE = os.environ.get('RDO_V1219_NORMALIZER', 'multiframe').strip().lower()
V12_18_NORMALIZER_FRAMES = int(os.environ.get('RDO_V1219_NORMALIZER_FRAMES', '3'))
V12_18_UNMATCHED_MODE = os.environ.get('RDO_V1219_UNMATCHED_MODE', 'sum_positive').strip().lower()
if V12_18_ADVANTAGE_MODE not in ('critic', 'loo_residual'):
    raise ValueError('RDO_V1219_ADVANTAGE_MODE must be critic or loo_residual.')
if V12_18_NORMALIZER_MODE not in ('legacy', 'multiframe') or V12_18_NORMALIZER_FRAMES < 1:
    raise ValueError('Use legacy/multiframe normalization and a positive probe-frame count.')
if V12_18_UNMATCHED_MODE not in ('max_relative', 'sum_positive'):
    raise ValueError('RDO_V1219_UNMATCHED_MODE must be max_relative or sum_positive.')
from radar_sim.ppo_v12_19_reward import preservation_weighted_reward_terms, balanced_curve_regret, DEFAULT_COMMON_SPAN_DB, LOSS_COMPONENT_NAMES
V12_19_PRESERVATION_MULTIPLIER = float(os.environ.get('RDO_V1219_PRESERVATION_MULTIPLIER', '1'))
V12_19_CURVE_MODE = os.environ.get('RDO_V1219_CURVE_MODE', 'balanced').strip().lower()
V12_19_COMMON_SPAN_DB = DEFAULT_COMMON_SPAN_DB
if not np.isfinite(V12_19_PRESERVATION_MULTIPLIER) or V12_19_PRESERVATION_MULTIPLIER <= 0:
    raise ValueError('A finite positive preservation multiplier is required.')
if V12_19_CURVE_MODE not in ('legacy', 'balanced'):
    raise ValueError('Use legacy for the control or balanced for the corrected objective.')

def save_publication_figure(fig, save_path: Optional[Path], dpi: int=180):
    """Render in memory, then write bytes with explicit cross-platform checks.

    Giving a mounted Windows path directly to Pillow can raise ``EINVAL`` on
    some WSL/Pillow combinations.  A BytesIO render prevents Pillow from
    opening the destination path.  The final write is verified before return.
    """
    if save_path is None:
        return None
    target = Path(save_path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix.lower()
    format_name = {'.jpg': 'jpeg', '.jpeg': 'jpeg', '.tif': 'tiff', '.tiff': 'tiff'}.get(suffix, suffix.lstrip('.') or 'png')
    buffer = io.BytesIO()
    fig.savefig(buffer, format=format_name, dpi=dpi, bbox_inches='tight')
    payload = buffer.getvalue()
    buffer.close()
    if not payload:
        raise OSError(f'Figure renderer returned no bytes for {target}')
    temporary = target.with_name(f'.{target.name}.{os.getpid()}.tmp')
    try:
        with temporary.open('wb') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.replace(temporary, target)
        except OSError:
            target.write_bytes(payload)
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    except OSError as exc:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise OSError(f'Could not save figure to {target}. The output filesystem rejected a binary image write. Set RDO_OUTPUT_ROOT to a writable Linux-native directory (for example /home/<user>/rdo_outputs) and rerun.') from exc
    if not target.is_file() or target.stat().st_size != len(payload):
        raise OSError(f'Figure write verification failed for {target}')
    if suffix == '.png':
        with target.open('rb') as handle:
            signature = handle.read(8)
            handle.seek(-12, os.SEEK_END)
            trailer = handle.read(12)
        if signature != b'\x89PNG\r\n\x1a\n' or b'IEND' not in trailer:
            raise OSError(f'PNG integrity verification failed for {target}')
    return target
try:
    import h5py
except Exception as exc:
    raise ImportError('Install h5py in the active Python environment: python -m pip install h5py') from exc
try:
    from scipy.io import loadmat as scipy_loadmat
except Exception:
    scipy_loadmat = None
try:
    from scipy.optimize import linear_sum_assignment
except Exception as exc:
    raise ImportError('SciPy is required for one-to-one label/peak assignment.') from exc
try:
    from radar_sim.config import RadarParams
except Exception:

    class RadarParams:

        def __init__(self, N, M, fc, lam, num_rx, fs, S, Tchirp, snr_db, c0=299792458.0):
            self.N = int(N)
            self.M = int(M)
            self.fc = float(fc)
            self.lam = float(lam)
            self.num_rx = int(num_rx)
            self.fs = float(fs)
            self.S = float(S)
            self.Tchirp = float(Tchirp)
            self.snr_db = float(snr_db)
            self.c0 = float(c0)
EXPECTED_DATASET_CUBES_V11 = 1000
EXPECTED_DATASET_NAME_V11 = 'automotive_fmcw_1000_cubes_v44'

def _matlab_cube_file_count(path: Path) -> int:
    cube_dir = Path(path) / 'cubes'
    return sum((1 for _ in cube_dir.rglob('*.mat'))) if cube_dir.is_dir() else 0

def _valid_matlab100_root(path, expected_cubes=1000):
    if expected_cubes != 1000:
        raise ValueError('V12.19 requires the completed 1000-cube V44 dataset.')
    v44_dataset.validated_manifest(path)
    return True

def _valid_v6_1_1000_root(path: Path, expected_cubes: int=200) -> bool:
    return _valid_matlab100_root(path, expected_cubes)

def _v6_1000_file_count(path: Path) -> int:
    return _matlab_cube_file_count(path)

def find_matlab_dataset_root(explicit=None):
    return v44_dataset.find_dataset_root(PROJECT_ROOT, explicit)
DATASET_ROOT = find_matlab_dataset_root()
with (DATASET_ROOT / 'metadata' / 'radar_spec.json').open('r', encoding='utf-8') as handle:
    MATLAB_RADAR_SPEC = json.load(handle)
with (DATASET_ROOT / 'metadata' / 'label_schema.json').open('r', encoding='utf-8') as handle:
    MATLAB_LABEL_SCHEMA = json.load(handle)

def _spec_get(spec: Dict[str, Any], *names, default=None):
    for name in names:
        if name in spec:
            return spec[name]
    if default is not None:
        return default
    raise KeyError(f'None of the radar-spec fields are present: {names}')

def matlab_radar_cfg(spec: Dict[str, Any]=MATLAB_RADAR_SPEC) -> RadarParams:
    c0 = 299792458.0
    fc = float(_spec_get(spec, 'CenterFrequencyHz'))
    lam = float(_spec_get(spec, 'WavelengthM', default=c0 / fc))
    return RadarParams(N=int(_spec_get(spec, 'NumFastTimeSamples')), M=int(_spec_get(spec, 'NumChirps')), fc=fc, lam=lam, num_rx=int(_spec_get(spec, 'NumReceiveElements')), fs=float(_spec_get(spec, 'SampleRateHz')), S=float(_spec_get(spec, 'SweepSlopeHzPerS')), Tchirp=float(_spec_get(spec, 'SweepTimeS')), snr_db=14.0, c0=c0)
RADAR_CFG = matlab_radar_cfg()

def cfg_range_axis_m(cfg: RadarParams) -> np.ndarray:
    fb = np.fft.fftfreq(int(cfg.N), d=1.0 / float(cfg.fs))
    positive = fb >= 0
    return float(cfg.c0) * fb[positive] / (2.0 * float(cfg.S))

def cfg_velocity_axis_mps(cfg: RadarParams) -> np.ndarray:
    fd = np.fft.fftshift(np.fft.fftfreq(int(cfg.M), d=float(cfg.Tchirp)))
    return float(cfg.lam) / 2.0 * fd

def cfg_for_cube_shape(cube: np.ndarray, base_cfg: RadarParams=None) -> RadarParams:
    arr = np.asarray(cube)
    if arr.ndim != 3:
        raise ValueError(f'Expected (Rx, chirp, fast-time), got {arr.shape}')
    base_cfg = RADAR_CFG if base_cfg is None else base_cfg
    nrx, M, N = map(int, arr.shape)
    return RadarParams(N=N, M=M, fc=base_cfg.fc, lam=base_cfg.lam, num_rx=nrx, fs=base_cfg.fs, S=base_cfg.S, Tchirp=base_cfg.Tchirp, snr_db=base_cfg.snr_db, c0=base_cfg.c0)
V5_REQUIRED_LABEL_COLUMNS = ['global_frame', 'scene_id', 'scene_name', 'frame_in_scene', 'object_id', 'object_type', 'x_m', 'y_m', 'range_m', 'azimuth_deg', 'radial_velocity_mps', 'range_bin', 'doppler_bin', 'angle_bin', 'expected_visible']

def load_cube_matlab100(cube_row):
    return v44_dataset.load_compact_cube(Path(cube_row['cube_path']), (RADAR_CFG.num_rx, RADAR_CFG.M, RADAR_CFG.N), expected_metadata=cube_row)

def load_labels_matlab100(dataset_root):
    return v44_dataset.load_labels(dataset_root, RADAR_CFG)

def build_cube_index_matlab100(dataset_root, labels, accepted_only=False):
    return v44_dataset.build_cube_index(dataset_root, labels)

def labels_for_cube_matlab100(labels: pd.DataFrame, cube_row: pd.Series, visible_only: bool=True) -> pd.DataFrame:
    out = labels[labels.global_frame.astype(int) == int(cube_row.global_frame)].copy()
    if visible_only:
        out = out[out.expected_visible.astype(bool)]
    return out.reset_index(drop=True)

def preview_dataset_v5(dataset_root: Path=DATASET_ROOT):
    labels = load_labels_matlab100(dataset_root)
    index = build_cube_index_matlab100(dataset_root, labels)
    print('V4.3 labels:', labels.shape, 'cubes:', index.shape)
    display(index.groupby('scene_name').size().reset_index(name='cubes'))
    display(labels.groupby(['scene_name', 'semantic_class']).size().reset_index(name='labels'))
    first = load_cube_matlab100(index.iloc[0])
    print('First compact cube reconstructed shape/dtype:', first.shape, first.dtype)
    return (labels, index)
try:
    from radar_sim.processing import PostProcessing
except Exception as e:
    PostProcessing = None
    warnings.warn(f'Could not import radar_sim.processing.PostProcessing: {e}')

def make_post_processor(cfg, theta_res=121):
    """Scientific comparisons use exactly the checked native processor."""
    if PostProcessing is None:
        raise RuntimeError('Native radar_sim.processing.PostProcessing is required; no substitute detector is used.')
    pp = PostProcessing(theta_res=theta_res, cfg=cfg)
    print('PostProcessing constructor used: PostProcessing(theta_res=theta_res, cfg=cfg)')
    return pp

def postprocess_cube(cube: np.ndarray, post_processor: Any) -> Dict[str, Any]:
    out = post_processor.compute_RD_RA(cube)
    if isinstance(out, dict):
        d = dict(out)
    else:
        RD, RA, ranges, vel, detections, clusters, point_cloud, point_list = out
        d = {'RD': RD, 'RA': RA, 'ranges': ranges, 'vel': vel, 'detections': detections, 'clusters': clusters, 'point_cloud': point_cloud, 'point_list': point_list}
    if 'RD' not in d and 'RD_db' in d:
        d['RD'] = d['RD_db']
    if 'RA' not in d and 'RA_db' in d:
        d['RA'] = d['RA_db']
    if 'ranges' not in d:
        d['ranges'] = d.get('range_axis', cfg_range_axis_m(cfg_for_cube_shape(cube, RADAR_CFG)))
    if 'vel' not in d:
        d['vel'] = d.get('velocity_axis', d.get('velocities', cfg_velocity_axis_mps(cfg_for_cube_shape(cube, RADAR_CFG))))
    if 'detections' not in d:
        RD_tmp, _, _ = standardize_rd_map(d['RD'], d['ranges'], d['vel'])
        d['detections'] = cfar2d_ca(as_linear_power(RD_tmp, assume_db=True), threshold_factor=8.0)
    if 'point_cloud' not in d:
        d['point_cloud'] = np.zeros((0, 6), dtype=float)
    if 'angle_axis_deg' not in d:
        RA = np.asarray(d['RA'])
        d['angle_axis_deg'] = np.linspace(-60.0, 60.0, RA.shape[0])
    return validate_postprocessing_v12_18(cube, post_processor, d)

def as_linear_power(x: np.ndarray, assume_db: bool=True) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if assume_db:
        return np.power(10.0, x / 10.0)
    return np.maximum(x, 0.0)

def _impl_standardize_rd_map_c4_l153(RD, ranges, vel):
    RD, ranges, vel = (np.asarray(RD), np.asarray(ranges, float), np.asarray(vel, float))
    if RD.shape == (len(vel), len(ranges)):
        return (RD.T, ranges, vel)
    if RD.shape == (len(ranges), len(vel)):
        return (RD, ranges, vel)
    raise ValueError('RD dimensions must exactly match physical axes; no silent truncation.')

def _impl_standardize_ra_map_c4_l181(RA, ranges, angle_axis_deg):
    RA, ranges, angles = (np.asarray(RA), np.asarray(ranges, float), np.asarray(angle_axis_deg, float))
    if RA.shape == (len(angles), len(ranges)):
        return (RA.T, ranges, angles)
    if RA.shape == (len(ranges), len(angles)):
        return (RA, ranges, angles)
    raise ValueError('RA dimensions must exactly match physical axes; no silent truncation.')

def uniform_quantize_complex(x: np.ndarray, q: int) -> Tuple[np.ndarray, Dict[str, float]]:
    """Symmetric scalar quantization of real and imaginary parts."""
    x = np.asarray(x)
    q = int(q)
    if q <= 0:
        return (np.zeros_like(x), {'q': q, 'scale': 0.0})
    max_abs = float(max(np.max(np.abs(x.real)) if x.size else 0.0, np.max(np.abs(x.imag)) if x.size else 0.0, EPS))
    levels = max(1, 2 ** (q - 1) - 1)
    real_q = np.clip(np.round(x.real / max_abs * levels), -levels, levels)
    imag_q = np.clip(np.round(x.imag / max_abs * levels), -levels, levels)
    x_hat = (real_q + 1j * imag_q) * (max_abs / levels)
    return (x_hat.astype(x.dtype, copy=False), {'q': q, 'scale': max_abs, 'levels': levels})

def block_nmse(x: np.ndarray, x_hat: np.ndarray) -> float:
    return float(np.sum(np.abs(x - x_hat) ** 2) / (np.sum(np.abs(x) ** 2) + EPS))
DEFAULT_FIXED_ACTION_SET = [(0.04, 4), (0.06, 4), (0.08, 5), (0.1, 5), (0.12, 6), (0.16, 6), (0.2, 6), (0.25, 7), (0.32, 7), (0.4, 8), (0.55, 8), (0.7, 9)]
SUPERGRID_ACTION_SET = [(rho, q) for rho in np.round(np.geomspace(0.025, 0.75, 18), 4) for q in range(3, 11)]

def iter_blocks(cube: np.ndarray, block_slow: int=32, block_fast: int=64):
    nrx, M, N = cube.shape
    for m0 in range(0, M, block_slow):
        m1 = min(M, m0 + block_slow)
        for n0 in range(0, N, block_fast):
            n1 = min(N, n0 + block_fast)
            yield (slice(0, nrx), slice(m0, m1), slice(n0, n1))

def _impl_cfar2d_ca_c7_l6(power_map: np.ndarray, threshold_factor: float=8.0, guard: Tuple[int, int]=(1, 1), train: Tuple[int, int]=(4, 4)) -> np.ndarray:
    """2D CA-CFAR diagnostic detector.

    Threshold model:

        power[cell] > threshold_factor * local_training_mean

    A low threshold factor is permissive and produces many detections. A high
    threshold factor is conservative and eventually suppresses nearly all
    detections. This is used as an evaluation sweep, not as a compression input.
    """
    P = np.asarray(power_map, dtype=float)
    P = np.maximum(P, 0.0)
    gr, gc = map(int, guard)
    tr, tc = map(int, train)
    kr, kc = (2 * (tr + gr) + 1, 2 * (tc + gc) + 1)
    kernel = np.ones((kr, kc), dtype=float)
    kernel[tr:tr + 2 * gr + 1, tc:tc + 2 * gc + 1] = 0.0
    if SCIPY_AVAILABLE:
        noise_sum = scipy_convolve(P, kernel, mode='constant', cval=0.0)
        count = scipy_convolve(np.ones_like(P), kernel, mode='constant', cval=0.0)
        noise = noise_sum / np.maximum(count, 1.0)
        det = P > float(threshold_factor) * (noise + EPS)
        det[count <= 0] = False
        return det
    det = np.zeros_like(P, dtype=bool)
    nr, nc = P.shape
    for i in range(nr):
        r0, r1 = (max(0, i - tr - gr), min(nr, i + tr + gr + 1))
        rg0, rg1 = (max(0, i - gr), min(nr, i + gr + 1))
        for j in range(nc):
            c0, c1 = (max(0, j - tc - gc), min(nc, j + tc + gc + 1))
            cg0, cg1 = (max(0, j - gc), min(nc, j + gc + 1))
            win = P[r0:r1, c0:c1].copy()
            win[rg0 - r0:rg1 - r0, cg0 - c0:cg1 - c0] = np.nan
            noise = np.nanmean(win)
            if np.isfinite(noise) and P[i, j] > float(threshold_factor) * (noise + EPS):
                det[i, j] = True
    return det

def nearest_axis_index(axis: np.ndarray, value: float) -> int:
    axis = np.asarray(axis, dtype=float)
    if axis.size == 0 or not np.isfinite(value):
        return 0
    return int(np.nanargmin(np.abs(axis - float(value))))
from collections import Counter
try:
    from scipy.ndimage import maximum_filter as scipy_maximum_filter
    from scipy.ndimage import label as scipy_connected_components
except Exception:
    scipy_maximum_filter = None
    scipy_connected_components = None
NUM_CUBES_TO_EVALUATE_V10 = int(os.environ.get('RDO_NUM_CUBES', '40'))
RUN_PIPELINE_V10 = os.environ.get('RDO_RUN_PIPELINE', '1').strip().lower() not in {'0', 'false', 'no', 'definitions_only'}
ALLOW_FALLBACK_POSTPROCESSOR_V10 = os.environ.get('RDO_ALLOW_FALLBACK_POSTPROCESSOR', '0').strip().lower() in {'1', 'true', 'yes'}
SAVE_FIGURES = os.environ.get('RDO_SAVE_FIGURES', '1').strip().lower() not in {'0', 'false', 'no'}
MAX_CUBES_V8 = NUM_CUBES_TO_EVALUATE_V10
CFAR_LABEL_GATE_RD_V8 = (2, 2)
CFAR_LABEL_GATE_RA_V8 = (2, 2)
METHOD_LEARNED_V8 = 'TaskAware_PPO'
METHOD_FIXED_V8 = 'TaskAware_RDO_training_calibrated'
METHOD_ALIASES.update({METHOD_LEARNED_V8: 'Task-aware PPO', METHOD_FIXED_V8: 'Task-aware calibrated RDO'})
load_labels_v6_1000 = load_labels_matlab100
build_cube_index_v6_1000 = build_cube_index_matlab100
load_cube_v6_1000 = load_cube_matlab100
labels_for_cube_v6_1000 = labels_for_cube_matlab100

def safe_tight_layout(fig, rect=None):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*tight_layout.*', category=UserWarning)
        try:
            fig.tight_layout(rect=rect) if rect is not None else fig.tight_layout()
        except Exception:
            fig.subplots_adjust(left=0.08, right=0.98, bottom=0.13, top=0.9)

def find_dataset_root_v8(explicit: Optional[str]=None) -> Path:
    return find_matlab_dataset_root(explicit)
DATASET_ROOT_V8 = find_dataset_root_v8()

def _collapse_peak_plateaus_v9(power: np.ndarray, candidates: np.ndarray, nms_radius: Tuple[int, int]=(1, 1)) -> np.ndarray:
    if not np.any(candidates):
        return np.zeros_like(candidates, dtype=bool)
    if scipy_connected_components is not None:
        components, n_components = scipy_connected_components(candidates)
        peaks = np.zeros_like(candidates, dtype=bool)
        for component_id in range(1, int(n_components) + 1):
            points = np.argwhere(components == component_id)
            if len(points):
                values = power[points[:, 0], points[:, 1]]
                best = points[int(np.argmax(values))]
                peaks[int(best[0]), int(best[1])] = True
        return peaks
    rr, cc = map(int, nms_radius)
    peaks = candidates.copy()
    for row, col in np.argwhere(candidates):
        r0, r1 = (max(0, row - rr), min(power.shape[0], row + rr + 1))
        c0, c1 = (max(0, col - cc), min(power.shape[1], col + cc + 1))
        points = np.argwhere(candidates[r0:r1, c0:c1])
        if len(points) > 1:
            absolute = points + np.array([r0, c0])
            keep = absolute[np.lexsort((absolute[:, 1], absolute[:, 0]))][0]
            if row != keep[0] or col != keep[1]:
                peaks[row, col] = False
    return peaks

def _unique_actions(actions: Sequence[Tuple[float, int]]) -> List[Tuple[float, int]]:
    out, seen = ([], set())
    for rho, q in actions:
        action = (round(float(rho), 4), int(q))
        if action not in seen:
            out.append(action)
            seen.add(action)
    return out
ACTION_SUPERGRID_V6 = _unique_actions([*DEFAULT_FIXED_ACTION_SET, *SUPERGRID_ACTION_SET])
from matplotlib.lines import Line2D

def select_cube_index_v8(cube_index: pd.DataFrame, max_cubes: Optional[int]=MAX_CUBES_V8) -> pd.DataFrame:
    """Select an exact deterministic, scene-balanced V6 subset.

    Frames are evenly spaced within every scene. When all cubes are requested,
    canonical scene/frame order is returned, so a 1,000-cube selected run and a
    complete run have identical cube identity and order.
    """
    if cube_index is None or len(cube_index) == 0:
        raise ValueError('cube_index is empty.')
    ci = cube_index.sort_values(['scene_id', 'frame_in_scene', 'record_name']).reset_index(drop=True)
    requested = len(ci) if max_cubes is None else int(max_cubes)
    if not 1 <= requested <= len(ci):
        raise ValueError(f'Requested {requested} cubes from an index containing {len(ci)} cubes.')
    if requested == len(ci):
        return ci
    scene_groups = [group.sort_values(['frame_in_scene', 'record_name']).reset_index(drop=True) for _, group in ci.groupby(['scene_id', 'scene_name'], sort=True)]
    scene_count = len(scene_groups)
    if requested < scene_count:
        warnings.warn(f'Only {requested} cubes were requested for {scene_count} scenes; not every scene can be represented. Use at least one cube per scene.')
    quotas = [requested // scene_count + (i < requested % scene_count) for i in range(scene_count)]
    deficit = 0
    for i, group in enumerate(scene_groups):
        if quotas[i] > len(group):
            deficit += quotas[i] - len(group)
            quotas[i] = len(group)
    while deficit > 0:
        progressed = False
        for i, group in enumerate(scene_groups):
            if quotas[i] < len(group):
                quotas[i] += 1
                deficit -= 1
                progressed = True
                if deficit == 0:
                    break
        if not progressed:
            raise RuntimeError('Unable to allocate the requested scene-stratified cube count.')
    selected = []
    for group, quota in zip(scene_groups, quotas):
        if quota <= 0:
            continue
        positions = np.linspace(0, len(group) - 1, quota).round().astype(int)
        if len(np.unique(positions)) != quota:
            raise AssertionError('Evenly-spaced scene selection produced duplicate frame positions.')
        selected.append(group.iloc[positions])
    out = pd.concat(selected, ignore_index=True)
    out = out.sort_values(['scene_id', 'frame_in_scene', 'record_name']).reset_index(drop=True)
    if len(out) != requested or out['record_name'].duplicated().any():
        raise AssertionError('Scene-stratified selection violated the exact unique-cube contract.')
    return out
RL_RUN_MODE_V11 = os.environ.get('RDO_RL_RUN_MODE', 'full').strip().lower()
if RL_RUN_MODE_V11 not in {'debug', 'fast_smoke', 'smoke', 'full', 'definitions_only'}:
    raise ValueError('RDO_RL_RUN_MODE must be debug, fast_smoke, smoke, full, or definitions_only.')
_mode_defaults_v11 = {'debug': dict(train=2, validation=2, test=2, passes=1, calibration=1, bc=8, ppo=1), 'definitions_only': dict(train=4, validation=2, test=2, passes=1, calibration=1, bc=8, ppo=2), 'fast_smoke': dict(train=4, validation=2, test=2, passes=1, calibration=1, bc=8, ppo=2), 'smoke': dict(train=12, validation=5, test=5, passes=2, calibration=3, bc=8, ppo=3), 'full': dict(train=600, validation=200, test=200, passes=2, calibration=14, bc=25, ppo=5)}[RL_RUN_MODE_V11]
RL_TRAIN_CUBES_V11 = int(os.environ.get('RDO_RL_TRAIN_CUBES', _mode_defaults_v11['train']))
RL_VAL_CUBES_V11 = int(os.environ.get('RDO_RL_VAL_CUBES', _mode_defaults_v11['validation']))
NUM_CUBES_TO_EVALUATE_V11 = int(os.environ.get('RDO_NUM_CUBES', _mode_defaults_v11['test']))
RL_PASSES_V11 = int(os.environ.get('RDO_RL_PASSES', _mode_defaults_v11['passes']))
RL_CALIBRATION_CUBES_V11 = int(os.environ.get('RDO_RL_CALIBRATION_CUBES', _mode_defaults_v11['calibration']))
RL_BC_EPOCHS_V11 = 0
RL_PPO_EPOCHS_V11 = int(os.environ.get('RDO_RL_PPO_EPOCHS', _mode_defaults_v11['ppo']))
RL_BATCH_EPISODES_V11 = int(os.environ.get('RDO_RL_BATCH_EPISODES', '5'))
RL_SEED_V11 = int(os.environ.get('RDO_RL_SEED', '20260825'))
RL_RATE_TARGET_BPCS_V11 = float(os.environ.get('RDO_RL_RATE_TARGET_BPCS', '4.8'))
RL_RATE_BETA_V11 = float(os.environ.get('RDO_RL_RATE_BETA', '5.0'))
RL_RATE_BUDGET_TOLERANCE_V11 = 0.0
RL_LEARNING_RATE_V11 = float(os.environ.get('RDO_RL_LR', '0.0005'))
RL_BC_LEARNING_RATE_V11 = float(os.environ.get('RDO_RL_BC_LR', '0.003'))
RL_BC_TARGET_ACCURACY_V11 = float(os.environ.get('RDO_RL_BC_TARGET_ACCURACY', '0.90'))
RL_ENTROPY_COEF_V11 = float(os.environ.get('RDO_RL_ENTROPY_COEF', '0.002'))
RL_PPO_CLIP_V11 = float(os.environ.get('RDO_RL_PPO_CLIP', '0.20'))
RL_GAMMA_V11 = float(os.environ.get('RDO_RL_GAMMA', '1.0'))
RL_GAE_LAMBDA_V11 = 1.0
RL_TARGET_KL_V11 = float(os.environ.get('RDO_RL_TARGET_KL', '0.02'))
RL_GRAD_CLIP_V11 = float(os.environ.get('RDO_RL_GRAD_CLIP', '1.0'))
RL_FALSE_CONSTRAINT_TOLERANCE_V11 = float(os.environ.get('RDO_RL_FALSE_CONSTRAINT_TOLERANCE', '0.02'))
RL_FALSE_CONSTRAINT_WEIGHT_V11 = float(os.environ.get('RDO_RL_FALSE_CONSTRAINT_WEIGHT', '2.0'))
RL_MAP_CONSTRAINT_WEIGHT_V11 = float(os.environ.get('RDO_RL_MAP_CONSTRAINT_WEIGHT', '1.0'))
RL_MAP_CONSTRAINT_RELATIVE_TOLERANCE_V11 = float(os.environ.get('RDO_RL_MAP_CONSTRAINT_RELATIVE_TOLERANCE', '0.10'))
RL_MAP_CONSTRAINT_ABSOLUTE_TOLERANCE_V11 = float(os.environ.get('RDO_RL_MAP_CONSTRAINT_ABSOLUTE_TOLERANCE', '0.01'))
RL_TRAIN_CFAR_TARGETS_V11 = (1.25, 2.0, 4.0, 8.0, 16.0, 32.0)
ALLOW_DIAGNOSTIC_TEST_V11 = os.environ.get('RDO_ALLOW_DIAGNOSTIC_TEST', '0').strip().lower() in {'1', 'true', 'yes'}
RL_SHAPING_CHECKPOINTS_V11 = 1
RL_VALIDATION_TASK_TOLERANCE_V11 = float(os.environ.get('RDO_RL_VALIDATION_TASK_TOLERANCE', '0.0'))
if not 1 <= RL_TRAIN_CUBES_V11 <= 600 or not 1 <= RL_VAL_CUBES_V11 <= 200:
    raise ValueError('Counts exceed whole-episode V44 split 600/200/200.')
if not 1 <= NUM_CUBES_TO_EVALUATE_V11 <= 200:
    raise ValueError('Test count must be in [1,200].')
if min(RL_PASSES_V11, RL_PPO_EPOCHS_V11, RL_BATCH_EPISODES_V11) < 1 or RL_BC_EPOCHS_V11 < 0:
    raise ValueError('PPO passes/epochs/batch must be positive and BC epochs non-negative.')
if RL_RATE_TARGET_BPCS_V11 <= 0 or RL_RATE_BETA_V11 < 0 or RL_RATE_BUDGET_TOLERANCE_V11 < 0:
    raise ValueError('Rate target must be positive; rate penalty/tolerance must be non-negative.')
if not 0.0 <= RL_GAE_LAMBDA_V11 <= 1.0 or RL_TARGET_KL_V11 <= 0:
    raise ValueError('GAE lambda must be in [0,1] and target KL must be positive.')
if not 0.0 < RL_BC_TARGET_ACCURACY_V11 <= 1.0:
    raise ValueError('Behavior-cloning target accuracy must be in (0,1].')
if RL_SHAPING_CHECKPOINTS_V11 < 1:
    raise ValueError('RDO_RL_SHAPING_CHECKPOINTS must be positive.')
V11_TASK_COMPONENT_NAMES = ['RD_log_power_mse', 'RA_log_power_mse', 'RD_true_label_miss', 'RA_true_label_miss', 'new_false_detection_cost']
V11_TASK_SCALE_FLOORS = {'RD_log_power_mse': 0.02, 'RA_log_power_mse': 0.02, 'RD_true_label_miss': 0.05, 'RA_true_label_miss': 0.05, 'new_false_detection_cost': 0.05}
V11_TASK_SCALE_CEILINGS = {'RD_log_power_mse': np.inf, 'RA_log_power_mse': np.inf, 'RD_true_label_miss': 1.0, 'RA_true_label_miss': 1.0, 'new_false_detection_cost': 1.0}
VERSION_NAME_V11 = f'v12_19_v44_preservation_balanced_ppo_{RL_RUN_MODE_V11}_train{RL_TRAIN_CUBES_V11:03d}_val{RL_VAL_CUBES_V11:03d}_test{NUM_CUBES_TO_EVALUATE_V11:03d}'
_run_tag_v11 = ''.join((character if character.isalnum() or character in '-_' else '_' for character in os.environ.get('RDO_RL_RUN_TAG', '').strip()))
if _run_tag_v11:
    VERSION_NAME_V11 += f'_{_run_tag_v11}'

def _is_wsl_mounted_path_v11(path: Path) -> bool:
    text = str(Path(path).expanduser().resolve(strict=False)).replace('\\', '/')
    return text == '/mnt' or text.startswith('/mnt/')
_output_root_text_v11 = os.environ.get('RDO_OUTPUT_ROOT', '').strip()
if _output_root_text_v11:
    OUTPUT_ROOT_V11 = Path(_output_root_text_v11).expanduser()
    if _is_wsl_mounted_path_v11(OUTPUT_ROOT_V11) and os.environ.get('RDO_ALLOW_MOUNTED_OUTPUT', '0').strip().lower() not in {'1', 'true', 'yes'}:
        raise OSError('RDO_OUTPUT_ROOT points to a WSL-mounted Windows filesystem. Full runs create many tables and figures and can fail with Errno 5/22 there. Use a Linux-native path such as /home/<user>/rdo_outputs, or set RDO_ALLOW_MOUNTED_OUTPUT=1 at your own risk.')
else:
    OUTPUT_ROOT_V11 = PROJECT_ROOT / 'outputs'
OUTPUT_DIR_V11 = OUTPUT_ROOT_V11 / VERSION_NAME_V11
FIG_DIR_V11 = OUTPUT_DIR_V11 / 'figures'
TABLE_DIR_V11 = OUTPUT_DIR_V11 / 'tables'
MAP_DIR_V11 = OUTPUT_DIR_V11 / 'map_moments'
MODEL_DIR_V11 = OUTPUT_DIR_V11 / 'models'

def _atomic_target_v11(path: Path) -> Path:
    return path.with_name(f'.{path.name}.{os.getpid()}.{time.time_ns()}.tmp')

def safe_write_csv_v11(frame: pd.DataFrame, path: Path, index: bool=False, retries: int=3) -> Path:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    if len(frame.columns) == 0:
        frame = pd.DataFrame(columns=['status'])
    last_error = None
    for attempt in range(1, retries + 1):
        temporary = _atomic_target_v11(target)
        try:
            with temporary.open('w', encoding='utf-8', newline='') as handle:
                frame.to_csv(handle, index=index)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            if not target.is_file() or target.stat().st_size <= 0:
                raise OSError('empty CSV after atomic replace')
            observed = pd.read_csv(target, nrows=min(3, max(len(frame), 1)))
            expected_columns = [str(value) for value in frame.reset_index().columns] if index else [str(value) for value in frame.columns]
            if list(map(str, observed.columns)) != expected_columns:
                raise OSError('CSV read-back header mismatch')
            return target
        except (OSError, pd.errors.ParserError, UnicodeError) as exc:
            last_error = exc
            try:
                temporary.unlink()
            except OSError:
                pass
            if attempt < retries:
                time.sleep(0.15 * attempt)
    raise OSError(f'Could not persist verified CSV {target} after {retries} attempts. Use a Linux-native RDO_OUTPUT_ROOT and confirm free disk space.') from last_error

def safe_write_json_v11(payload: Dict[str, Any], path: Path, retries: int=3) -> Path:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, allow_nan=False).encode('utf-8')
    last_error = None
    for attempt in range(1, retries + 1):
        temporary = _atomic_target_v11(target)
        try:
            with temporary.open('wb') as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            with target.open('r', encoding='utf-8') as handle:
                json.load(handle)
            return target
        except (OSError, json.JSONDecodeError) as exc:
            last_error = exc
            try:
                temporary.unlink()
            except OSError:
                pass
            if attempt < retries:
                time.sleep(0.15 * attempt)
    raise OSError(f'Could not persist verified JSON {target}.') from last_error

def safe_save_npz_v11(path: Path, retries: int=3, **arrays) -> Path:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(1, retries + 1):
        temporary = _atomic_target_v11(target)
        try:
            with temporary.open('wb') as handle:
                np.savez_compressed(handle, **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            with np.load(target, allow_pickle=False) as loaded:
                if set(loaded.files) != set(arrays):
                    raise OSError('NPZ key mismatch')
            return target
        except (OSError, ValueError) as exc:
            last_error = exc
            try:
                temporary.unlink()
            except OSError:
                pass
            if attempt < retries:
                time.sleep(0.15 * attempt)
    raise OSError(f'Could not persist verified NPZ {target}.') from last_error
for _directory in [OUTPUT_DIR_V11, FIG_DIR_V11, TABLE_DIR_V11, MAP_DIR_V11, MODEL_DIR_V11]:
    _directory.mkdir(parents=True, exist_ok=True)
_storage_probe_v11 = pd.DataFrame({'row': np.arange(4096), 'value': np.linspace(0.0, 1.0, 4096)})
safe_write_csv_v11(_storage_probe_v11, TABLE_DIR_V11 / '.rdo_storage_probe.csv')
try:
    (TABLE_DIR_V11 / '.rdo_storage_probe.csv').unlink()
except OSError:
    pass
METHOD_RL_V11 = 'TaskAware_PPO'
METHOD_FIXED_V11 = 'TaskAware_RDO_training_calibrated'
METHOD_ALIASES.update({METHOD_RL_V11: 'Task-aware PPO', METHOD_FIXED_V11: 'Task-aware calibrated RDO', 'FSLQ_6bit': 'FFT quantization, 6 bit', 'FSLQ_8bit': 'FFT quantization, 8 bit'})
from radar_sim.ppo_v12_14_codec import DEFAULT_ACTIONS
ACTION_LIBRARY_V11 = list(DEFAULT_ACTIONS) + [(0.4, 4), (0.4, 5)]
ACTION_TO_INDEX_V11 = {(round(float(rho), 6), int(q)): i for i, (rho, q) in enumerate(ACTION_LIBRARY_V11)}
np.random.seed(RL_SEED_V11)
RUN_PIPELINE_V11 = RL_RUN_MODE_V11 not in {'debug', 'definitions_only'} and os.environ.get('RDO_RUN_PIPELINE', '1').strip().lower() not in {'0', 'false', 'no'}
RUN_DEBUG_V11 = RL_RUN_MODE_V11 == 'debug'

def temporal_scene_split_v11(cube_index, train_fraction=0.6, validation_fraction=0.2):
    if train_fraction != 0.6 or validation_fraction != 0.2:
        raise ValueError('Use the declared whole-episode 600/200/200 protocol.')
    split = v44_dataset.episode_split(cube_index, seed=RL_SEED_V11)
    safe_write_json_v11(split['metadata'], TABLE_DIR_V11 / 'episode_split_protocol_v12_19.json')
    return {key: split[key] for key in ('train', 'validation', 'test')}

def select_split_rows_v11(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    return select_cube_index_v8(frame, max_cubes=int(count))

class StateNormalizerV11:
    """Train-only z-scores; bounded inputs protect tanh conditioning.

    Legacy mode preserves V17's normalizer transform/probe method. Clipping
    affects network inputs only, never codec costs, spectra, labels or masks.
    """

    def __init__(self):
        self.mean = None
        self.std = None
        self.clip = 5.0 if V12_18_NORMALIZER_MODE == 'multiframe' else None

    def fit(self, states):
        states = np.asarray(states, np.float32)
        if states.ndim != 2 or states.shape[1] != len(V11_STATE_NAMES) or (not len(states)) or (not np.isfinite(states).all()):
            raise ValueError('State-normalizer data must be finite, nonempty and correctly shaped.')
        self.mean = states.mean(axis=0)
        self.std = states.std(axis=0)
        self.std[self.std < 1e-05] = 1.0
        return self

    def transform(self, states):
        if self.mean is None or self.std is None:
            raise RuntimeError('Fit the training-only normalizer first.')
        states = np.asarray(states, np.float32)
        if states.shape[-1] != len(self.mean) or not np.isfinite(states).all():
            raise ValueError('Malformed network observations.')
        result = (states - self.mean) / self.std
        if self.clip is not None:
            result = np.clip(result, -self.clip, self.clip)
        return result.astype(np.float32)

    def as_dict(self):
        return dict(state_names=V11_STATE_NAMES, mean=self.mean.tolist(), std=self.std.tolist(), clip=self.clip, normalization_mode=V12_18_NORMALIZER_MODE, fitted_from='training episodes only; label-free uniform feasible allocations')

class RadarPPOPolicyV11:
    """One 32-unit actor and a separate one 32-unit critic; no shared gradients."""

    def __init__(self, state_dim, action_count, hidden_dim=32, seed=RL_SEED_V11):
        rng = np.random.default_rng(seed)
        self.state_dim = int(state_dim)
        scale1 = np.sqrt(2.0 / max(state_dim + hidden_dim, 1))
        scale_head = np.sqrt(2.0 / max(hidden_dim, 1))
        self.params = {'W1': (rng.normal(size=(state_dim, hidden_dim)) * scale1).astype(np.float32), 'b1': np.zeros(hidden_dim, np.float32), 'Wa': (rng.normal(size=(hidden_dim, action_count)) * 0.1 * scale_head).astype(np.float32), 'ba': np.zeros(action_count, np.float32), 'Wv': (rng.normal(size=(hidden_dim, 1)) * 0.1 * scale_head).astype(np.float32), 'bv': np.zeros(1, np.float32), 'Wc1': (rng.normal(size=(state_dim, hidden_dim)) * scale1).astype(np.float32), 'bc1': np.zeros(hidden_dim, np.float32)}
        self.allowed_action_mask = np.ones(action_count, bool)
        self.rng = np.random.default_rng(seed + 1)

    def set_allowed_actions(self, action_indices):
        indices = np.unique(np.asarray(action_indices, np.int64))
        if len(indices) < 1 or np.any(indices < 0) or np.any(indices >= len(self.allowed_action_mask)):
            raise ValueError('Allowed action indices are empty or out of range.')
        self.allowed_action_mask[:] = False
        self.allowed_action_mask[indices] = True

    def forward(self, state):
        x = np.asarray(state, np.float32)
        if x.ndim != 2 or x.shape[1] != self.state_dim:
            raise ValueError('Policy state shape does not match its input dimension.')
        hidden = np.tanh(x @ self.params['W1'] + self.params['b1'])
        logits = hidden @ self.params['Wa'] + self.params['ba']
        probability = np.exp(logits - logits.max(axis=1, keepdims=True))
        probability /= probability.sum(axis=1, keepdims=True)
        critic_hidden = np.tanh(x @ self.params['Wc1'] + self.params['bc1'])
        value = (critic_hidden @ self.params['Wv'] + self.params['bv']).reshape(-1)
        return (probability.astype(np.float32), value.astype(np.float32), hidden.astype(np.float32))

    def state_dict(self):
        return {name: value.copy() for name, value in self.params.items()}

    def load_state_dict(self, state):
        if set(state) != set(self.params):
            raise ValueError('V12.12 requires its own separate-critic checkpoint, not V12.10 weights.')
        for name, value in state.items():
            array = np.asarray(value, np.float32)
            if array.shape != self.params[name].shape or not np.all(np.isfinite(array)):
                raise ValueError('Invalid checkpoint tensor: ' + name)
            self.params[name][...] = array

class NumpyAdamV11:

    def __init__(self, params: Dict[str, np.ndarray], learning_rate: float):
        self.params = params
        self.learning_rate = float(learning_rate)
        self.m = {name: np.zeros_like(value) for name, value in params.items()}
        self.v = {name: np.zeros_like(value) for name, value in params.items()}
        self.t = 0

    def step(self, gradients: Dict[str, np.ndarray]):
        self.t += 1
        total_norm = float(np.sqrt(sum((np.sum(np.asarray(gradient, dtype=float) ** 2) for gradient in gradients.values()))))
        scale = min(1.0, RL_GRAD_CLIP_V11 / max(total_norm, EPS))
        for name, gradient in gradients.items():
            gradient = np.asarray(gradient, dtype=np.float32) * scale
            self.m[name] = 0.9 * self.m[name] + 0.1 * gradient
            self.v[name] = 0.999 * self.v[name] + 0.001 * (gradient * gradient)
            m_hat = self.m[name] / (1.0 - 0.9 ** self.t)
            v_hat = self.v[name] / (1.0 - 0.999 ** self.t)
            self.params[name] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-08)

def _network_gradients_v11(policy, x, grad_logits, grad_value=None):
    x = np.asarray(x, np.float32)
    _, _, hidden = policy.forward(x)
    n = max(len(x), 1)
    g = np.asarray(grad_logits, np.float32)
    gradients = {name: np.zeros_like(value) for name, value in policy.params.items()}
    gradients['Wa'] = hidden.T @ g / n
    gradients['ba'] = g.mean(axis=0)
    actor_pre = g @ policy.params['Wa'].T * (1 - hidden * hidden)
    gradients['W1'] = x.T @ actor_pre / n
    gradients['b1'] = actor_pre.mean(axis=0)
    if grad_value is not None:
        critic_hidden = np.tanh(x @ policy.params['Wc1'] + policy.params['bc1'])
        gv = np.asarray(grad_value, np.float32).reshape(-1, 1)
        gradients['Wv'] = critic_hidden.T @ gv / n
        gradients['bv'] = gv.mean(axis=0)
        critic_pre = gv @ policy.params['Wv'].T * (1 - critic_hidden * critic_hidden)
        gradients['Wc1'] = x.T @ critic_pre / n
        gradients['bc1'] = critic_pre.mean(axis=0)
    return {name: value.astype(np.float32) for name, value in gradients.items()}

def _postprocessor_for_cube_v11(cube: np.ndarray, cache: Dict[Tuple[int, ...], Any]):
    shape = tuple(map(int, cube.shape))
    if shape not in cache:
        cache[shape] = make_post_processor(cfg_for_cube_shape(cube, RADAR_CFG))
    return cache[shape]

def _nearest_training_thresholds_v11() -> Tuple[float, ...]:
    available = np.asarray(CFAR_THRESHOLD_FACTORS_V8, dtype=float)
    selected = [float(available[np.argmin(np.abs(available - target))]) for target in RL_TRAIN_CFAR_TARGETS_V11]
    return tuple(dict.fromkeys(selected))

def _normalized_log_power_mse_v11(map_original: np.ndarray, map_method: np.ndarray) -> float:
    """Scale-invariant MSE on log power; radar dynamic range cannot be dominated by one peak."""
    power_original = np.maximum(as_linear_power(map_original, assume_db=True), 0.0)
    power_method = np.maximum(as_linear_power(map_method, assume_db=True), 0.0)
    scale = float(np.mean(power_original) + EPS)
    log_original = np.log1p(power_original / scale)
    log_method = np.log1p(power_method / scale)
    return float(np.mean((log_method - log_original) ** 2) / (np.mean(log_original ** 2) + EPS))

def _apply_action_mask_v11(probability: np.ndarray, action_mask: np.ndarray) -> np.ndarray:
    """Renormalize categorical probabilities over budget-feasible actions only."""
    probability = np.asarray(probability, dtype=np.float32)
    action_mask = np.asarray(action_mask, dtype=bool)
    if probability.ndim == 1:
        probability = probability[None, :]
    if action_mask.ndim == 1:
        action_mask = action_mask[None, :]
    if probability.shape != action_mask.shape:
        raise ValueError('Probability and action-mask shapes must match.')
    if np.any(action_mask.sum(axis=1) == 0):
        raise RuntimeError('Every policy step must retain at least one feasible action.')
    masked = probability * action_mask.astype(np.float32)
    masked /= np.maximum(masked.sum(axis=1, keepdims=True), EPS)
    return masked.astype(np.float32)

def encode_policy_v11(cube, policy, normalizer, stochastic, source_context=None):
    """Only the actor chooses actions; shared antenna support, one RD inverse."""
    cube = np.asarray(cube)
    source = prepare_policy_source_v12_12(cube) if source_context is None else source_context
    if tuple(cube.shape) != source['cube_shape']:
        raise ValueError('Source context does not match cube shape.')
    allowed = np.asarray(policy.allowed_action_mask, bool)
    if allowed.shape != (len(ACTION_LIBRARY_V11),) or not allowed.any():
        raise ValueError('Invalid action mask.')
    costs = source['action_bits_by_block']
    minimum = np.where(allowed[None, :], costs, np.inf).min(axis=1)
    suffix = np.cumsum(minimum[::-1])[::-1]
    budget = RL_RATE_TARGET_BPCS_V11 * cube.size
    running = float(source['cube_header_bits'])
    if running + suffix[0] > budget + 1e-06:
        raise ValueError('This cube is too small for codec headers at the requested budget.')
    spectrum_hat = np.zeros_like(source['spectrum'])
    raw_states, states, masks, actions, logp, values, logs, contributions = ([] for _ in range(8))
    previous = None
    for index, (slc, prepared) in enumerate(zip(source['slices'], source['prepared_blocks'])):
        raw = _codec_control_state_v11(slc, cube.shape, running, cube.size, previous, prepared, source['cube_mean_power'], source['cube_mean_spectral_concentration'], source['cube_block_log_energy_std'], source['cube_spectral_concentration_std'])
        raw = np.r_[raw, remaining_budget_features(source, cube.size, index, running, budget, allowed)].astype(np.float32)
        state = normalizer.transform(raw[None, :])[0]
        probability, value, _ = policy.forward(state[None, :])
        future = suffix[index + 1] if index + 1 < len(suffix) else 0.0
        mask = allowed & (costs[index] <= budget - running - future + 1e-06)
        if not mask.any():
            raise RuntimeError('No budget-feasible action; do not silently exceed budget.')
        probability = _apply_action_mask_v11(probability, mask)[0]
        action = int(policy.rng.choice(len(ACTION_LIBRARY_V11), p=probability)) if stochastic else int(np.argmax(probability))
        rho, q = ACTION_LIBRARY_V11[action]
        tile, bits = coherent_codec_v14.materialize_action(prepared, rho, q)
        spectrum_hat[slc] = tile
        running += bits['realized_bits']
        raw_states.append(raw)
        states.append(state)
        masks.append(mask)
        actions.append(action)
        logp.append(float(np.log(probability[action] + EPS)))
        values.append(float(value[0]))
        contributions.append(float(prepared['action_distortion'][action]))
        logs.append(dict(**{**bits, 'block_idx': index, 'action_index': action, 'rho': rho, 'q': q}, block_log_energy_ratio=float(raw[4]), spectral_top10_energy_fraction=float(raw[5]), cube_mean_spectral_concentration=float(raw[6]), cube_block_log_energy_std=float(raw[7]), cube_spectral_concentration_std=float(raw[8]), feasible_action_count=int(mask.sum()), compact_action_count=int(allowed.sum()), masked_action_count=int((~mask).sum()), transform_distortion_contribution=contributions[-1], remaining_budget_fraction=float(raw[-4]), minimum_remaining_cost_fraction=float(raw[-3]), maximum_remaining_cost_fraction=float(raw[-2]), allowance_per_remaining_coefficient=float(raw[-1])))
        previous = action
    cube_hat = coherent_codec_v14.reconstruct(source, spectrum_hat)
    bpcs = running / cube.size
    if bpcs > RL_RATE_TARGET_BPCS_V11 + 1e-06:
        raise AssertionError('Hard budget violated.')
    meta = dict(method=METHOD_RL_V11, codec='global_RD_shared_antenna_support_v14', realized_bits=running, realized_bpcs=bpcs, cube_nmse=block_nmse(cube, cube_hat), rel_error=float(np.sqrt(block_nmse(cube, cube_hat))), compression_ratio_vs_v43_int16_iq_32bpc=32.0 / max(bpcs, EPS), memory_ratio_proxy_64b_complex=64.0 / max(bpcs, EPS), actual_codec_bitstream_saved=False, codec_size_is_analytical=True, rate_budget_bits=budget, cube_header_bits=source['cube_header_bits'], rate_budget_enforced_by_action_mask=True, compact_policy_action_count=int(allowed.sum()), raw_cube_metrics_used_for_training=True, refinement_used=False)
    return dict(cube_hat=cube_hat, states_raw=np.asarray(raw_states, np.float32), states=np.asarray(states, np.float32), actions=np.asarray(actions, np.int64), action_masks=np.asarray(masks, bool), old_log_probs=np.asarray(logp, np.float32), old_values=np.asarray(values, np.float32), rewards=np.zeros(len(actions), np.float64), block_log=pd.DataFrame(logs), meta=meta, distortion_contributions=np.asarray(contributions, np.float64))

def compress_uniform_action_v11(cube, action, method_name=METHOD_FIXED_V11):
    result = coherent_codec_v14.uniform_encode(cube, action)
    result['meta']['method'] = method_name
    return (result['cube_hat'], pd.DataFrame(result['block_log']), result['meta'])

def _teacher_state_trace_v11(cube, action_index):
    """For the independent imitation comparator only; no main actor training."""
    source = prepare_policy_source_v12_12(cube)
    sequence = np.full(len(source['slices']), int(action_index), dtype=np.int64) if np.isscalar(action_index) else np.asarray(action_index, np.int64)
    if sequence.shape != (len(source['slices']),) or np.any(sequence < 0) or np.any(sequence >= len(ACTION_LIBRARY_V11)):
        raise ValueError('Invalid comparator action sequence.')
    running, previous, states = (float(source['cube_header_bits']), None, [])
    for i, (slc, prepared) in enumerate(zip(source['slices'], source['prepared_blocks'])):
        raw = _codec_control_state_v11(slc, cube.shape, running, cube.size, previous, prepared, source['cube_mean_power'], source['cube_mean_spectral_concentration'], source['cube_block_log_energy_std'], source['cube_spectral_concentration_std'])
        features = remaining_budget_features(source, cube.size, i, running, RL_RATE_TARGET_BPCS_V11 * cube.size, np.ones(len(ACTION_LIBRARY_V11), bool))
        states.append(np.r_[raw, features].astype(np.float32))
        running += source['action_bits_by_block'][i, sequence[i]]
        previous = int(sequence[i])
    return (np.asarray(states, np.float32), sequence)

def _impl_calibrate_task_rdo_v11_c18_l729(rows: pd.DataFrame, labels: pd.DataFrame, postprocessors: Dict[Tuple[int, ...], Any]) -> Dict[str, Any]:
    """Training-only post-processing sweep; test encoding uses the frozen selected action."""
    records = []
    for position, (_, row) in enumerate(rows.iterrows(), start=1):
        cube = load_cube_v6_1000(row)
        processor = _postprocessor_for_cube_v11(cube, postprocessors)
        labels_cube = labels_for_cube_v6_1000(labels, row)
        post_original = postprocess_cube(cube, processor)
        for action_index, action in enumerate(ACTION_LIBRARY_V11):
            cube_hat, _, meta = compress_uniform_action_v11(cube, action)
            post_method = postprocess_cube(cube_hat, processor)
            components, detail = radar_task_components_v11(post_original, post_method, labels_cube)
            detail['cube_nmse'] = float(block_nmse_contributions_v12_11(cube, cube_hat).sum())
            records.append({'record_name': row['record_name'], 'scene_name': row['scene_name'], 'action_index': action_index, 'rho': action[0], 'q': action[1], 'realized_bpcs': meta['realized_bpcs'], **components, **detail})
            del cube_hat, post_method
        print(f"V12.3 task-RDO calibration [{position:03d}/{len(rows):03d}] {row['record_name']}")
        del post_original
        gc.collect()
    table = pd.DataFrame(records)
    scaler = TaskObjectiveScalerV11.fit(table)

    def objective_frame(frame: pd.DataFrame) -> pd.DataFrame:
        objective_rows = []
        for _, record in frame.iterrows():
            components = {name: float(record[name]) for name in V11_TASK_COMPONENT_NAMES}
            detail = record.to_dict()
            objective_rows.append(scaler.total(components, detail, float(record['realized_bpcs'])))
        return pd.DataFrame(objective_rows)
    table = pd.concat([table.reset_index(drop=True), objective_frame(table)], axis=1)
    provisional_summary = table.groupby(['action_index', 'rho', 'q'], as_index=False).agg(mean_task_loss=('task_loss', 'mean'), mean_rate_ratio=('rate_ratio', 'mean'), mean_total_objective=('total_objective', 'mean'), mean_realized_bpcs=('realized_bpcs', 'mean')).sort_values(['mean_total_objective', 'mean_task_loss', 'mean_realized_bpcs'])
    provisional_action_index = int(provisional_summary.iloc[0]['action_index'])
    provisional_rows = table.loc[table['action_index'] == provisional_action_index]
    reference_components = {name: float(provisional_rows[name].quantile(0.9)) for name in ['new_false_detection_cost', 'RD_log_power_mse', 'RA_log_power_mse']}
    scaler.set_safety_budgets(reference_components)
    objective_columns = list(objective_frame(table).columns)
    table = table.drop(columns=objective_columns)
    table = pd.concat([table.reset_index(drop=True), objective_frame(table)], axis=1)
    action_summary = table.groupby(['action_index', 'rho', 'q'], as_index=False).agg(mean_task_loss=('task_loss', 'mean'), mean_rate_ratio=('rate_ratio', 'mean'), mean_total_objective=('total_objective', 'mean'), mean_realized_bpcs=('realized_bpcs', 'mean'), mean_false_constraint_violation=('false_constraint_violation', 'mean'), mean_map_constraint_penalty=('map_constraint_penalty', 'mean'), mean_rate_excess=('rate_excess', 'mean')).sort_values(['mean_total_objective', 'mean_rate_excess', 'mean_false_constraint_violation', 'mean_map_constraint_penalty', 'mean_task_loss', 'mean_realized_bpcs'])
    action_summary['maximum_realized_bpcs'] = action_summary.action_index.map(table.groupby('action_index').realized_bpcs.max())
    action_summary['within_rate_budget'] = action_summary['maximum_realized_bpcs'] <= RL_RATE_TARGET_BPCS_V11 + 1e-06
    feasible_summary = action_summary[action_summary['within_rate_budget']].copy()
    if len(feasible_summary) == 0:
        raise RuntimeError('No calibrated codec action satisfies the declared bpcs budget.')
    selected_action_index = _choose_calibration_anchor_v12_8(table, feasible_summary)
    action_summary['selected_fixed_action'] = action_summary.action_index.eq(selected_action_index)
    selected_action = ACTION_LIBRARY_V11[selected_action_index]
    policy_action_indices = np.arange(len(ACTION_LIBRARY_V11), dtype=np.int64)
    table['calibration_task_loss'] = table['task_loss']
    table['calibration_objective_protocol'] = 'unchanged historical RDO calibration score; not the PPO objective'
    comparable = []
    for _, record in table.iterrows():
        parts = {key: float(record[key]) for key in V11_TASK_COMPONENT_NAMES}
        terms = _safety_rate_terminal_reward_v12_3(parts, record.to_dict(), {}, {}, float(record.realized_bpcs), None)
        comparable.append({'original_only_' + key: value for key, value in terms.items() if key.startswith('cost_') or key in ('total_loss', 'total_reward', 'cube_nmse', 'reward_components_sum', 'reward_components_error')})
    table = pd.concat([table, pd.DataFrame(comparable)], axis=1)
    safe_write_csv_v11(table, TABLE_DIR_V11 / 'task_rdo_training_calibration_v12.csv')
    safe_write_csv_v11(action_summary, TABLE_DIR_V11 / 'task_rdo_action_summary_v12.csv')
    scale_table = pd.DataFrame([{'component': name, 'scale': scaler.scales[name], 'source': 'training positive median with declared floor and safety ceiling'} for name in V11_TASK_COMPONENT_NAMES])
    safe_write_csv_v11(scale_table, TABLE_DIR_V11 / 'task_loss_scales_v12.csv')
    return {'table': table, 'action_summary': action_summary, 'scaler': scaler, 'selected_action_index': selected_action_index, 'selected_action': selected_action, 'policy_action_indices': policy_action_indices, 'teacher_states': np.empty((0, len(V11_STATE_NAMES)), dtype=np.float32), 'teacher_actions': np.empty(0, dtype=np.int64)}

def _evaluable_label_bins_v11_8(labels: pd.DataFrame, ranges: np.ndarray, second_axis: np.ndarray, map_type: str):
    """Map only finite, in-axis labels; never clamp out-of-axis labels to a border."""
    ranges = np.asarray(ranges, dtype=float)
    second_axis = np.asarray(second_axis, dtype=float)
    if ranges.size == 0 or second_axis.size == 0:
        return ([], [], int(len(labels)))
    r_lo, r_hi = (float(np.nanmin(ranges)), float(np.nanmax(ranges)))
    s_lo, s_hi = (float(np.nanmin(second_axis)), float(np.nanmax(second_axis)))
    pairs, source_indices = ([], [])
    for source_index, (_, label) in enumerate(labels.iterrows()):
        r = float(label.get('range_m', np.nan))
        second = float(label.get('radial_velocity_mps' if str(map_type).upper() == 'RD' else 'azimuth_deg', np.nan))
        if not (np.isfinite(r) and np.isfinite(second) and (r_lo <= r <= r_hi) and (s_lo <= second <= s_hi)):
            continue
        pairs.append((nearest_axis_index(ranges, r), nearest_axis_index(second_axis, second)))
        source_indices.append(source_index)
    return (pairs, source_indices, int(len(labels) - len(pairs)))

def evaluate_fft8_validation_reference_v11(rows: pd.DataFrame, labels: pd.DataFrame, scaler: 'TaskObjectiveScalerV11', postprocessors: Dict[Tuple[int, ...], Any]) -> pd.DataFrame:
    """Compute the fixed FFT8 validation benchmark once; test cubes are never used."""
    records = []
    for _, row in rows.iterrows():
        cube = load_cube_v6_1000(row)
        processor = _postprocessor_for_cube_v11(cube, postprocessors)
        labels_cube = labels_for_cube_v6_1000(labels, row)
        post_original = postprocess_cube(cube, processor)
        cube_hat, _, meta = compress_cube_fft_full_support_multi(cube, {'FSLQ_8bit': 8})['FSLQ_8bit']
        post_method = postprocess_cube(cube_hat, processor)
        components, detail = radar_task_components_v11(post_original, post_method, labels_cube)
        detail['cube_nmse'] = float(block_nmse_contributions_v12_11(cube, cube_hat).sum())
        objective = scaler.total(components, detail, float(meta['realized_bpcs']))
        records.append({'record_name': row['record_name'], 'scene_name': row['scene_name'], 'frame_in_scene': row['frame_in_scene'], 'realized_bpcs': float(meta['realized_bpcs']), **components, **detail, **objective})
        del cube, cube_hat, post_original, post_method
        gc.collect()
    return pd.DataFrame(records)

def _peak_points(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(np.asarray(mask, dtype=bool))
    return pts.astype(np.int64, copy=False)

def _object_peak_assignment(label_bins: Sequence[Tuple[int, int]], peak_mask: np.ndarray, gate: Tuple[int, int], peak_power: Optional[np.ndarray]=None):
    """Global one-to-one Hungarian assignment of geometry labels to peaks."""
    peaks = _peak_points(peak_mask)
    nlab = len(label_bins)
    assignment = np.full(nlab, -1, dtype=int)
    if nlab == 0 or len(peaks) == 0:
        return (assignment, peaks)
    gr, gc = (max(int(gate[0]), 1), max(int(gate[1]), 1))
    huge = 1000000.0
    cost = np.full((nlab, len(peaks)), huge, dtype=float)
    for i, (r, c) in enumerate(label_bins):
        dr = np.abs(peaks[:, 0] - int(r))
        dc = np.abs(peaks[:, 1] - int(c))
        inside = (dr <= gr) & (dc <= gc)
        geom = (dr / gr) ** 2 + (dc / gc) ** 2
        if peak_power is not None and np.any(inside):
            p = np.asarray(peak_power, dtype=float)[peaks[:, 0], peaks[:, 1]]
            finite = p[np.isfinite(p)]
            if len(finite):
                p = (p - np.nanmin(p)) / (np.nanmax(p) - np.nanmin(p) + 1e-12)
                geom = geom + 0.02 * (1.0 - p)
        cost[i, inside] = geom[inside]
    rows, cols = linear_sum_assignment(cost)
    for i, j in zip(rows, cols):
        if cost[i, j] < huge / 2:
            assignment[int(i)] = int(j)
    return (assignment, peaks)

def _evaluable_label_bins_v12(labels: pd.DataFrame, ranges: np.ndarray, second_axis: np.ndarray, map_type: str):
    return _evaluable_label_bins_v11_8(labels, ranges, second_axis, map_type)

def _impl_precompute_original_cfar_count_sweep_v11_2_c21_l44(map_original_db: np.ndarray, labels: pd.DataFrame, ranges: np.ndarray, second_axis: np.ndarray, map_type: str, threshold_factors: Sequence[float], gate: Tuple[int, int]) -> Dict[str, Any]:
    map_type = str(map_type).upper()
    label_bins, label_source_indices, excluded = _evaluable_label_bins_v12(labels, ranges, second_axis, map_type)
    power = as_linear_power(map_original_db, assume_db=True)
    masks = precompute_nested_cfar_peak_masks_v11_2(power, threshold_factors)
    thresholds = sorted((float(v) for v in threshold_factors))
    per = {}
    for t in thresholds:
        positive = masks[t]['positive']
        peaks = masks[t]['peaks']
        assign, points = _object_peak_assignment(label_bins, peaks, gate, power)
        eligible = assign >= 0
        used = set(assign[eligible].tolist())
        unmatched = int(len(points) - len(used))
        per[t] = {'cells0': positive, 'peaks0': peaks, 'assignment0': assign, 'eligible0': eligible, 'supported0': int(eligible.sum()), 'npeak0': int(len(points)), 'unmatched0': unmatched, 'num_positive0': int(np.sum(positive))}
    return {'map_type': map_type, 'label_bins': label_bins, 'label_source_indices': label_source_indices, 'num_input_visible_labels': int(len(labels)), 'num_evaluable_labels': int(len(label_bins)), 'num_excluded_out_of_axis_labels': int(excluded), 'num_visible_labels': int(len(label_bins)), 'threshold_factors': thresholds, 'gate': tuple(map(int, gate)), 'per_threshold': per, 'ranges': np.asarray(ranges, dtype=float), 'second_axis': np.asarray(second_axis, dtype=float)}
assert _impl_precompute_original_cfar_count_sweep_v11_2_c21_l44.__doc__ is None or 'cluster' not in (_impl_precompute_original_cfar_count_sweep_v11_2_c21_l44.__doc__ or '').lower()
V12_2_RATE_PENALTY_MAX = 4.0

class TaskObjectiveScalerV11:
    """Training-calibrated diagnostic scorer, not the active PPO reward."""
    MAP_NAMES = ('RD_log_power_mse', 'RA_log_power_mse')

    def __init__(self, scales: Dict[str, float], false_detection_budget: Optional[float]=None, map_error_budgets: Optional[Dict[str, float]]=None, supported_label_scale: float=1.0):
        self.scales = {name: float(scales[name]) for name in V11_TASK_COMPONENT_NAMES}
        self.false_detection_budget = None if false_detection_budget is None else float(false_detection_budget)
        self.map_error_budgets = {name: float(value) for name, value in (map_error_budgets or {}).items() if name in self.MAP_NAMES}
        self.supported_label_scale = float(max(supported_label_scale, 1.0))

    @classmethod
    def fit(cls, table: pd.DataFrame):
        scales = {}
        for name in V11_TASK_COMPONENT_NAMES:
            values = pd.to_numeric(table[name], errors='coerce').to_numpy(dtype=float)
            positive = values[np.isfinite(values) & (values > 0)]
            median = float(np.median(positive)) if len(positive) else 0.0
            robust_scale = max(median, float(V11_TASK_SCALE_FLOORS[name]))
            scales[name] = min(robust_scale, float(V11_TASK_SCALE_CEILINGS[name]))
        if {'record_name', 'RD_labels_supported_original', 'RA_labels_supported_original'}.issubset(table.columns):
            support = table.groupby('record_name', as_index=False).first()
            support_total = pd.to_numeric(support['RD_labels_supported_original'], errors='coerce').fillna(0.0) + pd.to_numeric(support['RA_labels_supported_original'], errors='coerce').fillna(0.0)
            supported_label_scale = float(max(support_total.mean(), 1.0))
        else:
            supported_label_scale = 1.0
        return cls(scales, supported_label_scale=supported_label_scale)

    def set_safety_budgets(self, reference_components: Dict[str, float]):
        self.false_detection_budget = float(max(float(reference_components['new_false_detection_cost']), 0.0) + RL_FALSE_CONSTRAINT_TOLERANCE_V11)
        self.map_error_budgets = {name: float(max(float(reference_components[name]), 0.0) * (1.0 + RL_MAP_CONSTRAINT_RELATIVE_TOLERANCE_V11) + RL_MAP_CONSTRAINT_ABSOLUTE_TOLERANCE_V11) for name in self.MAP_NAMES}

    def set_false_detection_budget(self, reference_cost: float):
        reference = {name: 0.0 for name in V11_TASK_COMPONENT_NAMES}
        reference['new_false_detection_cost'] = float(reference_cost)
        self.set_safety_budgets(reference)

    def task_terms(self, components: Dict[str, float], detail: Dict[str, float]) -> Dict[str, float]:
        supported_total = float(detail.get('true_labels_supported_total', 0.0))
        preserved_total = float(detail.get('true_labels_preserved_total', 0.0))
        missed_total = max(0.0, supported_total - preserved_total) if supported_total > 0 else float(sum((float(components[f'{p}_true_label_miss']) for p in ('RD', 'RA') if bool(detail.get(f'{p}_label_active', 0.0)))))
        base_detection_loss = missed_total / max(self.supported_label_scale, 1.0)
        false_violation = 0.0
        if self.false_detection_budget is not None:
            raw = max(0.0, float(components['new_false_detection_cost']) - self.false_detection_budget)
            false_violation = min(raw / max(self.scales['new_false_detection_cost'], 1.0), 1.0)
        false_penalty = float(false_violation)
        map_violations = {}
        for name in self.MAP_NAMES:
            budget = self.map_error_budgets.get(name)
            if budget is None:
                map_violations[name] = 0.0
            else:
                excess = max(0.0, float(components[name]) - float(budget))
                denom = max(abs(float(budget)), self.scales[name], 1e-06)
                map_violations[name] = min(excess / denom, 1.0)
        map_penalty = float(np.mean(list(map_violations.values())))
        result = {'true_labels_missed_total': float(missed_total), 'supported_label_scale': float(self.supported_label_scale), 'base_detection_loss': float(base_detection_loss), 'base_task_loss': float(base_detection_loss), 'false_constraint_budget': np.nan if self.false_detection_budget is None else float(self.false_detection_budget), 'false_constraint_violation': float(false_violation), 'false_constraint_penalty': float(false_penalty), 'RD_map_constraint_budget': self.map_error_budgets.get('RD_log_power_mse', np.nan), 'RA_map_constraint_budget': self.map_error_budgets.get('RA_log_power_mse', np.nan), 'RD_map_constraint_violation': float(map_violations['RD_log_power_mse']), 'RA_map_constraint_violation': float(map_violations['RA_log_power_mse']), 'map_constraint_penalty': float(map_penalty), 'task_loss': float(base_detection_loss + false_penalty + map_penalty)}
        thresholds = np.asarray(_nearest_training_thresholds_v11(), dtype=float)
        maximum_span = 2.0 * float(10.0 * np.log10(thresholds.max() / thresholds.min()))
        span = float(detail.get('alignment_span_sum_db', maximum_span))
        supported = max(float(detail.get('true_labels_supported_total', 0.0)), 1.0)
        deficit = float(detail.get('alignment_weighted_label_deficit', 0.0)) / supported
        span_shortfall = max(maximum_span - span, 0.0) / max(maximum_span, EPS)
        alignment_loss = V12_7_ALIGNMENT_TASK_LOSS_WEIGHT * (0.65 * span_shortfall + 0.35 * deficit)
        result['alignment_span_shortfall_normalized'] = float(span_shortfall)
        result['alignment_weighted_deficit_normalized'] = float(deficit)
        result['alignment_task_loss'] = float(alignment_loss)
        result['task_loss'] = float(result['task_loss'] + alignment_loss)
        return result

    def total(self, components: Dict[str, float], detail: Dict[str, float], bpcs: float) -> Dict[str, float]:
        task_terms = self.task_terms(components, detail)
        rate_ratio = float(bpcs / max(RL_RATE_TARGET_BPCS_V11, EPS))
        rate_excess = max(0.0, rate_ratio - 1.0)
        rate_penalty = min(float(RL_RATE_BETA_V11 * rate_excess ** 2), V12_2_RATE_PENALTY_MAX)
        return {**task_terms, 'rate_ratio': rate_ratio, 'rate_excess': float(rate_excess), 'rate_penalty': float(rate_penalty), 'rate_budget_bpcs': float(RL_RATE_TARGET_BPCS_V11), 'rate_within_budget': bool(bpcs <= RL_RATE_TARGET_BPCS_V11 * (1.0 + RL_RATE_BUDGET_TOLERANCE_V11)), 'total_objective': float(task_terms['task_loss'] + rate_penalty)}

    def as_dict(self):
        return dict(revision=NOTEBOOK_REVISION_V12_8, architecture='coherent-transform 32-unit actor; independent one 32-unit state-value critic', objective='budget-constrained perception/distortion; stationary task weights', main_initialization='random; no imitation or calibrated-action bias', behavior_cloning='separate frozen comparator only', main_imitation_epochs=0, refinement_used=False, codec='global_RD_shared_antenna_support', low_cfar_margin_factor=V12_18_MARGIN_FACTOR, low_cfar_margin_weight=V12_18_MARGIN_WEIGHT, state_names=V11_STATE_NAMES, action_library=ACTION_LIBRARY_V11, action_set='full; hard rate feasibility masks only', state_normalizer_source='training-only label-free exploration', training_cfar_thresholds=list(_nearest_training_thresholds_v11()), scales=dict(self.scales), supported_label_scale=self.supported_label_scale, map_error_budgets=self.map_error_budgets, false_detection_budget=self.false_detection_budget, constraints=V12_18_DUAL.as_dict(), map_excess_weight=V12_18_MAP_EXCESS_WEIGHT, dense_distortion_weight=V12_11_DISTORTION_WEIGHT, reward_rate_saving_weight=0.0, base_rate_cap_bpcs=RL_RATE_TARGET_BPCS_V11, advantage_estimator=V12_18_ADVANTAGE_MODE, state_normalization=V12_18_NORMALIZER_MODE, auxiliary_critic_only=False, rollouts_per_cube=V12_12_ROLLOUTS_PER_CUBE, fixed_validation_reward_weights=V12_18_INITIAL_WEIGHTS.tolist(), no_zero_miss_guarantee=True, experiment_seed=int(RL_SEED_V11))
V12_4_PPO_REWARD_SCALE = float(os.environ.get('RDO_V124_PPO_REWARD_SCALE', '5000'))
V12_4_VALUE_COEF = float(os.environ.get('RDO_V124_VALUE_COEF', '0.25'))
V12_4_FALSE_MEAN_LIMIT = float(os.environ.get('RDO_V124_FALSE_MEAN_LIMIT', '0.05'))
V12_4_FALSE_WORST_LIMIT = float(os.environ.get('RDO_V124_FALSE_WORST_LIMIT', '0.15'))
V12_4_MAX_UNIFORM_QUALITY_RATE_MULTIPLIER = float(os.environ.get('RDO_V124_MAX_UNIFORM_QUALITY_RATE_MULTIPLIER', '1.75'))
if V12_4_PPO_REWARD_SCALE <= 0 or V12_4_VALUE_COEF < 0:
    raise ValueError('V12.4 PPO reward scale must be positive and value coefficient non-negative.')
if V12_4_FALSE_MEAN_LIMIT <= 0 or V12_4_FALSE_WORST_LIMIT <= 0:
    raise ValueError('V12.4 false-peak limits must be positive.')

def _codec_control_state_v11(slc, cube_shape, running_bits, total_coeffs, previous_action_index, prepared, cube_mean_power, cube_mean_spectral_concentration, cube_block_log_energy_std, cube_spectral_concentration_std):
    """Label-free physical state; no action ordinal, detector or GT access."""
    previous = (0.0, 0.0) if previous_action_index is None else ACTION_LIBRARY_V11[int(previous_action_index)]
    prefix = [0.5 * (slc[1].start + slc[1].stop) / cube_shape[1], 0.5 * (slc[2].start + slc[2].stop) / cube_shape[2], running_bits / max(total_coeffs * RL_RATE_TARGET_BPCS_V11, EPS), previous[0], float(np.clip(np.log((prepared['block_mean_power'] + EPS) / (cube_mean_power + EPS)), -12, 12)), prepared['spectral_top10_fraction'], cube_mean_spectral_concentration, cube_block_log_energy_std, cube_spectral_concentration_std, prepared['spectral_top1_fraction'], prepared['spectral_top5_fraction'], prepared['spectral_entropy_normalized'], prepared['spectral_peak_to_mean_log'], previous[1] / 16.0, prepared['block_energy_fraction'], prepared['future_energy_fraction']]
    for bits, distortion in zip(prepared['action_bits'], prepared['action_distortion']):
        prefix.extend([bits / max(prepared['total_coeffs'] * RL_RATE_TARGET_BPCS_V11, EPS), float(np.log1p(1000.0 * distortion))])
    state = np.asarray(prefix, np.float32)
    if state.shape != (len(V11_STATE_NAMES) - 4,) or not np.isfinite(state).all():
        raise ValueError('Malformed physical/action state.')
    return state
_cfar2d_ca_v12_4_reference = _impl_cfar2d_ca_c7_l6

def _box_sum_zero_v12_4_1(a: np.ndarray, half_rows: int, half_cols: int) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    hr, hc = (int(half_rows), int(half_cols))
    padded = np.pad(a, ((hr, hr), (hc, hc)), mode='constant', constant_values=0.0)
    integral = np.pad(padded.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)), mode='constant', constant_values=0.0)
    kh, kw = (2 * hr + 1, 2 * hc + 1)
    return integral[kh:, kw:] - integral[:-kh, kw:] - integral[kh:, :-kw] + integral[:-kh, :-kw]

def _ca_cfar_score_v12_4_1(power_map: np.ndarray, guard: Tuple[int, int]=(1, 1), train: Tuple[int, int]=(4, 4)) -> Tuple[np.ndarray, np.ndarray]:
    power = np.maximum(np.asarray(power_map, dtype=np.float64), 0.0)
    gr, gc = map(int, guard)
    tr, tc = map(int, train)
    outer_r, outer_c = (tr + gr, tc + gc)
    outer_sum = _box_sum_zero_v12_4_1(power, outer_r, outer_c)
    guard_sum = _box_sum_zero_v12_4_1(power, gr, gc)
    noise_sum = outer_sum - guard_sum
    ones = np.ones_like(power, dtype=np.float64)
    outer_count = _box_sum_zero_v12_4_1(ones, outer_r, outer_c)
    guard_count = _box_sum_zero_v12_4_1(ones, gr, gc)
    count = outer_count - guard_count
    noise = noise_sum / np.maximum(count, 1.0)
    score = power / (noise + EPS)
    return (score, count)

def cfar2d_ca(power_map: np.ndarray, threshold_factor: float=8.0, guard: Tuple[int, int]=(1, 1), train: Tuple[int, int]=(4, 4)) -> np.ndarray:
    score, count = _ca_cfar_score_v12_4_1(power_map, guard=guard, train=train)
    return (count > 0.0) & (score > float(threshold_factor))

def _preflight_vectorized_cfar_v12_4_1() -> None:
    rng = np.random.default_rng(1241)
    test_power = rng.lognormal(mean=0.0, sigma=1.0, size=(13, 17))
    for threshold in (0.1, 1.0, 8.0, 100.0):
        reference = _cfar2d_ca_v12_4_reference(test_power, threshold_factor=threshold)
        candidate = cfar2d_ca(test_power, threshold_factor=threshold)
        if not np.array_equal(reference, candidate):
            raise AssertionError(f'Vectorized CA-CFAR parity failed at threshold {threshold}.')
_preflight_vectorized_cfar_v12_4_1()

def _prune_original_cfar_cache_v12_4_1(cache: Dict[str, Any]) -> Dict[str, Any]:
    for item in cache.get('per_threshold', {}).values():
        item.pop('cells0', None)
        item.pop('peaks0', None)
        item.pop('assignment0', None)
    return cache

def _ensure_original_task_cache_v12_4_1(post_original: Dict[str, Any], labels_cube: pd.DataFrame) -> Dict[str, Any]:
    cached = post_original.get('_task_cache_v12_4_1')
    if cached is not None:
        return cached
    thresholds = _nearest_training_thresholds_v11()
    rd0, rr, vv = standardize_rd_map(post_original['RD'], post_original['ranges'], post_original['vel'])
    ra0, rar, aa = standardize_ra_map(post_original['RA'], post_original['ranges'], post_original['angle_axis_deg'])
    rd_cache = precompute_original_cfar_count_sweep_v11_2(rd0, labels_cube, rr, vv, 'RD', thresholds, CFAR_LABEL_GATE_RD_V8)
    ra_cache = precompute_original_cfar_count_sweep_v11_2(ra0, labels_cube, rar, aa, 'RA', thresholds, CFAR_LABEL_GATE_RA_V8)
    cached = {'thresholds': tuple((float(x) for x in thresholds)), 'rd0': rd0, 'rr': rr, 'vv': vv, 'ra0': ra0, 'rar': rar, 'aa': aa, 'rd_cache': _prune_original_cfar_cache_v12_4_1(rd_cache), 'ra_cache': _prune_original_cfar_cache_v12_4_1(ra_cache)}
    post_original['_task_cache_v12_4_1'] = cached
    return cached

def _compact_task_post_v12_4_1(post: Dict[str, Any]) -> Dict[str, Any]:
    """Retain only arrays needed by task evaluation plus the original CFAR cache."""
    keep = {}
    for name in ('RD', 'RA', 'ranges', 'vel', 'angle_axis_deg'):
        if name in post:
            keep[name] = post[name]
    if '_task_cache_v12_4_1' in post:
        keep['_task_cache_v12_4_1'] = post['_task_cache_v12_4_1']
    return keep
_VALIDATION_REFERENCE_CACHE_V12_4_1: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

def evaluate_policy_rows_v11(rows: pd.DataFrame, labels: pd.DataFrame, policy: RadarPPOPolicyV11, normalizer: StateNormalizerV11, fixed_action: Tuple[float, int], scaler: TaskObjectiveScalerV11, postprocessors: Dict[Tuple[int, ...], Any]):
    candidate_records, fixed_records, action_rows = ([], [], [])
    fixed_action_key = (float(fixed_action[0]), int(fixed_action[1]))
    for _, row in rows.iterrows():
        cube = load_cube_v6_1000(row)
        processor = _postprocessor_for_cube_v11(cube, postprocessors)
        labels_cube = labels_for_cube_v6_1000(labels, row)
        cache_key = (str(row['record_name']), str(row.get('cube_path', '')), fixed_action_key, tuple((float(x) for x in _nearest_training_thresholds_v11())))
        reference = _VALIDATION_REFERENCE_CACHE_V12_4_1.get(cache_key)
        if reference is None:
            post_original_full = postprocess_cube(cube, processor)
            _ensure_original_task_cache_v12_4_1(post_original_full, labels_cube)
            fixed_hat, _, fixed_meta = compress_uniform_action_v11(cube, fixed_action)
            post_fixed = postprocess_cube(fixed_hat, processor)
            fixed_components, fixed_detail = radar_task_components_v11(post_original_full, post_fixed, labels_cube)
            fixed_meta['cube_nmse'] = float(block_nmse_contributions_v12_11(cube, fixed_hat).sum())
            fixed_detail['cube_nmse'] = fixed_meta['cube_nmse']
            reference = {'post_original': _compact_task_post_v12_4_1(post_original_full), 'fixed_components': dict(fixed_components), 'fixed_detail': dict(fixed_detail), 'fixed_meta': dict(fixed_meta)}
            _VALIDATION_REFERENCE_CACHE_V12_4_1[cache_key] = reference
            del post_original_full, fixed_hat, post_fixed
        post_original = reference['post_original']
        package = encode_policy_v11(cube, policy, normalizer, stochastic=False)
        package['meta']['cube_nmse'] = float(block_nmse_contributions_v12_11(cube, package['cube_hat']).sum())
        post_candidate = postprocess_cube(package['cube_hat'], processor)
        candidate_components, candidate_detail = radar_task_components_v11(post_original, post_candidate, labels_cube)
        candidate_detail['cube_nmse'] = package['meta']['cube_nmse']
        candidate_objective = scaler.total(candidate_components, candidate_detail, package['meta']['realized_bpcs'])
        candidate_records.append({'record_name': row['record_name'], 'scene_name': row['scene_name'], 'frame_in_scene': row['frame_in_scene'], 'realized_bpcs': package['meta']['realized_bpcs'], 'cube_nmse': float(package['meta']['cube_nmse']), **candidate_components, **candidate_detail, **candidate_objective})
        fixed_components = dict(reference['fixed_components'])
        fixed_detail = dict(reference['fixed_detail'])
        fixed_meta = reference['fixed_meta']
        fixed_detail['cube_nmse'] = fixed_meta['cube_nmse']
        fixed_objective = scaler.total(fixed_components, fixed_detail, fixed_meta['realized_bpcs'])
        fixed_records.append({'record_name': row['record_name'], 'scene_name': row['scene_name'], 'frame_in_scene': row['frame_in_scene'], 'realized_bpcs': fixed_meta['realized_bpcs'], 'cube_nmse': float(fixed_meta['cube_nmse']), **fixed_components, **fixed_detail, **fixed_objective})
        desired = ['realized_bpcs', 'realized_bits', 'feasible_action_count', 'block_idx', 'action_index', 'rho', 'q', 'block_log_energy_ratio', 'spectral_top10_energy_fraction', 'cube_mean_spectral_concentration', 'cube_block_log_energy_std', 'cube_spectral_concentration_std', 'spectral_top1_energy_fraction', 'spectral_top5_energy_fraction', 'spectral_entropy_normalized', 'spectral_peak_to_mean_log']
        cols = [name for name in desired if name in package['block_log'].columns]
        log = package['block_log'][cols].copy()
        log['record_name'] = row['record_name']
        log['scene_name'] = row['scene_name']
        action_rows.append(log)
        del cube, package, post_candidate
        gc.collect()
    return (pd.DataFrame(candidate_records), pd.DataFrame(fixed_records), pd.concat(action_rows, ignore_index=True) if action_rows else pd.DataFrame())

def train_taskaware_ppo_v11(train_rows, validation_rows, labels):
    """Train/restore main PPO first; construct all compressor comparators later."""
    global V12_18_DUAL
    V12_18_DUAL = PreservationConstraintsV12_18()
    preflight = run_preflight_debug_v11()
    check_learning_math_v12_11()
    validation_rows = select_split_rows_v11(validation_rows, min(V12_18_VALIDATION_PROBE_CUBES, len(validation_rows)))
    safe_write_csv_v11(validation_rows, TABLE_DIR_V11 / 'fixed_validation_probe_v12_19.csv')
    postprocessors = {}
    policy = RadarPPOPolicyV11(len(V11_STATE_NAMES), len(ACTION_LIBRARY_V11), seed=RL_SEED_V11)
    policy.set_allowed_actions(np.arange(len(ACTION_LIBRARY_V11)))
    scratch_initial_state = policy.state_dict()
    if not np.all(policy.params['ba'] == 0):
        raise AssertionError('No calibrated initialization is permitted.')
    exploration_rows = select_split_rows_v11(train_rows, min(RL_CALIBRATION_CUBES_V11, len(train_rows)))
    normalizer = fit_exploration_normalizer_v12_10(normalizer_probe_rows_v12_18(train_rows))
    scaler = OriginalTaskObjectiveV12_18()
    optimizer = PPOAdamV12_11(policy.params, RL_LEARNING_RATE_V11)
    safe_save_npz_v11(MODEL_DIR_V11 / 'scratch_initial_policy_v12_10.npz', **scratch_initial_state)
    safe_write_json_v11(normalizer.as_dict(), MODEL_DIR_V11 / 'state_normalizer_v12.json')
    safe_write_json_v11(scaler.as_dict(), MODEL_DIR_V11 / 'task_objective_v12.json')
    rng = np.random.default_rng(RL_SEED_V11)
    training_records, shaping_records, pending, checkpoints = ([], [], [], [])
    train_cache, validation_cache = ({}, {})
    episodes_seen, update_count = (0, 0)

    def validation_checkpoint(pass_number):
        candidate, original, actions = evaluate_original_policy_rows_v12_18(validation_rows, labels, policy, normalizer, scaler, postprocessors, validation_cache)
        measurements = validation_measurements_v12_10(candidate, None, scaler)
        gate = original_validation_gate_v12_18(candidate)
        eligible = bool(gate.passed.all())
        key = (measurements['low_misses'], -measurements['mean_fixed_validation_reward'], measurements['sweep_misses'], measurements['mean_unmatched'], measurements['actual_bpcs'])
        seq = actions.sort_values(['record_name', 'block_idx']).groupby('record_name').action_index.apply(tuple)
        checkpoint_id = len(checkpoints)
        safe_write_csv_v11(candidate, TABLE_DIR_V11 / f'ppo_fixed_validation_components_checkpoint_{checkpoint_id:03d}_v12_19.csv')
        checkpoints.append(dict(**measurements, checkpoint_id=checkpoint_id, episodes_seen=episodes_seen, ppo_update_batches=update_count, **{'pass': float(pass_number)}, deployment_eligible=eligible, safety_eligible=eligible, validation_low_threshold_misses_candidate=measurements['low_misses'], validation_missed_labels_candidate=measurements['sweep_misses'], validation_low_threshold_misses_rdo=np.nan, validation_missed_labels_rdo=np.nan, missed_label_count_improvement=np.nan, mean_task_improvement_vs_rdo=np.nan, false_detection_cost_improvement=np.nan, RD_RA_log_power_mse_improvement=np.nan, mean_total_objective=float(candidate.total_objective.mean()), mean_task_loss=float(candidate.task_loss.mean()), mean_rate_excess=float(candidate.rate_excess.mean()), mean_realized_bpcs=measurements['actual_bpcs'], mean_false_detection_cost=measurements['mean_unmatched'], mean_RD_RA_log_power_mse=float(candidate[['RD_log_power_mse', 'RA_log_power_mse']].mean().sum()), unique_validation_action_sequences=int(seq.nunique()), checkpoint_uses_compressor_comparison=False))
        history = pd.DataFrame(checkpoints)
        safe_write_csv_v11(history, TABLE_DIR_V11 / 'ppo_fixed_validation_progress_v12_10.csv')
        safe_write_csv_v11(candidate, TABLE_DIR_V11 / 'latest_scratch_validation_per_cube_v12_10.csv')
        safe_write_csv_v11(actions, TABLE_DIR_V11 / 'latest_validation_actions_v12_11.csv')
        safe_save_npz_v11(MODEL_DIR_V11 / f'validation_checkpoint_{checkpoint_id:03d}_v12_19.npz', **policy.state_dict())
        plot_original_fixed_learning_v12_18(history)
        return dict(constraint_weights=V12_18_DUAL.weights.copy(), key=key, checkpoint_id=checkpoint_id, state=policy.state_dict(), candidate=candidate, original=original, actions=actions, gate=gate, **{'pass': float(pass_number)}, deployment_eligible=eligible)
    best = validation_checkpoint(0.0)
    for pass_index in range(RL_PASSES_V11):
        order = rng.permutation(len(train_rows))
        for episode_index, row_index in enumerate(order, 1):
            row = train_rows.iloc[int(row_index)]
            cube = load_cube_v6_1000(row)
            processor = _postprocessor_for_cube_v11(cube, postprocessors)
            labs = labels_for_cube_v6_1000(labels, row)
            reference = _original_source_reference_v12_18(cube, row, labs, processor, train_cache)
            trajectories, records, shaping = collect_same_cube_rollouts_v12_12(cube, row, policy, normalizer, processor, reference['post_original'], labs, scaler, reference['original_components'], {}, pass_index + 1, episode_index, episodes_seen + 1)
            episodes_seen += 1
            pending.extend(trajectories)
            training_records.extend(records)
            shaping_records.append(shaping)
            if len(pending) >= RL_BATCH_EPISODES_V11 * V12_12_ROLLOUTS_PER_CUBE or episode_index == len(order):
                update = ppo_update_v11(policy, optimizer, pending)
                update.update(V12_18_DUAL.update(pending))
                for record in training_records[-len(pending):]:
                    record.update(update)
                pending = []
                update_count += 1
                persist_training_progress_v12_11(policy, normalizer, training_records, shaping_records, best)
                if V12_10_VAL_EVERY_UPDATES and update_count % V12_10_VAL_EVERY_UPDATES == 0 and (episode_index < len(order)):
                    checkpoint = validation_checkpoint(pass_index + episode_index / len(order))
                    if checkpoint['key'] < best['key']:
                        best = checkpoint
                    persist_training_progress_v12_11(policy, normalizer, training_records, shaping_records, best)
            print(f"V12.19 ORIGINAL-only PPO pass {pass_index + 1}/{RL_PASSES_V11} cube {episode_index:03d}/{len(order):03d} | mean_bpcs={np.mean([r['realized_bpcs'] for r in records]):.3f} mean_low_miss={np.mean([r['low_threshold_labels_missed_total'] for r in records]):.2f} mean_unmatched={np.mean([r['new_false_detections_total'] for r in records]):.1f} mean_reward={np.mean([r['episode_return'] for r in records]):+.3f}")
            del cube, trajectories, records
            gc.collect()
        checkpoint = validation_checkpoint(pass_index + 1.0)
        if checkpoint['key'] < best['key']:
            best = checkpoint
        persist_training_progress_v12_11(policy, normalizer, training_records, shaping_records, best)
    safe_save_npz_v11(MODEL_DIR_V11 / 'last_scratch_ppo_policy_v12_10.npz', **policy.state_dict())
    policy.load_state_dict(best['state'])
    V12_18_DUAL.weights = best['constraint_weights'].copy()
    selected_state = policy.state_dict()
    safe_save_npz_v11(MODEL_DIR_V11 / 'taskaware_ppo_policy_v12.npz', **selected_state, action_library=np.asarray(ACTION_LIBRARY_V11, np.float32), allowed_action_indices=np.flatnonzero(policy.allowed_action_mask).astype(np.int64))
    safe_write_csv_v11(best['gate'], TABLE_DIR_V11 / 'ppo_original_only_validation_gate_v12_19.csv')
    safe_write_csv_v11(best['original'], TABLE_DIR_V11 / 'original_validation_reference_v12_19.csv')
    train_cache.clear()
    validation_cache.clear()
    print('Main PPO training/checkpoint selection COMPLETE. Building reporting-only comparators now.')
    calibration = calibrate_task_rdo_v11(exploration_rows, labels, postprocessors)
    fixed_action = calibration['selected_action']
    calibration['comparators_built_after_main_training'] = True
    calibration['training_dependency'] = False
    frozen, baseline_bc_log = make_frozen_imitation_baseline_v12_10(calibration, normalizer)
    frozen_initial_state = frozen.state_dict()
    if any((np.shares_memory(policy.params[k], frozen.params[k]) for k in policy.params)):
        raise AssertionError('Comparator parameters share storage with the main actor.')
    baseline_candidate, baseline_fixed, baseline_actions = evaluate_policy_rows_v11(validation_rows, labels, frozen, normalizer, fixed_action, scaler, postprocessors)
    frozen_measurements = validation_measurements_v12_10(baseline_candidate, None, scaler)
    rdo_measurements = validation_measurements_v12_10(baseline_fixed, None, scaler)
    validation_fft8 = evaluate_fft8_validation_reference_v11(validation_rows, labels, scaler, postprocessors)
    if any((not np.array_equal(policy.params[k], selected_state[k]) for k in policy.params)):
        raise AssertionError('Reporting-only comparator construction changed the selected main actor.')
    safe_write_csv_v11(baseline_candidate, TABLE_DIR_V11 / 'frozen_imitation_validation_per_cube_v12_10.csv')
    safe_write_csv_v11(baseline_actions, TABLE_DIR_V11 / 'frozen_imitation_validation_actions_v12_10.csv')
    safe_write_csv_v11(validation_fft8, TABLE_DIR_V11 / 'fft8_validation_reference_v12.csv')
    training_log = pd.DataFrame(training_records)
    shaping_log = pd.concat(shaping_records, ignore_index=True)
    history = pd.DataFrame(checkpoints)
    history['validation_low_threshold_misses_rdo'] = rdo_measurements['low_misses']
    history['validation_missed_labels_rdo'] = rdo_measurements['sweep_misses']
    history['missed_label_count_improvement'] = rdo_measurements['sweep_misses'] - history.sweep_misses
    history['false_detection_cost_improvement'] = rdo_measurements['mean_unmatched'] - history.mean_unmatched
    history['selected'] = history.checkpoint_id.eq(best['checkpoint_id'])
    bc_log = pd.DataFrame(columns=['bc_epoch', 'bc_loss', 'bc_accuracy', 'bc_macro_f1'])
    for frame, name in [(training_log, 'ppo_training_episodes_v12.csv'), (shaping_log, 'ppo_reward_shaping_checkpoints_v12.csv'), (history, 'ppo_validation_checkpoint_history_v12.csv'), (bc_log, 'behavior_cloning_initialization_v12.csv'), (best['candidate'], 'ppo_validation_candidate_v12.csv'), (baseline_fixed, 'task_rdo_validation_reference_v12.csv'), (best['gate'], 'ppo_validation_gate_v12.csv'), (best['actions'], 'ppo_validation_actions_v12.csv')]:
        safe_write_csv_v11(frame, TABLE_DIR_V11 / name)
    safe_write_json_v11(scaler.as_dict(), MODEL_DIR_V11 / 'task_objective_v12.json')
    safe_write_csv_v11(pd.DataFrame([dict(stage='main reward/constraints', uses_original=True, uses_labels=True, uses_compressor_baseline=False, uses_test_data=False), dict(stage='main checkpoint selection', uses_original=True, uses_labels=True, uses_compressor_baseline=False, uses_test_data=False), dict(stage='post-training comparator calibration', uses_original=True, uses_labels=True, uses_compressor_baseline=True, uses_test_data=False), dict(stage='deployment action selection', uses_original=False, uses_labels=False, uses_compressor_baseline=False, uses_test_data=True)]), TABLE_DIR_V11 / 'postprocessing_and_label_leakage_audit_v12.csv')
    return dict(validation_probe_count=len(validation_rows), frozen_initial_state=frozen_initial_state, scratch_initial_state=scratch_initial_state, frozen_baseline_measurements=frozen_measurements, fixed_rdo_measurements=rdo_measurements, baseline_bc_log=baseline_bc_log, constraint_weights=best['constraint_weights'].tolist(), main_imitation_epochs=0, rollouts_per_cube=V12_12_ROLLOUTS_PER_CUBE, policy=policy, normalizer=normalizer, scaler=scaler, fixed_action=fixed_action, calibration=calibration, training_log=training_log, shaping_log=shaping_log, bc_log=bc_log, validation_candidate=best['candidate'], validation_fixed=baseline_fixed, validation_original=best['original'], validation_fft8=validation_fft8, validation_actions=best['actions'], validation_gate=best['gate'], validation_checkpoint_history=history, deployment_eligible=best['deployment_eligible'], selected_pass=best['pass'], postprocessors=postprocessors, preflight=preflight, comparators_built_after_main_training=True, checkpoint_selection_uses_compressor_comparison=False, training_uses_compressor_comparison=False)
V12_5_FALSE_ALLOWANCE_PER_THRESHOLD = float(os.environ.get('RDO_V125_FALSE_ALLOWANCE_PER_THRESHOLD', '2'))
V12_5_VIS_CFAR_THRESHOLD = float(os.environ.get('RDO_V125_VIS_CFAR_THRESHOLD', '8.0'))
if V12_5_FALSE_ALLOWANCE_PER_THRESHOLD < 0 or V12_5_VIS_CFAR_THRESHOLD <= 0:
    raise ValueError('V12.5 absolute false allowance must be nonnegative and visualization CFAR > 0.')
V12_5_LOW_MISS_HARD = 5000.0
V12_5_AGG_MISS_PER_COUNT = 500.0

def _associate_labels_and_peaks_v12_5(label_bins: Sequence[Tuple[int, int]], peak_mask: np.ndarray, gate: Tuple[int, int]):
    """Many-to-many geometric association used by the V12.5 metric contract.

    `label_has_peak[i]` is true when at least one peak is inside label i's gate.
    `peak_has_label[j]` is true when peak j lies inside at least one GT gate.
    No denominator or one-to-one assignment is used.
    """
    peaks = _peak_points(peak_mask)
    n_labels = len(label_bins)
    label_has_peak = np.zeros(n_labels, dtype=bool)
    peak_has_label = np.zeros(len(peaks), dtype=bool)
    if n_labels == 0 or len(peaks) == 0:
        return (label_has_peak, peak_has_label, peaks)
    gr, gc = (max(int(gate[0]), 1), max(int(gate[1]), 1))
    for i, (r, c) in enumerate(label_bins):
        dr = np.abs(peaks[:, 0] - int(r))
        dc = np.abs(peaks[:, 1] - int(c))
        inside = (dr <= gr) & (dc <= gc)
        label_has_peak[i] = bool(np.any(inside))
        peak_has_label |= inside
    return (label_has_peak, peak_has_label, peaks)

def precompute_original_cfar_count_sweep_v11_2(map_original_db: np.ndarray, labels: pd.DataFrame, ranges: np.ndarray, second_axis: np.ndarray, map_type: str, threshold_factors: Sequence[float], gate: Tuple[int, int]) -> Dict[str, Any]:
    map_type = str(map_type).upper()
    label_bins, label_source_indices, excluded = _evaluable_label_bins_v12(labels, ranges, second_axis, map_type)
    power = as_linear_power(map_original_db, assume_db=True)
    masks = precompute_nested_cfar_peak_masks_v11_2(power, threshold_factors)
    thresholds = sorted((float(v) for v in threshold_factors))
    per = {}
    for t in thresholds:
        positive = masks[t]['positive']
        peaks = masks[t]['peaks']
        eligible, peak_associated, points = _associate_labels_and_peaks_v12_5(label_bins, peaks, gate)
        per[t] = {'cells0': positive, 'peaks0': peaks, 'eligible0': eligible, 'supported0': int(np.sum(eligible)), 'npeak0': int(len(points)), 'unmatched0': int(np.sum(~peak_associated)), 'num_positive0': int(np.sum(positive))}
    return {'map_type': map_type, 'label_bins': label_bins, 'label_source_indices': label_source_indices, 'num_input_visible_labels': int(len(labels)), 'num_evaluable_labels': int(len(label_bins)), 'num_excluded_out_of_axis_labels': int(excluded), 'num_visible_labels': int(len(label_bins)), 'threshold_factors': thresholds, 'gate': tuple(map(int, gate)), 'per_threshold': per, 'ranges': np.asarray(ranges, dtype=float), 'second_axis': np.asarray(second_axis, dtype=float), 'metric_protocol': 'absolute_count_any_peak_gt_association_v12_5'}

def evaluate_cfar_count_sweep_method_v11_2(original_cache: Dict[str, Any], map_method_db: np.ndarray) -> pd.DataFrame:
    power = as_linear_power(map_method_db, assume_db=True)
    masks = precompute_nested_cfar_peak_masks_v11_2(power, original_cache['threshold_factors'])
    label_bins = original_cache['label_bins']
    gate = original_cache['gate']
    rows = []
    for t in original_cache['threshold_factors']:
        cached = original_cache['per_threshold'][float(t)]
        positive = masks[float(t)]['positive']
        peaks = masks[float(t)]['peaks']
        method_label_has_peak, method_peak_associated, method_points = _associate_labels_and_peaks_v12_5(label_bins, peaks, gate)
        eligible = np.asarray(cached['eligible0'], dtype=bool)
        preserved_mask = eligible & method_label_has_peak
        supported0 = int(np.sum(eligible))
        preserved = int(np.sum(preserved_mask))
        missed = int(max(supported0 - preserved, 0))
        unmatched_method = int(np.sum(~method_peak_associated))
        rows.append({'map_type': original_cache['map_type'], 'threshold_factor': float(t), 'cfar_threshold_db': float(10.0 * np.log10(max(float(t), EPS))), 'num_input_visible_labels': int(original_cache['num_input_visible_labels']), 'num_excluded_out_of_axis_labels': int(original_cache['num_excluded_out_of_axis_labels']), 'num_visible_labels': int(original_cache['num_evaluable_labels']), 'num_labels_supported_original': supported0, 'num_labels_supported_method': int(np.sum(method_label_has_peak)), 'num_labels_preserved_from_original': preserved, 'num_labels_missed_from_original': missed, 'num_cfar_positive_cells_original': int(cached['num_positive0']), 'num_cfar_positive_cells_method': int(np.sum(positive)), 'num_cfar_peaks_original': int(cached['npeak0']), 'num_cfar_peaks_method': int(len(method_points)), 'num_label_unmatched_peaks_original': int(cached['unmatched0']), 'num_label_unmatched_peaks_method': unmatched_method, 'num_new_false_detections_method': unmatched_method, 'original_label_support_fraction': np.nan, 'method_label_support_fraction': np.nan, 'conditional_label_preservation': np.nan, 'original_label_unmatched_fraction': np.nan, 'method_label_unmatched_fraction': np.nan, 'new_false_detection_fraction_method': np.nan, 'new_vs_original_fraction': np.nan, 'peak_preservation_vs_original': np.nan, 'num_det_original': int(cached['npeak0']), 'num_det_method': int(len(method_points)), 'preservation': np.nan, 'new_fraction': np.nan, 'metric_protocol': 'absolute_count_only'})
    return pd.DataFrame(rows)

def _v12_5_false_allowance(detail: Dict[str, Any]) -> float:
    n_thresholds = max(int(detail.get('training_cfar_threshold_count', len(_nearest_training_thresholds_v11()))), 1)
    return float(V12_5_FALSE_ALLOWANCE_PER_THRESHOLD * n_thresholds)

def calibrate_task_rdo_v11(rows, labels, postprocessors):
    """Calibrate the same codec/action library; do not restrict PPO's actions."""
    base = _impl_calibrate_task_rdo_v11_c18_l729(rows, labels, postprocessors)
    states, actions = ([], [])
    for _, row in rows.iterrows():
        cube = load_cube_v6_1000(row)
        x, a = _teacher_state_trace_v11(cube, base['selected_action_index'])
        states.append(x)
        actions.append(a)
    base['teacher_states'], base['teacher_actions'] = (np.concatenate(states), np.concatenate(actions))
    base['policy_action_indices'] = np.arange(len(ACTION_LIBRARY_V11), dtype=np.int64)
    return base

def validation_gate_v11(candidate, fixed=None, fft8=None):
    """Compatibility acceptance API; compressor comparator arguments are unused."""
    return original_validation_gate_v12_18(candidate)

def plot_ppo_training_v11(training_log: pd.DataFrame, save_path: Path):
    d = training_log.reset_index(drop=True).copy()
    if len(d) == 0:
        return None
    x = np.arange(1, len(d) + 1)
    fig, axes = plt.subplots(3, 2, figsize=(14.5, 12.0), sharex=True)
    ax = axes[0, 0]
    rate = pd.to_numeric(d.realized_bpcs, errors='coerce')
    ax.plot(x, rate, alpha=0.45, label='episode')
    ax.plot(x, rate.rolling(10, min_periods=1).mean(), linewidth=2, label='10-allocation mean')
    ax.axhline(RL_RATE_TARGET_BPCS_V11, linestyle='--', label='rate target')
    ax.set_ylabel('Packed bits per complex sample [bpcs]')
    ax.set_title('Communication rate')
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax = axes[0, 1]
    reward = pd.to_numeric(d.episode_return, errors='coerce')
    ax.plot(x, reward, alpha=0.4, label='raw absolute-count reward')
    ax.plot(x, reward.rolling(10, min_periods=1).mean(), linewidth=2, label='10-allocation mean')
    ax.axhline(0, linestyle='--', linewidth=1)
    ax.set_ylabel('Raw constrained reward')
    ax.set_title('PPO task/compression reward')
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax = axes[1, 0]
    if 'true_labels_missed_total' in d:
        ax.plot(x, pd.to_numeric(d.true_labels_missed_total, errors='coerce'), label='sweep missed object/map/threshold instances')
    if 'low_threshold_labels_missed_total' in d:
        ax.plot(x, pd.to_numeric(d.low_threshold_labels_missed_total, errors='coerce'), label='lowest-threshold missed labels')
    ax.axhline(0, linestyle='--', linewidth=1)
    ax.set_ylabel('GT label instances [count]')
    ax.set_title('Absolute object-preservation diagnostics')
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax = axes[1, 1]
    if 'false_peak_excess' in d:
        ax.plot(x, pd.to_numeric(d.false_peak_excess, errors='coerce'), label='absolute excess vs original')
    if 'method_unmatched_false_peaks' in d:
        ax.plot(x, pd.to_numeric(d.method_unmatched_false_peaks, errors='coerce'), alpha=0.55, label='compressed unmatched false peaks')
    ax.axhline(0, linestyle='--', linewidth=1)
    ax.set_ylabel('Unmatched false peaks [count]')
    ax.set_title('Absolute outside-GT peak counts')
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax = axes[2, 0]
    for col, label in (('policy_entropy', 'policy entropy'), ('approximate_kl', 'approx. KL'), ('clip_fraction', 'clip fraction')):
        if col in d:
            ax.plot(x, pd.to_numeric(d[col], errors='coerce'), label=label)
    ax.set_ylabel('Optimization diagnostic')
    ax.set_xlabel(f'Sampled allocation ({V12_12_ROLLOUTS_PER_CUBE} per cube visit)')
    ax.set_title('PPO update diagnostics (not monotonic objectives)')
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax = axes[2, 1]
    if 'unique_actions' in d:
        ax.plot(x, pd.to_numeric(d.unique_actions, errors='coerce'), label='unique actions per cube')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    corr_ax = ax.twinx()
    corr_ax.set_ylim(-1.05, 1.05)
    corr_ax.set_ylabel('Block bpcs vs concentration: Pearson r')
    if 'rate_spectral_correlation' in d:
        corr_ax.plot(x, pd.to_numeric(d.rate_spectral_correlation, errors='coerce'), color='tab:orange', alpha=0.65)
    ax.set_ylabel('Adaptivity diagnostic')
    ax.set_xlabel(f'Sampled allocation ({V12_12_ROLLOUTS_PER_CUBE} per cube visit)')
    ax.set_title('Content-adaptive behavior')
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.suptitle(f'V12.19 raw coherent-transform PPO — {RL_RUN_MODE_V11}', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_publication_figure(fig, Path(save_path), dpi=220)
    plt.close(fig)
    return Path(save_path)

def plot_v12_3_training_diagnostics(training_log: pd.DataFrame, save_dir: Path) -> List[Path]:
    path = Path(save_dir) / 'v12_5_training_absolute_count_diagnostics.png'
    plot_ppo_training_v11(training_log, path)
    return [path]
from matplotlib.ticker import MaxNLocator
CFAR_THRESHOLD_FACTORS_V8 = np.asarray([1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0], dtype=float)
if 'RDO_RL_PPO_EPOCHS' not in os.environ:
    RL_PPO_EPOCHS_V11 = max(int(RL_PPO_EPOCHS_V11), 4)
_standardize_rd_map_unbounded_v12_6 = _impl_standardize_rd_map_c4_l153
_standardize_ra_map_unbounded_v12_6 = _impl_standardize_ra_map_c4_l181
V12_6_RANGE_MIN_M = float(MATLAB_RADAR_SPEC.get('MinimumRangeM', 1.0))
V12_6_RANGE_MAX_M = float(MATLAB_RADAR_SPEC.get('MaximumRangeM', 150.0))
V12_6_AZ_HALF_FOV_DEG = 0.5 * float(MATLAB_RADAR_SPEC.get('FieldOfViewDeg', 120.0))

def standardize_rd_map(RD, ranges, vel):
    data, r, v = _standardize_rd_map_unbounded_v12_6(RD, ranges, vel)
    keep = np.isfinite(r) & (r >= V12_6_RANGE_MIN_M) & (r <= V12_6_RANGE_MAX_M)
    if not np.any(keep):
        raise ValueError("RD map has no bins inside the radar's declared physical range ROI.")
    return (np.asarray(data)[keep, :], np.asarray(r)[keep], np.asarray(v))

def standardize_ra_map(RA, ranges, angle_axis_deg):
    data, r, a = _standardize_ra_map_unbounded_v12_6(RA, ranges, angle_axis_deg)
    keep_r = np.isfinite(r) & (r >= V12_6_RANGE_MIN_M) & (r <= V12_6_RANGE_MAX_M)
    keep_a = np.isfinite(a) & (np.abs(a) <= V12_6_AZ_HALF_FOV_DEG + 1e-09)
    if not np.any(keep_r) or not np.any(keep_a):
        raise ValueError("RA map has no bins inside the radar's declared physical range/FOV ROI.")
    return (np.asarray(data)[np.ix_(keep_r, keep_a)], np.asarray(r)[keep_r], np.asarray(a)[keep_a])

def precompute_nested_cfar_peak_masks_v11_2(power_map, threshold_factors, guard=(1, 1), train=(4, 4), nms_radius=(2, 2)):
    power = np.maximum(np.asarray(power_map, dtype=float), 0.0)
    gr, gc = map(int, guard)
    tr, tc = map(int, train)
    rr, cc = map(int, nms_radius)
    if SCIPY_AVAILABLE and scipy_maximum_filter is not None:
        kr, kc = (2 * (tr + gr) + 1, 2 * (tc + gc) + 1)
        kernel = np.ones((kr, kc), dtype=float)
        kernel[tr:tr + 2 * gr + 1, tc:tc + 2 * gc + 1] = 0.0
        full_training_cells = float(kernel.sum())
        noise_sum = scipy_convolve(power, kernel, mode='constant', cval=0.0)
        count = scipy_convolve(np.ones_like(power), kernel, mode='constant', cval=0.0)
        valid = np.isclose(count, full_training_cells, rtol=0.0, atol=1e-09)
        noise = noise_sum / max(full_training_cells, 1.0)
        cfar_score = power / (noise + EPS)
        local_maximum = power >= scipy_maximum_filter(power, size=(2 * rr + 1, 2 * cc + 1), mode='nearest')
    else:
        cfar_score = np.zeros_like(power)
        valid = np.zeros_like(power, dtype=bool)
        local_maximum = np.zeros_like(power, dtype=bool)
        rpad, cpad = (tr + gr, tc + gc)
        for i in range(rpad, power.shape[0] - rpad):
            for j in range(cpad, power.shape[1] - cpad):
                patch = power[i - rpad:i + rpad + 1, j - cpad:j + cpad + 1].copy()
                patch[tr:tr + 2 * gr + 1, tc:tc + 2 * gc + 1] = np.nan
                cfar_score[i, j] = power[i, j] / (np.nanmean(patch) + EPS)
                local = power[max(0, i - rr):i + rr + 1, max(0, j - cc):j + cc + 1]
                local_maximum[i, j] = power[i, j] >= np.max(local)
                valid[i, j] = True
    base_peaks = _collapse_peak_plateaus_v9(power, local_maximum & valid & (power > 0.0), nms_radius=(rr, cc))
    output = {}
    previous = None
    for threshold in sorted((float(value) for value in threshold_factors)):
        positive = valid & (cfar_score > threshold)
        peaks = base_peaks & positive
        if previous is not None and np.any(peaks & ~previous):
            raise AssertionError('CFAR nesting invariant failed.')
        previous = peaks
        output[threshold] = {'positive': positive.astype(bool), 'peaks': peaks.astype(bool), 'cfar_score': cfar_score, 'power': power, 'valid': valid.astype(bool)}
    return output

def _exact_full_support_meta_v12_6(cube, cube_hat, q, method_name, header_bits=128.0):
    n = int(np.asarray(cube).size)
    bits = float(2 * int(q) * n + header_bits)
    bpcs = bits / max(n, 1)
    return {'method': method_name, 'realized_bits': bits, 'realized_bpcs': bpcs, 'payload_bits': float(2 * int(q) * n), 'support_bits': 0.0, 'header_bits': float(header_bits), 'bits_per_real_component': int(q), 'cube_nmse': block_nmse(cube, cube_hat), 'rel_error': float(np.linalg.norm((cube - cube_hat).reshape(-1)) / (np.linalg.norm(np.asarray(cube).reshape(-1)) + EPS)), 'compression_ratio_vs_v43_int16_iq_32bpc': float(32.0 / max(bpcs, EPS)), 'memory_ratio_proxy_64b_complex': float(64.0 / max(bpcs, EPS)), 'rate_accounting': '2*q*N payload + 128-bit scale/header; implicit full support'}

def compress_cube_fft_full_support_multi(cube, q_by_method):
    """Reuse one FFT while charging no support map for implicit full support."""
    cube = np.asarray(cube)
    spectrum = np.fft.fftn(cube, axes=(0, 1, 2))
    output = {}
    for method_name, q in q_by_method.items():
        spectrum_hat, _ = uniform_quantize_complex(spectrum, int(q))
        cube_hat = np.fft.ifftn(spectrum_hat, axes=(0, 1, 2)).astype(cube.dtype, copy=False)
        meta = _exact_full_support_meta_v12_6(cube, cube_hat, int(q), str(method_name))
        output[str(method_name)] = (cube_hat, pd.DataFrame([meta]), meta)
    return output

def compress_cube_raw_linear(cube, q=8, method_name='RawLinear_8bit_per_component'):
    cube_hat, _ = uniform_quantize_complex(cube, int(q))
    cube_hat = cube_hat.astype(cube.dtype, copy=False)
    meta = _exact_full_support_meta_v12_6(cube, cube_hat, q, method_name)
    return (cube_hat, pd.DataFrame([meta]), meta)

def _map_triplet_v12_6(post, map_type):
    if map_type == 'RD':
        return standardize_rd_map(post['RD'], post['ranges'], post['vel'])
    return standardize_ra_map(post['RA'], post['ranges'], post['angle_axis_deg'])

def _append_label_audit_v12_6(rows, cache, method_map, labs, method, record_name, scene_name):
    masks = precompute_nested_cfar_peak_masks_v11_2(as_linear_power(method_map, assume_db=True), cache['threshold_factors'])
    for threshold in cache['threshold_factors']:
        original_eligible = np.asarray(cache['per_threshold'][float(threshold)]['eligible0'], bool)
        method_has, _, _ = _associate_labels_and_peaks_v12_5(cache['label_bins'], masks[float(threshold)]['peaks'], cache['gate'])
        for local_i, source_i in enumerate(cache['label_source_indices']):
            label = labs.iloc[int(source_i)]
            eligible = bool(original_eligible[local_i])
            preserved = bool(eligible and method_has[local_i]) if method != 'Original' else eligible
            rows.append({'record_name': record_name, 'scene_name': scene_name, 'map_type': cache['map_type'], 'method': method, 'threshold_factor': float(threshold), 'cfar_threshold_db': float(10 * np.log10(float(threshold))), 'object_id': str(label.get('object_id', '')), 'actor_id': label.get('actor_id', np.nan), 'object_type': str(label.get('object_type', 'unknown')), 'range_m': float(label.get('range_m', np.nan)), 'radial_velocity_mps': float(label.get('radial_velocity_mps', np.nan)), 'azimuth_deg': float(label.get('azimuth_deg', np.nan)), 'original_detectable': eligible, 'preserved': preserved, 'missed_after_compression': bool(eligible and (not preserved)), 'status': 'preserved' if preserved else 'missed' if eligible else 'original_not_detected_excluded', 'counting_rule': 'one GT object counts once; any peak inside its gate preserves it'})

def _cube_quality_row_v12_6(record_name, scene_name, method, cube, cube_hat, meta, post0, post1):
    rd0, _, _ = _map_triplet_v12_6(post0, 'RD')
    rd1, _, _ = _map_triplet_v12_6(post1, 'RD')
    ra0, _, _ = _map_triplet_v12_6(post0, 'RA')
    ra1, _, _ = _map_triplet_v12_6(post1, 'RA')
    original_bits = float(32 * cube.size)
    bits = original_bits if method == 'Original' else float(meta['realized_bits'])
    bpcs = bits / cube.size
    error = np.asarray(cube) - np.asarray(cube_hat)
    nmse = float(np.vdot(error.reshape(-1), error.reshape(-1)).real / (np.vdot(np.asarray(cube).reshape(-1), np.asarray(cube).reshape(-1)).real + EPS))
    amp_ratio = float(np.mean(np.abs(cube_hat)) / (np.mean(np.abs(cube)) + EPS))
    peak_ratio = float(np.max(np.abs(cube_hat)) / (np.max(np.abs(cube)) + EPS))
    return {'record_name': record_name, 'scene_name': scene_name, 'method': method, 'method_display': V12_6_DISPLAY[method], 'source_bits': original_bits, 'realized_bits': bits, 'realized_bpcs': bpcs, 'compressed_MiB_per_cube': bits / 8.0 / 2 ** 20, 'saved_MiB_per_cube': (original_bits - bits) / 8.0 / 2 ** 20, 'compression_ratio_vs_32bpc': original_bits / max(bits, EPS), 'cube_nmse': nmse, 'cube_nmse_db': float(10 * np.log10(max(nmse, EPS))), 'relative_l2_error': float(np.sqrt(max(nmse, 0.0))), 'mean_absolute_amplitude_ratio': amp_ratio, 'peak_amplitude_ratio': peak_ratio, 'amplitude_attenuation_percent': 100.0 * (1.0 - amp_ratio), 'RD_log_power_mse_physical_roi': _normalized_log_power_mse_v11(rd0, rd1), 'RA_log_power_mse_physical_roi': _normalized_log_power_mse_v11(ra0, ra1), 'shield_selected_candidate': str(meta.get('shield_selected_candidate', 'n/a'))}

def _relative_db_map_v12_6(data, reference_peak=None):
    power = np.maximum(as_linear_power(data, assume_db=True), EPS)
    peak = float(np.max(power)) if reference_peak is None else float(reference_peak)
    return 10 * np.log10(power / max(peak, EPS))

def _status_vectors_v12_6(cache, method_map, threshold):
    c = cache['per_threshold'][float(threshold)]
    eligible = np.asarray(c['eligible0'], bool)
    masks = precompute_nested_cfar_peak_masks_v11_2(as_linear_power(method_map, assume_db=True), [float(threshold)])
    method_has, _, _ = _associate_labels_and_peaks_v12_5(cache['label_bins'], masks[float(threshold)]['peaks'], cache['gate'])
    return (eligible, method_has)

def _overlay_labels_v12_6(ax, labs, cache, eligible, method_has, map_type, original=False):
    for local_i, source_i in enumerate(cache['label_source_indices']):
        label = labs.iloc[int(source_i)]
        if not bool(eligible[local_i]):
            color, marker = ('0.65', 'x')
        elif original or bool(method_has[local_i]):
            color, marker = ('lime', 'o')
        else:
            color, marker = ('red', 'X')
        x = float(label.radial_velocity_mps if map_type == 'RD' else label.azimuth_deg)
        y = float(label.range_m)
        if marker == 'x':
            ax.scatter([x], [y], s=55, c=color, marker=marker, linewidths=1.4, zorder=5)
        else:
            ax.scatter([x], [y], s=55, facecolors='none' if marker == 'o' else color, edgecolors=color, marker=marker, linewidths=1.4, zorder=5)
        actor = label.get('actor_id', '')
        short = str(label.get('object_type', 'obj'))[:8]
        ax.annotate(f'{short}:{actor}', (x, y), xytext=(3, 3), textcoords='offset points', color=color, fontsize=6.5, zorder=6)

def _ra_cfar_point_cloud_v12_6(post, threshold):
    """Convert RA CFAR/local maxima to a Cartesian x/y detection cloud."""
    ra, ranges, azimuth = _map_triplet_v12_6(post, 'RA')
    masks = precompute_nested_cfar_peak_masks_v11_2(as_linear_power(ra, assume_db=True), [float(threshold)])
    peaks = np.argwhere(masks[float(threshold)]['peaks'])
    if len(peaks) == 0:
        return np.zeros((0, 2), dtype=float)
    r = ranges[peaks[:, 0]]
    theta = np.radians(azimuth[peaks[:, 1]])
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])

def _impl_run_preflight_debug_v11_c32_l787():
    """V12.6 synthetic invariants, sized for complete CA-CFAR windows."""
    rows = []

    def add(name, passed, value, insight):
        rows.append({'check': name, 'passed': bool(passed), 'value': value, 'insight': insight})
    labels = [(12, 12)]
    multiple = np.zeros((25, 25), dtype=bool)
    multiple[11, 12] = True
    multiple[13, 13] = True
    one = np.zeros((25, 25), dtype=bool)
    one[13, 13] = True
    one_label_many, assoc_many, _ = _associate_labels_and_peaks_v12_5(labels, multiple, (2, 2))
    one_label_one, assoc_one, _ = _associate_labels_and_peaks_v12_5(labels, one, (2, 2))
    add('one_object_counts_once_with_multiple_peaks', int(one_label_many.sum()) == 1, int(one_label_many.sum()), 'Several scattering peaks cannot inflate the preserved-label count.')
    add('one_remaining_peak_preserves_object', int(one_label_one.sum()) == 1, int(one_label_one.sum()), 'Removing some peaks does not lose an object while any in-gate peak remains.')
    add('associated_peak_is_not_unmatched', bool(assoc_one.all()), int((~assoc_one).sum()), 'A peak inside any GT gate is not an unmatched false peak.')
    with_false = one.copy()
    with_false[18, 18] = True
    _, assoc_false, _ = _associate_labels_and_peaks_v12_5(labels, with_false, (2, 2))
    add('truly_unmatched_peak_counted_false', int((~assoc_false).sum()) == 1, int((~assoc_false).sum()), 'A peak outside every GT gate is counted once as unmatched.')
    power = np.ones((25, 25), dtype=float)
    power[0, 0] = 1000000.0
    power[12, 12] = 1000000.0
    masks = precompute_nested_cfar_peak_masks_v11_2(power, [2.0, 4.0])
    add('incomplete_edge_window_rejected', not bool(masks[2.0]['positive'][0, 0]), bool(masks[2.0]['positive'][0, 0]), 'CFAR is evaluated only with the complete training-cell stencil.')
    add('valid_center_target_detected', bool(masks[2.0]['positive'][12, 12]), bool(masks[2.0]['positive'][12, 12]), 'A strong target with full training support is detected.')
    add('cfar_sweep_is_nested', not np.any(masks[4.0]['peaks'] & ~masks[2.0]['peaks']), 'nested', 'Raising the threshold cannot create a new peak.')
    tiny = (np.arange(64, dtype=np.float32).reshape(2, 4, 8) + 1j * np.arange(64, dtype=np.float32).reshape(2, 4, 8)).astype(np.complex64)
    _, _, raw_meta = compress_cube_raw_linear(tiny, q=8)
    expected = 16.0 + 128.0 / tiny.size
    add('raw_linear_8bit_exact_rate', abs(raw_meta['realized_bpcs'] - expected) < 1e-12, raw_meta['realized_bpcs'], 'Eight bits for I and eight for Q plus the explicit 128-bit header.')
    fft_meta = compress_cube_fft_full_support_multi(tiny, {'FFT8': 8})['FFT8'][2]
    add('fft8_full_support_exact_rate', abs(fft_meta['realized_bpcs'] - expected) < 1e-12, fft_meta['realized_bpcs'], 'Full FFT support is implicit; no one-bit mask per coefficient is charged.')
    add('physical_cfar_thresholds_above_background', float(np.min(CFAR_THRESHOLD_FACTORS_V8)) > 1.0, float(np.min(CFAR_THRESHOLD_FACTORS_V8)), 'No below-background threshold is reported as CFAR.')
    report = pd.DataFrame(rows)
    safe_write_csv_v11(report, TABLE_DIR_V11 / 'preflight_debug_checks_v12_6.csv')
    if not bool(report.passed.all()):
        raise AssertionError('V12.6 preflight failed: ' + str(report.loc[~report.passed, 'check'].tolist()))
    print('V12.6 metric/rate/CFAR preflight: PASS')
    display(report)
    return report
V12_7_METHODS = ['Original', 'PPO', 'CalibratedRDO', 'FFT14']
V12_7_DISPLAY = {'Original': 'Original uncompressed (32 b/complex sample)', 'PPO': 'Raw PPO (no refinement)', 'CalibratedRDO': 'Calibrated uniform RDO', 'FFT14': 'FFT scalar quantization, 14 b/component (28-bit complex)'}
V12_7_COLORS = {'Original': 'black', 'PPO': '#2a9d55', 'CalibratedRDO': '#f58518', 'FFT14': '#4c78a8'}
V12_6_DISPLAY = V12_7_DISPLAY
V12_7_ALIGNMENT_GAIN_WEIGHT = float(os.environ.get('RDO_V127_ALIGNMENT_GAIN_WEIGHT', '18.0'))
V12_7_WEIGHTED_DEFICIT_GAIN = float(os.environ.get('RDO_V127_WEIGHTED_DEFICIT_GAIN', '10.0'))
V12_7_ALIGNMENT_TASK_LOSS_WEIGHT = float(os.environ.get('RDO_V127_ALIGNMENT_TASK_LOSS_WEIGHT', '2.5'))
V12_7_BASE_CFAR_THRESHOLD_FACTORS = np.asarray(CFAR_THRESHOLD_FACTORS_V8, dtype=float).copy()
V12_7_RD_PRESERVATION_THRESHOLD_FACTORS = np.unique(np.concatenate([V12_7_BASE_CFAR_THRESHOLD_FACTORS, np.asarray([192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152], dtype=float)]))

def _alignment_metrics_v12_7(sweep):
    """Contiguous exact-match endpoint and a low-to-high weighted miss count."""
    ordered = sweep.sort_values('cfar_threshold_db').reset_index(drop=True)
    if ordered.empty:
        return {'alignment_endpoint_db': np.nan, 'alignment_span_db': 0.0, 'aligned_threshold_count': 0, 'weighted_label_deficit': 0.0}
    missed = pd.to_numeric(ordered['num_labels_missed_from_original'], errors='coerce').fillna(0.0).to_numpy(float)
    db = pd.to_numeric(ordered['cfar_threshold_db'], errors='coerce').to_numpy(float)
    first_miss = np.flatnonzero(missed > 0.0)
    aligned_count = int(first_miss[0]) if len(first_miss) else int(len(ordered))
    endpoint_index = max(0, aligned_count - 1)
    endpoint = float(db[endpoint_index])
    span = float(max(endpoint - db[0], 0.0)) if aligned_count else 0.0
    weights = np.linspace(1.0, 2.0, len(ordered), dtype=float)
    weighted_deficit = float(np.dot(weights, missed))
    return {'alignment_endpoint_db': endpoint, 'alignment_span_db': span, 'aligned_threshold_count': aligned_count, 'weighted_label_deficit': weighted_deficit}

def _map_detection_costs_v11(cache, method_map):
    """V12.7 object counts plus contiguous CFAR-alignment diagnostics."""
    sweep = evaluate_cfar_count_sweep_method_v11_2(cache, method_map)
    low = sweep.loc[sweep.threshold_factor.idxmin()]
    supported = float(sweep.num_labels_supported_original.sum())
    preserved = float(sweep.num_labels_preserved_from_original.sum())
    metrics = _alignment_metrics_v12_7(sweep)
    return {'label_active': 1.0 if supported > 0 else 0.0, 'labels_supported_original': supported, 'labels_preserved': preserved, 'label_miss': max(supported - preserved, 0.0), 'new_false': float(sweep.num_label_unmatched_peaks_method.sum()), 'original_unmatched': float(sweep.num_label_unmatched_peaks_original.sum()), 'original_peaks': float(sweep.num_cfar_peaks_original.sum()), 'low_labels_supported_original': float(low.num_labels_supported_original), 'low_labels_preserved': float(low.num_labels_preserved_from_original), 'low_labels_missed': float(low.num_labels_missed_from_original), 'low_unmatched_method': float(low.num_label_unmatched_peaks_method), 'low_unmatched_original': float(low.num_label_unmatched_peaks_original), 'low_threshold': float(low.threshold_factor), 'low_threshold_db': float(low.cfar_threshold_db), 'excluded_out_of_axis_labels': float(low.num_excluded_out_of_axis_labels), 'num_thresholds': int(len(sweep)), **metrics}

def radar_task_components_v11(post_original, post_method, labels_cube):
    """Task reward: absolute object counts, false peaks, maps, and alignment span."""
    thresholds = _nearest_training_thresholds_v11()
    rd0, rr, vv = standardize_rd_map(post_original['RD'], post_original['ranges'], post_original['vel'])
    rd1, _, _ = standardize_rd_map(post_method['RD'], post_method['ranges'], post_method['vel'])
    ra0, rar, aa = standardize_ra_map(post_original['RA'], post_original['ranges'], post_original['angle_axis_deg'])
    ra1, _, _ = standardize_ra_map(post_method['RA'], post_method['ranges'], post_method['angle_axis_deg'])
    rd = _map_detection_costs_v11(precompute_original_cfar_count_sweep_v11_2(rd0, labels_cube, rr, vv, 'RD', thresholds, CFAR_LABEL_GATE_RD_V8), rd1)
    ra = _map_detection_costs_v11(precompute_original_cfar_count_sweep_v11_2(ra0, labels_cube, rar, aa, 'RA', thresholds, CFAR_LABEL_GATE_RA_V8), ra1)
    components = {'RD_log_power_mse': _normalized_log_power_mse_v11(rd0, rd1), 'RA_log_power_mse': _normalized_log_power_mse_v11(ra0, ra1), 'RD_true_label_miss': float(rd['label_miss']), 'RA_true_label_miss': float(ra['label_miss']), 'new_false_detection_cost': float(rd['new_false'] + ra['new_false'])}
    detail = {'RD_label_active': rd['label_active'], 'RA_label_active': ra['label_active'], 'RD_labels_supported_original': rd['labels_supported_original'], 'RA_labels_supported_original': ra['labels_supported_original'], 'RD_labels_preserved': rd['labels_preserved'], 'RA_labels_preserved': ra['labels_preserved'], 'RD_labels_missed': rd['label_miss'], 'RA_labels_missed': ra['label_miss'], 'RD_low_threshold_labels_supported_original': rd['low_labels_supported_original'], 'RA_low_threshold_labels_supported_original': ra['low_labels_supported_original'], 'RD_low_threshold_labels_preserved': rd['low_labels_preserved'], 'RA_low_threshold_labels_preserved': ra['low_labels_preserved'], 'RD_low_threshold_labels_missed': rd['low_labels_missed'], 'RA_low_threshold_labels_missed': ra['low_labels_missed'], 'low_threshold_labels_supported_total': rd['low_labels_supported_original'] + ra['low_labels_supported_original'], 'low_threshold_labels_preserved_total': rd['low_labels_preserved'] + ra['low_labels_preserved'], 'low_threshold_labels_missed_total': rd['low_labels_missed'] + ra['low_labels_missed'], 'true_labels_supported_total': rd['labels_supported_original'] + ra['labels_supported_original'], 'true_labels_preserved_total': rd['labels_preserved'] + ra['labels_preserved'], 'true_labels_missed_total': rd['label_miss'] + ra['label_miss'], 'new_false_detections_total': rd['new_false'] + ra['new_false'], 'original_unmatched_peaks_total': rd['original_unmatched'] + ra['original_unmatched'], 'low_unmatched_peaks_method_total': rd['low_unmatched_method'] + ra['low_unmatched_method'], 'low_unmatched_peaks_original_total': rd['low_unmatched_original'] + ra['low_unmatched_original'], 'original_peaks_total': rd['original_peaks'] + ra['original_peaks'], 'low_cfar_threshold': rd['low_threshold'], 'low_cfar_threshold_db': rd['low_threshold_db'], 'RD_labels_excluded_out_of_axis': rd['excluded_out_of_axis_labels'], 'RA_labels_excluded_out_of_axis': ra['excluded_out_of_axis_labels'], 'RD_alignment_endpoint_db': rd['alignment_endpoint_db'], 'RA_alignment_endpoint_db': ra['alignment_endpoint_db'], 'RD_alignment_span_db': rd['alignment_span_db'], 'RA_alignment_span_db': ra['alignment_span_db'], 'alignment_span_sum_db': rd['alignment_span_db'] + ra['alignment_span_db'], 'RD_aligned_threshold_count': rd['aligned_threshold_count'], 'RA_aligned_threshold_count': ra['aligned_threshold_count'], 'alignment_weighted_label_deficit': rd['weighted_label_deficit'] + ra['weighted_label_deficit'], 'training_cfar_threshold_count': int(max(rd['num_thresholds'], ra['num_thresholds'])), 'training_cfar_thresholds': ','.join((f'{x:.6g}' for x in thresholds))}
    if not all((np.isfinite(value) and value >= 0.0 for value in components.values())):
        raise RuntimeError(f'Non-finite or negative V12.7 task component: {components}')
    deficits = []
    for map_type, original, method, ranges, second, gate in [('RD', rd0, rd1, rr, vv, CFAR_LABEL_GATE_RD_V8), ('RA', ra0, ra1, rar, aa, CFAR_LABEL_GATE_RA_V8)]:
        margin = cfar_curve_map_v12_18(original, method, labels_cube, ranges, second, map_type, gate)
        if margin['num_low_misses'] != int(detail[map_type + '_low_threshold_labels_missed']):
            raise AssertionError('Auxiliary margin and reported object counting disagree.')
        detail[map_type + '_low_cfar_margin_deficit'] = margin['margin_sum']
        deficits.append(margin['margin_sum'])
        detail[map_type + '_cfar_curve_regret'] = margin['normalized_regret_sum']
        detail[map_type + '_cfar_lost_support_db_sum'] = margin['regret_db_sum']
        detail[map_type + '_unmatched_curve_original'] = margin['original_unmatched_curve_mean']
        detail[map_type + '_unmatched_curve_method'] = margin['method_unmatched_curve_mean']
        detail[map_type + '_cfar_curve_high_threshold'] = margin['high_threshold']
    detail['low_cfar_margin_deficit'] = float(sum(deficits))
    detail['cfar_curve_regret'] = sum((detail[k + '_cfar_curve_regret'] for k in ('RD', 'RA')))
    detail['unmatched_curve_original_total'] = sum((detail[k + '_unmatched_curve_original'] for k in ('RD', 'RA')))
    detail['unmatched_curve_method_total'] = sum((detail[k + '_unmatched_curve_method'] for k in ('RD', 'RA')))
    detail['cfar_curve_regret_legacy'] = detail['cfar_curve_regret']
    detail['cfar_curve_regret_balanced'] = balanced_curve_regret(detail, reference_span_db=V12_19_COMMON_SPAN_DB)
    return (components, detail)

def encode_policy_safe_v12_6(cube, learned, processor, exclude_candidates=()):
    """Compatibility reporting API. No shield/refinement/alternative selection."""
    if exclude_candidates:
        raise ValueError('V12.19 reports raw PPO only; no candidate exclusions.')
    encoded = encode_policy_v11(cube, learned['policy'], learned['normalizer'], False)
    post0 = postprocess_cube(cube, processor)
    post1 = postprocess_cube(encoded['cube_hat'], processor)
    encoded['meta'].update(shield_selected_candidate='raw_ppo_only', shield_candidate_count=1)
    audit = pd.DataFrame([dict(candidate='raw_ppo_only', selected=True, realized_bpcs=encoded['meta']['realized_bpcs'], refinement_used=False)])
    return dict(cube_hat=encoded['cube_hat'], meta=encoded['meta'], blocks=encoded['block_log'], post=post1, post_original=post0, shield_audit=audit)

def evaluate_v12_7_split(rows, labels, learned, split_name='validation'):
    sweep_rows = []
    curve_rows = []
    label_rows = []
    quality_rows = []
    shield_rows = []
    visual_records = []
    visual_scenes = set()
    postprocessors = learned['postprocessors']
    for cube_idx, (_, row) in enumerate(rows.reset_index(drop=True).iterrows()):
        cube = load_cube_v6_1000(row)
        processor = _postprocessor_for_cube_v11(cube, postprocessors)
        labs = labels_for_cube_v6_1000(labels, row)
        safe = encode_policy_safe_v12_6(cube, learned, processor)
        post0 = safe['post_original']
        rdo_hat, _, rdo_meta = compress_uniform_action_v11(cube, learned['fixed_action'])
        fft14_hat, _, fft14_meta = compress_cube_fft_full_support_multi(cube, {'FFT14': 14})['FFT14']
        methods = {'Original': (cube, {'realized_bits': float(32 * cube.size), 'realized_bpcs': 32.0}, post0), 'PPO': (safe['cube_hat'], safe['meta'], safe['post']), 'CalibratedRDO': (rdo_hat, rdo_meta, postprocess_cube(rdo_hat, processor)), 'FFT14': (fft14_hat, fft14_meta, postprocess_cube(fft14_hat, processor))}
        audit = safe['shield_audit'].copy()
        audit.insert(0, 'record_name', row['record_name'])
        audit.insert(1, 'scene_name', row['scene_name'])
        shield_rows.append(audit)
        for method, (cube_hat, meta, post1) in methods.items():
            quality_rows.append(_cube_quality_row_v12_6(row['record_name'], row['scene_name'], method, cube, cube_hat, meta, post0, post1))
        for map_type in ('RD', 'RA'):
            map0, ranges, second = _map_triplet_v12_6(post0, map_type)
            gate = CFAR_LABEL_GATE_RD_V8 if map_type == 'RD' else CFAR_LABEL_GATE_RA_V8
            evaluation_thresholds = V12_7_RD_PRESERVATION_THRESHOLD_FACTORS if map_type == 'RD' else V12_7_BASE_CFAR_THRESHOLD_FACTORS
            cache = precompute_original_cfar_count_sweep_v11_2(map0, labs, ranges, second, map_type, evaluation_thresholds, gate)
            for method, (_, meta, post1) in methods.items():
                method_map, _, _ = _map_triplet_v12_6(post1, map_type)
                _append_label_audit_v12_6(label_rows, cache, method_map, labs, method, row['record_name'], row['scene_name'])
                curve = cfar_curve_map_v12_18(map0, method_map, labs, ranges, second, map_type, gate)
                curve_rows.append(dict(record_name=row['record_name'], scene_name=row['scene_name'], method=method, map_type=map_type, low_threshold=curve['low_threshold'], high_threshold=curve['high_threshold'], eligible_objects=curve['num_eligible'], lowest_misses=curve['num_low_misses'], normalized_label_regret=curve['normalized_regret_sum'], lost_support_db_sum=curve['regret_db_sum'], original_unmatched_curve_mean=curve['original_unmatched_curve_mean'], method_unmatched_curve_mean=curve['method_unmatched_curve_mean'], realized_bpcs=float(meta['realized_bpcs'])))
                if method == 'Original':
                    for threshold in cache['threshold_factors']:
                        c = cache['per_threshold'][float(threshold)]
                        sweep_rows.append({'cube_idx': cube_idx, 'record_name': row['record_name'], 'scene_name': row['scene_name'], 'map_type': map_type, 'method': method, 'method_display': V12_7_DISPLAY[method], 'threshold_factor': float(threshold), 'cfar_threshold_db': float(10 * np.log10(float(threshold))), 'eligible_original': int(c['supported0']), 'preserved_original_supported': int(c['supported0']), 'missed_original_supported': 0, 'unmatched_false_peaks': int(c['unmatched0']), 'num_cfar_peaks': int(c['npeak0']), 'realized_bpcs': 32.0})
                else:
                    evaluated = evaluate_cfar_count_sweep_method_v11_2(cache, method_map)
                    for _, value in evaluated.iterrows():
                        sweep_rows.append({'cube_idx': cube_idx, 'record_name': row['record_name'], 'scene_name': row['scene_name'], 'map_type': map_type, 'method': method, 'method_display': V12_7_DISPLAY[method], 'threshold_factor': float(value.threshold_factor), 'cfar_threshold_db': float(value.cfar_threshold_db), 'eligible_original': int(value.num_labels_supported_original), 'preserved_original_supported': int(value.num_labels_preserved_from_original), 'missed_original_supported': int(value.num_labels_missed_from_original), 'unmatched_false_peaks': int(value.num_label_unmatched_peaks_method), 'num_cfar_peaks': int(value.num_cfar_peaks_method), 'realized_bpcs': float(meta['realized_bpcs'])})
        if str(row['scene_name']) not in visual_scenes:
            visual_scenes.add(str(row['scene_name']))
            visual_records.append({'row': row.copy(), 'labs': labs.copy(), 'methods': methods})
        print(f"V12.7 {split_name} [{cube_idx + 1:03d}/{len(rows):03d}] {row['record_name']}")
        gc.collect()
    curve_audit = pd.DataFrame(curve_rows)
    safe_write_csv_v11(curve_audit, TABLE_DIR_V11 / f'{split_name}_continuous_cfar_curve_audit_v12_19.csv')
    sweep = pd.DataFrame(sweep_rows)
    label_audit = pd.DataFrame(label_rows)
    quality = pd.DataFrame(quality_rows)
    shield = pd.concat(shield_rows, ignore_index=True) if shield_rows else pd.DataFrame()
    safe_write_csv_v11(sweep, TABLE_DIR_V11 / f'{split_name}_cfar_sweep_physical_roi_v12_7.csv')
    safe_write_csv_v11(label_audit, TABLE_DIR_V11 / f'{split_name}_object_label_audit_v12_7.csv')
    safe_write_csv_v11(quality, TABLE_DIR_V11 / f'{split_name}_compression_quality_v12_7.csv')
    safe_write_csv_v11(shield, TABLE_DIR_V11 / f'{split_name}_ppo_label_free_shield_audit_v12_7.csv')
    return {'cfar_curve_audit': curve_audit, 'sweep': sweep, 'label_audit': label_audit, 'quality': quality, 'shield': shield, 'visual_records': visual_records}

def aggregate_v12_7_sweep(sweep, split_name='validation', include_extended_rd=False):
    source = sweep.copy()
    if not include_extended_rd:
        base_max = float(np.max(V12_7_BASE_CFAR_THRESHOLD_FACTORS))
        source = source[(source.map_type != 'RD') | (source.threshold_factor <= base_max + 1e-12)]
    summary = source.groupby(['map_type', 'method', 'method_display', 'threshold_factor', 'cfar_threshold_db'], as_index=False).agg(eligible_original=('eligible_original', 'sum'), preserved_original_supported=('preserved_original_supported', 'sum'), missed_original_supported=('missed_original_supported', 'sum'), unmatched_false_peaks=('unmatched_false_peaks', 'sum'), num_cfar_peaks=('num_cfar_peaks', 'sum'), mean_realized_bpcs=('realized_bpcs', 'mean'), cubes=('cube_idx', 'nunique'))
    summary['exact_original_alignment'] = summary.preserved_original_supported == summary.eligible_original
    if not include_extended_rd:
        safe_write_csv_v11(summary, TABLE_DIR_V11 / f'{split_name}_cfar_summary_v12_7.csv')
    return summary

def _alignment_summary_v12_7(summary, split_name='validation'):
    rows = []
    for (map_type, method), group in summary.groupby(['map_type', 'method']):
        ordered = group.sort_values('cfar_threshold_db')
        missed = ordered.missed_original_supported.to_numpy(float)
        db = ordered.cfar_threshold_db.to_numpy(float)
        first = np.flatnonzero(missed > 0)
        count = int(first[0]) if len(first) else len(ordered)
        endpoint = float(db[max(0, count - 1)])
        rows.append({'map_type': map_type, 'method': method, 'aligned_threshold_count': count, 'alignment_endpoint_db': endpoint, 'alignment_span_db': float(max(endpoint - db[0], 0.0)) if count else 0.0, 'total_preservation_deficit': int(missed.sum()), 'threshold_weighted_deficit': float(np.dot(np.linspace(1.0, 2.0, len(missed)), missed))})
    result = pd.DataFrame(rows)
    safe_write_csv_v11(result, TABLE_DIR_V11 / f'{split_name}_cfar_alignment_metrics_v12_7.csv')
    return result

def plot_v12_7_cfar_figures(sweep, split_name='validation'):
    summary = aggregate_v12_7_sweep(sweep, split_name)
    summary_extended_rd = aggregate_v12_7_sweep(sweep, split_name, include_extended_rd=True)
    alignment = _alignment_summary_v12_7(summary, split_name)
    paths = []
    metrics = (('preserved_original_supported', 'Preserved original-detectable GT objects [count]', 'preserved_true_labels'), ('unmatched_false_peaks', 'CFAR/local peaks outside every GT gate [count]', 'unmatched_false_peaks'))
    for map_type in ('RD', 'RA'):
        g = summary[summary.map_type == map_type]
        for metric, ylabel, stem in metrics:
            fig, ax = plt.subplots(figsize=(9.4, 5.8))
            for method in ('Original', 'PPO'):
                part = g[g.method == method].sort_values('cfar_threshold_db')
                ax.plot(part.cfar_threshold_db, part[metric], marker='o', markersize=4, linewidth=2.5 if method == 'PPO' else 2.0, color=V12_7_COLORS[method], linestyle='--' if method == 'Original' else '-', label=V12_7_DISPLAY[method])
            if metric == 'preserved_original_supported':
                endpoint = float(alignment.query("map_type == @map_type and method == 'PPO'").alignment_endpoint_db.iloc[0])
                x0 = float(g.cfar_threshold_db.min())
                ax.axvspan(x0, endpoint, color=V12_7_COLORS['PPO'], alpha=0.08, label=f'exact-alignment interval through {endpoint:.2f} dB')
            ax.set_xlabel('CA-CFAR threshold [dB above local training-cell mean]')
            ax.set_ylabel(ylabel)
            ax.set_title(f"{map_type}: Original vs PPO — {ylabel.split(' [')[0]}")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            fig.tight_layout()
            path = FIG_DIR_V11 / f'{map_type}_original_vs_PPO_{stem}_vs_cfar_threshold_v12_7.png'
            save_publication_figure(fig, path, dpi=220)
            plt.close(fig)
            paths.append(path)
            four_g = summary_extended_rd[summary_extended_rd.map_type == map_type] if map_type == 'RD' and metric == 'preserved_original_supported' else g
            fig, ax = plt.subplots(figsize=(9.8, 5.9))
            for method in V12_7_METHODS:
                part = four_g[four_g.method == method].sort_values('cfar_threshold_db')
                ax.plot(part.cfar_threshold_db, part[metric], marker='o', markersize=3.8, linewidth=2.4 if method == 'PPO' else 1.8, color=V12_7_COLORS[method], linestyle='--' if method == 'Original' else '-', label=V12_7_DISPLAY[method])
            ax.set_xlabel('CA-CFAR threshold [dB above local training-cell mean]')
            ax.set_ylabel(ylabel)
            ax.set_title(f"{map_type}: four-policy comparison — {ylabel.split(' [')[0]}")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            fig.tight_layout()
            path = FIG_DIR_V11 / f'{map_type}_four_policy_{stem}_vs_cfar_threshold_v12_7.png'
            save_publication_figure(fig, path, dpi=220)
            plt.close(fig)
            paths.append(path)
    return paths

def plot_v12_7_scene_comparisons(visual_records):
    paths = []
    threshold = float(np.min(CFAR_THRESHOLD_FACTORS_V8))
    for record in visual_records:
        row, labs, methods = (record['row'], record['labs'], record['methods'])
        for map_type in ('RD', 'RA'):
            map0, ranges, second = _map_triplet_v12_6(methods['Original'][2], map_type)
            gate = CFAR_LABEL_GATE_RD_V8 if map_type == 'RD' else CFAR_LABEL_GATE_RA_V8
            cache = precompute_original_cfar_count_sweep_v11_2(map0, labs, ranges, second, map_type, [threshold], gate)
            ref_peak = float(np.max(as_linear_power(map0, assume_db=True)))
            fig = plt.figure(figsize=(21.5, 5.6))
            grid = fig.add_gridspec(1, 5, left=0.045, right=0.975, bottom=0.13, top=0.78, wspace=0.18, width_ratios=[1, 1, 1, 1, 0.045])
            axes = [fig.add_subplot(grid[0, i]) for i in range(4)]
            color_axis = fig.add_subplot(grid[0, 4])
            for ax, method in zip(axes, V12_7_METHODS):
                method_map, _, _ = _map_triplet_v12_6(methods[method][2], map_type)
                image = _relative_db_map_v12_6(method_map, ref_peak)
                extent = [float(second.min()), float(second.max()), float(ranges.min()), float(ranges.max())]
                im = ax.imshow(image, origin='lower', aspect='auto', extent=extent, vmin=-60, vmax=0, cmap='viridis')
                eligible, method_has = _status_vectors_v12_6(cache, method_map, threshold)
                _overlay_labels_v12_6(ax, labs, cache, eligible, method_has, map_type, original=method == 'Original')
                ax.set_title(V12_7_DISPLAY[method], fontsize=8.5)
                ax.set_xlabel('Radial velocity [m/s]' if map_type == 'RD' else 'Azimuth [deg]')
                ax.grid(False)
            axes[0].set_ylabel('Range [m]')
            fig.colorbar(im, cax=color_axis, label='Power relative to original peak [dB]')
            fig.suptitle(f"{map_type} physical-ROI comparison — {row['record_name']}\ngreen=preserved, red=missed, gray=not detected in original", y=0.96)
            path = FIG_DIR_V11 / f"{row['record_name']}_{map_type}_four_policy_label_status_v12_7.png"
            save_publication_figure(fig, path, dpi=200)
            plt.close(fig)
            paths.append(path)
        fig, axes = plt.subplots(1, 4, figsize=(20, 5.2), sharex=True, sharey=True)
        rd0, rr, vv = _map_triplet_v12_6(methods['Original'][2], 'RD')
        ra0, rar, aa = _map_triplet_v12_6(methods['Original'][2], 'RA')
        crd = precompute_original_cfar_count_sweep_v11_2(rd0, labs, rr, vv, 'RD', [threshold], CFAR_LABEL_GATE_RD_V8)
        cra = precompute_original_cfar_count_sweep_v11_2(ra0, labs, rar, aa, 'RA', [threshold], CFAR_LABEL_GATE_RA_V8)
        for ax, method in zip(axes, V12_7_METHODS):
            pts = _ra_cfar_point_cloud_v12_6(methods[method][2], threshold)
            if len(pts):
                ax.scatter(pts[:, 0], pts[:, 1], s=7, c='#4c78a8', alpha=0.32, label='RA CFAR peak')
            rdm, _, _ = _map_triplet_v12_6(methods[method][2], 'RD')
            ram, _, _ = _map_triplet_v12_6(methods[method][2], 'RA')
            erd, hrd = _status_vectors_v12_6(crd, rdm, threshold)
            era, hra = _status_vectors_v12_6(cra, ram, threshold)
            rd_status = {src: (bool(erd[i]), bool(hrd[i])) for i, src in enumerate(crd['label_source_indices'])}
            ra_status = {src: (bool(era[i]), bool(hra[i])) for i, src in enumerate(cra['label_source_indices'])}
            for source_i, (_, label) in enumerate(labs.iterrows()):
                e1, h1 = rd_status.get(source_i, (False, False))
                e2, h2 = ra_status.get(source_i, (False, False))
                eligible = e1 and e2
                kept = eligible and (method == 'Original' or (h1 and h2))
                color = 'lime' if kept else 'red' if eligible else '0.65'
                marker = 'o' if kept else 'X' if eligible else 'x'
                if marker == 'x':
                    ax.scatter([float(label.x_m)], [float(label.y_m)], s=55, marker=marker, c=color, linewidths=1.4)
                else:
                    ax.scatter([float(label.x_m)], [float(label.y_m)], s=55, marker=marker, facecolors='none' if marker == 'o' else color, edgecolors=color, linewidths=1.4)
            ax.set_title(V12_7_DISPLAY[method], fontsize=8.5)
            ax.set_xlabel('Forward x [m]')
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, alpha=0.2)
        axes[0].set_ylabel('Lateral y [m]')
        fig.suptitle(f"Physical point-cloud comparison — {row['record_name']}\nGT green=preserved in both RD and RA, red=missed, gray=original-ineligible")
        safe_tight_layout(fig, rect=[0, 0, 1, 0.92])
        path = FIG_DIR_V11 / f"{row['record_name']}_point_cloud_four_policy_label_status_v12_7.png"
        save_publication_figure(fig, path, dpi=200)
        plt.close(fig)
        paths.append(path)
    return paths

def plot_v12_7_compression_effectiveness(quality, split_name='validation'):
    summary = quality.groupby(['method', 'method_display'], as_index=False).agg(mean_bpcs=('realized_bpcs', 'mean'), mean_MiB=('compressed_MiB_per_cube', 'mean'), mean_saved_MiB=('saved_MiB_per_cube', 'mean'), mean_ratio=('compression_ratio_vs_32bpc', 'mean'), mean_nmse_db=('cube_nmse_db', 'mean'), mean_RD_mse=('RD_log_power_mse_physical_roi', 'mean'), mean_RA_mse=('RA_log_power_mse_physical_roi', 'mean'), mean_amp_ratio=('mean_absolute_amplitude_ratio', 'mean'), mean_peak_ratio=('peak_amplitude_ratio', 'mean'))
    linear_nmse = quality.groupby('method').cube_nmse.mean()
    summary['linear_mean_nmse'] = summary.method.map(linear_nmse)
    summary['db_of_linear_mean_nmse'] = 10 * np.log10(summary.linear_mean_nmse.where(summary.linear_mean_nmse > 0))
    safe_write_csv_v11(summary, TABLE_DIR_V11 / f'{split_name}_compression_summary_v12_7.csv')
    summary['order'] = summary.method.map({m: i for i, m in enumerate(V12_7_METHODS)})
    summary = summary.sort_values('order')
    short = {'Original': 'Original\n32 b/complex', 'PPO': 'PPO\nraw coherent transform', 'CalibratedRDO': 'Calibrated\nRDO', 'FFT14': 'FFT14\n14 b/component'}
    labels = [short[m] for m in summary.method]
    colors = [V12_7_COLORS[m] for m in summary.method]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    axes[0].bar(labels, summary.mean_MiB, color=colors)
    axes[0].set_ylabel('Accounted codec size [MiB/cube]')
    axes[0].set_title('Analytical coded volume (not a saved bitstream)')
    axes[1].bar(labels, summary.mean_saved_MiB, color=colors)
    axes[1].set_ylabel('Saved relative to int16 I+Q [MiB/cube]')
    axes[1].set_title('Volume reduction')
    for ax in axes:
        ax.grid(True, axis='y', alpha=0.25)
        ax.tick_params(axis='x', labelsize=8)
    fig.suptitle('Declared value/support/header bit accounting; NPY/NPZ are decoded tensors')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = FIG_DIR_V11 / 'compression_bits_and_MiB_saved_per_cube_v12_7.png'
    save_publication_figure(fig, path, dpi=220)
    plt.close(fig)
    paths = [path]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    width = 0.36
    x = np.arange(len(summary))
    axes[0].bar(labels, summary.db_of_linear_mean_nmse, color=colors)
    axes[0].annotate('Original: exact\nNMSE = 0 (−∞ dB)', (0, 0), xytext=(4, -25), textcoords='offset points', fontsize=7)
    axes[0].set_ylabel('Cube NMSE [dB]')
    axes[0].set_title('10 log10(mean linear NMSE); lower is better')
    axes[1].bar(x - width / 2, summary.mean_RD_mse, width, label='RD')
    axes[1].bar(x + width / 2, summary.mean_RA_mse, width, label='RA')
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel('Normalized log-power MSE')
    axes[1].set_title('Physical-ROI map distortion')
    axes[1].legend()
    axes[2].bar(x - width / 2, summary.mean_amp_ratio, width, label='mean |IQ|')
    axes[2].bar(x + width / 2, summary.mean_peak_ratio, width, label='peak |IQ|')
    axes[2].axhline(1.0, color='black', linestyle='--', linewidth=1)
    axes[2].set_xticks(x, labels)
    axes[2].set_ylabel('Amplitude ratio to original')
    axes[2].set_title('Amplitude preservation')
    axes[2].legend()
    for ax in axes:
        ax.grid(True, axis='y', alpha=0.25)
        ax.tick_params(axis='x', labelsize=7)
    fig.suptitle('Rate-distortion and amplitude preservation')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = FIG_DIR_V11 / 'compression_distortion_and_amplitude_preservation_v12_7.png'
    save_publication_figure(fig, path, dpi=220)
    plt.close(fig)
    paths.append(path)
    return paths

def _rolling_series_v12_7(frame, column, window):
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    values = pd.to_numeric(frame[column], errors='coerce')
    return values.rolling(int(window), min_periods=1).mean()

def plot_v12_7_training_diagnostics(training_log):
    """One reader-oriented PPO training figure plus a compact numeric summary."""
    if training_log is None or len(training_log) == 0:
        return []
    d = training_log.reset_index(drop=True).copy()
    x = np.arange(1, len(d) + 1)
    window = int(np.clip(round(len(d) / 12), 3, 15))
    fig, axes = plt.subplots(3, 2, figsize=(14.5, 12.0), sharex=True)

    def raw_and_mean(ax, column, color, label, direction):
        raw = pd.to_numeric(d.get(column, pd.Series(np.nan, index=d.index)), errors='coerce')
        mean = _rolling_series_v12_7(d, column, window)
        ax.plot(x, raw, color=color, alpha=0.18, linewidth=0.8)
        ax.plot(x, mean, color=color, linewidth=2.2, label=f'{label} ({window}-episode mean)')
        ax.set_title(f'{label} — {direction}')
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    raw_and_mean(axes[0, 0], 'realized_bpcs', '#4c78a8', 'Packed rate [bpcs]', 'must stay below cap')
    axes[0, 0].axhline(RL_RATE_TARGET_BPCS_V11, color='red', linestyle=':', linewidth=1.3, label='hard cap')
    axes[0, 0].legend(fontsize=8)
    raw_and_mean(axes[0, 1], 'episode_return', '#2a9d55', 'PPO episode return', 'higher is better')
    axes[0, 1].axhline(0.0, color='black', linewidth=1, alpha=0.6)
    raw_and_mean(axes[1, 0], 'alignment_span_sum_db', '#9467bd', 'RD+RA exact-alignment span [dB]', 'higher is better')
    for column, color, label in (('low_threshold_labels_missed_total', '#d62728', 'low-threshold misses'), ('true_labels_missed_total', '#ff7f0e', 'all-threshold misses')):
        axes[1, 1].plot(x, _rolling_series_v12_7(d, column, window), color=color, linewidth=2.0, label=f'{label} ({window}-episode mean)')
    axes[1, 1].set_title('True-object misses — lower is better')
    axes[1, 1].grid(True, alpha=0.25)
    axes[1, 1].legend(fontsize=8)
    for column, color, label in (('new_false_detections_total', '#e45756', 'compressed unmatched peaks'), ('original_unmatched_peaks_total', 'black', 'original-map reference')):
        axes[2, 0].plot(x, _rolling_series_v12_7(d, column, window), color=color, linewidth=2.0, linestyle='--' if color == 'black' else '-', label=f'{label} ({window}-episode mean)')
    axes[2, 0].set_title('Peaks outside GT gates — lower is better')
    axes[2, 0].grid(True, alpha=0.25)
    axes[2, 0].legend(fontsize=8)
    diagnostic_columns = [c for c in ('policy_entropy', 'entropy', 'approx_kl', 'clip_fraction') if c in d]
    if diagnostic_columns:
        for column, color in zip(diagnostic_columns, ('#17becf', '#bcbd22', '#8c564b', '#7f7f7f')):
            axes[2, 1].plot(x, _rolling_series_v12_7(d, column, window), linewidth=2.0, color=color, label={'approx_kl': 'exact KL (choice states)', 'policy_entropy': 'entropy (choice states)', 'clip_fraction': 'clip fraction (choice states)'}.get(column, column.replace('_', ' ')))
        axes[2, 1].set_title('PPO update stability')
    else:
        raw_and_mean(axes[2, 1], 'unique_actions', '#17becf', 'Actions used per cube', 'diversity indicator')
    axes[2, 1].grid(True, alpha=0.25)
    axes[2, 1].legend(fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel(f'Sampled allocation ({V12_12_ROLLOUTS_PER_CUBE} per cube visit)')
    tail = d.tail(max(1, min(window, len(d))))

    def mean_or_nan(column):
        return float(pd.to_numeric(tail.get(column, pd.Series(dtype=float)), errors='coerce').mean())
    summary_text = f"episodes={len(d)} | passes={int(pd.to_numeric(d.get('pass', pd.Series([1])), errors='coerce').max())} | final-window rate={mean_or_nan('realized_bpcs'):.3f} bpcs | return={mean_or_nan('episode_return'):+.2f} | alignment span={mean_or_nan('alignment_span_sum_db'):.2f} dB | low-threshold misses={mean_or_nan('low_threshold_labels_missed_total'):.2f}"
    fig.suptitle('PPO training diagnostics\n' + summary_text, fontsize=13.5, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    path = FIG_DIR_V11 / 'ppo_training_diagnostics_v12_7.png'
    save_publication_figure(fig, path, dpi=220)
    plt.close(fig)
    columns = ['realized_bpcs', 'episode_return', 'alignment_span_sum_db', 'alignment_weighted_label_deficit', 'low_threshold_labels_missed_total', 'true_labels_missed_total', 'new_false_detections_total', 'original_unmatched_peaks_total', 'safety_feasible', 'unique_actions']
    rows = []
    head = d.head(max(1, min(window, len(d))))
    for column in columns:
        if column not in d.columns:
            continue
        values = pd.to_numeric(d[column], errors='coerce')
        rows.append({'metric': column, 'episodes': int(values.notna().sum()), 'first_window_mean': float(pd.to_numeric(head[column], errors='coerce').mean()), 'last_window_mean': float(pd.to_numeric(tail[column], errors='coerce').mean()), 'best': float(values.max() if column in {'episode_return', 'alignment_span_sum_db', 'safety_feasible'} else values.min())})
    safe_write_csv_v11(pd.DataFrame(rows), TABLE_DIR_V11 / 'ppo_training_quick_summary_v12_7.csv')
    return [path]

def save_v12_7_reconstructed_ppo_cubes(visual_records, max_cubes=None):
    """Persist representative dense PPO reconstructions and audit actual disk bytes.

    The .npy/.npz files are decoder-output tensors for inspection or downstream
    experiments. They are intentionally distinguished from the exact packed codec
    payload, whose byte count comes from the declared support/value/header syntax.
    """
    if max_cubes is None:
        max_cubes = int(os.environ.get('RDO_V127_SAVE_RECON_CUBES', '2'))
    max_cubes = max(0, int(max_cubes))
    cube_dir = OUTPUT_DIR_V11 / 'reconstructed_ppo_cubes'
    cube_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in list(visual_records)[:max_cubes]:
        row = record['row']
        cube_hat, meta, _ = record['methods']['PPO']
        cube_hat = np.asarray(cube_hat)
        stem = str(row['record_name']) + '_PPO_reconstructed'
        npy_path = cube_dir / f'{stem}.npy'
        npz_path = cube_dir / f'{stem}.npz'
        np.save(npy_path, cube_hat, allow_pickle=False)
        np.savez_compressed(npz_path, cube=cube_hat)
        packed_bits = float(meta['realized_bits'])
        original_reference_bytes = float(32 * cube_hat.size / 8.0)
        metadata = {'record_name': str(row['record_name']), 'scene_name': str(row['scene_name']), 'shape': list(map(int, cube_hat.shape)), 'dtype': str(cube_hat.dtype), 'actual_codec_bitstream_saved': False, 'codec_size_is_analytical': True, 'codec_exact_packed_bits': packed_bits, 'codec_exact_packed_bytes': int(math.ceil(packed_bits / 8.0)), 'codec_exact_packed_MiB': packed_bits / 8.0 / 2 ** 20, 'dense_npy_bytes_on_disk': int(npy_path.stat().st_size), 'compressed_npz_bytes_on_disk': int(npz_path.stat().st_size), 'original_int16_IQ_reference_bytes': int(original_reference_bytes), 'note': 'NPY/NPZ are saved decoder-output tensors, not the serialized sparse codec bitstream.'}
        safe_write_json_v11(metadata, cube_dir / f'{stem}_storage.json')
        rows.append({**metadata, 'npy_path': str(npy_path), 'npz_path': str(npz_path), 'codec_saved_MiB_vs_32bpc': (original_reference_bytes - packed_bits / 8.0) / 2 ** 20, 'npz_saved_MiB_vs_32bpc': (original_reference_bytes - npz_path.stat().st_size) / 2 ** 20})
    table = pd.DataFrame(rows)
    safe_write_csv_v11(table, TABLE_DIR_V11 / 'saved_PPO_reconstructed_cube_storage_v12_7.csv')
    return table

def run_preflight_debug_v11():
    report = _impl_run_preflight_debug_v11_c32_l787()
    if len(V11_STATE_NAMES) != 20 + 2 * len(ACTION_LIBRARY_V11):
        raise AssertionError('Physical/action observation dimension mismatch.')
    print('V12.19 raw coherent-codec/absolute-count preflight: PASS; no refinement.')
    return report

def validate_v12_7_invariants(report, split_name='validation'):
    checks = []

    def add(name, passed, detail):
        checks.append({'check': name, 'passed': bool(passed), 'detail': str(detail)})
    quality = report['quality']
    fft14 = quality[quality.method == 'FFT14']
    cube_size = float(load_cube_v6_1000(report['visual_records'][0]['row']).size)
    expected = 28.0 + 128.0 / cube_size
    add('fft14_exact_rate', np.allclose(fft14.realized_bpcs, expected, rtol=0, atol=1e-10), expected)
    add('requested_method_order', list(V12_7_METHODS) == ['Original', 'PPO', 'CalibratedRDO', 'FFT14'], V12_7_METHODS)
    sweep = report['sweep']
    add('preserved_never_exceeds_eligible', bool((sweep.preserved_original_supported <= sweep.eligible_original).all()), 'all rows')
    label_audit = report['label_audit']
    key = ['record_name', 'map_type', 'method', 'threshold_factor', 'object_id']
    add('one_row_per_gt_object_per_condition', not label_audit.duplicated(key).any(), 'one object counts once')
    add('physical_thresholds_above_background', float(sweep.threshold_factor.min()) > 1.0, sweep.threshold_factor.min())
    base_summary = aggregate_v12_7_sweep(sweep, split_name)
    lowest = base_summary[np.isclose(base_summary.threshold_factor, base_summary.threshold_factor.min())]
    for map_type in ('RD', 'RA'):
        qrow = lowest[(lowest.map_type == map_type) & (lowest.method == 'FFT14')].iloc[0]
        add(f'FFT14_{map_type}_initial_alignment', int(qrow.preserved_original_supported) == int(qrow.eligible_original), f'{int(qrow.preserved_original_supported)}/{int(qrow.eligible_original)}')
    later = base_summary[(base_summary.method == 'FFT14') & (base_summary.threshold_factor > base_summary.threshold_factor.min())]
    full_validation = RL_RUN_MODE_V11 == 'full' and int(sweep.record_name.nunique()) >= 40
    add('FFT14_not_identical_to_original_after_initial_point', not full_validation or bool((later.preserved_original_supported < later.eligible_original).any()), 'checked on full validation; reduced runs only exercise the code path')
    rd_terminal = sweep[(sweep.map_type == 'RD') & np.isclose(sweep.threshold_factor, V12_7_RD_PRESERVATION_THRESHOLD_FACTORS[-1])]
    add('RD_extended_preservation_curves_end_at_zero', len(rd_terminal) > 0 and bool((rd_terminal.preserved_original_supported == 0).all()), f'rows={len(rd_terminal)}, preserved_sum={(int(rd_terminal.preserved_original_supported.sum()) if len(rd_terminal) else -1)}')
    result = pd.DataFrame(checks)
    empirical = result['check'].str.startswith('FFT14_') | result['check'].eq('RD_extended_preservation_curves_end_at_zero')
    result['check_kind'] = np.where(empirical, 'empirical_outcome', 'implementation_invariant')
    safe_write_csv_v11(result, TABLE_DIR_V11 / f'{split_name}_invariants_v12_7.csv')
    if not bool(result.loc[~empirical, 'passed'].all()):
        raise AssertionError('V12.7 invariant failure: ' + str(result.loc[~result.passed, 'check'].tolist()))
    if not bool(result.loc[empirical, 'passed'].all()):
        print('Observed baseline/terminal-sweep outcomes did not meet expectations:', result.loc[empirical & ~result.passed, 'check'].tolist())
    return result

def validation_gate_v12_7(report):
    """Full-sweep ORIGINAL-only acceptance plus separate baseline observations.

    Compressor comparisons remain useful publication diagnostics, but none
    determines acceptance, rewards, constraints, or main checkpoint ranking.
    The extended RD terminal-preservation tail is not added to the count
    allowance for the ordinary CFAR operating sweep.
    """
    summary = aggregate_v12_7_sweep(report['sweep'], 'validation')
    quality = report['quality']
    raw_quality = quality[quality.method.eq('PPO')]
    raw = summary[summary.method.eq('PPO')]
    source = summary[summary.method.eq('Original')]
    rows = []

    def add(name, candidate, reference, rule, passed):
        rows.append(dict(criterion=name, candidate=float(candidate), reference=float(reference), rule=rule, passed=bool(passed), uses_compressor_comparison=False))
    max_rate = float(raw_quality.realized_bpcs.max())
    add('per_cube_exact_rate_cap', max_rate, RL_RATE_TARGET_BPCS_V11, 'every RAW PPO cube <= configured cap', max_rate <= RL_RATE_TARGET_BPCS_V11 + 1e-06)
    lowest = float(raw.threshold_factor.min())
    raw_low, source_low = (raw[np.isclose(raw.threshold_factor, lowest)], source[np.isclose(source.threshold_factor, lowest)])
    n_cubes = int(report['sweep'].record_name.nunique())
    for mt in ('RD', 'RA'):
        p = raw_low[raw_low.map_type.eq(mt)].iloc[0]
        original = source_low[source_low.map_type.eq(mt)].iloc[0]
        add(mt + '_lowest_threshold_exact_label_preservation', p.preserved_original_supported, p.eligible_original, 'preserved == original-detectable objects', int(p.preserved_original_supported) == int(p.eligible_original))
        allowance = V12_5_FALSE_ALLOWANCE_PER_THRESHOLD * n_cubes
        add(mt + '_lowest_unmatched_vs_original', p.unmatched_false_peaks, original.unmatched_false_peaks, f'original + {allowance:g} count allowance', p.unmatched_false_peaks <= original.unmatched_false_peaks + allowance)
    total_unmatched, original_unmatched = (float(raw.unmatched_false_peaks.sum()), float(source.unmatched_false_peaks.sum()))
    operating_rows = report['sweep'][report['sweep'].method.eq('PPO') & report['sweep'].threshold_factor.isin(V12_7_BASE_CFAR_THRESHOLD_FACTORS)]
    allowance = V12_5_FALSE_ALLOWANCE_PER_THRESHOLD * len(operating_rows)
    add('sweep_unmatched_vs_original', total_unmatched, original_unmatched, f'original + {allowance:g} base-operating-condition count allowance', total_unmatched <= original_unmatched + allowance)
    continuous = report['cfar_curve_audit']
    for mt in ('RD', 'RA'):
        measured = continuous[continuous.method.eq('PPO') & continuous.map_type.eq(mt)]
        value = float(measured.method_unmatched_curve_mean.sum())
        reference = float(measured.original_unmatched_curve_mean.sum())
        allowance = V12_5_FALSE_ALLOWANCE_PER_THRESHOLD * len(measured)
        add(mt + '_continuous_interval_unmatched_vs_original', value, reference, f'exact continuous dB-mean count <= original + {allowance:g}', value <= reference + allowance)
    for name in ('cube_nmse', 'RD_log_power_mse_physical_roi', 'RA_log_power_mse_physical_roi'):
        values = pd.to_numeric(raw_quality[name], errors='coerce').to_numpy(float)
        add(name + '_finite_nonnegative', float(np.mean(values)), 0.0, 'implementation check; magnitude remains a measured quality diagnostic', bool(np.isfinite(values).all() and (values >= 0).all()))
    observed = summary.groupby(['method', 'method_display'], as_index=False).agg(low_or_sweep_eligible=('eligible_original', 'sum'), preserved_object_conditions=('preserved_original_supported', 'sum'), missed_object_conditions=('missed_original_supported', 'sum'), unmatched_peaks=('unmatched_false_peaks', 'sum'))
    measured_quality = quality.groupby('method', as_index=False).agg(mean_bpcs=('realized_bpcs', 'mean'), maximum_bpcs=('realized_bpcs', 'max'), mean_nmse=('cube_nmse', 'mean'), worst_nmse=('cube_nmse', 'max'))
    observed = observed.merge(measured_quality, on='method', validate='one_to_one')
    observed['used_for_training_or_acceptance'] = False
    safe_write_csv_v11(observed, TABLE_DIR_V11 / 'posttraining_baseline_observations_v12_19.csv')
    gate = pd.DataFrame(rows)
    safe_write_csv_v11(gate, TABLE_DIR_V11 / 'raw_ppo_original_only_full_sweep_gate_v12_19.csv')
    safe_write_csv_v11(gate, TABLE_DIR_V11 / 'shielded_ppo_validation_gate_v12_7.csv')
    return gate

def _build_v12_7_report_for_current_dirs(rows, labels, learned, split_name):
    report = evaluate_v12_7_split(rows, labels, learned, split_name)
    figures = []
    figures += plot_v12_7_cfar_figures(report['sweep'], split_name)
    figures += plot_v12_7_scene_comparisons(report['visual_records'])
    figures += plot_v12_7_compression_effectiveness(report['quality'], split_name)
    checks = validate_v12_7_invariants(report, split_name)
    report.update({'figure_paths': figures, 'checks': checks})
    return report

def build_v12_7_heldout(outputs, labels):
    global FIG_DIR_V11, TABLE_DIR_V11
    original_fig, original_table = (FIG_DIR_V11, TABLE_DIR_V11)
    root = OUTPUT_DIR_V11 / 'heldout_test'
    FIG_DIR_V11 = root / 'figures'
    TABLE_DIR_V11 = root / 'tables'
    FIG_DIR_V11.mkdir(parents=True, exist_ok=True)
    TABLE_DIR_V11.mkdir(parents=True, exist_ok=True)
    try:
        report = _build_v12_7_report_for_current_dirs(outputs['test_rows'], labels, outputs['learned'], 'heldout')
        safe_write_json_v11({'revision': NOTEBOOK_REVISION_V12_7, 'split': 'heldout_diagnostic_no_test_tuning', 'cubes': int(len(outputs['test_rows'])), 'figures': int(len(report['figure_paths']))}, root / 'heldout_status_v12_7.json')
        return report
    finally:
        FIG_DIR_V11, TABLE_DIR_V11 = (original_fig, original_table)

def behavior_clone_v11(policy, states, actions, normalizer):
    """Ordinary empirical teacher likelihood, not inverse-frequency reweighting.

    A rare high-rate action must not receive the same total imitation mass as
    the commonly selected rate-efficient action. Macro-F1 remains diagnostic,
    not the checkpoint objective. Single-class teachers are valid demonstrations;
    the categorical policy and existing PPO entropy term retain exploration.
    No validation or test examples are used in this initialization.
    """
    if RL_BC_EPOCHS_V11 == 0 or len(states) == 0:
        return pd.DataFrame(columns=['bc_epoch', 'bc_loss', 'bc_accuracy', 'bc_macro_f1'])
    x = normalizer.transform(states)
    y = np.asarray(actions, dtype=np.int64)
    if len(x) != len(y) or not np.all(policy.allowed_action_mask[y]):
        raise ValueError('Invalid behavior-cloning demonstrations.')
    classes = np.unique(y)
    policy.params['ba'][policy.allowed_action_mask] *= np.float32(0.25)
    optimizer = NumpyAdamV11(policy.params, RL_BC_LEARNING_RATE_V11)
    rng = np.random.default_rng(RL_SEED_V11)
    logs = []
    best_loss = np.inf
    best_params = None
    for epoch in range(RL_BC_EPOCHS_V11):
        order = rng.permutation(len(x))
        for start in range(0, len(x), 256):
            idx = order[start:start + 256]
            probability = policy.forward(x[idx])[0]
            probability = _apply_action_mask_v11(probability, np.broadcast_to(policy.allowed_action_mask, probability.shape))
            gradient = probability.copy()
            gradient[np.arange(len(idx)), y[idx]] -= 1.0
            optimizer.step(_network_gradients_v11(policy, x[idx], gradient))
        probability = policy.forward(x)[0]
        probability = _apply_action_mask_v11(probability, np.broadcast_to(policy.allowed_action_mask, probability.shape))
        loss = float(-np.mean(np.log(probability[np.arange(len(y)), y] + EPS)))
        if not np.isfinite(loss):
            raise FloatingPointError('Nonfinite imitation likelihood.')
        pred = probability.argmax(axis=1)
        recalls = []
        precisions = []
        f1s = []
        for action in classes:
            tp = float(np.sum((pred == action) & (y == action)))
            recall = tp / max(float(np.sum(y == action)), 1.0)
            precision = tp / max(float(np.sum(pred == action)), 1.0)
            recalls.append(recall)
            precisions.append(precision)
            f1s.append(2.0 * precision * recall / max(precision + recall, EPS))
        logs.append({'bc_epoch': epoch + 1, 'bc_loss': loss, 'bc_accuracy': float(np.mean(pred == y)), 'bc_balanced_accuracy': float(np.mean(recalls)), 'bc_min_action_recall': float(np.min(recalls)), 'bc_macro_precision': float(np.mean(precisions)), 'bc_macro_f1': float(np.mean(f1s)), 'bc_demonstrated_classes': len(classes), 'bc_status': 'empirical_teacher_likelihood'})
        if loss < best_loss:
            best_loss = loss
            best_params = {name: value.copy() for name, value in policy.params.items()}
    for name, value in best_params.items():
        policy.params[name][...] = value
    log = pd.DataFrame(logs)
    log['selected_checkpoint'] = log.index == log.bc_loss.idxmin()
    return log

def _choose_calibration_anchor_v12_8(table, feasible_summary):
    """Train-only preservation-first anchor, with measured counts as tie-breaks.

    A clipped/hinge task loss can be zero for several codecs with substantially
    different unmatched counts. Do not then pick the cheapest as if their
    perception quality were identical. Rate remains a hard constraint and the
    last tie-break, as in the existing validation checkpoint ordering.
    """
    work = table.copy()
    work['positive_unmatched_excess'] = np.maximum(work.new_false_detections_total - work.original_unmatched_peaks_total, 0.0)
    rank = work.groupby('action_index', as_index=False).agg(low_misses=('low_threshold_labels_missed_total', 'sum'), sweep_misses=('true_labels_missed_total', 'sum'), unmatched_excess=('positive_unmatched_excess', 'sum'), rd_error=('RD_log_power_mse', 'mean'), ra_error=('RA_log_power_mse', 'mean'), rate=('realized_bpcs', 'mean'))
    rank['map_error'] = rank.rd_error + rank.ra_error
    rank = rank[rank.action_index.isin(feasible_summary.action_index)]
    if rank.empty:
        raise RuntimeError('No feasible training-calibrated fixed action.')
    rank = rank.sort_values(['low_misses', 'sweep_misses', 'unmatched_excess', 'map_error', 'rate'])
    chosen = int(rank.iloc[0].action_index)
    rank['selected_fixed_action'] = rank.action_index.eq(chosen)
    safe_write_csv_v11(rank, TABLE_DIR_V11 / 'calibration_count_ranking_v12_8.csv')
    return chosen

def plot_fixed_validation_learning_v12_8(learned):
    """Same validation cubes at every pass; no exploratory/shielded conflation."""
    history = learned['validation_checkpoint_history'].copy().sort_values('pass')
    if history.empty:
        return []
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    specs = [('validation_low_threshold_misses_candidate', 'Lowest-threshold missed object instances'), ('validation_missed_labels_candidate', 'Missed object-threshold instances across sweep'), ('mean_false_detection_cost', 'Mean unmatched peaks per cube across RD+RA sweep'), ('mean_realized_bpcs', 'Mean raw PPO rate [bpcs]')]
    for ax, (key, title) in zip(axes.flat, specs):
        ax.plot(history['pass'], history[key], marker='o', label='Raw PPO (no refinement)')
        ax.set_title(title)
        ax.set_xlabel('Completed training passes (0 = initialization)')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.25)
        if 'selected' in history:
            for value in history.loc[history.selected.astype(bool), 'pass']:
                ax.axvline(value, color='gray', ls=':', label='Selected checkpoint')
    axes[0, 0].axhline(0, color='black', ls='--', label='Required zero misses')
    ref = learned['validation_candidate']
    original = float(ref.original_unmatched_peaks_total.mean())
    axes[1, 0].axhline(original, color='black', ls='--', label='Original reference')
    axes[1, 1].axhline(RL_RATE_TARGET_BPCS_V11, color='black', ls='--', label='Base rate cap')
    for ax in axes.flat:
        ax.legend(fontsize=8)
    fig.suptitle('Fixed-validation learning evidence — not a smoothed training episode curve')
    path = FIG_DIR_V11 / 'ppo_fixed_validation_learning_v12_8.png'
    save_publication_figure(fig, path, dpi=180)
    plt.close(fig)
    safe_write_csv_v11(history, TABLE_DIR_V11 / 'ppo_fixed_validation_learning_v12_8.csv')
    return [path]

class _ContentBlindNormalizerV12_8:
    """Counterfactual: replace content features by training means, retain budget/position.

    This tests sensitivity, not causally proven useful adaptation. The changed
    bitrate and task metrics must also be inspected; distinct action strings alone
    can result from budget exhaustion or random initialization.
    """

    def __init__(self, normalizer):
        self.normalizer = normalizer

    def transform(self, states):
        result = self.normalizer.transform(states).copy()
        result[:, 4:13] = 0.0
        result[:, 14:-4] = 0.0
        return result

def audit_adaptability_v12_8(outputs, labels, max_cubes=None):
    learned = outputs['learned']
    count = int(os.environ.get('RDO_V128_ADAPTATION_AUDIT_CUBES', '5')) if max_cubes is None else int(max_cubes)
    if count <= 0:
        return []
    rows = select_split_rows_v11(outputs['validation_rows'], min(count, len(outputs['validation_rows'])))
    records = []
    action_records = []
    for _, row in rows.iterrows():
        cube = load_cube_v6_1000(row)
        processor = _postprocessor_for_cube_v11(cube, learned['postprocessors'])
        post0 = postprocess_cube(cube, processor)
        labs = labels_for_cube_v6_1000(labels, row)
        normal = encode_policy_v11(cube, learned['policy'], learned['normalizer'], stochastic=False)
        blind = encode_policy_v11(cube, learned['policy'], _ContentBlindNormalizerV12_8(learned['normalizer']), stochastic=False)
        fixed_hat, _, fixed_meta = compress_uniform_action_v11(cube, learned['fixed_action'])
        changed = float(np.mean(normal['actions'] != blind['actions']))
        for name, hat, meta in [('RawPPO', normal['cube_hat'], normal['meta']), ('ContentBlindPPO', blind['cube_hat'], blind['meta']), ('CalibratedRDO', fixed_hat, fixed_meta)]:
            comp, detail = radar_task_components_v11(post0, postprocess_cube(hat, processor), labs)
            records.append({'record_name': row.record_name, 'scene_name': row.scene_name, **comp, **detail, **meta, 'method': name, 'content_ablation_action_change_fraction': changed})
        block = normal['block_log'].copy()
        block['record_name'] = row.record_name
        block['scene_name'] = row.scene_name
        block['content_blind_action_index'] = blind['actions']
        block['content_sensitive_action'] = normal['actions'] != blind['actions']
        action_records.append(block)
        print(f'V12.8 validation adaptation audit: {row.record_name} | content-sensitive actions={changed:.1%}', flush=True)
    result = pd.DataFrame(records)
    safe_write_csv_v11(result, TABLE_DIR_V11 / 'ppo_content_adaptation_audit_v12_8.csv')
    safe_write_csv_v11(pd.concat(action_records, ignore_index=True), TABLE_DIR_V11 / 'ppo_content_adaptation_actions_v12_8.csv')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    names = result.record_name.drop_duplicates().tolist()
    x = np.arange(len(names))
    for ax, metric, title in zip(axes.flat, ['low_threshold_labels_missed_total', 'new_false_detections_total', 'realized_bpcs', 'content_ablation_action_change_fraction'], ['Lowest-threshold misses', 'Unmatched peaks across RD+RA sweep', 'Actual raw-codec rate [bpcs]', 'Fraction of decisions changed by content ablation']):
        for name in ['RawPPO', 'ContentBlindPPO', 'CalibratedRDO']:
            if metric == 'content_ablation_action_change_fraction' and name != 'RawPPO':
                continue
            series = result[result.method.eq(name)].set_index('record_name').loc[names, metric]
            ax.plot(x, series, marker='o', label=name)
        ax.set_title(title)
        ax.set_xlabel('Fixed validation cube')
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(fontsize=8)
    fig.suptitle('Adaptation audit: content dependence is not automatically useful adaptation\nSame rate cap, actual rates shown; no refinement or test-set tuning')
    path = FIG_DIR_V11 / 'ppo_content_adaptation_audit_v12_8.png'
    save_publication_figure(fig, path, dpi=180)
    plt.close(fig)
    return [path]

def run_v12_7_pipeline(dataset_root):
    outputs = _run_training_pipeline_v12_7(Path(dataset_root))
    globals()['training_outputs_v12_8'] = outputs
    labels = load_labels_v6_1000(Path(dataset_root))
    outputs['v12_7_reports'] = build_v12_7_reports(outputs, labels)
    outputs['test_evaluated'] = True
    print('V12.19 complete report:', OUTPUT_DIR_V11)
    display(outputs['v12_7_reports']['validation_gate'])
    return outputs
V12_9_ACTION_SET = 'full'
if V12_9_ACTION_SET not in ('full', 'legacy'):
    raise ValueError('RDO_V129_ACTION_SET must be full or legacy (controlled ablation).')
V12_5_FALSE_VIOLATION_PER_PEAK = float(os.environ.get('RDO_V129_FALSE_WEIGHT', '1.0'))
if not np.isfinite(V12_5_FALSE_VIOLATION_PER_PEAK) or V12_5_FALSE_VIOLATION_PER_PEAK < 0:
    raise ValueError('RDO_V129_FALSE_WEIGHT must be finite and nonnegative.')

def _safety_rate_terminal_reward_v12_3(candidate_components, candidate_detail, reference_components, reference_detail, candidate_bpcs, scaler, multipliers=None):
    """Original-only V12.19 loss. Comparator arguments remain unused.

    One multiplier changes only four preservation terms. Labels, detector,
    unmatched/map/NMSE costs, rate cap and feasibility masks are unchanged.
    """
    if V12_18_UNMATCHED_MODE != 'sum_positive':
        raise ValueError('The controlled V12.19 comparison requires V12.18 sum_positive unmatched costs.')
    weights = V12_18_DUAL.weights if multipliers is None else np.asarray(multipliers, float)
    detail = dict(candidate_detail)
    detail['cfar_curve_regret'] = detail.get('cfar_curve_regret_legacy', detail['cfar_curve_regret'])
    terms = preservation_weighted_reward_terms(candidate_components, detail, candidate_bpcs, preservation_multiplier=V12_19_PRESERVATION_MULTIPLIER, curve_mode=V12_19_CURVE_MODE, common_span_db=V12_19_COMMON_SPAN_DB, rate_cap=RL_RATE_TARGET_BPCS_V11, low_weight=float(weights[0]), unmatched_weight=float(weights[1]), sweep_weight=500.0, map_weight=V12_18_MAP_WEIGHT, margin_weight=V12_18_MARGIN_WEIGHT, curve_weight=V12_18_MARGIN_WEIGHT * V12_18_CURVE_LOSS_SCALE, nmse_weight=V12_11_DISTORTION_WEIGHT, false_allowance_per_threshold=V12_5_FALSE_ALLOWANCE_PER_THRESHOLD, training_threshold_count=len(_nearest_training_thresholds_v11()))
    terms['reward_cfar_curve_regret'] = terms['cfar_curve_regret']
    terms['cfar_curve_regret'] = detail['cfar_curve_regret']
    terms['cfar_curve_regret_legacy'] = detail['cfar_curve_regret']
    terms['cfar_curve_regret_balanced'] = terms['balanced_cfar_curve_regret']
    recomputed = -math.fsum((terms[name] for name in LOSS_COMPONENT_NAMES))
    if not np.isclose(recomputed, terms['terminal_label_first_reward'], rtol=1e-12, atol=1e-08):
        raise AssertionError('Independently summed reward components disagree with total.')
    terms['reward_components_sum'] = recomputed
    terms['reward_components_error'] = recomputed - terms['terminal_label_first_reward']
    return terms

def safe_correlation_v12_9(x, y):
    """Undefined correlations remain NaN, not invented zero evidence."""
    x, y = (np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    good = np.isfinite(x) & np.isfinite(y)
    x, y = (x[good], y[good])
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])

def action_diagnostics_v12_9(trajectory):
    """Physical decisions, not categorical action IDs; no change to the reward."""
    log = trajectory['block_log']
    concentration = np.asarray(log.spectral_top10_energy_fraction)
    rate = np.asarray(log.realized_bpcs)
    return dict(rate_spectral_correlation=safe_correlation_v12_9(rate, concentration), q_spectral_correlation=safe_correlation_v12_9(log.q, concentration), rho_spectral_correlation=safe_correlation_v12_9(log.rho, concentration), rate_energy_correlation=safe_correlation_v12_9(rate, log.block_log_energy_ratio), cube_nmse=float(trajectory['meta']['cube_nmse']))

def ppo_update_v11(policy, optimizer, trajectories):
    before, offset = (int(optimizer.t), len(optimizer.gradient_records))
    result = update_mc_ppo(policy, optimizer, trajectories, epochs=RL_PPO_EPOCHS_V11, clip=RL_PPO_CLIP_V11, entropy_coef=RL_ENTROPY_COEF_V11, target_kl=RL_TARGET_KL_V11, value_coef=V12_4_VALUE_COEF, reward_scale=V12_4_PPO_REWARD_SCALE, gradients_fn=_network_gradients_v11, mask_fn=_apply_action_mask_v11, seed=RL_SEED_V11, eps=EPS, advantage_mode=V12_18_ADVANTAGE_MODE)
    norms = np.asarray(optimizer.gradient_records[offset:], float)
    grouped = {}
    for t in trajectories:
        grouped.setdefault(t['rollout_group'], []).append(float(np.sum(t['rewards'])))
    groups = list(grouped.values())
    result.update(optimizer_step_before=before, optimizer_step_after=int(optimizer.t), actor_gradient_norm=float(norms[:, 0].mean()), critic_gradient_norm=float(norms[:, 1].mean()), mean_cube_nmse=float(np.mean([t['meta']['cube_nmse'] for t in trajectories])), within_cube_return_std=float(np.mean([np.std(g) for g in groups])), between_cube_mean_return_std=float(np.std([np.mean(g) for g in groups])), zero_contrast_group_fraction=float(np.mean([np.ptp(g) <= 1e-08 for g in groups])), paired_advantage_rms=result['mc_advantage_rms'], rollout_groups_per_update=len(groups), rollouts_per_cube=V12_12_ROLLOUTS_PER_CUBE)
    return result

def _save_diagnostic_figure_v12_9(fig, name):
    path = FIG_DIR_V11 / (name + '_v12_9.png')
    save_publication_figure(fig, path, dpi=180)
    plt.close(fig)
    return path

def plot_learning_evidence_v12_9(learned):
    d = learned['training_log'].copy()
    if d.empty:
        return []
    paths = []
    updates = d.drop_duplicates('optimizer_step_after').reset_index(drop=True)
    x = np.arange(1, len(updates) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')
    for ax, key, title in zip(axes.flat, ['actor_loss', 'value_loss', 'exact_categorical_kl_choice', 'policy_entropy', 'clip_fraction', 'critic_explained_variance_after'], ['PPO surrogate loss (not a global task loss)', 'Critic mean squared return error', 'Exact policy KL (choice states)', 'Policy entropy (choice states)', 'Clip fraction', 'Critic explained variance']):
        ax.plot(x, updates[key], marker='.', lw=1.2)
        ax.set_title(title)
        ax.set_xlabel('PPO update batch')
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if key in ('value_loss', 'exact_categorical_kl_choice'):
            ax.set_ylim(bottom=0)
        if key == 'policy_entropy':
            ax.set_ylim(0, np.log(len(ACTION_LIBRARY_V11)) * 1.05)
        if key == 'clip_fraction':
            ax.set_ylim(0, max(0.05, float(updates[key].max()) * 1.1))
        if key == 'critic_explained_variance_after':
            ax.axhline(0, color='gray', ls=':', lw=1)
    fig.suptitle('Optimizer diagnostics — movement is not proof of task improvement')
    paths.append(_save_diagnostic_figure_v12_9(fig, 'ppo_optimizer_losses'))
    safe_write_csv_v11(updates, TABLE_DIR_V11 / 'ppo_unique_optimizer_updates_v12_9.csv')
    d['low_miss_rate'] = d.low_threshold_labels_missed_total / d.low_threshold_labels_supported_total.replace(0, np.nan)
    d['sweep_miss_rate'] = d.true_labels_missed_total / d.true_labels_supported_total.replace(0, np.nan)
    d['positive_unmatched_excess'] = np.maximum(d.new_false_detections_total - d.original_unmatched_peaks_total, 0.0)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')
    grouped = d.groupby(['pass', 'scene_name'], as_index=False).agg(low_miss_rate=('low_miss_rate', 'mean'), sweep_miss_rate=('sweep_miss_rate', 'mean'), positive_unmatched_excess=('positive_unmatched_excess', 'mean'), cube_nmse=('cube_nmse', 'mean'), realized_bpcs=('realized_bpcs', 'mean'), episode_return=('episode_return', 'mean'))
    for ax, key, title in zip(axes.flat, ['low_miss_rate', 'sweep_miss_rate', 'positive_unmatched_excess', 'cube_nmse', 'realized_bpcs', 'episode_return'], ['Lowest-threshold missed-label fraction', 'Sweep missed-label fraction', 'Excess unmatched peaks/cube', 'Complex-cube NMSE (linear mean)', 'Actual base rate [bpcs]', 'Episode return']):
        for scene, g in grouped.groupby('scene_name'):
            ax.plot(g['pass'], g[key], marker='o', label=scene)
        ax.set_title(title)
        ax.set_xlabel('Completed training pass')
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if key != 'episode_return':
            ax.set_ylim(bottom=0)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('Per-scene training errors — stochastic rollouts, not held-out performance')
    paths.append(_save_diagnostic_figure_v12_9(fig, 'ppo_scene_learning_errors'))
    safe_write_csv_v11(grouped, TABLE_DIR_V11 / 'ppo_scene_learning_errors_v12_9.csv')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout='constrained')
    for ax, key, title in zip(axes, ['reward_low_miss_penalty', 'reward_unmatched_penalty', 'reward_rate_saving'], ['Lowest-threshold label penalty', 'Unmatched-peak penalty', 'Rate-saving reward']):
        x = np.arange(1, len(d) + 1)
        values = d[key]
        ax.plot(x, values, alpha=0.3, label='episode')
        ax.plot(x, values.rolling(10, min_periods=1).mean(), label='10-allocation mean')
        ax.set_title(title)
        ax.set_xlabel(f'Sampled allocation ({V12_12_ROLLOUTS_PER_CUBE} per cube visit)')
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle('Reward terms in their actual units — no rescaled visual balance')
    paths.append(_save_diagnostic_figure_v12_9(fig, 'ppo_reward_tradeoff'))
    return paths

def plot_action_physics_v12_9(learned):
    d = learned['validation_actions'].copy()
    required = {'rho', 'q', 'realized_bpcs', 'spectral_top10_energy_fraction', 'record_name'}
    if not required.issubset(d):
        path = TABLE_DIR_V11 / 'ppo_content_adaptation_actions_v12_8.csv'
        d = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not required.issubset(d):
        raise RuntimeError('Physical action diagnostics need the saved per-block audit.')
    records = []
    for name, g in d.groupby('record_name'):
        for key in ['rho', 'q', 'realized_bpcs']:
            records.append(dict(record_name=name, quantity=key, spectral_correlation=safe_correlation_v12_9(g[key], g.spectral_top10_energy_fraction)))
    safe_write_csv_v11(pd.DataFrame(records), TABLE_DIR_V11 / 'action_physical_correlations_v12_9.csv')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout='constrained')
    for ax, key, label in zip(axes, ['rho', 'q', 'realized_bpcs'], ['Retained coefficient fraction', 'Quantizer bits/component', 'Actual block rate [bpcs]']):
        ax.scatter(d.spectral_top10_energy_fraction, d[key], s=9, alpha=0.3)
        ax.set_xlabel('Spectral top-10% energy fraction')
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle('Selected-policy content/action relationship — descriptive, not a causal claim')
    return [_save_diagnostic_figure_v12_9(fig, 'ppo_physical_action_adaptation')]

def audit_isolated_contribution_v12_9(outputs, labels, max_cubes=None):
    """Reuse ALL fixed-validation results; no extra refinement or recompression."""
    learned = outputs['learned']
    original = pd.read_csv(TABLE_DIR_V11 / 'frozen_imitation_validation_per_cube_v12_10.csv')
    result = pd.concat([original.assign(experiment_method='FrozenBC'), learned['validation_candidate'].assign(experiment_method='RawPPO'), learned['validation_fixed'].assign(experiment_method='CalibratedRDO')], ignore_index=True)
    safe_write_csv_v11(result, TABLE_DIR_V11 / 'ppo_isolated_contribution_v12_9.csv')
    return plot_isolated_contribution_v12_9(result, learned['validation_candidate'].record_name.tolist())

def plot_isolated_contribution_v12_9(result, names):
    """Historical figure families retained, now comparing raw policies directly."""
    methods = ['FrozenBC', 'RawPPO', 'CalibratedRDO']
    paths = []
    for filename, mode in [('ppo_isolated_contribution', 'per_cube'), ('ppo_learning_same_refinement', 'per_scene')]:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
        for ax, key, title in zip(axes.flat, ['low_threshold_labels_missed_total', 'new_false_detections_total', 'realized_bpcs', 'cube_nmse'], ['Lowest-threshold misses', 'Outside-GT peaks across training sweep', 'Analytical packed rate [bpcs]', 'Complex-cube NMSE']):
            for method in methods:
                d = result[result.experiment_method.eq(method)]
                values = d.set_index('record_name').loc[names, key] if mode == 'per_cube' else d.groupby('scene_name')[key].mean()
                ax.plot(np.arange(len(values)), values, '-o', label=method)
                if mode == 'per_scene':
                    ax.set_xticks(np.arange(len(values)), values.index, rotation=20, fontsize=7)
            ax.set(title=title, xlabel='Fixed validation cube' if mode == 'per_cube' else 'Traffic episode (episode-disjoint split)')
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        fig.suptitle('Raw policy comparison: same coherent codec, full actions and cap; NO refinement\nActual rates shown; common cap is not rate-matched superiority')
        paths.append(_save_diagnostic_figure_v12_9(fig, filename))
    summary = result.groupby('experiment_method', as_index=False).agg(cubes=('record_name', 'nunique'), low_misses=('low_threshold_labels_missed_total', 'sum'), sweep_misses=('true_labels_missed_total', 'sum'), mean_unmatched=('new_false_detections_total', 'mean'), mean_bpcs=('realized_bpcs', 'mean'), mean_nmse=('cube_nmse', 'mean'), worst_nmse=('cube_nmse', 'max'))
    summary['evaluation_population'] = 'fixed_validation_probe'
    safe_write_csv_v11(summary, TABLE_DIR_V11 / 'ppo_full_validation_isolation_summary_v12_19.csv')
    return paths

def distortion_tail_audit_v12_9():
    for scope, folder in [('validation', OUTPUT_DIR_V11), ('heldout', OUTPUT_DIR_V11 / 'heldout_test')]:
        path = folder / 'tables' / f'{scope}_compression_quality_v12_7.csv'
        if not path.exists():
            continue
        d = pd.read_csv(path)
        if 'cube_nmse' not in d:
            continue
        records = []
        for method, g in d.groupby('method'):
            values = g.cube_nmse.to_numpy(float)
            records.append(dict(method=method, linear_mean_nmse=float(np.mean(values)), db_of_mean_nmse=float(10 * np.log10(max(np.mean(values), 1e-12))), median_nmse=float(np.median(values)), p95_nmse=float(np.quantile(values, 0.95)), worst_nmse=float(np.max(values))))
        safe_write_csv_v11(pd.DataFrame(records), folder / 'tables' / 'distortion_tail_summary_v12_9.csv')

def build_v12_7_reports(outputs, labels):
    """All retained reports; raw training-grid and full-sweep acceptance are distinct.

    Final test is diagnostic even when a validation gate fails. It is never
    used to select policies, multipliers, actions, or thresholds.
    """
    learned = outputs['learned']
    report = _build_v12_7_report_for_current_dirs(outputs['validation_rows'], labels, learned, 'validation')
    gate = validation_gate_v12_7(report)
    report['validation_gate'] = gate
    report['reconstructed_ppo_storage'] = save_v12_7_reconstructed_ppo_cubes(report['visual_records'])
    report['heldout_test'] = build_v12_7_heldout(outputs, labels)
    raw, fixed, actions = evaluate_policy_rows_v11(outputs['test_rows'], labels, learned['policy'], learned['normalizer'], learned['fixed_action'], learned['scaler'], learned['postprocessors'])
    for frame, name in [(raw, 'raw_PPO'), (fixed, 'calibrated_RDO'), (actions, 'raw_PPO_actions')]:
        safe_write_csv_v11(frame, OUTPUT_DIR_V11 / 'heldout_test' / 'tables' / f'{name}_heldout_diagnostic_v12_19.csv')
    raw_test_gate = validation_gate_v11(raw, fixed)
    safe_write_csv_v11(raw_test_gate, OUTPUT_DIR_V11 / 'heldout_test' / 'tables' / 'raw_PPO_heldout_acceptance_v12_19.csv')
    report['raw_heldout_metrics'] = validation_measurements_v12_10(raw, fixed, learned['scaler'])
    extra = plot_v12_7_training_diagnostics(learned.get('training_log'))
    extra += plot_fixed_validation_learning_v12_8(learned)
    extra += audit_adaptability_v12_8(outputs, labels)
    extra += plot_learning_evidence_v12_9(learned)
    extra += plot_action_physics_v12_9(learned)
    extra += audit_isolated_contribution_v12_9(outputs, labels)
    extra += plot_scratch_vs_imitation_v12_10(learned)
    extra += plot_credit_assignment_v12_11(learned)
    extra += plot_same_cube_credit_v12_12(learned)
    extra += plot_constraint_learning_v12_13(learned)
    extra += plot_margin_learning_v14(learned)
    extra += plot_original_only_learning_v12_18(learned)
    extra += plot_cfar_curve_learning_v12_18(learned)
    safe_write_json_v11(postprocessing_contract_v12_18(), TABLE_DIR_V11 / 'postprocessing_contract_v12_19.json')
    report['figure_paths'].extend(extra)
    distortion_tail_audit_v12_9()
    for scope, folder in [('validation', OUTPUT_DIR_V11), ('heldout', OUTPUT_DIR_V11 / 'heldout_test')]:
        path = folder / 'tables' / f'{scope}_ppo_label_free_shield_audit_v12_7.csv'
        audit = pd.read_csv(path)
        selected = audit[audit.selected.astype(str).str.lower().eq('true')]
        counts = selected.groupby('candidate').size().rename('selected_cubes').reset_index()
        safe_write_csv_v11(counts, folder / 'tables' / 'ppo_candidate_attribution_v12_8.csv')
        display(counts)
    report['status'] = dict(revision=NOTEBOOK_REVISION_V12_8, dataset_root=str(DATASET_ROOT_V8), raw_training_grid_validation_passed=bool(learned['deployment_eligible']), raw_ppo_heldout_criteria_passed=bool(raw_test_gate.passed.all()), raw_ppo_heldout_metrics=report['raw_heldout_metrics'], full_sweep_raw_validation_passed=bool(gate.passed.all()), refinement_used=False, heldout_test_evaluated=True, heldout_is_diagnostic=True, publication_ready=False, actual_codec_bitstream_saved=False, label_count_rule='one original-detectable GT object per map/threshold; any in-gate peak preserves it', deployment_encoder_uses_labels=False, main_cfar_method='RAW PPO ONLY: no refinement, selector or fallback', validation_figure_count=len(list(FIG_DIR_V11.glob('*.png'))), caveat='No monotonic-learning, feasibility or zero-heldout-miss guarantee; matched-rate multiseed evidence is required')
    for name in ['run_status_v12_19.json', 'run_status_v12.json', 'run_status_v12_7.json']:
        safe_write_json_v11(report['status'], OUTPUT_DIR_V11 / name)
    files = sorted((p for p in OUTPUT_DIR_V11.rglob('*') if p.is_file()))
    safe_write_csv_v11(pd.DataFrame({'relative_path': [str(p.relative_to(OUTPUT_DIR_V11)) for p in files], 'bytes': [p.stat().st_size for p in files]}), TABLE_DIR_V11 / 'output_manifest_v12_7.csv')
    return report
V12_10_BASELINE_BC_EPOCHS = int(os.environ.get('RDO_V1210_BASELINE_BC_EPOCHS', '25'))
V12_10_VAL_EVERY_UPDATES = int(os.environ.get('RDO_V1219_VAL_EVERY_UPDATES', '60' if RL_RUN_MODE_V11 == 'full' else '20'))
if V12_10_BASELINE_BC_EPOCHS < 1 or V12_10_VAL_EVERY_UPDATES < 0:
    raise ValueError('Baseline BC epochs must be positive; validation interval nonnegative.')

def fit_exploration_normalizer_v12_10(rows):
    """Label-free training-cube statistics, not teacher-selected state traces.

    Uniform probabilities over all rate-feasible actions supply running-rate
    and previous-action features. These probes do NOT update either network.
    Normalize physical features for tanh conditioning; do not alter the state
    representation, use labels, or inspect validation/test cubes.
    """
    identity = StateNormalizerV11()
    identity.mean = np.zeros(len(V11_STATE_NAMES), np.float32)
    identity.std = np.ones(len(V11_STATE_NAMES), np.float32)
    probe = RadarPPOPolicyV11(len(V11_STATE_NAMES), len(ACTION_LIBRARY_V11), seed=RL_SEED_V11 + 101)
    probe.params['Wa'].fill(0.0)
    probe.params['ba'].fill(0.0)
    states, audit = ([], [])
    for _, row in rows.iterrows():
        cube = load_cube_v6_1000(row)
        package = encode_policy_v11(cube, probe, identity, stochastic=True)
        states.append(package['states_raw'].copy())
        audit.append(dict(record_name=row.record_name, scene_name=row.scene_name, state_count=len(package['states_raw']), uses_labels=False, uses_teacher_actions=False, uses_validation_or_test=False, probe_rate_bpcs=package['meta']['realized_bpcs']))
        del cube, package
    normalizer = StateNormalizerV11().fit(np.concatenate(states))
    safe_write_csv_v11(pd.DataFrame(audit), TABLE_DIR_V11 / 'normalizer_training_probe_audit_v12_10.csv')
    return normalizer

def make_frozen_imitation_baseline_v12_10(calibration, normalizer):
    """Separate object/optimizer. Its weights cannot initialize the main PPO."""
    global RL_BC_EPOCHS_V11
    baseline = RadarPPOPolicyV11(len(V11_STATE_NAMES), len(ACTION_LIBRARY_V11), seed=RL_SEED_V11)
    baseline.set_allowed_actions(np.arange(len(ACTION_LIBRARY_V11)))
    baseline.params['ba'][int(calibration['selected_action_index'])] = 1.0
    saved_epochs = RL_BC_EPOCHS_V11
    try:
        RL_BC_EPOCHS_V11 = V12_10_BASELINE_BC_EPOCHS
        log = behavior_clone_v11(baseline, calibration['teacher_states'], calibration['teacher_actions'], normalizer)
    finally:
        RL_BC_EPOCHS_V11 = saved_epochs
    safe_save_npz_v11(MODEL_DIR_V11 / 'frozen_imitation_policy_v12_10.npz', **baseline.state_dict(), allowed_action_indices=np.flatnonzero(baseline.allowed_action_mask))
    safe_write_csv_v11(log, TABLE_DIR_V11 / 'frozen_imitation_baseline_training_v12_10.csv')
    return (baseline, log)

def validation_measurements_v12_10(candidate, fixed, scaler):
    """Fixed original-only weights; the historical comparator argument is unused."""
    rewards, reward_records = ([], [])
    for _, row in candidate.iterrows():
        components = {name: float(row[name]) for name in V11_TASK_COMPONENT_NAMES}
        reward = _safety_rate_terminal_reward_v12_3(components, row.to_dict(), {}, {}, float(row.realized_bpcs), scaler, multipliers=V12_18_INITIAL_WEIGHTS)
        rewards.append(reward['terminal_label_first_reward'])
        reward_records.append(reward)
    component_means = {'mean_' + key: float(np.mean([record[key] for record in reward_records])) for key in LOSS_COMPONENT_NAMES}
    component_means['mean_balanced_cfar_curve_regret'] = float(candidate.cfar_curve_regret_balanced.mean())
    component_means['mean_lost_original_support_object_db'] = float((candidate.RD_cfar_lost_support_db_sum + candidate.RA_cfar_lost_support_db_sum).mean())
    return dict(**component_means, mean_cfar_curve_regret=float(candidate.cfar_curve_regret.mean()), mean_unmatched_curve_method_total=float(candidate.unmatched_curve_method_total.mean()), mean_unmatched_curve_original_total=float(candidate.unmatched_curve_original_total.mean()), mean_low_cfar_margin_deficit=float(candidate.low_cfar_margin_deficit.mean()), mean_fixed_validation_reward=float(np.mean(rewards)), low_supported=float(candidate.low_threshold_labels_supported_total.sum()), low_preserved=float(candidate.low_threshold_labels_preserved_total.sum()), low_misses=float(candidate.low_threshold_labels_missed_total.sum()), sweep_supported=float(candidate.true_labels_supported_total.sum()), sweep_preserved=float(candidate.true_labels_preserved_total.sum()), sweep_misses=float(candidate.true_labels_missed_total.sum()), mean_unmatched=float(candidate.new_false_detections_total.mean()), mean_positive_unmatched_excess=float(np.maximum(candidate.new_false_detections_total - candidate.original_unmatched_peaks_total, 0.0).mean()), mean_original_unmatched=float(candidate.original_unmatched_peaks_total.mean()), actual_bpcs=float(candidate.realized_bpcs.mean()), mean_cube_nmse=float(candidate.cube_nmse.mean()), worst_cube_nmse=float(candidate.cube_nmse.max()), low_zero_miss_observed=bool(candidate.low_threshold_labels_missed_total.sum() == 0))

def plot_scratch_vs_imitation_v12_10(learned):
    """One extra figure: fixed-validation outcomes, never monotonic envelopes.

    Horizontal frozen-BC/RDO curves are static baselines, not PPO checkpoints.
    Same dataset, state normalization, action library and cap. Actual rates may
    differ and are explicitly plotted; no rate-matched superiority is assumed.
    """
    history = learned['validation_checkpoint_history'].sort_values('episodes_seen')
    baseline = learned['frozen_baseline_measurements']
    rdo = learned['fixed_rdo_measurements']
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout='constrained')
    specs = [('low_misses', 'Lowest-threshold lost labels (RD + RA)'), ('sweep_misses', 'Lost object/map/threshold instances'), ('mean_positive_unmatched_excess', 'Positive unmatched excess per cube'), ('mean_unmatched', 'All unmatched peaks per cube'), ('mean_fixed_validation_reward', 'Reward on fixed validation cubes'), ('actual_bpcs', 'Actual base rate [bits/complex sample]')]
    for ax, (key, title) in zip(axes.flat, specs):
        ax.plot(history.episodes_seen, history[key], '-o', label='PPO from scratch (raw)', lw=1.8)
        ax.axhline(baseline[key], color='tab:orange', ls='--', label='Frozen imitation (raw)')
        ax.axhline(rdo[key], color='tab:green', ls=':', label='Calibrated RDO (raw)')
        ax.set(title=title, xlabel='Training cube episodes completed')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if key != 'mean_fixed_validation_reward':
            ax.set_ylim(bottom=0)
        if key in ('low_misses', 'sweep_misses'):
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_ylim(0, max(1.0, ax.get_ylim()[1]))
        if key == 'low_misses':
            ax.axhline(0, color='black', ls=':', label='Required zero losses')
        if key == 'actual_bpcs':
            ax.axhline(RL_RATE_TARGET_BPCS_V11, color='black', ls=':', label='Same cap')
            ax.set_ylim(0, max(ax.get_ylim()[1], 1.05 * RL_RATE_TARGET_BPCS_V11))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle('Does PPO learn without imitation? Fixed validation, no residual or candidate selector\nFrozen baselines use no PPO updates; curves are measured, not forced to improve')
    path = FIG_DIR_V11 / 'ppo_scratch_vs_frozen_imitation_learning_v12_10.png'
    save_publication_figure(fig, path, dpi=180)
    plt.close(fig)
    audit = history.copy()
    for prefix, values in [('frozen_imitation', baseline), ('calibrated_rdo', rdo)]:
        for key, value in values.items():
            audit[prefix + '_' + key] = value
    audit['actual_rate_gap_vs_frozen_bpcs'] = audit.actual_bpcs - baseline['actual_bpcs']
    safe_write_csv_v11(audit, TABLE_DIR_V11 / 'ppo_scratch_vs_frozen_imitation_learning_v12_10.csv')
    return [path]

class PPOAdamV12_11:
    """Separate Adam state and gradient clipping; critic cannot clip the actor."""
    actor_names = ('W1', 'b1', 'Wa', 'ba')
    critic_names = ('Wc1', 'bc1', 'Wv', 'bv')

    def __init__(self, params, learning_rate):
        self.actor = NumpyAdamV11({k: params[k] for k in self.actor_names}, learning_rate)
        self.critic = NumpyAdamV11({k: params[k] for k in self.critic_names}, learning_rate)
        self.gradient_records = []

    @property
    def t(self):
        return self.actor.t

    def step(self, gradients):
        actor = {k: gradients[k] for k in self.actor_names}
        critic = {k: gradients[k] for k in self.critic_names}
        norm = lambda d: float(np.sqrt(sum((np.sum(v.astype(float) ** 2) for v in d.values()))))
        self.gradient_records.append((norm(actor), norm(critic)))
        self.actor.step(actor)
        self.critic.step(critic)

def block_nmse_contributions_v12_11(cube, cube_hat):
    """Disjoint block errors / WHOLE-cube energy: sum == cube NMSE.

    Do not average individually normalized block NMSEs: small/quiet blocks
    would then receive a disproportionate weight. Accumulate energy in FP64.
    """
    denominator = float(np.sum(np.abs(np.asarray(cube, np.complex128)) ** 2, dtype=np.float64) + EPS)
    return np.asarray([float(np.sum(np.abs(np.asarray(cube[slc], np.complex128) - np.asarray(cube_hat[slc], np.complex128)) ** 2, dtype=np.float64)) / denominator for slc in iter_blocks(cube)], np.float64)

def persist_training_progress_v12_11(policy, normalizer, training_records, shaping_records, best):
    """Save after every completed update, not only after all training finishes."""
    safe_write_csv_v11(pd.DataFrame(training_records), TABLE_DIR_V11 / 'ppo_training_episodes_v12.csv')
    safe_write_csv_v11(pd.concat(shaping_records, ignore_index=True), TABLE_DIR_V11 / 'ppo_reward_shaping_checkpoints_v12.csv')
    safe_save_npz_v11(MODEL_DIR_V11 / 'latest_training_policy_v12_19.npz', **policy.state_dict())
    safe_save_npz_v11(MODEL_DIR_V11 / 'best_validation_policy_v12_19.npz', **best['state'])
    safe_write_json_v11(normalizer.as_dict(), MODEL_DIR_V11 / 'state_normalizer_v12.json')
    safe_write_json_v11(dict(checkpoint_id=best['checkpoint_id'], selected_pass=best['pass'], deployment_eligible=best['deployment_eligible'], resumable_optimizer=False), MODEL_DIR_V11 / 'best_validation_checkpoint_v12_19.json')

def check_learning_math_v12_11():
    """Small runtime guards for the user's preflight; no dataset/training needed."""
    policy = RadarPPOPolicyV11(3, 4, hidden_dim=5, seed=19)
    x = np.asarray([[0.1, -0.2, 0.3], [0.4, 0.2, -0.1]], np.float32)
    critic_grad = _network_gradients_v11(policy, x, np.zeros((2, 4), np.float32), np.ones(2, np.float32))
    actor_names = PPOAdamV12_11.actor_names
    critic_names = PPOAdamV12_11.critic_names
    if not all((np.all(critic_grad[k] == 0) for k in actor_names)):
        raise AssertionError('Critic gradients leaked into the actor.')
    logits_grad = policy.forward(x)[0].copy()
    logits_grad[:, 0] -= 1
    actor_grad = _network_gradients_v11(policy, x, logits_grad)
    if not all((np.all(actor_grad[k] == 0) for k in critic_names)):
        raise AssertionError('Actor gradients leaked into the critic.')
    before = {k: policy.params[k].copy() for k in actor_names}
    optimizer = PPOAdamV12_11(policy.params, 0.001)
    optimizer.step(critic_grad)
    if not all((np.array_equal(before[k], policy.params[k]) for k in actor_names)):
        raise AssertionError('A critic-only update moved the actor.')
    cube = (np.arange(60).reshape(3, 4, 5) + 1j).astype(np.complex64)
    cube_hat = 0.75 * cube
    contributions = block_nmse_contributions_v12_11(cube, cube_hat)
    exact = float(np.sum(np.abs(cube.astype(np.complex128) - cube_hat.astype(np.complex128)) ** 2) / (np.sum(np.abs(cube.astype(np.complex128)) ** 2) + EPS))
    if not np.isclose(contributions.sum(), exact, rtol=1e-12, atol=1e-12):
        raise AssertionError('Block distortion contributions do not sum to whole-cube NMSE.')
    check = pd.DataFrame([dict(check='independent_actor_critic_gradients', passed=True), dict(check='critic_only_update_preserves_actor', passed=True), dict(check='whole_cube_nmse_block_sum', passed=True)])
    safe_write_csv_v11(check, TABLE_DIR_V11 / 'preflight_learning_math_v12_11.csv')
    print('Retained independent-gradient/dense-distortion preflight: PASS')
    return check

def plot_credit_assignment_v12_11(learned):
    d = learned['training_log'].drop_duplicates('optimizer_step_after').copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout='constrained')
    x = np.arange(1, len(d) + 1)
    for key in ('actor_gradient_norm', 'critic_gradient_norm'):
        axes[0].plot(x, d[key], label=key.replace('_', ' '))
    axes[0].set(title='Independent gradient norms (before clipping)', ylabel='L2 norm')
    axes[0].legend(fontsize=8)
    axes[1].plot(x, d.mean_cube_nmse, marker='.', label='Stochastic training batches')
    axes[1].set(title='Raw reconstruction distortion', ylabel='Whole-cube NMSE')
    history = learned['validation_checkpoint_history']
    axes[2].plot(history.ppo_update_batches, history.mean_fixed_validation_reward, '-o')
    axes[2].set(title='Fixed validation: task + distortion reward', ylabel='Reward (higher is better)')
    for ax in axes:
        ax.set_xlabel('PPO update batch')
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle('V12.19: measured credit/quality diagnostics; no forced learning trend')
    path = FIG_DIR_V11 / 'ppo_credit_assignment_and_distortion_v12_11.png'
    save_publication_figure(fig, path, dpi=180)
    plt.close(fig)
    return [path]
V12_11_DISTORTION_WEIGHT = float(os.environ.get('RDO_V1211_DISTORTION_WEIGHT', '5000'))
if not np.isfinite(V12_11_DISTORTION_WEIGHT) or V12_11_DISTORTION_WEIGHT < 0:
    raise ValueError('RDO_V1211_DISTORTION_WEIGHT must be finite and nonnegative.')
if RL_GAMMA_V11 != 1.0:
    raise ValueError('V12.11 task-potential telescoping requires RDO_RL_GAMMA=1.')

def prepare_policy_source_v12_12(cube):
    source = coherent_codec_v14.prepare_source(cube, ACTION_LIBRARY_V11)
    fractions = np.asarray([p['tile_energy'] for p in source['prepared_blocks']], float) / max(source['whole_cube_energy'], EPS)
    suffix = np.cumsum(fractions[::-1])[::-1]
    for i, prepared in enumerate(source['prepared_blocks']):
        prepared['block_energy_fraction'] = float(fractions[i])
        prepared['future_energy_fraction'] = float(suffix[i + 1]) if i + 1 < len(suffix) else 0.0
        prepared['action_distortion'] = prepared['action_distortion_contributions']
    return source

def collect_same_cube_rollouts_v12_12(cube, row, policy, normalizer, processor, post_original, labels_cube, scaler, fixed_components, fixed_objective, pass_number, episode_index, cube_episode):
    """Original-only rewards on independent on-policy same-cube allocations.

    Historical reference parameters carry ORIGINAL diagnostics only. No
    competitor allocation/reward is evaluated, subtracted, or imitated.
    """
    source = prepare_policy_source_v12_12(cube)
    group_id = f'{pass_number}:{episode_index}:{row.record_name}'
    trajectories, records, shaping = ([], [], [])
    for sample in range(V12_12_ROLLOUTS_PER_CUBE):
        trajectory = encode_policy_v11(cube, policy, normalizer, True, source_context=source)
        contributions = trajectory['distortion_contributions']
        nmse = float(contributions.sum())
        trajectory['meta']['cube_nmse'] = nmse
        post = postprocess_cube(trajectory['cube_hat'], processor)
        components, detail = radar_task_components_v11(post_original, post, labels_cube)
        detail['cube_nmse'] = nmse
        objective = scaler.total(components, detail, trajectory['meta']['realized_bpcs'])
        terms = _safety_rate_terminal_reward_v12_3(components, detail, {}, {}, trajectory['meta']['realized_bpcs'], scaler)
        rewards = -V12_11_DISTORTION_WEIGHT * contributions
        rewards[-1] += terms['reward_task_only']
        actual, expected = (float(rewards.sum()), terms['terminal_label_first_reward'])
        if not np.isclose(actual, expected, rtol=1e-12, atol=1e-08):
            raise AssertionError('Original-only dense and terminal rewards do not sum to the declared objective.')
        trajectory.update(rewards=rewards, rollout_group=group_id, record_name=str(row.record_name), rollout_index=sample, constraint_costs=np.asarray([terms['constraint_low_cost'], terms['constraint_unmatched_cost']]))
        records.append({**{'pass': pass_number, 'episode_in_pass': episode_index, 'cube_episode': cube_episode, 'rollout_index': sample, 'rollout_group': group_id, 'record_name': row.record_name, 'scene_name': row.scene_name, 'realized_bpcs': trajectory['meta']['realized_bpcs'], 'episode_return': actual, 'reference_task_loss': np.nan, 'task_improvement_vs_rdo': np.nan, 'true_label_miss_count_improvement_vs_rdo': np.nan, 'shaping_checkpoints': 1, 'shaping_telescoping_error': abs(actual - expected), 'unique_actions': int(np.unique(trajectory['actions']).size)}, **action_diagnostics_v12_9(trajectory), **components, **detail, **objective, **terms})
        shaping.append(dict(checkpoint=1, end_block_exclusive=len(rewards), checkpoint_bpcs=trajectory['meta']['realized_bpcs'], original_only_task_reward=terms['reward_task_only'], low_threshold_labels_missed_total=detail['low_threshold_labels_missed_total'], low_label_gain=terms['low_label_gain'], aggregate_label_gain=terms['aggregate_label_gain'], new_false_detection_cost=components['new_false_detection_cost'], false_peak_excess=terms['false_peak_excess'], false_penalty=terms['false_penalty'], RD_log_power_mse=components['RD_log_power_mse'], RA_log_power_mse=components['RA_log_power_mse'], map_error_excess=terms['map_error_excess'], map_penalty=terms['map_penalty'], rate_penalty_for_reward=terms['rate_penalty_for_reward'], secondary_penalty=terms['secondary_penalty'], telescoping_error=abs(actual - expected), telescoping_expected_total=expected, telescoping_actual_total=actual, dense_cube_nmse=nmse, dense_distortion_reward=terms['reward_cube_distortion'], **{'pass': pass_number}, episode_in_pass=episode_index, record_name=row.record_name, rollout_index=sample, rollout_group=group_id))
        del trajectory['cube_hat']
        trajectories.append(trajectory)
        del post
    return (trajectories, records, pd.DataFrame(shaping))

def plot_same_cube_credit_v12_12(learned):
    d = learned['training_log'].drop_duplicates('optimizer_step_after').copy()
    h = learned['validation_checkpoint_history'].sort_values('episodes_seen')
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
    ax = axes[0, 0]
    ax.plot(d.cube_episode, d.within_cube_return_std, label='Within same cube (sampled actions)')
    ax.plot(d.cube_episode, d.between_cube_mean_return_std, label='Between cubes in update')
    ax.set(title='Reward variability: action contrast versus cube difficulty', ylabel='Reward standard deviation')
    ax = axes[0, 1]
    ax.plot(d.cube_episode, d.paired_advantage_rms)
    ax.set(title='Actor advantage after selected control variate', ylabel='Advantage RMS before normalization')
    ax = axes[1, 0]
    ax.plot(h.episodes_seen, h.low_misses, '-o', label='Raw PPO')
    ax.axhline(learned['frozen_baseline_measurements']['low_misses'], ls='--', label='Frozen imitation')
    ax.axhline(0, color='black', ls=':', label='Required zero misses')
    ax.set(title='Fixed validation: lowest-threshold missed labels', ylabel='Object/map count')
    ax = axes[1, 1]
    ax.plot(h.episodes_seen, h.mean_positive_unmatched_excess, '-o', label='Raw PPO')
    ax.axhline(learned['frozen_baseline_measurements']['mean_positive_unmatched_excess'], ls='--', label='Frozen imitation')
    ax.set(title='Fixed validation: unmatched excess outside annotation gates', ylabel='Mean excess peaks per cube')
    for ax in axes.flat:
        ax.set_xlabel('Training cube visits (each has independent rollouts)')
        ax.grid(alpha=0.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    fig.suptitle('V12.19 variance-aware PPO: measured learning, no residual refinement')
    path = FIG_DIR_V11 / 'ppo_same_cube_credit_and_learning_v12_12.png'
    save_publication_figure(fig, path, dpi=180)
    plt.close(fig)
    safe_write_csv_v11(d, TABLE_DIR_V11 / 'ppo_same_cube_update_diagnostics_v12_12.csv')
    return [path]
V12_12_ROLLOUTS_PER_CUBE = int(os.environ.get('RDO_V1219_ROLLOUTS_PER_CUBE', '4'))
if V12_12_ROLLOUTS_PER_CUBE < 1:
    raise ValueError('At least one on-policy allocation per cube is required.')
if V12_18_ADVANTAGE_MODE == 'loo_residual' and V12_12_ROLLOUTS_PER_CUBE < 2:
    raise ValueError('Leave-one-out credit requires >=2 independent allocations per cube.')
if float(os.environ.get('RDO_RL_GAMMA', '1.0')) != 1.0:
    raise ValueError('V12.12 uses undiscounted whole-cube rewards: gamma must equal 1.')
if 'RDO_RL_GAE_LAMBDA' in os.environ and float(os.environ['RDO_RL_GAE_LAMBDA']) != 1.0:
    raise ValueError('Unset RDO_RL_GAE_LAMBDA: V12.12 uses full Monte-Carlo returns, not GAE bootstrap.')
if 'RDO_RL_SHAPING_CHECKPOINTS' in os.environ and int(os.environ['RDO_RL_SHAPING_CHECKPOINTS']) != 1:
    raise ValueError('Unset RDO_RL_SHAPING_CHECKPOINTS: V12.12 evaluates completed cubes only.')

class PreservationConstraintsV12_18:
    """Stationary task weights. This is a penalty objective, not a safety guarantee."""

    def __init__(self):
        self.weights = V12_18_INITIAL_WEIGHTS.copy()
        self.updates = 0

    def update(self, trajectories):
        costs = np.asarray([t['constraint_costs'] for t in trajectories], float)
        if costs.ndim != 2 or costs.shape[1] != 2 or (not np.isfinite(costs).all()):
            raise ValueError('Malformed task-cost diagnostics.')
        self.updates += 1
        return dict(dual_update=self.updates, lambda_low_before=self.weights[0], lambda_low_after=self.weights[0], lambda_unmatched_before=self.weights[1], lambda_unmatched_after=self.weights[1], mean_low_constraint_cost=float(costs[:, 0].mean()), mean_unmatched_constraint_cost=float(costs[:, 1].mean()), dual_low_saturated=False, dual_unmatched_saturated=False)

    def as_dict(self):
        return dict(weights=self.weights.tolist(), adaptive_weights=False, semantics='stationary original-only penalty objective; zero misses must be measured')

def plot_constraint_learning_v12_13(learned):
    """Fixed-unit outcomes, not an artificial monotonically improving curve."""
    d = learned['training_log'].groupby('cube_episode', as_index=False).mean(numeric_only=True)
    h = learned['validation_checkpoint_history'].sort_values('episodes_seen')
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
    for ax, col, label in [(axes[0, 0], 'fixed_monitor_return', 'Fixed-weight monitor return'), (axes[0, 1], 'constraint_unmatched_cost', f'Unmatched constraint cost ({V12_18_UNMATCHED_MODE})')]:
        ax.plot(d.cube_episode, d[col], alpha=0.25, label='Mean independent samples')
        ax.plot(d.cube_episode, d[col].rolling(15, min_periods=1).mean(), label='15-cube mean')
        ax.set_title(label)
    axes[0, 1].axhline(0, color='black', ls=':', label='Original + count allowance')
    axes[1, 0].plot(h.episodes_seen, h.low_misses, '-o', label='Raw greedy PPO: missed labels')
    axes[1, 0].axhline(0, color='black', ls=':', label='Required zero')
    axes[1, 0].set_title('Same fixed validation cubes; no refinement')
    updates = learned['training_log'].drop_duplicates('optimizer_step_after')
    axes[1, 1].plot(updates.cube_episode, updates.lambda_low_after / V12_18_INITIAL_WEIGHTS[0], label='Low-miss multiplier / initial')
    axes[1, 1].plot(updates.cube_episode, updates.lambda_unmatched_after / V12_18_INITIAL_WEIGHTS[1], label='Unmatched multiplier / initial')
    axes[1, 1].set_title('Fixed task weights (stationary return)')
    for ax in axes.flat:
        ax.set_xlabel('Training cube visits')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle('V12.19 raw PPO — observed learning and constraint pressure')
    path = FIG_DIR_V11 / 'ppo_constraint_learning_v12_13.png'
    save_publication_figure(fig, path, dpi=180)
    plt.close(fig)
    safe_write_csv_v11(updates, TABLE_DIR_V11 / 'ppo_constraint_updates_v12_19.csv')
    return [path]

def _run_training_pipeline_v12_7(dataset_root: Path) -> Dict[str, Any]:
    dataset_root = Path(dataset_root)
    if not _valid_v6_1_1000_root(dataset_root, EXPECTED_DATASET_CUBES_V11):
        raise RuntimeError(f'V12 V4.3 requires the validated {EXPECTED_DATASET_NAME_V11}; received {dataset_root}.')
    labels = load_labels_v6_1000(dataset_root)
    cube_index_all = build_cube_index_v6_1000(dataset_root, labels, accepted_only=False)
    disk_count = _v6_1000_file_count(dataset_root)
    if disk_count != EXPECTED_DATASET_CUBES_V11 or len(cube_index_all) != EXPECTED_DATASET_CUBES_V11:
        raise RuntimeError(f'V12 requires exactly 1000 validated V44 MATLAB cubes; found files/index={disk_count}/{len(cube_index_all)}.')
    split = temporal_scene_split_v11(cube_index_all)
    train_rows = select_split_rows_v11(split['train'], RL_TRAIN_CUBES_V11)
    validation_rows = select_split_rows_v11(split['validation'], RL_VAL_CUBES_V11)
    test_rows = select_split_rows_v11(split['test'], NUM_CUBES_TO_EVALUATE_V11)
    for name, frame in [('train', train_rows), ('validation', validation_rows), ('test', test_rows)]:
        safe_write_csv_v11(frame.assign(split=name), TABLE_DIR_V11 / f'cube_index_{name}_v12.csv')
    cube_index_all = cube_index_all.sort_values(['scene_id', 'frame_in_scene', 'record_name']).reset_index(drop=True)
    safe_write_csv_v11(cube_index_all, TABLE_DIR_V11 / 'cube_index_all_matlab100_v12.csv')
    selected_names = {name: set(frame.record_name.astype(str)) for name, frame in [('train', train_rows), ('validation', validation_rows), ('test', test_rows)]}
    if selected_names['train'] & selected_names['validation'] or selected_names['train'] & selected_names['test'] or selected_names['validation'] & selected_names['test']:
        raise AssertionError('Train/validation/test leakage detected.')
    print('Training V12.19 original-only coherent-transform PPO ...')
    learned = train_taskaware_ppo_v11(train_rows, validation_rows, labels)
    display(learned['validation_gate'])
    plot_ppo_training_v11(learned['training_log'], FIG_DIR_V11 / 'ppo_training_diagnostics_v12_3.png')
    plot_v12_3_training_diagnostics(learned['training_log'], FIG_DIR_V11)
    return dict(cube_index_all=cube_index_all, split=split, train_rows=train_rows, validation_rows=validation_rows, test_rows=test_rows, learned=learned, test_evaluated=False)
NOTEBOOK_REVISION_V12_8 = '12.19.0-v44-preservation-balanced-ppo'
NOTEBOOK_REVISION_V12_7 = NOTEBOOK_REVISION_V12_8
V12_18_INITIAL_WEIGHTS = np.asarray([float(os.environ.get('RDO_V1219_LOW_INITIAL', '5000')), float(os.environ.get('RDO_V1219_UNMATCHED_INITIAL', '2000'))])
V12_18_MAP_EXCESS_WEIGHT = float(os.environ.get('RDO_V1219_MAP_EXCESS_WEIGHT', '500'))
if not np.isfinite(V12_18_INITIAL_WEIGHTS).all() or np.any(V12_18_INITIAL_WEIGHTS <= 0):
    raise ValueError('Stationary task weights must be finite and positive.')
V12_18_DUAL = PreservationConstraintsV12_18()
if (TABLE_DIR_V11 / 'ppo_training_episodes_v12.csv').exists():
    raise RuntimeError('This V12.19 output already contains training results. Set a new RDO_RL_RUN_TAG and restart the kernel to preserve them.')
print('FINAL ACTIVE BUILD:', NOTEBOOK_REVISION_V12_8)
print(f'Main PPO has no imitation; full action library; configured cap {RL_RATE_TARGET_BPCS_V11:g} bpcs.')
print('Stationary original-only task weights; learned state-value baseline; no underspending bonus.')
from radar_sim import ppo_v12_14_codec as coherent_codec_v14
from radar_sim.ppo_v12_14_margin import lowest_cfar_margin_loss_v12_14

def plot_margin_learning_v14(learned):
    d = learned['training_log'].groupby('cube_episode', as_index=False).mean(numeric_only=True)
    h = learned['validation_checkpoint_history']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout='constrained')
    for ax, key, title in zip(axes[:2], ['low_cfar_margin_deficit', 'cube_nmse'], ['Auxiliary low-CFAR peak-margin deficit', 'Whole-cube distortion (NMSE)']):
        ax.plot(d.cube_episode, d[key], alpha=0.25, label='Mean sampled allocations')
        ax.plot(d.cube_episode, d[key].rolling(15, min_periods=1).mean(), label='15-cube mean')
        ax.set(title=title, xlabel='Training cube visits')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[2].plot(h.episodes_seen, h.mean_low_cfar_margin_deficit, '-o', label='Fixed raw greedy validation')
    axes[2].set(title='Same validation cubes: peak-margin deficit', xlabel='Training cube visits')
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=8)
    return [_save_diagnostic_figure_v12_9(fig, 'ppo_low_cfar_margin_and_distortion')]
V12_18_MARGIN_FACTOR = float(os.environ.get('RDO_V1219_MARGIN_FACTOR', '1.25'))
V12_18_MARGIN_WEIGHT = float(os.environ.get('RDO_V1219_MARGIN_WEIGHT', '2500'))
if not np.isfinite([V12_18_MARGIN_FACTOR, V12_18_MARGIN_WEIGHT]).all() or V12_18_MARGIN_FACTOR < 1 or V12_18_MARGIN_WEIGHT < 0:
    raise ValueError('Invalid experimental low-CFAR auxiliary margin settings.')
V11_STATE_NAMES = ['doppler_tile_center', 'range_frequency_tile_center', 'running_rate_fraction', 'previous_keep_fraction', 'block_log_energy_ratio', 'spectral_top10_energy_fraction', 'cube_mean_spectral_concentration', 'cube_block_log_energy_std', 'cube_spectral_concentration_std', 'spectral_top1_energy_fraction', 'spectral_top5_energy_fraction', 'spectral_entropy_normalized', 'spectral_peak_to_mean_log', 'previous_q_div_16', 'block_energy_fraction', 'future_energy_fraction']
V11_STATE_NAMES += [f'action_{i}_{name}' for i in range(len(ACTION_LIBRARY_V11)) for name in ('rate_fraction', 'log1p_scaled_parseval_error')]
V12_18_MAP_WEIGHT = float(os.environ.get('RDO_V1219_MAP_WEIGHT', '500'))
if not np.isfinite(V12_18_MAP_WEIGHT) or V12_18_MAP_WEIGHT < 0:
    raise ValueError('The fixed original-map distortion weight must be finite and nonnegative.')

class OriginalTaskObjectiveV12_18:
    """Fixed physical/count units, never fitted to calibrated compressor data."""

    def __init__(self):
        self.scales = {name: 1.0 for name in V11_TASK_COMPONENT_NAMES}
        self.map_error_budgets = {}
        self.false_detection_budget = None
        self.supported_label_scale = 1.0

    def total(self, components, detail, realized_bpcs):
        terms = _safety_rate_terminal_reward_v12_3(components, detail, {}, {}, realized_bpcs, self, multipliers=V12_18_INITIAL_WEIGHTS)
        loss = -float(terms['fixed_monitor_return'])
        return dict(**terms, task_loss=loss, base_task_loss=loss, base_detection_loss=loss, total_objective=loss, alignment_task_loss=loss, rate_ratio=float(realized_bpcs) / RL_RATE_TARGET_BPCS_V11, rate_excess=terms['rate_hard_violation'], rate_penalty=terms['rate_penalty_for_reward'], rate_budget_bpcs=RL_RATE_TARGET_BPCS_V11, rate_within_budget=bool(realized_bpcs <= RL_RATE_TARGET_BPCS_V11 + 1e-06), objective_source='original_only_fixed_units', objective_uses_compressor_baseline=False)

    def as_dict(self):
        return dict(revision=NOTEBOOK_REVISION_V12_8, architecture='one 32-unit actor; independent learned state-value critic', objective='original-only task/distortion under an exact rate cap', preservation_multiplier=V12_19_PRESERVATION_MULTIPLIER, preservation_curve_mode=V12_19_CURVE_MODE, preservation_common_span_db=V12_19_COMMON_SPAN_DB, preservation_normalization='summed original-supported lost object-dB / one fixed common span', reward_component_export=True, scales=self.scales, scale_source='fixed units; no calibration', map_error_budgets={}, false_detection_budget=None, supported_label_scale=1.0, main_initialization='random', main_imitation_epochs=0, state_names=V11_STATE_NAMES, action_library=ACTION_LIBRARY_V11, training_cfar_thresholds=list(_nearest_training_thresholds_v11()), map_distortion_weight=V12_18_MAP_WEIGHT, dense_cube_nmse_weight=V12_11_DISTORTION_WEIGHT, low_cfar_margin_weight=V12_18_MARGIN_WEIGHT, low_cfar_margin_factor=V12_18_MARGIN_FACTOR, cfar_curve_weight=V12_18_MARGIN_WEIGHT * V12_18_CURVE_LOSS_SCALE, cfar_curve_bounds=dict(RD=[1.25, float(np.max(V12_7_RD_PRESERVATION_THRESHOLD_FACTORS))], RA=[1.25, float(np.max(V12_7_BASE_CFAR_THRESHOLD_FACTORS))]), unmatched_constraint_covers_continuous_interval=True, base_rate_cap_bpcs=RL_RATE_TARGET_BPCS_V11, training_uses_compressor_comparison=False, advantage_estimator=V12_18_ADVANTAGE_MODE, state_normalization=V12_18_NORMALIZER_MODE, reward_weights_stationary=True, unmatched_objective_mode=V12_18_UNMATCHED_MODE, checkpoint_selection_uses_compressor_comparison=False, comparators_built_after_main_training=True, constraints=V12_18_DUAL.as_dict())

def _original_source_reference_v12_18(cube, row, labels_cube, processor, cache):
    """Cache original maps/GT-count reference only; never compress a baseline."""
    key = (str(row.record_name), str(row.get('cube_path', '')))
    if key not in cache:
        post = postprocess_cube(cube, processor)
        _ensure_original_task_cache_v12_4_1(post, labels_cube)
        components, detail = radar_task_components_v11(post, post, labels_cube)
        detail['cube_nmse'] = 0.0
        components['_detail_v12_4_1'] = dict(detail)
        cache[key] = dict(post_original=_compact_task_post_v12_4_1(post), original_components=components, original_detail=detail)
        if len(cache) > 8:
            cache.pop(next(iter(cache)))
    return cache[key]

def evaluate_original_policy_rows_v12_18(rows, labels, policy, normalizer, scaler, postprocessors, cache=None):
    """Greedy fixed-validation evaluation against ORIGINAL data only."""
    cache = {} if cache is None else cache
    candidates, originals, action_logs = ([], [], [])
    for _, row in rows.iterrows():
        cube = load_cube_v6_1000(row)
        processor = _postprocessor_for_cube_v11(cube, postprocessors)
        labs = labels_for_cube_v6_1000(labels, row)
        ref = _original_source_reference_v12_18(cube, row, labs, processor, cache)
        encoded = encode_policy_v11(cube, policy, normalizer, False)
        post = postprocess_cube(encoded['cube_hat'], processor)
        components, detail = radar_task_components_v11(ref['post_original'], post, labs)
        detail['cube_nmse'] = float(block_nmse_contributions_v12_11(cube, encoded['cube_hat']).sum())
        metadata = dict(record_name=row.record_name, scene_name=row.scene_name, frame_in_scene=row.frame_in_scene)
        candidates.append({**metadata, 'realized_bpcs': encoded['meta']['realized_bpcs'], **components, **detail, **scaler.total(components, detail, encoded['meta']['realized_bpcs'])})
        original_components = {k: v for k, v in ref['original_components'].items() if not k.startswith('_')}
        originals.append({**metadata, 'realized_bpcs': 32.0, **original_components, **ref['original_detail']})
        log = encoded['block_log'].copy()
        log['record_name'], log['scene_name'] = (row.record_name, row.scene_name)
        action_logs.append(log)
        del cube, encoded, post
    return (pd.DataFrame(candidates), pd.DataFrame(originals), pd.concat(action_logs, ignore_index=True))

def original_validation_gate_v12_18(candidate):
    """Measured ORIGINAL-relative criteria; no comparator eligibility or budgets."""
    rows = []
    for mt in ('RD', 'RA'):
        missed = float(candidate[mt + '_low_threshold_labels_missed'].sum())
        rows.append(dict(criterion=mt + '_zero_low_threshold_misses', candidate_mean=missed, reference_mean=0.0, rule='missed original-detectable labels == 0', passed=missed == 0))
    n_thresholds = len(_nearest_training_thresholds_v11())
    allowance = V12_5_FALSE_ALLOWANCE_PER_THRESHOLD * n_thresholds * len(candidate)
    unmatched, original = (float(candidate.new_false_detections_total.sum()), float(candidate.original_unmatched_peaks_total.sum()))
    rows.append(dict(criterion='unmatched_vs_original', candidate_mean=unmatched, reference_mean=original, rule=f'original + {allowance:g} count allowance', passed=unmatched <= original + allowance))
    low, low_original = (float(candidate.low_unmatched_peaks_method_total.sum()), float(candidate.low_unmatched_peaks_original_total.sum()))
    low_allowance = 2 * V12_5_FALSE_ALLOWANCE_PER_THRESHOLD * len(candidate)
    rows.append(dict(criterion='lowest_unmatched_vs_original', candidate_mean=low, reference_mean=low_original, rule=f'original + {low_allowance:g} count allowance', passed=low <= low_original + low_allowance))
    for mt in ('RD', 'RA'):
        continuous_method = float(candidate[mt + '_unmatched_curve_method'].sum())
        continuous_source = float(candidate[mt + '_unmatched_curve_original'].sum())
        continuous_allowance = V12_5_FALSE_ALLOWANCE_PER_THRESHOLD * len(candidate)
        rows.append(dict(criterion=mt + '_continuous_interval_unmatched_vs_original', candidate_mean=continuous_method, reference_mean=continuous_source, rule=f'continuous dB-mean count <= original + {continuous_allowance:g}', passed=continuous_method <= continuous_source + continuous_allowance))
    worst_rate = float(candidate.realized_bpcs.max())
    rows.append(dict(criterion='per_cube_exact_rate_cap', candidate_mean=worst_rate, reference_mean=RL_RATE_TARGET_BPCS_V11, rule='every cube <= cap', passed=worst_rate <= RL_RATE_TARGET_BPCS_V11 + 1e-06))
    gate = pd.DataFrame(rows)
    gate['deployment_eligible'] = bool(gate.passed.all())
    gate['uses_compressor_comparison'] = False
    return gate

def plot_original_fixed_learning_v12_18(history):
    """Fixed greedy learning measured while NO comparator yet exists."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')
    for ax, key, title in zip(axes.flat, ['low_misses', 'sweep_misses', 'mean_unmatched', 'mean_fixed_validation_reward', 'actual_bpcs', 'mean_cube_nmse'], ['Lowest-CFAR missed labels', 'Sweep missed labels', 'Unmatched peaks', 'Fixed original-only return', 'Analytical rate [bpcs]', 'Whole-cube NMSE']):
        ax.plot(history.episodes_seen, history[key], '-o', label='Raw greedy PPO')
        if 'selected' in history and history.selected.astype(bool).any():
            selected = float(history.loc[history.selected.astype(bool), 'episodes_seen'].iloc[0])
            ax.axvline(selected, color='tab:green', ls=':', label='Selected checkpoint')
        if key == 'mean_unmatched':
            ax.plot(history.episodes_seen, history.mean_original_unmatched, '--', label='Uncompressed original')
        if key == 'low_misses':
            ax.axhline(0, color='black', ls=':', label='Required zero losses')
        ax.set(title=title, xlabel='Training cube visits')
        if key in ('low_misses', 'sweep_misses'):
            ax.set_ylim(0, max(1.0, float(history[key].max()) * 1.1))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle('Original-only PPO: fixed validation; no baseline controls training or selection')
    path = FIG_DIR_V11 / 'ppo_original_only_fixed_validation_v12_19.png'
    save_publication_figure(fig, path, dpi=180)
    plt.close(fig)
    return [path]

def plot_original_only_learning_v12_18(learned):
    """Publication-facing wrapper for the original-only fixed-validation figure."""
    return plot_original_fixed_learning_v12_18(learned['validation_checkpoint_history'])
from radar_sim.ppo_v12_16_cfar import cfar_preservation_regret_v12_16 as cfar_preservation_regret_v12_18

def cfar_curve_map_v12_18(map_original_db, map_method_db, labels, ranges, second_axis, map_type, gate):
    """Same fixed-NMS CA-CFAR scores as the integer publication sweeps."""
    threshold = float(min(_nearest_training_thresholds_v11()))
    original = precompute_nested_cfar_peak_masks_v11_2(as_linear_power(map_original_db), [0.0])[0.0]
    method = precompute_nested_cfar_peak_masks_v11_2(as_linear_power(map_method_db), [0.0])[0.0]
    bins, _, _ = _evaluable_label_bins_v12(labels, ranges, second_axis, map_type)
    if not np.array_equal(original['valid'], method['valid']):
        raise AssertionError('Original and decoded CFAR complete-window validity differ.')
    high = float(np.max(V12_7_RD_PRESERVATION_THRESHOLD_FACTORS if map_type == 'RD' else V12_7_BASE_CFAR_THRESHOLD_FACTORS))
    result = cfar_preservation_regret_v12_18(original['cfar_score'], method['cfar_score'], original['peaks'], method['peaks'], original['valid'], bins, gate, low_threshold=threshold, high_threshold=high, margin_factor=V12_18_MARGIN_FACTOR)
    result['margin_sum'] = result['low_margin_sum']
    return result

def validate_postprocessing_v12_18(cube, processor, post):
    """Reject silent axis truncation/sign/layout changes; never alter map values.

    This adapter is specifically for radar_sim.processing.PostProcessing, not a
    generic processor with unknown conventions. All policies use this same path.
    The analytical axes are derived from the actual per-cube radar specification.
    """
    x = np.asarray(cube)
    expected = (int(processor.num_rx), int(processor.M), int(processor.N))
    if x.shape != expected or not np.iscomplexobj(x) or (not np.isfinite(x).all()):
        raise ValueError('Finite complex IQ must have [Rx, chirp, fast-time] shape ' + str(expected))
    n, m = (expected[2], expected[1])
    ranges = np.asarray(post['ranges'], float)
    velocity = np.asarray(post['vel'], float)
    angles = np.asarray(post['angle_axis_deg'], float)
    rd, ra = (np.asarray(post['RD']), np.asarray(post['RA']))
    if rd.shape != (m, n) or ra.shape != (int(processor.theta_res), n):
        raise ValueError('RD must be [Doppler, range] and RA [angle, range]; no axis truncation allowed.')
    if ranges.shape != (n,) or velocity.shape != (m,) or angles.shape != (int(processor.theta_res),):
        raise ValueError('Map dimensions and physical axes disagree.')
    if not all((np.isfinite(a).all() for a in (rd, ra, ranges, velocity, angles))):
        raise ValueError('Post-processing maps/axes must be finite.')
    expected_range = processor.c0 * np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / processor.fs)) / (2.0 * processor.S)
    expected_velocity = processor.lam / 2.0 * np.fft.fftshift(np.fft.fftfreq(m, d=processor.Tchirp))
    expected_angle = np.linspace(-60.0, 60.0, int(processor.theta_res))
    for actual, required, name in [(ranges, expected_range, 'range'), (velocity, expected_velocity, 'radial velocity'), (angles, expected_angle, 'Bartlett azimuth')]:
        if not np.allclose(actual, required, rtol=1e-10, atol=1e-10):
            raise ValueError('Unexpected ' + name + ' axis; verify the generator/processor convention.')
    return post

def plot_cfar_curve_learning_v12_18(learned):
    """Unsmoothed fixed validation plus honestly labeled sampled training curves."""
    data = learned['training_log'].groupby('cube_episode', as_index=False).mean(numeric_only=True)
    history = learned['validation_checkpoint_history']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    specs = [('cfar_curve_regret', 'mean_cfar_curve_regret', 'Lost label support: continuous CFAR-curve area'), ('unmatched_curve_method_total', 'mean_unmatched_curve_method_total', 'Outside-GT peaks: continuous CFAR-curve mean')]
    for column, (key, hkey, title) in enumerate(specs):
        axes[0, column].plot(data.cube_episode, data[key], marker='.', alpha=0.3, label=f'Mean of {V12_12_ROLLOUTS_PER_CUBE} sampled allocations')
        axes[0, column].plot(data.cube_episode, data[key].rolling(15, min_periods=1).mean(), marker='.', label='15-cube moving mean')
        axes[1, column].plot(history.episodes_seen, history[hkey], '-o', label='Same validation cubes; greedy raw PPO')
        if column == 1:
            axes[1, column].plot(history.episodes_seen, history.mean_unmatched_curve_original_total, '--', label='Uncompressed original')
        for ax in axes[:, column]:
            ax.set(title=title, xlabel='Training cube visits')
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
    fig.suptitle('Full-interval task loss: RD to 46.92 dB; RA to 21.07 dB\nContinuous integrals, not finite-grid integer counts; trends are not forced')
    path = FIG_DIR_V11 / 'ppo_full_interval_cfar_learning_v12_19.png'
    save_publication_figure(fig, path, dpi=180)
    plt.close(fig)
    return [path]

def postprocessing_contract_v12_18():
    return dict(cube_layout='[Rx, chirp, fast-time]', range_axis='c0 * fftshift(fftfreq(N, 1/fs)) / (2*S)', velocity_axis='lambda/2 * fftshift(fftfreq(M, Tchirp)); generator radial sign retained', windows='Hann fast and slow', clutter_filter='nonperiodic two-tap MTI, first chirp zero', ra='ULA Bartlett, lambda/2 spacing, noncoherent power integrated over Doppler; NOT Capon', publication_detector='same CA-CFAR complete-window validity and fixed power NMS for every method', count_rule='one original-supported object per map/threshold; any valid in-gate NMS peak preserves it', unmatched='outside union of ALL evaluable annotation gates; not confirmed physical false alarms', point_cloud_figures='RA CA-CFAR projection; inherited native GO-CFAR/DBSCAN PC metrics are a separate protocol', interval_metric='unchanged exact object-dB integral; legacy per-map normalized metrics retained for reports', preservation_loss_normalization=V12_19_CURVE_MODE, preservation_common_span_db=V12_19_COMMON_SPAN_DB, preservation_multiplier=V12_19_PRESERVATION_MULTIPLIER, integer_sweep_metrics_unchanged=True, refinement_used=False, deployment_encoder_uses_labels=False, label_metric='geometric gate support; overlapping gates may share a peak, NOT unique object identification', dataset='V44 25 complete traffic episodes, unchanged V43 signal model', splitting='whole-episode disjoint; seen templates, not unseen-layout generalization')
V12_18_CURVE_LOSS_SCALE = float(os.environ.get('RDO_V1219_CURVE_LOSS_SCALE', '1.0'))
if not np.isfinite(V12_18_CURVE_LOSS_SCALE) or V12_18_CURVE_LOSS_SCALE < 0:
    raise ValueError('The CFAR-curve loss scale must be finite and nonnegative.')
print('CFAR interval task loss: RD', V12_7_RD_PRESERVATION_THRESHOLD_FACTORS[[0, -1]], 'RA', V12_7_BASE_CFAR_THRESHOLD_FACTORS[[0, -1]])

def summarize_learning_v12_18(outputs):
    """Export measured changes, never assert an improving curve into existence."""
    h = outputs['learned']['validation_checkpoint_history'].sort_values('episodes_seen')
    first, final = (h.iloc[0], h.iloc[-1])
    selected = h.loc[h.selected.astype(bool)].iloc[0]
    rows = []
    for key in ('low_misses', 'sweep_misses', 'mean_unmatched', 'mean_fixed_validation_reward', 'actual_bpcs', 'mean_cube_nmse'):
        rows.append(dict(metric=key, initial=float(first[key]), last=float(final[key]), selected=float(selected[key]), last_minus_initial=float(final[key] - first[key])))
    summary = pd.DataFrame(rows)
    safe_write_csv_v11(summary, TABLE_DIR_V11 / 'measured_learning_summary_v12_19.csv')
    safe_write_json_v11(dict(zero_probe_misses_observed=bool(selected.low_misses == 0), last_probe_misses_decreased=bool(final.low_misses < first.low_misses), last_probe_unmatched_decreased=bool(final.mean_unmatched < first.mean_unmatched), last_probe_reward_increased=bool(final.mean_fixed_validation_reward > first.mean_fixed_validation_reward), probe_cubes=int(outputs['learned']['validation_probe_count']), full_validation_cubes=len(outputs['validation_rows']), full_test_cubes=len(outputs['test_rows']), guarantees_zero_misses=False, refinement_used=False, bitrates_are_analytical=True), TABLE_DIR_V11 / 'learning_outcomes_v12_19.json')
    display(summary)
    return summary

def run_experiment():
    if RL_RUN_MODE_V11 == 'definitions_only' or (not RUN_PIPELINE_V11 and (not RUN_DEBUG_V11)):
        return dict(dataset_root=DATASET_ROOT, revision=NOTEBOOK_REVISION_V12_8)
    if RUN_DEBUG_V11:
        return dict(preflight=run_preflight_debug_v11(), learning_math=check_learning_math_v12_11())
    outputs = run_v12_7_pipeline(DATASET_ROOT)
    outputs['learning_summary'] = summarize_learning_v12_18(outputs)
    return outputs
V11_STATE_NAMES += ['remaining_budget_fraction', 'minimum_remaining_cost_fraction', 'maximum_remaining_cost_fraction', 'allowance_per_remaining_coefficient']

def normalizer_probe_rows_v12_18(train_rows):
    """Cover beginning/middle/end of each TRAIN episode, without holdout access."""
    if V12_18_NORMALIZER_MODE == 'legacy':
        return select_split_rows_v11(train_rows, min(RL_CALIBRATION_CUBES_V11, len(train_rows)))
    selected = []
    for _, episode in train_rows.groupby('scene_name', sort=True):
        episode = episode.sort_values('frame_in_scene')
        positions = np.unique(np.rint(np.linspace(0, len(episode) - 1, min(V12_18_NORMALIZER_FRAMES, len(episode)))).astype(int))
        selected.append(episode.iloc[positions])
    if not selected:
        raise ValueError('No training episodes for state normalization.')
    return pd.concat(selected, ignore_index=True)
