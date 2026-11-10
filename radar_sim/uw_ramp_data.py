"""UW raw-ADC sequences as RAMP-CNN inputs: split, labels, 3D slices and label maps.

Slices follow slice3d.py of the RAMP-CNN repository: Hamming windows, 128-point range,
Doppler and angle FFTs on the 8 virtual channels, a complex RA slice at chirps 0 and
128 and magnitude RV and VA slices from the first and last 128 chirps. Only the frames
in the store differ between the original and a decoded sequence.

Differences from the repository's UWCR conversion, each needed for this data:
  * ids are read as floats: 12 of 22 sequences write ``4.00,0.00,...``, which its
    ``int(line[1])`` refuses;
  * positions stay in metres. It truncates px and py to whole metres, up to 1 m of error
    at a 0.22 m range bin;
  * person maps to pedestrian (its class table has no ``person``), bus to car and
    motorbike to cyclist, so these objects are not background;
  * the range and angle axes are the ones of the 128-point FFTs (see ramp_cnn/PATCHES.md).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import torch

from . import uw_detection_physics as uw

N_FFT = 128
WINDOW = 16
RANGE_LIMITS, ANGLE_LIMIT = (1., 24.), 90.
TO_RAMP = {0: 0, 80: 1, 3: 1, 2: 2, 5: 2, 7: 2}  # pedestrian 0, cyclist 1, car 2
RAMP_CLASSES = ('pedestrian', 'cyclist', 'car')
SIGMAS = dict(pedestrian=15, cyclist=20, car=30)
SIGMA_INTERVALS = dict(pedestrian=(5, 15), cyclist=(8, 20), car=(10, 30))
LENGTHS = dict(pedestrian=1, cyclist=2, car=3)
GAUSSIAN_THRESHOLD = 36
RANGE_GRID = np.arange(N_FFT)*uw.RANGE_BIN_M
ANGLE_GRID = np.degrees(np.arcsin(np.arange(-N_FFT//2, N_FFT//2)/(N_FFT/2)))

# Fixed from label counts and camera scenes, before any detector ran. Whole sequences.
SPLIT = {
    'train': ('2019_04_09_bms1000', '2019_04_09_css1000', '2019_04_09_pms1000', '2019_04_09_pms2000', '2019_04_30_cm1s000',
              '2019_04_30_mlms000', '2019_04_30_pbss000', '2019_04_30_pcms001', '2019_05_09_bm1s007', '2019_05_09_cm1s003',
              '2019_05_09_pcms002', '2019_05_29_cm1s014', '2019_05_29_pbms007', '2019_05_29_pcms005'),
    'validation': ('2019_04_09_pms3000', '2019_04_30_mlms001', '2019_05_09_pbms004', '2019_05_29_bcms000'),
    'test': ('2019_04_09_cms1000', '2019_04_30_pbms002', '2019_05_09_mlms003', '2019_05_29_mlms006')}
PARKING_LOT = ('2019_04_09_bms1000', '2019_04_09_cms1000', '2019_04_09_css1000', '2019_04_09_pms1000', '2019_04_09_pms2000',
               '2019_04_09_pms3000', '2019_04_30_cm1s000', '2019_04_30_pbms002', '2019_04_30_pbss000', '2019_05_29_pbms007')


def scenario(sequence):
    return 'parking_lot' if sequence in PARKING_LOT else 'curbside'


@dataclass
class Sequence:
    name: str
    role: str
    root: Path
    frames: np.ndarray

    def radar(self, frame):
        return self.root/'radar_raw_frame'/f'{int(frame):06d}.mat'

    def label(self, frame):
        return self.root/'text_labels'/f'{int(frame):010d}.csv'

    def objects(self, frame):
        path = self.label(frame)
        return objects_of(uw.read_labels(path)) if path.is_file() else []


def sequences(root):
    """Consecutive radar frames of every sequence, in split order; bodies are not read."""
    out = {}
    for role, names in SPLIT.items():
        out[role] = []
        for name in names:
            folder = Path(root)/name
            frames = np.array(sorted(int(p.stem) for p in (folder/'radar_raw_frame').glob('*.mat')))
            if not len(frames) or not np.array_equal(frames, np.arange(frames[0], frames[0]+len(frames))):
                raise ValueError(f'{name}: radar frames are not consecutive.')
            out[role].append(Sequence(name, role, folder, frames))
    return out


def objects_of(rows):
    """Label rows -> [range index, angle index, RAMP class] inside the network's field of view."""
    objects = []
    for _, cls, px, py, _, _ in rows:
        if cls not in TO_RAMP:  # the repository's converter skips ids outside its label map too
            continue
        distance, angle = math.hypot(px, py), math.degrees(math.atan2(px, py))
        if RANGE_LIMITS[0] <= distance <= RANGE_LIMITS[1] and abs(angle) <= ANGLE_LIMIT:
            objects.append([int(np.abs(RANGE_GRID-distance).argmin()), int(np.abs(ANGLE_GRID-angle).argmin()), TO_RAMP[cls]])
    return objects


class Slicer:
    """slice3d.py on the device for [batch, channel, chirp, sample] cubes."""

    def __init__(self, device):
        real = dict(dtype=torch.float64, device=device)
        self.range_window = torch.hamming_window(uw.N_SAMPLES, periodic=False, **real)[None, None, None, :]
        self.doppler_window = torch.hamming_window(N_FFT, periodic=False, **real)[None, None, :, None]
        self.angle_window = torch.hamming_window(uw.N_CHANNELS, periodic=False, **real)[None, :, None, None]

    @torch.no_grad()
    def __call__(self, cube):
        """-> RA [b, 2 chirps, range, angle, re/im], RV [b, range, velocity, 2], VA [b, angle, velocity, 2]."""
        ranged = torch.fft.fft(cube.to(torch.complex128)*self.range_window, N_FFT, dim=3)
        rv, va = [], []
        for chirps in (ranged[:, :, :N_FFT], ranged[:, :, N_FFT-1:]):
            doppler = torch.fft.fftshift(torch.fft.fft(chirps*self.doppler_window, N_FFT, dim=2), dim=2)
            rv.append(doppler.abs().sum(1).transpose(1, 2)/uw.N_CHANNELS)
            angle = torch.fft.fftshift(torch.fft.fft(doppler*self.angle_window, N_FFT, dim=1), dim=1)
            va.append(angle.abs().sum(3)/N_FFT)
        selected = ranged[:, :, (0, N_FFT)]*self.angle_window
        ra = torch.fft.fftshift(torch.fft.fft(selected, N_FFT, dim=1), dim=1).permute(0, 2, 3, 1)
        return (torch.view_as_real(ra).to(torch.float32), torch.stack(rv, 3).to(torch.float32), torch.stack(va, 3).to(torch.float32))


class SliceStore:
    """Three float32 arrays of a sequence on disk; frames are addressed by position."""
    SHAPES = dict(ra=(2, N_FFT, N_FFT, 2), rv=(N_FFT, N_FFT, 2), va=(N_FFT, N_FFT, 2))

    def __init__(self, folder, count=None):
        self.folder = Path(folder)
        if count is not None:
            self.folder.mkdir(parents=True, exist_ok=False)
            self.arrays = {k: np.lib.format.open_memmap(self.folder/f'{k}.npy', 'w+', np.float32, (count, *s)) for k, s in self.SHAPES.items()}
        else:
            self.arrays = {k: np.load(self.folder/f'{k}.npy', mmap_mode='r') for k in self.SHAPES}
        self.count = len(self.arrays['ra'])

    def write(self, at, ra, rv, va):
        for key, value in dict(ra=ra, rv=rv, va=va).items():
            self.arrays[key][at:at+len(value)] = value.cpu().numpy()

    def flush(self):
        for array in self.arrays.values():
            array.flush()

    def window(self, start):
        """[16 frames] -> network layout: RA [2, 32, r, a], RV and VA [1, 32, ...], chirps interleaved."""
        ra = np.moveaxis(self.arrays['ra'][start:start+WINDOW].reshape(2*WINDOW, N_FFT, N_FFT, 2), 3, 0)
        rv = np.moveaxis(self.arrays['rv'][start:start+WINDOW], 3, 1).reshape(1, 2*WINDOW, N_FFT, N_FFT)
        va = np.moveaxis(self.arrays['va'][start:start+WINDOW], 3, 1).reshape(1, 2*WINDOW, N_FFT, N_FFT)
        return ra, rv, va


def confmap(objects):
    """generate_confmap + normalize_confmap of the repository, without the per-cell loops."""
    out = np.zeros((len(RAMP_CLASSES), N_FFT, N_FFT))
    if not objects:
        return out
    rows, columns = np.mgrid[:N_FFT, :N_FFT]
    for r, a, c in objects:
        name = RAMP_CLASSES[c]
        low, high = SIGMA_INTERVALS[name]
        sigma = min(max(2*np.arctan(LENGTHS[name]/(2*RANGE_GRID[r]))*SIGMAS[name], low), high)
        distance = (((r-rows)*2)**2+(a-columns)**2)/sigma**2
        out[c] = np.maximum(out[c], np.where(distance < GAUSSIAN_THRESHOLD, np.exp(-distance/2)/(2*math.pi), 0))
    low, high = out.min(), out.max()
    return (out-low)/(high-low) if high != low else out


def normalization(stores, every=8):
    """Single mean and std per slice type, as the repository's mean1/std1; TRAIN only."""
    out = {}
    for key in SliceStore.SHAPES:
        total, square, count = 0., 0., 0
        for store in stores:
            for block in range(0, store.count, every):
                value = np.asarray(store.arrays[key][block], dtype=np.float64)
                total, square, count = total+value.sum(), square+(value**2).sum(), count+value.size
        mean = total/count
        out[key] = dict(mean=float(mean), std=float(math.sqrt(square/count-mean**2)))
    return out


class Windows(torch.utils.data.Dataset):
    """16-frame windows with label maps of one split; inputs are normalized here."""

    def __init__(self, items, norm, stride):
        self.items, self.norm = items, norm                     # [(Sequence, SliceStore, objects per frame)]
        self.index = [(i, s) for i, (_, store, _) in enumerate(items) for s in range(0, store.count-WINDOW+1, stride)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, position):
        i, start = self.index[position]
        _, store, objects = self.items[i]
        ra, rv, va = store.window(start)
        n = self.norm
        scale = lambda value, key: torch.from_numpy((value-n[key]['mean'])/n[key]['std']).float()
        maps = np.stack([confmap(objects[start+k]) for k in range(WINDOW)], 1)
        return scale(ra, 'ra'), scale(rv, 'rv'), scale(va, 'va'), torch.from_numpy(maps).float()
