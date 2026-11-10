"""RADIal raw ADC boundary for V12.7; no training or dataset writes on import.

Reference: valeoai/RADIal DBReader/DBReader.py, SensorsReaders.py and
SignalProcessing/rpl.py. RD coordinates are *unshifted cyclic bins*, NOT m/s.
The supplied CSV's misleading radar_D_mps header is preserved as source metadata.
"""
from __future__ import annotations

import ast
import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from scipy import fft, signal

SHAPE = (16, 256, 512)  # receiver, chirp, fast-time sample
CHIP_BYTES = 512 * 256 * 4 * 2 * 2
RANGE_EXTENT_M = 103.0  # exact convention in official __get_PCL
SENSOR_OFFSETS_US = {"camera": 0, "scala": -40000, "radar_ch0": -180000,
                     "radar_ch1": -180000, "radar_ch2": -180000,
                     "radar_ch3": -180000, "gps": 0, "can": 0}


def portable_path(value):
    text = str(value).strip().strip('"').replace("\\", "/")
    match = re.match(r"^([A-Za-z]):/(.*)$", text)
    if match and os.name != "nt":
        return Path("/mnt") / match[1].lower() / match[2]
    return Path(text).expanduser()


@dataclass(frozen=True)
class RadialCubeConfig:
    N: int = 512
    M: int = 256
    num_rx: int = 16
    # No guessed chirp timing, velocity conversion or uniform-array parameters.
    coordinate_contract: str = "official_range_metres_and_cyclic_doppler_bins"


def decode_adc_chips(chips):
    """Lossless reshaping of the four int16 streams; returned working dtype complex64."""
    frames = {}
    for ch in range(4):
        adc = np.asarray(chips[ch])
        if adc.ndim != 1 or adc.size != CHIP_BYTES//2 or adc.dtype.kind != "i" or adc.dtype.itemsize != 2:
            raise ValueError(f"Chip {ch} is not a complete int16 frame")
        iq = adc[0::2].astype(np.float32)+1j*adc[1::2].astype(np.float32)
        frames[ch] = iq.reshape((512,4,256),order="F").transpose(0,2,1)
    cube = np.concatenate([frames[3],frames[0],frames[1],frames[2]],axis=2).transpose(2,1,0)
    return np.ascontiguousarray(cube,dtype=np.complex64)


def parse_events(folder, *, allow_truncated_final_event=False, audit=None):
    """Metadata-only counterpart of official parse_recording, same offsets/order."""
    folder = Path(folder)
    records = {}
    for p in folder.iterdir():
        sensor_name = p.stem[len(folder.name) + 1:]
        if p.is_file() and sensor_name in SENSOR_OFFSETS_US:
            records[sensor_name] = {"filename": p, "events": []}
    event_path = folder / (folder.name + "_events_log.rec")
    lines = event_path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, 1):
        fields = line.split()
        if len(fields) not in (11, 17):
            # A downloaded prefix may end halfway through the final log event.
            # Discard only that unparseable EOF fragment, never an internal row.
            # Earlier synchronized indices retain their original order. Any
            # unavailable ADC body is still excluded by the separate byte audit.
            if (allow_truncated_final_event and line_number == len(lines)
                    and len(fields) >= 11 and fields[0] == 'timestamp:'
                    and fields[9] == 'sensor:' and fields[10] in records):
                item = dict(scene_name=folder.name, event_log=str(event_path),
                            line_number=line_number, reason='truncated_final_event_excluded',
                            sensor=fields[10], frame_indices_renumbered=False)
                if audit is not None:
                    audit.append(item)
                warnings.warn(f'Excluded truncated final event in {event_path}; exported in event audit.')
                continue
            raise ValueError(f"Unexpected event format: {event_path}: {line[:120]}")
        sensor_name = fields[10]
        if sensor_name not in records:
            continue
        row = dict(timestamp=int(fields[1]), timeofissue=int(fields[4]), sample=int(fields[7]))
        if len(fields) == 17:
            row.update(offset=int(fields[13]), datasize=int(fields[16]))
        records[sensor_name]["events"].append(row)
    for sensor_name, value in records.items():
        value["events"] = pd.DataFrame(value["events"])
        table = value["events"]
        if len(table):
            offset = int(table.timeofissue.iloc[0] - table.timestamp.iloc[0]) + SENSOR_OFFSETS_US[sensor_name]
            table["timestamp"] += offset
    return records


def synchronized_frames(events, tolerance_us=40000):
    """Official radar-master matching; include camera/scala/CAN in row filtering.

    Labels index the filtered SyncReader table, not raw chip ordinal positions.
    Do not renumber that table after rejecting unavailable/truncated ADC bytes.
    """
    chips = [f"radar_ch{i}" for i in range(4)]
    if any(c not in events for c in chips):
        raise ValueError("All four radar chip files are required")
    auxiliary = [s for s in events if s not in chips and s not in {"gps", "preview", "None"}]
    lookup = {}
    for c in chips:
        lookup[c] = {}
        for i, sample in enumerate(events[c]["events"]["sample"]):
            lookup[c].setdefault(int(sample), i)  # official np.where(...)[0][0]
    output = []
    for i, master in events["radar_ch3"]["events"].iterrows():
        n = int(master["sample"])
        match = {"radar_ch3": int(i), "radar_ch0": lookup["radar_ch0"].get(n+1, -1),
                 "radar_ch2": lookup["radar_ch2"].get(n+2, -1),
                 "radar_ch1": lookup["radar_ch1"].get(n+3, -1)}
        for sensor_name in auxiliary:
            table = events[sensor_name]["events"]
            if table.empty:
                match[sensor_name] = -1
            else:
                delta = np.abs(table.timestamp.to_numpy() - int(master.timestamp))
                j = int(delta.argmin())
                match[sensor_name] = j if delta[j] <= tolerance_us else -1
        if all(j >= 0 for j in match.values()):
            output.append(match)
    return output


class RadialDataset:
    source_bpcs = 32.0
    cfg = RadialCubeConfig()

    def __init__(self, root, reference_root, tolerance_us=40000, *,
                 allow_truncated_final_event=False):
        self.root = portable_path(root).resolve()
        self.reference_root = portable_path(reference_root).resolve()
        self.tolerance_us = int(tolerance_us)
        self.allow_truncated_final_event = bool(allow_truncated_final_event)
        self.event_audit = []
        if self.tolerance_us <= 0:
            raise ValueError("RADIAL_SYNC_TOLERANCE_US must be positive")
        self.calibration_path = self.reference_root / "SignalProcessing/CalibrationTable.npy"
        self.rpl_path = self.reference_root / "SignalProcessing/rpl.py"
        self.label_path = self.root / "labels_CVPR.csv"
        for path in [self.calibration_path, self.rpl_path, self.label_path]:
            if not path.is_file():
                raise FileNotFoundError(path)
        raw = pd.read_csv(self.label_path)
        required = {"numSample", "dataset", "index", "radar_R_m", "radar_A_deg",
                    "radar_D_mps", "radar_X_m", "radar_Y_m", "Difficult",
                    "laser_X_m", "laser_Y_m", "Annotation"}
        if not required.issubset(raw):
            raise ValueError(f"Missing label columns: {required-set(raw)}")
        raw["source_csv_row"] = np.arange(len(raw))
        self._tracked = {self.label_path, self.calibration_path, self.rpl_path}
        self.sync_reference_path = self.reference_root/'DBReader/DBReader/DBReader.py'
        if self.sync_reference_path.is_file():
            self._tracked.add(self.sync_reference_path)
        self.events = {}
        cube_rows, label_rows, audit_rows, alternative_rows = [], [], [], []
        for scene_id, folder in enumerate(sorted(self.root.glob("RECORD*"))):
            if not folder.is_dir():
                continue
            group = raw.loc[raw.dataset == folder.name]
            if group.empty:
                continue  # unannotated frames are NOT negative-labelled training data
            events = parse_events(folder, allow_truncated_final_event=self.allow_truncated_final_event,
                                  audit=self.event_audit)
            self.events[folder.name] = events
            self._tracked.add(folder / (folder.name + "_events_log.rec"))
            for data in events.values():
                self._tracked.add(data["filename"])
            table = synchronized_frames(events, self.tolerance_us)
            alternate_us = 40000 if self.tolerance_us == 20000 else 20000
            alternate = synchronized_frames(events, alternate_us)
            for frame, frame_labels in group.groupby("index", sort=True):
                if not np.isfinite(frame) or int(frame) != frame:
                    raise ValueError(f"Noninteger synchronized index in {folder.name}")
                frame = int(frame)
                if frame < 0 or frame >= len(table):
                    audit_rows.append(dict(scene_name=folder.name, frame_in_scene=frame,
                                           valid=False, reason="label_index_outside_sync_table"))
                    continue
                match = table[frame]
                record_name = f"{folder.name}_frame_{frame:06d}"
                num_samples = frame_labels.numSample.unique()
                if len(num_samples) != 1:
                    raise ValueError(f"Multiple numSample values at {record_name}")
                row = dict(record_name=record_name, scene_name=folder.name, scene_id=scene_id,
                           frame_in_scene=frame, global_frame=int(num_samples[0]),
                           cube_path=str(folder), accepted=True, split_role="unassigned",
                           source_bytes=4*CHIP_BYTES, source_bpcs=32.0)
                row["annotation_types"] = ";".join(sorted(frame_labels.Annotation.astype(str).unique()))
                reasons = []
                for ch in range(4):
                    data = events[f"radar_ch{ch}"]
                    event = data["events"].iloc[match[f"radar_ch{ch}"]]
                    offset, length = int(event.offset), int(event.datasize)
                    row.update({f"chip{ch}_path": str(data["filename"]),
                                f"chip{ch}_offset": offset, f"chip{ch}_sample": int(event["sample"]),
                                f"chip{ch}_ordinal": match[f"radar_ch{ch}"]})
                    if length != CHIP_BYTES or offset < 0 or offset+length > data["filename"].stat().st_size:
                        reasons.append(f"chip{ch}_incomplete")
                row["timestamp_us"] = int(events["radar_ch3"]["events"].iloc[match["radar_ch3"]].timestamp)
                same = frame < len(alternate) and all(match[c] == alternate[frame][c] for c in match)
                alternative_rows.append(dict(record_name=record_name, selected_tolerance_us=self.tolerance_us,
                                             alternative_tolerance_us=alternate_us, same_frame=bool(same)))
                audit_rows.append(dict(scene_name=folder.name, frame_in_scene=frame,
                                       valid=not reasons, reason=";".join(reasons) or "complete"))
                if reasons:
                    continue
                cube_rows.append(row)
                for _, label in frame_labels.iterrows():
                    r, a, d = float(label.radar_R_m), float(label.radar_A_deg), float(label.radar_D_mps)
                    # -1 sentinel is valid only as an explicit annotation of no objects.
                    empty = all(float(label[c]) == -1 for c in
                                ("radar_R_m", "radar_A_deg", "radar_D_mps", "laser_X_m", "laser_Y_m"))
                    if empty:
                        continue
                    valid = bool(np.isfinite([r,a,d]).all() and 0 <= r < RANGE_EXTENT_M
                                 and -75 <= a <= 75 and 0 <= d < 256)
                    item = label.to_dict()
                    item.update(scene_name=folder.name, scene_id=scene_id, record_name=record_name,
                                frame_in_scene=frame, global_frame=int(num_samples[0]),
                                object_id=f"{folder.name}:annotation:{int(label.source_csv_row)}",
                                actor_id=int(label.source_csv_row), object_type="vehicle", semantic_class="vehicle",
                                range_m=r, azimuth_deg=a, doppler_bin=d,
                                range_bin=r*512/RANGE_EXTENT_M,
                                x_m=r*np.cos(np.deg2rad(a)), y_m=r*np.sin(np.deg2rad(a)),
                                expected_visible=valid, annotation_evaluable=valid,
                                label_coordinate_note="radar visible-face annotation; Doppler bin, not m/s")
                    label_rows.append(item)
        if not cube_rows:
            raise ValueError("No complete, synchronized, annotated RADIal frames found")
        self.index = pd.DataFrame(cube_rows)
        self.labels = pd.DataFrame(label_rows)
        self.audit = pd.DataFrame(audit_rows)
        self.sync_audit = pd.DataFrame(alternative_rows)
        if self.index.record_name.duplicated().any() or self.labels.object_id.duplicated().any():
            raise ValueError("Duplicate frame/annotation identity")
        self._signature = self._get_signature()
        self._splits = record_disjoint_split(self.index)
        if not self.sync_audit.same_frame.all():
            warnings.warn("RADIal synchronization choices change some labelled frame identities. "
                          f"This snapshot uses {self.tolerance_us} us; the alternative mapping is audit-only. "
                          "See radial_synchronization_audit.csv; this is informational, not a training blocker.")

    def _get_signature(self):
        return tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(self._tracked))

    def assert_unchanged(self):
        if self._get_signature() != self._signature:
            raise RuntimeError("RADIal source files changed since startup. Restart with a fresh run directory.")

    def split_capacities(self):
        return {name: len(rows) for name, rows in self._splits.items()}

    def load_cube(self, row):
        chips = {}
        for ch in range(4):
            with portable_path(row[f"chip{ch}_path"]).open("rb") as handle:
                handle.seek(int(row[f"chip{ch}_offset"]))
                adc = np.fromfile(handle, dtype="<i2", count=CHIP_BYTES//2)
            if adc.size != CHIP_BYTES//2:
                raise IOError(f"Short read for {row['record_name']}, chip {ch}")
            chips[ch] = adc
        # Exact official RPL ordering; every int16 value is exactly representable.
        cube = decode_adc_chips(chips)
        if cube.shape != SHAPE or not np.isfinite(cube).all():
            raise ValueError("Invalid decoded ADC frame")
        return np.ascontiguousarray(cube, dtype=np.complex64)

    @staticmethod
    def labels_for_cube(labels, row, visible_only=True):
        selected = labels.loc[(labels.scene_name == row["scene_name"]) &
                              (labels.frame_in_scene == int(row["frame_in_scene"]))].copy()
        if visible_only:
            selected = selected.loc[selected.annotation_evaluable]
        return selected.reset_index(drop=True)

    def report(self):
        return dict(dataset_root=str(self.root), reference_root=str(self.reference_root),
                    shape_rx_chirp_fast=list(SHAPE), source_dtype="int16 I + int16 Q",
                    processing_dtype="complex64 working array; no requantization at load",
                    source_bpcs=32, source_bytes_per_cube=4*CHIP_BYTES,
                    annotated_complete_frames=len(self.index), stored_object_annotations=len(self.labels),
                    annotation_counts=self.labels.Annotation.value_counts().to_dict(),
                    frame_annotation_counts=self.index.annotation_types.value_counts().to_dict(),
                    label_policy="all supplied annotation qualities retained for exploratory evaluation; not all strong GT",
                    excluded_invalid_annotations=int((~self.labels.annotation_evaluable).sum()),
                    excluded_frames=int((~self.audit.valid).sum()),
                    truncated_final_events=self.event_audit,
                    allow_truncated_final_event=self.allow_truncated_final_event,
                    records=self.index.groupby("scene_name").size().to_dict(),
                    split_capacities=self.split_capacities(),
                    synchronization_tolerance_us=self.tolerance_us,
                    synchronization_reference=f"label producer generate_database_2_updated.py: tolerance=20000; selected {self.tolerance_us}",
                    alternative_sync_differs_frames=int((~self.sync_audit.same_frame).sum()),
                    annotation_provenance="supplied labels_CVPR.csv; synchronization protocol must match its producer",
                    doppler_units="unshifted cyclic bins 0..255; radar_D_mps is not interpreted as m/s",
                    labels="radar-assisted vehicle visible-face annotations; not independent geometric object centres",
                    unmatched="peaks outside annotated vehicle gates; not confirmed false alarms",
                    array_processing="official RPL DC removal, Hamming, unshifted FFT, DDMA, calibrated azimuth",
                    rpl_sha256=hashlib.sha256(self.rpl_path.read_bytes()).hexdigest(),
                    labels_sha256=hashlib.sha256(self.label_path.read_bytes()).hexdigest(),
                    calibration_sha256=hashlib.sha256(self.calibration_path.read_bytes()).hexdigest())


def record_disjoint_split(index):
    records = sorted(index.scene_name.unique())
    if len(records) < 3:
        raise ValueError("At least three recordings required for disjoint train/validation/test")
    nt = min(max(1, math.ceil(.6*len(records))), len(records)-2)
    nv = min(max(1, math.floor(.2*len(records))), len(records)-nt-1)
    groups = dict(train=records[:nt], validation=records[nt:nt+nv], test=records[nt+nv:])
    split = {name: index.loc[index.scene_name.isin(names)].reset_index(drop=True)
             for name,names in groups.items()}
    if sum(map(len, split.values())) != len(index):
        raise AssertionError("Record-disjoint partition lost frames")
    return split


def load_reference_classes(path):
    """Use the supplied classes, without forcing optional CuPy/MKL dependencies.

    Only the two named class definitions are loaded from the user-supplied source.
    SciPy FFT implements the same unnormalised FFT API. No edits to RADIal-main.
    """
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n,ast.ClassDef)
             and n.name in {"RadarSignalProcessing","CA_CFAR"}]
    if len(nodes) != 2:
        raise ValueError("Unsupported RPL source: expected RadarSignalProcessing and CA_CFAR")
    namespace = dict(np=np, math=math, signal=signal, mkl_fft=fft)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),"exec"),namespace)
    return namespace["RadarSignalProcessing"], namespace["CA_CFAR"]


class RadialPostProcessor:
    """Reference calibrated RPL processing, continuous RA, explicit linear power.

    RA is the official summed magnitude squared before 10log10. RD similarly
    squares the official sum of RX magnitudes. The square matches the reference
    CA_CFAR's magnitude-to-power convention. No per-map normalization/GT fitting.
    Range chunks only reduce temporary memory; all 256 Doppler hypotheses remain.
    """
    def __init__(self, reference_root, range_chunk=16):
        root = portable_path(reference_root)
        cls, cfar_cls = load_reference_classes(root / "SignalProcessing/rpl.py")
        self.reference = cls(str(root / "SignalProcessing/CalibrationTable.npy"), method="RA",device="cpu",lib="CuPy")
        self.angle_axis_deg = np.asarray(self.reference.AoA_mat["Azimuth_table"],float)
        self.range_chunk = int(range_chunk)
        if self.range_chunk <= 0:
            raise ValueError("RADIAL_RA_RANGE_CHUNK must be positive")
        if self.reference.CalibMat.shape != (751,192):
            raise ValueError("Expected official (751 azimuth, 192 virtual antenna) calibration")
        table = self.reference.AoA_mat["Signal"]
        self.pc_calibration = np.rollaxis(table,2,1).reshape(table.shape[0]*table.shape[2],table.shape[1])
        self.pc_cfar = cfar_cls((9,9,3,3),2,(512,16))

    def point_cloud(self, spectrum, amplitude):
        """Official PC equations, chunked only at the large calibration product."""
        ref = self.reference
        reduced = amplitude.reshape(512,16,16).sum(axis=1)
        rbin, reduced_bin = np.where(self.pc_cfar(reduced))
        detection = np.zeros((512,256),bool)
        if not len(rbin):
            return np.empty((0,6)),np.empty((0,4)),detection
        dbin = ref._RadarSignalProcessing__find_TX0_position(amplitude,rbin,reduced_bin)
        detection[rbin,dbin] = True
        azimuth, elevation = [], []
        for start in range(0,len(rbin),256):
            rb,db = rbin[start:start+256],dbin[start:start+256]
            sequence = (db[:,None]+ref.dividend_constant_arr[None,:]) % 256
            sequence = np.concatenate([sequence[:,:1],sequence[:,5:]],axis=1).astype(int)
            mimo = spectrum[rb[:,None],sequence,:].reshape(len(rb),192)*ref.window
            response = np.abs(self.pc_calibration @ mimo.T)
            response[np.isnan(response)] = 0
            az,el = np.unravel_index(response.argmax(axis=0),(751,11))
            azimuth.extend(ref.AoA_mat["Azimuth_table"][az])
            elevation.extend(ref.AoA_mat["Elevation_table"][el])
        a,e = np.deg2rad(azimuth),np.deg2rad(elevation)
        r = rbin*RANGE_EXTENT_M/512
        polar = np.column_stack([r,dbin,a,e])
        cartesian = np.column_stack([r*np.cos(e)*np.cos(a),r*np.cos(e)*np.sin(a),
                                     r*np.sin(e),dbin,rbin,dbin])
        return cartesian,polar,detection

    def compute_RD_RA(self, cube):
        cube = np.asarray(cube)
        if cube.shape != SHAPE:
            raise ValueError(f"Expected {SHAPE}, got {cube.shape}")
        # Official int16 + 1j*int16 construction promotes to complex128 before DC.
        adc = np.asarray(cube.transpose(2,1,0),dtype=np.complex128)
        adc = adc - np.mean(adc,axis=(0,1))
        ref = self.reference
        spectrum = fft.fft(fft.fft(adc*ref.range_fft_coef,512,axis=0)*ref.doppler_fft_coef,256,axis=1)
        rd_amplitude = np.sum(np.abs(spectrum),axis=2)
        ra_amplitude = np.empty((512,len(self.angle_axis_deg)),float)
        for start in range(0,512,self.range_chunk):
            ra_amplitude[start:start+self.range_chunk] = ref._RadarSignalProcessing__get_RA(spectrum[start:start+self.range_chunk])
        point_cloud,polar_points,detections = self.point_cloud(spectrum,rd_amplitude)
        tiny = np.finfo(float).tiny
        return dict(RD=20*np.log10(np.maximum(rd_amplitude,tiny)),
                    RA=20*np.log10(np.maximum(ra_amplitude,tiny)),
                    ranges=np.arange(512)*RANGE_EXTENT_M/512,
                    vel=np.arange(256,dtype=float), angle_axis_deg=self.angle_axis_deg.copy(),
                    point_cloud=point_cloud, point_list=point_cloud.copy(), detections=detections,
                    reference_polar_point_cloud=polar_points,
                    doppler_units="cyclic_bin", power_convention="official amplitude squared")


class CyclicLabelBins(list):
    def __init__(self, values, period=None):
        super().__init__(values)
        self.period = period


def label_bins(labels, ranges, second_axis, map_type):
    pairs, indices = [], []
    rr, ss = np.asarray(ranges,float), np.asarray(second_axis,float)
    rd = str(map_type).upper() == "RD"
    if not len(rr) or not len(ss):
        return CyclicLabelBins([],256 if rd else None), [], len(labels)
    for i,(_,lab) in enumerate(labels.iterrows()):
        r = float(lab.range_m)
        second = float(lab.doppler_bin if rd else lab.azimuth_deg)
        if np.isfinite([r,second]).all() and rr.min() <= r <= rr.max() and ss.min() <= second <= ss.max():
            pairs.append((int(np.argmin(abs(rr-r))),int(np.argmin(abs(ss-second)))))
            indices.append(i)
    return CyclicLabelBins(pairs,len(ss) if rd else None), indices, len(labels)-len(pairs)


def associate_labels_and_peaks(bins, peak_mask, gate):
    peaks = np.argwhere(np.asarray(peak_mask,bool))
    kept, matched = np.zeros(len(bins),bool), np.zeros(len(peaks),bool)
    gr,gc = max(int(gate[0]),1),max(int(gate[1]),1)
    for i,(r,c) in enumerate(bins):
        dr = np.abs(peaks[:,0]-r)
        dc = np.abs(peaks[:,1]-c)
        if getattr(bins,"period",None) is not None:
            dc = np.minimum(dc,bins.period-dc)
        inside = (dr<=gr)&(dc<=gc)
        kept[i] = bool(inside.any())
        matched |= inside
    return kept,matched,peaks


def fft_indices(range_m, doppler_bin, shape=SHAPE):
    # Official RPL does not fftshift either axis.
    fast = int(np.rint(float(range_m)*shape[2]/RANGE_EXTENT_M)) % shape[2]
    slow = int(np.rint(float(doppler_bin))) % shape[1]
    return slow,fast
