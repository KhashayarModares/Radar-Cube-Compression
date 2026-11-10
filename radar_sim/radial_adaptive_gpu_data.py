"""Metadata/label boundary for the separate adaptive RADIal experiment.

The existing RADIal parser, 40-ms synchronization, chip decoder and label
coordinates are reused unchanged. This module never reads ADC bodies during
prepare/split/identity. Recording-disjoint percentages concern usable annotated
frames, not object counts. Exact proportions are attempted, never fabricated by
dropping frames, repeating frames or sharing a recording between populations.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from .radial_compression_dataset import (
    CHIP_BYTES, RANGE_EXTENT_M, SHAPE, RadialDataset,
    parse_events, portable_path, synchronized_frames,
)

SEED = 20261022
ROLES = ("train", "validation", "test")
# labels_CVPR.csv 'index' is the row of SyncReader(..., tolerance=20000) in the official
# generator (scripts/generate_RADIal_from_raw_sequences/generate_database_2_updated.py).
# On real frames, 20 ms gives CFAR label recall 0.71 at +-2 bins vs 0.25 for 40 ms.
TOLERANCE_US = 20000
LABEL_FIELDS = ("radar_R_m", "radar_A_deg", "radar_D_mps", "laser_X_m", "laser_Y_m")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _annotation_reason(row) -> str:
    r, a, d = float(row.radar_R_m), float(row.radar_A_deg), float(row.radar_D_mps)
    if not np.isfinite([r, a, d]).all():
        return "nonfinite_radar_annotation"
    reasons = []
    if not 0 <= r < RANGE_EXTENT_M:
        reasons.append("range_outside_official_extent")
    if not -75 <= a <= 75:
        reasons.append("angle_outside_annotation_domain")
    if not 0 <= d < 256:
        reasons.append("cyclic_doppler_bin_outside_0_255")
    return ";".join(reasons)


class AdaptiveRadialDataset(RadialDataset):
    """Native dataset with explicit partition access at the ADC-read boundary."""

    def load_cube(self, row):
        assert_partition_access(self, pd.DataFrame([dict(row)]))
        return super().load_cube(row)


def prepare_dataset(root, reference) -> AdaptiveRadialDataset:
    """Audit all recording metadata and retain every usable complete labelled cube.

    Explicit -1 no-object frames remain valid examples. Invalid object labels
    are not converted into negatives: invalid-only frames remain in the complete
    inventory and exclusion audit but are not admitted to learning/evaluation.
    """
    root, reference = portable_path(root).resolve(), portable_path(reference).resolve()
    raw = pd.read_csv(root / "labels_CVPR.csv")
    required = {"dataset", "index", "numSample", *LABEL_FIELDS}
    if not required.issubset(raw.columns):
        raise ValueError(f"Missing RADIal label columns: {sorted(required-set(raw))}")
    present = sorted(p for p in root.glob("RECORD@*") if p.is_dir())
    if not present:
        raise ValueError("No RADIal RECORD@ recording directories found")
    local = raw[raw.dataset.isin([p.name for p in present])].copy()
    for name in ("index", "numSample", *LABEL_FIELDS):
        local[name] = pd.to_numeric(local[name], errors="raise")
    if (not np.isfinite(local[["index", "numSample"]].to_numpy()).all()
            or (local[["index", "numSample"]].to_numpy() % 1 != 0).any()):
        raise ValueError("RADIal frame identities must be finite integers; no implicit index coercion")
    sentinel = local[list(LABEL_FIELDS)].eq(-1).all(axis=1)
    reasons = local.apply(_annotation_reason, axis=1)
    valid = reasons.eq("") & ~sentinel
    if not valid.any():
        raise ValueError("Preservation study requires at least one evaluable object annotation globally; all-negative/invalid dataset is unsupported")

    # Do not alter the parser, synchronization function, ADC decoder or native
    # annotation geometry. The only tolerated log damage is the native explicit
    # opt-in for a malformed final EOF fragment, never a malformed internal row.
    try:
        dataset = AdaptiveRadialDataset(root, reference, tolerance_us=TOLERANCE_US,
                                       allow_truncated_final_event=True)
    except AttributeError as exc:
        # The unchanged historical constructor accesses object_id on an empty
        # annotation table. Avoid copying/reimplementing its label extraction.
        if "object_id" in str(exc):
            raise ValueError("No object annotations survive the complete-frame filter; all-negative/invalid complete population is unsupported") from exc
        raise
    dataset.complete_index = dataset.index.copy()
    dataset.annotation_audit = dataset.labels.loc[~dataset.labels.annotation_evaluable,
        ["record_name", "object_id", "source_csv_row"]].copy()
    if not dataset.annotation_audit.empty:
        by_id = dataset.labels.set_index("object_id")
        dataset.annotation_audit["reason"] = [
            _annotation_reason(by_id.loc[name]) for name in dataset.annotation_audit.object_id]
    else:
        dataset.annotation_audit["reason"] = pd.Series(dtype=str)

    local = local.assign(explicit_negative=sentinel, evaluable_object=valid)
    group = local.groupby(["dataset", "index"], sort=True)[["explicit_negative", "evaluable_object"]].any()
    unusable = set()
    explicit_negative = set()
    for (recording, frame), flags in group.iterrows():
        name = f"{recording}_frame_{int(frame):06d}"
        if bool(flags.explicit_negative) and bool(flags.evaluable_object):
            raise ValueError(f"Conflicting no-object sentinel and evaluable objects at {name}")
        if bool(flags.explicit_negative):
            explicit_negative.add(name)
        elif not bool(flags.evaluable_object):
            unusable.add(name)
    dataset.invalid_only_frames = dataset.complete_index.loc[
        dataset.complete_index.record_name.isin(unusable)].copy()
    dataset.invalid_only_frames["reason"] = "no_evaluable_object_and_no_explicit_negative_annotation"
    dataset.index = dataset.complete_index.loc[~dataset.complete_index.record_name.isin(unusable)].copy().reset_index(drop=True)
    if dataset.index.empty:
        raise ValueError("No usable complete labelled RADIal frames remain after annotation audit")
    # Native source_csv_row/object_id and synchronized frame indices survive
    # exclusions unchanged. episode_id is recording identity, never frame time.
    recording_ids = {p.name: i+1 for i, p in enumerate(present)}
    dataset.index["episode_id"] = dataset.index.scene_name.map(recording_ids).astype(int)
    dataset.complete_index["episode_id"] = dataset.complete_index.scene_name.map(recording_ids).astype(int)
    dataset.index["explicit_negative"] = dataset.index.record_name.isin(explicit_negative)

    record_audit = []
    for folder in present:
        events = dataset.events.get(folder.name)
        if events is None:
            # Unannotated recordings are audited but never invented negatives.
            events = parse_events(folder, allow_truncated_final_event=True, audit=dataset.event_audit)
        required_chips = {f"radar_ch{ch}" for ch in range(4)}
        missing = required_chips - set(events)
        if missing:
            raise ValueError(f"{folder.name}: missing radar chip streams {sorted(missing)}")
        sizes = {ch: events[f"radar_ch{ch}"]["filename"].stat().st_size for ch in range(4)}
        frames = synchronized_frames(events, TOLERANCE_US)
        complete = 0
        for match in frames:
            good = True
            for ch in range(4):
                entry = events[f"radar_ch{ch}"]["events"].iloc[match[f"radar_ch{ch}"]]
                offset, length = int(entry.offset), int(entry.datasize)
                good &= length == CHIP_BYTES and offset >= 0 and offset+length <= sizes[ch]
            complete += int(good)
        dataset._tracked.add(folder / f"{folder.name}_events_log.rec")
        dataset._tracked.update(item["filename"] for item in events.values())
        record_audit.append(dict(recording=folder.name, synchronized_frames=len(frames), complete_frames=complete,
            complete_label_indexed_frames=int(dataset.complete_index.scene_name.eq(folder.name).sum()),
            usable_label_indexed_frames=int(dataset.index.scene_name.eq(folder.name).sum()),
            explicit_negative_frames=int((dataset.index.scene_name.eq(folder.name) & dataset.index.explicit_negative).sum()),
            invalid_only_frames=int(dataset.invalid_only_frames.scene_name.eq(folder.name).sum()),
            available_auxiliary_sensors=";".join(sorted(set(events)-required_chips)),
            synchronization_tolerance_us=TOLERANCE_US))
    dataset.recording_audit = pd.DataFrame(record_audit)
    dataset._signature = dataset._get_signature()
    dataset._adaptive_partitions = None
    dataset._adaptive_test_records = set()
    dataset._adaptive_authorized_test_records = set()
    dataset.metadata_audit = dict(
        metadata_only=True, ADC_bodies_read=False, synchronization_tolerance_us=TOLERANCE_US,
        synchronization_function="unchanged radial_compression_dataset.synchronized_frames",
        frame_indices_renumbered=False,
        chip_sample_contract="CH3 n; CH0 n+1; CH2 n+2; CH1 n+3",
        doppler_contract="Unshifted cyclic bins 0..255; radar_D_mps is not converted to m/s",
        label_policy="All supplied annotation qualities retained; invalid geometry excluded with reasons",
        complete_label_indexed_frames=len(dataset.complete_index), usable_label_indexed_frames=len(dataset.index),
        invalid_only_frames_excluded=len(dataset.invalid_only_frames),
        invalid_object_annotations=len(dataset.annotation_audit),
        explicit_negative_frames=int(dataset.index.explicit_negative.sum()),
        truncated_final_events=dataset.event_audit,
        auxiliary_sensor_policy="All available recognized auxiliary sensors retained in native synchronization; absent CAN is not invented",
        recording_counts=record_audit,
        caveat="Metadata and byte availability do not prove physical annotation accuracy or ADC payload integrity")
    return dataset


def _subsets(weights: list[int], order: list[int], cap: int | None = None) -> dict[int, int]:
    """One-dimensional subset sum: one deterministic bit mask per reachable sum."""
    sums = {0: 0}
    for position in order:
        for total, mask in list(sums.items()):
            value = total + weights[position]
            if (cap is None or value <= cap) and value not in sums:
                sums[value] = mask | (1 << position)
    return sums


def _partition_counts(names: list[str], weights: list[int], seed: int) -> tuple[list[set[str]], dict]:
    if len(names) < 3 or any(w <= 0 for w in weights):
        raise ValueError("At least three nonempty recordings are required for disjoint TRAIN/validation/TEST")
    total = sum(weights)
    ideal = (3*total/5, total/5, total/5)
    rng = random.Random(int(seed))
    orders = [list(range(len(names))), sorted(range(len(names)), key=lambda i: weights[i])]
    for _ in range(30):
        order = list(range(len(names)))
        rng.shuffle(order)
        orders.append(order)
    best = None
    # Exact 60/20/20 percentages require an integer target. Finding no solution
    # with these reconstructions is not a proof that no exact partition exists.
    if total % 5 == 0:
        target = total // 5
        for order in orders:
            val = _subsets(weights, order, target).get(target)
            if val is None:
                continue
            remaining = [i for i in order if not val & (1 << i)]
            test = _subsets(weights, remaining, target).get(target)
            if test is not None and val | test != (1 << len(names))-1:
                best = ((0,0,0), val, test)
                break
    if best is None:
        # Bounded, reproducible nearest-target search: varied TRAIN-only-free
        # metadata orders, up to eight validation sums near its desired count.
        for order in orders:
            possible_val = _subsets(weights, order)
            candidates = sorted((s for s in possible_val if 0 < s < total), key=lambda s: (abs(s-ideal[1]), s))[:8]
            for val_size in candidates:
                val = possible_val[val_size]
                remaining = [i for i in order if not val & (1 << i)]
                if len(remaining) < 2:
                    continue
                possible_test = _subsets(weights, remaining)
                for test_size in sorted((s for s in possible_test if 0 < s < total-val_size),
                                        key=lambda s: (abs(s-ideal[2]), s))[:2]:
                    test = possible_test[test_size]
                    counts = (total-val_size-test_size, val_size, test_size)
                    errors = [abs(5*n-t) for n,t in zip(counts, (3*total, total, total))]
                    score = (sum(errors), max(errors), errors[0])
                    if best is None or score < best[0]:
                        best = (score, val, test)
    if best is None:
        raise ValueError("Could not construct three nonempty recording-disjoint populations")
    _, val, test = best
    groups = [set(), set(), set()]
    for i, name in enumerate(names):
        groups[1 if val & (1 << i) else 2 if test & (1 << i) else 0].add(name)
    counts = [sum(weights[i] for i,n in enumerate(names) if n in group) for group in groups]
    audit = dict(seed=int(seed), target_fractions=[.6,.2,.2], target_cube_counts=list(ideal),
        actual_cube_counts=counts, actual_fractions=[n/total for n in counts],
        exact_60_20_20=all(5*n == t for n,t in zip(counts, (3*total,total,total))),
        all_usable_frames_retained=True, recording_disjoint=True, data_repeated=False,
        partition_method="Seeded two-stage one-dimensional subset-sum reconstructions; bounded nearest-target fallback",
        optimality_claim=False, search_orders=len(orders),
        limitation="If exact proportions were not found, the deterministic nearest candidate is reported; this bounded search does not prove global optimality or infeasibility",
        recordings={role: sorted(group) for role,group in zip(ROLES,groups)})
    return groups, audit


def _tiny_rows(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    groups = [g.sort_values(["frame_in_scene", "record_name"]) for _,g in frame.groupby("scene_name", sort=True)]
    random.Random(seed).shuffle(groups)
    selected = []
    for offset in range(max(map(len,groups))):
        for group in groups:
            if offset < len(group):
                selected.append(group.iloc[[offset]])
                if len(selected) == min(2,len(frame)):
                    return pd.concat(selected, ignore_index=True)
    raise AssertionError("Tiny roster construction failed")


def split_populations(dataset, seed=SEED, profile="full") -> list[pd.DataFrame]:
    if profile not in ("full", "tiny"):
        raise ValueError("profile must be full or tiny")
    index = dataset.index
    if index.record_name.duplicated().any():
        raise ValueError("Duplicate complete-cube identities")
    counts = index.groupby("scene_name", sort=True).size()
    groups, audit = _partition_counts(list(counts.index), list(map(int,counts)), int(seed))
    frames = []
    for role,group in zip(ROLES,groups):
        frame = index.loc[index.scene_name.isin(group)].copy().sort_values(["scene_name", "frame_in_scene"]).reset_index(drop=True)
        frame["split_role"] = role
        if "episode_id" not in frame:
            names = {name: i+1 for i,name in enumerate(sorted(index.scene_name.unique()))}
            frame["episode_id"] = frame.scene_name.map(names).astype(int)
        frames.append(frame)
    if sum(map(len,frames)) != len(index) or len(set().union(*(set(f.record_name) for f in frames))) != len(index):
        raise AssertionError("Recording partition discarded or repeated frames")
    dataset._adaptive_partitions = frames
    dataset._adaptive_test_records = set(frames[2].record_name)
    dataset._adaptive_authorized_test_records = set()
    dataset.partition_audit = audit
    dataset._splits = dict(zip(ROLES,frames))
    if profile == "tiny":
        audit = dict(audit, profile="tiny", full_population_counts=list(map(len,frames)))
        frames = [_tiny_rows(frame, int(seed)+i) for i,frame in enumerate(frames)]
        audit["used_cube_counts"] = list(map(len,frames))
        dataset.tiny_partition_audit = audit
    return [frame.copy() for frame in frames]


def assert_partition_access(dataset, rows: pd.DataFrame, *, frozen=None) -> None:
    if getattr(dataset, "_adaptive_partitions", None) is None:
        raise RuntimeError("Establish recording-disjoint populations before loading ADC bodies or scoped labels")
    required = {"record_name", "scene_name", "frame_in_scene"}
    if not required.issubset(rows):
        raise ValueError(f"Scoped roster missing {sorted(required-set(rows))}")
    if rows.record_name.duplicated().any():
        raise ValueError("Scoped roster repeats a cube")
    known = dataset.index.set_index("record_name")
    for _,row in rows.iterrows():
        if row.record_name not in known.index:
            raise ValueError(f"Unknown or unusable annotated cube: {row.record_name}")
        native = known.loc[row.record_name]
        frame = float(row.frame_in_scene)
        if (not math_is_integer(frame) or str(row.scene_name) != str(native.scene_name)
                or int(frame) != int(native.frame_in_scene)):
            raise ValueError("Scoped record-name/frame identity mismatch")
        for name in ("global_frame", "episode_id", *(f"chip{ch}_{field}" for ch in range(4)
                                                     for field in ("offset", "sample", "ordinal"))):
            if name in rows and name in native and row[name] != native[name]:
                raise ValueError(f"Scoped cube metadata mismatch: {name}")
        for name in (f"chip{ch}_path" for ch in range(4)):
            if name in rows and name in native and portable_path(row[name]).resolve() != portable_path(native[name]).resolve():
                raise ValueError(f"Scoped cube stream path mismatch: {name}")
    test = set(rows.record_name) & dataset._adaptive_test_records
    if test and frozen is not None:
        path = Path(frozen)
        if not path.is_file():
            raise ValueError("TEST access requires an existing frozen-policy JSON receipt")
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt.get("test_used_for_selection") is not False or not receipt.get("checkpoint"):
            raise ValueError("Frozen receipt must identify a checkpoint selected without TEST")
        checkpoint = Path(receipt["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = path.parent/checkpoint
        if not checkpoint.is_file() or not receipt.get("checkpoint_sha256") or _hash(checkpoint) != receipt["checkpoint_sha256"]:
            raise ValueError("Frozen checkpoint is missing or its SHA256 differs")
        dataset._adaptive_authorized_test_records.update(test)
    if test - dataset._adaptive_authorized_test_records:
        raise PermissionError("TEST ADC/label access denied until a verified validation-frozen checkpoint is supplied")


def math_is_integer(value: float) -> bool:
    return bool(np.isfinite(value) and value == int(value))


def scoped_labels(dataset, rows: pd.DataFrame, *, frozen=None) -> pd.DataFrame:
    """Exact frame join, invalid-annotation filtering, no TEST data before freeze."""
    assert_partition_access(dataset, rows, frozen=frozen)
    labels = dataset.labels.loc[dataset.labels.record_name.isin(set(rows.record_name))].copy()
    labels = labels.loc[labels.annotation_evaluable].reset_index(drop=True)
    if labels.object_id.duplicated().any():
        raise ValueError("Duplicate scoped annotation identities")
    labels.attrs["scoped_loading"] = dict(record_names=sorted(set(rows.record_name)),
        label_coordinate_protocol="unchanged native RADIal coordinates; cyclic Doppler bins",
        labels_used_for_policy_observations=False, invalid_annotations_excluded=True,
        test_included=bool(set(rows.record_name)&dataset._adaptive_test_records))
    return labels


def dataset_identity(dataset) -> dict:
    """Hash metadata/calibration only; ADC/sensor bodies contribute stat entries."""
    dataset.assert_unchanged()
    metadata = {dataset.label_path, dataset.calibration_path, dataset.rpl_path}
    metadata.update(p for p in dataset._tracked if p.name.endswith("_events_log.rec"))
    if dataset.sync_reference_path.is_file():
        metadata.add(dataset.sync_reference_path)
    return dict(dataset_root=str(dataset.root), reference_root=str(dataset.reference_root),
        shape_rx_chirp_fast=list(SHAPE), source_bpcs=32., synchronization_tolerance_us=TOLERANCE_US,
        complete_label_indexed_frames=len(dataset.complete_index), usable_frames=len(dataset.index),
        metadata_sha256={str(p): _hash(p) for p in sorted(metadata)},
        file_stat_inventory=[dict(path=str(p), bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
                             for p in sorted(dataset._tracked)],
        ADC_body_hashing=False, ADC_bodies_read=False,
        annotation_audit=dataset.metadata_audit,
        partition_audit=getattr(dataset,"partition_audit",None))
