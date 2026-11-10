"""Explicit 20-ms RADIal metadata boundary for the corrected LMS smoke study.

Historical runners remain separate. This module never changes a process-global
legacy tolerance and never reads ADC bodies during preparation or identity.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .radial_compression_dataset import CHIP_BYTES, SHAPE, parse_events, portable_path, synchronized_frames
from .radial_adaptive_gpu_data import (
    AdaptiveRadialDataset, LABEL_FIELDS, ROLES, SEED, _annotation_reason, _hash,
    assert_partition_access, scoped_labels, split_populations, math_is_integer,
)

TOLERANCE_US = 20000
SYNC_CONTRACT = 'RADIal-label-producer-20ms-v1'


def verify_pairing(dataset):
    """Verify every eligible frame against 20-ms chip metadata, not CSV order."""
    if dataset.tolerance_us != TOLERANCE_US:
        raise ValueError('The label producer requires a 20,000-us matching tolerance.')
    audits=[]
    for recording,rows in dataset.index.groupby('scene_name',sort=True):
        events=dataset.events[str(recording)]
        correct=synchronized_frames(events,TOLERANCE_US)
        historical=synchronized_frames(events,40000)
        for row in rows.itertuples():
            index=int(row.frame_in_scene)
            if not 0<=index<len(correct):
                raise AssertionError('Eligible label index is absent from the 20-ms table.')
            match=correct[index]
            for channel in range(4):
                entry=events[f'radar_ch{channel}']['events'].iloc[match[f'radar_ch{channel}']]
                if (int(getattr(row,f'chip{channel}_ordinal'))!=match[f'radar_ch{channel}']
                        or int(getattr(row,f'chip{channel}_offset'))!=int(entry.offset)
                        or int(getattr(row,f'chip{channel}_sample'))!=int(entry['sample'])):
                    raise AssertionError('ADC chip identity does not match the 20-ms synchronized row.')
            previous=historical[index] if index<len(historical) else None
            same=previous is not None and all(previous[f'radar_ch{channel}']==match[f'radar_ch{channel}'] for channel in range(4))
            master=events['radar_ch3']['events']
            correct_timestamp=int(master.iloc[match['radar_ch3']].timestamp)
            old_timestamp=int(master.iloc[previous['radar_ch3']].timestamp) if previous is not None else None
            audits.append(dict(record_name=str(row.record_name),scene_name=str(recording),
                frame_in_scene=index,synchronization_tolerance_us=TOLERANCE_US,
                correct_master_ordinal=match['radar_ch3'],all_four_chip_identities_verified=True,
                historical_40ms_index_available=previous is not None,same_frame_as_40ms=same,
                correct_timestamp_us=correct_timestamp,historical_40ms_timestamp_us=old_timestamp,
                correct_minus_40ms_seconds=(correct_timestamp-old_timestamp)/1e6 if old_timestamp is not None else None))
    dataset.sync20_pairing_audit=pd.DataFrame(audits)
    dataset.metadata_audit['label_synchronization_contract']=dict(
        version=SYNC_CONTRACT,tolerance_us=TOLERANCE_US,
        producer='generate_database_2_updated.py and generate_database_add_FP_updated.py: SyncReader(tolerance=20000)',
        CSV_index='Index in the filtered 20-ms radar-master synchronization table',
        all_four_chip_identities_verified=True,verified_eligible_frames=len(audits),
        different_from_historical_40ms=sum(not row['same_frame_as_40ms'] for row in audits),
        sensor_offsets_unchanged=True,ADC_bodies_read=False)

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
    verify_pairing(dataset)
    return dataset


def dataset_identity(dataset) -> dict:
    """Hash metadata/calibration only; ADC/sensor bodies contribute stat entries."""
    if dataset.tolerance_us != TOLERANCE_US:
        raise ValueError("Corrected identity requires an actual 20-ms dataset.")
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
