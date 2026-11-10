"""Read native V4.4 label bodies only for explicitly approved radar records.

The native dataset adapter validates the complete dataset's completion metadata,
file inventory, sizes and CSV headers. That metadata-only inventory includes
TEST-associated paths, but this helper never loads unselected truth/measurement
bodies or raw cubes. Permitted record names come from the existing historical
TRAIN and development-validation manifests; no TEST manifest is read.

The per-frame normalization loop below is intentionally copied verbatim from
``matlab_v44_dataset.load_labels`` and checked for AST identity before use.
Dataset-wide association totals are not rewritten to make a subset pass. Native
per-frame checks remain intact; only subset-specific completeness is checked at
the end. No native function, audit, cache or module global is monkey-patched.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path

import numpy as np
import pandas as pd

from radar_sim import matlab_v44_dataset as native
from radar_sim.matlab_v44_dataset import (
    EXPECTED_SHAPE, MEASUREMENT_COLUMNS, TRUTH_COLUMNS, VEHICLE_CLASSES,
    _csv, _flags, _json, _path, _require, validated_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_TABLES = PROJECT_ROOT / (
    'outputs/v12_21_v44_uncapped_perception_ppo_full_train300_val100_test100/tables'
)


def verify_native_label_loop_parity():
    """Check source-level per-frame parity without loading label bodies."""
    def loop_dump(function):
        tree = ast.parse(inspect.getsource(function))
        definition = tree.body[0]
        loops = [node for node in definition.body if isinstance(node, ast.For)
                 and isinstance(node.target, ast.Name) and node.target.id == 'meta']
        _require(len(loops) == 1, 'Expected exactly one native-style per-frame label loop')
        module = ast.Module(body=loops[0].body, type_ignores=[])
        return ast.dump(module, include_attributes=False)

    reference = loop_dump(native.load_labels)
    scoped = loop_dump(load_record_labels)
    passed = scoped == reference
    _require(passed, 'Scoped per-frame label normalization differs from native V4.4')
    return dict(passed=passed,
        native_source=str(Path(inspect.getsourcefile(native.load_labels)).resolve()),
        scoped_source=str(Path(__file__).resolve()),
        per_frame_AST_sha256=hashlib.sha256(reference.encode('utf-8')).hexdigest(),
        scope='Exact per-frame validation/geometry/binning/association AST; no label-body reads')


def _permitted_records():
    """Use split metadata only, not labels or the held-out TEST manifest."""
    frames = []
    columns = {'record_name', 'global_frame', 'episode_id', 'frame_in_scene'}
    for name in ('cube_index_train_v12_21.csv', 'cube_index_validation_v12_21.csv'):
        path = HISTORICAL_TABLES / name
        _require(path.is_file(), f'Approved historical split manifest missing: {path}')
        frame = pd.read_csv(path)
        _require(columns <= set(frame), f'Approved split manifest schema mismatch: {path}')
        frames.append(frame)
    allowed = pd.concat(frames, ignore_index=True)
    _require(allowed.record_name.is_unique, 'Historical TRAIN/development records overlap')
    return allowed.set_index('record_name')


def load_record_labels(root, cfg, record_names):
    """Normalize selected TRAIN or explicitly supplied development records.

    The caller owns stage authorization and supplies its preregistered record
    list. This API rejects unknown/TEST record names, duplicates and empty
    selections. It returns the native label columns and native manifest order.
    """
    _require(not isinstance(record_names, (str, bytes)), 'Provide a collection of record names')
    requested = list(record_names)
    _require(bool(requested) and all(isinstance(value, str) and value.strip() == value
                                    and bool(value) for value in requested),
             'Explicit nonempty record names are required')
    _require(len(requested) == len(set(requested)), 'Duplicate requested radar records')
    permitted = _permitted_records()
    _require(set(requested) <= set(permitted.index),
             'Requested labels include an unknown or held-out TEST record')
    parity = verify_native_label_loop_parity()
    root = _path(root).resolve()
    manifest = validated_manifest(root)
    manifest = manifest.loc[manifest.record_name.isin(requested)].copy()
    _require(len(manifest) == len(requested) and manifest.record_name.is_unique,
             'Selected records are absent or duplicated in the native dataset manifest')
    approved = permitted.loc[manifest.record_name]
    for actual, expected in (('global_frame', 'global_frame'), ('SceneID', 'episode_id'),
                              ('frame_in_scene', 'frame_in_scene')):
        _require(np.array_equal(manifest[actual].to_numpy(), approved[expected].to_numpy()),
                 f'Selected dataset/split identity mismatch: {actual}')
    _require((int(cfg.num_rx), int(cfg.M), int(cfg.N)) == EXPECTED_SHAPE, "Wrong radar dimensions")
    spec = _json(root, "radar_spec.json")
    for attr, key in (("fs", "SampleRateHz"), ("S", "SweepSlopeHzPerS"),
                      ("Tchirp", "SweepTimeS"), ("lam", "WavelengthM")):
        _require(np.isclose(float(getattr(cfg, attr)), spec[key], rtol=1e-10, atol=0), f"Wrong radar {attr}")
    _require(float(cfg.c0) == 299792458., "Wrong speed of light")
    range_axis = float(cfg.c0) * np.fft.fftshift(np.fft.fftfreq(cfg.N, d=1 / cfg.fs)) / (2 * cfg.S)
    velocity_axis = cfg.lam / 2 * np.fft.fftshift(np.fft.fftfreq(cfg.M, d=cfg.Tchirp))
    angle_axis = np.linspace(-60., 60., 241)
    frames = []
    for meta in manifest.itertuples(index=False):
        truth = _csv(meta.TruthPath, TRUTH_COLUMNS)
        m = _csv(meta.MeasurementPath, MEASUREMENT_COLUMNS)
        _require(len(truth) == meta.NumRadarTargets and truth.ActorID.is_unique,
                 f"Truth count/actor identity mismatch: {meta.TruthPath}")
        numeric = truth.drop(columns="ClassName").to_numpy(float)
        _require(np.isfinite(numeric).all(), f"Nonfinite truth: {meta.TruthPath}")
        _require((truth.ActorID > 0).all()
                 and np.array_equal(truth.ActorID, truth.ActorID.astype(int))
                 and np.array_equal(truth.ClassID, truth.ClassID.astype(int)), "Invalid physical actor/class IDs")
        xyz = truth[["X_RadarM", "Y_RadarM", "Z_RadarM"]].to_numpy(float)
        ego = truth[["X_EgoM", "Y_EgoM", "Z_EgoM"]].to_numpy(float)
        velocity = truth[["VxRelative_EgoMps", "VyRelative_EgoMps", "VzRelative_EgoMps"]].to_numpy(float)
        ranges = np.linalg.norm(xyz, axis=1)
        geometry = np.column_stack((ranges, np.rad2deg(np.arctan2(xyz[:, 1], xyz[:, 0])),
                                    np.rad2deg(np.arctan2(xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1]))),
                                    np.sum(velocity * xyz, axis=1) / ranges))
        _require(np.allclose(ego - [3.7, 0, .55], xyz, rtol=0, atol=1e-8)
                 and np.allclose(geometry, truth[["RangeFromRadarM", "AzimuthFromRadarDeg",
                                                  "ElevationFromRadarDeg", "RadialSpeedMps"]],
                                 rtol=0, atol=1e-8), f"Physical geometry mismatch: {meta.TruthPath}")
        _require(((ranges >= 1) & (ranges <= 150) & (xyz[:, 0] > 0)
                  & (np.abs(geometry[:, 1]) <= 60)
                  & (np.abs(geometry[:, 3]) < spec["UnambiguousRadialSpeedMps"])).all(),
                 f"Out-of-coverage truth: {meta.TruthPath}")
        vehicles = truth[truth.ClassID.isin(VEHICLE_CLASSES)].set_index("ActorID")
        _require(m.ActorID.is_unique and set(m.ActorID) == set(vehicles.index)
                 and (m.SceneName == meta.SceneName).all() and (m.FrameIndex == meta.FrameIndex).all(),
                 f"Measurement episode/frame/actor join failed: {meta.MeasurementPath}")
        joined = vehicles.reindex(m.ActorID)
        _require(np.allclose(m[["GeometryRangeM", "GeometryRangeRateMps", "GeometryAzimuthDeg"]],
                                joined[["RangeFromRadarM", "RadialSpeedMps", "AzimuthFromRadarDeg"]],
                                rtol=0, atol=1e-8)
                 and np.array_equal(m.ClassID, joined.ClassID)
                 and np.array_equal(m.ClassName, joined.ClassName), "Measurement truth mismatch")
        valid, cfar = _flags(m.AssociationValid), _flags(m.CFARDetected)
        _require(m.loc[valid, "ComponentID"].is_unique
                 and (m.loc[valid, ["ComponentID", "PeakID"]] > 0).all().all()
                 and m.loc[valid & cfar, "AssociationStatus"].eq("CFAR_PEAK_AOA_ASSOCIATED").all()
                 and m.loc[valid & ~cfar, "AssociationStatus"].eq("PEAK_AOA_ASSOCIATED_CFAR_MISS").all()
                 and m.loc[~valid, "AssociationStatus"].str.startswith("MISS_").all(),
                 "Invalid measurement component/status contract")
        for flag in ("RDLocalPeakPassed", "RAPeakPassed", "AoARelativePowerPassed", "PeakSNRPassed",
                     "AngleProminencePassed", "RangeRateSignPassed", "AzimuthSignPassed",
                     "PointCloudConsistencyPassed", "MeasurementContractPassed"):
            _require(_flags(m.loc[valid, flag]).all(), f"Associated measurement flag failed: {flag}")
        _require(not ((~valid) & _flags(m.NearbyAngleCompatiblePeakExists)
                    & m.AssociationStatus.ne("MISS_UNRESOLVED_AOA_COMPONENT")).any(),
                 "Available compatible component was left as an unexplained miss")
        for error, gate in (("RangeErrorM", "RangeGateM"), ("RangeRateErrorMps", "RangeRateGateMps"),
                            ("AzimuthErrorDeg", "AzimuthGateDeg")):
            _require((m.loc[valid, error] <= m.loc[valid, gate] + 1e-9).all(), "Measurement outside gate")
        normalized = pd.DataFrame({
            "global_frame": meta.global_frame, "scene_id": meta.scene_id, "scene_name": meta.SceneName,
            "frame_in_scene": meta.frame_in_scene, "simulation_time_s": meta.SimulationTimeS,
            "episode_id": meta.SceneID, "template_id": meta.SceneTemplateID,
            "template_name": meta.SceneTemplateName, "episode_index": meta.EpisodeIndex,
            "episode_seed": meta.EpisodeSeed, "traffic_time_scale": meta.TrafficTimeScale,
            "object_id": [f"{meta.scene_id}_actor_{int(actor)}" for actor in truth.ActorID],
            "label_key": [f"{meta.record_name}_actor_{int(actor)}" for actor in truth.ActorID],
            "actor_id": truth.ActorID.astype(int), "class_id": truth.ClassID.astype(int),
            "object_type": truth.ClassName.astype(str).str.lower(), "semantic_class": truth.ClassName,
            "x_m": xyz[:, 0], "y_m": xyz[:, 1], "z_m": xyz[:, 2],
            "range_m": geometry[:, 0], "azimuth_deg": geometry[:, 1], "elevation_deg": geometry[:, 2],
            "radial_velocity_mps": geometry[:, 3], "expected_visible": True,
            "source_truth_path": meta.TruthPath,
        })
        for name, axis, values in (("range_bin", range_axis, geometry[:, 0]),
                                   ("doppler_bin", velocity_axis, geometry[:, 3]),
                                   ("angle_bin", angle_axis, geometry[:, 1])):
            normalized[name] = np.abs(axis[None, :] - values[:, None]).argmin(axis=1)
        original = m[["ActorID", "AssociationStatus", "ComponentID", "PeakID"]].copy()
        original["AssociationValid"] = valid
        original["CFARDetected"] = cfar
        original = original.rename(columns={"ActorID": "actor_id", "AssociationValid": "original_association_valid",
            "CFARDetected": "original_cfar_detected", "AssociationStatus": "original_association_status",
            "ComponentID": "original_component_id", "PeakID": "original_peak_id"})
        normalized = normalized.merge(original, on="actor_id", how="left", validate="one_to_one")
        frames.append(normalized)
    labels = pd.concat(frames, ignore_index=True)
    _require(labels.label_key.is_unique and not labels.duplicated(["episode_id", "frame_in_scene", "actor_id"]).any(),
             "Duplicate normalized label identity")
    counts = labels.groupby('global_frame').size().reindex(manifest.global_frame, fill_value=0).to_numpy()
    _require(np.array_equal(counts, manifest.NumRadarTargets.to_numpy())
             and len(labels) == int(manifest.NumRadarTargets.sum())
             and set(labels.global_frame) == set(manifest.global_frame),
             'Selected label counts differ from selected native manifest records')
    labels.attrs['scoped_loading'] = dict(record_names=manifest.record_name.tolist(),
        records=len(manifest), normalized_rows=len(labels), native_per_frame_AST=parity,
        TEST_truth_and_measurement_bodies_read=False, TEST_raw_cubes_read=False,
        complete_dataset_metadata_and_CSV_headers_inventoried=True,
        datasetwide_audit_totals_modified=False)
    return labels
