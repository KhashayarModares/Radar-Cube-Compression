"""Strict, read-only adapter for the audited V4.4 1,000-cube dataset.

V4.4 changes traffic episodes, not the V4.3 signal or physical-truth contract.
MATLAB stores [fast time, receive element, chirp]; Python uses [Rx, M, N].
"""

from pathlib import Path
import csv
import fnmatch
import json
import os
import re
import warnings

import numpy as np
import pandas as pd


DATASET_NAME = "automotive_fmcw_1000_cubes_v44"
DATASET_REVISION = "4.4.0-diverse-episodes-v43-signal-contract"
SIGNAL_REVISION = "4.3.0-phase-center-multiaoa-200cubes"
TRUTH_VERSION = "automotive_fmcw_truth_v43"
EXPECTED_SHAPE = (24, 96, 1670)
TEMPLATE_EPISODES = {1: 4, 2: 4, 3: 3, 5: 3, 6: 3, 7: 4, 8: 4}
VEHICLE_CLASSES = {1, 2, 3, 12, 13}
MANIFEST_COLUMNS = (
    "SceneID SceneName FrameIndex SimulationTimeS NumRadarTargets CubePath TruthPath "
    "MeasurementPath PeakPointCloudPath CFARPointCloudPath AssociationPointCloudPath "
    "CubeSizeMiB QuantizationNMSEdB ScenarioMinimumCenterDistanceM LabelValidationPassed "
    "MeasurementValidationPassed PeakAssociationFraction CFARAssociationFraction "
    "SceneTemplateID SceneTemplateName EpisodeIndex EpisodeSeed TrafficTimeScale"
).split()
PLAN_COLUMNS = (
    "SceneID SceneTemplateID EpisodeIndex EpisodeSeed TrafficTimeScale FramesPerEpisode "
    "FramePeriodS SceneTemplateName SceneName"
).split()
TRUTH_COLUMNS = (
    "ActorID ClassID ClassName LengthM WidthM HeightM YawRelativeEgoDeg X_EgoM Y_EgoM "
    "Z_EgoM X_RadarM Y_RadarM Z_RadarM VxRelative_EgoMps VyRelative_EgoMps "
    "VzRelative_EgoMps RangeFromRadarM AzimuthFromRadarDeg ElevationFromRadarDeg "
    "RadialSpeedMps MeanRCSdBsm"
).split()
MEASUREMENT_COLUMNS = (
    "SceneName FrameIndex ActorID ClassID ClassName GeometryRangeM GeometryRangeRateMps "
    "GeometryAzimuthDeg GeometryX_RadarM GeometryY_RadarM GeometryYawDeg RangeGateM "
    "RangeRateGateMps AzimuthGateDeg AssociationValid AssociationStatus ComponentID "
    "PeakID AoAComponentRank CFARDetected MeasuredRangeM MeasuredRangeRateMps "
    "MeasuredAzimuthDeg MeasuredX_RadarM MeasuredY_RadarM PeakPowerDBRel PeakLocalSNRDB "
    "AngleProminenceDB AoARelativePowerDB RangeErrorM RangeRateErrorMps AzimuthErrorDeg "
    "RangeRateSignPassed AzimuthSignPassed NearbyRDPeakExists NearbyAngleCompatiblePeakExists "
    "NearestPeakID NearestComponentID NearestPeakRangeErrorM NearestPeakRangeRateErrorMps "
    "NearestPeakAzimuthErrorDeg RDLocalPeakPassed RAPeakPassed PointCloudConsistencyPassed "
    "CFARCellPassed MeasurementContractPassed PeakSNRPassed AngleProminencePassed "
    "AoARelativePowerPassed PointCloudConsistencyErrorM"
).split()
POINT_CLOUD_COLUMNS = (
    "X_RadarM Y_RadarM RangeRateMps PowerDBRel ComponentID ParentPeakID CFARDetected "
    "AngleProminenceDB RelativeAnglePowerDB"
).split()
PATH_FAMILIES = {
    "CubePath": ("cubes", "*.mat"),
    "TruthPath": ("labels", "*_truth.csv"),
    "MeasurementPath": ("measurements", "*_measurements.csv"),
    "PeakPointCloudPath": ("pointclouds/qualified_peaks", "*_qualified_peak_pointcloud.csv"),
    "CFARPointCloudPath": ("pointclouds/cfar", "*_cfar_pointcloud.csv"),
    "AssociationPointCloudPath": ("pointclouds/associated", "*_associated_pointcloud.csv"),
}
# Completed generator outputs are immutable during a PPO run. Cache checked
# snapshots only in this process, invalidate when any controlling metadata
# changes, and return copies so callers cannot modify the cached snapshot.
_MANIFEST_CACHE = {}
_LABEL_CACHE = {}


def _cache_key(root):
    names = ("manifest.csv", "episode_plan.csv", "generation_complete.json", "audit_report.json",
             "radar_spec.json", "label_schema.json")
    stats = [(root / "metadata" / name).stat() for name in names]
    return str(root), tuple((stat.st_mtime_ns, stat.st_size) for stat in stats)


def _family_files(folder, pattern):
    """Inventory each directory once; avoid thousands of WSL path-stat calls."""
    files, pending = {}, [folder]
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                _require(not entry.is_symlink(), f"Symbolic link in dataset family: {entry.path}")
                if entry.is_dir(follow_symlinks=False):
                    pending.append(entry.path)
                elif fnmatch.fnmatchcase(entry.name, pattern) and entry.is_file(follow_symlinks=False):
                    files[str(Path(entry.path))] = entry
    return files


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(root, name):
    with (Path(root) / "metadata" / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def _csv(path, columns):
    data = pd.read_csv(path, sep=",")
    _require(list(data.columns) == columns, f"Unexpected CSV schema: {path}")
    return data


def _flags(series):
    text = series.astype(str).str.strip().str.lower()
    _require(text.isin({"0", "1", "false", "true"}).all(), "Invalid boolean flag")
    return text.isin({"1", "true"})


def _path(value):
    text = str(value).strip().strip('"').replace("\\", "/")
    drive = re.match(r"^([A-Za-z]):/(.*)$", text)
    if drive and os.name != "nt":
        return Path("/mnt") / drive.group(1).lower() / drive.group(2)
    return Path(text).expanduser()


def _file(root, value, folder, scene, parents, files):
    # Resolve within the selected dataset even if its original Windows paths
    # still exist elsewhere. This also supports relocated datasets and WSL.
    name = str(value).replace("\\", "/").rsplit("/", 1)[-1]
    key = (folder, str(scene))
    if key not in parents:
        parents[key] = (root / folder / str(scene)).resolve()
        _require(parents[key].is_relative_to(root), "Path escapes dataset root")
    candidate = parents[key] / name
    _require(bool(name) and name not in {".", ".."}, "Invalid dataset filename")
    if str(candidate) not in files:
        raise FileNotFoundError(f"Missing dataset file: {candidate}")
    return str(candidate)


def find_dataset_root(project_root, explicit=None):
    """Find V4.4 only; invalid explicit/nonempty environment overrides never fall back."""
    override = explicit if explicit is not None else os.environ.get("MATLAB_RADAR_DATASET_ROOT")
    if explicit is not None:
        _require(bool(str(explicit).strip()), "Explicit dataset root is empty")
    if override is not None and str(override).strip():
        root = _path(override).resolve()
        validated_manifest(root)
        return root
    project = _path(project_root).resolve()
    relative = Path("matlab_automotive_fmcw/matlab_fmcw_v44_1000_bundle/outputs") / DATASET_NAME
    candidates = [project / "src" / relative, project / relative,
                  project.parent / relative, project.parent / "src" / relative]
    errors = []
    for root in dict.fromkeys(candidates):
        if not root.is_dir():
            continue
        try:
            validated_manifest(root)
        except (ValueError, FileNotFoundError, KeyError, json.JSONDecodeError) as error:
            errors.append(f"{root}: {error}")
        else:
            return root.resolve()
    raise FileNotFoundError("No audited V4.4 1,000-cube dataset found. " + "; ".join(errors))


def validated_manifest(root):
    """Validate completion, provenance, schemas, paths and the signal contract."""
    root = _path(root).resolve()
    cache_key = _cache_key(root)
    if cache_key in _MANIFEST_CACHE:
        return _MANIFEST_CACHE[cache_key].copy(deep=True)
    manifest = _csv(root / "metadata/manifest.csv", MANIFEST_COLUMNS)
    plan = _csv(root / "metadata/episode_plan.csv", PLAN_COLUMNS)
    completion = _json(root, "generation_complete.json")
    audit = _json(root, "audit_report.json")
    spec = _json(root, "radar_spec.json")
    schema = _json(root, "label_schema.json")
    _require(completion.get("GenerationComplete") is True and audit.get("Passed") is True,
             "Dataset generation and saved-data audit must both pass")
    for record in (completion, spec):
        _require(record.get("DatasetRevision") == DATASET_REVISION
                 and record.get("ExpectedCubeCount") == 1000
                 and record.get("NumTrafficEpisodes") == 25
                 and record.get("NumSceneTemplates") == 7, "Wrong V4.4 identity/counts")
    _require(audit.get("NumCubes") == 1000 and audit.get("NumEpisodes") == 25
             and audit.get("NumSceneTemplates") == 7, "Incomplete saved-data audit")
    _require(spec.get("GeneratorRevision") == SIGNAL_REVISION
             and spec.get("LabelSchemaVersion") == TRUTH_VERSION
             and schema.get("Version") == TRUTH_VERSION
             and spec.get("CubeLayout") == "fast_time_x_receive_element_x_chirp"
             and (spec.get("NumReceiveElements"), spec.get("NumChirps"),
                  spec.get("NumFastTimeSamples")) == EXPECTED_SHAPE,
             "V4.3 signal/truth contract changed")
    _require(schema.get("RadialSpeedConvention") == "positive_receding_negative_approaching"
             and np.allclose(spec.get("PhysicalMountingLocationM"), [3.7, 0, .55], atol=1e-12)
             and np.allclose(spec.get("SignalModelTransceiverMountingLocationM"), 0, atol=1e-12)
             and np.allclose(spec.get("SignalModelTransceiverMountingAnglesDeg"), 0, atol=1e-12),
             "Radar coordinate convention changed")
    calibration = spec.get("ProcessingCalibration", {})
    _require(calibration.get("Passed") is True and spec.get("MultiAoASelfTest", {}).get("Passed") is True
             and calibration.get("RangeScale") == 1
             and abs(calibration.get("RangeRateScale", 0)) == 1
             and abs(calibration.get("AzimuthSign", 0)) == 1
             and all(calibration.get(key) == 0 for key in
                     ("RangeOffsetM", "RangeRateOffsetMps", "AzimuthOffsetDeg")),
             "Phase-center processing checks did not pass")
    _require(len(manifest) == 1000 and len(plan) == 25, "Expected 1,000 cubes and 25 episodes")
    _require(not manifest.duplicated(["SceneID", "FrameIndex"]).any()
             and plan.SceneID.is_unique and plan.SceneName.is_unique
             and not plan.duplicated(["SceneTemplateID", "EpisodeIndex"]).any(),
             "Duplicate frame/episode identity")
    _require(plan.groupby("SceneTemplateID").size().to_dict() == TEMPLATE_EPISODES
             and set(plan.SceneID) == set(range(1, 26)) and plan.EpisodeSeed.is_unique,
             "Unexpected template/episode plan")
    _require((plan.FramesPerEpisode == 40).all() and (plan.FramePeriodS == .25).all()
             and np.isfinite(plan.TrafficTimeScale).all()
             and plan.TrafficTimeScale.between(.5, 1).all(), "Invalid episode cadence/scales")
    _require(_flags(manifest.LabelValidationPassed).all()
             and _flags(manifest.MeasurementValidationPassed).all(), "A saved validation flag failed")
    _require(np.isfinite(manifest.select_dtypes(include="number").to_numpy()).all()
             and manifest.CubeSizeMiB.between(0, 16, inclusive="right").all(),
             "Nonfinite manifest metadata or oversized cube")
    provenance = ["SceneName", "SceneTemplateID", "SceneTemplateName", "EpisodeIndex",
                  "EpisodeSeed", "TrafficTimeScale"]
    _require(set(manifest.SceneID) == set(plan.SceneID), "Manifest contains unplanned episodes")
    for episode in plan.itertuples(index=False):
        rows = manifest[manifest.SceneID == episode.SceneID].sort_values("FrameIndex")
        _require(len(rows) == 40 and np.array_equal(rows.FrameIndex.to_numpy(), np.arange(1, 41))
                 and np.allclose(rows.SimulationTimeS, np.arange(40) * .25, rtol=0, atol=1e-7),
                 f"Incomplete episode frames/cadence: {episode.SceneName}")
        _require(all((rows[column] == getattr(episode, column)).all() for column in provenance),
                 f"Episode provenance mismatch: {episode.SceneName}")
    manifest = manifest.sort_values(["SceneID", "FrameIndex"]).reset_index(drop=True)
    parents, cube_files = {}, {}
    for column, (folder, pattern) in PATH_FAMILIES.items():
        files = _family_files(root / folder, pattern)
        _require(len(files) == 1000, f"Extra or missing files in {folder}")
        suffix = pattern.removeprefix("*")
        expected_names = [f"{scene}_frame_{int(frame):04d}{suffix}"
                          for scene, frame in zip(manifest.SceneName, manifest.FrameIndex)]
        actual_names = [str(path).replace("\\", "/").rsplit("/", 1)[-1] for path in manifest[column]]
        _require(actual_names == expected_names, f"{column} filenames differ from their episode/frame keys")
        manifest[column] = [_file(root, value, folder, scene, parents, files)
                            for value, scene in zip(manifest[column], manifest.SceneName)]
        _require(manifest[column].is_unique, f"Duplicate {column}")
        if column == "CubePath":
            cube_files = files
        if column != "CubePath":
            columns = (TRUTH_COLUMNS if column == "TruthPath" else MEASUREMENT_COLUMNS
                       if column == "MeasurementPath" else POINT_CLOUD_COLUMNS)
            for path in manifest[column]:
                with Path(path).open(encoding="utf-8", newline="") as handle:
                    _require(next(csv.reader(handle), None) == columns, f"Unexpected CSV schema: {path}")
    sizes = np.asarray([cube_files[path].stat(follow_symlinks=False).st_size / 1024**2
                        for path in manifest.CubePath])
    _require(np.allclose(sizes, manifest.CubeSizeMiB, rtol=0, atol=1e-6), "Cube byte sizes differ")
    manifest["global_frame"] = np.arange(1000)
    manifest["scene_id"] = [f"episode_{int(value):02d}" for value in manifest.SceneID]
    manifest["frame_in_scene"] = manifest.FrameIndex.astype(int) - 1
    manifest["record_name"] = [f"{name}_frame_{int(frame):04d}"
                               for name, frame in zip(manifest.SceneName, manifest.FrameIndex)]
    manifest["cube_path"] = manifest.CubePath
    manifest["truth_path"] = manifest.TruthPath
    if len(_MANIFEST_CACHE) >= 4:
        _MANIFEST_CACHE.clear()
    _MANIFEST_CACHE[cache_key] = manifest.copy(deep=True)
    return manifest


def load_compact_cube(path, expected_shape=EXPECTED_SHAPE, expected_metadata=None):
    """Decode compact classic MAT or MATLAB v7.3 HDF5 into complex64 [Rx,M,N]."""
    import h5py
    path = _path(path)
    scalar_fields = ("sceneID", "frameIndex", "simulationTime")
    episode_fields = ("SceneID", "SceneTemplateID", "EpisodeIndex", "EpisodeSeed", "TrafficTimeScale")
    spec_fields = ("ExpectedCubeCount", "NumTrafficEpisodes", "NumSceneTemplates", "NumReceiveElements",
                   "NumChirps", "NumFastTimeSamples")
    metadata = {}
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as handle:
            _require(all(key in handle for key in ("adcI", "adcQ", "adcScale")),
                     f"Missing compact I/Q fields: {path}")
            i, q, scale = (np.asarray(handle[key][()]) for key in ("adcI", "adcQ", "adcScale"))
            if expected_metadata is not None:
                def numeric_value(node):
                    value = np.asarray(node[()])
                    if h5py.check_dtype(ref=value.dtype) is not None:
                        _require(value.size == 1, "Invalid MATLAB scalar reference")
                        value = np.asarray(handle[value.item()][()])
                    return value
                _require(all(key in handle for key in (*scalar_fields, "episodeInfo", "radarSpec")),
                         "Missing embedded V4.4 cube provenance")
                metadata = {key: numeric_value(handle[key]) for key in scalar_fields}
                for record, fields in (("episodeInfo", episode_fields), ("radarSpec", spec_fields)):
                    group = handle[record]
                    if isinstance(group, h5py.Dataset):
                        group = handle[np.asarray(group[()]).item()]
                    _require(all(key in group for key in fields), f"Incomplete {record}")
                    metadata[record] = {key: numeric_value(group[key]) for key in fields}
        axes = (1, 0, 2)  # HDF5 exposes MATLAB's dimensions in reverse order.
    else:
        from scipy.io import loadmat
        from scipy.io.matlab import MatReadWarning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message='Duplicate variable name "None" in stream.*',
                                    category=MatReadWarning)
            names = ["adcI", "adcQ", "adcScale"]
            if expected_metadata is not None:
                names += [*scalar_fields, "episodeInfo", "radarSpec"]
            data = loadmat(path, variable_names=names, struct_as_record=False)
        _require(all(key in data for key in ("adcI", "adcQ", "adcScale")),
                 f"Missing compact I/Q fields: {path}")
        i, q, scale = (np.asarray(data[key]) for key in ("adcI", "adcQ", "adcScale"))
        if expected_metadata is not None:
            _require(all(key in data for key in (*scalar_fields, "episodeInfo", "radarSpec")),
                     "Missing embedded V4.4 cube provenance")
            metadata = {key: data[key] for key in scalar_fields}
            for record, fields in (("episodeInfo", episode_fields), ("radarSpec", spec_fields)):
                value = np.asarray(data[record]).item()
                _require(all(hasattr(value, key) for key in fields), f"Incomplete {record}")
                metadata[record] = {key: getattr(value, key) for key in fields}
        axes = (1, 2, 0)
    _require(i.ndim == 3 and q.shape == i.shape
             and all(value.dtype.kind == "i" and value.dtype.itemsize == 2 for value in (i, q)),
             f"Compact I/Q must be matching three-dimensional int16 arrays: {path}")
    _require(scale.size == 1 and scale.dtype.kind == "f" and scale.dtype.itemsize == 4
             and np.isfinite(scale).all() and float(scale.item()) > 0,
             f"adcScale must be a finite positive single scalar: {path}")
    stored = (i.astype(np.float32) + 1j * q.astype(np.float32)) * np.float32(scale.item())
    cube = np.transpose(stored, axes).astype(np.complex64, copy=False)
    _require(tuple(cube.shape) == tuple(expected_shape) and np.isfinite(cube).all(),
             f"Malformed decoded cube {path}: expected {expected_shape}, got {cube.shape}")
    if expected_metadata is not None:
        def scalar(record, key):
            value = np.asarray(record[key])
            _require(value.size == 1 and value.dtype.kind in "ifu" and np.isfinite(value).all(),
                     f"Invalid numeric provenance field: {key}")
            return float(value.item())
        expected = expected_metadata
        frame = expected["FrameIndex"] if "FrameIndex" in expected else int(expected["frame_in_scene"]) + 1
        scene = expected.get("SceneID", expected.get("episode_id"))
        time = expected.get("SimulationTimeS", expected.get("simulation_time_s"))
        _require(scalar(metadata, "sceneID") == scene and scalar(metadata, "frameIndex") == frame
                 and np.isclose(scalar(metadata, "simulationTime"), time, rtol=0, atol=1e-8),
                 f"Embedded cube scene/frame/time differs from index: {path}")
        for field, normalized in (("SceneID", "episode_id"), ("SceneTemplateID", "template_id"),
                                  ("EpisodeIndex", "episode_index"), ("EpisodeSeed", "episode_seed"),
                                  ("TrafficTimeScale", "traffic_time_scale")):
            _require(np.isclose(scalar(metadata["episodeInfo"], field),
                                expected.get(field, expected.get(normalized)), rtol=0, atol=1e-12),
                     f"Embedded cube episode provenance differs: {field}")
        spec = metadata["radarSpec"]
        _require([scalar(spec, key) for key in spec_fields] == [1000, 25, 7, *expected_shape],
                 "Embedded cube dataset counts/dimensions differ")
    return cube


def load_labels(root, cfg):
    """Normalize all physical truth, retaining independent original detector flags."""
    root = _path(root).resolve()
    manifest = validated_manifest(root)
    cache_key = (_cache_key(root), tuple(getattr(cfg, name) for name in
                                       ("num_rx", "M", "N", "fs", "S", "Tchirp", "lam", "c0")))
    if cache_key in _LABEL_CACHE:
        return _LABEL_CACHE[cache_key].copy(deep=True)
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
    audit = _json(root, "audit_report.json")
    _require(labels.label_key.is_unique and not labels.duplicated(["episode_id", "frame_in_scene", "actor_id"]).any(),
             "Duplicate normalized label identity")
    _require(len(labels) == audit["NumTruthRows"]
             and int(labels.class_id.isin(VEHICLE_CLASSES).sum()) == audit["NumVehicleTruthRows"]
             and int(labels.original_association_valid.eq(True).sum()) == audit["NumPeakAssociations"]
             and int((labels.original_association_valid.eq(True)
                      & labels.original_cfar_detected.eq(True)).sum()) == audit["NumCFARAssociations"],
             "Label/association counts differ from the saved-data audit")
    if len(_LABEL_CACHE) >= 4:
        _LABEL_CACHE.clear()
    _LABEL_CACHE[cache_key] = labels.copy(deep=True)
    return labels


def build_cube_index(root, labels):
    """Create the existing PPO index schema with explicit episode provenance."""
    m = validated_manifest(root)
    required = {"global_frame", "scene_id", "scene_name", "frame_in_scene", "episode_id",
                "source_truth_path", "actor_id", "label_key"}
    _require(required.issubset(labels.columns) and labels.label_key.is_unique
             and not labels.duplicated(["episode_id", "frame_in_scene", "actor_id"]).any()
             and set(labels.global_frame) == set(m.global_frame), "Malformed or duplicate label keys")
    expected = m.set_index("global_frame").reindex(labels.global_frame)
    for actual, original in (("scene_id", "scene_id"), ("scene_name", "SceneName"),
                             ("frame_in_scene", "frame_in_scene"), ("episode_id", "SceneID"),
                             ("source_truth_path", "TruthPath")):
        _require(np.array_equal(labels[actual].to_numpy(), expected[original].to_numpy()),
                 f"Label/index provenance differs: {actual}")
    counts = labels.groupby("global_frame").size().reindex(m.global_frame, fill_value=0).to_numpy()
    _require(np.array_equal(counts, m.NumRadarTargets), "Labels do not cover all indexed frames")
    return pd.DataFrame({
        "cube_path": m.cube_path, "truth_path": m.truth_path, "record_name": m.record_name,
        "scene_id": m.scene_id, "scene_name": m.SceneName, "frame_in_scene": m.frame_in_scene,
        "global_frame": m.global_frame, "simulation_time_s": m.SimulationTimeS,
        "num_radar_targets": counts, "label_validation_passed": True, "accepted": True,
        "template_id": m.SceneTemplateID, "template_name": m.SceneTemplateName,
        "episode_id": m.SceneID, "episode_index": m.EpisodeIndex, "episode_seed": m.EpisodeSeed,
        "traffic_time_scale": m.TrafficTimeScale,
    })


def episode_split(index, seed=42):
    """600/200/200 cubes in whole episodes, with all seven templates in train.

    Validation/test each cover five templates and their union covers all seven.
    These are held-out traffic episodes of seen layouts, not unseen-scene tests.
    """
    _require(len(index) == 1000 and index.record_name.is_unique
             and index.groupby("episode_id").size().eq(40).all()
             and index.episode_id.nunique() == 25, "Expected 25 complete 40-frame episodes")
    episodes = index[["episode_id", "template_id", "template_name", "episode_index",
                      "episode_seed", "traffic_time_scale"]].drop_duplicates()
    _require(len(episodes) == 25 and episodes.episode_id.is_unique
             and episodes.groupby("template_id").size().to_dict() == TEMPLATE_EPISODES,
             "Inconsistent split provenance")
    _require(all(np.array_equal(np.sort(group.frame_in_scene.to_numpy()), np.arange(40))
                 for _, group in index.groupby("episode_id")), "Incomplete or repeated episode frames")
    rng = np.random.default_rng(int(seed))
    assignments = {"train": [], "validation": [], "test": []}
    # Randomize which templates receive both held-out partitions; this avoids
    # assigning a fixed speed factor or a fixed scene type to a split.
    for count, allocation in ((3, [(1, 1, 1), (2, 1, 0), (2, 0, 1)]),
                              (4, [(2, 1, 1), (2, 1, 1), (3, 1, 0), (3, 0, 1)])):
        templates = sorted(key for key, value in TEMPLATE_EPISODES.items() if value == count)
        templates = rng.permutation(templates)
        for template, sizes in zip(templates, allocation):
            ids = rng.permutation(sorted(episodes.loc[episodes.template_id == template, "episode_id"]))
            start = 0
            for name, size in zip(assignments, sizes):
                assignments[name].extend(int(value) for value in ids[start:start + size])
                start += size
    split = {name: index[index.episode_id.isin(ids)].sort_values(["episode_id", "frame_in_scene"])
             .reset_index(drop=True) for name, ids in assignments.items()}
    _require([len(split[name]) for name in assignments] == [600, 200, 200]
             and split["train"].template_id.nunique() == 7
             and len(set().union(*(set(ids) for ids in assignments.values()))) == 25,
             "Invalid episode split coverage")
    _require(all(not set(assignments[a]) & set(assignments[b]) for a, b in
                 (("train", "validation"), ("train", "test"), ("validation", "test"))),
             "Episode split overlap")
    split["metadata"] = {
        "seed": int(seed), "method": "template_stratified_whole_episode",
        "generalization_scope": "held_out_traffic_episodes_of_seen_templates",
        "unseen_scene_generalization": False,
        "note": "Each 200-cube holdout contains five templates; 280 cubes are required to cover all seven.",
        "episode_ids": {name: sorted(ids) for name, ids in assignments.items()},
        "cube_counts": {name: len(split[name]) for name in assignments},
        "template_ids": {name: sorted(int(value) for value in split[name].template_id.unique())
                         for name in assignments},
        "episode_provenance": {name: episodes[episodes.episode_id.isin(ids)]
                               .sort_values("episode_id").to_dict("records")
                               for name, ids in assignments.items()},
    }
    return split
