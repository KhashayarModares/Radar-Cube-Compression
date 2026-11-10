"""Frozen Arm E population reports for the explicit full V44 experiment.

Reporting reuses the V12.22 codec, detector, gates, metrics, baselines and figure
functions. It neither chooses a checkpoint nor updates a policy. TRAIN/DEV/TEST
permission is owned by the separate full-run driver, not relaxed here.
"""
from __future__ import annotations

import gc
from pathlib import Path
import time

import pandas as pd

from radar_sim import ppo_v12_22_codec as codec
from radar_sim import ppo_v12_22_run as cpu
from radar_sim.ppo_v12_22_storage import (
    actor_hash, digest, load_tree, read_json, receipt, save_tree,
    verify_receipt, write_csv, write_json,
)


def _report_identity(runtime, actor, norm, actions, cfar, split, status,
                     save_reconstructed):
    """Bind resumable cases to one policy, source, roster and save protocol."""
    return digest(dict(
        actor=actor_hash(actor), normalizer=norm.as_dict(), actions=actions,
        cfar=bool(cfar), split=split, arm='E', seed=int(status['seed']),
        selected_update=int(status['selected_update']),
        roster=runtime.rows.record_name.astype(str).tolist(),
        frozen_rdo=runtime.manifest['fixed_rdo_action'],
        source=runtime.manifest['source_hashes'],
        save_reconstructed=int(save_reconstructed),
    ))


def export_population(runtime, actor, norm, actions, cfar, job, split='test',
                      *, save_reconstructed=0):
    """Report a validation-selected, frozen E actor on one authorized roster.

    ``runtime.rows`` contains the runner's explicitly permitted population and
    ``runtime.root`` must be its separate validation or TEST evaluation folder.
    Existing small cached cases are reused only under an identical identity.
    No decoded cube tensors are cached. Optional decoded examples are ordinary
    reconstruction files, not codec payloads or evidence of transmitted size.
    """
    from radar_sim.ppo_v12_21_pipeline import report_directories
    from radar_sim.ppo_v12_21_run import plot_cfar
    from radar_sim.ppo_v12_21_reporting import (
        collect_detection_frame, export_detection_reports,
    )
    from radar_sim.ppo_v12_19_diagnostics import _label_rows
    from radar_sim.ppo_v12_22_reporting import plot_job_learning

    if split not in ('test', 'validation'):
        raise ValueError('Population split must be test or validation.')
    if isinstance(save_reconstructed, bool) or not isinstance(save_reconstructed, int):
        raise TypeError('save_reconstructed must be a nonnegative integer.')
    if save_reconstructed < 0:
        raise ValueError('save_reconstructed must be nonnegative.')
    root, job = Path(runtime.root), Path(job)
    rows = runtime.rows
    if rows.empty or rows.record_name.duplicated().any():
        raise ValueError('Evaluation requires a nonempty unique cube roster.')
    status = read_json(job/'job_status.json')
    if not status.get('completed') or status.get('arm') != 'E':
        raise ValueError('Frozen population reporting requires completed Arm E training.')
    if not cfar:
        raise ValueError('The unchanged Arm E policy requires CFAR observations.')
    policy_before = actor_hash(actor)
    identity = _report_identity(runtime, actor, norm, actions, cfar, split, status,
                                save_reconstructed)
    complete = root/'REPORT_COMPLETED.json'
    if complete.is_file():
        result = verify_receipt(complete)
        if result.get('identity') != identity:
            raise ValueError('Completed report refers to another frozen evaluation.')
        return result
    destination = root/'reports'/(
        split+'_'+time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
        +'_'+str(time.time_ns()%1000000000))
    destination.mkdir(parents=True)
    ex = runtime.ex
    sweeps, quality, audit, frames, perception, peaks, action_rows = (
        [], [], [], [], [], [], [])
    seen, visuals = set(), []
    with report_directories(ex, destination):
        for ci, (_, row) in enumerate(rows.iterrows()):
            cached = root/'cases'/f'{row.record_name}.npz'
            representative = str(row.scene_name) not in seen
            seen.add(str(row.scene_name))
            save_example = ci < save_reconstructed
            case = load_tree(cached) if cached.is_file() else None
            if case is not None and case['identity'] != identity:
                raise ValueError('Cached case belongs to a different frozen evaluation.')
            if case is None or representative or save_example:
                cube = runtime.load_cube(row)
                original = runtime.original(row, cube)
                labs = ex.labels_for_cube_v6_1000(runtime.labels, row)
                source = codec.prepare_source(ex, cube, actions)
                features = runtime.features(row, original, source) if cfar else None
                encoded = codec.encode(ex, cube, actor, norm, False, source, features)
                post = runtime.post(encoded['cube_hat'], row, 'frozen_'+split+'_report',
                                    arm='E', seed=int(status['seed']))
                baseline = getattr(runtime, 'baseline', None)
                baselines = {
                    method: (baseline(row, cube, original, method)
                             if baseline is not None
                             else cpu._baseline(runtime, row, cube, original, method))
                    for method in ('CalibratedRDO', 'FFT14')}
                methods = {
                    'Original': (cube, dict(realized_bits=32*cube.size,
                                           realized_bpcs=32.), original),
                    'PPO': (encoded['cube_hat'], encoded['meta'], post)}
                methods.update({name: (None, value['meta'], value['post'])
                                for name, value in baselines.items()})
                if representative:
                    ex.plot_v12_7_scene_comparisons([
                        dict(row=row.copy(), labs=labs.copy(), methods=methods)])
                if save_example:
                    visuals.append(dict(row=row.copy(), methods={'PPO': (
                        encoded['cube_hat'].copy(), encoded['meta'].copy(), None)}))
                if case is None:
                    case = dict(
                        identity=identity, sweeps=[], quality=[], audit=[], frames=[],
                        perception=dict(record_name=row.record_name,
                                        scene_name=row.scene_name,
                                        **runtime.measure(original, post, labs, encoded)),
                        peaks=_label_rows(ex, original, post, labs,
                                          str(row.record_name), str(row.scene_name)
                                          ).to_dict('records'),
                        actions=encoded['block_log'].assign(
                            record_name=row.record_name, scene_name=row.scene_name
                            ).to_dict('records'))
                    for method, (decoded, meta, current) in methods.items():
                        case['quality'].append(
                            baselines[method]['quality'] if method in baselines else
                            ex._cube_quality_row_v12_6(
                                row.record_name, row.scene_name, method,
                                cube, decoded, meta, original, current))
                    for kind in ('RD', 'RA'):
                        data0, ranges, second = ex._map_triplet_v12_6(original, kind)
                        gate = (ex.CFAR_LABEL_GATE_RD_V8 if kind == 'RD'
                                else ex.CFAR_LABEL_GATE_RA_V8)
                        factors = (ex.V12_7_RD_PRESERVATION_THRESHOLD_FACTORS
                                   if kind == 'RD'
                                   else ex.V12_7_BASE_CFAR_THRESHOLD_FACTORS)
                        cache = ex.precompute_original_cfar_count_sweep_v11_2(
                            data0, labs, ranges, second, kind, factors, gate)
                        for method, (_, meta, current) in methods.items():
                            data, _, _ = ex._map_triplet_v12_6(current, kind)
                            ex._append_label_audit_v12_6(
                                case['audit'], cache, data, labs, method,
                                row.record_name, row.scene_name)
                            counts = ex.evaluate_cfar_count_sweep_method_v11_2(cache, data)
                            for item in counts.itertuples():
                                case['sweeps'].append(dict(
                                    cube_idx=ci, record_name=row.record_name,
                                    scene_name=row.scene_name, map_type=kind,
                                    method=method, method_display=ex.V12_7_DISPLAY[method],
                                    threshold_factor=item.threshold_factor,
                                    cfar_threshold_db=item.cfar_threshold_db,
                                    eligible_original=int(item.num_labels_supported_original),
                                    preserved_original_supported=int(
                                        item.num_labels_preserved_from_original),
                                    missed_original_supported=int(
                                        item.num_labels_missed_from_original),
                                    unmatched_false_peaks=int(
                                        item.num_label_unmatched_peaks_method),
                                    num_cfar_peaks=int(item.num_cfar_peaks_method),
                                    realized_bpcs=meta['realized_bpcs']))
                            case['frames'].append(collect_detection_frame(
                                ex, current, labs, kind, row.record_name, method))
                    save_tree(cached, case)
                del cube, original, source, encoded, post, methods, baselines
                gc.collect()
            sweeps.extend(case['sweeps'])
            quality.extend(case['quality'])
            audit.extend(case['audit'])
            frames.extend(case['frames'])
            perception.append(case['perception'])
            peaks.extend(case['peaks'])
            action_rows.extend(case['actions'])
            print(f'V12.22 Arm E frozen {split.upper()} reports {ci+1}/{len(rows)}',
                  flush=True)
        sweep, qtable = pd.DataFrame(sweeps), pd.DataFrame(quality)
        for name, values in [
            ('cfar_sweep', sweep), ('compression_quality', qtable),
            ('object_label_audit', audit), ('PPO_components', perception),
            ('PPO_label_peak_noise', peaks), ('PPO_actions', action_rows),
        ]:
            write_csv(destination/'tables'/f'{split}_{name}_v12_22.csv', values)
        plot_cfar(ex, sweep, split)
        ex.plot_v12_7_compression_effectiveness(qtable, split)
        if visuals:
            ex.save_v12_7_reconstructed_ppo_cubes(
                visuals, max_cubes=save_reconstructed)
        export_detection_reports(ex, frames, destination/'detection_metrics', split)
        plot_job_learning(ex, job, destination/'figures')
    if actor_hash(actor) != policy_before:
        raise ValueError('Frozen actor changed during population reporting.')
    cpu._rename_report_files(destination)
    write_json(destination/'REPORT_PROTOCOL.json', dict(
        arm='E', seed=int(status['seed']), split=split, cubes=len(rows),
        test_evaluated=split == 'test', frozen_policy=True,
        selected_policy_hash=policy_before, checkpoint=status['selected_update'],
        selected_using='validation_only', inherited_figures_preserved=True,
        RDO_shared_and_frozen=True, FFT14_unchanged=True,
        saved_reconstructed_cubes=len(visuals),
        decoded_npy_npz_are_not_codec_payload=True,
        raw_cube_layout='Rx,chirp,fast-time'))
    receipt(complete, [path for path in sorted(destination.rglob('*')) if path.is_file()],
            directory=str(destination), split=split, identity=identity,
            test_evaluated=split == 'test')
    return read_json(complete)
