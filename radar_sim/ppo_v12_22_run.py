"""V12.22 controlled 16/21-action study. No work is performed on import.

All six arms retain the V12.21 terminal objective. TEST evaluation is forbidden.
The existing sionna-multiscene-gpu kernel runs this NumPy/SciPy implementation
on CPU; its name does not claim CUDA acceleration. Full runs require opt-in.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
import uuid

import numpy as np
import pandas as pd

from radar_sim import ppo_v12_22_codec as codec
from radar_sim.ppo_v12_22_storage import (
    actor_hash, digest, load_tree, optimizer_state, read_json, receipt,
    restore_optimizer, save_tree, sha, verify_receipt, write_csv, write_json,
)

ROOT = Path(__file__).resolve().parents[1]
REVISION = '12.22.0-controlled-21-action-study'
HISTORY = ROOT / 'outputs/v12_21_v44_uncapped_perception_ppo_full_train300_val100_test100'
TRAIN_PANEL = ROOT / 'outputs/p26a_train_census_20261021T191732Z_246237/P26A_TRAIN16_ROSTER.csv'
DEV_PANEL = ROOT / 'outputs/p25a_generalization_20261021T155414Z/P25A_DEV32_POPULATION.csv'
RDO_REFERENCE = ROOT / 'outputs/v12_19_v44_preservation_balanced_ppo_full_train600_val200_test200/tables/task_rdo_action_summary_v12.csv'
SPLIT_SEED = 20260825
DEFAULT_SEEDS = (20260825, 20261022)
ARMS = {
    'A': dict(menu='original16', cfar=False, auxiliary='native'),
    'B': dict(menu='proposed21', cfar=False, auxiliary='native'),
    'C': dict(menu='proposed21', cfar=True, auxiliary='native'),
    'D': dict(menu='proposed21', cfar=True, auxiliary='margin'),
    'E': dict(menu='proposed21', cfar=True, auxiliary='compatible'),
    'F': dict(menu='proposed21', cfar=True, auxiliary='risk'),
}


def _definition(arm):
    if arm in ARMS:
        return ARMS[arm]
    if arm in ('D0','E0','F0'):
        return {**ARMS[arm[0]],'cfar':False}
    raise ValueError('Unknown arm.')


def _job_pairs(runtime):
    pairs=[(arm,int(seed)) for arm in runtime.manifest['arms'] for seed in runtime.manifest['seeds']]
    path=runtime.root/'INTERACTION_RESERVE.json'
    if path.is_file():
        values=read_json(path)
        pairs += [(values['control_arm'],int(seed)) for seed in runtime.manifest['seeds']]
    return pairs


def profile(name):
    if name == 'full':
        return dict(train=300, validation=100, passes=2, batch_cubes=5, rollouts=2,
                    epochs=5, updates=120, checkpoints=[0, 30, 60, 90, 120],
                    train_probes=16, dev_probes=32, normalizer_cubes=15,
                    calibration_cubes=16, diagnostic_suffixes=4, rate_sweep_cubes=8)
    if name == 'tiny':
        return dict(train=2, validation=2, passes=1, batch_cubes=2, rollouts=2,
                    epochs=1, updates=1, checkpoints=[0, 1], train_probes=2,
                    dev_probes=2, normalizer_cubes=2, calibration_cubes=2,
                    diagnostic_suffixes=1, rate_sweep_cubes=1)
    raise ValueError('Unknown profile.')


def selection_keys(frame):
    excess = (frame.new_false_detections_total-frame.original_unmatched_peaks_total).clip(lower=0)
    low = int(frame.low_threshold_labels_missed_total.sum())
    six = int(frame.true_labels_missed_total.sum())
    return dict(new=[low, six, float(excess.mean()), float(frame.realized_bpcs.mean())],
                historical=[low, float(excess.mean()), six, float(frame.cfar_curve_regret.mean())])


def canonical(record, row, **identity):
    result = dict(record)
    result.update(record_name=str(row.record_name), cube_id=str(row.record_name),
                  episode_id=int(row.episode_id), scene_name=str(row.scene_name),
                  low_miss=int(record['low_threshold_labels_missed_total']),
                  six_miss=int(record['true_labels_missed_total']),
                  unmatched_excess=max(float(record['new_false_detections_total'])-
                                       float(record['original_unmatched_peaks_total']), 0.),
                  unmatched=float(record['new_false_detections_total']),
                  nmse=float(record.get('decoded_cube_nmse', record['cube_nmse'])),
                  bpcs=float(record['realized_bpcs']),
                  reward=float(record['terminal_label_first_reward']), **identity)
    result['nmse_fails_005'] = result['nmse'] > .05
    result['nmse_exceeds_001'] = result['nmse'] > .01
    return result


def summarize(frame):
    nmse = frame.nmse.to_numpy(float)
    return dict(n_cubes=len(frame), low_miss=int(frame.low_miss.sum()),
        six_miss=int(frame.six_miss.sum()), unmatched_excess=float(frame.unmatched_excess.mean()),
        unmatched=float(frame.unmatched.mean()), nmse=float(nmse.mean()),
        nmse_median=float(np.median(nmse)), nmse_p95=float(np.quantile(nmse, .95)),
        nmse_max=float(nmse.max()), nmse_exceed_005=float(np.mean(nmse>.05)),
        nmse_exceed_001=float(np.mean(nmse>.01)), bpcs=float(frame.bpcs.mean()),
        reward=float(frame.reward.mean()),
        mean_RD_log_power_mse=float(frame.RD_log_power_mse.mean()),
        mean_RA_log_power_mse=float(frame.RA_log_power_mse.mean()))


def source_hashes():
    paths = list((ROOT/'radar_sim').glob('ppo_v12_22_*.py'))
    paths += [ROOT/'radar_sim'/name for name in (
        'ppo_v12_21_pipeline.py', 'ppo_v12_21_learning.py', 'ppo_v12_21_codec.py',
        'ppo_v12_21_objective.py', 'ppo_v12_21_reporting.py', 'ppo_v12_21_run.py',
        'ppo_v12_19_pipeline.py', 'ppo_v12_19_reward.py', 'ppo_v12_19_diagnostics.py','ppo_v12_19_rate_sweep.py',
        'ppo_v12_14_codec.py', 'processing.py', 'matlab_v44_dataset.py',
        'ppo_priority2_scoped_labels_v2.py', 'ppo_priority3_cfar_features.py',
        'ppo_v12_21_prefixes.py', 'ppo_prefix_rollouts.py',
        'ppo_advantage_diagnostics.py', 'ppo_conditional_action_learning.py')]
    # radar_sim is a namespace package: no __init__.py is required or present.
    paths += [ROOT/'radar_sim'/name for name in ('config.py',
        'ppo_v12_18_learning.py','ppo_v12_14_margin.py','ppo_v12_16_cfar.py')]
    return {str(path.relative_to(ROOT)): sha(path) for path in sorted(set(paths))}


def package_versions():
    return dict(python=platform.python_version(),**{name:importlib.metadata.version(name)
        for name in ('numpy','scipy','pandas','matplotlib','h5py','scikit-learn')})


def verify_sources(manifest):
    current = source_hashes()
    if current != manifest['source_hashes']:
        raise RuntimeError('Source files changed after preparation. Preserve this run; prepare a new directory.')


class Runtime:
    """One sequential experiment; immutable native physics/baseline primitives."""
    def __init__(self, root, *, verify=True):
        self.root = Path(root).resolve()
        self.manifest = read_json(self.root/'manifest.json')
        if self.manifest.get('revision') != REVISION or self.manifest.get('test_evaluation') is not False:
            raise ValueError('Wrong experiment revision or TEST protocol.')
        if verify:
            verify_sources(self.manifest)
            if self.manifest['package_versions'] != package_versions():
                raise ValueError('Python/scientific-package versions changed since preparation.')
            for name, value in self.manifest['input_hashes'].items():
                if sha(ROOT/name) != value:
                    raise ValueError('A frozen experiment input changed: ' + name)
            for name,value in self.manifest['dataset_metadata_hashes'].items():
                if sha(Path(self.manifest['dataset_root'])/'metadata'/name)!=value:
                    raise ValueError('Dataset metadata changed: '+name)
        os.environ['MATLAB_RADAR_DATASET_ROOT'] = self.manifest['dataset_root']
        os.environ['RDO_RL_SEED'] = str(SPLIT_SEED)
        from radar_sim.ppo_v12_21_pipeline import load_primitives
        self.ex = load_primitives(self.root/'primitive_reports')
        if tuple(map(tuple, self.ex.ACTION_LIBRARY_V11)) != codec.ORIGINAL_ACTIONS:
            raise ValueError('The independent original action menu changed.')
        if tuple(self.ex._nearest_training_thresholds_v11()) != (1.25, 2., 4., 8., 16., 32.):
            raise ValueError('The original training detector grid changed.')
        self.config = self.manifest['configuration']
        self.train_rows = pd.read_csv(self.root/'populations/TRAIN.csv')
        self.validation_rows = pd.read_csv(self.root/'populations/VALIDATION.csv')
        self.train_probe = pd.read_csv(self.root/'populations/TRAIN_PROBE.csv')
        self.dev_probe = pd.read_csv(self.root/'populations/DEV_PROBE.csv')
        permitted = set(self.manifest['approved_records'])
        for frame in (self.train_rows, self.validation_rows, self.train_probe, self.dev_probe):
            if not frame.record_name.is_unique or not set(frame.record_name) <= permitted:
                raise ValueError('Unapproved/duplicated records in a population.')
        for column,frames in (('cube_path',(self.train_rows,self.validation_rows)),
                              ('truth_path',(self.train_rows,self.validation_rows))):
            if any(not Path(value).is_file() for frame in frames for value in frame[column]):
                raise FileNotFoundError('A frozen TRAIN/VALIDATION input path no longer exists.')
        if set(self.train_rows.episode_id) & set(self.validation_rows.episode_id):
            raise ValueError('TRAIN/VALIDATION episode leakage.')
        from radar_sim.ppo_priority2_scoped_labels_v2 import load_record_labels
        names = self.train_rows.record_name.tolist()+self.validation_rows.record_name.tolist()
        self.labels = load_record_labels(self.manifest['dataset_root'], self.ex.RADAR_CFG, names)
        self.processors = {}

    def load_cube(self, row):
        if str(row.record_name) not in self.manifest['approved_records']:
            raise ValueError('TEST/unknown cube body access is prohibited.')
        return self.ex.load_cube_v6_1000(row)

    def post(self, decoded, row, kind, **metadata):
        processor = self.ex._postprocessor_for_cube_v11(decoded, self.processors)
        started = time.perf_counter()
        post = self.ex.postprocess_cube(decoded, processor)
        event = dict(kind=kind, record_name=str(row.record_name),
                     seconds=time.perf_counter()-started, **metadata)
        with (self.root/'physical_evaluations.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(event, allow_nan=False)+'\n')
            stream.flush()
        return post

    def original(self, row, cube):
        path = self.root/'cache/original'/f'{row.record_name}.npz'
        if path.is_file():
            return load_tree(path)
        post = self.post(cube, row, 'original_processing')
        # Only native postprocessing arrays/lists, not raw cubes or policy maps.
        save_tree(path, post)
        return post

    def features(self, row, original, source):
        path = self.root/'cache/features'/f'{row.record_name}.npz'
        if path.is_file():
            return load_tree(path)
        result = codec.feature_bundle(self.ex, original, source)
        save_tree(path, result)
        return result

    def measure(self, original, post, labs, encoded):
        from radar_sim.ppo_v12_21_pipeline import measure
        if 'distortion_contributions' not in encoded:
            encoded = {**encoded, 'distortion_contributions': np.asarray([encoded['meta']['cube_nmse']])}
        result = measure(self.ex, original, post, labs, encoded, 'perception_first')
        result['decoded_cube_nmse'] = encoded['meta'].get('decoded_cube_nmse', result['cube_nmse'])
        result['realized_bpcs'] = float(encoded['meta']['realized_bpcs'])
        return result

    def load_job(self, arm, seed, selected=True):
        return load_job(self, arm, seed, selected)


def _relocate(frame, inventory):
    frame = frame.copy()
    current = inventory.set_index('record_name').loc[frame.record_name]
    if not np.array_equal(frame.global_frame.to_numpy(), current.global_frame.to_numpy()):
        raise ValueError('Historical split and current dataset frame identities disagree.')
    frame['cube_path'] = current.CubePath.to_numpy()
    frame['truth_path'] = current.TruthPath.to_numpy()
    return frame


def prepare(out, profile_name, arms, seeds, *, resume=False):
    out = Path(out).resolve()
    if (out/'PREPARE_COMPLETED.json').is_file():
        if not resume:
            raise FileExistsError('Existing experiment; use --resume or a NEW output folder.')
        verify_receipt(out/'PREPARE_COMPLETED.json')
        runtime = Runtime(out)
        if runtime.manifest['profile'] != profile_name or runtime.manifest['arms'] != arms or runtime.manifest['seeds'] != seeds:
            raise ValueError('Cannot change arms/seeds/profile on resume.')
        return runtime
    if out.exists() and not (out/'manifest.json').is_file():
        raise FileExistsError('Refusing an unrelated existing output folder.')
    if out.exists() and not resume:
        raise FileExistsError('Partial preparation exists. Use --resume with identical settings.')
    out.mkdir(parents=True, exist_ok=True)
    cfg = profile(profile_name)
    from radar_sim import matlab_v44_dataset as v44
    data = v44.find_dataset_root(ROOT)
    inventory = v44.validated_manifest(data)  # metadata/inventory only, no TEST bodies
    paths = dict(train=HISTORY/'tables/cube_index_train_v12_21.csv',
                 validation=HISTORY/'tables/cube_index_validation_v12_21.csv',
                 train_panel=TRAIN_PANEL, dev_panel=DEV_PANEL, frozen_rdo=RDO_REFERENCE)
    input_hashes = {str(path.relative_to(ROOT)): sha(path) for path in paths.values()}
    fulltrain = _relocate(pd.read_csv(paths['train']), inventory)
    fullval = _relocate(pd.read_csv(paths['validation']), inventory)
    if len(fulltrain) != 300 or len(fullval) != 100:
        raise ValueError('Require the original 300/100 historical populations; no new split.')
    train_names = pd.read_csv(TRAIN_PANEL).record_name.tolist()
    dev_names = pd.read_csv(DEV_PANEL).record_name.tolist()
    if len(train_names) != 16 or len(dev_names) != 32 or not set(train_names) <= set(fulltrain.record_name) or not set(dev_names) <= set(fullval.record_name):
        raise ValueError('The diverse preregistered TRAIN16/DEV32 panels changed.')
    tprobe = fulltrain.set_index('record_name').loc[train_names].reset_index()
    dprobe = fullval.set_index('record_name').loc[dev_names].reset_index()
    train, val = (fulltrain, fullval) if profile_name == 'full' else (tprobe.iloc[:2], dprobe.iloc[:2])
    tprobe, dprobe = tprobe.iloc[:cfg['train_probes']], dprobe.iloc[:cfg['dev_probes']]
    frozen = pd.read_csv(RDO_REFERENCE)
    chosen = frozen.loc[frozen.selected_fixed_action.astype(str).str.lower().eq('true')]
    if len(chosen) != 1:
        raise ValueError('Frozen independent RDO must select exactly one action.')
    fixed = [float(chosen.iloc[0].rho), int(chosen.iloc[0].q)]
    if not np.allclose(fixed,codec.ORIGINAL_ACTIONS[int(chosen.iloc[0].action_index)]):
        raise ValueError('Frozen RDO index and physical original action tuple disagree.')
    manifest = dict(revision=REVISION, profile=profile_name, configuration=cfg, arms=arms,
        seeds=seeds, split_seed=SPLIT_SEED, dataset_root=str(data), input_hashes=input_hashes,
        source_hashes=source_hashes(),package_versions=package_versions(),
        dataset_metadata_hashes={name:sha(data/'metadata'/name) for name in (
            'manifest.csv','episode_plan.csv','generation_complete.json','audit_report.json','radar_spec.json','label_schema.json')},
        approved_records=train.record_name.tolist()+val.record_name.tolist(),
        arm_definitions=ARMS, proposed_actions=codec.PROPOSED_ACTIONS,
        original_actions=codec.ORIGINAL_ACTIONS, fixed_rdo_action=fixed,
        objective='UNCHANGED V12.21 PerceptionObjective', test_evaluation=False,
        test_body_and_annotation_reads=False, independent_baseline_recalibration=False,
        checkpoint_order=['lowest_miss_sum', 'six_miss_sum', 'mean_positive_unmatched_excess', 'mean_bpcs'],
        rate_cap=None, rate_reward_weight=0., nmse_reward_weight=0.,
        cpu_backend=True, no_imitation=True, no_refinement=True, preparation_did_not_train=True,
        provisional_nmse_failure=.05, preferred_nmse=.01,
        interaction_reserve='Not launched automatically; optional two original-state auxiliary runs need explicit CLI opt-in.',
        planned_base_reconstruction_evaluations=56856,
        diagnostics_additional_greedy_baselines_up_to=48,
        budget_caveat='Original processing, failed/repeated attempts and report encodes are logged additionally. Rate sweep and figure costs are reported, never hidden.')
    if (out/'manifest.json').is_file() and digest(read_json(out/'manifest.json')) != digest(manifest):
        raise ValueError('Partial preparation configuration or source changed.')
    write_json(out/'manifest.json', manifest)
    for name, frame in [('TRAIN',train),('VALIDATION',val),('TRAIN_PROBE',tprobe),('DEV_PROBE',dprobe)]:
        write_csv(out/'populations'/f'{name}.csv', frame)
    write_csv(out/'frozen_RDO_reference.csv', frozen)
    runtime = Runtime(out)
    prepare_models(runtime)
    # Full-cube physical checks precede any trained expanded-menu actor.
    # This gate checks completion/finite measurements, not superiority over RDO.
    from radar_sim.ppo_v12_22_diagnostics import physical_menu_checks
    physical=physical_menu_checks(runtime)
    if not physical.get('completed') or physical['rows']!=physical['expected_rows']:
        raise RuntimeError('Physical menu engineering checks are incomplete; training is not authorized.')
    write_json(out/'PHYSICAL_MENU_GATE.json',dict(completed=True,
        physical_rows=physical['rows'],test_evaluated=False,
        observed_zero_miss_guarantee=False,
        criterion='All preregistered full-cube physical probes completed; no performance-dependent pruning.',
        table_sha256=sha(out/'diagnostics/physical_menu/PHYSICAL_MENU_OUTCOMES.csv')))
    artifacts = [out/'manifest.json', out/'frozen_RDO_reference.csv', *sorted((out/'populations').glob('*.csv'))]
    artifacts.append(out/'PHYSICAL_MENU_GATE.json')
    artifacts += sorted((out/'preparation').rglob('*.npz'))+sorted((out/'preparation').rglob('*.json'))
    artifacts += sorted((out/'preparation').rglob('*.csv'))
    receipt(out/'PREPARE_COMPLETED.json', artifacts, stage='prepare', test_evaluated=False)
    return runtime


def _initial_actor(ex, actions, seed):
    actor = ex.RadarPPOPolicyV11(20+2*len(actions), len(actions), seed=int(seed))
    if len(actions) == 21:
        actor.params['W1'][-4:] = 0.
        actor.params['Wc1'][-4:] = 0.
    return actor


def prepare_models(runtime):
    """Fit TRAIN-only statistics and auxiliary coefficients, without updates."""
    from radar_sim.ppo_v12_22_learning import calibrate_auxiliaries, NoninformativeAuxiliaryCalibration
    from radar_sim.ppo_v12_21_codec import ForcedAction
    from radar_sim.ppo_v12_21_prefixes import choose_training_prefixes
    ex, cfg, root = runtime.ex, runtime.config, runtime.root
    panel = ex.select_split_rows_v11(runtime.train_rows, min(cfg['normalizer_cubes'], len(runtime.train_rows)))
    for seed in runtime.manifest['seeds']:
        destination = root/'preparation'/f'seed{seed}'
        for menu, actions in [('original16',codec.ORIGINAL_ACTIONS),('proposed21',codec.PROPOSED_ACTIONS)]:
            norm_path, init_path = destination/f'{menu}_normalizer.json', destination/f'{menu}_initial.npz'
            norm_receipt=destination/f'{menu}_NORMALIZER_COMPLETED.json'
            if norm_receipt.is_file():
                verify_receipt(norm_receipt)
                continue
            dim = 20+2*len(actions)
            identity = codec.Normalizer(np.zeros(dim),np.ones(dim))
            probe = ex.RadarPPOPolicyV11(dim,len(actions),seed=int(seed)+101)
            probe.params['Wa'].fill(0.); probe.params['ba'].fill(0.)
            raw, features, covered, audit = [], [], [], []
            for _, row in panel.iterrows():
                print(f'V12.22 TRAIN normalizer {menu} seed {seed}: {row.record_name}',flush=True)
                cube = runtime.load_cube(row); source = codec.prepare_source(ex,cube,actions)
                bundle = runtime.features(row,runtime.original(row,cube),source) if menu=='proposed21' else None
                encoded = codec.encode(ex,cube,probe,identity,True,source)
                raw.append(encoded['states_raw'])
                if bundle is not None:
                    features.append(bundle['values']); covered.append(bundle['covered'])
                audit.append(dict(record_name=row.record_name,uses_labels=False,full_library_uniform=True,
                                  feature_extraction_seconds=bundle['contract'].get('feature_extraction_seconds',0.) if bundle else 0.))
                del cube,source,encoded; gc.collect()
            norm = codec.fit_normalizer(np.concatenate(raw),
                np.concatenate(features) if features else None, np.concatenate(covered) if covered else None)
            write_json(norm_path,norm.as_dict())
            save_tree(init_path,_initial_actor(ex,actions,seed).state_dict())
            write_csv(destination/f'{menu}_normalizer_audit.csv',audit)
            receipt(norm_receipt,[norm_path,init_path,destination/f'{menu}_normalizer_audit.csv'],fit_split='TRAIN')
        # All expanded arms have the SAME initial model and base mapping.
        target = destination/'auxiliary_calibration.json'
        calibration_receipt=destination/'CALIBRATION_COMPLETED.json'
        if calibration_receipt.is_file():
            verify_receipt(calibration_receipt)
            continue
        if not any(ARMS[arm]['auxiliary']!='native' for arm in runtime.manifest['arms']):
            write_json(target,dict(coefficients=dict(native=0.,margin=.30,compatible=None,risk=None),
                informative=False,noninformative_auxiliaries=[],contexts=0,
                reason='No auxiliary arm requested; no calibration reconstructions performed.',
                source_partition='TRAIN',frozen_after_calibration=True))
            receipt(calibration_receipt,[target],auxiliary_requested=False)
            continue
        actor = _initial_actor(ex,codec.PROPOSED_ACTIONS,seed)
        norm_values = read_json(destination/'proposed21_normalizer.json')
        norm_values['cfar_enabled'] = True
        norm = codec.Normalizer.from_dict(norm_values)
        states, packets, outcomes = [], [], []
        for context, (_,row) in enumerate(runtime.train_probe.iloc[:cfg['calibration_cubes']].iterrows()):
            print(f'V12.22 initial TRAIN calibration seed {seed} context {context+1}/{cfg["calibration_cubes"]}: {row.record_name}',flush=True)
            cube = runtime.load_cube(row); original = runtime.original(row,cube)
            labs = ex.labels_for_cube_v6_1000(runtime.labels,row)
            source = codec.prepare_source(ex,cube,codec.PROPOSED_ACTIONS)
            bundle = runtime.features(row,original,source)
            rng = np.random.default_rng(np.random.SeedSequence([seed,1222,801,context]))
            tiles,_ = choose_training_prefixes(ex,original,labs,source,1,rng,row.record_name,row.scene_name,'original_supported')
            tile = tiles[0]
            primary = codec.encode(ex,cube,actor,norm,True,source,bundle)
            state = primary['states'][tile:tile+1]
            p = actor.forward(state)[0][0].astype(float); p /= p.sum()
            measurements=[]
            for action in range(len(codec.PROPOSED_ACTIONS)):
                facade = ForcedAction(actor,primary['actions'],tile,action,
                    np.random.SeedSequence([seed,1222,802,context]))
                branch = codec.encode(ex,cube,facade,norm,True,source,bundle)
                post = runtime.post(branch['cube_hat'],row,'auxiliary_calibration',seed=seed,action=action)
                result = runtime.measure(original,post,labs,branch)
                measurements.append(result)
                outcomes.append(dict(record_name=row.record_name,context=context,action=action,
                                     tile_index=tile,seed=seed,**result))
                del branch,post
                if (action+1)%7==0 or action+1==len(codec.PROPOSED_ACTIONS):
                    print(f'  calibration actions {action+1}/{len(codec.PROPOSED_ACTIONS)}',flush=True)
            states.append(state[0])
            packets.append(_packet(context,p,measurements))
            del cube,source,original,primary;gc.collect()
        state_array = np.asarray(states,np.float32)
        try:
            calibration = calibrate_auxiliaries(actor,state_array,packets,ex._network_gradients_v11)
        except NoninformativeAuxiliaryCalibration as exc:
            calibration = exc.report
        save_tree(destination/'calibration_contexts.npz',dict(states=state_array,packets=packets))
        write_csv(destination/'calibration_outcomes.csv',outcomes)
        write_json(target,calibration)
        receipt(calibration_receipt,[target,destination/'calibration_contexts.npz',destination/'calibration_outcomes.csv'],
                fit_split='TRAIN',auxiliary_requested=True)


def _packet(primary_row, probabilities, measurements):
    return dict(primary_row=int(primary_row),old_probabilities=np.asarray(probabilities,float),
        q_values_unscaled=np.asarray([r['terminal_label_first_reward'] for r in measurements]),
        lowest_misses=np.asarray([r['low_threshold_labels_missed_total'] for r in measurements]),
        six_misses=np.asarray([r['true_labels_missed_total'] for r in measurements]),
        unmatched=np.asarray([r['new_false_detections_total'] for r in measurements]),
        bpcs=np.asarray([r.get('realized_bpcs',r.get('_bpcs',0.)) for r in measurements]))


def _job_directory(root, arm, seed):
    _definition(arm)
    return Path(root)/'jobs'/f'{arm}_seed{int(seed)}'


def _job_settings(runtime, arm, seed):
    definition = _definition(arm)
    actions = codec.ORIGINAL_ACTIONS if definition['menu']=='original16' else codec.PROPOSED_ACTIONS
    folder = runtime.root/'preparation'/f'seed{seed}'
    norm_values = read_json(folder/f'{definition["menu"]}_normalizer.json')
    norm_values['cfar_enabled'] = definition['cfar']
    norm = codec.Normalizer.from_dict(norm_values)
    actor = _initial_actor(runtime.ex, actions, seed)
    actor.load_state_dict(load_tree(folder/f'{definition["menu"]}_initial.npz'))
    calibration = read_json(folder/'auxiliary_calibration.json')
    auxiliary = definition['auxiliary']
    coefficient = calibration['coefficients'][auxiliary]
    if auxiliary != 'native' and (auxiliary in calibration.get('noninformative_auxiliaries',[]) or coefficient is None):
        raise RuntimeError(f'Arm {arm} seed {seed}: noninformative {auxiliary} TRAIN calibration. '
                           'This arm is blocked, not a successful inactive auxiliary.')
    return actor,norm,actions,float(coefficient)


def load_job(runtime, arm, seed, selected=True):
    folder = _job_directory(runtime.root,arm,seed)
    actor,norm,actions,_ = _job_settings(runtime,arm,seed)
    if selected:
        status = read_json(folder/'job_status.json')
        if not status.get('completed'):
            raise ValueError('Selected-policy evaluation requires a complete training job.')
        verify_receipt(folder/'TRAIN_COMPLETED.json')
        saved = load_tree(folder/'models/selected.npz')
    else:
        receipt_path = sorted((folder/'updates').glob('update_*/COMPLETE.json'))[-1]
        verify_receipt(receipt_path)
        saved = load_tree(receipt_path.parent/'state.npz')
    actor.load_state_dict(saved['policy'])
    if 'actor_rng' in saved:
        actor.rng.bit_generator.state = saved['actor_rng']
    return actor,norm,actions,_definition(arm)['cfar'],folder


def _state(actor,optimizer,selection_rng,update):
    return dict(policy=actor.state_dict(),optimizer=optimizer_state(optimizer),
                actor_rng=copy.deepcopy(actor.rng.bit_generator.state),
                selection_rng=copy.deepcopy(selection_rng.bit_generator.state),update=int(update))


def _restore(actor,optimizer,selection_rng,saved):
    actor.load_state_dict(saved['policy'])
    restore_optimizer(optimizer,saved['optimizer'])
    actor.rng.bit_generator.state = saved['actor_rng']
    selection_rng.bit_generator.state = saved['selection_rng']


def _evaluation_count(root, arm=None, seed=None):
    path = Path(root)/'physical_evaluations.jsonl'
    if not path.exists():
        return 0
    count=0
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            event=json.loads(line)
            if (arm is None or event.get('arm')==arm) and (seed is None or event.get('seed')==int(seed)):
                count+=1
    return count


def _evaluate_checkpoint(runtime,arm,seed,actor,norm,actions,update):
    """Greedy fixed populations. Per-cube commits survive evaluation interruption."""
    folder = _job_directory(runtime.root,arm,seed)/'checkpoints'/f'update_{update:04d}'
    identity = dict(policy=actor_hash(actor),normalizer=digest(norm.as_dict()),arm=arm,seed=int(seed),update=update)
    if (folder/'COMPLETE.json').is_file():
        values = verify_receipt(folder/'COMPLETE.json')
        if values['identity'] != digest(identity):
            raise ValueError('Checkpoint policy differs from committed evaluation.')
        return load_tree(folder/'results.npz')
    ex, all_records, all_actions = runtime.ex, {}, {}
    for split, population in [('train_probe',runtime.train_probe),('validation',runtime.validation_rows)]:
        records, logs = [], []
        for _,row in population.iterrows():
            path = folder/split/f'{row.record_name}.npz'
            if path.is_file():
                cached = load_tree(path)
                if cached['identity'] != digest(identity):
                    raise ValueError('Partial checkpoint belongs to another policy.')
                record,log = cached['record'],cached['actions']
            else:
                cube = runtime.load_cube(row); original = runtime.original(row,cube)
                labs = ex.labels_for_cube_v6_1000(runtime.labels,row)
                source = codec.prepare_source(ex,cube,actions)
                features = runtime.features(row,original,source) if norm.cfar_enabled else None
                encoded = codec.encode(ex,cube,actor,norm,False,source,features)
                post = runtime.post(encoded['cube_hat'],row,'checkpoint',arm=arm,seed=int(seed),update=update,split=split)
                record = canonical(runtime.measure(original,post,labs,encoded),row,
                                   arm=arm,seed=int(seed),update=update,split=split)
                log = encoded['block_log'].assign(record_name=row.record_name).to_dict('records')
                save_tree(path,dict(identity=digest(identity),record=record,actions=log))
                del cube,source,post,encoded;gc.collect()
            records.append(record); logs.extend(log)
        all_records[split],all_actions[split] = records,logs
    frame = pd.DataFrame(all_records['validation'])
    result = dict(records=all_records,actions=all_actions,keys=selection_keys(frame),
                  summaries={name:summarize(pd.DataFrame(records)) for name,records in all_records.items()},
                  policy=actor.state_dict(),actor_rng=copy.deepcopy(actor.rng.bit_generator.state),
                  update=update,identity=identity)
    save_tree(folder/'results.npz',result)
    receipt(folder/'COMPLETE.json',[folder/'results.npz'],identity=digest(identity),test_evaluated=False)
    summary = result['summaries']['validation']
    print(f'V12.22 {arm} seed {seed} greedy VALIDATION update {update}: '
          f'LOW={summary["low_miss"]} SIX={summary["six_miss"]} '
          f'unmatched_excess={summary["unmatched_excess"]:.3f} bpcs={summary["bpcs"]:.3f}',flush=True)
    return result


def _collect_batch(runtime,arm,seed,actor,norm,actions,selection_rng,update,positions,pass_index):
    from radar_sim.ppo_v12_21_prefixes import choose_training_prefixes
    from radar_sim.ppo_v12_21_codec import ForcedAction
    from radar_sim.ppo_prefix_rollouts import assert_prefix_matches
    ex,cfg = runtime.ex,runtime.config
    trajectories,records,packets,branches,placement,clipping = [],[],[],[],[],[]
    for batch_position,position in enumerate(positions):
        row = runtime.train_rows.iloc[int(position)]
        cube = runtime.load_cube(row);original = runtime.original(row,cube)
        labs = ex.labels_for_cube_v6_1000(runtime.labels,row)
        source = codec.prepare_source(ex,cube,actions)
        features = runtime.features(row,original,source) if norm.cfar_enabled else None
        primary_for_branch = int(selection_rng.integers(cfg['rollouts']))
        if batch_position == 0:
            tiles,audit = choose_training_prefixes(ex,original,labs,source,1,selection_rng,
                row.record_name,row.scene_name,'original_supported')
            placement.extend(dict(update=update,record_name=row.record_name,**value) for value in audit)
        else:
            tiles=[]
        flat_offset = sum(len(item['actions']) for item in trajectories)
        for sample in range(cfg['rollouts']):
            encoded = codec.encode(ex,cube,actor,norm,True,source,features)
            post = runtime.post(encoded['cube_hat'],row,'primary_rollout',arm=arm,seed=int(seed),update=update)
            measured = runtime.measure(original,post,labs,encoded)
            encoded['rewards'][-1] = measured['terminal_label_first_reward']
            encoded.update(rollout_group=f'{pass_index}:{position}:{row.record_name}',
                           rollout_index=sample,record_name=str(row.record_name))
            record = canonical(measured,row,arm=arm,seed=int(seed),update=update,split='TRAIN',
                               rollout_index=sample,episode=(update-1)*cfg['batch_cubes']+batch_position+1,
                               pass_number=pass_index+1,unique_actions=int(np.unique(encoded['actions']).size),
                               **ex.action_diagnostics_v12_9(encoded))
            clipping.append(dict(update=update,record_name=row.record_name,rollout=sample,
                                  statistics=encoded['normalizer_clipping']))
            encoded.pop('cube_hat')
            encoded['block_log'] = encoded['block_log'].to_dict('records')
            trajectories.append(encoded); records.append(record)
            if sample == primary_for_branch:
                primary = encoded
                branch_start = flat_offset+sample*len(encoded['actions'])
            del post
        for tile in tiles:
            old = actor.forward(primary['states'][tile:tile+1])[0][0].astype(float)
            old /= old.sum()
            measured_actions=[]
            print(f'V12.22 {arm} seed {seed} update {update}: querying {len(actions)} fresh action continuations',flush=True)
            for action in range(len(actions)):
                # CRN across current alternatives; a fresh on-policy suffix,
                # never the historical fixed-allocation physical table.
                entropy = [int(seed),1221,pass_index,update,tile,0]
                facade = ForcedAction(actor,primary['actions'],tile,action,np.random.SeedSequence(entropy))
                alternative = codec.encode(ex,cube,facade,norm,True,source,features)
                assert_prefix_matches(primary,alternative,tile,source['action_bits_by_block'])
                post = runtime.post(alternative['cube_hat'],row,'conditional_action',
                                    arm=arm,seed=int(seed),update=update,action=action)
                outcome = runtime.measure(original,post,labs,alternative)
                measured_actions.append(outcome)
                branches.append(dict(arm=arm,seed=int(seed),update=update,record_name=row.record_name,
                    primary_row=branch_start+tile,tile_index=tile,action=action,rho=actions[action][0],
                    q=actions[action][1],old_probability=float(old[action]),seed_entropy=entropy,**outcome))
                del alternative,post
                if (action+1)%7==0 or action+1==len(actions):
                    print(f'  conditional actions {action+1}/{len(actions)}',flush=True)
            packets.append(_packet(branch_start+tile,old,measured_actions))
        del cube,source,original;gc.collect()
    return dict(trajectories=trajectories,records=records,packets=packets,branches=branches,
                placement=placement,clipping=clipping,actor_rng_after=copy.deepcopy(actor.rng.bit_generator.state),
                selection_rng_after=copy.deepcopy(selection_rng.bit_generator.state))


def _logged_training_records(records, statistics):
    """Keep perception-objective losses separate from optimizer losses."""
    scalars = {('optimizer_total_loss' if key == 'total_loss' else key): value
               for key, value in statistics.items()
               if value is None or isinstance(value, (str, bool, int, float, np.generic))}
    result = []
    for record in records:
        overlap = set(record) & set(scalars)
        if overlap:
            raise ValueError('Optimizer statistics would overwrite measured outcomes: ' + str(sorted(overlap)))
        result.append({**record, **scalars})
    return result


def _export_job_tables(folder):
    """Derive tables solely from committed updates; never duplicate resumed rows."""
    records,branches,credits,actions,auxiliary,contexts,placement,clipping = ([] for _ in range(8))
    for receipt_path in sorted((folder/'updates').glob('update_*/COMPLETE.json')):
        verify_receipt(receipt_path)
        result = load_tree(receipt_path.parent/'log.npz')
        records.extend(result['records']);branches.extend(result['branches'])
        credits.extend(result['credits']);actions.extend(result['actions'])
        auxiliary.extend(result['auxiliary']);contexts.extend(result['contexts'])
        placement.extend(result['placement']);clipping.extend(result['clipping'])
    for name,rows in [('training',records),('conditional_actions',branches),('actor_credit_rows',credits),
                      ('conditional_action_credit',actions),('auxiliary_updates',auxiliary),
                      ('auxiliary_contexts',contexts),('query_placement',placement)]:
        write_csv(folder/'tables'/f'{name}.csv',rows)
    write_json(folder/'tables/normalizer_clipping.json',clipping)
    histories,per_cube = [],[]
    for path in sorted((folder/'checkpoints').glob('update_*/COMPLETE.json')):
        verify_receipt(path)
        checkpoint = load_tree(path.parent/'results.npz')
        for split,summary in checkpoint['summaries'].items():
            histories.append(dict(arm=checkpoint['identity']['arm'],seed=checkpoint['identity']['seed'],
                                  update=checkpoint['update'],split=split,**summary))
            per_cube.extend(checkpoint['records'][split])
    write_csv(folder/'tables/checkpoints.csv',histories)
    write_csv(folder/'tables/per_cube.csv',per_cube)
    return pd.DataFrame(records),pd.DataFrame(histories)


def train_job(runtime,arm,seed,*,resume=False):
    from radar_sim.ppo_v12_22_learning import update_ppo_v12_22
    verify_receipt(runtime.root/'PREPARE_COMPLETED.json')
    folder = _job_directory(runtime.root,arm,seed)
    if (folder/'TRAIN_COMPLETED.json').is_file():
        if not resume:
            raise FileExistsError('Job already completed; use --resume to skip it: '+str(folder))
        verify_receipt(folder/'TRAIN_COMPLETED.json')
        print('Already complete; preserving '+str(folder),flush=True)
        return read_json(folder/'job_status.json')
    if folder.exists() and not resume:
        raise FileExistsError('Partial job exists; explicit --resume required: '+str(folder))
    folder.mkdir(parents=True,exist_ok=True)
    actor,norm,actions,coefficient = _job_settings(runtime,arm,seed)
    optimizer = runtime.ex.PPOAdamV12_11(actor.params,.0005)
    selection_rng = np.random.default_rng(np.random.SeedSequence([int(seed),1221,77]))
    initial = folder/'models/initial.npz'
    if not initial.is_file():
        save_tree(initial,_state(actor,optimizer,selection_rng,0))
    else:
        _restore(actor,optimizer,selection_rng,load_tree(initial))
    write_json(folder/'models/normalizer.json',norm.as_dict())
    cfg = runtime.config
    order_rng = np.random.default_rng(int(seed))
    orders = [order_rng.permutation(len(runtime.train_rows)) for _ in range(cfg['passes'])]
    schedule = [(p,order[i:i+cfg['batch_cubes']]) for p,order in enumerate(orders)
                for i in range(0,len(order),cfg['batch_cubes'])]
    if len(schedule) != cfg['updates']:
        raise AssertionError('Optimizer schedule does not match frozen profile.')
    schedule_path = folder/'models/permutations.npz'
    if schedule_path.is_file():
        if any(not np.array_equal(a,b) for a,b in zip(load_tree(schedule_path)['orders'],orders)):
            raise ValueError('Training permutation changed on resume.')
    else:
        save_tree(schedule_path,dict(orders=orders))
    _evaluate_checkpoint(runtime,arm,seed,actor,norm,actions,0)
    for update,(pass_index,positions) in enumerate(schedule,1):
        directory = folder/'updates'/f'update_{update:04d}'
        if (directory/'COMPLETE.json').is_file():
            verify_receipt(directory/'COMPLETE.json')
            _restore(actor,optimizer,selection_rng,load_tree(directory/'state.npz'))
        else:
            directory.mkdir(parents=True,exist_ok=True)
            before = directory/'pre_update.npz'
            if before.is_file():
                _restore(actor,optimizer,selection_rng,load_tree(before))
            else:
                save_tree(before,_state(actor,optimizer,selection_rng,update-1))
            frozen_hash = actor_hash(actor)
            batch_file = directory/'frozen_batch.npz'
            if batch_file.is_file():
                batch = load_tree(batch_file)
                if batch['behavior_policy_hash'] != frozen_hash or batch['update'] != update:
                    raise ValueError('Frozen batch and pre-update behavior policy differ.')
                actor.rng.bit_generator.state = batch['actor_rng_after']
                selection_rng.bit_generator.state = batch['selection_rng_after']
            else:
                batch = _collect_batch(runtime,arm,seed,actor,norm,actions,selection_rng,update,positions,pass_index)
                batch.update(behavior_policy_hash=frozen_hash,update=update,
                             positions=np.asarray(positions),pass_index=pass_index)
                save_tree(batch_file,batch)
            stats = update_ppo_v12_22(actor,optimizer,batch['trajectories'],batch['packets'],
                auxiliary=_definition(arm)['auxiliary'],auxiliary_coefficient=coefficient,
                epochs=cfg['epochs'],clip=.2,entropy_coef=.002,target_kl=.02,value_coef=.5,
                reward_scale=1.,gradients_fn=runtime.ex._network_gradients_v11,
                mask_fn=runtime.ex._apply_action_mask_v11,seed=int(seed),eps=runtime.ex.EPS)
            flat = np.concatenate([item['states'] for item in batch['trajectories']])
            probability_after = actor.forward(flat)[0]
            credit_rows = [dict(update=update,**row) for row in stats.pop('credit_rows')]
            action_rows=[]
            for item in stats.pop('conditional_action_rows'):
                after = float(probability_after[item['primary_row'],item['action']])
                action_rows.append(dict(update=update,**item,post_update_probability=after,
                                       probability_change=after-item['old_probability']))
            auxiliary_rows = [dict(update=update,**row) for row in stats.pop('auxiliary_gradient_rows',[])]
            context_rows = [dict(update=update,**row) for row in stats.pop('auxiliary_context_rows',[])]
            # Keep scalar update diagnostics in CSV; rich arrays stay in NPZ.
            records = _logged_training_records(batch['records'], stats)
            log = dict(records=records,branches=batch['branches'],credits=credit_rows,actions=action_rows,
                       auxiliary=auxiliary_rows,contexts=context_rows,placement=batch['placement'],
                       clipping=batch['clipping'],update_statistics=stats)
            save_tree(directory/'log.npz',log)
            save_tree(directory/'state.npz',_state(actor,optimizer,selection_rng,update))
            receipt(directory/'COMPLETE.json',[before,batch_file,directory/'state.npz',directory/'log.npz'],
                    arm=arm,seed=int(seed),update=update,behavior_policy_hash=frozen_hash,
                    collected_before_optimizer=True,critic_branch_targets=False)
            avg = pd.DataFrame(records)
            print(f'V12.22 {arm} seed {seed} update {update}/{len(schedule)} stochastic: '
                  f'LOW={avg.low_miss.mean():.3f} SIX={avg.six_miss.mean():.3f} '
                  f'unmatched_excess={avg.unmatched_excess.mean():.2f} '
                  f'reward={avg.reward.mean():+.5f} bpcs={avg.bpcs.mean():.3f}',flush=True)
            _export_job_tables(folder)
        if update in cfg['checkpoints']:
            _evaluate_checkpoint(runtime,arm,seed,actor,norm,actions,update)
            _export_job_tables(folder)
    checkpoints=[]
    for path in sorted((folder/'checkpoints').glob('update_*/COMPLETE.json')):
        verify_receipt(path)
        checkpoints.append(load_tree(path.parent/'results.npz'))
    # Earliest update wins exact ties. A winning initialization is explicit.
    best = min(checkpoints,key=lambda value:(tuple(value['keys']['new']),value['update']))
    historical = min(checkpoints,key=lambda value:(tuple(value['keys']['historical']),value['update']))
    save_tree(folder/'models/last.npz',_state(actor,optimizer,selection_rng,cfg['updates']))
    save_tree(folder/'models/selected.npz',dict(policy=best['policy'],actor_rng=best['actor_rng'],update=best['update']))
    _export_job_tables(folder)
    selected = pd.DataFrame(best['records']['validation'])
    write_csv(folder/'tables/validation_per_cube.csv',selected)
    write_csv(folder/'tables/selected_validation_actions.csv',best['actions']['validation'])
    status = dict(completed=True,arm=arm,seed=int(seed),selected_update=best['update'],
        historical_selected_update=historical['update'],new_selection_key=best['keys']['new'],
        historical_selection_key=historical['keys']['historical'],
        selected_initialization=best['update']==0,selected_summary=best['summaries']['validation'],
        final_summary=checkpoints[-1]['summaries']['validation'],
        auxiliary=_definition(arm)['auxiliary'],auxiliary_coefficient=coefficient,
        test_evaluated=False,no_imitation=True,no_refinement=True,rate_cap=None,
        optimizer_resumable=True,primary_critic_only=True,
        physical_evaluations=_evaluation_count(runtime.root,arm,seed))
    write_json(folder/'job_status.json',status)
    receipt(folder/'TRAIN_COMPLETED.json',[folder/'job_status.json',folder/'models/selected.npz',
            folder/'models/last.npz',folder/'models/normalizer.json',schedule_path],test_evaluated=False)
    return status


def train_interaction(runtime,parent,*,resume=False):
    """Optional TWO-run state control; never automatically choose an auxiliary."""
    if parent not in ('D','E','F') or runtime.manifest['profile']!='full' or len(runtime.manifest['seeds'])!=2:
        raise ValueError('The reserve requires full profile, two seeds and one explicit D/E/F parent.')
    if runtime.manifest['arms']!=list(ARMS):
        raise ValueError('Complete the six-arm base comparison before the interaction reserve.')
    for arm in ARMS:
        for seed in runtime.manifest['seeds']:
            verify_receipt(_job_directory(runtime.root,arm,seed)/'TRAIN_COMPLETED.json')
    values=dict(parent_arm=parent,control_arm=parent+'0',seeds=runtime.manifest['seeds'],
        difference='Four CFAR observation slots are zero; menu/init/reward/auxiliary/coefficient/schedule unchanged.',
        coefficients='Frozen coefficients from the original TRAIN calibration; not recalibrated for this control.',
        test_evaluation=False,max_additional_jobs=2)
    path=runtime.root/'INTERACTION_RESERVE.json'
    if path.is_file() and digest(read_json(path))!=digest(values):
        raise ValueError('The two-run reserve was already assigned; additional parents exceed the approved scope.')
    write_json(path,values)
    return [train_job(runtime,parent+'0',seed,resume=resume) for seed in runtime.manifest['seeds']]


def _baseline(runtime,row,cube,original,method):
    """Frozen RDO/FFT baselines shared across jobs; never consult the actor."""
    path=runtime.root/'cache/baselines'/str(row.record_name)/(method+'.npz')
    if path.is_file():
        return load_tree(path)
    ex=runtime.ex
    if method=='CalibratedRDO':
        decoded,_,meta=ex.compress_uniform_action_v11(cube,tuple(runtime.manifest['fixed_rdo_action']))
    elif method=='FFT14':
        decoded,_,meta=ex.compress_cube_fft_full_support_multi(cube,{'FFT14':14})['FFT14']
    else:
        raise ValueError('Unknown independent baseline.')
    post=runtime.post(decoded,row,'independent_baseline',method=method)
    quality=ex._cube_quality_row_v12_6(row.record_name,row.scene_name,method,cube,decoded,meta,original,post)
    result=dict(post=post,meta=meta,quality=quality)
    save_tree(path,result)
    return result


def _rename_report_files(directory):
    """Version names indicate THIS run, with an explicit function-lineage map."""
    import re
    rows=[]
    for path in sorted(Path(directory).rglob('*')):
        if not path.is_file():
            continue
        name=re.sub(r'_v12_\d+(?:_\d+)?(?=\.(?:png|csv|json)$)','_v12_22',path.name)
        if name!=path.name:
            target=path.with_name(name)
            if target.exists():
                raise FileExistsError('Distinct report artifacts would collide: '+str(target))
            path.rename(target)
            rows.append(dict(inherited_function_filename=path.name,run_filename=name))
    write_csv(Path(directory)/'figure_and_table_lineage.csv',rows)


def report_job(runtime,arm,seed):
    """All inherited four-policy figures, using explicit generalized encoding."""
    from radar_sim.ppo_v12_21_pipeline import report_directories
    from radar_sim.ppo_v12_21_run import plot_cfar
    from radar_sim.ppo_v12_21_reporting import collect_detection_frame,export_detection_reports
    from radar_sim.ppo_v12_19_diagnostics import _label_rows
    from radar_sim.ppo_v12_22_reporting import plot_job_learning
    actor,norm,actions,cfar,job=load_job(runtime,arm,seed,True)
    complete=job/'REPORT_COMPLETED.json'
    if complete.is_file():
        return verify_receipt(complete)
    # A failed attempt stays recoverable; inherited exporters refuse overwrite.
    destination=job/'reports'/('validation_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+str(time.time_ns()%1000000000))
    destination.mkdir(parents=True)
    ex=runtime.ex
    identity=digest(dict(actor=actor_hash(actor),normalizer=norm.as_dict(),actions=actions,
                         frozen_rdo=runtime.manifest['fixed_rdo_action'],source=runtime.manifest['source_hashes']))
    sweeps,quality,label_audit,frames,perception,peaks,action_rows=[],[],[],[],[],[],[]
    seen=set();visuals=[]
    with report_directories(ex,destination):
        for ci,(_,row) in enumerate(runtime.validation_rows.iterrows()):
            cached_path=job/'reports/cases'/f'{row.record_name}.npz'
            representative=str(row.scene_name) not in seen
            seen.add(str(row.scene_name))
            case=load_tree(cached_path) if cached_path.is_file() else None
            if case is not None and case['identity']!=identity:
                raise ValueError('Saved report case refers to a different selected policy.')
            if case is None or representative:
                cube=runtime.load_cube(row);original=runtime.original(row,cube)
                labs=ex.labels_for_cube_v6_1000(runtime.labels,row)
                source=codec.prepare_source(ex,cube,actions)
                features=runtime.features(row,original,source) if cfar else None
                encoded=codec.encode(ex,cube,actor,norm,False,source,features)
                post=runtime.post(encoded['cube_hat'],row,'selected_policy_report',arm=arm,seed=int(seed))
                baselines={method:_baseline(runtime,row,cube,original,method) for method in ('CalibratedRDO','FFT14')}
                methods={'Original':(cube,dict(realized_bits=32*cube.size,realized_bpcs=32.),original),
                         'PPO':(encoded['cube_hat'],encoded['meta'],post)}
                methods.update({name:(None,value['meta'],value['post']) for name,value in baselines.items()})
                if representative:
                    visual=dict(row=row.copy(),labs=labs.copy(),methods=methods)
                    ex.plot_v12_7_scene_comparisons([visual])
                    if len(visuals)<2:
                        # Only two explicitly requested decoded tensors, not every cube.
                        visuals.append(dict(row=row.copy(),methods={'PPO':(encoded['cube_hat'].copy(),encoded['meta'].copy(),None)}))
                if case is None:
                    case=dict(identity=identity,sweeps=[],quality=[],label_audit=[],frames=[],
                              perception=dict(record_name=row.record_name,scene_name=row.scene_name,
                                              **runtime.measure(original,post,labs,encoded)),
                              peaks=_label_rows(ex,original,post,labs,str(row.record_name),str(row.scene_name)).to_dict('records'),
                              actions=encoded['block_log'].assign(record_name=row.record_name,scene_name=row.scene_name).to_dict('records'))
                    for method,(decoded,meta,current) in methods.items():
                        q=baselines[method]['quality'] if method in baselines else ex._cube_quality_row_v12_6(
                            row.record_name,row.scene_name,method,cube,decoded,meta,original,current)
                        case['quality'].append(q)
                    for kind in ('RD','RA'):
                        original_map,ranges,second=ex._map_triplet_v12_6(original,kind)
                        gate=ex.CFAR_LABEL_GATE_RD_V8 if kind=='RD' else ex.CFAR_LABEL_GATE_RA_V8
                        factors=ex.V12_7_RD_PRESERVATION_THRESHOLD_FACTORS if kind=='RD' else ex.V12_7_BASE_CFAR_THRESHOLD_FACTORS
                        cache=ex.precompute_original_cfar_count_sweep_v11_2(original_map,labs,ranges,second,kind,factors,gate)
                        for method,(_,meta,current) in methods.items():
                            data,_,_=ex._map_triplet_v12_6(current,kind)
                            ex._append_label_audit_v12_6(case['label_audit'],cache,data,labs,method,row.record_name,row.scene_name)
                            counts=ex.evaluate_cfar_count_sweep_method_v11_2(cache,data)
                            for item in counts.itertuples():
                                case['sweeps'].append(dict(cube_idx=ci,record_name=row.record_name,scene_name=row.scene_name,
                                    map_type=kind,method=method,method_display=ex.V12_7_DISPLAY[method],
                                    threshold_factor=item.threshold_factor,cfar_threshold_db=item.cfar_threshold_db,
                                    eligible_original=int(item.num_labels_supported_original),
                                    preserved_original_supported=int(item.num_labels_preserved_from_original),
                                    missed_original_supported=int(item.num_labels_missed_from_original),
                                    unmatched_false_peaks=int(item.num_label_unmatched_peaks_method),
                                    num_cfar_peaks=int(item.num_cfar_peaks_method),realized_bpcs=meta['realized_bpcs']))
                            case['frames'].append(collect_detection_frame(ex,current,labs,kind,row.record_name,method))
                    save_tree(cached_path,case)
                del cube,source,encoded,post,original,methods,baselines;gc.collect()
            sweeps.extend(case['sweeps']);quality.extend(case['quality']);label_audit.extend(case['label_audit'])
            frames.extend(case['frames']);perception.append(case['perception']);peaks.extend(case['peaks']);action_rows.extend(case['actions'])
            print(f'V12.22 {arm} seed {seed} VALIDATION reports {ci+1}/{len(runtime.validation_rows)}',flush=True)
        sweep=pd.DataFrame(sweeps);qtable=pd.DataFrame(quality)
        for name,values in [('validation_cfar_sweep',sweep),('validation_compression_quality',qtable),
                            ('validation_object_label_audit',label_audit),('validation_PPO_components',perception),
                            ('validation_PPO_label_peak_noise',peaks),('validation_PPO_actions',action_rows)]:
            write_csv(destination/'tables'/f'{name}_v12_22.csv',values)
        plot_cfar(ex,sweep,'validation')
        ex.plot_v12_7_compression_effectiveness(qtable,'validation')
        if visuals:
            ex.save_v12_7_reconstructed_ppo_cubes(visuals,max_cubes=2)
        export_detection_reports(ex,frames,destination/'detection_metrics','validation')
        plot_job_learning(ex,job,destination/'figures')
    _rename_report_files(destination)
    write_json(destination/'REPORT_PROTOCOL.json',dict(arm=arm,seed=int(seed),test_evaluated=False,
        selected_policy_hash=actor_hash(actor),checkpoint=read_json(job/'job_status.json')['selected_update'],
        inherited_figures_preserved=True,RDO_shared_and_frozen=True,FFT14_unchanged=True,
        decoded_npy_npz_are_not_codec_payload=True,raw_cube_layout='Rx,chirp,fast-time'))
    receipt(complete,[path for path in sorted(destination.rglob('*')) if path.is_file()],
            directory=str(destination),test_evaluated=False)
    return read_json(complete)


def rate_reports(runtime):
    """One shared measured action sweep; twelve PPO operating points, not a curve."""
    from radar_sim.ppo_v12_19_rate_sweep import FFT_BITS,summarize_operating_points
    from radar_sim.ppo_v12_22_reporting import plot_selected_rate_reports
    destination=runtime.root/('rate_sweep_interaction' if (runtime.root/'INTERACTION_RESERVE.json').is_file() else 'rate_sweep')
    cases_dir=runtime.root/'rate_sweep/cases'
    if (destination/'COMPLETE.json').is_file():
        return verify_receipt(destination/'COMPLETE.json')
    destination.mkdir(parents=True,exist_ok=True)
    ex=runtime.ex
    population=ex.select_split_rows_v11(runtime.validation_rows,min(runtime.config['rate_sweep_cubes'],len(runtime.validation_rows)))
    quality=[];sweeps=[]
    for _,row in population.iterrows():
        cube=runtime.load_cube(row);original=runtime.original(row,cube)
        labs=ex.labels_for_cube_v6_1000(runtime.labels,row)
        source=codec.prepare_source(ex,cube,codec.PROPOSED_ACTIONS)
        caches={}
        for kind in ('RD','RA'):
            data,rr,second=ex._map_triplet_v12_6(original,kind)
            gate=ex.CFAR_LABEL_GATE_RD_V8 if kind=='RD' else ex.CFAR_LABEL_GATE_RA_V8
            caches[kind]=ex.precompute_original_cfar_count_sweep_v11_2(data,labs,rr,second,kind,ex._nearest_training_thresholds_v11(),gate)

        def record(name,family,decoded,meta,action=-1,rho=1.,q=0,post=None):
            case_path=cases_dir/f'{row.record_name}_{name}.npz'
            if case_path.is_file():
                case=load_tree(case_path)
            else:
                current=post if post is not None else runtime.post(decoded,row,'rate_sweep',policy=name)
                item=dict(record_name=row.record_name,scene_name=row.scene_name,policy_id=name,family=family,
                    action_index=action,rho=rho,q=q,complex_samples=cube.size,source_bits=32*cube.size,
                    realized_bits=meta['realized_bits'],accounted_bpcs=meta['realized_bits']/cube.size,
                    MiB_saved=(32*cube.size-meta['realized_bits'])/(8*1024**2),cube_nmse=meta.get('cube_nmse',0.))
                rows=[]
                for kind in ('RD','RA'):
                    data,_,_=ex._map_triplet_v12_6(current,kind)
                    data0,_,_=ex._map_triplet_v12_6(original,kind)
                    counts=ex.evaluate_cfar_count_sweep_method_v11_2(caches[kind],data)
                    counts=counts.assign(record_name=row.record_name,policy_id=name,family=family,map_type=kind)
                    rows.extend(counts.to_dict('records'))
                    item[kind+'_log_power_mse']=ex._normalized_log_power_mse_v11(data0,data)
                case=dict(quality=item,sweeps=rows)
                save_tree(case_path,case)
            quality.append(case['quality']);sweeps.extend(case['sweeps'])
        record('Original','Original',cube,dict(realized_bits=32*cube.size,cube_nmse=0.),post=original)
        for bits in FFT_BITS:
            name=f'FFT{bits}'
            if (cases_dir/f'{row.record_name}_{name}.npz').is_file():
                record(name,'FFT_full_support',None,{})
            else:
                decoded,_,meta=ex.compress_cube_fft_full_support_multi(cube,{'FFT':bits})['FFT']
                record(name,'FFT_full_support',decoded,meta,q=bits)
                del decoded
        for index,(rho,q) in enumerate(codec.PROPOSED_ACTIONS):
            name=f'Uniform{index:02d}'
            if (cases_dir/f'{row.record_name}_{name}.npz').is_file():
                record(name,'Uniform_library',None,{})
            else:
                encoded=ex.coherent_codec_v14.encode_sequence(cube,np.full(len(source['slices']),index),source=source)
                record(name,'Uniform_library',encoded['cube_hat'],encoded['meta'],index,rho,q)
                del encoded
        for arm,seed in _job_pairs(runtime):
                actor,norm,menu,cfar,_=load_job(runtime,arm,seed)
                name=f'PPO_{arm}_seed{seed}'
                if (cases_dir/f'{row.record_name}_{name}.npz').is_file():
                    record(name,'Trained_PPO',None,{})
                else:
                    own_source=source if menu==codec.PROPOSED_ACTIONS else codec.prepare_source(ex,cube,menu)
                    bundle=runtime.features(row,original,own_source) if cfar else None
                    encoded=codec.encode(ex,cube,actor,norm,False,own_source,bundle)
                    record(name,'Trained_PPO',encoded['cube_hat'],encoded['meta'])
                    del encoded,own_source
        del cube,source,original;gc.collect()
    qframe,sframe=pd.DataFrame(quality),pd.DataFrame(sweeps)
    summary=summarize_operating_points(sframe,qframe).drop(columns=['above_PPO_4p8_cap'],errors='ignore')
    write_csv(destination/'per_cube_v12_22.csv',qframe)
    write_csv(destination/'per_threshold_v12_22.csv',sframe)
    write_csv(destination/'summary_v12_22.csv',summary)
    # Separate panels per learned policy avoid twelve overlapping annotations.
    fixed=summary.loc[summary.family.ne('Trained_PPO')]
    for arm,seed in _job_pairs(runtime):
            name=f'PPO_{arm}_seed{seed}'
            selected=summary.loc[summary.policy_id.eq(name)]
            policy_dir=destination/f'{arm}_seed{seed}'
            if (policy_dir/'COMPLETE.json').is_file():
                verify_receipt(policy_dir/'COMPLETE.json')
                continue
            attempt=policy_dir/f'attempt_{time.time_ns()}'
            status=read_json(_job_directory(runtime.root,arm,seed)/'job_status.json')
            plot_selected_rate_reports(ex,pd.concat([fixed,selected],ignore_index=True),attempt,
                                       arm=arm,selected_update=int(status['selected_update']))
            _rename_report_files(attempt)
            expected=[attempt/f'{metric}_vs_bits_saved_v12_22.png' for metric in ('label_preservation','unmatched_peaks')]
            if not all(path.is_file() for path in expected):
                raise RuntimeError('A rate-sweep figure family is incomplete.')
            receipt(policy_dir/'COMPLETE.json',[path for path in sorted(attempt.rglob('*')) if path.is_file()],directory=str(attempt))
    receipt(destination/'COMPLETE.json',[path for path in sorted(destination.rglob('*')) if path.is_file()],
            test_evaluated=False,diagnostic_only=True,uniform_actions_not_used_for_deployment=True)
    return read_json(destination/'COMPLETE.json')


def export_execution_cost(runtime):
    path=runtime.root/'physical_evaluations.jsonl'
    if path.exists():
        frame=pd.read_json(path,lines=True)
        for column in ('arm','seed'):
            if column not in frame:
                frame[column]='shared'
            frame[column]=frame[column].fillna('shared')
        summary=frame.groupby(['kind','arm','seed'],dropna=False).agg(
            physical_evaluations=('kind','size'),seconds=('seconds','sum')).reset_index()
    else:
        summary=pd.DataFrame(columns=['kind','arm','seed','physical_evaluations','seconds'])
    write_csv(runtime.root/'tables/physical_evaluation_ledger.csv',summary)
    return summary


@contextmanager
def _exclusive_run(out,resume):
    """Prevent a notebook and launcher from writing the same experiment."""
    out=Path(out).resolve()
    lock=out.with_name(out.name+'.RUNNING.lock')
    lock.parent.mkdir(parents=True,exist_ok=True)
    token=uuid.uuid4().hex
    try:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError:
        old=read_json(lock)
        if not resume or old.get('schema')!='V1222_PROCESS_LOCK' or old.get('host')!=platform.node():
            raise RuntimeError('Experiment is locked. Do not run notebook/CLI concurrently: '+str(lock))
        try:
            os.kill(int(old['pid']),0)
            alive=True
        except ProcessLookupError:
            alive=False
        except PermissionError:
            alive=True
        if alive:
            raise RuntimeError('The experiment owner process is still alive. Refusing concurrent resume.')
        # Explicit resume can recover this one known stale control file only.
        lock.unlink()
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'w',encoding='utf-8') as stream:
        json.dump(dict(schema='V1222_PROCESS_LOCK',host=platform.node(),pid=os.getpid(),token=token),stream)
        stream.flush();os.fsync(stream.fileno())
    try:
        yield
    finally:
        if lock.is_file() and read_json(lock).get('token')==token:
            lock.unlink()


def _execute_stage(stage,out,profile_name='full',arms=None,seeds=None,*,resume=False,acknowledge_scope=False,interaction_arm=None):
    arms=list(ARMS) if arms is None else list(arms)
    seeds=list(DEFAULT_SEEDS) if seeds is None else [int(value) for value in seeds]
    if not arms or len(set(arms))!=len(arms) or not set(arms)<=set(ARMS) or not seeds or len(set(seeds))!=len(seeds) or any(s<0 for s in seeds):
        raise ValueError('Use distinct known arms and nonnegative training seeds.')
    if stage=='check':
        from radar_sim.ppo_priority2_scoped_labels_v2 import verify_native_label_loop_parity
        checks=dict(revision=REVISION,configuration=profile(profile_name),arms=arms,seeds=seeds,
                    original_actions=codec.ORIGINAL_ACTIONS,proposed_actions=codec.PROPOSED_ACTIONS,
                    native_label_normalization=verify_native_label_loop_parity(),
                    test_evaluation=False,source_hashes=source_hashes(),backend='CPU NumPy/SciPy',
                    output_created=False,training_started=False)
        print(json.dumps(checks,indent=2),flush=True)
        return checks
    if profile_name=='full' and stage in ('train','all') and not acknowledge_scope:
        raise ValueError('Full training requires --acknowledge-scope (six arms/two seeds by default).')
    if stage in ('prepare','all'):
        runtime=prepare(out,profile_name,arms,seeds,resume=resume)
    else:
        verify_receipt(Path(out)/'PREPARE_COMPLETED.json')
        runtime=Runtime(out)
        if runtime.manifest['profile']!=profile_name or runtime.manifest['arms']!=arms or runtime.manifest['seeds']!=seeds:
            raise ValueError('CLI populations/arms/seeds differ from the frozen experiment.')
    try:
        if stage in ('train','all'):
            if interaction_arm is not None:
                if stage!='train':
                    raise ValueError('Use --interaction-arm only with the explicit train stage.')
                train_interaction(runtime,interaction_arm,resume=resume)
            else:
                for arm in arms:
                    for seed in seeds:
                        train_job(runtime,arm,seed,resume=resume)
        if stage in ('diagnostics','all'):
            from radar_sim.ppo_v12_22_diagnostics import physical_menu_checks,selected_policy_diagnostics
            physical_menu_checks(runtime)
            selected_policy_diagnostics(runtime)
        if stage in ('report','all'):
            for arm,seed in _job_pairs(runtime):
                report_job(runtime,arm,seed)
            rate_reports(runtime)
            export_execution_cost(runtime)
            from radar_sim.ppo_v12_22_reporting import build_comparison_report
            build_comparison_report(runtime.root)
            write_json(runtime.root/'EXPERIMENT_COMPLETED.json',dict(completed=True,test_evaluated=False,
                profile=profile_name,arms=arms,seeds=seeds,physical_evaluations=_evaluation_count(runtime.root),
                scientific_superiority_not_assumed=True))
    finally:
        export_execution_cost(runtime)
    return dict(output=str(runtime.root),stage=stage,test_evaluated=False)


def run_stage(stage,out,profile_name='full',arms=None,seeds=None,*,resume=False,acknowledge_scope=False,interaction_arm=None):
    if interaction_arm is not None and stage!='train':
        raise ValueError('Use --interaction-arm only with the explicit train stage.')
    if stage=='check':
        return _execute_stage(stage,out,profile_name,arms,seeds,resume=resume,acknowledge_scope=acknowledge_scope,interaction_arm=interaction_arm)
    with _exclusive_run(out,resume):
        return _execute_stage(stage,out,profile_name,arms,seeds,resume=resume,acknowledge_scope=acknowledge_scope,interaction_arm=interaction_arm)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('check','prepare','train','diagnostics','report','all'),required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--profile',choices=('full','tiny'),default='full')
    parser.add_argument('--arms',nargs='+',choices=tuple(ARMS),default=list(ARMS))
    parser.add_argument('--seeds',nargs='+',type=int,default=list(DEFAULT_SEEDS))
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--acknowledge-scope',action='store_true')
    parser.add_argument('--interaction-arm',choices=('D','E','F'),help='Optional original-state control, after all base jobs complete; two additional runs maximum.')
    args=parser.parse_args(argv)
    return run_stage(args.stage,args.out,args.profile,args.arms,args.seeds,
                     resume=args.resume,acknowledge_scope=args.acknowledge_scope,interaction_arm=args.interaction_arm)


if __name__=='__main__':
    main()
