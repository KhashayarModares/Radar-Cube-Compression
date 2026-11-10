"""V12.21: uncapped, perception-first PPO with refreshed conditional credit.

One training loop, one encoder, one objective. The V12.19 module is used ONLY
as a frozen library of dataset/codec/network/detector/plotting primitives.
Its training loop, masking, reward, checkpoint selector and refinement are not
called or replaced. Existing source files and experiment outputs stay intact.
"""
from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
from dataclasses import asdict, dataclass
import gc
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from radar_sim.ppo_v12_21_codec import encode_uncapped, ForcedAction
from radar_sim.ppo_v12_21_objective import PerceptionObjective, checkpoint_key, perception_terms
from radar_sim.ppo_v12_21_learning import update_ppo_v12_21
from radar_sim.ppo_prefix_rollouts import assert_prefix_matches
from radar_sim.ppo_v12_19_reward import LOSS_COMPONENT_NAMES, preservation_weighted_reward_terms
from radar_sim.ppo_v12_21_prefixes import choose_training_prefixes

ROOT = Path(__file__).resolve().parents[1]
REVISION = '12.21.0-uncapped-perception-first-conditional-ppo'


@dataclass(frozen=True)
class RunConfig:
    mode: str = 'full'
    train_cubes: int = 600
    validation_cubes: int = 200
    test_cubes: int = 200
    passes: int = 2
    rollouts: int = 2
    batch_cubes: int = 5
    ppo_epochs: int = 5
    seed: int = 20260825
    learning_rate: float = .0005
    conditional_contexts_per_batch: int = 1
    branch_suffixes: int = 1
    validation_probe_cubes: int = 40
    validation_every_updates: int = 30
    normalizer_cubes: int = 15
    rate_sweep_cubes: int = 8
    objective: str = 'perception_first'
    conditional_credit: bool = True
    prefix_sampling: str = 'original_supported'

    def __post_init__(self):
        for name in ('train_cubes', 'validation_cubes', 'passes', 'rollouts', 'batch_cubes',
                     'ppo_epochs', 'branch_suffixes', 'validation_probe_cubes', 'normalizer_cubes'):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer.')
        if self.rollouts < 2 or self.test_cubes < 0 or self.conditional_contexts_per_batch < 0:
            raise ValueError('Use >=2 primary rollouts and nonnegative test/conditional counts.')
        if self.objective not in ('perception_first', 'old_objective_control'):
            raise ValueError('Unknown objective.')
        if self.prefix_sampling not in ('original_supported','uniform'):
            raise ValueError('Use original_supported TRAIN measurement placement or uniform control.')

    @classmethod
    def from_environment(cls):
        mode = os.environ.get('RDO_RL_RUN_MODE', 'full')
        defaults = cls() if mode == 'full' else cls(mode=mode, train_cubes=4,
            validation_cubes=2, test_cubes=0, passes=2, batch_cubes=2,
            validation_probe_cubes=2, validation_every_updates=1,
            normalizer_cubes=4, conditional_contexts_per_batch=1, rate_sweep_cubes=2)
        names = dict(train_cubes='RDO_RL_TRAIN_CUBES', validation_cubes='RDO_RL_VAL_CUBES',
            test_cubes='RDO_NUM_CUBES', passes='RDO_RL_PASSES', rollouts='RDO_V1221_ROLLOUTS',
            batch_cubes='RDO_RL_BATCH_EPISODES', ppo_epochs='RDO_RL_PPO_EPOCHS',
            seed='RDO_RL_SEED', conditional_contexts_per_batch='RDO_V1221_CONDITIONAL_CONTEXTS',
            branch_suffixes='RDO_V1221_BRANCH_SUFFIXES', validation_probe_cubes='RDO_V1221_VAL_PROBES',
            validation_every_updates='RDO_V1221_VAL_EVERY_UPDATES', normalizer_cubes='RDO_V1221_NORMALIZER_CUBES',
            rate_sweep_cubes='RDO_V1221_RATE_SWEEP_CUBES')
        values = asdict(defaults)
        values.update({name:int(os.environ[key]) for name,key in names.items() if key in os.environ})
        return cls(**values)


def load_primitives(output):
    """Disable legacy execution before importing physics/report primitives."""
    settings=dict(RDO_RUN_PIPELINE='0', RDO_RL_RUN_MODE='definitions_only',
        RDO_RL_RUN_TAG='v1221_library_definitions_only', RDO_ALLOW_MOUNTED_OUTPUT='1',
        RDO_OUTPUT_ROOT=str(Path(output)/'_library_definitions'), RDO_RL_RATE_TARGET_BPCS='4.8',
        RDO_V1219_NORMALIZER='multiframe',RDO_V1219_ADVANTAGE_MODE='loo_residual',
        RDO_V1219_UNMATCHED_MODE='sum_positive',RDO_V1219_CURVE_MODE='balanced',
        RDO_V1219_PRESERVATION_MULTIPLIER='1',RDO_V1219_ROLLOUTS_PER_CUBE='2',
        RDO_RL_GAMMA='1',RDO_RL_GAE_LAMBDA='1',RDO_RL_SHAPING_CHECKPOINTS='1')
    saved={key:os.environ.get(key) for key in settings}
    try:
        os.environ.update(settings)
        with redirect_stdout(io.StringIO()):
            ex = importlib.import_module('radar_sim.ppo_v12_19_pipeline')
    finally:
        for key,value in saved.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value
    return ex


@contextmanager
def report_directories(ex, root):
    """Change output destinations only; never replace legacy function hooks."""
    root = Path(root)
    old = (ex.OUTPUT_DIR_V11, ex.TABLE_DIR_V11, ex.FIG_DIR_V11, ex.MODEL_DIR_V11)
    ex.OUTPUT_DIR_V11, ex.TABLE_DIR_V11, ex.FIG_DIR_V11, ex.MODEL_DIR_V11 = (
        root, root/'tables', root/'figures', root/'models')
    for directory in (ex.TABLE_DIR_V11, ex.FIG_DIR_V11, ex.MODEL_DIR_V11):
        directory.mkdir(parents=True, exist_ok=True)
    try:
        yield
    finally:
        ex.OUTPUT_DIR_V11, ex.TABLE_DIR_V11, ex.FIG_DIR_V11, ex.MODEL_DIR_V11 = old


def measure(ex, original, post, labs, encoded, objective='perception_first'):
    components, detail = ex.radar_task_components_v11(original, post, labs)
    detail['cube_nmse'] = float(np.sum(encoded['distortion_contributions']))
    if objective == 'perception_first':
        terms = perception_terms(components, detail, encoded['meta']['realized_bpcs'])
    else:
        # Same uncapped allocation, original U2000 objective minus cap penalty.
        terms = preservation_weighted_reward_terms(components, detail, encoded['meta']['realized_bpcs'],
            preservation_multiplier=1., rate_cap=1e9)
        terms['total_objective'] = terms['total_loss']
        terms['task_loss'] = terms['total_loss']
        terms['rate_excess'] = 0.
    independent = -sum(float(terms[key]) for key in LOSS_COMPONENT_NAMES)
    if not np.isclose(independent, terms['terminal_label_first_reward'], rtol=1e-12, atol=1e-9):
        raise AssertionError('Reward components do not sum independently to the declared return.')
    return {**components, **detail, **terms}


def fit_normalizer(ex, rows, config, output):
    identity = ex.StateNormalizerV11()
    identity.mean = np.zeros(52, np.float32); identity.std = np.ones(52, np.float32)
    actor = ex.RadarPPOPolicyV11(52, len(ex.ACTION_LIBRARY_V11), seed=config.seed+101)
    actor.params['Wa'].fill(0.); actor.params['ba'].fill(0.)
    states, audits = [], []
    for row in ex.select_split_rows_v11(rows, min(config.normalizer_cubes,len(rows))).itertuples():
        cube = ex.load_cube_v6_1000(pd.Series(row._asdict()))
        encoded = encode_uncapped(ex, cube, actor, identity, True)
        states.append(encoded['states_raw'])
        audits.append(dict(record_name=row.record_name, uses_labels=False,
            uniform_full_library=True, bitrate_cap=None, state_count=len(encoded['states'])))
        del cube, encoded
    normalizer = ex.StateNormalizerV11().fit(np.concatenate(states))
    ex.safe_write_csv_v11(pd.DataFrame(audits), Path(output)/'tables/normalizer_TRAIN_audit_v12_21.csv')
    return normalizer


def evaluate_greedy(ex, rows, labels, actor, normalizer, config, processors):
    records, action_frames = [], []
    for _, row in rows.iterrows():
        cube = ex.load_cube_v6_1000(row)
        labs = ex.labels_for_cube_v6_1000(labels, row)
        processor = ex._postprocessor_for_cube_v11(cube, processors)
        original = ex.postprocess_cube(cube, processor)
        encoded = encode_uncapped(ex, cube, actor, normalizer, False)
        post = ex.postprocess_cube(encoded['cube_hat'], processor)
        record = measure(ex, original, post, labs, encoded, config.objective)
        records.append(dict(record_name=row.record_name, scene_name=row.scene_name,
            realized_bpcs=encoded['meta']['realized_bpcs'], **record))
        logs = encoded['block_log'].copy(); logs['record_name'] = row.record_name
        action_frames.append(logs)
        del cube, encoded, post, original
        gc.collect()
    return pd.DataFrame(records), pd.concat(action_frames, ignore_index=True)


def validation_summary(frame):
    return dict(low_misses=int(frame.low_threshold_labels_missed_total.sum()),
        sweep_misses=int(frame.true_labels_missed_total.sum()),
        mean_unmatched=float(frame.new_false_detections_total.mean()),
        mean_positive_unmatched_excess=float(np.maximum(frame.new_false_detections_total-frame.original_unmatched_peaks_total,0).mean()),
        actual_bpcs=float(frame.realized_bpcs.mean()),
        mean_fixed_validation_reward=float(frame.terminal_label_first_reward.mean()),
        mean_cube_nmse=float(frame.cube_nmse.mean()),
        mean_RD_log_power_mse=float(frame.RD_log_power_mse.mean()),
        mean_RA_log_power_mse=float(frame.RA_log_power_mse.mean()),
        mean_continuous_regret=float(frame.cfar_curve_regret.mean()))


def train(ex, train_rows, validation_rows, labels, config, output, *, normalizer=None, initial_state=None):
    """Only primary trajectories train the critic; branches supply actor Q only."""
    output = Path(output)
    for name in ('models','tables','figures'):
        (output/name).mkdir(parents=True, exist_ok=True)
    actor = ex.RadarPPOPolicyV11(52, len(ex.ACTION_LIBRARY_V11), seed=config.seed)
    if initial_state is not None:
        actor.load_state_dict(initial_state)
    initial_state = actor.state_dict()
    normalizer = fit_normalizer(ex,train_rows,config,output) if normalizer is None else normalizer
    optimizer = ex.PPOAdamV12_11(actor.params, config.learning_rate)
    processors, records, history, branches, credits, action_credits = {}, [], [], [], [], []
    updates, episodes, pending, packets = 0, 0, [], []
    started = time.monotonic()
    probe_rows = ex.select_split_rows_v11(validation_rows,min(config.validation_probe_cubes,len(validation_rows)))
    ex.safe_write_csv_v11(probe_rows, output/'tables/fixed_validation_population_v12_21.csv')
    ex.safe_save_npz_v11(output/'models/scratch_initial_policy_v12_21.npz', **initial_state)
    ex.safe_write_json_v11(normalizer.as_dict(), output/'models/state_normalizer_v12_21.json')
    ex.safe_write_json_v11(PerceptionObjective().as_dict() if config.objective=='perception_first' else
        dict(objective='old_U2000_balanced_control',rate_penalty=0.,bitrate_cap=None,
             reward_scale=5000.,NMSE_reward_weight=5000.,configuration=asdict(config)),
        output/'models/task_objective_v12_21.json')

    def checkpoint(pass_number):
        frame, actions = evaluate_greedy(ex,probe_rows,labels,actor,normalizer,config,processors)
        index = len(history)
        summary = validation_summary(frame)
        history.append(dict(**summary, checkpoint_id=index, episodes_seen=episodes,
            ppo_update_batches=updates, **{'pass':pass_number}, validation_low_threshold_misses_candidate=summary['low_misses'],
            mean_realized_bpcs=summary['actual_bpcs'], mean_false_detection_cost=summary['mean_unmatched'],
            mean_total_objective=float(frame.total_objective.mean()), mean_task_loss=float(frame.task_loss.mean()),
            mean_RD_RA_log_power_mse=summary['mean_RD_log_power_mse']+summary['mean_RA_log_power_mse'],
            selected_by='lowest misses then positive unmatched inflation then sweep misses; ORIGINAL only'))
        ex.safe_write_csv_v11(frame, output/f'tables/fixed_validation_checkpoint_{index:03d}_v12_21.csv')
        ex.safe_write_csv_v11(pd.DataFrame(history), output/'tables/fixed_validation_progress_v12_21.csv')
        ex.safe_save_npz_v11(output/f'models/validation_checkpoint_{index:03d}_v12_21.npz', **actor.state_dict())
        print(f"V12.21 fixed greedy validation checkpoint {index}: low_misses={summary['low_misses']} "
              f"sweep_misses={summary['sweep_misses']} unmatched={summary['mean_unmatched']:.2f} "
              f"bpcs={summary['actual_bpcs']:.3f} reward={summary['mean_fixed_validation_reward']:+.4f}",flush=True)
        return dict(key=checkpoint_key(frame), state=actor.state_dict(), candidate=frame,
            actions=actions, checkpoint_id=index, selected_pass=pass_number)

    best = checkpoint(0.)
    order_rng = np.random.default_rng(config.seed)
    selection_rng = np.random.default_rng(np.random.SeedSequence([config.seed,1221,77]))
    for pass_index in range(config.passes):
        order = order_rng.permutation(len(train_rows))
        for visit, position in enumerate(order,1):
            row = train_rows.iloc[int(position)]
            cube = ex.load_cube_v6_1000(row)
            labs = ex.labels_for_cube_v6_1000(labels,row)
            processor = ex._postprocessor_for_cube_v11(cube,processors)
            original = ex.postprocess_cube(cube,processor)
            source = ex.prepare_policy_source_v12_12(cube)
            first_row = sum(len(t['actions']) for t in pending)
            primary_for_branch = int(selection_rng.integers(config.rollouts))
            at_branch_cube = visit % config.batch_cubes == 1 or config.batch_cubes == 1
            prefixes,placement_audit = choose_training_prefixes(ex,original,labs,source,
                config.conditional_contexts_per_batch,selection_rng,row.record_name,row.scene_name,config.prefix_sampling) \
                if config.conditional_credit and at_branch_cube else ([],[])
            # A uniformly chosen TRAJECTORY supplies the actual on-policy
            # history. Query locations above depend only on original TRAIN
            # information, never the current action or its resulting loss.
            eligible_rows = []
            for sample in range(config.rollouts):
                encoded = encode_uncapped(ex,cube,actor,normalizer,True,source)
                post = ex.postprocess_cube(encoded['cube_hat'],processor)
                measured = measure(ex,original,post,labs,encoded,config.objective)
                encoded['rewards'][-1] = measured['terminal_label_first_reward']
                encoded.update(rollout_group=f'{pass_index}:{visit}:{row.record_name}',
                    rollout_index=sample, record_name=str(row.record_name))
                record = {**measured, 'record_name':row.record_name,'scene_name':row.scene_name,
                    'realized_bpcs':encoded['meta']['realized_bpcs'],
                    'episode_return':float(encoded['rewards'].sum()), 'rollout_index':sample,
                    'rollout_group':encoded['rollout_group'], 'cube_episode':episodes+1,
                    'episode_in_pass':visit, 'pass':pass_index+1,
                    'unique_actions':int(np.unique(encoded['actions']).size),
                    **ex.action_diagnostics_v12_9(encoded)}
                if sample == primary_for_branch:
                    eligible_rows = list(range(len(encoded['actions'])))
                    branch_primary = encoded
                    branch_flat_start = first_row+sample*len(encoded['actions'])
                # Avoid keeping reconstructed 30-MiB cubes in pending batches.
                encoded.pop('cube_hat')
                pending.append(encoded); records.append(record)
                del post
            # Predeclared amortized budget: branch on one cube per batch,
            # chosen by batch position, never its loss, label or current action.
            if config.conditional_credit and at_branch_cube:
                for tile in prefixes:
                    old = actor.forward(branch_primary['states'][tile:tile+1])[0][0].astype(float)
                    old /= old.sum()
                    q_values = []
                    for action in range(len(ex.ACTION_LIBRARY_V11)):
                        returns = []
                        for rep in range(config.branch_suffixes):
                            entropy = [config.seed,1221,pass_index,visit,tile,rep]
                            facade = ForcedAction(actor,branch_primary['actions'],tile,action,np.random.SeedSequence(entropy))
                            alternative = encode_uncapped(ex,cube,facade,normalizer,True,source)
                            assert_prefix_matches(branch_primary,alternative,tile,source['action_bits_by_block'])
                            post = ex.postprocess_cube(alternative['cube_hat'],processor)
                            result = measure(ex,original,post,labs,alternative,config.objective)
                            returns.append(result['terminal_label_first_reward'])
                            branches.append(dict(update=updates+1,record_name=row.record_name,
                                tile_index=tile,action=action,replicate=rep,
                                measurement_placement=config.prefix_sampling,
                                placement_reason=next(r['reason'] for r in placement_audit if r['tile_index']==tile),
                                old_probability=old[action],seed_entropy=json.dumps(entropy),
                                realized_bpcs=alternative['meta']['realized_bpcs'],**result))
                            del alternative,post
                        q_values.append(float(np.mean(returns)))
                    packets.append(dict(primary_row=branch_flat_start+tile,
                        q_values_unscaled=np.asarray(q_values),old_probabilities=old))
            episodes += 1
            batch_end = visit % config.batch_cubes == 0 or visit == len(order)
            if batch_end:
                stats = update_ppo_v12_21(actor,optimizer,pending,packets,
                    epochs=config.ppo_epochs,clip=.2,entropy_coef=.002,target_kl=.02,
                    value_coef=.5,reward_scale=1. if config.objective=='perception_first' else 5000.,
                    gradients_fn=ex._network_gradients_v11,mask_fn=ex._apply_action_mask_v11,
                    seed=config.seed,eps=ex.EPS)
                updates += 1
                # Observe the actual shared update, not just the desired sign
                # of a single row's advantage. This does not change gradients.
                flat=np.concatenate([t['states'] for t in pending])
                after=actor.forward(flat)[0]
                for item in stats['conditional_action_rows']:
                    item['post_update_probability']=float(after[item['primary_row'],item['action']])
                    item['probability_change']=item['post_update_probability']-item['old_probability']
                credits.extend(dict(update=updates,**r) for r in stats.pop('credit_rows'))
                action_credits.extend(dict(update=updates,**r) for r in stats.pop('conditional_action_rows'))
                for record in records[-len(pending):]:
                    record.update(stats,update=updates,mean_cube_nmse=float(np.mean([t['meta']['cube_nmse'] for t in pending])))
                pending,packets = [],[]
                ex.safe_write_csv_v11(pd.DataFrame(records),output/'tables/ppo_training_episodes_v12_21.csv')
                ex.safe_write_csv_v11(pd.DataFrame(branches),output/'tables/conditional_action_outcomes_v12_21.csv')
                ex.safe_write_csv_v11(pd.DataFrame(credits),output/'tables/actor_credit_rows_v12_21.csv')
                ex.safe_write_csv_v11(pd.DataFrame(action_credits),output/'tables/conditional_action_credit_v12_21.csv')
                ex.safe_save_npz_v11(output/'models/latest_policy_v12_21.npz',**actor.state_dict())
                if config.validation_every_updates and updates % config.validation_every_updates == 0 and visit < len(order):
                    candidate = checkpoint(pass_index+visit/len(order))
                    if candidate['key'] < best['key']:
                        best = candidate
            last = records[-config.rollouts:]
            print(f"V12.21 pass {pass_index+1}/{config.passes} cube {visit}/{len(order)} | "
                f"stochastic low_miss={np.mean([r['low_threshold_labels_missed_total'] for r in last]):.2f} "
                f"unmatched={np.mean([r['new_false_detections_total'] for r in last]):.1f} "
                f"bpcs={np.mean([r['realized_bpcs'] for r in last]):.3f} | no cap",flush=True)
            del cube,source,original
            gc.collect()
        candidate = checkpoint(pass_index+1.)
        if candidate['key'] < best['key']:
            best = candidate
    ex.safe_save_npz_v11(output/'models/last_policy_v12_21.npz',**actor.state_dict())
    actor.load_state_dict(best['state'])
    ex.safe_save_npz_v11(output/'models/selected_policy_v12_21.npz',**actor.state_dict())
    ex.safe_write_json_v11(dict(checkpoint_id=best['checkpoint_id'],pass_number=best['selected_pass'],
        validation_lowest_misses=int(best['candidate'].low_threshold_labels_missed_total.sum()),
        selected_without_RDO=True, optimizer_resumable=False),output/'models/selected_checkpoint_v12_21.json')
    learned = dict(policy=actor,normalizer=normalizer,scaler=PerceptionObjective(),
        training_log=pd.DataFrame(records),validation_checkpoint_history=pd.DataFrame(history),
        validation_candidate=best['candidate'],validation_actions=best['actions'],
        scratch_initial_state=initial_state,postprocessors=processors,
        elapsed_seconds=time.monotonic()-started,selected_checkpoint=best['checkpoint_id'],
        main_imitation_epochs=0,refinement_used=False,rate_cap=None)
    return learned
