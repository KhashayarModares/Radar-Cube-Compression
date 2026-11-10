"""Bounded V12.22 physical-menu and selected-policy diagnostics.

Interventions never alter a deployed policy, its menu, objective or checkpoint.
All comparisons use existing TRAIN/VALIDATION rosters; TEST is never accessed.
One-tile results are contextual measurements, not full-sequence guarantees.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import ppo_v12_22_codec as codec
from . import ppo_v12_22_storage as store
from .ppo_prefix_rollouts import assert_prefix_matches
from .ppo_v12_19_diagnostics import _label_rows, tile_error_decomposition


def _menu_probes(source, cube_size):
    """Four preregistered complete-cube allocations; no labels or detector input."""
    bits = np.asarray(source['action_bits_by_block'], np.float64)
    sse = np.asarray(source['action_distortion_by_block'], np.float64)
    if bits.shape != sse.shape or bits.ndim != 2:
        raise ValueError('Per-tile, per-action physical descriptors are required.')
    relative_bits = bits / (32. * int(cube_size))
    return dict(min_bits=np.argmin(bits, axis=1), min_signal_sse=np.argmin(sse, axis=1),
                signal_sse_plus_001_rate=np.argmin(sse + .01 * relative_bits, axis=1),
                signal_sse_plus_010_rate=np.argmin(sse + .10 * relative_bits, axis=1))


def _native_labels(runtime, row):
    return runtime.ex.labels_for_cube_v6_1000(runtime.labels, row)


def _save_case(folder, outcome, labels=None, peaks=None, *, contract):
    folder = Path(folder)
    files = [folder / 'outcome.csv']
    store.write_csv(files[0], [outcome])
    for name, frame in (('labels', labels), ('peaks', peaks)):
        if frame is not None:
            files.append(folder / (name + '.csv'))
            # A stable column avoids pandas' EmptyDataError when no GT supports.
            if not len(frame.columns):
                frame = pd.DataFrame(columns=['record_name'])
            store.write_csv(files[-1], frame)
    store.receipt(folder / 'COMPLETE.json', files, contract=store.digest(contract))


def _case_exists(folder, contract):
    path = Path(folder) / 'COMPLETE.json'
    if not path.exists():
        return False
    saved = store.verify_receipt(path)
    if saved.get('contract') != store.digest(contract):
        raise ValueError('Diagnostic inputs changed for a committed case: ' + str(folder))
    return True


def _merge_cases(folders, name):
    frames = []
    for folder in folders:
        file = Path(folder) / (name + '.csv')
        if file.exists():
            frames.append(pd.read_csv(file))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=['record_name'])


def physical_menu_checks(runtime):
    """48 cubes x three menus x four fixed label-free allocations in full mode.

    Every probe reconstructs the entire cube. It is a physical feasibility
    diagnostic, not an oracle, training allocation or action-pruning procedure.
    Each completed case is committed separately and skipped on resume.
    """
    root = Path(runtime.root) / 'diagnostics' / 'physical_menu'
    populations = [('TRAIN', runtime.train_probe), ('VALIDATION', runtime.dev_probe)]
    if runtime.manifest['profile'] == 'full' and (len(runtime.train_probe), len(runtime.dev_probe)) != (16, 32):
        raise ValueError('Full physical-menu diagnostics require the frozen 16 TRAIN / 32 VALIDATION probes.')
    menus = dict(original16=codec.ORIGINAL_ACTIONS, compact16=codec.PROPOSED_ACTIONS[:16],
                 proposed21=codec.PROPOSED_ACTIONS)
    folders, newly_evaluated = [], 0
    for split, population in populations:
        for _, row in population.iterrows():
            print(f'V12.22 physical-menu check {split}: {row.record_name}',flush=True)
            cube = runtime.load_cube(row)
            original = runtime.original(row, cube)
            labs = _native_labels(runtime, row)
            for menu_name, actions in menus.items():
                source = codec.prepare_source(runtime.ex, cube, actions)
                for probe, sequence in _menu_probes(source, cube.size).items():
                    contract = dict(split=split, record_name=str(row.record_name), menu=menu_name,
                                    actions=actions, probe=probe, sequence=sequence.tolist())
                    folder = root / 'cases' / store.digest(contract)[:20]
                    folders.append(folder)
                    if _case_exists(folder, contract):
                        continue
                    encoded = codec.physical.encode_sequence(cube, sequence, source=source)
                    post = runtime.post(encoded['cube_hat'], row, 'physical_menu_check',
                                        menu=menu_name, probe=probe, split=split)
                    metrics = runtime.measure(original, post, labs, encoded)
                    info = dict(split=split, record_name=str(row.record_name), scene_name=str(row.scene_name),
                                menu=menu_name, action_count=len(actions), probe=probe,
                                chosen_action_count=int(np.unique(sequence).size),
                                complete_cube_reconstruction=True, labels_used_for_allocation=False,
                                diagnostic_only=True, **metrics)
                    _save_case(folder, info, contract=contract)
                    newly_evaluated += 1
    frame = _merge_cases(folders, 'outcome')
    table = root / 'PHYSICAL_MENU_OUTCOMES.csv'
    store.write_csv(table, frame)
    result = dict(completed=True, rows=len(frame), expected_rows=len(folders),
                  newly_evaluated=newly_evaluated, profile=runtime.manifest['profile'],
                  full_protocol_rows=576, complete_cube_oracle_claim=False,
                  diagnostic_only=True, TEST_evaluated=False)
    store.receipt(root / 'STATUS.json', [table], **{k: v for k, v in result.items() if k != 'completed'})
    return result


def _suspect_tile(source, original, labels, sequence):
    """Choose one diagnostic tile; association is nominal, not exact influence."""
    if len(labels):
        rd = labels.loc[labels.map_type == 'RD'].copy()
        if len(rd):
            rd = rd.sort_values(['missed_low', 'lost_original_support_db', 'original_score'],
                                ascending=[False, False, True], kind='stable')
            for label in rd.to_dict('records'):
                r, v = float(label['original_peak_range_m']), float(label['original_peak_second_axis'])
                if not np.isfinite([r, v]).all():
                    continue
                ri = int(np.argmin(abs(np.asarray(original['ranges']) - r)))
                vi = int(np.argmin(abs(np.asarray(original['vel']) - v)))
                for tile, slc in enumerate(source['slices']):
                    if slc[1].start <= vi < slc[1].stop and slc[2].start <= ri < slc[2].stop:
                        return tile, 'original-supported_RD_peak; nominal_bin_not_exact_damage_footprint'
    # Label-free fallback when no original-supported RD object lies in ROI.
    damage = [prepared['action_distortion'][int(action)]
              for prepared, action in zip(source['prepared_blocks'], sequence)]
    return int(np.argmax(damage)), 'largest_spectral_SSE; no_supported_RD_peak'


def _noise(detector, r, c, eps):
    score, power = float(detector['cfar_score'][r, c]), float(detector['power'][r, c])
    return max(power / score - eps, 0.) if score > 0 else np.nan


def _unmatched_peak_rows(ex, original, method, labels, maximum=5):
    """Highest-confidence outside-GT-gate peaks with same-position Original data.

    Outside annotated gates means unmatched, not proven physical false alarms.
    Missing annotations and real clutter can produce legitimate unmatched peaks.
    """
    rows = []
    for kind, gate in (('RD', ex.CFAR_LABEL_GATE_RD_V8), ('RA', ex.CFAR_LABEL_GATE_RA_V8)):
        p0, ranges, secondary = ex._map_triplet_v12_6(original, kind)
        p1, ranges1, secondary1 = ex._map_triplet_v12_6(method, kind)
        if not np.array_equal(ranges, ranges1) or not np.array_equal(secondary, secondary1):
            raise AssertionError('Original and method physical diagnostic axes differ.')
        detectors = [ex.precompute_nested_cfar_peak_masks_v11_2(ex.as_linear_power(p), [0.])[0.]
                     for p in (p0, p1)]
        first, second = detectors
        occupied = np.zeros(np.shape(second['peaks']), bool)
        bins, _, _ = ex._evaluable_label_bins_v12(labels, ranges, secondary, kind)
        for r, c in bins:
            r, c = int(r), int(c)
            occupied[max(0, r-gate[0]):r+gate[0]+1, max(0, c-gate[1]):c+gate[1]+1] = True
        valid = (np.asarray(second['peaks'], bool) & np.asarray(second['valid'], bool) & ~occupied)
        positions = np.argwhere(valid)
        if len(positions):
            scores = second['cfar_score'][positions[:, 0], positions[:, 1]]
            positions = positions[np.argsort(-scores, kind='stable')[:int(maximum)]]
        for rank, (r, c) in enumerate(positions):
            r, c = int(r), int(c)
            row = dict(map_type=kind, confidence_rank=rank + 1, map_row=r, map_col=c,
                       range_m=float(ranges[r]), second_axis=float(secondary[c]),
                       outside_annotated_gates=True, confirmed_false_alarm=False)
            for prefix, detector in (('original_same_position', first), ('method', second)):
                row.update({prefix+'_power': float(detector['power'][r, c]),
                            prefix+'_score': float(detector['cfar_score'][r, c]),
                            prefix+'_CA_CFAR_background': _noise(detector, r, c, float(ex.EPS)),
                            prefix+'_fixed_NMS_peak': bool(detector['peaks'][r, c])})
            rows.append(row)
    return pd.DataFrame(rows)


def _annotate(frame, context):
    frame = frame.copy()
    for name, value in context.items():
        frame[name] = value
    return frame


def selected_policy_diagnostics(runtime):
    """Four predetermined cubes; all actions; fixed others plus four fresh suffixes.

    Full: 4,840 action-intervention reconstructions plus 48 explicit greedy
    baseline reconstructions. Tiny engineering profile uses one suffix instead.
    Nothing is selected for deployment from any diagnostic outcome.
    """
    root = Path(runtime.root) / 'diagnostics' / 'selected_policy'
    rows = [('TRAIN', row) for _, row in runtime.train_probe.head(2).iterrows()]
    rows += [('VALIDATION', row) for _, row in runtime.dev_probe.head(2).iterrows()]
    if len(rows) != 4:
        raise ValueError('Diagnostics require the first two predefined TRAIN and two VALIDATION probes.')
    replicas = 1 if runtime.manifest['profile'] == 'tiny' else 4
    all_cases, baselines, evaluations = [], [], 0
    # Validate EVERY selected job before any physical reconstruction. Tiny
    # policies are cheap to retain; do not partly diagnose A while B is active.
    jobs = {(arm, int(seed)): runtime.load_job(arm, int(seed), selected=True)
            for arm in runtime.manifest['arms'] for seed in runtime.manifest['seeds']}
    for arm in runtime.manifest['arms']:
        for seed in runtime.manifest['seeds']:
            policy, norm, actions, cfar_enabled, jobpath = jobs[(arm, int(seed))]
            actions = tuple(tuple(v) for v in actions)
            policy_hash = store.actor_hash(policy)
            for split, row in rows:
                cube = runtime.load_cube(row)
                original = runtime.original(row, cube)
                labs = _native_labels(runtime, row)
                source = codec.prepare_source(runtime.ex, cube, actions)
                features = runtime.features(row, original, source) if cfar_enabled else None
                common = dict(arm=arm, seed=int(seed), split=split,
                              record_name=str(row.record_name), scene_name=str(row.scene_name),
                              policy_hash=policy_hash, normalizer_hash=store.digest(norm.as_dict()),
                              menu_hash=store.digest(actions), selected_job=str(jobpath))
                folder = root / 'baselines' / store.digest(common)[:20]
                # Baseline encoding is cheap relative to detector evaluation;
                # encode afresh to obtain exact states for prefix identity tests.
                base = codec.encode(runtime.ex, cube, policy, norm, False, source, features)
                baselines.append(folder)
                if _case_exists(folder, common):
                    base_labels = pd.read_csv(folder / 'labels.csv')
                else:
                    post = runtime.post(base['cube_hat'], row, 'diagnostic_baseline', arm=arm, seed=int(seed))
                    base_labels = _label_rows(runtime.ex, original, post, labs, str(row.record_name), str(row.scene_name))
                    base_peaks = _unmatched_peak_rows(runtime.ex, original, post, labs)
                    _save_case(folder, dict(**common, **runtime.measure(original, post, labs, base)),
                               _annotate(base_labels, common), _annotate(base_peaks, common), contract=common)
                    evaluations += 1
                tile, reason = _suspect_tile(source, original, base_labels, base['actions'])
                slc = source['slices'][tile]
                prefix_hash = store.digest(base['actions'][:tile])
                for mode in ('fixed_allocation', 'fresh_stochastic_suffix'):
                    for replicate in range(1 if mode == 'fixed_allocation' else replicas):
                        entropy = [int(seed), int(store.digest(str(row.record_name))[:8], 16), tile, replicate, 1222]
                        suffix_seed = int(np.random.SeedSequence(entropy).generate_state(1)[0])
                        for action, (rho, q) in enumerate(actions):
                            context = dict(**common, intervention_mode=mode, replicate=replicate,
                                tile_index=tile, tile_selection=reason, action_index=action, rho=float(rho), q=int(q),
                                baseline_action=int(base['actions'][tile]),
                                baseline_action_probability=float(base['action_probabilities'][tile, action]),
                                suffix_seed=suffix_seed, prefix_actions_hash=prefix_hash,
                                doppler_start=slc[1].start, doppler_stop=slc[1].stop,
                                range_frequency_start=slc[2].start, range_frequency_stop=slc[2].stop,
                                diagnostic_only=True, deployment_refinement=False)
                            case = root / 'cases' / store.digest(context)[:24]
                            all_cases.append(case)
                            if _case_exists(case, context):
                                continue
                            if mode == 'fixed_allocation':
                                sequence = base['actions'].copy()
                                sequence[tile] = action
                                other = np.arange(len(sequence)) != tile
                                if not np.array_equal(sequence[other], base['actions'][other]):
                                    raise AssertionError('A fixed-allocation intervention changed another tile.')
                                enc = codec.physical.encode_sequence(cube, sequence, source=source)
                                prefix_verified = True
                            else:
                                facade = codec.ForcedAction(policy, base['actions'], tile, action, suffix_seed)
                                enc = codec.encode(runtime.ex, cube, facade, norm, True, source, features)
                                assert_prefix_matches(base, enc, tile, source['action_bits_by_block'])
                                prefix_verified = True
                            post = runtime.post(enc['cube_hat'], row, 'selected_policy_intervention',
                                                arm=arm, seed=int(seed), mode=mode,
                                                action=action, replicate=replicate)
                            metrics = runtime.measure(original, post, labs, enc)
                            decomposition = tile_error_decomposition(runtime.ex.coherent_codec_v14,
                                source['prepared_blocks'][tile], actions[action], source['whole_cube_energy'])
                            labels = _label_rows(runtime.ex, original, post, labs, str(row.record_name), str(row.scene_name))
                            peaks = _unmatched_peak_rows(runtime.ex, original, post, labs)
                            outcome = dict(**context, **metrics, **decomposition,
                                           same_predecision_prefix_verified=prefix_verified,
                                           all_other_greedy_decisions_fixed=mode == 'fixed_allocation')
                            _save_case(case, outcome, _annotate(labels, context), _annotate(peaks, context), contract=context)
                            evaluations += 1
    artifacts = []
    for name, cases, table_name in (
        ('outcome', all_cases, 'ACTION_INTERVENTIONS.csv'),
        ('labels', all_cases, 'LABEL_PEAK_NOISE.csv'),
        ('peaks', all_cases, 'UNMATCHED_PEAK_NOISE.csv'),
        ('outcome', baselines, 'GREEDY_DIAGNOSTIC_BASELINES.csv')):
        path = root / table_name
        store.write_csv(path, _merge_cases(cases, name))
        artifacts.append(path)
    result = dict(completed=True, intervention_cases=len(all_cases), greedy_baseline_cases=len(baselines),
                  suffix_replicates=replicas, newly_evaluated=evaluations,
                  full_protocol_intervention_cases=4840, full_protocol_additional_baselines=48,
                  diagnostics_not_deployment_refinement=True, TEST_evaluated=False)
    store.receipt(root / 'STATUS.json', artifacts, **{k: v for k, v in result.items() if k != 'completed'})
    return result
