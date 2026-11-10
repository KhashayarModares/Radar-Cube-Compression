"""Runnable V12.21 experiment and publication reports; never runs on import."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd

from radar_sim.ppo_v12_21_pipeline import (ROOT, REVISION, RunConfig, load_primitives,
    report_directories, train, measure, validation_summary)
from radar_sim.ppo_v12_21_codec import encode_uncapped
from radar_sim.ppo_v12_21_reporting import (collect_detection_frame,
    export_detection_reports, plot_percentage_rate_reports)
from radar_sim.ppo_v12_19_rate_sweep import summarize_operating_points, FFT_BITS


def preserve_baseline(ex, train_rows, labels, processors, output):
    """Freeze the prior TRAIN-calibrated RDO; PPO cannot alter this baseline."""
    reference = Path(os.environ.get('RDO_V1221_RDO_REFERENCE', str(ROOT/
        'outputs/v12_19_v44_preservation_balanced_ppo_full_train600_val200_test200/tables/task_rdo_action_summary_v12.csv')))
    if reference.is_file():
        table = pd.read_csv(reference)
        chosen = table.loc[table.selected_fixed_action.astype(str).str.lower().eq('true')]
        if len(chosen) != 1:
            raise ValueError('The frozen RDO reference must select exactly one action.')
        row = chosen.iloc[0]; index = int(row.action_index)
        action = tuple(ex.ACTION_LIBRARY_V11[index])
        if not np.allclose(action,[row.rho,row.q]):
            raise AssertionError('Frozen RDO action does not match the unchanged codec library.')
        ex.safe_write_csv_v11(table,Path(output)/'tables/frozen_RDO_action_summary_v12_21.csv')
        contract = dict(source=str(reference),sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
            selected_action=index,action=list(action),PPO_training_dependency=False,
            protocol='Frozen prior TRAIN-only RDO calibration, including its historical 4.8-bpcs feasibility rule.',
            comparison_rate_matched=False,PPO_rate_cap=None)
    else:
        # Fresh machines can reproduce the historical independent baseline;
        # calibration is after PPO training and never supplies its reward.
        with report_directories(ex,output):
            calibration=ex.calibrate_task_rdo_v11(ex.select_split_rows_v11(train_rows,min(15,len(train_rows))),labels,processors)
        action=tuple(calibration['selected_action'])
        contract=dict(source='new TRAIN-only historical RDO calibration',action=list(action),
            PPO_training_dependency=False,comparison_rate_matched=False,PPO_rate_cap=None)
    ex.safe_write_json_v11(contract,Path(output)/'tables/independent_RDO_protocol_v12_21.json')
    return action


def canonicalize_plot_names(ex, directory):
    """One run/version label; retain an explicit legacy-function filename map."""
    directory=Path(directory)
    mapping=[]
    for path in sorted(directory.rglob('*.png')):
        name=re.sub(r'_v12_\d+(?:_\d+)?(?=\.png$)','_v12_21',path.name)
        target=path.with_name(name)
        if target != path:
            if target.exists():
                raise FileExistsError('Two distinct figures map to the same canonical filename: '+str(target))
            path.rename(target)
            mapping.append(dict(library_function_filename=str(path),run_figure_filename=str(target),run_revision=REVISION))
    if mapping:
        ex.safe_write_csv_v11(pd.DataFrame(mapping),directory/'figure_name_provenance_v12_21.csv')


def summarize_conditional_credit(table):
    """Report probability MASS of all tied-best actions, not an arbitrary ID.

    Q ties use a numerical-only tolerance (rtol=atol=1e-12).  An unstable
    sort must not change the diagnostic when physically equivalent actions
    have identical measured returns.  This summary never affects training.
    """
    rows=[]
    for (update,primary_row),group in table.groupby(['update','primary_row'],sort=True):
        feasible=group.feasible.astype(str).str.lower().eq('true')
        group=group.loc[feasible].copy()
        q=group.q_value_unscaled.to_numpy(dtype=float)
        if not len(q) or not np.isfinite(q).all():
            raise ValueError('Conditional credit report needs finite feasible Q estimates.')
        best=group.loc[np.isclose(q,q.max(),rtol=1e-12,atol=1e-12)].sort_values('action')
        before=float(best.old_probability.sum())
        after=float(best.post_update_probability.sum())
        rows.append(dict(update=int(update),primary_row=int(primary_row),
            estimated_best_actions='|'.join(str(int(a)) for a in best.action),
            tied_best_count=len(best),q_range=float(q.max()-q.min()),
            old_best_set_probability=before,post_best_set_probability=after,
            best_set_probability_change=after-before))
    return pd.DataFrame(rows)


def plot_learning(ex,learned,output):
    import matplotlib.pyplot as plt
    d=learned['training_log'];h=learned['validation_checkpoint_history']
    updates=d.drop_duplicates('optimizer_step_after')
    paths=[]
    specs=[('low_threshold_labels_missed_total','Lowest-threshold missed object/map instances'),
           ('new_false_detections_total','Unmatched RD+RA peaks across six thresholds'),
           ('episode_return','Perception-first reward (higher is better)'),
           ('value_loss','Critic MC mean-square error'),
           ('approximate_kl','Categorical KL to behavior policy'),
           ('realized_bpcs','Actual accounted bits / complex sample; no cap')]
    for fixed in (False,True):
        fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
        if not fixed:
            for ax,(key,title) in zip(axes.flat,specs):
                table=updates if key in ('value_loss','approximate_kl') else d
                x=table['update'] if key in ('value_loss','approximate_kl') else np.arange(1,len(table)+1)
                ax.plot(x,table[key],lw=.9,marker='.' if key in ('value_loss','approximate_kl') else None)
                if key=='new_false_detections_total':
                    ax.plot(x,table.original_unmatched_peaks_total,color='black',lw=.7,label='Original reference');ax.legend(fontsize=8)
                ax.set(title=title,xlabel='PPO update' if key in ('value_loss','approximate_kl') else 'Stochastic TRAIN rollout')
                ax.grid(alpha=.2)
            fig.suptitle('V12.21 raw stochastic training; failures remain visible; no forced monotonicity')
            path=Path(output)/'figures/ppo_training_diagnostics_v12_21.png'
        else:
            columns=[('low_misses','Lowest-threshold misses; fixed validation population'),
                     ('mean_unmatched','Unmatched peaks / cube; fixed validation'),
                     ('mean_fixed_validation_reward','Greedy validation reward'),
                     ('sweep_misses','Six-threshold missed object/map instances'),
                     ('actual_bpcs','Actual accounted bpcs; no cap'),
                     ('mean_cube_nmse','Cube NMSE; reported, not rewarded')]
            for ax,(key,title) in zip(axes.flat,columns):
                ax.plot(h.ppo_update_batches,h[key],'-o')
                selected=h.loc[h.checkpoint_id.eq(learned['selected_checkpoint'])]
                ax.scatter(selected.ppo_update_batches,selected[key],s=100,marker='*',color='orange',zorder=5)
                ax.set(title=title,xlabel='PPO update batch');ax.grid(alpha=.2)
            fig.suptitle('V12.21 deterministic greedy validation; orange star = selected checkpoint')
            path=Path(output)/'figures/fixed_validation_learning_v12_21.png'
        ex.save_publication_figure(fig,path);plt.close(fig);paths.append(path)
    fig,axes=plt.subplots(1,3,figsize=(14,4.2),layout='constrained')
    for ax,key,title in zip(axes,('actor_loss','value_loss','policy_entropy'),
            ('Clipped actor loss; not a monotone task score','Critic MC loss','Policy entropy')):
        ax.plot(updates['update'],updates[key],'-o',ms=3);ax.set(title=title,xlabel='PPO update');ax.grid(alpha=.2)
    path=Path(output)/'figures/ppo_losses_and_entropy_v12_21.png'
    ex.save_publication_figure(fig,path);plt.close(fig);paths.append(path)
    fig,axes=plt.subplots(1,3,figsize=(14,4.2),layout='constrained')
    for ax,key,label in zip(axes,('rate_spectral_correlation','q_spectral_correlation','rho_spectral_correlation'),
            ('Allocated rate','Quantizer bits','Retained fraction')):
        ax.plot(np.arange(1,len(d)+1),d[key],lw=.8)
        ax.axhline(0,color='black',lw=.7);ax.set_ylim(-1.05,1.05)
        ax.set(title=label+' vs spectral concentration',xlabel='TRAIN rollout',ylabel='Physical-quantity correlation')
        ax.grid(alpha=.2)
    fig.suptitle('Content adaptation; numeric action IDs are not used as ordered quantities')
    path=Path(output)/'figures/ppo_physical_action_adaptivity_v12_21.png'
    ex.save_publication_figure(fig,path);plt.close(fig);paths.append(path)
    credit_path=Path(output)/'tables/conditional_action_credit_v12_21.csv'
    if credit_path.is_file() and credit_path.stat().st_size>5:
        table=pd.read_csv(credit_path)
        if len(table) and 'post_update_probability' in table:
            best=summarize_conditional_credit(table)
            best.to_csv(Path(output)/'tables/conditional_best_action_set_audit_v12_21.csv',index=False)
            fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
            x=np.arange(1,len(best)+1)
            axes[0].bar(x,best.q_range.to_numpy(),color='#4c78a8')
            axes[0].set(title='Measured action-return contrast',xlabel='TRAIN query context',ylabel='Max Q estimate minus min Q estimate')
            change=100*best.best_set_probability_change.to_numpy()
            axes[1].bar(x,change,color=np.where(change>=0,'#2a9d55','#c44e52'))
            axes[1].axhline(0,color='black',lw=1)
            if len(best)<=12:
                for i,row in enumerate(best.itertuples()):
                    axes[1].annotate(f'{100*row.old_best_set_probability:.2f}% → {100*row.post_best_set_probability:.2f}%',
                        (x[i],change[i]),xytext=(0,8 if change[i]>=0 else -16),
                        textcoords='offset points',ha='center',fontsize=8)
            axes[1].set(title='Does the shared update favor the estimated best action set?',
                xlabel='TRAIN query context (different states; not a learning curve)',
                ylabel='Tied-best action probability mass change [percentage points]')
            axes[1].margins(y=.3)
            for ax in axes:ax.grid(alpha=.2)
            fig.suptitle('Conditional credit audit; finite noisy Q estimates, not a guaranteed oracle ranking')
            path=Path(output)/'figures/ppo_conditional_credit_audit_v12_21.png'
            ex.save_publication_figure(fig,path);plt.close(fig);paths.append(path)
    return paths


def plot_cfar(ex,sweep,split_name):
    """Same integer metrics; never shade an alignment interval that does not exist."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    base=ex.aggregate_v12_7_sweep(sweep,split_name)
    extended=ex.aggregate_v12_7_sweep(sweep,split_name,include_extended_rd=True)
    paths=[]
    for kind in ('RD','RA'):
        for metric,ylabel,stem in (
            ('preserved_original_supported','Preserved original-supported object instances [count]','preserved_true_labels'),
            ('unmatched_false_peaks','Peaks outside all annotation gates [count]','unmatched_false_peaks')):
            for methods,prefix in ((['Original','PPO'],'original_vs_PPO'),(ex.V12_7_METHODS,'four_policy')):
                frame=extended if kind=='RD' and metric=='preserved_original_supported' and len(methods)>2 else base
                frame=frame.loc[frame.map_type.eq(kind)]
                fig,ax=plt.subplots(figsize=(10,6),layout='constrained')
                for method in methods:
                    points=frame.loc[frame.method.eq(method)].sort_values('cfar_threshold_db')
                    ax.plot(points.cfar_threshold_db,points[metric],'-o',ms=3,lw=2 if method=='PPO' else 1.4,
                        color=ex.V12_7_COLORS[method],label=ex.V12_7_DISPLAY[method])
                if metric=='preserved_original_supported' and len(methods)==2:
                    original=frame.loc[frame.method.eq('Original')].sort_values('cfar_threshold_db')
                    ppo=frame.loc[frame.method.eq('PPO')].sort_values('cfar_threshold_db')
                    difference=original[metric].to_numpy()-ppo[metric].to_numpy()
                    first=np.flatnonzero(difference!=0)
                    aligned=int(first[0]) if len(first) else len(difference)
                    if aligned:
                        ax.axvspan(float(original.cfar_threshold_db.iloc[0]),float(original.cfar_threshold_db.iloc[aligned-1]),
                            color=ex.V12_7_COLORS['PPO'],alpha=.08,label='Contiguous exact low-threshold alignment')
                    else:
                        ax.text(.02,.04,'PPO is not aligned at the lowest threshold',transform=ax.transAxes,fontsize=9,color='darkred')
                ax.set(title=f'{kind}: Original vs PPO' if len(methods)==2 else f'{kind}: four independent policies',
                    xlabel='CA-CFAR threshold [dB above local background]',ylabel=ylabel)
                ax.yaxis.set_major_locator(MaxNLocator(integer=True));ax.grid(alpha=.2);ax.legend(fontsize=8)
                path=ex.FIG_DIR_V11/f'{kind}_{prefix}_{stem}_vs_cfar_threshold_v12_21.png'
                ex.save_publication_figure(fig,path);plt.close(fig);paths.append(path)
    return paths


def evaluate_split(ex,rows,labels,learned,fixed_action,output,split_name):
    """Same raw detector/labels for Original, uncapped PPO, frozen RDO and FFT14."""
    output=Path(output);sweeps=[];quality=[];label_audit=[];frames=[];perception=[];visuals=[];peak_diagnostics=[];action_rows=[]
    seen=set()
    with report_directories(ex,output):
        for ci,(_,row) in enumerate(rows.iterrows()):
            cube=ex.load_cube_v6_1000(row);labs=ex.labels_for_cube_v6_1000(labels,row)
            processor=ex._postprocessor_for_cube_v11(cube,learned['postprocessors'])
            original=ex.postprocess_cube(cube,processor)
            encoded=encode_uncapped(ex,cube,learned['policy'],learned['normalizer'],False)
            action_rows.append(encoded['block_log'].assign(record_name=row.record_name,scene_name=row.scene_name))
            rdo,_,rmeta=ex.compress_uniform_action_v11(cube,fixed_action)
            fft,_,fmeta=ex.compress_cube_fft_full_support_multi(cube,{'FFT14':14})['FFT14']
            methods={'Original':(cube,dict(realized_bits=32*cube.size,realized_bpcs=32.),original),
                'PPO':(encoded['cube_hat'],encoded['meta'],ex.postprocess_cube(encoded['cube_hat'],processor)),
                'CalibratedRDO':(rdo,rmeta,ex.postprocess_cube(rdo,processor)),
                'FFT14':(fft,fmeta,ex.postprocess_cube(fft,processor))}
            perception.append(dict(record_name=row.record_name,scene_name=row.scene_name,
                realized_bpcs=encoded['meta']['realized_bpcs'],
                **measure(ex,original,methods['PPO'][2],labs,encoded)))
            from radar_sim.ppo_v12_19_diagnostics import _label_rows
            peak_diagnostics.append(_label_rows(ex,original,methods['PPO'][2],labs,str(row.record_name),str(row.scene_name)))
            for method,(decoded,meta,post) in methods.items():
                quality.append(ex._cube_quality_row_v12_6(row.record_name,row.scene_name,method,cube,decoded,meta,original,post))
            for kind in ('RD','RA'):
                map0,ranges,second=ex._map_triplet_v12_6(original,kind)
                gate=ex.CFAR_LABEL_GATE_RD_V8 if kind=='RD' else ex.CFAR_LABEL_GATE_RA_V8
                factors=ex.V12_7_RD_PRESERVATION_THRESHOLD_FACTORS if kind=='RD' else ex.V12_7_BASE_CFAR_THRESHOLD_FACTORS
                cache=ex.precompute_original_cfar_count_sweep_v11_2(map0,labs,ranges,second,kind,factors,gate)
                for method,(_,meta,post) in methods.items():
                    data,_,_=ex._map_triplet_v12_6(post,kind)
                    ex._append_label_audit_v12_6(label_audit,cache,data,labs,method,row.record_name,row.scene_name)
                    counts=ex.evaluate_cfar_count_sweep_method_v11_2(cache,data)
                    for item in counts.itertuples():
                        sweeps.append(dict(cube_idx=ci,record_name=row.record_name,scene_name=row.scene_name,
                            map_type=kind,method=method,method_display=ex.V12_7_DISPLAY[method],
                            threshold_factor=item.threshold_factor,cfar_threshold_db=item.cfar_threshold_db,
                            eligible_original=int(item.num_labels_supported_original),
                            preserved_original_supported=int(item.num_labels_preserved_from_original),
                            missed_original_supported=int(item.num_labels_missed_from_original),
                            unmatched_false_peaks=int(item.num_label_unmatched_peaks_method),
                            num_cfar_peaks=int(item.num_cfar_peaks_method),realized_bpcs=meta['realized_bpcs']))
                    frames.append(collect_detection_frame(ex,post,labs,kind,row.record_name,method))
            if str(row.scene_name) not in seen:
                seen.add(str(row.scene_name))
                visual=dict(row=row.copy(),labs=labs.copy(),methods=methods)
                ex.plot_v12_7_scene_comparisons([visual])
                if len(visuals)<2:
                    visuals.append(visual)
            print(f'V12.21 {split_name} {ci+1}/{len(rows)}: {row.record_name}',flush=True)
            del cube,encoded,rdo,fft,original,methods
            gc.collect()
        sweep=pd.DataFrame(sweeps);qtable=pd.DataFrame(quality);pp=pd.DataFrame(perception)
        ex.safe_write_csv_v11(sweep,ex.TABLE_DIR_V11/f'{split_name}_cfar_sweep_v12_21.csv')
        ex.safe_write_csv_v11(qtable,ex.TABLE_DIR_V11/f'{split_name}_compression_quality_v12_21.csv')
        ex.safe_write_csv_v11(pd.DataFrame(label_audit),ex.TABLE_DIR_V11/f'{split_name}_object_label_audit_v12_21.csv')
        ex.safe_write_csv_v11(pp,ex.TABLE_DIR_V11/f'{split_name}_PPO_perception_components_v12_21.csv')
        ex.safe_write_csv_v11(pd.concat(action_rows,ignore_index=True),ex.TABLE_DIR_V11/f'{split_name}_PPO_actions_v12_21.csv')
        ex.safe_write_csv_v11(pd.concat(peak_diagnostics,ignore_index=True),
            ex.TABLE_DIR_V11/f'{split_name}_PPO_label_peak_noise_diagnostics_v12_21.csv')
        plot_cfar(ex,sweep,split_name)
        ex.plot_v12_7_compression_effectiveness(qtable,split_name)
        if visuals:
            ex.save_v12_7_reconstructed_ppo_cubes(visuals,max_cubes=2)
        ap=export_detection_reports(ex,frames,output/'detection_metrics',split_name)
    canonicalize_plot_names(ex,output/'figures')
    return dict(sweep=sweep,quality=qtable,perception=pp,AP=ap['summary'],summary=validation_summary(pp))


def rate_sweep(ex,rows,labels,learned,output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    quality=[];sweeps=[]
    for _,row in rows.iterrows():
        cube=ex.load_cube_v6_1000(row);labs=ex.labels_for_cube_v6_1000(labels,row)
        processor=ex._postprocessor_for_cube_v11(cube,learned['postprocessors'])
        original=ex.postprocess_cube(cube,processor);source=ex.prepare_policy_source_v12_12(cube)
        caches={}
        for kind in ('RD','RA'):
            data,rr,second=ex._map_triplet_v12_6(original,kind)
            gate=ex.CFAR_LABEL_GATE_RD_V8 if kind=='RD' else ex.CFAR_LABEL_GATE_RA_V8
            caches[kind]=ex.precompute_original_cfar_count_sweep_v11_2(data,labs,rr,second,kind,
                ex._nearest_training_thresholds_v11(),gate)
        def record(name,family,decoded,meta,action=-1,rho=1.,q=0):
            post=original if family=='Original' else ex.postprocess_cube(decoded,processor)
            item=dict(record_name=row.record_name,scene_name=row.scene_name,policy_id=name,family=family,
                action_index=action,rho=rho,q=q,complex_samples=cube.size,source_bits=32*cube.size,
                realized_bits=meta['realized_bits'],accounted_bpcs=meta['realized_bits']/cube.size,
                MiB_saved=(32*cube.size-meta['realized_bits'])/(8*1024**2),cube_nmse=meta.get('cube_nmse',0.))
            for kind in ('RD','RA'):
                data,_,_=ex._map_triplet_v12_6(post,kind)
                original_map,_,_=ex._map_triplet_v12_6(original,kind)
                count=ex.evaluate_cfar_count_sweep_method_v11_2(caches[kind],data)
                count=count.assign(record_name=row.record_name,policy_id=name,family=family)
                sweeps.append(count)
                item[kind+'_log_power_mse']=ex._normalized_log_power_mse_v11(original_map,data)
            quality.append(item)
        record('Original','Original',cube,dict(realized_bits=32*cube.size,cube_nmse=0.))
        for bits in FFT_BITS:
            decoded,_,meta=ex.compress_cube_fft_full_support_multi(cube,{'FFT':bits})['FFT']
            record(f'FFT{bits}','FFT_full_support',decoded,meta,q=bits)
            del decoded
        for index,(rho,q) in enumerate(ex.ACTION_LIBRARY_V11):
            encoded=ex.coherent_codec_v14.encode_sequence(cube,np.full(len(source['slices']),index),source=source)
            record(f'A{index:02d}','Uniform_library',encoded['cube_hat'],encoded['meta'],index,rho,q)
            del encoded
        encoded=encode_uncapped(ex,cube,learned['policy'],learned['normalizer'],False,source)
        record('PPO_selected','Trained_PPO',encoded['cube_hat'],encoded['meta'])
        del cube,original,encoded,source
        gc.collect()
    qframe=pd.DataFrame(quality);sframe=pd.concat(sweeps,ignore_index=True)
    summary=summarize_operating_points(sframe,qframe)
    summary=summary.drop(columns=['above_PPO_4p8_cap'],errors='ignore')
    for name,frame in [('per_cube',qframe),('per_threshold',sframe),('summary',summary)]:
        ex.safe_write_csv_v11(frame,output/f'rate_sweep_{name}_v12_21.csv')
    plot_percentage_rate_reports(ex,summary,output)
    ex.safe_write_json_v11(dict(rate_cap=None,trained_PPO='one measured operating point, not a variable-rate-trained curve',
        diagnostic_only=True,actions_pruned=False,source_bpcs=32,thresholds=list(ex._nearest_training_thresholds_v11()),
        cubes=rows.record_name.tolist(),heldout_test_used=False),output/'rate_sweep_protocol_v12_21.json')
    return summary


def run_experiment(config=None,output=None,*,reports=True):
    config=RunConfig.from_environment() if config is None else config
    base=Path(os.environ.get('RDO_OUTPUT_ROOT',str(ROOT/'outputs'))).expanduser()
    tag=os.environ.get('RDO_V1221_RUN_TAG','').strip()
    name=f'v12_21_v44_uncapped_perception_ppo_{config.mode}_train{config.train_cubes:03d}_val{config.validation_cubes:03d}_test{config.test_cubes:03d}'
    output=Path(output) if output else base/(name+('_'+tag if tag else ''))
    output=output.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite an existing run: {output}. Set a NEW RDO_V1221_RUN_TAG.')
    output.mkdir(parents=True)
    ex=load_primitives(output)
    hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
        [Path(__file__),*(ROOT/'radar_sim').glob('ppo_v12_21_*.py'),ROOT/'radar_sim/ppo_v12_19_pipeline.py',
         *[ROOT/'radar_sim'/name for name in ('ppo_v12_14_codec.py','processing.py',
             'ppo_v12_19_diagnostics.py','ppo_v12_19_reward.py','ppo_v12_18_learning.py',
             'ppo_conditional_action_learning.py','ppo_advantage_diagnostics.py',
             'ppo_prefix_rollouts.py','matlab_v44_dataset.py')]]}
    labels=ex.load_labels_v6_1000(ex.DATASET_ROOT)
    index=ex.build_cube_index_v6_1000(ex.DATASET_ROOT,labels,accepted_only=False)
    split=ex.temporal_scene_split_v11(index)
    populations={name:ex.select_split_rows_v11(split[name],count) if count else split[name].iloc[:0]
        for name,count in [('train',config.train_cubes),('validation',config.validation_cubes),('test',config.test_cubes)]}
    if set(populations['train'].record_name)&set(populations['validation'].record_name):
        raise AssertionError('TRAIN/validation leakage.')
    with report_directories(ex,output):
        for name,rows in populations.items():ex.safe_write_csv_v11(rows,ex.TABLE_DIR_V11/f'cube_index_{name}_v12_21.csv')
        ex.safe_write_json_v11(dict(revision=REVISION,configuration=asdict(config),bitrate_cap=None,
            dataset_root=str(ex.DATASET_ROOT),kernel_python=os.sys.executable,source_hashes=hashes,
            old_pipeline_execution=False,no_imitation=True,no_refinement=True,no_RDO_rewards=True,
            old_numeric_plot_suffixes='function lineage only; canonical figures renamed to v12_21'),output/'run_protocol_v12_21.json')
    learned=train(ex,populations['train'],populations['validation'],labels,config,output)
    plot_learning(ex,learned,output)
    results=dict(output=output,learned=learned,**populations)
    if reports:
        fixed=preserve_baseline(ex,split['train'],labels,learned['postprocessors'],output)
        results['validation_report']=evaluate_split(ex,populations['validation'],labels,learned,fixed,output,'validation')
        if len(populations['test']):
            results['test_report']=evaluate_split(ex,populations['test'],labels,learned,fixed,output/'heldout_test','heldout')
        sweep_rows=ex.select_split_rows_v11(populations['validation'],min(config.rate_sweep_cubes,len(populations['validation'])))
        if len(sweep_rows):results['rate_sweep']=rate_sweep(ex,sweep_rows,labels,learned,output/'rate_sweep')
    status=dict(completed=True,revision=REVISION,bitrate_cap=None,full_training=config.mode=='full',
        test_evaluated='test_report' in results,refinement_used=False,main_imitation=False,
        selected_probe_summary=validation_summary(learned['validation_candidate']),
        publication_ready=False,actual_serialized_codec_payload_saved=False,
        warning='A complete execution is not a zero-miss guarantee or evidence of matched-rate superiority.')
    for name in ('validation_report','test_report'):
        if name in results:status[name]=results[name]['summary']
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in hashes.items()):
        raise AssertionError('An implementation changed during this run; do not mix results and code revisions.')
    ex.safe_write_json_v11(status,output/'run_status_v12_21.json')
    print('\nV12.21 COMPLETE:',output,flush=True)
    print(json.dumps(status,indent=2),flush=True)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('full','smoke'),default=os.environ.get('RDO_RL_RUN_MODE','full'))
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    os.environ['RDO_RL_RUN_MODE']=args.mode
    run_experiment(output=args.output)


if __name__=='__main__':main()
