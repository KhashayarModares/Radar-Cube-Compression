"""TRAIN-only placement of expensive conditional-return measurements.

Placement uses original supported annotations BEFORE any current policy action.
It never selects an action, modifies actor inputs or runs at deployment. The
updater still replaces just one actor term per measured primary state, so
directing measurement effort does not duplicate/overweight annotation states.
"""
from __future__ import annotations

import numpy as np


def peak_tile(source, full_ranges, full_velocities, range_m, velocity_mps):
    r=int(np.argmin(np.abs(np.asarray(full_ranges)-float(range_m))))
    v=int(np.argmin(np.abs(np.asarray(full_velocities)-float(velocity_mps))))
    # RADIal detector maps use the official unshifted FFT coordinates, whereas
    # the unchanged codec tiles live on fftshift axes. MATLAB axes are already
    # shifted and retain the historical path when this explicit flag is absent.
    if source.get('radial_unshifted_detector_axes', False):
        r=(r+len(full_ranges)//2)%len(full_ranges)
        v=(v+len(full_velocities)//2)%len(full_velocities)
    found=[i for i,s in enumerate(source['slices'])
           if s[1].start<=v<s[1].stop and s[2].start<=r<s[2].stop]
    if len(found)!=1:
        raise AssertionError('The actual original peak must belong to exactly one codec tile.')
    return found[0]


def supported_object_tiles(source,post,label_frame):
    """Order unique RD peak tiles by the SAME object's weakest RD/RA support."""
    if label_frame.empty:
        return []
    rd=label_frame.loc[label_frame.map_type.eq('RD')]
    if rd.source_label_index.duplicated().any():
        raise ValueError('Original RD association must be unique for each annotation.')
    ordered=[]
    for source_index,rows in label_frame.groupby('source_label_index'):
        counterpart=rd.loc[rd.source_label_index.eq(source_index)]
        if len(counterpart)!=1:
            continue  # no guessed RD association for an RA-only supported object
        item=counterpart.iloc[0]
        if not rows.object_id.astype(str).eq(str(item.object_id)).all():
            raise AssertionError('RD/RA counterpart belongs to a different object.')
        tile=peak_tile(source,post['ranges'],post['vel'],item.original_peak_range_m,
                       item.original_peak_second_axis)
        ordered.append(dict(tile_index=tile,object_id=str(item.object_id),
            source_label_index=int(source_index),original_weakest_score=float(rows.original_score.min()),
            reason='TRAIN_original_supported_object_peak; before policy actions',
            actor_uses_GT=False,action_selected_by_GT=False))
    ordered.sort(key=lambda r:(r['original_weakest_score'],r['source_label_index']))
    unique=[]
    for row in ordered:
        if row['tile_index'] not in [r['tile_index'] for r in unique]:unique.append(row)
    return unique


def choose_training_prefixes(ex,original,labels,source,count,rng,record_name,scene_name,strategy):
    count=min(int(count),len(source['slices']))
    if count<=0:return [],[]
    if strategy=='uniform':
        tiles=sorted(rng.choice(len(source['slices']),count,replace=False).tolist())
        return tiles,[dict(tile_index=t,reason='uniform_pre_action_control',actor_uses_GT=False,
                           action_selected_by_GT=False) for t in tiles]
    if strategy!='original_supported':raise ValueError('Unknown conditional measurement placement.')
    from radar_sim.ppo_v12_19_diagnostics import _label_rows
    frame=_label_rows(ex,original,original,labels,str(record_name),str(scene_name))
    objects=supported_object_tiles(source,original,frame)
    chosen=objects[:1]
    # A second contrast targets original outside-GT clutter peaks. This is a
    # TRAIN query location, not an action or a deployed reconstruction repair.
    data,rr,vv=ex._map_triplet_v12_6(original,'RD')
    cache=ex.precompute_original_cfar_count_sweep_v11_2(data,labels,rr,vv,'RD',[1.25],ex.CFAR_LABEL_GATE_RD_V8)
    _,associated,points=ex._associate_labels_and_peaks_v12_5(cache['label_bins'],cache['per_threshold'][1.25]['peaks0'],cache['gate'])
    counts={}
    for r,v in np.asarray(points)[~np.asarray(associated,bool)]:
        tile=peak_tile(source,original['ranges'],original['vel'],rr[int(r)],vv[int(v)])
        counts[tile]=counts.get(tile,0)+1
    for tile,number in sorted(counts.items(),key=lambda x:(-x[1],x[0])):
        if len(chosen)>=count:break
        if tile not in [r['tile_index'] for r in chosen]:
            chosen.append(dict(tile_index=tile,reason='TRAIN_original_outside_gate_peak_tile; before policy actions',
                original_unmatched_peak_count=number,actor_uses_GT=False,action_selected_by_GT=False))
    for row in objects[1:]:
        if len(chosen)>=count:break
        if row['tile_index'] not in [r['tile_index'] for r in chosen]:chosen.append(row)
    if len(chosen)<count:
        remaining=[t for t in range(len(source['slices'])) if t not in [r['tile_index'] for r in chosen]]
        for tile in rng.choice(remaining,count-len(chosen),replace=False):
            chosen.append(dict(tile_index=int(tile),reason='uniform_control_no_additional_original_peak_tile',
                actor_uses_GT=False,action_selected_by_GT=False))
    chosen=sorted(chosen,key=lambda r:r['tile_index'])
    return [r['tile_index'] for r in chosen],chosen
