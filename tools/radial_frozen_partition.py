"""Metadata-only historical global RADIal partitions; never rebalance a subset.

Use after native dataset construction and before any ADC/feature fitting. A
legacy Runtime may have constructed temporary local partitions; replace its rows
AND refresh its scoped labels using the returned frozen roles before body reads.
Bounded calls return no TEST rows, while native access guards retain every
available historical TEST identity. No annotation or chip metadata is changed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from radar_sim import radial_adaptive_gpu_data as native

ROLES=('train','validation','test')


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def load_global_assignment(path):
    """Load historical SPLIT_AUDIT JSON or an explicit recording/role CSV."""
    path=Path(path).resolve()
    if not path.is_file():raise FileNotFoundError(path)
    if path.suffix.lower()=='.json':
        source=json.loads(path.read_text(encoding='utf-8-sig'))
        groups=source.get('recordings')
        if not isinstance(groups,dict) or set(groups)!=set(ROLES):
            raise ValueError('Historical JSON must contain train/validation/test recording lists.')
        rows=[(str(scene),role) for role in ROLES for scene in groups[role]]
        global_counts=source.get('actual_cube_counts')
    elif path.suffix.lower()=='.csv':
        frame=pd.read_csv(path)
        scene='recording' if 'recording' in frame else 'scene_name' if 'scene_name' in frame else None
        role='role' if 'role' in frame else 'split_role' if 'split_role' in frame else None
        if scene is None or role is None:raise ValueError('Assignment CSV needs recording/scene_name and role/split_role.')
        rows=[(str(s),str(r)) for s,r in zip(frame[scene],frame[role])]
        global_counts=None
    else:raise ValueError('Provide the frozen global JSON or CSV assignment.')
    mapping={}
    for scene,role in rows:
        if not scene or role not in ROLES:raise ValueError('Unknown or empty global recording/role assignment.')
        if scene in mapping and mapping[scene]!=role:raise ValueError('A historical recording appears in multiple partitions: '+scene)
        mapping[scene]=role
    if set(mapping.values())!=set(ROLES):raise ValueError('The GLOBAL assignment must cover all three nonempty roles.')
    return mapping,dict(source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        global_recording_role_hash=_digest(sorted(mapping.items())),global_cube_counts=global_counts,
        global_recording_counts={role:sum(r==role for r in mapping.values()) for role in ROLES})


def assign_frozen_partitions(dataset,assignment_path,*,bounded=True,out=None):
    """Assign every eligible native row using immutable global recording roles.

    Missing recordings change availability counts only. Unknown eligible
    recordings are rejected before mutating the dataset; explicitly extend a
    global assignment rather than silently fitting a new60/20/20 partition.
    This function does not require a nonempty TRAIN subset: returning zero is
    deliberate, so the caller must obtain genuine TRAIN data before proceeding.
    """
    mapping,provenance=load_global_assignment(assignment_path)
    original=dataset.index.copy(deep=True)
    required={'record_name','scene_name','frame_in_scene'}
    if not required.issubset(original):raise ValueError('Native complete-cube index is missing frame identity fields.')
    if original.record_name.duplicated().any():raise ValueError('Duplicate eligible native cube identity.')
    unknown=set(original.scene_name.astype(str))-set(mapping)
    if unknown:raise ValueError('Eligible recordings absent from the frozen global assignment: '+', '.join(sorted(unknown)))
    index=original.copy(deep=True);index['split_role']=index.scene_name.map(mapping)
    # Role assignment is the sole permitted row modification. Label tables are
    # neither copied/filtered nor renumbered by this module.
    unchanged=[name for name in original.columns if name!='split_role']
    pd.testing.assert_frame_equal(index[unchanged],original[unchanged],check_exact=True)
    frames=[index.loc[index.split_role.eq(role)].copy().sort_values(['scene_name','frame_in_scene']).reset_index(drop=True) for role in ROLES]
    names=[set(frame.record_name) for frame in frames]
    if sum(map(len,frames))!=len(original) or any(names[i]&names[j] for i in range(3) for j in range(i)):
        raise AssertionError('Frozen assignment lost or repeated native rows.')
    available=set(index.scene_name)
    counts=list(map(len,frames));total=len(index)
    audit=dict(**provenance,assignment_method='Frozen historical global recording roles; no subset rebalancing',
        metadata_only=True,ADC_bodies_read=False,labels_modified=False,frame_indices_renumbered=False,
        recording_disjoint=True,data_repeated=False,all_eligible_frames_assigned=True,
        actual_cube_counts=counts,actual_fractions=[value/total if total else 0. for value in counts],
        available_recording_counts={role:int(frames[i].scene_name.nunique()) for i,role in enumerate(ROLES)},
        recordings={role:sorted(scene for scene,r in mapping.items() if r==role) for role in ROLES},
        available_recordings={role:sorted(scene for scene in available if mapping[scene]==role) for role in ROLES},
        missing_recordings={role:sorted(scene for scene,r in mapping.items() if r==role and scene not in available) for role in ROLES},
        available_cube_role_hash=_digest(sorted(zip(index.record_name.astype(str),index.split_role.astype(str)))),
        available_subset_fractions_are_not_rebalanced=True,bounded=bool(bounded),
        used_cube_counts=[counts[0],counts[1],0 if bounded else counts[2]],
        TEST_returned=not bounded and counts[2]>0,
        historical_TEST_already_used_in_research='This assignment prevents further role drift; it does not erase earlier TEST exposure.')
    dataset.index=index
    dataset._adaptive_partitions=frames
    dataset._adaptive_test_records=names[2]
    dataset._adaptive_authorized_test_records=set()
    dataset._splits=dict(zip(ROLES,frames))
    dataset.partition_audit=audit
    dataset._frozen_recording_roles=dict(mapping)
    dataset._frozen_partition_bounded=bool(bounded)
    if hasattr(dataset,'tiny_partition_audit'):delattr(dataset,'tiny_partition_audit')
    result=[frame.copy() for frame in frames]
    if bounded:result[2]=result[2].iloc[:0].copy()
    if out is not None:
        out=Path(out);out.mkdir(parents=True,exist_ok=True)
        with (out/'FROZEN_PARTITION_AUDIT.json').open('x',encoding='utf-8') as stream:
            json.dump(audit,stream,indent=2,allow_nan=False)
        pd.DataFrame([dict(recording=scene,role=role,available=scene in available) for scene,role in sorted(mapping.items())]).to_csv(out/'GLOBAL_RECORDING_ROLES.csv',index=False,mode='x')
        index.to_csv(out/'AVAILABLE_CUBE_ROLES.csv',index=False,mode='x')
    return result


def assert_frozen_role_access(dataset,rows,role):
    """Check native identity/access and prevent historical VAL->TRAIN promotion."""
    if role not in ROLES or not hasattr(dataset,'_frozen_recording_roles'):
        raise ValueError('Establish the frozen global assignment and use a declared role.')
    native.assert_partition_access(dataset,rows)
    if role=='test' and dataset._frozen_partition_bounded:raise PermissionError('Bounded experiments never use TEST rows.')
    expected=rows.scene_name.map(dataset._frozen_recording_roles)
    if expected.isna().any() or not expected.eq(role).all():
        raise PermissionError('Requested roster crosses frozen global partition roles.')
    if 'split_role' in rows and not rows.split_role.eq(role).all():
        raise PermissionError('Roster role labels disagree with its declared frozen role.')
