"""Compiler unit fixtures do not certify native capture fidelity."""
from dataclasses import replace
import gzip
import json

import pytest
import torch

from data_pipeline.r0_compile import ExpertCommand,battle_partition,canonical_cell,compile_label,index_capture,reconcile_plan
from training_r0.catalog import CardVocabulary
from training_r0.config import ModelConfig
from training_r0.observation import ObservationBuilder
from training_r0.synthetic import rich_public_frame


@pytest.fixture(scope='module')
def batch():
    builder=ObservationBuilder(CardVocabulary.from_native(),ModelConfig(allow_incomplete_capture=True))
    b=builder.batch([rich_public_frame(builder,tick=100)])
    return replace(b,candidate_uids=torch.tensor([[0,1,2,3,15000001]]))


def command(tick=101,deck=0,side=0,x=500,y=500,accepted=True):
    return ExpertCommand(tick,side,dict(type='play',side=side,deck_index=deck,x=x,y=y),accepted)


def test_offsets_at_window_edges_and_public_grid(batch):
    action,known,reason=compile_label(batch,[command(101),command(105,deck=1)])
    assert known and reason=='action' and action.offset_bin.tolist()==[[0,4]]
    assert action.count.tolist()==[2] and action.target_cell.tolist()==[[0,0]]
    assert command(105).window==100 and command(106).window==105
    assert canonical_cell(17499,31499,1)==0
    assert canonical_cell(499,500,0)==0
    with pytest.raises(ValueError,match='not representable'):canonical_cell(700,500,0)


def test_rejected_unknown_and_incomplete_are_not_wait_labels(batch):
    for commands,complete,reason in [([command(accepted=False)],True,'source_action_rejected_or_unresolved'),
        ([ExpertCommand(101,0,None,False,'unresolved')],True,'source_action_rejected_or_unresolved'),
        ([],False,'incomplete_window')]:
        action,known,actual=compile_label(batch,commands,window_complete=complete)
        assert not known and reason==actual and action.count.item()==0
    assert compile_label(batch,[])[1:] == (True,'observed_wait')


def test_shadow_resource_and_reuse_fail_closed(batch):
    assert compile_label(batch,[command(),command(102)])[1] is False
    assert compile_label(batch,[command(deck=3),command(102,deck=2)])[1] is False  # 4+4 > 7
    assert compile_label(batch,[command(deck=7)])[1] is False
    assert compile_label(batch,[command(y=30500)])[1] is False
    assert compile_label(batch,[command(),command(102,deck=1),command(103,deck=2)])[1] is False


def test_index_requires_receipt_source_identity_and_retains_missing(tmp_path):
    path=tmp_path/'capture.gz'
    rows=[dict(kind='episode'),dict(kind='frame',state=dict(tick=0)),
          dict(kind='source_action',source_kind='play',event=dict(tick=0,side=0),requested_execution_tick=1),
          dict(kind='frame',state=dict(tick=1)),
          dict(kind='source_action',source_kind='ability',event=dict(tick=1,side=1),requested_execution_tick=2),
          dict(kind='source_action_not_executed',tick=2,side=1,reason='ability_unresolved')]
    with gzip.open(path,'wt',encoding='utf-8') as f:
        f.writelines(json.dumps(r)+'\n' for r in rows)
    _,index,trace=index_capture(path)
    assert trace['continuous'] and index[0,0][0].reason=='missing_receipt'
    assert not index[0,1][0].accepted
    rows.append(dict(kind='command_receipt',tick=2,request=dict(action=dict(side=1)),response=dict(result=dict(accepted=True,tick=2))))
    with gzip.open(path,'wt',encoding='utf-8') as f:f.writelines(json.dumps(r)+'\n' for r in rows)
    with pytest.raises(ValueError,match='matched'):index_capture(path)


def test_battle_partition_is_stable():
    assert battle_partition('same-game')==battle_partition('same-game')
    assert {battle_partition(str(i)) for i in range(1000)}=={'train','validation','test'}


def test_missing_source_marker_masks_owner_not_false_wait():
    from collections import defaultdict
    index=defaultdict(list);trace=dict(last_tick=200,source_actions=[])
    reconcile_plan(index,trace,dict(actions=[dict(tick=101,side=1)],ability_events=[]))
    assert trace['source_markers_missing']==1
    assert index[100,1][0].reason=='source_marker_missing_from_capture'
    assert not index[100,1][0].accepted and not index[100,0]


def test_explicit_compilation_prefix_has_aligned_bounded_end():
    from data_pipeline.r0_compile import compilation_end_tick
    assert compilation_end_tick(4360,1200)==1200
    assert compilation_end_tick(4360,None)==4360
    for invalid in (-5,1201,5000,True):
        with pytest.raises(ValueError):compilation_end_tick(4360,invalid)
