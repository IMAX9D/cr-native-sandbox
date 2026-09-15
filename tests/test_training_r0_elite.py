from dataclasses import replace
import pytest
import torch
from training_r0.config import ModelConfig
from training_r0.catalog import CardVocabulary
from training_r0.observation import ObservationBuilder
from training_r0.model import R0Policy
from training_r0.synthetic import rich_public_frame,DECK
from training_r0.semantics import EntityDetail,CombatEvent,OwnCommand,GroupRuntime
from training_r0.feature_contract import CHILD_FEATURES,TOWER_FEATURES,GROUP_FEATURES,CARD_FEATURES,MATCH_FEATURES,COMBAT_KINDS
from training_r0.elite_tensorizer import REQUIRED_CAPTURE
from training_r0.native_graph import native_definitions,node_id
from training_r0.lifecycle import DamageTracker,DamageEvent


@pytest.fixture(autouse=True)
def cpu_threads():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def parts():
    torch.manual_seed(325)
    config=ModelConfig(width=32,layers=1,heads=4,hidden=64,spatial_channels=16,learned_scatter_channels=8,native_graph_layers=1,allow_incomplete_capture=True)
    vocabulary=CardVocabulary.from_native();builder=ObservationBuilder(vocabulary,config)
    return builder,R0Policy(vocabulary,config)


def test_complete_reference_feature_bank_and_expanded_capacity():
    assert [len(v) for v in (CHILD_FEATURES,TOWER_FEATURES,GROUP_FEATURES,CARD_FEATURES,MATCH_FEATURES,COMBAT_KINDS)]==[72,21,22,16,18,39]
    elite=ModelConfig();reference=ModelConfig.reference()
    assert (elite.width,elite.layers,elite.heads,elite.hidden,elite.learned_scatter_channels,elite.spatial_channels,elite.local_pool_slots)==(384,6,12,768,64,128,8)
    assert reference.width==256 and reference.hidden==512
    assert not elite.allow_incomplete_capture


def test_native_graph_contains_real_definitions_and_all_attributes(parts):
    builder,model=parts;raw,sha=native_definitions()
    assert len(raw['attributes'])>20000 and len(raw['edges'])>2000 and len(sha)==64
    assert node_id('CHARACTER.HogRider')>1
    memory=model.native_graph()
    assert memory.shape[0]==len(raw['node_names']) and memory.shape[1]==model.config.width
    connected=torch.unique(model.native_graph.edge_source[:16])
    memory[connected].square().sum().backward()
    assert model.native_graph.attribute[0].weight.grad.abs().sum()>0
    assert model.native_graph.forward_messages[0].weight.grad.abs().sum()>0


def test_all_72_values_and_known_bits_survive_tensorization(parts):
    builder,model=parts;frame=rich_public_frame(builder)
    detail=replace(frame.semantics.entities[0],archetype='CHARACTER.HogRider',child_features=tuple((name,float(i)/100) for i,name in enumerate(CHILD_FEATURES)))
    frame=replace(frame,semantics=replace(frame.semantics,entities=(detail,*frame.semantics.entities[1:])))
    batch=builder.batch([frame]);values=batch.semantic.elite.child[0,0]
    assert torch.allclose(values[::2],torch.arange(72)/100) and torch.equal(values[1::2],torch.ones(72))
    original=model.encode(batch).policy
    changed=replace(detail,child_features=tuple((name,1. if i==60 else float(i)/100) for i,name in enumerate(CHILD_FEATURES)))
    other=replace(frame,semantics=replace(frame.semantics,entities=(changed,*frame.semantics.entities[1:])))
    assert not torch.allclose(original,model.encode(builder.batch([other])).policy,atol=1e-7,rtol=0)


def test_combat_event_type_source_and_amount_affect_context(parts):
    builder,model=parts;frame=rich_public_frame(builder)
    event=CombatEvent('hit','damage',99,0,5000001,5000002,DECK[0],0,(('amount_sum_over_5000',.2),))
    frame=replace(frame,semantics=replace(frame.semantics,combat_events=(event,)))
    baseline=model.encode(builder.batch([frame])).policy
    changed=replace(frame,semantics=replace(frame.semantics,combat_events=(replace(event,kind='heal'),)))
    assert not torch.allclose(baseline,model.encode(builder.batch([changed])).policy,atol=1e-7,rtol=0)
    with pytest.raises(ValueError,match='future'):
        replace(frame.semantics,combat_events=(replace(event,observed_tick=101),))


def test_complete_previous_action_offset_order_and_identity(parts):
    builder,model=parts;frame=rich_public_frame(builder)
    command=OwnCommand('one',95,96,'play',DECK[0],(.2,.3),4.,effective_card=DECK[0],effective_form=0,
        source_position=(.3,.4),delay_offset_bin=1,sequence_index=0)
    frame=replace(frame,semantics=replace(frame.semantics,previous=(command,)))
    baseline=model.encode(builder.batch([frame])).policy
    other=replace(frame,semantics=replace(frame.semantics,previous=(replace(command,delay_offset_bin=4,sequence_index=1),)))
    assert not torch.allclose(baseline,model.encode(builder.batch([other])).policy,atol=1e-7,rtol=0)


def test_elite_production_rejects_missing_capture_streams(parts):
    builder,diagnostic=parts
    production=R0Policy(builder.vocabulary,replace(builder.config,allow_incomplete_capture=False))
    frame=rich_public_frame(builder)
    with pytest.raises(ValueError,match='capture streams incomplete'):production(builder.batch([frame]))
    # A capability declaration alone cannot hide missing archetypes/tower details.
    declared=replace(frame,semantics=replace(frame.semantics,capture_capabilities=REQUIRED_CAPTURE))
    with pytest.raises(ValueError,match='capture streams incomplete'):production(builder.batch([declared]))


def test_compact_tower_tokens_do_not_keep_non_tower_slots(parts):
    builder,model=parts;batch=builder.batch([rich_public_frame(builder)])
    context=model.encode(batch)
    expected=1+8+batch.semantic.enemy_tokens.shape[1]+model.config.max_towers+batch.semantic.group_mask.shape[1]
    assert context.tokens.shape[1]==expected


def test_damage_attribution_windows_and_no_future_leakage():
    tracker=DamageTracker('g',continuous_damage_capture=True)
    tracker.spawn(1,0,observed_tick=0,birth_tick=0,deployment_id='deploy')
    tracker.spawn(2,0,observed_tick=10,birth_tick=10,parent_key=1,kind='projectile')
    early=DamageEvent('a',15,16,2,'unit',120.)
    assert tracker.record(early) and not tracker.record(early)
    tracker.record(DamageEvent('b',50,65,1,'tower',200.))
    assert tracker.snapshot(1,60).values[:2]==(120.,0.)
    late=tracker.snapshot(1,65)
    assert late.values[:2]==(120.,200.) and len(late.values)==6
    assert tracker.deployment_totals('deploy',65)['unit_damage']==120.
    tracker.mark_dead(1,death_tick=55,observed_tick=56)
    tracker.record(DamageEvent('after-death-projectile',60,66,2,'tower',10.))
    assert tracker.snapshot(1,66).values[1]==210.


def test_summoned_units_separate_lifetimes_but_shared_deployment():
    tracker=DamageTracker('g',continuous_damage_capture=True)
    tracker.spawn(1,0,observed_tick=0,birth_tick=0,deployment_id='deploy')
    tracker.spawn(3,0,observed_tick=5,birth_tick=5,parent_key=1,kind='unit')
    tracker.record(DamageEvent('child',10,10,3,'unit',50.))
    assert tracker.snapshot(1,10).values[0]==0.
    assert tracker.snapshot(3,10).values[0]==50.
    assert tracker.deployment_totals('deploy',10)['unit_damage']==50.
    tracker.record(DamageEvent('unknown',11,11,None,'unit',15.))
    assert not tracker.snapshot(3,11).damage_complete
    assert tracker.deployment_totals('deploy',11)['unit_damage']==50.


def test_damage_gaps_partial_capture_and_lifecycle_input_binding(parts):
    builder,model=parts
    tracker=DamageTracker('synthetic-game',continuous_damage_capture=True,capture_start_tick=90)
    tracker.spawn(5000001,0,observed_tick=90,birth_tick=80)
    assert not tracker.snapshot(5000001,100).damage_complete
    complete=DamageTracker('synthetic-game',continuous_damage_capture=True)
    complete.spawn(5000001,0,observed_tick=80,birth_tick=80)
    complete.record(DamageEvent('hit',95,96,5000001,'tower',200.))
    frame=rich_public_frame(builder)
    detail=replace(frame.semantics.entities[0],damage=complete.snapshot(5000001,100))
    frame=replace(frame,semantics=replace(frame.semantics,entities=(detail,*frame.semantics.entities[1:])))
    batch=builder.batch([frame]);assert batch.semantic.elite.lifecycle.shape[-1]==6
    assert batch.semantic.elite.lifecycle[0,0,1].item()==pytest.approx(.02)
    model.encode(batch).value.sum().backward();assert model.elite.lifecycle[0].weight.grad.abs().sum()>0
    wrong=replace(frame,semantics=replace(frame.semantics,entities=(replace(detail,damage=complete.snapshot(5000001,105)),*frame.semantics.entities[1:])))
    with pytest.raises(ValueError,match='lifecycle'):builder.batch([wrong])
    complete.mark_capture_gap(90,92);assert not complete.snapshot(5000001,100).damage_complete
