"""Behavioral checks for the rich adaptation, not a claim of playing strength."""
from dataclasses import replace
from copy import deepcopy
import json

import pytest
import torch

from training_r0.catalog import CardVocabulary
from training_r0.config import ModelConfig
from training_r0.model import R0Policy
from training_r0.observation import ObservationBuilder
from training_r0.synthetic import public_frame,native_frame,candidates,DECK
from training_r0.history import ConfirmedEvent
from training_r0.semantics import (CardProfile,MechanicOperation,SemanticCatalog,
    EntityDetail,PublicEffect,PublicRelation,OwnCommand,PublicSemantics,DYNAMIC_SCALES)


@pytest.fixture(autouse=True)
def cpu():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def parts():
    torch.manual_seed(410)
    vocab=CardVocabulary.from_native()
    config=ModelConfig(width=32,heads=4,layers=2,hidden=64,spatial_channels=16,learned_scatter_channels=8,native_graph_layers=1,allow_incomplete_capture=True)
    return ObservationBuilder(vocab,config),R0Policy(vocab,config)


def rich_frame(builder):
    frame=public_frame(builder)
    semantic=PublicSemantics(frame.episode_uid,0,100,
        entities=(EntityDetail(5000001,'troop','spawnA','deployment',radius_tiles=.5,
                    values=(('shield',300.),('velocity_x',2.),('attack_cooldown_ms',800.))),
                  EntityDetail(5000002,'troop',radius_tiles=0.,values=(('flying',1.),))),
        effects=(PublicEffect(5000001,'slow',95,magnitude=.5,remaining_ms=1800.,stacks=1),),
        relations=(PublicRelation(5000001,5000002,'targets',99),),
        previous=(OwnCommand('done',90,92,'play',DECK[0],(.2,.25),4.),),
        pending=(OwnCommand('sent',100,100,'ability',26000038,cost=2.),),
        revealed_enemy_cards=(DECK[2],))
    return replace(frame,semantics=semantic,events=(ConfirmedEvent('e',95,99,1,'play',DECK[0]),))


def context(builder,model,frame):
    return model.encode(builder.batch([frame]))


def test_each_new_public_branch_changes_policy_context(parts):
    builder,model=parts;frame=rich_frame(builder)
    with torch.no_grad():
        baseline=context(builder,model,frame).policy
        variants=[replace(frame.semantics,effects=()),replace(frame.semantics,relations=()),
            replace(frame.semantics,previous=()),replace(frame.semantics,pending=()),
            replace(frame.semantics,revealed_enemy_cards=()),
            replace(frame.semantics,entities=(replace(frame.semantics.entities[0],values=(),radius_tiles=2.),frame.semantics.entities[1]))]
        for changed in variants:
            assert not torch.allclose(baseline,context(builder,model,replace(frame,semantics=changed)).policy,atol=1e-7,rtol=0)


@pytest.mark.parametrize('switch,field',[('use_effects','effects'),('use_relation_edges','relations'),('use_command_memory','pending')])
def test_ablation_switch_removes_corresponding_information(parts,switch,field):
    builder,old=parts
    config=replace(old.config,**{switch:False})
    model=R0Policy(builder.vocabulary,config);builder=ObservationBuilder(builder.vocabulary,config)
    frame=rich_frame(builder)
    with torch.no_grad():
        a=context(builder,model,frame)
        b=context(builder,model,replace(frame,semantics=replace(frame.semantics,**{field:()})))
    assert torch.allclose(a.policy,b.policy,atol=1e-6,rtol=0)


def test_static_mechanic_operations_are_used_and_catalogs_bound(parts):
    builder,base=parts
    catalog=builder.semantic_catalog
    selected=next(p for p in catalog.profiles if p.card_id==DECK[0])
    enhanced=replace(selected,operations=(MechanicOperation('damage',1200.,radius_tiles=2.),))
    changed=replace(catalog,profiles=tuple(enhanced if p.card_id==selected.card_id else p for p in catalog.profiles))
    other=ObservationBuilder(builder.vocabulary,builder.config,changed)
    model=R0Policy(builder.vocabulary,builder.config,changed)
    assert builder.schema_hash!=other.schema_hash
    with pytest.raises(ValueError,match='schema mismatch'):
        model(builder.batch([public_frame(builder)]))
    frame=rich_frame(other)
    out=model(other.batch([frame]));out.value.sum().backward()
    assert model.cards.operation_kind.weight.grad.abs().sum()>0
    with pytest.raises(RuntimeError,match='contract mismatch'):
        model.load_state_dict(base.state_dict())


def test_unknown_is_distinct_from_known_zero_and_nominal_data_loaded(parts):
    builder,model=parts;frame=rich_frame(builder)
    unknown=replace(frame,semantics=replace(frame.semantics,entities=(replace(frame.semantics.entities[0],values=()),frame.semantics.entities[1])))
    zero=replace(unknown,semantics=replace(unknown.semantics,entities=(replace(unknown.semantics.entities[0],values=(('shield',0.),)),unknown.semantics.entities[1])))
    a,b=(builder.batch([f]) for f in (unknown,zero))
    field=list(DYNAMIC_SCALES).index('shield')
    assert a.semantic.dynamic[0,0,2*field:2*field+2].tolist()==[0.,0.]
    assert b.semantic.dynamic[0,0,2*field:2*field+2].tolist()==[0.,1.]
    with torch.no_grad():assert not torch.allclose(model.encode(a).policy,model.encode(b).policy)
    profile=next(p for p in builder.semantic_catalog.profiles if p.card_id==26000021)
    assert dict(profile.values)['targets_buildings_only']==1
    assert dict(profile.values)['hit_interval_ms']==1600


def test_bilinear_scatter_conserves_features_and_responds_within_cell(parts):
    _,model=parts;scatter=model.spatial.scatter
    values=torch.ones(1,1,1,requires_grad=True)
    xy=torch.tensor([[[4.25/18,8.5/32]]],requires_grad=True)
    mask=torch.ones(1,1,dtype=torch.bool);kind=torch.ones(1,1,dtype=torch.long)
    result=scatter(values,xy,mask,kind,torch.zeros(1,1),torch.zeros(1,1,2))
    assert result[0,0,8:10,4:6].detach().tolist()==[[.375,.125],[.375,.125]]
    assert result.sum().item()==1
    moved=scatter(values,xy+torch.tensor([[[.25/18,0.]]]),mask,kind,torch.zeros(1,1),torch.zeros(1,1,2))
    assert not torch.equal(result,moved)
    result[0,0,8,4].backward()
    assert torch.isfinite(xy.grad).all() and xy.grad.abs().sum()>0
    assert values.grad.abs().sum()>0


def test_circle_rectangle_coverage_and_boundary(parts):
    _,model=parts;scatter=model.spatial.scatter
    values=torch.ones(1,1,1);mask=torch.ones(1,1,dtype=torch.bool)
    xy=torch.tensor([[[4.5/18,8.5/32]]])
    circle=scatter(values,xy,mask,torch.full((1,1),2),torch.tensor([[1.1]]),torch.zeros(1,1,2))
    rectangle=scatter(values,xy,mask,torch.full((1,1),3),torch.zeros(1,1),torch.tensor([[[1.,1.]]]))
    assert circle.sum()==5 and rectangle.sum()==9
    edge=scatter(values,torch.ones(1,1,2),mask,torch.ones(1,1,dtype=torch.long),torch.zeros(1,1),torch.zeros(1,1,2))
    assert edge.sum()==1 and edge[0,0,-1,-1]==1


def test_groups_pool_real_members_and_opaque_keys_not_learned(parts):
    builder,model=parts
    raw=native_frame();extra=deepcopy(raw['entities'][0]);extra['category']=5000003;extra['x']+=400
    raw['entities'].append(extra);raw['entity_count']=3
    frame=builder.from_native(raw,episode_uid='g',actor_side=0,own_deck=DECK,candidates=candidates(),
        semantics=PublicSemantics('g',0,100,entities=(EntityDetail(5000001,'troop','A','deployment'),EntityDetail(5000003,'troop','A','deployment'))))
    batch=builder.batch([frame])
    assert sorted(batch.semantic.membership.sum(-1)[0].tolist())==[1,2]
    renamed=replace(frame,semantics=replace(frame.semantics,entities=tuple(replace(e,group_id='different opaque id') for e in frame.semantics.entities)))
    with torch.no_grad():assert torch.equal(model.encode(batch).policy,context(builder,model,renamed).policy)
    singleton=replace(frame,semantics=replace(frame.semantics,entities=tuple(replace(e,group_id=None) for e in frame.semantics.entities)))
    with torch.no_grad():assert not torch.allclose(model.encode(batch).policy,context(builder,model,singleton).policy)
    cfg=replace(model.config,use_groups=False)
    separate_model=R0Policy(builder.vocabulary,cfg);separate_builder=ObservationBuilder(builder.vocabulary,cfg)
    assert torch.isfinite(context(separate_builder,separate_model,frame).policy).all()


def test_mixed_batch_padding_does_not_change_real_frame(parts):
    builder,model=parts;first=rich_frame(builder)
    other=public_frame(builder,episode_uid='second')
    other=replace(other,view=replace(other.view,entities=(),towers=()),native_entity_count=0,candidates=())
    with torch.no_grad():
        single=context(builder,model,first)
        joined=model.encode(builder.batch([other,first]))
    assert torch.allclose(single.policy[0],joined.policy[1],atol=1e-5,rtol=0)
    assert torch.isfinite(joined.policy).all()


def test_public_boundary_staleness_unknown_references_and_capacity(parts):
    builder,model=parts;frame=rich_frame(builder)
    raw=native_frame();private=deepcopy(raw)
    private['players'][1]['elixir_raw']=5000
    private['players'][1]['hand_deck_indices']=[4,5,6,7];private['players'][1]['next_deck_index']=0
    private['entities'][1]['ability_cooldown_remaining_ms']=99000
    with torch.no_grad():
        a,b=[context(builder,model,builder.from_native(r,episode_uid=frame.episode_uid,actor_side=0,own_deck=DECK,candidates=candidates(),semantics=frame.semantics)) for r in (raw,private)]
    assert torch.equal(a.policy,b.policy) and torch.equal(a.value,b.value)
    for semantic in [replace(frame.semantics,tick=101),replace(frame.semantics,actor_side=1),replace(frame.semantics,episode_uid='wrong')]:
        with pytest.raises(ValueError,match='episode/side/tick'):
            builder.batch([replace(frame,semantics=semantic)])
    with pytest.raises(ValueError,match='absent'):
        builder.batch([replace(frame,semantics=replace(frame.semantics,effects=(PublicEffect(999,'slow',99),)))])
    with pytest.raises(ValueError,match='future'):
        replace(frame.semantics,relations=(PublicRelation(5000001,5000002,'targets',101),))
    small=ObservationBuilder(builder.vocabulary,replace(builder.config,max_effects=1))
    with pytest.raises(OverflowError):
        small.batch([replace(frame,semantics=replace(frame.semantics,effects=frame.semantics.effects*2))])


def test_rich_branches_receive_finite_gradients_and_joint_logp_matches(parts):
    builder,model=parts;frame=rich_frame(builder)
    child=replace(frame.view.entities[0],key=5000003,x=4600,hp=800)
    frame=replace(frame,view=replace(frame.view,entities=(*frame.view.entities,child)),native_entity_count=3,
        semantics=replace(frame.semantics,entities=(*frame.semantics.entities,EntityDetail(5000003,'troop','spawnA','deployment'))))
    batch=builder.batch([frame])
    from training_r0.actions import ActionSequence
    action=ActionSequence(torch.tensor([2]),torch.tensor([[100,101]]),torch.tensor([[20,30]]),torch.tensor([[0,1]]))
    output=model(batch,forced=action)
    (-output.logp.mean()+output.value.square().mean()).backward()
    for name,parameter in [('effects',model.rich.effect_kind.weight),('groups',model.rich.local_queries),
        ('relations',model.relation_blocks[0].edge_bias.weight),('commands',model.rich.command_encoder[0].weight),
        ('spatial',model.spatial.scatter_value.weight)]:
        assert parameter.grad is not None and parameter.grad.abs().sum()>0,name
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    with torch.no_grad():
        ctx=model.encode(batch)
        for seed in range(10):
            sampled=model.decode(batch,ctx,generator=torch.Generator().manual_seed(seed))
            evaluated=model.decode(batch,ctx,forced=sampled.actions)
            assert torch.allclose(sampled.logp,evaluated.logp,atol=1e-6)


def test_rich_checkpoint_roundtrip_and_legacy_rejection(parts):
    builder,model=parts;frame=rich_frame(builder)
    other=R0Policy(builder.vocabulary,builder.config)
    other.load_state_dict(model.state_dict())
    with torch.no_grad():assert torch.equal(context(builder,model,frame).policy,context(builder,other,frame).policy)
    legacy={k:v for k,v in model.state_dict().items() if k!='_contract'}
    legacy['gate_head.2.weight']=torch.ones_like(model.gate_head.weight)
    before=other.gate_head.weight.detach().clone()
    with pytest.raises(RuntimeError,match='contract mismatch'):other.load_state_dict(legacy,strict=False)
    assert torch.equal(before,other.gate_head.weight)


def test_native_table_hash_schema_and_known_feature_validation(parts,tmp_path):
    from training_r0.import_nominal_combat import convert,EXPECTED_FIELDS,EXPECTED_SCALES,RUNTIME
    raw=dict(schema='cr_nominal_static_combat_v1',source_libg_sha256=RUNTIME,fields=EXPECTED_FIELDS,
        scales=EXPECTED_SCALES,card_vocabulary=['x@26000021'],features=[[.4]+[0.]*9+[1]+[0]*9])
    source=tmp_path/'source.json';target=tmp_path/'table.json';source.write_text(json.dumps(raw))
    convert(source,target)
    assert json.loads(target.read_text())['cards']['26000021']=={'cost':4.}
    raw['source_libg_sha256']='a'*64;source.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='runtime'):convert(source,target)
