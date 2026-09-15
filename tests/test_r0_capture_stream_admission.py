"""Synthetic regression checks for raw-stream semantics, not admission proof."""
from copy import deepcopy

import pytest

from training_r0.lifecycle import DamageTracker,DamageEvent
from training_r0.replay_adapter import ReplayFeatureAssembler


def assembler():
    cards=[26000000,26000001,26000002,26000003,26000004,26000005,26000006,26000007]
    replay={'battle':{'gamemode':72000006,**{f'deck{s}':{'sp':[{'d':c,'l':10} for c in cards]} for s in (0,1)}}}
    return ReplayFeatureAssembler(replay,'synthetic')


def frame(tick):
    entities=[dict(id=f'0x{i}',category=5000000+i,kind=15,side=i-1,x=5000,y=10000+i*1000,card_id=26000000,hp=100,max_hp=100) for i in (1,2)]
    details=[dict(key=e['category'],native_data_id=34000000,native_name='Knight',created_tick_candidate=0) for e in entities]
    return dict(state=dict(tick=tick,coherent=True,entities=entities,effects=[],projectiles=[],episode=dict(crown_towers=[])),
        details=dict(tick=tick,objects=details),telemetry=dict(epoch=1,rejected=0,unattributed=0,gap=False,events=[],
            lifecycle=dict(events=[],gap=False,rejected=0)),masks=[])


def damage(sequence=1,tick=2):
    return dict(sequence=sequence,tick=tick,source_key=5000001,attacker_key=5000001,target_key=5000002,
        source_side=0,target_side=1,target_kind=15,effective_damage=12,lethal=False)


def test_rejected_damage_receipt_does_not_mutate_or_hide_retry():
    tracker=DamageTracker('x');tracker.spawn(1,0,observed_tick=0,birth_tick=0)
    event=DamageEvent('invalid',1,1,1,'unit',4,confirmed_nonunit_source=True)
    for _ in range(2):
        with pytest.raises(ValueError,match='nonunit source'):tracker.record(event)
        assert tracker.events=={} and tracker._by_source=={} and tracker._unattributed==[]


def test_middle_tick_damage_and_heal_reach_next_decision_without_invented_healer():
    a=assembler();snapshots={}
    a._actor=lambda f,*args:snapshots.setdefault((f['state']['tick'],args[-1]),tuple(deepcopy(a.event_window)))
    for tick in range(11):
        raw=frame(tick)
        if tick==2:raw['telemetry']['events']=[damage()]
        if tick==3:raw['telemetry']['lifecycle']['events']=[dict(sequence=1,operation=3,key=5000002,side=1,heal_amount=17,source_candidate_key=5000001)]
        a.ingest(raw)
    assert {e['kind'] for e in snapshots[5,0]}=={'damage','heal'}
    heal=next(e for e in snapshots[5,0] if e['kind']=='heal')
    assert heal['source'] is None and heal['target']==5000002 and heal['amount']==17
    assert snapshots[10,0]==()


def test_post_command_duplicate_tick_does_not_erase_next_decision_event():
    a=assembler();snapshots={}
    a._actor=lambda f,*args:snapshots.setdefault((f['state']['tick'],args[-1]),tuple(deepcopy(a.event_window)))
    for tick in range(6):a.ingest(frame(tick))
    duplicate=frame(5);duplicate['telemetry']['events']=[damage(tick=5)]
    assert a.ingest(duplicate)==[]
    for tick in range(6,11):a.ingest(frame(tick))
    assert snapshots[5,0]==()
    assert snapshots[10,0][0]['tick']==6


def capability_frame():
    return dict(state=dict(tick=0,coherent=True,players=[dict(side=0,hand_deck_indices=[0,1,2,-1]),dict(side=1,hand_deck_indices=[0,1,2,3])],
        episode=dict(tower_snapshot_complete=True,result_source='native_logic_terminal_and_crown_tower_entities'),effects_classified=True),
        details=dict(valid=True),telemetry=dict(enabled=True),masks=[dict(side=0,deck_index=i) for i in range(3)])


def test_dealing_empty_slot_is_not_a_missing_probe_but_missing_valid_slot_is():
    a=assembler();raw=capability_frame()
    capabilities,_=a._capabilities(raw,{}, {},[],0)
    assert 'card_runtime' in capabilities
    raw['masks'].pop()
    capabilities,reasons=a._capabilities(raw,{}, {},[],0)
    assert 'card_runtime' not in capabilities and 'card_runtime' in reasons


def test_unpinned_identity_cannot_certify_static_resources_or_timeline():
    a=assembler();raw=capability_frame()
    capabilities,reasons=a._capabilities(raw,{}, {},[],0)
    assert not {'archetypes','geometry','match_runtime'} & set(capabilities)
    assert a._match_features(raw['state'])=={}


def test_legacy_area_classification_requires_all_three_native_proofs():
    a=assembler();a.capture_identity={'libg_sha256':a.timelines['runtime_sha256']}
    raw=capability_frame();raw['state'].update(effects_classified=False,unclassified_effect_count=1,
        effects=[dict(category=3000001,vtable_rva='0x19691f8')],projectiles=[])
    details={3000001:dict(native_data_id=22000008,component_count=0)}
    capabilities,_=a._capabilities(raw,{},details,[],0)
    assert 'effects' in capabilities
    for field,value in [('vtable_rva','0xBAD'),('category',3000002)]:
        changed=deepcopy(raw);changed['state']['effects'][0][field]=value
        capabilities,_=a._capabilities(changed,{},details,[],0)
        assert 'effects' not in capabilities


def test_public_card_runtime_does_not_consume_enemy_hand_or_elixir():
    a=assembler();state=dict(entities=[],players=[dict(side=0),dict(side=1,elixir_raw=100000,hand_deck_indices=[0,1,2,3])])
    before=a._card_runtime(state,0,0)
    state['players'][1].update(elixir_raw=0,hand_deck_indices=[4,5,6,7])
    assert a._card_runtime(state,0,0)==before


def test_pinned_timeline_transition_uses_resource_boundaries():
    a=assembler();a.capture_identity={'libg_sha256':a.timelines['runtime_sha256']}
    assert a._match_features(dict(tick=2399))['elixir_multiplier_over_3']==pytest.approx(1/3)
    assert a._match_features(dict(tick=2400))['elixir_multiplier_over_3']==pytest.approx(2/3)
    assert a._match_features(dict(tick=3599))['overtime_or_sudden_death']==0
    assert a._match_features(dict(tick=3600))['overtime_or_sudden_death']==1
    # A runtime snapshot beyond the finite gameplay schedule is not assumed to
    # be a verified tiebreak merely because a timestamp passed the boundary.
    assert 'tiebreak' not in a._match_features(dict(tick=6000))


def test_verified_native_projectile_extent_is_preserved_without_changing_legacy():
    from training_r0.synthetic import native_frame
    from training_r0.native_projection import normalize_native_state,actor_projection
    from expert_v1.tick_store_v1.schema import normalize_native_state as legacy_normalize
    raw=native_frame();raw['entities'].append(dict(category=4000001,vtable_rva='0x1969b38',side=0,x=18142,y=20689))
    raw['entity_count']=len(raw['entities'])
    with pytest.raises(ValueError,match='outside native arena'):legacy_normalize(raw)
    state=normalize_native_state(raw)
    own=next(e for e in actor_projection(state,actor_side=0).entities if e.key==4000001)
    enemy=next(e for e in actor_projection(state,actor_side=1).entities if e.key==4000001)
    assert own.x==18142 and enemy.x==-143 and enemy.y==11310
    raw['entities'][-1]['vtable_rva']='0xBAD'
    with pytest.raises(ValueError,match='unverified'):normalize_native_state(raw)


def test_offboard_projectile_remains_entity_token_but_not_an_arena_cell():
    import torch
    from training_r0.synthetic import native_frame,candidates,DECK
    from training_r0.config import ModelConfig
    from training_r0.catalog import CardVocabulary
    from training_r0.observation import ObservationBuilder
    raw=native_frame();builder=ObservationBuilder(CardVocabulary.from_native(),ModelConfig(allow_incomplete_capture=True))
    base=builder.batch([builder.from_native(raw,episode_uid='x',actor_side=0,own_deck=DECK,candidates=candidates())])
    raw['entities'].append(dict(category=4000001,vtable_rva='0x1969b38',side=0,x=18142,y=20689))
    raw['entity_count']=len(raw['entities'])
    updated=builder.batch([builder.from_native(raw,episode_uid='x',actor_side=0,own_deck=DECK,candidates=candidates())])
    assert updated.entity_mask.sum()==base.entity_mask.sum()+1
    assert (updated.entity_positions[...,0]>1).any()
    assert torch.equal(base.grid,updated.grid)
    assert torch.equal(base.semantic.planes,updated.semantic.planes)


def test_capture_entry_refuses_implicit_diagnostic_projection(tmp_path,monkeypatch):
    """Production capture must not silently fall back to a projected ruleset."""
    import sys
    from data_pipeline import pilot100
    manifest=tmp_path/'manifest.json'
    base=['pilot100','capture','--output',str(tmp_path),'--bundle-manifest',str(manifest)]
    monkeypatch.setattr(sys,'argv',base)
    with pytest.raises(SystemExit) as refused:
        pilot100.main()
    assert refused.value.code==2, "capture without the processing contract must fail closed"
    # The same invocation is allowed once the operator asks for it explicitly.
    monkeypatch.setattr(sys,'argv',base+['--allow-diagnostic-projection'])
    with pytest.raises(FileNotFoundError):
        pilot100.main()


def test_creation_context_supplies_lineage_only_with_native_evidence():
    """An effect with no parent slot takes its lineage from the native spawn
    context, and that context is rejected when the owner side disagrees."""
    a=assembler()
    for tick in range(5):
        raw=frame(tick)
        if tick==1:
            raw['telemetry']['lifecycle']['events']=[dict(sequence=1,operation=1,key=3000001,side=0,tick=1,
                parent_key=0,command_sequence=0,birth_tick=-1,native_data_id=22000001,
                context_caller='0xf821f8',context_key=5000001)]
        if tick==2:
            # A context entity on the other side must not become a parent.
            raw['telemetry']['lifecycle']['events']=[dict(sequence=2,operation=1,key=3000002,side=0,tick=2,
                parent_key=0,command_sequence=0,birth_tick=-1,native_data_id=22000001,
                context_caller='0xf821f8',context_key=5000002)]
        if tick==3:
            # No native context at all: the lineage must stay unresolved.
            raw['telemetry']['lifecycle']['events']=[dict(sequence=3,operation=1,key=3000003,side=0,tick=3,
                parent_key=0,command_sequence=0,birth_tick=-1,native_data_id=22000001)]
        if tick==4:
            raw['telemetry']['events']=[dict(damage(sequence=1,tick=4),source_key=3000003,attacker_key=3000003)]
        a.ingest(raw)
    assert a.damage.births[3000001].parent_key==5000001
    assert a.issues['native_creation_context_parents']==1
    assert a.damage.births[3000002].parent_key is None
    assert a.issues['native_creation_context_owner_mismatch']==1
    assert a.damage.births[3000003].parent_key is None
    # An effect with no native lineage stays uncredited instead of being
    # attached to a unit by proximity.
    event=next(e for e in a.damage.events.values() if e.source_key==3000003)
    assert a.damage.credited_source(event) is None
    assert event in a.damage._unattributed


def test_destroyed_tower_keeps_static_identity_not_stale_runtime_values():
    from training_r0.synthetic import native_frame
    a=assembler();state=native_frame(100);state['entities']=[];state['entity_count']=0
    prior=deepcopy(state['episode']['crown_towers'][0]);state['episode']['crown_towers'][0].update(hp=0,destroyed=True)
    a.tower_native[5000100]=prior
    a.tower_details[5000100]=dict(native_data_id=35000000,native_name='KingTower',builtin_shield=999,
        phase_runtime=dict(load_remaining_ms=1234))
    a.damage.spawn(5000100,0,observed_tick=0,birth_tick=0)
    raw=dict(state=state,details=dict(valid=True),masks=[],telemetry=dict(enabled=True))
    a._actor(raw,{}, {},{},0)
    dead=next(row for row in a.last_inputs[0].semantics.entities if row.key==0)
    assert dead.kind=='king' and dead.archetype=='BUILDING.KingTower'
    assert dict(dead.tower_features)=={'active':0.,'has_visible_target':0.}
    assert dict(dead.values)=={} and dead.damage.damage_complete
