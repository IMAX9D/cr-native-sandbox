from dataclasses import replace
import pytest
import torch

from training_r0.replay_facts import actor_velocity,effect_kind,projectile_nominal_features,state_objects
from data_pipeline.r0_admission import current_contract,require_manifest,require_batch,R0AdmissionError
from training_r0.config import ModelConfig
from training_r0.catalog import CardVocabulary
from training_r0.observation import ObservationBuilder
from training_r0.model import R0Policy
from training_r0.synthetic import rich_public_frame


def test_velocity_uses_elapsed_tick_not_duplicate_observation():
    old=dict(id='0x1',x=1000,y=1000);new=dict(id='0x1',x=1500,y=1000)
    assert actor_velocity(new,old,tick=10,previous_tick=10,side=0)=={}
    assert actor_velocity(new,old,tick=10,previous_tick=9,side=0)['velocity_x']==10
    assert actor_velocity(new,old,tick=10,previous_tick=9,side=1)['velocity_x']==-10
    assert actor_velocity(dict(new,id='0x2'),old,tick=10,previous_tick=9,side=0)=={}


def test_raw_projectile_fields_are_not_overwritten_by_missing_unit_fields():
    r=state_objects(dict(projectiles=[dict(generation_key=4000001,id='x',x=1,y=2)],entities=[],effects=[]))
    assert r[4000001]['x']==1 and r[4000001]['hp']==-1


def test_effects_use_pinned_properties_not_name_substrings():
    assert effect_kind('Rage')=='speed'
    assert effect_kind('IceWizardSlowDown')=='slow'
    assert effect_kind('ZapFreeze')=='stun'
    assert effect_kind('UnknownRage')=='unknown'


def test_projectile_nominal_fallback_does_not_forge_scaled_damage():
    values=projectile_nominal_features('PROJECTILE.chr_wizardProjectile')
    assert values['projectile_homing']==1
    assert values['projectile_radius_over_5_tiles']==.3
    assert 'projectile_damage_over_5000' not in values


def test_old_unattributed_damage_does_not_poison_a_later_born_unit():
    from training_r0.lifecycle import DamageTracker,DamageEvent
    tracker=DamageTracker('x',continuous_damage_capture=True)
    tracker.spawn(1,0,observed_tick=0,birth_tick=0)
    tracker.record(DamageEvent('unknown',2,2,None,'unit',10))
    tracker.spawn(2,0,observed_tick=10,birth_tick=10)
    assert not tracker.snapshot(1,20).damage_complete
    assert tracker.snapshot(2,20).damage_complete


def test_current_contract_tracks_latest_r0_and_rejects_pilot():
    c=current_contract()
    assert c['decision_ticks']==5 and c['micro_actions']==2
    assert c['model_config']['allow_incomplete_capture'] is False
    assert len(c['features']['CHILD_FEATURE_NAMES'])==72
    with pytest.raises(R0AdmissionError,match='not_compiled_r0_sequences'):
        require_manifest(dict(training_ready=False,schema='cr-raw-pilot-summary.v1'),c)
    with pytest.raises(R0AdmissionError,match='no_training_shards'):
        require_manifest(dict(training_ready=True),c)


def test_real_production_model_guard_rejects_incomplete_feature_batch():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:
        vocabulary=CardVocabulary.from_native()
        config=ModelConfig(width=32,layers=1,heads=4,hidden=64,spatial_channels=16,learned_scatter_channels=8,native_graph_layers=1,allow_incomplete_capture=True)
        builder=ObservationBuilder(vocabulary,config);batch=builder.batch([rich_public_frame(builder)])
        model=R0Policy(vocabulary,replace(config,allow_incomplete_capture=False))
        with pytest.raises(ValueError,match='capture streams incomplete'):require_batch(batch,model)
        diagnostic=R0Policy(vocabulary,config)
        with pytest.raises(R0AdmissionError,match='diagnostic R0'):require_batch(batch,diagnostic)
    finally:torch.set_num_threads(previous)
