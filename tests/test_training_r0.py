from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from training_r0.actions import ActionSequence, DecisionEnvelope, ShadowLegality, TimingCertificate, command_plan
from training_r0.catalog import CardVocabulary
from training_r0.config import CELLS, ModelConfig, Temperatures
from training_r0.history import ConfirmedEvent, EventHistory, LabelValidity, trusted_window
from training_r0.model import R0Policy
from training_r0.observation import ObservationBuilder
from training_r0.synthetic import DECK, candidates, native_frame, public_frame
from training_r0.synthetic import rollout
from training_r0.learning import gate_imitation_loss, update_minibatch
from training_r0.config import TrainingConfig
from training_r0.rollout import LaneIdentity, advantages
from training_r0.session import FrozenPolicySession
from training_r0.rewards import RewardControl, TowerTotals


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def components():
    torch.manual_seed(71)
    vocabulary = CardVocabulary.from_native()
    config = ModelConfig(width=32,layers=1,heads=4,hidden=64,spatial_channels=16)
    builder = ObservationBuilder(vocabulary,config)
    model = R0Policy(vocabulary,config)
    return builder,model


def test_unknown_history_is_action_not_wait():
    history = EventHistory('game')
    event = ConfirmedEvent('command1',95,100,1,kind='ability')
    assert history.record(event)
    assert not history.record(event)
    assert history.snapshot(99) == ()
    assert history.snapshot(100) == (event,)
    with pytest.raises(ValueError):
        history.record(replace(event,kind='play'))
    labels = LabelValidity(action_occurred_known=True)
    assert not labels.candidate_known
    assert trusted_window(50,300) and not trusted_window(100,300)


def test_public_projection_ignores_enemy_private_fields(components):
    builder,model = components
    raw = native_frame()
    changed = deepcopy(raw)
    changed['players'][1]['elixir_raw'] = 10000
    changed['players'][1]['hand_deck_indices'] = [4,5,6,7]
    changed['players'][1]['next_deck_index'] = 0
    changed['entities'][1]['ability_cooldown_remaining_ms'] = 7777
    frames = [builder.from_native(item,episode_uid='game',actor_side=0,own_deck=DECK,candidates=candidates()) for item in (raw,changed)]
    batches = [builder.batch([frame]) for frame in frames]
    with torch.no_grad():
        first,second = (model.encode(batch) for batch in batches)
    assert torch.equal(first.policy,second.policy)
    assert torch.equal(first.value,second.value)


def test_no_empty_fallback_unknown_entities_retained(components):
    builder,_ = components
    raw = native_frame()
    raw['entity_count'] = 3
    with pytest.raises(ValueError,match='entity count'):
        builder.from_native(raw,episode_uid='g',actor_side=0,own_deck=DECK,candidates=candidates())
    raw['entity_count'] = 2
    raw['entities'][1]['card_id'] = 999999999
    frame = builder.from_native(raw,episode_uid='g',actor_side=0,own_deck=DECK,candidates=candidates())
    batch = builder.batch([frame])
    assert batch.entity_mask.sum() == 8
    assert batch.entity_tokens[0,1] == 1
    with pytest.raises(OverflowError):
        ObservationBuilder(builder.vocabulary,replace(builder.config,max_entities=1)).batch([frame])


def test_candidates_must_match_actual_own_hand_and_abilities(components):
    builder,_ = components
    wrong = list(candidates())
    wrong[0] = replace(wrong[0],card_id=DECK[-1])
    with pytest.raises(ValueError,match='current hand'):
        builder.from_native(native_frame(),episode_uid='g',actor_side=0,own_deck=DECK,candidates=wrong)
    wrong = list(candidates())
    wrong[-1] = replace(wrong[-1],source_entity=5000002)
    with pytest.raises(ValueError,match='own entity'):
        builder.from_native(native_frame(),episode_uid='g',actor_side=0,own_deck=DECK,candidates=wrong)


def test_current_scene_affects_context_and_uids_are_not_features(components):
    builder,model = components
    frame = public_frame(builder)
    moved = replace(frame,view=replace(frame.view,entities=tuple(replace(e,x=2000) if e.relation else e for e in frame.view.entities)))
    relabelled = replace(frame,candidates=tuple(replace(c,uid=c.uid+10000) for c in frame.candidates))
    with torch.no_grad():
        original,changed,renamed = (model.encode(builder.batch([item])) for item in (frame,moved,relabelled))
    assert not torch.allclose(original.policy,changed.policy,atol=1e-7,rtol=0)
    assert torch.equal(original.policy,renamed.policy)


def test_sample_and_teacher_force_joint_log_probability(components):
    builder,model = components
    batch = builder.batch([public_frame(builder),public_frame(builder,episode_uid='g2')])
    with torch.no_grad():
        context = model.encode(batch)
        for seed in range(12):
            sampled = model.decode(batch,context,generator=torch.Generator().manual_seed(seed))
            evaluated = model.decode(batch,context,forced=sampled.actions)
            assert torch.allclose(sampled.logp,evaluated.logp,atol=1e-6)
            assert torch.isfinite(evaluated.entropy).all()


def test_equal_offset_supported_but_native_plan_requires_certificate(components):
    builder,model = components
    frame = public_frame(builder)
    actions = ActionSequence(torch.tensor([2]),torch.tensor([[100,101]]),torch.tensor([[20,30]]),torch.tensor([[0,0]]))
    batch = builder.batch([frame])
    output = model(batch,forced=actions)
    assert output.actions.count.item() == 2
    decision = output.decision
    with pytest.raises(ValueError,match='not certified'):
        command_plan(frame,decision,0,certificate=None,runtime_sha256='a'*64,observation_schema_hash=builder.schema_hash)
    cert = TimingCertificate('a'*64,builder.schema_hash,1)
    with pytest.raises(ValueError,match='same-offset'):
        command_plan(frame,decision,0,certificate=cert,runtime_sha256='a'*64,observation_schema_hash=builder.schema_hash)
    cert = replace(cert,equal_offset_order_verified=True)
    plan = command_plan(frame,decision,0,certificate=cert,runtime_sha256='a'*64,observation_schema_hash=builder.schema_hash)
    assert [row['submit_tick'] for row in plan] == [101,101]
    assert [row['sequence_order'] for row in plan] == [0,1]


def test_cannot_reuse_candidate_or_overspend(components):
    builder,model = components
    batch = builder.batch([public_frame(builder)])
    duplicate = ActionSequence(torch.tensor([2]),torch.tensor([[100,100]]),torch.tensor([[20,30]]),torch.tensor([[0,1]]))
    with pytest.raises(ValueError,match='illegal'):
        model(batch,forced=duplicate)
    expensive = replace(duplicate,candidate_uid=torch.tensor([[100,102]]))
    with pytest.raises(ValueError,match='illegal'):
        model(batch,forced=expensive)


def test_no_legal_action_and_empty_scene_remain_finite(components):
    builder,model = components
    frame = public_frame(builder)
    frame = replace(frame,view=replace(frame.view,entities=(),towers=()),native_entity_count=0,candidates=())
    batch = builder.batch([frame])
    output = model(batch)
    assert output.actions.count.item() == 0
    assert output.logp.item() == 0
    assert torch.isfinite(output.value).all()
    loss = output.value.square().mean()-output.entropy.mean()*.01
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_gradient_flows_through_forced_action(components):
    builder,model = components
    batch = builder.batch([public_frame(builder)])
    actions = ActionSequence(torch.tensor([2]),torch.tensor([[100,101]]),torch.tensor([[20,30]]),torch.tensor([[0,1]]))
    output = model(batch,forced=actions)
    (-output.logp.mean()+output.value.square().mean()).backward()
    assert model.core.weight_ih.grad is not None
    assert model.entity_encoder[0].weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_gate_only_imitation_does_not_require_fabricated_candidate(components):
    builder,model = components
    batch = builder.batch([public_frame(builder)])
    context = model.encode(batch)
    loss = gate_imitation_loss(model,batch,context,torch.tensor([True]),torch.tensor([True]))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.gate_head.weight.grad.abs().sum() > 0


def test_time_chunks_carry_state_and_only_one_optimizer_step(components):
    builder,model = components
    segment = rollout(model,builder,steps=7,sides=(0,1))
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-5)
    original_step = optimizer.step
    calls = []
    def step(*args,**kwargs):
        calls.append(True)
        return original_step(*args,**kwargs)
    optimizer.step = step
    metrics = update_minibatch(model,optimizer,segment,expected_behavior=segment.behavior,config=TrainingConfig(tbptt_steps=3))
    assert metrics['updated']
    assert metrics['valid_decisions'] == 14
    assert metrics['tbptt_chunks'] == 3
    assert len(calls) == metrics['optimizer_steps'] == 1


def test_frozen_opponent_samples_and_cross_episode_lanes_rejected(components):
    builder,model = components
    segment = rollout(model,builder,steps=3)
    bad = replace(segment,lanes=(LaneIdentity('synthetic-game',0,'current_frozen',1),))
    with pytest.raises(ValueError,match='frozen opponent'):
        bad.validate()
    starts = segment.episode_start.clone()
    starts[1] = True
    with pytest.raises(ValueError,match='new logical lane'):
        replace(segment,episode_start=starts).validate()


def test_bootstrap_terminal_truncation_and_missing_value(components):
    builder,model = components
    segment = rollout(model,builder,steps=1)
    next_value = torch.full_like(segment.next_values,7.)
    segment = replace(segment,next_values=next_value)
    terminal,_ = advantages(segment,gamma=.9,gae_lambda=.98)
    truncated = replace(segment,terminated=torch.zeros_like(segment.terminated),truncated=segment.valid,
                        bootstrap_known=segment.valid)
    truncated.validate()
    adv,_ = advantages(truncated,gamma=.9,gae_lambda=.98)
    assert torch.allclose(adv-terminal,torch.tensor([[6.3]]),atol=1e-5)
    with pytest.raises(ValueError,match='bootstrap'):
        replace(truncated,bootstrap_known=torch.zeros_like(segment.valid)).validate()


def test_behavior_contract_and_kl_guard_do_not_publish_updates(components):
    builder,model = components
    segment = rollout(model,builder,steps=3)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-5)
    with pytest.raises(ValueError,match='contract mismatch'):
        update_minibatch(model,optimizer,segment,expected_behavior=replace(segment.behavior,temperatures=Temperatures(gate=.2)))
    old = model.core.weight_ih.detach().clone()
    inconsistent = replace(segment,logp=segment.logp-3)
    metrics = update_minibatch(model,optimizer,inconsistent,expected_behavior=segment.behavior)
    assert not metrics['updated'] and metrics['optimizer_steps'] == 0
    assert torch.equal(old,model.core.weight_ih)


def test_padding_not_counted_and_not_reset_in_middle(components):
    builder,model = components
    segment = rollout(model,builder,steps=4)
    valid = segment.valid.clone()
    valid[-1] = False
    terminal = segment.terminated.clone()
    terminal[-1] = False
    terminal[-2] = True
    segment = replace(segment,valid=valid,terminated=terminal,bootstrap_known=~terminal)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-5)
    metrics = update_minibatch(model,optimizer,segment,expected_behavior=segment.behavior)
    assert metrics['valid_decisions'] == 3


def test_frozen_sessions_isolate_episodes_ticks_and_learner_weights(components):
    builder,model = components
    session = FrozenPolicySession(model,max_sessions=2)
    first = builder.batch([public_frame(builder,episode_uid='A'),public_frame(builder,episode_uid='B')])
    output = session.act(first,sample=False)
    assert session.state_count == 2
    with pytest.raises(ValueError,match='inference tick'):
        session.act(first)
    with torch.no_grad():
        model.gate_head.bias[1] += 100
    another = FrozenPolicySession(model,max_sessions=2)
    assert another.weights_sha256 != session.weights_sha256
    next_batch = builder.batch([public_frame(builder,tick=105,episode_uid='A')])
    assert torch.isfinite(session.act(next_batch).logp).all()
    with pytest.raises(OverflowError):
        session.act(builder.batch([public_frame(builder,episode_uid='C')]))
    session.release_episode('B')
    assert session.state_count == 1
    assert torch.isfinite(output.value).all()


def test_opaque_candidate_reordering_preserves_recorded_action_probability(components):
    builder,model = components
    frame = public_frame(builder)
    action = ActionSequence(torch.tensor([1]),torch.tensor([[100,-1]]),torch.tensor([[20,-1]]),torch.tensor([[2,-1]]))
    with torch.no_grad():
        a = model(builder.batch([frame]),forced=action)
        b = model(builder.batch([replace(frame,candidates=tuple(reversed(frame.candidates)))]),forced=action)
    assert torch.allclose(a.logp,b.logp,atol=2e-5,rtol=0)


def test_shadow_geometry_is_conservative_when_unknown(components):
    builder,_ = components
    frame = public_frame(builder)
    changed = list(frame.candidates)
    changed[0] = replace(changed[0],is_building=True)
    changed[1] = replace(changed[1],is_building=True)
    shadow = ShadowLegality(builder.batch([replace(frame,candidates=tuple(changed))]))
    shadow.apply(torch.tensor([True]),torch.tensor([0]),torch.tensor([20]),torch.tensor([0]))
    assert not shadow.candidate_mask(1)[0,1]
    assert shadow.conservative_geometry[0]


def test_native_plan_checks_budget_and_duplicate_candidates(components):
    builder,_ = components
    frame = public_frame(builder)
    action = ActionSequence(torch.tensor([2]),torch.tensor([[100,102]]),torch.tensor([[20,30]]),torch.tensor([[0,1]]))
    cert = TimingCertificate('a'*64,builder.schema_hash,1,True)
    decision = DecisionEnvelope(action,((frame.episode_uid,0,100),),builder.schema_hash)
    with pytest.raises(ValueError,match='elixir'):
        command_plan(frame,decision,0,certificate=cert,runtime_sha256='a'*64,observation_schema_hash=builder.schema_hash)


def test_reward_terminal_not_chunk_and_no_overflow_bonus():
    control = RewardControl(TowerTotals(100,100),gamma_per_decision=1.)
    first = control.transition(TowerTotals(100,100),TowerTotals(100,80),elapsed_ticks=5,terminated=False)
    second = control.transition(TowerTotals(100,80),TowerTotals(90,0),elapsed_ticks=3,terminated=True,outcome=1)
    assert abs(first['total']+second['total']-1) < 1e-12
    assert first['tower_shaping'] > 0 and first['terminal'] == 0
    assert first['overflow'] == 0
    with pytest.raises(ValueError):
        control.transition(TowerTotals(100,100),TowerTotals(100,90),elapsed_ticks=5,terminated=False,outcome=0)


def test_future_events_rejected_and_history_known_bits_encoded(components):
    builder,_ = components
    frame = public_frame(builder)
    event = ConfirmedEvent('unknown-command',95,100,1,kind='ability')
    batch = builder.batch([replace(frame,events=(event,))])
    assert batch.event_kinds[0,0] == 3 and batch.event_tokens[0,0] == 1
    assert batch.event_features[0,0,-1] == 1
    with pytest.raises(ValueError,match='future event'):
        builder.batch([replace(frame,events=(replace(event,observed_tick=101),))])


def test_config_guard_and_elapsed_tick_integrity(components):
    with pytest.raises(ValueError):
        ModelConfig(width=31,heads=4)
    with pytest.raises(ValueError):
        Temperatures(gate=float('nan'))
    builder,model = components
    segment = rollout(model,builder,steps=2)
    elapsed = segment.elapsed_ticks.clone()
    elapsed[0] = 5.5
    with pytest.raises(ValueError,match='integral'):
        replace(segment,elapsed_ticks=elapsed).validate()


def test_stale_inference_cannot_be_relabelled_for_a_later_tick(components):
    builder,model = components
    frame = public_frame(builder)
    action = ActionSequence(torch.tensor([1]),torch.tensor([[100,-1]]),torch.tensor([[20,-1]]),torch.tensor([[0,-1]]))
    output = model(builder.batch([frame]),forced=action)
    later = public_frame(builder,tick=105)
    certificate = TimingCertificate('a'*64,builder.schema_hash,1,True)
    with pytest.raises(ValueError,match='stale'):
        command_plan(later,output.decision,0,certificate=certificate,runtime_sha256='a'*64,observation_schema_hash=builder.schema_hash)


def test_missing_spatial_label_is_not_silently_cell_zero(components):
    builder,model = components
    action = ActionSequence(torch.tensor([1]),torch.tensor([[100,-1]]),torch.tensor([[-1,-1]]),torch.tensor([[0,-1]]))
    with pytest.raises(ValueError,match='missing its target'):
        model(builder.batch([public_frame(builder)]),forced=action)
