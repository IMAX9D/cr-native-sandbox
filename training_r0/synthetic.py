"""Small synthetic observe_train_v1 fixtures, never evidence of native fidelity."""
from __future__ import annotations

from dataclasses import replace

import torch

from native_core.card_catalog import card_cost
from .config import CELLS, WIDTH
from .observation import Candidate, ObservationBuilder
from .session import weights_hash

DECK = (26000021,28000011,28000000,26000014,26000030,26000010,27000000,26000038)


def native_frame(tick: int = 100) -> dict:
    towers = []
    for side in (0,1):
        for role,lane,x,y in (('king',None,9000,3000),('princess','left',3500,6500),('princess','right',14500,6500)):
            towers.append(dict(side=side,type=role,lane=lane,x=x if side==0 else 17999-x,
                               y=y if side==0 else 31999-y,hp=3000,max_hp=3000))
    return dict(kind='libg_native_train_state_v1',schema_version=1,coherent=True,tick=tick,entity_count=2,
        players=[dict(side=side,elixir_raw=70000,hand_deck_indices=[0,1,2,3],next_deck_index=4,refill_timer=0) for side in (0,1)],
        entities=[dict(category=5000001,side=0,x=4000,y=8000,card_id=26000038,level=11,hp=1000,max_hp=2000,
                       ability_slot=1,ability_available=True,ability_mana_cost=2),
                  dict(category=5000002,side=1,x=13000,y=22000,card_id=26000021,level=11,hp=1500,max_hp=3000)],
        episode=dict(terminated=False,crowns=[0,0],commands_allowed=True,command_gate_code=0,
                     native_phase=dict(battle=1,logic=0,logic_substate=0,flag_1e9=0),crown_towers=towers))


def candidates(*, side=0) -> tuple[Candidate, ...]:
    mask = tuple(cell//WIDTH < 16 for cell in range(CELLS))
    result = [Candidate(100+slot,card,'play',float(card_cost(card)),True,'grid',mask,hand_slot=slot,form_flags=0)
              for slot,card in enumerate(DECK[:4])]
    if side == 0:
        result.append(Candidate(200,26000038,'ability',2.,True,'none',(False,)*CELLS,source_entity=5000001))
    return tuple(result)


def public_frame(builder: ObservationBuilder, *, tick=100, side=0, episode_uid='synthetic-game'):
    frame = builder.from_native(native_frame(tick),episode_uid=episode_uid,actor_side=side,own_deck=DECK,candidates=candidates(side=side))
    return replace(frame, source='synthetic_fixture_not_native_validation')


def rich_public_frame(builder: ObservationBuilder, *, tick=100, side=0, episode_uid='synthetic-game'):
    from .semantics import PublicSemantics,EntityDetail,PublicEffect,PublicRelation,OwnCommand
    raw=native_frame(tick)
    own=raw['entities'][side]
    raw['entities'].append({**own,'category':5000003,'x':own['x']+300,'hp':own['hp']-200})
    raw['entity_count']=3
    target=raw['entities'][1-side]['category']
    semantic=PublicSemantics(episode_uid,side,tick,
        entities=(EntityDetail(own['category'],'troop','synthetic-group','deployment',radius_tiles=.4,
                values=(('shield',200.),('velocity_x',1.5),('attack_cooldown_ms',500.))),
            EntityDetail(5000003,'troop','synthetic-group','deployment',half_extent_tiles=(.5,.75)),
            EntityDetail(target,'troop',radius_tiles=0.)),
        effects=(PublicEffect(own['category'],'slow',tick-1,magnitude=.5,remaining_ms=1000.),),
        relations=(PublicRelation(own['category'],target,'targets',tick-1),),
        previous=(OwnCommand('confirmed',tick-5,tick-4,'play',DECK[0],(.2,.25),4.),),
        pending=(OwnCommand('pending',tick,tick,'play',DECK[1],(.3,.25),2.),))
    frame=builder.from_native(raw,episode_uid=episode_uid,actor_side=side,own_deck=DECK,candidates=candidates(side=side),semantics=semantic)
    return replace(frame,source='synthetic_rich_fixture_not_native_validation')


@torch.no_grad()
def rollout(model, builder: ObservationBuilder, *, steps=6, sides=(0,), rich=False):
    from .rollout import BehaviorIdentity, LaneIdentity, RolloutSegment
    state = initial = model.initial_state(len(sides))
    observations,actions,logp,values = [],[],[],[]
    for step in range(steps):
        make_frame=rich_public_frame if rich else public_frame
        observation = builder.batch([make_frame(builder,tick=100+step*5,side=side) for side in sides])
        output = model(observation,state)
        observations.append(observation)
        actions.append(output.actions)
        logp.append(output.logp.detach().clone())
        values.append(output.value.detach().clone())
        state = output.next_state.detach()
    values = torch.stack(values)
    rewards = torch.zeros_like(values)
    rewards[-1] = torch.tensor([1. if side == 0 else -1. for side in sides])
    valid = torch.ones(steps,len(sides),dtype=torch.bool)
    terminal = torch.zeros_like(valid)
    terminal[-1] = True
    starts = torch.zeros_like(valid)
    starts[0] = True
    return RolloutSegment(tuple(observations),tuple(actions),
        tuple(LaneIdentity('synthetic-game',side,'current_current') for side in sides),
        BehaviorIdentity('synthetic-policy',0,weights_hash(model),'a'*64,builder.schema_hash,model.model_schema_hash),
        initial,0,torch.stack(logp),values,torch.cat((values[1:],torch.zeros_like(values[:1]))),
        rewards,torch.full_like(values,5.),valid,starts,terminal,torch.zeros_like(valid),~terminal)
