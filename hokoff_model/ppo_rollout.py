"""Complete native self-play games with exact stochastic behavior probabilities."""
from collections import Counter
from copy import deepcopy
import json
import math
import numpy as np
import torch
from expert_selfplay_v1.gae import variable_time_gae
from training.schema import DefensiveTowerReward
from run_hokoff_fixed import advance_fixed,accepted_actions,show
from .live import NativeMaskProvider
from .ppo_actions import native_action


def advance_with_elixir_ticks(env,raw,commands,period=4):
    """Inspect native pre-tick elixir without extra policy/LSTM decisions.

    Full means >=100000 raw (10 elixir). Commands execute only on the first
    tick. Reuse the existing bounded terminal-latch handling for each tick.
    """
    current=raw;first_receipt=None;elixir_ticks=[]
    for offset in range(period):
        players={int(p['side']):p for p in current['players']}
        if set(players)!={0,1}:raise RuntimeError('missing per-side elixir state')
        elixir=[players[s]['elixir_raw'] for s in (0,1)]
        if any(not isinstance(v,int) or v<0 for v in elixir):
            raise RuntimeError('invalid native elixir_raw')
        submitted=commands if offset==0 else []
        transition,advanced=advance_fixed(env,current,submitted,period=1)
        episode=transition['step']['episode']
        accepted_actions(transition['joint_action'],submitted,
                         terminal_gate=bool(episode.get('terminated')) and advanced==0)
        if first_receipt is None:first_receipt=deepcopy(transition['joint_action'])
        if advanced:elixir_ticks.append(elixir)
        if episode.get('terminated') or episode.get('truncated'):break
        current=transition['state']
    transition['joint_action']=first_receipt
    return transition,len(elixir_ticks),elixir_ticks


def full_elixir_idle_ticks(elixir_ticks,commands,accepted):
    """A successful card play exempts its execution tick, for that side only.

    An ability is not a card play; its cost lowers elixir for following ticks.
    A mid-interval refill is charged only from its first full pre-tick state.
    """
    deployed={a['side'] for a in commands if a['type']=='play' and a['side'] in accepted}
    full={s:sum(values[s]>=100000 for values in elixir_ticks) for s in (0,1)}
    idle={s:full[s]-int(bool(elixir_ticks) and elixir_ticks[0][s]>=100000 and s in deployed)
          for s in (0,1)}
    return full,idle


def episode_stream(agent,env,replay,*,learner_side,log=None,episode_id=0,max_decisions=2500,full_elixir_penalty_per_tick=0.,opponent_number=None,progress=True):
    if not math.isfinite(full_elixir_penalty_per_tick) or full_elixir_penalty_per_tick<0:
        raise ValueError("full elixir penalty must be finite and nonnegative")
    env.reset(replay,warmup_steps=100);raw=env.observe_train()
    if raw['tick']!=100:raise RuntimeError('native reset did not start at tick100')
    agent.reset();provider=NativeMaskProvider(env);reward=DefensiveTowerReward()
    rows=[];counts=Counter();terminal=None;first_card_tick=None
    for decision in range(max_decisions):
        state,encoded,batch=agent.observations(raw,env.decks)
        legal=[provider.for_side(state,s,env.decks[s],encoded.ability_entity_keys[s],
                                  encoded.ability_mask[s,0].numpy()) for s in (0,1)]
        masks={k:torch.from_numpy(np.stack([getattr(m,k) for m in legal]))
               for k in ('cards','positions','abilities')}
        prediction=yield dict(batch=batch,hidden=agent.hidden,masks=masks,learner_side=learner_side)
        agent.hidden=prediction['hidden'];agent.last_tick=state.tick
        indices=prediction['indices'];probabilities=prediction['probabilities']
        features=prediction['features'];value=prediction['value']
        actions=[native_action(indices[s],s,legal[s]) for s in (0,1)]
        commands=[a for a in actions if a is not None]
        if full_elixir_penalty_per_tick:
            transition,advanced,elixir_ticks=advance_with_elixir_ticks(env,raw,commands)
        else:
            transition,advanced=advance_fixed(env,raw,commands);elixir_ticks=None
        episode=transition['step']['episode']
        done=bool(episode.get('terminated'))
        if episode.get('truncated'):raise RuntimeError('truncated native game is not admitted to PPO')
        accepted=accepted_actions(transition['joint_action'],commands,terminal_gate=done and advanced==0)
        if advanced==0 and accepted:raise RuntimeError('accepted action at zero-tick terminal is ambiguous for PPO')
        agent.record_accepted(commands,accepted,env.decks)
        terminal_rewards={s:float(episode['rewards'][s]) for s in (0,1)} if done else {0:0.,1:0.}
        current={'episode':episode} if done else transition['state']
        if not all(s.get('episode',{}).get('tower_snapshot_complete') is True for s in (raw,current)):
            raise RuntimeError('incomplete tower snapshot cannot supply PPO rewards')
        base_rewards=reward.transition(raw,current,terminal_rewards=terminal_rewards,done=done)
        full,idle=full_elixir_idle_ticks(elixir_ticks or [],commands,accepted)
        penalties={s:-full_elixir_penalty_per_tick*idle[s] for s in (0,1)}
        rewards={s:base_rewards[s]+penalties[s] for s in (0,1)}
        if log is not None:
            log.write(json.dumps(dict(episode=episode_id,learner_side=learner_side,opponent_number=opponent_number,tick=state.tick,
                advanced_ticks=advanced,action_indices=indices.tolist(),old_log_prob=probabilities.tolist(),
                commands=commands,receipt=transition['joint_action'],rewards=rewards,
                base_rewards=base_rewards,full_elixir_penalties=penalties,
                elixir_before_ticks=elixir_ticks,full_elixir_idle_ticks=idle if elixir_ticks is not None else None,
                terminal=episode if done else None))+'\n')
        if advanced==0:
            if not done or not rows:raise RuntimeError('zero-tick transition cannot form a PPO sample')
            rows[-1]['reward']+=rewards[learner_side];rows[-1]['terminated']=True
            rows[-1]['base_reward']+=base_rewards[learner_side]
        else:
            row={k:v[learner_side].detach().cpu().clone() for k,v in dict(features,**masks).items()}
            row.update(action=int(indices[learner_side]),old_log_prob=float(probabilities[learner_side]),
                       value=value,reward=rewards[learner_side],delta_ticks=advanced,terminated=done,
                       base_reward=base_rewards[learner_side],full_elixir_penalty=penalties[learner_side],
                       full_elixir_idle_ticks=idle[learner_side] if elixir_ticks is not None else -1)
            rows.append(row)
        counts['attempted_actions']+=len(commands);counts['accepted_actions']+=len(accepted)
        counts['ability_actions']+=sum(a['type']=='ability' for a in commands)
        counts['native_ticks']+=advanced
        counts['full_elixir_ticks']+=full[learner_side]
        counts['full_elixir_idle_ticks']+=idle[learner_side]
        learner_play=any(a['type']=='play' and a['side']==learner_side and a['side'] in accepted for a in commands)
        counts['learner_card_plays']+=int(learner_play)
        if learner_play and first_card_tick is None:first_card_tick=state.tick
        if progress and (decision+1)%200==0:
            show('PPO collection progress',dict(episode=episode_id,decisions=decision+1,
                 tick=state.tick+advanced,accepted_actions=counts['accepted_actions']))
            if log is not None:log.flush()
        if done:
            if episode.get('outcome') in (None,'ongoing'):raise RuntimeError('unresolved native terminal')
            terminal=episode;break
        raw=transition['state']
    if terminal is None:raise RuntimeError('decision limit reached; incomplete rollout discarded')
    advantages,returns=variable_time_gae(np.array([r['reward'] for r in rows]),
        np.array([r['value'] for r in rows]),np.array([r['terminated'] for r in rows]),
        np.array([r['delta_ticks'] for r in rows]),bootstrap_value=0.)
    data={k:torch.stack([r[k] for r in rows]) for k in
          ('context','hand_tokens','ability_tokens','cards','positions','abilities')}
    for k in ('action','old_log_prob','reward','delta_ticks','terminated','value',
              'base_reward','full_elixir_penalty','full_elixir_idle_ticks'):
        data[k]=torch.tensor([r[k] for r in rows])
    data.update(advantage=torch.from_numpy(advantages),**{'return':torch.from_numpy(returns)})
    return data,dict(episode=episode_id,learner_side=learner_side,opponent_number=opponent_number,decisions=len(rows),
                     reward_sum=float(data['reward'].sum()),base_reward_sum=float(data['base_reward'].sum()),
                     full_elixir_penalty_sum=float(data['full_elixir_penalty'].sum()),
                     first_card_tick=first_card_tick,**dict(counts),
                     full_elixir_ticks_measured=bool(full_elixir_penalty_per_tick),terminal=terminal)


def collect_episode(agent,trainer,env,replay,*,learner_side,generator,opponent=None,**kwargs):
    """Single-env compatibility adapter; shares all transitions with vector mode."""
    from .ppo_vector import infer_requests
    stream=episode_stream(agent,env,replay,learner_side=learner_side,**kwargs)
    try:
        request=next(stream)
        while True:
            prediction=infer_requests(agent.model,trainer,[request],[generator],opponent)[0]
            request=stream.send(prediction)
    except StopIteration as done:return done.value
    finally:stream.close()
