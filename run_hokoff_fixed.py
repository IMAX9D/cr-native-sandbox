#!/usr/bin/env python3
"""Run fixed4 BC on both sides of one native Worker; no optimizer or PPO update."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import time
import traceback
import torch
from policy_v1.data import digest
from native_core.env import NativeRoyaleEnv
from hokoff_model.live import FixedLiveAgent, NativeMaskProvider

LIBG_SHA256='fa6704b83cb9c5b8eecb7b56c9671b834d636a3a6d9ac446e698e1262dc246ba'


def advance_fixed(env,state,actions,period=4):
    """Keep the fixed observation grid even when a native terminal latch is late.

    Submit commands exactly once. Bounded follow-ups contain no commands and
    finish the remaining ticks, or return the authenticated early terminal.
    """
    tick=int(state['tick']);first_receipt=None;followups=[]
    result=None
    for attempt in range(3):
        previous_advance=0 if result is None else int(result['state']['tick'])-tick
        result=env.joint_training_transition(actions if attempt==0 else [],steps=period-previous_advance)
        episode=result['step']['episode']
        if first_receipt is None: first_receipt=deepcopy(result['joint_action'])
        else: followups.append(deepcopy(result['joint_action']))
        if episode.get('terminated') and episode.get('truncated'):
            raise RuntimeError('native episode cannot be both terminated and truncated')
        done=bool(episode.get('terminated') or episode.get('truncated'))
        end=int(episode['terminal_tick']) if done else int(result['state']['tick'])
        advanced=end-tick
        if not previous_advance <= advanced <= period:
            raise RuntimeError(f'invalid native tick advance: {tick} -> {end}')
        if done or advanced==period:
            result['joint_action']=first_receipt
            result['pending_followups']=followups
            return result,advanced
    raise RuntimeError('native tick stalled after bounded empty-action follow-ups')


def accepted_actions(receipt,expected,*,terminal_gate=False):
    rows=receipt.get('actions')
    if not isinstance(rows,list) or len(rows)!=len(expected):
        raise RuntimeError('native action receipt count differs from submitted commands')
    expected_by_side={a['side']:a for a in expected};accepted=set();seen=set()
    for row in rows:
        side=int(row['side'])
        if side not in expected_by_side or side in seen: raise RuntimeError('native receipt side mismatch')
        seen.add(side);result=row['result']
        if result.get('accepted'):
            command=expected_by_side[side]
            fields=('deck_index','x','y') if command.get('type')=='play' else ('entity_id',) if command.get('type')=='ability' else ()
            if any(result.get(key)!=command[key] for key in fields):
                raise RuntimeError('native accepted a different command than submitted')
            accepted.add(side)
        elif not (terminal_gate and result.get('result_code') in (3,4)):
            raise RuntimeError('native rejected masked action: '+json.dumps(row))
    return accepted


def write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def show(title,values):
    print('#'*72+'\n'+title.center(72),flush=True)
    for key,value in values.items():
        formatted=f'{value:.4f}' if isinstance(value,float) else str(value)
        print(f'{key:>30}: {formatted}',flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    base=Path('/home/lenovo/cr-data')
    p.add_argument('--checkpoint',type=Path,default=base/'runs/hokoff-fixed-p4/last.pt')
    p.add_argument('--manifest',type=Path,default=base/'expert-dataset/native-bc-v1/manifest.json')
    p.add_argument('--worker-dir',type=Path,default=Path('/data/local/tmp/cr-native-direct-0'))
    p.add_argument('--replay',type=Path,help='deck/arena fixture; defaults to worker bootstrap-replay.json; cmd is cleared')
    p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',type=int,default=39031)
    p.add_argument('--episodes',type=int,default=1)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--timing-threshold',type=float,default=.5)
    p.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--max-decisions',type=int,default=2500)
    p.add_argument('--log-every',type=int,default=100)
    p.add_argument('--timeout',type=float,default=30)
    p.add_argument('--output',type=Path)
    p.add_argument('--save-observations',action='store_true',help='also save compact pre-decision states for offline auditing')
    return p


def run(args):
    if min(args.episodes,args.max_decisions,args.log_every,args.cpu_threads)<1 or args.timeout<=0:
        raise ValueError('positive episode/decision/log/thread/timeout limits required')
    torch.set_num_threads(args.cpu_threads);torch.manual_seed(args.seed)
    agent=FixedLiveAgent.load(args.checkpoint,args.manifest,device=args.device,threshold=args.timing_threshold)
    libg=args.worker_dir/'libg.so'
    if digest(libg)!=LIBG_SHA256: raise ValueError('local worker libg does not match the frozen x86_64 runtime')
    replay_path=args.replay or args.worker_dir/'bootstrap-replay.json'
    fixture=NativeRoyaleEnv.read_replay(replay_path)
    fixture['cmd']=[]
    for side in (0,1): agent.encoder._deck(fixture['battle'][f'deck{side}']['sp'])
    output=args.output or Path('/home/lenovo/cr-data/runs')/('hokoff-fixed-live-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    output.mkdir(parents=True,exist_ok=False)
    contract=dict(kind='hokoff_fixed_live_bc_v1',**agent.identity,host=args.host,port=args.port,
                  device=args.device,seed=args.seed,episodes=args.episodes,
                  max_decisions=args.max_decisions,local_libg_sha256=digest(libg),
                  local_worker_dir=str(args.worker_dir.resolve()),replay_source=str(replay_path.resolve()),
                  replay_source_sha256=digest(replay_path),replay_commands_cleared=True,
                  position_mask='audited_native_probe_and_current_towers',skill_has_position=False,
                  recurrent_state='independent_per_side_reset_each_episode',
                  save_observations=args.save_observations,ppo_rollout=False)
    write_json(output/'contract.json',contract)
    summary=dict(status='running',completed_episodes=0,episodes=[],output=str(output))
    exit_code=0;started=time.perf_counter()
    observations=None
    try:
        with NativeRoyaleEnv(host=args.host,port=args.port,timeout=args.timeout) as env, (output/'decisions.jsonl').open('w') as log:
            ping=env._request({'op':'ping'});write_json(output/'worker_ping.json',ping)
            if args.save_observations: observations=(output/'observations.jsonl').open('w')
            for episode_index in range(args.episodes):
                replay=deepcopy(fixture);replay['rndSeed']=args.seed+episode_index
                write_json(output/f'episode-{episode_index:03d}-replay.json',replay)
                env.reset(replay,warmup_steps=100)
                state=env.observe_train()
                if int(state['tick'])!=100: raise RuntimeError('reset/warmup did not reach tick100')
                agent.reset();masks=NativeMaskProvider(env);counts=Counter()
                profile=Counter();episode_started=time.perf_counter();terminal=None
                for decision in range(1,args.max_decisions+1):
                    if observations is not None:
                        observations.write(json.dumps(dict(episode=episode_index,decision=decision,state=state))+'\n')
                    stage=time.perf_counter();actions,records,audit=agent.decide(state,env.decks,masks)
                    profile['inference_and_masks_seconds']+=time.perf_counter()-stage
                    stage=time.perf_counter();transition,advanced=advance_fixed(env,state,actions)
                    profile['native_transition_seconds']+=time.perf_counter()-stage
                    episode=transition['step']['episode']
                    done=bool(episode.get('terminated') or episode.get('truncated'))
                    row=dict(episode=episode_index,decision=decision,tick=state['tick'],advanced_ticks=advanced,
                             decisions=records,entity_audit=audit,native_receipt=transition['joint_action'],
                             pending_followups=transition['pending_followups'],terminal=episode if done else None)
                    # Persist the receipt before validating: errors remain inspectable.
                    log.write(json.dumps(row,ensure_ascii=False)+'\n')
                    accepted=accepted_actions(transition['joint_action'],actions,terminal_gate=done and advanced==0)
                    agent.record_accepted(actions,accepted,env.decks)
                    counts['attempted_actions']+=len(actions);counts['accepted_actions']+=len(accepted)
                    counts['native_ticks']+=advanced
                    for record in records:
                        counts[record['action']+'_decisions']+=1
                        if record['action']=='wait': counts[record['reason']]+=1
                    if decision%args.log_every==0 or done:
                        log.flush()
                        if observations is not None: observations.flush()
                        elapsed=time.perf_counter()-episode_started
                        show(f'BC episode {episode_index+1}/{args.episodes} | decision {decision}',
                             dict(tick=int(state['tick'])+advanced,accepted_actions=counts['accepted_actions'],
                                  attempted_actions=counts['attempted_actions'],skill_decisions=counts['ability_decisions'],
                                  wait_decisions=counts['wait_decisions'],elapsed_seconds=elapsed,
                                  native_ticks_per_second=counts['native_ticks']/max(elapsed,1e-6)))
                    if done:
                        terminal=episode
                        break
                    state=transition['state']
                result=dict(episode=episode_index,seed=args.seed+episode_index,decisions=decision,
                            **dict(counts),profile=dict(profile),elapsed_seconds=time.perf_counter()-episode_started,
                            terminal=terminal,rpc=env.client.profile_summary())
                summary['episodes'].append(result)
                write_json(output/f'episode-{episode_index:03d}-summary.json',result)
                if terminal is None:
                    raise RuntimeError('decision limit reached; partial match was saved, not counted as complete')
                if terminal.get('truncated'):
                    raise RuntimeError('native truncated the match; not counted as a completed game')
                if terminal.get('outcome') in (None,'ongoing'):
                    raise RuntimeError('terminal has no resolved outcome')
                summary['completed_episodes']+=1
            summary['status']='completed'
    except KeyboardInterrupt:
        summary['status']='interrupted';exit_code=130
    except Exception as error:
        summary.update(status='failed',error=f'{type(error).__name__}: {error}')
        (output/'error.txt').write_text(traceback.format_exc())
        exit_code=1
    finally:
        if observations is not None: observations.close()
        summary['elapsed_seconds']=time.perf_counter()-started
        summary['total_accepted_actions']=sum(e.get('accepted_actions',0) for e in summary['episodes'])
        summary['checkpoint_unchanged']=digest(args.checkpoint)==agent.identity['checkpoint_sha256']
        if not summary['checkpoint_unchanged']:summary['status']='failed';exit_code=1
        write_json(output/'summary.json',summary)
        show('BC run finished',dict(status=summary['status'],completed_episodes=summary['completed_episodes'],output=str(output)))
        if summary.get('error'): print(summary['error'],flush=True)
    return exit_code


if __name__=='__main__':raise SystemExit(run(parser().parse_args()))
