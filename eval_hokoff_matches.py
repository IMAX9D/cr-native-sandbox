"""Frozen BC checkpoint versus release: mirrored random decks, balanced sides."""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import random
import time
import torch
from expert_selfplay_v1.native_observation import NativeObservationEncoder
from hokoff_model.match_agent import (DEFAULT_CONTRACT, LIBG_SHA, HistoryOnlinePolicy,
    MatchAgent, file_sha, load_release, verify_runtime)
from hokoff_model.match_lock import match_lease
from hokoff_model.ppo_decks import opponent_deck_config
from hokoff_model.train_fixed import FixedConfig
from native_core.env import NativeRoyaleEnv
from policy_v1.train import load_checkpoint
from run_hokoff_fixed import advance_fixed, accepted_actions

ROOT = Path(__file__).resolve().parent


def load_candidate(path, contract, device):
    sha = file_sha(path)
    if sha in contract['allowed_weights']:
        return load_release(path, device=device)
    # Full checkpoints contain optimizer/RNG Python objects: only use your own checkpoints.
    saved = load_checkpoint(path)
    bound = saved.get('contract', {})
    if bound.get('manifest_sha256') != contract['source_manifest_sha256']:
        raise ValueError('candidate dataset differs from historical release')
    if saved['config'].get('architecture') != FixedConfig.architecture:
        raise ValueError('candidate is not fixed BC')
    config = FixedConfig(**saved['config'])
    if config.history_length != 4 or config.decision_period != 4:
        raise ValueError('candidate must use fixed4/history4')
    for key in ('card_vocab_size','ability_vocab_size','public_scalar_size','entity_numeric_size','grid_channels'):
        if getattr(config,key) != contract['encoder']['dimensions'][key]:
            raise ValueError('candidate encoder dimensions differ')
    table = json.loads((ROOT/'hokoff_model/combat_features.json').read_text())
    if config.combat_features != table['features']:
        raise ValueError('candidate combat table differs')
    model = HistoryOnlinePolicy(config)
    model.load_state_dict(saved['model'], strict=True)
    if not all(torch.isfinite(v).all() for v in model.state_dict().values() if v.is_floating_point()):
        raise ValueError('non-finite candidate')
    model.to(device).eval().requires_grad_(False)
    if file_sha(path) != sha: raise ValueError('candidate changed during load')
    return model, NativeObservationEncoder.from_manifest(contract['encoder']), dict(
        checkpoint_sha256=sha, training_step=saved['step'],
        source_step=bound.get('weights_restart',{}).get('source_step'),
        step_semantics=bound.get('weights_restart',{}).get('step_semantics','checkpoint_step'))


def schedule(fixture, pool, games, seed):
    rng=random.Random(seed); seen=set(); rows=[]
    champions=set(pool['champion_ids'])
    for i in range(games):
        while True:
            cards=rng.sample(pool['card_ids'],8)
            signature=tuple(sorted(cards))
            if signature not in seen and len(set(cards)&champions)<=1: break
        seen.add(signature)
        replay=deepcopy(fixture); replay['cmd']=[]; replay['rndSeed']=seed+i
        deck=deepcopy(replay['battle']['deck0'])
        deck['sp']=[dict(d=card,l=10) for card in cards]
        for side in (0,1): replay['battle']['deck'+str(side)]=deepcopy(deck)
        rows.append(dict(game=i+1,candidate_side=i%2,replay=replay))
    return rows


def result_name(terminal, side):
    if not terminal.get('terminated') or terminal.get('truncated'):
        raise ValueError('incomplete/truncated game cannot count towards win rate')
    outcome=terminal.get('outcome')
    if outcome not in ('side0_win','side1_win','draw'): raise ValueError('unknown terminal outcome')
    return 'draw' if outcome=='draw' else 'win' if outcome=='side%d_win'%side else 'loss'


def play(env, agents, replay, max_ticks, log, viewer=None):
    state=env.reset(deepcopy(replay),warmup_steps=0)
    tick=int(state['tick'])
    if not 0<=tick<100: raise ValueError('unexpected opening tick')
    for agent in agents: agent.reset(tick)
    counts=[0,0]
    while tick<max_ticks:
        if viewer is not None: viewer.show(state)
        actions=[]
        for agent in agents:
            action,_=agent.decide(state,env.decks,env)
            if action is not None:
                if tick<100: raise ValueError('action before opening')
                actions.append(action)
        # Both policies see the same pre-action state. Native receipts advance both histories.
        transition,advanced=advance_fixed(env,state,actions,period=1)
        terminal=transition['step']['episode']
        done=bool(terminal.get('terminated') or terminal.get('truncated'))
        after=int(terminal['terminal_tick']) if done else int(transition['state']['tick'])
        if actions or done:
            log.write(json.dumps(dict(tick=tick,after=after,actions=actions,
                receipt=transition['joint_action'],pending_followups=transition['pending_followups'],terminal=terminal if done else None))+'\n');log.flush()
        for agent in agents:
            agent.record_transition(tick,after,actions,transition['joint_action'],env.decks,terminal=done)
        accepted=accepted_actions(transition['joint_action'],actions,terminal_gate=done and advanced==0)
        for side in accepted: counts[side]+=1
        if done: return dict(terminal=terminal,actions_by_side=counts,tick=after)
        if after != tick+1: raise ValueError('unexpected tick advance')
        tick=after;state=transition['state']
        if tick%1000==0: print('  tick=%d actions=%s'%(tick,counts),flush=True)
    raise RuntimeError('tick limit reached without full terminal; not counted')


def write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def summarize(rows):
    n=len(rows);w=sum(r['result']=='win' for r in rows);d=sum(r['result']=='draw' for r in rows)
    return dict(completed=n,wins=w,losses=n-w-d,draws=d,
                win_rate=w/n if n else None,score_rate=(w+.5*d)/n if n else None,
                candidate_blue_games=sum(r['candidate_side']==0 for r in rows),
                candidate_red_games=sum(r['candidate_side']==1 for r in rows))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True,help='your trusted full BC checkpoint')
    p.add_argument('--opponent',type=Path,default=ROOT/'models/hokoff-bc-step1037042.pt')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--games',type=int,default=100);p.add_argument('--seed',type=int,default=20260912)
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=39031)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--gui',action='store_true',help='read-only live spectator; default is headless')
    p.add_argument('--max-ticks',type=int,default=10000)
    args=p.parse_args()
    if args.games<2 or args.games%2 or args.max_ticks<100: p.error('use a positive even game count and max-ticks >=100')
    torch.set_num_threads(1)
    contract=json.loads(DEFAULT_CONTRACT.read_text())
    candidate,encoder,ci=load_candidate(args.candidate,contract,args.device)
    opponent,_,oi=load_release(args.opponent,device=args.device)
    pool=opponent_deck_config('random',contract['encoder'])
    fixture=json.loads((ROOT/'hokoff_model/fixtures/hog_2_6.json').read_text())
    plan=schedule(fixture,pool,args.games,args.seed)
    run_contract=dict(kind='frozen_bc_mirrored_decks_v1',candidate=ci,opponent=oi,
        games=args.games,seed=args.seed,device=args.device,max_ticks=args.max_ticks,
        encoder_contract_sha256=file_sha(DEFAULT_CONTRACT),pool=pool,
        timing_threshold=.5,decision_ticks=4,deck_mode='unique_random_base_8_same_both_sides',plan=plan)
    with match_lease(args.host,args.port), NativeRoyaleEnv(host=args.host,port=args.port,timeout=30) as env:
        runtime=verify_runtime(env)
        args.output.mkdir(parents=True,exist_ok=True)
        contract_path=args.output/'contract.json'
        if contract_path.exists():
            if json.loads(contract_path.read_text()) != run_contract: raise ValueError('existing evaluation contract differs')
        else:
            if any(args.output.iterdir()): raise ValueError('output is not an evaluation directory')
            write_json(contract_path,run_contract)
        write_json(args.output/'runtime.json',runtime)
        completed=[]
        viewer=None
        if args.gui:
            from hokoff_model.eval_viewer import EvalViewer
            viewer=EvalViewer()
        try:
            for row in plan:
                target=args.output/('game-%03d.json'%row['game'])
                if target.exists():
                    saved=json.loads(target.read_text())
                    if saved['game']!=row['game'] or saved['candidate_side']!=row['candidate_side'] or saved['result']!=result_name(saved['terminal'],row['candidate_side']):
                        raise ValueError('saved game result differs')
                    completed.append(saved);continue
                side=row['candidate_side']
                agents=[MatchAgent(candidate if s==side else opponent,encoder,side=s,device=args.device) for s in (0,1)]
                if viewer is not None:
                    stats=summarize(completed)
                    viewer.label="第 %d/%d 场 · 新模型%s方 · %d胜 %d负 %d平"%(row["game"],args.games,"蓝" if side==0 else "红",stats["wins"],stats["losses"],stats["draws"])
                started=time.monotonic()
                with (args.output/('game-%03d-actions.jsonl'%row['game'])).open('w') as log:
                    result=play(env,agents,row['replay'],args.max_ticks,log,viewer)
                result.update(game=row['game'],candidate_side=side,result=result_name(result['terminal'],side),elapsed_seconds=time.monotonic()-started)
                write_json(target,result);completed.append(result)
                summary=dict(status='running',**summarize(completed))
                write_json(args.output/'summary.json',summary);print(json.dumps(summary),flush=True)
            if file_sha(args.candidate)!=ci['checkpoint_sha256'] or file_sha(args.opponent)!=oi['checkpoint_sha256']:
                raise ValueError('model file changed during evaluation')
            write_json(args.output/'summary.json',dict(status='complete',**summarize(completed)))
            if viewer is not None: viewer.finish('评估完成：'+json.dumps(summarize(completed),ensure_ascii=False))
        except BaseException as error:
            write_json(args.output/'summary.json',dict(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error),**summarize(completed)))
            raise


if __name__=='__main__': main()
