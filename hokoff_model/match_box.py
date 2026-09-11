"""Human (blue) vs a pinned BC Release (red), on the native match GUI."""
from __future__ import annotations
import argparse
import os
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid
import torch

from native_core.env import NativeRoyaleEnv, CARD_NAMES
from .match_agent import MatchAgent, load_release, verify_runtime, DEFAULT_CONTRACT
from .match_lock import match_lease

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHT = PROJECT/'models'/'hokoff-bc-step1037042.pt'
DEFAULT_REPLAY = PROJECT/'examples'/'hog-2.6-evo-hero.json'


def prime(agent,env,replay):
    if replay.get('cmd'): raise ValueError('match box requires a fresh battle, not a replay with past commands')
    identity = verify_runtime(env)
    state = env.reset(deepcopy(replay),warmup_steps=0)
    agent.reset(int(state['tick']))
    if not 0 <= int(state['tick']) < 100:
        raise ValueError('reset must return the fresh opening before match-box controls unlock')
    # The frozen host bootstraps at Tick 10. Do not invent Tick 0 observations;
    # align from the actual native clock. The adapter, not the engine flag,
    # keeps all actions disabled until Tick 100 (same fixed-policy contract).
    if int(state['tick']) % 4:
        before=int(state['tick']);steps=4-before%4
        result=env.joint_transition([],steps=steps);state=result['state']
        agent.record_transition(before,int(state['tick']),[],result['joint_action'],env.decks)
    while int(state['tick']) < 100:
        before = int(state['tick'])
        action,_ = agent.decide(state,env.decks,env)
        if action is not None: raise ValueError('native accepted command gate before battle opening')
        result = env.joint_transition([],steps=4)
        state = result['state']
        agent.record_transition(before,int(state['tick']),[],result['joint_action'],env.decks)
    return state,identity


def smoke(env,model,encoder,replay,*,ticks=160):
    """Short real-native checks, not the user's two interactive matches."""
    results=[]
    for seed in (113,227):
        template=deepcopy(replay);template['rndSeed']=seed
        agent=MatchAgent(model,encoder,side=1,device=next(model.parameters()).device)
        state,identity=prime(agent,env,template)
        assert agent.history.accepted_plays == [0,0]
        human_sent=ai_sent=0;maximum_entities=0;maximum_history=0
        for index in range(ticks):
            before=int(state['tick'])
            action,audit=agent.decide(state,env.decks,env)
            actions=[]
            if action is not None: actions.append(action)
            if index in (0,60,120):
                player=next(p for p in state['players'] if p['side']==0)
                # Choose a real legal in-hand card using native masks, no debug resource edits.
                for raw_index in player['hand_deck_indices']:
                    if int(raw_index)<0: continue
                    probe=env.probe_grid(side=0,deck_index=int(raw_index))
                    if int(player['elixir_raw']) < int(probe.get('card_cost_raw',10**9)): continue
                    choices=[(r,c) for r,row in enumerate(probe['rows']) for c,value in enumerate(row)
                             if value=='1' and 2<=r<=12 and 2<=c<=15]
                    if choices:
                        r,c=choices[len(choices)//2]
                        actions.insert(0,dict(side=0,deck_index=int(raw_index),x=c*1000+500,y=r*1000+500))
                        break
            result=env.joint_transition(actions,steps=1)
            episode=result['step']['episode'];done=bool(episode.get('terminated') or episode.get('truncated'))
            after=int(episode['terminal_tick']) if done else int(result['state']['tick'])
            agent.record_transition(before,after,actions,result['joint_action'],env.decks,terminal=done)
            for item in result['joint_action']['actions']:
                if not item['result']['accepted']: raise RuntimeError('native rejected smoke action: '+str(item))
                if item['side']==0: human_sent+=1
                else: ai_sent+=1
            maximum_entities=max(maximum_entities,int(agent.last_audit.get('encoded_entities',0)))
            maximum_history=max(maximum_history,int(agent.last_audit.get('history_slots',0)))
            if done:break
            state=result['state']
        results.append(dict(seed=seed,tick=after,human_actions=human_sent,ai_actions=ai_sent,
                     dynamic_entities_max=maximum_entities,history_slots_max=maximum_history,
                     confirmed_history=list(agent.history.accepted_plays),decisions=agent.decisions,
                     hidden_nonzero=bool(torch.count_nonzero(agent.hidden[0])),runtime=identity))
        assert human_sent>0 and maximum_entities>0 and maximum_history>0
        assert results[-1]['hidden_nonzero']
    return dict(passed=True,matches=results,scope='short native integration smoke; not win-rate evaluation')


def open_box(env,model,encoder,metadata,checkpoint,replay_path,session_root):
    import tkinter as tk
    from tkinter import ttk
    from native_core.human_vs_ai import HumanVsAiGui

    class MatchBox(HumanVsAiGui):
        def __init__(self,root):
            self.agent=MatchAgent(model,encoder,side=1,device=next(model.parameters()).device)
            self.session_root=session_root
            self.native_identity={}
            self.last_error=None
            self.ready=False
            super().__init__(root,env,replay_path,checkpoint=checkpoint,model=model,
                             model_meta=metadata,device=next(model.parameters()).device,
                             policy_seed=113,autostart=False)
            root.title(f"CR 对局盒子 · 你（蓝）vs BC {metadata['training_step']:,}步（红）")
            control=ttk.Frame(root,padding=6);control.pack(side='bottom',fill='x')
            self.play_button=ttk.Button(control,text='开始对局 / 暂停',command=self.toggle_play)
            self.play_button.pack(side='left')
            ttk.Label(control,text='每200ms决策 · 原始阈值0.5 · 选蓝方卡牌后点击战场 · 重置开始下一局').pack(side='left',padx=10)
            self.root.after(1000,self.write_status)

        def _reset_native_battle(self):
            if self.state is not None and self.session_path is None:
                self.save_session(partial=not bool(self.state.get('episode',{}).get('terminated')))
            self.ready=False
            self.last_error=None
            self.loop_generation+=1;self.running=False
            self.state,self.native_identity=prime(self.agent,self.env,self.replay_template_with_seed())
            self.ai_hidden=self.agent.hidden
            self.pending_human_action=None;self.public_actions={0:None,1:None}
            self.terminal_announced=False;self.session_path=None
            self.human_plays=self.ai_plays=self.human_abilities=self.ai_abilities=self.unexpected_rejections=0
            self.action_log=[];self.ai_last_action='等待开始';self.ai_last_value=0.0
            self.selected_deck.set(-1);self.deployment_mask=None;self.raw_deployment_mask=None
            self.last_deploy_marker=None;self.ready=True
            self.render();self.write_status(schedule=False)

        def attach(self):
            try:self._reset_native_battle()
            except Exception as error:self._stop_with_error(error)

        def _stop_with_error(self,error):
            self.ready=False;self.last_error=str(error)
            self.write_status(schedule=False)
            super()._stop_with_error(error)

        def replay_template_with_seed(self):
            value=deepcopy(self.replay_template);value['rndSeed']=int(self.seed.get());return value

        def toggle_play(self):
            if not self.ready or self.state is None:return
            if self.state.get('episode',{}).get('terminated'):
                self.status.set('本局已结束，请点击重置再开一局');return
            self.loop_generation+=1;self.running=not self.running
            if self.running:
                generation=self.loop_generation
                self.next_deadline=time.perf_counter()+self.TICK_SECONDS
                self.root.after(1,lambda:self._game_loop(generation))
            self.render();self.write_status(schedule=False)

        def _sample_ai(self):
            action,audit=self.agent.decide(self.state,self.env.decks,self.env)
            self.ai_hidden=self.agent.hidden
            if not audit.get('policy_skipped'):
                self.ai_last_action=('WAIT' if action is None else '技能' if action.get('type')=='ability'
                                     else CARD_NAMES.get(action['card_id'],str(action['card_id'])))
            return action,audit

        def _advance_one_tick(self):
            before=int(self.state['tick'])
            done=super()._advance_one_tick()
            row=self.action_log[-1]
            actions=[action for action in (row['human_action'],row['ai_action']) if action is not None]
            self.agent.record_transition(before,int(self.state['tick']),actions,row['native'],self.env.decks,terminal=done)
            return done

        def render(self):
            super().render()
            if self.state is not None:
                audit=self.agent.last_audit
                self.status.set(self.status.get().replace(' V=+0.000','')+f" | ACT={audit.get('play_probability',0):.3f} 历史={audit.get('history_slots',0)}/8"
                                + (' | 点击开始对局' if not self.running and self.ready else ''))

        def _session_payload(self,*,partial):
            result=super()._session_payload(partial=partial)
            result.pop('expert_choice_mode',None);result.pop('expert_play_rate_scale',None)
            result.update(kind='native_human_vs_hokoff_release_v1',bc_decoding_mode='greedy-masked-fixed4',bc_timing_threshold=.5)
            result.update(native_identity=self.native_identity,online_history_contract='strict_past_acknowledged_normal_plays',
                          confirmed_history=self.agent.history.accepted_plays,last_model_audit=self.agent.last_audit)
            return result

        def save_session(self,*,partial):
            if self.session_path is not None and not partial:return self.session_path
            self.session_root.mkdir(parents=True,exist_ok=True)
            target=self.session_root/f"bc-{metadata['training_step']}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}.json"
            target.write_text(json.dumps(self._session_payload(partial=partial),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            self.session_path=target;return target

        def write_status(self,*,schedule=True):
            self.session_root.mkdir(parents=True,exist_ok=True)
            target=self.session_root/'box-status.json';temporary=target.with_suffix('.tmp')
            temporary.write_text(json.dumps(dict(pid=os.getpid(),ready=self.ready,running=self.running,window_visible=bool(self.root.winfo_viewable()),
                 tick=(self.state or {}).get('tick'),model=metadata,history=self.agent.history.accepted_plays,
                 audit=self.agent.last_audit,error=self.last_error,updated_utc=datetime.now(timezone.utc).isoformat()),ensure_ascii=False,indent=2),encoding='utf-8')
            temporary.replace(target)
            if schedule:self.root.after(1000,self.write_status)

    root=tk.Tk();box=MatchBox(root)
    root.mainloop()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=DEFAULT_WEIGHT)
    p.add_argument('--encoder-contract',type=Path,default=DEFAULT_CONTRACT)
    p.add_argument('--replay',type=Path,default=DEFAULT_REPLAY)
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=37031)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--session-root',type=Path,default=PROJECT/'artifacts'/'hokoff-match-box')
    p.add_argument('--smoke',action='store_true');p.add_argument('--smoke-ticks',type=int,default=160)
    args=p.parse_args(argv);torch.set_num_threads(1)
    with match_lease(args.host,args.port):
        return run(args)


def run(args):
    model,encoder,identity=load_release(args.checkpoint,args.encoder_contract,device=args.device)
    replay=json.loads(args.replay.read_text(encoding='utf-8-sig'))
    if replay.get('cmd'):raise ValueError('use a deck preset, not a recorded replay')
    env=NativeRoyaleEnv(host=args.host,port=args.port,timeout=30)
    try:
        if args.smoke:
            result=smoke(env,model,encoder,replay,ticks=args.smoke_ticks)
            args.session_root.mkdir(parents=True,exist_ok=True)
            (args.session_root/'smoke.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            print(json.dumps(result,ensure_ascii=False));return 0
        open_box(env,model,encoder,identity,args.checkpoint,args.replay,args.session_root)
    finally:env.close()
    return 0


if __name__=='__main__':raise SystemExit(main())
