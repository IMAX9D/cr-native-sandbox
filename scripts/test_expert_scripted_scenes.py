"""Bounded native-core regression matches, without opening a GUI or training.

Compares stochastic selection with greedy card/placement selection. Timing and
ability-kind sampling remain unchanged. A scripted opponent is not a skill rating.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from collections import Counter, defaultdict
import torch

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path: sys.path.insert(0,str(PROJECT))
from native_core.env import NativeRoyaleEnv
from native_core.human_vs_ai import HumanVsAiGui, _load_policy, CARD_COSTS
from training.schema import ActionMaskCache
from expert_v1.training_v1.train import _atomic_json


def make_driver(env, model, meta, checkpoint, replay, seed, mode, rate_scale, device):
    gui = HumanVsAiGui.__new__(HumanVsAiGui)
    gui.env, gui.model, gui.model_meta = env, model, meta
    gui.policy_version = meta['policy_version']
    gui.policy_label = 'epoch-002-fixed-scene'
    gui.checkpoint = checkpoint
    gui.device, gui.policy_seed = device, seed
    gui.expert_choice_mode = mode
    gui.expert_play_rate_scale = float(rate_scale)
    gui.expert_card_id_to_token = {int(k):int(v) for k,v in meta['card_id_to_token'].items()}
    gui.expert_ability_id_to_token = {int(k):int(v) for k,v in meta['ability_id_to_token'].items()}
    gui.expert_revealed_enemy_tokens = []
    gui.expert_generator = torch.Generator(device=device).manual_seed(seed)
    gui.ai_hidden = model.initial_hidden(1,device=device)
    gui.native_masks, gui.mask_cache = {}, ActionMaskCache()
    gui.public_actions = {0:None,1:None}
    gui.pending_human_action = None
    gui.action_log = []
    gui.human_plays = gui.ai_plays = gui.human_abilities = gui.ai_abilities = gui.unexpected_rejections = 0
    gui.render = lambda: None
    gui.status = SimpleNamespace(set=lambda message: None)
    value = deepcopy(replay); value['rndSeed'] = seed
    gui.state = env.reset(value,warmup_steps=100)
    return gui


def opponent_action(gui, lane_x):
    state = gui.state
    if not state['episode'].get('commands_allowed',True): return None
    player = next(p for p in state['players'] if int(p['side']) == 0)
    enemies = [e for e in state['entities'] if int(e.get('side',-1))==1 and
               int(e.get('card_id',-1))>=0 and int(e.get('hp',0))>0]
    for e in state['entities']:
        if int(e.get('side',-1))==0 and e.get('ability_available') and any(
            (int(e['x'])-int(q['x']))**2+(int(e['y'])-int(q['y']))**2<6000**2 for q in enemies):
            return {'type':'ability','side':0,'entity_id':int(e['entity_id']),'card_id':int(e['card_id'])}
    priority = [26000021,26000072,26000000,26000003,26000014,26000001,26000010,26000030]
    hand = sorted(player['hand'],key=lambda c:priority.index(c['card_id']) if c['card_id'] in priority else 100+c['deck_index'])
    for card in hand:
        cid = int(card['card_id'])
        if float(player['elixir_exact']) < CARD_COSTS[cid]: continue
        if cid >= 28000000:
            target = min(enemies,key=lambda e:int(e['y']),default=None)
            desired = (int(target['x']),int(target['y'])) if target else (lane_x,25500)
        elif cid >= 27000000:
            desired = (8500,11000)
        else:
            desired = (lane_x,14500)
        rows = gui.env.probe_grid(side=0,deck_index=card['deck_index'])['rows']
        cells = [(c*1000+500,r*1000+500) for r,row in enumerate(rows) for c,ok in enumerate(row) if ok=='1']
        if not cells: continue
        x,y = min(cells,key=lambda xy:(xy[0]-desired[0])**2+(xy[1]-desired[1])**2)
        return {'side':0,'deck_index':card['deck_index'],'card_id':cid,'x':x,'y':y}
    return None


def tower_damage(state,side):
    return sum(max(0,int(t['max_hp'])-int(t['hp'])) for t in state['episode']['crown_towers'] if int(t['side'])==side)


def aggregate(results):
    groups=defaultdict(list)
    for row in results: groups[(row['preset'],row['mode'],row['play_rate_scale'])].append(row)
    output={}
    for key,rows in groups.items():
        seconds=sum(x['native_seconds'] for x in rows)
        cards=Counter()
        for row in rows: cards.update(row['ai_card_counts'])
        output['|'.join(map(str,key))]={
            'games':len(rows), 'native_seconds':seconds,
            'ai_actions_per_minute':sum(x['ai_plays']+x['ai_abilities'] for x in rows)*60/max(seconds,1e-9),
            'ai_tower_damage_received_mean':sum(x['ai_tower_damage_received'] for x in rows)/len(rows),
            'opponent_tower_damage_received_mean':sum(x['opponent_tower_damage_received'] for x in rows)/len(rows),
            'ai_crown_delta_mean':sum(x['ai_crowns']-x['opponent_crowns'] for x in rows)/len(rows),
            'longest_gap_max_seconds':max(x['longest_gap_seconds'] for x in rows),
            'unexpected_rejections':sum(x['unexpected_rejections'] for x in rows),
            'ai_card_counts':dict(cards),
        }
    return output


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--port',type=int,default=37031)
    p.add_argument('--ticks',type=int,default=1200)
    p.add_argument('--seeds',type=int,nargs='+',default=[20260831,20260832])
    p.add_argument('--presets',nargs='+',default=['examples/eight-card-bootstrap.json','examples/queen-hog-control.json'])
    p.add_argument('--play-rate-scales',type=float,nargs='+',default=[1.0])
    p.add_argument('--choice-modes',nargs='+',choices=['sample','greedy-placement'],default=['sample','greedy-placement'])
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(2)
    model,meta=_load_policy(a.checkpoint,device=device,cuda_graph=False,expert_dataset_root=a.dataset_root)
    if any(not 0 < value <= 10 for value in a.play_rate_scales): raise ValueError('invalid play rate scale')
    results=[]; total=len(a.presets)*len(a.seeds)*len(a.choice_modes)*len(a.play_rate_scales)
    env=NativeRoyaleEnv(port=a.port,timeout=30)
    try:
        for preset in a.presets:
            replay=json.loads((PROJECT/preset).read_text())
            for seed in a.seeds:
                for mode in a.choice_modes:
                  for rate_scale in a.play_rate_scales:
                    case=f'{Path(preset).stem}-{seed}-{mode}-rate{rate_scale:g}x'
                    gui=make_driver(env,model,meta,a.checkpoint,replay,seed,mode,rate_scale,device)
                    initial=deepcopy(gui.state); initial_hash=initial.get('state_hash')
                    began=time.perf_counter(); threats=0; threat_actions=0
                    for index in range(a.ticks):
                        if index%40==0: gui.pending_human_action=opponent_action(gui,3500 if seed%2 else 14500)
                        threat=any(int(e.get('side',-1))==0 and int(e.get('card_id',-1))>=0 and
                                   int(e.get('y',0))>16000 and int(e.get('hp',0))>0 for e in gui.state['entities'])
                        threats+=int(threat)
                        done=gui._advance_one_tick()
                        threat_actions+=int(threat and gui.action_log[-1]['ai_action'] is not None)
                        if (index+1)%100==0 or done:
                            progress={'case':case,'finished_cases':len(results),'total_cases':total,
                                      'ticks':index+1,'target_ticks':a.ticks,'terminated':done}
                            _atomic_json(a.output/'progress.json',progress)
                            print(json.dumps(progress),flush=True)
                        if done: break
                    action_ticks=[int(x['tick']) for x in gui.action_log if x['ai_action']]
                    boundaries=[int(gui.action_log[0]['tick']),*action_ticks,int(gui.action_log[-1]['tick'])+1]
                    gaps=[(right-left)*.05 for left,right in zip(boundaries,boundaries[1:])]
                    ai_cards=Counter(str(x['ai_action']['card_id']) for x in gui.action_log if x['ai_action'] and x['ai_action'].get('type')!='ability')
                    row={'case':case,'mode':mode,'play_rate_scale':rate_scale,'seed':seed,'preset':preset,'ticks':index+1,
                         'native_seconds':len(gui.action_log)*.05,'longest_gap_seconds':max(gaps,default=0.),
                         'ai_card_counts':dict(ai_cards),
                         'initial_state_hash':initial_hash,'wall_seconds':time.perf_counter()-began,
                         'ai_plays':gui.ai_plays,'ai_abilities':gui.ai_abilities,'opponent_plays':gui.human_plays,
                         'opponent_abilities':gui.human_abilities,'unexpected_rejections':gui.unexpected_rejections,
                         'ai_tower_damage_received':tower_damage(gui.state,1),'opponent_tower_damage_received':tower_damage(gui.state,0),
                         'enemy_in_ai_half_ticks':threats,'ai_actions_under_threat':threat_actions,
                         'episode':gui.state['episode']}
                    _atomic_json(a.output/(case+'.json'),{'summary':row,'initial_state':initial,
                        'final_state':gui.state,'actions':[x for x in gui.action_log if x['human_action'] or x['ai_action']]})
                    results.append(row)
                    crowns=gui.state['episode'].get('crowns',[0,0]);row['opponent_crowns']=int(crowns[0]);row['ai_crowns']=int(crowns[1])
                    _atomic_json(a.output/'summary.json',{'results':results,'aggregates':aggregate(results),'complete':len(results)==total,
                        'limits':'Short scripted-opponent native regressions, not human skill or win-rate evaluation; damage and pressure counters are descriptive, not a defense score.'})
                    print(json.dumps({'event':'case_complete',**row}),flush=True)
    finally: env.close()


if __name__=='__main__':main()
