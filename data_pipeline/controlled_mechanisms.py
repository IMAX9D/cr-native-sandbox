"""Exercise actual evolution cycles with legal commands, without modifying game state.

Synthetic mechanism probes are not expert training data.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
from native_core.client import JsonLineClient
from native_core.card_catalog import catalog
from .prepare_600k import atomic_json


def run(host,port,output,mechanism,maximum):
    output.mkdir(parents=True,exist_ok=False)
    template=json.loads((Path(__file__).resolve().parents[1]/'examples/eight-card-bootstrap.json').read_text('utf-8'))
    target=27000012 if mechanism=='capture' else 27000013
    deck=[target,26000010,26000030,26000031,26000084,28000016,28000008,28000017]
    template['battle']['gamemode']=72000006
    template['battle']['deck0']['sp']=[dict(d=c,l=10,**({'el':1} if i==0 else {})) for i,c in enumerate(deck)]
    template['battle']['deck1']['sp'][0]={'d':26000003,'l':10}
    # Original fixture already contains Giant in slot 2; retain eight uniques.
    template['battle']['deck1']['sp'][2]={'d':26000000,'l':10}
    facts=catalog();counts=Counter();cursor=life_cursor=0;last_play=-100;giant=False;actions=[]
    with JsonLineClient(host=host,port=port,timeout=60) as client:
        def call(payload):
            r=client.request(payload)
            if not r.get('ok'):raise ValueError(r)
            return r
        for seed in range(1,129):
            template['rndSeed']=seed
            state=call(dict(op='reset',replay=template,capture_from_tick_zero=True))['state']
            if all(0 in p['hand_deck_indices'] for p in state['players']):break
        else:raise ValueError('controlled opening not found')
        call(dict(op='elite_begin_v1'))
        with (output/'frames.jsonl').open('x',encoding='utf-8') as stream:
            for tick in range(maximum+1):
                frame=call(dict(op='elite_observe_v1',after_sequence=cursor,after_lifecycle_sequence=life_cursor,capture_masks=False))
                cursor=frame['telemetry']['last_sequence'];life_cursor=frame['telemetry']['lifecycle']['last_sequence']
                stream.write(json.dumps(frame,separators=(',',':'))+'\n');state=frame['state']
                if state['tick']!=tick:raise ValueError('controlled clock mismatch')
                active=[]
                for detail in frame['details']['objects']:
                    special=detail.get('special_runtime',{})
                    for name in ('capture','relocation'):
                        row=special.get(name,{})
                        if row.get('present'):
                            counts[name+'_frames']+=1;counts[name+'_valid_frames']+=row.get('valid') is True
                            if name==mechanism:active.append(row)
                            if name=='capture':counts['captured_target_frames']+=bool(row.get('targets'))
                            if name=='relocation':counts['relocating_frames']+=row.get('stage',1)%2==0
                if active and not giant:
                    opponent=state['players'][1]
                    if 0 in opponent['hand_deck_indices'] and opponent['elixir']>=5:
                        x,y=(8500,17500) if mechanism=='capture' else (3500,23000)
                        request=dict(type='play',side=1,deck_index=0,x=x,y=y,account_hi=2,account_lo=2,dry_run=False)
                        response=call(dict(op='act',action=request))['result'];actions.append(dict(tick=tick,action=request,receipt=response))
                        giant=bool(response.get('accepted'))
                if tick>=100 and tick-last_play>=20 and not active:
                    player=state['players'][0];hand=[i for i in player['hand_deck_indices'] if i>=0]
                    choices=sorted(hand,key=lambda i:(i!=0,facts[deck[i]]['elixir'],i))
                    if choices and player['elixir']>=facts[deck[choices[0]]]['elixir']:
                        index=choices[0];x,y=((8500,14000) if mechanism=='capture' else (3500,25000)) if index==0 else (500,500)
                        request=dict(type='play',side=0,deck_index=index,x=x,y=y,account_hi=1,account_lo=1,dry_run=False)
                        response=call(dict(op='act',action=request))['result'];actions.append(dict(tick=tick,action=request,receipt=response));last_play=tick
                        counts['accepted_actions']+=bool(response.get('accepted'));counts['rejected_actions']+=not response.get('accepted')
                if state['episode']['terminated']:break
                if tick<maximum:call(dict(op='step',steps=1))
    result=dict(kind='r0-controlled-mechanism-probe.v1',training_data=False,mechanism=mechanism,seed=seed,last_tick=tick,counts=dict(counts),actions=actions)
    atomic_json(output/'report.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--host',required=True);p.add_argument('--port',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--mechanism',choices=['capture','relocation'],required=True)
    p.add_argument('--maximum',type=int,default=3000);a=p.parse_args();r=run(a.host,a.port,a.output,a.mechanism,a.maximum)
    print(json.dumps({k:v for k,v in r.items() if k!='actions'}))
