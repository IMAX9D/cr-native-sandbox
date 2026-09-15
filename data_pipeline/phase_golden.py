"""Owned native duel: compare hook-off/on physics and collect phase evidence."""
import argparse
import json
from pathlib import Path
import time
from native_core.client import JsonLineClient
from .prepare_600k import atomic_json


def run(host,port,out,steps,scenario='knight'):
    template=json.loads((Path(__file__).resolve().parents[1]/'examples/eight-card-bootstrap.json').read_text('utf-8'))
    if scenario=='projectile':
        for side in (0,1):
            deck=template['battle'][f'deck{side}']['sp'];deck[0],deck[4]=deck[4],deck[0]
    if scenario=='heal':template['battle']['deck0']['sp'][4]={'d':28000016,'l':10}
    results=[]
    with JsonLineClient(host=host,port=port,timeout=30) as client:
        def call(payload):
            r=client.request(payload)
            if not r.get('ok'):raise ValueError(r)
            return r
        for seed in range(1,129):
            template['rndSeed']=seed
            state=call(dict(op='reset',replay=template,capture_from_tick_zero=True))['state']
            if all(0 in p['hand_deck_indices'] for p in state['players']) and (scenario!='heal' or 4 in state['players'][0]['hand_deck_indices']):break
        else:raise ValueError('no suitable controlled opening')
        for enabled in (False,True):
            path=out/('hook-on.jsonl' if enabled else 'hook-off.jsonl');records=[];cursor=0;life_cursor=0;started=time.monotonic()
            call(dict(op='reset',replay=template,capture_from_tick_zero=True))
            if enabled:call(dict(op='elite_begin_v1'))
            with path.open('x',encoding='utf-8') as f:
                for tick in range(steps+1):
                    if tick==100:
                        for side,y in ((0,14000),(1,18000)):
                            r=call(dict(op='act',action=dict(type='play',side=side,deck_index=0,x=4500,y=y,
                                account_hi=side+1,account_lo=side+1,dry_run=False)))
                            if not r['result'].get('accepted'):raise ValueError('golden action rejected')
                    if scenario=='heal' and tick==190:
                        r=call(dict(op='act',action=dict(type='play',side=0,deck_index=4,x=4500,y=14000,
                            account_hi=1,account_lo=1,dry_run=False)))
                        if not r['result'].get('accepted'):raise ValueError('golden healing deployment rejected')
                    frame=call(dict(op='elite_observe_v1',after_sequence=cursor,after_lifecycle_sequence=life_cursor,capture_masks=False))
                    tel=frame['telemetry'];cursor=tel['last_sequence']
                    life_cursor=tel.get('lifecycle',{}).get('last_sequence',0)
                    state=frame['state']
                    if state['tick']!=tick:raise ValueError('golden clock mismatch')
                    f.write(json.dumps(frame,separators=(',',':'))+'\n')
                    records.append(dict(tick=tick,players=state['players'],entities=[{k:e.get(k) for k in (
                        'generation_key','card_id','side','x','y','hp','max_hp','behavior_state','attack_progress_ms','attack_load_timer_ms')}
                        for e in state['entities']]))
                    if tick<steps:call(dict(op='step',steps=1))
            results.append(dict(enabled=enabled,states=records,elapsed=time.monotonic()-started))
    differences=[i for i,(a,b) in enumerate(zip(results[0]['states'],results[1]['states'])) if a!=b]
    summary=dict(scenario=scenario,seed=template['rndSeed'],frames=steps+1,hook_off_on_core_equal=not differences,different_ticks=differences[:20],
        hook_off_seconds=results[0]['elapsed'],hook_on_seconds=results[1]['elapsed'])
    atomic_json(out/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--host',required=True);p.add_argument('--port',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=350)
    p.add_argument('--scenario',choices=['knight','projectile','heal'],default='knight')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);run(a.host,a.port,a.output,a.steps,a.scenario)
