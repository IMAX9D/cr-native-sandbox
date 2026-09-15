"""Capture real teacher-forced replay telemetry before admitting new IL data.

Uses the existing validated replay planner/executor. Does not train a policy,
replace old datasets, or label a partially instrumented runtime production-ready.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import argparse,gzip,hashlib,json,time

from native_core.env import NativeRoyaleEnv
from expert_v1.native_replay_plan import compile_battle
from expert_v1.native_replay_runner import execute_plan,load_template
from expert_v1.native_seed_search import resolve_native_seed
from expert_v1.native_replay_plan import materialize_replay

RUNTIME_SHA='fa6704b83cb9c5b8eecb7b56c9671b834d636a3a6d9ac446e698e1262dc246ba'


class EliteReplayEnv(NativeRoyaleEnv):
    def __init__(self,*,output:Path,capture_masks:bool=False,**kwargs):
        super().__init__(**kwargs);self.output=output;self.capture=False;self.stream=None;self.cursor=0;self.epoch=None
        self.frames=0;self.last_tick=None;self.stats=Counter();self.commands=[];self.last_frame=None
        self.capture_masks=capture_masks;self.lifecycle_cursor=0

    def arm(self):
        identity=super()._request({'op':'runtime_identity_v1'})['identity']
        if identity.get('libg_sha256')!=RUNTIME_SHA or identity.get('elite_telemetry')!='x86-v1':raise ValueError('elite host/runtime contract mismatch')
        self.identity=identity;self.capture=True

    def capture_request(self):
        return dict(after_sequence=self.cursor,after_lifecycle_sequence=self.lifecycle_cursor,capture_masks=self.capture_masks,accounts=[dict(hi=hi,lo=lo) for hi,lo in self.accounts])

    def _record(self,frame):
        state=frame['state'];telemetry=frame['telemetry'];tick=state['tick']
        if not state.get('coherent') or not frame['details'].get('valid') or frame['details']['tick']!=tick:raise ValueError('incoherent elite frame')
        if self.epoch!=telemetry['epoch'] or telemetry['gap']:raise ValueError('telemetry epoch/cursor gap')
        if self.last_tick is not None and tick not in (self.last_tick,self.last_tick+1):raise ValueError('nonconsecutive elite trace')
        if telemetry['last_sequence']<self.cursor:raise ValueError('damage cursor moved backwards')
        self.cursor=telemetry['last_sequence']
        self.lifecycle_cursor=telemetry.get('lifecycle',{}).get('last_sequence',0)
        self.stats['rejected_damage']=max(self.stats['rejected_damage'],telemetry['rejected'])
        self.stats['unattributed_damage']=max(self.stats['unattributed_damage'],telemetry['unattributed'])
        for event in telemetry['events']:
            if event['effective_damage']<=0:raise ValueError('nonpositive damage receipt')
            self.stats['damage_events']+=1
            self.stats['damage_total']+=event['effective_damage']
            is_tower=bool(event.get('target_is_tower',event['target_kind'] in (12,13)))
            self.stats['tower_damage' if is_tower else 'unit_damage']+=event['effective_damage']
        self.stats['frames_with_projectiles']+=bool(state.get('projectiles'))
        self.stats['max_entities']=max(self.stats['max_entities'],state['entity_count'])
        self.stats['mask_records']+=len(frame.get('masks',()))
        self.stats['objects_with_native_names']+=sum(bool(d['native_name']) for d in frame['details']['objects'])
        self.stats['active_effect_records']+=sum(len(d.get('active_effects',())) for d in frame['details']['objects'])
        self.stream.write(json.dumps({'kind':'frame',**frame},ensure_ascii=True,separators=(',',':'))+'\n')
        self.frames+=1;self.last_tick=tick;self.last_frame=frame

    def _request(self,payload):
        response=super()._request(payload)
        if not self.capture:return response
        if payload['op']=='reset':
            if self.stream:self.stream.close()
            self.output.mkdir(parents=True,exist_ok=True)
            self.stream=gzip.open(self.output/'native-frames.jsonl.gz','wt',encoding='utf-8')
            self.stream.write(json.dumps({'kind':'episode','replay':payload['replay'],'identity':self.identity},ensure_ascii=True)+'\n')
            begin=super()._request({'op':'elite_begin_v1'})['telemetry']
            self.epoch=begin['epoch'];self.cursor=0;self.lifecycle_cursor=0;self.last_tick=None;self.frames=0;self.stats.clear();self.commands=[]
            self._record(super()._request({'op':'elite_observe_v1',**self.capture_request()}))
        elif payload['op'] in ('act','ability','joint_act'):
            record={'kind':'command_receipt','tick':self.last_tick,'request':payload,'response':response}
            self.commands.append(record);self.stream.write(json.dumps(record,ensure_ascii=True,separators=(',',':'))+'\n')
        return response

    def step(self,steps=1):
        if not self.capture:return super().step(steps)
        if type(steps) is not int or steps<0:raise ValueError('invalid elite step count')
        if steps==0:return super().step(0)
        first_tick=self.last_tick;advanced=0;last=None
        while advanced<steps:
            result=super()._request({'op':'elite_step_trace_v1','steps':min(32,steps-advanced),**self.capture_request()})['result']
            for frame in result['frames']:self._record(frame)
            last=result['step'];progress=last['tick_after']-(first_tick+advanced)
            if progress<0:raise ValueError('native clock regressed')
            advanced+=progress
            if last.get('episode',{}).get('terminated') or progress==0:break
        return {**last,'tick_before':first_tick,'tick_after':self.last_tick,'stepped':advanced}

    def close(self):
        if self.stream:self.stream.close();self.stream=None
        super().close()


def run(source:Path,output:Path,port:int,template:Path,maximum_seeds:int=4096):
    raw=source.read_bytes();plan=compile_battle(json.loads(raw))
    if not plan.native_replay_ready:raise ValueError('source is not a native replay candidate')
    replay=load_template(template);output.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    with EliteReplayEnv(output=output,port=port,timeout=60) as env:
        selected=resolve_native_seed(env,plan,replay,maximum_seeds_to_test=maximum_seeds,warmup_tick=10)
        env.arm()
        result=execute_plan(env,plan,replay,fixed_seed=selected.chosen_seed,capture_decisions=False)
        report=dict(source=str(source.resolve()),source_sha256=hashlib.sha256(raw).hexdigest(),battle_tag=plan.battle_tag,
            chosen_seed=selected.chosen_seed,result=result.json(),frames=env.frames,telemetry=dict(env.stats),
            schema='elite-real-replay-capture.v1',training_ready=False,
            remaining='Full 78-field semantic coverage and capture certificates must pass before compilation.',
            elapsed_seconds=time.perf_counter()-started)
    (output/'capture-report.json').write_text(json.dumps(report,ensure_ascii=True,indent=2)+'\n',encoding='utf-8')
    return report


def extract(source:Path,output:Path,port:int,template:Path,seed:int=1):
    """Extraction only: rejected/unresolved commands are logged, never fatal.

    Native execution constraints remain native; no mana/hand/rule patches are
    used to force an inconsistent source action to execute.
    """
    raw=source.read_bytes();plan=compile_battle(json.loads(raw));replay,mappings=materialize_replay(plan,load_template(template),seed=seed)
    output.mkdir(parents=True,exist_ok=True);started=time.perf_counter();counts=Counter();stop_reason='source_duration_reached'
    events=sorted([('play',e) for e in plan.actions]+[('ability',e) for e in plan.ability_events],key=lambda pair:(pair[1].tick,pair[1].side,pair[0]))
    with EliteReplayEnv(output=output,port=port,timeout=60,capture_masks=False) as env:
        env.arm();env.reset(replay,warmup_steps=0)
        for kind,event in events:
            execution_tick=event.tick+1
            if execution_tick>env.last_tick:
                before=env.last_tick;step=env.step(execution_tick-before)
                if env.last_tick==before or step.get('episode',{}).get('terminated'):
                    stop_reason='native_terminal' if step.get('episode',{}).get('terminated') else 'native_clock_stopped';break
            if kind=='play':
                result=env.act(side=event.side,deck_index=mappings[event.side][event.logical_card_index],x=event.x,y=event.y)
                counts['accepted_deploys' if result.get('accepted') else 'rejected_deploys']+=1
            else:
                candidates=[e for e in env.last_frame['state']['entities'] if e['side']==event.side and e.get('ability_slot',0)>0 and e.get('ability_available')]
                if len(candidates)==1:
                    result=env.use_ability(side=event.side,entity_id=candidates[0]['category'])
                    counts['accepted_abilities' if result.get('accepted') else 'rejected_abilities']+=1
                else:
                    counts['unresolved_abilities']+=1
                    env.stream.write(json.dumps({'kind':'source_action_not_executed','tick':execution_tick,'side':event.side,'reason':'ability_source_not_unique_or_available'},ensure_ascii=True)+'\n')
        if stop_reason=='source_duration_reached' and plan.duration_ticks>env.last_tick:
            step=env.step(plan.duration_ticks-env.last_tick)
            if step.get('episode',{}).get('terminated'):stop_reason='native_terminal'
        report=dict(schema='elite-replay-extraction.v1',source=str(source.resolve()),source_sha256=hashlib.sha256(raw).hexdigest(),
            battle_tag=plan.battle_tag,seed=seed,frames=env.frames,last_tick=env.last_tick,source_duration=plan.duration_ticks,
            source_actions=len(events),command_results=dict(counts),telemetry=dict(env.stats),stop_reason=stop_reason,
            legality_checked=False,replay_consistency_checked=False,candidate_masks_collected=False,
            elapsed_seconds=time.perf_counter()-started)
    (output/'extraction-report.json').write_text(json.dumps(report,ensure_ascii=True,indent=2)+'\n',encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('source',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--port',type=int,default=37032);parser.add_argument('--template',type=Path,default=Path(__file__).resolve().parents[1]/'examples/eight-card-bootstrap.json')
    parser.add_argument('--maximum-seeds',type=int,default=4096)
    parser.add_argument('--validate-replay',action='store_true',help='opt-in historical legality/consistency audit; default only extracts')
    parser.add_argument('--seed',type=int,default=1)
    args=parser.parse_args();report=run(args.source,args.output,args.port,args.template,args.maximum_seeds) if args.validate_replay else extract(args.source,args.output,args.port,args.template,args.seed)
    print(json.dumps({k:v for k,v in report.items() if k!='result'},ensure_ascii=True,indent=2))
