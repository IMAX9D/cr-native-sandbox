"""Compile retained native captures into losslessly stored, NOT auto-admitted R0 IL.

Observations are assembled in capture order. A separate command index supplies
future *labels*, never future observation facts. Rejected or unrepresentable
commands invalidate their owner/window loss, rather than teaching a false WAIT.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch

from training_r0.actions import ActionSequence, ShadowLegality
from training_r0.config import DECISION_TICKS, OFFSET_TICKS, WIDTH, HEIGHT
from training_r0.il_dataset import ILSequence, load_sequence, save_sequence
from training_r0.replay_adapter import ReplayFeatureAssembler, raw_capture_lines
from .prepare_600k import atomic_json, digest
from .r0_admission import current_contract


def battle_partition(battle_uid: str) -> str:
    """Identical battles, either perspective and every overlapping chunk stay together."""
    bucket = int(hashlib.sha256(('r0-battle-split-v1:'+battle_uid).encode()).hexdigest()[:16], 16) % 10000
    return 'train' if bucket < 9600 else 'validation' if bucket < 9800 else 'test'


@dataclass(frozen=True)
class ExpertCommand:
    execution_tick: int
    side: int
    action: dict | None
    accepted: bool
    reason: str | None = None
    source_event: dict | None = None

    @property
    def window(self):
        return ((self.execution_tick-1)//DECISION_TICKS)*DECISION_TICKS


def index_capture(path):
    episode = None; pending = None; commands = []; last_tick = None; first_tick = None
    frames = 0; bad_ticks = set(); continuity = True;source_actions=[]
    with raw_capture_lines(path) as lines:
        for line in lines:
            row = json.loads(line); kind = row['kind']
            if kind == 'episode':
                if episode is not None: raise ValueError('multiple episodes in one capture')
                episode = row
            elif kind == 'frame':
                tick = row['state']['tick']; frames += 1
                if first_tick is None: first_tick = tick
                if last_tick is not None and tick not in (last_tick, last_tick+1): continuity = False
                last_tick = tick
                if row.get('capture_quality_flags'): bad_ticks.add(tick)
            elif kind == 'source_action':
                source_actions.append(dict(source_kind=row['source_kind'],event=row['event']))
                if pending is not None:
                    commands.append(ExpertCommand(pending['requested_execution_tick'],pending['event']['side'],None,False,'missing_receipt'))
                pending = row
                if row['requested_execution_tick'] != row['event']['tick']+1: raise ValueError('uncalibrated source/native timing')
            elif kind == 'command_receipt':
                request = row['request']; response = row['response'].get('result',{})
                actions = request.get('actions',[request.get('action',{})])
                results = response.get('actions',[{'result':response}])
                if len(actions)!=len(results): raise ValueError('command receipt count mismatch')
                for action,result in zip(actions,results):
                    result = result.get('result',{}); side = action['side']; tick = int(result.get('tick',row['tick']))
                    if pending is None or pending['event']['side']!=side or pending['requested_execution_tick']!=tick:
                        raise ValueError('receipt not matched to an explicit source action')
                    expected_kind='play' if pending['source_kind']=='play' else 'ability'
                    if action.get('type','play')!=expected_kind:raise ValueError('receipt action type differs from source')
                    if expected_kind=='play' and any(action.get(axis)!=pending['event'][axis] for axis in ('x','y')):
                        raise ValueError('receipt target differs from source')
                    accepted = result.get('accepted') is True
                    if expected_kind=='play':
                        deck=episode['replay']['battle'][f'deck{side}']['sp']
                        if type(action.get('deck_index')) is not int or not 0<=action['deck_index']<len(deck):raise ValueError('receipt deck index outside calibrated deck')
                    commands.append(ExpertCommand(tick,side,action,accepted,None if accepted else result.get('result_reason','native_rejected'),pending['event']))
                    pending = None
            elif kind == 'source_action_not_executed':
                if pending is None or pending['event']['side']!=row['side']: raise ValueError('unresolved action without source marker')
                commands.append(ExpertCommand(row['tick'],row['side'],None,False,row['reason'])); pending=None
    if pending is not None: commands.append(ExpertCommand(pending['requested_execution_tick'],pending['event']['side'],None,False,'missing_receipt'))
    if episode is None or first_tick is None: raise ValueError('capture has no episode/frames')
    by_window = defaultdict(list)
    for command in commands:
        if command.execution_tick<1 or command.side not in (0,1): raise ValueError('invalid command identity')
        by_window[(command.window,command.side)].append(command)
    return episode,by_window,dict(first_tick=first_tick,last_tick=last_tick,frames=frames,continuous=continuity,bad_ticks=sorted(bad_ticks),source_actions=source_actions)


def reconcile_plan(index,trace,plan,episode=None):
    """Do not turn a dropped source-action log line into an apparently valid WAIT."""
    def identity(row):return json.dumps(row,sort_keys=True,separators=(',',':'))
    expected=[dict(source_kind=kind,event=event) for kind,key in (('play','actions'),('ability','ability_events'))
              for event in plan[key] if event['tick']+1<=trace['last_tick']]
    observed=Counter(identity(row) for row in trace.pop('source_actions'))
    expected_counts=Counter(identity(row) for row in expected)
    if observed-expected_counts:raise ValueError('capture source markers differ from frozen replay plan')
    if episode is not None:
        for commands in index.values():
            for command in commands:
                if command.action is None or command.action.get('type','play')!='play':continue
                expected_card=plan['sides'][command.side]['deck'][command.source_event['logical_card_index']]['card_id']
                actual_card=episode['replay']['battle'][f'deck{command.side}']['sp'][command.action['deck_index']]['d']
                if actual_card!=expected_card:raise ValueError('executed deck card differs from expert source card')
    missing=expected_counts-observed
    for encoded,count in missing.items():
        row=json.loads(encoded);event=row['event']
        for _ in range(count):
            command=ExpertCommand(event['tick']+1,event['side'],None,False,'source_marker_missing_from_capture')
            index[(command.window,command.side)].append(command)
    trace['source_markers_missing']=sum(missing.values())
    trace['source_markers_verified']=sum(observed.values())


def empty_actions(batch_size=1):
    return ActionSequence(torch.zeros(batch_size,dtype=torch.long),*(torch.full((batch_size,2),-1,dtype=torch.long) for _ in range(3)))


def canonical_cell(x,y,side):
    if type(x) is not int or type(y) is not int: raise ValueError('noninteger source target')
    if side: x,y = 17999-x,31999-y
    if not 0<=x<WIDTH*1000 or not 0<=y<HEIGHT*1000: raise ValueError('source target outside board')
    # Public replay half-cell centers use 499/500 due to perspective reflection.
    # Never silently quantize arbitrary coordinates to an invented expert action.
    if abs(x%1000-500)>1 or abs(y%1000-500)>1: raise ValueError('source target not representable on R0 grid')
    return (y//1000)*WIDTH+x//1000


def compile_label(batch,commands,*,window_complete=True):
    if batch.batch_size!=1: raise ValueError('compiler expects one explicit actor lane')
    unknown = empty_actions()
    if not window_complete: return unknown,False,'incomplete_window'
    if len(commands)>2: return unknown,False,'more_than_two_actions'
    if any(not c.accepted for c in commands): return unknown,False,'source_action_rejected_or_unresolved'
    result = empty_actions(); shadow = ShadowLegality(batch)
    try:
        for order,command in enumerate(commands):
            tick = batch.ticks[0]
            if command.side!=batch.sides[0] or command.window!=tick: raise ValueError('wrong owner/window')
            action=command.action; kind=action.get('type','play')
            uid=action['deck_index'] if kind=='play' else 10000000+action['entity_id'] if kind=='ability' else -1
            matches = (batch.candidate_uids[0]==uid) & shadow.candidate_mask(order)[0]
            if int(matches.sum())!=1: raise ValueError('expert candidate unavailable at decision boundary')
            index=int(matches.nonzero()[0,0]); selected=torch.tensor([index])
            target=canonical_cell(action['x'],action['y'],command.side) if kind=='play' else -1
            if bool(batch.grid_targets[0,index]) != (target>=0): raise ValueError('target mode mismatch')
            if target>=0 and not bool(shadow.placement(selected)[0,target]): raise ValueError('expert target illegal at decision boundary')
            offset=command.execution_tick-1-tick
            if offset not in OFFSET_TICKS or not bool(shadow.offset_mask(order)[0,offset]): raise ValueError('expert offset/order invalid')
            result.candidate_uid[0,order]=uid; result.target_cell[0,order]=target; result.offset_bin[0,order]=offset
            shadow.apply(torch.tensor([True]),selected,torch.tensor([target]),torch.tensor([offset]))
        result.count[0]=len(commands); result.validate(1)
        return result,True,'action' if commands else 'observed_wait'
    except (ValueError,KeyError,IndexError) as error:
        return unknown,False,str(error)


def compilation_end_tick(raw_last_tick,requested):
    if requested is None:return raw_last_tick
    if type(requested) is not int or requested<0 or requested>raw_last_tick or requested%DECISION_TICKS:
        raise ValueError('max_tick must be a nonnegative decision boundary within the raw capture')
    return requested


def compile_capture(capture_dir,output,*,sequence_steps=32,source=None,max_tick=None):
    capture_dir=Path(capture_dir);output=Path(output)
    if sequence_steps<1: raise ValueError('sequence_steps must be positive')
    if output.exists(): raise FileExistsError('create a new immutable compilation directory')
    path=capture_dir/'native-frames.jsonl.zst'
    if not path.exists(): path=capture_dir/'native-frames.jsonl.gz'
    episode,index,trace=index_capture(path)
    end_tick=compilation_end_tick(trace['last_tick'],max_tick)
    report=json.loads((capture_dir/'report.json').read_text('utf-8'))
    plan=json.loads((capture_dir/'plan.json').read_text('utf-8'))
    reconcile_plan(index,trace,plan,episode)
    source_path=Path(source or episode['source_file'])
    source_hash=digest(source_path)
    if source_hash!=episode['source_sha256'] or source_hash!=report['source_sha256']: raise ValueError('source hash mismatch')
    if report.get('raw_sha256') and digest(path)!=report['raw_sha256']: raise ValueError('raw hash mismatch')
    battle=json.loads(source_path.read_bytes()).get('battle_tag') or capture_dir.name
    if not trace['continuous'] or trace['first_tick']!=0: raise ValueError('continuous Tick-0 capture required')
    output.mkdir(parents=True);assembler=ReplayFeatureAssembler(episode['replay'],battle,capture_identity=episode.get('identity'))
    buffers={s:[] for s in (0,1)}; shards=[];counts=Counter(); previous={s:None for s in (0,1)};breaks=[]
    excluded=Counter()
    partition=battle_partition(battle)

    def flush(side):
        samples=buffers[side]
        if not samples:return
        sequence=ILSequence(tuple(s[0] for s in samples),tuple(s[1] for s in samples),torch.tensor([[s[2]] for s in samples],dtype=torch.bool),samples[0][0].ticks==(0,),source_hash)
        relative=f'{partition}/{battle}-side{side}-tick{samples[0][0].ticks[0]:06d}.npz'
        sha=save_sequence(sequence,output/relative);loaded=load_sequence(output/relative,expected_sha256=sha)
        if loaded.observations[-1].ticks!=sequence.observations[-1].ticks:raise ValueError('shard readback mismatch')
        shards.append(dict(path=relative,sha256=sha,battle_uid=battle,side=side,partition=partition,first_tick=samples[0][0].ticks[0],last_tick=samples[-1][0].ticks[0],steps=len(samples),known_labels=int(sequence.label_known.sum()),starts_episode=sequence.starts_episode,requires_recurrent_predecessor=not sequence.starts_episode))
        buffers[side]=[]

    with raw_capture_lines(path) as lines:
        for line in lines:
            row=json.loads(line)
            if row['kind']=='command_receipt': assembler.command(row)
            elif row['kind']=='frame':
                if row['state']['tick']>end_tick:break
                for batch in assembler.ingest(row):
                    side=batch.sides[0];tick=batch.ticks[0]
                    # One sample with incomplete capture streams makes the strict
                    # model refuse an entire batch, so it can never carry training
                    # signal. Drop exactly that sample, break the recurrent chain
                    # (no shard may pretend to have observed it), and count why.
                    # Every sample that does enter the dataset is capture-complete.
                    if not bool(batch.semantic.elite.capture_complete.all()):
                        flush(side);previous[side]=None
                        reasons=assembler.last_capability_reasons.get(side) or {}
                        excluded[','.join(sorted(reasons)) or 'entity_identity_incomplete']+=1
                        continue
                    if previous[side] is not None and tick!=previous[side]+DECISION_TICKS:
                        flush(side);breaks.append(dict(side=side,previous_tick=previous[side],next_tick=tick))
                    complete=tick+DECISION_TICKS<=end_tick and not any(tick<=t<=tick+DECISION_TICKS for t in trace['bad_ticks'])
                    action,known,reason=compile_label(batch,index.get((tick,side),()),window_complete=complete)
                    counts[reason]+=1;buffers[side].append((batch,action,known));previous[side]=tick
                    if len(buffers[side])>=sequence_steps:flush(side)
    for side in (0,1):flush(side)
    coverage=assembler.report()
    coverage['retained_samples']=len(shards) and sum(shard['steps'] for shard in shards)
    coverage['excluded_samples']=sum(excluded.values())
    coverage['exclusion_reasons']=dict(excluded)
    atomic_json(output/'coverage.json',coverage)
    contract=current_contract();atomic_json(output/'contract.json',contract)
    manifest=dict(kind='r0-compiled-sequences.v1',contract_sha256=contract['sha256'],training_ready=False,
        compiler_sha256=digest(__file__),
        source_stage='retained_raw_capture_pending_admission',allow_incomplete_capture=False,
        capture_certificates={},checks=dict(source_hashes=True,tick_zero_capture=True,continuous_frames=True,
            opening_cycle_compatible=report.get('opening_calibration',{}).get('both_cycles_compatible') is True,
            legal_candidate_alignment=True,rejected_action_loss_masks=True,action_offsets=True,
            sequence_boundaries=not breaks,battle_level_split=True,shard_readback=True,strict_r0_forward=False,
            sample_capture_complete=True),
        retained_samples=coverage['retained_samples'],excluded_samples=coverage['excluded_samples'],
        exclusion_reasons=coverage['exclusion_reasons'],
        source=dict(path=str(source_path.resolve()),sha256=source_hash,battle_uid=battle),
        raw_capture=dict(path=str(path.resolve()),sha256=digest(path)),shards=shards,label_counts=dict(counts),
        replay_plan=dict(path=str((capture_dir/'plan.json').resolve()),sha256=digest(capture_dir/'plan.json')),
        sequence_gaps=breaks,trace=trace,raw_truncated=report.get('truncated',False),
        compiled_scope=dict(first_tick=0,last_tick=end_tick,requested_max_tick=max_tick,raw_last_tick=trace['last_tick'],
            prefix_only=end_tick<trace['last_tick'],full_raw_preserved=True,whole_battle_training_ready=False,
            final_decision_window_loss_masked=True),
        semantics='masked labels use inactive sentinels only as storage; label_known=False contributes no imitation loss',
        admission_note='Real tensors and expert targets compiled; capture certificates and strict R0 forward/backward are separate mandatory admission gates.')
    atomic_json(output/'manifest.json',manifest)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('capture',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sequence-steps',type=int,default=32);p.add_argument('--source',type=Path);p.add_argument('--max-tick',type=int)
    args=p.parse_args();result=compile_capture(args.capture,args.output,sequence_steps=args.sequence_steps,source=args.source,max_tick=args.max_tick)
    print(json.dumps({k:v for k,v in result.items() if k not in ('shards','capture_certificates')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
