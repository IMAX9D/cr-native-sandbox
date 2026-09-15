"""Mechanism-stratified raw extraction pilot; never admits training samples."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict,replace
import io
import json
import lzma
from pathlib import Path
import os
import tarfile
import time

import zstandard as zstd

from .prepare_600k import atomic_json,digest,encoded


GROUPS={
    'melee':{'knight','mini-pekka','pekka','valkyrie','barbarians'},
    'ranged_projectile':{'musketeer','archers','princess','dart-goblin','magic-archer'},
    'area_damage':{'wizard','bomber','valkyrie','fireball','executioner'},
    'summon':{'witch','night-witch','skeleton-barrel','graveyard','tombstone','goblin-hut'},
    'shield':{'guards','dark-prince','royal-recruits','cannon-cart'},
    'persistent_effect':{'poison','earthquake','rage','freeze','tornado'},
    'building':{'cannon','tesla','inferno-tower','bomb-tower','goblin-cage'},
}


def tags_for(value):
    plays=value.get('card_plays',[])
    cards={str(e.get('card_base') or e.get('card') or '') for e in plays}
    tags={name for name,tokens in GROUPS.items() if tokens&cards}
    forms={str(e.get('card_form') or e.get('card') or '') for e in plays}
    if any('ev1' in f or 'evo' in f for f in forms):tags.add('evolution')
    if any('hero' in f for f in forms):tags.add('hero')
    if value.get('ability_plays'):tags.add('active_ability_markers')
    stamp=value.get('battle_time_utc') or ''
    tags.add('month_'+stamp[:7])
    return tags,cards


def select(args):
    out=args.output.resolve()
    if (out/'selection.json').exists():raise ValueError('selection already frozen; do not replace it')
    candidates=[]
    for shard in sorted((args.normalized/'normalized').glob('shard-*.jsonl.zst')):
        with shard.open('rb') as f,zstd.ZstdDecompressor().stream_reader(f) as raw,io.TextIOWrapper(raw,encoding='utf-8') as text:
            for line in text:
                r=json.loads(line); tags,cards=tags_for(r['source_payload'])
                candidates.append(dict(source=r['source'],tags=tags,cards=cards))
    counts=Counter();card_counts=Counter();chosen=[]
    while len(chosen)<min(args.count,len(candidates)):
        # Deterministic, diversity-oriented sample, not a random pass-rate sample.
        best=max(candidates,key=lambda x:(sum(1/(1+counts[t]) for t in x['tags'])+.08*sum(1/(1+card_counts[c]) for c in x['cards']),x['source']['battle_tag']))
        candidates.remove(best);chosen.append(best);counts.update(best['tags']);card_counts.update(best['cards'])
    expected={r['source']['member']:r for r in chosen};saved={}
    sources=out/'sources';sources.mkdir(parents=True,exist_ok=True)
    with lzma.open(args.archive,'rb') as f,tarfile.open(fileobj=f,mode='r|') as tar:
        for member in tar:
            if member.name not in expected:continue
            r=expected[member.name];b=tar.extractfile(member).read()
            import hashlib
            if hashlib.sha256(b).hexdigest()!=r['source']['sha256']:raise ValueError('source hash changed')
            path=sources/(r['source']['battle_tag']+'.json')
            with path.open('xb') as target:target.write(b)
            saved[member.name]=str(path)
            if len(saved)==len(expected):break
    if len(saved)!=len(chosen):raise ValueError('selected source missing')
    rows=[dict(**r['source'],mechanism_candidates=sorted(r['tags']),source_file=saved[r['source']['member']]) for r in chosen]
    report=dict(schema='cr-raw-pilot-selection.v1',selected=len(rows),candidate_count=len(candidates)+len(rows),
                selection_method='deterministic greedy card/mechanism diversity from first normalized records',
                mechanism_evidence='played card names/forms are coverage candidates, not proof a mechanism activated',
                categories=dict(counts),unique_played_card_bases=len(card_counts),records=rows,training_ready=False)
    atomic_json(out/'selection.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='records'},ensure_ascii=True),flush=True)


def capture_one(source:Path,out:Path,host,port,seed,bundle_manifest,ability_policy='unique',
                calibrate_opening=True,maximum_seeds=512,capture_from_tick_zero=True,
                downloader_root=None,capture_masks=False,max_capture_tick=None,allow_diagnostic_projection=False):
    from training_r0.replay_capture import EliteReplayEnv
    from native_core.env import NativeRoyaleEnv
    from expert_v1.native_replay_plan import compile_battle,materialize_replay
    from expert_v1.native_replay_runner import load_template

    class RawEnv(EliteReplayEnv):
        def __init__(self,**kwargs):
            super().__init__(**kwargs);self.anomalies=Counter();self.detail_keys=Counter();self.present=Counter();self.first_tick=None

        def _record(self,frame):
            state=frame['state'];tel=frame['telemetry'];tick=state['tick'];detail=frame['details']
            if self.first_tick is None:self.first_tick=tick
            flags=[]
            if not state.get('coherent') or detail.get('tick')!=tick or not detail.get('valid'):flags.append('incoherent_frame')
            if self.epoch!=tel['epoch']:flags.append('telemetry_epoch_changed')
            if tel.get('gap'):flags.append('telemetry_gap')
            if self.last_tick is not None and tick not in (self.last_tick,self.last_tick+1):flags.append('tick_gap_or_regression')
            if tel['last_sequence']<self.cursor:flags.append('event_cursor_regression')
            life=tel.get('lifecycle',{})
            if life.get('gap'):flags.append('lifecycle_event_gap')
            self.lifecycle_cursor=life.get('last_sequence',0)
            self.anomalies.update(flags)
            self.stream.write(encoded({'kind':'frame','capture_quality_flags':flags,**frame}).decode())
            self.frames+=1;self.last_tick=tick;self.last_frame=frame;self.epoch=tel['epoch'];self.cursor=tel['last_sequence']
            self.stats['damage_events']+=len(tel.get('events',[]));self.stats['max_entities']=max(self.stats['max_entities'],state.get('entity_count',0))
            self.stats['unattributed_damage_reported_max']=max(self.stats['unattributed_damage_reported_max'],tel.get('unattributed',0))
            for field in ['entities','projectiles','effects']:
                self.present[field+'_nonempty_frames']+=bool(state.get(field))
            for obj in detail.get('objects',[]):
                self.detail_keys.update(obj.keys())
                self.present['objects']+=1
                self.present['objects_with_active_effects']+=bool(obj.get('active_effects'))

        def _request(self,payload):
            if self.capture and payload['op']=='reset' and capture_from_tick_zero:
                if not self.identity.get('capture_from_tick_zero'):raise ValueError('host lacks Tick-0 capture contract')
                payload={**payload,'capture_from_tick_zero':True}
            response=NativeRoyaleEnv._request(self,payload)
            if not self.capture:return response
            if payload['op']=='reset':
                if self.stream:self.stream.close()
                raw=(out/'native-frames.jsonl.zst').open('xb')
                self.stream=io.TextIOWrapper(zstd.ZstdCompressor(level=3).stream_writer(raw),encoding='utf-8')
                self.stream.write(encoded(dict(kind='episode',replay=payload['replay'],identity=self.identity,
                    source_file=str(source),source_sha256=digest(source),runtime_bundle=bundle_manifest,
                    capture_code_sha256=digest(__file__),state_precision='original native JSON numeric values',
                    training_ready=False,ruleset_match='not_certified',opening_calibration=calibration,
                    capture_mode=capture_mode,replay_tier=str(getattr(plan,'replay_tier','')),
                    native_replay_ready=bool(getattr(plan,'native_replay_ready',False)),
                    seed_provenance='native-verified compatible cycle' if calibration else 'fixed diagnostic seed, not recovered original')).decode())
                begin=NativeRoyaleEnv._request(self,{'op':'elite_begin_v1'})['telemetry']
                self.epoch=begin['epoch'];self.cursor=0;self.lifecycle_cursor=0;self.last_tick=None
                self._record(NativeRoyaleEnv._request(self,{'op':'elite_observe_v1',**self.capture_request()}))
            elif payload['op'] in ('act','ability','joint_act'):
                self.stream.write(encoded(dict(kind='command_receipt',tick=self.last_tick,request=payload,response=response)).decode())
            return response

    started=time.monotonic();counts=Counter();out.mkdir(parents=True,exist_ok=False)
    # capture_mode/plan must exist even if the plan never resolved, otherwise the
    # report line below masks the real failure with an UnboundLocalError.
    env=None;status='capture_failed';error=None;total=0;attempted=0;calibration=None
    capture_mode='not_started';plan=None
    try:
        value=json.loads(source.read_bytes())
        if downloader_root:
            from .processing_contract import prepare_for_native
            value=prepare_for_native(value,downloader_root)
            atomic_json(out/'processing-input.json',value)
        try:
            if downloader_root:
                from .processing_contract import compile_processing_battle
                plan=compile_processing_battle(value);capture_mode='production_processing_contract'
            else:
                plan=compile_battle(value);capture_mode='source_contract'
        except ValueError as original_error:
            if value.get('schema_version')!=5 or 'authoritative native contract is missing' not in str(original_error):raise
            # A projected plan inherits template ruleset defaults, so it is a
            # different battle from the same source. Refuse it unless the
            # operator explicitly asked for a diagnostics-only capture: a raw
            # stream that silently changed ruleset must never look qualified.
            if not allow_diagnostic_projection:
                raise ValueError(
                    'source does not carry the authoritative native contract; production capture requires '
                    '--downloader-root, and the diagnostics-only compatibility projection must be requested '
                    'explicitly with --allow-diagnostic-projection') from original_error
            projected={**value,'schema_version':3}
            p=compile_battle(projected)
            plan=replace(p,source_schema_version=5,native_replay_ready=False,original_state_exact=False,
                replay_tier='diagnostic_source5_without_native_contract',
                state_provenance='diagnostic_unanchored_native_state',
                limitations=tuple(p.limitations)+('source5_missing_native_contract_explicit_legacy_parser_projection',))
            capture_mode='diagnostic_projection'
        atomic_json(out/'plan.json',plan.json())
        template=Path(__file__).resolve().parents[1]/'examples/eight-card-bootstrap.json'
        replay,mappings=materialize_replay(plan,load_template(template),seed=seed)
        events=sorted([('play',e) for e in plan.actions]+[('ability',e) for e in plan.ability_events],
                      key=lambda p:(p[1].tick,p[1].source_marker_index,p[1].side))
        total=len(events);env=RawEnv(output=out,host=host,port=port,timeout=60,capture_masks=capture_masks)
        calibration_seconds=0.0
        if calibrate_opening:
            _calibration_started=time.monotonic()
            from expert_v1.native_seed_search import resolve_native_seed
            resolved=resolve_native_seed(env,plan,load_template(template),preferred_seed=seed,maximum_seeds_to_test=maximum_seeds,warmup_tick=10)
            seed=resolved.chosen_seed;replay=resolved.replay;mappings=resolved.mappings
            calibration_seconds=time.monotonic()-_calibration_started
            calibration=dict(chosen_seed=seed,seeds_tested=resolved.seeds_tested,native_resets=resolved.native_resets,
                both_cycles_compatible=True,source_seed_recovered=False,observed_players=resolved.state['players'])
            atomic_json(out/'opening-calibration.json',calibration)
        env.arm()
        expected_files={f['name']:f['sha256'] for f in bundle_manifest['worker_files']}
        for field,name in [('libg_sha256','libg.so'),('host_sha256','lifecycle-probe.jar'),('bridge_sha256','libnative_host_bridge.so')]:
            expected=expected_files.get(name) or (bundle_manifest.get('libg_sha256') if name=='libg.so' else None)
            if expected is None or env.identity.get(field)!=expected:
                raise ValueError('running capture artifact does not match supplied bundle: '+name)
        env.reset(replay,warmup_steps=0)
        if calibrate_opening:
            from expert_v1.native_seed_search import layouts_accept_plan
            if not all(layouts_accept_plan(plan,sorted(env.last_frame['state']['players'],key=lambda p:p['side']))):
                raise ValueError('captured initial hand differs from verified compatible cycle')
        if capture_from_tick_zero and env.first_tick!=0:raise ValueError('capture did not begin at Tick 0')
        status='source_duration_reached'
        for kind,event in events:
            execution_tick=event.tick+1
            if max_capture_tick is not None and execution_tick>max_capture_tick:
                status='capture_limit';break
            if execution_tick>env.last_tick:
                before=env.last_tick;step=env.step(execution_tick-before)
                if env.last_tick==before or step.get('episode',{}).get('terminated'):
                    status='native_terminal' if step.get('episode',{}).get('terminated') else 'native_clock_stopped';break
            attempted+=1
            env.stream.write(encoded(dict(kind='source_action',source_kind=kind,event=asdict(event),requested_execution_tick=execution_tick)).decode())
            if kind=='play':
                r=env.act(side=event.side,deck_index=mappings[event.side][event.logical_card_index],x=event.x,y=event.y)
                counts['accepted_deploy' if r.get('accepted') else 'rejected_deploy']+=1
            else:
                # Refresh at the same Tick after preceding same-Tick commands.
                env._record(NativeRoyaleEnv._request(env,{'op':'elite_observe_v1',**env.capture_request()}))
                candidates=[e for e in env.last_frame['state']['entities'] if e['side']==event.side and e.get('ability_slot',0)>0 and e.get('ability_available')]
                selected=candidates[0]['category'] if len(candidates)==1 else None
                if ability_policy=='newest-eligible':
                    from .firstlight_resolution import select_ability
                    from expert_v1.native_capabilities import ability_cards
                    bindings=ability_cards(plan.sides[event.side].deck)
                    # Live hero entities expose category-203 form IDs, not
                    # necessarily the category-26 base card ID.
                    allowed={identity for c in bindings for identity in (c.base_card_id,c.native_form_id)}
                    resolution=select_ability(env.last_frame['state']['entities'],side=event.side,allowed_card_ids=allowed)
                    selected=resolution['selected']
                    env.stream.write(encoded(dict(kind='ability_resolution',tick=execution_tick,side=event.side,policy=ability_policy,**resolution)).decode())
                if selected is not None:
                    r=env.use_ability(side=event.side,entity_id=selected)
                    counts['accepted_ability' if r.get('accepted') else 'rejected_ability']+=1
                else:
                    counts['unresolved_ability']+=1
                    env.stream.write(encoded(dict(kind='source_action_not_executed',tick=execution_tick,side=event.side,
                        reason='ability_source_not_unique_or_available',candidate_count=len(candidates))).decode())
        if status=='source_duration_reached' and plan.duration_ticks>env.last_tick:
            step=env.step(plan.duration_ticks-env.last_tick)
            if step.get('episode',{}).get('terminated'):status='native_terminal'
        if status=='capture_limit' and max_capture_tick>env.last_tick:
            step=env.step(max_capture_tick-env.last_tick)
            if step.get('episode',{}).get('terminated'):status='native_terminal'
    except Exception as e:
        status='capture_failed';error=type(e).__name__+': '+str(e)
    finally:
        if env:env.close()
    frames=env.frames if env else 0
    report=dict(schema='cr-raw-pilot-capture.v1',status=status,error=error,source_sha256=digest(source),seed=seed,
                calibration_seconds=round(calibration_seconds,3),
                replay_seconds=round(time.monotonic()-started-calibration_seconds,3),
                seeds_tested=(calibration or {}).get('seeds_tested'),
                ability_policy=ability_policy,
                opening_calibration=calibration,first_tick=env.first_tick if env else None,
                candidate_masks_collected=capture_masks,truncated=status=='capture_limit',
                source_events=total,reached_events=attempted,unreached_events=total-attempted,command_counts=dict(counts),
                frames=frames,last_tick=env.last_tick if env else None,anomalies=dict(env.anomalies) if env else {},
                telemetry_counts=dict(env.stats) if env else {},presence_counts=dict(env.present) if env else {},
                detail_field_presence=dict(env.detail_keys) if env else {},training_ready=False,ruleset_match='not_certified',
                capture_mode=capture_mode,replay_tier=str(getattr(plan,'replay_tier','')),
                native_replay_ready=bool(getattr(plan,'native_replay_ready',False)),
                capture_code_sha256=digest(__file__),elapsed_seconds=round(time.monotonic()-started,3),
                raw_bytes=(out/'native-frames.jsonl.zst').stat().st_size if (out/'native-frames.jsonl.zst').exists() else 0)
    # Readback verifies decompression, event/frame counts and parsed JSON.
    frame_count=0
    if (out/'native-frames.jsonl.zst').exists():
        with (out/'native-frames.jsonl.zst').open('rb') as f,zstd.ZstdDecompressor().stream_reader(f) as raw,io.TextIOWrapper(raw,encoding='utf-8') as text:
            for line in text:
                d=json.loads(line);frame_count+=d.get('kind')=='frame'
        if frame_count!=frames:raise ValueError('raw frame readback mismatch')
        report['raw_sha256']=digest(out/'native-frames.jsonl.zst')
    report['readback_frames']=frame_count
    atomic_json(out/'report.json',report)
    return report


def capture(args):
    from concurrent.futures import ThreadPoolExecutor,as_completed
    from queue import Queue
    selection=json.loads((args.output/'selection.json').read_text('utf-8'))
    bundle=json.loads(args.bundle_manifest.read_text('utf-8'));reports=[]
    available=Queue()
    for port in range(args.port,args.port+args.workers):available.put(port)

    def one(r):
        target=args.output/args.attempt/r['battle_tag']
        if (target/'report.json').exists():
            report=json.loads((target/'report.json').read_text('utf-8'))
            if report.get('capture_code_sha256') not in [digest(__file__),*args.accept_existing_code_hash]:raise ValueError('capture code changed: keep old attempt and explicitly create a new one')
            if digest(Path(r['source_file']))!=report['source_sha256']:raise ValueError('source changed')
            raw=target/'native-frames.jsonl.zst'
            if report.get('raw_sha256') and digest(raw)!=report['raw_sha256']:raise ValueError('previous raw capture changed')
        else:
            temporary=target.with_name(target.name+'.partial')
            port=available.get()
            try:report=capture_one(Path(r['source_file']),temporary,args.host,port,args.seed,bundle,args.ability_policy,
                                  args.calibrate_opening,args.maximum_seeds,args.capture_from_tick_zero,
                                  args.downloader_root,args.capture_masks,args.max_capture_tick,
                                  args.allow_diagnostic_projection)
            finally:available.put(port)
            os.replace(temporary,target)
        return r['battle_tag'],report

    tasks=[r for r in selection['records'] if not args.battle_tag or r['battle_tag'] in args.battle_tag]
    if args.limit:tasks=tasks[:args.limit]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(one,r) for r in tasks]
        for future in as_completed(futures):
            tag,report=future.result();reports.append(report)
            summary=dict(schema='cr-raw-pilot-summary.v1',selected=selection['selected'],attempted=len(reports),
                statuses=dict(Counter(r['status'] for r in reports)),frames=sum(r['frames'] for r in reports),
                compressed_bytes=sum(r['raw_bytes'] for r in reports),capture_seconds=sum(r['elapsed_seconds'] for r in reports),
                command_counts=dict(sum((Counter(r['command_counts']) for r in reports),Counter())),
                anomalies=dict(sum((Counter(r['anomalies']) for r in reports),Counter())),training_ready=False)
            summary_path='capture-summary.json' if args.attempt=='raw-capture-v2' else args.attempt+'-summary.json'
            atomic_json(args.output/summary_path,summary)
            print(json.dumps(dict(battle=tag,**summary)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['select','capture'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--normalized',type=Path)
    p.add_argument('--archive',type=Path);p.add_argument('--count',type=int,default=100)
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=39431)
    p.add_argument('--seed',type=int,default=1);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--bundle-manifest',type=Path)
    p.add_argument('--attempt',default='raw-capture')
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--ability-policy',choices=['unique','newest-eligible'],default='newest-eligible')
    p.add_argument('--battle-tag',action='append',default=[])
    p.add_argument('--calibrate-opening',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--maximum-seeds',type=int,default=512)
    p.add_argument('--capture-from-tick-zero',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--downloader-root',type=Path)
    p.add_argument('--capture-masks',action='store_true')
    p.add_argument('--max-capture-tick',type=int)
    p.add_argument('--accept-existing-code-hash',action='append',default=[])
    p.add_argument('--allow-diagnostic-projection',action='store_true',
        help='explicitly run a diagnostics-only capture on a source without the authoritative native contract')
    a=p.parse_args()
    if not a.attempt or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in a.attempt):
        p.error('--attempt must be a single safe directory name')
    if a.action=='capture' and not a.downloader_root and not a.allow_diagnostic_projection:
        p.error('capture requires --downloader-root for the authoritative processing contract; pass '
                '--allow-diagnostic-projection to run an explicitly diagnostics-only capture')
    if a.action=='select':select(a)
    else:capture(a)


if __name__=='__main__':main()
