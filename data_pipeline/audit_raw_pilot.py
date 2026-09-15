"""Read native facts; export lifecycle observations and causal damage summaries.

Disappearance is NOT asserted to be death; source links are never inferred
from proximity. A field being present is NOT evidence its semantics are proven.
"""
from collections import Counter,defaultdict,deque
from pathlib import Path
import argparse
import io
import json
import time

import zstandard as zstd

from .prepare_600k import atomic_json,encoded,digest


def audit(path,out):
    started=time.monotonic();entities={};damage=Counter();recent=defaultdict(deque)
    counts=Counter();fields=Counter();issues=Counter();known=Counter();events_seen=set();last_tick=None
    out.mkdir(parents=True,exist_ok=False)
    with (out/'damage-history-5tick.jsonl.zst').open('xb') as dst,zstd.ZstdCompressor(level=3).stream_writer(dst,closefd=False) as writer:
        with path.open('rb') as src,zstd.ZstdDecompressor().stream_reader(src) as raw,io.TextIOWrapper(raw,encoding='utf-8') as text:
            for line in text:
                r=json.loads(line);kind=r['kind'];counts['record_'+kind]+=1
                if kind!='frame':continue
                state=r['state'];tick=state['tick'];tel=r['telemetry'];detail=r['details'];counts['frames']+=1
                if last_tick is not None and tick not in (last_tick,last_tick+1):issues['nonconsecutive_tick']+=1
                for flag in r.get('capture_quality_flags',[]):issues[flag]+=1
                objects=list(state.get('entities',[]))+list(state.get('projectiles',[]))+list(state.get('effects',[]))
                detail_map={o['key']:o for o in detail.get('objects',[])}
                current=[]
                for e in objects:
                    key=e.get('generation_key',e.get('category'))
                    if key is None:issues['object_missing_stable_key']+=1;continue
                    current.append(key);d=detail_map.get(key,{})
                    old=entities.get(key)
                    if old is not None and old.get('native_pointer')!=e.get('id'):issues['same_generation_key_different_pointer']+=1
                    if old is None:
                        old=entities[key]=dict(key=key,native_pointer=e.get('id'),first_observed_tick=tick,
                            born_tick_candidate=d.get('created_tick_candidate'),kind=e.get('kind'),side=e.get('side'),
                            native_name=d.get('native_name'),source_key=d.get('source_key'),
                            source_pointer=e.get('source'),attached_owner_pointer=e.get('attached_owner'),
                            target_at_first_observation=e.get('target'),birth_confirmed=False,death_confirmed=False)
                    old['last_observed_tick']=tick
                    fields.update('state.'+k for k in e);fields.update('detail.'+k for k in d)
                    for k in ['hp','max_hp','level','ability_cooldown_remaining_ms','ability_charges_remaining','attack_progress_ms','attack_load_timer_ms']:
                        if type(e.get(k)) in (int,float) and e[k]>=0:known[k]+=1
                    for k in ['builtin_shield','max_builtin_shield']:
                        if type(d.get(k)) in (int,float) and d[k]>=0:known[k]+=1
                for event in tel.get('events',[]):
                    ident=(tel['epoch'],event['sequence'])
                    if ident in events_seen:issues['duplicate_damage_event']+=1;continue
                    events_seen.add(ident);counts['damage_events']+=1
                    event_tick=event['tick'];amount=event['effective_damage']
                    if event_tick>tick:issues['future_damage_event']+=1;continue
                    if amount<=0:issues['nonpositive_effective_damage']+=1;continue
                    key=event.get('attacker_key',0)
                    if not key:issues['damage_attacker_unknown']+=1;continue
                    if key not in entities:issues['damage_attacker_never_observed']+=1
                    tower=bool(event.get('target_is_tower',event.get('target_kind') in (12,13)))
                    damage[(key,tower)]+=amount
                    recent[(key,tower)].append((event_tick,amount,tick))
                    if event.get('lethal') and event.get('target_key') in entities:
                        target=entities[event['target_key']];target['lethal_damage_observed']=True;target['lethal_event_tick']=event_tick
                # Once per Tick even if ability resolution adds same-Tick observations.
                # Write audit-only six damage values; no policy features/labels.
                if tick%5==0 and tick!=last_tick:
                    rows=[]
                    for key in current:
                        values=[]
                        for tower in (False,True):
                            q=recent[(key,tower)]
                            # Event order is preserved; no future event is used.
                            while q and q[0][0]<=tick-60:q.popleft()
                            values.extend([damage[(key,tower)],sum(v for t,v,o in q if tick-20<t<=tick),sum(v for t,v,o in q if tick-60<t<=tick)])
                        rows.append(dict(key=key,values=values))
                    writer.write(encoded(dict(tick=tick,rows=rows,training_ready=False,
                        completeness='not_certified',field_order=['unit_total','unit_last1s','unit_last3s','tower_total','tower_last1s','tower_last3s'])))
                last_tick=tick
    with (out/'lifecycle.jsonl.zst').open('xb') as f,zstd.ZstdCompressor(level=3).stream_writer(f) as writer:
        for key,e in sorted(entities.items()):writer.write(encoded(e))
    report=dict(source_sha256=digest(path),counts=dict(counts),issues=dict(issues),
                raw_field_presence=dict(fields),nonnegative_numeric_counts=dict(known),entities=len(entities),
                elapsed_seconds=round(time.monotonic()-started,3),training_ready=False,
                missing_semantic_certifications=['attack_phase_mapping','deployment_phase_mapping','complete_effect_classification',
                    'projectile_damage_parameters','birth_deployment_causal_binding','all_damage_sources'],
                note='Presence counters are not the full 72-feature semantic coverage. Unknown damage is retained only in original events, never spatially assigned. Lifecycle first/last observation is not asserted birth/death.')
    atomic_json(out/'coverage.json',report);return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--attempt',default='raw-capture-v2')
    p.add_argument('--limit',type=int,default=0);a=p.parse_args();reports=[]
    for path in sorted((a.root/a.attempt).glob('*/native-frames.jsonl.zst')):
        if path.parent.name.endswith('.partial'):continue
        if not (path.parent/'report.json').exists():continue
        out=a.root/'raw-audit-v2'/path.parent.name
        if (out/'coverage.json').exists():r=json.loads((out/'coverage.json').read_text('utf-8'))
        else:r=audit(path,out)
        reports.append(r)
        if a.limit and len(reports)>=a.limit:break
    summary=dict(audited_captures=len(reports),frames=sum(r['counts']['frames'] for r in reports),
        entities=sum(r['entities'] for r in reports),issues=dict(sum((Counter(r['issues']) for r in reports),Counter())),
        counts=dict(sum((Counter(r['counts']) for r in reports),Counter())),training_ready=False,
        missing_semantic_certifications=reports[0]['missing_semantic_certifications'] if reports else [])
    atomic_json(a.root/'raw-audit-summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
