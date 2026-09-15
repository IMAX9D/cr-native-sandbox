"""Stream a catalogued TAR.XZ into validated, resumable Zstandard shards.

This stage does NOT certify rulesets, reconstruct native observations, or
generate WAIT/training labels. Source data is preserved inside each record.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import lzma
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tarfile
import time

import zstandard as zstd

SCHEMA = 'cr-source-normalized.v1'
MEMBER = re.compile(r'^battles/[A-Za-z0-9]{2}/([A-Za-z0-9]+)\.json$')


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n').encode('utf-8')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('wb') as f:
        f.write(encoded(value)); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def normalize(value, source):
    issues = []
    if not isinstance(value, dict):
        return dict(schema=SCHEMA, source=source, source_payload=value,
                    issues=['root_not_object'], normalization_status='quarantined',
                    ruleset_status='unassigned', training_ready=False, events=[])
    tag = source['battle_tag']
    if value.get('battle_tag') != tag: issues.append('battle_identity_mismatch')
    version = value.get('schema_version')
    if version not in (3, 5): issues.append('unsupported_source_schema')
    stamp = source['catalog_timestamp']
    time_fields = {k:value.get(k) for k in ('timestamp','version_timestamp','battle_time_utc')}
    found_time = False
    for key in ('timestamp', 'version_timestamp'):
        v = time_fields[key]
        if v is not None:
            found_time = True
            if type(v) is not int or v != stamp: issues.append(key+'_conflicts_with_catalog')
    if time_fields['battle_time_utc'] is not None:
        found_time = True
        try:
            dt = datetime.fromisoformat(str(time_fields['battle_time_utc']).replace('Z','+00:00'))
            if dt.tzinfo is None or dt.timestamp() != stamp: issues.append('utc_time_conflicts_with_catalog')
        except (ValueError, TypeError, OverflowError): issues.append('invalid_utc_time')
    if not found_time: issues.append('source_timestamp_missing')
    duration = value.get('duration_seconds')
    duration_ticks = round(duration*20) if finite_number(duration) and duration >= 0 else None
    if duration_ticks is None: issues.append('duration_missing_or_invalid')
    elif abs(duration_ticks-duration*20)>1e-6: issues.append('duration_not_integer_tick')
    events = []
    markers = []
    for kind, field in [('deploy','card_plays'), ('ability','ability_plays')]:
        values = value.get(field)
        if not isinstance(values, list):
            issues.append(field+'_missing_or_invalid'); continue
        for i, e in enumerate(values):
            if not isinstance(e, dict):
                issues.append(kind+'_event_not_object'); continue
            tick = e.get('time_raw')
            if type(tick) is not int or tick < 0:
                issues.append(kind+'_integer_tick_missing'); tick = None
            if tick is not None and duration_ticks is not None and tick > duration_ticks:
                issues.append(kind+'_after_declared_duration')
            seconds = e.get('time')
            if seconds is not None and (not finite_number(seconds) or tick is None or abs(seconds*20-tick)>1e-5):
                issues.append(kind+'_tick_seconds_disagree')
            if e.get('side') not in ('team','opponent'): issues.append(kind+'_side_unknown')
            marker = e.get('marker_index')
            if type(marker) is int and marker >= 0: markers.append(marker)
            else: issues.append(kind+'_marker_order_unknown')
            event = dict(kind=kind, tick=tick, side=e.get('side'), source_list=field,
                         source_index=i, marker_index=marker, identity=e.get('card') if kind=='deploy' else e.get('ability_id'))
            if kind == 'deploy':
                x,y,flag = e.get('x_raw'),e.get('y_raw'),e.get('data_i')
                if type(flag) is int and flag in (0,1) and finite_number(x) and finite_number(y):
                    # Same transform as native_replay_plan.CoordinateAudit.
                    event['native_xy_candidate'] = [18000-x,32000-y] if flag==0 else [x,y]
                    event['coordinate_provenance'] = 'royaleapi_raw_data_i_to_native_v1'
                    if not (0<=x<=18000 and 0<=y<=32000): issues.append('deploy_coordinates_outside_arena')
                else:
                    event['native_xy_candidate'] = None
                    event['coordinate_provenance'] = 'unverified_legacy_coordinates'
                    issues.append('deploy_raw_coordinate_provenance_missing')
                if not e.get('card'): issues.append('deploy_card_identity_missing')
            else:
                event['identity_status'] = 'source_declared' if e.get('ability_id') is not None else 'needs_native_resolution'
            events.append(event)
    if len(markers) != len(set(markers)): issues.append('duplicate_marker_index')
    events.sort(key=lambda e:(e['tick'] if e['tick'] is not None else -1,
                              e['marker_index'] if type(e['marker_index']) is int else -1,
                              e['source_list'],e['source_index']))
    for field in ('team_deck','opponent_deck'):
        if not value.get(field): issues.append(field+'_missing')
    # Calendar buckets are intentionally NOT a historical-ruleset classifier.
    month = datetime.fromtimestamp(stamp, timezone.utc).strftime('%Y-%m')
    return dict(schema=SCHEMA, source=source, source_payload=value,
                source_schema_version=version, timestamp=stamp, calendar_bucket=month,
                duration_ticks=duration_ticks, events=events, issues=sorted(set(issues)),
                normalization_status='clean_source' if not issues else 'needs_review',
                ruleset_id=None, ruleset_status='needs_verified_runtime_and_resource_mapping',
                training_ready=False)


def verify_shard(path, expected):
    import io
    actual = []
    with path.open('rb') as f, zstd.ZstdDecompressor().stream_reader(f) as raw, io.TextIOWrapper(raw,encoding='utf-8') as text:
        for line in text:
            d = json.loads(line)
            actual.append((d['source']['battle_tag'], d['source']['sha256']))
            if d['training_ready'] is not False: raise ValueError('stage1 cannot admit training data')
    if actual != expected: raise ValueError('shard readback identity/order mismatch')


def run(args):
    archive = args.archive.resolve(); index = args.catalog.resolve(); output = args.output.resolve()
    if output == archive.parent or output == output.parent or output in archive.parents:
        raise ValueError('output must be a separate dedicated directory')
    output.mkdir(parents=True,exist_ok=True)
    control=Path(getattr(args,'control_dir',None) or output).resolve()
    control.mkdir(parents=True,exist_ok=True)
    shards = output/'normalized'; shards.mkdir(exist_ok=True)
    lock_file = control/'processing.lock'
    # OS lock releases on crash; the empty marker may safely remain.
    lock = lock_file.open('a+b'); lock.seek(0)
    if os.name == 'nt':
        import msvcrt
        if lock_file.stat().st_size == 0: lock.write(b'0'); lock.flush()
        lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    else:
        import fcntl
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        print(json.dumps({'stage':'source_hash_verification','archive':str(archive)}),flush=True)
        archive_hash = digest(archive)
        if archive_hash != args.expected_sha256.lower(): raise ValueError('archive SHA256 mismatch')
        contract = dict(schema=SCHEMA, archive=str(archive), archive_sha256=archive_hash,
                        catalog=str(index), catalog_sha256=digest(index), code_sha256=digest(__file__),
                        training_ready=False, shard_records=args.shard_records, control_dir=str(control))
        contract_path=output/'contract.json'
        if contract_path.exists() and json.loads(contract_path.read_text('utf-8')) != contract:
            # A failed pre-commit attempt contains no reusable data. Preserve
            # its contract before restarting with a corrected reader.
            previous=json.loads(contract_path.read_text('utf-8'))
            ledger=control/'processing.sqlite3'
            existing=sqlite3.connect(str(ledger)) if ledger.exists() else None
            if existing:existing.execute('pragma query_only=ON')
            count=existing.execute('select count(*) from records').fetchone()[0] if existing else 0
            if existing:existing.close()
            if count or any(shards.iterdir()):
                raise ValueError('resume contract mismatch; use a new output directory')
            atomic_json(output/('empty-attempt-contract-'+previous['code_sha256'][:12]+'.json'),previous)
        atomic_json(contract_path,contract)
        if not index.is_file():raise ValueError('source catalog not found')
        source=sqlite3.connect(str(index));source.execute('pragma query_only=ON')
        rows=source.execute('select tag,member,sha256,size,timestamp,schema_version,source from battles where archive=?',(archive.name,)).fetchall()
        expected={r[1]:r for r in rows}
        if len(expected)!=len(rows): raise ValueError('catalog duplicate archive member')
        db=sqlite3.connect(control/'processing.sqlite3')
        db.execute('pragma journal_mode=WAL')
        db.execute('create table if not exists shards(id integer primary key,path text unique,sha256 text,count integer,bytes integer)')
        db.execute('create table if not exists records(tag text primary key,member text unique,source_sha256 text,shard integer,line integer,timestamp integer,source_schema integer,status text,issues text,duration_ticks integer,deploys integer,abilities integer,unresolved_abilities integer)')
        db.execute('create table if not exists run_status(key text primary key,value text)')
        completed={r[0] for r in db.execute('select member from records')}
        for sid,relative,sha,count,size in db.execute('select * from shards'):
            f=output/relative
            if not f.is_file() or f.stat().st_size!=size or digest(f)!=sha:
                raise ValueError('previous shard missing or modified')
        seen=set(); batch=[]; added=0; started=time.monotonic(); last_print=started

        def flush():
            nonlocal batch,added,last_print
            if not batch:return
            if shutil.disk_usage(output).free < args.min_free_gib*1024**3: raise RuntimeError('disk low-water guard')
            sid=db.execute('select coalesce(max(id),-1)+1 from shards').fetchone()[0]
            final=shards/f'shard-{sid:06d}.jsonl.zst'
            if final.exists(): raise ValueError('uncommitted shard exists; preserve it and reconcile before resume')
            temp=final.with_suffix(final.suffix+'.partial')
            if temp.exists(): raise ValueError('partial shard exists; preserve/reconcile before resume')
            with temp.open('xb') as raw:
                with zstd.ZstdCompressor(level=3,threads=0).stream_writer(raw,closefd=False) as writer:
                    for record in batch:writer.write(encoded(record))
                raw.flush();os.fsync(raw.fileno())
            verify_shard(temp,[(r['source']['battle_tag'],r['source']['sha256']) for r in batch])
            sha=digest(temp); size=temp.stat().st_size
            os.replace(temp,final)
            with db:
                db.execute('insert into shards values(?,?,?,?,?)',(sid,str(final.relative_to(output)),sha,len(batch),size))
                for line,r in enumerate(batch):
                    events=r['events'];s=r['source']
                    db.execute('insert into records values(?,?,?,?,?,?,?,?,?,?,?,?,?)',(
                        s['battle_tag'],s['member'],s['sha256'],sid,line,s['catalog_timestamp'],s['catalog_schema'],
                        r['normalization_status'],json.dumps(r['issues']),r.get('duration_ticks'),
                        sum(e['kind']=='deploy' for e in events),sum(e['kind']=='ability' for e in events),
                        sum(e['kind']=='ability' and e['identity'] is None for e in events)))
            added+=len(batch);batch=[]
            print(json.dumps({'stage':'normalize','committed':len(completed)+added,'expected':len(rows),
                              'elapsed_seconds':round(time.monotonic()-started,1),'last_shard_bytes':size}),flush=True)

        reached_eof=True
        # LZMAFile handles concatenated XZ streams produced by parallel packers;
        # tarfile's internal streaming XZ decoder stops at the first stream.
        with lzma.open(archive,'rb') as decompressed, tarfile.open(fileobj=decompressed,mode='r|') as tar:
            for m in tar:
                if not m.isfile():continue
                match=MEMBER.fullmatch(m.name)
                if not match:
                    if m.name.startswith('battles/'):raise ValueError('unsafe/noncanonical battle member')
                    continue
                if m.name in seen:raise ValueError('duplicate TAR battle member')
                seen.add(m.name)
                row=expected.get(m.name)
                if row is None or row[0]!=match.group(1) or m.size!=row[3]:raise ValueError('member/catalog mismatch')
                if m.name in completed:continue
                if m.size > 16*1024*1024:raise ValueError('oversized source record')
                b=tar.extractfile(m).read()
                if hashlib.sha256(b).hexdigest()!=row[2]:raise ValueError('source member hash mismatch')
                provenance=dict(battle_tag=row[0],member=m.name,sha256=row[2],bytes=row[3],
                                catalog_timestamp=row[4],catalog_schema=row[5],collection_source=row[6],archive_sha256=archive_hash)
                record=normalize(json.loads(b),provenance)
                if record.get('source_schema_version')!=row[5]:
                    record['issues'].append('schema_conflicts_with_catalog');record['normalization_status']='needs_review'
                batch.append(record)
                if len(batch)>=args.shard_records:flush()
                if args.limit and added+len(batch)>=args.limit:
                    reached_eof=False;break
        flush()
        if reached_eof and seen != set(expected):raise ValueError('archive/catalog membership incomplete')
        count=db.execute('select count(*) from records').fetchone()[0]
        issues=Counter()
        for raw,count_issue in db.execute('select issues,count(*) from records group by issues'):
            for issue in json.loads(raw):issues[issue]+=count_issue
        summary=dict(schema=SCHEMA, stage='source_normalization', archive_hash_verified=True,
                     source_members_sha256_verified=count, shards_readback_verified=True,
                     committed_records=count, expected_records=len(rows), source_stage_complete=reached_eof and count==len(rows),
                     added_this_run=added, normalized_bytes=db.execute('select coalesce(sum(bytes),0) from shards').fetchone()[0],
                     statuses=dict(db.execute('select status,count(*) from records group by status').fetchall()),
                     source_schemas=dict(db.execute('select source_schema,count(*) from records group by source_schema').fetchall()),
                     issues=dict(issues.most_common()), action_totals=db.execute('select sum(deploys),sum(abilities),sum(unresolved_abilities) from records').fetchone(),
                     duration_ticks_total=db.execute('select sum(duration_ticks) from records').fetchone()[0],
                     elapsed_this_run_seconds=round(time.monotonic()-started,2),
                     native_replayed=0, training_ready_records=0,
                     blocked_next_stage=['verified_ruleset_mapping','matching_native_service','rich_observation_and_action_contract'])
        atomic_json(output/'summary.json',summary)
        db.close();source.close()
        print(json.dumps(summary,ensure_ascii=True),flush=True)
        return summary
    finally:
        lock.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--expected-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--control-dir',type=Path,help='Local SQLite/lock directory; required in practice when output is an SMB share')
    p.add_argument('--limit',type=int,default=4096,help='New records this run; 0 means whole source')
    p.add_argument('--shard-records',type=int,default=2048)
    p.add_argument('--min-free-gib',type=int,default=50)
    args=p.parse_args()
    if args.limit<0 or args.shard_records<1 or args.min_free_gib<1:p.error('invalid limit/shard/space bounds')
    run(args)


if __name__=='__main__':main()
