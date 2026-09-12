"""Read-only Linux audit of explicit native-worker PIDs; no signals or hooks.

Separates live thread count from measured CPU, reports PSS/private pages, and
checks whether libg mappings use the same backing inode. Not a performance
benchmark: /proc sampling has its own cost and should not run at high frequency.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time


def parse_stat(text: str) -> dict:
    end=text.rfind(')')
    start=text.find('(')
    if start<0 or end<start:raise ValueError('invalid proc stat')
    fields=text[end+1:].split()
    if len(fields)<20:raise ValueError('short proc stat')
    return {'name':text[start+1:end],'start_ticks':int(fields[19]),
            'cpu_ticks':int(fields[11])+int(fields[12]),
            'processor':int(fields[36]) if len(fields)>36 else None}


def parse_memory(text: str) -> dict:
    result={}
    for line in text.splitlines():
        fields=line.split()
        if fields and fields[0] in {'Rss:','Pss:','Private_Dirty:','Private_Clean:','Shared_Clean:','Shared_Dirty:'}:
            result[fields[0].rstrip(':')]=int(fields[1])*1024
    return result


def libg_mappings(text: str) -> list[dict]:
    found={}
    for line in text.splitlines():
        fields=line.split(maxsplit=5)
        if len(fields)<6 or not fields[5].removesuffix(' (deleted)').endswith('/libg.so'):continue
        identity=(fields[3],int(fields[4]))
        found[identity]={'device':identity[0],'inode':identity[1],'path':fields[5]}
    return list(found.values())


def snapshot(pid: int, *, proc_root: Path=Path('/proc')) -> dict:
    process=proc_root/str(pid)
    args=process.joinpath('cmdline').read_bytes().split(b'\0')
    if b'royale.nativehost.JniHost' not in args or b'serve-direct' not in args:
        raise ValueError(f'PID {pid} is not an expected native host')
    result={'pid':pid,'monotonic':time.perf_counter(),
            'process':parse_stat((process/'stat').read_text()),'threads':{},'read_failures':0}
    for task in (process/'task').iterdir():
        if not task.name.isdigit():continue
        try: result['threads'][task.name]=parse_stat((task/'stat').read_text())
        except (OSError,ValueError):result['read_failures']+=1
    try:result['memory_bytes']=parse_memory((process/'smaps_rollup').read_text())
    except OSError:result['memory_bytes']=None
    try:result['libg_mappings']=libg_mappings((process/'maps').read_text())
    except OSError:result['libg_mappings']=None
    final=parse_stat((process/'stat').read_text())
    if final['start_ticks']!=result['process']['start_ticks']:
        raise ValueError('process identity changed while reading')
    return result


def compare(first: dict, second: dict, *, clock_hz: int) -> dict:
    if first['pid']!=second['pid'] or first['process']['start_ticks']!=second['process']['start_ticks']:
        raise ValueError('PID was reused between samples')
    elapsed=second['monotonic']-first['monotonic']
    if elapsed<=0 or clock_hz<=0:raise ValueError('invalid sample clock')
    groups=defaultdict(lambda:{'threads_at_end':0,'matched_threads':0,'cpu_seconds':0.0})
    unmatched=0
    for tid,current in second['threads'].items():
        group=groups[current['name']]
        group['threads_at_end']+=1
        old=first['threads'].get(tid)
        if old is None or old['start_ticks']!=current['start_ticks'] or current['cpu_ticks']<old['cpu_ticks']:
            unmatched+=1
            continue
        group['matched_threads']+=1
        group['cpu_seconds']+=(current['cpu_ticks']-old['cpu_ticks'])/clock_hz
    rows=[]
    for name,group in groups.items():
        rows.append({'name':name,**group,'mean_cpu_cores':group['cpu_seconds']/elapsed})
    rows.sort(key=lambda row:(-row['cpu_seconds'],-row['threads_at_end'],row['name']))
    delta=second['process']['cpu_ticks']-first['process']['cpu_ticks']
    if delta<0:raise ValueError('process CPU counter regressed')
    return {'pid':first['pid'],'sample_seconds':elapsed,
            'process_mean_cpu_cores':delta/clock_hz/elapsed,
            'threads_at_end':len(second['threads']),'unmatched_end_threads':unmatched,
            'ended_or_missing_threads':len(set(first['threads'])-set(second['threads'])),
            'thread_groups':rows,'memory_bytes':second['memory_bytes'],
            'libg_mappings':second['libg_mappings'],
            'read_failures':first['read_failures']+second['read_failures']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pids',required=True,help='explicit comma-separated worker PIDs')
    p.add_argument('--seconds',type=float,default=2)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if os.name!='posix':raise RuntimeError('live /proc audit requires Linux')
    pids=[int(x) for x in a.pids.split(',')]
    if not pids or len(pids)>128 or len(set(pids))!=len(pids) or min(pids)<=0 or not 0.2<=a.seconds<=10:
        raise ValueError('invalid PID set or sampling interval')
    if a.output.exists():raise FileExistsError(a.output)
    first={pid:snapshot(pid) for pid in pids}
    time.sleep(a.seconds)
    rows=[]
    for pid in pids:
        try:rows.append(compare(first[pid],snapshot(pid),clock_hz=os.sysconf('SC_CLK_TCK')))
        except (OSError,ValueError) as error:rows.append({'pid':pid,'invalid_sample':str(error)})
    identities={(m['device'],m['inode']) for row in rows for m in (row.get('libg_mappings') or [])}
    result={'kind':'native_worker_read_only_cost_audit_v1','workers':rows,
            'observed_distinct_libg_inodes':len(identities),
            'notes':['thread count is not CPU consumption','thread lifetime changes can leave unaccounted CPU',
                     'PSS/private pages include host and match; they are not all shareable',
                     'same inode is a sharing prerequisite, not proof all mapped pages are shared']}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
