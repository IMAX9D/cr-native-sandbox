import json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(name):return json.loads((ROOT/name).read_text())
def compare(rows,nested=False):
    groups={}
    for row in rows:
        r=row['result'] if nested else row;name=row['variant'];groups.setdefault(name,[]).append(r)
    out={}
    for name,items in groups.items():
        seconds=sum(r.get('seconds',r.get('wall_seconds',0)) for r in items)
        out[name]=dict(runs=len(items),mean_seconds=seconds/len(items),aggregate_tps=sum(r['ticks'] for r in items)/seconds)
        if 'max_gpu_mib' in items[0]:
            out[name]['max_gpu_mib']=max(r['max_gpu_mib'] for r in items)
            out[name]['mean_cpu_cores']=statistics.mean(r['mean_cpu_cores'] for r in items)
    return out
result={}
for name,file,nested in [('single','suite-summary.json',True),('binary','wire-summary.json',True),('compact','compact-summary.json',True),('load','load-summary.json',False),('transport','transport-summary.json',False),('compiled','compiled-full-summary.json',True)]:
    if (ROOT/file).exists():result[name]=compare(read(file),nested)
for name,file in [('compact_correctness','compact-verified.json'),('compiler','compiled-full-done.json'),('learner','learner-audited/result.json'),('checkpoints','checkpoint-verified.json')]:
    if (ROOT/file).exists():result[name]=read(file)
if (ROOT/'learner-audited/learner.json').exists():
    l=read('learner-audited/learner.json');c=read('learner-audited/collector.json');u=l['updates'];s=c['segments']
    result['learner_timing']=dict(trainable_parameters=l['trainable_parameters'],
        update_seconds=[x['ended']-x['started'] for x in u],queue_age_seconds=[x['queue_age_seconds'] for x in u],
        max_kl=max(x['kl'] for x in u),total_reward_abs=sum(x['reward_abs_sum'] for x in u),
        collection_samples_per_second=len(s)*128/(s[-1]['end']-s[0]['start']),
        accepted_training_samples_per_update_second=sum(x['samples'] for x in u)/sum(x['ended']-x['started'] for x in u),
        queue_backpressure_seconds=c['queue_wait_seconds'])
(ROOT/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('checkpoints',)},indent=2))
