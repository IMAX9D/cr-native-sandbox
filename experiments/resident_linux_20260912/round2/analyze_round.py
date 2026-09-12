"""Same-workload performance comparison; keeps CPU work and CUDA activity separate."""
import json, statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(name):return json.loads((ROOT/name).read_text())
paired=read('paired-summary.json')
base=[r['result'] for r in paired if r['variant']=='baseline']
opt=[r['result'] for r in paired if r['variant']=='dense']
result={'single_worker':dict(baseline_mean_seconds=statistics.mean(x['seconds'] for x in base),
        optimized_mean_seconds=statistics.mean(x['seconds'] for x in opt),
        throughput_ratio=statistics.mean(x['seconds'] for x in base)/statistics.mean(x['seconds'] for x in opt),
        all_games_equal=all(x['result']['finals']==paired[0]['result']['finals'] for x in paired))}
if (ROOT/'profiles-done.json').exists():
    b=read('baseline-control/summary.json');o=read('dense-control/summary.json');p=read('dense-profile/summary.json')
    assert b['finals']==o['finals']==p['finals']
    phases={}
    for name in b['exclusive_phases']:
        old=b['exclusive_phases'][name];new=o['exclusive_phases'][name]
        phases[name]=dict(baseline_wall_seconds=old['wall_ns']/1e9,optimized_wall_seconds=new['wall_ns']/1e9,
                         baseline_cpu_seconds=old['cpu_ns']/1e9,optimized_cpu_seconds=new['cpu_ns']/1e9)
    result['profile_controls']=dict(baseline_wall=b['wall_seconds'],optimized_wall=o['wall_seconds'],
            baseline_actor_cpu=b['actor_process_cpu_seconds'],optimized_actor_cpu=o['actor_process_cpu_seconds'],
            phases=phases,optimized_native_profile=p['native_delta'])
if (ROOT/'dense-profile/trace-analysis.json').exists():result['gpu_trace']=read('dense-profile/trace-analysis.json')
if (ROOT/'load-done.json').exists():
    load=read('load-summary.json');b=[r for r in load if r['variant']=='baseline'];o=[r for r in load if r['variant']=='dense']
    assert all(r['games']==load[0]['games'] for r in load)
    result['load']=dict(workers=12,resident_games=48,games_per_run=load[0]['completed_games'],runs=len(load),
                       all_games_equal=True,baseline_mean_seconds=statistics.mean(r['seconds'] for r in b),
                       optimized_mean_seconds=statistics.mean(r['seconds'] for r in o),
                       baseline_tps=sum(r['ticks'] for r in b)/sum(r['seconds'] for r in b),
                       optimized_tps=sum(r['ticks'] for r in o)/sum(r['seconds'] for r in o),
                       throughput_ratio=(sum(r['ticks'] for r in o)/sum(r['seconds'] for r in o))/(sum(r['ticks'] for r in b)/sum(r['seconds'] for r in b)),
                       cases=[{k:v for k,v in r.items() if k!='games'} for r in load])
(ROOT/'analysis.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ('gpu_trace','profile_controls')},indent=2))
