import json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def stats(text):return {k:int(v) for k,v in (line.split() for line in text.splitlines())}
for name in sys.argv[1:] or ['scaling']:
    path=ROOT/(name+'-summary.json')
    if not path.exists():continue
    rows=json.loads(path.read_text());out=[]
    for row in rows:
        samples=json.loads((Path(row['source'])/'resources.json').read_text())
        a,b=samples[0],samples[-1];dt=b['time']-a['time']
        gpu=[s['gpu'].split(',') for s in samples]
        r=dict(row)
        r.update(mean_cpu_cores=(stats(b['cpu.stat'])['usage_usec']-stats(a['cpu.stat'])['usage_usec'])/1e6/dt,
                 max_cgroup_gib=max(int(s['memory.current']) for s in samples)/2**30,
                 mean_gpu_util=statistics.mean(float(x[0].strip().split()[0]) for x in gpu),
                 max_gpu_mib=max(float(x[1].strip().split()[0]) for x in gpu),
                 oom_delta=stats(b['memory.events'])['oom']-stats(a['memory.events'])['oom'])
        out.append(r)
    (ROOT/(name+'-analysis.json')).write_text(json.dumps(out,indent=2))
    print(json.dumps(out),flush=True)
