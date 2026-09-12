"""12x4 full-game ABBA throughput; all actions/finals must match by seed."""
import json, os, statistics, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;BASE=Path('/root/autodl-tmp/resident-linux-20260912')

def usage():
    result={k:Path('/sys/fs/cgroup',k).read_text() for k in ('cpu.stat','memory.current','memory.events')}
    result['time']=time.time()
    result['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu','--format=csv,noheader'],text=True).strip()
    return result

def stats(value):return {k:int(v) for k,v in (line.split() for line in value.splitlines())}
def native(which,op,count):
    script=ROOT/'launch_binary.py'
    cmd=[sys.executable,str(script),op,'--count',str(count)]
    subprocess.run(cmd,check=True,timeout=240)

def run(index,variant,count=12,rounds=1):
    native(variant,'start',count)
    out=ROOT/f'transport-{index}-{variant}';out.mkdir();barrier=out/'go'
    jobs=[];samples=[]
    try:
        for i in range(count):
            result=out/f'policy-{i}.json';ready=out/f'ready-{i}.json'
            cmd=[sys.executable,str(ROOT/'run_variant.py'),'--port',str(26431+i),
                 '--raw','--device','cuda','--modes',','.join(['packed']*rounds),'--slots','4','--seed',str(424242+i*4),
                 '--seed-stride','1000','--timeout','240','--barrier',str(barrier),'--ready-file',str(ready),'--output',str(result),
                 '--checkpoint',str(BASE/'model319051/inference.pt'),'--contract',str(BASE/'model319051/encoder-contract.json')]
            with (out/f'policy-{i}.log').open('w') as log:
                p=subprocess.Popen(cmd,stdout=log,stderr=log,env={**os.environ,'CR_FULL_VARIANT':variant})
            jobs.append((p,result,ready))
        (out/'pids.json').write_text(json.dumps([p.pid for p,_,_ in jobs]))
        deadline=time.monotonic()+150
        while not all(r.exists() for _,_,r in jobs):
            if any(p.poll() is not None for p,_,_ in jobs):raise RuntimeError('readiness failed')
            if time.monotonic()>deadline:raise TimeoutError('readiness')
            time.sleep(.2)
        samples.append(usage());start=time.perf_counter();barrier.touch();deadline=time.monotonic()+600;next_sample=time.monotonic()+2
        while any(p.poll() is None for p,_,_ in jobs):
            if any(p.poll() not in (None,0) for p,_,_ in jobs):raise RuntimeError('policy failure')
            if time.monotonic()>deadline:raise TimeoutError('load test')
            if time.monotonic()>=next_sample:samples.append(usage());next_sample=time.monotonic()+2
            time.sleep(.1)
        wall=time.perf_counter()-start;samples.append(usage())
        data=[json.loads(f.read_text()) for _,f,_ in jobs]
        assert len({d['identity']['checkpoint_sha256'] for d in data})==1
        games={};ticks=plays=0
        for d in data:
            for row in d['rows']:
                assert row['complete'];ticks+=row['ticks'];plays+=row['plays']
                for slot,game in row['finals'].items():games[str(row['seed']+int(slot))]=game
        assert len(games)==count*rounds*4
        a,b=samples[0],samples[-1];dt=b['time']-a['time'];gpu=[s['gpu'].split(',') for s in samples]
        result=dict(variant=variant,workers=count,resident_games=count*4,completed_games=len(games),ticks=ticks,plays=plays,
                    seconds=wall,tps=ticks/wall,games=games,
                    mean_cpu_cores=(stats(b['cpu.stat'])['usage_usec']-stats(a['cpu.stat'])['usage_usec'])/1e6/dt,
                    max_cgroup_gib=max(int(s['memory.current']) for s in samples)/2**30,
                    mean_gpu_util=statistics.mean(float(g[0].split()[0]) for g in gpu),
                    max_gpu_mib=max(float(g[1].split()[0]) for g in gpu),
                    oom_delta=stats(b['memory.events'])['oom']-stats(a['memory.events'])['oom'])
        (out/'summary.json').write_text(json.dumps(result,indent=2));return result
    finally:
        for p,_,_ in jobs:
            if p.poll() is None:p.terminate()
        for p,_,_ in jobs:
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        (out/'resources.json').write_text(json.dumps(samples,indent=2));native(variant,'stop',count)

if __name__=='__main__':
    native('control','stop',24)
    results=[]
    try:
        for i,variant in enumerate(('graph','graph-compact','graph-compact','graph')):
            result=run(i,variant)
            if results:assert result['games']==results[0]['games'],'paired game divergence'
            results.append(result)
            (ROOT/'transport-summary.json').write_text(json.dumps(results,indent=2))
            print(json.dumps({k:v for k,v in result.items() if k!='games'}),flush=True)
        (ROOT/'transport-done.json').write_text(json.dumps(dict(complete=True,time=time.time())))
    finally:
        native('control','stop',24)
