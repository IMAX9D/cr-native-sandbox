"""Tiered real-policy collection; no learner and no model weight updates."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
from pool_launch import start_one,stop_one,main as unused,POOL,BASE_PORT

def usage():
    r={k:Path('/sys/fs/cgroup',k).read_text() for k in ('cpu.stat','memory.current','memory.events')}
    r['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu','--format=csv,noheader'],text=True).strip()
    r['time']=time.time();return r

def tier(count,rounds,checkpoint,contract,slots):
    import concurrent.futures
    tag=f'w{count}-{int(time.time())}';out=ROOT/'scaling'/tag;out.mkdir(parents=True)
    subprocess.run([sys.executable,str(ROOT/'pool_launch.py'),'start','--count',str(count)],check=True,timeout=240)
    jobs=[];barrier=out/'go';samples=[]
    try:
        for i in range(count):
            result=out/f'policy-{i}.json';ready=out/f'ready-{i}.json'
            log=(out/f'policy-{i}.log').open('w')
            cmd=[sys.executable,str(ROOT/'policy_test.py'),'--port',str(BASE_PORT+i),'--raw','--device','cuda',
                 '--modes',','.join(['packed']*rounds),'--slots',str(slots),'--seed',str(424242+i*slots),'--seed-stride','1000',
                 '--timeout','300','--barrier',str(barrier),'--ready-file',str(ready),'--output',str(result),
                 '--checkpoint',checkpoint,'--contract',contract]
            proc=subprocess.Popen(cmd,stdout=log,stderr=log);log.close()
            jobs.append((proc,result,ready));
        (out/'pids.json').write_text(json.dumps([p.pid for p,_,_ in jobs]))
        deadline=time.monotonic()+120
        while not all(r.exists() for _,_,r in jobs):
            if any(p.poll() is not None for p,_,_ in jobs):raise RuntimeError('policy failed during readiness')
            if time.monotonic()>deadline:raise TimeoutError('model readiness')
            time.sleep(.2)
        samples.append(usage());started=time.perf_counter();barrier.touch()
        deadline=time.monotonic()+rounds*310;next_sample=time.monotonic()+2
        while any(p.poll() is None for p,_,_ in jobs):
            if time.monotonic()>deadline:raise TimeoutError('tier budget')
            if any(p.poll() not in (None,0) for p,_,_ in jobs):raise RuntimeError('policy process failure')
            time.sleep(.1)
            if time.monotonic()>=next_sample:
                samples.append(usage());next_sample=time.monotonic()+2
        wall=time.perf_counter()-started;data=[json.loads(r.read_text()) for _,r,_ in jobs]
        rows=[r for d in data for r in d['rows']]
        assert len(rows)==count*rounds and all(r['complete'] for r in rows)
        ids={d['identity']['checkpoint_sha256'] for d in data};assert len(ids)==1
        result=dict(workers=count,slots_per_worker=slots,resident_games=count*slots,model_copies=count,rounds=rounds,
                    complete_games=len(rows)*slots,wall_seconds=wall,ticks=sum(r['ticks'] for r in rows),
                    plays=sum(r['plays'] for r in rows),model_sha256=list(ids)[0],
                    tps=sum(r['ticks'] for r in rows)/wall,source=str(out))
        (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
        return result
    finally:
        for p,_,_ in jobs:
            if p.poll() is None:p.terminate()
        for p,_,_ in jobs:
            try:p.wait(timeout=20)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        (out/'resources.json').write_text(json.dumps(samples,indent=2))
        for i in range(count):stop_one(i)

def main():
    p=argparse.ArgumentParser();p.add_argument('--counts',default='1,2,4,8');p.add_argument('--rounds',type=int,default=2)
    p.add_argument('--checkpoint',default='/root/autodl-tmp/bc-cloud-bench-20260911/model.pt')
    p.add_argument('--contract',default='/root/autodl-tmp/bc-cloud-bench-20260911/core/hokoff_model/match_encoder_contract.json')
    p.add_argument('--label',default='scaling')
    p.add_argument('--slots',type=int,default=4,choices=[1,4])
    a=p.parse_args();assert a.label.replace('-','').isalnum();POOL.mkdir(exist_ok=True);results=[]
    for n in map(int,a.counts.split(',')):
        if n not in (1,2,4,8,12,16,24):raise ValueError('unapproved tier')
        results.append(tier(n,a.rounds,a.checkpoint,a.contract,a.slots))
        (ROOT/(a.label+'-summary.json')).write_text(json.dumps(results,indent=2))
    (ROOT/(a.label+'-done.json')).write_text(json.dumps(dict(complete=True,time=time.time())))
if __name__=='__main__':main()
