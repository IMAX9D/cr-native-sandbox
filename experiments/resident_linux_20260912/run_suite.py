"""Bounded, observed benchmark subprocess; preserves logs and resource evidence."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
name=sys.argv[1];args=sys.argv[2:]
if not name.replace('-','').isalnum():raise ValueError('case name')
def stats():
    r={}
    for key in ('cpu.stat','memory.current','memory.events'):
        r[key]=(Path('/sys/fs/cgroup')/key).read_text()
    pids=[]
    launch=ROOT/'launch.json'
    if launch.exists():pids.append(json.loads(launch.read_text())['pid'])
    for pid in pids:
        try:r[f'worker-{pid}-smaps']=Path(f'/proc/{pid}/smaps_rollup').read_text()
        except OSError:pass
    r['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu','--format=csv,noheader'],text=True).strip()
    r['time']=time.time();return r
samples=[stats()]
with (ROOT/f'{name}.log').open('w') as f:
    p=subprocess.Popen([sys.executable,str(ROOT/'policy_test.py'),*args,'--output',name+'.json'],stdout=f,stderr=f)
    (ROOT/f'{name}-pid.json').write_text(json.dumps(dict(pid=p.pid,args=args,started=time.time())))
    try:
        deadline=time.monotonic()+900
        while p.poll() is None:
            if time.monotonic()>deadline:raise TimeoutError('suite time limit')
            time.sleep(2);samples.append(stats())
    except BaseException:
        p.terminate();p.wait(timeout=20);raise
    finally:
        samples.append(stats())
        (ROOT/f'{name}-resources.json').write_text(json.dumps(dict(exit_code=p.poll(),samples=samples),indent=2))
    (ROOT/f'{name}-done.json').write_text(json.dumps(dict(exit_code=p.returncode,ended=time.time())))
    sys.exit(p.returncode)
