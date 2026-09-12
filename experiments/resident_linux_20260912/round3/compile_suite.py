"""Isolated process-group-bounded compiler validation and matched full-game timing."""
import json,os,sys,subprocess,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parent;BASE=Path('/root/autodl-tmp/resident-linux-20260912')
def case(label,variant,verify=False):
    path=ROOT/(label+'.json')
    cmd=[sys.executable,str(ROOT/'run_variant.py'),'--device','cuda','--port','25431','--raw','--modes','packed',
         '--checkpoint',str(BASE/'model319051/inference.pt'),'--contract',str(BASE/'model319051/encoder-contract.json'),
         '--output',str(path),'--timeout','180']
    if verify:cmd+=['--verify','--max-ticks','512']
    with (ROOT/(label+'.log')).open('w') as log:
        p=subprocess.Popen(cmd,stdout=log,stderr=log,start_new_session=True,
            env={**os.environ,'CR_FULL_VARIANT':variant,'CR_FULL_VERIFY':'1' if verify else '0','TORCHINDUCTOR_COMPILE_THREADS':'2'})
        try:
            code=p.wait(timeout=220)
            if code:raise RuntimeError((label,code))
        except BaseException:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            raise
    return json.loads(path.read_text())['rows'][0]
try:
    v=case('compile-verified','compile',True)
    results=[]
    for i,variant in enumerate(('control','compile','compile','control')):
        row=case(f'compiled-full-{i}',variant);assert row['complete']
        if results:assert row['finals']==results[0]['result']['finals'],'compiled trajectory mismatch'
        results.append(dict(variant=variant,result=row));print(json.dumps(dict(variant=variant,seconds=row['seconds'],tps=row['tps'])),flush=True)
    (ROOT/'compiled-full-summary.json').write_text(json.dumps(results,indent=2))
    (ROOT/'compiled-full-done.json').write_text(json.dumps(dict(passed=True,time=time.time())))
except BaseException as e:
    (ROOT/'compiled-full-done.json').write_text(json.dumps(dict(passed=False,error=repr(e),time=time.time())))
    print('COMPILER_REJECTED '+repr(e),flush=True)
