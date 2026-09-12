"""Binary/control ABBA, bounded compiler trial, then separate real learner canary."""
import os,sys,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;BASE=Path('/root/autodl-tmp/resident-linux-20260912')
rows=[]
for i,(variant,port) in enumerate([('vector',26431),('binary',26431),('binary',26431),('vector',26431)]):
    path=ROOT/f'wire-{i}-{variant}.json'
    cmd=[sys.executable,str(ROOT/'run_variant.py'),'--device','cuda','--port',str(port),'--raw','--modes','packed',
         '--checkpoint',str(BASE/'model319051/inference.pt'),'--contract',str(BASE/'model319051/encoder-contract.json'),
         '--output',str(path),'--timeout','180']
    with (ROOT/f'wire-{i}-{variant}.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=f,env={**os.environ,'CR_FULL_VARIANT':variant},check=True,timeout=240)
    r=json.loads(path.read_text())['rows'][0];assert r['complete'];assert not rows or r['finals']==rows[0]['result']['finals']
    rows.append(dict(variant=variant,result=r));(ROOT/'wire-summary.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(dict(wire=variant,seconds=r['seconds'],tps=r['tps'])),flush=True)
cmd=[sys.executable,str(ROOT/'run_variant.py'),'--device','cuda','--port','25431','--raw','--modes','packed','--max-ticks','128',
     '--checkpoint',str(BASE/'model319051/inference.pt'),'--contract',str(BASE/'model319051/encoder-contract.json'),
     '--output',str(ROOT/'compile.json'),'--timeout','90']
try:
    with (ROOT/'compile.log').open('w') as f:
        subprocess.run(cmd,stdout=f,stderr=f,env={**os.environ,'CR_FULL_VARIANT':'compile','TORCHINDUCTOR_COMPILE_THREADS':'2'},check=True,timeout=120)
    compile_result=dict(completed=True)
except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as e:compile_result=dict(completed=False,error=repr(e))
(ROOT/'compile-result.json').write_text(json.dumps(compile_result));print(json.dumps(dict(compile=compile_result)),flush=True)
with (ROOT/'learner-driver.log').open('w') as f:subprocess.run([sys.executable,str(ROOT/'learner_canary.py')],stdout=f,stderr=f,check=True,timeout=330)
(ROOT/'extra-done.json').write_text(json.dumps(dict(complete=True,time=time.time())))
