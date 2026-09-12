"""Bounded opt-in variants. Compile costs included, failure preserved, no training."""
import os,sys,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;BASE=Path('/root/autodl-tmp/resident-linux-20260912')
rows=[]
plan=['control','vector','buffers','graph','graph','control']
for i,variant in enumerate(plan):
    path=ROOT/f'case-{i}-{variant}.json'
    cmd=[sys.executable,str(ROOT/'run_variant.py'),'--device','cuda','--port','25431','--raw','--modes','packed',
         '--checkpoint',str(BASE/'model319051/inference.pt'),'--contract',str(BASE/'model319051/encoder-contract.json'),
         '--output',str(path),'--timeout','240']
    with (ROOT/f'case-{i}-{variant}.log').open('w') as log:
        done=subprocess.run(cmd,stdout=log,stderr=log,env={**os.environ,'CR_FULL_VARIANT':variant},timeout=300)
    if done.returncode:raise RuntimeError((variant,done.returncode))
    result=json.loads(path.read_text())['rows'][0];assert result['complete']
    assert not rows or result['finals']==rows[0]['result']['finals'],'action/final mismatch '+variant
    rows.append(dict(variant=variant,result=result))
    (ROOT/'suite-summary.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(dict(variant=variant,seconds=result['seconds'],tps=result['tps'],matched=True)),flush=True)
(ROOT/'suite-done.json').write_text(json.dumps(dict(complete=True,time=time.time())))
