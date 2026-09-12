import os,sys,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;BASE=Path('/root/autodl-tmp/resident-linux-20260912');rows=[]
subprocess.run([sys.executable,str(ROOT/'compact_verify.py')],check=True,timeout=120)
for i,variant in enumerate(('vector','compact','compact-binary','compact-binary','compact','vector')):
    path=ROOT/f'compact-case-{i}.json'
    cmd=[sys.executable,str(ROOT/'run_variant.py'),'--device','cuda','--port','26431','--raw','--modes','packed',
         '--checkpoint',str(BASE/'model319051/inference.pt'),'--contract',str(BASE/'model319051/encoder-contract.json'),
         '--output',str(path),'--timeout','180']
    with (ROOT/f'compact-case-{i}.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=f,env={**os.environ,'CR_FULL_VARIANT':variant},check=True,timeout=220)
    r=json.loads(path.read_text())['rows'][0];assert r['complete'];assert not rows or r['finals']==rows[0]['result']['finals']
    rows.append(dict(variant=variant,result=r));print(json.dumps(dict(variant=variant,seconds=r['seconds'],tps=r['tps'])),flush=True)
(ROOT/'compact-summary.json').write_text(json.dumps(rows,indent=2))
