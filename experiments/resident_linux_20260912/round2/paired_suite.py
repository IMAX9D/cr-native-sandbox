"""Paired full-game runs across CPU and CUDA scheduling variants."""
import json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;BASE=Path('/root/autodl-tmp/resident-linux-20260912')
plan=[('baseline',41431),('combined',41431),('combined',42431),('dense',42431),('dense',42431),('baseline',41431)]
rows=[]
for i,(variant,port) in enumerate(plan):
    path=ROOT/f'paired-{i}-{variant}.json'
    cmd=[sys.executable,str(ROOT/'run_variant.py'),'--device','cuda','--port',str(port),'--raw','--modes','packed',
         '--checkpoint',str(BASE/'model319051/inference.pt'),'--contract',str(BASE/'model319051/encoder-contract.json'),
         '--output',str(path),'--timeout','150']
    with (ROOT/f'paired-{i}-{variant}.log').open('w') as f:
        subprocess.run(cmd,env={**os.environ,'CR_OPT_VARIANT':variant},stdout=f,stderr=f,check=True,timeout=180)
    row=json.loads(path.read_text())['rows'][0];assert row['complete']
    if rows:assert row['finals']==rows[0]['result']['finals'] and row['action_digest']==rows[0]['result']['action_digest'],variant
    rows.append(dict(variant=variant,port=port,result=row))
    (ROOT/'paired-summary.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(dict(variant=variant,port=port,seconds=row['seconds'],tps=row['tps'],matched=True)),flush=True)
(ROOT/'paired-done.json').write_text(json.dumps(dict(complete=True,time=time.time())))
