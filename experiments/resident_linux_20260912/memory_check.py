"""Run immediately after a fresh isolated worker start; no model process."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,'/root/autodl-tmp/bc-cloud-bench-20260911/core')
from native_core.client import JsonLineClient
pid=json.loads((ROOT/'launch.json').read_text())['pid']
def memory():
    values={}
    for line in Path(f'/proc/{pid}/smaps_rollup').read_text().splitlines():
        if ':' in line:
            k,v=line.split(':',1)
            if v.strip().endswith('kB'):values[k]=int(v.split()[0])
    return values
c=JsonLineClient(host='127.0.0.1',port=39331,timeout=20)
try:
    rows=[dict(slots=0,memory_kib=memory())]
    replay=json.loads((ROOT/'trace-424242.json').read_text())['replay']
    for i in range(4):
        replay['rndSeed']=424242+i
        c.request(dict(op='resident',slot=i,mode='create',replay=replay))
        c.request(dict(op='resident',slot=i,mode='step',steps=12))
        rows.append(dict(slots=i+1,memory_kib=memory()))
    (ROOT/'resident-memory.json').write_text(json.dumps(dict(pid=pid,rows=rows,includes_bootstrap_world=True),indent=2))
    print(json.dumps(rows))
finally:c.close()
