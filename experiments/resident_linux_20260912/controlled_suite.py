"""Same-core ABBA comparison for the optional tower capture optimization."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
aff=json.loads((ROOT/'affinity.json').read_text());rows=[]
for index,fast in enumerate((False,True,True,False)):
    name=f'controlled-{index}-'+('fast' if fast else 'base')
    command=['taskset','-c',str(aff['native_cpu']),sys.executable,str(ROOT/'launch.py')]
    if fast:command.append('--tower-fast')
    subprocess.run(command,check=True,timeout=60)
    subprocess.run(['taskset','-c',str(aff['policy_cpu']),sys.executable,str(ROOT/'run_suite.py'),name,
                    '--device','cuda','--raw','--modes','packed'],check=True,timeout=280)
    r=json.loads((ROOT/(name+'.json')).read_text());r['tower_fast']=fast;r['case']=name
    rows.append(r)
    (ROOT/'controlled-summary.json').write_text(json.dumps(dict(affinity=aff,cases=rows),indent=2))
    print(name,'complete',flush=True)
(ROOT/'controlled-done.json').write_text(json.dumps(dict(complete=True,time=time.time())))
