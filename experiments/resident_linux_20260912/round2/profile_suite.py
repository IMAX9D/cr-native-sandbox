"""Matched phase accounting followed by one GPU activity trace on optimized path."""
import json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;BASE=Path('/root/autodl-tmp/resident-linux-20260912')
cases=[('baseline-control','baseline',24431,False),('dense-control','dense',25431,False),('dense-profile','dense',25431,True)]
results=[]
for label,variant,port,trace in cases:
    if not trace:
        launcher=ROOT/('launch_control.py' if variant=='baseline' else 'launch_opt.py')
        command=[sys.executable,str(launcher),'start','--count','1']
        if variant!='baseline':command.append('--cache')
        subprocess.run(command,check=True,timeout=150)
    if trace:
        subprocess.run([sys.executable,str(ROOT/'launch_opt.py'),'start','--count','1','--cache','--profile'],check=True,timeout=150)
    cmd=[sys.executable,str(ROOT/'profile_variant.py'),'--label',str(ROOT/label)]
    if trace:cmd+=['--trace','--native-profile']
    with (ROOT/(label+'.log')).open('w') as f:
        subprocess.run(cmd,stdout=f,stderr=f,env={**os.environ,'CR_OPT_VARIANT':variant,'CR_OPT_PORT':str(port)},check=True,timeout=180)
    row=json.loads((ROOT/label/'summary.json').read_text())
    assert len(row['finals'])==4
    if results:assert row['finals']==results[0]['finals']
    results.append(row)
    print(json.dumps(dict(label=label,seconds=row['wall_seconds'],matched=True)),flush=True)
(ROOT/'profiles-done.json').write_text(json.dumps(dict(complete=True,time=time.time())))
