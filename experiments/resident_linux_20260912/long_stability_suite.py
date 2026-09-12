"""Matched seeds: grouped waves vs immediate per-slot refill, frozen policy."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
common=['--checkpoint',str(ROOT/'model319051/inference.pt'),'--contract',str(ROOT/'model319051/encoder-contract.json')]
subprocess.run([sys.executable,str(ROOT/'scale_suite.py'),'--counts','12','--slots','4','--rounds','6','--label','wave-long',*common],check=True,timeout=2100)
subprocess.run([sys.executable,str(ROOT/'scale_continuous_suite.py'),'--counts','12','--rounds','1','--games','24',
                '--local-policy','--label','continuous-long',*common],check=True,timeout=1500)
wave=json.loads((ROOT/'wave-long-summary.json').read_text())[0]
continuous=json.loads((ROOT/'continuous-long-summary.json').read_text())[0]
gold={};actual={}
for p in Path(wave['source']).glob('policy-*.json'):
    if p.name.endswith('.progress.json'):continue
    for r in json.loads(p.read_text())['rows']:
        for slot,value in r['finals'].items():gold[r['seed']+int(slot)]=value
for p in Path(continuous['source']).glob('policy-*.json'):
    if p.name.endswith('.progress.json'):continue
    for r in json.loads(p.read_text())['rows']:
        for value in r['finals']:actual[value['seed']]={k:v for k,v in value.items() if k not in ('slot','game','seed')}
assert gold.keys()==actual.keys()
diff=[dict(seed=s,wave=gold[s],continuous=actual[s]) for s in gold if gold[s]!=actual[s]]
out=dict(wave=wave,continuous=continuous,matched_games=len(gold),different_games=diff,
         speed_ratio=continuous['tps']/wave['tps'],completed=time.time())
(ROOT/'long-stability-comparison.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
