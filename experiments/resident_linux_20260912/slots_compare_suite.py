import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
common=['--checkpoint',str(ROOT/'model319051/inference.pt'),'--contract',str(ROOT/'model319051/encoder-contract.json')]
for workers,slots,label in ((4,4,'slots4equal16'),(16,1,'slots1equal16')):
    subprocess.run([sys.executable,str(ROOT/'scale_suite.py'),'--counts',str(workers),'--slots',str(slots),
                    '--rounds','2','--label',label,*common],check=True,timeout=900)
def games(label):
    summary=json.loads((ROOT/(label+'-summary.json')).read_text())[0];g={}
    for p in Path(summary['source']).glob('policy-*.json'):
        if p.name.endswith('progress.json'):continue
        for row in json.loads(p.read_text())['rows']:
            for slot,result in row['finals'].items():g[row['seed']+int(slot)]=result
    return summary,g
a,ga=games('slots4equal16');b,gb=games('slots1equal16')
assert ga.keys()==gb.keys()
differences=[dict(seed=s,multi=ga[s],single=gb[s]) for s in ga if ga[s]!=gb[s]]
out=dict(four_slots=a,one_slot=b,matched_seed_count=len(ga),different_game_results=differences,
         four_slot_speed_ratio=a['tps']/b['tps'])
(ROOT/'slots-comparison.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
