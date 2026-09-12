"""Compare checked-read work for the same active battlefield / 4-tick slices."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,'/root/autodl-tmp/bc-cloud-bench-20260911/core')
from native_core.client import JsonLineClient
trace=json.loads((ROOT/'trace-424242.json').read_text())
c=JsonLineClient(host='127.0.0.1',port=39331,timeout=20)
try:
    def req(mode,**kw):return c.request(dict(op='resident',slot=0,mode=mode,**kw))
    req('create',replay=trace['replay']);req('step',steps=10)
    begin=c.request(dict(op='memory_read_stats'))['result']
    actions=0
    for tick in range(10,1010,4):
        a=next((e['actions'] for e in trace['events'] if e['tick']==tick),[])
        r=c.request(dict(op='resident_batch',raw_response=True,entries=[dict(slot=0,steps=4,actions=a)]))['results'][0]
        for item in r['joint_action']['actions']:assert item['result']['accepted']
        actions+=len(a);assert r['state']['tick']==tick+4
    end=c.request(dict(op='memory_read_stats'))['result']
    delta={k:end[k]-begin[k] for k in end if type(end[k]) is int}
    name='profile-fast.json' if json.loads((ROOT/'launch.json').read_text())['tower_fast']=='1' else 'profile-base.json'
    out=dict(begin=begin,end=end,delta=delta,actions=actions)
    (ROOT/name).write_text(json.dumps(out,indent=2));print(json.dumps(out))
finally:c.close()
