"""Compare original wrapper and detached inner manager, same real commands."""
import gzip,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,'/root/autodl-tmp/bc-cloud-bench-20260911/core')
from native_core.client import JsonLineClient
from native_core.env import NativeRoyaleEnv

def normalize(state):
    ids={e['id']:['entity',e['generation_key']] for e in state['entities']}
    for group in ('effects','projectiles'):
        for n,e in enumerate(state.get(group,[])):
            if 'id' in e:ids[e['id']]=[group,e.get('generation_key',n)]
    def walk(x):
        if isinstance(x,dict):return {k:walk(v) for k,v in x.items() if k!='state_hash'}
        if isinstance(x,list):return [walk(v) for v in x]
        if isinstance(x,str) and x.startswith('0x'):return ids.get(x,'null' if int(x,16)==0 else 'unresolved-pointer')
        return x
    return walk(state)
def differences(a,b,p=''):
    # JSONObject reserializes integral doubles as integers; raw JSON keeps 0.0.
    # Compare numeric values exactly while keeping booleans type-sensitive.
    if type(a) in (int,float) and type(b) in (int,float) and a==b:return []
    if type(a)!=type(b):return [(p,a,b)]
    if isinstance(a,dict):
        out=[]
        for k in a.keys()|b.keys():out+=differences(a.get(k),b.get(k),p+'/'+k)
        return out
    if isinstance(a,list):
        if len(a)!=len(b):return [(p+'/length',len(a),len(b))]
        return [d for n,(x,y) in enumerate(zip(a,b)) for d in differences(x,y,p+'/'+str(n))]
    return [] if a==b else [(p,a,b)]
def resident(c,mode,**kw):
    return c.request(dict(op='resident',slot=0,mode=mode,**kw))
def checked(actions,r):
    assert len(actions)==len(r['actions'])
    for a,b in zip(actions,r['actions']):
        assert b['result']['accepted'] and a['side']==b['side'],b
        assert all(a[k]==b['result'][k] for k in ('deck_index','x','y')),b

def run(seed):
    trace=json.loads((ROOT/f'trace-{seed}.json').read_text());events={e['tick']:e['actions'] for e in trace['events']}
    e=NativeRoyaleEnv(port=39331,timeout=30);baseline=[]
    try:
        e.reset(trace['replay'],warmup_steps=0)
        state=e.client.request({'op':'observe'})['state'];baseline.append(normalize(state))
        for _ in range(2000):
            tick=state['tick'];acts=events.get(tick,[])
            if acts:checked(acts,e.joint_act(acts))
            r=e.step(4);state=e.client.request({'op':'observe'})['state'];baseline.append(normalize(state))
            if r['episode']['terminated']:break
        else:raise AssertionError('baseline did not finish')
    finally:e.close()
    with gzip.open(ROOT/f'baseline-{seed}.json.gz','wt') as f:json.dump(baseline,f)
    c=JsonLineClient(host='127.0.0.1',port=39331,timeout=30)
    try:
        resident(c,'create',replay=trace['replay']);resident(c,'step',steps=10)
        mismatches=[];matched=0
        for n,expected in enumerate(baseline):
            state=resident(c,'observe')['state'];actual=normalize(state)
            diff=differences(expected,actual)
            if diff:
                mismatches.append(dict(index=n,tick=state['tick'],differences=diff[:30]))
                if len(mismatches)==1:
                    (ROOT/f'first-difference-{seed}.json').write_text(json.dumps(dict(expected=expected,actual=actual,differences=diff),indent=2))
            else:matched+=1
            if n==len(baseline)-1:break
            acts=events.get(state['tick'],[])
            if acts:checked(acts,resident(c,'act',actions=acts)['result'])
            resident(c,'step',steps=4)
        result=dict(seed=seed,compared=len(baseline),equal=matched,first_differences=mismatches[:5],
                    baseline_final=baseline[-1]['episode'],resident_final=actual['episode'])
        (ROOT/f'comparison-{seed}.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(dict(seed=seed,compared=len(baseline),equal=matched,first_differences=mismatches[:2])),flush=True)
    finally:c.close()
if __name__=='__main__':
    for seed in (424242,717):run(seed)
