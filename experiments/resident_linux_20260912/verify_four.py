"""Linux four-slot batch stepping against the old-wrapper golden states."""
import gzip,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,'/root/autodl-tmp/bc-cloud-bench-20260911/core')
from native_core.client import JsonLineClient
from compare import normalize,differences,checked

def main():
    traces=[json.loads((ROOT/f'trace-{s}.json').read_text()) for s in (424242,717)]
    gold=[]
    for s in (424242,717):
        with gzip.open(ROOT/f'baseline-{s}.json.gz','rt') as f:gold.append(json.load(f))
    c=JsonLineClient(host='127.0.0.1',port=39331,timeout=20)
    def req(i,mode,**kw):return c.request(dict(op='resident',slot=i,mode=mode,**kw))
    def reset(i):
        req(i,'create',replay=traces[i%2]['replay']);req(i,'step',steps=10)
    try:
        for i in range(4):reset(i)
        states=[req(i,'observe')['state'] for i in range(4)];active=list(range(4));counts=[0]*4;actions=0;compared=0
        started=time.perf_counter()
        for cycle in range(2000):
            entries=[];want={}
            for i in active:
                expected=gold[i%2][counts[i]];actual=normalize(states[i])
                diff=differences(expected,actual)
                if diff:raise AssertionError(dict(slot=i,cycle=cycle,differences=diff[:4]))
                compared+=1
                want[i]=next((e['actions'] for e in traces[i%2]['events'] if e['tick']==states[i]['tick']),[])
                entries.append(dict(slot=i,steps=4,actions=want[i]))
            values=c.request(dict(op='resident_batch',entries=entries,raw_response='--raw' in sys.argv))['results']
            assert [v['slot'] for v in values]==active
            for v in values:
                i=v['slot'];checked(want[i],v['joint_action']);actions+=len(want[i]);counts[i]+=1;states[i]=v['state']
                if states[i]['episode']['terminated']:
                    assert normalize(states[i])==gold[i%2][-1],('final mismatch',i)
                    compared+=1;active.remove(i)
            if not active:break
        else:raise AssertionError('cycle bound')
        assert actions==604
        for i in range(4):
            reset(i);req(i,'step',steps=100)
            acts=next(e['actions'] for e in traces[i%2]['events'] if e['tick']==110)
            checked(acts,req(i,'act',actions=acts)['result']);req(i,'step',steps=4)
        before=[req(i,'observe')['state'] for i in range(4)]
        for _ in range(30):reset(0)
        after=[req(i,'observe')['state'] for i in range(4)]
        assert before[1:]==after[1:],'live-slot reset contamination'
        req(0,'step',steps=4)
        assert after[1:]==[req(i,'observe')['state'] for i in range(1,4)],'live-slot step contamination'
        result=dict(passed=True,raw_response='--raw' in sys.argv,platform='Linux/Bionic x86_64',slots=4,interval=4,observations_compared=compared,
                    accepted_actions=actions,live_reset_cycles=30,other_slots_unchanged=True,seconds=time.perf_counter()-started)
        (ROOT/('four-slot-raw-correctness.json' if '--raw' in sys.argv else 'four-slot-correctness.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    finally:c.close()
if __name__=='__main__':main()
