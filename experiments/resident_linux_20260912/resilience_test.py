"""Terminate one owned native worker, verify its peer, then recreate failed slots."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
from pool_launch import stop_one,start_one,BASE_PORT
p=argparse.ArgumentParser()
p.add_argument('--checkpoint',default='/root/autodl-tmp/bc-cloud-bench-20260911/model.pt')
p.add_argument('--contract',default='/root/autodl-tmp/bc-cloud-bench-20260911/core/hokoff_model/match_encoder_contract.json')
p.add_argument('--reference-summary',default='scaling-summary.json');args=p.parse_args()
out=ROOT/'resilience';out.mkdir(exist_ok=True)
jobs=[]
def policy(index,name):
    result=out/(name+'.json');ready=out/(name+'-ready.json');barrier=out/(name+'-go')
    # Unique invocation names; never reuse stale ready/go files.
    assert not ready.exists() and not barrier.exists()
    cmd=[sys.executable,str(ROOT/'policy_test.py'),'--port',str(BASE_PORT+index),'--raw','--device','cuda',
         '--modes','packed','--seed',str(424242+index*4),'--timeout','240','--output',str(result),
         '--ready-file',str(ready),'--barrier',str(barrier),'--checkpoint',args.checkpoint,'--contract',args.contract]
    with (out/(name+'.log')).open('w') as f:p=subprocess.Popen(cmd,stdout=f,stderr=f)
    jobs.append(p);return p,result,ready,barrier
try:
    subprocess.run([sys.executable,str(ROOT/'pool_launch.py'),'start','--count','2'],check=True,timeout=240)
    stamp=str(int(time.time()));a=policy(0,'victim-'+stamp);b=policy(1,'peer-'+stamp)
    until=time.monotonic()+90
    while not a[2].exists() or not b[2].exists():
        assert all(p.poll() is None for p in jobs),'readiness failed'
        if time.monotonic()>until:raise TimeoutError('readiness')
        time.sleep(.1)
    a[3].touch();b[3].touch();time.sleep(5)
    assert a[0].poll() is None and b[0].poll() is None
    stop_one(0);failed_code=a[0].wait(timeout=40);assert failed_code!=0
    start_one(0);recovered=policy(0,'recovered-'+stamp)
    until=time.monotonic()+90
    while not recovered[2].exists():
        if recovered[0].poll() is not None:raise RuntimeError('recovery boot failure')
        if time.monotonic()>until:raise TimeoutError('recovery readiness')
        time.sleep(.1)
    recovered[3].touch()
    assert b[0].wait(timeout=240)==0
    assert recovered[0].wait(timeout=240)==0
    peer=json.loads(b[1].read_text())['rows'][0];recovery=json.loads(recovered[1].read_text())['rows'][0]
    assert peer['complete'] and recovery['complete']
    summaries=json.loads((ROOT/args.reference_summary).read_text())
    w2=next(x for x in summaries if x['workers']>=2)
    gold=json.loads((Path(w2['source'])/'policy-1.json').read_text())['rows'][0]
    assert peer['action_digest']==gold['action_digest'],'peer changed after unrelated worker death'
    w1=next(x for x in summaries if x['workers']==1)
    initial=json.loads((Path(w1['source'])/'policy-0.json').read_text())['rows'][0]
    assert recovery['action_digest']==initial['action_digest']
    result=dict(passed=True,failed_process_exit=failed_code,peer_complete_games=4,recovered_complete_games=4,
                peer_action_sequence_unchanged=True,recovered_fresh_state_matches=True,
                scope='native CPU worker termination only; incomplete victim matches discarded')
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
finally:
    for p in jobs:
        if p.poll() is None:p.terminate()
    for p in jobs:
        try:p.wait(timeout=20)
        except subprocess.TimeoutExpired:p.kill();p.wait()
    for i in (0,1):stop_one(i)
