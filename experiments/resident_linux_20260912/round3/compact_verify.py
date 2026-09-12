"""Full debug vs native train-schema observations on the same unadvanced battle."""
import sys,json,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parent;BASE=Path('/root/autodl-tmp/resident-linux-20260912')
sys.path[:0]=['/root/autodl-tmp/bc-cloud-bench-20260911/core',str(BASE)]
import torch
from native_core.client import JsonLineClient
from expert_v1.tick_store_v1.schema import normalize_native_state
from expert_selfplay_v1.native_observation import NativeObservationEncoder,NativeActorFrame
from policy_test import ResidentEnv
contract=json.loads((BASE/'model319051/encoder-contract.json').read_text());enc=NativeObservationEncoder.from_manifest(contract['encoder'])
c=JsonLineClient(port=26431);env=ResidentEnv(c,0,26431);frames=0;full_bytes=compact_bytes=0;actions=0
try:
    for seed in (424242,717):
        trace=json.loads((BASE/f'trace-{seed}.json').read_text());env._configure_replay(trace['replay'])
        c.request(dict(op='resident',slot=0,mode='create',replay=trace['replay']));c.request(dict(op='resident',slot=0,mode='step',steps=10))
        for t in range(400):
            full=c.request(dict(op='resident',slot=0,mode='observe'))['state']
            compact=c.request(dict(op='resident',slot=0,mode='observe_train'))['state']
            a,b=normalize_native_state(full),normalize_native_state(compact);assert a==b,'train-schema state mismatch'
            fa=[NativeActorFrame(a,s,env.decks[s]) for s in (0,1)];fb=[NativeActorFrame(b,s,env.decks[s]) for s in (0,1)]
            ea,eb=enc.encode_batch(fa),enc.encode_batch(fb)
            for k in ea:assert torch.equal(ea[k],eb[k]),k
            assert ea.ability_entity_keys==eb.ability_entity_keys and torch.equal(ea.ability_mask,eb.ability_mask)
            full_bytes+=len(json.dumps(full,separators=(',',':')).encode());compact_bytes+=len(json.dumps(compact,separators=(',',':')).encode());frames+=1
            act=next((e['actions'] for e in trace['events'] if e['tick']==a.tick),[])
            result=c.request(dict(op='resident_batch',entries=[dict(slot=0,steps=4,actions=act)],raw_response=True,observation_schema='train-v1'))['results'][0]
            assert all(x['result']['accepted'] for x in result['joint_action']['actions']);actions+=len(act)
    r=dict(passed=True,frames=frames,actor_views=frames*2,accepted_actions=actions,full_json_bytes=full_bytes,compact_json_bytes=compact_bytes,byte_ratio=compact_bytes/full_bytes)
    (ROOT/'compact-verified.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
finally:c.close()
