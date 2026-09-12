import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,'/root/autodl-tmp/bc-cloud-bench-20260911/core')
from native_core.env import NativeRoyaleEnv
from native_core.client import JsonLineClient
from compare import normalize,differences
r=json.loads((ROOT/'trace-424242.json').read_text())['replay']
e=NativeRoyaleEnv(port=39331)
try:
    e.reset(r,warmup_steps=0);e.step(2)
    for _ in range(2000):
        result=e.step(4)
        if result['episode']['terminated']:break
    else:raise AssertionError('original draw timeout')
    expected=normalize(e.client.request({'op':'observe'})['state'])
finally:e.close()
c=JsonLineClient(host='127.0.0.1',port=39331)
try:
    c.request(dict(op='resident',slot=0,mode='create',replay=r));c.request(dict(op='resident',slot=0,mode='step',steps=12))
    extra=0
    for _ in range(2000):
        result=c.request(dict(op='resident_batch',entries=[dict(slot=0,steps=4,actions=[])]))['results'][0]
        extra+=result['step']['settlement_updates']
        if result['step']['episode']['terminated']:break
        assert result['step']['tick_after']-result['step']['tick_before']==4
    else:raise AssertionError('resident draw timeout')
    actual=normalize(result['state']);diff=differences(expected,actual)
    assert not diff,diff
    assert extra>0,'fixture did not exercise partial interval settlement'
    out=dict(passed=True,tick=actual['tick'],outcome=actual['episode']['outcome'],settlement_updates=extra)
    (ROOT/'draw-correctness.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
finally:c.close()
