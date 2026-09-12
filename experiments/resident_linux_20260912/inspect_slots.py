import json,sys
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/bc-cloud-bench-20260911/core')
from native_core.client import JsonLineClient
c=JsonLineClient(host='127.0.0.1',port=39331,timeout=10)
try:
    states=[c.request(dict(op='resident',slot=i,mode='observe'))['state'] for i in range(4)]
    Path(__file__).with_name('pause-slot-states.json').write_text(json.dumps(states,indent=2))
    for i,s in enumerate(states):print(json.dumps(dict(slot=i,tick=s['tick'],episode=s['episode'])))
finally:c.close()
