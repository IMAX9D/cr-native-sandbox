"""Read-only full state equality; then exact golden-state transition verification."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;BASE=Path('/root/autodl-tmp/resident-linux-20260912')
sys.path.insert(0,str(BASE));sys.path.insert(0,'/root/autodl-tmp/bc-cloud-bench-20260911/core')
from native_core.client import JsonLineClient
from binary_client import BinaryClient

# Server accepts one client at a time, close each before the next comparison.
for slot in range(4):
    for wire in (False,True):
        c=(BinaryClient if wire else JsonLineClient)(port=26431)
        try:r=c.request(dict(op='resident',slot=slot,mode='create',replay=json.loads((BASE/'trace-424242.json').read_text())['replay']))
        finally:c.close()
    a=JsonLineClient(port=26431)
    try:r1=a.request(dict(op='resident',slot=slot,mode='observe'))
    finally:a.close()
    b=BinaryClient(port=26431)
    try:r2=b.request(dict(op='resident',slot=slot,mode='observe'))
    finally:b.close()
    assert r1==r2,'binary state mismatch'
import verify_four
for seed in (424242,717):
    for name in (f'trace-{seed}.json',f'baseline-{seed}.json.gz'):
        p=ROOT/name
        if not p.exists():p.symlink_to(BASE/name)
verify_four.ROOT=ROOT
verify_four.JsonLineClient=lambda **kw:BinaryClient(port=26431)
verify_four.main()
