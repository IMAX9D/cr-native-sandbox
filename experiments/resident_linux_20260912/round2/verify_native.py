"""Golden-state and reset-isolation gates on the NEW port, original evidence read-only."""
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;BASE=Path('/root/autodl-tmp/resident-linux-20260912')
sys.path.insert(0,str(BASE))
import verify_four
for seed in (424242,717):
    for name in (f'trace-{seed}.json',f'baseline-{seed}.json.gz'):
        target=ROOT/name
        if not target.exists():target.symlink_to(BASE/name)
verify_four.ROOT=ROOT
OriginalClient=verify_four.JsonLineClient
def client(**kwargs):
    kwargs['port']=42431
    return OriginalClient(**kwargs)
verify_four.JsonLineClient=client
if __name__=='__main__':verify_four.main()
