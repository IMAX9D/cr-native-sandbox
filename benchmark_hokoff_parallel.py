#!/usr/bin/env python3
"""Compare complete collection+PPO iterations; requires ready native workers."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,nargs='+',default=[4,8,12])
    p.add_argument('--episodes',type=int,default=12)
    p.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output',type=Path,default=None)
    a=p.parse_args()
    if not a.workers or min(a.workers)<1 or a.episodes<max(a.workers):raise ValueError('episodes must be >= every worker count')
    output=a.output or Path.home()/'cr-data/runs'/('hokoff-parallel-benchmark-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
    output.mkdir(parents=True,exist_ok=False);rows=[]
    for workers in a.workers:
        run=output/f'workers-{workers}'
        subprocess.run([sys.executable,str(Path(__file__).with_name('train_hokoff_ppo.py')),
            '--device',a.device,'--seed',str(a.seed),'--workers',str(workers),
            '--episodes-per-iteration',str(a.episodes),'--iterations','1','--output',str(run)],check=True)
        result=json.loads((run/'iteration-0000-update.json').read_text())
        row=dict(workers=workers,**result['collection'],update_seconds=result['update_seconds'],
            iteration_seconds=result['elapsed_seconds'],peak_cuda_mb=result['peak_cuda_mb'])
        row['iteration_ticks_per_second']=row['native_ticks']/row['iteration_seconds']
        rows.append(row);(output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
