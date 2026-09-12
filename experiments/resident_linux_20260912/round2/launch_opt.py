"""Separate native pool and ports; no replacement of the certified baseline pool."""
import argparse, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE=Path('/root/autodl-tmp/resident-linux-20260912')
sys.path.insert(0,str(BASE))
import pool_launch
pool_launch.ROOT=ROOT
pool_launch.POOL=ROOT/'pool'
pool_launch.BASE_PORT=25431
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('op',choices=['start','stop']);p.add_argument('--count',type=int,default=1)
    p.add_argument('--cache',action='store_true');p.add_argument('--profile',action='store_true');a=p.parse_args()
    os.environ['CR_NATIVE_RESIDENT_BIND_CACHE']='1' if a.cache else '0'
    # The baseline launcher explicitly sets its timing option.
    pool_launch.PROFILE_TIMING=a.profile
    args=['launch_opt.py',a.op,'--count',str(a.count)]
    if a.profile:args.append('--profile-timing')
    sys.argv=args;pool_launch.main()
