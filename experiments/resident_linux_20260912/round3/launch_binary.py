"""Separate binary-wire Java host over unchanged round2 native library."""
import sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,'/root/autodl-tmp/resident-linux-20260912')
import pool_launch
pool_launch.ROOT=ROOT;pool_launch.POOL=ROOT/'pool';pool_launch.BASE_PORT=26431
os.environ['CR_NATIVE_RESIDENT_BIND_CACHE']='1'
if __name__=='__main__':pool_launch.main()
