"""Baseline binary/flags unchanged; benchmark listeners below ephemeral port range."""
import sys
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/resident-linux-20260912')
import pool_launch
pool_launch.BASE_PORT=24431
if __name__=='__main__':pool_launch.main()
