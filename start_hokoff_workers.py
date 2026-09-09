#!/usr/bin/env python3
"""Start user-owned Bionic workers on consecutive ports; reuse existing services."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=12)
    p.add_argument('--base-port',type=int,default=39031)
    p.add_argument('--runtime-project',type=Path,default=Path.home()/'gh/cr-native-linux-bionic')
    p.add_argument('--template',type=Path,default=Path('/opt/cr-native-bionic/worker0'))
    p.add_argument('--directory',type=Path,default=Path.home()/'cr-data/native-workers/hokoff-ppo')
    p.add_argument('--stop',action='store_true',help='stop only processes previously launched by this helper')
    a=p.parse_args()
    if a.workers<1 or not 1<=a.base_port<=65536-a.workers:raise ValueError('invalid worker count/port range')
    sys.path.insert(0,str(a.runtime_project))
    from cr_native_bionic.runtime import build_worker_command,worker_environment,ready,sha256_file,atomic_json
    a.directory.mkdir(parents=True,exist_ok=True);record_path=a.directory/'workers.json'
    saved=json.loads(record_path.read_text()) if record_path.exists() else {}
    if a.stop:
        for port,row in list(saved.items()):
            if not a.base_port<=int(port)<a.base_port+a.workers:continue
            pid=row['pid'];cmdline=Path(f'/proc/{pid}/cmdline')
            if cmdline.exists():
                command=cmdline.read_bytes().replace(b'\0',b' ').decode()
                if row['directory'] not in command or f'serve-direct {port}' not in command:
                    raise RuntimeError(f'PID identity changed: {pid}')
                os.kill(pid,signal.SIGTERM)
            saved.pop(port)
        atomic_json(record_path,saved);return
    expected='fa6704b83cb9c5b8eecb7b56c9671b834d636a3a6d9ac446e698e1262dc246ba'
    if sha256_file(a.template/'libg.so')!=expected:raise ValueError('wrong frozen libg template')
    for port in range(a.base_port,a.base_port+a.workers):
        if ready(port):
            print(f'port {port}: reuse ready worker',flush=True);continue
        direct=a.directory/f'port-{port}'
        if not direct.exists():shutil.copytree(a.template,direct)
        if sha256_file(direct/'libg.so')!=expected:raise ValueError('worker libg changed')
        (direct/'cache').mkdir(exist_ok=True)
        with (direct/'worker.log').open('ab',buffering=0) as log:
            process=subprocess.Popen(build_worker_command(direct,port),cwd=direct/'assets',
                env=worker_environment(direct),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        saved[str(port)]=dict(pid=process.pid,directory=str(direct.resolve()))
        atomic_json(record_path,saved)
        print(f'port {port}: launched pid {process.pid}',flush=True)
    deadline=time.monotonic()+90
    pending=list(range(a.base_port,a.base_port+a.workers))
    while pending and time.monotonic()<deadline:
        pending=[port for port in pending if not ready(port)]
        if pending:time.sleep(.25)
    if pending:raise RuntimeError(f'workers not ready: {pending}; inspect {a.directory}/port-*/worker.log')
    print(f'ready: {a.workers}; ports: {a.base_port}-{a.base_port+a.workers-1}',flush=True)


if __name__=='__main__':main()
