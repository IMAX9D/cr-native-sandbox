"""Isolated pool: one Bionic process / four native matches / unique endpoint."""
import argparse,concurrent.futures,hashlib,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;POOL=ROOT/'pool';BASE_PORT=41431
PROFILE_TIMING=False
sys.path.insert(0,'/root/autodl-tmp/linux-arch-test-20260910/linux')
from cr_native_bionic.runtime import build_worker_command,worker_environment,ready,ensure_properties

def stop_one(index):
    path=POOL/f'worker-{index:02d}.json'
    if not path.exists():return
    r=json.loads(path.read_text());p=Path(f"/proc/{r['pid']}/cmdline")
    if p.exists() and p.read_bytes():
        cmd=p.read_bytes().split(b'\0');expected=[x.encode() for x in r['command']]
        if cmd[:len(expected)]!=expected:raise RuntimeError(('PID ownership mismatch',index))
        os.kill(r['pid'],signal.SIGTERM)
        for _ in range(50):
            if not p.exists() or not p.read_bytes():break
            time.sleep(.1)
        else:raise RuntimeError(('worker did not stop',index))

def start_one(index):
    stop_one(index);port=BASE_PORT+index;direct=POOL/f'slot-{index:02d}'
    if ready(port):raise RuntimeError(('occupied port',port))
    if not direct.exists():shutil.copytree('/data/local/tmp/cr-native-direct-0',direct)
    for name in ('libnative_host_bridge.so','lifecycle-probe.jar'):
        src=('libnative_host_bridge_profile.so' if name.endswith('.so') else 'lifecycle-probe-profile.jar') if PROFILE_TIMING else name
        shutil.copy2(ROOT/src,direct/name)
    assert hashlib.sha256((direct/'libg.so').read_bytes()).hexdigest()=='fa6704b83cb9c5b8eecb7b56c9671b834d636a3a6d9ac446e698e1262dc246ba'
    command=build_worker_command(direct,port,execution_mode='jit');env=worker_environment(direct)
    env.update(CR_NATIVE_EPISODE_READ_CACHE='0',CR_NATIVE_TOWER_CAPTURE_FAST='0',CR_NATIVE_PROFILE_MEMORY_READS='0')
    env['CR_NATIVE_PROFILE_TIMING']='1' if PROFILE_TIMING else '0'
    with (POOL/f'worker-{index:02d}.log').open('ab') as log:
        p=subprocess.Popen(command,cwd=direct/'assets',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    record=dict(index=index,pid=p.pid,port=port,command=command,started=time.time())
    (POOL/f'worker-{index:02d}.json').write_text(json.dumps(record))
    for _ in range(180):
        if p.poll() is not None:raise RuntimeError(('worker exited',index,p.returncode))
        if ready(port):return record
        time.sleep(.5)
    stop_one(index);raise TimeoutError(('worker bootstrap',index))

def main():
    global PROFILE_TIMING
    p=argparse.ArgumentParser();p.add_argument('op',choices=['start','stop']);p.add_argument('--count',type=int,default=1)
    p.add_argument('--profile-timing',action='store_true')
    a=p.parse_args();PROFILE_TIMING=a.profile_timing;assert 1<=a.count<=24;POOL.mkdir(exist_ok=True)
    if a.op=='stop':
        for i in range(a.count):stop_one(i)
        print('owned pool stopped',a.count,flush=True);return
    ensure_properties(Path('/root/autodl-tmp/linux-arch-test-20260910/cr-native-linux-bionic-runtime-150535029'))
    rows=[]
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4,a.count)) as pool:
            for row in pool.map(start_one,range(a.count)):
                rows.append(row);print('ready',row['index'],row['pid'],flush=True)
    except BaseException:
        for i in range(a.count):stop_one(i)
        raise
    (POOL/'active.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
