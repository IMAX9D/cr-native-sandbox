"""Isolated Linux/Bionic worker; never targets the cloned production slots."""
import hashlib,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,'/root/autodl-tmp/linux-arch-test-20260910/linux')
from cr_native_bionic.runtime import build_worker_command,worker_environment,ready,ensure_properties
WORKER=ROOT/'worker';PORT=39331
record=ROOT/'launch.json'
def stop():
    if not record.exists():return
    r=json.loads(record.read_text());pid=r['pid'];p=Path(f'/proc/{pid}/cmdline')
    if p.exists():
        cmd=p.read_bytes().replace(b'\0',b' ').decode()
        if str(WORKER) not in cmd or str(PORT) not in cmd:raise RuntimeError('refuse unrelated PID')
        os.kill(pid,signal.SIGTERM)
        for _ in range(50):
            if not p.exists() or not p.read_bytes():break
            time.sleep(.1)
        else:raise RuntimeError('worker did not stop')
    print('isolated worker stopped',flush=True)
def start():
    stop()
    if ready(PORT):raise RuntimeError('port occupied')
    if not WORKER.exists():shutil.copytree('/data/local/tmp/cr-native-direct-0',WORKER)
    for name in ('libnative_host_bridge.so','lifecycle-probe.jar'):
        shutil.copy2(ROOT/name,WORKER/name)
    assert hashlib.sha256((WORKER/'libg.so').read_bytes()).hexdigest()=='fa6704b83cb9c5b8eecb7b56c9671b834d636a3a6d9ac446e698e1262dc246ba'
    ensure_properties(Path('/root/autodl-tmp/linux-arch-test-20260910/cr-native-linux-bionic-runtime-150535029'))
    cmd=build_worker_command(WORKER,PORT,execution_mode='jit');env=worker_environment(WORKER)
    env['CR_NATIVE_EPISODE_READ_CACHE']='0'
    env['CR_NATIVE_TOWER_CAPTURE_FAST']='1' if '--tower-fast' in sys.argv else '0'
    env['CR_NATIVE_PROFILE_MEMORY_READS']='1' if '--profile' in sys.argv else '0'
    with (ROOT/'worker.log').open('ab') as log:
        p=subprocess.Popen(cmd,cwd=WORKER/'assets',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    record.write_text(json.dumps(dict(pid=p.pid,command=cmd,started=time.time(),tower_fast=env['CR_NATIVE_TOWER_CAPTURE_FAST'],profile=env['CR_NATIVE_PROFILE_MEMORY_READS'])))
    for _ in range(60):
        if p.poll() is not None:raise RuntimeError(f'worker exit {p.returncode}')
        if ready(PORT):print('ready',p.pid,flush=True);return
        time.sleep(.5)
    stop();raise TimeoutError('worker startup')
if __name__=='__main__':stop() if 'stop' in sys.argv else start()
