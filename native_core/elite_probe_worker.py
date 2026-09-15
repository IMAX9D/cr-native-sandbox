"""Deploy an isolated replay extraction host to an explicitly selected ADB device."""
from __future__ import annotations
import argparse,hashlib,json,shlex,subprocess,time
from pathlib import Path,PurePosixPath
from .client import JsonLineClient

RUNTIME_SHA='fa6704b83cb9c5b8eecb7b56c9671b834d636a3a6d9ac446e698e1262dc246ba'


def start(*,adb:Path,serial:str,artifacts:Path,port:int=37032,source_root:str='/data/local/tmp/cr-native-direct-0'):
    if not serial or not 1024<=port<=65535:raise ValueError('explicit serial and port required')
    root=f'/data/local/tmp/cr-native-direct-elite-{port}'
    if '..' in PurePosixPath(source_root).parts or not source_root.startswith('/data/local/tmp/cr-native-direct-') or source_root==root:raise ValueError('invalid source runtime directory')
    def call(*args,timeout=30):
        result=subprocess.run([str(adb),'-s',serial,*args],capture_output=True,timeout=timeout,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if result.returncode:raise RuntimeError(result.stderr.decode('utf-8',errors='replace') or result.stdout.decode('utf-8',errors='replace'))
        return result.stdout.decode('utf-8',errors='strict').strip()
    actual=call('shell','sha256sum',source_root+'/libg.so').split()[0]
    if actual!=RUNTIME_SHA:raise ValueError('replay telemetry requires the frozen x86_64 library')
    for name in ('lifecycle-probe.jar','libnative_core_probe.so'):
        if not (artifacts/name).is_file():raise FileNotFoundError(artifacts/name)
    q=shlex.quote
    exists=call('shell',f'if [ -d {q(root)} ]; then echo yes; else echo no; fi')=='yes'
    if not exists:
        call('shell',f'cp -a {q(source_root)} {q(root)} && echo elite-replay-v1 >{q(root)}/.elite-replay-owner')
    elif call('shell','cat',root+'/.elite-replay-owner')!='elite-replay-v1':raise ValueError('refusing to replace a non-owned host')
    # Exact root/port match only; no global app_process or emulator shutdown.
    for line in call('shell','ps -A -o PID,ARGS').splitlines():
        if 'royale.nativehost.JniHost '+root+' serve-direct '+str(port) in line:
            pid=line.split()[0]
            if not pid.isdigit():raise ValueError('invalid owned host PID')
            call('shell','kill',pid)
    for local,remote in [('lifecycle-probe.jar','lifecycle-probe.jar'),('libnative_core_probe.so','libnative_host_bridge.so')]:
        path=artifacts/local;call('push',str(path),root+'/'+remote)
        if call('shell','sha256sum',root+'/'+remote).split()[0]!=hashlib.sha256(path.read_bytes()).hexdigest():raise ValueError('host upload hash differs')
    call('forward',f'tcp:{port}',f'tcp:{port}')
    settings='CR_BINDERLESS_ANDROID=1 CR_BINDERLESS_SKIP_FRAMEWORK_REGISTRATION=1 CR_BINDERLESS_NATIVE_CONFIG_GETTERS=1 CR_BINDERLESS_NATIVE_CONFIG_POSTPROCESS=1 CR_NATIVE_LOADING_MAX_FRAMES=10000 CR_NATIVE_LOADING_TIMEOUT_MS=30000 CR_NATIVE_LOADING_SLEEP_MS=5'
    cp=f'{root}/lifecycle-probe.jar:{root}/base.apk:/system/framework/android.test.base.jar:/system/framework/android.test.mock.jar'
    command=f'cd {q(root)} && env {settings} CLASSPATH={q(cp)} LD_LIBRARY_PATH={q(root)} nohup app_process /system/bin royale.nativehost.JniHost {q(root)} serve-direct {port} >{q(root)}/service-elite.log 2>&1 </dev/null &'
    launch=subprocess.Popen([str(adb),'-s',serial,'shell',command],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    deadline=time.monotonic()+50;last_error=''
    while time.monotonic()<deadline:
        client=JsonLineClient(port=port,timeout=.5)
        try:
            response=client.request({'op':'runtime_identity_v1'})
            if response.get('ok') and response.get('identity',{}).get('libg_sha256')==RUNTIME_SHA:
                return dict(ready=True,serial=serial,port=port,remote_root=root,identity=response['identity'],launch_adb_pid=launch.pid)
        except (OSError,ValueError,ConnectionError) as error:last_error=str(error)
        finally:client.close()
        time.sleep(.5)
    raise RuntimeError('isolated elite host did not become ready: '+last_error)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--adb',type=Path,required=True);parser.add_argument('--serial',required=True)
    parser.add_argument('--artifacts',type=Path,required=True);parser.add_argument('--port',type=int,default=37032);parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=start(adb=args.adb,serial=args.serial,artifacts=args.artifacts,port=args.port)
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True,indent=2))
