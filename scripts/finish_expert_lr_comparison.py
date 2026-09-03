"""Finish an already-started control arm, validate both arms, and optionally shut down.

All training is performed by the unchanged trainer in separate run directories.
Failure is preserved and never automatically restarted.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path: sys.path.insert(0, str(PROJECT))
from expert_v1.training_v1.train import _atomic_json, _certify_checkpoint
from expert_v1.training_v1.schema import sha256_file
from scripts.finish_expert_epoch_shutdown import identity, power_off_autodl


def command_for_arm(control_command, run_root, rate, target):
    command = list(control_command)
    if command[:4] != [sys.executable, '-u', '-m', 'expert_v1.training_v1']:
        raise ValueError('Unexpected training entry point')
    for flag in ('--stop-after-epoch', '--stop-at-step'):
        if flag in command:
            i = command.index(flag)
            del command[i:i+2]
    for flag, value in (('--run-id', run_root.name), ('--output-root', str(run_root.parent)),
                        ('--learning-rate', str(rate))):
        if command.count(flag) != 1: raise ValueError(f'Expected one {flag}')
        command[command.index(flag)+1] = value
    return command + ['--stop-at-step', str(target)]


def require_paused(progress, target):
    if progress.get('status') != 'paused' or progress.get('global_step') != target:
        raise RuntimeError('Training exited without the exact requested saved boundary')
    if progress.get('checkpoint', {}).get('status') != 'saved':
        raise RuntimeError('Missing checkpoint save acknowledgement')


def verified_products(root, step):
    import torch
    torch.set_num_threads(2)
    manifest = json.loads((root/'manifest.json').read_text())
    latest = root/'checkpoints/latest.pt'
    state = torch.load(latest, map_location='cpu', mmap=True, weights_only=False)
    _certify_checkpoint(latest, state, dataset_manifest_sha256=manifest['dataset_manifest_sha256'],
        run_signature_sha256=manifest['run_signature_sha256'], model_config=manifest['model'],
        run_id=root.name, optimizer_identity_sha256=manifest['optimizer']['identity_sha256'])
    if state['global_step'] != step: raise RuntimeError('Wrong checkpoint step')
    if not all(math.isfinite(float(x)) for x in state['validation_metrics'].values()):
        raise RuntimeError('Nonfinite validation result')
    if not all(bool(torch.isfinite(x).all()) for x in state['model_state'].values() if x.is_floating_point()):
        raise RuntimeError('Nonfinite model parameters')
    ack = json.loads((root/'control/checkpoint-response.json').read_text())
    products = {'latest': {'path':str(latest), 'sha256':sha256_file(latest)}}
    for name, pathkey, hashkey in (
        ('preserved_checkpoint','preserved_checkpoint','checkpoint_sha256'),
        ('fp16_export','inference_export','inference_export_sha256')):
        path = Path(ack[pathkey])
        digest = sha256_file(path)
        if digest != ack[hashkey]: raise RuntimeError(f'Corrupt {name}')
        products[name] = {'path':str(path), 'sha256':digest, 'bytes':path.stat().st_size}
    return products


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiment-root', type=Path, required=True)
    p.add_argument('--shutdown-on-success', action='store_true')
    args = p.parse_args()
    root = args.experiment_root.resolve()
    control = root/'runs/control-lr1e-4'
    candidate = root/'runs/candidate-lr5e-5'
    start, target = 154674, 157674
    env = os.environ.copy()
    env['CR_EXPERT_TRUST_EXISTING_INTEGRITY'] = '1'
    base = root.parents[1]
    os.chdir(PROJECT)
    def emit(phase, **extra):
        value = {'phase':phase, 'updated_utc':datetime.now(timezone.utc).isoformat(), **extra}
        _atomic_json(root/'comparison-progress.json', value)
        print(json.dumps(value, ensure_ascii=False), flush=True)
    def wait_training(run, pid):
        first = identity(pid)
        previous = None
        while first is not None and identity(pid) == first:
            status = json.loads((run/'training-progress.json').read_text())
            key = (status.get('global_step'), status.get('status'))
            if key != previous:
                previous = key
                emit('training', run_id=run.name, added_steps=int(status.get('global_step',start))-start,
                     target_steps=target-start, progress=status)
            time.sleep(5)
        require_paused(json.loads((run/'training-progress.json').read_text()), target)
    def evaluate(run):
        command = [sys.executable,'-u','-m','expert_v1.training_v1.fork_position_run',
                   'evaluate','--run-root',str(run)]
        with (run/'full-validation.log').open('ab',buffering=0) as log:
            process = subprocess.Popen(command, cwd=PROJECT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        previous = None
        while process.poll() is None:
            path = run/'validation-progress.json'
            value = json.loads(path.read_text()) if path.exists() else {}
            if value.get('global_step') == target and value.get('completed') != previous:
                previous = value.get('completed')
                emit('full_validation',run_id=run.name,progress=value)
            time.sleep(5)
        if process.returncode: raise RuntimeError(f'{run.name} validation failed; see full-validation.log')
        return json.loads((run/f'validation-step-{target}.json').read_text())
    try:
        launch = json.loads((control/'launch.json').read_text())
        if launch['target_step'] != target: raise RuntimeError('Unexpected control target')
        for run in (control, candidate):
            receipt = json.loads((run/'fork-receipt.json').read_text())
            if receipt['resume_step'] != start or not receipt['checkpoint_state_verified']:
                raise RuntimeError('Unauthenticated fork')
        baseline = json.loads((control/f'validation-step-{start}.json').read_text())
        wait_training(control, launch['trainer_pid'])
        results = {'control': evaluate(control)}
        products = {'control': verified_products(control, target)}
        if shutil.disk_usage(root).free < 9*1024**3:
            raise RuntimeError('Not enough free disk for the second arm; preserving first arm')
        command = command_for_arm(launch['command'], candidate, 5e-5, target)
        with (candidate/'train.log').open('ab',buffering=0) as log:
            process = subprocess.Popen(command,cwd=PROJECT,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        cache_cmd = [sys.executable,'-u','scripts/keep_expert_dataset_hot.py','--run-root',str(candidate),
            '--trainer-pid',str(process.pid),'--cache-gib','68','--reserve-gib','12',
            '--status',str(candidate/'ram-cache-status.json')]
        mirror_cmd = [sys.executable,'-u','scripts/mirror_training_progress_to_tensorboard.py',
            '--progress',str(candidate/'training-progress.json'),'--events',str(candidate/'events.jsonl'),
            '--log-dir','/root/tf-logs/lr-ab-20260831/'+candidate.name,
            '--cache-status',str(candidate/'ram-cache-status.json')]
        with (candidate/'cache.log').open('ab',buffering=0) as log:
            cache = subprocess.Popen(cache_cmd,cwd=PROJECT,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        with (candidate/'mirror.log').open('ab',buffering=0) as log:
            mirror = subprocess.Popen(mirror_cmd,cwd=PROJECT,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        record = {'run_id':candidate.name,'run_root':str(candidate),'trainer_pid':process.pid,
            'cache_pid':cache.pid,'mirror_pid':mirror.pid,'resume_step':start,'target_step':target,
            'learning_rate':5e-5,'command':command,'cache_command':cache_cmd,'mirror_command':mirror_cmd,
            'runtime_env':{'CR_EXPERT_TRUST_EXISTING_INTEGRITY':'1'}}
        _atomic_json(candidate/'launch.json',record)
        _atomic_json(base/'active-training-run.json',record)
        wait_training(candidate,process.pid)
        process.wait()
        if process.returncode: raise RuntimeError('Candidate trainer failed')
        results['candidate'] = evaluate(candidate)
        products['candidate'] = verified_products(candidate,target)
        metrics = ('loss','card_top1','card_top3','position_mean_cell_error','position_within_1_cell')
        report = {'status':'complete','source_step':start,'final_step_per_arm':target,'steps_per_arm':3000,
            'baseline':baseline,'results':results,'artifacts':products,
            'candidate_minus_control':{k:results['candidate']['validation'][k]-results['control']['validation'][k] for k in metrics},
            'ability_label_audit':json.loads((root/'ability-label-audit.json').read_text()),
            'limits':'Single-seed short run; no automatic deployment or extension; not a win-rate evaluation'}
        _atomic_json(root/'report.json',report)
        emit('complete',report=report)
        os.sync()
        if args.shutdown_on_success:
            trash = Path('/root/.local/share/Trash')
            if trash.exists() and (not trash.is_dir() or any(trash.iterdir())):
                raise RuntimeError('Refusing provider shutdown because it would empty nonempty Trash')
            emit('shutdown_requested',report_path=str(root/'report.json'))
            os.sync()
            power_off_autodl()
    except Exception as error:
        emit('failed_no_restart',error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__': main()
