"""One-shot cloud job: verify a completed epoch, preserve artifacts, then power off.

Not a timer: never powers off on elapsed time, missing progress, or trainer failure.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[19] if fields[0] not in ('Z', 'X') else None
    except FileNotFoundError:
        return None


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require_completed(progress, epoch, step):
    if not (progress.get('status') == 'paused' and progress.get('reason') == 'stop_after_epoch'
            and progress.get('epoch') == epoch and progress.get('global_step') == step):
        raise RuntimeError('Trainer did not reach the requested completed-epoch boundary; NOT shutting down')


def power_off_autodl():
    # AutoDL installs a shell script without a shebang. Direct execve gives
    # ENOEXEC; use the fixed interpreter and fixed provider script, no shell string.
    subprocess.run(['/bin/bash', '/usr/bin/shutdown'], check=True, timeout=30)


def validate_and_preserve(root, epoch, step, tensorboard_root=Path('/root/tf-logs')):
    import torch
    torch.set_num_threads(4)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['run_id'] != root.name:
        raise RuntimeError('Run identity mismatch')
    paths = {'latest': root / 'checkpoints/latest.pt', 'best': root / 'checkpoints/best.pt',
             'epoch': root / f'checkpoints/epochs/epoch-{epoch:03d}.pt',
             'export': root / f'exports/epochs/epoch-{epoch:03d}-fp16.pt'}
    def finite(value):
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError('Nonfinite checkpoint tensor')
        elif isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError('Nonfinite checkpoint metric')
        elif isinstance(value, dict):
            for item in value.values(): finite(item)
        elif isinstance(value, (list, tuple)):
            for item in value: finite(item)
    metadata = {}
    validation = None
    for label, path in paths.items():
        checkpoint = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        if checkpoint.get('run_id') != root.name:
            raise RuntimeError(f'{label}: wrong run')
        if label != 'best' and (checkpoint.get('epoch') != epoch or checkpoint.get('global_step') != step):
            raise RuntimeError(f'{label}: wrong epoch or step')
        if not checkpoint.get('model_state'):
            raise RuntimeError(f'{label}: missing model')
        finite(checkpoint['model_state'])
        if label != 'export':
            for key in ('optimizer_state', 'scheduler_state', 'rng', 'normalizer_state'):
                if not checkpoint.get(key): raise RuntimeError(f'{label}: missing {key}')
            finite(checkpoint['optimizer_state'])
            finite(checkpoint['validation_metrics'])
            if label != 'best' and not checkpoint.get('epoch_complete'):
                raise RuntimeError(f'{label}: incomplete epoch')
            if label == 'latest': validation = checkpoint['validation_metrics']
        metadata[label] = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)}
        del checkpoint
    backup = root / f'checkpoints/manual/completed-epoch-{epoch:03d}-shutdown-{step}'
    if backup.exists(): raise RuntimeError('Backup destination already exists; refusing overwrite')
    required = sum(paths[k].stat().st_size for k in ('latest', 'best', 'export'))
    if shutil.disk_usage(root).free < required + 2 * 1024**3:
        raise RuntimeError('Insufficient free disk for protected backup')
    backup.mkdir(parents=True)
    for key in ('latest', 'best', 'export'):
        dest = backup / paths[key].name
        shutil.copy2(paths[key], dest)
        if digest(dest) != metadata[key]['sha256']: raise RuntimeError('Backup hash mismatch')
    for name in ('manifest.json', 'events.jsonl', 'training-progress.json', 'launch.json'):
        shutil.copy2(root / name, backup / name)
    tb = tensorboard_root / root.name
    if not tb.is_dir() or not any(tb.glob('events.out.tfevents.*')):
        raise RuntimeError('TensorBoard event files missing')
    shutil.copytree(tb, backup / 'tensorboard')
    result = {'epoch': epoch, 'global_step': step, 'validation': validation,
              'artifacts': metadata, 'protected_backup': str(backup)}
    atomic_json(backup / 'receipt.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--trainer-pid', type=int, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--execute-shutdown', action='store_true')
    args = parser.parse_args()
    root = args.run_root.resolve()
    status_path = root / 'control/epoch-shutdown-status.json'
    import fcntl
    with (root / 'control/epoch-shutdown.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def status(state, **extra):
            value = {'status': state, 'run_id': root.name, 'epoch': args.epoch,
                     'target_step': args.step, 'trainer_pid': args.trainer_pid,
                     'updated_utc': datetime.now(timezone.utc).isoformat(), **extra}
            atomic_json(status_path, value)
            print(json.dumps(value, ensure_ascii=False), flush=True)
        try:
            launch = json.loads((root / 'launch.json').read_text())
            if launch['trainer_pid'] != args.trainer_pid or launch.get('stop_after_epoch') != args.epoch:
                raise RuntimeError('Launch does not match authorized epoch stop')
            start = identity(args.trainer_pid)
            if start is None: raise RuntimeError('Trainer not alive when arming shutdown')
            status('armed', shutdown_authorized=args.execute_shutdown)
            while identity(args.trainer_pid) == start:
                time.sleep(20)
            require_completed(json.loads((root / 'training-progress.json').read_text()), args.epoch, args.step)
            status('verifying_and_preserving')
            receipt = validate_and_preserve(root, args.epoch, args.step)
            status('saved_and_verified', receipt=receipt)
            os.sync()
            if args.execute_shutdown:
                # AutoDL's documented wrapper also empties Trash: do not discard user data.
                trash = Path('/root/.local/share/Trash')
                if trash.exists() and (not trash.is_dir() or any(trash.iterdir())):
                    raise RuntimeError('AutoDL shutdown would empty nonempty Trash; refusing')
                status('shutdown_requested', receipt=receipt)
                os.sync()
                power_off_autodl()
        except Exception as exc:
            status('failed_no_shutdown', error=f'{type(exc).__name__}: {exc}')
            raise


if __name__ == '__main__':
    main()
