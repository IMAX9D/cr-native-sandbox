"""Explicit, audited LR-only fork of a fully resumable expert checkpoint."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
import numpy as np

from . import train
from .schema import sha256_file


def args_from_manifest(manifest: Mapping[str, Any]) -> argparse.Namespace:
    args = train.build_parser().parse_args([])
    for name in vars(args):
        if name in manifest['training']:
            setattr(args, name, manifest['training'][name])
    args.device = manifest['training']['device_request']
    args.dataset_root = Path(manifest['dataset_root'])
    args.expected_source_manifest = Path(manifest['source_manifest']['path'])
    args.run_id = manifest['run_id']
    args.resume = True
    return args


def migrate_lr_checkpoint(source: Mapping[str, Any], *, learning_rate: float,
                          run_id: str, signature: str, optimizer_identity: str) -> dict[str, Any]:
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('learning rate must be finite and positive')
    groups = deepcopy(source['optimizer_state']['param_groups'])
    if len(groups) != 1:
        raise ValueError('LR-only migration supports this trainer\'s single AdamW group')
    scheduler = deepcopy(source['scheduler_state'])
    if len(scheduler.get('base_lrs', [])) != 1 or len(scheduler.get('_last_lr', [])) != 1:
        raise ValueError('incompatible constant-LR scheduler state')
    groups[0]['lr'] = learning_rate
    groups[0]['initial_lr'] = learning_rate
    scheduler['base_lrs'] = [learning_rate]
    scheduler['_last_lr'] = [learning_rate]
    # Tensors and moment buffers are deliberately not rebuilt or reinitialized.
    value = dict(source)
    value['optimizer_state'] = {**source['optimizer_state'], 'param_groups': groups}
    value['scheduler_state'] = scheduler
    value['run_id'] = run_id
    value['run_signature_sha256'] = signature
    value['optimizer_identity_sha256'] = optimizer_identity
    value['fork_origin'] = {
        'run_id': source['run_id'], 'step': source['global_step'],
        'learning_rate': source['optimizer_state']['param_groups'][0]['lr'],
        'optimizer_moments_preserved': True, 'rng_preserved': True,
    }
    return value


def assert_tensors_equal(source: Mapping[str, Any], restored: Mapping[str, Any], *,
                         expected_model_config: Mapping[str, Any] | None = None) -> None:
    for key, value in source['model_state'].items():
        if not torch.equal(value.cpu(), restored['model_state'][key].cpu()):
            raise RuntimeError(f'model weight changed in LR-only fork: {key}')
    for parameter, state in source['optimizer_state']['state'].items():
        for key, value in state.items():
            other = restored['optimizer_state']['state'][parameter][key]
            equal = torch.equal(value.cpu(), other.cpu()) if isinstance(value, torch.Tensor) else value == other
            if not equal:
                raise RuntimeError(f'optimizer state changed: {parameter}/{key}')
    for key in ['torch', 'train_loader_generator']:
        if not torch.equal(source['rng'][key].cpu(), restored['rng'][key].cpu()):
            raise RuntimeError(f'RNG state changed: {key}')
    if source['rng']['python'] != restored['rng']['python']:
        raise RuntimeError('Python RNG state changed')
    first_numpy, second_numpy = source['rng']['numpy'], restored['rng']['numpy']
    if (first_numpy[0] != second_numpy[0] or first_numpy[2:] != second_numpy[2:]
            or not np.array_equal(first_numpy[1], second_numpy[1])):
        raise RuntimeError('NumPy RNG state changed')
    if len(source['rng'].get('cuda') or []) != len(restored['rng'].get('cuda') or []):
        raise RuntimeError('CUDA RNG device count changed')
    for first, second in zip(source['rng'].get('cuda') or [], restored['rng'].get('cuda') or []):
        if not torch.equal(first.cpu(), second.cpu()):
            raise RuntimeError('CUDA RNG state changed')
    if not torch.equal(source['epoch_start_train_generator_state'].cpu(),
                       restored['epoch_start_train_generator_state'].cpu()):
        raise RuntimeError('epoch shuffle RNG changed')
    for key in ['global_step', 'step', 'epoch', 'epoch_complete', 'batch_in_epoch',
                'batches_in_epoch', 'normalizer_state']:
        if source[key] != restored[key]:
            raise RuntimeError(f'checkpoint counter/config changed: {key}')
    expected_config = source['model_config'] if expected_model_config is None else dict(expected_model_config)
    if restored['model_config'] != expected_config:
        raise RuntimeError('checkpoint model configuration differs from the explicitly expected contract')


def create_fork(source_run: Path, checkpoint: Path, output_root: Path,
                run_id: str, learning_rate: float, expected_step: int, *,
                allow_equal_learning_rate: bool = False) -> dict[str, Any]:
    manifest = json.loads((source_run/'manifest.json').read_text(encoding='utf-8-sig'))
    args = args_from_manifest(manifest)
    source_signature, _ = train._run_signature(args,
        dataset_manifest_sha256=manifest['dataset_manifest_sha256'],
        observation_mode=manifest['training']['observation_mode'])
    source_identity, _ = train._optimizer_identity(args, run_id=args.run_id, model_config=manifest['model'])
    if source_signature != manifest['run_signature_sha256'] or source_identity != manifest['optimizer']['identity_sha256']:
        raise RuntimeError('source runtime does not match the frozen run contract')
    original = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    train._certify_checkpoint(checkpoint, original,
        dataset_manifest_sha256=manifest['dataset_manifest_sha256'],
        run_signature_sha256=source_signature, model_config=manifest['model'],
        run_id=args.run_id, optimizer_identity_sha256=source_identity)
    if original['global_step'] != expected_step:
        raise RuntimeError('requested rollback step does not match the checkpoint')
    if learning_rate > args.learning_rate or (learning_rate == args.learning_rate and not allow_equal_learning_rate):
        raise ValueError('this recovery command only permits reducing the learning rate')
    if run_id == args.run_id:
        raise ValueError('a recovery fork requires a distinct run ID')
    if Path(run_id).name != run_id or run_id in {'.', '..'} or '/' in run_id or '\\' in run_id:
        raise ValueError('run ID must be a single directory name')
    args.learning_rate = learning_rate
    args.run_id = run_id
    args.output_root = output_root.resolve()
    signature, training_config = train._run_signature(args,
        dataset_manifest_sha256=manifest['dataset_manifest_sha256'],
        observation_mode=manifest['training']['observation_mode'])
    changed = {key for key in training_config if training_config[key] != manifest['training'].get(key)}
    expected_changes = set() if learning_rate == manifest['training']['learning_rate'] else {'learning_rate'}
    if changed != expected_changes:
        raise RuntimeError(f'unexpected hyperparameter changes: {changed}')
    identity, optimizer_config = train._optimizer_identity(args, run_id=run_id, model_config=manifest['model'])
    destination = args.output_root / run_id
    destination.mkdir(parents=True, exist_ok=False)
    lineage = {'source_run':str(source_run.resolve()), 'source_run_id':manifest['run_id'],
               'source_checkpoint':str(checkpoint.resolve()), 'source_checkpoint_sha256':sha256_file(checkpoint),
               'source_step':expected_step, 'changes':{'learning_rate':[manifest['training']['learning_rate'],learning_rate]},
               'optimizer_moments_preserved':True, 'rng_preserved':True}
    new_manifest = {**manifest, 'run_id':run_id, 'created_utc':datetime.now(timezone.utc).isoformat(),
                    'training':training_config, 'run_signature_sha256':signature,
                    'optimizer':{**optimizer_config,'identity_sha256':identity}, 'fork_lineage':lineage}
    train._atomic_json(destination/'manifest.json', new_manifest)
    target = migrate_lr_checkpoint(original, learning_rate=learning_rate, run_id=run_id,
                                   signature=signature, optimizer_identity=identity)
    target['checkpoint_role'] = 'latest'
    train._atomic_torch(destination/'checkpoints/latest.pt', target)
    restored = torch.load(destination/'checkpoints/latest.pt', map_location='cpu', weights_only=False, mmap=True)
    assert_tensors_equal(original, restored)
    best_source = source_run/'checkpoints/best.pt'
    if best_source.is_file():
        best = torch.load(best_source, map_location='cpu', weights_only=False, mmap=True)
        train._certify_checkpoint(best_source, best, dataset_manifest_sha256=manifest['dataset_manifest_sha256'],
            run_signature_sha256=source_signature, model_config=manifest['model'],
            run_id=manifest['run_id'], optimizer_identity_sha256=source_identity)
        if best['global_step'] > expected_step:
            raise RuntimeError('source best checkpoint is newer than rollback target')
        migrated_best = migrate_lr_checkpoint(best, learning_rate=learning_rate, run_id=run_id,
                                              signature=signature, optimizer_identity=identity)
        migrated_best['checkpoint_role'] = 'best'
        train._atomic_torch(destination/'checkpoints/best.pt', migrated_best)
    _, selected = train._load_resume_checkpoint(destination, torch.device('cpu'),
        dataset_manifest_sha256=manifest['dataset_manifest_sha256'], run_signature_sha256=signature,
        model_config=manifest['model'], run_id=run_id, optimizer_identity_sha256=identity)
    if selected['global_step'] != expected_step:
        raise RuntimeError('resume selected a different step than requested')
    train._append_jsonl(destination/'events.jsonl', {'event':'lr_only_fork_created', 'global_step':expected_step,
        'created_utc':datetime.now(timezone.utc).isoformat(), **lineage})
    train._atomic_json(destination/'training-progress.json', {'kind':'cr_expert_training_progress_v1',
        'status':'ready_to_resume', 'epoch':original['epoch'], 'epochs':args.epochs,
        'batch':original['batch_in_epoch'], 'batches':original['batches_in_epoch'], 'global_step':expected_step,
        'learning_rate':learning_rate, 'updated_utc':datetime.now(timezone.utc).isoformat()})
    receipt = {'ok':True, 'run_root':str(destination), 'run_id':run_id,
               'resume_step':expected_step, 'learning_rate':learning_rate,
               'checkpoint_state_verified':True, 'lineage':lineage}
    train._atomic_json(destination/'fork-receipt.json', receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--learning-rate', type=float, required=True)
    parser.add_argument('--expected-step', type=int, required=True)
    parser.add_argument('--allow-equal-learning-rate', action='store_true',
                        help='explicitly permit an unchanged-LR control arm; never permits increasing LR')
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(create_fork(args.source_run, args.checkpoint, args.output_root,
        args.run_id, args.learning_rate, args.expected_step,
        allow_equal_learning_rate=args.allow_equal_learning_rate), ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
