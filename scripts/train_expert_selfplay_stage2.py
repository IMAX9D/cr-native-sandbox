"""Execute one guarded Stage-2 recurrent PPO update from one CLOSED shard."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import shutil
import sys
import traceback
from typing import Any, Mapping

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from expert_selfplay_v1.actor_adapter import actor_state_digest
from expert_selfplay_v1.contracts import BatchManifest
from expert_selfplay_v1.ledger import RolloutLedger
from expert_selfplay_v1.stage2_training import (
    Stage2PPOTrainer,
    Stage2TrainingConfig,
    _file_sha256,
    _state_digest,
)
from scripts.run_expert_selfplay_v1 import RunJournal, atomic_json


RESULT_KIND = "cr_native_expert_selfplay_stage2_update_result_v1"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def _collection_run(shard: Path) -> Path:
    run = shard.resolve(strict=True).parents[1]
    if shard.parent.name != "rollouts" or not (run / "rollout-ledger.sqlite").is_file():
        raise RuntimeError("shard is not inside a collection run")
    return run


def _link_pre_update(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"Stage-2 run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    journal = RunJournal(run_dir)
    ledgers: list[tuple[RolloutLedger, str]] = []
    try:
        torch.set_num_threads(args.cpu_threads)
        shards = [value.resolve(strict=True) for value in args.shard]
        if not shards or len(set(shards)) != len(shards):
            raise ValueError("Stage-2 requires distinct rollout shards")
        collection_runs = [_collection_run(shard) for shard in shards]
        batch_manifests = []
        for collection_run in collection_runs:
            collection_manifest = _read(collection_run / "manifest.json")
            batch_manifest = BatchManifest(**dict(collection_manifest["batch_manifest"]))
            batch_manifest.validate()
            ledger = RolloutLedger(collection_run / "rollout-ledger.sqlite")
            if ledger.state(batch_manifest.batch_id) != "CLOSED":
                ledger.close()
                raise RuntimeError("Stage-2 accepts only CLOSED, unconsumed rollout batches")
            batch_manifests.append(batch_manifest)
            ledgers.append((ledger, batch_manifest.batch_id))

        config = Stage2TrainingConfig(
            ppo_epochs=args.ppo_epochs,
            chunk_batch_size=args.chunk_batch_size,
            retain_checkpoints=args.retain_checkpoints,
        )
        trainer = Stage2PPOTrainer(
            base_inference_checkpoint=args.base_checkpoint,
            continuation_checkpoint=args.continuation_checkpoint,
            expert_manifest=args.expert_manifest,
            device=args.device,
            config=config,
        )
        if any(
            batch_manifest.behavior_actor_sha256 != trainer.behavior_actor_sha256
            for batch_manifest in batch_manifests
        ):
            raise RuntimeError("rollout/continuation behavior Actor hashes differ")
        if any(
            batch_manifest.policy_version != trainer.policy_version
            for batch_manifest in batch_manifests
        ):
            raise RuntimeError("rollout/continuation policy versions differ")

        manifest = {
            "kind": "cr_native_expert_selfplay_stage2_update_manifest_v1",
            "run_id": run_dir.name,
            "stage": "stage2_reaction",
            "base_checkpoint": str(args.base_checkpoint.resolve(strict=True)),
            "base_checkpoint_sha256": _file_sha256(args.base_checkpoint.resolve(strict=True)),
            "continuation_checkpoint": str(args.continuation_checkpoint.resolve(strict=True)),
            "continuation_checkpoint_sha256": _file_sha256(
                args.continuation_checkpoint.resolve(strict=True)
            ),
            "expert_manifest": str(args.expert_manifest.resolve(strict=True)),
            "expert_manifest_sha256": _file_sha256(args.expert_manifest.resolve(strict=True)),
            "collection_runs": [str(value) for value in collection_runs],
            "shards": [str(value) for value in shards],
            "batch_manifests": [asdict(value) for value in batch_manifests],
            "config": asdict(config),
            "pre_policy_version": trainer.policy_version,
            "pre_global_update": trainer.global_update,
            "pre_behavior_actor_sha256": trainer.behavior_actor_sha256,
            "pre_master_actor_sha256": trainer.master_actor_sha256,
        }
        atomic_json(run_dir / "manifest.json", manifest)
        _link_pre_update(
            args.continuation_checkpoint.resolve(strict=True),
            run_dir / "pre-update" / "checkpoint.pt",
        )
        journal.progress("preparing_rollout", policy_version=trainer.policy_version)
        chunks = []
        rollout_rows = []
        for shard, expected_manifest in zip(shards, batch_manifests, strict=True):
            shard_chunks, admitted_manifest, rollout = trainer.prepare_rollout(shard)
            if admitted_manifest.digest() != expected_manifest.digest():
                raise RuntimeError("prepared rollout manifest identity changed")
            chunks.extend(shard_chunks)
            rollout_rows.append({"shard": str(shard), **rollout})
        rollout = {
            "shards": rollout_rows,
            "episodes": sum(value["episodes"] for value in rollout_rows),
            "decisions": sum(value["decisions"] for value in rollout_rows),
            "chunks": len(chunks),
        }

        actor_before = {
            name: value.detach().cpu().clone()
            for name, value in trainer.model.actor.state_dict().items()
        }
        bc_before = actor_state_digest(trainer.bc_actor)
        for ledger, batch_id in ledgers:
            ledger.transition(batch_id, "UPDATING")
        journal.progress(
            "training", chunks=len(chunks), decisions=rollout["decisions"],
            policy_version=trainer.policy_version,
        )
        metrics, guard, retry_attempt = trainer.train_update(chunks)

        changed = [
            f"actor_adapter.actor.{name}"
            for name, value in trainer.model.actor.state_dict().items()
            if not torch.equal(value.detach().cpu(), actor_before[name])
        ]
        allowed = set(trainer.stage_report["trainable_names"])
        unexpected = sorted(set(changed).difference(allowed))
        if unexpected:
            raise RuntimeError(f"Stage-2 changed frozen Actor parameters: {unexpected}")
        if not changed:
            raise RuntimeError("Stage-2 accepted an update that changed no Actor parameter")
        if actor_state_digest(trainer.bc_actor) != bc_before:
            raise RuntimeError("Stage-2 changed the frozen BC Actor")

        for ledger, batch_id in ledgers:
            ledger.transition(batch_id, "VALIDATING")
        checkpoint, export = trainer.save(
            run_dir,
            metrics=metrics,
            guard=guard,
            retry_attempt=retry_attempt,
            rollout=rollout,
        )
        export_value = torch.load(export, map_location="cpu", weights_only=False, mmap=True)
        if _state_digest(export_value["model_state"]) != trainer.behavior_actor_sha256:
            raise RuntimeError("published FP16 Actor export hash differs from checkpoint")
        if not all(
            math.isfinite(float(value))
            for value in metrics.values()
            if isinstance(value, (int, float))
        ):
            raise FloatingPointError("Stage-2 result metrics contain NaN/Inf")
        for ledger, batch_id in ledgers:
            ledger.commit(batch_id)
        result = {
            "kind": RESULT_KIND,
            "status": "completed",
            "run_id": run_dir.name,
            "policy_version": trainer.policy_version,
            "global_update": trainer.global_update,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _file_sha256(checkpoint),
            "behavior_export": str(export),
            "behavior_export_sha256": _file_sha256(export),
            "behavior_actor_sha256": trainer.behavior_actor_sha256,
            "master_actor_sha256": trainer.master_actor_sha256,
            "changed_actor_parameters": changed,
            "retry_attempt": retry_attempt,
            "guard": asdict(guard),
            "metrics": metrics,
            "rollout": rollout,
            "ledger_states": [ledger.state(batch_id) for ledger, batch_id in ledgers],
        }
        atomic_json(run_dir / "result.json", result)
        journal.progress(
            "completed",
            policy_version=trainer.policy_version,
            global_update=trainer.global_update,
            checkpoint=str(checkpoint),
            behavior_export=str(export),
            ledger_states=result["ledger_states"],
        )
        journal.event("stage2_update_completed", result=result)
        return result
    except BaseException as error:
        journal.progress("failed", error_type=type(error).__name__, error=str(error))
        journal.event(
            "stage2_update_failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback="".join(traceback.format_exception(error)),
        )
        raise
    finally:
        for ledger, _batch_id in ledgers:
            ledger.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--continuation-checkpoint", type=Path, required=True)
    parser.add_argument("--expert-manifest", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=12)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--chunk-batch-size", type=int, default=2)
    parser.add_argument("--retain-checkpoints", type=int, default=3)
    args = parser.parse_args()
    if min(
        args.cpu_threads, args.ppo_epochs,
        args.chunk_batch_size, args.retain_checkpoints,
    ) < 1:
        raise ValueError("Stage-2 runtime values must be positive")
    print(json.dumps(run(args), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
