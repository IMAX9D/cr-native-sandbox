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
import time
import traceback
from typing import Any, Callable, Mapping

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from expert_selfplay_v1.actor_adapter import actor_state_digest
from expert_selfplay_v1.critic_training import _capture_rng_state, _restore_rng_state, _clone_state
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


def _admit_collection_batches(
    shards: list[Path],
) -> tuple[list[BatchManifest], list[tuple[RolloutLedger, str]]]:
    """Admit every shard of a CLOSED batch, opening each ledger only once."""
    grouped: dict[Path, list[Path]] = {}
    for shard in shards:
        grouped.setdefault(_collection_run(shard), []).append(shard)
    manifests: dict[Path, BatchManifest] = {}
    ledgers: list[tuple[RolloutLedger, str]] = []
    try:
        for collection_run, selected in grouped.items():
            manifest = BatchManifest(**dict(
                _read(collection_run / "manifest.json")["batch_manifest"]
            ))
            manifest.validate()
            ledger = RolloutLedger(collection_run / "rollout-ledger.sqlite")
            ledgers.append((ledger, manifest.batch_id))
            if ledger.state(manifest.batch_id) != "CLOSED":
                raise RuntimeError("Stage-2 accepts only CLOSED, unconsumed rollout batches")
            registered = {
                uuid: (digest, consumed)
                for uuid, digest, consumed in ledger.shards(manifest.batch_id)
            }
            if set(registered) != {shard.name for shard in selected}:
                raise RuntimeError("Stage-2 requires all registered shards of each batch")
            episode_ids: set[str] = set()
            for shard in selected:
                metadata = _read(shard / "manifest.json")
                if (
                    metadata.get("shard_uuid") != shard.name
                    or metadata.get("batch_manifest_sha256") != manifest.digest()
                    or registered[shard.name] != (metadata.get("content_sha256"), False)
                ):
                    raise RuntimeError("shard metadata does not match its CLOSED batch ledger")
                ids = metadata.get("episode_ids")
                if (
                    not isinstance(ids, list) or not ids
                    or any(not isinstance(value, str) or not value for value in ids)
                    or len(set(ids)) != len(ids)
                    or metadata.get("episode_count") != len(ids)
                    or episode_ids.intersection(ids)
                ):
                    raise RuntimeError("batch shards contain duplicate or invalid episode coverage")
                episode_ids.update(ids)
            if len(episode_ids) != manifest.episode_count:
                raise RuntimeError("batch shards do not cover the scheduled episode count")
            manifests[collection_run] = manifest
        return [manifests[_collection_run(shard)] for shard in shards], ledgers
    except BaseException:
        for ledger, _batch_id in ledgers:
            ledger.close()
        raise


class PersistentStage2Learner:
    """Keep the accepted Actor, BC teacher and optimizers resident between batches."""

    def __init__(self) -> None:
        self.trainer: Stage2PPOTrainer | None = None
        self._rng: dict[str, Any] | None = None
        self._prepared: dict[Path, tuple[Any, Any, Any]] = {}

    def initialize(self, args: argparse.Namespace) -> None:
        self._provide(
            base_inference_checkpoint=args.base_checkpoint,
            continuation_checkpoint=args.continuation_checkpoint,
            expert_manifest=args.expert_manifest, device=args.device,
            config=_training_config(args),
        )

    def prepare(self, shard: Path) -> None:
        if self.trainer is None:
            raise RuntimeError("initialize the resident learner before preparing rollouts")
        path = shard.resolve(strict=True)
        if path not in self._prepared:
            self._prepared[path] = self.trainer.prepare_rollout(path)

    def _take_prepared(self, shard: Path):
        prepared = self._prepared.pop(shard.resolve(strict=True), None)
        if prepared is not None:
            return prepared
        if self.trainer is None:
            raise RuntimeError("resident learner is not initialized")
        return self.trainer.prepare_rollout(shard)

    def _provide(self, **values: Any) -> Stage2PPOTrainer:
        if self.trainer is None:
            self.trainer = Stage2PPOTrainer(**values)
        else:
            trainer = self.trainer
            if (
                trainer.base_path != Path(values["base_inference_checkpoint"]).resolve()
                or trainer.continuation_path != Path(values["continuation_checkpoint"]).resolve()
                or trainer.manifest_path != Path(values["expert_manifest"]).resolve()
                or trainer.device != torch.device(values["device"])
                or trainer.config != values["config"]
            ):
                raise RuntimeError("resident learner cannot switch training lineage or configuration")
            if self._rng is not None:
                _restore_rng_state(self._rng)
        return self.trainer

    def run(self, args: argparse.Namespace) -> dict[str, Any]:
        result = run(args, trainer_factory=self._provide, prepare_rollout=self._take_prepared)
        self._rng = _capture_rng_state()
        self.trainer.actor_optimizer.zero_grad(set_to_none=True)
        self.trainer.critic_optimizer.zero_grad(set_to_none=True)
        if self.trainer.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return result

    def close(self) -> None:
        device = None if self.trainer is None else self.trainer.device
        self.trainer = None
        self._rng = None
        self._prepared.clear()
        if device is not None and device.type == "cuda":
            torch.cuda.empty_cache()


def _training_config(args: argparse.Namespace) -> Stage2TrainingConfig:
    return Stage2TrainingConfig(
        ppo_epochs=args.ppo_epochs, chunk_batch_size=args.chunk_batch_size,
        preprocess_window_size=args.preprocess_window_size,
        preprocess_batch_size=args.preprocess_batch_size,
        prepared_cache_gib=float(getattr(args, "prepared_cache_gib", 4.0)),
        training_precision=str(getattr(args, "training_precision", "float32")),
        fused_optimizer=bool(getattr(args, "fused_optimizer", False)),
        chunk_padding_multiple=int(getattr(args, "chunk_padding_multiple", 0)),
        retain_checkpoints=args.retain_checkpoints,
    )


def run(
    args: argparse.Namespace,
    *,
    trainer_factory: Callable[..., Stage2PPOTrainer] = Stage2PPOTrainer,
    prepare_rollout: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    total_started_at = time.monotonic()
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
        collection_runs = list(dict.fromkeys(_collection_run(shard) for shard in shards))
        batch_manifests, ledgers = _admit_collection_batches(shards)

        initialize_started_at = time.monotonic()
        config = _training_config(args)
        trainer = trainer_factory(
            base_inference_checkpoint=args.base_checkpoint,
            continuation_checkpoint=args.continuation_checkpoint,
            expert_manifest=args.expert_manifest,
            device=args.device,
            config=config,
        )
        initialized_at = time.monotonic()
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
        prepare_started_at = time.monotonic()
        for shard, expected_manifest in zip(shards, batch_manifests, strict=True):
            shard_chunks, admitted_manifest, rollout = (
                trainer.prepare_rollout if prepare_rollout is None else prepare_rollout
            )(shard)
            if admitted_manifest.digest() != expected_manifest.digest():
                raise RuntimeError("prepared rollout manifest identity changed")
            chunks.extend(shard_chunks)
            rollout_rows.append({"shard": str(shard), **rollout})
        prepared_at = time.monotonic()
        rollout = {
            "shards": rollout_rows,
            "episodes": sum(value["episodes"] for value in rollout_rows),
            "decisions": sum(value["decisions"] for value in rollout_rows),
            "chunks": len(chunks),
        }

        bc_before = trainer.bc_master_sha256
        for ledger, batch_id in ledgers:
            ledger.transition(batch_id, "UPDATING")
        journal.progress(
            "training", chunks=len(chunks), decisions=rollout["decisions"],
            policy_version=trainer.policy_version,
        )
        ppo_started_at = time.monotonic()
        metrics, guard, retry_attempt = trainer.train_update(chunks)
        ppo_finished_at = time.monotonic()

        actor_before = trainer.last_actor_before_update
        if actor_before is None:
            raise RuntimeError("guarded update did not preserve its pre-update Actor")
        actor_after = _clone_state(trainer.model.actor, fp32=True)
        changed = [
            f"actor_adapter.actor.{name}"
            for name, value in actor_after.items()
            if not torch.equal(value, actor_before[name])
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
        publish_started_at = time.monotonic()
        checkpoint, export = trainer.save(
            run_dir,
            metrics=metrics,
            guard=guard,
            retry_attempt=retry_attempt,
            rollout=rollout,
            actor_master_state=actor_after,
        )
        published_at = time.monotonic()
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
        hash_started_at = time.monotonic()
        checkpoint_sha256 = _file_sha256(checkpoint)
        behavior_export_sha256 = _file_sha256(export)
        hashed_at = time.monotonic()
        timings = {
            "initialize_seconds": initialized_at - initialize_started_at,
            "prepare_rollout_seconds": prepared_at - prepare_started_at,
            "ppo_and_guard_seconds": ppo_finished_at - ppo_started_at,
            "publish_seconds": published_at - publish_started_at,
            "artifact_hash_seconds": hashed_at - hash_started_at,
            "total_seconds": hashed_at - total_started_at,
        }
        result = {
            "kind": RESULT_KIND,
            "status": "completed",
            "run_id": run_dir.name,
            "policy_version": trainer.policy_version,
            "global_update": trainer.global_update,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "behavior_export": str(export),
            "behavior_export_sha256": behavior_export_sha256,
            "behavior_actor_sha256": trainer.behavior_actor_sha256,
            "master_actor_sha256": trainer.master_actor_sha256,
            "changed_actor_parameters": changed,
            "retry_attempt": retry_attempt,
            "guard": asdict(guard),
            "metrics": metrics,
            "rollout": rollout,
            "ledger_states": [ledger.state(batch_id) for ledger, batch_id in ledgers],
            "timings": timings,
        }
        atomic_json(run_dir / "result.json", result)
        journal.progress(
            "completed",
            policy_version=trainer.policy_version,
            global_update=trainer.global_update,
            checkpoint=str(checkpoint),
            behavior_export=str(export),
            ledger_states=result["ledger_states"],
            timings=timings,
        )
        journal.event("stage2_update_completed", result=result)
        trainer.continuation_path = checkpoint
        trainer.continuation_sha256 = checkpoint_sha256
        trainer.continuation_kind = "cr_native_expert_selfplay_stage2_checkpoint_v1"
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--continuation-checkpoint", type=Path, required=True)
    parser.add_argument("--expert-manifest", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=12)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--chunk-batch-size", type=int, default=8)
    parser.add_argument("--preprocess-window-size", type=int, default=256)
    parser.add_argument("--preprocess-batch-size", type=int, default=3)
    parser.add_argument("--prepared-cache-gib", type=float, default=4.0)
    parser.add_argument("--training-precision", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--fused-optimizer", action="store_true")
    parser.add_argument("--chunk-padding-multiple", type=int, default=0)
    parser.add_argument("--retain-checkpoints", type=int, default=3)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if min(
        args.cpu_threads, args.ppo_epochs,
        args.chunk_batch_size, args.preprocess_window_size,
        args.preprocess_batch_size, args.retain_checkpoints,
    ) < 1:
        raise ValueError("Stage-2 runtime values must be positive")
    print(json.dumps(run(args), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
