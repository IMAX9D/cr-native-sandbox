"""Recover a Stage-1 run whose immutable rollout committed before its update.

This command never recollects data.  It accepts only a ledger batch left in
``UPDATING``, verifies the existing shard and the recorded continuation
checkpoint, performs one Critic update, and commits the original batch.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import traceback

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from expert_selfplay_v1.actor_adapter import actor_state_digest
from expert_selfplay_v1.contracts import BatchManifest
from expert_selfplay_v1.critic_training import (
    CriticTrainingConfig,
    Stage1CriticTrainer,
)
from expert_selfplay_v1.ledger import RolloutLedger
from expert_selfplay_v1.rollout_storage import verify_rollout_shard
from scripts.run_expert_selfplay_v1 import RunJournal, atomic_json, sha256_file


def _read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def recover(args: argparse.Namespace) -> dict:
    run_dir = args.run_dir.resolve(strict=True)
    journal = RunJournal(run_dir)
    try:
        existing_result = run_dir / "result.json"
        if existing_result.exists():
            prior = _read_object(existing_result)
            if prior.get("status") == "completed":
                raise RuntimeError("run is already completed")

        manifest = _read_object(run_dir / "manifest.json")
        batch_manifest = BatchManifest(**dict(manifest["batch_manifest"]))
        batch_manifest.validate()
        if batch_manifest.run_id != run_dir.name:
            raise RuntimeError("run directory and BatchManifest run_id differ")

        base_checkpoint = args.checkpoint.resolve(strict=True)
        recorded_base = manifest.get("checkpoint", {})
        if sha256_file(base_checkpoint) != recorded_base.get("file_sha256"):
            raise RuntimeError("BASE Actor checkpoint differs from failed run")
        resume_checkpoint = args.resume_checkpoint.resolve(strict=True)
        recorded_resume = manifest.get("resume_checkpoint")
        if not isinstance(recorded_resume, dict):
            raise RuntimeError("failed run was not a continuation batch")
        if sha256_file(resume_checkpoint) != recorded_resume.get("sha256"):
            raise RuntimeError("resume checkpoint differs from failed run")

        shards = sorted((run_dir / "rollouts").glob("shard-*"))
        if len(shards) != 1:
            raise RuntimeError(f"recovery requires exactly one shard, found {len(shards)}")
        verified = verify_rollout_shard(
            shards[0], expected_batch_manifest=batch_manifest
        )
        payload = torch.load(
            shards[0] / "rollout.pt", map_location="cpu", weights_only=False
        )
        episodes = payload.get("episodes")
        if not isinstance(episodes, list) or len(episodes) != batch_manifest.episode_count:
            raise RuntimeError("rollout episode count differs from BatchManifest")
        chunks = [
            chunk
            for episode in episodes
            for chunk in episode.get("chunks", [])
        ]
        if not chunks:
            raise RuntimeError("verified rollout contains no recurrent chunks")

        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        trainer = Stage1CriticTrainer.from_expert_checkpoint(
            base_checkpoint,
            device=device,
            actor_inference_dtype=(torch.float16 if device.type == "cuda" else None),
            config=CriticTrainingConfig(
                retain_checkpoints=args.retain_checkpoints
            ),
            run_config={"recovered_run": run_dir.name},
        )
        expected_actor = batch_manifest.behavior_actor_sha256
        if trainer.actor_sha256 != expected_actor:
            raise RuntimeError("BASE Actor state hash differs from rollout behavior policy")
        previous_metrics = trainer.restore_checkpoint(resume_checkpoint)
        journal.event(
            "recovery_update_started",
            shard=str(shards[0]),
            chunks=len(chunks),
            resume_checkpoint=str(resume_checkpoint),
            resume_global_update=trainer.global_update,
        )
        journal.progress(
            "recovery_training",
            episodes=len(episodes),
            chunks=len(chunks),
            resume_global_update=trainer.global_update,
        )
        before = actor_state_digest(trainer.model.actor)
        metrics = dict(trainer.train_update(chunks))
        after = actor_state_digest(trainer.model.actor)
        if before != expected_actor or after != before:
            raise RuntimeError("recovery update changed the frozen BASE Actor")
        if not all(
            math.isfinite(float(value))
            for value in metrics.values()
            if isinstance(value, (int, float))
        ):
            raise FloatingPointError("recovery metrics contain NaN/Inf")
        checkpoint = trainer.save_checkpoint(run_dir / "checkpoints", metrics)
        if not checkpoint.is_file():
            raise RuntimeError("recovery did not publish a checkpoint")

        ledger = RolloutLedger(run_dir / "rollout-ledger.sqlite")
        try:
            if ledger.state(batch_manifest.batch_id) != "UPDATING":
                raise RuntimeError("recovery ledger is not left in UPDATING")
            ledger.transition(batch_manifest.batch_id, "VALIDATING")
            ledger.commit(batch_manifest.batch_id)
            ledger_state = ledger.state(batch_manifest.batch_id)
        finally:
            ledger.close()

        result = {
            "kind": "cr_native_expert_selfplay_stage1_result_v1",
            "status": "completed",
            "recovered_from_failure": True,
            "run_id": run_dir.name,
            "episodes": len(episodes),
            "decisions": sum(int(row["decision_count"]) for row in episodes),
            "chunks": len(chunks),
            "updates": 1,
            "global_update": trainer.global_update,
            "actor_sha256_before": before,
            "actor_sha256_after": after,
            "actor_unchanged": True,
            "shard": str(shards[0]),
            "shard_content_sha256": verified["content_sha256"],
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "previous_metrics": previous_metrics,
            "metrics": [metrics],
            "ledger_state": ledger_state,
        }
        atomic_json(run_dir / "result.json", result)
        journal.progress(
            "completed",
            recovered_from_failure=True,
            global_update=trainer.global_update,
            checkpoint=str(checkpoint),
            ledger_state=ledger_state,
        )
        journal.event("run_recovered", result=result)
        return result
    except BaseException as error:
        journal.progress(
            "recovery_failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        journal.event(
            "recovery_failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback="".join(traceback.format_exception(error)),
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--retain-checkpoints", type=int, default=3)
    args = parser.parse_args()
    if args.retain_checkpoints < 1:
        raise ValueError("--retain-checkpoints must be positive")
    print(json.dumps(recover(args), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
