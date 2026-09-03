"""Train one Stage-1 Critic update from parallel collect-only runs.

Every input run must contain a verified immutable rollout shard whose ledger is
still CLOSED.  The update restores one explicit predecessor checkpoint, trains
over the union of all recurrent chunks, then commits every child ledger only
after the new checkpoint and finite metrics have been published.

The script is restart-safe for a failed update: it always restores the recorded
pre-update checkpoint, accepts child ledgers left in UPDATING/VALIDATING, and
therefore cannot advance the optimizer twice in the checkpoint chain.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import sys
import traceback
from typing import Any, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from expert_selfplay_v1.actor_adapter import actor_state_digest
from expert_selfplay_v1.contracts import BatchManifest
from expert_selfplay_v1.critic_training import CriticTrainingConfig, Stage1CriticTrainer
from expert_selfplay_v1.ledger import RolloutLedger
from expert_selfplay_v1.rollout_storage import verify_rollout_shard
from scripts.run_expert_selfplay_v1 import RunJournal, atomic_json, sha256_file


RESULT_KIND = "cr_native_expert_selfplay_stage1_parallel_update_v1"


def _object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read JSON object: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _finite_metrics(metrics: dict[str, Any]) -> bool:
    return all(
        math.isfinite(float(value))
        for value in metrics.values()
        if isinstance(value, (int, float))
    )


def _load_collection(
    run_dir: Path,
    *,
    base_checkpoint_sha256: str,
    expert_manifest_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], RolloutLedger]:
    run_dir = run_dir.resolve(strict=True)
    collection = _object(run_dir / "collection-result.json")
    if collection.get("kind") != "cr_native_expert_selfplay_stage1_collection_v1":
        raise RuntimeError(f"not a Stage-1 collection run: {run_dir}")
    if collection.get("status") != "collected":
        raise RuntimeError(f"collection run is not complete: {run_dir}")
    manifest = _object(run_dir / "manifest.json")
    if manifest.get("collection_only") is not True:
        raise RuntimeError(f"collection-only marker missing: {run_dir}")
    if manifest.get("checkpoint", {}).get("file_sha256") != base_checkpoint_sha256:
        raise RuntimeError(f"BASE checkpoint differs across collection run: {run_dir}")
    if manifest.get("expert_manifest", {}).get("sha256") != expert_manifest_sha256:
        raise RuntimeError(f"expert manifest differs across collection run: {run_dir}")

    batch_manifest = BatchManifest(**dict(manifest["batch_manifest"]))
    batch_manifest.validate()
    if batch_manifest.run_id != run_dir.name:
        raise RuntimeError(f"BatchManifest run_id differs from directory: {run_dir}")
    shard = Path(str(collection.get("shard", ""))).resolve(strict=True)
    if not _inside(shard, run_dir / "rollouts"):
        raise RuntimeError(f"collection shard escapes its run directory: {shard}")
    verified = verify_rollout_shard(
        shard,
        expected_batch_manifest=batch_manifest,
        return_payload=True,
        mmap=True,
        # The collect-only writer already performed the full semantic digest
        # pass immediately before closing this local shard.  Ingestion repeats
        # the authenticated file SHA-256 and full structural/terminal checks,
        # but avoids hashing every tensor a second time on the same host.
        verify_semantic_digest=False,
    )
    if verified["content_sha256"] != collection.get("shard_content_sha256"):
        raise RuntimeError(f"collection shard digest differs from result: {run_dir}")
    payload = verified.pop("_payload")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != batch_manifest.episode_count:
        raise RuntimeError(f"collection episode count differs from manifest: {run_dir}")
    chunks = [
        chunk
        for episode in episodes
        for chunk in episode.get("chunks", [])
    ]
    if not chunks:
        raise RuntimeError(f"collection contains no recurrent chunks: {run_dir}")
    if sum(int(episode.get("decision_count", 0)) for episode in episodes) != int(
        collection.get("decisions", -1)
    ):
        raise RuntimeError(f"collection decision count differs from payload: {run_dir}")

    ledger = RolloutLedger(run_dir / "rollout-ledger.sqlite")
    state = ledger.state(batch_manifest.batch_id)
    if state not in {"CLOSED", "UPDATING", "VALIDATING", "COMMITTED"}:
        ledger.close()
        raise RuntimeError(f"collection ledger cannot be trained from state {state}: {run_dir}")
    return {
        "run_dir": str(run_dir),
        "batch_id": batch_manifest.batch_id,
        "actor_sha256": batch_manifest.behavior_actor_sha256,
        "shard": str(shard),
        "shard_content_sha256": verified["content_sha256"],
        "episodes": len(episodes),
        "decisions": int(collection["decisions"]),
        "chunks": len(chunks),
    }, chunks, ledger


def train(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    journal = RunJournal(run_dir)
    result_path = run_dir / "result.json"
    if result_path.is_file():
        result = _object(result_path)
        if result.get("kind") == RESULT_KIND and result.get("status") == "completed":
            return result
        raise RuntimeError(f"incompatible result already exists: {result_path}")

    base_checkpoint = args.checkpoint.resolve(strict=True)
    expert_manifest = args.expert_manifest.resolve(strict=True)
    resume_checkpoint = args.resume_checkpoint.resolve(strict=True)
    base_sha256 = sha256_file(base_checkpoint)
    expert_sha256 = sha256_file(expert_manifest)
    resume_sha256 = sha256_file(resume_checkpoint)
    collection_dirs = [path.resolve(strict=True) for path in args.collection_run]
    if not collection_dirs or len(set(collection_dirs)) != len(collection_dirs):
        raise ValueError("update requires one or more distinct collection runs")

    attempt = {
        "kind": "cr_native_expert_selfplay_stage1_parallel_attempt_v1",
        "checkpoint": str(base_checkpoint),
        "checkpoint_sha256": base_sha256,
        "expert_manifest": str(expert_manifest),
        "expert_manifest_sha256": expert_sha256,
        "resume_checkpoint": str(resume_checkpoint),
        "resume_checkpoint_sha256": resume_sha256,
        "collection_runs": [str(path) for path in collection_dirs],
    }
    attempt_path = run_dir / "update-attempt.json"
    if attempt_path.is_file():
        if _object(attempt_path) != attempt:
            raise RuntimeError("parallel update invocation differs from recorded attempt")
    else:
        atomic_json(attempt_path, attempt)

    ledgers: list[tuple[RolloutLedger, str]] = []
    try:
        torch.set_num_threads(args.cpu_threads)
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")

        records: list[dict[str, Any]] = []
        chunks: list[dict[str, Any]] = []
        chunk_groups: list[list[dict[str, Any]]] = []
        actor_hashes: set[str] = set()
        journal.progress("loading_collections", collections=len(collection_dirs))
        for collection_dir in collection_dirs:
            record, collection_chunks, ledger = _load_collection(
                collection_dir,
                base_checkpoint_sha256=base_sha256,
                expert_manifest_sha256=expert_sha256,
            )
            records.append(record)
            chunks.extend(collection_chunks)
            chunk_groups.append(collection_chunks)
            actor_hashes.add(record["actor_sha256"])
            ledgers.append((ledger, record["batch_id"]))
        if len(actor_hashes) != 1:
            raise RuntimeError("parallel collections used different behavior Actors")

        trainer = Stage1CriticTrainer.from_expert_checkpoint(
            base_checkpoint,
            device=device,
            actor_inference_dtype=(torch.float16 if device.type == "cuda" else None),
            config=CriticTrainingConfig(retain_checkpoints=args.retain_checkpoints),
            run_config={"parallel_collection_runs": [str(path) for path in collection_dirs]},
        )
        expected_actor = next(iter(actor_hashes))
        if trainer.actor_sha256 != expected_actor:
            raise RuntimeError("BASE Actor hash differs from collected behavior Actor")
        previous_metrics = trainer.restore_checkpoint(resume_checkpoint)
        previous_update = trainer.global_update

        for ledger, batch_id in ledgers:
            state = ledger.state(batch_id)
            if state == "CLOSED":
                ledger.transition(batch_id, "UPDATING")
            elif state == "COMMITTED":
                # A prior attempt may have committed ledgers just before the
                # parent result write.  Replaying from the recorded predecessor
                # remains safe and repairs that narrow crash window.
                continue
            elif state not in {"UPDATING", "VALIDATING"}:
                raise RuntimeError(f"unexpected collection ledger state: {state}")

        validation_stride = max(1, len(chunks) // args.validation_chunks)
        validation_chunks = chunks[::validation_stride][:args.validation_chunks]
        journal.progress(
            "validating_fresh",
            collections=len(records),
            validation_chunks=len(validation_chunks),
            resume_global_update=previous_update,
        )
        validation_metrics = dict(trainer.evaluate_chunks(validation_chunks))
        if not _finite_metrics(validation_metrics):
            raise FloatingPointError("fresh validation metrics contain NaN/Inf")

        journal.progress(
            "training",
            collections=len(records),
            episodes=sum(row["episodes"] for row in records),
            decisions=sum(row["decisions"] for row in records),
            chunks=len(chunks),
            resume_global_update=previous_update,
        )
        before = actor_state_digest(trainer.model.actor)
        update_metrics: list[dict[str, Any]] = []
        for update_index, collection_chunks in enumerate(chunk_groups, start=1):
            metrics = dict(trainer.train_update(collection_chunks))
            update_metrics.append(metrics)
            journal.progress(
                "training",
                collections=len(records),
                collection_updates_completed=update_index,
                collection_updates=len(chunk_groups),
                episodes=sum(row["episodes"] for row in records),
                decisions=sum(row["decisions"] for row in records),
                chunks=len(chunks),
                current_global_update=trainer.global_update,
            )
        after = actor_state_digest(trainer.model.actor)
        if before != expected_actor or after != before:
            raise RuntimeError("parallel Stage-1 update changed the frozen BASE Actor")
        if trainer.global_update != previous_update + len(chunk_groups):
            raise RuntimeError("parallel Stage-1 update number is not contiguous")
        if not all(_finite_metrics(metrics) for metrics in update_metrics):
            raise FloatingPointError("parallel Stage-1 metrics contain NaN/Inf")
        metrics = update_metrics[-1]
        checkpoint = trainer.save_checkpoint(run_dir / "checkpoints", metrics)
        if not checkpoint.is_file():
            raise RuntimeError("parallel Stage-1 trainer did not publish a checkpoint")

        for ledger, batch_id in ledgers:
            state = ledger.state(batch_id)
            if state == "UPDATING":
                ledger.transition(batch_id, "VALIDATING")
                state = "VALIDATING"
            if state == "VALIDATING":
                ledger.commit(batch_id)
                state = "COMMITTED"
            if state != "COMMITTED":
                raise RuntimeError(f"collection ledger did not commit: {state}")

        result = {
            "kind": RESULT_KIND,
            "status": "completed",
            "run_id": run_dir.name,
            "collections": records,
            "episodes": sum(row["episodes"] for row in records),
            "decisions": sum(row["decisions"] for row in records),
            "chunks": len(chunks),
            "global_update": trainer.global_update,
            "updates_committed": len(update_metrics),
            "actor_sha256_before": before,
            "actor_sha256_after": after,
            "actor_unchanged": True,
            "resume_checkpoint": str(resume_checkpoint),
            "resume_checkpoint_sha256": resume_sha256,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "previous_metrics": previous_metrics,
            "fresh_validation_before_update": validation_metrics,
            "metrics": metrics,
            "update_metrics": update_metrics,
            "ledger_states": ["COMMITTED"] * len(ledgers),
        }
        atomic_json(result_path, result)
        journal.progress(
            "completed",
            global_update=trainer.global_update,
            checkpoint=str(checkpoint),
            episodes=result["episodes"],
            decisions=result["decisions"],
            chunks=result["chunks"],
        )
        journal.event("parallel_update_completed", result=result)
        return result
    except BaseException as error:
        journal.progress("failed", error_type=type(error).__name__, error=str(error))
        journal.event(
            "parallel_update_failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback="".join(traceback.format_exception(error)),
        )
        raise
    finally:
        for ledger, _ in ledgers:
            ledger.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--collection-run", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expert-manifest", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--retain-checkpoints", type=int, default=3)
    parser.add_argument("--validation-chunks", type=int, default=128)
    args = parser.parse_args(argv)
    if min(args.cpu_threads, args.retain_checkpoints, args.validation_chunks) < 1:
        raise ValueError("cpu thread and checkpoint retention values must be positive")
    result = train(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
