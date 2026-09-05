"""One-click cloud entry for Stage-2 Canary or guarded formal training."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ensure_bionic_workers import ensure
from scripts.run_expert_selfplay_stage2_loop import run, run_isolated


DEFAULT_BASE = PROJECT_ROOT / "models/expert-v1.1/candidate-lr5e-5-step157674-fp16.pt"
DEFAULT_MANIFEST = PROJECT_ROOT / "models/expert-v1.1/manifest.json"
DEFAULT_STAGE1 = (
    PROJECT_ROOT
    / "formal-runs/stage1-hog26-stream-formal-v1/updates/update-00000025/"
      "checkpoints/checkpoint-000000000052.pt"
)
DEFAULT_DECK = PROJECT_ROOT / "examples/hog-2.6-evo-hero.json"
DEFAULT_OPPONENT_DECKS = PROJECT_ROOT / "top-deck-presets-v1"


def _completed_run_continuation(root: Path) -> tuple[Path, Path]:
    progress_path = root.resolve(strict=True) / "progress.json"
    value = json.loads(progress_path.read_text(encoding="utf-8-sig"))
    completed = value.get("completed_updates")
    if not (
        value.get("kind") == "cr_native_expert_selfplay_stage2_loop_v1"
        and value.get("status") == "completed"
        and value.get("completion_reason") == "requested_updates_committed"
        and isinstance(completed, list)
        and completed
    ):
        raise RuntimeError("resume mode requires a completed guarded Stage-2 run")
    last = completed[-1]
    checkpoint = Path(value["latest_checkpoint"]).resolve(strict=True)
    behavior = Path(value["latest_behavior_export"]).resolve(strict=True)
    if str(checkpoint) != str(Path(last["checkpoint"]).resolve(strict=True)) or (
        str(behavior) != str(Path(last["behavior_export"]).resolve(strict=True))
    ):
        raise RuntimeError("resume run latest artifacts do not match its last commit")
    return checkpoint, behavior


def _canary_continuation(root: Path) -> tuple[Path, Path]:
    progress_path = root.resolve(strict=True) / "progress.json"
    value = json.loads(progress_path.read_text(encoding="utf-8-sig"))
    if value.get("status") != "completed" or value.get("completion_reason") != (
        "requested_updates_committed"
    ):
        raise RuntimeError("formal mode requires a completed Stage-2 Canary")
    completed = value.get("completed_updates")
    if not isinstance(completed, list) or len(completed) < 3:
        raise RuntimeError("formal mode requires at least three guarded Canary updates")
    if any(row.get("metrics") is None for row in completed[-3:]):
        raise RuntimeError("Canary update metrics are incomplete")
    checkpoint = Path(value["latest_checkpoint"]).resolve(strict=True)
    behavior = Path(value["latest_behavior_export"]).resolve(strict=True)
    attestation_path = root / "local-rtx3080-attestation.json"
    if not attestation_path.is_file():
        raise RuntimeError("formal mode requires local RTX 3080 attestation")
    attestation = json.loads(attestation_path.read_text(encoding="utf-8-sig"))
    digest = hashlib.sha256(behavior.read_bytes()).hexdigest()
    if not (
        attestation.get("status") == "passed"
        and "RTX 3080" in str(attestation.get("device", ""))
        and attestation.get("weights_sha256") == digest
        and attestation.get("finite") is True
    ):
        raise RuntimeError("local RTX 3080 attestation does not match Canary export")
    return checkpoint, behavior


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("canary", "formal"), default="canary")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--canary-root", type=Path)
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--base-port", type=int, default=19031)
    parser.add_argument("--prepared-cache-gib", type=float)
    parser.add_argument("--training-precision", choices=("float32", "bfloat16", "float16"))
    parser.add_argument("--chunk-batch-size", type=int)
    parser.add_argument("--chunk-padding-multiple", type=int)
    parser.add_argument("--fused-optimizer", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--persistent-learner", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--overlap-preparation", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--isolate-updates", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--profile",
        choices=("throughput", "conservative"),
        default="throughput",
        help=(
            "96-worker JIT/MPS, BF16 padded-batch profile or the original "
            "48-worker profile"
        ),
    )
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--expert-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--stage1-checkpoint", type=Path, default=DEFAULT_STAGE1)
    parser.add_argument("--learner-deck", type=Path, default=DEFAULT_DECK)
    parser.add_argument("--opponent-deck-root", type=Path, default=DEFAULT_OPPONENT_DECKS)
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=PROJECT_ROOT / "bionic-runtime",
    )
    args = parser.parse_args()

    if args.mode == "formal":
        if args.resume_run is not None and args.canary_root is not None:
            raise ValueError("use either --resume-run or --canary-root, not both")
        if args.resume_run is not None:
            continuation, behavior = _completed_run_continuation(args.resume_run)
        else:
            if args.canary_root is None:
                raise ValueError(
                    "--canary-root or --resume-run is required in formal mode"
                )
            continuation, behavior = _canary_continuation(args.canary_root)
        updates = 100 if args.updates is None else args.updates
    else:
        continuation = args.stage1_checkpoint.resolve(strict=True)
        behavior = args.base_checkpoint.resolve(strict=True)
        updates = 3 if args.updates is None else args.updates
    if updates < 1:
        raise ValueError("--updates must be positive")
    if args.run_root is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_root = PROJECT_ROOT / "formal-runs" / f"stage2-{args.mode}-{stamp}"
    else:
        run_root = args.run_root

    if args.profile == "throughput":
        worker_count = 96
        collector_count = 6
        step_ticks = 12
        collection_waves = 2
        worker_execution_mode = "jit"
        enable_mps = True
        defaults = {
            "prepared_cache_gib": 8.0, "training_precision": "bfloat16",
            "chunk_batch_size": 32, "chunk_padding_multiple": 80,
            "fused_optimizer": True, "persistent_learner": True,
            "overlap_preparation": True, "isolate_updates": True,
        }
        preprocess_window_size, preprocess_batch_size = 128, 2
    else:
        worker_count = 48
        collector_count = 6
        step_ticks = 4
        collection_waves = 1
        worker_execution_mode = "interpreter"
        enable_mps = False
        defaults = {
            "prepared_cache_gib": 4.0, "training_precision": "float32",
            "chunk_batch_size": 8, "chunk_padding_multiple": 0,
            "fused_optimizer": False, "persistent_learner": False,
            "overlap_preparation": False, "isolate_updates": False,
        }
        preprocess_window_size, preprocess_batch_size = 256, 3
    for name, default in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, default)
    if not args.persistent_learner:
        args.overlap_preparation = False

    worker_result = ensure(argparse.Namespace(
        runtime_root=args.runtime_root,
        base_port=args.base_port,
        count=worker_count,
        slot_offset=0,
        execution_mode=worker_execution_mode,
        ready_timeout=120.0,
    ))
    loop_args = argparse.Namespace(
        base_checkpoint=args.base_checkpoint,
        base_opponent_checkpoint=args.base_checkpoint,
        initial_continuation=continuation,
        initial_behavior_export=behavior,
        expert_manifest=args.expert_manifest,
        ports=f"{args.base_port}-{args.base_port + worker_count - 1}",
        collectors=collector_count,
        updates=updates,
        run_root=run_root,
        learner_deck=args.learner_deck,
        opponent_deck_root=args.opponent_deck_root,
        host="127.0.0.1",
        step_ticks=step_ticks,
        idle_step_ticks=None,
        collection_waves=collection_waves,
        async_shard_writes=False,
        rolling_collection=False,
        compile_actor=False,
        compile_batch_size=None,
        compile_entity_slots=None,
        dense_policy_sampling=False,
        enable_mps=enable_mps,
        mps_root=None,
        max_decisions=3000,
        timeout=30.0,
        seed=20600000 if args.mode == "canary" else 20700000,
        device="cuda",
        collector_cpu_threads=2,
        trainer_cpu_threads=8,
        ppo_epochs=2,
        chunk_batch_size=args.chunk_batch_size,
        chunk_padding_multiple=args.chunk_padding_multiple,
        preprocess_window_size=preprocess_window_size,
        preprocess_batch_size=preprocess_batch_size,
        prepared_cache_gib=args.prepared_cache_gib,
        training_precision=args.training_precision,
        fused_optimizer=args.fused_optimizer,
        persistent_learner=args.persistent_learner,
        overlap_preparation=args.overlap_preparation,
        isolate_updates=args.isolate_updates,
        retain_checkpoints=3,
        retain_rollout_updates=2,
        retain_artifact_updates=3,
        minimum_free_gb=25.0,
        python=sys.executable,
        collect_script=PROJECT_ROOT / "scripts/run_expert_selfplay_v1.py",
        train_script=PROJECT_ROOT / "scripts/train_expert_selfplay_stage2.py",
        ensure_workers_script=PROJECT_ROOT / "scripts/ensure_bionic_workers.py",
        worker_runtime_root=args.runtime_root,
        worker_ready_timeout=120.0,
        worker_execution_mode=worker_execution_mode,
    )
    result = (run_isolated if args.isolate_updates else run)(loop_args)
    print(json.dumps({
        "mode": args.mode,
        "profile": args.profile,
        "workers": worker_result,
        "run_root": str(run_root.resolve()),
        "result": result,
    }, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
