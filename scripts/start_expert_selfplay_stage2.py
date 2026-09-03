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
from scripts.run_expert_selfplay_stage2_loop import run


DEFAULT_BASE = PROJECT_ROOT / "models/expert-v1.1/candidate-lr5e-5-step157674-fp16.pt"
DEFAULT_MANIFEST = PROJECT_ROOT / "models/expert-v1.1/manifest.json"
DEFAULT_STAGE1 = (
    PROJECT_ROOT
    / "formal-runs/stage1-hog26-stream-formal-v1/updates/update-00000025/"
      "checkpoints/checkpoint-000000000052.pt"
)
DEFAULT_DECK = PROJECT_ROOT / "examples/hog-2.6-evo-hero.json"
DEFAULT_OPPONENT_DECKS = PROJECT_ROOT / "top-deck-presets-v1"


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
    parser.add_argument("--updates", type=int)
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
        if args.canary_root is None:
            raise ValueError("--canary-root is required in formal mode")
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

    worker_result = ensure(argparse.Namespace(
        runtime_root=args.runtime_root,
        base_port=39031,
        count=48,
        ready_timeout=120.0,
    ))
    loop_args = argparse.Namespace(
        base_checkpoint=args.base_checkpoint,
        base_opponent_checkpoint=args.base_checkpoint,
        initial_continuation=continuation,
        initial_behavior_export=behavior,
        expert_manifest=args.expert_manifest,
        ports="39031-39078",
        collectors=3,
        updates=updates,
        run_root=run_root,
        learner_deck=args.learner_deck,
        opponent_deck_root=args.opponent_deck_root,
        host="127.0.0.1",
        step_ticks=4,
        max_decisions=3000,
        timeout=30.0,
        seed=20600000 if args.mode == "canary" else 20700000,
        device="cuda",
        collector_cpu_threads=2,
        trainer_cpu_threads=12,
        ppo_epochs=2,
        chunk_batch_size=2,
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
    )
    result = run(loop_args)
    print(json.dumps({
        "mode": args.mode,
        "workers": worker_result,
        "run_root": str(run_root.resolve()),
        "result": result,
    }, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
