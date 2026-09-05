"""Benchmark Stage-2 rollout verification and preprocessing without training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from expert_selfplay_v1.stage2_training import (
    Stage2PPOTrainer,
    Stage2TrainingConfig,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--continuation-checkpoint", type=Path, required=True)
    parser.add_argument("--expert-manifest", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--preprocess-window-size", type=int, default=256)
    parser.add_argument("--preprocess-batch-size", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(
        args.cpu_threads,
        args.preprocess_window_size,
        args.preprocess_batch_size,
    ) < 1:
        raise ValueError("benchmark sizes must be positive")
    torch.set_num_threads(args.cpu_threads)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA benchmark requested but unavailable")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    total_started = time.perf_counter()
    init_started = time.perf_counter()
    trainer = Stage2PPOTrainer(
        base_inference_checkpoint=args.base_checkpoint,
        continuation_checkpoint=args.continuation_checkpoint,
        expert_manifest=args.expert_manifest,
        device=args.device,
        config=Stage2TrainingConfig(
            ppo_epochs=1,
            chunk_batch_size=1,
            preprocess_window_size=args.preprocess_window_size,
            preprocess_batch_size=args.preprocess_batch_size,
        ),
    )
    initialized = time.perf_counter()

    rows = []
    decisions = 0
    chunks = 0
    advantage_sum = 0.0
    return_sum = 0.0
    preprocess_started = time.perf_counter()
    for shard in args.shard:
        shard_started = time.perf_counter()
        prepared, manifest, rollout = trainer.prepare_rollout(shard)
        shard_finished = time.perf_counter()
        rows.append({
            "shard": str(shard.resolve()),
            "policy_version": manifest.policy_version,
            "episodes": int(rollout["episodes"]),
            "decisions": int(rollout["decisions"]),
            "chunks": len(prepared),
            "seconds": shard_finished - shard_started,
        })
        decisions += int(rollout["decisions"])
        chunks += len(prepared)
        advantage_sum += sum(
            float(torch.as_tensor(chunk["advantages"]).double().sum())
            for chunk in prepared
        )
        return_sum += sum(
            float(torch.as_tensor(chunk["returns"]).double().sum())
            for chunk in prepared
        )
        del prepared
    preprocess_finished = time.perf_counter()
    if args.device == "cuda":
        torch.cuda.synchronize()
    finished = time.perf_counter()
    preprocess_seconds = preprocess_finished - preprocess_started
    result = {
        "kind": "cr_native_stage2_preprocessing_benchmark_v1",
        "status": "completed",
        "training_performed": False,
        "device": (
            torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu"
        ),
        "preprocess_window_size": args.preprocess_window_size,
        "preprocess_batch_size": args.preprocess_batch_size,
        "shards": rows,
        "episodes": sum(row["episodes"] for row in rows),
        "decisions": decisions,
        "chunks": chunks,
        "initialize_seconds": initialized - init_started,
        "preprocess_seconds": preprocess_seconds,
        "total_seconds": finished - total_started,
        "decisions_per_second": decisions / max(preprocess_seconds, 1e-9),
        "advantage_sum": advantage_sum,
        "return_sum": return_sum,
        "peak_allocated_mib": (
            torch.cuda.max_memory_allocated() / 1024**2
            if args.device == "cuda" else 0.0
        ),
        "peak_reserved_mib": (
            torch.cuda.max_memory_reserved() / 1024**2
            if args.device == "cuda" else 0.0
        ),
    }
    if any(
        not math.isfinite(float(value))
        for value in result.values()
        if isinstance(value, float)
    ):
        raise FloatingPointError("preprocessing benchmark emitted NaN/Inf")
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
