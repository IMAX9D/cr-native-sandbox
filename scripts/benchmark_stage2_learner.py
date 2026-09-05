"""Compare guarded PPO execution modes on identical immutable native rollouts."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from expert_selfplay_v1.actor_adapter import actor_state_digest
from expert_selfplay_v1.stage2_training import Stage2PPOTrainer, Stage2TrainingConfig
from scripts.run_expert_selfplay_v1 import atomic_json
from scripts.train_expert_selfplay_stage2 import _admit_collection_batches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--continuation-checkpoint", type=Path, required=True)
    parser.add_argument("--expert-manifest", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", default="fp32,fp32-cache,bf16-fused")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--chunk-batch-size", type=int, default=8)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--chunk-padding-multiple", type=int, default=0)
    parser.add_argument("--preprocess-window-size", type=int, default=256)
    parser.add_argument("--preprocess-batch-size", type=int, default=3)
    parser.add_argument("--cache-gib", type=float, default=4.0)
    parser.add_argument("--save-artifacts", action="store_true")
    args = parser.parse_args()
    variants = args.variants.split(",")
    if not variants or set(variants) - {"fp32", "fp32-cache", "bf16-fused", "fp16-fused"}:
        raise ValueError("unknown learner benchmark variant")
    if args.output.exists():
        raise FileExistsError(args.output)
    shards = [path.resolve(strict=True) for path in args.shard]
    _manifests, ledgers = _admit_collection_batches(shards)
    for ledger, _batch_id in ledgers:
        ledger.close()
    torch.set_num_threads(args.cpu_threads)
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires the cloud CUDA device")
    result: dict[str, Any] = {
        "kind": "cr_native_stage2_learner_benchmark_v1", "status": "running",
        "formal_policy_modified": False, "rollouts_consumed": False,
        "device": torch.cuda.get_device_name(0), "variants": [],
    }
    atomic_json(args.output, result)
    chunks = None
    preparation_seconds = 0.0
    try:
        for variant in variants:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            config = Stage2TrainingConfig(
                ppo_epochs=args.ppo_epochs, chunk_batch_size=args.chunk_batch_size,
                preprocess_window_size=args.preprocess_window_size,
                preprocess_batch_size=args.preprocess_batch_size,
                prepared_cache_gib=0.0 if variant == "fp32" else args.cache_gib,
                training_precision={"bf16-fused": "bfloat16", "fp16-fused": "float16"}.get(variant, "float32"),
                fused_optimizer=variant in ("bf16-fused", "fp16-fused"),
                chunk_padding_multiple=args.chunk_padding_multiple,
            )
            trainer = Stage2PPOTrainer(
                base_inference_checkpoint=args.base_checkpoint,
                continuation_checkpoint=args.continuation_checkpoint,
                expert_manifest=args.expert_manifest, device="cuda", config=config,
            )
            initialized = time.perf_counter()
            if chunks is None:
                preparation_started = time.perf_counter()
                chunks = []
                episodes = decisions = 0
                for shard in shards:
                    prepared, _batch, summary = trainer.prepare_rollout(shard)
                    chunks.extend(prepared)
                    episodes += summary["episodes"]
                    decisions += summary["decisions"]
                preparation_seconds = time.perf_counter() - preparation_started
                result.update(episodes=episodes, decisions=decisions, chunks=len(chunks),
                              preparation_seconds=preparation_seconds)
            row: dict[str, Any] = {
                "variant": variant, "config": asdict(config),
                "initialization_seconds": initialized - started,
            }
            result["active_variant"] = variant
            atomic_json(args.output, result)
            training_started = time.perf_counter()
            try:
                metrics, guard, retry = trainer.train_update(chunks)
                torch.cuda.synchronize()
                row.update(status="accepted", metrics=metrics,
                           guard=asdict(guard), retry=retry,
                           actor_master_sha256=actor_state_digest(trainer.model.actor))
            except (RuntimeError, FloatingPointError) as error:
                row.update(status="rejected", error=f"{type(error).__name__}: {error}")
            row.update(
                training_seconds=time.perf_counter() - training_started,
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
            )
            if row["status"] == "accepted" and args.save_artifacts:
                publish_started = time.perf_counter()
                checkpoint, export = trainer.save(
                    args.output.parent / variant,
                    metrics=metrics, guard=guard, retry_attempt=retry,
                    rollout={"episodes": episodes, "decisions": decisions,
                             "shards": [str(path) for path in shards]},
                )
                row.update(publish_seconds=time.perf_counter() - publish_started,
                           checkpoint=str(checkpoint), behavior_export=str(export))
            row["full_update_seconds"] = (
                row["initialization_seconds"] + preparation_seconds
                + row["training_seconds"] + row.get("publish_seconds", 0.0)
            )
            result["variants"].append(row)
            result.pop("active_variant", None)
            atomic_json(args.output, result)
            print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
            del trainer
        result["status"] = "completed"
        atomic_json(args.output, result)
        return 0
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        atomic_json(args.output, result)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
