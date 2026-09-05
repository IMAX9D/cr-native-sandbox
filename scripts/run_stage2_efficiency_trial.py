"""Bounded cloud efficiency experiment with durable results and optional power-off."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_expert_selfplay_v1 import atomic_json


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--continuation-checkpoint", type=Path, required=True)
    p.add_argument("--behavior-checkpoint", type=Path, required=True)
    p.add_argument("--expert-manifest", type=Path, required=True)
    p.add_argument("--runtime-root", type=Path, required=True)
    p.add_argument("--policy-version", type=int, required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--collectors", type=int, default=2)
    p.add_argument("--base-port", type=int, default=19031)
    p.add_argument("--step-ticks", type=int, default=12)
    p.add_argument("--max-seconds", type=float, default=600)
    p.add_argument("--reuse-collection-root", type=Path)
    p.add_argument("--variants", default="fp32,fp32-cache,bf16-fused")
    p.add_argument("--shutdown-grace-seconds", type=float, default=20)
    p.add_argument("--pipeline-updates", type=int, default=0)
    p.add_argument("--preprocess-window-size", type=int, default=256)
    p.add_argument("--preprocess-batch-size", type=int, default=3)
    p.add_argument("--prepared-cache-gib", type=float, default=4.0)
    p.add_argument("--chunk-batch-size", type=int, default=8)
    p.add_argument("--chunk-padding-multiple", type=int, default=0)
    p.add_argument("--training-precision", choices=("float32", "bfloat16", "float16"), default="float32")
    p.add_argument("--fused-optimizer", action="store_true")
    p.add_argument("--isolate-updates", action="store_true")
    p.add_argument("--shutdown-on-finish", action="store_true")
    args = p.parse_args()
    root = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = {"status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
             "formal_policy_modified": False, "stages": [],
             "shutdown_authorized": args.shutdown_on_finish}

    def save():
        state["elapsed_seconds"] = time.monotonic() - started
        atomic_json(root / "trial-progress.json", state)
        print(json.dumps(state, ensure_ascii=False, allow_nan=False), flush=True)

    def stage(name, command):
        remaining = args.max_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("efficiency trial time limit reached")
        state["active_stage"] = name
        save()
        before = time.monotonic()
        with (root / f"{name}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log,
                                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    timeout=remaining, check=False)
        state["stages"].append({"name": name, "returncode": result.returncode,
                                "seconds": time.monotonic() - before})
        save()
        if result.returncode:
            tail = (root / f"{name}.log").read_text(encoding="utf-8", errors="replace")[-5000:]
            raise RuntimeError(f"{name} exited with {result.returncode}: {tail}")

    save()
    exit_code = 0
    try:
        if args.pipeline_updates:
            stage("workers", [sys.executable, "scripts/ensure_bionic_workers.py",
                  "--runtime-root", str(args.runtime_root), "--base-port", str(args.base_port),
                  "--count", str(args.workers), "--execution-mode", "jit"])
            pipeline_command = [sys.executable, "scripts/run_expert_selfplay_stage2_loop.py",
                  "--base-checkpoint", str(args.base_checkpoint),
                  "--base-opponent-checkpoint", str(args.base_checkpoint),
                  "--initial-continuation", str(args.continuation_checkpoint),
                  "--initial-behavior-export", str(args.behavior_checkpoint),
                  "--expert-manifest", str(args.expert_manifest),
                  "--ports", f"{args.base_port}-{args.base_port + args.workers - 1}",
                  "--collectors", str(args.collectors), "--updates", str(args.pipeline_updates),
                  "--run-root", str(root / "pipeline"),
                  "--learner-deck", "examples/hog-2.6-evo-hero.json",
                  "--opponent-deck-root", "top-deck-presets-v1",
                  "--step-ticks", str(args.step_ticks), "--collection-waves", "2",
                  "--device", "cuda", "--collector-cpu-threads", "2",
                  "--trainer-cpu-threads", "8", "--ppo-epochs", "2",
                  "--chunk-batch-size", str(args.chunk_batch_size),
                  "--chunk-padding-multiple", str(args.chunk_padding_multiple),
                  "--training-precision", args.training_precision,
                  "--preprocess-batch-size", str(args.preprocess_batch_size),
                  "--preprocess-window-size", str(args.preprocess_window_size),
                  "--prepared-cache-gib", str(args.prepared_cache_gib),
                  "--persistent-learner", "--overlap-preparation", "--enable-mps",
                  "--mps-root", "/root/autodl-tmp/mps-eff-pipeline",
                  "--worker-runtime-root", str(args.runtime_root),
                  "--worker-execution-mode", "jit"]
            if args.fused_optimizer:
                pipeline_command.append("--fused-optimizer")
            if args.isolate_updates:
                pipeline_command.append("--isolate-updates")
            stage("pipeline", pipeline_command)
            state["status"] = "completed"
            return 0
        collection = (root / "collection" if args.reuse_collection_root is None
                      else args.reuse_collection_root.resolve(strict=True))
        if args.reuse_collection_root is None:
            stage("workers", [sys.executable, "scripts/ensure_bionic_workers.py",
                  "--runtime-root", str(args.runtime_root), "--base-port", str(args.base_port),
                  "--count", str(args.workers), "--execution-mode", "jit"])
            stage("collection", [sys.executable, "scripts/benchmark_stage2_collection.py",
              "--base-opponent-checkpoint", str(args.base_checkpoint),
              "--behavior-checkpoint", str(args.behavior_checkpoint),
              "--expert-manifest", str(args.expert_manifest),
              "--ports", f"{args.base_port}-{args.base_port + args.workers - 1}",
              "--collectors", str(args.collectors), "--collection-waves", "2",
              "--run-root", str(collection), "--learner-deck", "examples/hog-2.6-evo-hero.json",
              "--opponent-deck-root", "top-deck-presets-v1", "--host", "127.0.0.1",
              "--step-ticks", str(args.step_ticks), "--seed", "20905410",
              "--policy-version", str(args.policy_version), "--device", "cuda",
              "--collector-cpu-threads", "2", "--enable-mps",
                  "--mps-root", "/root/autodl-tmp/mps-eff-trial"])
        shards = sorted(collection.glob("collect-p*/rollouts/shard-*"))
        command = [sys.executable, "scripts/benchmark_stage2_learner.py",
                   "--base-checkpoint", str(args.base_checkpoint),
                   "--continuation-checkpoint", str(args.continuation_checkpoint),
                   "--expert-manifest", str(args.expert_manifest),
                   "--output", str(root / "learner-benchmark.json"),
                   "--variants", args.variants, "--save-artifacts",
                   "--chunk-batch-size", str(args.chunk_batch_size),
                   "--chunk-padding-multiple", str(args.chunk_padding_multiple),
                   "--preprocess-window-size", str(args.preprocess_window_size),
                   "--preprocess-batch-size", str(args.preprocess_batch_size),
                   "--cache-gib", str(args.prepared_cache_gib)]
        for shard in shards:
            command.extend(("--shard", str(shard)))
        stage("learner", command)
        state["status"] = "completed"
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        exit_code = 1
    finally:
        state.pop("active_stage", None)
        save()
        if args.shutdown_on_finish:
            trash = Path("/root/.local/share/Trash")
            if trash.exists() and (not trash.is_dir() or any(trash.iterdir())):
                state["shutdown_error"] = "provider wrapper would erase nonempty Trash"
                save()
            else:
                state["shutdown_requested"] = True
                save()
                os.sync()
                time.sleep(max(0.0, args.shutdown_grace_seconds))
                # AutoDL's existing provider wrapper stops the billed container.
                subprocess.run(["/bin/bash", "/usr/bin/shutdown"], check=True, timeout=30)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
