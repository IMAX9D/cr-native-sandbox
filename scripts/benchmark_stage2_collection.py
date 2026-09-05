"""Benchmark one full multi-process Stage-2 collection wave without training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_expert_selfplay_stage2_loop import (
    collector_command,
    parse_ports,
    split_ports,
    validate_collection,
)
from expert_selfplay_v1.mps_runtime import ManagedMPS


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _terminate(processes: list[subprocess.Popen[str]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


def run(args: argparse.Namespace) -> dict[str, Any]:
    ports = parse_ports(args.ports)
    groups = split_ports(ports, args.collectors)
    run_root = args.run_root.resolve()
    if run_root.exists():
        raise FileExistsError(f"benchmark output already exists: {run_root}")
    run_root.mkdir(parents=True)

    processes: list[subprocess.Popen[str]] = []
    logs = []
    collection_dirs: list[Path] = []
    mps_runtime = ManagedMPS(
        enabled=bool(args.enable_mps),
        root=(args.mps_root or (run_root / "mps-runtime")),
    )
    process_environment = mps_runtime.start()
    mps_stop: dict[str, Any] = {
        "requested": False, "returncode": None, "output": ""
    }
    started = time.perf_counter()
    try:
        for collector_index, group in enumerate(groups):
            collection_dir = run_root / f"collect-p{collector_index:02d}"
            collection_dirs.append(collection_dir)
            log = (run_root / f"collect-p{collector_index:02d}.log").open(
                "w", encoding="utf-8"
            )
            logs.append(log)
            process = subprocess.Popen(
                collector_command(
                    args,
                    ports=group,
                    run_dir=collection_dir,
                    seed=args.seed + collector_index,
                    policy_version=args.policy_version,
                    behavior_checkpoint=args.behavior_checkpoint.resolve(strict=True),
                ),
                cwd=str(PROJECT_ROOT),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
                env=process_environment,
            )
            processes.append(process)
        codes = [process.wait() for process in processes]
    except BaseException:
        _terminate(processes)
        raise
    finally:
        for log in logs:
            log.close()
        mps_stop = mps_runtime.stop()
    finished = time.perf_counter()

    if any(code != 0 for code in codes):
        raise RuntimeError(f"collector failure codes: {codes}")

    rows = []
    total_episodes = 0
    total_decisions = 0
    total_chunks = 0
    total_bytes = 0
    for collector_index, collection_dir in enumerate(collection_dirs):
        result = _read_json(collection_dir / "collection-result.json")
        raw_shards = result.get("shards")
        shard_paths = (
            [Path(str(row["directory"])) for row in raw_shards]
            if isinstance(raw_shards, list) and raw_shards
            else [validate_collection(collection_dir)]
        )
        if any(not path.is_dir() for path in shard_paths):
            raise RuntimeError(f"collector shard is missing: {collection_dir}")
        rollout_bytes = sum(
            (path / "rollout.pt").stat().st_size for path in shard_paths
        )
        row = {
            "collector": collector_index,
            "ports": groups[collector_index],
            "episodes": int(result["episodes"]),
            "decisions": int(result["decisions"]),
            "chunks": int(result["chunks"]),
            "rollout_bytes": rollout_bytes,
            "timings": result.get("timings", {}),
            "shard": str(shard_paths[0]),
            "shards": [str(path) for path in shard_paths],
            "ledger_state": result.get("ledger_state"),
        }
        rows.append(row)
        total_episodes += row["episodes"]
        total_decisions += row["decisions"]
        total_chunks += row["chunks"]
        total_bytes += rollout_bytes

    wall_seconds = finished - started
    result = {
        "kind": "cr_native_stage2_collection_benchmark_v1",
        "status": "completed",
        "training_performed": False,
        "policy_version": args.policy_version,
        "collectors": args.collectors,
        "workers": len(ports),
        "episodes_per_collector": len(groups[0]) * int(args.collection_waves),
        "episodes_per_wave": len(groups[0]),
        "collection_waves": int(args.collection_waves),
        "async_shard_writes": bool(args.async_shard_writes),
        "rolling_collection": bool(args.rolling_collection),
        "episodes": total_episodes,
        "decisions": total_decisions,
        "chunks": total_chunks,
        "rollout_bytes": total_bytes,
        "wall_seconds": wall_seconds,
        "games_per_second": total_episodes / max(wall_seconds, 1e-9),
        "games_per_hour": total_episodes * 3600.0 / max(wall_seconds, 1e-9),
        "games_per_day": total_episodes * 86400.0 / max(wall_seconds, 1e-9),
        "decisions_per_second": total_decisions / max(wall_seconds, 1e-9),
        "bytes_per_decision": total_bytes / max(total_decisions, 1),
        "mps": {
            "enabled": bool(args.enable_mps),
            "root": str(mps_runtime.root),
            "stop": mps_stop,
        },
        "collector_rows": rows,
    }
    for name, value in result.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise FloatingPointError(f"benchmark field {name} is NaN/Inf")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-opponent-checkpoint", type=Path, required=True)
    parser.add_argument("--behavior-checkpoint", type=Path, required=True)
    parser.add_argument("--expert-manifest", type=Path, required=True)
    parser.add_argument("--ports", required=True)
    parser.add_argument("--collectors", type=int, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--learner-deck", type=Path, required=True)
    parser.add_argument("--opponent-deck-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--step-ticks", type=int, default=4)
    parser.add_argument("--idle-step-ticks", type=int)
    parser.add_argument("--collection-waves", type=int, default=1)
    parser.add_argument("--async-shard-writes", action="store_true")
    parser.add_argument("--rolling-collection", action="store_true")
    parser.add_argument("--compile-actor", action="store_true")
    parser.add_argument("--compile-batch-size", type=int)
    parser.add_argument("--compile-entity-slots", type=int)
    parser.add_argument("--dense-policy-sampling", action="store_true")
    parser.add_argument("--enable-mps", action="store_true")
    parser.add_argument("--mps-root", type=Path)
    parser.add_argument("--max-decisions", type=int, default=3000)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=20904000)
    parser.add_argument("--policy-version", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--collector-cpu-threads", type=int, default=2)
    parser.add_argument("--policy-server-address")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--collect-script",
        type=Path,
        default=PROJECT_ROOT / "scripts/run_expert_selfplay_v1.py",
    )
    args = parser.parse_args()
    if min(args.collectors, args.collector_cpu_threads) < 1:
        raise ValueError("collector counts must be positive")
    result = run(args)
    output = args.run_root.resolve() / "benchmark-result.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
