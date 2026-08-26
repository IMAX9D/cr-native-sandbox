"""Small real-Worker benchmark for compact batched native Tick traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from expert_v1.tick_store_v1 import TickTraceAccumulator, encode_episode
from native_core.env import NativeRoyaleEnv


def wire_bytes(value: object) -> int:
    return len(json.dumps(value, separators=(",", ":")).encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=38031)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--replay",
        type=Path,
        default=PROJECT_ROOT / "examples" / "eight-card-bootstrap.json",
    )
    args = parser.parse_args()
    if not 1 <= args.steps <= 64 or args.repeats <= 0:
        raise ValueError("steps must be in 1..64 and repeats must be positive")
    replay = json.loads(args.replay.read_text(encoding="utf-8-sig"))
    samples: dict[str, list[dict[str, float]]] = {
        "step_plus_observe_per_tick": [],
        "compact_trace_1": [],
        f"compact_trace_{args.steps}": [],
        f"full_trace_{args.steps}": [],
    }
    with NativeRoyaleEnv(port=args.port, timeout=30) as env:
        for _ in range(args.repeats):
            env.reset(replay, warmup_steps=10)
            started = time.perf_counter()
            response_bytes = 0
            for _tick in range(args.steps):
                response_bytes += wire_bytes(env.step(1))
                response_bytes += wire_bytes(env.observe_train())
            elapsed = time.perf_counter() - started
            samples["step_plus_observe_per_tick"].append({
                "seconds": elapsed,
                "ticks_per_second": args.steps / elapsed,
                "response_bytes": float(response_bytes),
                "rpc_count": float(args.steps * 2),
            })

            env.reset(replay, warmup_steps=10)
            accumulator = TickTraceAccumulator()
            started = time.perf_counter()
            response_bytes = 0
            for _tick in range(args.steps):
                value = env.trace_train(1)
                response_bytes += wire_bytes(value)
                accumulator.extend(value)
            elapsed = time.perf_counter() - started
            samples["compact_trace_1"].append({
                "seconds": elapsed,
                "ticks_per_second": args.steps / elapsed,
                "response_bytes": float(response_bytes),
                "rpc_count": float(args.steps),
            })

            env.reset(replay, warmup_steps=10)
            accumulator = TickTraceAccumulator()
            started = time.perf_counter()
            value = env.trace_train(args.steps)
            elapsed = time.perf_counter() - started
            accumulator.extend(value)
            blob, _stats = encode_episode(
                accumulator.states,
                {"battle_tag": "TRACE-BENCHMARK"},
                anchor_interval=64,
                compression_level=1,
            )
            samples[f"compact_trace_{args.steps}"].append({
                "seconds": elapsed,
                "ticks_per_second": args.steps / elapsed,
                "response_bytes": float(wire_bytes(value)),
                "rpc_count": 1.0,
                "tick_store_bytes": float(len(blob)),
            })

            env.reset(replay, warmup_steps=10)
            started = time.perf_counter()
            value = env.trace(args.steps)
            elapsed = time.perf_counter() - started
            samples[f"full_trace_{args.steps}"].append({
                "seconds": elapsed,
                "ticks_per_second": args.steps / elapsed,
                "response_bytes": float(wire_bytes(value)),
                "rpc_count": 1.0,
            })

    summary = {}
    for name, rows in samples.items():
        summary[name] = {
            key: statistics.median(row[key] for row in rows if key in row)
            for key in sorted({key for row in rows for key in row})
        }
        summary[name]["response_bytes_per_tick"] = (
            summary[name]["response_bytes"] / args.steps
        )
        if "tick_store_bytes" in summary[name]:
            summary[name]["tick_store_bytes_per_state"] = (
                summary[name]["tick_store_bytes"] / (args.steps + 1)
            )
    print(json.dumps({
        "schema_version": 1,
        "kind": "compact_native_tick_trace_benchmark_v1",
        "port": args.port,
        "native_tick_hz": 20,
        "steps": args.steps,
        "repeats": args.repeats,
        "results": summary,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
