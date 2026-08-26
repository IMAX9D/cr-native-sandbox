"""Benchmark bounded native seed resolution on a fixed expert selection."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from queue import Empty, Queue
import statistics
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from expert_v1.native_replay_plan import compile_battle
from expert_v1.native_replay_runner import load_template
from expert_v1.native_seed_search import (
    clear_native_seed_cache,
    native_seed_cache_size,
    resolve_native_seed,
)
from native_core.env import NativeRoyaleEnv


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ports", default="38031,38032,38033,38034")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--maximum-seeds", type=int, default=4096)
    parser.add_argument(
        "--template", type=Path,
        default=PROJECT_ROOT / "examples" / "eight-card-bootstrap.json",
    )
    args = parser.parse_args()
    ports = [int(value) for value in args.ports.split(",") if value.strip()]
    if not ports:
        raise ValueError("at least one port is required")
    rows = [
        json.loads(line)
        for line in args.selection.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ][: args.limit]
    tasks: Queue[tuple[int, dict[str, Any]]] = Queue()
    for index, row in enumerate(rows):
        tasks.put((index, row))
    template = load_template(args.template)
    clear_native_seed_cache()
    started = time.perf_counter()

    def worker(port: int) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        with NativeRoyaleEnv(port=port, timeout=60.0) as env:
            while True:
                try:
                    index, row = tasks.get_nowait()
                except Empty:
                    break
                item_started = time.perf_counter()
                try:
                    source_path = Path(row["source_path"])
                    source = json.loads(source_path.read_text(encoding="utf-8-sig"))
                    plan = compile_battle(
                        source,
                        terminal_crowns=(
                            int(row["team_crowns"]),
                            int(row["opponent_crowns"]),
                        ),
                    )
                    result = resolve_native_seed(
                        env,
                        plan,
                        template,
                        maximum_seeds_to_test=args.maximum_seeds,
                        warmup_tick=10,
                    )
                    output.append({
                        "selection_index": index,
                        "battle_tag": plan.battle_tag,
                        "port": port,
                        "success": True,
                        **result.audit(),
                        "wall_seconds": time.perf_counter() - item_started,
                    })
                except Exception as error:
                    output.append({
                        "selection_index": index,
                        "battle_tag": row.get("battle_tag"),
                        "port": port,
                        "success": False,
                        "failure": f"{type(error).__name__}: {error}",
                        "wall_seconds": time.perf_counter() - item_started,
                    })
        return output

    with ThreadPoolExecutor(max_workers=len(ports)) as executor:
        results = [
            item
            for batch in executor.map(worker, ports)
            for item in batch
        ]
    results.sort(key=lambda item: int(item["selection_index"]))
    wall = time.perf_counter() - started
    successes = [item for item in results if item["success"]]
    tested = [float(item["seeds_tested"]) for item in successes]
    resets = [float(item["seed_search_native_resets"]) for item in successes]
    item_wall = [float(item["wall_seconds"]) for item in successes]
    total_resets = int(sum(resets))
    extrapolated_resets = (
        None if not successes else int(round(statistics.mean(resets) * 100_000))
    )
    throughput = len(results) / wall if wall else 0.0
    summary = {
        "schema_version": 1,
        "kind": "native_seed_search_benchmark_v1",
        "selection": str(args.selection.resolve()),
        "battles": len(results),
        "successes": len(successes),
        "failures": len(results) - len(successes),
        "workers": len(ports),
        "ports": ports,
        "maximum_seeds_to_test": args.maximum_seeds,
        "wall_seconds": wall,
        "battles_per_second": throughput,
        "total_native_resets": total_resets,
        "cache_entries": native_seed_cache_size(),
        "cache_hits": sum(bool(item.get("seed_search_cache_hit")) for item in successes),
        "chosen_seed": {
            "min": None if not tested else int(min(item["chosen_seed"] for item in successes)),
            "median": _percentile(
                [float(item["chosen_seed"]) for item in successes], 0.5
            ),
            "p95": _percentile(
                [float(item["chosen_seed"]) for item in successes], 0.95
            ),
            "p99": _percentile(
                [float(item["chosen_seed"]) for item in successes], 0.99
            ),
            "max": None if not tested else int(max(item["chosen_seed"] for item in successes)),
        },
        "seeds_tested": {
            "mean": None if not tested else statistics.mean(tested),
            "median": _percentile(tested, 0.5),
            "p95": _percentile(tested, 0.95),
            "p99": _percentile(tested, 0.99),
            "max": None if not tested else int(max(tested)),
        },
        "per_battle_wall_seconds": {
            "mean": None if not item_wall else statistics.mean(item_wall),
            "median": _percentile(item_wall, 0.5),
            "p95": _percentile(item_wall, 0.95),
            "max": None if not item_wall else max(item_wall),
        },
        "extrapolated_100k": {
            "native_resets": extrapolated_resets,
            "wall_seconds_at_measured_four_worker_rate": (
                None if throughput <= 0 else 100_000 / throughput
            ),
            "hours_at_measured_four_worker_rate": (
                None if throughput <= 0 else 100_000 / throughput / 3600.0
            ),
            "scope": "seed_resolution_only_not_tick_replay",
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    return 0 if not summary["failures"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
