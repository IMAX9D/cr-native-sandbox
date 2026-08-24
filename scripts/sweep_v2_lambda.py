"""No-gradient native coverage sweep for the v0.2 initial action rate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from native_core.env import NativeRoyaleEnv
from native_core.worker import MultiAvdWorkerPool
from selfplay_v2.baselines import RandomRateLegalPolicy
from selfplay_v2.rollout import aggregate_timed_behavior
from selfplay_v2.vector_rollout import ContinuousRateVectorCollector
from training.schema import RunStore


DEFAULT_OUTPUT = Path(
    r"D:\AI_data\cr-native-core\selfplay-v0.2\lambda-sweeps"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rates", type=float, nargs="+", default=[0.10, 0.20, 0.30, 0.50]
    )
    parser.add_argument("--episodes-per-rate", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=710_000)
    parser.add_argument("--max-ticks", type=int, default=7200)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.workers < 1 or args.workers > 4:
        raise ValueError("lambda sweep uses 1..4 Workers in one AVD")
    replay = json.loads(
        (PROJECT_ROOT / "examples" / "eight-card-bootstrap.json").read_text(
            encoding="utf-8-sig"
        )
    )
    sweep_id = "lambda-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_root / sweep_id
    output.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "kind": "v2_no_gradient_lambda_sweep",
        "sweep_id": sweep_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rates": args.rates,
        "episodes_per_rate": args.episodes_per_rate,
        "actor_updates": 0,
        "critic_updates": 0,
        "results": [],
    }
    RunStore._atomic_json(output / "sweep.json", summary)
    pool = MultiAvdWorkerPool(avds=1, workers_per_avd=args.workers)
    try:
        ready = pool.ensure_ready(configure_direct=True)
        RunStore._atomic_json(output / "workers-ready.json", ready)
        envs = [
            NativeRoyaleEnv(port=port, timeout=30)
            for port in pool.environment_ports("direct")
        ]
        next_seed = args.seed
        for rate_index, rate in enumerate(args.rates):
            all_results = []
            pending = args.episodes_per_rate
            while pending:
                count = min(args.workers, pending)
                seeds = list(range(next_seed, next_seed + count))
                next_seed += count
                pending -= count
                policy = RandomRateLegalPolicy(
                    rate=rate,
                    seed=args.seed + rate_index * 1000 + len(all_results),
                )
                collector = ContinuousRateVectorCollector(
                    envs[:count],
                    policy,  # type: ignore[arg-type]
                    replay,
                    device=torch.device("cpu"),
                    reward_mode="terminal",
                    max_ticks=args.max_ticks,
                )
                all_results.extend(collector.collect(seeds))
            behavior, _histogram = aggregate_timed_behavior(all_results)
            cards = behavior["timing_v2"]["cards"]
            high_ids = (26000003, 26000014, 26000021)
            high_in_hand = sum(cards[str(card)]["ticks_in_hand"] for card in high_ids)
            high_playable = sum(cards[str(card)]["playable_ticks"] for card in high_ids)
            high_selected = sum(cards[str(card)]["selected_count"] for card in high_ids)
            result = {
                "rate": rate,
                "episodes": len(all_results),
                "seeds": [item.seed for item in all_results],
                "normal_terminals": sum(item.terminated for item in all_results),
                "truncated": sum(item.truncated for item in all_results),
                "draw_rate": behavior["draw_rate"],
                "average_episode_ticks": behavior["average_episode_ticks"],
                "average_elixir": behavior["average_elixir"],
                "elixir_leak_ratio": behavior["elixir_leak_ratio"],
                "native_action_rejections": behavior["native_action_rejections"],
                "play_events_per_side_second": behavior["timing_v2"][
                    "play_events_per_side_second"
                ],
                "high_cost_playable_given_in_hand": (
                    high_playable / high_in_hand if high_in_hand else 0.0
                ),
                "high_cost_selected": high_selected,
                "cards": cards,
                "behavior": behavior,
            }
            summary["results"].append(result)
            RunStore._atomic_json(output / f"rate-{rate:.2f}.json", result)
            RunStore._atomic_json(output / "sweep.json", summary)
            print(json.dumps({
                "event": "lambda_rate_complete",
                **{key: result[key] for key in (
                    "rate", "episodes", "normal_terminals", "average_elixir",
                    "play_events_per_side_second",
                    "high_cost_playable_given_in_hand",
                    "high_cost_selected", "native_action_rejections",
                )},
            }, ensure_ascii=False), flush=True)
        summary["completed_utc"] = datetime.now(timezone.utc).isoformat()
        RunStore._atomic_json(output / "sweep.json", summary)
        print(json.dumps({"sweep": str(output), "complete": True}), flush=True)
        return 0
    finally:
        pool.stop(keep_vms=False)


if __name__ == "__main__":
    raise SystemExit(main())
