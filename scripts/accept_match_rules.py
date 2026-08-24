"""Certify the frozen standard-1v1 clock, elixir phases and tiebreak."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from native_core.env import NativeRoyaleEnv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=37032)
    parser.add_argument(
        "--output", type=Path,
        default=Path(os.environ.get(
            "CR_SANDBOX_DATA", r"D:\AI_data\cr-native-sandbox"
        )) / "acceptance-match-rules.json",
    )
    args = parser.parse_args()
    template = json.loads(
        (PROJECT_ROOT / "examples" / "eight-card-bootstrap.json").read_text(
            encoding="utf-8-sig"
        )
    )
    env = NativeRoyaleEnv(port=args.port, timeout=90)

    def reset(seed: int, tick: int) -> dict[str, Any]:
        replay = json.loads(json.dumps(template))
        replay["rndSeed"] = seed
        return env.reset(replay, warmup_steps=tick)

    elixir = []
    for tick, multiplier, expected_gain in (
        (100, 1, 1780), (2400, 2, 3570), (4800, 3, 5370),
    ):
        state = reset(10003, tick)
        player = next(item for item in state["players"] if item["side"] == 0)
        arrows = next(item for item in player["hand"] if item["card_id"] == 28000001)
        action = env.act(
            side=0, deck_index=int(arrows["deck_index"]), x=9000, y=3000
        )
        after_action = env.observe()["players"][0]["elixir_raw"]
        env.step(10)
        after_ticks = env.observe()["players"][0]["elixir_raw"]
        measured_gain = int(after_ticks) - int(after_action)
        item = {
            "tick": tick, "multiplier": multiplier,
            "ten_tick_elixir_raw_gain": measured_gain,
            "expected_gain": expected_gain, "action_accepted": action["accepted"],
        }
        if measured_gain != expected_gain or not action["accepted"]:
            raise RuntimeError(f"native elixir phase mismatch: {item}")
        elixir.append(item)

    reset(10004, 100)
    draw = env.step(6200)["episode"]
    if not (
        draw["terminated"] and draw["outcome"] == "draw"
        and draw["termination_reason"] == "native_tiebreak_exact_draw"
    ):
        raise RuntimeError(f"native exact-draw mismatch: {draw}")

    state = reset(10003, 100)
    arrows = next(
        item for item in state["players"][0]["hand"]
        if item["card_id"] == 28000001
    )
    env.act(side=0, deck_index=int(arrows["deck_index"]), x=3500, y=25500)
    env.step(30)
    tiebreak = env.step(6070)["episode"]
    if not (
        tiebreak["terminated"] and tiebreak["winner"] == 0
        and tiebreak["termination_reason"] == "native_tiebreak_hp_drain"
    ):
        raise RuntimeError(f"native tiebreak mismatch: {tiebreak}")
    reset_after_terminal = reset(10003, 100)

    result = {
        "schema_version": 1,
        "kind": "native_standard_1v1_match_rules_acceptance",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "schedule": {
            "regulation_ticks": 3600, "overtime_ticks": 2400,
            "total_ticks": 6000, "tick_seconds": 0.05,
            "double_elixir_tick": 2400, "triple_elixir_tick": 4800,
        },
        "elixir": elixir,
        "exact_draw": {
            key: draw[key] for key in (
                "outcome", "winner", "crowns", "terminal_tick",
                "termination_reason",
            )
        },
        "asymmetric_hp_tiebreak": {
            key: tiebreak[key] for key in (
                "outcome", "winner", "crowns", "terminal_tick",
                "termination_reason", "crown_towers",
            )
        },
        "reset_after_terminal": {
            "tick": reset_after_terminal["tick"],
            "entities": reset_after_terminal["entity_count"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
