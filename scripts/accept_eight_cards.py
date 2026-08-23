"""Exercise every frozen card through libg's native command path."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from native_core.env import NativeRoyaleEnv
from training.schema import CARD_IDS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=37031)
    parser.add_argument("--seed-base", type=int, default=10000)
    parser.add_argument(
        "--output", type=Path,
        default=Path(r"D:\AI_data\cr-native-core\acceptance-eight-cards.json"),
    )
    args = parser.parse_args()
    replay = json.loads(
        (PROJECT_ROOT / "examples" / "eight-card-bootstrap.json").read_text(
            encoding="utf-8-sig"
        )
    )
    env = NativeRoyaleEnv(port=args.port, timeout=30)
    results: list[dict[str, Any]] = []
    for card_id in CARD_IDS:
        for offset in range(1, 65):
            episode = json.loads(json.dumps(replay))
            episode["rndSeed"] = args.seed_base + offset
            state = env.reset(episode, warmup_steps=100)
            player = next(item for item in state["players"] if item["side"] == 0)
            matches = [item for item in player["hand"] if item["card_id"] == card_id]
            if not matches:
                continue
            deck_index = int(matches[0]["deck_index"])
            grid = env.probe_grid(side=0, deck_index=deck_index)
            cells = [
                (column * 1000 + 500, row * 1000 + 500)
                for row, values in enumerate(grid["rows"])
                for column, value in enumerate(values)
                if value == "1"
            ]
            x, y = min(cells, key=lambda point: abs(point[0] - 9000) + abs(point[1] - 10000))
            before_hand = tuple(player["hand_deck_indices"])
            before_elixir = int(player["elixir_raw"])
            before_effects = int(state["effect_count"])
            action = env.act(side=0, deck_index=deck_index, x=x, y=y)
            env.step(5)
            after = env.observe()
            after_player = next(item for item in after["players"] if item["side"] == 0)
            result = {
                "card_id": card_id,
                "seed": args.seed_base + offset,
                "accepted": bool(action["accepted"]),
                "result_code": int(action["result_code"]),
                "point": [x, y],
                "valid_cells": int(grid["valid_cells"]),
                "hand_rotated": tuple(after_player["hand_deck_indices"]) != before_hand,
                "elixir_spent": int(after_player["elixir_raw"]) < before_elixir,
                "native_entities": sum(
                    1 for entity in after["entities"] if entity.get("card_id") == card_id
                ),
                "effects_delta": int(after["effect_count"]) - before_effects,
            }
            if not (result["accepted"] and result["hand_rotated"] and result["elixir_spent"]):
                raise RuntimeError(f"native card acceptance failed: {result}")
            results.append(result)
            break
        else:
            raise RuntimeError(f"card {card_id} was absent from all probed openings")
    envelope = {
        "schema_version": 1,
        "kind": "native_eight_card_action_acceptance",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": len(results) == len(CARD_IDS),
        "cards": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(envelope, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(envelope, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
