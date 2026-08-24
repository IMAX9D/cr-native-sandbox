"""Exercise every visible standard card through the original libg action path."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from native_core.card_catalog import card_cost, catalog, standard_card_ids
from native_core.env import NativeRoyaleEnv


DEFAULT_REPLAY = PROJECT_ROOT / "examples" / "eight-card-bootstrap.json"
DEFAULT_OUTPUT = Path(r"D:\AI_data\cr-native-core\full-card-acceptance.json")
SAFE_FILLERS = (
    26000000, 26000001, 26000002, 26000010,
    26000013, 26000030, 26000031, 28000008,
)


def _deck(target: int) -> list[int]:
    values = [target]
    values.extend(card_id for card_id in SAFE_FILLERS if card_id != target)
    return values[:8]


def _set_deck(replay: dict[str, Any], side: int, values: list[int]) -> None:
    replay["battle"][f"deck{side}"]["sp"] = [
        {"d": card_id, "l": 10} for card_id in values
    ]


def _first_cell(rows: list[str]) -> tuple[int, int] | None:
    for row_index, row in enumerate(rows):
        for column, value in enumerate(row):
            if value == "1":
                return row_index, column
    return None


def _wait_for_cost(env: NativeRoyaleEnv, side: int, cost: int) -> dict[str, Any]:
    for _ in range(80):
        state = env.observe()
        player = next(item for item in state["players"] if int(item["side"]) == side)
        if int(player["elixir"]) >= cost:
            return state
        env.step(10)
    raise RuntimeError(f"side {side} never reached {cost} elixir")


def _play(env: NativeRoyaleEnv, deck_index: int, card_id: int) -> dict[str, Any]:
    _wait_for_cost(env, 0, card_cost(card_id))
    grid = env.probe_grid(side=0, deck_index=deck_index)
    cell = _first_cell([str(row) for row in grid["rows"]])
    if cell is None:
        return {"accepted": False, "reason": "empty_native_grid"}
    row, column = cell
    return env.act(
        side=0, deck_index=deck_index,
        x=column * 1000 + 500, y=row * 1000 + 500,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=37031)
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    template = json.loads(args.replay.read_text(encoding="utf-8-sig"))
    opponent = list(SAFE_FILLERS)
    records: list[dict[str, Any]] = []
    with NativeRoyaleEnv(host=args.host, port=args.port, timeout=30) as env:
        for card_id in standard_card_ids():
            replay = deepcopy(template)
            values = _deck(card_id)
            _set_deck(replay, 0, values)
            _set_deck(replay, 1, opponent)
            selected_seed = 2
            replay["rndSeed"] = selected_seed
            state = env.reset(replay, warmup_steps=100)
            player = next(
                item for item in state["players"] if int(item["side"]) == 0
            )
            preplayed = False
            for _ in range(5):
                if 0 in player["hand_deck_indices"]:
                    break
                filler_index = next(
                    index for index in player["hand_deck_indices"]
                    if index > 0
                )
                filler_id = values[filler_index]
                filler = _play(env, filler_index, filler_id)
                if not bool(filler.get("accepted")):
                    raise RuntimeError(f"cycle pre-play failed: {filler}")
                preplayed = True
                env.step(1)
                state = env.observe()
                player = next(
                    item for item in state["players"] if int(item["side"]) == 0
                )
            if 0 not in player["hand_deck_indices"]:
                raise RuntimeError(f"card did not cycle into hand: {card_id}")
            if card_id == 28_000_006 and not preplayed:
                # Mirror needs a previous spell selection.
                filler_index = next(
                    index for index in player["hand_deck_indices"]
                    if index > 0
                )
                filler_id = values[filler_index]
                filler = _play(env, filler_index, filler_id)
                if not bool(filler.get("accepted")):
                    raise RuntimeError(f"mirror pre-play failed: {filler}")
                env.step(1)
            before = env.observe()
            result = _play(env, 0, card_id)
            env.step(1)
            after = env.observe()
            records.append({
                "card_id": card_id,
                "name": catalog()[card_id]["display_name"],
                "type": catalog()[card_id]["type"],
                "elixir": card_cost(card_id),
                "seed": selected_seed,
                "accepted": bool(result.get("accepted")),
                "result": result,
                "tick_before": int(before["tick"]),
                "tick_after": int(after["tick"]),
            })
    output = {
        "schema_version": 1,
        "kind": "libg_full_card_action_acceptance_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "game_version": "15.535.29",
        "tested": len(records),
        "accepted": sum(bool(item["accepted"]) for item in records),
        "failed": [item for item in records if not item["accepted"]],
        "cards": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "tested": output["tested"],
        "accepted": output["accepted"],
        "failed": len(output["failed"]),
    }))
    if output["failed"]:
        raise RuntimeError("one or more visible standard cards were rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
