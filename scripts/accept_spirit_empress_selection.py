"""Accept Spirit Empress's original 3/6-elixir native choice selector."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from native_core.decks import build_replay
from native_core.env import NativeRoyaleEnv


TEMPLATE = ROOT / "examples" / "eight-card-bootstrap.json"
DECK = (
    "MergeMaiden", "Skeletons", "ElectroSpirit", "FireSpirits",
    "Goblins", "Archer", "Musketeer", "Zap",
)
CONTROL_FILLERS = (
    "Skeletons", "ElectroSpirit", "FireSpirits", "Goblins",
    "Archer", "Musketeer", "Zap",
)


def _player(state: dict[str, Any]) -> dict[str, Any]:
    return next(item for item in state["players"] if int(item["side"]) == 0)


def _spawn_ids(state: dict[str, Any]) -> list[int]:
    return [
        int(item["card_id"])
        for item in state["entities"]
        if int(item.get("side", -1)) == 0 and int(item.get("card_id", -1)) > 0
    ]


def _run(env: NativeRoyaleEnv, replay: dict[str, Any], *, low: bool) -> dict[str, Any]:
    state = env.reset(replay, warmup_steps=100)
    if low:
        spent = env.act(side=0, deck_index=6, x=5500, y=23000)
        if not spent.get("accepted"):
            raise RuntimeError(f"4-elixir setup play failed: {spent}")
        state = env.observe()
    before_raw = int(_player(state)["elixir_raw"])
    grid = env.probe_grid(side=0, deck_index=0)
    result = env.act(side=0, deck_index=0, x=12500, y=24000)
    after_raw = int(_player(env.observe())["elixir_raw"])
    env.step(1)
    spawned = _spawn_ids(env.observe())
    expected_id = 26000104 if low else 26000105
    expected_cost_raw = 30000 if low else 60000
    checks = {
        "accepted": result.get("accepted") is True,
        "resolved": int(result.get("resolved_data_id", -1)) == expected_id,
        "probe_resolved": int(grid.get("resolved_data_id", -1)) == expected_id,
        "cost": int(grid.get("card_cost_raw", -1)) == expected_cost_raw,
        "deduction": before_raw - after_raw == expected_cost_raw,
        "spawned": expected_id in spawned,
        "native_strategy": result.get("selection_strategy") == "native_dynamic_choice",
        "native_builder": result.get("selection_builder_rva") == "0xd71800",
    }
    if not all(checks.values()):
        raise RuntimeError(f"Spirit Empress native selection mismatch: {checks}")
    return {
        "tier": "low_ground" if low else "high_flying",
        "elixir_raw_before": before_raw,
        "elixir_raw_after": after_raw,
        "deducted_raw": before_raw - after_raw,
        "resolved_data_id": int(result["resolved_data_id"]),
        "spawned_card_ids": spawned,
        "packed_selection": int(result["packed_selection"]),
        "selection_form_index": int(result["selection_form_index"]),
        "selection_strategy": result["selection_strategy"],
        "selection_builder_rva": result["selection_builder_rva"],
        "selection_root_vtable_rva": result["selection_root_vtable_rva"],
        "probe": {
            key: grid[key] for key in (
                "resolved_data_id", "card_cost_raw", "selection_form_index",
                "selection_strategy", "selection_builder_rva",
            )
        },
        "checks": checks,
    }


def _control(
    env: NativeRoyaleEnv, template: dict[str, Any], *, name: str,
    target: str, form: str, expected_id: int,
) -> dict[str, Any]:
    first: str | dict[str, str] = target
    if form != "base":
        first = {"card_id": target, "form": form}
    deck = (first, *CONTROL_FILLERS)
    replay = build_replay(template, deck, deck, seed=2)
    state = env.reset(replay, warmup_steps=100)
    if 0 not in _player(state)["hand_deck_indices"]:
        raise RuntimeError(f"control target absent from hand: {name}")
    grid = env.probe_grid(side=0, deck_index=0)
    result = env.act(side=0, deck_index=0, x=12500, y=24000)
    env.step(1)
    spawned = _spawn_ids(env.observe())
    checks = {
        "accepted": result.get("accepted") is True,
        "resolved": int(result.get("resolved_data_id", -1)) == expected_id,
        "probe_resolved": int(grid.get("resolved_data_id", -1)) == expected_id,
        "canonical": result.get("selection_strategy") == "canonical",
        "spawned": expected_id in spawned,
    }
    if not all(checks.values()):
        raise RuntimeError(f"selection regression control failed: {name}: {checks}")
    return {
        "name": name,
        "configured_form": form,
        "resolved_data_id": int(result["resolved_data_id"]),
        "selection_strategy": result["selection_strategy"],
        "selection_builder_rva": result["selection_builder_rva"],
        "checks": checks,
    }
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=38035)
    parser.add_argument(
        "--output", type=Path,
        default=Path(r"D:\AI_data\cr-native-core\spirit-empress-selection-acceptance.json"),
    )
    args = parser.parse_args()
    template = json.loads(TEMPLATE.read_text(encoding="utf-8-sig"))
    replay = build_replay(template, DECK, DECK, seed=2)
    with NativeRoyaleEnv(port=args.port, timeout=30) as env:
        scenarios = [_run(env, replay, low=True), _run(env, replay, low=False)]
        controls = [
            _control(
                env, template, name="ordinary_knight", target="Knight",
                form="base", expected_id=26000000,
            ),
            _control(
                env, template, name="evolution_knight_first_cycle",
                target="Knight", form="evolution", expected_id=26000000,
            ),
            _control(
                env, template, name="hero_knight", target="Knight",
                form="hero", expected_id=203000000,
            ),
            _control(
                env, template, name="champion_archer_queen",
                target="ArcherQueen", form="base", expected_id=26000072,
            ),
        ]
    result = {
        "schema_version": 1,
        "kind": "libg_spirit_empress_dynamic_selection_acceptance_v1",
        "game_version": "15.535.29",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(
            all(item["checks"].values()) for item in (*scenarios, *controls)
        ),
        "scenarios": scenarios,
        "regression_controls": controls,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "passed": result["passed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
