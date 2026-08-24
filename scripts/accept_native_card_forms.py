"""Certify native Evo/Hero forms and active ability commands against libg."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from native_core.card_catalog import card_cost, catalog
from native_core.decks import build_replay
from native_core.env import NativeRoyaleEnv


TEMPLATE = json.loads(
    (PROJECT_ROOT / "examples" / "eight-card-bootstrap.json").read_text(
        encoding="utf-8-sig"
    )
)
OPPONENT = [
    "Knight", "Archer", "Giant", "Skeletons",
    "Musketeer", "HogRider", "Cannon", "Arrows",
]
FILLER_POOL = [
    "Goblins", "Skeletons", "Knight", "IceSpirits", "FireSpirits",
    "ElectroSpirit", "Bats", "SpearGoblins", "Bomber", "IceGolemite",
]


def ability_fields(entity: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in entity.items()
        if key.startswith("ability_")
    }


def deck_with_target(card_id: int, form: str) -> list[Any]:
    fillers = [name for name in FILLER_POOL if catalog()[card_id][
        "internal_name"
    ] != name][:7]
    return [
        fillers[0], fillers[1], {"card_id": card_id, "form": form},
        *fillers[2:],
    ]


def valid_cell(env: NativeRoyaleEnv, deck_index: int) -> tuple[int, int]:
    grid = env.probe_grid(side=0, deck_index=deck_index)
    candidates: list[tuple[int, int, int]] = []
    for y, row in enumerate(grid["rows"]):
        for x, value in enumerate(row):
            if value == "1":
                candidates.append((abs(x - 9) + abs(y - 10), x, y))
    if not candidates:
        raise RuntimeError(f"native mask is empty for deck index {deck_index}")
    _, x, y = min(candidates)
    return x * int(grid["cell_size"]), y * int(grid["cell_size"])


def play_current(env: NativeRoyaleEnv, deck_index: int) -> dict[str, Any]:
    x, y = valid_cell(env, deck_index)
    return env.act(side=0, deck_index=deck_index, x=x, y=y)


def hero_ability(env: NativeRoyaleEnv) -> dict[str, Any]:
    replay = build_replay(
        TEMPLATE,
        [
            {"card_id": "Berserker", "form": "hero"},
            "Knight", "Archer", "Giant", "Skeletons", "Musketeer",
            "HogRider", "Cannon",
        ],
        OPPONENT,
        seed=42,
    )
    env.reset(replay, warmup_steps=100)
    for deck_index, steps in ((4, 1), (2, 20)):
        result = env.act(
            side=0, deck_index=deck_index, x=9000, y=10000
        )
        if not result.get("accepted"):
            raise RuntimeError(f"hero setup play rejected: {result}")
        env.step(steps)
    deploy = env.act(side=0, deck_index=0, x=9000, y=10500)
    if not deploy.get("accepted"):
        raise RuntimeError(f"hero form deployment rejected: {deploy}")
    env.step(1)
    state = env.observe()
    hero = next(
        item for item in state["entities"]
        if item.get("native_card_id") == 203000076 and item["side"] == 0
    )
    low_elixir = env.use_ability(side=0, entity_id=hero["entity_id"])
    if low_elixir.get("result_code") != 0x41A:
        raise RuntimeError(f"hero low-elixir gate mismatch: {low_elixir}")
    env.step(50)
    ready_state = env.observe()
    ready = next(
        item for item in ready_state["entities"]
        if item.get("entity_id") == hero["entity_id"]
    )
    if ready.get("ability_available") is not True:
        raise RuntimeError(f"hero ability did not become ready: {ready}")
    accepted = env.use_ability(side=0, entity_id=hero["entity_id"])
    if not accepted.get("accepted") or accepted.get("native_mana_cost") != 3:
        raise RuntimeError(f"hero ability rejected: {accepted}")
    env.step(1)
    cast_state = env.observe()
    cast = next(
        item for item in cast_state["entities"]
        if item.get("entity_id") == hero["entity_id"]
    )
    if cast.get("behavior_state") != 10:
        raise RuntimeError(f"hero did not enter native casting state: {cast}")
    second = env.use_ability(side=0, entity_id=hero["entity_id"])
    if second.get("result_code") != 0x3F6:
        raise RuntimeError(f"hero charge gate mismatch: {second}")
    return {
        "resolved_form_id": deploy["resolved_data_id"],
        "entity_id": hero["entity_id"],
        "low_elixir_result": low_elixir["result_code"],
        "ready": ability_fields(ready),
        "accepted": accepted,
        "cast_behavior_state": cast["behavior_state"],
        "second_press_result": second["result_code"],
    }


def champion_ability(env: NativeRoyaleEnv) -> dict[str, Any]:
    replay = build_replay(
        TEMPLATE,
        [
            "Berserker", "Knight", "ArcherQueen", "Giant", "Skeletons",
            "Musketeer", "HogRider", "Cannon",
        ],
        OPPONENT,
        seed=42,
    )
    env.reset(replay, warmup_steps=100)
    deploy = env.act(side=0, deck_index=2, x=9000, y=10500)
    if not deploy.get("accepted"):
        raise RuntimeError(f"champion deployment rejected: {deploy}")
    env.step(1)
    state = env.observe()
    queen = next(
        item for item in state["entities"]
        if item.get("native_card_id") == 26000072 and item["side"] == 0
    )
    before = ability_fields(queen)
    accepted = env.use_ability(side=0, entity_id=queen["entity_id"])
    if not accepted.get("accepted") or accepted.get("native_mana_cost") != 1:
        raise RuntimeError(f"champion ability rejected: {accepted}")
    env.step(25)
    cast_state = env.observe()
    cast = next(
        item for item in cast_state["entities"]
        if item.get("entity_id") == queen["entity_id"]
    )
    if cast.get("behavior_state") != 10:
        raise RuntimeError(f"champion pending cast did not fire: {cast}")
    return {
        "entity_id": queen["entity_id"],
        "before": before,
        "accepted": accepted,
        "cast_behavior_state": cast["behavior_state"],
    }


def evolution_cycle(env: NativeRoyaleEnv) -> dict[str, Any]:
    replay = build_replay(
        TEMPLATE,
        [
            "Goblins", "Skeletons",
            {"card_id": "Knight", "form": "evolution"},
            "IceSpirits", "FireSpirits", "ElectroSpirit", "Bats",
            "SpearGoblins",
        ],
        OPPONENT,
        seed=42,
    )
    env.reset(replay, warmup_steps=100)
    history: list[dict[str, Any]] = []
    for attempt in range(30):
        state = env.observe()
        hand = sorted(
            state["players"][0]["hand"],
            key=lambda item: (
                0 if item["deck_index"] == 2 else 1,
                card_cost(item["card_id"]),
            ),
        )
        if not hand:
            env.step(20)
            continue
        selected = hand[0]
        result = env.act(
            side=0,
            deck_index=selected["deck_index"],
            x=7000 + (attempt % 5) * 1000,
            y=9000 + (attempt % 3) * 500,
        )
        if not result.get("accepted"):
            env.step(100)
            continue
        row = {
            "tick": state["tick"],
            "deck_index": selected["deck_index"],
            "resolved_data_id": result.get("resolved_data_id"),
        }
        history.append(row)
        env.step(20)
        if result.get("resolved_data_id") == 13000000:
            break
    knight_plays = [item for item in history if item["deck_index"] == 2]
    if [item["resolved_data_id"] for item in knight_plays] != [
        26000000, 26000000, 13000000
    ]:
        raise RuntimeError(f"native evolution cycle mismatch: {knight_plays}")
    state = env.observe()
    evolved = [
        item for item in state["entities"]
        if item.get("native_card_id") == 13000000
    ]
    if not evolved or any(item.get("ability_slot") != 0 for item in evolved):
        raise RuntimeError(f"evolution incorrectly exposed a button: {evolved}")
    return {
        "knight_plays": knight_plays,
        "evolved_entity": {
            "entity_id": evolved[-1]["entity_id"],
            "card_form": evolved[-1]["card_form"],
            "ability_slot": evolved[-1]["ability_slot"],
        },
    }


def all_hero_forms(env: NativeRoyaleEnv) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for card_id, metadata in catalog().items():
        form_id = metadata.get("hero_form_id")
        if form_id is None:
            continue
        replay = build_replay(
            TEMPLATE, deck_with_target(card_id, "hero"), OPPONENT, seed=42
        )
        state = env.reset(replay, warmup_steps=100)
        if not any(item["deck_index"] == 2 for item in state["players"][0]["hand"]):
            raise RuntimeError(f"hero target is absent from opening hand: {card_id}")
        deploy = play_current(env, 2)
        if not deploy.get("accepted") or deploy.get("resolved_data_id") != form_id:
            raise RuntimeError(f"hero form resolution failed: {card_id}: {deploy}")
        env.step(1)
        state = env.observe()
        entities = [
            item for item in state["entities"]
            if item.get("native_card_id") == form_id and item["side"] == 0
        ]
        rows.append({
            "base_card_id": card_id,
            "name": metadata["internal_name"],
            "hero_form_id": form_id,
            "resolved": True,
            "entity_count": len(entities),
            "ability_slots": sorted({
                int(item["ability_slot"]) for item in entities
                if int(item.get("ability_slot", 0)) > 0
            }),
        })
    expected = sum(1 for item in catalog().values() if item.get("hero_form_id"))
    if len(rows) != expected:
        raise RuntimeError(f"hero form coverage mismatch: {len(rows)} != {expected}")
    return {
        "tested": len(rows),
        "resolved": sum(bool(item["resolved"]) for item in rows),
        "rows": rows,
    }


def all_evolution_forms(env: NativeRoyaleEnv) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for card_id, metadata in catalog().items():
        form_id = metadata.get("evolution_form_id")
        if form_id is None:
            continue
        replay = build_replay(
            TEMPLATE, deck_with_target(card_id, "evolution"), OPPONENT,
            seed=42,
        )
        env.reset(replay, warmup_steps=100)
        target_sequence: list[int] = []
        for _ in range(30):
            state = env.observe()
            hand = sorted(
                state["players"][0]["hand"],
                key=lambda item: (
                    0 if item["deck_index"] == 2 else 1,
                    card_cost(item["card_id"]),
                ),
            )
            if not hand:
                env.step(20)
                continue
            selected = hand[0]
            result = play_current(env, selected["deck_index"])
            if not result.get("accepted"):
                env.step(100)
                continue
            if selected["deck_index"] == 2:
                target_sequence.append(int(result["resolved_data_id"]))
            env.step(20)
            if result.get("resolved_data_id") == form_id:
                break
        if not target_sequence or target_sequence[-1] != form_id:
            raise RuntimeError(
                f"evolution form did not resolve: {card_id}: {target_sequence}"
            )
        rows.append({
            "base_card_id": card_id,
            "name": metadata["internal_name"],
            "evolution_form_id": form_id,
            "configured_cycles": metadata["evolution_cycles"],
            "target_play_sequence": target_sequence,
        })
    expected = sum(
        1 for item in catalog().values() if item.get("evolution_form_id")
    )
    if len(rows) != expected:
        raise RuntimeError(
            f"evolution form coverage mismatch: {len(rows)} != {expected}"
        )
    return {"tested": len(rows), "resolved": len(rows), "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=37031)
    parser.add_argument(
        "--output", type=Path,
        default=Path(r"D:\AI_data\cr-native-core\card-form-acceptance.json"),
    )
    args = parser.parse_args()
    with NativeRoyaleEnv(port=args.port) as env:
        result = {
            "schema_version": 1,
            "kind": "libg_card_form_and_ability_acceptance_v1",
            "game_version": "15.535.29",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "native_command": {
                "type": 0x5A,
                "constructor_rva": "0xD8F360",
                "execute_rva": "0xD8F3C0",
            },
            "catalog_counts": json.loads(
                (PROJECT_ROOT / "native_core" / "data" /
                 "live_card_catalog.json").read_text(encoding="utf-8")
            )["counts"],
            "hero_ability": hero_ability(env),
            "champion_ability": champion_ability(env),
            "evolution_cycle": evolution_cycle(env),
            "all_hero_forms": all_hero_forms(env),
            "all_evolution_forms": all_evolution_forms(env),
        }
    result["accepted"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "accepted": True,
        "output": str(args.output.resolve()),
        "hero_command": result["hero_ability"]["accepted"]["result_code"],
        "champion_command": result["champion_ability"]["accepted"][
            "result_code"
        ],
        "evolution_sequence": [
            item["resolved_data_id"]
            for item in result["evolution_cycle"]["knight_plays"]
        ],
        "hero_forms": result["all_hero_forms"]["resolved"],
        "evolution_forms": result["all_evolution_forms"]["resolved"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
