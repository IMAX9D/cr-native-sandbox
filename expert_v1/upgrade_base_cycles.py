"""Recover exact base-card hand cycles from schema-v1 full-play replays."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any, Iterable

import numpy as np

try:
    import orjson
except ImportError:  # pragma: no cover
    orjson = None


def loads(raw: bytes) -> dict[str, Any]:
    value = orjson.loads(raw) if orjson is not None else json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError("JSON root is not an object")
    return value


def dump_line(value: Any) -> bytes:
    encoded = (
        orjson.dumps(value) if orjson is not None
        else json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    return encoded + b"\n"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initial_states() -> tuple[np.ndarray, np.ndarray]:
    states: list[tuple[int, tuple[int, ...]]] = []
    for hand in itertools.combinations(range(8), 4):
        remaining = [index for index in range(8) if index not in hand]
        mask = sum(1 << index for index in hand)
        for queue in itertools.permutations(remaining):
            states.append((mask, queue))
    return (
        np.asarray([state[0] for state in states], dtype=np.uint16),
        np.asarray([state[1] for state in states], dtype=np.uint8),
    )


INITIAL_MASKS, INITIAL_QUEUES = initial_states()
FORM_SUFFIX_RE = re.compile(r"-(?:ev\d+|hero)$")


def base_card(value: str) -> str:
    return FORM_SUFFIX_RE.sub("", str(value).lower())


def solve_cycle(card_indices: list[int]) -> dict[str, Any]:
    """Infer pre-action hand and next-card state using the full future sequence."""
    masks = INITIAL_MASKS.copy()
    queues = INITIAL_QUEUES.copy()
    hand_masks: list[int | None] = []
    next_cards: list[int | None] = []
    failure_action_index: int | None = None

    for action_index, played in enumerate(card_indices):
        hand_unique = bool(len(masks) and np.all(masks == masks[0]))
        next_unique = bool(len(queues) and np.all(queues[:, 0] == queues[0, 0]))
        hand_masks.append(int(masks[0]) if hand_unique else None)
        next_cards.append(int(queues[0, 0]) if next_unique else None)

        keep = ((masks >> np.uint16(played)) & np.uint16(1)).astype(bool)
        masks = masks[keep]
        queues = queues[keep]
        if len(masks) == 0:
            failure_action_index = action_index
            # A unique-looking state is not a usable label if the observed play
            # is illegal in every surviving candidate.
            hand_masks[-1] = None
            next_cards[-1] = None
            remaining = len(card_indices) - action_index - 1
            hand_masks.extend([None] * remaining)
            next_cards.extend([None] * remaining)
            break
        incoming = queues[:, 0].astype(np.uint16)
        masks = (
            masks & np.uint16(~(1 << played) & 0xFFFF)
        ) | (np.uint16(1) << incoming)
        queues = np.concatenate(
            (
                queues[:, 1:],
                np.full((len(queues), 1), played, dtype=np.uint8),
            ),
            axis=1,
        )
        # Different initial states commonly converge to one identical current
        # state after four plays. Collapse it so long replays become O(actions).
        if (
            len(masks) > 1
            and np.all(masks == masks[0])
            and np.all(queues == queues[0])
        ):
            masks = masks[:1]
            queues = queues[:1]

    exact = [
        index for index, (hand, next_card) in enumerate(zip(hand_masks, next_cards))
        if hand is not None and next_card is not None
    ]
    return {
        "cycle_valid": failure_action_index is None,
        "failure_action_index": failure_action_index,
        "first_exact_action_index": exact[0] if exact else None,
        "exact_action_count": len(exact),
        "hand_masks_before": hand_masks,
        "next_card_indices_before": next_cards,
    }


def actor_coordinates(event: dict[str, Any], side: str) -> tuple[int, int]:
    x = int(event["x"])
    y = int(event["y"])
    if side == "opponent":
        x = 17_999 - x
        y = 31_999 - y
    return max(0, min(17_999, x)), max(0, min(31_999, y))


def ability_count(value: dict[str, Any], side: str) -> int:
    raw = (
        (((value.get("elixir_stats") or {}).get(side) or {}).get("Ability") or {})
        .get("count")
    )
    return int(raw or 0)


def recorded_ability_events(value: dict[str, Any], side: str) -> list[dict[str, Any]]:
    events = value.get("ability_plays") or []
    return [event for event in events if event.get("side") == side]


def process_battle(record: dict[str, Any]) -> list[dict[str, Any]]:
    source = Path(str(record["source_path"])).resolve(strict=True)
    value = loads(source.read_bytes())
    battle_tag = str(value.get("battle_tag") or "")
    if battle_tag != str(record["battle_tag"]):
        raise ValueError(f"battle tag mismatch: {source}")
    plays = value.get("card_plays") or []
    counts = value.get("card_counts") or {}
    output: list[dict[str, Any]] = []
    source_schema_version = int(value.get("schema_version") or 1)
    for side in ("team", "opponent"):
        deck = list((counts.get(side) or {}).keys())
        events = [
            (int(event.get("marker_index", event_index)), event)
            for event_index, event in enumerate(plays)
            if event.get("side") == side
        ]
        if len(deck) != 8:
            raise ValueError(f"{battle_tag} {side}: expected eight base cards")
        card_index = {card: index for index, card in enumerate(deck)}
        indices = [card_index[str(event["card"])] for _, event in events]
        solved = solve_cycle(indices)
        actor_xy = [actor_coordinates(event, side) for _, event in events]
        exact_source_deck = value.get(f"{side}_deck") or []
        variants = {base_card(card): str(card) for card in exact_source_deck}
        exact_deck = [variants.get(base_card(card)) for card in deck]
        exact_forms = len(exact_source_deck) == 8 and all(exact_deck)
        player: dict[str, Any] = {}
        rounds = value.get("rounds") or []
        if len(rounds) == 1 and isinstance(rounds[0], dict):
            players = rounds[0].get(side) or []
            if len(players) == 1 and isinstance(players[0], dict):
                player = players[0]
        level_map = player.get("card_levels") or {}
        card_levels = [level_map.get(card) for card in exact_deck] if exact_forms else []
        exact_levels = exact_forms and all(isinstance(level, int) for level in card_levels)
        tower_troop = player.get("tower_troop")
        ability_events = recorded_ability_events(value, side)
        expected_abilities = ability_count(value, side)
        ability_log_complete = (
            source_schema_version >= 3
            and len(ability_events) == expected_abilities
        ) or expected_abilities == 0
        output.append({
            "schema_version": 1,
            "kind": "expert_base_cycle_side_v1",
            "battle_tag": battle_tag,
            "side": side,
            "source_path": str(source),
            "source_schema_version": source_schema_version,
            "base_deck": deck,
            "base_deck_complete": True,
            "exact_card_forms": exact_deck if exact_forms else None,
            "exact_forms": exact_forms,
            "card_levels": card_levels if exact_levels else None,
            "exact_levels": exact_levels,
            "tower_troop": tower_troop,
            "exact_tower_troop": bool(tower_troop),
            "ability_events_complete": ability_log_complete,
            "missing_ability_event_count": max(
                0, expected_abilities - len(ability_events)
            ),
            "ability_ticks": [int(event["time_raw"]) for event in ability_events],
            "ability_event_indices": [
                int(event.get("marker_index", index))
                for index, event in enumerate(ability_events)
            ],
            "action_count": len(events),
            "event_indices": [event_index for event_index, _ in events],
            "ticks": [int(event["time_raw"]) for _, event in events],
            "card_indices": indices,
            "actor_x": [xy[0] for xy in actor_xy],
            "actor_y": [xy[1] for xy in actor_xy],
            **solved,
        })
    return output


def schema_records(path: Path, schema_version: int, limit: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for raw in handle:
            value = loads(raw)
            if int(value.get("schema_version") or 1) != schema_version:
                continue
            records.append(value)
            if limit and len(records) >= limit:
                break
    return records


def run(args: argparse.Namespace) -> dict[str, Any]:
    union = args.union_manifest.resolve(strict=True)
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    building = output.with_name(output.name + ".building")
    if building.exists():
        raise FileExistsError(f"staging output already exists: {building}")
    records = schema_records(union, args.schema_version, args.limit)
    if args.expected_count and len(records) != args.expected_count:
        raise RuntimeError(
            f"expected {args.expected_count} schema-v{args.schema_version} battles, "
            f"got {len(records)}"
        )

    started = time.perf_counter()
    counters: Counter[str] = Counter()
    building.mkdir(parents=True)
    sequences_path = building / "side-sequences.jsonl"
    valid_path = building / "valid-sides.jsonl"
    invalid_path = building / "invalid-sides.jsonl"
    try:
        with (
            sequences_path.open("wb") as all_handle,
            valid_path.open("wb") as valid_handle,
            invalid_path.open("wb") as invalid_handle,
            ThreadPoolExecutor(max_workers=args.workers) as executor,
        ):
            for battle_index, sides in enumerate(
                executor.map(process_battle, records), 1
            ):
                counters["battles"] += 1
                battle_valid_sides = 0
                for side_record in sides:
                    encoded = dump_line(side_record)
                    all_handle.write(encoded)
                    counters["sides"] += 1
                    counters["actions"] += int(side_record["action_count"])
                    counters["reconstructed_exact_actions"] += int(
                        side_record["exact_action_count"]
                    )
                    counters["missing_abilities"] += int(
                        side_record["missing_ability_event_count"]
                    )
                    if side_record["cycle_valid"]:
                        valid_handle.write(encoded)
                        counters["valid_sides"] += 1
                        counters["usable_actions"] += int(side_record["action_count"])
                        counters["usable_exact_actions"] += int(
                            side_record["exact_action_count"]
                        )
                        battle_valid_sides += 1
                    else:
                        invalid_handle.write(encoded)
                        counters["invalid_sides"] += 1
                counters[f"battles_with_{battle_valid_sides}_valid_sides"] += 1
                if args.progress_every and battle_index % args.progress_every == 0:
                    print(
                        f"processed {battle_index}/{len(records)} battles",
                        flush=True,
                    )
        summary = {
            "schema_version": 1,
            "kind": "expert_base_cycle_upgrade_summary_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "union_manifest": str(union),
            "union_manifest_sha256": sha256(union),
            "source_schema_version": args.schema_version,
            **counters,
            "exact_action_fraction": (
                counters["usable_exact_actions"] / counters["usable_actions"]
                if counters["usable_actions"] else 0.0
            ),
            "workers": args.workers,
            "elapsed_seconds": time.perf_counter() - started,
            "source_files_modified": False,
        }
        (building / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(building, output)
        return summary
    except Exception:
        if building.exists():
            shutil.rmtree(building)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--union-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--expected-count", type=int, default=0)
    parser.add_argument("--schema-version", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=5_000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 1 <= args.workers <= 64:
        raise ValueError("workers must be in 1..64")
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
