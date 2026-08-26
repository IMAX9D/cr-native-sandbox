"""Pure helpers for auditing expert replay elixir/tick disagreements.

This module does not alter replay timing.  It only turns native rejection
evidence and source actions into an auditable diagnostic table.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from native_core.card_catalog import catalog

from .native_replay_plan import BattlePlan


@dataclass(frozen=True)
class Code13Case:
    battle_tag: str
    source_path: str
    tick: int
    source_event_index: int
    side: int
    base_token: str
    elixir_raw: int
    cost_raw: int
    deficit_raw: int
    hand_deck_indices: tuple[int, ...]
    next_deck_index: int
    refill_timer: int
    resolved_data_id: int

    def json(self) -> dict[str, Any]:
        return asdict(self)


def packed_card_cost_raw(native_result: Mapping[str, Any]) -> int:
    """Decode the cost nibble used by the authoritative native command.

    New diagnostic Hosts expose the same value as
    ``resource_before.card_cost``.  Older v3/v6 pilot output still contains
    ``packed_selection``; its high nibble is what ``D8D7C1`` compares against
    ``player+0x2f8``.
    """
    resource = native_result.get("resource_before")
    if isinstance(resource, Mapping) and isinstance(resource.get("card_cost"), int):
        return int(resource["card_cost"]) * 10_000
    packed = int(native_result["packed_selection"]) & 0xFFFFFFFF
    return ((packed >> 28) & 0xF) * 10_000


def code13_cases(result_rows: Iterable[Mapping[str, Any]]) -> list[Code13Case]:
    cases: list[Code13Case] = []
    for row in result_rows:
        rejection = row.get("first_rejection")
        if not isinstance(rejection, Mapping):
            continue
        for event in rejection.get("events", []):
            if not isinstance(event, Mapping):
                continue
            native = event.get("native_result")
            if not isinstance(native, Mapping) or int(native.get("result_code", -1)) != 13:
                continue
            cost_raw = packed_card_cost_raw(native)
            elixir_raw = int(event["pre_action_elixir_raw"])
            cases.append(Code13Case(
                battle_tag=str(row["battle_tag"]),
                source_path=str(row["source_path"]),
                tick=int(rejection["tick"]),
                source_event_index=int(event["source_event_index"]),
                side=int(event["side"]),
                base_token=str(event["base_token"]),
                elixir_raw=elixir_raw,
                cost_raw=cost_raw,
                deficit_raw=max(0, cost_raw - elixir_raw),
                hand_deck_indices=tuple(
                    int(value) for value in event["pre_action_hand_deck_indices"]
                ),
                next_deck_index=int(event.get("pre_action_next_deck_index", -1)),
                refill_timer=int(event.get("pre_action_refill_timer", -1)),
                resolved_data_id=int(native.get("resolved_data_id", -1)),
            ))
    return sorted(cases, key=lambda item: (item.tick, item.battle_tag))


def load_result_rows(root: Path) -> list[dict[str, Any]]:
    preferred = root / "results.jsonl"
    paths = [preferred] if preferred.is_file() else sorted(root.glob("worker-*.results.jsonl"))
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8-sig") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def elixir_multiplier(tick: int) -> int:
    if tick < 2400:
        return 1
    if tick < 4800:
        return 2
    return 3


def lower_bound_regen_ticks(deficit_raw: int, tick: int) -> int:
    """Lower bound assuming no cap/leak and the frozen default timeline.

    Default x1 fills 100000 raw in 560 ticks, x2 in 280, and x3 in
    560/3 ticks.  The real accumulator is integer/fixed-point; this result is
    only a bound used to size an A/B search, never a replay correction.
    """
    if deficit_raw <= 0:
        return 0
    multiplier = elixir_multiplier(tick)
    return (deficit_raw * 560 + 100_000 * multiplier - 1) // (
        100_000 * multiplier
    )


def source_prefix(plan: BattlePlan, case: Code13Case) -> list[dict[str, Any]]:
    """Describe every same-side deployment through the rejected marker."""
    rows: list[dict[str, Any]] = []
    previous_tick: int | None = None
    cumulative_cost_raw = 0
    for action in plan.actions:
        if action.side != case.side:
            continue
        spec = plan.sides[action.side].deck[action.logical_card_index]
        cost_raw = int(catalog()[spec.card_id]["elixir"]) * 10_000
        cumulative_cost_raw += cost_raw
        rows.append({
            "source_event_index": int(action.source_event_index),
            "tick": int(action.tick),
            "delta_ticks": None if previous_tick is None else int(action.tick - previous_tick),
            "card": action.base_token,
            "source_deck_token": spec.source_token,
            "form_flags": int(spec.form_flags),
            "catalog_cost_raw": cost_raw,
            "cumulative_catalog_cost_raw": cumulative_cost_raw,
            "is_target": int(action.source_event_index) == case.source_event_index,
        })
        previous_tick = action.tick
        if int(action.source_event_index) == case.source_event_index:
            break
    return rows


def source_resource_flags(plan: BattlePlan) -> list[str]:
    """Flag cards that can change elixir outside passive regeneration."""
    known = {"elixir-golem", "elixir-collector"}
    return sorted({
        spec.base_token
        for side in plan.sides
        for spec in side.deck
        if spec.base_token in known
    })
