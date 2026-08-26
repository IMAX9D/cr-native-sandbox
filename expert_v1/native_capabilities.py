"""Evidence-backed native mappings for expert teacher-forced replay.

Source ability markers contain only ``tick`` and ``side``.  Identity is
therefore resolved at that native tick from the live, legal entity set.  A
non-unique set is returned explicitly for branching; callers must never pick
an arbitrary entity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from native_core.card_catalog import catalog, observed_card


@dataclass(frozen=True)
class TowerTroopSpec:
    source_token: str
    support_card_id: int
    spawn_group: str
    runtime_probed: bool = True


TOWER_TROOPS: dict[str, TowerTroopSpec] = {
    "tower-princess": TowerTroopSpec(
        "tower-princess", 159_000_000, "King_PrincessTowers"
    ),
    "cannoneer": TowerTroopSpec(
        "cannoneer", 159_000_001, "King_CannonTowers"
    ),
    "dagger-duchess": TowerTroopSpec(
        "dagger-duchess", 159_000_002, "King_KnifeTowers"
    ),
    # Slot 159000003 is the hidden Goblin Queen support action.
    "royal-chef": TowerTroopSpec(
        "royal-chef", 159_000_004, "King_ChefTowers"
    ),
}


def tower_troop(token: str | None) -> TowerTroopSpec:
    normalized = str(token or "").strip().lower().replace("_", "-")
    try:
        return TOWER_TROOPS[normalized]
    except KeyError as error:
        raise ValueError(f"unmapped native tower troop: {token!r}") from error


@dataclass(frozen=True)
class AbilityCard:
    logical_card_index: int
    base_card_id: int
    native_form_id: int
    source_token: str
    ability_name: str
    mana_cost: int | None


def ability_cards(deck: Sequence[Any]) -> tuple[AbilityCard, ...]:
    """Return all button-bearing cards from a compiled ``CardSpec`` deck."""
    result: list[AbilityCard] = []
    rows = catalog()
    for logical_index, spec in enumerate(deck):
        row = rows[int(spec.card_id)]
        hero = bool(int(spec.form_flags) & 2)
        name = row.get("hero_active_ability" if hero else "active_ability")
        if not name:
            continue
        form_id = row.get("hero_form_id") if hero else int(spec.card_id)
        cost = row.get(
            "hero_ability_mana_cost" if hero else "active_ability_mana_cost"
        )
        result.append(AbilityCard(
            logical_card_index=logical_index,
            base_card_id=int(spec.card_id),
            native_form_id=int(form_id),
            source_token=str(spec.source_token),
            ability_name=str(name),
            mana_cost=None if cost is None else int(cost),
        ))
    return tuple(result)


@dataclass(frozen=True)
class AbilityResolution:
    status: str
    side: int
    tick: int
    candidate_entity_ids: tuple[int, ...]
    candidate_card_ids: tuple[int, ...]

    def json(self) -> dict[str, Any]:
        return asdict(self)


def resolve_live_ability(
    state: Mapping[str, Any], *, side: int, tick: int,
    allowed_cards: Sequence[AbilityCard] = (),
) -> AbilityResolution:
    """Resolve a source marker against native legal candidates.

    ``unique`` is executable. ``branch_required`` must be replayed as explicit
    alternatives by a caller capable of state cloning/reset. ``no_legal_*``
    is fail-closed and must not synthesize a button press.
    """
    if int(state.get("tick", -1)) != int(tick):
        raise ValueError("ability resolution requires the exact source tick")
    allowed_base = {item.base_card_id for item in allowed_cards}
    allowed_forms = {item.native_form_id for item in allowed_cards}
    candidates: list[tuple[int, int]] = []
    for entity in state.get("entities", []):
        if int(entity.get("side", -1)) != int(side):
            continue
        if int(entity.get("ability_slot", 0)) <= 0:
            continue
        if entity.get("ability_available") is not True:
            continue
        native_id = int(entity.get("native_card_id", entity.get("card_id", -1)))
        identity = observed_card(native_id)
        base_id = int(identity["base_card_id"])
        if allowed_cards and native_id not in allowed_forms and base_id not in allowed_base:
            continue
        entity_id = int(entity.get("entity_id", entity.get("category", -1)))
        if entity_id >= 0:
            candidates.append((entity_id, base_id))
    candidates.sort()
    if len(candidates) == 1:
        status = "unique"
    elif len(candidates) > 1:
        status = "branch_required"
    elif allowed_cards:
        status = "no_legal_matching_entity"
    else:
        status = "no_legal_ability_entity"
    return AbilityResolution(
        status=status, side=int(side), tick=int(tick),
        candidate_entity_ids=tuple(item[0] for item in candidates),
        candidate_card_ids=tuple(item[1] for item in candidates),
    )


def ability_log_tier(value: Mapping[str, Any]) -> str:
    """Classify source ability completeness without inventing old events."""
    counts = []
    for side in ("team", "opponent"):
        try:
            counts.append(int(value["elixir_stats"][side]["Ability"]["count"] or 0))
        except (KeyError, TypeError, ValueError):
            counts.append(0)
    events = value.get("ability_plays")
    if not any(counts):
        return "source_reports_zero"
    if not isinstance(events, list):
        return "count_only_missing_ticks"
    observed = [
        sum(1 for event in events if isinstance(event, Mapping) and event.get("side") == side)
        for side in ("team", "opponent")
    ]
    return (
        "observed_ticks_identity_runtime_resolved"
        if observed == counts else "observed_tick_count_mismatch"
    )
