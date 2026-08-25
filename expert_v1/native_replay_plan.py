"""Compile scraped expert battles into fail-closed native replay plans.

This module deliberately stops before calling ``libg``.  It separates facts
present in the RoyaleAPI artifact from assumptions needed to synthesize a
native battle.  In particular, a compatible card cycle is enough to make an
action stream executable, but it does *not* recover the original libg RNG
seed, build, tower level, game-mode id, or omitted ability button presses.

The resulting plan is consumed by :mod:`expert_v1.native_replay_runner` after
one native shuffle-layout calibration per Worker.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from native_core.card_catalog import catalog
from native_core.decks import resolve_card

from .upgrade_base_cycles import INITIAL_MASKS, INITIAL_QUEUES


SIDE_NAMES = ("team", "opponent")
TOWER_PRINCESS_ID = 159_000_000
DEFAULT_NATIVE_SEED = 424_242

# The frozen DataTables retain several pre-release/internal names.  Keep this
# translation local to expert ingestion: changing the public sandbox resolver
# would be an unrelated compatibility change.  IDs are from the same frozen
# ``live_card_catalog.json`` used by libg acceptance.
ROYALEAPI_CARD_ALIASES = {
    "archers": 26000001,
    "guards": 26000025,
    "ice-spirit": 26000030,
    "fire-spirit": 26000031,
    "sparky": 26000033,
    "lumberjack": 26000035,
    "ice-golem": 26000038,
    "dart-goblin": 26000040,
    "elite-barbarians": 26000043,
    "executioner": 26000045,
    "bandit": 26000046,
    "night-witch": 26000048,
    "royal-ghost": 26000050,
    "zappies": 26000052,
    "cannon-cart": 26000054,
    "skeleton-barrel": 26000056,
    "flying-machine": 26000057,
    "magic-archer": 26000062,
    "mother-witch": 26000083,
    "rune-giant": 26000101,
    "furnace": 27000010,
    "the-log": 28000011,
    "barbarian-barrel": 28000015,
    "heal-spirit": 28000016,
    "giant-snowball": 28000017,
    "void": 28000023,
    "spirit-empress": 28000025,
}


class ReplayPlanError(ValueError):
    """The source cannot be converted into an internally consistent plan."""


@dataclass(frozen=True)
class CardSpec:
    source_token: str
    base_token: str
    card_id: int
    level: int | None
    form_flags: int
    runtime_form_supported: bool


@dataclass(frozen=True)
class CycleCandidate:
    initial_hand: tuple[int, int, int, int]
    initial_queue: tuple[int, int, int, int]
    compatible_initial_state_count: int


@dataclass(frozen=True)
class ExpertAction:
    tick: int
    side: int
    logical_card_index: int
    base_token: str
    x: int
    y: int
    source_event_index: int


@dataclass(frozen=True)
class SidePlan:
    side: int
    source_side: str
    deck: tuple[CardSpec, ...]
    cycle: CycleCandidate
    action_count: int
    missing_ability_event_count: int
    tower_troop: str | None


@dataclass(frozen=True)
class BattlePlan:
    schema_version: int
    kind: str
    battle_tag: str
    source_schema_version: int
    duration_ticks: int
    sides: tuple[SidePlan, SidePlan]
    actions: tuple[ExpertAction, ...]
    replay_tier: str
    native_replay_ready: bool
    original_state_exact: bool
    state_provenance: str
    action_provenance: str
    hand_provenance: str
    ability_provenance: str
    terminal_provenance: str
    terminal_crowns: tuple[int, int] | None
    limitations: tuple[str, ...]

    def json(self) -> dict[str, Any]:
        return asdict(self)


def split_card_token(token: str) -> tuple[str, int]:
    """Return the base RoyaleAPI token and native ``el`` form flags."""
    value = str(token).strip().lower()
    flags = 0
    # Be tolerant of future serializers that emit both suffixes.
    changed = True
    while changed:
        changed = False
        if value.endswith("-ev1"):
            value = value[:-4]
            flags |= 1
            changed = True
        if value.endswith("-hero"):
            value = value[:-5]
            flags |= 2
            changed = True
    if not value:
        raise ReplayPlanError(f"empty card token after form parsing: {token!r}")
    return value, flags


def card_spec(token: str, level: int | None) -> CardSpec:
    base, flags = split_card_token(token)
    card_id = ROYALEAPI_CARD_ALIASES.get(base)
    if card_id is None:
        try:
            card_id = resolve_card(base)
        except KeyError as error:
            raise ReplayPlanError(
                f"card is absent from frozen libg catalog: {token}"
            ) from error
    row = catalog()[card_id]
    runtime_form_supported = not (
        (flags & 1 and row.get("evolution_form_id") is None)
        or (flags & 2 and row.get("hero_form_id") is None)
    )
    if level is not None and not 1 <= int(level) <= 16:
        raise ReplayPlanError(f"invalid card level {level!r} for {token}")
    return CardSpec(
        source_token=str(token), base_token=base, card_id=card_id,
        level=None if level is None else int(level), form_flags=flags,
        runtime_form_supported=runtime_form_supported,
    )


def compatible_cycle(played: Sequence[int]) -> CycleCandidate:
    """Choose one initial hand/queue compatible with the complete play stream.

    Multiple initial states often converge after four plays.  Choosing one of
    them makes a deterministic *synthetic* replay possible; the count remains
    in the plan so it cannot be mistaken for recovered source truth.
    """
    masks = INITIAL_MASKS.copy()
    queues = INITIAL_QUEUES.copy()
    origin_masks = masks.copy()
    origin_queues = queues.copy()
    for action_index, raw_index in enumerate(played):
        index = int(raw_index)
        if not 0 <= index < 8:
            raise ReplayPlanError(f"invalid logical card index at action {action_index}")
        keep = ((masks >> np.uint16(index)) & np.uint16(1)).astype(bool)
        masks, queues = masks[keep], queues[keep]
        origin_masks, origin_queues = origin_masks[keep], origin_queues[keep]
        if len(masks) == 0:
            raise ReplayPlanError(
                f"observed card sequence violates the eight-card cycle at action "
                f"{action_index}"
            )
        incoming = queues[:, 0].astype(np.uint16)
        masks = (
            masks & np.uint16(~(1 << index) & 0xFFFF)
        ) | (np.uint16(1) << incoming)
        queues = np.concatenate(
            (queues[:, 1:], np.full((len(queues), 1), index, dtype=np.uint8)),
            axis=1,
        )
    first_mask = int(origin_masks[0])
    hand = tuple(index for index in range(8) if first_mask & (1 << index))
    queue = tuple(int(value) for value in origin_queues[0])
    if len(hand) != 4 or len(queue) != 4:
        raise AssertionError("cycle candidate is not a 4+4 partition")
    return CycleCandidate(
        initial_hand=hand,  # type: ignore[arg-type]
        initial_queue=queue,  # type: ignore[arg-type]
        compatible_initial_state_count=int(len(origin_masks)),
    )


def _ability_count(value: Mapping[str, Any], source_side: str) -> int:
    try:
        raw = value["elixir_stats"][source_side]["Ability"]["count"]
    except (KeyError, TypeError):
        return 0
    return int(raw or 0)


def _ability_provenance(value: Mapping[str, Any], counts: Sequence[int]) -> str:
    events = value.get("ability_plays")
    if not any(counts):
        return "source_reports_zero"
    if not isinstance(events, list):
        return "missing_observed_count_only"
    observed = [
        sum(1 for event in events if isinstance(event, Mapping) and event.get("side") == side)
        for side in SIDE_NAMES
    ]
    if observed != list(counts):
        return "observed_tick_count_mismatch"
    if all(
        event.get("ability_id") not in (None, "")
        for event in events if isinstance(event, Mapping)
    ):
        return "observed_ticks_with_source_identity_unmapped"
    return "observed_ticks_identity_unresolved"


def _schema_two_player(
    value: Mapping[str, Any], source_side: str
) -> Mapping[str, Any]:
    rounds = value.get("rounds")
    if not isinstance(rounds, list) or len(rounds) != 1:
        raise ReplayPlanError("schema-v2 battle must contain exactly one round")
    players = rounds[0].get(source_side) if isinstance(rounds[0], Mapping) else None
    if not isinstance(players, list) or len(players) != 1:
        raise ReplayPlanError(f"{source_side} must contain exactly one player")
    player = players[0]
    if not isinstance(player, Mapping) or player.get("complete") is not True:
        raise ReplayPlanError(f"{source_side} deck metadata is incomplete")
    return player


def _side_deck(
    value: Mapping[str, Any], source_side: str, source_schema: int
) -> tuple[tuple[CardSpec, ...], str | None]:
    if source_schema >= 2:
        player = _schema_two_player(value, source_side)
        tokens = player.get("full_deck")
        levels = player.get("card_levels")
        if not isinstance(tokens, list) or len(tokens) != 8 or not isinstance(levels, Mapping):
            raise ReplayPlanError(f"{source_side} requires eight cards and levels")
        result = tuple(card_spec(str(token), int(levels[str(token)])) for token in tokens)
        tower = str(player.get("tower_troop") or "") or None
    else:
        counts = value.get("card_counts")
        side_counts = counts.get(source_side) if isinstance(counts, Mapping) else None
        if not isinstance(side_counts, Mapping) or len(side_counts) != 8:
            raise ReplayPlanError(f"{source_side} does not expose eight played cards")
        result = tuple(card_spec(str(token), None) for token in side_counts)
        tower = None
    bases = [item.base_token for item in result]
    if len(set(bases)) != 8:
        raise ReplayPlanError(f"{source_side} deck has duplicate base-card identities")
    return result, tower


def compile_battle(
    value: Mapping[str, Any],
    *,
    terminal_crowns: Sequence[int] | None = None,
) -> BattlePlan:
    battle_tag = str(value.get("battle_tag") or "")
    if not battle_tag:
        raise ReplayPlanError("battle tag is missing")
    crowns: tuple[int, int] | None = None
    if terminal_crowns is not None:
        if (
            len(terminal_crowns) != 2
            or any(
                isinstance(value, bool) or not 0 <= int(value) <= 3
                for value in terminal_crowns
            )
        ):
            raise ReplayPlanError("terminal crowns must contain two values in 0..3")
        crowns = (int(terminal_crowns[0]), int(terminal_crowns[1]))
    source_schema = int(value.get("schema_version") or 1)
    if value.get("draft") is True:
        raise ReplayPlanError("draft battles are not supported")
    duration_seconds = value.get("duration_seconds")
    if not isinstance(duration_seconds, (int, float)) or not 1 <= duration_seconds <= 360:
        raise ReplayPlanError("battle duration is missing or outside 1..360 seconds")
    duration_ticks = int(round(float(duration_seconds) * 20.0))
    plays = value.get("card_plays")
    if not isinstance(plays, list) or not plays:
        raise ReplayPlanError("battle has no card play events")

    side_decks: list[tuple[CardSpec, ...]] = []
    tower_troops: list[str | None] = []
    logical_indices: list[dict[str, int]] = []
    for source_side in SIDE_NAMES:
        deck, tower = _side_deck(value, source_side, source_schema)
        side_decks.append(deck)
        tower_troops.append(tower)
        logical_indices.append({item.base_token: index for index, item in enumerate(deck)})

    actions: list[ExpertAction] = []
    per_side_indices: list[list[int]] = [[], []]
    same_side_ticks: set[tuple[int, int]] = set()
    for event_index, event in enumerate(plays):
        if not isinstance(event, Mapping):
            raise ReplayPlanError(f"event {event_index} is not an object")
        source_side = str(event.get("side"))
        if source_side not in SIDE_NAMES:
            raise ReplayPlanError(f"event {event_index} has invalid side")
        side = SIDE_NAMES.index(source_side)
        base_token, _ignored_form = split_card_token(str(event.get("card") or ""))
        if base_token not in logical_indices[side]:
            raise ReplayPlanError(
                f"event {event_index} card {base_token!r} is absent from {source_side} deck"
            )
        tick = int(event.get("time_raw", -1))
        if tick < 0 or tick > duration_ticks + 20:
            raise ReplayPlanError(f"event {event_index} tick is outside battle duration")
        tick_key = (tick, side)
        if tick_key in same_side_ticks:
            raise ReplayPlanError(
                f"multiple actions for side {side} at native tick {tick}"
            )
        same_side_ticks.add(tick_key)
        logical = logical_indices[side][base_token]
        per_side_indices[side].append(logical)
        try:
            x, y = int(event["x"]), int(event["y"])
        except (KeyError, TypeError, ValueError) as error:
            raise ReplayPlanError(f"event {event_index} coordinates are missing") from error
        if not 0 <= x <= 17_999 or not 0 <= y <= 31_999:
            raise ReplayPlanError(f"event {event_index} coordinates are outside the arena")
        actions.append(ExpertAction(
            tick=tick, side=side, logical_card_index=logical,
            base_token=base_token, x=x, y=y, source_event_index=event_index,
        ))
    if any(
        actions[index].tick > actions[index + 1].tick
        for index in range(len(actions) - 1)
    ):
        raise ReplayPlanError("card events are not sorted by tick")

    cycles = [compatible_cycle(indices) for indices in per_side_indices]
    ability_counts = [_ability_count(value, source_side) for source_side in SIDE_NAMES]
    side_plans = tuple(
        SidePlan(
            side=side, source_side=SIDE_NAMES[side], deck=side_decks[side],
            cycle=cycles[side], action_count=len(per_side_indices[side]),
            missing_ability_event_count=ability_counts[side],
            tower_troop=tower_troops[side],
        )
        for side in range(2)
    )

    limitations = [
        "source_native_rng_seed_missing",
        "source_game_build_missing",
        "source_numeric_game_mode_missing",
        "source_initial_hand_not_observed",
        "source_tower_level_missing",
        "source_tick_state_anchors_missing",
    ]
    if source_schema < 2:
        limitations.extend((
            "card_forms_missing", "card_levels_missing", "tower_troops_missing",
        ))
    if any(ability_counts):
        limitations.append("ability_button_events_missing")
    if any(tower not in (None, "tower-princess") for tower in tower_troops):
        limitations.append("non_princess_tower_runtime_mapping_missing")
    unsupported_forms = sorted({
        spec.source_token
        for deck in side_decks for spec in deck
        if not spec.runtime_form_supported
    })
    if unsupported_forms:
        limitations.append(
            "runtime_card_forms_unsupported:" + ",".join(unsupported_forms)
        )
    if any(cycle.compatible_initial_state_count != 1 for cycle in cycles):
        limitations.append("multiple_compatible_initial_cycles")
    limitations = list(dict.fromkeys(limitations))

    metadata_exact = source_schema >= 2
    replay_ready = (
        metadata_exact
        and not any(ability_counts)
        and all(tower == "tower-princess" for tower in tower_troops)
        and not unsupported_forms
    )
    tier = (
        "native_replay_candidate"
        if replay_ready
        else "synthetic_base_cycle"
        if source_schema < 2 and not any(ability_counts)
        else "action_sequence_only"
    )
    return BattlePlan(
        schema_version=1,
        kind="expert_native_replay_plan_v1",
        battle_tag=battle_tag,
        source_schema_version=source_schema,
        duration_ticks=duration_ticks,
        sides=side_plans,  # type: ignore[arg-type]
        actions=tuple(actions),
        replay_tier=tier,
        native_replay_ready=replay_ready,
        # No current RoyaleAPI artifact has the seed/build/state anchors needed
        # to make this claim, even when all deployment actions are accepted.
        original_state_exact=False,
        state_provenance="native_generated_unanchored",
        action_provenance="observed_deployments",
        hand_provenance=(
            "inferred_cycle_unique"
            if all(cycle.compatible_initial_state_count == 1 for cycle in cycles)
            else "inferred_cycle_compatible_initial"
        ),
        ability_provenance=_ability_provenance(value, ability_counts),
        terminal_provenance=(
            "source_index_crowns" if crowns is not None else "unknown_no_source_anchor"
        ),
        terminal_crowns=crowns,
        limitations=tuple(limitations),
    )


def native_layout_order(player: Mapping[str, Any]) -> tuple[int, ...]:
    """Validate a calibrated native 4-card hand plus 4-card refill queue."""
    hand = tuple(int(value) for value in player.get("hand_deck_indices", ()))
    queue = tuple(int(value) for value in player.get("cycle_deck_indices", ()))
    if len(hand) != 4 or len(queue) != 4 or set(hand + queue) != set(range(8)):
        raise ReplayPlanError(
            f"native shuffle layout is not a complete 4+4 partition: {hand}, {queue}"
        )
    return hand + queue


def logical_to_native_mapping(
    side_plan: SidePlan, player: Mapping[str, Any]
) -> tuple[int, ...]:
    """Map logical source cards to replay deck slots for one calibrated seed."""
    native_order = native_layout_order(player)
    desired_order = side_plan.cycle.initial_hand + side_plan.cycle.initial_queue
    mapping = [-1] * 8
    for native_index, logical_index in zip(native_order, desired_order, strict=True):
        mapping[logical_index] = native_index
    if set(mapping) != set(range(8)):
        raise AssertionError("logical/native mapping is not bijective")
    return tuple(mapping)


def materialize_replay(
    plan: BattlePlan,
    template: Mapping[str, Any],
    calibrated_players: Sequence[Mapping[str, Any]],
    *,
    seed: int = DEFAULT_NATIVE_SEED,
    fallback_level: int = 11,
) -> tuple[dict[str, Any], tuple[tuple[int, ...], tuple[int, ...]]]:
    """Build a libg replay whose shuffled 4+4 layout matches the plan."""
    import copy

    if len(calibrated_players) != 2:
        raise ReplayPlanError("native calibration requires two player states")
    unsupported = sorted({
        spec.source_token
        for side in plan.sides for spec in side.deck
        if not spec.runtime_form_supported
    })
    if unsupported:
        raise ReplayPlanError(
            "cannot materialize forms absent from frozen libg: "
            + ", ".join(unsupported)
        )
    replay = copy.deepcopy(dict(template))
    replay["rndSeed"] = int(seed)
    battle = replay["battle"]
    mappings: list[tuple[int, ...]] = []
    for side, side_plan in enumerate(plan.sides):
        mapping = logical_to_native_mapping(side_plan, calibrated_players[side])
        mappings.append(mapping)
        native_deck: list[dict[str, int] | None] = [None] * 8
        for logical_index, spec in enumerate(side_plan.deck):
            row = {
                "d": int(spec.card_id),
                "l": int((spec.level or fallback_level) - 1),
            }
            if spec.form_flags:
                row["el"] = int(spec.form_flags)
            native_deck[mapping[logical_index]] = row
        if any(item is None for item in native_deck):
            raise AssertionError("materialized native deck has an empty slot")
        battle[f"deck{side}"]["sp"] = native_deck
        # Only Tower Princess is currently mapped/certified by this runtime.
        battle[f"deck{side}"]["sc"] = [{
            "d": TOWER_PRINCESS_ID,
            "l": int(max((spec.level or fallback_level) for spec in side_plan.deck) - 1),
            "t": 0,
            "c": 0,
        }]
    return replay, (mappings[0], mappings[1])


def grouped_actions(
    plan: BattlePlan, mappings: Sequence[Sequence[int]]
) -> Iterable[tuple[int, list[dict[str, int]]]]:
    """Yield canonical joint native actions grouped by source tick."""
    current_tick: int | None = None
    batch: list[dict[str, int]] = []
    for action in plan.actions:
        if current_tick is not None and action.tick != current_tick:
            yield current_tick, sorted(batch, key=lambda item: item["side"])
            batch = []
        current_tick = action.tick
        batch.append({
            "type": "play",
            "side": action.side,
            "deck_index": int(mappings[action.side][action.logical_card_index]),
            "x": action.x,
            "y": action.y,
            "source_event_index": action.source_event_index,
        })
    if current_tick is not None:
        yield current_tick, sorted(batch, key=lambda item: item["side"])
