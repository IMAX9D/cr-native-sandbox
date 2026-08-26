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
from functools import lru_cache
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from native_core.card_catalog import catalog
from native_core.decks import resolve_card

from .native_capabilities import ability_cards, ability_log_tier, tower_troop
from .upgrade_base_cycles import INITIAL_MASKS, INITIAL_QUEUES


SIDE_NAMES = ("team", "opponent")
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
    # These two public slugs only differ from the frozen internal name by a
    # word boundary.  ``resolve_card`` historically accepted them after
    # punctuation folding, but the authoritative ingest contract needs the
    # exact RoyaleAPI token so a crawler can validate without duplicating its
    # own card table.
    "wall-breakers": 26000058,
    "furnace": 27000010,
    "x-bow": 27000008,
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
    first_exact_action_index: int | None
    exact_action_count: int


@dataclass(frozen=True)
class ExpertAction:
    tick: int
    side: int
    logical_card_index: int
    base_token: str
    x: int
    y: int
    source_event_index: int
    source_marker_index: int
    side_action_index: int


@dataclass(frozen=True)
class ExpertAbilityEvent:
    """One observed native ability-button marker from a schema-v3 replay."""

    tick: int
    side: int
    source_event_index: int
    source_marker_index: int
    source_ability_id: str | None


@dataclass(frozen=True)
class CoordinateAudit:
    """How RoyaleAPI marker coordinates became native arena coordinates.

    RoyaleAPI's ``data-i`` is a per-marker coordinate inversion flag.  The
    replay viewer first rotates raw coordinates when it is ``1``.  Native
    side 0 uses the opposite arena orientation from that viewer, so the two
    rotations cancel for ``data-i == 1`` and only ``data-i == 0`` is rotated
    here.  Older captures did not retain the raw flag and remain an explicit,
    unverified compatibility fallback.
    """

    transform: str
    raw_data_i_events: int
    data_i_zero_events: int
    data_i_one_events: int
    legacy_xy_fallback_events: int
    data_i_values: tuple[int, ...]


@dataclass(frozen=True)
class FinalTowerHp:
    """Authoritative source terminal HP without an invented lane mapping.

    RoyaleAPI's list popup exposes two Princess Tower slots as ``princess0``
    and ``princess1``.  Their mapping to native arena X coordinates has not
    been proved, so the plan deliberately preserves the source slots and the
    runner compares them as a multiset.
    """

    king: int
    princess0: int
    princess1: int
    total: int
    provenance: str
    slot_mapping_provenance: str


@dataclass(frozen=True)
class SidePlan:
    side: int
    source_side: str
    deck: tuple[CardSpec, ...]
    cycle: CycleCandidate
    action_count: int
    observed_ability_event_count: int
    missing_ability_event_count: int
    tower_troop: str | None
    tower_troop_level: int | None
    tower_troop_level_provenance: str
    king_tower_level: int | None
    king_tower_level_provenance: str
    final_tower_hp: FinalTowerHp | None


@dataclass(frozen=True)
class BattlePlan:
    schema_version: int
    kind: str
    battle_tag: str
    source_schema_version: int
    numeric_game_mode_id: int | None
    numeric_game_mode_provenance: str
    native_execution_game_mode_id: int | None
    native_execution_game_mode_provenance: str
    battle_index: int | None
    battle_index_provenance: str
    authoritative_contract_game_version: str | None
    authoritative_contract_sha256: str | None
    authoritative_contract_file_sha256: str | None
    authoritative_contract_provenance: str
    duration_ticks: int
    sides: tuple[SidePlan, SidePlan]
    actions: tuple[ExpertAction, ...]
    ability_events: tuple[ExpertAbilityEvent, ...]
    ability_log_tier: str
    replay_tier: str
    native_replay_ready: bool
    original_state_exact: bool
    state_provenance: str
    action_provenance: str
    coordinate_provenance: str
    coordinate_audit: CoordinateAudit
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


def _native_event_coordinates(
    event: Mapping[str, Any], *, event_index: int,
) -> tuple[int, int, int | None]:
    """Return ``(native_x, native_y, data_i)`` from one replay marker.

    Frozen schema-v3 sources retain the authoritative marker fields.  Do not
    trust their historical derived ``x/y`` values: the old crawler always
    rotated those values and therefore mirrored every ``data-i == 1`` battle.
    Schema-v1/v2 records may lack one or all raw fields; only those records use
    the old ``x/y`` compatibility path and are surfaced in ``CoordinateAudit``.
    """

    raw_x = event.get("x_raw")
    raw_y = event.get("y_raw")
    data_i = event.get("data_i")
    has_exact_marker = (
        isinstance(raw_x, int) and not isinstance(raw_x, bool)
        and isinstance(raw_y, int) and not isinstance(raw_y, bool)
        and isinstance(data_i, int) and not isinstance(data_i, bool)
    )
    if has_exact_marker:
        if data_i not in (0, 1):
            raise ReplayPlanError(
                f"event {event_index} data_i must be 0 or 1"
            )
        if not 0 <= raw_x <= 18_000 or not 0 <= raw_y <= 32_000:
            raise ReplayPlanError(
                f"event {event_index} raw coordinates are outside the arena"
            )
        if data_i == 0:
            x, y = 18_000 - raw_x, 32_000 - raw_y
        else:
            x, y = raw_x, raw_y
        marker_flag: int | None = data_i
    else:
        try:
            x, y = int(event["x"]), int(event["y"])
        except (KeyError, TypeError, ValueError) as error:
            raise ReplayPlanError(
                f"event {event_index} coordinates are missing"
            ) from error
        marker_flag = None
    if not 0 <= x <= 17_999 or not 0 <= y <= 31_999:
        raise ReplayPlanError(
            f"event {event_index} coordinates are outside the arena"
        )
    return x, y, marker_flag


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
    first_exact_action_index: int | None = None
    exact_action_count = 0
    for action_index, raw_index in enumerate(played):
        index = int(raw_index)
        if not 0 <= index < 8:
            raise ReplayPlanError(f"invalid logical card index at action {action_index}")
        hand_unique = bool(len(masks) and np.all(masks == masks[0]))
        next_unique = bool(len(queues) and np.all(queues[:, 0] == queues[0, 0]))
        if hand_unique and next_unique:
            if first_exact_action_index is None:
                first_exact_action_index = action_index
            exact_action_count += 1
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
        first_exact_action_index=first_exact_action_index,
        exact_action_count=exact_action_count,
    )


def _ability_count(value: Mapping[str, Any], source_side: str) -> int:
    try:
        raw = value["elixir_stats"][source_side]["Ability"]["count"]
    except (KeyError, TypeError):
        return 0
    return int(raw or 0)


def _ability_provenance(value: Mapping[str, Any], counts: Sequence[int]) -> str:
    tier = ability_log_tier(value)
    if tier == "observed_ticks_identity_runtime_resolved":
        return "observed_ticks_identity_runtime_resolved"
    return tier


def _compile_ability_events(
    value: Mapping[str, Any], *, duration_ticks: int,
) -> tuple[ExpertAbilityEvent, ...]:
    raw_events = value.get("ability_plays")
    if not isinstance(raw_events, list):
        return ()
    result: list[ExpertAbilityEvent] = []
    marker_indices: set[int] = set()
    previous_key: tuple[int, int] | None = None
    for source_event_index, raw in enumerate(raw_events):
        if not isinstance(raw, Mapping):
            raise ReplayPlanError(
                f"ability event {source_event_index} is not an object"
            )
        source_side = str(raw.get("side"))
        if source_side not in SIDE_NAMES:
            raise ReplayPlanError(
                f"ability event {source_event_index} has invalid side"
            )
        tick = int(raw.get("time_raw", -1))
        if tick < 0 or tick > duration_ticks + 20:
            raise ReplayPlanError(
                f"ability event {source_event_index} tick is outside battle duration"
            )
        marker_index = int(raw.get("marker_index", source_event_index))
        if marker_index < 0 or marker_index in marker_indices:
            raise ReplayPlanError(
                f"ability event {source_event_index} has invalid/duplicate marker index"
            )
        marker_indices.add(marker_index)
        key = (tick, marker_index)
        if previous_key is not None and key < previous_key:
            raise ReplayPlanError("ability events are not sorted by tick/marker index")
        previous_key = key
        raw_identity = raw.get("ability_id")
        result.append(ExpertAbilityEvent(
            tick=tick,
            side=SIDE_NAMES.index(source_side),
            source_event_index=source_event_index,
            source_marker_index=marker_index,
            source_ability_id=(
                None if raw_identity in (None, "") else str(raw_identity)
            ),
        ))
    return tuple(result)


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


def _strict_int(value: Any, *, minimum: int = 0) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= minimum
    )


@lru_cache(maxsize=1)
def _default_native_ingest_contract() -> Any:
    # Lazy import avoids the intentional dependency in the opposite direction:
    # the contract generator derives RoyaleAPI aliases from this module.
    from .native_ingest_contract import load_native_ingest_contract

    return load_native_ingest_contract()


def _schema_five_terminal_hp(
    value: Mapping[str, Any], source_side: str,
) -> FinalTowerHp:
    root = value.get("final_tower_hp")
    if not isinstance(root, Mapping):
        raise ReplayPlanError("schema-v5 final_tower_hp is missing")
    if root.get("provenance") != "list_hp_both_popup":
        raise ReplayPlanError("schema-v5 final tower HP provenance is invalid")
    if root.get("slot_mapping_provenance") != "source_slots_unmapped":
        raise ReplayPlanError(
            "schema-v5 Princess Tower slot mapping provenance is invalid"
        )
    raw_mapping = root.get("raw_princess_mapping")
    if not isinstance(raw_mapping, Mapping) or set(raw_mapping) != {
        "princess0", "princess1",
    }:
        raise ReplayPlanError("schema-v5 raw Princess Tower slot mapping is missing")
    side = root.get(source_side)
    if not isinstance(side, Mapping):
        raise ReplayPlanError(f"schema-v5 {source_side} final tower HP is missing")
    keys = ("king", "princess0", "princess1", "total")
    if any(not _strict_int(side.get(key)) for key in keys):
        raise ReplayPlanError(f"schema-v5 {source_side} final tower HP is invalid")
    if int(side["total"]) != sum(int(side[key]) for key in keys[:3]):
        raise ReplayPlanError(
            f"schema-v5 {source_side} final tower HP total does not match"
        )
    return FinalTowerHp(
        king=int(side["king"]),
        princess0=int(side["princess0"]),
        princess1=int(side["princess1"]),
        total=int(side["total"]),
        provenance="list_hp_both_popup",
        slot_mapping_provenance="source_slots_unmapped",
    )


def _schema_five_metadata(
    value: Mapping[str, Any], native_ingest_contract: Any | None,
) -> dict[str, Any]:
    """Validate the complete authoritative crawler/native contract boundary."""
    contract = (
        _default_native_ingest_contract()
        if native_ingest_contract is None
        else native_ingest_contract
    )
    stamp = value.get("authoritative_native_contract")
    if not isinstance(stamp, Mapping):
        raise ReplayPlanError("schema-v5 authoritative native contract is missing")
    expected_game_version = str(contract.value.get("game_version") or "")
    expected_contract_sha = str(contract.value.get("contract_sha256") or "")
    if str(stamp.get("game_version") or "") != expected_game_version:
        raise ReplayPlanError("schema-v5 authoritative game version mismatch")
    if str(stamp.get("contract_sha256") or "") != expected_contract_sha:
        raise ReplayPlanError("schema-v5 authoritative contract SHA-256 mismatch")
    if str(stamp.get("contract_file_sha256") or "") != str(contract.file_sha256):
        raise ReplayPlanError("schema-v5 authoritative contract file SHA-256 mismatch")
    eligibility = value.get("authoritative_eligibility")
    if not isinstance(eligibility, Mapping) or (
        eligibility.get("status") != "accepted"
        or eligibility.get("gate") != "native_static_v2"
    ):
        raise ReplayPlanError("schema-v5 authoritative eligibility stamp is invalid")

    rounds = value.get("rounds")
    deck_metadata = value.get("deck_metadata")
    if (
        value.get("draft") is not False
        or value.get("matchup_players") != "1v1"
        or value.get("normal_1v1") is not True
        or value.get("battle_type") == "riverRaceDuel"
        or not isinstance(rounds, list)
        or len(rounds) != 1
        or not isinstance(deck_metadata, Mapping)
        or deck_metadata.get("authoritative_complete") is not True
        or deck_metadata.get("source") != "battle_list_html"
        or value.get("deck_crosscheck_complete") is not True
    ):
        raise ReplayPlanError("schema-v5 authoritative 1v1/deck metadata is incomplete")
    timestamp = value.get("version_timestamp")
    if (
        not _strict_int(timestamp, minimum=1)
        or value.get("version_timestamp_provenance")
        != "battle_list_exact_timestamp"
        or not isinstance(value.get("battle_time_utc"), str)
        or not str(value["battle_time_utc"]).endswith("Z")
        or value.get("battle_time_utc_provenance") not in {
            "battle_timestamp_popup_utc",
            "derived_from_exact_list_timestamp",
        }
    ):
        raise ReplayPlanError("schema-v5 exact version timestamp is incomplete")

    mode = value.get("numeric_game_mode_id")
    if not _strict_int(mode, minimum=1):
        raise ReplayPlanError("schema-v5 numeric game mode is missing or not allowed")
    mode_provenance = str(value.get("numeric_game_mode_provenance") or "")
    if mode_provenance != "list_matchup_button_joined_by_data_index":
        raise ReplayPlanError("schema-v5 numeric game mode provenance is invalid")
    execution_mode = value.get("native_execution_game_mode_id")
    execution_mode_provenance = str(
        value.get("native_execution_game_mode_provenance") or ""
    )
    execution_issues = contract.validate_execution_game_mode(
        mode, execution_mode, execution_mode_provenance
    )
    if execution_issues:
        details = ",".join(issue.code for issue in execution_issues)
        raise ReplayPlanError(
            "schema-v5 native execution game mode is invalid: " + details
        )
    battle_index = value.get("battle_index")
    if not _strict_int(battle_index, minimum=1):
        raise ReplayPlanError("schema-v5 battle index is missing")
    battle_index_provenance = str(value.get("battle_index_provenance") or "")
    if battle_index_provenance != "list_replay_and_matchup_button_data_index":
        raise ReplayPlanError("schema-v5 battle index provenance is invalid")

    coordinates = value.get("coordinate_provenance")
    if not isinstance(coordinates, Mapping) or (
        coordinates.get("transform_id")
        != "royaleapi_data_i_to_libg_native_v1"
        or coordinates.get("target") != "libg_native_arena"
        or coordinates.get("missing_or_invalid_data_i") != "reject"
    ):
        raise ReplayPlanError("schema-v5 coordinate contract is invalid")
    if not isinstance(value.get("ability_plays"), list):
        raise ReplayPlanError("schema-v5 exact ability event array is missing")
    elixir = value.get("elixir_stats")
    for source_side in SIDE_NAMES:
        try:
            ability_count = elixir[source_side]["Ability"]["count"]
        except (KeyError, TypeError):
            raise ReplayPlanError(
                f"schema-v5 {source_side} ability count is missing"
            ) from None
        if not _strict_int(ability_count):
            raise ReplayPlanError(
                f"schema-v5 {source_side} ability count is invalid"
            )

    return {
        "contract": contract,
        "numeric_game_mode_id": int(mode),
        "numeric_game_mode_provenance": mode_provenance,
        "native_execution_game_mode_id": int(execution_mode),
        "native_execution_game_mode_provenance": execution_mode_provenance,
        "battle_index": int(battle_index),
        "battle_index_provenance": battle_index_provenance,
        "contract_game_version": expected_game_version,
        "contract_sha256": expected_contract_sha,
        "contract_file_sha256": str(contract.file_sha256),
        "contract_provenance": "schema5_authoritative_native_contract_verified",
    }


def _side_deck(
    value: Mapping[str, Any], source_side: str, source_schema: int
) -> tuple[
    tuple[CardSpec, ...], str | None, int | None, str, int | None, str,
]:
    if source_schema >= 2:
        player = _schema_two_player(value, source_side)
        tokens = player.get("full_deck")
        levels = player.get("card_levels")
        if not isinstance(tokens, list) or len(tokens) != 8 or not isinstance(levels, Mapping):
            raise ReplayPlanError(f"{source_side} requires eight cards and levels")
        result = tuple(card_spec(str(token), int(levels[str(token)])) for token in tokens)
        tower = str(player.get("tower_troop") or "") or None
        tower_level = (
            int(player["tower_troop_level"])
            if source_schema == 5
            and _strict_int(player.get("tower_troop_level"), minimum=1)
            else None
        )
        tower_level_provenance = (
            "schema5_round_player_battle_list_tower_card"
            if tower_level is not None
            else "legacy_not_observed"
        )
        king_tower_level = (
            int(player["king_tower_level"])
            if source_schema == 5
            and _strict_int(player.get("king_tower_level"), minimum=1)
            else None
        )
        king_tower_level_provenance = (
            str(player.get("king_tower_level_provenance") or "")
            if source_schema == 5
            else "legacy_not_observed"
        )
        if source_schema == 5 and king_tower_level != 16:
            raise ReplayPlanError(
                f"schema-v5 {source_side} King Tower level evidence is invalid"
            )
        if source_schema == 5:
            top_deck = value.get(f"{source_side}_deck")
            if top_deck != tokens:
                raise ReplayPlanError(
                    f"schema-v5 {source_side} top-level/round deck mismatch"
                )
            deck_cards = player.get("deck_cards")
            if not isinstance(deck_cards, list) or len(deck_cards) != 8:
                raise ReplayPlanError(
                    f"schema-v5 {source_side} deck_cards are missing"
                )
            for slot, (token, item) in enumerate(zip(tokens, deck_cards, strict=True)):
                if not isinstance(item, Mapping):
                    raise ReplayPlanError(
                        f"schema-v5 {source_side} deck card {slot} is invalid"
                    )
                base, flags = split_card_token(str(token))
                expected_form = (
                    "ev1" if flags == 1 else
                    "hero" if flags == 2 else "base"
                )
                if (
                    item.get("slot") != slot
                    or item.get("slug") != token
                    or item.get("base_slug") != base
                    or item.get("form") != expected_form
                    or item.get("level") != levels.get(str(token))
                ):
                    raise ReplayPlanError(
                        f"schema-v5 {source_side} deck card {slot} is inconsistent"
                    )
    else:
        counts = value.get("card_counts")
        side_counts = counts.get(source_side) if isinstance(counts, Mapping) else None
        if not isinstance(side_counts, Mapping) or len(side_counts) != 8:
            raise ReplayPlanError(f"{source_side} does not expose eight played cards")
        result = tuple(card_spec(str(token), None) for token in side_counts)
        tower = None
        tower_level = None
        tower_level_provenance = "legacy_not_observed"
        king_tower_level = None
        king_tower_level_provenance = "legacy_not_observed"
    bases = [item.base_token for item in result]
    if len(set(bases)) != 8:
        raise ReplayPlanError(f"{source_side} deck has duplicate base-card identities")
    return (
        result,
        tower,
        tower_level,
        tower_level_provenance,
        king_tower_level,
        king_tower_level_provenance,
    )


def compile_battle(
    value: Mapping[str, Any],
    *,
    terminal_crowns: Sequence[int] | None = None,
    native_ingest_contract: Any | None = None,
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
    schema_five = (
        _schema_five_metadata(value, native_ingest_contract)
        if source_schema == 5
        else None
    )
    if source_schema == 5:
        source_crowns = (value.get("team_crowns"), value.get("opponent_crowns"))
        if any(
            not _strict_int(item) or int(item) > 3
            for item in source_crowns
        ):
            raise ReplayPlanError("schema-v5 terminal crowns are missing or invalid")
        exact_crowns = (int(source_crowns[0]), int(source_crowns[1]))
        if crowns is not None and crowns != exact_crowns:
            raise ReplayPlanError("schema-v5 terminal crowns disagree with caller")
        crowns = exact_crowns
    if value.get("draft") is True:
        raise ReplayPlanError("draft battles are not supported")
    duration_seconds = value.get("duration_seconds")
    if not isinstance(duration_seconds, (int, float)) or not 1 <= duration_seconds <= 360:
        raise ReplayPlanError("battle duration is missing or outside 1..360 seconds")
    if source_schema == 5 and not _strict_int(duration_seconds, minimum=1):
        raise ReplayPlanError("schema-v5 battle duration must be an exact integer")
    duration_ticks = int(round(float(duration_seconds) * 20.0))
    plays = value.get("card_plays")
    if not isinstance(plays, list) or not plays:
        raise ReplayPlanError("battle has no card play events")

    side_decks: list[tuple[CardSpec, ...]] = []
    tower_troops: list[str | None] = []
    tower_troop_levels: list[int | None] = []
    tower_troop_level_provenance: list[str] = []
    king_tower_levels: list[int | None] = []
    king_tower_level_provenance: list[str] = []
    final_tower_hp: list[FinalTowerHp | None] = []
    logical_indices: list[dict[str, int]] = []
    for source_side in SIDE_NAMES:
        (
            deck,
            tower,
            tower_level,
            tower_level_source,
            king_tower_level,
            king_tower_level_source,
        ) = _side_deck(
            value, source_side, source_schema
        )
        if source_schema == 5 and tower_level is None:
            raise ReplayPlanError(
                f"schema-v5 {source_side} tower troop level is missing"
            )
        side_decks.append(deck)
        tower_troops.append(tower)
        tower_troop_levels.append(tower_level)
        tower_troop_level_provenance.append(tower_level_source)
        king_tower_levels.append(king_tower_level)
        king_tower_level_provenance.append(king_tower_level_source)
        terminal_hp = (
            _schema_five_terminal_hp(value, source_side)
            if source_schema == 5
            else None
        )
        if source_schema == 5:
            assert schema_five is not None
            evidence_issues = schema_five[
                "contract"
            ].validate_king_tower_level_evidence(
                king_tower_level=king_tower_level,
                provenance=king_tower_level_source,
                tower_troop_level=tower_level,
                final_king_hp=(None if terminal_hp is None else terminal_hp.king),
            )
            if evidence_issues:
                raise ReplayPlanError(
                    f"schema-v5 {source_side} King Tower level evidence is "
                    f"inconsistent: {evidence_issues[0].code}"
                )
        final_tower_hp.append(terminal_hp)
        logical_indices.append({item.base_token: index for index, item in enumerate(deck)})

    if source_schema == 5:
        assert schema_five is not None
        contract = schema_five["contract"]
        for side, source_side in enumerate(SIDE_NAMES):
            for spec in side_decks[side]:
                issues = contract.validate_card_token(spec.source_token)
                if issues:
                    raise ReplayPlanError(
                        f"schema-v5 {source_side} card violates native contract: "
                        f"{issues[0].code}:{spec.source_token}"
                    )
            if tower_troops[side] is None:
                raise ReplayPlanError(
                    f"schema-v5 {source_side} tower troop is missing"
                )
            tower_issues = contract.validate_tower_troop(tower_troops[side])
            if tower_issues:
                raise ReplayPlanError(
                    f"schema-v5 {source_side} tower troop violates native contract"
                )

    actions: list[ExpertAction] = []
    per_side_indices: list[list[int]] = [[], []]
    same_side_ticks: set[tuple[int, int]] = set()
    side_action_counts = [0, 0]
    coordinate_flag_counts = [0, 0]
    legacy_coordinate_events = 0
    source_marker_indices: set[int] = set()
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
        x, y, coordinate_flag = _native_event_coordinates(
            event, event_index=event_index
        )
        marker_index = event.get("marker_index", event_index)
        if source_schema == 5:
            if not _strict_int(marker_index):
                raise ReplayPlanError(
                    f"schema-v5 event {event_index} marker index is missing"
                )
            if int(marker_index) in source_marker_indices:
                raise ReplayPlanError(
                    f"schema-v5 duplicate marker index {marker_index}"
                )
            source_marker_indices.add(int(marker_index))
            expected_token = side_decks[side][logical_indices[side][base_token]].source_token
            if event.get("card_form") != expected_token:
                raise ReplayPlanError(
                    f"schema-v5 event {event_index} card form is inconsistent"
                )
            expected_transform = "rotate_180" if coordinate_flag == 0 else "identity"
            if (
                event.get("coordinate_provenance")
                != "royaleapi_data_i_to_libg_native_v1"
                or event.get("coordinate_transform") != expected_transform
                or event.get("x") != x
                or event.get("y") != y
            ):
                raise ReplayPlanError(
                    f"schema-v5 event {event_index} coordinate derivation mismatch"
                )
        if coordinate_flag is None:
            legacy_coordinate_events += 1
        else:
            coordinate_flag_counts[coordinate_flag] += 1
        actions.append(ExpertAction(
            tick=tick, side=side, logical_card_index=logical,
            base_token=base_token, x=x, y=y, source_event_index=event_index,
            source_marker_index=int(marker_index),
            side_action_index=side_action_counts[side],
        ))
        side_action_counts[side] += 1
    if any(
        actions[index].tick > actions[index + 1].tick
        for index in range(len(actions) - 1)
    ):
        raise ReplayPlanError("card events are not sorted by tick")
    if source_schema == 5:
        for side, source_side in enumerate(SIDE_NAMES):
            observed_bases = {
                action.base_token for action in actions if action.side == side
            }
            expected_bases = {spec.base_token for spec in side_decks[side]}
            if observed_bases != expected_bases:
                raise ReplayPlanError(
                    f"schema-v5 {source_side} does not expose the complete eight-card cycle"
                )

    raw_coordinate_events = sum(coordinate_flag_counts)
    if raw_coordinate_events and not legacy_coordinate_events:
        coordinate_provenance = "royaleapi_raw_data_i_to_native_v1"
        coordinate_transform = (
            "data_i=0:rotate_18000_32000;data_i=1:identity"
        )
    elif raw_coordinate_events:
        coordinate_provenance = "mixed_raw_data_i_and_legacy_xy_fallback"
        coordinate_transform = (
            "raw:data_i=0_rotate,data_i=1_identity;legacy:stored_xy"
        )
    else:
        coordinate_provenance = "legacy_stored_xy_fallback_unverified"
        coordinate_transform = "legacy:stored_xy"
    coordinate_audit = CoordinateAudit(
        transform=coordinate_transform,
        raw_data_i_events=raw_coordinate_events,
        data_i_zero_events=coordinate_flag_counts[0],
        data_i_one_events=coordinate_flag_counts[1],
        legacy_xy_fallback_events=legacy_coordinate_events,
        data_i_values=tuple(
            flag for flag, count in enumerate(coordinate_flag_counts) if count
        ),
    )

    cycles = [compatible_cycle(indices) for indices in per_side_indices]
    ability_counts = [_ability_count(value, source_side) for source_side in SIDE_NAMES]
    compiled_abilities = _compile_ability_events(
        value, duration_ticks=duration_ticks
    )
    if source_schema == 5:
        for event in compiled_abilities:
            if event.source_marker_index in source_marker_indices:
                raise ReplayPlanError(
                    f"schema-v5 duplicate marker index {event.source_marker_index}"
                )
            source_marker_indices.add(event.source_marker_index)
    observed_ability_counts = [
        sum(1 for event in compiled_abilities if event.side == side)
        for side in range(2)
    ]
    if source_schema == 5:
        assert schema_five is not None
        contract = schema_five["contract"]
        for side, source_side in enumerate(SIDE_NAMES):
            issues = contract.validate_ability_source(
                [spec.source_token for spec in side_decks[side]],
                observed_ability_events=observed_ability_counts[side],
            )
            if issues:
                raise ReplayPlanError(
                    f"schema-v5 {source_side} ability source violates native contract"
                )
    ability_tier = ability_log_tier(value)
    if source_schema == 5 and ability_tier not in {
        "source_reports_zero", "observed_ticks_identity_runtime_resolved",
    }:
        raise ReplayPlanError(
            f"schema-v5 exact ability contract is incomplete: {ability_tier}"
        )
    side_plans = tuple(
        SidePlan(
            side=side, source_side=SIDE_NAMES[side], deck=side_decks[side],
            cycle=cycles[side], action_count=len(per_side_indices[side]),
            observed_ability_event_count=observed_ability_counts[side],
            missing_ability_event_count=max(
                0, ability_counts[side] - observed_ability_counts[side]
            ),
            tower_troop=tower_troops[side],
            tower_troop_level=tower_troop_levels[side],
            tower_troop_level_provenance=tower_troop_level_provenance[side],
            king_tower_level=king_tower_levels[side],
            king_tower_level_provenance=king_tower_level_provenance[side],
            final_tower_hp=final_tower_hp[side],
        )
        for side in range(2)
    )

    # A player can submit at most one command in a native Tick.  Preserve
    # simultaneous opposing actions, but reject a same-side deploy/ability or
    # duplicate ability instead of inventing an order inside that Tick.
    occupied_ticks = {(action.tick, action.side) for action in actions}
    for event in compiled_abilities:
        key = (event.tick, event.side)
        if key in occupied_ticks:
            raise ReplayPlanError(
                f"multiple deploy/ability actions for side {event.side} "
                f"at native tick {event.tick}"
            )
        occupied_ticks.add(key)

    limitations = [
        "source_native_rng_seed_missing",
        "source_exact_game_build_missing",
        "source_initial_hand_not_observed",
        "source_tick_state_anchors_missing",
    ]
    if source_schema != 5:
        limitations.extend((
            "source_numeric_game_mode_missing",
            "source_king_tower_level_missing",
            "source_tower_troop_level_missing",
            "source_final_six_tower_hp_missing",
        ))
    if legacy_coordinate_events:
        limitations.append("legacy_precomputed_coordinates_unverified")
    if source_schema < 2:
        limitations.extend((
            "card_forms_missing", "card_levels_missing", "tower_troops_missing",
        ))
    if ability_tier == "count_only_missing_ticks":
        limitations.extend((
            "ability_button_events_missing",
            "ability_button_event_ticks_missing",
        ))
    elif ability_tier == "observed_tick_count_mismatch":
        limitations.append("ability_button_event_count_mismatch")
    for side, expected in enumerate(ability_counts):
        if expected and not ability_cards(side_decks[side]):
            limitations.append(f"ability_card_runtime_mapping_missing_side_{side}")
    unmapped_towers: list[str] = []
    for tower in tower_troops:
        if tower is None:
            continue
        try:
            tower_troop(tower)
        except ValueError:
            unmapped_towers.append(tower)
    if unmapped_towers:
        limitations.append(
            "tower_troop_runtime_mapping_missing:" + ",".join(sorted(set(unmapped_towers)))
        )
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
    ability_events_complete = ability_tier in {
        "source_reports_zero", "observed_ticks_identity_runtime_resolved",
    }
    replay_ready = (
        metadata_exact
        and ability_events_complete
        and all(not count or ability_cards(side_decks[side])
                for side, count in enumerate(ability_counts))
        and not unmapped_towers
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
        numeric_game_mode_id=(
            None if schema_five is None
            else int(schema_five["numeric_game_mode_id"])
        ),
        numeric_game_mode_provenance=(
            "legacy_unobserved" if schema_five is None
            else str(schema_five["numeric_game_mode_provenance"])
        ),
        native_execution_game_mode_id=(
            None if schema_five is None
            else int(schema_five["native_execution_game_mode_id"])
        ),
        native_execution_game_mode_provenance=(
            "legacy_template_mode_unmodified" if schema_five is None
            else str(schema_five["native_execution_game_mode_provenance"])
        ),
        battle_index=(
            None if schema_five is None else int(schema_five["battle_index"])
        ),
        battle_index_provenance=(
            "legacy_unobserved" if schema_five is None
            else str(schema_five["battle_index_provenance"])
        ),
        authoritative_contract_game_version=(
            None if schema_five is None
            else str(schema_five["contract_game_version"])
        ),
        authoritative_contract_sha256=(
            None if schema_five is None
            else str(schema_five["contract_sha256"])
        ),
        authoritative_contract_file_sha256=(
            None if schema_five is None
            else str(schema_five["contract_file_sha256"])
        ),
        authoritative_contract_provenance=(
            "legacy_unobserved" if schema_five is None
            else str(schema_five["contract_provenance"])
        ),
        duration_ticks=duration_ticks,
        sides=side_plans,  # type: ignore[arg-type]
        actions=tuple(actions),
        ability_events=compiled_abilities,
        ability_log_tier=ability_tier,
        replay_tier=tier,
        native_replay_ready=replay_ready,
        # No current RoyaleAPI artifact has the seed/build/state anchors needed
        # to make this claim, even when all deployment actions are accepted.
        original_state_exact=False,
        state_provenance=(
            "native_generated_schema5_authoritative_metadata_anchored"
            if source_schema == 5 else "native_generated_unanchored"
        ),
        action_provenance=(
            "schema5_observed_exact_tick_deployments"
            if source_schema == 5 else "observed_deployments"
        ),
        coordinate_provenance=coordinate_provenance,
        coordinate_audit=coordinate_audit,
        hand_provenance=(
            "inferred_exact_all_actions"
            if all(cycle.first_exact_action_index == 0 for cycle in cycles)
            else "inferred_mixed_ambiguous_prefix_then_exact"
        ),
        ability_provenance=_ability_provenance(value, ability_counts),
        terminal_provenance=(
            "schema5_source_crowns_and_final_six_tower_hp"
            if source_schema == 5
            else "source_index_crowns"
            if crowns is not None
            else "unknown_no_source_anchor"
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


def materialize_replay(
    plan: BattlePlan,
    template: Mapping[str, Any],
    calibrated_players: Sequence[Mapping[str, Any]] | None = None,
    *,
    seed: int = DEFAULT_NATIVE_SEED,
    fallback_level: int = 11,
) -> tuple[dict[str, Any], tuple[tuple[int, ...], tuple[int, ...]]]:
    """Build a source-order libg replay without moving form-designated slots.

    ``calibrated_players`` is retained as a compatibility-only parameter for
    older callers.  The former implementation permuted the eight input deck
    slots to force one arbitrarily selected compatible 4+4 state.  Real libg
    evidence shows that shuffle order depends on card identity as well as the
    slot, so that fixed-point transform can oscillate and, more importantly,
    moves Evo/Hero form-slot semantics away from the source ``full_deck``.

    The replay now preserves logical/source deck order exactly and returns an
    identity logical-to-native mapping.  The missing initial-hand seed is
    resolved separately against authoritative libg by bounded seed search.
    """
    import copy

    if calibrated_players is not None and len(calibrated_players) != 2:
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
    if plan.native_execution_game_mode_id is not None:
        # Schema 5 keeps the list-page mode as source truth but executes the
        # explicit frozen contract transform.  Older sources deliberately
        # retain the template mode instead.
        battle["gamemode"] = int(plan.native_execution_game_mode_id)
    mappings: list[tuple[int, ...]] = []
    for side, side_plan in enumerate(plan.sides):
        mapping = tuple(range(8))
        mappings.append(mapping)
        native_deck: list[dict[str, int]] = []
        for logical_index, spec in enumerate(side_plan.deck):
            row = {
                "d": int(spec.card_id),
                "l": int((spec.level or fallback_level) - 1),
            }
            if spec.form_flags:
                row["el"] = int(spec.form_flags)
            if mapping[logical_index] != logical_index:
                raise AssertionError("source-order mapping must remain identity")
            native_deck.append(row)
        battle[f"deck{side}"]["sp"] = native_deck
        support = tower_troop(side_plan.tower_troop or "tower-princess")
        support_level = (
            side_plan.tower_troop_level
            if side_plan.tower_troop_level is not None
            else max((spec.level or fallback_level) for spec in side_plan.deck)
        )
        battle[f"deck{side}"]["sc"] = [{
            "d": support.support_card_id,
            # Tower Troop level is not King level and is not inferred from the
            # highest card in authoritative schema 5.  The legacy fallback is
            # retained only for schema 1..4 compatibility.
            "l": int(support_level - 1),
            "t": 0,
            "c": 0,
        }]
        if side_plan.king_tower_level is not None:
            king_level = int(side_plan.king_tower_level)
            avatar = battle.get(f"avatar{side}")
            home_battle_data = battle.get("hbd")
            if (
                not isinstance(avatar, dict)
                or not isinstance(home_battle_data, list)
                or len(home_battle_data) <= side
                or not isinstance(home_battle_data[side], dict)
            ):
                raise ReplayPlanError(
                    "native template lacks avatar/home battle data for King level"
                )
            avatar["expLevel"] = king_level
            avatar["kt"] = king_level
            home_battle_data[side]["kt"] = king_level
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


def grouped_replay_events(
    plan: BattlePlan, mappings: Sequence[Sequence[int]],
) -> Iterable[tuple[int, list[dict[str, Any]]]]:
    """Yield deployments and unresolved ability markers on one Tick timeline."""
    events: list[dict[str, Any]] = []
    for action in plan.actions:
        events.append({
            "type": "play",
            "tick": action.tick,
            "side": action.side,
            "deck_index": int(mappings[action.side][action.logical_card_index]),
            "x": action.x,
            "y": action.y,
            "source_event_index": action.source_event_index,
            "source_marker_index": action.source_marker_index,
        })
    for event in plan.ability_events:
        events.append({
            "type": "ability_marker",
            "tick": event.tick,
            "side": event.side,
            "source_event_index": event.source_event_index,
            "source_marker_index": event.source_marker_index,
            "source_ability_id": event.source_ability_id,
        })
    events.sort(key=lambda item: (
        int(item["tick"]), int(item["source_marker_index"]), int(item["side"])
    ))
    current_tick: int | None = None
    batch: list[dict[str, Any]] = []
    for event in events:
        tick = int(event["tick"])
        if current_tick is not None and tick != current_tick:
            yield current_tick, sorted(batch, key=lambda item: int(item["side"]))
            batch = []
        current_tick = tick
        batch.append(event)
    if current_tick is not None:
        yield current_tick, sorted(batch, key=lambda item: int(item["side"]))
