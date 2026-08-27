"""Content-addressed native deployment masks for offline expert labels.

``probe_grid`` can only inspect a card while that deck slot is in the native
hand.  During a teacher-forced replay we therefore probe each ``(side,
deck_index)`` exactly once, the first time it is visible.  The immutable raw
18x32 result is stored once by content hash.  Per-Tick ownership, destroyed
Princess-Tower pockets, and living Tower footprints are reconstructed from
the lossless :class:`~expert_v1.tick_store_v1.schema.TickState` stream.

This module intentionally contains no game simulation.  The base mask comes
from libg, and the dynamic layer is the same deterministic projection used by
the online action-mask path.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any, Iterable, Mapping, Sequence

from .schema import TickState


MASK_SCHEMA_VERSION = 1
BASE_MASK_KIND = "cr_native_deployment_base_mask_v1"
EPISODE_MASK_KIND = "cr_native_episode_deployment_masks_v1"
MASK_STORE_KIND = "cr_native_deployment_mask_store_v1"
EPISODE_METADATA_KEY = "native_deployment_masks_v1"
MASK_STORE_DIRECTORY = "deployment-masks-v1"

ARENA_COLUMNS = 18
ARENA_ROWS = 32
CELL_SIZE = 1000
EXPECTED_SIDES = (0, 1)
EXPECTED_DECK_INDICES = tuple(range(8))
EXPECTED_SLOT_COUNT = len(EXPECTED_SIDES) * len(EXPECTED_DECK_INDICES)
POCKET_DEPTH_CELLS = 5
LANE_SPLIT_COLUMN = 9
GLOBAL_DEPLOY_CARD_IDS = frozenset({26_000_032, 27_000_013})
DYNAMIC_RULE = "native_base_and_tower_state_projection_v2"
CAPTURE_STRATEGY = (
    "probe_once_when_each_side_deck_index_first_enters_native_hand_plus_"
    "exact_play_tick_variants_for_native_dynamic_choice"
)
ENTRY_ENCODING = (
    "side",
    "deck_index",
    "card_id",
    "level",
    "form_flags",
    "capture_tick",
    "content_sha256",
    "native_selection_strategy",
    "tick_variant_required",
    "dynamic_label_variants",
)
VARIANT_ENCODING = ("tick", "content_sha256")
CONTENT_PATH_RULE = (
    f"{MASK_STORE_DIRECTORY}/<sha256[0:2]>/<sha256>.json"
)
MIRROR_CARD_ID = 28_000_006

# Compilation constructs short-lived store objects per episode/actor.  A
# per-instance cache therefore still caused one authenticated JSON read for
# every Tick x four cards.  These caches are process-wide and keyed by the
# resolved store root so independent corpora cannot alias each other.
_PROCESS_CACHE_LOCK = threading.RLock()
_PROCESS_PAYLOAD_CACHE: dict[tuple[str, str], dict[str, Any]] = {}
_PROCESS_DERIVED_CACHE: dict[
    tuple[str, str, int, tuple[tuple[int, int, int, int, int], ...]],
    tuple[str, ...],
] = {}


class DeploymentMaskContractError(ValueError):
    """A native mask, reference, or offline label is not authoritative."""


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DeploymentMaskContractError(f"{name} must be an integer")
    return int(value)


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _validate_sha256(value: Any, name: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise DeploymentMaskContractError(f"{name} is not lowercase SHA-256")
    return text


def normalize_native_probe(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one libg ``probe_grid`` response without guessing fields."""
    width = _integer(value.get("width"), "probe.width")
    height = _integer(value.get("height"), "probe.height")
    cell_size = _integer(value.get("cell_size"), "probe.cell_size")
    if (width, height, cell_size) != (ARENA_COLUMNS, ARENA_ROWS, CELL_SIZE):
        raise DeploymentMaskContractError(
            "native deployment mask grid is not 18x32 at 1000-unit cells"
        )
    raw_rows = value.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != ARENA_ROWS:
        raise DeploymentMaskContractError("native deployment mask needs 32 rows")
    rows = []
    for index, row in enumerate(raw_rows):
        if (
            not isinstance(row, str)
            or len(row) != ARENA_COLUMNS
            or any(cell not in "01" for cell in row)
        ):
            raise DeploymentMaskContractError(
                f"native deployment mask row {index} is not 18 binary cells"
            )
        rows.append(row)
    valid_cells = sum(row.count("1") for row in rows)
    if _integer(value.get("valid_cells"), "probe.valid_cells") != valid_cells:
        raise DeploymentMaskContractError("native valid_cells disagrees with rows")

    packed_selection = _integer(
        value.get("packed_selection"), "probe.packed_selection"
    )
    card_cost = _integer(value.get("card_cost"), "probe.card_cost")
    card_cost_raw = _integer(value.get("card_cost_raw"), "probe.card_cost_raw")
    if card_cost < 0 or card_cost_raw != card_cost * 10_000:
        raise DeploymentMaskContractError("native card-cost fields disagree")
    strategy = str(value.get("selection_strategy") or "")
    if strategy not in {"canonical", "native_dynamic_choice"}:
        raise DeploymentMaskContractError(
            f"unsupported native selection strategy: {strategy!r}"
        )
    resolved_data_id = _integer(
        value.get("resolved_data_id"), "probe.resolved_data_id"
    )
    if resolved_data_id <= 0:
        raise DeploymentMaskContractError(
            "native resolved_data_id must be positive"
        )
    result = {
        "schema_version": MASK_SCHEMA_VERSION,
        "kind": BASE_MASK_KIND,
        "width": width,
        "height": height,
        "cell_size": cell_size,
        "valid_cells": valid_cells,
        "rows": rows,
        "resolved_data_id": resolved_data_id,
        "packed_selection": packed_selection,
        "card_cost": card_cost,
        "card_cost_raw": card_cost_raw,
        "selection_form_index": _integer(
            value.get("selection_form_index"), "probe.selection_form_index"
        ),
        "selection_strategy": strategy,
        "selection_builder_rva": str(value.get("selection_builder_rva") or ""),
        "selection_root_vtable_rva": str(
            value.get("selection_root_vtable_rva") or ""
        ),
    }
    for name in ("selection_builder_rva", "selection_root_vtable_rva"):
        if not result[name].startswith("0x"):
            raise DeploymentMaskContractError(f"probe.{name} lacks RVA provenance")
    return result


def sidecar_sha256(value: Mapping[str, Any]) -> str:
    normalized = normalize_sidecar(value)
    return _sha256(_canonical_bytes(normalized))


def normalize_sidecar(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a persisted sidecar and return its canonical representation."""
    if value.get("kind") != BASE_MASK_KIND:
        raise DeploymentMaskContractError("deployment-mask sidecar kind changed")
    if (
        _integer(value.get("schema_version"), "sidecar.schema_version")
        != MASK_SCHEMA_VERSION
    ):
        raise DeploymentMaskContractError("deployment-mask sidecar schema changed")
    return normalize_native_probe(value)


def _sidecar_relative_path(content_sha256: str) -> str:
    digest = _validate_sha256(content_sha256, "content_sha256")
    return f"{MASK_STORE_DIRECTORY}/{digest[:2]}/{digest}.json"


@dataclass(frozen=True, slots=True)
class DeckSlot:
    side: int
    deck_index: int
    card_id: int
    level: int | None
    form_flags: int
    source_token: str
    base_token: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DeckSlot":
        side = _integer(value.get("side"), "deck.side")
        deck_index = _integer(value.get("deck_index"), "deck.deck_index")
        card_id = _integer(value.get("card_id"), "deck.card_id")
        form_flags = _integer(value.get("form_flags", 0), "deck.form_flags")
        level_value = value.get("level")
        level = None if level_value is None else _integer(level_value, "deck.level")
        if side not in EXPECTED_SIDES or deck_index not in EXPECTED_DECK_INDICES:
            raise DeploymentMaskContractError("deck slot is outside side/deck range")
        if card_id <= 0 or form_flags < 0 or (level is not None and level <= 0):
            raise DeploymentMaskContractError("deck slot card fields are invalid")
        source_token = str(value.get("source_token") or "")
        base_token = str(value.get("base_token") or "")
        if not source_token or not base_token:
            raise DeploymentMaskContractError("deck slot lacks card-token provenance")
        return cls(
            side, deck_index, card_id, level, form_flags,
            source_token, base_token,
        )


class NativeDeploymentMaskCapture:
    """Capture every native deck slot with at most one probe RPC per slot."""

    def __init__(self, deck_slots: Sequence[Mapping[str, Any]]) -> None:
        slots = [DeckSlot.from_mapping(value) for value in deck_slots]
        keys = [(slot.side, slot.deck_index) for slot in slots]
        expected = [
            (side, deck_index)
            for side in EXPECTED_SIDES
            for deck_index in EXPECTED_DECK_INDICES
        ]
        if sorted(keys) != expected:
            raise DeploymentMaskContractError(
                "mask capture requires exactly both sides' eight deck slots"
            )
        self._slots = {(slot.side, slot.deck_index): slot for slot in slots}
        self._entries: dict[tuple[int, int], dict[str, Any]] = {}
        self._payloads: dict[str, dict[str, Any]] = {}
        self.probe_rpc_count = 0
        self.dynamic_label_probe_rpc_count = 0

    @property
    def complete(self) -> bool:
        return len(self._entries) == EXPECTED_SLOT_COUNT

    @property
    def captured_slots(self) -> int:
        return len(self._entries)

    @property
    def missing_slots(self) -> tuple[tuple[int, int], ...]:
        return tuple(sorted(set(self._slots) - set(self._entries)))

    @property
    def payloads(self) -> dict[str, dict[str, Any]]:
        return {digest: dict(value) for digest, value in self._payloads.items()}

    def capture_available(self, env: Any, state: Mapping[str, Any]) -> int:
        """Probe unseen slots currently in hand; repeated calls are RPC-free."""
        tick = _integer(state.get("tick"), "state.tick")
        players = state.get("players")
        if not isinstance(players, list):
            raise DeploymentMaskContractError("native state lacks players")
        by_side: dict[int, Mapping[str, Any]] = {}
        for player in players:
            if not isinstance(player, Mapping):
                raise DeploymentMaskContractError("native player is not an object")
            side = _integer(player.get("side"), "player.side")
            if side in by_side or side not in EXPECTED_SIDES:
                raise DeploymentMaskContractError("native players are not sides 0/1")
            by_side[side] = player
        if set(by_side) != set(EXPECTED_SIDES):
            raise DeploymentMaskContractError("native state must expose two players")

        pending: list[tuple[int, int]] = []
        for side in EXPECTED_SIDES:
            raw_hand = by_side[side].get("hand_deck_indices")
            if not isinstance(raw_hand, list) or len(raw_hand) != 4:
                raise DeploymentMaskContractError("native hand must contain four slots")
            hand = [_integer(value, "hand.deck_index") for value in raw_hand]
            # libg exposes -1 while a just-played slot waits for its refill
            # timer.  This is a valid, losslessly stored transient state; probe
            # only the currently materialized unique deck indices.
            visible = [index for index in hand if index != -1]
            if (
                not visible
                or len(set(visible)) != len(visible)
                or any(index not in EXPECTED_DECK_INDICES for index in visible)
            ):
                raise DeploymentMaskContractError("native hand deck indices are invalid")
            pending.extend(
                (side, deck_index)
                for deck_index in sorted(visible)
                if (side, deck_index) not in self._entries
            )

        for side, deck_index in pending:
            # Increment before the call so an exception is still accounted as
            # an attempted native probe in diagnostics.
            self.probe_rpc_count += 1
            raw = env.probe_grid(side=side, deck_index=deck_index)
            if not isinstance(raw, Mapping):
                raise DeploymentMaskContractError("probe_grid returned no object")
            payload = normalize_native_probe(raw)
            encoded = _canonical_bytes(payload)
            digest = _sha256(encoded)
            slot = self._slots[(side, deck_index)]
            self._payloads.setdefault(digest, payload)
            self._entries[(side, deck_index)] = {
                "side": side,
                "deck_index": deck_index,
                "card_id": slot.card_id,
                "level": slot.level,
                "form_flags": slot.form_flags,
                "source_token": slot.source_token,
                "base_token": slot.base_token,
                "capture_tick": tick,
                "content_sha256": digest,
                "content_path": _sidecar_relative_path(digest),
                "native_resolved_data_id": payload["resolved_data_id"],
                "native_card_cost": payload["card_cost"],
                "native_selection_form_index": payload["selection_form_index"],
                "native_selection_strategy": payload["selection_strategy"],
                "tick_variant_required": bool(
                    payload["selection_strategy"] == "native_dynamic_choice"
                    or slot.card_id == MIRROR_CARD_ID
                    or int(payload["resolved_data_id"]) != slot.card_id
                ),
                "dynamic_label_variants": [],
            }
        return len(pending)

    def capture_label_variants(
        self,
        env: Any,
        state: Mapping[str, Any],
        labels: Iterable[Mapping[str, Any]],
    ) -> int:
        """Pin dynamic-choice selection to each actual expert play Tick.

        Ordinary cards require no further RPC after their base slot probe.
        A wrapper such as Spirit Empress or Mirror can resolve to a different
        form/cost/mask as resources or the previous card change.  At each
        deployment label we therefore probe every tick-variant slot in that
        actor's current hand: card-head supervision needs the whole legal
        hand, not only the card that was selected.
        """
        tick = _integer(state.get("tick"), "state.tick")
        players = state.get("players")
        if not isinstance(players, list):
            raise DeploymentMaskContractError("native state lacks players")
        hands: dict[int, set[int]] = {}
        for player in players:
            if not isinstance(player, Mapping):
                raise DeploymentMaskContractError("native player is not an object")
            side = _integer(player.get("side"), "player.side")
            raw_hand = player.get("hand_deck_indices")
            if side not in EXPECTED_SIDES or not isinstance(raw_hand, list):
                raise DeploymentMaskContractError("native player hand is invalid")
            hand = [_integer(value, "hand.deck_index") for value in raw_hand]
            visible = [value for value in hand if value != -1]
            if (
                not visible
                or len(set(visible)) != len(visible)
                or any(value not in EXPECTED_DECK_INDICES for value in visible)
            ):
                raise DeploymentMaskContractError(
                    "native player hand deck indices are invalid"
                )
            hands[side] = set(visible)
        calls = 0
        labelled_slots = {
            (
                _integer(label.get("side"), "label.side"),
                _integer(label.get("deck_index"), "label.deck_index"),
            )
            for label in labels
        }
        for side, deck_index in labelled_slots:
            if (
                (side, deck_index) not in self._entries
                or deck_index not in hands.get(side, set())
            ):
                raise DeploymentMaskContractError(
                    "expert label slot is not captured/currently in hand"
                )
        actor_sides = {side for side, _deck_index in labelled_slots}
        uncaptured_hand = [
            (side, deck_index)
            for side in actor_sides
            for deck_index in hands.get(side, set())
            if (side, deck_index) not in self._entries
        ]
        if uncaptured_hand:
            raise DeploymentMaskContractError(
                "deployment label hand contains uncaptured mask slots"
            )
        keys = sorted(
            (side, deck_index)
            for side in actor_sides
            for deck_index in hands.get(side, set())
            if self._entries[(side, deck_index)]["tick_variant_required"]
        )
        for key in keys:
            side, deck_index = key
            entry = self._entries[key]
            variants = entry["dynamic_label_variants"]
            if any(int(value["tick"]) == tick for value in variants):
                continue
            if int(entry["capture_tick"]) == tick:
                payload = self._payloads[str(entry["content_sha256"])]
                digest = str(entry["content_sha256"])
            else:
                self.probe_rpc_count += 1
                self.dynamic_label_probe_rpc_count += 1
                calls += 1
                raw = env.probe_grid(side=side, deck_index=deck_index)
                if not isinstance(raw, Mapping):
                    raise DeploymentMaskContractError(
                        "dynamic probe_grid returned no object"
                    )
                payload = normalize_native_probe(raw)
                if (
                    payload["selection_strategy"]
                    != entry["native_selection_strategy"]
                ):
                    raise DeploymentMaskContractError(
                        "native tick-variant slot changed selection strategy"
                    )
                encoded = _canonical_bytes(payload)
                digest = _sha256(encoded)
                self._payloads.setdefault(digest, payload)
            variants.append({
                "tick": tick,
                "content_sha256": digest,
                "content_path": _sidecar_relative_path(digest),
                "native_resolved_data_id": payload["resolved_data_id"],
                "native_card_cost": payload["card_cost"],
                "native_selection_form_index": payload[
                    "selection_form_index"
                ],
            })
        return calls

    def metadata(self, *, require_complete: bool = True) -> dict[str, Any]:
        if require_complete and not self.complete:
            raise DeploymentMaskContractError(
                "native deployment-mask capture is incomplete; missing "
                + ",".join(f"{side}:{deck}" for side, deck in self.missing_slots)
            )
        return {
            "schema_version": MASK_SCHEMA_VERSION,
            "kind": EPISODE_MASK_KIND,
            "complete": self.complete,
            "expected_slots": EXPECTED_SLOT_COUNT,
            "captured_slots": self.captured_slots,
            "base_probe_rpc_count": self.captured_slots,
            "probe_rpc_count": self.probe_rpc_count,
            "dynamic_label_probe_rpc_count": (
                self.dynamic_label_probe_rpc_count
            ),
            "capture_strategy": CAPTURE_STRATEGY,
            "dynamic_rule": DYNAMIC_RULE,
            "grid": {
                "width": ARENA_COLUMNS,
                "height": ARENA_ROWS,
                "cell_size": CELL_SIZE,
            },
            "content_path_rule": CONTENT_PATH_RULE,
            "entry_encoding": list(ENTRY_ENCODING),
            "variant_encoding": list(VARIANT_ENCODING),
            "entries": [
                [
                    self._entries[key]["side"],
                    self._entries[key]["deck_index"],
                    self._entries[key]["card_id"],
                    self._entries[key]["level"],
                    self._entries[key]["form_flags"],
                    self._entries[key]["capture_tick"],
                    self._entries[key]["content_sha256"],
                    self._entries[key]["native_selection_strategy"],
                    self._entries[key]["tick_variant_required"],
                    [
                        [variant["tick"], variant["content_sha256"]]
                        for variant in self._entries[key][
                            "dynamic_label_variants"
                        ]
                    ],
                ]
                for key in sorted(self._entries)
            ],
        }

    def audit_label_position(
        self,
        state: TickState | Mapping[str, Any],
        label: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Check one expert deployment point against the captured exact mask."""
        tick = (
            state.tick
            if isinstance(state, TickState)
            else _integer(state.get("tick"), "state.tick")
        )
        side = _integer(label.get("side"), "label.side")
        deck_index = _integer(label.get("deck_index"), "label.deck_index")
        x = _integer(label.get("x"), "label.x")
        y = _integer(label.get("y"), "label.y")
        entry = self._entries.get((side, deck_index))
        if entry is None:
            raise DeploymentMaskContractError(
                "deployment label has no captured deck-slot mask"
            )
        reference = resolve_deployment_reference(
            entry, tick=tick, require_dynamic_exact=True
        )
        assert reference is not None
        digest = str(reference["content_sha256"])
        payload = self._payloads.get(digest)
        if payload is None:
            raise DeploymentMaskContractError(
                "deployment label sidecar payload is not staged"
            )
        rows = derive_deployment_rows(
            payload, state, side=side, card_id=int(entry["card_id"])
        )
        legal = deployment_label_is_legal(rows, x=x, y=y)
        return {
            "tick": tick,
            "side": side,
            "deck_index": deck_index,
            "card_id": int(entry["card_id"]),
            "x": x,
            "y": y,
            "content_sha256": digest,
            "legal": legal,
        }


def validate_episode_mask_metadata(
    value: Mapping[str, Any], *, require_complete: bool = True
) -> dict[str, Any]:
    if value.get("kind") != EPISODE_MASK_KIND:
        raise DeploymentMaskContractError("episode mask metadata kind changed")
    if _integer(
        value.get("schema_version"), "episode_mask.schema_version"
    ) != MASK_SCHEMA_VERSION:
        raise DeploymentMaskContractError("episode mask metadata schema changed")
    if not isinstance(value.get("complete"), bool):
        raise DeploymentMaskContractError("episode mask complete flag is invalid")
    if require_complete and value.get("complete") is not True:
        raise DeploymentMaskContractError("episode mask capture is incomplete")
    captured_slots = _integer(value.get("captured_slots"), "captured_slots")
    base_probe_count = _integer(
        value.get("base_probe_rpc_count"), "base_probe_rpc_count"
    )
    if (
        _integer(value.get("expected_slots"), "expected_slots")
        != EXPECTED_SLOT_COUNT
        or not 1 <= captured_slots <= EXPECTED_SLOT_COUNT
        or base_probe_count != captured_slots
        or (require_complete and captured_slots != EXPECTED_SLOT_COUNT)
        or bool(value.get("complete"))
        != (captured_slots == EXPECTED_SLOT_COUNT)
    ):
        raise DeploymentMaskContractError("episode mask slot/RPC accounting is open")
    probe_rpc_count = _integer(value.get("probe_rpc_count"), "probe_rpc_count")
    dynamic_probe_count = _integer(
        value.get("dynamic_label_probe_rpc_count"),
        "dynamic_label_probe_rpc_count",
    )
    if (
        dynamic_probe_count < 0
        or probe_rpc_count != captured_slots + dynamic_probe_count
    ):
        raise DeploymentMaskContractError(
            "episode dynamic mask probe accounting is open"
        )
    if value.get("capture_strategy") != CAPTURE_STRATEGY:
        raise DeploymentMaskContractError("episode mask capture strategy changed")
    if value.get("dynamic_rule") != DYNAMIC_RULE:
        raise DeploymentMaskContractError("episode dynamic mask rule changed")
    if value.get("grid") != {
        "width": ARENA_COLUMNS, "height": ARENA_ROWS, "cell_size": CELL_SIZE,
    }:
        raise DeploymentMaskContractError("episode mask grid changed")
    if str(value.get("content_path_rule") or "") != CONTENT_PATH_RULE:
        raise DeploymentMaskContractError("episode mask content path rule changed")
    if tuple(value.get("entry_encoding") or ()) != ENTRY_ENCODING:
        raise DeploymentMaskContractError("episode mask entry encoding changed")
    if tuple(value.get("variant_encoding") or ()) != VARIANT_ENCODING:
        raise DeploymentMaskContractError("episode mask variant encoding changed")
    raw_entries = value.get("entries")
    if not isinstance(raw_entries, list) or len(raw_entries) != captured_slots:
        raise DeploymentMaskContractError(
            "episode mask metadata entry count disagrees with captured slots"
        )
    entries: list[dict[str, Any]] = []
    keys: set[tuple[int, int]] = set()
    for raw in raw_entries:
        if not isinstance(raw, list) or len(raw) != len(ENTRY_ENCODING):
            raise DeploymentMaskContractError(
                "episode mask entry does not match compact encoding"
            )
        (
            side_value,
            deck_index_value,
            card_id_value,
            level_value,
            form_flags_value,
            capture_tick_value,
            digest_value,
            strategy_value,
            tick_variant_required_value,
            raw_variants,
        ) = raw
        side = _integer(side_value, "entry.side")
        deck_index = _integer(deck_index_value, "entry.deck_index")
        card_id = _integer(card_id_value, "entry.card_id")
        level = (
            None
            if level_value is None
            else _integer(level_value, "entry.level")
        )
        form_flags = _integer(form_flags_value, "entry.form_flags")
        if (
            side not in EXPECTED_SIDES
            or deck_index not in EXPECTED_DECK_INDICES
            or card_id <= 0
            or (level is not None and level <= 0)
            or form_flags < 0
        ):
            raise DeploymentMaskContractError(
                "episode mask compact deck fields are invalid"
            )
        key = (side, deck_index)
        if key in keys:
            raise DeploymentMaskContractError("duplicate episode mask deck slot")
        keys.add(key)
        digest = _validate_sha256(digest_value, "content_sha256")
        capture_tick = _integer(capture_tick_value, "capture_tick")
        if capture_tick < 0:
            raise DeploymentMaskContractError("capture_tick cannot be negative")
        strategy = str(strategy_value or "")
        if strategy not in {
            "canonical", "native_dynamic_choice",
        }:
            raise DeploymentMaskContractError(
                "episode mask entry has unsupported native selection strategy"
            )
        if not isinstance(tick_variant_required_value, bool):
            raise DeploymentMaskContractError(
                "episode mask tick-variant flag is not boolean"
            )
        tick_variant_required = bool(tick_variant_required_value)
        if strategy == "native_dynamic_choice" and not tick_variant_required:
            raise DeploymentMaskContractError(
                "native dynamic-choice slot is not marked tick-variant"
            )
        if not isinstance(raw_variants, list):
            raise DeploymentMaskContractError(
                "episode mask entry lacks dynamic label variants"
            )
        variants: list[dict[str, Any]] = []
        variant_ticks: set[int] = set()
        for raw_variant in raw_variants:
            if (
                not isinstance(raw_variant, list)
                or len(raw_variant) != len(VARIANT_ENCODING)
            ):
                raise DeploymentMaskContractError(
                    "dynamic label variant does not match compact encoding"
                )
            variant_tick = _integer(raw_variant[0], "variant.tick")
            if variant_tick < 0 or variant_tick in variant_ticks:
                raise DeploymentMaskContractError(
                    "dynamic label variant Tick is invalid/duplicate"
                )
            variant_ticks.add(variant_tick)
            variant_sha = _validate_sha256(
                raw_variant[1], "variant.content_sha256"
            )
            variants.append({
                "tick": variant_tick,
                "content_sha256": variant_sha,
                "content_path": _sidecar_relative_path(variant_sha),
            })
        if (
            not tick_variant_required
            and variants
        ):
            raise DeploymentMaskContractError(
                "static slot unexpectedly has Tick label variants"
            )
        entries.append({
            "side": side,
            "deck_index": deck_index,
            "card_id": card_id,
            "level": level,
            "form_flags": form_flags,
            "capture_tick": capture_tick,
            "content_sha256": digest,
            "content_path": _sidecar_relative_path(digest),
            "native_selection_strategy": strategy,
            "tick_variant_required": tick_variant_required,
            "dynamic_label_variants": variants,
        })
    expected = {
        (side, deck_index)
        for side in EXPECTED_SIDES
        for deck_index in EXPECTED_DECK_INDICES
    }
    if not keys <= expected or (require_complete and keys != expected):
        raise DeploymentMaskContractError("episode mask deck slots are incomplete")
    observed_dynamic_probe_count = sum(
        int(variant["tick"]) != int(entry["capture_tick"])
        for entry in entries
        for variant in entry["dynamic_label_variants"]
    )
    if observed_dynamic_probe_count != dynamic_probe_count:
        raise DeploymentMaskContractError(
            "episode dynamic variant/probe accounting disagrees"
        )
    return {**dict(value), "entries": sorted(entries, key=lambda row: (
        int(row["side"]), int(row["deck_index"])
    ))}


class DeploymentMaskStore:
    """Atomic, idempotent content store shared by all generator workers."""

    def __init__(self, tick_store_root: Path, *, create: bool = True) -> None:
        self.tick_store_root = tick_store_root.resolve()
        self.root = self.tick_store_root / MASK_STORE_DIRECTORY
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
        elif not self.root.is_dir():
            raise DeploymentMaskContractError(
                "deployment-mask content directory is missing"
            )
        self._lock = threading.Lock()
        self._payload_cache: dict[str, dict[str, Any]] = {}

    def _process_key(self, digest: str) -> tuple[str, str]:
        return str(self.root), digest

    def path_for(self, content_sha256: str) -> Path:
        digest = _validate_sha256(content_sha256, "content_sha256")
        return self.root / digest[:2] / f"{digest}.json"

    def publish(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        normalized = normalize_sidecar(payload)
        raw = _canonical_bytes(normalized)
        digest = _sha256(raw)
        path = self.path_for(digest)
        with self._lock:
            cached = self._payload_cache.get(digest)
            if cached is not None:
                if cached != normalized:
                    raise DeploymentMaskContractError(
                        f"cached content-addressed mask diverged: {digest}"
                    )
                return {
                    "content_sha256": digest,
                    "content_path": _sidecar_relative_path(digest),
                    "bytes": len(raw),
                }
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if path.read_bytes() != raw:
                    raise DeploymentMaskContractError(
                        f"content-addressed mask diverged: {digest}"
                    )
            else:
                temporary = path.with_name(
                    path.name + f".{os.getpid()}.{threading.get_ident()}.tmp"
                )
                with temporary.open("wb") as output:
                    output.write(raw)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, path)
            self._payload_cache[digest] = normalized
            with _PROCESS_CACHE_LOCK:
                _PROCESS_PAYLOAD_CACHE[self._process_key(digest)] = normalized
        return {
            "content_sha256": digest,
            "content_path": _sidecar_relative_path(digest),
            "bytes": len(raw),
        }

    def publish_many(
        self, payloads: Mapping[str, Mapping[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for expected_sha, payload in sorted(payloads.items()):
            published = self.publish(payload)
            if published["content_sha256"] != expected_sha:
                raise DeploymentMaskContractError(
                    "staged deployment-mask content hash changed"
                )
            result[expected_sha] = published
        return result

    def load(
        self, content_sha256: str, *, allow_cached: bool = False
    ) -> dict[str, Any]:
        digest = _validate_sha256(content_sha256, "content_sha256")
        if allow_cached:
            with _PROCESS_CACHE_LOCK:
                cached = _PROCESS_PAYLOAD_CACHE.get(self._process_key(digest))
            if cached is None:
                with self._lock:
                    cached = self._payload_cache.get(digest)
            if cached is not None:
                return dict(cached)
        path = self.path_for(digest)
        try:
            raw = path.read_bytes()
        except FileNotFoundError as error:
            raise DeploymentMaskContractError(
                f"deployment-mask sidecar is missing: {digest}"
            ) from error
        if _sha256(raw) != digest:
            raise DeploymentMaskContractError(
                f"deployment-mask sidecar SHA changed: {digest}"
            )
        try:
            value = json.loads(raw)
        except Exception as error:
            raise DeploymentMaskContractError(
                f"deployment-mask sidecar JSON is invalid: {digest}"
            ) from error
        if not isinstance(value, Mapping):
            raise DeploymentMaskContractError("deployment-mask sidecar is not an object")
        normalized = normalize_sidecar(value)
        if _canonical_bytes(normalized) != raw:
            raise DeploymentMaskContractError(
                f"deployment-mask sidecar is not canonical: {digest}"
            )
        with self._lock:
            self._payload_cache[digest] = normalized
        with _PROCESS_CACHE_LOCK:
            _PROCESS_PAYLOAD_CACHE[self._process_key(digest)] = normalized
        return normalized

    def derive(
        self,
        content_sha256: str,
        state: TickState | Mapping[str, Any],
        *,
        side: int,
        card_id: int,
    ) -> tuple[str, ...]:
        """Return a bit-identical derived mask with tower-state memoization.

        Deployment projection depends only on the authenticated sidecar, actor
        side, and living tower footprints.  HP magnitude and all troop/effect
        state are irrelevant, so a normal battle has only a handful of cache
        states even though it contains thousands of native Ticks.
        """
        digest = _validate_sha256(content_sha256, "content_sha256")
        towers = _tower_rows(state)
        signature = tuple(
            sorted(
                (
                    int(tower["side"]),
                    int(tower["role"]),
                    int(tower["x"]),
                    int(tower["y"]),
                    int(int(tower["hp"]) > 0),
                )
                for tower in towers
            )
        )
        key = (str(self.root), digest, int(side), signature)
        with _PROCESS_CACHE_LOCK:
            cached = _PROCESS_DERIVED_CACHE.get(key)
        if cached is not None:
            return cached
        payload = self.load(digest, allow_cached=True)
        result = derive_deployment_rows(
            payload, state, side=side, card_id=card_id
        )
        with _PROCESS_CACHE_LOCK:
            previous = _PROCESS_DERIVED_CACHE.setdefault(key, result)
        if previous != result:
            raise DeploymentMaskContractError(
                "derived deployment-mask cache diverged"
            )
        return previous

    def verify_episode_metadata(
        self,
        metadata: Mapping[str, Any],
        *,
        allow_cached: bool = True,
        require_complete: bool = True,
    ) -> dict[str, Any]:
        value = metadata.get(EPISODE_METADATA_KEY)
        if not isinstance(value, Mapping):
            raise DeploymentMaskContractError(
                f"episode lacks {EPISODE_METADATA_KEY}"
            )
        normalized = validate_episode_mask_metadata(
            value, require_complete=require_complete
        )
        seen: set[str] = set()
        for entry in normalized["entries"]:
            references = [entry, *entry["dynamic_label_variants"]]
            for reference_index, reference in enumerate(references):
                digest = str(reference["content_sha256"])
                payload: dict[str, Any] | None = None
                if digest not in seen:
                    payload = self.load(digest, allow_cached=allow_cached)
                    seen.add(digest)
                if payload is None:
                    payload = self.load(digest, allow_cached=True)
                expected_strategy = (
                    str(entry["native_selection_strategy"])
                    if reference_index == 0
                    else str(entry["native_selection_strategy"])
                )
                if payload["selection_strategy"] != expected_strategy:
                    raise DeploymentMaskContractError(
                        "episode mask reference selection strategy disagrees"
                    )
                if reference_index == 0:
                    inferred_tick_variant = bool(
                        payload["selection_strategy"]
                        == "native_dynamic_choice"
                        or int(entry["card_id"]) == MIRROR_CARD_ID
                        or int(payload["resolved_data_id"])
                        != int(entry["card_id"])
                    )
                    if inferred_tick_variant != bool(
                        entry["tick_variant_required"]
                    ):
                        raise DeploymentMaskContractError(
                            "episode mask tick-variant classification disagrees"
                        )
        return normalized

    def build_manifest(self) -> dict[str, Any]:
        entries: list[dict[str, Any]] = []
        for path in sorted(self.root.glob("[0-9a-f][0-9a-f]/*.json")):
            if path.name == "manifest.json":
                continue
            digest = path.stem
            payload = self.load(digest)
            entries.append({
                "content_sha256": digest,
                "content_path": _sidecar_relative_path(digest),
                "bytes": path.stat().st_size,
                "valid_cells": int(payload["valid_cells"]),
            })
        content_sha = _sha256(_canonical_bytes({"entries": entries}))
        manifest = {
            "schema_version": MASK_SCHEMA_VERSION,
            "kind": MASK_STORE_KIND,
            "sidecars": len(entries),
            "bytes": sum(int(entry["bytes"]) for entry in entries),
            "content_sha256": content_sha,
            "entries": entries,
        }
        path = self.root / "manifest.json"
        raw = _canonical_bytes(manifest)
        temporary = path.with_name(
            path.name + f".{os.getpid()}.{threading.get_ident()}.tmp"
        )
        with temporary.open("wb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        return manifest

    def verify_manifest(self) -> dict[str, Any]:
        path = self.root / "manifest.json"
        try:
            raw = path.read_bytes()
            value = json.loads(raw)
        except Exception as error:
            raise DeploymentMaskContractError("deployment-mask manifest is missing/invalid") from error
        if not isinstance(value, Mapping) or value.get("kind") != MASK_STORE_KIND:
            raise DeploymentMaskContractError("deployment-mask manifest kind changed")
        entries = value.get("entries")
        if not isinstance(entries, list):
            raise DeploymentMaskContractError("deployment-mask manifest entries missing")
        if int(value.get("sidecars", -1)) != len(entries):
            raise DeploymentMaskContractError("deployment-mask manifest count changed")
        total_bytes = 0
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise DeploymentMaskContractError("malformed mask manifest entry")
            digest = _validate_sha256(entry.get("content_sha256"), "content_sha256")
            if str(entry.get("content_path") or "") != _sidecar_relative_path(digest):
                raise DeploymentMaskContractError("mask manifest path changed")
            payload = self.load(digest)
            path_size = self.path_for(digest).stat().st_size
            if path_size != int(entry.get("bytes", -1)):
                raise DeploymentMaskContractError("mask manifest byte count changed")
            if int(payload["valid_cells"]) != int(entry.get("valid_cells", -1)):
                raise DeploymentMaskContractError("mask manifest payload summary changed")
            total_bytes += path_size
        if total_bytes != int(value.get("bytes", -1)):
            raise DeploymentMaskContractError("mask store total bytes changed")
        expected_content = _sha256(_canonical_bytes({"entries": entries}))
        if str(value.get("content_sha256") or "") != expected_content:
            raise DeploymentMaskContractError("mask store content digest changed")
        if _canonical_bytes(dict(value)) != raw:
            raise DeploymentMaskContractError("deployment-mask manifest is not canonical")
        return dict(value)


def _tower_rows(state: TickState | Mapping[str, Any]) -> list[dict[str, int]]:
    if isinstance(state, TickState):
        return [
            {
                "side": tower.side,
                "role": tower.role,
                "x": tower.x,
                "y": tower.y,
                "hp": tower.hp,
            }
            for tower in state.towers
        ]
    episode = state.get("episode")
    raw_towers = (
        episode.get("crown_towers", [])
        if isinstance(episode, Mapping) else []
    )
    result = []
    for tower in raw_towers:
        if not isinstance(tower, Mapping):
            raise DeploymentMaskContractError("crown tower is not an object")
        role_name = str(tower.get("type") or "").lower()
        role = 0 if "king" in role_name else 1 if "princess" in role_name else -1
        side = _integer(tower.get("side"), "tower.side")
        if side not in EXPECTED_SIDES or role < 0:
            raise DeploymentMaskContractError("invalid crown tower identity")
        result.append({
            "side": side,
            "role": role,
            "x": _integer(tower.get("x"), "tower.x"),
            "y": _integer(tower.get("y"), "tower.y"),
            "hp": _integer(tower.get("hp"), "tower.hp"),
        })
    return result


def derive_deployment_rows(
    sidecar: Mapping[str, Any],
    state: TickState | Mapping[str, Any],
    *,
    side: int,
    card_id: int,
) -> tuple[str, ...]:
    """Reconstruct the exact cached online mask from a base sidecar+Tick."""
    if side not in EXPECTED_SIDES:
        raise DeploymentMaskContractError("side must be 0 or 1")
    payload = normalize_sidecar(sidecar)
    native_rows = payload["rows"]
    # Public/base spell identity remains authoritative for deployment-mask
    # *category*: hero/evolution wrappers can resolve to a 203-series form ID
    # without turning a spell such as Barbarian Barrel into a troop footprint.
    # Mirror is the deliberate exception: it follows its exact Tick-resolved
    # troop/building/spell selection rather than its public wrapper namespace.
    # Miner/Goblin Drill remain explicit global-deploy exceptions in the
    # ordinary troop/building namespaces.
    resolved_data_id = int(payload["resolved_data_id"])
    if resolved_data_id <= 0:
        raise DeploymentMaskContractError(
            "native resolved_data_id is invalid for deployment semantics"
        )
    if (
        (
            int(card_id) // 1_000_000 == 28
            and int(card_id) != MIRROR_CARD_ID
        )
        or resolved_data_id // 1_000_000 == 28
        or int(card_id) in GLOBAL_DEPLOY_CARD_IDS
        or resolved_data_id in GLOBAL_DEPLOY_CARD_IDS
    ):
        return tuple(native_rows)

    result = [
        [
            all(
                native_rows[source_row][source_column] == "1"
                for source_row, source_column in (
                    (row, column),
                    (row, ARENA_COLUMNS - 1 - column),
                    (ARENA_ROWS - 1 - row, column),
                    (ARENA_ROWS - 1 - row, ARENA_COLUMNS - 1 - column),
                )
            )
            for column in range(ARENA_COLUMNS)
        ]
        for row in range(ARENA_ROWS)
    ]
    towers = _tower_rows(state)
    allowed = [[False] * ARENA_COLUMNS for _ in range(ARENA_ROWS)]
    own_rows = range(0, 15) if side == 0 else range(17, ARENA_ROWS)
    for row in own_rows:
        allowed[row] = [True] * ARENA_COLUMNS
    living_enemy_princesses = [
        tower for tower in towers
        if tower["side"] != side and tower["role"] == 1 and tower["hp"] > 0
    ]
    left_alive = any(tower["x"] < 9000 for tower in living_enemy_princesses)
    right_alive = any(tower["x"] >= 9000 for tower in living_enemy_princesses)
    # Native/source coordinates use the river-edge half-cell boundary.  The
    # unlocked five-row pocket is 16..20 for side 0 and its exact 180-degree
    # mirror 11..15 for side 1.  Starting at 17/ending at 14 rejects valid
    # source placements such as (3500, 16501) after the left tower falls.
    pocket_rows = (
        range(16, 16 + POCKET_DEPTH_CELLS)
        if side == 0
        else range(16 - POCKET_DEPTH_CELLS, 16)
    )
    if not left_alive:
        for row in pocket_rows:
            for column in range(0, LANE_SPLIT_COLUMN):
                allowed[row][column] = True
    if not right_alive:
        for row in pocket_rows:
            for column in range(LANE_SPLIT_COLUMN, ARENA_COLUMNS):
                allowed[row][column] = True
    for tower in towers:
        if tower["hp"] <= 0:
            continue
        footprint = 4 if tower["role"] == 0 else 3
        half_extent = footprint * CELL_SIZE // 2
        column_start = max(0, (tower["x"] - half_extent) // CELL_SIZE)
        column_stop = min(
            ARENA_COLUMNS, (tower["x"] + half_extent + CELL_SIZE - 1) // CELL_SIZE
        )
        row_start = max(0, (tower["y"] - half_extent) // CELL_SIZE)
        row_stop = min(
            ARENA_ROWS, (tower["y"] + half_extent + CELL_SIZE - 1) // CELL_SIZE
        )
        for row in range(row_start, row_stop):
            for column in range(column_start, column_stop):
                allowed[row][column] = False
    return tuple(
        "".join(
            "1" if result[row][column] and allowed[row][column] else "0"
            for column in range(ARENA_COLUMNS)
        )
        for row in range(ARENA_ROWS)
    )


def deployment_label_is_legal(rows: Sequence[str], *, x: int, y: int) -> bool:
    if len(rows) != ARENA_ROWS or any(len(row) != ARENA_COLUMNS for row in rows):
        raise DeploymentMaskContractError("derived mask is not 18x32")
    if not 0 <= x < ARENA_COLUMNS * CELL_SIZE or not 0 <= y < ARENA_ROWS * CELL_SIZE:
        return False
    return rows[y // CELL_SIZE][x // CELL_SIZE] == "1"


def resolve_deployment_reference(
    entry: Mapping[str, Any],
    *,
    tick: int,
    require_dynamic_exact: bool = True,
) -> Mapping[str, Any] | None:
    """Resolve one deck entry for a supervised Tick without guessing.

    Static selections use the base reference.  A tick-variant wrapper
    (including native dynamic-choice cards and canonical Mirror) returns its
    exact play-Tick variant.  At non-card-supervised Ticks a caller may pass
    ``require_dynamic_exact=False`` and receive ``None`` so it can mask that
    slot out while retaining independent timing supervision.
    """
    if entry.get("tick_variant_required") is not True:
        return entry
    variants = [
        variant for variant in entry.get("dynamic_label_variants", [])
        if isinstance(variant, Mapping) and int(variant.get("tick", -1)) == tick
    ]
    if len(variants) == 1:
        return variants[0]
    if not require_dynamic_exact and not variants:
        return None
    raise DeploymentMaskContractError(
        "tick-variant card supervision lacks one exact native play-Tick variant"
    )


def verify_deployment_labels(
    states: Iterable[TickState],
    episode_metadata: Mapping[str, Any],
    store: DeploymentMaskStore,
    labels: Iterable[Mapping[str, Any]],
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Offline, fail-closed label legality audit with no native RPCs."""
    metadata = store.verify_episode_metadata(
        episode_metadata, require_complete=require_complete
    )
    references = {
        (int(entry["side"]), int(entry["deck_index"])): entry
        for entry in metadata["entries"]
    }
    by_tick = {state.tick: state for state in states}
    checked = 0
    violations: list[dict[str, Any]] = []
    for raw in labels:
        tick = _integer(raw.get("tick"), "label.tick")
        side = _integer(raw.get("side"), "label.side")
        deck_index = _integer(raw.get("deck_index"), "label.deck_index")
        x = _integer(raw.get("x"), "label.x")
        y = _integer(raw.get("y"), "label.y")
        state = by_tick.get(tick)
        reference = references.get((side, deck_index))
        if state is None or reference is None:
            raise DeploymentMaskContractError(
                f"label lacks exact Tick/mask reference: {tick}/{side}/{deck_index}"
            )
        selected_reference = resolve_deployment_reference(
            reference, tick=tick, require_dynamic_exact=True
        )
        assert selected_reference is not None
        digest = str(selected_reference["content_sha256"])
        payload = store.load(digest, allow_cached=True)
        rows = store.derive(
            digest, state, side=side, card_id=int(reference["card_id"])
        )
        player = state.players[side]
        reasons = []
        if deck_index not in player.hand:
            reasons.append("deck_index_not_in_hand")
        if state.tick < 100:
            reasons.append("deployment_gate_not_open")
        if not state.episode.commands_allowed:
            reasons.append("native_commands_not_allowed")
        if player.elixir_raw < int(payload["card_cost_raw"]):
            reasons.append("insufficient_native_elixir")
        if not deployment_label_is_legal(rows, x=x, y=y):
            reasons.append("position_not_in_derived_native_mask")
        legal = not reasons
        checked += 1
        if not legal:
            violations.append({
                "tick": tick, "side": side, "deck_index": deck_index,
                "card_id": int(reference["card_id"]), "x": x, "y": y,
                "reasons": reasons,
            })
    return {
        "schema_version": MASK_SCHEMA_VERSION,
        "kind": "cr_native_deployment_label_audit_v1",
        "checked": checked,
        "legal": checked - len(violations),
        "violations": violations,
        "all_legal": not violations,
        "native_rpc_count": 0,
        "dynamic_rule": DYNAMIC_RULE,
    }


__all__ = [
    "ARENA_COLUMNS",
    "ARENA_ROWS",
    "BASE_MASK_KIND",
    "CAPTURE_STRATEGY",
    "CELL_SIZE",
    "DYNAMIC_RULE",
    "DeploymentMaskContractError",
    "DeploymentMaskStore",
    "EPISODE_MASK_KIND",
    "EPISODE_METADATA_KEY",
    "EXPECTED_SLOT_COUNT",
    "MASK_SCHEMA_VERSION",
    "NativeDeploymentMaskCapture",
    "deployment_label_is_legal",
    "derive_deployment_rows",
    "normalize_native_probe",
    "resolve_deployment_reference",
    "validate_episode_mask_metadata",
    "verify_deployment_labels",
]
