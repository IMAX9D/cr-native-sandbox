"""On-disk contract for native expert behaviour-cloning shards.

The actor schema is intentionally public-information only.  Hidden opponent
hands, exact opponent elixir and any privileged libg state are prohibited.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = 1
DATASET_KIND = "cr_native_expert_bc_dataset_v1"
SHARD_KIND = "cr_native_expert_bc_shard_v1"
OBSERVATION_NATIVE = "native_state_v1"
OBSERVATION_SEQUENCE = "sequence_only_v1"
ARENA_ROWS = 32
ARENA_COLUMNS = 18
POSITION_COUNT = ARENA_ROWS * ARENA_COLUMNS
POSITION_MASK_BYTES = POSITION_COUNT // 8
DECK_SIZE = 8
HAND_SIZE = 4

# A shard is a directory of plain .npy arrays.  This keeps large arrays
# memory-mappable and avoids loading a compressed archive into every worker.
NATIVE_REQUIRED_ARRAYS = {
    "sequence_offsets",
    "grid",
    "public_scalars",
    "own_deck_tokens",
    "hand_tokens",
    "next_card_token",
    "revealed_enemy_tokens",
    "ability_tokens",
    "delta_ticks",
    "timing_exposure_ticks",
    "card_mask",
    "action_kind_mask",
    "ability_mask",
    "selected_position_mask_packed",
    "ability_position_mask_packed",
    "play_now",
    "action_kind",
    "card_slot",
    "position",
    "ability_slot",
    "ability_position",
    "timing_label_mask",
    "kind_label_mask",
    "card_label_mask",
    "position_label_mask",
    "ability_label_mask",
    "ability_position_label_mask",
    "sample_weight",
}

# Sequence-only data is deliberately a different physical contract.  In
# particular it does not contain a fabricated all-zero native grid, native
# legality masks, abilities or privileged state.  The three previous-event
# fields are public replay history and let the recurrent model retain spatial
# and opponent-play context without pretending to know the libg scene.
SEQUENCE_REQUIRED_ARRAYS = {
    "sequence_offsets",
    "public_scalars",
    "own_deck_tokens",
    "hand_tokens",
    "next_card_token",
    "revealed_enemy_tokens",
    "previous_event_card_token",
    "previous_event_side",
    "previous_event_position",
    "delta_ticks",
    "timing_exposure_ticks",
    "card_mask",
    "play_now",
    "card_slot",
    "position",
    "timing_label_mask",
    "card_label_mask",
    "position_label_mask",
    "sample_weight",
}

# Backwards-compatible public name used by existing native smoke fixtures.
REQUIRED_ARRAYS = NATIVE_REQUIRED_ARRAYS

# Fail closed if a compiler accidentally adds privileged arrays to an actor
# dataset.  Future public fields must be explicitly added to REQUIRED_ARRAYS.
FORBIDDEN_FRAGMENTS = (
    "enemy_hand",
    "opponent_hand",
    "enemy_elixir",
    "opponent_elixir",
    "privileged",
    "hidden_state",
    "native_rng",
    "unrevealed",
)


class DatasetContractError(ValueError):
    """The compiled training data violates the actor data contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.is_file():
        raise DatasetContractError(f"dataset manifest is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    validate_manifest(value, root=root)
    return value


def validate_manifest(value: Mapping[str, Any], *, root: Path) -> None:
    if value.get("kind") != DATASET_KIND:
        raise DatasetContractError("unexpected dataset kind")
    if int(value.get("schema_version", -1)) != SCHEMA_VERSION:
        raise DatasetContractError("unsupported expert dataset schema")
    observation_mode = str(value.get("observation_mode") or OBSERVATION_NATIVE)
    if observation_mode not in (OBSERVATION_NATIVE, OBSERVATION_SEQUENCE):
        raise DatasetContractError(f"unsupported observation mode: {observation_mode}")
    dimensions = value.get("dimensions") or {}
    for key in ("public_scalar_size", "card_vocab_size"):
        if int(dimensions.get(key, 0)) <= 0:
            raise DatasetContractError(f"invalid dataset dimension: {key}")
    grid_channels = int(dimensions.get("grid_channels", -1))
    if observation_mode == OBSERVATION_NATIVE and grid_channels <= 0:
        raise DatasetContractError("native-state datasets require grid channels")
    if observation_mode == OBSERVATION_SEQUENCE and grid_channels != 0:
        raise DatasetContractError("sequence-only datasets must declare zero grid channels")
    if int(dimensions.get("ability_vocab_size", 0)) <= 0:
        raise DatasetContractError("ability vocab must reserve at least PAD")
    if int(dimensions.get("max_ability_slots", 0)) <= 0:
        raise DatasetContractError("max_ability_slots must be positive")
    splits = value.get("splits")
    if not isinstance(splits, Mapping):
        raise DatasetContractError("splits must be an object")
    if not all(name in splits for name in ("train", "validation", "test")):
        raise DatasetContractError("train/validation/test splits are required")
    seen: set[str] = set()
    for split, shards in splits.items():
        if not isinstance(shards, list):
            raise DatasetContractError(f"split {split} must be a list")
        for relative in shards:
            relative = str(relative).replace("\\", "/")
            if relative in seen:
                raise DatasetContractError(f"shard appears in multiple splits: {relative}")
            seen.add(relative)
            path = (root / relative).resolve()
            if root.resolve() not in path.parents:
                raise DatasetContractError(f"shard escapes dataset root: {relative}")
            if not path.is_dir():
                raise DatasetContractError(f"shard directory is missing: {path}")
    contract = value.get("split_contract") or {}
    if contract.get("battle_tag_disjoint") is not True:
        raise DatasetContractError("battle_tag_disjoint must be proven true")
    if contract.get("source_file_disjoint") is not True:
        raise DatasetContractError("source_file_disjoint must be proven true")
    if value.get("actor_information") != "public_only_v1":
        raise DatasetContractError("actor information contract must be public_only_v1")
    provenance = value.get("state_provenance") or {}
    if observation_mode == OBSERVATION_SEQUENCE:
        if provenance.get("mode") != "sequence_only":
            raise DatasetContractError("sequence-only provenance must be explicit")
        if int(provenance.get("native_grid_rows", -1)) != 0:
            raise DatasetContractError("sequence-only data cannot claim native grid rows")
        if value.get("native_replay_validated") is not False:
            raise DatasetContractError(
                "sequence-only data must not claim native replay validation"
            )
        if value.get("timing_target") != "piecewise_exponential_event_v1":
            raise DatasetContractError("unexpected sequence-only timing target")
    feature_schema = value.get("feature_schema") or {}
    grid_names = feature_schema.get("grid_channels")
    scalar_names = feature_schema.get("public_scalars")
    if not isinstance(grid_names, list) or len(grid_names) != int(dimensions["grid_channels"]):
        raise DatasetContractError("grid channel names must describe every channel")
    if not isinstance(scalar_names, list) or len(scalar_names) != int(dimensions["public_scalar_size"]):
        raise DatasetContractError("public scalar names must describe every scalar")
    feature_names = [str(name).lower() for name in grid_names + scalar_names]
    forbidden_features = sorted(
        name
        for name in feature_names
        if any(fragment in name for fragment in FORBIDDEN_FRAGMENTS)
    )
    if forbidden_features:
        raise DatasetContractError(
            f"privileged actor features are forbidden: {forbidden_features}"
        )


def required_arrays(manifest: Mapping[str, Any]) -> set[str]:
    mode = str(manifest.get("observation_mode") or OBSERVATION_NATIVE)
    return SEQUENCE_REQUIRED_ARRAYS if mode == OBSERVATION_SEQUENCE else NATIVE_REQUIRED_ARRAYS


def _load_arrays(
    shard: Path,
    manifest: Mapping[str, Any],
    *,
    mmap: bool = True,
) -> dict[str, np.ndarray]:
    names = {path.stem for path in shard.glob("*.npy")}
    forbidden = sorted(
        name for name in names if any(fragment in name.lower() for fragment in FORBIDDEN_FRAGMENTS)
    )
    if forbidden:
        raise DatasetContractError(f"privileged arrays are forbidden: {forbidden}")
    approved = required_arrays(manifest)
    missing = sorted(approved - names)
    extra = sorted(names - approved)
    if missing:
        raise DatasetContractError(f"missing arrays in {shard}: {missing}")
    if extra:
        raise DatasetContractError(f"unapproved arrays in {shard}: {extra}")
    mode = "r" if mmap else None
    return {name: np.load(shard / f"{name}.npy", mmap_mode=mode) for name in names}


def validate_shard(shard: Path, manifest: Mapping[str, Any]) -> dict[str, int]:
    arrays = _load_arrays(shard, manifest, mmap=True)
    offsets = arrays["sequence_offsets"]
    if offsets.ndim != 1 or len(offsets) < 2 or int(offsets[0]) != 0:
        raise DatasetContractError(f"bad sequence offsets: {shard}")
    if np.any(np.diff(offsets) <= 0):
        raise DatasetContractError(f"empty or reversed sequence in {shard}")
    rows = int(offsets[-1])
    dimensions = manifest["dimensions"]
    if str(manifest.get("observation_mode") or OBSERVATION_NATIVE) == OBSERVATION_SEQUENCE:
        return _validate_sequence_shard(shard, manifest, arrays, offsets, rows)
    grid_shape = (
        rows,
        int(dimensions["grid_channels"]),
        ARENA_ROWS,
        ARENA_COLUMNS,
    )
    shapes = {
        "grid": grid_shape,
        "public_scalars": (rows, int(dimensions["public_scalar_size"])),
        "own_deck_tokens": (rows, DECK_SIZE),
        "hand_tokens": (rows, HAND_SIZE),
        "next_card_token": (rows,),
        "revealed_enemy_tokens": (rows, DECK_SIZE),
        "ability_tokens": (rows, int(dimensions["max_ability_slots"])),
        "card_mask": (rows, HAND_SIZE),
        "action_kind_mask": (rows, 2),
        "ability_mask": (rows, int(dimensions["max_ability_slots"])),
        "selected_position_mask_packed": (rows, POSITION_MASK_BYTES),
        "ability_position_mask_packed": (rows, POSITION_MASK_BYTES),
    }
    one_dimensional = REQUIRED_ARRAYS - set(shapes) - {"sequence_offsets"}
    shapes.update({name: (rows,) for name in one_dimensional})
    for name, expected in shapes.items():
        if tuple(arrays[name].shape) != expected:
            raise DatasetContractError(
                f"{shard}/{name}.npy shape {arrays[name].shape} != {expected}"
            )
    if len(offsets) and rows != arrays["grid"].shape[0]:
        raise DatasetContractError("offset terminal does not match row count")
    if arrays["grid"].dtype != np.uint8:
        raise DatasetContractError("grid must be uint8")
    for name in ("own_deck_tokens", "hand_tokens", "next_card_token", "revealed_enemy_tokens"):
        values = arrays[name]
        if values.min(initial=0) < 0 or values.max(initial=0) >= int(dimensions["card_vocab_size"]):
            raise DatasetContractError(f"card token outside vocabulary: {name}")
    ability_tokens = arrays["ability_tokens"]
    if ability_tokens.min(initial=0) < 0 or ability_tokens.max(initial=0) >= int(dimensions["ability_vocab_size"]):
        raise DatasetContractError("ability token outside vocabulary")
    ability_mask_values = arrays["ability_mask"].astype(bool)
    if np.any(ability_mask_values & (ability_tokens == 0)):
        raise DatasetContractError("PAD ability slot cannot be legal")

    valid = arrays["timing_label_mask"].astype(bool)
    if np.any(arrays["timing_exposure_ticks"][valid] <= 0):
        raise DatasetContractError("timing exposure must be positive")
    if np.any(~np.isfinite(arrays["public_scalars"])):
        raise DatasetContractError("public scalar contains NaN/Inf")
    if np.any(~np.isfinite(arrays["sample_weight"])) or np.any(arrays["sample_weight"] < 0):
        raise DatasetContractError("sample weights must be finite and non-negative")
    if np.any(arrays["sample_weight"][valid] <= 0):
        raise DatasetContractError("supervised timing rows require positive weight")
    own_deck = arrays["own_deck_tokens"]
    hand = arrays["hand_tokens"]
    next_card = arrays["next_card_token"]
    if np.any(own_deck == 0) or np.any(hand == 0) or np.any(next_card == 0):
        raise DatasetContractError("own deck, current hand and next card cannot contain PAD")
    if np.any(np.sort(own_deck, axis=1)[:, 1:] == np.sort(own_deck, axis=1)[:, :-1]):
        raise DatasetContractError("own deck contains duplicate card tokens")
    if np.any(np.sort(hand, axis=1)[:, 1:] == np.sort(hand, axis=1)[:, :-1]):
        raise DatasetContractError("current hand contains duplicate card tokens")
    hand_in_deck = (hand[:, :, None] == own_deck[:, None, :]).any(axis=-1)
    if np.any(~hand_in_deck):
        raise DatasetContractError("current hand is not a subset of own deck")
    if np.any(~(next_card[:, None] == own_deck).any(axis=-1)):
        raise DatasetContractError("next card is not in own deck")
    if np.any((next_card[:, None] == hand).any(axis=-1)):
        raise DatasetContractError("next card is already in current hand")
    card_mask = arrays["card_mask"].astype(bool)
    if np.any(card_mask & (hand == 0)):
        raise DatasetContractError("PAD hand slot cannot be legal")
    kind_rows = arrays["kind_label_mask"].astype(bool)
    kinds = arrays["action_kind"].astype(np.int64)
    if np.any((kinds[kind_rows] < 0) | (kinds[kind_rows] >= 2)):
        raise DatasetContractError("action kind label outside deploy/ability")
    if np.any(~arrays["action_kind_mask"][np.flatnonzero(kind_rows), kinds[kind_rows]].astype(bool)):
        raise DatasetContractError("expert action kind is masked illegal")
    play_now = arrays["play_now"].astype(bool)
    available_kinds = arrays["action_kind_mask"].astype(bool)
    if np.any(available_kinds[:, 0] != card_mask.any(axis=-1)):
        raise DatasetContractError("deploy availability disagrees with card mask")
    if np.any(available_kinds[:, 1] != ability_mask_values.any(axis=-1)):
        raise DatasetContractError("ability availability disagrees with ability mask")
    if np.any(valid & ~available_kinds.any(axis=-1)):
        raise DatasetContractError("timing label exists when no action kind is legal")
    if np.any(kind_rows & ~play_now):
        raise DatasetContractError("conditional action-kind label requires play_now")
    card_rows = arrays["card_label_mask"].astype(bool)
    slots = arrays["card_slot"].astype(np.int64)
    if np.any((slots[card_rows] < 0) | (slots[card_rows] >= HAND_SIZE)):
        raise DatasetContractError("card label outside hand slots")
    if np.any(~arrays["card_mask"][np.flatnonzero(card_rows), slots[card_rows]].astype(bool)):
        raise DatasetContractError("expert card is masked illegal")
    if np.any(card_rows & (~kind_rows | (kinds != 0))):
        raise DatasetContractError("card label requires a deploy action-kind label")
    position_rows = arrays["position_label_mask"].astype(bool)
    positions = arrays["position"].astype(np.int64)
    if np.any((positions[position_rows] < 0) | (positions[position_rows] >= POSITION_COUNT)):
        raise DatasetContractError("position label outside arena")
    if np.any(position_rows):
        unpacked = np.unpackbits(
            arrays["selected_position_mask_packed"][position_rows],
            axis=-1,
            count=POSITION_COUNT,
            bitorder="little",
        ).astype(bool)
        if np.any(~unpacked[np.arange(len(unpacked)), positions[position_rows]]):
            raise DatasetContractError("expert position is masked illegal")
    if np.any(position_rows & ~card_rows):
        raise DatasetContractError("position label requires a card label")
    ability_rows = arrays["ability_label_mask"].astype(bool)
    ability_slots = arrays["ability_slot"].astype(np.int64)
    max_slots = int(dimensions["max_ability_slots"])
    if np.any((ability_slots[ability_rows] < 0) | (ability_slots[ability_rows] >= max_slots)):
        raise DatasetContractError("ability label outside candidate slots")
    if np.any(~arrays["ability_mask"][np.flatnonzero(ability_rows), ability_slots[ability_rows]].astype(bool)):
        raise DatasetContractError("expert ability is masked illegal")
    if np.any(ability_rows & (~kind_rows | (kinds != 1))):
        raise DatasetContractError("ability label requires an ability action-kind label")
    ability_position_rows = arrays["ability_position_label_mask"].astype(bool)
    if np.any(ability_position_rows & ~ability_rows):
        raise DatasetContractError("ability position label requires an ability label")
    ability_positions = arrays["ability_position"].astype(np.int64)
    if np.any(
        (ability_positions[ability_position_rows] < 0)
        | (ability_positions[ability_position_rows] >= POSITION_COUNT)
    ):
        raise DatasetContractError("ability position label outside arena")
    if np.any(ability_position_rows):
        unpacked = np.unpackbits(
            arrays["ability_position_mask_packed"][ability_position_rows],
            axis=-1,
            count=POSITION_COUNT,
            bitorder="little",
        ).astype(bool)
        if np.any(
            ~unpacked[
                np.arange(len(unpacked)), ability_positions[ability_position_rows]
            ]
        ):
            raise DatasetContractError("expert ability position is masked illegal")
    any_supervision = (
        valid
        | kind_rows
        | card_rows
        | position_rows
        | ability_rows
        | ability_position_rows
    )
    if np.any(arrays["sample_weight"][any_supervision] <= 0):
        raise DatasetContractError("every supervised row requires positive sample weight")
    return {"sequences": len(offsets) - 1, "rows": rows}


def unpack_position_masks(packed: np.ndarray) -> np.ndarray:
    return np.unpackbits(
        packed,
        axis=-1,
        count=POSITION_COUNT,
        bitorder="little",
    ).astype(np.bool_)


def _validate_sequence_shard(
    shard: Path,
    manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    offsets: np.ndarray,
    rows: int,
) -> dict[str, int]:
    dimensions = manifest["dimensions"]
    shapes = {
        "public_scalars": (rows, int(dimensions["public_scalar_size"])),
        "own_deck_tokens": (rows, DECK_SIZE),
        "hand_tokens": (rows, HAND_SIZE),
        "next_card_token": (rows,),
        "revealed_enemy_tokens": (rows, DECK_SIZE),
        "card_mask": (rows, HAND_SIZE),
    }
    shapes.update(
        {
            name: (rows,)
            for name in SEQUENCE_REQUIRED_ARRAYS - set(shapes) - {"sequence_offsets"}
        }
    )
    for name, expected in shapes.items():
        if tuple(arrays[name].shape) != expected:
            raise DatasetContractError(
                f"{shard}/{name}.npy shape {arrays[name].shape} != {expected}"
            )
    vocab_size = int(dimensions["card_vocab_size"])
    for name in (
        "own_deck_tokens",
        "hand_tokens",
        "next_card_token",
        "revealed_enemy_tokens",
        "previous_event_card_token",
    ):
        values = arrays[name]
        if values.min(initial=0) < 0 or values.max(initial=0) >= vocab_size:
            raise DatasetContractError(f"card token outside vocabulary: {name}")
    previous_side = arrays["previous_event_side"]
    if np.any((previous_side < 0) | (previous_side > 2)):
        raise DatasetContractError("previous event side outside NONE/OWN/ENEMY")
    previous_position = arrays["previous_event_position"]
    if np.any((previous_position < 0) | (previous_position > POSITION_COUNT)):
        raise DatasetContractError("previous event position outside arena/sentinel")
    if np.any(~np.isfinite(arrays["public_scalars"])):
        raise DatasetContractError("public scalar contains NaN/Inf")
    weights = arrays["sample_weight"]
    if np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise DatasetContractError("sample weights must be finite and non-negative")
    timing_rows = arrays["timing_label_mask"].astype(bool)
    if np.any(arrays["timing_exposure_ticks"][timing_rows] <= 0):
        raise DatasetContractError("timing exposure must be positive")
    if np.any(weights[timing_rows] <= 0):
        raise DatasetContractError("supervised timing rows require positive weight")

    own_deck = arrays["own_deck_tokens"]
    hand = arrays["hand_tokens"]
    next_card = arrays["next_card_token"]
    if np.any(own_deck == 0) or np.any(hand == 0) or np.any(next_card == 0):
        raise DatasetContractError("exact cycle fields cannot contain PAD")
    if np.any(np.sort(own_deck, axis=1)[:, 1:] == np.sort(own_deck, axis=1)[:, :-1]):
        raise DatasetContractError("own deck contains duplicate card tokens")
    if np.any(np.sort(hand, axis=1)[:, 1:] == np.sort(hand, axis=1)[:, :-1]):
        raise DatasetContractError("current hand contains duplicate card tokens")
    if np.any(~(hand[:, :, None] == own_deck[:, None, :]).any(axis=-1)):
        raise DatasetContractError("current hand is not a subset of own deck")
    if np.any(~(next_card[:, None] == own_deck).any(axis=-1)):
        raise DatasetContractError("next card is not in own deck")
    if np.any((next_card[:, None] == hand).any(axis=-1)):
        raise DatasetContractError("next card is already in current hand")

    card_rows = arrays["card_label_mask"].astype(bool)
    position_rows = arrays["position_label_mask"].astype(bool)
    play_now = arrays["play_now"].astype(bool)
    slots = arrays["card_slot"].astype(np.int64)
    if np.any(card_rows & ~play_now):
        raise DatasetContractError("card label requires an observed play event")
    if np.any((slots[card_rows] < 0) | (slots[card_rows] >= HAND_SIZE)):
        raise DatasetContractError("card label outside hand slots")
    if np.any(~arrays["card_mask"][np.flatnonzero(card_rows), slots[card_rows]].astype(bool)):
        raise DatasetContractError("expert card is outside the exact hand")
    positions = arrays["position"].astype(np.int64)
    if np.any((positions[position_rows] < 0) | (positions[position_rows] >= POSITION_COUNT)):
        raise DatasetContractError("position label outside arena")
    if np.any(position_rows & ~card_rows):
        raise DatasetContractError("position label requires a card label")
    any_supervision = timing_rows | card_rows | position_rows
    if np.any(weights[any_supervision] <= 0):
        raise DatasetContractError("every supervised row requires positive sample weight")
    return {"sequences": len(offsets) - 1, "rows": rows}


def load_shard_arrays(
    shard: Path, manifest: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    """Public loader used by the mmap dataset after validation."""
    return _load_arrays(shard, manifest, mmap=True)
