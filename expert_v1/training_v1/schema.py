"""On-disk contract for native expert behaviour-cloning shards.

The actor schema is intentionally public-information only.  Hidden opponent
hands, exact opponent elixir and any privileged libg state are prohibited.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = 2
DATASET_KIND = "cr_native_expert_bc_dataset_v2"
SHARD_KIND = "cr_native_expert_bc_shard_v2"
OBSERVATION_NATIVE = "native_state_v1"
OBSERVATION_SEQUENCE = "sequence_only_v1"
ARENA_ROWS = 32
ARENA_COLUMNS = 18
POSITION_COUNT = ARENA_ROWS * ARENA_COLUMNS
POSITION_MASK_BYTES = POSITION_COUNT // 8
DECK_SIZE = 8
HAND_SIZE = 4
GRID_STORAGE = "actor_row_csr_flat_u16_u8_v1"
POSITION_MASK_STORAGE = "supervised_rows_packbits_little_v1"

# A shard is a directory of plain .npy arrays.  This keeps large arrays
# memory-mappable and avoids loading a compressed archive into every worker.
NATIVE_REQUIRED_ARRAYS = {
    "sequence_offsets",
    # Lossless actor-row CSR over the flattened [channels, 32, 18] uint8
    # tensor.  The dense grid is reconstructed only for the requested mmap
    # window; storing 4,608 mostly-zero bytes for every 20 Hz actor row made
    # the production corpus several terabytes.
    "grid_offsets",
    "grid_indices",
    "grid_values",
    "public_scalars",
    "own_deck_tokens",
    "hand_tokens",
    "next_card_token",
    "revealed_enemy_tokens",
    # Public native entities use a ragged categorical representation.  Card
    # identity is never written into a continuous grid channel.
    "entity_offsets",
    "entity_tokens",
    "entity_positions",
    "entity_relations",
    "entity_numeric",
    "ability_tokens",
    "delta_ticks",
    "timing_exposure_ticks",
    "card_mask",
    "action_kind_mask",
    "ability_mask",
    "selected_position_mask_rows",
    "selected_position_mask_packed",
    "ability_position_mask_rows",
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
CONTINUOUS_IDENTIFIER_FRAGMENTS = (
    "card_id",
    "card_token",
    "entity_id",
    "native_id",
    "data_id",
)


class DatasetContractError(ValueError):
    """The compiled training data violates the actor data contract."""


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_dataset_integrity(
    root: Path,
    *,
    workers: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fail closed unless the immutable compiled dataset is byte-exact.

    This is intentionally a separate, one-shot startup gate rather than part
    of :func:`read_manifest`: the trainer constructs three mmap datasets and
    must not hash every shard three times.  Every referenced ``.npy`` file
    must be covered by ``shard_file_sha256`` and every declared digest is
    checked before a model or DataLoader is created.
    """

    root = root.resolve()
    manifest_path = root / "manifest.json"
    sidecar_path = root / "manifest.sha256"
    if not manifest_path.is_file():
        raise DatasetContractError(f"dataset manifest is missing: {manifest_path}")
    if not sidecar_path.is_file():
        raise DatasetContractError(f"dataset manifest checksum is missing: {sidecar_path}")
    fields = sidecar_path.read_text(encoding="ascii").split()
    if (
        len(fields) != 2
        or fields[1] != "manifest.json"
        or not _SHA256_RE.fullmatch(fields[0].lower())
    ):
        raise DatasetContractError(f"invalid manifest.sha256 format: {sidecar_path}")
    manifest_digest = sha256_file(manifest_path)
    if fields[0].lower() != manifest_digest:
        raise DatasetContractError(
            f"manifest checksum mismatch: expected {fields[0].lower()}, got {manifest_digest}"
        )

    value = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    validate_manifest(value, root=root)
    declared_value = value.get("shard_file_sha256")
    if not isinstance(declared_value, Mapping) or not declared_value:
        raise DatasetContractError("manifest lacks shard_file_sha256 coverage")

    declared: dict[str, str] = {}
    for raw_relative, raw_digest in declared_value.items():
        relative = str(raw_relative).replace("\\", "/")
        digest = str(raw_digest).lower()
        if relative in declared:
            raise DatasetContractError(f"duplicate shard checksum path: {relative}")
        if not _SHA256_RE.fullmatch(digest):
            raise DatasetContractError(f"invalid shard SHA-256 for {relative}")
        path = (root / relative).resolve()
        if root not in path.parents or path.suffix != ".npy":
            raise DatasetContractError(f"invalid shard checksum path: {relative}")
        declared[relative] = digest

    actual: set[str] = set()
    for shards in value["splits"].values():
        for relative_shard in shards:
            shard = (root / str(relative_shard)).resolve()
            for path in shard.glob("*.npy"):
                actual.add(path.relative_to(root).as_posix())
    missing_coverage = sorted(actual - declared.keys())
    stale_entries = sorted(declared.keys() - actual)
    if missing_coverage or stale_entries:
        raise DatasetContractError(
            "shard checksum coverage mismatch: "
            f"missing={missing_coverage[:5]}, stale={stale_entries[:5]}"
        )

    paths = sorted(declared)
    maximum_workers = int(workers) if workers > 0 else min(32, max(1, (os.cpu_count() or 1) * 2))

    def digest_one(relative: str) -> tuple[str, str]:
        path = root / relative
        if not path.is_file():
            raise DatasetContractError(f"checksummed shard file is missing: {path}")
        return relative, sha256_file(path)

    mismatches: list[tuple[str, str, str]] = []
    with ThreadPoolExecutor(max_workers=maximum_workers) as executor:
        for relative, digest in executor.map(digest_one, paths):
            expected = declared[relative]
            if digest != expected:
                mismatches.append((relative, expected, digest))
    if mismatches:
        relative, expected, actual_digest = mismatches[0]
        raise DatasetContractError(
            "shard checksum mismatch: "
            f"{relative}: expected {expected}, got {actual_digest} "
            f"({len(mismatches)} mismatched file(s))"
        )
    return value, {
        "manifest_sha256": manifest_digest,
        "shard_files": len(paths),
        "integrity_workers": maximum_workers,
    }


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
    if observation_mode == OBSERVATION_NATIVE:
        storage = value.get("storage_schema") or {}
        if storage.get("grid") != GRID_STORAGE:
            raise DatasetContractError(
                f"native grid storage must be {GRID_STORAGE}"
            )
        if storage.get("selected_position_mask") != POSITION_MASK_STORAGE:
            raise DatasetContractError(
                "selected-position mask storage contract changed"
            )
        if storage.get("ability_position_mask") != POSITION_MASK_STORAGE:
            raise DatasetContractError(
                "ability-position mask storage contract changed"
            )
        if value.get("production_ready") is True:
            capacity = value.get("capacity_preflight") or {}
            expected_capacity_path = (root / "capacity-preflight.json").resolve()
            try:
                capacity_path = Path(str(capacity["path"])).resolve()
                capacity_sha = str(capacity["file_sha256"])
            except Exception as error:
                raise DatasetContractError(
                    "production native dataset lacks capacity preflight binding"
                ) from error
            if capacity_path != expected_capacity_path or not capacity_path.is_file():
                raise DatasetContractError(
                    "capacity preflight path is outside/missing from dataset root"
                )
            if not _SHA256_RE.fullmatch(capacity_sha) or sha256_file(capacity_path) != capacity_sha:
                raise DatasetContractError("capacity preflight checksum changed")
        if feature_schema.get("entity_identity") != "categorical_card_vocabulary_v1":
            raise DatasetContractError(
                "native entity identity must use categorical card-vocabulary tokens"
            )
        entity_numeric = feature_schema.get("entity_numeric")
        if (
            not isinstance(entity_numeric, list)
            or len(entity_numeric) != int(dimensions.get("entity_numeric_size", 0))
        ):
            raise DatasetContractError(
                "entity numeric names must describe every continuous entity feature"
            )
        continuous_names = [
            str(name).lower() for name in grid_names + scalar_names + entity_numeric
        ]
        identifiers = sorted(
            name
            for name in continuous_names
            if any(fragment in name for fragment in CONTINUOUS_IDENTIFIER_FRAGMENTS)
        )
        if identifiers:
            raise DatasetContractError(
                "discrete identifiers are forbidden in continuous features: "
                f"{identifiers}"
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
    shapes = {
        "public_scalars": (rows, int(dimensions["public_scalar_size"])),
        "own_deck_tokens": (rows, DECK_SIZE),
        "hand_tokens": (rows, HAND_SIZE),
        "next_card_token": (rows,),
        "revealed_enemy_tokens": (rows, DECK_SIZE),
        "ability_tokens": (rows, int(dimensions["max_ability_slots"])),
        "card_mask": (rows, HAND_SIZE),
        "action_kind_mask": (rows, 2),
        "ability_mask": (rows, int(dimensions["max_ability_slots"])),
    }
    grid_offsets = arrays["grid_offsets"]
    if (
        grid_offsets.dtype != np.int64
        or
        grid_offsets.ndim != 1
        or tuple(grid_offsets.shape) != (rows + 1,)
        or int(grid_offsets[0]) != 0
        or np.any(np.diff(grid_offsets) < 0)
    ):
        raise DatasetContractError("grid_offsets must be monotonic rows+1 CSR offsets")
    grid_entries = int(grid_offsets[-1])
    shapes.update({
        "grid_indices": (grid_entries,),
        "grid_values": (grid_entries,),
    })
    entity_offsets = arrays["entity_offsets"]
    if (
        entity_offsets.ndim != 1
        or tuple(entity_offsets.shape) != (rows + 1,)
        or int(entity_offsets[0]) != 0
        or np.any(np.diff(entity_offsets) < 0)
    ):
        raise DatasetContractError("entity_offsets must be monotonic rows+1 ragged offsets")
    entities = int(entity_offsets[-1])
    entity_numeric_size = int(dimensions.get("entity_numeric_size", 0))
    if entity_numeric_size <= 0:
        raise DatasetContractError("native-state datasets require entity_numeric_size")
    shapes.update({
        "entity_tokens": (entities,),
        "entity_positions": (entities,),
        "entity_relations": (entities,),
        "entity_numeric": (entities, entity_numeric_size),
    })
    position_rows_count = int(arrays["position_label_mask"].astype(bool).sum())
    ability_position_rows_count = int(
        arrays["ability_position_label_mask"].astype(bool).sum()
    )
    shapes.update({
        "selected_position_mask_rows": (position_rows_count,),
        "selected_position_mask_packed": (
            position_rows_count,
            POSITION_MASK_BYTES,
        ),
        "ability_position_mask_rows": (ability_position_rows_count,),
        "ability_position_mask_packed": (
            ability_position_rows_count,
            POSITION_MASK_BYTES,
        ),
    })
    one_dimensional = REQUIRED_ARRAYS - set(shapes) - {
        "sequence_offsets", "entity_offsets", "grid_offsets"
    }
    shapes.update({name: (rows,) for name in one_dimensional})
    for name, expected in shapes.items():
        if tuple(arrays[name].shape) != expected:
            raise DatasetContractError(
                f"{shard}/{name}.npy shape {arrays[name].shape} != {expected}"
            )
    if arrays["grid_indices"].dtype != np.uint16:
        raise DatasetContractError("grid_indices must be uint16")
    if arrays["grid_values"].dtype != np.uint8:
        raise DatasetContractError("grid_values must be uint8")
    grid_indices = arrays["grid_indices"]
    grid_size = int(dimensions["grid_channels"]) * ARENA_ROWS * ARENA_COLUMNS
    if grid_indices.max(initial=0) >= grid_size:
        raise DatasetContractError("grid CSR index outside flattened grid")
    if np.any(arrays["grid_values"] == 0):
        raise DatasetContractError("grid CSR may not store explicit zero values")
    if grid_entries > 1:
        differences = np.diff(grid_indices.astype(np.int32, copy=False))
        boundaries = np.asarray(grid_offsets[1:-1], dtype=np.int64) - 1
        boundaries = boundaries[(boundaries >= 0) & (boundaries < len(differences))]
        differences[boundaries] = 1
        if np.any(differences <= 0):
            raise DatasetContractError(
                "grid CSR indices must be strictly increasing within each row"
            )
    for name in ("own_deck_tokens", "hand_tokens", "next_card_token", "revealed_enemy_tokens"):
        values = arrays[name]
        if values.min(initial=0) < 0 or values.max(initial=0) >= int(dimensions["card_vocab_size"]):
            raise DatasetContractError(f"card token outside vocabulary: {name}")
    entity_tokens = arrays["entity_tokens"]
    if (
        entity_tokens.min(initial=1) <= 0
        or entity_tokens.max(initial=0) >= int(dimensions["card_vocab_size"])
    ):
        raise DatasetContractError("native entity token outside non-PAD card vocabulary")
    entity_positions = arrays["entity_positions"]
    if (
        entity_positions.min(initial=0) < 0
        or entity_positions.max(initial=0) >= POSITION_COUNT
    ):
        raise DatasetContractError("native entity position outside arena")
    entity_relations = arrays["entity_relations"]
    if np.any((entity_relations < 0) | (entity_relations > 1)):
        raise DatasetContractError("native entity relation outside own/enemy")
    if np.any(~np.isfinite(arrays["entity_numeric"])):
        raise DatasetContractError("native entity numeric feature contains NaN/Inf")
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
    if np.any(play_now & ~valid):
        raise DatasetContractError("an observed expert action requires a timing label")
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
        stored_rows = arrays["selected_position_mask_rows"].astype(np.int64)
        expected_rows = np.flatnonzero(position_rows)
        if not np.array_equal(stored_rows, expected_rows):
            raise DatasetContractError(
                "selected-position sparse row index disagrees with label mask"
            )
        unpacked = np.unpackbits(
            arrays["selected_position_mask_packed"],
            axis=-1,
            count=POSITION_COUNT,
            bitorder="little",
        ).astype(bool)
        if np.any(~unpacked[np.arange(len(unpacked)), positions[position_rows]]):
            raise DatasetContractError("expert position is masked illegal")
    if np.any(position_rows & ~card_rows):
        raise DatasetContractError("position label requires a card label")
    if arrays["selected_position_mask_rows"].dtype != np.int64:
        raise DatasetContractError("selected-position sparse rows must be int64")
    if arrays["selected_position_mask_packed"].dtype != np.uint8:
        raise DatasetContractError("selected-position packed masks must be uint8")
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
        stored_rows = arrays["ability_position_mask_rows"].astype(np.int64)
        expected_rows = np.flatnonzero(ability_position_rows)
        if not np.array_equal(stored_rows, expected_rows):
            raise DatasetContractError(
                "ability-position sparse row index disagrees with label mask"
            )
        unpacked = np.unpackbits(
            arrays["ability_position_mask_packed"],
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
    if arrays["ability_position_mask_rows"].dtype != np.int64:
        raise DatasetContractError("ability-position sparse rows must be int64")
    if arrays["ability_position_mask_packed"].dtype != np.uint8:
        raise DatasetContractError("ability-position packed masks must be uint8")
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


def pack_sparse_grid(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode dense ``[rows, channels, 32, 18]`` uint8 losslessly as CSR."""
    value = np.asarray(grid)
    if value.ndim != 4 or tuple(value.shape[-2:]) != (ARENA_ROWS, ARENA_COLUMNS):
        raise ValueError("grid must have shape [rows, channels, 32, 18]")
    if value.dtype != np.uint8:
        raise ValueError("grid must be uint8")
    flattened = value.reshape(value.shape[0], -1)
    row_indices, flat_indices = np.nonzero(flattened)
    counts = np.bincount(row_indices, minlength=value.shape[0])
    offsets = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64))
    )
    return (
        offsets,
        flat_indices.astype(np.uint16, copy=False),
        flattened[row_indices, flat_indices].astype(np.uint8, copy=False),
    )


def unpack_sparse_grid(
    offsets: np.ndarray,
    indices: np.ndarray,
    values: np.ndarray,
    *,
    start: int,
    stop: int,
    channels: int,
) -> np.ndarray:
    """Decode an actor-row CSR slice into byte-identical dense uint8 grids."""
    if not 0 <= start <= stop < len(offsets):
        raise ValueError("invalid sparse-grid row slice")
    count = stop - start
    result = np.zeros(
        (count, channels, ARENA_ROWS, ARENA_COLUMNS), dtype=np.uint8
    )
    if count == 0:
        return result
    first = int(offsets[start])
    last = int(offsets[stop])
    local_offsets = np.asarray(offsets[start : stop + 1], dtype=np.int64) - first
    selected_indices = np.asarray(indices[first:last], dtype=np.int64)
    selected_values = np.asarray(values[first:last], dtype=np.uint8)
    if int(local_offsets[-1]) != len(selected_indices) or len(selected_indices) != len(selected_values):
        raise DatasetContractError("sparse-grid slice offsets disagree with payload")
    if len(selected_indices):
        row_ids = np.repeat(np.arange(count, dtype=np.int64), np.diff(local_offsets))
        result.reshape(count, -1)[row_ids, selected_indices] = selected_values
    return result


def unpack_sparse_position_masks(
    rows: np.ndarray,
    packed: np.ndarray,
    *,
    start: int,
    stop: int,
) -> np.ndarray:
    """Reconstruct zero-default position masks for one actor-row window."""
    result = np.zeros((stop - start, POSITION_COUNT), dtype=np.bool_)
    lower = int(np.searchsorted(rows, start, side="left"))
    upper = int(np.searchsorted(rows, stop, side="left"))
    if lower == upper:
        return result
    local_rows = np.asarray(rows[lower:upper], dtype=np.int64) - start
    result[local_rows] = unpack_position_masks(np.asarray(packed[lower:upper]))
    return result


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
