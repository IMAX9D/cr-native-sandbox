"""Create a deterministic, contract-valid dataset for offline smoke tests."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from .schema import DATASET_KIND, POSITION_COUNT, SCHEMA_VERSION, SHARD_KIND


def _pack(mask: np.ndarray) -> np.ndarray:
    return np.packbits(mask, axis=-1, bitorder="little")


def _shard(path: Path, *, seed: int, sequences: int, steps: int, dimensions: dict[str, int]) -> None:
    rng = np.random.default_rng(seed)
    rows = sequences * steps
    cards = dimensions["card_vocab_size"]
    abilities = dimensions["ability_vocab_size"]
    slots = dimensions["max_ability_slots"]
    arrays: dict[str, np.ndarray] = {
        "sequence_offsets": np.arange(0, rows + 1, steps, dtype=np.int64),
        "grid": rng.integers(0, 256, (rows, dimensions["grid_channels"], 32, 18), dtype=np.uint8),
        "public_scalars": rng.normal(size=(rows, dimensions["public_scalar_size"])).astype(np.float32),
        "own_deck_tokens": np.empty((rows, 8), dtype=np.int16),
        "hand_tokens": np.empty((rows, 4), dtype=np.int16),
        "next_card_token": np.empty(rows, dtype=np.int16),
        "revealed_enemy_tokens": np.zeros((rows, 8), dtype=np.int16),
        "ability_tokens": np.zeros((rows, slots), dtype=np.int16),
        "delta_ticks": np.ones(rows, dtype=np.float32),
        "timing_exposure_ticks": np.ones(rows, dtype=np.float32),
        "card_mask": np.ones((rows, 4), dtype=np.uint8),
        "action_kind_mask": np.ones((rows, 2), dtype=np.uint8),
        "ability_mask": np.zeros((rows, slots), dtype=np.uint8),
        "play_now": np.zeros(rows, dtype=np.uint8),
        "action_kind": np.full(rows, -100, dtype=np.int16),
        "card_slot": np.full(rows, -100, dtype=np.int16),
        "position": np.full(rows, -100, dtype=np.int16),
        "ability_slot": np.full(rows, -100, dtype=np.int16),
        "ability_position": np.full(rows, -100, dtype=np.int16),
        "timing_label_mask": np.ones(rows, dtype=np.uint8),
        "kind_label_mask": np.zeros(rows, dtype=np.uint8),
        "card_label_mask": np.zeros(rows, dtype=np.uint8),
        "position_label_mask": np.zeros(rows, dtype=np.uint8),
        "ability_label_mask": np.zeros(rows, dtype=np.uint8),
        "ability_position_label_mask": np.zeros(rows, dtype=np.uint8),
        "sample_weight": np.ones(rows, dtype=np.float32),
    }
    selected_positions = np.zeros((rows, POSITION_COUNT), dtype=np.uint8)
    ability_positions = np.zeros((rows, POSITION_COUNT), dtype=np.uint8)
    for row in range(rows):
        deck = rng.choice(np.arange(2, cards), size=8, replace=False).astype(np.int16)
        hand = deck[:4]
        arrays["own_deck_tokens"][row] = deck
        arrays["hand_tokens"][row] = hand
        arrays["next_card_token"][row] = deck[4]
        revealed = rng.choice(deck, size=int(rng.integers(0, 5)), replace=False)
        arrays["revealed_enemy_tokens"][row, : len(revealed)] = revealed
        arrays["ability_tokens"][row, 0] = int(rng.integers(1, abilities))
        arrays["ability_mask"][row, 0] = 1
        legal = rng.choice(POSITION_COUNT, size=160, replace=False)
        selected_positions[row, legal] = 1
        ability_positions[row, legal] = 1
        if row % 5 != 0:
            continue
        arrays["play_now"][row] = 1
        arrays["kind_label_mask"][row] = 1
        if row % 15 == 0:
            arrays["action_kind"][row] = 1  # ability
            arrays["ability_label_mask"][row] = 1
            arrays["ability_slot"][row] = 0
            arrays["ability_position_label_mask"][row] = 1
            arrays["ability_position"][row] = int(legal[0])
        else:
            slot = int(rng.integers(0, 4))
            arrays["action_kind"][row] = 0  # deploy
            arrays["card_label_mask"][row] = 1
            arrays["position_label_mask"][row] = 1
            arrays["card_slot"][row] = slot
            arrays["position"][row] = int(legal[0])
    arrays["selected_position_mask_packed"] = _pack(selected_positions)
    arrays["ability_position_mask_packed"] = _pack(ability_positions)
    path.mkdir(parents=True)
    for name, value in arrays.items():
        np.save(path / f"{name}.npy", value, allow_pickle=False)
    (path / "shard.json").write_text(
        json.dumps({"kind": SHARD_KIND, "rows": rows, "sequences": sequences}, indent=2) + "\n",
        encoding="utf-8",
    )


def create_smoke_dataset(root: Path, *, replace: bool = False) -> Path:
    root = root.resolve()
    if root.exists():
        if not replace:
            raise FileExistsError(root)
        shutil.rmtree(root)
    dimensions = {
        "grid_channels": 6,
        "public_scalar_size": 24,
        "card_vocab_size": 160,
        "ability_vocab_size": 32,
        "max_ability_slots": 2,
    }
    splits = {
        "train": ["shards/train-00000"],
        "validation": ["shards/validation-00000"],
        "test": ["shards/test-00000"],
    }
    for index, (split, paths) in enumerate(splits.items()):
        _shard(
            root / paths[0],
            seed=1234 + index,
            sequences=4 if split == "train" else 2,
            steps=20,
            dimensions=dimensions,
        )
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": DATASET_KIND,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "production_ready": False,
        "native_replay_validated": False,
        "actor_information": "public_only_v1",
        "dimensions": dimensions,
        "splits": splits,
        "split_contract": {
            "battle_tag_disjoint": True,
            "source_file_disjoint": True,
            "player_holdout_test": True,
            "synthetic_smoke": True,
        },
        "feature_schema": {
            "grid_channels": [
                f"public_grid_channel_{index}"
                for index in range(dimensions["grid_channels"])
            ],
            "public_scalars": [
                f"public_scalar_{index}"
                for index in range(dimensions["public_scalar_size"])
            ],
        },
        "quality_gates": {
            "split_collisions": 0,
            "forbidden_actor_features": 0,
            "nonfinite_features": 0,
            "expert_label_mask_violations": 0,
            "native_action_rejections": 0,
            "terminal_mismatches": 0,
            "terminal_validation_unknown": 0,
        },
        "state_provenance": {
            "authoritative_rows": 0,
            "native_generated_unanchored_rows": 0,
            "synthetic_smoke_rows": 160,
        },
        "source_manifest": {
            "path": "synthetic://expert-v1-smoke",
            "sha256": "synthetic",
        },
        "card_vocab": {"0": "PAD", "1": "UNK"},
        "ability_vocab": {"0": "PAD"},
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return root
