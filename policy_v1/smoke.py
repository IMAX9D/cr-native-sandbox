"""One-command CPU training check on explicitly synthetic data."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from .data import SCALARS, prepare, digest
from .train import parser as train_parser, run


def create_fixture(root, steps=24):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    dimensions = {
        "card_vocab_size": 12,
        "ability_vocab_size": 4,
        "public_scalar_size": 16,
        "entity_numeric_size": 3,
        "grid_channels": 8,
        "max_ability_slots": 2,
        "max_entities": 2,
    }
    m = {
        "kind": "cr_native_expert_bc_dataset_v3",
        "observation_mode": "native_state_v1",
        "actor_information": "public_only_v1",
        "production_ready": False,
        "smoke_only": True,
        "dimensions": dimensions,
        "feature_schema": {
            "public_scalars": SCALARS,
            "entity_numeric": ["level_ratio", "hp_ratio", "log_max_hp"],
        },
        "splits": {},
    }
    for split in ("train", "validation"):
        relative = "shards/" + split + "-00000"
        p = root / relative
        p.mkdir(parents=True)
        m["splits"][split] = [relative]
        rows = steps * 2
        a = {
            "sequence_offsets": np.array([0, steps, rows], dtype=np.int64),
            "public_scalars": np.zeros((rows, 16), np.float32),
            "delta_ticks": np.ones(rows, np.float32),
            "timing_exposure_ticks": np.ones(rows, np.float32),
            "sample_weight": np.ones(rows, np.float32),
            "own_deck_tokens": np.tile(np.arange(1, 9, dtype=np.int16), (rows, 1)),
            "hand_tokens": np.tile(np.arange(1, 5, dtype=np.int16), (rows, 1)),
            "next_card_token": np.full(rows, 5, np.int16),
            "revealed_enemy_tokens": np.zeros((rows, 8), np.int16),
            "ability_tokens": np.tile(np.array([1, 0], np.int16), (rows, 1)),
            "card_mask": np.ones((rows, 4), np.uint8),
            "action_kind_mask": np.ones((rows, 2), np.uint8),
            "ability_mask": np.tile(np.array([1, 0], np.uint8), (rows, 1)),
            "entity_offsets": np.arange(0, 2 * rows + 1, 2, dtype=np.int64),
            "entity_tokens": np.tile(np.array([1, 2], np.int16), rows),
            "entity_positions": np.tile(np.array([200, 300], np.int16), rows),
            "entity_relations": np.tile(np.array([0, 1], np.uint8), rows),
            "entity_numeric": np.ones((2 * rows, 3), np.float32) * 0.5,
            "grid_offsets": np.arange(rows + 1, dtype=np.int64),
            "grid_indices": np.zeros(rows, np.uint16),
            "grid_values": np.full(rows, 255, np.uint8),
            "replay_extent": np.ones(rows, np.uint8),
        }
        a["public_scalars"][:, 0] = np.tile(np.arange(100, 100 + steps) / 6000.0, 2)
        a["public_scalars"][:, 1] = 0.7
        a["public_scalars"][:, 2] = 1
        for name in (
            "play_now",
            "timing_label_mask",
            "kind_label_mask",
            "card_label_mask",
            "position_label_mask",
            "ability_label_mask",
            "ability_position_label_mask",
        ):
            a[name] = np.zeros(rows, np.uint8)
        a["timing_label_mask"][:] = 1
        for name in (
            "action_kind",
            "card_slot",
            "position",
            "ability_slot",
            "ability_position",
        ):
            a[name] = np.full(rows, -100, np.int64)
        deploy = []
        skill = []
        for side in (0, 1):
            for t in range(2 + side, steps, 6):
                r = side * steps + t
                is_skill = t % 3 == 0
                a["play_now"][r] = a["kind_label_mask"][r] = 1
                a["action_kind"][r] = int(is_skill)
                if is_skill:
                    a["ability_label_mask"][r] = a["ability_position_label_mask"][r] = 1
                    a["ability_slot"][r] = 0
                    a["ability_position"][r] = 100
                    skill.append(r)
                else:
                    a["card_label_mask"][r] = a["position_label_mask"][r] = 1
                    a["card_slot"][r] = 0
                    a["position"][r] = 200
                    deploy.append(r)
        for prefix, rr in (
            ("selected_position_mask", deploy),
            ("ability_position_mask", skill),
        ):
            a[prefix + "_rows"] = np.array(sorted(rr), np.int64)
            a[prefix + "_packed"] = np.full((len(rr), 72), 255, np.uint8)
        for k, v in a.items():
            np.save(p / (k + ".npy"), v, allow_pickle=False)
        metadata = {
            "sequence_identity": [
                {"actor_side": s, "battle_tag": "synthetic-" + split} for s in (0, 1)
            ],
            "file_sha256": {k + ".npy": digest(p / (k + ".npy")) for k in a},
        }
        (p / "shard.json").write_text(json.dumps(metadata))
    (root / "manifest.json").write_text(json.dumps(m))
    return root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    base = args.output or Path(tempfile.mkdtemp(prefix="cr-policy-smoke-"))
    base.mkdir(parents=True, exist_ok=True)
    data = create_fixture(base / "data")
    prepare(data, base / "cache", allow_smoke=True, verify_hashes=True)
    argv = [
        "--data",
        str(data),
        "--cache",
        str(base / "cache"),
        "--run",
        str(base / "run"),
        "--device",
        "cpu",
        "--allow-smoke",
        "--width",
        "32",
        "--heads",
        "4",
        "--layers",
        "1",
        "--frame-window",
        "8",
        "--event-window",
        "8",
        "--targets",
        "4",
        "--batch-size",
        "2",
        "--workers",
        "0",
        "--max-steps",
        "3",
        "--log-every",
        "1",
        "--eval-batches",
        "2",
    ]
    run(train_parser().parse_args(argv))
    print(
        "Synthetic smoke passed; checkpoint:",
        base / "run/last.pt",
        "(not a trained game policy)",
    )


if __name__ == "__main__":
    main()
