"""Differential acceptance for compact observations and vector collection."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from native_core.env import NativeRoyaleEnv
from training.model import RecurrentPolicyValueNet
from training.rollout import NativeSelfPlayCollector
from training.schema import (
    ActionMaskCache, ObservationEncoder, PotentialReward, build_action_masks,
)
from training.vector_rollout import VectorNativeSelfPlayCollector


class _DeterministicPolicy(RecurrentPolicyValueNet):
    def sample_batch(self, *args, **kwargs):
        kwargs["deterministic"] = True
        return super().sample_batch(*args, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=37031)
    parser.add_argument(
        "--output", type=Path,
        default=Path(r"D:\AI_data\cr-native-core\acceptance-training-fast-path.json"),
    )
    args = parser.parse_args()
    replay = json.loads(
        (PROJECT_ROOT / "examples" / "eight-card-bootstrap.json").read_text(
            encoding="utf-8-sig"
        )
    )
    replay["rndSeed"] = 777
    env = NativeRoyaleEnv(port=args.port, timeout=30)
    env.reset(replay, warmup_steps=100)
    encoder = ObservationEncoder()
    reward = PotentialReward(gamma=0.99995)
    mask_cache = ActionMaskCache()
    comparisons = []

    def compare(label: str) -> dict[str, Any]:
        full = env.observe()
        compact = env.observe_train()
        fields = (
            "category", "side", "x", "y", "card_id", "hp", "max_hp",
            "behavior_state",
        )
        normalize = lambda state: sorted(
            tuple(entity[field] for field in fields)
            for entity in state["entities"]
        )
        if not (
            full["tick"] == compact["tick"]
            and full["entity_count"] == compact["entity_count"]
            and normalize(full) == normalize(compact)
            and full["episode"] == compact["episode"]
            and reward.potential(full, 0) == reward.potential(compact, 0)
        ):
            raise RuntimeError(f"compact state mismatch at {label}")
        native_masks = {}
        for player in full["players"]:
            side = int(player["side"])
            for deck_index in player["hand_deck_indices"]:
                if deck_index >= 0 and (side, deck_index) not in native_masks:
                    native_masks[(side, deck_index)] = env.probe_grid(
                        side=side, deck_index=deck_index
                    )["rows"]
        for side in (0, 1):
            full_player = next(p for p in full["players"] if p["side"] == side)
            compact_player = next(p for p in compact["players"] if p["side"] == side)
            for field in ("side", "elixir", "elixir_raw", "hand_deck_indices"):
                if full_player[field] != compact_player[field]:
                    raise RuntimeError(f"compact player mismatch: {field}")
            full_grid, full_scalars = encoder.encode(full, side=side)
            compact_grid, compact_scalars = encoder.encode(compact, side=side)
            if not (
                np.array_equal(full_grid, compact_grid)
                and np.array_equal(full_scalars, compact_scalars)
                and np.array_equal(
                    encoder.privileged(full, side=side),
                    encoder.privileged(compact, side=side),
                )
            ):
                raise RuntimeError(f"compact encoding mismatch for side {side}")
            legacy = build_action_masks(
                full, side=side, native_masks=native_masks, decks=env.decks
            )
            cached = build_action_masks(
                compact, side=side, native_masks=native_masks, decks=env.decks,
                cache=mask_cache,
            )
            if not (
                np.array_equal(legacy[0], cached[0])
                and np.array_equal(legacy[1], cached[1])
                and legacy[2] == cached[2]
            ):
                raise RuntimeError(f"cached mask mismatch for side {side}")
        row = {
            "label": label, "tick": full["tick"],
            "entities": full["entity_count"],
            "full_bytes": len(json.dumps(full, separators=(",", ":"))),
            "compact_bytes": len(json.dumps(compact, separators=(",", ":"))),
        }
        comparisons.append(row)
        return full

    opening = compare("opening")
    actions = []
    for side in (0, 1):
        player = next(p for p in opening["players"] if p["side"] == side)
        deck_index = int(player["hand_deck_indices"][0])
        rows = env.probe_grid(side=side, deck_index=deck_index)["rows"]
        cells = [
            (column * 1000 + 500, row * 1000 + 500)
            for row, values in enumerate(rows)
            for column, value in enumerate(values) if value == "1"
        ]
        x, y = cells[len(cells) // 3]
        actions.append({
            "side": side, "deck_index": deck_index, "x": x, "y": y,
        })
    env.joint_act(actions)
    env.step(20)
    compare("deployed")
    env.step(300)
    compare("combat")

    torch.manual_seed(99)
    model = _DeterministicPolicy().to("cuda").eval()
    replay["rndSeed"] = 9001
    scalar = NativeSelfPlayCollector(
        env, model, replay, device=torch.device("cuda"),
        reward_mode="potential", max_ticks=500,
    ).collect(9001)
    vector = VectorNativeSelfPlayCollector(
        [env], model, replay, device=torch.device("cuda"),
        reward_mode="potential", max_ticks=500,
    ).collect([9001])[0]
    for side in (0, 1):
        left, right = scalar.trajectories[side].arrays(), vector.trajectories[side].arrays()
        for field in (
            "grid", "scalars", "privileged", "card_masks", "position_masks",
            "cards", "positions", "rewards", "dones",
        ):
            if not np.array_equal(left[field], right[field]):
                raise RuntimeError(f"vector differential mismatch: side={side} {field}")

    result = {
        "schema_version": 1,
        "kind": "native_training_fast_path_acceptance",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "compact_comparisons": comparisons,
        "deterministic_vector_steps_per_side": len(
            scalar.trajectories[0].cards
        ),
        "deterministic_actions": scalar.actions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
