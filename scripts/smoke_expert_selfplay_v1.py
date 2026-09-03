"""Stage-0 expert self-play admission; performs no optimization or cloud work."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from expert_selfplay_v1.actor_adapter import assert_actor_equivalence
from expert_selfplay_v1.contracts import EntityInputGuard, canonical_schema_hash
from expert_selfplay_v1.critic import PrivilegedCritic, PrivilegedCriticConfig
from expert_selfplay_v1.decks import DeckScheduler
from expert_selfplay_v1.league import LeagueState, OpponentScheduler
from expert_v1.training_v1.dataset import NativeExpertSequenceDataset, collate_sequences
from expert_v1.training_v1.model import (
    ExpertPolicyConfig,
    RecurrentExpertPolicy,
    configure_position_precision,
)
from expert_v1.training_v1.schema import read_manifest
from training.schema import DefensiveTowerReward


DEFAULT_CHECKPOINT = Path(
    r"D:\AI_data\cr-native-core\expert-v1\downloaded\lr-ab-20260831"
    r"\candidate-lr5e-5-step157674-fp16.pt"
)
DEFAULT_DATASET = Path(
    r"D:\AI_data\cr-native-core\expert-v1"
    r"\one-click-schema5-v3-current-frontier-v5\compiled\native-bc-v1"
)
DEFAULT_DECK_ROOT = Path(
    r"D:\AI_data\cr-native-core\expert-v1\audits\top-deck-presets-v1"
)


def _inputs(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, Any]:
    names = (
        "grid", "public_scalars", "own_deck_tokens", "hand_tokens",
        "next_card_token", "revealed_enemy_tokens", "ability_tokens",
        "delta_ticks", "entity_tokens", "entity_positions",
        "entity_relations", "entity_numeric", "entity_mask",
    )
    return {
        name: batch[name][:, :1].to(device)
        for name in names
    }


def _synthetic_inputs(config: ExpertPolicyConfig, device: torch.device) -> dict[str, Any]:
    batch, steps, entities = 1, 1, 2
    return {
        "grid": torch.zeros(batch, steps, config.grid_channels, 32, 18, device=device),
        "public_scalars": torch.zeros(
            batch, steps, config.public_scalar_size, device=device
        ),
        "own_deck_tokens": torch.arange(1, 9, device=device).view(1, 1, 8),
        "hand_tokens": torch.arange(1, 5, device=device).view(1, 1, 4),
        "next_card_token": torch.full((batch, steps), 5, device=device, dtype=torch.long),
        "revealed_enemy_tokens": torch.zeros(batch, steps, 8, device=device, dtype=torch.long),
        "ability_tokens": torch.zeros(
            batch, steps, config.max_ability_slots, device=device, dtype=torch.long
        ),
        "delta_ticks": torch.ones(batch, steps, device=device),
        "entity_tokens": torch.tensor([[[1, 2]]], device=device),
        "entity_positions": torch.tensor([[[100, 477]]], device=device),
        "entity_relations": torch.tensor([[[0, 1]]], device=device),
        "entity_numeric": torch.tensor(
            [[[[0.7, 0.8, 0.4], [0.8, 0.5, 0.6]]]], device=device
        ),
        "entity_mask": torch.ones(batch, steps, entities, device=device, dtype=torch.bool),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--learner-deck", type=Path,
        default=PROJECT_ROOT / "examples" / "user-selected-heavy-control.json",
    )
    parser.add_argument("--opponent-deck-root", type=Path, default=DEFAULT_DECK_ROOT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("kind") != "cr_native_expert_inference_weights_v1":
        raise RuntimeError("Stage-0 requires an expert inference checkpoint")
    manifest_sha = str(checkpoint.get("dataset_manifest_sha256"))
    if len(manifest_sha) != 64:
        raise RuntimeError("expert checkpoint lacks its dataset manifest hash")
    config = ExpertPolicyConfig(**checkpoint["model_config"])
    configure_position_precision(config)
    actor = RecurrentExpertPolicy(config)
    actor.load_state_dict(checkpoint["model_state"], strict=True)
    actor.eval().to(device)

    guard = EntityInputGuard()
    if args.synthetic:
        actor_inputs = _synthetic_inputs(config, device)
        guard.observe(
            native_eligible_entities=2,
            entity_tokens=[1, 2],
            entity_positions=[100, 477],
            entity_mask=[True, True],
        )
    else:
        manifest = read_manifest(args.dataset_root)
        manifest_bytes = (args.dataset_root / "manifest.json").read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha:
            raise RuntimeError("expert checkpoint/dataset vocabulary hashes differ")
        dataset = NativeExpertSequenceDataset(
            args.dataset_root, split="validation", sequence_length=64, burn_in=0
        )
        batch = collate_sequences([dataset[0]])
        mask = batch["entity_mask"][0].numpy()
        tokens = batch["entity_tokens"][0].numpy()
        positions = batch["entity_positions"][0].numpy()
        for index in range(mask.shape[0]):
            guard.observe(
                native_eligible_entities=int(mask[index].sum()),
                entity_tokens=tokens[index],
                entity_positions=positions[index],
                entity_mask=mask[index],
            )
        actor_inputs = _inputs(batch, device)
        dataset.close()
    if guard.summary()["native_nonempty_frames"] < 1:
        raise RuntimeError("Stage-0 sample contains no dynamic entity frame")
    equivalence = assert_actor_equivalence(actor, actor_inputs)

    critic = PrivilegedCritic(PrivilegedCriticConfig(
        actor_latent_size=config.hidden_size,
        card_vocab_size=config.card_vocab_size,
        public_grid_channels=config.grid_channels,
        entity_numeric_size=config.entity_numeric_size,
        scalar_size=32,
    ))
    critic_parameters = sum(value.numel() for value in critic.parameters())
    deck_scheduler = DeckScheduler(
        learner_preset=args.learner_deck,
        opponent_presets=sorted(args.opponent_deck_root.glob("deck-*.json")),
    )
    deck_rows = deck_scheduler.build_batch(episode_count=20, seed=20260902)
    league = LeagueState(base_policy_id="BASE", champion_policy_id="BASE")
    opponent_rows = OpponentScheduler(league).build_batch(
        episode_count=20, candidate_id="BASE-CANDIDATE", seed=20260902
    )
    reward_contract = {
        "schema": DefensiveTowerReward.schema_version,
        "damage_dealt_scale": DefensiveTowerReward.damage_dealt_scale,
        "damage_received_scale": DefensiveTowerReward.damage_received_scale,
        "tower_destroyed_reward": DefensiveTowerReward.tower_destroyed_reward,
        "terminal_win_reward": DefensiveTowerReward.terminal_win_reward,
    }
    result = {
        "kind": "cr_native_expert_selfplay_stage0_smoke_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "admitted": True,
        "device": str(device),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": int(checkpoint["global_step"]),
        "checkpoint_run_id": str(checkpoint["run_id"]),
        "dataset_manifest_sha256": manifest_sha,
        "actor_parameters": sum(value.numel() for value in actor.parameters()),
        "critic_parameters": critic_parameters,
        "actor_equivalence": equivalence,
        "entity_guard": guard.summary(),
        "reward_contract": reward_contract,
        "reward_schema_sha256": canonical_schema_hash(reward_contract),
        "deck_batch": {
            "episodes": len(deck_rows),
            "learner_side0": sum(row.learner_side == 0 for row in deck_rows),
            "opponent_presets": len({row.opponent_preset for row in deck_rows}),
        },
        "opponent_bootstrap": {
            "episodes": len(opponent_rows),
            "champion": sum(row.category == "champion" for row in opponent_rows),
            "base": sum(row.category == "base" for row in opponent_rows),
            "history": sum(row.category == "history" for row in opponent_rows),
        },
        "optimization_performed": False,
        "cloud_started": False,
        "synthetic_observation": bool(args.synthetic),
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
