from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

import torch

from expert_v1.training_v1.model import ExpertPolicyConfig, RecurrentExpertPolicy
from expert_selfplay_v1.actions import ExpertActionMasks
from expert_selfplay_v1.batched_policy import BatchedPolicyService, PolicyRequest
from expert_selfplay_v1.contracts import BatchManifest
from expert_selfplay_v1.critic import ExpertActorCritic, PrivilegedCritic, PrivilegedCriticConfig
from expert_selfplay_v1.critic_training import Stage1CriticTrainer
from expert_selfplay_v1.rollout import DecisionRecord, EpisodeHeader, LearnerEpisodeBuffer
from expert_selfplay_v1.rollout_storage import ImmutableRolloutShardWriter
from expert_selfplay_v1.stage2_training import (
    EXPERT_INFERENCE_KIND,
    Stage2PPOTrainer,
    Stage2TrainingConfig,
    _state_digest,
)


class Stage2TrainingTests(unittest.TestCase):
    @staticmethod
    def _config() -> ExpertPolicyConfig:
        return ExpertPolicyConfig(
            grid_channels=2,
            public_scalar_size=5,
            card_vocab_size=16,
            ability_vocab_size=3,
            max_ability_slots=2,
            hidden_size=12,
            card_embedding_size=8,
            spatial_size=8,
            lambda_initial=4.0,
            lambda_max=20.0,
        )

    @staticmethod
    def _actor_inputs(index: int) -> dict[str, torch.Tensor]:
        return {
            "grid": torch.randn(1, 2, 32, 18),
            "public_scalars": torch.tensor([[0.1, 0.2, 0.3, 0.4, index / 10]]),
            "own_deck_tokens": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
            "hand_tokens": torch.tensor([[1, 2, 3, 4]]),
            "next_card_token": torch.tensor([5]),
            "revealed_enemy_tokens": torch.tensor([[6, 7, 0, 0, 0, 0, 0, 0]]),
            "ability_tokens": torch.zeros(1, 2, dtype=torch.long),
            "delta_ticks": torch.tensor([4.0]),
            "entity_tokens": torch.tensor([[1, 2]]),
            "entity_positions": torch.tensor([[10, 20]]),
            "entity_relations": torch.tensor([[0, 1]]),
            "entity_numeric": torch.ones(1, 2, 3),
            "entity_mask": torch.ones(1, 2, dtype=torch.bool),
        }

    @staticmethod
    def _masks() -> ExpertActionMasks:
        positions = torch.zeros(4, 576, dtype=torch.bool)
        positions[:, 17] = True
        return ExpertActionMasks(
            action_kind=torch.tensor([True, False]),
            cards=torch.ones(4, dtype=torch.bool),
            positions=positions,
            abilities=torch.zeros(2, dtype=torch.bool),
            ability_positions=torch.zeros(2, 576, dtype=torch.bool),
            ability_requires_target=torch.zeros(2, dtype=torch.bool),
        )

    @staticmethod
    def _critic_inputs(index: int) -> dict[str, torch.Tensor]:
        return {
            "grid": torch.randn(1, 2, 32, 18),
            "entity_tokens": torch.tensor([[1, 2]]),
            "entity_positions": torch.tensor([[10, 20]]),
            "entity_relations": torch.tensor([[0, 1]]),
            "entity_numeric": torch.ones(1, 2, 3),
            "entity_mask": torch.ones(1, 2, dtype=torch.bool),
            "private_card_tokens": torch.tensor([[1, 2, 3, 4]]),
            "private_card_owners": torch.tensor([[0, 0, 1, 1]]),
            "private_card_slots": torch.tensor([[0, 1, 2, 3]]),
            "private_card_mask": torch.ones(1, 4, dtype=torch.bool),
            "scalars": torch.tensor([[index / 10] * 32]),
        }

    def _fixture(self, root: Path) -> tuple[Path, Path, Path, str]:
        manifest = root / "manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
        config = self._config()
        actor = RecurrentExpertPolicy(config)
        half_state = {
            name: value.detach().half().clone()
            for name, value in actor.state_dict().items()
        }
        behavior_sha = _state_digest(half_state)
        base = root / "base-fp16.pt"
        torch.save({
            "kind": EXPERT_INFERENCE_KIND,
            "dataset_manifest_sha256": manifest_sha,
            "model_config": config.to_dict(),
            "model_state": half_state,
            "global_step": 10,
            "run_id": "test-base",
        }, base)

        behavior_actor = RecurrentExpertPolicy(config)
        behavior_actor.load_state_dict(half_state)
        behavior_actor.half()
        critic_config = PrivilegedCriticConfig(
            actor_latent_size=config.hidden_size,
            card_vocab_size=config.card_vocab_size,
            public_grid_channels=config.grid_channels,
            entity_numeric_size=config.entity_numeric_size,
            scalar_size=32,
        )
        stage1 = Stage1CriticTrainer(
            ExpertActorCritic(behavior_actor, PrivilegedCritic(critic_config)),
            device="cpu",
            actor_source_reference=base,
            global_update=52,
        )
        stage1_path = stage1.save_checkpoint(root / "stage1", {"loss": 1.0})
        return base, manifest, stage1_path, behavior_sha

    def _shard(
        self, root: Path, actor: RecurrentExpertPolicy, behavior_sha: str,
        *, include_hidden: bool = True,
    ) -> Path:
        service = BatchedPolicyService(
            device="cpu", deterministic=True, deterministic_event_threshold=0.0
        )
        service.register_actor(actor, actor_sha256=behavior_sha)
        header = EpisodeHeader(
            episode_id="episode-1",
            batch_id="batch-1",
            seed=1,
            learner_side=0,
            behavior_policy_version=0,
            behavior_actor_sha256=behavior_sha,
            opponent_policy_id="BASE",
            opponent_actor_sha256=behavior_sha,
            learner_deck_sha256="c" * 64,
            opponent_deck_sha256="d" * 64,
            curriculum_stage="stage2_reaction",
            initial_hidden_sha256="e" * 64,
        )
        episode = LearnerEpisodeBuffer(header)
        payloads = []
        for index in range(3):
            inputs = self._actor_inputs(index)
            masks = self._masks()
            action = service.act([PolicyRequest(
                worker_id="worker-1",
                side=0,
                actor_sha256=behavior_sha,
                actor_inputs=inputs,
                masks=masks,
                delta_ticks=4,
            )])[0]
            hidden = service.last_pre_action_hidden(
                actor_sha256=behavior_sha, worker_id="worker-1", side=0
            )
            terminal = index == 2
            episode.append(DecisionRecord(
                tick=100 + index * 4,
                delta_ticks=4,
                side=0,
                event_happened=action.event_happened,
                action_kind=action.action_kind,
                card_slot=action.card_slot,
                position=action.position,
                ability_slot=action.ability_slot,
                ability_position=action.ability_position,
                old_logp_total=action.old_logp_total,
                old_logp_timing=action.old_logp_timing,
                old_logp_action_type=action.old_logp_action_type,
                old_logp_slot=action.old_logp_slot,
                old_logp_position=action.old_logp_position,
                reward_damage_dealt=0.0,
                reward_damage_received=0.0,
                reward_towers_dealt=0.0,
                reward_towers_received=0.0,
                reward_terminal=1.0 if terminal else 0.0,
                reward_total=1.0 if terminal else 0.0,
                value=0.0,
                terminated=terminal,
                truncated=False,
                native_entity_count=2,
                encoded_entity_count=2,
            ))
            actor_inputs = {
                name: value.detach().clone() for name, value in inputs.items()
            }
            if include_hidden:
                actor_inputs["hidden"] = tuple(value.detach().clone() for value in hidden)
            payloads.append({
                "actor_inputs": actor_inputs,
                "critic_inputs": self._critic_inputs(index),
                "action_masks": {
                    name: getattr(masks, name).clone()
                    for name in masks.__dataclass_fields__
                },
                "recorded_action": {
                    "event_happened": torch.tensor(action.event_happened),
                    "action_kind": torch.tensor(action.action_kind),
                    "card_slot": torch.tensor(action.card_slot),
                    "position": torch.tensor(action.position),
                    "ability_slot": torch.tensor(action.ability_slot),
                    "ability_position": torch.tensor(action.ability_position),
                    "ability_requires_target": torch.tensor(action.ability_requires_target),
                    "old_logp_total": torch.tensor(action.old_logp_total),
                },
                "critic_targets": {
                    "wdl_class": 2,
                    "crown_difference": 1.0,
                    "tower_hp_difference": 0.2,
                    "future_damage": [0.1, 0.0],
                },
            })
        frozen = episode.freeze()
        batch = BatchManifest(
            run_id="stage2-test-collection",
            batch_id="batch-1",
            policy_version=0,
            behavior_actor_sha256=behavior_sha,
            encoder_schema_sha256="1" * 64,
            action_schema_sha256="2" * 64,
            reward_schema_sha256="3" * 64,
            native_lib_sha256="4" * 64,
            episode_count=1,
        )
        writer = ImmutableRolloutShardWriter(root / "rollouts", batch)
        return writer.write(
            "shard-1", [frozen], step_payloads_by_episode={"episode-1": payloads}
        ).directory

    def test_hidden_anchored_rollout_runs_one_guarded_actor_update(self) -> None:
        torch.manual_seed(7)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, manifest, stage1, behavior_sha = self._fixture(root)
            config = self._config()
            actor = RecurrentExpertPolicy(config)
            actor.load_state_dict(torch.load(base, weights_only=False)["model_state"])
            shard = self._shard(root, actor, behavior_sha)
            trainer = Stage2PPOTrainer(
                base_inference_checkpoint=base,
                continuation_checkpoint=stage1,
                expert_manifest=manifest,
                device="cpu",
                config=Stage2TrainingConfig(
                    ppo_epochs=1,
                    chunk_batch_size=1,
                    bc_kl_soft_limit=1.0,
                ),
            )

            chunks, batch, rollout = trainer.prepare_rollout(shard)
            metrics, guard, retry = trainer.train_update(chunks)

            self.assertEqual(batch.policy_version, 0)
            self.assertEqual(rollout["decisions"], 3)
            self.assertEqual(trainer.policy_version, 1)
            self.assertEqual(trainer.global_update, 53)
            self.assertEqual(guard.action, "accept")
            self.assertIn(retry, (0, 1))
            self.assertTrue(all(
                torch.isfinite(torch.tensor(value))
                for value in metrics.values() if isinstance(value, (int, float))
            ))
            checkpoint, export = trainer.save(
                root / "stage2",
                metrics=metrics,
                guard=guard,
                retry_attempt=retry,
                rollout=rollout,
            )
            self.assertTrue(checkpoint.is_file())
            exported = torch.load(export, map_location="cpu", weights_only=False)
            self.assertEqual(
                _state_digest(exported["model_state"]), trainer.behavior_actor_sha256
            )
            resumed = Stage2PPOTrainer(
                base_inference_checkpoint=base,
                continuation_checkpoint=checkpoint,
                expert_manifest=manifest,
                device="cpu",
                config=Stage2TrainingConfig(
                    ppo_epochs=1,
                    chunk_batch_size=1,
                    bc_kl_soft_limit=1.0,
                ),
            )
            self.assertEqual(resumed.policy_version, 1)
            self.assertEqual(resumed.global_update, 53)
            self.assertEqual(
                resumed.behavior_actor_sha256, trainer.behavior_actor_sha256
            )

    def test_stage2_rejects_rollout_without_exact_hidden_anchor(self) -> None:
        torch.manual_seed(9)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, manifest, stage1, behavior_sha = self._fixture(root)
            actor = RecurrentExpertPolicy(self._config())
            actor.load_state_dict(torch.load(base, weights_only=False)["model_state"])
            shard = self._shard(root, actor, behavior_sha, include_hidden=False)
            trainer = Stage2PPOTrainer(
                base_inference_checkpoint=base,
                continuation_checkpoint=stage1,
                expert_manifest=manifest,
                device="cpu",
                config=Stage2TrainingConfig(ppo_epochs=1, chunk_batch_size=1),
            )
            with self.assertRaisesRegex(RuntimeError, "pre-action hidden"):
                trainer.prepare_rollout(shard)


if __name__ == "__main__":
    unittest.main()
