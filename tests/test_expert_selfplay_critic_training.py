from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch
from torch import Tensor, nn

from expert_v1.training_v1.model import ExpertPolicyConfig, RecurrentExpertPolicy
from expert_selfplay_v1.actor_adapter import actor_state_digest
from expert_selfplay_v1.critic import CriticOutput, ExpertActorCritic
from expert_selfplay_v1.critic_training import (
    CriticTrainingConfig,
    Stage1CriticTrainer,
    _capture_rng_state,
    _restore_rng_state,
)
from expert_selfplay_v1.rollout import DecisionRecord, EpisodeHeader, LearnerEpisodeBuffer
from expert_selfplay_v1.rollout_storage import LearnerEpisodeChunker


class _TinyCritic(nn.Module):
    def __init__(self, actor_latent_size: int, scalar_size: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(actor_latent_size + scalar_size, 24), nn.SiLU()
        )
        self.value = nn.Linear(24, 1)
        self.wdl = nn.Linear(24, 3)
        self.crown = nn.Linear(24, 1)
        self.tower = nn.Linear(24, 1)
        self.damage = nn.Linear(24, 2)

    def forward(self, *, actor_latent: Tensor, scalars: Tensor, **_unused) -> CriticOutput:
        hidden = self.body(torch.cat((actor_latent, scalars), dim=-1))
        return CriticOutput(
            self.value(hidden).squeeze(-1),
            self.wdl(hidden),
            self.crown(hidden).squeeze(-1),
            self.tower(hidden).squeeze(-1),
            self.damage(hidden),
        )


class Stage1CriticTrainingTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for mapped-RNG regression")
    def test_restore_rng_normalizes_cuda_mapped_byte_tensors_to_cpu(self) -> None:
        state = _capture_rng_state()
        state["torch_cpu"] = state["torch_cpu"].cuda()
        state["torch_cuda"] = [value.cuda() for value in state["torch_cuda"]]
        _restore_rng_state(state)

    @staticmethod
    def _model() -> ExpertActorCritic:
        actor = RecurrentExpertPolicy(ExpertPolicyConfig(
            grid_channels=2,
            public_scalar_size=5,
            card_vocab_size=16,
            ability_vocab_size=5,
            max_ability_slots=2,
            hidden_size=12,
            card_embedding_size=8,
            spatial_size=8,
        ))
        return ExpertActorCritic(actor, _TinyCritic(12, 4))

    @staticmethod
    def _header() -> EpisodeHeader:
        return EpisodeHeader(
            episode_id="episode-1",
            batch_id="batch-1",
            seed=1,
            learner_side=0,
            behavior_policy_version=1,
            behavior_actor_sha256="a" * 64,
            opponent_policy_id="opponent",
            opponent_actor_sha256="b" * 64,
            learner_deck_sha256="c" * 64,
            opponent_deck_sha256="d" * 64,
            curriculum_stage="stage1_critic",
            initial_hidden_sha256="e" * 64,
        )

    @staticmethod
    def _decision(index: int, count: int) -> DecisionRecord:
        terminal = index == count - 1
        return DecisionRecord(
            tick=index + 1,
            delta_ticks=1,
            side=0,
            event_happened=False,
            action_kind=0,
            card_slot=0,
            position=0,
            ability_slot=0,
            ability_position=0,
            old_logp_total=-0.1,
            old_logp_timing=-0.1,
            old_logp_action_type=0.0,
            old_logp_slot=0.0,
            old_logp_position=0.0,
            reward_damage_dealt=0.0,
            reward_damage_received=0.0,
            reward_towers_dealt=0.0,
            reward_towers_received=0.0,
            reward_terminal=1.0 if terminal else 0.0,
            reward_total=1.0 if terminal else 0.0,
            value=0.0,
            terminated=terminal,
            truncated=False,
            native_entity_count=1,
            encoded_entity_count=1,
        )

    @staticmethod
    def _payload(index: int) -> dict[str, object]:
        # Online service convention: singleton time axis, no batch axis.
        actor_inputs = {
            "grid": torch.randn(1, 2, 32, 18),
            "public_scalars": torch.randn(1, 5),
            "own_deck_tokens": torch.randint(1, 16, (1, 8)),
            "hand_tokens": torch.randint(1, 16, (1, 4)),
            "next_card_token": torch.randint(1, 16, (1,)),
            "revealed_enemy_tokens": torch.zeros(1, 8, dtype=torch.long),
            "ability_tokens": torch.zeros(1, 2, dtype=torch.long),
            "delta_ticks": torch.ones(1),
            "entity_tokens": torch.ones(1, 1, dtype=torch.long),
            "entity_positions": torch.full((1, 1), index, dtype=torch.long),
            "entity_relations": torch.zeros(1, 1, dtype=torch.long),
            "entity_numeric": torch.ones(1, 1, 3),
            "entity_mask": torch.ones(1, 1, dtype=torch.bool),
        }
        critic_inputs = {
            "grid": torch.zeros(1, 2, 32, 18),
            "entity_tokens": torch.ones(1, 1, dtype=torch.long),
            "entity_positions": torch.full((1, 1), index, dtype=torch.long),
            "entity_relations": torch.zeros(1, 1, dtype=torch.long),
            "entity_numeric": torch.ones(1, 1, 3),
            "entity_mask": torch.ones(1, 1, dtype=torch.bool),
            "private_card_tokens": torch.ones(1, 2, dtype=torch.long),
            "private_card_owners": torch.tensor([[0, 1]]),
            "private_card_slots": torch.tensor([[0, 1]]),
            "private_card_mask": torch.ones(1, 2, dtype=torch.bool),
            "scalars": torch.tensor([[0.1, 0.2, 0.3, index / 10.0]]),
        }
        return {
            "actor_inputs": actor_inputs,
            "critic_inputs": critic_inputs,
            "critic_targets": {
                "wdl_class": 2,
                "crown_difference": 1.0,
                "tower_hp_difference": 0.25,
                "future_damage": [0.2, 0.0],
            },
        }

    @classmethod
    def _chunks(cls) -> list[dict[str, object]]:
        buffer = LearnerEpisodeBuffer(cls._header())
        count = 3
        for index in range(count):
            buffer.append(cls._decision(index, count))
        return LearnerEpisodeChunker().chunk(
            buffer, step_payloads=[cls._payload(index) for index in range(count)]
        )

    def test_one_cpu_bf16_update_trains_critic_and_keeps_actor_bit_exact(self) -> None:
        torch.manual_seed(5)
        model = self._model()
        trainer = Stage1CriticTrainer(
            model,
            device="cpu",
            config=CriticTrainingConfig(max_grad_norm=0.25),
        )
        actor_before = actor_state_digest(model.actor)
        critic_before = {
            name: value.detach().clone() for name, value in model.critic.state_dict().items()
        }
        metrics = trainer.train_update(self._chunks())
        self.assertEqual(metrics["global_update"], 1)
        self.assertEqual(metrics["loss_steps"], 3)
        self.assertEqual(metrics["bf16_autocast"], 1)
        self.assertTrue(all(
            torch.isfinite(torch.tensor(value))
            for value in metrics.values() if isinstance(value, (int, float))
        ))
        self.assertEqual(actor_before, actor_state_digest(model.actor))
        self.assertTrue(all(parameter.grad is None for parameter in model.actor.parameters()))
        self.assertTrue(any(
            not torch.equal(value, critic_before[name])
            for name, value in model.critic.state_dict().items()
        ))

    def test_equal_length_ragged_chunks_are_batched_for_gpu_efficiency(self) -> None:
        chunks = self._chunks()
        wider = deepcopy(chunks[0])
        for payload in wider["step_payloads"]:
            for inputs_name in ("actor_inputs", "critic_inputs"):
                inputs = payload[inputs_name]
                inputs["entity_tokens"] = torch.ones(1, 2, dtype=torch.long)
                inputs["entity_positions"] = torch.ones(1, 2, dtype=torch.long)
                inputs["entity_relations"] = torch.zeros(1, 2, dtype=torch.long)
                inputs["entity_numeric"] = torch.ones(1, 2, 3)
                inputs["entity_mask"] = torch.ones(1, 2, dtype=torch.bool)
        trainer = Stage1CriticTrainer(
            self._model(),
            device="cpu",
            config=CriticTrainingConfig(chunk_batch_size=2),
        )

        metrics = trainer.train_update([chunks[0], wider])

        self.assertEqual(metrics["loss_steps"], 6)
        self.assertEqual(metrics["chunks"], 2)
        self.assertEqual(metrics["chunk_batches"], 1)
        self.assertEqual(metrics["chunk_batch_size"], 2)

    def test_fresh_chunk_evaluation_is_finite_and_does_not_update_critic(self) -> None:
        trainer = Stage1CriticTrainer(
            self._model(), device="cpu", config=CriticTrainingConfig(chunk_batch_size=2)
        )
        before = {
            name: value.detach().clone()
            for name, value in trainer.model.critic.state_dict().items()
        }

        metrics = trainer.evaluate_chunks(self._chunks() * 2)

        self.assertEqual(metrics["loss_steps"], 6)
        self.assertEqual(metrics["chunk_batches"], 1)
        self.assertTrue(all(
            torch.isfinite(torch.tensor(value))
            for value in metrics.values() if isinstance(value, (int, float))
        ))
        self.assertTrue(all(
            torch.equal(value, before[name])
            for name, value in trainer.model.critic.state_dict().items()
        ))

    def test_explained_variance_is_global_not_chunk_batch_average(self) -> None:
        chunks = self._chunks() * 2
        first_model = self._model()
        second_model = self._model()
        second_model.load_state_dict(first_model.state_dict())
        single = Stage1CriticTrainer(
            first_model, device="cpu", config=CriticTrainingConfig(chunk_batch_size=1)
        ).evaluate_chunks(chunks)
        batched = Stage1CriticTrainer(
            second_model, device="cpu", config=CriticTrainingConfig(chunk_batch_size=2)
        ).evaluate_chunks(chunks)

        self.assertAlmostEqual(
            single["explained_variance"], batched["explained_variance"], places=6
        )

    def test_atomic_bundle_is_complete_restorable_and_retains_three(self) -> None:
        model = self._model()
        trainer = Stage1CriticTrainer(model, device="cpu")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for update in range(1, 5):
                trainer.global_update = update
                trainer.save_checkpoint(root, {"loss": float(update)})
            versions = sorted(root.glob("checkpoint-*.pt"))
            self.assertEqual(
                [path.name for path in versions],
                [
                    "checkpoint-000000000002.pt",
                    "checkpoint-000000000003.pt",
                    "checkpoint-000000000004.pt",
                ],
            )
            latest = torch.load(root / "latest.pt", map_location="cpu", weights_only=False)
            required = {
                "actor_fp32_master", "actor_inference_state", "critic",
                "optimizer", "rng", "config", "metrics", "global_update",
            }
            self.assertFalse(required.difference(latest))
            self.assertEqual(latest["global_update"], 4)
            self.assertEqual(latest["actor_sha256"], actor_state_digest(model.actor))
            self.assertIsNotNone(latest["actor_fp32_master"])
            self.assertFalse(any(root.glob(".*.tmp")))

            restored = Stage1CriticTrainer(self._model(), device="cpu")
            # The Actor initialization is deterministic only after restoring the
            # exact immutable state referenced by this bundle.
            restored.model.actor.load_state_dict(latest["actor_inference_state"])
            restored.actor_sha256 = actor_state_digest(restored.model.actor)
            metrics = restored.restore_checkpoint(root / "latest.pt", restore_rng=False)
            self.assertEqual(restored.global_update, 4)
            self.assertEqual(metrics["loss"], 4.0)

    def test_nonfinite_target_fails_before_optimizer_step(self) -> None:
        model = self._model()
        trainer = Stage1CriticTrainer(model, device="cpu")
        chunks = self._chunks()
        chunks[0]["step_payloads"][0]["critic_targets"]["future_damage"] = [
            float("nan"), 0.0
        ]
        before = {
            name: value.detach().clone() for name, value in model.critic.state_dict().items()
        }
        with self.assertRaisesRegex(FloatingPointError, "NaN/Inf"):
            trainer.train_update(chunks)
        self.assertTrue(all(
            torch.equal(value, before[name])
            for name, value in model.critic.state_dict().items()
        ))


if __name__ == "__main__":
    unittest.main()
