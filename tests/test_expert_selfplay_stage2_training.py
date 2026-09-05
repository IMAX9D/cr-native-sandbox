from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from expert_v1.training_v1.model import ExpertPolicyConfig, RecurrentExpertPolicy
from expert_selfplay_v1.actions import ExpertActionMasks
from expert_selfplay_v1.batched_policy import BatchedPolicyService, PolicyRequest
from expert_selfplay_v1.contracts import BatchManifest
from expert_selfplay_v1.ledger import RolloutLedger
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
    def _two_wave_collection(self, root, actor, behavior_sha, *, policy_version=0):
        source = self._shard(root / "source", actor, behavior_sha, policy_version=policy_version)
        stored = torch.load(source / "rollout.pt", weights_only=False)["episodes"][0]
        chunk = stored["chunks"][0]
        batch = BatchManifest(
            run_id="two-wave-collection", batch_id="batch-multi", policy_version=policy_version,
            behavior_actor_sha256=behavior_sha, encoder_schema_sha256="1" * 64,
            action_schema_sha256="2" * 64, reward_schema_sha256="3" * 64,
            native_lib_sha256="4" * 64, episode_count=2,
        )
        collection = root / "collection"
        ledger = RolloutLedger(collection / "rollout-ledger.sqlite")
        ledger.open_batch(batch.batch_id, policy_version=policy_version, actor_sha256=behavior_sha)
        ledger.transition(batch.batch_id, "COLLECTING")
        writer = ImmutableRolloutShardWriter(collection / "rollouts", batch, ledger=ledger)
        shards = []
        try:
            for index in range(2):
                header = dict(stored["header"])
                header.update(episode_id=f"episode-{index}", batch_id=batch.batch_id)
                episode = LearnerEpisodeBuffer(EpisodeHeader(**header))
                for decision in chunk["decisions"]:
                    episode.append(DecisionRecord(**decision))
                shards.append(writer.write(
                    f"shard-{index}", [episode.freeze()],
                    step_payloads_by_episode={
                        header["episode_id"]: deepcopy(chunk["step_payloads"]),
                    },
                ).directory)
            ledger.close_collection(batch.batch_id, minimum_shards=2)
        finally:
            ledger.close()
        (collection / "manifest.json").write_text(
            json.dumps({"batch_manifest": asdict(batch)}), encoding="utf-8"
        )
        return shards

    def test_two_waves_complete_one_real_guarded_ppo_update(self):
        from scripts.train_expert_selfplay_stage2 import run

        torch.manual_seed(7)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, manifest, stage1, behavior_sha = self._fixture(root)
            actor = RecurrentExpertPolicy(self._config())
            actor.load_state_dict(torch.load(base, weights_only=False)["model_state"])
            shards = self._two_wave_collection(root, actor, behavior_sha)
            result = run(argparse.Namespace(
                base_checkpoint=base, continuation_checkpoint=stage1,
                expert_manifest=manifest, shard=shards, run_dir=root / "learner",
                device="cpu", cpu_threads=2, ppo_epochs=2, chunk_batch_size=2,
                preprocess_window_size=2, preprocess_batch_size=2,
                retain_checkpoints=3,
            ))
            self.assertEqual(result["policy_version"], 1)
            self.assertEqual(result["rollout"]["episodes"], 2)
            self.assertEqual(result["rollout"]["decisions"], 6)
            self.assertEqual(result["ledger_states"], ["COMMITTED"])
            self.assertTrue(Path(result["checkpoint"]).is_file())
            self.assertTrue(Path(result["behavior_export"]).is_file())

    def test_partial_wave_batch_is_rejected_before_loading_models(self):
        from scripts.train_expert_selfplay_stage2 import _admit_collection_batches

        torch.manual_seed(7)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, _manifest, _stage1, behavior_sha = self._fixture(root)
            actor = RecurrentExpertPolicy(self._config())
            actor.load_state_dict(torch.load(base, weights_only=False)["model_state"])
            shards = self._two_wave_collection(root, actor, behavior_sha)
            with self.assertRaisesRegex(RuntimeError, "all registered shards"):
                _admit_collection_batches(shards[:1])
            ledger = RolloutLedger(root / "collection" / "rollout-ledger.sqlite")
            try:
                self.assertEqual(ledger.state("batch-multi"), "CLOSED")
            finally:
                ledger.close()

    def test_prepared_cache_preserves_exact_cpu_ppo_update(self):
        torch.manual_seed(19)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, manifest, stage1, behavior_sha = self._fixture(root)
            actor = RecurrentExpertPolicy(self._config())
            actor.load_state_dict(torch.load(base, weights_only=False)["model_state"])
            shard = self._shard(root, actor, behavior_sha)
            outcomes = []
            for cache_gib in (0.0, 0.01):
                trainer = Stage2PPOTrainer(
                    base_inference_checkpoint=base, continuation_checkpoint=stage1,
                    expert_manifest=manifest, device="cpu",
                    config=Stage2TrainingConfig(
                        ppo_epochs=2, chunk_batch_size=1,
                        prepared_cache_gib=cache_gib,
                    ),
                )
                chunks, _batch, _summary = trainer.prepare_rollout(shard)
                metrics, guard, retry = trainer.train_update(chunks)
                outcomes.append((
                    {name: value.detach().clone() for name, value in trainer.model.state_dict().items()},
                    metrics, guard, retry,
                ))
                self.assertIsNone(trainer._prepared_cache)
            for name, value in outcomes[0][0].items():
                self.assertTrue(torch.equal(value, outcomes[1][0][name]), name)
            self.assertEqual(outcomes[0][2:], outcomes[1][2:])
            self.assertEqual(outcomes[0][1]["loss"], outcomes[1][1]["loss"])
            self.assertEqual(outcomes[1][1]["prepared_cache_misses"], 1)
            self.assertGreaterEqual(outcomes[1][1]["prepared_cache_hits"], 3)

    def test_resident_learner_reuses_models_across_two_real_policy_updates(self):
        from scripts.train_expert_selfplay_stage2 import PersistentStage2Learner

        torch.manual_seed(7)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, manifest, continuation, behavior_sha = self._fixture(root)
            export = base
            resident = PersistentStage2Learner()
            initial_trainer = None
            try:
                for version in range(2):
                    actor = RecurrentExpertPolicy(self._config())
                    state = torch.load(export, weights_only=False)["model_state"]
                    actor.load_state_dict(state)
                    behavior_sha = _state_digest(state)
                    shards = self._two_wave_collection(
                        root / f"batch-{version}", actor, behavior_sha,
                        policy_version=version,
                    )
                    run_args = argparse.Namespace(
                        base_checkpoint=base, continuation_checkpoint=continuation,
                        expert_manifest=manifest, shard=shards,
                        run_dir=root / f"learner-{version}", device="cpu",
                        cpu_threads=2, ppo_epochs=2, chunk_batch_size=2,
                        preprocess_window_size=2, preprocess_batch_size=2,
                        retain_checkpoints=3,
                    )
                    resident.initialize(run_args)
                    with patch.object(
                        resident.trainer, "prepare_rollout",
                        wraps=resident.trainer.prepare_rollout,
                    ) as prepare:
                        for shard in shards:
                            resident.prepare(shard)
                        result = resident.run(run_args)
                        self.assertEqual(prepare.call_count, len(shards))
                    if initial_trainer is None:
                        initial_trainer = resident.trainer
                    self.assertIs(resident.trainer, initial_trainer)
                    self.assertEqual(result["policy_version"], version + 1)
                    self.assertEqual(result["ledger_states"], ["COMMITTED"])
                    continuation = Path(result["checkpoint"])
                    export = Path(result["behavior_export"])
            finally:
                resident.close()

    def test_sequence_padding_preserves_valid_policy_outputs(self):
        torch.manual_seed(23)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, manifest, stage1, behavior_sha = self._fixture(root)
            actor = RecurrentExpertPolicy(self._config())
            actor.load_state_dict(torch.load(base, weights_only=False)["model_state"])
            shard = self._shard(root, actor, behavior_sha)
            trainer = Stage2PPOTrainer(
                base_inference_checkpoint=base, continuation_checkpoint=stage1,
                expert_manifest=manifest, device="cpu",
                config=Stage2TrainingConfig(chunk_batch_size=32, chunk_padding_multiple=80),
            )
            chunks, _manifest, _result = trainer.prepare_rollout(shard)
            short = deepcopy(chunks[0])
            for name in ("step_payloads", "decisions", "advantages", "returns", "loss_mask"):
                short[name] = short[name][:1]
            short["critic_targets"] = {
                name: value[:1] for name, value in short.get("critic_targets", {}).items()
            } if short.get("critic_targets") else short.get("critic_targets")
            if short.get("critic_targets") is None:
                short.pop("critic_targets", None)
            rows = [chunks[0], short]
            prepared = trainer._combine(rows)
            self.assertEqual(tuple(prepared["loss_mask"].shape), (2, 3))
            self.assertFalse(prepared["loss_mask"][1, 1:].any())
            self.assertTrue((prepared["actor_inputs"]["delta_ticks"][1, 1:] == 1).all())
            self.assertEqual(tuple(prepared["actor_inputs"]["hidden"][0].shape), (1, 2, 12))
            with torch.no_grad():
                combined = trainer.model.actor.forward_sequence(**prepared["actor_inputs"])
                single = trainer.model.actor.forward_sequence(**trainer._prepare_chunk(short)["actor_inputs"])
            for name in ("rate_logits", "card_logits", "position_logits"):
                torch.testing.assert_close(getattr(combined, name)[1:2, :1],
                                           getattr(single, name), rtol=1e-5, atol=1e-6)
            metrics, guard, _retry = trainer.train_update(rows)
            self.assertEqual(guard.action, "accept")
            self.assertEqual(metrics["evaluated_steps"], 4)

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
        *, include_hidden: bool = True, policy_version: int = 0,
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
            behavior_policy_version=policy_version,
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
            policy_version=policy_version,
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
                    preprocess_window_size=2,
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
            with self.assertRaisesRegex(RuntimeError, "hidden anchor"):
                trainer.prepare_rollout(shard)

    def test_preprocessing_batches_multiple_episodes_without_losing_rows(self) -> None:
        torch.manual_seed(11)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, manifest, stage1, behavior_sha = self._fixture(root)
            actor = RecurrentExpertPolicy(self._config())
            actor.load_state_dict(torch.load(base, weights_only=False)["model_state"])
            source = self._shard(root / "source", actor, behavior_sha)
            stored = torch.load(
                source / "rollout.pt", map_location="cpu", weights_only=False
            )["episodes"][0]
            chunk = stored["chunks"][0]
            mask = chunk["loss_mask"].bool().tolist()
            raw_decisions = [
                dict(row) for row, selected in zip(
                    chunk["decisions"], mask, strict=True
                ) if selected
            ]
            raw_payloads = [
                deepcopy(row) for row, selected in zip(
                    chunk["step_payloads"], mask, strict=True
                ) if selected
            ]
            episodes = []
            payloads_by_episode = {}
            for index in range(2):
                header = dict(stored["header"])
                header["episode_id"] = f"episode-{index + 1}"
                header["batch_id"] = "batch-multi"
                buffer = LearnerEpisodeBuffer(EpisodeHeader(**header))
                for decision in raw_decisions:
                    buffer.append(DecisionRecord(**decision))
                episodes.append(buffer.freeze())
                payloads_by_episode[header["episode_id"]] = deepcopy(raw_payloads)
            batch = BatchManifest(
                run_id="stage2-test-multi",
                batch_id="batch-multi",
                policy_version=0,
                behavior_actor_sha256=behavior_sha,
                encoder_schema_sha256="1" * 64,
                action_schema_sha256="2" * 64,
                reward_schema_sha256="3" * 64,
                native_lib_sha256="4" * 64,
                episode_count=2,
            )
            multi = ImmutableRolloutShardWriter(root / "multi", batch).write(
                "shard-1", episodes,
                step_payloads_by_episode=payloads_by_episode,
            ).directory
            serial_trainer = Stage2PPOTrainer(
                base_inference_checkpoint=base,
                continuation_checkpoint=stage1,
                expert_manifest=manifest,
                device="cpu",
                config=Stage2TrainingConfig(
                    ppo_epochs=1,
                    chunk_batch_size=1,
                    preprocess_window_size=2,
                    preprocess_batch_size=1,
                    bc_kl_soft_limit=1.0,
                ),
            )
            trainer = Stage2PPOTrainer(
                base_inference_checkpoint=base,
                continuation_checkpoint=stage1,
                expert_manifest=manifest,
                device="cpu",
                config=Stage2TrainingConfig(
                    ppo_epochs=1,
                    chunk_batch_size=1,
                    preprocess_window_size=2,
                    preprocess_batch_size=2,
                    bc_kl_soft_limit=1.0,
                ),
            )

            serial_chunks, serial_admitted, serial_rollout = (
                serial_trainer.prepare_rollout(multi)
            )
            chunks, admitted, rollout = trainer.prepare_rollout(multi)

            self.assertEqual(serial_admitted.digest(), admitted.digest())
            self.assertEqual(serial_rollout, rollout)
            self.assertEqual(admitted.episode_count, 2)
            self.assertEqual(rollout["episodes"], 2)
            self.assertEqual(rollout["decisions"], 6)
            self.assertEqual(len(chunks), 2)
            for serial_chunk, batched_chunk in zip(
                serial_chunks, chunks, strict=True
            ):
                torch.testing.assert_close(
                    serial_chunk["advantages"], batched_chunk["advantages"],
                    rtol=1e-5, atol=1e-6,
                )
                torch.testing.assert_close(
                    serial_chunk["returns"], batched_chunk["returns"],
                    rtol=1e-5, atol=1e-6,
                )
                for serial_payload, batched_payload in zip(
                    serial_chunk["step_payloads"],
                    batched_chunk["step_payloads"],
                    strict=True,
                ):
                    for name in serial_payload["bc_output"]:
                        torch.testing.assert_close(
                            serial_payload["bc_output"][name],
                            batched_payload["bc_output"][name],
                            rtol=1e-4, atol=1e-4,
                        )


if __name__ == "__main__":
    unittest.main()
