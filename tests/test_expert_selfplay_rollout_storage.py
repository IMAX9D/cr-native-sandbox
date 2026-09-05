from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from expert_selfplay_v1.contracts import BatchManifest
from expert_selfplay_v1.critic import PrivilegedCritic, PrivilegedCriticConfig
from expert_selfplay_v1.ledger import RolloutLedger
from expert_selfplay_v1.rollout import DecisionRecord, EpisodeHeader, LearnerEpisodeBuffer
from expert_selfplay_v1.rollout_storage import (
    CriticObservationAdapter,
    CriticPrivateObservation,
    ImmutableRolloutShardWriter,
    LearnerEpisodeChunker,
    verify_rollout_shard,
)


class RolloutStorageTests(unittest.TestCase):
    def test_semantic_hash_accepts_scalar_tensor_step_payloads(self) -> None:
        episode = self._episode(count=2)
        payloads = [
            {
                "value": torch.tensor(0.25),
                "action_kind": torch.tensor(1, dtype=torch.long),
                "legal": torch.tensor(True),
                "expanded_stride_zero": torch.tensor(7).expand(1),
                "empty_entities": torch.empty(0, dtype=torch.long),
            },
            {
                "value": torch.tensor(-0.5),
                "action_kind": torch.tensor(0, dtype=torch.long),
            },
        ]
        chunks = LearnerEpisodeChunker().chunk(
            episode,
            step_payloads=payloads,
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["step_payloads"][0]["value"].ndim, 0)

    @staticmethod
    def _critic_observation(*, entities: int = 2, private: int = 6):
        return CriticPrivateObservation(
            grid=np.zeros((8, 32, 18), dtype=np.float32),
            entity_tokens=np.arange(1, entities + 1, dtype=np.int64),
            entity_positions=np.arange(entities, dtype=np.int64),
            entity_relations=np.arange(entities, dtype=np.int64) % 2,
            entity_numeric=np.ones((entities, 3), dtype=np.float32),
            entity_mask=np.ones(entities, dtype=np.bool_),
            private_card_tokens=np.arange(1, private + 1, dtype=np.int64),
            private_card_owners=np.arange(private, dtype=np.int64) % 2,
            private_card_slots=np.arange(private, dtype=np.int64),
            private_card_mask=np.ones(private, dtype=np.bool_),
            scalars=np.zeros(20, dtype=np.float32),
        )

    @staticmethod
    def _header(*, episode_id: str = "E1", batch_id: str = "B1") -> EpisodeHeader:
        return EpisodeHeader(
            episode_id=episode_id,
            batch_id=batch_id,
            seed=7,
            learner_side=0,
            behavior_policy_version=3,
            behavior_actor_sha256="a" * 64,
            opponent_policy_id="OPP",
            opponent_actor_sha256="b" * 64,
            learner_deck_sha256="c" * 64,
            opponent_deck_sha256="d" * 64,
            curriculum_stage="stage1_critic",
            initial_hidden_sha256="e" * 64,
        )

    @staticmethod
    def _decision(index: int, count: int, **changes) -> DecisionRecord:
        kwargs = dict(
            tick=index + 1,
            delta_ticks=(index % 3) + 1,
            side=0,
            event_happened=bool(index % 2),
            action_kind=0,
            card_slot=0,
            position=0,
            ability_slot=0,
            ability_position=0,
            old_logp_total=-0.5,
            old_logp_timing=-0.2,
            old_logp_action_type=-0.1,
            old_logp_slot=-0.1,
            old_logp_position=-0.1,
            reward_damage_dealt=0.1,
            reward_damage_received=-0.2,
            reward_towers_dealt=0.0,
            reward_towers_received=0.0,
            reward_terminal=1.0 if index == count - 1 else 0.0,
            reward_total=0.9 if index == count - 1 else -0.1,
            value=0.05,
            terminated=index == count - 1,
            truncated=False,
            native_entity_count=2,
            encoded_entity_count=2,
        )
        kwargs.update(changes)
        return DecisionRecord(**kwargs)

    @classmethod
    def _episode(cls, count: int = 145, *, episode_id: str = "E1"):
        buffer = LearnerEpisodeBuffer(cls._header(episode_id=episode_id))
        for index in range(count):
            buffer.append(cls._decision(index, count))
        return buffer

    @staticmethod
    def _manifest() -> BatchManifest:
        return BatchManifest(
            run_id="RUN",
            batch_id="B1",
            policy_version=3,
            behavior_actor_sha256="a" * 64,
            encoder_schema_sha256="1" * 64,
            action_schema_sha256="2" * 64,
            reward_schema_sha256="3" * 64,
            native_lib_sha256="4" * 64,
            episode_count=32,
        )

    def test_critic_adapter_pads_ragged_sequence_and_forward_is_finite(self):
        config = PrivilegedCriticConfig(
            actor_latent_size=16,
            card_vocab_size=32,
            scalar_size=20,
        )
        adapter = CriticObservationAdapter(config)
        batch = adapter.encode_sequence([
            self._critic_observation(entities=2, private=6),
            self._critic_observation(entities=4, private=8),
        ])
        self.assertEqual(batch["grid"].shape, (1, 2, 8, 32, 18))
        self.assertEqual(batch["entity_tokens"].shape, (1, 2, 4))
        self.assertEqual(batch["private_card_tokens"].shape, (1, 2, 8))
        self.assertEqual(batch["entity_mask"].sum(dim=-1).tolist(), [[2, 4]])
        critic = PrivilegedCritic(config).eval()
        with torch.inference_mode():
            output = critic(actor_latent=torch.zeros(1, 2, 16), **batch)
        self.assertEqual(output.values.shape, (1, 2))
        self.assertTrue(torch.isfinite(output.values).all())
        self.assertTrue(torch.isfinite(output.wdl_logits).all())

    def test_critic_adapter_rejects_bad_shape_nonfinite_and_private_overflow(self):
        config = PrivilegedCriticConfig(
            actor_latent_size=8, card_vocab_size=32, scalar_size=20,
            private_slot_count=8,
        )
        adapter = CriticObservationAdapter(config)
        bad_grid = replace(
            self._critic_observation(),
            grid=np.zeros((8, 18, 32), dtype=np.float32),
        )
        with self.assertRaisesRegex(ValueError, "grid shape"):
            adapter.encode(bad_grid)
        bad_scalar = replace(
            self._critic_observation(),
            scalars=np.array([float("nan")] * 20, dtype=np.float32),
        )
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            adapter.encode(bad_scalar)
        with self.assertRaisesRegex(ValueError, "exceeds configured"):
            adapter.encode(self._critic_observation(private=9))
        masked_bad_token = replace(
            self._critic_observation(),
            entity_tokens=np.array([1, 99]),
            entity_mask=np.array([True, False]),
        )
        with self.assertRaisesRegex(ValueError, "out of vocabulary"):
            adapter.encode(masked_bad_token)

    def test_chunker_is_16_burn_in_64_unroll_with_full_episode_gae(self):
        chunks = LearnerEpisodeChunker().chunk(self._episode())
        self.assertEqual(len(chunks), 3)
        self.assertEqual(
            [(c["sequence_start"], c["loss_start"], c["loss_end"]) for c in chunks],
            [(0, 0, 64), (48, 64, 128), (112, 128, 145)],
        )
        self.assertEqual([c["burn_in"] for c in chunks], [0, 16, 16])

    def test_chunker_keeps_hidden_only_at_each_chunk_anchor(self):
        payloads = [
            {
                "actor_inputs": {
                    "hidden": (
                        torch.full((2, 1, 8), float(index)),
                        torch.full((2, 1, 8), float(-index)),
                    )
                }
            }
            for index in range(145)
        ]
        chunks = LearnerEpisodeChunker().chunk(
            self._episode(), step_payloads=payloads
        )
        for chunk in chunks:
            rows = chunk["step_payloads"]
            self.assertIn("hidden", rows[0]["actor_inputs"])
            self.assertTrue(all(
                "hidden" not in row["actor_inputs"] for row in rows[1:]
            ))
            anchor = int(chunk["sequence_start"])
            self.assertEqual(
                float(rows[0]["actor_inputs"]["hidden"][0][0, 0, 0]),
                float(anchor),
            )
        self.assertEqual([c["unroll"] for c in chunks], [64, 64, 17])
        self.assertEqual(int(chunks[1]["loss_mask"].sum()), 64)
        self.assertFalse(bool(chunks[1]["loss_mask"][:16].any()))
        self.assertTrue(bool(chunks[-1]["decisions"][-1]["terminated"]))
        self.assertTrue(torch.isfinite(chunks[-1]["advantages"]).all())
        self.assertTrue(torch.isfinite(chunks[-1]["returns"]).all())

    def test_chunker_rejects_partial_truncated_and_opponent_trajectories(self):
        partial = LearnerEpisodeBuffer(self._header())
        partial.append(self._decision(0, 2))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            LearnerEpisodeChunker().chunk(partial)

        truncated = LearnerEpisodeBuffer(self._header())
        truncated.append(self._decision(
            0, 1, terminated=False, truncated=True,
            reward_terminal=0.0, reward_total=-0.1,
        ))
        with self.assertRaisesRegex(ValueError, "time-truncated"):
            LearnerEpisodeChunker().chunk(truncated)

        frozen = self._episode(count=1).freeze()
        frozen["decisions"][0]["side"] = 1
        # Reseal to demonstrate that learner-side validation is independent of hash.
        unhashed = {key: value for key, value in frozen.items() if key != "content_sha256"}
        import hashlib
        frozen["content_sha256"] = hashlib.sha256(json.dumps(
            unhashed, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        with self.assertRaisesRegex(ValueError, "opponent trajectory"):
            LearnerEpisodeChunker().chunk(frozen)

    def test_atomic_shard_binds_manifest_is_verifiable_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = RolloutLedger(root / "ledger.sqlite")
            manifest = self._manifest()
            ledger.open_batch(
                manifest.batch_id,
                policy_version=manifest.policy_version,
                actor_sha256=manifest.behavior_actor_sha256,
            )
            ledger.transition(manifest.batch_id, "COLLECTING")
            writer = ImmutableRolloutShardWriter(
                root / "shards", manifest, ledger=ledger
            )
            first = writer.write("S-001", [self._episode(count=65)])
            self.assertTrue(first.created)
            self.assertTrue(first.ledger_recorded)
            self.assertTrue(first.manifest_path.is_file())
            self.assertTrue(first.torch_path.is_file())
            self.assertFalse(any(
                path.name.startswith(".S-001.") for path in (root / "shards").iterdir()
            ))
            verified = verify_rollout_shard(
                first.directory, expected_batch_manifest=manifest
            )
            self.assertEqual(verified["content_sha256"], first.content_sha256)
            self.assertEqual(ledger.shards("B1"), [("S-001", first.content_sha256, False)])
            second = writer.write("S-001", [self._episode(count=65)])
            self.assertFalse(second.created)
            self.assertFalse(second.ledger_recorded)
            self.assertEqual(second.content_sha256, first.content_sha256)
            ledger.close()

    def test_immutable_shard_rejects_conflict_and_manifest_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = ImmutableRolloutShardWriter(Path(temporary), self._manifest())
            writer.write("SAME", [self._episode(count=2)])
            with self.assertRaisesRegex(RuntimeError, "conflicting payload"):
                writer.write("SAME", [self._episode(count=3)])

            wrong = self._episode(count=1).freeze()
            wrong["header"]["behavior_policy_version"] = 4
            import hashlib
            unhashed = {key: value for key, value in wrong.items() if key != "content_sha256"}
            wrong["content_sha256"] = hashlib.sha256(json.dumps(
                unhashed, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest()
            with self.assertRaisesRegex(ValueError, "policy version"):
                writer.write("WRONG", [wrong])

    def test_tampered_torch_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = ImmutableRolloutShardWriter(Path(temporary), self._manifest())
            result = writer.write("TAMPER", [self._episode(count=2)])
            with result.torch_path.open("ab") as output:
                output.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "torch hash mismatch"):
                verify_rollout_shard(result.directory)


if __name__ == "__main__":
    unittest.main()
