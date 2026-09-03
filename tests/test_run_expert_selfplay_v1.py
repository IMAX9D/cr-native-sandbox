from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from expert_selfplay_v1.actor_adapter import actor_state_digest
from expert_selfplay_v1.rollout import DecisionRecord, LearnerEpisodeBuffer
from expert_selfplay_v1.native_observation import NativeObservationContractError
from expert_v1.training_v1.model import ExpertPolicyConfig, RecurrentExpertPolicy
from scripts.run_expert_selfplay_v1 import (
    EXPERT_INFERENCE_KIND,
    _runtime_identity,
    load_base,
    parse_ports,
    run,
    RuntimeDependencies,
)


class Stage1RunnerContractTests(unittest.TestCase):
    def test_ports_are_explicit_ordered_and_unambiguous(self) -> None:
        self.assertEqual(parse_ports("39031,39033-39035"), [39031, 39033, 39034, 39035])
        for value, message in (
            ("39031,39031", "duplicate"),
            ("39035-39031", "descending"),
            ("0", "outside"),
            ("39031,", "empty"),
        ):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, message):
                parse_ports(value)

    @staticmethod
    def _manifest() -> dict:
        return {
            "kind": "test",
            "dimensions": {
                "grid_channels": 8,
                "public_scalar_size": 16,
                "card_vocab_size": 9,
                "ability_vocab_size": 2,
                "max_ability_slots": 2,
                "entity_numeric_size": 3,
            },
            "card_vocabulary": [
                "<PAD>",
                "knight@26000000", "archers@26000001", "goblins@26000002",
                "giant@26000003", "pekka@26000004", "minions@26000005",
                "balloon@26000006", "witch@26000007",
            ],
            "ability_vocabulary": ["<PAD>", "knight-hero@203000000"],
        }

    @staticmethod
    def _config() -> ExpertPolicyConfig:
        return ExpertPolicyConfig(
            grid_channels=8,
            public_scalar_size=16,
            card_vocab_size=9,
            ability_vocab_size=2,
            max_ability_slots=2,
            card_embedding_size=8,
            spatial_size=8,
            hidden_size=16,
        )

    def _write_fixture(self, root: Path) -> tuple[Path, Path]:
        manifest_path = root / "manifest.json"
        manifest_path.write_text(
            json.dumps(self._manifest(), sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        model = RecurrentExpertPolicy(self._config())
        checkpoint_path = root / "base.pt"
        torch.save({
            "kind": EXPERT_INFERENCE_KIND,
            "dataset_manifest_sha256": digest,
            "model_config": self._config().to_dict(),
            "model_state": model.state_dict(),
            "global_step": 7,
            "run_id": "expert-test",
        }, checkpoint_path)
        return checkpoint_path, manifest_path

    def test_checkpoint_manifest_encoder_are_bound_and_actor_is_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, manifest = self._write_fixture(Path(temporary))
            loaded = load_base(checkpoint, manifest, device=torch.device("cpu"))
            self.assertEqual(loaded.actor_sha256, actor_state_digest(loaded.actor))
            self.assertEqual(loaded.expert_manifest_sha256, hashlib.sha256(manifest.read_bytes()).hexdigest())
            self.assertTrue(all(not value.requires_grad for value in loaded.actor.parameters()))
            self.assertEqual(loaded.checkpoint_step, 7)

    def test_manifest_byte_change_fails_closed_before_collection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, manifest = self._write_fixture(Path(temporary))
            manifest.write_bytes(manifest.read_bytes() + b"\n")
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                load_base(checkpoint, manifest, device=torch.device("cpu"))

    def test_encoder_dimension_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, manifest = self._write_fixture(root)
            value = json.loads(manifest.read_text(encoding="utf-8"))
            value["dimensions"]["grid_channels"] = 7
            manifest.write_text(json.dumps(value), encoding="utf-8")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            payload["dataset_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
            torch.save(payload, checkpoint)
            with self.assertRaises(NativeObservationContractError):
                load_base(checkpoint, manifest, device=torch.device("cpu"))

    def test_runtime_manifest_requires_frozen_x86_64_lib_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "runtime.json"
            path.write_text(json.dumps({
                "runtime_version": "150535029",
                "abi": "x86_64",
                "libg_sha256": "a" * 64,
            }), encoding="utf-8")
            self.assertEqual(_runtime_identity(path)["libg_sha256"], "a" * 64)
            path.write_text(json.dumps({"abi": "arm64", "libg_sha256": "a" * 64}), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "x86_64"):
                _runtime_identity(path)

    @staticmethod
    def _deck_replay() -> dict:
        cards = [
            {"d": 26_000_000 + index, "l": 10, "el": 0}
            for index in range(8)
        ]
        return {
            "rndSeed": 1,
            "battle": {
                "deck0": {"sp": cards},
                "deck1": {"sp": list(reversed(cards))},
            },
        }

    @dataclass(frozen=True)
    class _Spec:
        worker_id: str
        env: object
        fixture: object
        header: object
        actor_hashes: object

    class _Client:
        def request(self, payload):
            if payload != {"op": "status"}:
                raise AssertionError("runner preflight must be status-only")
            return {"ok": True, "state": {
                "current_state_type": 4,
                "tick": 10,
                "battle": "0x1234",
                "replay_data": "0x5678",
                "read_ok": {
                    "root": True,
                    "context": True,
                    "manager_fields": True,
                    "battle": True,
                    "tick": True,
                },
            }}

    class _Env:
        def __init__(self, *, host, port, timeout, profile_native):
            self.host, self.port, self.client = host, port, Stage1RunnerContractTests._Client()
            self.closed = False

        def close(self):
            self.closed = True

    class _Collector:
        def __init__(self, *, encoder, policy_service, reward, max_decisions, rpc_workers, step_ticks):
            if rpc_workers != 1:
                raise AssertionError("smoke collector must use the selected Worker count")
            if not 1 <= step_ticks <= 16:
                raise AssertionError("runner passed an invalid step_ticks")

        def collect_batch(self, specs):
            rows = []
            for spec in specs:
                episode = LearnerEpisodeBuffer(spec.header)
                episode.append(DecisionRecord(
                    tick=101,
                    delta_ticks=1,
                    side=spec.header.learner_side,
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
                    reward_terminal=10.0,
                    reward_total=10.0,
                    value=0.0,
                    terminated=True,
                    truncated=False,
                    native_entity_count=1,
                    encoded_entity_count=1,
                ))
                rows.append(SimpleNamespace(
                    episode=episode,
                    step_payloads=({"source": "real-native-test-double"},),
                    terminal_episode={"terminated": True, "outcome": "side0"},
                ))
            return rows

    @dataclass(frozen=True)
    class _TrainerConfig:
        retain_checkpoints: int = 3

    class _Trainer:
        mutate_actor = False

        def __init__(self, model, *, config, device, actor_source_reference):
            self.model = model
            self.update = 0

        def train_update(self, chunks):
            self.update += 1
            if self.mutate_actor:
                with torch.no_grad():
                    next(self.model.actor.parameters()).add_(1.0)
            return {"loss": 1.0, "chunks": len(chunks), "global_update": self.update}

        def save_checkpoint(self, directory, metrics=None):
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / f"checkpoint-{self.update:012d}.pt"
            torch.save({"metrics": metrics}, target)
            return target

        def restore_checkpoint(self, checkpoint):
            value = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.update = int(value["global_update"])
            return dict(value.get("metrics", {}))

    def _run_args(self, root: Path, checkpoint: Path, manifest: Path) -> argparse.Namespace:
        runtime = root / "runtime.json"
        runtime.write_text(json.dumps({
            "runtime_version": "150535029", "abi": "x86_64", "libg_sha256": "a" * 64,
        }), encoding="utf-8")
        learner = root / "learner.json"
        learner.write_text(json.dumps(self._deck_replay()), encoding="utf-8")
        pool = root / "pool"
        pool.mkdir()
        (pool / "deck-01.json").write_text(json.dumps(self._deck_replay()), encoding="utf-8")
        return argparse.Namespace(
            checkpoint=checkpoint,
            expert_manifest=manifest,
            ports="39031",
            host="127.0.0.1",
            run_dir=root / "run-1",
            learner_deck=learner,
            opponent_deck_root=pool,
            runtime_manifest=runtime,
            episodes=None,
            smoke_workers=1,
            updates=1,
            step_ticks=1,
            max_decisions=10,
            timeout=1.0,
            seed=9,
            device="cpu",
            cpu_threads=1,
            retain_checkpoints=3,
            resume_checkpoint=None,
        )

    def test_resume_continues_update_number_in_a_fresh_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, manifest = self._write_fixture(root)
            resume = root / "resume.pt"
            torch.save({"global_update": 7, "metrics": {"loss": 2.0}}, resume)
            args = self._run_args(root, checkpoint, manifest)
            args.resume_checkpoint = resume
            dependencies = RuntimeDependencies(
                self._Collector, self._Spec, self._Trainer, self._TrainerConfig
            )
            result = run(args, dependencies=dependencies, env_type=self._Env)
            self.assertEqual(result["metrics"][0]["global_update"], 8)
            self.assertTrue(Path(result["checkpoint"]).name.endswith("000000000008.pt"))

    def test_multiple_updates_without_fresh_rollouts_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, manifest = self._write_fixture(root)
            args = self._run_args(root, checkpoint, manifest)
            args.updates = 2
            with self.assertRaisesRegex(ValueError, "one fresh rollout batch"):
                run(args, dependencies=RuntimeDependencies(
                    self._Collector, self._Spec, self._Trainer, self._TrainerConfig
                ), env_type=self._Env)

    def test_one_batch_commits_only_real_complete_rollout_and_critic_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, manifest = self._write_fixture(root)
            args = self._run_args(root, checkpoint, manifest)
            dependencies = RuntimeDependencies(
                self._Collector, self._Spec, self._Trainer, self._TrainerConfig
            )
            result = run(args, dependencies=dependencies, env_type=self._Env)
            self.assertEqual(result["status"], "completed")
            self.assertTrue(result["actor_unchanged"])
            self.assertEqual(result["ledger_state"], "COMMITTED")
            self.assertTrue((args.run_dir / "rollouts/shard-000001/rollout.pt").is_file())
            self.assertTrue(Path(result["checkpoint"]).is_file())
            self.assertEqual(json.loads((args.run_dir / "progress.json").read_text())["status"], "completed")

    def test_collect_only_publishes_closed_shard_without_constructing_trainer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, manifest = self._write_fixture(root)
            args = self._run_args(root, checkpoint, manifest)
            args.collect_only = True

            class ForbiddenTrainer(self._Trainer):
                def __init__(self, *args, **kwargs):
                    raise AssertionError("collect-only must not construct a Critic trainer")

            dependencies = RuntimeDependencies(
                self._Collector, self._Spec, ForbiddenTrainer, self._TrainerConfig
            )
            result = run(args, dependencies=dependencies, env_type=self._Env)

            self.assertEqual(result["status"], "collected")
            self.assertEqual(result["ledger_state"], "CLOSED")
            self.assertTrue((args.run_dir / "collection-result.json").is_file())
            self.assertTrue((args.run_dir / "rollouts/shard-000001/rollout.pt").is_file())
            self.assertFalse((args.run_dir / "checkpoints").exists())
            self.assertEqual(
                json.loads((args.run_dir / "progress.json").read_text())["status"],
                "collected",
            )

    def test_collect_only_can_version_learner_against_distinct_frozen_opponent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, manifest = self._write_fixture(root)
            opponent = root / "opponent.pt"
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state = dict(payload["model_state"])
            first_name = next(iter(state))
            state[first_name] = state[first_name].clone()
            state[first_name].view(-1)[0] += 0.25
            payload["model_state"] = state
            torch.save(payload, opponent)
            args = self._run_args(root, checkpoint, manifest)
            args.collect_only = True
            args.opponent_checkpoint = opponent
            args.policy_version = 7
            args.curriculum_stage = "stage2_reaction"
            args.opponent_policy_id = "BASE"
            dependencies = RuntimeDependencies(
                self._Collector, self._Spec, self._Trainer, self._TrainerConfig
            )

            result = run(args, dependencies=dependencies, env_type=self._Env)

            shard = torch.load(
                Path(result["shard"]) / "rollout.pt",
                map_location="cpu",
                weights_only=False,
            )
            header = shard["episodes"][0]["header"]
            self.assertEqual(header["behavior_policy_version"], 7)
            self.assertEqual(header["curriculum_stage"], "stage2_reaction")
            self.assertNotEqual(
                header["behavior_actor_sha256"], header["opponent_actor_sha256"]
            )
            self.assertEqual(header["opponent_policy_id"], "BASE")

    def test_actor_mutation_aborts_before_ledger_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, manifest = self._write_fixture(root)
            args = self._run_args(root, checkpoint, manifest)
            class MutatingTrainer(self._Trainer):
                mutate_actor = True
            dependencies = RuntimeDependencies(
                self._Collector, self._Spec, MutatingTrainer, self._TrainerConfig
            )
            with self.assertRaisesRegex(RuntimeError, "mutated"):
                run(args, dependencies=dependencies, env_type=self._Env)
            progress = json.loads((args.run_dir / "progress.json").read_text())
            self.assertEqual(progress["status"], "failed")


if __name__ == "__main__":
    unittest.main()
