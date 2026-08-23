from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from training.model import RecurrentPolicyValueNet
from training.ppo import PPOTrainer
from training.rollout import AgentTrajectory, EpisodeResult
from training.run_contract import (
    aggregate_behavior,
    build_checkpoint,
    model_digest,
    restore_checkpoint,
    semantic_digest,
)
from training.schema import RunStore
from training.train import _candidate_name, _crossed_thresholds


def _config(*, reward: str = "tower_hp_potential_v1") -> dict:
    return {
        "algorithm": "persistent_native_recurrent_ppo",
        "workers": 1,
        "avds": 1,
        "workers_per_avd": 1,
        "max_ticks": 7200,
        "environment_ports": [38031],
        "transport": "direct",
        "device": "cpu",
        "reward": reward,
        "reward_contract": {"schema": "tower_hp_only_v1"},
        "ppo": {"gamma": 0.99995},
        "episode_reset": "native",
        "truth_source": "libg",
        "action_legality": "native",
        "observation_schema": "compact_train_v1",
        "network_schema": "recurrent_policy_value_v1",
        "native_tick_hz": 20,
        "decision_frequency_hz": 20,
        "cuda_graph_inference": False,
    }


class RunContractTests(unittest.TestCase):
    def test_milestones_are_named_from_crossed_native_tick_thresholds(self):
        self.assertEqual(
            _crossed_thresholds(249_999, 1_010_000, 250_000),
            [250_000, 500_000, 750_000, 1_000_000],
        )
        self.assertEqual(_candidate_name(500_000), "P005")
        self.assertEqual(_candidate_name(1_000_000), "P010")

    def test_run_manifest_is_immutable_and_can_be_reopened(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = RunStore(Path(temporary))
            created = store.create({"reward": "tower_hp_potential_v1"}, run_id="a")
            reopened, manifest = store.open("a")
            self.assertEqual(reopened.root, created.root)
            self.assertEqual(manifest["config"]["reward"], "tower_hp_potential_v1")
            with self.assertRaises(FileExistsError):
                store.create({}, run_id="a")

    def test_checkpoint_restores_model_optimizer_counters_and_all_rngs(self):
        random.seed(11)
        np.random.seed(12)
        torch.manual_seed(13)
        model = RecurrentPolicyValueNet(hidden_size=16)
        trainer = PPOTrainer(model, device=torch.device("cpu"))
        loss = sum(parameter.square().mean() for parameter in model.parameters())
        loss.backward()
        trainer.optimizer.step()
        config = _config()
        digest = model_digest(model)
        checkpoint = build_checkpoint(
            model=model,
            optimizer=trainer.optimizer,
            iteration=7,
            native_ticks=250123,
            agent_steps=500246,
            completed_episodes=40,
            next_seed=1051,
            config=config,
            run_manifest={"kind": "native_eight_card_selfplay_run"},
            initial_model_digest="initial",
        )
        expected_random = random.random()
        expected_numpy = float(np.random.random())
        expected_torch = float(torch.rand(()))

        restored_model = RecurrentPolicyValueNet(hidden_size=16)
        restored_trainer = PPOTrainer(
            restored_model, device=torch.device("cpu")
        )
        counters = restore_checkpoint(
            checkpoint,
            model=restored_model,
            optimizer=restored_trainer.optimizer,
            expected_semantic_digest=semantic_digest(config),
        )
        self.assertEqual(counters["iteration"], 7)
        self.assertEqual(counters["native_ticks"], 250123)
        self.assertEqual(counters["agent_steps"], 500246)
        self.assertEqual(counters["completed_episodes"], 40)
        self.assertEqual(counters["next_seed"], 1051)
        self.assertEqual(model_digest(restored_model), digest)
        self.assertIsNone(checkpoint["scheduler"])
        self.assertEqual(random.random(), expected_random)
        self.assertEqual(float(np.random.random()), expected_numpy)
        self.assertEqual(float(torch.rand(())), expected_torch)
        self.assertTrue(restored_trainer.optimizer.state)

    def test_checkpoint_rejects_semantic_change(self):
        model = RecurrentPolicyValueNet(hidden_size=16)
        trainer = PPOTrainer(model, device=torch.device("cpu"))
        checkpoint = build_checkpoint(
            model=model,
            optimizer=trainer.optimizer,
            iteration=0,
            native_ticks=0,
            agent_steps=0,
            completed_episodes=0,
            next_seed=1,
            config=_config(),
            run_manifest={},
            initial_model_digest=model_digest(model),
        )
        with self.assertRaisesRegex(RuntimeError, "semantics"):
            restore_checkpoint(
                checkpoint,
                model=model,
                optimizer=trainer.optimizer,
                expected_semantic_digest=semantic_digest(
                    _config(reward="terminal")
                ),
            )

    def test_episode_failure_counts_each_nonterminal_episode_once(self):
        trajectories = (AgentTrajectory(0, 1), AgentTrajectory(1, 1))
        result = EpisodeResult(
            seed=1,
            tick=100,
            winner=None,
            outcome="truncated",
            terminated=False,
            truncated=True,
            wall_seconds=1.0,
            actions=0,
            trajectories=trajectories,
            action_log=[],
            state_hash=None,
            profile={},
            behavior={"match_ticks": 100},
        )
        behavior, histogram = aggregate_behavior([result])
        self.assertEqual(behavior["episode_failure_rate"], 1.0)
        self.assertEqual(histogram.shape, (8, 32, 18))


if __name__ == "__main__":
    unittest.main()
