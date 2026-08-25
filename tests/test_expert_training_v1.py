from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from expert_v1.training_v1.dataset import NativeExpertSequenceDataset, collate_sequences
from expert_v1.training_v1.losses import behaviour_cloning_loss
from expert_v1.training_v1.model import ExpertPolicyConfig, RecurrentExpertPolicy
from expert_v1.training_v1.schema import DatasetContractError, read_manifest, validate_shard
from expert_v1.training_v1.smoke_data import create_smoke_dataset


class ExpertTrainingV1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = create_smoke_dataset(Path(self.temporary.name) / "dataset")
        self.manifest = read_manifest(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_contract_and_recurrent_loss(self) -> None:
        shard = self.root / self.manifest["splits"]["train"][0]
        summary = validate_shard(shard, self.manifest)
        self.assertEqual(summary, {"sequences": 4, "rows": 80})
        dataset = NativeExpertSequenceDataset(
            self.root, split="train", sequence_length=16, burn_in=4
        )
        batch = collate_sequences([dataset[0], dataset[1]])
        dimensions = self.manifest["dimensions"]
        config = ExpertPolicyConfig(
            grid_channels=dimensions["grid_channels"],
            public_scalar_size=dimensions["public_scalar_size"],
            card_vocab_size=dimensions["card_vocab_size"],
            ability_vocab_size=dimensions["ability_vocab_size"],
            max_ability_slots=dimensions["max_ability_slots"],
            hidden_size=32,
            card_embedding_size=16,
            spatial_size=16,
        )
        model = RecurrentExpertPolicy(config)
        output = model.forward_batch(batch)
        loss, metrics = behaviour_cloning_loss(output, batch, config)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(metrics["timing_count"], 0)
        self.assertGreater(metrics["card_count"], 0)
        self.assertEqual(output.card_logits.shape[-1], 4)
        self.assertEqual(output.position_logits.shape[-1], 32 * 18)
        expected_rate = torch.sigmoid(model.rate_head.bias) * config.lambda_max
        self.assertAlmostEqual(float(expected_rate.item()), config.lambda_initial, places=5)

    def test_privileged_array_is_rejected(self) -> None:
        shard = self.root / self.manifest["splits"]["train"][0]
        np.save(shard / "enemy_hand.npy", np.zeros((80, 4), dtype=np.int16))
        with self.assertRaises(DatasetContractError):
            validate_shard(shard, self.manifest)


if __name__ == "__main__":
    unittest.main()

