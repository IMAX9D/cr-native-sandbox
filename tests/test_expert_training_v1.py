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

    def test_multiple_refill_pads_are_valid_but_never_selectable(self) -> None:
        shard = self.root / self.manifest["splits"]["train"][0]
        hand_path = shard / "hand_tokens.npy"
        card_mask_path = shard / "card_mask.npy"
        kind_mask_path = shard / "action_kind_mask.npy"
        hand = np.load(hand_path)
        card_mask = np.load(card_mask_path)
        kind_mask = np.load(kind_mask_path)
        hand[1, 1:3] = 0
        card_mask[1, 1:3] = 0
        hand[2, :] = 0
        card_mask[2, :] = 0
        kind_mask[2, 0] = 0
        np.save(hand_path, hand, allow_pickle=False)
        np.save(card_mask_path, card_mask, allow_pickle=False)
        np.save(kind_mask_path, kind_mask, allow_pickle=False)
        self.assertEqual(
            validate_shard(shard, self.manifest),
            {"sequences": 4, "rows": 80},
        )

        labels_path = shard / "card_label_mask.npy"
        slots_path = shard / "card_slot.npy"
        labels = np.load(labels_path)
        slots = np.load(slots_path)
        labels[2] = 1
        slots[2] = 0
        np.save(labels_path, labels, allow_pickle=False)
        np.save(slots_path, slots, allow_pickle=False)
        with self.assertRaisesRegex(DatasetContractError, "empty hand slot"):
            validate_shard(shard, self.manifest)


if __name__ == "__main__":
    unittest.main()
