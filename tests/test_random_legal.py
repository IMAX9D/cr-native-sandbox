from __future__ import annotations

import math
import unittest

import torch

from training.baselines import RandomLegalPolicy
from training.schema import GRID_CHANNELS, PRIVILEGED_SIZE, SCALAR_SIZE


class RandomLegalTests(unittest.TestCase):
    @staticmethod
    def _inputs(batch: int = 32):
        grid = torch.zeros(batch, GRID_CHANNELS, 32, 18)
        scalars = torch.zeros(batch, SCALAR_SIZE)
        privileged = torch.zeros(batch, PRIVILEGED_SIZE)
        cards = torch.zeros(batch, 5, dtype=torch.bool)
        cards[:, 0] = True
        cards[:, 2] = True
        positions = torch.zeros(batch, 4, 32 * 18, dtype=torch.bool)
        positions[:, 1, 7] = True
        positions[:, 1, 91] = True
        return grid, scalars, privileged, cards, positions

    def test_never_leaves_card_or_position_mask(self):
        inputs = self._inputs()
        policy = RandomLegalPolicy(seed=123)
        hidden = policy.initial_hidden(32, device="cpu")
        samples = policy.sample_batch(*inputs, hidden)
        for sample in samples:
            self.assertIn(sample.card, (0, 2))
            if sample.card:
                self.assertIn(sample.position, (7, 91))
                self.assertAlmostEqual(sample.log_probability, -math.log(4.0))
            else:
                self.assertEqual(sample.position, 0)
                self.assertAlmostEqual(sample.log_probability, -math.log(2.0))

    def test_seed_is_reproducible_and_wait_only_is_supported(self):
        inputs = self._inputs(batch=8)
        left = RandomLegalPolicy(seed=9)
        right = RandomLegalPolicy(seed=9)
        left_actions = left.sample_batch(
            *inputs, left.initial_hidden(8, device="cpu")
        )
        right_actions = right.sample_batch(
            *inputs, right.initial_hidden(8, device="cpu")
        )
        self.assertEqual(
            [(item.card, item.position) for item in left_actions],
            [(item.card, item.position) for item in right_actions],
        )

        grid, scalars, privileged, cards, positions = self._inputs(batch=1)
        cards[:, 2] = False
        sample = left.sample_batch(
            grid,
            scalars,
            privileged,
            cards,
            positions,
            left.initial_hidden(1, device="cpu"),
        )[0]
        self.assertEqual((sample.card, sample.position), (0, 0))
        self.assertEqual(sample.log_probability, 0.0)


if __name__ == "__main__":
    unittest.main()
