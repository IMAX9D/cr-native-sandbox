from __future__ import annotations

import unittest

import torch

from selfplay_v2.baselines import RandomRateLegalPolicy
from training.schema import GRID_CHANNELS, PRIVILEGED_SIZE, SCALAR_SIZE


class RandomRateLegalTests(unittest.TestCase):
    def test_actions_are_reproducible_and_strictly_legal(self):
        batch = 256
        grid = torch.zeros(batch, GRID_CHANNELS, 32, 18)
        scalars = torch.zeros(batch, SCALAR_SIZE)
        privileged = torch.zeros(batch, PRIVILEGED_SIZE)
        cards = torch.zeros(batch, 4, dtype=torch.bool)
        cards[:, 1] = True
        positions = torch.zeros(batch, 4, 32 * 18, dtype=torch.bool)
        positions[:, 1, [7, 91]] = True
        left = RandomRateLegalPolicy(rate=20.0, seed=10)
        right = RandomRateLegalPolicy(rate=20.0, seed=10)
        left_values = left.sample_batch(
            grid, scalars, privileged, cards, positions,
            left.initial_hidden(batch, device="cpu"),
        )
        right_values = right.sample_batch(
            grid, scalars, privileged, cards, positions,
            right.initial_hidden(batch, device="cpu"),
        )
        self.assertEqual(
            [(item.card, item.position) for item in left_values],
            [(item.card, item.position) for item in right_values],
        )
        for item in left_values:
            if item.play_now:
                self.assertEqual(item.card, 2)
                self.assertIn(item.position, (7, 91))

    def test_no_legal_card_forces_zero_log_probability(self):
        policy = RandomRateLegalPolicy(rate=0.3, seed=1)
        sample = policy.sample_batch(
            torch.zeros(1, GRID_CHANNELS, 32, 18),
            torch.zeros(1, SCALAR_SIZE),
            torch.zeros(1, PRIVILEGED_SIZE),
            torch.zeros(1, 4, dtype=torch.bool),
            torch.zeros(1, 4, 32 * 18, dtype=torch.bool),
            policy.initial_hidden(1, device="cpu"),
        )[0]
        self.assertFalse(sample.timing_valid)
        self.assertFalse(sample.play_now)
        self.assertEqual(sample.log_probability, 0.0)


if __name__ == "__main__":
    unittest.main()
