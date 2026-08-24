from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from selfplay_v2.action import (
    initial_rate_bias,
    rate_distribution,
    timing_log_probability,
)
from selfplay_v2.model import ContinuousRatePolicyValueNet
from training.schema import GRID_CHANNELS, PRIVILEGED_SIZE, SCALAR_SIZE


class ContinuousActionRateTests(unittest.TestCase):
    def test_rate_is_invariant_to_time_partition(self):
        for rate in (0.01, 0.1, 0.3, 1.0, 10.0):
            for duration in (0.05, 0.5, 3.0):
                direct = math.exp(-rate * duration)
                for parts in (1, 2, 5, 20, 100):
                    partitioned = math.exp(-rate * duration / parts) ** parts
                    self.assertAlmostEqual(direct, partitioned, places=12)

    def test_initial_bias_recovers_requested_rate(self):
        lambda_max = 20.0
        for initial in (0.1, 0.2, 0.3, 0.5):
            bias = initial_rate_bias(initial, lambda_max)
            value = torch.tensor([bias], dtype=torch.float64)
            distribution = rate_distribution(
                value, lambda_max=lambda_max, tick_seconds=0.05
            )
            self.assertAlmostEqual(distribution.rate.item(), initial, places=12)
            self.assertAlmostEqual(
                distribution.play_probability.item(),
                1.0 - math.exp(-initial * 0.05),
                places=12,
            )

    def test_forced_no_play_has_zero_actor_log_probability(self):
        distribution = rate_distribution(torch.tensor([-100.0, 0.0, 100.0]))
        valid = torch.tensor([False, False, False])
        play = torch.tensor([False, False, False])
        result = timing_log_probability(
            distribution, play_now=play, timing_valid=valid
        )
        self.assertTrue(torch.equal(result, torch.zeros_like(result)))
        self.assertTrue(torch.isfinite(distribution.log_play).all())

    def test_sample_and_evaluate_log_probability_match(self):
        torch.manual_seed(123)
        model = ContinuousRatePolicyValueNet(
            hidden_size=16, lambda_max=20.0, lambda_initial=19.0
        ).eval()
        batch = 2
        grid = torch.zeros(batch, GRID_CHANNELS, 32, 18)
        scalars = torch.zeros(batch, SCALAR_SIZE)
        privileged = torch.zeros(batch, PRIVILEGED_SIZE)
        card_masks = torch.tensor([
            [True, False, True, False],
            [False, False, False, False],
        ])
        position_masks = torch.zeros(batch, 4, 32 * 18, dtype=torch.bool)
        position_masks[0, 0, [7, 19]] = True
        position_masks[0, 2, [35, 77, 91]] = True
        hidden = model.initial_hidden(batch, device="cpu")
        samples = model.sample_batch(
            grid,
            scalars,
            privileged,
            card_masks,
            position_masks,
            hidden,
            deterministic=True,
        )
        self.assertTrue(samples[0].play_now)
        self.assertFalse(samples[1].play_now)
        self.assertEqual(samples[1].log_probability, 0.0)
        cards = torch.tensor([[item.card] for item in samples])
        positions = torch.tensor([[item.position] for item in samples])
        evaluated, entropy, _values, _next_hidden = model.evaluate_actions(
            grid.unsqueeze(1),
            scalars.unsqueeze(1),
            privileged.unsqueeze(1),
            card_masks.unsqueeze(1),
            position_masks.unsqueeze(1),
            cards,
            positions,
            hidden,
        )
        expected = np.asarray([item.log_probability for item in samples])
        self.assertTrue(np.allclose(evaluated[:, 0].detach().numpy(), expected))
        self.assertEqual(float(entropy[1, 0].detach()), 0.0)


if __name__ == "__main__":
    unittest.main()
