from __future__ import annotations

import unittest

import torch

from training.model import RecurrentPolicyValueNet
from training.schema import GRID_CHANNELS, PRIVILEGED_SIZE, SCALAR_SIZE


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class CudaGraphInferenceTests(unittest.TestCase):
    def test_graph_forward_and_hidden_lifetime_match_eager(self):
        torch.manual_seed(7)
        device = torch.device("cuda")
        model = RecurrentPolicyValueNet().to(device).eval()
        batch = 8
        grid = torch.randn(batch, GRID_CHANNELS, 32, 18, device=device)
        scalars = torch.randn(batch, SCALAR_SIZE, device=device)
        privileged = torch.randn(batch, PRIVILEGED_SIZE, device=device)
        hidden = model.initial_hidden(batch, device=device)
        card_mask = torch.ones(batch, 5, dtype=torch.bool, device=device)
        position_masks = torch.ones(
            batch, 4, 32 * 18, dtype=torch.bool, device=device
        )

        eager = model.sample_batch(
            grid, scalars, privileged, card_mask, position_masks, hidden,
            deterministic=True,
        )
        model.enable_cuda_graph_inference()
        graphed = model.sample_batch(
            grid, scalars, privileged, card_mask, position_masks, hidden,
            deterministic=True,
        )
        self.assertEqual(
            [(item.card, item.position) for item in eager],
            [(item.card, item.position) for item in graphed],
        )
        for expected, actual in zip(eager, graphed, strict=True):
            self.assertEqual(expected.log_probability, actual.log_probability)
            self.assertEqual(expected.value, actual.value)
            torch.testing.assert_close(
                expected.hidden[0], actual.hidden[0], rtol=0, atol=0
            )
            torch.testing.assert_close(
                expected.hidden[1], actual.hidden[1], rtol=0, atol=0
            )

        retained = (
            graphed[0].hidden[0].clone(),
            graphed[0].hidden[1].clone(),
        )
        model.sample_batch(
            grid + 1.0, scalars, privileged, card_mask, position_masks, hidden,
            deterministic=True,
        )
        torch.testing.assert_close(
            graphed[0].hidden[0], retained[0], rtol=0, atol=0
        )
        torch.testing.assert_close(
            graphed[0].hidden[1], retained[1], rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
