from __future__ import annotations

import unittest

import numpy as np
import torch

from selfplay_v2.model import ContinuousRatePolicyValueNet
from selfplay_v2.ppo import ContinuousRatePPOTrainer
from selfplay_v2.rollout import TimedAgentTrajectory
from training.ppo import PPOConfig
from training.run_contract import model_digest
from training.schema import GRID_CHANNELS, PRIVILEGED_SIZE, SCALAR_SIZE


class V2PPOTests(unittest.TestCase):
    def test_recurrent_update_is_finite_and_changes_parameters(self):
        torch.manual_seed(77)
        model = ContinuousRatePolicyValueNet(
            hidden_size=16, lambda_initial=3.0
        )
        trajectory = TimedAgentTrajectory(side=0, seed=1)
        hidden = model.initial_hidden(1, device="cpu")
        for step in range(8):
            grid = np.zeros((GRID_CHANNELS, 32, 18), dtype=np.float32)
            grid[0, step, step] = 1.0
            scalars = np.zeros(SCALAR_SIZE, dtype=np.float32)
            privileged = np.zeros(PRIVILEGED_SIZE, dtype=np.float32)
            card_mask = np.asarray([True, True, False, False], dtype=np.bool_)
            position_mask = np.zeros((4, 32 * 18), dtype=np.bool_)
            position_mask[0, [7, 19]] = True
            position_mask[1, [35, 91]] = True
            h_before = (
                hidden[0][0, 0].detach().numpy().copy(),
                hidden[1][0, 0].detach().numpy().copy(),
            )
            sample = model.sample_batch(
                torch.from_numpy(grid).unsqueeze(0),
                torch.from_numpy(scalars).unsqueeze(0),
                torch.from_numpy(privileged).unsqueeze(0),
                torch.from_numpy(card_mask).unsqueeze(0),
                torch.from_numpy(position_mask).unsqueeze(0),
                hidden,
            )[0]
            hidden = sample.hidden
            trajectory.grid.append(grid)
            trajectory.scalars.append(scalars)
            trajectory.privileged.append(privileged)
            trajectory.card_masks.append(card_mask)
            trajectory.position_masks.append(position_mask)
            trajectory.cards.append(sample.card)
            trajectory.positions.append(sample.position)
            trajectory.timing_valids.append(sample.timing_valid)
            trajectory.rates.append(sample.rate)
            trajectory.play_probabilities.append(sample.play_probability)
            trajectory.policy_entropies.append(sample.entropy)
            trajectory.log_probabilities.append(sample.log_probability)
            trajectory.values.append(sample.value)
            trajectory.rewards.append(0.1 if step == 7 else 0.0)
            trajectory.dones.append(step == 7)
            trajectory.hidden_h.append(h_before[0])
            trajectory.hidden_c.append(h_before[1])
        config = PPOConfig(
            epochs=1,
            burn_in=0,
            train_length=8,
            chunk_batch_size=1,
        )
        trainer = ContinuousRatePPOTrainer(
            model, device=torch.device("cpu"), config=config
        )
        before = model_digest(model)
        metrics = trainer.update([trajectory])
        after = model_digest(model)
        self.assertNotEqual(before, after)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertGreater(metrics["rate_mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
