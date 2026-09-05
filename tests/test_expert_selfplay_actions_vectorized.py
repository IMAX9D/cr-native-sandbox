from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

import expert_selfplay_v1.actions as actions
from expert_v1.training_v1.model import ExpertPolicyConfig, ExpertPolicyOutput


class VectorizedActionTests(unittest.TestCase):
    def test_all_slot_probabilities_and_gradients_match_scalar_evaluation(self):
        torch.manual_seed(31)
        prefix = (2, 3)
        config = ExpertPolicyConfig(
            grid_channels=8, public_scalar_size=16, card_vocab_size=32,
            ability_vocab_size=42, max_ability_slots=16,
        )
        shapes = [prefix, (*prefix, 2), (*prefix, 4), (*prefix, 4, 576),
                  (*prefix, 16), (*prefix, 16, 576)]
        source_values = [torch.randn(shape) * 0.2 for shape in shapes]
        target = ExpertPolicyOutput(*(torch.randn(shape) * 0.2 for shape in shapes),
                                    (torch.empty(0), torch.empty(0)))
        masks = actions.ExpertActionMasks(
            action_kind=torch.ones(*prefix, 2, dtype=torch.bool),
            cards=torch.ones(*prefix, 4, dtype=torch.bool),
            positions=torch.ones(*prefix, 4, 576, dtype=torch.bool),
            abilities=torch.ones(*prefix, 16, dtype=torch.bool),
            ability_positions=torch.ones(*prefix, 16, 576, dtype=torch.bool),
            ability_requires_target=torch.ones(*prefix, 16, dtype=torch.bool),
        )
        # Include legal WAIT-only frames and unavailable child slots.
        masks.action_kind[0, 0] = False
        masks.cards[0, 0] = False
        masks.positions[0, 0] = False
        masks.abilities[0, 0] = False
        masks.ability_positions[0, 0] = False
        event = torch.tensor([[False, True, True], [True, False, True]])
        action = actions.RecordedExpertAction(
            event_happened=event,
            action_kind=torch.tensor([[0, 0, 1], [1, 0, 1]]),
            card_slot=torch.ones(prefix, dtype=torch.long),
            position=torch.full(prefix, 17, dtype=torch.long),
            ability_slot=torch.full(prefix, 2, dtype=torch.long),
            ability_position=torch.full(prefix, 35, dtype=torch.long),
            ability_requires_target=torch.ones(prefix, dtype=torch.bool),
        )
        original = actions._distribution

        def scalar_slots(logits, mask):
            if logits.ndim != 4:
                return original(logits, mask)
            rows = [original(logits[..., slot, :], mask[..., slot, :])
                    for slot in range(logits.shape[-2])]
            return (torch.stack([row[0] for row in rows], -2),
                    torch.stack([row[1] for row in rows], -2),
                    torch.stack([row[2] for row in rows], -1))

        outcomes = []
        for scalar in (True, False):
            values = [value.clone().requires_grad_() for value in source_values]
            source = ExpertPolicyOutput(*values, (torch.empty(0), torch.empty(0)))
            with patch.object(actions, "_distribution", scalar_slots if scalar else original):
                evaluated = actions.evaluate_expert_action(
                    output=source, config=config, masks=masks, action=action,
                    delta_ticks=torch.full(prefix, 12),
                )
                kl = actions.expert_policy_kl(
                    source=source, target=target, config=config, masks=masks,
                    delta_ticks=torch.full(prefix, 12),
                )
                loss = evaluated.log_prob.total.sum() + .01 * evaluated.entropy.sum() + kl.sum()
                gradients = torch.autograd.grad(loss, values)
                outcomes.append((evaluated.log_prob.total.detach(), evaluated.entropy.detach(),
                                 kl.detach(), gradients))
        for left, right in zip(outcomes[0][:3], outcomes[1][:3], strict=True):
            torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-6)
        for left, right in zip(outcomes[0][3], outcomes[1][3], strict=True):
            self.assertTrue(torch.isfinite(right).all())
            torch.testing.assert_close(left, right, rtol=2e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
