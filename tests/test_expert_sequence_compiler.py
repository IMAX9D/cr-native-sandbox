from __future__ import annotations

import unittest

import numpy as np
import torch

from expert_v1.compile_sequence_dataset import compile_side_sequence
from expert_v1.training_v1.losses import behaviour_cloning_loss
from expert_v1.training_v1.model import ExpertPolicyConfig, RecurrentExpertPolicy
from expert_v1.training_v1.schema import OBSERVATION_SEQUENCE, POSITION_COUNT
from expert_v1.upgrade_base_cycles import solve_cycle


def side_record(side: str, offset: int) -> dict[str, object]:
    deck = [f"card-{index}" for index in range(8)]
    indices = [0, 1, 2, 3, 4, 5, 0, 1, 6, 2]
    solved = solve_cycle(indices)
    return {
        "battle_tag": "BATTLE",
        "side": side,
        "base_deck": deck,
        "action_count": len(indices),
        "ticks": [100 + offset + index * 40 for index in range(len(indices))],
        "card_indices": indices,
        "actor_x": [500 + index * 1000 for index in range(len(indices))],
        "actor_y": [17_500 + (index % 4) * 1000 for index in range(len(indices))],
        "event_indices": [index * 2 + offset for index in range(len(indices))],
        "ability_events_complete": True,
        "ability_ticks": [],
        "ability_event_indices": [],
        **solved,
    }


class ExpertSequenceCompilerTests(unittest.TestCase):
    def test_exact_cycle_sequence_has_no_native_grid(self) -> None:
        own = side_record("team", 0)
        enemy = side_record("opponent", 10)
        vocabulary = {f"card-{index}": index + 1 for index in range(8)}
        sequence, reason = compile_side_sequence(own, enemy, vocabulary)
        self.assertIsNone(reason)
        assert sequence is not None
        self.assertNotIn("grid", sequence)
        self.assertGreater(len(sequence["play_now"]), 0)
        self.assertTrue(np.all(sequence["timing_exposure_ticks"] > 0))
        supervised = sequence["card_label_mask"]
        slots = sequence["card_slot"][supervised]
        hands = sequence["hand_tokens"][supervised]
        self.assertTrue(np.all((slots >= 0) & (slots < 4)))
        selected = hands[np.arange(len(slots)), slots]
        self.assertTrue(np.all(selected > 0))
        self.assertTrue(
            np.all(sequence["previous_event_position"] <= POSITION_COUNT)
        )

    def test_sequence_model_and_point_process_loss(self) -> None:
        batch_size, steps = 2, 5
        config = ExpertPolicyConfig(
            grid_channels=0,
            public_scalar_size=7,
            card_vocab_size=16,
            ability_vocab_size=1,
            max_ability_slots=1,
            card_embedding_size=8,
            spatial_size=8,
            hidden_size=16,
            observation_mode=OBSERVATION_SEQUENCE,
        )
        model = RecurrentExpertPolicy(config)
        own_deck = torch.arange(1, 9).repeat(batch_size, steps, 1)
        hand = own_deck[..., :4].clone()
        next_card = own_deck[..., 4].clone()
        play = torch.tensor([[0, 1, 0, 1, 0], [0, 0, 1, 0, 1]], dtype=torch.bool)
        card_slot = torch.where(play, torch.zeros_like(play, dtype=torch.long), -100)
        position = torch.where(play, torch.full_like(card_slot, 100), -100)
        batch = {
            "public_scalars": torch.zeros(batch_size, steps, 7),
            "own_deck_tokens": own_deck,
            "hand_tokens": hand,
            "next_card_token": next_card,
            "revealed_enemy_tokens": torch.zeros(batch_size, steps, 8, dtype=torch.long),
            "previous_event_card_token": torch.ones(batch_size, steps, dtype=torch.long),
            "previous_event_side": torch.ones(batch_size, steps, dtype=torch.long),
            "previous_event_position": torch.full(
                (batch_size, steps), POSITION_COUNT, dtype=torch.long
            ),
            "delta_ticks": torch.full((batch_size, steps), 20.0),
            "timing_exposure_ticks": torch.full((batch_size, steps), 20.0),
            "card_mask": torch.ones(batch_size, steps, 4, dtype=torch.bool),
            "play_now": play,
            "card_slot": card_slot,
            "position": position,
            "timing_label_mask": torch.ones_like(play),
            "card_label_mask": play,
            "position_label_mask": play,
            "sample_weight": torch.ones(batch_size, steps),
            "loss_mask": torch.ones_like(play),
        }
        output = model.forward_batch(batch)
        loss, metrics = behaviour_cloning_loss(output, batch, config)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(output.position_logits.shape, (batch_size, steps, 4, 576))
        self.assertEqual(metrics["card_count"], float(play.sum()))
        self.assertEqual(metrics["ability_count"], 0.0)


if __name__ == "__main__":
    unittest.main()
