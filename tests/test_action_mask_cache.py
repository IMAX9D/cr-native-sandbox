from __future__ import annotations

import random
import unittest

import numpy as np

from training.schema import ActionMaskCache, build_action_masks, deployment_mask


def _state(dead: set[tuple[int, int]]) -> dict:
    entities = []
    for side, y in ((0, 6500), (1, 25500)):
        king_y = 3000 if side == 0 else 29000
        entities.append({
            "category": 5_000_000 + side * 3, "card_id": -1,
            "side": side, "x": 9000, "y": king_y, "hp": 4824,
        })
        for lane, x in enumerate((3500, 14500), start=1):
            entities.append({
                "category": 5_000_000 + side * 3 + lane, "card_id": -1,
                "side": side, "x": x, "y": y,
                "hp": 0 if (side, lane) in dead else 3052,
            })
    return {"entities": entities}


class ActionMaskCacheTests(unittest.TestCase):
    def test_native_command_gate_forces_wait_only(self):
        state = _state(set())
        state.update({
            "tick": 3600,
            "episode": {"commands_allowed": False, "command_gate_code": 4},
            "players": [{
                "side": 0,
                "elixir": 10,
                "hand_deck_indices": [0, 1, 2, 3],
            }],
        })
        card_ids = [
            26000000, 26000001, 26000003, 26000010,
            26000014, 26000021, 27000000, 28000001,
        ]
        decks = [[{"card_id": card_id} for card_id in card_ids]]
        native_masks = {
            (0, index): ["1" * 18 for _ in range(32)]
            for index in range(4)
        }
        card_mask, _positions, _hand = build_action_masks(
            state,
            side=0,
            native_masks=native_masks,
            decks=decks,
        )
        self.assertEqual(card_mask.tolist(), [True, False, False, False, False])

    def test_cached_mask_is_bit_exact_for_tower_states_and_cards(self):
        rng = random.Random(12345)
        cache = ActionMaskCache()
        tower_states = (
            set(), {(1, 1)}, {(1, 2)}, {(1, 1), (1, 2)},
            {(0, 1)}, {(0, 2)}, {(0, 1), (0, 2)},
        )
        comparisons = 0
        for index in range(80):
            rows = [
                "".join("1" if rng.random() > 0.25 else "0" for _ in range(18))
                for _ in range(32)
            ]
            for side in (0, 1):
                for card_id in (26000000, 27000000, 28000001):
                    for dead in tower_states:
                        state = _state(dead)
                        expected = np.fromiter(
                            (
                                cell == "1"
                                for row in deployment_mask(
                                    rows, state, side=side, card_id=card_id
                                )
                                for cell in row
                            ),
                            dtype=np.bool_, count=32 * 18,
                        )
                        actual = cache.position_mask(
                            rows, state, side=side, deck_index=index,
                            card_id=card_id,
                        )
                        self.assertTrue(np.array_equal(expected, actual))
                        comparisons += 1
        self.assertEqual(comparisons, 3360)


if __name__ == "__main__":
    unittest.main()
