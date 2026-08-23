from __future__ import annotations

import random
import unittest

import numpy as np

from training.schema import ActionMaskCache, deployment_mask


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
