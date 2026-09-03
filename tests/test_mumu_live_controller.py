from __future__ import annotations

import unittest
import os

import numpy as np

from native_core.mumu_live_controller import (
    ScreenLayout,
    _native_state,
    _position_masks,
)
from native_core.mumu_live_monitor import _pid_alive


class MuMuLiveControllerTests(unittest.TestCase):
    def test_monitor_recognizes_current_windows_process(self) -> None:
        self.assertTrue(_pid_alive(os.getpid()))

    def test_portrait_layout_keeps_touches_inside_game_view(self) -> None:
        layout = ScreenLayout.from_size(1080, 1920)
        self.assertEqual(layout.hand_point(0), (297, 1751))
        for position in (0, 17, 558, 575):
            x, y = layout.deployment_point(position)
            self.assertGreaterEqual(x, 0)
            self.assertLess(x, 1080)
            self.assertGreaterEqual(y, 0)
            self.assertLess(y, 1920)

    def test_landscape_layout_never_touches_side_bars(self) -> None:
        layout = ScreenLayout.from_size(1920, 1080)
        left = (1920 - 1080 * 9 / 16) / 2
        right = left + 1080 * 9 / 16
        for slot in range(4):
            x, _ = layout.hand_point(slot)
            self.assertGreater(x, left)
            self.assertLess(x, right)

    def test_masks_limit_troops_to_canonical_own_half(self) -> None:
        decks = [[
            {"card_id": 26000021},
            {"card_id": 28000000},
            *({"card_id": 26000000} for _ in range(6)),
        ]] * 2
        card_mask, positions = _position_masks(
            decks, 1, [0, 1, -1, -1], 100_000
        )
        np.testing.assert_array_equal(card_mask, [True, True, False, False])
        self.assertTrue(positions[0, : 16 * 18].all())
        self.assertFalse(positions[0, 16 * 18 :].any())
        self.assertTrue(positions[1].all())

    def test_native_state_projects_private_players_and_towers(self) -> None:
        entity = {
            "coherent": True,
            "game_tick": 100,
            "entities": [
                {"category": 5000000, "side": 0, "x": 9000, "y": 3000,
                 "card_id": -1, "level": 11, "hp": 4000, "max_hp": 4000},
                {"category": 5000001, "side": 1, "x": 9000, "y": 29000,
                 "card_id": -1, "level": 11, "hp": 4000, "max_hp": 4000},
                {"category": 5000006, "side": 1, "x": 9000, "y": 20000,
                 "card_id": 26000021, "level": 11, "hp": 1000, "max_hp": 1000},
            ],
        }
        private = {
            "coherent": True,
            "players": [
                {"side": 0, "elixir_raw": 50000, "refill_timer": 0,
                 "next_deck_index": 4, "hand_deck_indices": [0, 1, 2, 3]},
                {"side": 1, "elixir_raw": 60000, "refill_timer": 0,
                 "next_deck_index": 4, "hand_deck_indices": [0, 1, 2, 3]},
            ],
        }
        state = _native_state(entity, private)
        self.assertEqual(state["tick"], 100)
        self.assertEqual(len(state["players"]), 2)
        self.assertEqual(len(state["episode"]["crown_towers"]), 2)
        self.assertEqual(state["entities"][2]["card_id"], 26000021)


if __name__ == "__main__":
    unittest.main()
