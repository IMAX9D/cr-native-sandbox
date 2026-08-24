from __future__ import annotations

import unittest

from native_core.deployment import deployment_mask


def opening_entities() -> list[dict[str, int]]:
    return [
        {"category": 5_000_000, "side": 0, "x": 9000, "y": 3000,
         "card_id": -1, "hp": 4824},
        {"category": 5_000_001, "side": 0, "x": 3500, "y": 6500,
         "card_id": -1, "hp": 3052},
        {"category": 5_000_002, "side": 0, "x": 14500, "y": 6500,
         "card_id": -1, "hp": 3052},
        {"category": 5_000_003, "side": 1, "x": 9000, "y": 29000,
         "card_id": -1, "hp": 4824},
        {"category": 5_000_004, "side": 1, "x": 3500, "y": 25500,
         "card_id": -1, "hp": 3052},
        {"category": 5_000_005, "side": 1, "x": 14500, "y": 25500,
         "card_id": -1, "hp": 3052},
    ]


class DeploymentRulesTests(unittest.TestCase):
    def test_units_are_restricted_to_friendly_half_and_tower_footprints(self):
        rows = deployment_mask(
            ["1" * 18 for _ in range(32)],
            {"entities": opening_entities()},
            side=0,
            card_id=26000000,
        )
        self.assertTrue(all("1" not in row for row in rows[17:]))
        self.assertEqual(rows[3][9], "0")
        self.assertEqual(rows[6][3], "0")
        self.assertEqual(rows[10][9], "1")

    def test_destroyed_enemy_left_princess_opens_only_left_pocket(self):
        entities = opening_entities()
        entities[4]["hp"] = 0
        rows = deployment_mask(
            ["1" * 18 for _ in range(32)],
            {"entities": entities},
            side=0,
            card_id=26000000,
        )
        self.assertEqual(rows[18][3], "1")
        self.assertEqual(rows[18][14], "0")

    def test_spells_keep_native_target_mask(self):
        native = ["10" * 9 for _ in range(32)]
        self.assertEqual(
            deployment_mask(
                native, {"entities": opening_entities()},
                side=0, card_id=28000001,
            ),
            native,
        )

    def test_invalid_grid_fails_closed(self):
        with self.assertRaises(ValueError):
            deployment_mask(
                ["1" * 18], {"entities": []}, side=0, card_id=26000000
            )


if __name__ == "__main__":
    unittest.main()
