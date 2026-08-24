from __future__ import annotations

import json
from pathlib import Path
import unittest

from native_core.card_catalog import (
    card_cost,
    catalog,
    standard_card_ids,
)
from native_core.decks import build_replay, parse_deck_text, resolve_card


ROOT = Path(__file__).resolve().parents[1]


class FullCardCatalogTests(unittest.TestCase):
    def test_live_catalog_is_versioned_and_complete(self) -> None:
        values = catalog()
        self.assertEqual(len(values), 152)
        self.assertEqual(len(standard_card_ids()), 122)
        self.assertEqual(values[26000102]["internal_name"], "Berserker")
        self.assertEqual(card_cost(26000102), 2)
        self.assertEqual(
            values[26000069]["active_ability"], "SkeletonKing"
        )
        self.assertEqual(values[26000072]["active_ability_mana_cost"], 1)
        self.assertEqual(
            values[26000102]["hero_active_ability"],
            "BerserkerHeroAbility",
        )
        self.assertEqual(values[26000102]["hero_ability_mana_cost"], 3)

    def test_names_resolve_without_a_fixed_eight_card_table(self) -> None:
        self.assertEqual(resolve_card("Hog Rider"), 26000021)
        self.assertEqual(resolve_card("goblin-drill"), 27000013)
        self.assertEqual(resolve_card("26000102"), 26000102)

    def test_arbitrary_decks_build_the_native_replay_contract(self) -> None:
        template = json.loads(
            (ROOT / "examples" / "eight-card-bootstrap.json").read_text()
        )
        left = [
            "IceSpirits", "Berserker", "Skeletons", "Cannon",
            "Fireball", "Log", "Musketeer", "HogRider",
        ]
        right = [
            "Cannon", "IceGolemite", "IceSpirits", "Log",
            "Fireball", "Skeletons", "Musketeer", "HogRider",
        ]
        replay = build_replay(template, left, right, seed=99, level=11)
        self.assertEqual(replay["rndSeed"], 99)
        self.assertEqual(
            [item["d"] for item in replay["battle"]["deck0"]["sp"]],
            [resolve_card(item) for item in left],
        )
        self.assertEqual(
            [item["l"] for item in replay["battle"]["deck1"]["sp"]],
            [10] * 8,
        )

    def test_native_card_form_flags_are_explicit(self) -> None:
        template = json.loads(
            (ROOT / "examples" / "eight-card-bootstrap.json").read_text()
        )
        left = parse_deck_text(
            "Berserker@hero,Skeletons@evolution,Knight,Cannon,"
            "Fireball,Log,Musketeer,HogRider"
        )
        right = [
            "Cannon", "IceGolemite", "IceSpirits", "Log",
            "Fireball", "Skeletons", "Musketeer", "HogRider",
        ]
        replay = build_replay(template, left, right)
        self.assertEqual(replay["battle"]["deck0"]["sp"][0]["el"], 2)
        self.assertEqual(replay["battle"]["deck0"]["sp"][1]["el"], 1)


if __name__ == "__main__":
    unittest.main()
