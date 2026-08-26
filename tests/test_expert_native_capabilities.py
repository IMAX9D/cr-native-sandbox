from __future__ import annotations

import unittest

from expert_v1.native_capabilities import (
    ability_cards, ability_log_tier, resolve_live_ability, tower_troop,
)
from expert_v1.native_replay_plan import card_spec
from native_core.card_catalog import catalog


class ExpertNativeCapabilitiesTests(unittest.TestCase):
    def test_all_live_tower_troops_have_native_support_ids(self) -> None:
        self.assertEqual(tower_troop("tower-princess").support_card_id, 159000000)
        self.assertEqual(tower_troop("cannoneer").support_card_id, 159000001)
        self.assertEqual(tower_troop("dagger-duchess").support_card_id, 159000002)
        self.assertEqual(tower_troop("royal-chef").support_card_id, 159000004)

    def test_goblinstein_second_summon_owns_ability(self) -> None:
        cards = ability_cards((card_spec("goblinstein", 16),))
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0].ability_name, "goblinstein_ability")
        self.assertEqual(cards[0].mana_cost, 2)

    def test_native_probed_elite_barbarian_evolution_is_mapped(self) -> None:
        row = catalog()[26000043]
        self.assertEqual(row["evolution_form_id"], 13000043)
        self.assertEqual(row["evolution_cycles"], 2)
        self.assertTrue(card_spec("elite-barbarians-ev1", 16).runtime_form_supported)

    def test_runtime_resolution_is_unique_or_explicit_branch(self) -> None:
        allowed = ability_cards((card_spec("berserker-hero", 16),))
        state = {
            "tick": 123,
            "entities": [{
                "side": 0, "entity_id": 50, "native_card_id": 203000076,
                "ability_slot": 1, "ability_available": True,
            }],
        }
        self.assertEqual(
            resolve_live_ability(state, side=0, tick=123, allowed_cards=allowed).status,
            "unique",
        )
        state["entities"].append({
            "side": 0, "entity_id": 51, "native_card_id": 203000076,
            "ability_slot": 2, "ability_available": True,
        })
        self.assertEqual(
            resolve_live_ability(state, side=0, tick=123, allowed_cards=allowed).status,
            "branch_required",
        )

    def test_old_missing_ticks_stay_explicit(self) -> None:
        value = {"elixir_stats": {
            "team": {"Ability": {"count": 2}},
            "opponent": {"Ability": {"count": 0}},
        }}
        self.assertEqual(ability_log_tier(value), "count_only_missing_ticks")


if __name__ == "__main__":
    unittest.main()
