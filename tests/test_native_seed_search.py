from __future__ import annotations

from dataclasses import replace
import unittest

from expert_v1.native_replay_plan import compile_battle, materialize_replay
from expert_v1.native_seed_search import (
    NativeSeedSearchError,
    clear_native_seed_cache,
    layout_accepts_sequence,
    resolve_fixed_native_seed,
    resolve_native_seed,
    seed_cache_key,
)


def source_battle() -> dict:
    cards = [
        "knight-ev1", "archers", "giant", "skeletons",
        "musketeer", "hog-rider", "cannon", "arrows",
    ]
    return {
        "schema_version": 3,
        "battle_tag": "SEED-SEARCH-TEST",
        "duration_seconds": 10,
        "draft": False,
        "card_plays": [
            {"time_raw": 20, "side": side, "card": "knight", "x": 1, "y": 1}
            for side in ("team", "opponent")
        ],
        "ability_plays": [],
        "elixir_stats": {
            "team": {"Ability": {"count": 0}},
            "opponent": {"Ability": {"count": 0}},
        },
        "rounds": [{
            side: [{
                "full_deck": cards,
                "card_levels": {card: 11 for card in cards},
                "tower_troop": "tower-princess",
                "complete": True,
            }]
            for side in ("team", "opponent")
        }],
    }


def template() -> dict:
    return {
        "rndSeed": 1,
        "battle": {
            "deck0": {"sp": [], "sc": []},
            "deck1": {"sp": [], "sc": []},
        },
    }


class FakeSeedEnv:
    def __init__(self, compatible_seed: int) -> None:
        self.compatible_seed = compatible_seed
        self.seeds: list[int] = []

    def reset(self, replay: dict, *, warmup_steps: int) -> dict:
        self.assert_source_order(replay)
        seed = int(replay["rndSeed"])
        self.seeds.append(seed)
        if seed == self.compatible_seed:
            layout = [0, 1, 2, 3, 4, 5, 6, 7]
        else:
            layout = [1, 2, 3, 4, 0, 5, 6, 7]
        return {
            "tick": warmup_steps,
            "players": [
                {
                    "side": side,
                    "hand_deck_indices": layout[:4],
                    "cycle_deck_indices": layout[4:],
                }
                for side in range(2)
            ],
        }

    @staticmethod
    def assert_source_order(replay: dict) -> None:
        for side in range(2):
            rows = replay["battle"][f"deck{side}"]["sp"]
            assert rows[0]["d"] == 26_000_000
            assert rows[0]["el"] == 1


class NativeSeedSearchTest(unittest.TestCase):
    def setUp(self) -> None:
        clear_native_seed_cache()

    def test_layout_transition_checks_complete_sequence(self) -> None:
        self.assertTrue(
            layout_accepts_sequence((0, 1, 2, 3, 4, 5, 6, 7), (0, 4, 5))
        )
        self.assertFalse(
            layout_accepts_sequence((0, 1, 2, 3, 4, 5, 6, 7), (4,))
        )

    def test_search_is_bounded_ascending_and_preserves_source_slots(self) -> None:
        plan = compile_battle(source_battle())
        env = FakeSeedEnv(compatible_seed=3)
        result = resolve_native_seed(
            env, plan, template(), preferred_seed=424_242,
            maximum_seeds_to_test=3,
            warmup_tick=7,
        )
        self.assertEqual(env.seeds, [1, 2, 3])
        self.assertEqual(result.chosen_seed, 3)
        self.assertEqual(result.seeds_tested, 3)
        self.assertFalse(result.source_seed_recovered)
        self.assertEqual(result.state["tick"], 7)
        self.assertEqual(result.mappings, (tuple(range(8)), tuple(range(8))))

    def test_cache_key_and_validation(self) -> None:
        plan = compile_battle(source_battle())
        first = resolve_native_seed(
            FakeSeedEnv(3), plan, template(), maximum_seeds_to_test=4
        )
        second_env = FakeSeedEnv(3)
        second = resolve_native_seed(
            second_env, plan, template(), maximum_seeds_to_test=4
        )
        self.assertEqual(first.chosen_seed, second.chosen_seed)
        self.assertTrue(second.cache_hit)
        self.assertTrue(second.cache_validated)
        self.assertEqual(second_env.seeds, [3])
        higher_level = replace(
            plan,
            sides=tuple(
                replace(
                    side,
                    deck=tuple(replace(card, level=16) for card in side.deck),
                )
                for side in plan.sides
            ),
        )
        self.assertNotEqual(seed_cache_key(plan), seed_cache_key(higher_level))

    def test_fixed_seed_replay_resets_once_without_repeating_search(self) -> None:
        plan = compile_battle(source_battle())
        searched_env = FakeSeedEnv(compatible_seed=3)
        searched = resolve_native_seed(
            searched_env, plan, template(), maximum_seeds_to_test=4
        )
        self.assertEqual(searched_env.seeds, [1, 2, 3])

        replay_env = FakeSeedEnv(compatible_seed=3)
        replayed = resolve_fixed_native_seed(
            replay_env,
            plan,
            template(),
            chosen_seed=searched.chosen_seed,
            warmup_tick=7,
        )
        self.assertEqual(replay_env.seeds, [3])
        self.assertEqual(replayed.chosen_seed, searched.chosen_seed)
        self.assertEqual(replayed.native_resets, 1)
        self.assertEqual(replayed.seeds_tested, 0)
        self.assertEqual(
            replayed.resolution_mode, "fixed_preflight_seed_replay"
        )
        self.assertEqual(replayed.state["tick"], 7)

    def test_exhaustion_fails_closed(self) -> None:
        plan = compile_battle(source_battle())
        with self.assertRaisesRegex(
            NativeSeedSearchError, "native_compatible_seed_not_found"
        ) as caught:
            resolve_native_seed(
                FakeSeedEnv(3), plan, template(), maximum_seeds_to_test=2
            )
        self.assertEqual(caught.exception.seeds_tested, 2)

    def test_cache_never_bypasses_a_stricter_search_bound(self) -> None:
        plan = compile_battle(source_battle())
        resolve_native_seed(
            FakeSeedEnv(3), plan, template(), maximum_seeds_to_test=3
        )
        bounded = FakeSeedEnv(3)
        with self.assertRaises(NativeSeedSearchError):
            resolve_native_seed(
                bounded, plan, template(), maximum_seeds_to_test=2
            )
        self.assertEqual(bounded.seeds, [1, 2])

    def test_materialization_ignores_legacy_calibration_permutation(self) -> None:
        plan = compile_battle(source_battle())
        legacy = tuple({
            "hand_deck_indices": [7, 6, 5, 4],
            "cycle_deck_indices": [3, 2, 1, 0],
        } for _ in range(2))
        replay, mappings = materialize_replay(plan, template(), legacy, seed=7)
        self.assertEqual(mappings, (tuple(range(8)), tuple(range(8))))
        self.assertEqual(replay["battle"]["deck0"]["sp"][0]["el"], 1)


if __name__ == "__main__":
    unittest.main()
