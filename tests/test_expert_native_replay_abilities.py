from __future__ import annotations

import unittest

from expert_v1.native_replay_plan import compile_battle
from expert_v1.native_replay_runner import execute_plan


def ability_battle() -> dict:
    team = [
        "golden-knight", "archers", "giant", "skeletons",
        "musketeer", "hog-rider", "cannon", "arrows",
    ]
    opponent = [
        "knight", "archers", "giant", "skeletons",
        "musketeer", "hog-rider", "cannon", "arrows",
    ]
    return {
        "schema_version": 3,
        "battle_tag": "ABILITY000001",
        "duration_seconds": 10,
        "draft": False,
        "card_plays": [
            {
                "time_raw": 20, "side": "team", "card": "golden-knight",
                "x": 8500, "y": 17500, "marker_index": 0,
            },
            {
                "time_raw": 20, "side": "opponent", "card": "knight",
                "x": 8500, "y": 14500, "marker_index": 1,
            },
        ],
        "ability_plays": [{
            "time_raw": 25, "side": "team", "ability_id": None,
            "marker_index": 2,
        }],
        "elixir_stats": {
            "team": {"Ability": {"count": 1}},
            "opponent": {"Ability": {"count": 0}},
        },
        "rounds": [{
            "team": [{
                "full_deck": team,
                "card_levels": {card: 11 for card in team},
                "tower_troop": "tower-princess", "complete": True,
            }],
            "opponent": [{
                "full_deck": opponent,
                "card_levels": {card: 11 for card in opponent},
                "tower_troop": "tower-princess", "complete": True,
            }],
        }],
    }


class FakeNativeEnv:
    def __init__(self, *, branch: bool = False) -> None:
        self.branch = branch
        self.tick = 10
        self.players = [
            {
                "side": side, "elixir": 10, "elixir_raw": 100000,
                "hand_deck_indices": [0, 1, 2, 3],
                "cycle_deck_indices": [4, 5, 6, 7],
                "next_deck_index": 4, "refill_timer": 0,
            }
            for side in range(2)
        ]
        self.submitted: list[list[dict]] = []

    def _state(self) -> dict:
        entities = [{
            "side": 0, "category": 50, "entity_id": 50,
            "card_id": 26000074, "native_card_id": 26000074,
            "ability_slot": 1, "ability_available": True,
            "x": 8500, "y": 17500, "hp": 100, "max_hp": 100,
            "behavior_state": 0,
        }]
        if self.branch:
            entities.append({**entities[0], "category": 51, "entity_id": 51})
        return {
            "tick": self.tick,
            "state_hash": f"tick-{self.tick}",
            "players": self.players,
            "entities": entities,
            "episode": {
                "terminated": False, "truncated": False,
                "crown_towers": [],
            },
        }

    def reset(self, replay: dict, *, warmup_steps: int) -> dict:
        del replay
        self.tick = warmup_steps
        return self._state()

    def step(self, steps: int) -> dict:
        self.tick += steps
        return {
            "tick_after": self.tick, "stepped": steps,
            "episode": {"terminated": False, "truncated": False},
        }

    def observe_train(self) -> dict:
        return self._state()

    def joint_act(self, actions: list[dict]) -> dict:
        self.submitted.append(actions)
        for action in actions:
            if action["type"] != "play":
                continue
            player = self.players[int(action["side"])]
            played = int(action["deck_index"])
            player["hand_deck_indices"].remove(played)
            incoming = player["cycle_deck_indices"].pop(0)
            player["hand_deck_indices"].append(incoming)
            player["cycle_deck_indices"].append(played)
            player["next_deck_index"] = player["cycle_deck_indices"][0]
        return {"actions": [
            {"result": {"accepted": True, "result_code": 0}}
            for _ in actions
        ]}


def calibration() -> list[dict]:
    return [
        {
            "side": side,
            "hand_deck_indices": [0, 1, 2, 3],
            "cycle_deck_indices": [4, 5, 6, 7],
        }
        for side in range(2)
    ]


def template() -> dict:
    empty_deck = {"sp": [], "sc": []}
    return {"battle": {"deck0": dict(empty_deck), "deck1": dict(empty_deck)}}


class ExpertNativeReplayAbilityTests(unittest.TestCase):
    def test_schema3_unique_live_entity_executes_native_ability(self) -> None:
        plan = compile_battle(ability_battle())
        self.assertTrue(plan.native_replay_ready)
        self.assertEqual(plan.ability_log_tier, "observed_ticks_identity_runtime_resolved")
        self.assertEqual(plan.ability_events[0].source_marker_index, 2)

        env = FakeNativeEnv()
        result = execute_plan(env, plan, template(), calibration())
        self.assertTrue(result.accepted, result.failure)
        self.assertEqual(result.source_deploy_actions, 2)
        self.assertEqual(result.source_ability_events, 1)
        self.assertEqual(result.accepted_ability_actions, 1)
        self.assertTrue(result.ability_replay_complete)
        self.assertEqual(result.ability_resolution_counts, {"unique": 1})
        self.assertEqual(env.submitted[-1], [
            {"type": "ability", "side": 0, "entity_id": 50}
        ])

    def test_branch_required_stops_without_guessing(self) -> None:
        plan = compile_battle(ability_battle())
        env = FakeNativeEnv(branch=True)
        result = execute_plan(env, plan, template(), calibration())
        self.assertFalse(result.accepted)
        self.assertIn("ability_branch_required_marker_2", result.failure or "")
        self.assertEqual(result.ability_resolution_counts, {"branch_required": 1})
        self.assertEqual(
            result.ability_resolutions[0]["candidate_entity_ids"], (50, 51)
        )
        self.assertEqual(
            result.ability_resolutions[0]["execution"],
            "branch_required_unselected",
        )

    def test_explicit_branch_is_validated_and_executed(self) -> None:
        plan = compile_battle(ability_battle())
        env = FakeNativeEnv(branch=True)
        result = execute_plan(
            env, plan, template(), calibration(),
            ability_branch_choices={2: 51},
        )
        self.assertTrue(result.accepted, result.failure)
        self.assertEqual(result.accepted_ability_actions, 1)
        self.assertEqual(
            result.ability_resolutions[0]["execution"],
            "explicit_branch_executed",
        )
        self.assertEqual(env.submitted[-1][0]["entity_id"], 51)

    def test_terminal_diagnostic_does_not_invalidate_teacher_forcing(self) -> None:
        plan = compile_battle(ability_battle(), terminal_crowns=(1, 0))
        result = execute_plan(FakeNativeEnv(), plan, template(), calibration())
        self.assertTrue(result.teacher_forced_success, result.failure)
        self.assertTrue(result.accepted)
        self.assertFalse(result.terminal_validated)
        self.assertEqual(result.terminal_diagnostic_status, "native_terminal_missing")


if __name__ == "__main__":
    unittest.main()
