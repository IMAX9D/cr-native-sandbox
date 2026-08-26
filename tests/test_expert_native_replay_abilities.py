from __future__ import annotations

import unittest

from expert_v1.native_replay_plan import ReplayPlanError, compile_battle
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
        self.force_bad_layout = False
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
        self.submitted_ticks: list[int] = []
        self.reset_seeds: list[int] = []
        self.probed_slots: list[tuple[int, int]] = []
        self.deck_ids = [[26_000_000 + index for index in range(8)] for _ in range(2)]

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
        self.reset_seeds.append(int(replay.get("rndSeed", -1)))
        self.players = [
            {
                "side": side, "elixir": 10, "elixir_raw": 100000,
                "hand_deck_indices": [0, 1, 2, 3],
                "cycle_deck_indices": [4, 5, 6, 7],
                "next_deck_index": 4, "refill_timer": 0,
            }
            for side in range(2)
        ]
        if self.force_bad_layout:
            for player in self.players:
                player["hand_deck_indices"] = [1, 2, 3, 4]
                player["cycle_deck_indices"] = [0, 5, 6, 7]
                player["next_deck_index"] = 0
        battle = replay.get("battle", {})
        for side in (0, 1):
            rows = battle.get(f"deck{side}", {}).get("sp", [])
            if len(rows) == 8:
                self.deck_ids[side] = [int(row["d"]) for row in rows]
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
        self.submitted_ticks.append(self.tick)
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

    def probe_grid(self, *, side: int, deck_index: int) -> dict:
        self.probed_slots.append((side, deck_index))
        rows = ["1" * 18 for _ in range(32)]
        return {
            "width": 18, "height": 32, "cell_size": 1000,
            "valid_cells": 18 * 32,
            "resolved_data_id": self.deck_ids[side][deck_index],
            "packed_selection": 3 << 28,
            "card_cost": 3, "card_cost_raw": 30_000,
            "selection_form_index": -1,
            "selection_strategy": "canonical",
            "selection_builder_rva": "0xd5b770",
            "selection_root_vtable_rva": "0x1234",
            "rows": rows,
        }


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
    def test_fixed_seed_second_pass_preserves_action_and_terminal_semantics(self) -> None:
        plan = compile_battle(ability_battle())
        env = FakeNativeEnv()
        preflight = execute_plan(env, plan, template(), calibration(), seed=19)
        reset_count_after_preflight = len(env.reset_seeds)
        full = execute_plan(
            env,
            plan,
            template(),
            calibration(),
            seed=999,
            fixed_seed=preflight.chosen_seed,
        )

        self.assertTrue(preflight.teacher_forced_success, preflight.failure)
        self.assertTrue(full.teacher_forced_success, full.failure)
        self.assertEqual(
            preflight.action_acceptance_sequence,
            full.action_acceptance_sequence,
        )
        self.assertEqual(preflight.terminal_match, full.terminal_match)
        self.assertEqual(
            preflight.terminal_tower_hp_match,
            full.terminal_tower_hp_match,
        )
        self.assertEqual(len(env.reset_seeds), reset_count_after_preflight + 1)
        self.assertEqual(env.reset_seeds[-1], preflight.chosen_seed)
        self.assertEqual(full.seed_search_native_resets, 1)
        self.assertEqual(full.seeds_tested, 0)
        self.assertEqual(
            full.layout_resolution_mode, "fixed_preflight_seed_replay"
        )

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
        self.assertEqual(result.action_execution_tick_offset, 1)
        self.assertEqual(result.ability_resolutions[0]["source_tick"], 25)
        self.assertEqual(result.ability_resolutions[0]["execution_tick"], 26)
        self.assertEqual(env.submitted_ticks, [21, 26])
        self.assertEqual(env.submitted[-1], [
            {"type": "ability", "side": 0, "entity_id": 50}
        ])
        profile = result.json()["native_teacher_forced_profile"]
        self.assertEqual(profile["name"], "royaleapi_native_teacher_forced")
        self.assertEqual(profile["version"], 1)
        self.assertFalse(profile["diagnostic_override"])

    def test_fixed_seed_layout_is_revalidated_before_trace_or_mask_probe(self) -> None:
        plan = compile_battle(ability_battle())
        env = FakeNativeEnv()
        preflight = execute_plan(env, plan, template(), calibration())
        self.assertTrue(preflight.teacher_forced_success, preflight.failure)
        env.force_bad_layout = True
        env.probed_slots.clear()

        full = execute_plan(
            env,
            plan,
            template(),
            calibration(),
            fixed_seed=preflight.chosen_seed,
            capture_deployment_masks=True,
        )
        self.assertFalse(full.teacher_forced_success)
        self.assertIn(
            "native_seed_search_layout_revalidation_failed_sides_",
            full.failure or "",
        )
        self.assertEqual(full.layout_resolution_mode, "fixed_preflight_seed_replay")
        self.assertEqual(full.seed_search_native_resets, 1)
        self.assertEqual(full.deployment_mask_probe_rpc_count, 0)
        self.assertEqual(env.probed_slots, [])

    def test_explicit_offset_zero_remains_a_diagnostic_override(self) -> None:
        plan = compile_battle(ability_battle())
        env = FakeNativeEnv()
        result = execute_plan(
            env,
            plan,
            template(),
            calibration(),
            action_execution_tick_offset=0,
        )

        self.assertTrue(result.accepted, result.failure)
        self.assertEqual(env.submitted_ticks, [20, 25])
        self.assertEqual(result.action_execution_tick_offset, 0)
        profile = result.json()["native_teacher_forced_profile"]
        self.assertEqual(profile["native_execution_boundary"], "source_tick+0")
        self.assertTrue(profile["diagnostic_override"])

    def test_offset_one_keeps_source_ticks_and_uses_data_i_coordinates(self) -> None:
        source = ability_battle()
        source["card_plays"][0].update({
            "x_raw": 1_250, "y_raw": 2_500, "data_i": 1,
        })
        source["card_plays"][1].update({
            "x_raw": 1_000, "y_raw": 2_000, "data_i": 0,
        })
        plan = compile_battle(source)
        env = FakeNativeEnv()
        result = execute_plan(
            env, plan, template(), calibration(),
            action_execution_tick_offset=1,
        )

        self.assertTrue(result.accepted, result.failure)
        self.assertEqual(env.submitted_ticks, [21, 26])
        self.assertEqual(
            env.submitted[0],
            [
                {
                    "type": "play", "side": 0, "deck_index": 0,
                    "x": 1_250, "y": 2_500,
                },
                {
                    "type": "play", "side": 1, "deck_index": 0,
                    "x": 17_000, "y": 30_000,
                },
            ],
        )
        self.assertEqual(result.action_execution_tick_offset, 1)
        self.assertIn("source_tick+1", result.action_tick_provenance)
        self.assertEqual(result.coordinate_provenance, "royaleapi_raw_data_i_to_native_v1")
        self.assertEqual(result.coordinate_audit["data_i_zero_events"], 1)
        self.assertEqual(result.coordinate_audit["data_i_one_events"], 1)
        resolution = result.ability_resolutions[0]
        self.assertEqual(resolution["source_tick"], 25)
        self.assertEqual(resolution["execution_tick"], 26)
        self.assertEqual(resolution["tick"], 26)
        first_decision = result.decision_records[0]
        self.assertEqual(first_decision["source_tick"], 20)
        self.assertEqual(first_decision["execution_tick"], 21)
        self.assertEqual(first_decision["tick"], 21)

    def test_deploy_ability_conflict_is_checked_on_source_tick(self) -> None:
        source = ability_battle()
        source["ability_plays"][0]["time_raw"] = 20
        with self.assertRaisesRegex(
            ReplayPlanError, "multiple deploy/ability actions.*native tick 20"
        ):
            compile_battle(source)

    def test_offset_rejects_values_outside_audited_boundaries(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 0 or 1"):
            execute_plan(
                FakeNativeEnv(), compile_battle(ability_battle()),
                template(), calibration(), action_execution_tick_offset=2,
            )

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

    def test_requested_mask_capture_fails_closed_when_full_deck_never_appears(self) -> None:
        source = ability_battle()
        source["card_plays"][0]["y"] = 9_500
        source["card_plays"][1]["y"] = 22_500
        plan = compile_battle(source)
        env = FakeNativeEnv()
        result = execute_plan(
            env, plan, template(), calibration(),
            capture_deployment_masks=True,
        )
        self.assertFalse(result.accepted)
        self.assertIn(
            "native_deployment_mask_capture_incomplete_slots_",
            result.failure or "",
        )
        self.assertEqual(result.deployment_mask_probe_rpc_count, 10)
        self.assertEqual(len(env.probed_slots), 10)
        self.assertEqual(len(set(env.probed_slots)), 10)
        self.assertFalse(result.deployment_mask_capture_complete)


if __name__ == "__main__":
    unittest.main()
