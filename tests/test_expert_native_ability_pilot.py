from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from expert_v1.native_ability_pilot import (
    AbilityPilotTask,
    execute_ability_task,
    select_ability_positive_tasks,
    sha256_file,
)
from expert_v1.native_replay_plan import compile_battle
from expert_v1.tick_store_v1.shard import ShardReader, WorkerShardSink


def ability_battle(tag: str = "ABILITY-PILOT-001") -> dict:
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
        "battle_tag": tag,
        "duration_seconds": 2,
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


def template() -> dict:
    empty = {"sp": [], "sc": []}
    return {"battle": {"deck0": dict(empty), "deck1": dict(empty)}}


def calibration() -> tuple[dict, dict]:
    return tuple({
        "side": side,
        "hand_deck_indices": [0, 1, 2, 3],
        "cycle_deck_indices": [4, 5, 6, 7],
    } for side in range(2))  # type: ignore[return-value]


class FakeTraceNativeEnv:
    def __init__(self, *, branch: bool = False, reject_ability: bool = False) -> None:
        self.branch = branch
        self.reject_ability = reject_ability
        self.tick = 10
        self.champion_spawned = False
        self.players: list[dict] = []
        self.submitted: list[list[dict]] = []
        self._reset_players()

    def _reset_players(self) -> None:
        self.players = [{
            "side": side,
            "elixir": 10,
            "elixir_raw": 100_000,
            "hand_deck_indices": [0, 1, 2, 3],
            "cycle_deck_indices": [4, 5, 6, 7],
            "next_deck_index": 4,
            "refill_timer": 0,
        } for side in range(2)]

    def _state(self) -> dict:
        entities = []
        if self.champion_spawned:
            entities.append({
                "side": 0, "category": 50, "entity_id": 50,
                "card_id": 26000074, "native_card_id": 26000074,
                "level": 11, "ability_slot": 1,
                "ability_state_code": 0, "ability_available": True,
                "ability_cooldown_remaining_ms": 0,
                "ability_charges_remaining": 1,
                "ability_pending_ms": 0, "ability_mana_cost": 1,
                "x": 8500, "y": 17500, "hp": 100, "max_hp": 100,
                "behavior_state": 0,
            })
            if self.branch:
                entities.append({**entities[0], "category": 51, "entity_id": 51})
        return {
            "schema_version": 1,
            "kind": "libg_native_train_state_v1",
            "coherent": True,
            "tick": self.tick,
            "state_hash": f"tick-{self.tick}",
            "players": self.players,
            "entities": entities,
            "episode": {
                "terminated": False,
                "truncated": False,
                "crowns": [0, 0],
                "crown_towers": [],
                "commands_allowed": True,
                "command_gate_code": 0,
                "native_phase": {
                    "battle": 1, "logic": 1, "logic_substate": 0,
                    "flag_1e9": 0,
                },
            },
        }

    def reset(self, replay: dict, *, warmup_steps: int) -> dict:
        del replay
        self.tick = warmup_steps
        self.champion_spawned = False
        self._reset_players()
        return self._state()

    def observe_train(self) -> dict:
        return self._state()

    def trace_train(self, steps: int) -> dict:
        initial = {
            "frame_index": 0,
            "advanced_steps": 0,
            "observation_complete": True,
            "state": self._state(),
        }
        frames = []
        for index in range(1, steps + 1):
            self.tick += 1
            frames.append({
                "frame_index": index,
                "advanced_steps": index,
                "observation_complete": True,
                "state": self._state(),
            })
        return {
            "schema_version": 1,
            "trace_schema_version": 1,
            "kind": "libg_native_train_tick_trace_v1",
            "encoding": "compact-train-v1",
            "fixed_dt": 0.05,
            "initial_frame": initial,
            "frames": frames,
            "stepped": steps,
            "final_frame_index": steps,
            "final_tick": self.tick,
            "terminal": False,
        }

    def joint_act(self, actions: list[dict]) -> dict:
        self.submitted.append([dict(action) for action in actions])
        response = []
        for action in actions:
            if action["type"] == "ability" and self.reject_ability:
                response.append({"result": {"accepted": False, "result_code": 17}})
                continue
            if action["type"] == "play":
                player = self.players[int(action["side"])]
                played = int(action["deck_index"])
                player["hand_deck_indices"].remove(played)
                incoming = player["cycle_deck_indices"].pop(0)
                player["hand_deck_indices"].append(incoming)
                player["cycle_deck_indices"].append(played)
                player["next_deck_index"] = player["cycle_deck_indices"][0]
                if int(action["side"]) == 0:
                    self.champion_spawned = True
            response.append({"result": {"accepted": True, "result_code": 0}})
        return {"actions": response}


def make_task(path: Path) -> AbilityPilotTask:
    source = json.loads(path.read_text(encoding="utf-8"))
    plan = compile_battle(source, terminal_crowns=(1, 0))
    return AbilityPilotTask(
        selection_index=0,
        selection_digest="0" * 64,
        battle_tag=plan.battle_tag,
        source_path=str(path.resolve()),
        source_sha256=sha256_file(path),
        source_schema_version=3,
        team_crowns=1,
        opponent_crowns=0,
        deploy_action_count=len(plan.actions),
        ability_event_count=len(plan.ability_events),
        duration_ticks=plan.duration_ticks,
    )


class ExpertNativeAbilityPilotTests(unittest.TestCase):
    def test_selection_is_order_independent_and_exact_ability_positive(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            rows = []
            for index in range(4):
                source = ability_battle(f"ABILITY-SELECT-{index}")
                if index == 0:
                    source["ability_plays"] = []
                    source["elixir_stats"]["team"]["Ability"]["count"] = 0
                if index == 1:
                    source["schema_version"] = 2
                path = root / f"source-{index}.json"
                path.write_text(json.dumps(source), encoding="utf-8")
                rows.append({
                    "battle_tag": source["battle_tag"],
                    "source_path": str(path),
                    "schema_version": source["schema_version"],
                    "team_crowns": 1,
                    "opponent_crowns": 0,
                })
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            first.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            second.write_text(
                "\n".join(json.dumps(row) for row in reversed(rows)) + "\n"
            )
            selected_a, summary_a = select_ability_positive_tasks(first, limit=2)
            selected_b, summary_b = select_ability_positive_tasks(second, limit=2)
            self.assertEqual(
                [task.battle_tag for task in selected_a],
                [task.battle_tag for task in selected_b],
            )
            self.assertTrue(all(task.ability_event_count > 0 for task in selected_a))
            self.assertEqual(summary_a["selected_battles"], 2)
            self.assertEqual(summary_b["selected_ability_events"], 2)

    def test_unique_ability_writes_a_consecutive_tick_store(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            source_path = root / "source.json"
            source_path.write_text(json.dumps(ability_battle()), encoding="utf-8")
            task = make_task(source_path)
            sink = WorkerShardSink(root / "store", "worker", episodes_per_shard=10)
            record, diagnostic = execute_ability_task(
                FakeTraceNativeEnv(), task, template(), calibration(), sink,
                seed=424242, trace_batch_steps=16,
            )
            manifests = sink.finalize()
            self.assertTrue(record["teacher_forced_success"], record["failure"])
            self.assertIsNone(diagnostic)
            self.assertEqual(record["accepted_ability_actions"], 1)
            self.assertEqual(record["ability_resolution_counts"], {"unique": 1})
            self.assertTrue(record["tick_store_integrity"])
            self.assertEqual(len(manifests), 1)
            manifest = manifests[0]
            with ShardReader(
                root / "store" / manifest["data_file"],
                root / "store" / manifest["index_file"],
            ) as reader:
                states = list(reader.episode(task.battle_tag).iter_ticks())
            self.assertEqual(
                [state.tick for state in states],
                list(range(states[0].tick, states[-1].tick + 1)),
            )

    def test_branch_required_preserves_candidates_and_native_state(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            source_path = root / "source.json"
            source_path.write_text(json.dumps(ability_battle()), encoding="utf-8")
            sink = WorkerShardSink(root / "store", "worker")
            record, diagnostic = execute_ability_task(
                FakeTraceNativeEnv(branch=True), make_task(source_path),
                template(), calibration(), sink, seed=424242,
            )
            sink.finalize()
            self.assertFalse(record["teacher_forced_success"])
            self.assertEqual(record["failure_class"], "ability_branch_required")
            self.assertEqual(
                record["ability_resolutions"][0]["candidate_entity_ids"], (50, 51)
            )
            assert diagnostic is not None
            snapshot = diagnostic["native_boundary_snapshot"]["latest_state"]
            self.assertEqual(snapshot["tick"], 25)
            self.assertEqual(len(snapshot["entities"]), 2)

    def test_native_rejection_preserves_request_response_and_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            source_path = root / "source.json"
            source_path.write_text(json.dumps(ability_battle()), encoding="utf-8")
            sink = WorkerShardSink(root / "store", "worker")
            record, diagnostic = execute_ability_task(
                FakeTraceNativeEnv(reject_ability=True), make_task(source_path),
                template(), calibration(), sink, seed=424242,
            )
            sink.finalize()
            self.assertFalse(record["teacher_forced_success"])
            self.assertEqual(record["failure_class"], "native_action_rejected")
            assert diagnostic is not None
            last = diagnostic["native_boundary_snapshot"]["recent_action_history"][-1]
            self.assertEqual(last["request"][0]["type"], "ability")
            self.assertEqual(last["pre_action_state"]["tick"], 25)
            self.assertEqual(
                last["response"]["actions"][0]["result"]["result_code"], 17
            )


if __name__ == "__main__":
    unittest.main()
