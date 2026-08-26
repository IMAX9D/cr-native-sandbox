from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from expert_v1.native_replay_plan import (
    ReplayPlanError,
    compile_battle,
    grouped_actions,
    materialize_replay,
    split_card_token,
)
from expert_v1.native_replay_runner import _compact_decision_state
from expert_v1.source_terminal_anchors import index_terminal_anchors


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def battle(*, ability_count: int = 0) -> dict:
    team = [
        "knight-ev1", "archers", "giant", "skeletons",
        "musketeer", "hog-rider", "cannon", "arrows",
    ]
    opponent = [
        "knight", "archers", "giant", "skeletons",
        "musketeer", "hog-rider", "cannon", "arrows",
    ]
    # Every prefix action is legal in at least one 4+4 initial state and the
    # second cycle removes any ambiguity relevant to future hand labels.
    events = []
    tick = 120
    for round_index in range(2):
        for index, token in enumerate(team):
            for side, deck in (("team", team), ("opponent", opponent)):
                events.append({
                    "time_raw": tick,
                    "side": side,
                    "card": deck[index].removesuffix("-ev1"),
                    "x": 8_500,
                    "y": 10_500 if side == "team" else 21_500,
                })
            tick += 20
    levels0 = {token: 11 for token in team}
    levels1 = {token: 11 for token in opponent}
    return {
        "schema_version": 2,
        "battle_tag": "TEST00000001",
        "duration_seconds": 60,
        "draft": False,
        "card_plays": events,
        "elixir_stats": {
            "team": {"Ability": {"count": ability_count}},
            "opponent": {"Ability": {"count": 0}},
        },
        "rounds": [{
            "team": [{
                "full_deck": team, "card_levels": levels0,
                "tower_troop": "tower-princess", "complete": True,
            }],
            "opponent": [{
                "full_deck": opponent, "card_levels": levels1,
                "tower_troop": "tower-princess", "complete": True,
            }],
        }],
    }


class ExpertNativeReplayPlanTest(unittest.TestCase):
    def test_form_suffixes(self) -> None:
        self.assertEqual(split_card_token("Knight-EV1"), ("knight", 1))
        self.assertEqual(split_card_token("Knight-Hero"), ("knight", 2))
        self.assertEqual(split_card_token("Knight-hero-ev1"), ("knight", 3))

    def test_compile_and_materialize_preserves_source_form_slots(self) -> None:
        plan = compile_battle(battle())
        self.assertTrue(plan.native_replay_ready)
        self.assertFalse(plan.original_state_exact)
        self.assertEqual(plan.sides[0].cycle.first_exact_action_index, 4)
        self.assertEqual(plan.sides[1].cycle.first_exact_action_index, 4)
        template = json.loads(
            (PROJECT_ROOT / "examples" / "eight-card-bootstrap.json").read_text(
                encoding="utf-8"
            )
        )
        players = [
            {
                "hand_deck_indices": [5, 0, 6, 2],
                "cycle_deck_indices": [1, 4, 3, 7],
            },
            {
                "hand_deck_indices": [4, 2, 7, 0],
                "cycle_deck_indices": [5, 6, 1, 3],
            },
        ]
        replay, mappings = materialize_replay(plan, template, players)
        self.assertEqual(mappings, (tuple(range(8)), tuple(range(8))))
        self.assertEqual(len(replay["battle"]["deck0"]["sp"]), 8)
        batches = list(grouped_actions(plan, mappings))
        self.assertEqual(len(batches), 16)
        self.assertTrue(all(len(actions) == 2 for _, actions in batches))
        self.assertEqual(
            replay["battle"]["deck0"]["sp"][0]["el"], 1
        )
        self.assertEqual(
            [row["d"] for row in replay["battle"]["deck0"]["sp"]],
            [card.card_id for card in plan.sides[0].deck],
        )

    def test_missing_ability_is_fail_closed_from_native_candidate(self) -> None:
        plan = compile_battle(battle(ability_count=2))
        self.assertFalse(plan.native_replay_ready)
        self.assertEqual(plan.replay_tier, "action_sequence_only")
        self.assertIn("ability_button_events_missing", plan.limitations)

    def test_terminal_crowns_are_explicit_source_provenance(self) -> None:
        plan = compile_battle(battle(), terminal_crowns=(2, 1))
        self.assertEqual(plan.terminal_crowns, (2, 1))
        self.assertEqual(plan.terminal_provenance, "source_index_crowns")

    def test_same_side_same_tick_is_rejected(self) -> None:
        value = battle()
        duplicate = dict(value["card_plays"][0])
        value["card_plays"].insert(1, duplicate)
        with self.assertRaisesRegex(ReplayPlanError, "multiple actions"):
            compile_battle(value)

    def test_actor_projection_drops_opponent_hidden_information(self) -> None:
        state = {
            "tick": 120,
            "state_hash": "audit-only",
            "players": [
                {
                    "side": 0, "elixir": 7, "elixir_raw": 7000,
                    "hand_deck_indices": [0, 1, 2, 3],
                    "next_deck_index": 4, "refill_timer": 0,
                },
                {
                    "side": 1, "elixir": 9, "elixir_raw": 9000,
                    "hand_deck_indices": [4, 5, 6, 7],
                    "next_deck_index": 0, "refill_timer": 0,
                },
            ],
            "entities": [],
            "episode": {"crown_towers": []},
        }
        record = _compact_decision_state(
            state, actor_side=0, source_tick=120, wait_ticks=10,
            expert_action=None,
        )
        self.assertEqual(record["own_player"]["elixir"], 7)
        encoded = json.dumps(record).lower()
        self.assertNotIn("9000", encoded)
        self.assertNotIn("[4, 5, 6, 7]", encoded)
        self.assertNotIn("opponent_hand", encoded)

    def test_terminal_anchor_is_recovered_from_source_index_url(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.jsonl"
            path.write_text(json.dumps({
                "kind": "battle",
                "url": (
                    "https://royaleapi.com/data/replay?tag=TAG123"
                    "&team_crowns=3&opponent_crowns=1"
                ),
            }) + "\n", encoding="utf-8")
            self.assertEqual(index_terminal_anchors(path), {"TAG123": (3, 1)})


if __name__ == "__main__":
    unittest.main()
