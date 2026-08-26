from __future__ import annotations

from copy import deepcopy
import unittest

from expert_v1.tick_store_v1 import TickTraceAccumulator
from expert_v1.native_replay_runner import _advance_native
from native_core.client import MAX_TRACE_RESPONSE_BYTES
from native_core.env import NativeHostError, NativeRoyaleEnv


def compact_state(tick: int, *, elixir_raw: int = 50_000) -> dict:
    return {
        "schema_version": 1,
        "kind": "libg_native_train_state_v1",
        "tick": tick,
        "applied_replay_tick": -1,
        "entities": [
            {
                "category": 5_000_100,
                "side": 0,
                "x": 9_000,
                "y": 5_000 + tick,
                "card_id": 26_000_000,
                "level": 15,
                "hp": 2_000,
                "max_hp": 2_000,
                "behavior_state": 1,
                "ability_slot": 0,
                "ability_state_code": -1,
                "ability_available": False,
                "ability_cooldown_remaining_ms": -1,
                "ability_charges_remaining": -1,
                "ability_pending_ms": -1,
                "ability_mana_cost": -1,
            }
        ],
        "players": [
            {
                "side": 0,
                "elixir": elixir_raw // 10_000,
                "elixir_raw": elixir_raw,
                "refill_timer": 0,
                "next_deck_index": 4,
                "hand_deck_indices": [0, 1, 2, 3],
            },
            {
                "side": 1,
                "elixir": 5,
                "elixir_raw": 50_000,
                "refill_timer": 0,
                "next_deck_index": 0,
                "hand_deck_indices": [4, 5, 6, 7],
            },
        ],
        "entity_count": 1,
        "coherent": True,
        "tick_after": tick,
        "episode": {
            "terminated": False,
            "truncated": False,
            "outcome": "ongoing",
            "winner": None,
            "crowns": [0, 0],
            "rewards": [0.0, 0.0],
            "commands_allowed": True,
            "command_gate_code": 0,
            "native_phase": {
                "battle": 1,
                "logic": 0,
                "logic_substate": 0,
                "flag_1e9": 0,
            },
            "terminal_tick": tick,
            "crown_towers": [
                {
                    "side": side,
                    "type": "princess",
                    "lane": lane,
                    "x": x,
                    "y": y,
                    "hp": 3052,
                    "max_hp": 3052,
                }
                for side, lane, x, y in (
                    (0, "left", 3500, 6500),
                    (0, "right", 14500, 6500),
                    (1, "left", 3500, 25500),
                    (1, "right", 14500, 25500),
                )
            ],
        },
    }


def trace(start: int, count: int, *, initial_elixir: int = 50_000) -> dict:
    return {
        "schema_version": 1,
        "kind": "libg_native_train_tick_trace_v1",
        "trace_schema_version": 1,
        "encoding": "compact-train-v1",
        "fixed_dt": 0.05,
        "initial_tick": start,
        "requested_steps": count,
        "max_response_bytes": MAX_TRACE_RESPONSE_BYTES,
        "initial_frame": {
            "frame_index": 0,
            "advanced_steps": 0,
            "observation_complete": True,
            "state": compact_state(start, elixir_raw=initial_elixir),
        },
        "frames": [
            {
                "frame_index": index,
                "advanced_steps": index,
                "observation_complete": True,
                "state": compact_state(start + index),
            }
            for index in range(1, count + 1)
        ],
        "stepped": count,
        "terminal": False,
        "final_frame_index": count,
        "final_tick": start + count,
    }


class FakeEnv(NativeRoyaleEnv):
    def __init__(self, response: dict) -> None:
        self.response = response
        self.last_payload: dict | None = None
        self.last_episode = None

    def _request(self, payload: dict) -> dict:
        self.last_payload = deepcopy(payload)
        return {"result": deepcopy(self.response)}


class CompactTickTraceTest(unittest.TestCase):
    def test_one_rpc_returns_raw_consecutive_compact_frames(self) -> None:
        env = FakeEnv(trace(100, 3))
        result = env.trace_train(3)
        self.assertEqual(env.last_payload["op"], "step_train_trace_v1")
        self.assertEqual([frame["state"]["tick"] for frame in result["frames"]], [101, 102, 103])
        self.assertNotIn("step", result["frames"][0])
        self.assertNotIn("elapsed_seconds", result["frames"][0]["state"])

    def test_accumulator_ignores_mutated_duplicate_boundary(self) -> None:
        accumulator = TickTraceAccumulator()
        self.assertEqual(accumulator.extend(trace(100, 2)), 2)
        self.assertEqual(
            accumulator.extend(trace(102, 2, initial_elixir=20_000)), 2
        )
        self.assertEqual([state.tick for state in accumulator.states], [100, 101, 102, 103, 104])
        self.assertEqual(accumulator.batches, 2)
        self.assertEqual(accumulator.states[2].players[0].elixir_raw, 50_000)

    def test_nonconsecutive_frame_fails_closed(self) -> None:
        value = trace(100, 2)
        value["frames"][1]["state"]["tick"] = 104
        value["final_tick"] = 104
        env = FakeEnv(value)
        with self.assertRaisesRegex(NativeHostError, "not consecutive"):
            env.trace_train(2)

    def test_terminal_no_progress_frame_is_metadata_not_duplicate_tick(self) -> None:
        value = trace(100, 1)
        value["frames"][0] = {
            "frame_index": 1,
            "advanced_steps": 1,
            "observation_complete": False,
            "state": {"tick": 100, "episode": {"terminated": True}},
        }
        value["terminal"] = True
        value["final_tick"] = 100
        env = FakeEnv(value)
        decoded = env.trace_train(1)
        accumulator = TickTraceAccumulator()
        self.assertEqual(accumulator.extend(decoded), 0)
        self.assertEqual([state.tick for state in accumulator.states], [100])
        self.assertEqual(accumulator.incomplete_terminal_frames, 1)

    def test_step_bound_is_enforced_before_rpc(self) -> None:
        env = FakeEnv(trace(100, 1))
        with self.assertRaises(ValueError):
            env.trace_train(65)
        self.assertIsNone(env.last_payload)

    def test_runner_chunks_long_gap_without_per_tick_rpc(self) -> None:
        class SequenceEnv:
            def __init__(self) -> None:
                self.tick = 100
                self.calls: list[int] = []

            def trace_train(self, steps: int) -> dict:
                self.calls.append(steps)
                value = trace(self.tick, steps)
                self.tick += steps
                return value

        env = SequenceEnv()
        accumulator = TickTraceAccumulator()
        accumulator.start(compact_state(100))
        result = _advance_native(  # type: ignore[arg-type]
            env, 130, accumulator, trace_batch_steps=64
        )
        self.assertEqual(env.calls, [64, 64, 2])
        self.assertEqual(result["tick_after"], 230)
        self.assertEqual(len(accumulator.states), 131)


if __name__ == "__main__":
    unittest.main()
