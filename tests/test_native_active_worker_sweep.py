from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import patch

from training.schema import ActionMaskCache, CARD_COSTS

from scripts.run_native_active_worker_sweep import (
    atomic_json,
    deterministic_legal_actions,
    load_ports,
    parse_integer_set,
    recommended_tier,
    run_phase,
    summarize_tier,
    validate_transition,
    workload_name,
)


class _Slot:
    def __init__(self, tick: int) -> None:
        self.state = {"tick": tick}


class _MaskEnv:
    def __init__(self) -> None:
        card_ids = list(CARD_COSTS)[:8]
        self.decks = [
            [{"card_id": card_id} for card_id in card_ids],
            [{"card_id": card_id} for card_id in card_ids],
        ]

    def probe_grid(self, *, side: int, deck_index: int) -> dict:
        del side, deck_index
        return {"rows": ["1" * 18 for _ in range(32)]}


class _SummaryResources:
    def summary(self) -> dict:
        return {"sample_count": 1}


class NativeActiveWorkerSweepTests(unittest.TestCase):
    def test_port_sources_and_ranges_are_explicit(self) -> None:
        self.assertEqual(parse_integer_set("38031-38033,38040", label="port"), [38031, 38032, 38033, 38040])
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "ports.json"
            path.write_text('{"ports":[38031,38032]}', encoding="utf-8")
            self.assertEqual(load_ports(None, path), [38031, 38032])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_integer_set("1,1", label="port")

    def test_transition_requires_real_tick_progress(self) -> None:
        slot = _Slot(100)
        transition = {
            "step": {"episode": {"terminated": False, "truncated": False}},
            "state": {"tick": 101},
        }
        self.assertEqual(validate_transition(slot, transition, requested_steps=1), (1, False))
        transition["state"]["tick"] = 100
        with self.assertRaisesRegex(RuntimeError, "freeze"):
            validate_transition(slot, transition, requested_steps=1)

    def test_deterministic_mode_selects_one_legal_action_per_side(self) -> None:
        slot = SimpleNamespace(
            env=_MaskEnv(),
            state={
                "tick": 100,
                "episode": {"commands_allowed": True},
                "entities": [
                    {
                        "category": 5_000_000 + side * 3 + lane,
                        "card_id": -1,
                        "side": side,
                        "x": x,
                        "y": 6_500 if side == 0 else 25_500,
                        "hp": 3_052,
                    }
                    for side in (0, 1)
                    for lane, x in ((1, 3_500), (2, 14_500))
                ],
                "players": [
                    {"side": 0, "elixir": 10, "hand_deck_indices": [0, 1, 2, 3]},
                    {"side": 1, "elixir": 10, "hand_deck_indices": [0, 1, 2, 3]},
                ],
            },
            masks={},
            mask_cache=ActionMaskCache(),
        )
        actions = deterministic_legal_actions(slot)
        self.assertEqual([action["side"] for action in actions], [0, 1])
        self.assertEqual(actions[0]["y"], 500)
        self.assertEqual(actions[1]["y"], 15_500)

    def test_terminal_contract_and_atomic_report(self) -> None:
        slot = _Slot(7199)
        transition = {"step": {"episode": {
            "terminated": True, "truncated": False, "terminal_tick": 7200,
            "outcome": "side_0_win", "crowns": [1, 0], "rewards": [1, -1],
        }}}
        self.assertEqual(validate_transition(slot, transition, requested_steps=1), (1, True))
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "result.json"
            atomic_json(path, {"status": "complete"})
            self.assertIn('"complete"', path.read_text(encoding="utf-8"))

    def test_late_terminal_metadata_may_repeat_the_last_tick(self) -> None:
        slot = _Slot(6147)
        transition = {"step": {"episode": {
            "terminated": True, "truncated": False, "terminal_tick": 6147,
            "outcome": "side_1_win", "crowns": [0, 1], "rewards": [-1, 1],
        }}}
        self.assertEqual(
            validate_transition(slot, transition, requested_steps=16),
            (0, True),
        )

    def test_recommendation_stops_below_five_percent_gain(self) -> None:
        rows = [
            {"status": "complete", "workers": 32, "native_ticks_per_second": 1000.0},
            {"status": "complete", "workers": 48, "native_ticks_per_second": 1100.0},
            {"status": "complete", "workers": 64, "native_ticks_per_second": 1140.0},
            {"status": "complete", "workers": 96, "native_ticks_per_second": 1400.0},
        ]
        self.assertEqual(recommended_tier(rows), 48)

    def test_workload_names_distinguish_action_semantics(self) -> None:
        self.assertEqual(
            workload_name("deterministic-legal"),
            "deterministic_legal_native_actions_v1",
        )
        self.assertEqual(workload_name("wait"), "wait_native_tick_v1")

    def test_run_phase_does_not_barrier_fast_workers(self) -> None:
        slots = [SimpleNamespace(name="fast", calls=0), SimpleNamespace(name="slow", calls=0)]

        def transition(slot, *, mode, decision_ticks):
            del mode, decision_ticks
            slot.calls += 1
            time.sleep(0.001 if slot.name == "fast" else 0.04)
            return 0.001, False

        with patch(
            "scripts.run_native_active_worker_sweep.one_transition",
            side_effect=transition,
        ):
            run_phase(
                slots,
                seconds=0.09,
                mode="wait",
                decision_ticks=1,
                output_interval=1.0,
            )
        self.assertGreater(slots[0].calls, slots[1].calls + 10)

    def test_summary_reports_worker_tick_rate_distribution(self) -> None:
        slots = []
        for ticks in (10, 20, 30):
            slots.append(SimpleNamespace(
                advanced_ticks=ticks,
                actions_attempted=0,
                actions_accepted=0,
                actions_rejected=0,
                unexpected_rejections=0,
                result_codes={},
                native_timing_ns={},
                transitions=ticks,
                terminals=0,
                resets=0,
                env=SimpleNamespace(
                    rpc_latency_samples=lambda: {"total": [], "receive": []},
                    rpc_profile={},
                ),
            ))
        summary = summarize_tier(
            slots, [], elapsed=10.0,
            resources=_SummaryResources(), reasons=[],
        )
        self.assertEqual(summary["worker_ticks_per_second_mean"], 2.0)
        self.assertEqual(summary["worker_ticks_per_second_p50"], 2.0)
        self.assertAlmostEqual(summary["worker_ticks_per_second_p05"], 1.1)
        self.assertAlmostEqual(summary["worker_ticks_per_second_p95"], 2.9)


if __name__ == "__main__":
    unittest.main()
