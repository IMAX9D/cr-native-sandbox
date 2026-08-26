import unittest

from expert_v1.elixir_tick_diagnostics import (
    code13_cases,
    elixir_multiplier,
    lower_bound_regen_ticks,
    packed_card_cost_raw,
)


class ElixirTickDiagnosticsTest(unittest.TestCase):
    def test_packed_card_cost_uses_native_command_nibble(self) -> None:
        self.assertEqual(
            packed_card_cost_raw({"packed_selection": 0x50000002}), 50_000
        )
        self.assertEqual(packed_card_cost_raw({
            "packed_selection": 0x50000002,
            "resource_before": {"card_cost": 4},
        }), 40_000)


    def test_code13_uses_runtime_cost_not_catalog_guess(self) -> None:
        rows = [{
        "battle_tag": "B",
        "source_path": "source.json",
        "first_rejection": {
            "tick": 674,
            "events": [{
                "source_event_index": 6,
                "side": 1,
                "base_token": "balloon",
                "pre_action_elixir_raw": 49_972,
                "pre_action_hand_deck_indices": [6, 3, 5, 4],
                "pre_action_next_deck_index": 0,
                "pre_action_refill_timer": 0,
                "native_result": {
                    "result_code": 13,
                    "packed_selection": 0x50000002,
                    "resolved_data_id": 203000006,
                },
            }],
        },
        }]
        case = code13_cases(rows)[0]
        self.assertEqual(case.cost_raw, 50_000)
        self.assertEqual(case.deficit_raw, 28)


    def test_timeline_multiplier_and_regen_lower_bound(self) -> None:
        self.assertEqual(
            [elixir_multiplier(tick) for tick in (0, 2399, 2400, 4799, 4800)],
            [1, 1, 2, 2, 3],
        )
        self.assertEqual(lower_bound_regen_ticks(28, 674), 1)
        self.assertEqual(lower_bound_regen_ticks(305, 2435), 1)
        self.assertEqual(lower_bound_regen_ticks(10_400, 941), 59)


if __name__ == "__main__":
    unittest.main()
