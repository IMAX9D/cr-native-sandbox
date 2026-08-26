from __future__ import annotations

import unittest

from scripts.compare_expert_action_phase_ab import category, failure_codes


class CompareExpertActionPhaseAbTest(unittest.TestCase):
    def test_rejection_codes_prefer_structured_evidence(self) -> None:
        row = {
            "failure": "renamed-diagnostic",
            "first_rejection": {"result_codes": [13]},
        }
        self.assertEqual(failure_codes(row), [13])
        self.assertEqual(category(row), "code13")

    def test_both_terminal_before_names_are_supported(self) -> None:
        self.assertEqual(category({
            "failure": "native_terminal_before_source_tick_100",
        }), "terminal_before")
        self.assertEqual(category({
            "failure": (
                "native_terminal_before_execution_tick_101_source_tick_100"
            ),
        }), "terminal_before")

    def test_success_dominates_diagnostic_text(self) -> None:
        self.assertEqual(category({
            "teacher_forced_success": True,
            "failure": "native_rejected_tick_1_codes_[13]",
        }), "success")


if __name__ == "__main__":
    unittest.main()
