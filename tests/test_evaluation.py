from __future__ import annotations

import unittest

from training.evaluate import summarize_evaluation, wilson_interval


class EvaluationTests(unittest.TestCase):
    def test_side_swapped_pairs_and_integrity_are_reported(self):
        records = [
            {
                "seed": 10,
                "outcome": "win",
                "score": 1.0,
                "terminated": True,
                "native_action_rejections": 0,
                "crown_difference": 1,
                "tower_hp_difference": 100,
                "match_ticks": 4000,
            },
            {
                "seed": 10,
                "outcome": "draw",
                "score": 0.5,
                "terminated": True,
                "native_action_rejections": 0,
                "crown_difference": 0,
                "tower_hp_difference": 0,
                "match_ticks": 6000,
            },
        ]
        summary = summarize_evaluation(records)
        self.assertEqual(summary["games"], 2)
        self.assertEqual(summary["paired_seed_count"], 1)
        self.assertEqual(summary["score_rate"], 0.75)
        self.assertEqual(summary["average_match_seconds"], 250.0)
        self.assertTrue(summary["passed_integrity"])

    def test_missing_swap_or_rejection_fails_integrity(self):
        summary = summarize_evaluation([{
            "seed": 10,
            "outcome": "loss",
            "score": 0.0,
            "terminated": True,
            "native_action_rejections": 1,
            "crown_difference": -1,
            "tower_hp_difference": -100,
            "match_ticks": 3000,
        }])
        self.assertFalse(summary["passed_integrity"])
        self.assertEqual(summary["incomplete_paired_seeds"], [10])

    def test_wilson_interval_contains_observed_rate(self):
        low, high = wilson_interval(7, 10)
        self.assertLess(low, 0.7)
        self.assertGreater(high, 0.7)


if __name__ == "__main__":
    unittest.main()
