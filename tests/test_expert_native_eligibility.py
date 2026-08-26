from __future__ import annotations

import unittest

from expert_v1.audit_native_eligibility import _failure_class, _wilson


class NativeEligibilityAuditTests(unittest.TestCase):
    def test_wilson_intervals_cover_observed_rate(self) -> None:
        for successes in (58, 89):
            low, high = _wilson(successes, 100)
            self.assertLess(low, successes / 100)
            self.assertGreater(high, successes / 100)
            self.assertGreaterEqual(low, 0.0)
            self.assertLessEqual(high, 1.0)

    def test_same_tick_failures_remain_distinct(self) -> None:
        self.assertEqual(
            _failure_class("multiple actions for side 1 at native tick 3011"),
            "same_tick_multiple_deployments",
        )
        self.assertEqual(
            _failure_class("multiple deploy/ability actions for side 0 at native tick 941"),
            "same_tick_deploy_ability_collision",
        )


if __name__ == "__main__":
    unittest.main()
