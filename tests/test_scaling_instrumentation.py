from __future__ import annotations

import unittest

from native_core.worker import MultiAvdWorkerPool
from training.vector_rollout import summarize_barrier


class ScalingInstrumentationTests(unittest.TestCase):
    def test_multi_avd_ports_are_isolated_and_direct_ports_are_contiguous(self):
        pool = MultiAvdWorkerPool(avds=3, workers_per_avd=4)
        self.assertEqual(pool.workers, 12)
        self.assertEqual(pool.environment_ports("direct"), list(range(38031, 38043)))
        self.assertEqual(pool.environment_ports("adb"), [
            37031, 37032, 37033, 37034,
            37131, 37132, 37133, 37134,
            37231, 37232, 37233, 37234,
        ])
        self.assertEqual(
            [item.config.serial for item in pool.pools],
            ["emulator-5554", "emulator-5556", "emulator-5558"],
        )

    def test_barrier_summary_preserves_fastest_median_slowest(self):
        rows = [
            (0, 4, 8, 0.001, 0.002, 0.004),
            (1, 3, 6, 0.002, 0.003, 0.008),
        ]
        summary = summarize_barrier(rows)
        self.assertEqual(summary["vector_round_count"], 2.0)
        self.assertEqual(summary["policy_batch_size_max"], 8.0)
        self.assertAlmostEqual(
            summary["worker_transition_fastest_mean_ms"], 1.5
        )
        self.assertAlmostEqual(
            summary["worker_transition_median_mean_ms"], 2.5
        )
        self.assertAlmostEqual(
            summary["worker_transition_slowest_mean_ms"], 6.0
        )
        self.assertAlmostEqual(
            summary["worker_transition_barrier_wait_mean_ms"], 4.5
        )

    def test_dense_two_avd_twenty_worker_layout(self):
        pool = MultiAvdWorkerPool(
            avds=2,
            workers_per_avd=10,
            cores_per_avd=10,
            memory_mb_per_avd=7168,
        )
        self.assertEqual(pool.workers, 20)
        self.assertEqual(
            pool.environment_ports("direct"), list(range(38031, 38051))
        )
        self.assertEqual(
            pool.environment_ports("adb"),
            [*range(37031, 37041), *range(37131, 37141)],
        )
        self.assertTrue(all(item.config.cores == 10 for item in pool.pools))
        self.assertTrue(all(item.config.memory_mb == 7168 for item in pool.pools))


if __name__ == "__main__":
    unittest.main()
