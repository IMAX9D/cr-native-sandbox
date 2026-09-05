import unittest
from unittest.mock import patch

import torch

from policy_v1.timing import StageTimer, timed_batches


class TimingTests(unittest.TestCase):
    def test_data_wait_excludes_consumer_work(self):
        now = [0.0]

        def loader():
            now[0] += 3.0  # Iterator startup, first batch only.
            for i in range(2):
                now[0] += 0.25
                yield i

        with patch("policy_v1.timing.time.perf_counter", side_effect=lambda: now[0]):
            batches = iter(timed_batches(loader()))
            self.assertEqual(next(batches), (0, 3.25))
            now[0] += 10.0  # Model computation, not data wait.
            self.assertEqual(next(batches), (1, 0.25))

    def test_warmup_sync_and_measurements(self):
        timer = StageTimer(torch.device("cuda"), enabled=True, warmup=1)
        now = [0.0]
        with patch(
            "policy_v1.timing.time.perf_counter", side_effect=lambda: now[0]
        ), patch("policy_v1.timing.torch.cuda.synchronize") as sync:
            timer.begin(0.3, 4)
            with timer.stage("forward"):
                now[0] += 0.1
            self.assertIsNone(timer.report())
            timer.begin(0.3, 4)
            with timer.stage("forward"):
                now[0] += 0.1
            report = timer.report()
            self.assertEqual(sync.call_count, 4)
            self.assertEqual(report["timed_batches"], 1)
            self.assertAlmostEqual(report["stage_ms_per_batch"]["forward"], 100.0)
            self.assertAlmostEqual(report["data_wait_percent"], 75.0)
            self.assertAlmostEqual(report["windows_per_timed_second"], 10.0)
            timer.clear()
            self.assertIsNone(timer.report())

    def test_disabled_timer_never_synchronizes(self):
        timer = StageTimer(torch.device("cuda"))
        with patch("policy_v1.timing.torch.cuda.synchronize") as sync:
            timer.begin(0.1, 2)
            with timer.stage("forward"):
                pass
            sync.assert_not_called()
            self.assertIsNone(timer.report())


if __name__ == "__main__":
    unittest.main()
