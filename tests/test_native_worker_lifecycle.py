from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from native_core.worker import HeadlessWorkerPool, WorkerConfig


class NativeWorkerLifecycleTests(unittest.TestCase):
    def test_stop_vm_is_idempotent_when_vm_is_already_down(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pool = HeadlessWorkerPool(
                WorkerConfig(data_root=Path(temporary), service_base_port=37031)
            )
            with (
                mock.patch.object(pool, "vm_ready", return_value=False),
                mock.patch.object(pool, "service_ready", return_value=False),
                mock.patch("native_core.worker.request", side_effect=OSError("down")),
                mock.patch("native_core.worker.subprocess.run"),
            ):
                result = pool.stop(1, keep_vm=False)
            self.assertTrue(result["vm_stopped"])
            self.assertEqual(
                result["services"],
                [{"slot": 0, "port": 37031, "stopped": True}],
            )


if __name__ == "__main__":
    unittest.main()
