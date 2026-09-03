from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from native_core.worker import HeadlessWorkerPool, WorkerConfig, WorkerError


class NativeWorkerLifecycleTests(unittest.TestCase):
    def test_reused_avd_restores_root_without_starting_second_vm(self) -> None:
        pool = HeadlessWorkerPool()
        with (mock.patch.object(pool, "vm_ready", return_value=True),
              mock.patch.object(pool, "_ensure_adb_root") as root,
              mock.patch("native_core.worker.subprocess.Popen") as launch):
            result = pool.start_vm()
        root.assert_called_once_with()
        launch.assert_not_called()
        self.assertFalse(result["started"])

    def test_already_root_does_not_restart_adbd(self) -> None:
        pool = HeadlessWorkerPool()
        with mock.patch.object(pool, "_adb", return_value="0\n") as adb:
            pool._ensure_adb_root()
        adb.assert_called_once_with("shell", "id", "-u", timeout=3)

    def test_shell_identity_becomes_root(self) -> None:
        pool = HeadlessWorkerPool()
        with mock.patch.object(pool, "_adb", side_effect=["2000\n", "restarting adbd as root", "0\n"]) as adb:
            pool._ensure_adb_root()
        self.assertEqual(adb.call_args_list[1].args, ("root",))

    def test_production_device_fails_closed(self) -> None:
        pool = HeadlessWorkerPool()
        with mock.patch.object(pool, "_adb", side_effect=["2000\n", "adbd cannot run as root in production builds"]):
            with self.assertRaisesRegex(WorkerError, "root-capable"):
                pool._ensure_adb_root()

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
