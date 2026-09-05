from __future__ import annotations

from pathlib import PurePosixPath
import unittest

from scripts.ensure_bionic_workers import (
    BOOT_JARS,
    command,
    environment,
    port_bindable,
)


class EnsureBionicWorkersTests(unittest.TestCase):
    def test_ephemeral_port_is_bindable(self) -> None:
        import socket

        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        self.assertTrue(port_bindable(port))

    def test_command_matches_headless_android_runtime_contract(self) -> None:
        direct = PurePosixPath("/data/local/tmp/cr-native-direct-7")
        value = command(direct, 39038)
        self.assertEqual(value[0], "/apex/com.android.runtime/bin/linker64")
        self.assertEqual(value[1], "/apex/com.android.art/bin/dalvikvm64")
        self.assertIn("-Xint", value)
        self.assertIn(f"-Xbootclasspath:{':'.join(BOOT_JARS)}", value)
        self.assertEqual(value[-3:], [str(direct), "serve-direct", "39038"])
        self.assertTrue(any(
            str(direct / "lifecycle-probe.jar") in item for item in value
        ))
        jit = command(direct, 39038, execution_mode="jit")
        self.assertNotIn("-Xint", jit)
        self.assertEqual(jit[-3:], [str(direct), "serve-direct", "39038"])

    def test_environment_is_binderless_and_worker_isolated(self) -> None:
        direct = PurePosixPath("/data/local/tmp/cr-native-direct-3")
        value = environment(direct)
        self.assertEqual(value["CR_BINDERLESS_ANDROID"], "1")
        self.assertEqual(value["CR_BINDERLESS_NATIVE_CONFIG_GETTERS"], "1")
        self.assertEqual(value["CR_BINDERLESS_NATIVE_CONFIG_POSTPROCESS"], "1")
        self.assertEqual(value["CR_NATIVE_LOADING_TIMEOUT_MS"], "30000")
        self.assertEqual(value["LD_LIBRARY_PATH"], str(direct))
        self.assertEqual(value["TMPDIR"], str(direct / "cache"))
        self.assertEqual(value["ANDROID_ROOT"], "/system")


if __name__ == "__main__":
    unittest.main()
