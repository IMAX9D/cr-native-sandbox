import unittest
from unittest.mock import patch

from scripts.preflight_cloud_native_runtime import _cgroup_limit


class CloudRuntimePreflightTests(unittest.TestCase):
    def test_cgroup_v2_max_is_unlimited(self):
        with patch(
            "scripts.preflight_cloud_native_runtime._read",
            side_effect=lambda path: "max\n" if path.endswith("memory.max") else "",
        ):
            self.assertIsNone(_cgroup_limit())

    def test_cgroup_numeric_limit_is_parsed(self):
        with patch(
            "scripts.preflight_cloud_native_runtime._read",
            side_effect=lambda path: "68719476736\n" if path.endswith("memory.max") else "",
        ):
            self.assertEqual(_cgroup_limit(), 64 * 1024**3)


if __name__ == "__main__":
    unittest.main()
