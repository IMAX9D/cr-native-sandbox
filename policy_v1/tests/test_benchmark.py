from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

from policy_v1.benchmark import parser, run
from policy_v1.data import prepare
from policy_v1.smoke import create_fixture


class BenchmarkTests(unittest.TestCase):
    def test_bounded_measurement_without_artifact_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = create_fixture(root / "data")
            cache = root / "cache"
            prepare(data, cache, allow_smoke=True)
            before = {
                p.relative_to(root): p.read_bytes()
                for p in root.rglob("*")
                if p.is_file()
            }
            args = parser().parse_args(
                [
                    "--data",
                    str(data),
                    "--cache",
                    str(cache),
                    "--allow-smoke",
                    "--device",
                    "cpu",
                    "--width",
                    "16",
                    "--heads",
                    "4",
                    "--layers",
                    "1",
                    "--frame-window",
                    "8",
                    "--event-window",
                    "8",
                    "--targets",
                    "4",
                    "--batch-size",
                    "2",
                    "--workers",
                    "0",
                    "--cpu-threads",
                    "1",
                    "--warmup",
                    "1",
                    "--steps",
                    "2",
                ]
            )
            with redirect_stdout(io.StringIO()):
                result = run(args)
            self.assertEqual(result["timed_batches"], 2)
            self.assertEqual(result["timed_windows"], 4)
            self.assertEqual(result["successful_updates"], 2)
            self.assertEqual(result["skipped_overflows"], 0)
            self.assertGreater(result["stage_ms_per_batch"]["backward"], 0)
            after = {
                p.relative_to(root): p.read_bytes()
                for p in root.rglob("*")
                if p.is_file()
            }
            self.assertEqual(before, after)
