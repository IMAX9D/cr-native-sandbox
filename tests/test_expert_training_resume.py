from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from expert_v1.training_v1 import train as training
from expert_v1.training_v1.schema import (
    DatasetContractError,
    sha256_file,
    verify_dataset_integrity,
)
from expert_v1.training_v1.smoke_data import create_smoke_dataset


class ExpertTrainingIntegrityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = create_smoke_dataset(Path(self.temporary.name) / "dataset")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_full_dataset_integrity_passes(self) -> None:
        manifest, summary = verify_dataset_integrity(self.root, workers=2)
        self.assertEqual(summary["manifest_sha256"], sha256_file(self.root / "manifest.json"))
        self.assertEqual(summary["shard_files"], len(manifest["shard_file_sha256"]))

    def test_manifest_sidecar_mismatch_fails_closed(self) -> None:
        with (self.root / "manifest.json").open("a", encoding="utf-8") as handle:
            handle.write(" ")
        with self.assertRaisesRegex(DatasetContractError, "manifest checksum mismatch"):
            verify_dataset_integrity(self.root, workers=1)

    def test_shard_byte_mismatch_fails_closed(self) -> None:
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        relative = next(iter(manifest["shard_file_sha256"]))
        path = self.root / relative
        with path.open("r+b") as handle:
            handle.seek(-1, 2)
            value = handle.read(1)
            handle.seek(-1, 2)
            handle.write(bytes([value[0] ^ 0x01]))
        with self.assertRaisesRegex(DatasetContractError, "shard checksum mismatch"):
            verify_dataset_integrity(self.root, workers=2)

    def test_incomplete_hash_coverage_fails_closed(self) -> None:
        manifest_path = self.root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["shard_file_sha256"].pop(next(iter(manifest["shard_file_sha256"])))
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        (self.root / "manifest.sha256").write_text(
            f"{sha256_file(manifest_path)}  manifest.json\n", encoding="ascii"
        )
        with self.assertRaisesRegex(DatasetContractError, "checksum coverage mismatch"):
            verify_dataset_integrity(self.root, workers=1)

    def test_process_lock_rejects_duplicate_instance(self) -> None:
        lock_path = Path(self.temporary.name) / "runs" / ".expert-training-v1.lock"
        with training.TrainingInstanceLock(lock_path, run_id="first"):
            with self.assertRaisesRegex(RuntimeError, "already owns"):
                with training.TrainingInstanceLock(lock_path, run_id="second"):
                    self.fail("duplicate lock unexpectedly succeeded")


class ExpertTrainingResumeTests(unittest.TestCase):
    def test_incomplete_checkpoint_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            checkpoint = run_root / "checkpoints" / "latest.pt"
            checkpoint.parent.mkdir()
            torch.save(
                {
                    "kind": training.CHECKPOINT_KIND,
                    "schema_version": training.CHECKPOINT_SCHEMA_VERSION,
                    "epoch": 1,
                    "global_step": 1,
                },
                checkpoint,
            )
            with self.assertRaisesRegex(RuntimeError, "checkpoint is incomplete"):
                training._load_resume_checkpoint(run_root, torch.device("cpu"))

    def test_legacy_cli_remains_non_resuming_by_default(self) -> None:
        args = training.build_parser().parse_args(
            ["--run-id", "legacy", "--epochs", "1", "--batch-size", "2"]
        )
        self.assertEqual(args.run_id, "legacy")
        self.assertFalse(args.resume)

    def test_interrupted_epoch_boundary_resumes_full_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            output = root / "runs"
            command = [
                "--smoke",
                "--dataset-root",
                str(dataset),
                "--output-root",
                str(output),
                "--run-id",
                "resume-test",
                "--epochs",
                "2",
                "--batch-size",
                "2",
                "--sequence-length",
                "16",
                "--burn-in",
                "4",
                "--workers",
                "0",
                "--integrity-workers",
                "2",
                "--max-train-batches",
                "1",
                "--max-eval-batches",
                "1",
                "--hidden-size",
                "32",
                "--card-embedding-size",
                "16",
                "--device",
                "cpu",
            ]
            args = training.build_parser().parse_args(command)
            append = training._append_jsonl

            def interrupt_after_checkpoint(path: Path, value: dict[str, object]) -> None:
                if value.get("event") == "epoch_complete":
                    raise RuntimeError("simulated interruption")
                append(path, value)

            with mock.patch.object(training, "_append_jsonl", interrupt_after_checkpoint):
                with redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                        training.run(args)

            run_root = output / "resume-test"
            first = torch.load(
                run_root / "checkpoints" / "latest.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(first["epoch"], 1)
            self.assertEqual(first["step"], 1)
            self.assertEqual(first["global_step"], 1)
            for key in (
                "model_state",
                "optimizer_state",
                "scheduler_state",
                "normalizer_state",
                "rng",
            ):
                self.assertIn(key, first)
            self.assertEqual(first["scheduler_state"]["last_epoch"], 1)
            self.assertEqual(
                first["normalizer_state"]["kind"],
                training.DatasetPrecomputedNormalizer.KIND,
            )
            self.assertEqual(
                set(first["rng"]),
                {
                    "python",
                    "numpy",
                    "torch",
                    "cuda",
                    "train_loader_generator",
                },
            )

            resumed_args = training.build_parser().parse_args([*command, "--resume"])
            with redirect_stdout(io.StringIO()):
                self.assertEqual(training.run(resumed_args), run_root)
            final = json.loads((run_root / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(final["epochs_completed"], 2)
            self.assertEqual(final["global_step"], 2)
            latest = torch.load(
                run_root / "checkpoints" / "latest.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(latest["scheduler_state"]["last_epoch"], 2)
            events = [
                json.loads(line)
                for line in (run_root / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            resumed = next(event for event in events if event["event"] == "run_resumed")
            self.assertEqual(resumed["epoch"], 1)
            self.assertEqual(resumed["global_step"], 1)
            result_mtime = (run_root / "result.json").stat().st_mtime_ns
            with mock.patch.object(
                training,
                "RecurrentExpertPolicy",
                side_effect=AssertionError("completed run must not restart training"),
            ):
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(training.run(resumed_args), run_root)
            self.assertEqual((run_root / "result.json").stat().st_mtime_ns, result_mtime)

    def test_resume_without_run_id_is_stable(self) -> None:
        args = training.build_parser().parse_args(["--resume"])
        digest = "a" * 64
        signature, _ = training._run_signature(
            args,
            dataset_manifest_sha256=digest,
            observation_mode="native_state_v1",
        )
        first = training._stable_run_id(
            observation_mode="native_state_v1",
            dataset_manifest_sha256=digest,
            run_signature_sha256=signature,
        )
        second = training._stable_run_id(
            observation_mode="native_state_v1",
            dataset_manifest_sha256=digest,
            run_signature_sha256=signature,
        )
        self.assertEqual(first, second)

    def test_native_production_launcher_passes_explicit_admission_contract(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts" / "start_expert_training_v1.ps1"
        ).read_text(encoding="utf-8")
        self.assertIn("--expected-source-manifest $ExpectedSourceManifest", script)
        self.assertIn("--allow-unanchored-native-states", script)
        self.assertIn("$datasetManifest.source_manifest.path", script)


if __name__ == "__main__":
    unittest.main()
