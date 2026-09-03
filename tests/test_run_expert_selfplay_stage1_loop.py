from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from scripts.run_expert_selfplay_stage1_loop import (
    LOCK_FILENAME,
    LOOP_KIND,
    RunRootLock,
    _validated_child_result,
    run,
)


def _argument(command: list[str], name: str) -> str | None:
    try:
        return command[command.index(name) + 1]
    except ValueError:
        return None


class FakeStage1Runner:
    def __init__(self, *, fail_call: int | None = None) -> None:
        self.commands: list[list[str]] = []
        self.fail_call = fail_call

    def __call__(self, command, **kwargs):
        del kwargs
        command = list(command)
        self.commands.append(command)
        call = len(self.commands)
        run_dir = Path(_argument(command, "--run-dir"))
        run_dir.mkdir(parents=True)
        if self.fail_call == call:
            (run_dir / "progress.json").write_text(
                json.dumps({"status": "failed", "error": "injected"}),
                encoding="utf-8",
            )
            return SimpleNamespace(returncode=17, stdout="partial", stderr="injected")
        checkpoint = run_dir / "checkpoints" / f"checkpoint-{call:012d}.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(f"checkpoint-{call}".encode("ascii"))
        result = {
            "kind": "cr_native_expert_selfplay_stage1_result_v1",
            "status": "completed",
            "run_id": run_dir.name,
            "episodes": int(_argument(command, "--episodes")),
            "decisions": 100 + call,
            "chunks": 10 + call,
            "updates": 1,
            "checkpoint": str(checkpoint.resolve()),
            "ledger_state": "COMMITTED",
        }
        (run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout=json.dumps(result), stderr="")


class InterruptingStage1Runner:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def __call__(self, command, **kwargs):
        del kwargs
        command = list(command)
        self.commands.append(command)
        run_dir = Path(_argument(command, "--run-dir"))
        run_dir.mkdir(parents=True)
        (run_dir / "progress.json").write_text(
            json.dumps({"status": "collecting"}), encoding="utf-8"
        )
        raise KeyboardInterrupt


class Stage1FreshBatchLoopTests(unittest.TestCase):
    def _args(self, root: Path, *, batches: int = 3) -> argparse.Namespace:
        for name in ("base.pt", "expert-manifest.json", "learner.json", "runtime.json"):
            (root / name).write_bytes(b"fixture")
        pool = root / "decks"
        pool.mkdir(exist_ok=True)
        (pool / "deck-01.json").write_text("{}", encoding="utf-8")
        return argparse.Namespace(
            checkpoint=root / "base.pt",
            expert_manifest=root / "expert-manifest.json",
            ports="39031-39034",
            worker_count=4,
            step_ticks=1,
            run_root=root / "loop",
            batches=batches,
            start_resume=None,
            host="127.0.0.1",
            learner_deck=root / "learner.json",
            opponent_deck_root=pool,
            runtime_manifest=root / "runtime.json",
            max_decisions=100,
            timeout=2.0,
            seed=700,
            device="cpu",
            cpu_threads=4,
            retain_checkpoints=3,
            python="python-test",
            runner_script=root / "fake-stage1.py",
        )

    def test_every_update_has_fresh_run_seed_and_checkpoint_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self._args(Path(temporary))
            fake = FakeStage1Runner()
            result = run(args, runner=fake)

            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["completed_count"], 3)
            self.assertEqual(len(fake.commands), 3)
            run_dirs = [Path(_argument(row, "--run-dir")) for row in fake.commands]
            self.assertEqual(len(set(run_dirs)), 3)
            self.assertEqual(
                [int(_argument(row, "--seed")) for row in fake.commands],
                [700, 701, 702],
            )
            for command in fake.commands:
                self.assertEqual(_argument(command, "--updates"), "1")
                self.assertEqual(_argument(command, "--episodes"), "4")
                self.assertEqual(_argument(command, "--ports"), "39031-39034")
                self.assertEqual(_argument(command, "--step-ticks"), "1")
            self.assertIsNone(_argument(fake.commands[0], "--resume-checkpoint"))
            self.assertEqual(
                Path(_argument(fake.commands[1], "--resume-checkpoint")),
                Path(result["completed_batches"][0]["checkpoint"]),
            )
            self.assertEqual(
                Path(_argument(fake.commands[2], "--resume-checkpoint")),
                Path(result["completed_batches"][1]["checkpoint"]),
            )
            progress = json.loads((args.run_root / "progress.json").read_text())
            self.assertEqual(progress["kind"], LOOP_KIND)
            self.assertEqual(progress["completed_count"], 3)

    def test_start_resume_is_only_the_first_link(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self._args(root, batches=2)
            start = root / "start-resume.pt"
            start.write_bytes(b"resume")
            args.start_resume = start
            fake = FakeStage1Runner()

            result = run(args, runner=fake)

            self.assertEqual(
                Path(_argument(fake.commands[0], "--resume-checkpoint")),
                start.resolve(),
            )
            self.assertEqual(
                Path(_argument(fake.commands[1], "--resume-checkpoint")),
                Path(result["completed_batches"][0]["checkpoint"]),
            )

    def test_failure_stops_immediately_and_manual_resume_skips_commits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self._args(root, batches=3)
            first = FakeStage1Runner(fail_call=2)
            with self.assertRaisesRegex(RuntimeError, "exit code 17"):
                run(args, runner=first)
            self.assertEqual(len(first.commands), 2, "failure must not auto-restart")
            failed_dir = Path(_argument(first.commands[1], "--run-dir"))
            self.assertTrue(failed_dir.is_dir(), "failed scene must be preserved")
            progress = json.loads((args.run_root / "progress.json").read_text())
            self.assertEqual(progress["status"], "failed")
            self.assertEqual(progress["completed_count"], 1)

            second = FakeStage1Runner()
            result = run(args, runner=second)
            self.assertEqual(len(second.commands), 2)
            self.assertEqual(int(_argument(second.commands[0], "--seed")), 701)
            self.assertNotEqual(Path(_argument(second.commands[0], "--run-dir")), failed_dir)
            self.assertEqual(
                Path(_argument(second.commands[0], "--resume-checkpoint")),
                Path(result["completed_batches"][0]["checkpoint"]),
            )
            self.assertEqual(result["completed_count"], 3)

            no_repeat = FakeStage1Runner()
            final = run(args, runner=no_repeat)
            self.assertEqual(no_repeat.commands, [])
            self.assertEqual(final["completed_count"], 3)

    def test_parent_interruption_adopts_already_committed_active_child(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self._args(root, batches=2)
            first = FakeStage1Runner()
            completed = run(args, runner=first)

            # Emulate the narrow crash window after child commit but before the
            # parent copied that result into completed_batches.
            first_record = completed["completed_batches"][0]
            second_record = completed["completed_batches"][1]
            progress_path = args.run_root / "progress.json"
            progress = json.loads(progress_path.read_text())
            progress["status"] = "running"
            progress["completed_batches"] = [first_record]
            progress["completed_count"] = 1
            progress["next_batch_index"] = 2
            progress["latest_checkpoint"] = first_record["checkpoint"]
            progress["active_batch"] = {
                "batch_index": 2,
                "seed": 701,
                "run_dir": second_record["run_dir"],
                "resume_checkpoint": first_record["checkpoint"],
                "state": "running",
            }
            progress_path.write_text(json.dumps(progress), encoding="utf-8")

            fake = FakeStage1Runner()
            recovered = run(args, runner=fake)
            self.assertEqual(fake.commands, [])
            self.assertEqual(recovered["completed_count"], 2)
            events = (args.run_root / "events.jsonl").read_text(encoding="utf-8")
            self.assertIn("committed_batch_recovered", events)

    def test_parent_failure_is_not_evidence_that_child_failed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self._args(root, batches=1)
            interrupted = InterruptingStage1Runner()
            with self.assertRaises(KeyboardInterrupt):
                run(args, runner=interrupted)

            retry = FakeStage1Runner()
            with self.assertRaisesRegex(RuntimeError, "outcome is unknown"):
                run(args, runner=retry)
            self.assertEqual(retry.commands, [], "unknown child must not be recollected")

    def test_run_root_lock_rejects_a_second_owner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / LOCK_FILENAME
            with RunRootLock(lock_path):
                with self.assertRaisesRegex(RuntimeError, "already active"):
                    with RunRootLock(lock_path):
                        self.fail("second lock owner must never enter")
            with RunRootLock(lock_path):
                self.assertTrue(lock_path.is_file())

    def test_child_checkpoint_must_be_inside_child_checkpoint_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "batch-000001-seed-700"
            run_dir.mkdir()
            outside = root / "previous-checkpoint.pt"
            outside.write_bytes(b"stale")
            result = {
                "status": "completed",
                "ledger_state": "COMMITTED",
                "run_id": run_dir.name,
                "updates": 1,
                "checkpoint": str(outside),
            }
            (run_dir / "result.json").write_text(
                json.dumps(result), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "outside its checkpoints"):
                _validated_child_result(run_dir)


if __name__ == "__main__":
    unittest.main()
