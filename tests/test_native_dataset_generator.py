from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from expert_v1.native_dataset_generator import (
    COORDINATE_PROVENANCE,
    NativeDatasetTask,
    RecordingCountingEnv,
    StagedTickSink,
    StoredFrameRegistry,
    execute_task,
    atomic_json,
    prepare_run,
    reconcile_result_files,
    select_tasks,
    summarize_results,
)
from expert_v1.native_profile import native_teacher_forced_profile
from expert_v1.tick_store_v1.schema import (
    EpisodeState,
    PlayerPrivate,
    TickState,
)
from expert_v1.tick_store_v1.shard import WorkerShardSink
from expert_v1.tick_store_v1.work_queue import TickStoreWorkQueue


def tick_states(count: int = 4) -> list[TickState]:
    return [
        TickState(
            tick=10 + index,
            players=(
                PlayerPrivate(0, 50_000, (0, 1, 2, 3), 4),
                PlayerPrivate(1, 50_000, (4, 5, 6, 7), 0),
            ),
            towers=(),
            entities=(),
            episode=EpisodeState(1, 0, 1, 0, 0, 0, 0, 0, 0),
        )
        for index in range(count)
    ]


def candidate(tag: str, path: Path, *, abilities: int) -> dict:
    return {
        "ability_count_reported": abilities,
        "ability_events_observed": abilities,
        "ability_log_tier": (
            "observed_ticks_identity_runtime_resolved"
            if abilities else "source_reports_zero"
        ),
        "authoritative_native_full_candidate": True,
        "battle_tag": tag,
        "compiler_native_replay_ready": True,
        "coordinate_tier": "all_card_events_raw_data_i",
        "deployment_actions": 10,
        "duration_ticks": 100,
        "eligibility_tier": (
            "authoritative_native_ability_exact"
            if abilities else "authoritative_native_deployment_only"
        ),
        "source_path": str(path),
        "source_schema_version": 3,
        "source_sha256": "a" * 64,
    }


def write_candidates(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class NativeDatasetGeneratorTest(unittest.TestCase):
    def test_limited_selection_is_deterministic_and_mixed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue = root / "candidates.jsonl"
            rows = [
                candidate("P1", root / "p1.json", abilities=2),
                candidate("P2", root / "p2.json", abilities=1),
                candidate("Z1", root / "z1.json", abilities=0),
                candidate("Z2", root / "z2.json", abilities=0),
            ]
            write_candidates(queue, rows)
            first, summary = select_tasks(queue, limit=2, selection_seed="smoke")
            second, _ = select_tasks(queue, limit=2, selection_seed="smoke")
            self.assertEqual(first, second)
            self.assertEqual(sum(task.ability_positive for task in first), 1)
            self.assertEqual(sum(not task.ability_positive for task in first), 1)
            self.assertEqual(summary["selected_rows"], 2)
            self.assertNotIn("source_json", first[0].json())

    def test_explicit_stratum_quotas_are_exact_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue = root / "candidates.jsonl"
            rows = [
                candidate("P1", root / "p1.json", abilities=2),
                candidate("P2", root / "p2.json", abilities=1),
                candidate("P3", root / "p3.json", abilities=3),
                candidate("Z1", root / "z1.json", abilities=0),
                candidate("Z2", root / "z2.json", abilities=0),
            ]
            write_candidates(queue, rows)
            selected, summary = select_tasks(
                queue, limit=4, selection_seed="quota",
                deployment_zero_quota=1, ability_exact_quota=3,
            )
            self.assertEqual(sum(not task.ability_positive for task in selected), 1)
            self.assertEqual(sum(task.ability_positive for task in selected), 3)
            self.assertEqual(summary["explicit_stratum_quotas"], {
                "authoritative_native_deployment_only": 1,
                "authoritative_native_ability_exact": 3,
            })
            with self.assertRaises(ValueError):
                select_tasks(
                    queue, limit=4, deployment_zero_quota=1,
                    ability_exact_quota=2,
                )
            with self.assertRaises(RuntimeError):
                select_tasks(
                    queue, limit=5, deployment_zero_quota=3,
                    ability_exact_quota=2,
                )

    def test_prepare_is_idempotent_and_contract_change_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_queue = root / "candidates.jsonl"
            write_candidates(source_queue, [
                candidate("P", root / "p.json", abilities=1),
                candidate("Z", root / "z.json", abilities=0),
            ])
            template = root / "template.json"
            template.write_text('{"battle":{}}\n', encoding="utf-8")
            output = root / "output"
            args = dict(
                candidate_queue=source_queue,
                output_root=output,
                template_path=template,
                limit=2,
                selection_seed="stable",
                seed=1,
                maximum_seeds_to_test=16,
                trace_batch_steps=8,
                episodes_per_shard=4,
            )
            first = prepare_run(**args)
            second = prepare_run(**args)
            self.assertEqual(first[0], second[0])
            with TickStoreWorkQueue(first[2]) as queue:
                self.assertEqual(queue.counts(), {"pending": 2})
            changed = dict(args)
            changed["selection_seed"] = "different"
            with self.assertRaises(RuntimeError):
                prepare_run(**changed)

    def test_staged_episode_commits_once_and_can_be_reused_after_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = StoredFrameRegistry(root)
            sink = WorkerShardSink(root, "worker-a", episodes_per_shard=10)
            first_stage = StagedTickSink()
            first_stage.append("BATTLE", tick_states(), {"stable": 1})
            self.assertIsNotNone(first_stage.episode)
            first = registry.commit_or_reuse(sink, first_stage.episode)  # type: ignore[arg-type]
            second_stage = StagedTickSink()
            second_stage.append("BATTLE", tick_states(), {"stable": 1})
            second = registry.commit_or_reuse(sink, second_stage.episode)  # type: ignore[arg-type]
            self.assertFalse(first["resume_reused_existing_frame"])
            self.assertTrue(second["resume_reused_existing_frame"])
            self.assertEqual(first["payload_sha256"], second["payload_sha256"])
            manifests = sink.finalize()
            self.assertEqual(manifests[0]["episode_count"], 1)

    def test_source_sha_mismatch_never_writes_a_tick_frame_or_source_copy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            source.write_text('{"secret":"must-not-copy"}\n', encoding="utf-8")
            task = NativeDatasetTask(
                0, "b" * 64, "BAD-SHA", str(source), "a" * 64,
                3, 100, 10, 0, 0, "source_reports_zero",
                "all_card_events_raw_data_i",
                "authoritative_native_deployment_only",
            )
            shard_root = root / "shards"
            sink = WorkerShardSink(shard_root, "worker", episodes_per_shard=2)
            execution = execute_task(
                object(), task, {}, sink, StoredFrameRegistry(shard_root),
                worker_id="worker", port=1, attempt=1,
            )
            self.assertFalse(execution.record["teacher_forced_success"])
            self.assertEqual(execution.record["failure_class"], "source_sha_mismatch")
            self.assertEqual(sink.writer.episode_count, 0)
            diagnostic_text = json.dumps(execution.diagnostic, ensure_ascii=False)
            self.assertNotIn("must-not-copy", diagnostic_text)
            self.assertFalse(execution.diagnostic["source_identity"]["source_json_copied"])
            self.assertEqual(sink.finalize(), [])

    def test_result_frame_checkpoint_reconciles_interrupted_queue_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue_path = root / "queue.sqlite3"
            with TickStoreWorkQueue(queue_path) as queue:
                queue.add_tasks([{
                    "battle_tag": "CRASH-WINDOW",
                    "source_path": "source.json",
                    "source_sha256": "a" * 64,
                    "payload": {},
                }])
                claim = queue.claim(
                    "worker", limit=1, lease_seconds=300,
                    maximum_attempts=10,
                )[0]
                self.assertEqual(claim.battle_tag, "CRASH-WINDOW")
            shards = root / "shards"
            sink = WorkerShardSink(shards, "worker", episodes_per_shard=1)
            entry = sink.append("CRASH-WINDOW", tick_states(), {"stable": 1})
            sink.finalize()
            atomic_json(root / "results" / "CRASH_WINDOW.json", {
                "schema_version": 1,
                "kind": "expert_authoritative_native_tick_result_v1",
                "battle_tag": "CRASH-WINDOW",
                "teacher_forced_success": True,
                "tick_store_entry": entry,
            })
            self.assertEqual(reconcile_result_files(root, queue_path), 1)
            with TickStoreWorkQueue(queue_path) as queue:
                self.assertEqual(queue.counts(), {"done": 1})
                row = queue.connection.execute(
                    "SELECT episode_sha256 FROM tasks WHERE battle_tag=?",
                    ("CRASH-WINDOW",),
                ).fetchone()
                self.assertEqual(row["episode_sha256"], entry["payload_sha256"])
            # Reconciliation is idempotent and never appends another frame.
            self.assertEqual(reconcile_result_files(root, queue_path), 0)
            manifests = list(shards.glob("*.manifest.json"))
            self.assertEqual(len(manifests), 1)
            self.assertEqual(
                json.loads(manifests[0].read_text())["episode_count"], 1
            )

    def test_native_action_metrics_use_attempted_not_planned_denominator(self) -> None:
        class FakeEnv:
            def joint_act(self, actions):
                return {
                    "actions": [
                        {"result": {"accepted": True, "result_code": 0}},
                        {"result": {"accepted": False, "result_code": 4}},
                    ]
                }

        recorder = RecordingCountingEnv(FakeEnv())
        recorder.joint_act([
            {"type": "play", "side": 0},
            {"type": "ability", "side": 1},
        ])
        metrics = recorder.metrics()
        self.assertEqual(metrics["native_actions_attempted"], 2)
        self.assertEqual(metrics["native_actions_accepted"], 1)
        self.assertEqual(metrics["true_attempted_acceptance_rate"], 0.5)
        self.assertEqual(metrics["native_deploy_actions_accepted"], 1)
        self.assertEqual(metrics["native_ability_actions_accepted"], 0)

    def test_summary_separates_semantic_rejection_from_infrastructure(self) -> None:
        task = NativeDatasetTask(
            0, "c" * 64, "A", "a.json", "d" * 64, 3, 100, 10,
            1, 1, "observed_ticks_identity_runtime_resolved",
            "all_card_events_raw_data_i", "authoritative_native_ability_exact",
        )
        base = {
            "battle_tag": "A",
            "source_sha_verified": True,
            "native_teacher_forced_profile": native_teacher_forced_profile(1),
            "coordinate_provenance": COORDINATE_PROVENANCE,
            "native_actions_attempted": 4,
            "native_actions_accepted": 3,
            "native_deploy_actions_attempted": 3,
            "native_deploy_actions_accepted": 3,
            "native_ability_actions_attempted": 1,
            "native_ability_actions_accepted": 0,
            "teacher_forced_success": False,
            "failure_class": "ability_branch_required",
            "terminal_diagnostic_status": "not_reached",
        }
        summary = summarize_results(
            [task], [base], queue_counts={"failed": 1}, worker_reports=[{
                "worker_error": None,
            }], wall_seconds=2.0, missing_tags=[], unexpected_tags=[],
        )
        self.assertTrue(summary["infrastructure_complete"])
        self.assertTrue(summary["publication_ready"])
        self.assertEqual(summary["true_attempted_acceptance_rate"], 0.75)
        self.assertEqual(summary["branch_required_battles"], 1)


if __name__ == "__main__":
    unittest.main()
