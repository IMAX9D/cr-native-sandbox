from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from expert_v1.native_dataset_generator import (
    COORDINATE_PROVENANCE,
    _failure_class,
    _failure_domain,
    NativeDatasetTask,
    RecordingCountingEnv,
    StagedTickSink,
    StoredFrameRegistry,
    execute_task,
    atomic_json,
    prepare_run,
    reconcile_result_files,
    recover_unmanifested_final_shards,
    requeue_failed_infrastructure,
    select_tasks,
    should_retry_failure,
    summarize_results,
    verify_published_tick_store,
)
from expert_v1.native_profile import native_teacher_forced_profile
from expert_v1.tick_store_v1.schema import (
    EpisodeState,
    PlayerPrivate,
    TickState,
)
from expert_v1.tick_store_v1.shard import WorkerShardSink, build_store_manifest
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
    def test_failure_domains_keep_infrastructure_out_of_semantic_rejections(self) -> None:
        timeout = _failure_class(
            None, TimeoutError("RPC timeout"), "native_teacher_forced_replay"
        )
        disk = _failure_class(
            None, OSError("disk full"), "immutable_tick_store_commit"
        )
        source_sha = _failure_class(
            None, RuntimeError("changed"), "source_sha_verification"
        )
        self.assertEqual(_failure_domain(timeout), "infrastructure")
        self.assertEqual(_failure_domain(disk), "infrastructure")
        self.assertEqual(_failure_domain(source_sha), "infrastructure")
        self.assertEqual(_failure_domain("native_action_rejected"), "semantic")
        protocol = _failure_class(
            SimpleNamespace(failure="native_action_count_mismatch_tick_20"),
            None,
            "first_native_difference",
        )
        hand = _failure_class(
            SimpleNamespace(failure="hand_mismatch_event_7"),
            None,
            "first_native_difference",
        )
        unknown = _failure_class(
            SimpleNamespace(failure="new_unrecognized_runner_failure"),
            None,
            "first_native_difference",
        )
        terminal = _failure_class(
            SimpleNamespace(failure="native_terminal_before_execution_tick_80"),
            None,
            "first_native_difference",
        )
        self.assertEqual(_failure_domain(protocol), "infrastructure")
        self.assertEqual(_failure_domain(hand), "source_integrity")
        self.assertEqual(_failure_domain(unknown), "infrastructure")
        self.assertEqual(_failure_domain(terminal), "semantic")
        self.assertTrue(should_retry_failure({
            "teacher_forced_success": False,
            "failure_domain": "infrastructure",
        }, 1))
        self.assertFalse(should_retry_failure({
            "teacher_forced_success": False,
            "failure_domain": "infrastructure",
        }, 3))
        self.assertFalse(should_retry_failure({
            "teacher_forced_success": False,
            "failure_domain": "semantic",
        }, 1))

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
            self.assertEqual(execution.record["failure_domain"], "infrastructure")
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

    def test_new_run_requeues_only_infrastructure_with_fresh_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue_path = root / "queue.sqlite3"
            with TickStoreWorkQueue(queue_path) as queue:
                queue.add_tasks([
                    {
                        "battle_tag": tag, "source_path": f"{tag}.json",
                        "source_sha256": "a" * 64, "payload": {},
                    }
                    for tag in ("INFRA", "SEMANTIC")
                ])
                queue.connection.execute(
                    "UPDATE tasks SET status='failed', attempts=3"
                )
            for tag, domain in (
                ("INFRA", "infrastructure"), ("SEMANTIC", "semantic")
            ):
                atomic_json(root / "results" / f"{tag}.json", {
                    "schema_version": 1,
                    "kind": "expert_authoritative_native_tick_result_v1",
                    "battle_tag": tag,
                    "teacher_forced_success": False,
                    "failure_domain": domain,
                })
            self.assertEqual(
                requeue_failed_infrastructure(root, queue_path), 1
            )
            with TickStoreWorkQueue(queue_path) as queue:
                rows = {
                    row["battle_tag"]: (row["status"], row["attempts"])
                    for row in queue.connection.execute(
                        "SELECT battle_tag,status,attempts FROM tasks"
                    )
                }
            self.assertEqual(rows["INFRA"], ("pending", 0))
            self.assertEqual(rows["SEMANTIC"], ("failed", 3))

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
        self.assertEqual(metrics["native_actions_responded"], 2)
        self.assertEqual(metrics["native_actions_rejected"], 1)
        self.assertEqual(metrics["native_actions_no_response"], 0)
        self.assertEqual(metrics["true_attempted_acceptance_rate"], 0.5)
        self.assertEqual(metrics["native_deploy_actions_accepted"], 1)
        self.assertEqual(metrics["native_ability_actions_accepted"], 0)

        class TimeoutEnv:
            def joint_act(self, actions):
                raise TimeoutError("transport stalled")

        timed_out = RecordingCountingEnv(TimeoutEnv())
        with self.assertRaises(TimeoutError):
            timed_out.joint_act([{"type": "play", "side": 0}])
        timeout_metrics = timed_out.metrics()
        self.assertEqual(timeout_metrics["native_actions_attempted"], 1)
        self.assertEqual(timeout_metrics["native_actions_responded"], 0)
        self.assertEqual(timeout_metrics["native_actions_no_response"], 1)
        self.assertEqual(timeout_metrics["native_action_exceptions"], 1)

    def test_finalize_crash_window_rebuilds_missing_shard_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sink = WorkerShardSink(root, "worker", episodes_per_shard=1)
            sink.append("RECOVER", tick_states(), {"stable": 1})
            sink.finalize()
            manifest = next(root.glob("worker-*.manifest.json"))
            manifest.unlink()
            self.assertEqual(
                recover_unmanifested_final_shards(root, "worker"), 1
            )
            recovered = json.loads(manifest.read_text())
            self.assertTrue(recovered["recovered_after_finalize_crash"])
            self.assertEqual(recovered["episode_count"], 1)
            self.assertEqual(
                recover_unmanifested_final_shards(root, "worker"), 0
            )
            selection = root / "selection.jsonl"
            selection.write_text("{}\n", encoding="utf-8")
            build_store_manifest(
                root, source_manifest=selection,
                expected_episodes=1, expected_ticks=len(tick_states()),
            )
            self.assertEqual(
                verify_published_tick_store(root)["episodes"], 1
            )
            data = next(root.glob("worker-*.crts"))
            with data.open("r+b") as handle:
                handle.seek(-1, 2)
                value = handle.read(1)
                handle.seek(-1, 2)
                handle.write(bytes([value[0] ^ 0x01]))
            with self.assertRaises(RuntimeError):
                verify_published_tick_store(root)

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
            "native_actions_responded": 4,
            "native_actions_accepted": 3,
            "native_actions_rejected": 1,
            "native_actions_no_response": 0,
            "native_action_response_excess": 0,
            "native_action_exceptions": 0,
            "native_deploy_actions_attempted": 3,
            "native_deploy_actions_accepted": 3,
            "native_ability_actions_attempted": 1,
            "native_ability_actions_accepted": 0,
            "teacher_forced_success": False,
            "failure_class": "ability_branch_required",
            "failure_domain": "semantic",
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
        self.assertTrue(summary["native_action_accounting_closed"])

        infrastructure = dict(base)
        infrastructure.update({
            "failure_class": "infrastructure_native_replay_exception",
            "failure_domain": "infrastructure",
            "native_actions_responded": 3,
            "native_actions_rejected": 0,
            "native_actions_no_response": 1,
        })
        blocked = summarize_results(
            [task], [infrastructure], queue_counts={"failed": 1},
            worker_reports=[{"worker_error": None}], wall_seconds=2.0,
            missing_tags=[], unexpected_tags=[],
        )
        self.assertFalse(blocked["infrastructure_complete"])
        self.assertFalse(blocked["publication_ready"])
        self.assertEqual(
            blocked["failure_domain_counts"], {"infrastructure": 1}
        )


if __name__ == "__main__":
    unittest.main()
