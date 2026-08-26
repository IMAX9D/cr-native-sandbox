from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from expert_v1.one_click_v1 import (
    OneClickConfig,
    OneClickError,
    OneClickLock,
    OneClickOrchestrator,
    STAGES,
    _crawler_active,
    _default_config,
    build_parser,
    main,
    StageJournal,
    compile_command,
    evaluate_ability_positive_coverage,
    file_fingerprint,
    formal_training_command,
    native_contract_binding,
    native_generation_command,
    native_worker_command,
    training_smoke_command,
    validate_native_result_records,
    validate_schema5_candidate_queue,
    value_fingerprint,
)


class ExpertOneClickV1Test(unittest.TestCase):
    @staticmethod
    def _contract(
        path: Path,
        *,
        schema_version: int = 3,
        kind: str = "cr_native_authoritative_contract_v3",
    ) -> tuple[str, str]:
        import hashlib

        payload = {
            "schema_version": schema_version,
            "kind": kind,
        }
        canonical = hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        value = {**payload, "contract_sha256": canonical}
        raw = (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")
        path.write_bytes(raw)
        file_sha = hashlib.sha256(raw).hexdigest()
        path.with_suffix(path.suffix + ".sha256").write_text(
            f"{file_sha}  {path.name}\n", encoding="ascii"
        )
        return canonical, file_sha

    def test_contract_v2_is_rejected_before_pipeline_use(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            contract = Path(temporary) / "contract-v2.json"
            self._contract(
                contract,
                schema_version=2,
                kind="cr_native_authoritative_contract_v2",
            )
            with self.assertRaisesRegex(OneClickError, "requires.*contract v3"):
                native_contract_binding(contract)

    def test_legacy_v2_state_is_rejected_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "state.json"
            state.write_text(
                json.dumps({
                    "schema_version": 1,
                    "kind": "cr_expert_one_click_state_v1",
                    "stages": {"collect_schema5_v2": {"status": "completed"}},
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(OneClickError, "mixing is forbidden"):
                StageJournal(state)

    def test_parser_rejects_legacy_v2_output_namespaces(self) -> None:
        args = build_parser().parse_args([
            "--data-root",
            str(Path(tempfile.gettempdir()) / "one-click-schema5-v2"),
        ])
        with self.assertRaisesRegex(OneClickError, "legacy v2 state"):
            _default_config(args)

        args = build_parser().parse_args([
            "--authoritative-root",
            str(Path(tempfile.gettempdir()) / "authoritative-schema5-v2"),
        ])
        with self.assertRaisesRegex(OneClickError, "old v2 output is read-only"):
            _default_config(args)

    def test_single_instance_lock_rejects_second_owner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock = Path(temporary) / "run.lock"
            with OneClickLock(lock):
                with self.assertRaisesRegex(OneClickError, "already running"):
                    with OneClickLock(lock):
                        self.fail("second lock owner must never enter")

    def test_native_layout_is_selected_once_and_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = OneClickConfig(
                project_root=root,
                data_root=root / "run",
                crawler_root=root / "crawler",
                crawler_python=root / "crawler-python.exe",
                training_python=root / "python.exe",
                crawler_config=root / "crawler.toml",
                authoritative_db=root / "db.sqlite3",
                authoritative_root=root / "authoritative-schema5-v3",
                native_contract=root / "contract.json",
                template=root / "template.json",
                workers=0,
                avds=0,
                ports=(),
                native_layout_reason="pending_post_collection_ram_preflight",
            )
            orchestrator = OneClickOrchestrator(config)
            with mock.patch(
                "expert_v1.one_click_v1.available_physical_memory_bytes",
                return_value=17 * 1024**3,
            ):
                orchestrator._ensure_native_layout()
            self.assertEqual(orchestrator.config.avds, 2)
            self.assertEqual(orchestrator.config.workers, 8)
            self.assertEqual(
                orchestrator.config.ports, tuple(range(38031, 38039))
            )
            frozen = dict(orchestrator.journal.value["native_layout"])
            with mock.patch(
                "expert_v1.one_click_v1.available_physical_memory_bytes",
                side_effect=AssertionError("resume must not re-probe RAM"),
            ):
                orchestrator._ensure_native_layout()
            self.assertEqual(orchestrator.journal.value["native_layout"], frozen)

    def test_config_construction_and_status_do_not_probe_ram(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary) / "fresh"
            args = build_parser().parse_args(
                ["--status", "--data-root", str(data_root)]
            )
            with mock.patch(
                "expert_v1.one_click_v1.available_physical_memory_bytes",
                side_effect=AssertionError("status must not select native layout"),
            ):
                config = _default_config(args)
            self.assertEqual(config.avds, 0)
            self.assertEqual(config.workers, 0)
            self.assertEqual(config.ports, ())

    def test_native_hardware_lock_is_global_across_data_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            common = dict(
                project_root=root,
                crawler_root=root / "crawler",
                crawler_python=root / "crawler-python.exe",
                training_python=root / "python.exe",
                crawler_config=root / "crawler.toml",
                authoritative_db=root / "db.sqlite3",
                authoritative_root=root / "authoritative-schema5-v3",
                native_contract=root / "contract.json",
                template=root / "template.json",
            )
            first = OneClickConfig(data_root=root / "one", **common)
            second = OneClickConfig(data_root=root / "two", **common)
            self.assertEqual(
                first.native_hardware_lock_path,
                second.native_hardware_lock_path,
            )

    def test_run_stops_native_before_validation_and_compile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = OneClickConfig(
                project_root=root,
                data_root=root / "run",
                crawler_root=root / "crawler",
                crawler_python=root / "crawler-python.exe",
                training_python=root / "python.exe",
                crawler_config=root / "crawler.toml",
                authoritative_db=root / "db.sqlite3",
                authoritative_root=root / "authoritative-schema5-v3",
                native_contract=root / "contract.json",
                template=root / "template.json",
                workers=0,
                avds=0,
                ports=(),
                native_layout_reason="pending_post_collection_ram_preflight",
            )
            orchestrator = OneClickOrchestrator(config)
            order: list[str] = []
            methods = (
                "collect", "freeze", "audit", "generate_native",
                "stop_workers", "validate_tick_store", "compile", "smoke", "train",
            )
            patches = [
                mock.patch.object(
                    orchestrator, name,
                    side_effect=lambda name=name: order.append(name),
                )
                for name in methods
            ]
            with ExitStack() as stack:
                stack.enter_context(mock.patch(
                    "expert_v1.one_click_v1.DEFAULT_NATIVE_HARDWARE_LOCK",
                    root / "hardware.lock",
                ))
                stack.enter_context(mock.patch(
                    "expert_v1.one_click_v1.available_physical_memory_bytes",
                    return_value=17 * 1024**3,
                ))
                for patcher in patches:
                    stack.enter_context(patcher)
                orchestrator.run()
            self.assertEqual(order, list(methods))

    def test_completed_stage_resumes_only_when_all_shas_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            output = root / "output.json"
            source.write_text('{"source":1}\n', encoding="utf-8")
            output.write_text('{"output":1}\n', encoding="utf-8")
            journal = StageJournal(root / "state.json")
            inputs = [file_fingerprint(source), value_fingerprint("target", 3)]
            self.assertTrue(journal.begin("freeze_schema5_v3", inputs))
            journal.complete(
                "freeze_schema5_v3", [file_fingerprint(output)], {"rows": 3}
            )
            reopened = StageJournal(root / "state.json")
            self.assertFalse(reopened.begin("freeze_schema5_v3", inputs))
            output.write_text('{"output":2}\n', encoding="utf-8")
            with self.assertRaisesRegex(OneClickError, "artifact SHA changed"):
                reopened.begin("freeze_schema5_v3", inputs)

    def test_interrupted_stage_refuses_changed_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.json"
            source.write_text('{"source":1}\n', encoding="utf-8")
            journal = StageJournal(root / "state.json")
            self.assertTrue(
                journal.begin("audit_schema5_v3", [file_fingerprint(source)])
            )
            source.write_text('{"source":2}\n', encoding="utf-8")
            with self.assertRaisesRegex(OneClickError, "input SHA changed"):
                StageJournal(root / "state.json").begin(
                    "audit_schema5_v3", [file_fingerprint(source)]
                )

    def test_native_queue_rejects_any_schema3_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "battle.json"
            source.write_text("{}\n", encoding="utf-8")
            queue = root / "queue.jsonl"
            queue.write_text(
                json.dumps(
                    {
                        "battle_tag": "old",
                        "source_path": str(source),
                        "source_sha256": file_fingerprint(source)["sha256"],
                        "source_schema_version": 3,
                        "schema5_authoritative_contract_verified": False,
                        "authoritative_native_full_candidate": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(OneClickError, "legacy/non-authoritative"):
                validate_schema5_candidate_queue(queue, authoritative_root=root)

    def test_schema5_queue_is_source_sha_and_root_fenced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "battle.json"
            source.write_text('{"schema_version":5}\n', encoding="utf-8")
            queue = root / "queue.jsonl"
            queue.write_text(
                json.dumps(
                    {
                        "battle_tag": "new",
                        "source_path": str(source),
                        "source_sha256": file_fingerprint(source)["sha256"],
                        "source_schema_version": 5,
                        "schema5_authoritative_contract_verified": True,
                        "authoritative_native_full_candidate": True,
                        "ability_events_observed": 2,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                validate_schema5_candidate_queue(queue, authoritative_root=root),
                {"rows": 1, "ability_positive": 1, "ability_zero": 0},
            )
            source.write_text('{"schema_version":5,"changed":true}\n', encoding="utf-8")
            with self.assertRaisesRegex(OneClickError, "source SHA changed"):
                validate_schema5_candidate_queue(queue, authoritative_root=root)

    def test_candidate_queue_is_exact_frozen_manifest_join(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "battle.json"
            source.write_text('{"schema_version":5}\n', encoding="utf-8")
            sha = file_fingerprint(source)["sha256"]
            frozen = root / "frozen.jsonl"
            frozen.write_text(json.dumps({
                "battle_tag": "B1",
                "source_path": str(source),
                "source_sha256": sha,
                "source_schema_version": 5,
                "contract_sha256": "a" * 64,
                "contract_file_sha256": "b" * 64,
            }) + "\n", encoding="utf-8")
            queue = root / "queue.jsonl"
            row = {
                "battle_tag": "B1",
                "source_path": str(source),
                "source_sha256": sha,
                "source_schema_version": 5,
                "schema5_authoritative_contract_verified": True,
                "authoritative_native_full_candidate": True,
                "contract_sha256": "a" * 64,
                "contract_file_sha256": "b" * 64,
            }
            queue.write_text(json.dumps(row) + "\n", encoding="utf-8")
            self.assertEqual(
                validate_schema5_candidate_queue(
                    queue,
                    authoritative_root=root,
                    frozen_manifest=frozen,
                    expected_rows=1,
                )["rows"],
                1,
            )
            row["source_sha256"] = "c" * 64
            queue.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(OneClickError, "candidate/frozen"):
                validate_schema5_candidate_queue(
                    queue,
                    authoritative_root=root,
                    verify_source_bytes=False,
                    frozen_manifest=frozen,
                    expected_rows=1,
                )

    def test_native_results_cover_every_candidate_with_final_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue = root / "queue.jsonl"
            queue.write_text(
                "".join(
                    json.dumps(
                        {
                            "battle_tag": tag,
                            "ability_events_observed": 1 if tag == "A" else 0,
                        }
                    )
                    + "\n"
                    for tag in ("A", "B")
                ),
                encoding="utf-8",
            )
            results = root / "results.jsonl"
            rows = [
                {
                    "kind": "expert_authoritative_native_tick_result_v1",
                    "battle_tag": "A",
                    "final_attempt": True,
                    "teacher_forced_success": True,
                },
                {
                    "kind": "expert_authoritative_native_tick_result_v1",
                    "battle_tag": "B",
                    "final_attempt": True,
                    "teacher_forced_success": False,
                    "failure_class": "semantic",
                },
            ]
            results.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            self.assertEqual(
                validate_native_result_records(
                    results, queue, expected_rows=2
                ),
                {
                    "rows": 2,
                    "successes": 1,
                    "failures": 1,
                    "failure_class_counts": {"semantic": 1},
                    "success_tags": ["A"],
                    "audit_prefix_tags": [],
                    "unframed_tags": ["B"],
                    "audit_tick_episodes": 1,
                    "ability_positive": {
                        "candidates": 1,
                        "attempted": 1,
                        "successes": 1,
                        "failures": 0,
                        "failure_class_counts": {},
                        "success_rate": 1.0,
                    },
                    "ability_zero": {
                        "candidates": 1,
                        "attempted": 1,
                        "successes": 0,
                        "failures": 1,
                        "failure_class_counts": {"semantic": 1},
                        "success_rate": 0.0,
                    },
                },
            )
            audit = validate_native_result_records(
                results, queue, expected_rows=2
            )
            admitted = evaluate_ability_positive_coverage(
                {"ability_positive": 1, "ability_zero": 1},
                audit,
                minimum_success_count=1,
                minimum_success_rate=0.10,
                waived=False,
                waiver_reason=None,
            )
            self.assertTrue(admitted["gate"]["admitted"])
            rows[1]["final_attempt"] = False
            results.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(OneClickError, "final attempt"):
                validate_native_result_records(
                    results, queue, expected_rows=2
                )

    def test_ability_positive_failures_cannot_hide_behind_overall_success(self) -> None:
        result = {
            "ability_positive": {
                "candidates": 2,
                "attempted": 2,
                "successes": 0,
                "failures": 2,
                "failure_class_counts": {"semantic": 2},
                "success_rate": 0.0,
            },
            "ability_zero": {
                "candidates": 98,
                "attempted": 98,
                "successes": 98,
                "failures": 0,
                "failure_class_counts": {},
                "success_rate": 1.0,
            },
        }
        coverage = evaluate_ability_positive_coverage(
            {"ability_positive": 2, "ability_zero": 98},
            result,
            minimum_success_count=1,
            minimum_success_rate=0.10,
            waived=False,
            waiver_reason=None,
        )
        self.assertFalse(coverage["gate"]["raw_passed"])
        self.assertFalse(coverage["gate"]["admitted"])
        waived = evaluate_ability_positive_coverage(
            {"ability_positive": 2, "ability_zero": 98},
            result,
            minimum_success_count=1,
            minimum_success_rate=0.10,
            waived=True,
            waiver_reason="known native ability replay defect CR-123",
        )
        self.assertTrue(waived["gate"]["admitted"])
        self.assertTrue(waived["gate"]["waiver_applied"])

    def test_native_result_audit_prefix_union_and_tamper_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            queue = root / "queue.jsonl"
            queue.write_text(
                "".join(
                    json.dumps({"battle_tag": tag, "ability_events_observed": 0})
                    + "\n"
                    for tag in ("FULL", "PREFIX")
                ),
                encoding="utf-8",
            )
            results = root / "results.jsonl"
            prefix = {
                "kind": "expert_authoritative_native_tick_result_v1",
                "battle_tag": "PREFIX",
                "final_attempt": True,
                "teacher_forced_success": False,
                "failure_class": "native_action_rejected",
                "failure_domain": "semantic",
                "failure_prefix_semantic_match": True,
                "audit_prefix_tick_store_entry": {"ticks": 12},
                "audit_prefix_extent": {
                    "kind": "cr_native_replay_extent_v1",
                    "extent": "valid_prefix",
                    "training_admission": "audit_only",
                    "terminal_target": "unknown_censored",
                    "failure_tick_has_labels": False,
                },
            }
            rows = [
                {
                    "kind": "expert_authoritative_native_tick_result_v1",
                    "battle_tag": "FULL",
                    "final_attempt": True,
                    "teacher_forced_success": True,
                },
                prefix,
            ]
            results.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            audit = validate_native_result_records(
                results, queue, expected_rows=2
            )
            self.assertEqual(audit["success_tags"], ["FULL"])
            self.assertEqual(audit["audit_prefix_tags"], ["PREFIX"])
            self.assertEqual(audit["unframed_tags"], [])
            prefix["audit_prefix_extent"]["training_admission"] = "full_bc"
            results.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(OneClickError, "audit-prefix"):
                validate_native_result_records(results, queue, expected_rows=2)

    def test_ability_positive_waiver_requires_a_reason(self) -> None:
        result = {
            "ability_positive": {
                "candidates": 1,
                "attempted": 1,
                "successes": 0,
                "failures": 1,
                "failure_class_counts": {"semantic": 1},
            },
            "ability_zero": {
                "candidates": 0,
                "attempted": 0,
                "successes": 0,
                "failures": 0,
                "failure_class_counts": {},
            },
        }
        with self.assertRaisesRegex(OneClickError, "requires a reason"):
            evaluate_ability_positive_coverage(
                {"ability_positive": 1, "ability_zero": 0},
                result,
                minimum_success_count=1,
                minimum_success_rate=0.10,
                waived=True,
                waiver_reason=None,
            )

    def test_one_click_defaults_keep_ability_gate_enabled(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.minimum_ability_positive_success_count, 1)
        self.assertEqual(args.minimum_ability_positive_success_rate, 0.10)
        self.assertFalse(args.waive_ability_positive_coverage)
        with self.assertRaisesRegex(OneClickError, "lowering ability-positive"):
            main(["--minimum-ability-positive-success-rate", "0"])
        with self.assertRaisesRegex(OneClickError, "requires.*reason"):
            main(["--waive-ability-positive-coverage"])

    def test_active_crawler_lock_must_match_config_and_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "logs").mkdir()
            contract = root / "contract.json"
            canonical, _ = self._contract(contract)
            configured = root / "crawler.toml"
            configured.write_text("", encoding="utf-8")
            other = root / "other.toml"
            other.write_text("", encoding="utf-8")
            (root / "logs" / "authoritative-production.lock").write_text(
                json.dumps({
                    "pid": 123,
                    "config": str(other),
                    "contract_sha256": canonical,
                }),
                encoding="utf-8",
            )
            config = OneClickConfig(
                project_root=root,
                data_root=root / "run",
                crawler_root=root,
                crawler_python=root / "python.exe",
                training_python=root / "python.exe",
                crawler_config=configured,
                authoritative_db=root / "db.sqlite3",
                authoritative_root=root / "authoritative-schema5-v3",
                native_contract=contract,
                template=root / "template.json",
            )
            with mock.patch(
                "expert_v1.one_click_v1._pid_alive", return_value=True
            ):
                with self.assertRaisesRegex(OneClickError, "another config"):
                    _crawler_active(config)

    def test_commands_use_isolated_schema5_queue_and_resume_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = OneClickConfig(
                project_root=root / "project",
                data_root=root / "one-click",
                crawler_root=root / "crawler",
                crawler_python=root / "crawler-python.exe",
                training_python=root / "python.exe",
                crawler_config=root / "crawler.toml",
                authoritative_db=root / "authoritative.sqlite3",
                authoritative_root=root / "authoritative-schema5-v3",
                native_contract=root / "contract.json",
                template=root / "template.json",
            )
            compiled_manifest = config.compiled_root / "manifest.json"
            compiled_manifest.parent.mkdir(parents=True)
            compiled_manifest.write_text('{"kind":"compiled"}\n', encoding="utf-8")

            native = native_generation_command(config)
            self.assertEqual(native[native.index("--queue") + 1], str(config.candidate_queue))
            self.assertTrue(config.candidate_queue.is_relative_to(config.data_root))
            self.assertNotEqual(
                config.candidate_queue,
                Path(
                    r"D:\AI_data\cr-native-core\expert-v1\native-eligibility-v1"
                    r"\queues\authoritative-native-full.jsonl"
                ),
            )
            self.assertEqual(
                native[native.index("--native-contract") + 1],
                str(config.native_contract),
            )
            worker = native_worker_command(config, "start")
            self.assertEqual(worker[worker.index("--avds") + 1], "1")
            self.assertEqual(
                worker[worker.index("--workers-per-avd") + 1], "4"
            )
            self.assertLess(
                STAGES.index("stop_native_workers"),
                STAGES.index("validate_tick_store_and_masks"),
            )

            compile_args = compile_command(config)
            self.assertEqual(
                compile_args[compile_args.index("--schema5-manifest") + 1],
                str(config.frozen_manifest),
            )
            self.assertEqual(
                compile_args[
                    compile_args.index("--native-generation-receipt") + 1
                ],
                str(config.native_generation_receipt),
            )
            smoke = training_smoke_command(config)
            self.assertIn("--smoke", smoke)
            self.assertIn("--resume", smoke)
            self.assertEqual(
                smoke[smoke.index("--dataset-root") + 1], str(config.compiled_root)
            )
            self.assertEqual(
                smoke[smoke.index("--expected-source-manifest") + 1],
                str(config.frozen_manifest),
            )
            self.assertIn("--allow-unanchored-native-states", smoke)
            formal = formal_training_command(config)
            self.assertIn("--resume", formal)
            self.assertNotIn("--smoke", formal)
            self.assertIn("--allow-unanchored-native-states", formal)
            self.assertEqual(
                formal[formal.index("--expected-source-manifest") + 1],
                str(config.frozen_manifest),
            )


if __name__ == "__main__":
    unittest.main()
