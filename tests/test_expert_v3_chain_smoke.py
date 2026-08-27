from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.run_expert_v3_chain_smoke import (
    ChainSmokeError,
    SmokeConfig,
    V3ChainSmokeRunner,
    build_config,
    build_parser,
    greedy_select,
    status,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _contract(path: Path) -> None:
    payload = {
        "schema_version": 3,
        "kind": "cr_native_authoritative_contract_v3",
    }
    canonical = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    raw = (json.dumps({**payload, "contract_sha256": canonical}, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    file_sha = hashlib.sha256(raw).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{file_sha}  {path.name}\n", encoding="ascii"
    )


class ExpertV3ChainSmokeTest(unittest.TestCase):
    def test_selector_is_deterministic_multicover_and_balanced(self) -> None:
        index = {
            "allowed": ("c1", "c2", "f1"),
            "allowed_set": {"c1", "c2", "f1"},
            "evolution": ("f1",),
            "hero": (),
            "form_tokens": ("f1",),
            "ability": ("a1",),
            "ability_set": {"a1"},
        }
        candidates = []
        for number in range(6):
            positive = number < 3
            candidates.append({
                "battle_tag": f"B{number}",
                "played_cards": frozenset({"c1", "c2", "f1"}),
                "played_forms": frozenset({"f1"}),
                "ability_candidates": frozenset({"a1"}) if positive else frozenset(),
                "ability_singletons": frozenset({"a1"}) if positive else frozenset(),
                "ability_positive": positive,
                "player_tags": frozenset({f"P{number}"}),
                "stable_rank": hashlib.sha256(f"B{number}".encode()).hexdigest(),
                "manifest_row": {"battle_tag": f"B{number}"},
            })
        with mock.patch(
            "scripts.run_expert_v3_chain_smoke._contract_index",
            return_value=index,
        ):
            first, report = greedy_select(candidates, {}, limit=4)
            second, _ = greedy_select(list(reversed(candidates)), {}, limit=4)
        self.assertEqual(
            [row["battle_tag"] for row in first],
            [row["battle_tag"] for row in second],
        )
        self.assertEqual(report["deficits"], {"card": [], "form": [], "ability": []})
        self.assertGreaterEqual(report["cohorts"]["ability_positive"], 1)
        self.assertGreaterEqual(report["cohorts"]["ability_zero"], 1)

    def test_snapshot_isolated_and_does_not_modify_live_db_or_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            crawler = root / "crawler"
            authoritative = root / "authoritative-schema5-v3"
            crawler.mkdir()
            (crawler / "logs").mkdir()
            authoritative.mkdir()
            db = root / "authoritative.sqlite3"
            index = authoritative / "index.jsonl"
            db.write_bytes(b"immutable-db")
            index.write_bytes(b"immutable-index\n")
            contract = root / "contract.json"
            _contract(contract)
            sources = []
            for number in range(2):
                source = root / f"battle-{number}.json"
                source.write_text(
                    json.dumps({"battle_tag": f"B{number}"}) + "\n",
                    encoding="utf-8",
                )
                sources.append(source)
            config = SmokeConfig(
                project_root=PROJECT_ROOT,
                data_root=root / "smoke",
                crawler_root=crawler,
                crawler_python=root / "crawler-python.exe",
                training_python=root / "python.exe",
                crawler_config=root / "crawler.toml",
                authoritative_db=db,
                authoritative_root=authoritative,
                native_contract=contract,
                template=root / "template.json",
                target=2,
                workers=8,
                avds=2,
                workers_per_avd=4,
                ports=tuple(range(38031, 38039)),
            )
            db_sha = hashlib.sha256(db.read_bytes()).hexdigest()
            index_sha = hashlib.sha256(index.read_bytes()).hexdigest()

            def fake_freeze(**kwargs):
                rows = []
                for number, source in enumerate(sources):
                    rows.append({
                        "battle_tag": f"B{number}",
                        "source_path": str(source),
                        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "source_schema_version": 5,
                        "player_tags": [f"P{number}"],
                    })
                kwargs["output"].parent.mkdir(parents=True, exist_ok=True)
                kwargs["output"].write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                kwargs["output"].with_suffix(".jsonl.manifest.json").write_text(
                    json.dumps({"accepted_battles": 2}) + "\n",
                    encoding="utf-8",
                )
                return {
                    "accepted_battles": 2,
                    "manifest_sha256": hashlib.sha256(
                        kwargs["output"].read_bytes()
                    ).hexdigest(),
                    "authoritative_index_sha256": hashlib.sha256(
                        index.read_bytes()
                    ).hexdigest(),
                }

            features = [
                {
                    "battle_tag": f"B{number}",
                    "manifest_row": {
                        "battle_tag": f"B{number}",
                        "source_path": str(source),
                        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "source_schema_version": 5,
                        "player_tags": [f"P{number}"],
                    },
                    "source_path": str(source),
                    "ability_positive": bool(number),
                }
                for number, source in enumerate(sources)
            ]
            token_receipt = {
                "canonical_sha256": "a" * 64,
                "source_coverage": {
                    "observed_card_tokens": list(range(180)),
                    "observed_form_tokens": list(range(58)),
                    "observed_ability_tokens": list(range(25)),
                },
            }
            runner = V3ChainSmokeRunner(config)
            with (
                mock.patch(
                    "scripts.run_expert_v3_chain_smoke._crawler_active",
                    return_value=False,
                ),
                mock.patch(
                    "scripts.run_expert_v3_chain_smoke.formal_freeze",
                    side_effect=fake_freeze,
                ),
                mock.patch(
                    "scripts.run_expert_v3_chain_smoke.candidate_features",
                    side_effect=features,
                ),
                mock.patch(
                    "scripts.run_expert_v3_chain_smoke._contract_index",
                    return_value={},
                ),
                mock.patch(
                    "scripts.run_expert_v3_chain_smoke.greedy_select",
                    return_value=(features, {"selected": 2, "deficits": {}}),
                ),
                mock.patch(
                    "scripts.run_expert_v3_chain_smoke.build_frozen_source_token_coverage_receipt",
                    return_value=token_receipt,
                ),
            ):
                runner.snapshot()
            self.assertEqual(hashlib.sha256(db.read_bytes()).hexdigest(), db_sha)
            self.assertEqual(hashlib.sha256(index.read_bytes()).hexdigest(), index_sha)
            self.assertEqual(
                len(config.frozen_manifest.read_text(encoding="utf-8").splitlines()),
                2,
            )
            state = json.loads(config.state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["stages"]["freeze_schema5_v3"]["status"], "completed")

    def test_failure_always_requests_native_shutdown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = SmokeConfig(
                project_root=PROJECT_ROOT,
                data_root=root / "smoke",
                crawler_root=root,
                crawler_python=root / "crawler-python.exe",
                training_python=root / "python.exe",
                crawler_config=root / "crawler.toml",
                authoritative_db=root / "db.sqlite3",
                authoritative_root=root / "authoritative-schema5-v3",
                native_contract=root / "contract.json",
                template=root / "template.json",
                target=2,
                workers=8,
                avds=2,
                workers_per_avd=4,
                ports=tuple(range(38031, 38039)),
            )
            calls = []

            class FakeOrchestrator:
                def audit(self): calls.append("audit")
                def generate_native(self):
                    calls.append("generate")
                    raise RuntimeError("boom")
                def _best_effort_stop_native(self): calls.append("emergency-stop")

            runner = V3ChainSmokeRunner(config, orchestrator=FakeOrchestrator())
            with (
                mock.patch.object(runner, "snapshot", side_effect=lambda: calls.append("snapshot")),
                mock.patch(
                    "scripts.run_expert_v3_chain_smoke._crawler_active",
                    return_value=False,
                ),
                mock.patch(
                    "expert_v1.one_click_v1.DEFAULT_NATIVE_HARDWARE_LOCK",
                    root / "hardware.lock",
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    runner.run()
            self.assertEqual(
                calls, ["snapshot", "audit", "generate", "emergency-stop"]
            )

    def test_success_reuses_formal_chain_in_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = SmokeConfig(
                project_root=PROJECT_ROOT,
                data_root=root / "smoke",
                crawler_root=root,
                crawler_python=root / "crawler-python.exe",
                training_python=root / "python.exe",
                crawler_config=root / "crawler.toml",
                authoritative_db=root / "db.sqlite3",
                authoritative_root=root / "authoritative-schema5-v3",
                native_contract=root / "contract.json",
                template=root / "template.json",
                target=2,
                workers=8,
                avds=2,
                workers_per_avd=4,
                ports=tuple(range(38031, 38039)),
            )
            calls = []

            class FakeOrchestrator:
                def audit(self): calls.append("audit")
                def generate_native(self): calls.append("generate")
                def stop_workers(self): calls.append("stop")
                def validate_tick_store(self): calls.append("validate")
                def compile(self): calls.append("compile")
                def smoke(self): calls.append("train-smoke")
                def _best_effort_stop_native(self): calls.append("unexpected-stop")

            runner = V3ChainSmokeRunner(config, orchestrator=FakeOrchestrator())
            with (
                mock.patch.object(
                    runner, "snapshot", side_effect=lambda: calls.append("snapshot")
                ),
                mock.patch(
                    "scripts.run_expert_v3_chain_smoke._crawler_active",
                    return_value=False,
                ),
                mock.patch(
                    "expert_v1.one_click_v1.DEFAULT_NATIVE_HARDWARE_LOCK",
                    root / "hardware.lock",
                ),
            ):
                runner.run()
            self.assertEqual(calls, [
                "snapshot", "audit", "generate", "stop",
                "validate", "compile", "train-smoke",
            ])

    def test_status_is_read_only_and_config_is_fixed_2x4(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "absent"
            args = build_parser().parse_args(["--output-root", str(root)])
            config = build_config(args)
            value = status(config)
            self.assertFalse(root.exists())
            self.assertEqual(config.avds, 2)
            self.assertEqual(config.workers, 8)
            self.assertEqual(config.ports, tuple(range(38031, 38039)))
            self.assertEqual(value["stages"]["freeze_schema5_v3"], "pending")


if __name__ == "__main__":
    unittest.main()
