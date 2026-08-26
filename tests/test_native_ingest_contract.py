from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from expert_v1.native_ingest_contract import (
    CONTRACT_KIND,
    CONTRACT_SCHEMA_VERSION,
    GAME_VERSION,
    NATIVE_EXECUTION_GAME_MODE_PROVENANCE,
    NativeIngestContractError,
    build_native_ingest_contract,
    contract_payload_sha256,
    load_native_ingest_contract,
    validate_ingest_metadata,
    write_native_ingest_contract,
)


class NativeIngestContractTest(unittest.TestCase):
    def test_generated_contract_is_complete_and_crawler_compatible(self) -> None:
        value = build_native_ingest_contract()
        self.assertEqual(value["schema_version"], CONTRACT_SCHEMA_VERSION)
        self.assertEqual(value["kind"], CONTRACT_KIND)
        self.assertEqual(value["game_version"], GAME_VERSION)
        self.assertEqual(
            value["source_numeric_game_mode_ids"],
            [72000006, 72000450, 72000464],
        )
        self.assertEqual(
            value["native_execution_mode_by_source"],
            {
                "72000006": 72000006,
                "72000450": 72000006,
                "72000464": 72000006,
            },
        )
        self.assertEqual(
            value["king_tower_max_hp_by_level"],
            {
                "1": 2400, "2": 2568, "3": 2736, "4": 2904,
                "5": 3096, "6": 3312, "7": 3528, "8": 3768,
                "9": 4008, "10": 4392, "11": 4824, "12": 5304,
                "13": 5832, "14": 6408, "15": 7032, "16": 7728,
            },
        )
        evidence = value["king_tower_level_evidence"]
        self.assertEqual(evidence["ranked_template_level_cap"], 16)
        self.assertEqual(
            evidence["precedence"],
            ["tower_troop_level", "final_king_hp"],
        )
        self.assertEqual(
            evidence["forbidden_inference_fields"],
            ["card_levels", "deck_cards.level"],
        )
        self.assertEqual(value["counts"]["supported_base_cards"], 122)
        self.assertEqual(value["counts"]["allowed_card_tokens"], 180)
        self.assertEqual(value["counts"]["tower_troops"], 4)
        self.assertEqual(value["counts"]["ability_source_tokens"], 25)
        self.assertEqual(
            value["contract_sha256"], contract_payload_sha256(value)
        )
        self.assertTrue(
            set(value["ability_source_tokens"])
            <= set(value["allowed_card_tokens"])
        )

    def test_exact_royaleapi_slugs_and_forms_are_derived(self) -> None:
        value = build_native_ingest_contract()
        allowed = set(value["allowed_card_tokens"])
        for token in (
            "archers", "x-bow", "wall-breakers", "elite-barbarians-ev1",
            "knight-hero", "spirit-empress",
        ):
            self.assertIn(token, allowed)
        self.assertNotIn("archer", allowed)
        self.assertNotIn("xbow", allowed)
        self.assertNotIn("party-hut", allowed)
        self.assertNotIn("giant-ev1", allowed)

        knight = next(
            row for row in value["cards"] if row["card_id"] == 26000000
        )
        self.assertEqual(knight["allowed_form_flags"], [0, 1, 2, 3])
        giant = next(
            row for row in value["cards"] if row["card_id"] == 26000003
        )
        self.assertEqual(giant["allowed_form_flags"], [0, 2])

    def test_tower_and_ability_sources_are_programmatic(self) -> None:
        value = build_native_ingest_contract()
        towers = {
            item["slug"]: item["support_card_id"]
            for item in value["tower_troops"]
        }
        self.assertEqual(towers["tower-princess"], 159000000)
        self.assertEqual(towers["cannoneer"], 159000001)
        self.assertEqual(towers["dagger-duchess"], 159000002)
        self.assertEqual(towers["royal-chef"], 159000004)
        abilities = set(value["ability_source_tokens"])
        self.assertIn("goblinstein", abilities)
        self.assertIn("knight-hero", abilities)
        self.assertNotIn("knight", abilities)
        self.assertNotIn("knight-ev1", abilities)

    def test_roundtrip_reader_and_fail_closed_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            result = write_native_ingest_contract(path)
            contract = load_native_ingest_contract(path)
            self.assertEqual(contract.file_sha256, result["file_sha256"])
            self.assertEqual(contract.validate_card_token("knight-ev1"), ())
            self.assertEqual(contract.validate_tower_troop("royal-chef"), ())
            self.assertEqual(contract.validate_game_mode(72000006), ())
            self.assertEqual(contract.validate_game_mode(72000450), ())
            self.assertEqual(
                contract.execution_game_mode_for_source(72000464), 72000006
            )
            self.assertEqual(
                contract.validate_execution_game_mode(
                    72000450,
                    72000006,
                    NATIVE_EXECUTION_GAME_MODE_PROVENANCE,
                ),
                (),
            )
            self.assertEqual(
                {issue.code for issue in contract.validate_execution_game_mode(
                    72000450, 72000450, "guessed"
                )},
                {
                    "native_execution_game_mode_mismatch",
                    "native_execution_game_mode_provenance_invalid",
                },
            )
            self.assertEqual(contract.king_tower_max_hp_by_level[16], 7728)
            self.assertEqual(
                contract.validate_king_tower_level_evidence(
                    king_tower_level=16,
                    provenance=(
                        "ranked_template_cap16_and_tower_troop_level16_v1"
                    ),
                    tower_troop_level=16,
                    final_king_hp=1,
                ),
                (),
            )
            self.assertEqual(
                contract.validate_king_tower_level_evidence(
                    king_tower_level=16,
                    provenance="ranked_template_cap16_and_full_king_hp_v1",
                    tower_troop_level=15,
                    final_king_hp=7728,
                ),
                (),
            )
            self.assertEqual(
                contract.validate_king_tower_level_evidence(
                    king_tower_level=16,
                    provenance="ranked_template_cap16_and_full_king_hp_v1",
                    tower_troop_level=15,
                    final_king_hp=7000,
                )[0].code,
                "king_tower_level_exact_evidence_missing",
            )
            self.assertEqual(
                contract.validate_ability_source(
                    ["goblinstein"], observed_ability_events=1
                ),
                (),
            )

            issues = validate_ingest_metadata(
                contract,
                deck_tokens=("party-hut", "giant-ev1"),
                tower_troop="unknown-tower",
                numeric_game_mode_id=72000007,
                native_execution_game_mode_id=72000007,
                native_execution_game_mode_provenance="untrusted",
                observed_ability_events=1,
            )
            self.assertEqual(
                {item.code for item in issues},
                {
                    "native_card_mapping_missing",
                    "native_form_mapping_missing",
                    "native_tower_troop_mapping_missing",
                    "numeric_game_mode_not_allowed",
                    "native_ability_source_mapping_missing",
                },
            )

    def test_reader_rejects_tampering_even_without_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            write_native_ingest_contract(path)
            value = json.loads(path.read_text(encoding="utf-8"))
            value["source_numeric_game_mode_ids"].append(72000007)
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(
                NativeIngestContractError, "canonical SHA-256 mismatch"
            ):
                load_native_ingest_contract(path, verify_sidecar=False)


if __name__ == "__main__":
    unittest.main()
