from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from expert_v1.audit_native_eligibility import audit_one
from expert_v1.native_dataset_generator import _validate_candidate_row
from expert_v1.native_ingest_contract import (
    load_native_ingest_contract,
    write_native_ingest_contract,
)
from expert_v1.native_replay_plan import (
    ReplayPlanError,
    compile_battle,
    materialize_replay,
)
from expert_v1.native_replay_runner import (
    _observed_final_tower_hp,
    _source_final_tower_hp,
    _tower_hp_matches,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "expert_schema5_authoritative.json"


def authoritative_fixture(directory: Path) -> tuple[dict, object]:
    contract_path = directory / "native-contract.json"
    published = write_native_ingest_contract(contract_path)
    contract = load_native_ingest_contract(contract_path)
    value = json.loads(FIXTURE.read_text(encoding="utf-8"))
    value["authoritative_native_contract"] = {
        "game_version": contract.value["game_version"],
        "contract_sha256": contract.value["contract_sha256"],
        "contract_file_sha256": published["file_sha256"],
    }
    return value, contract


class ExpertSchema5NativeIngestTests(unittest.TestCase):
    def test_damaged_king_is_exact_when_tower_troop_is_level_16(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, contract = authoritative_fixture(Path(directory))
            player = source["rounds"][0]["opponent"][0]
            player["tower_troop_level"] = 16
            player["king_tower_level_provenance"] = (
                "ranked_template_cap16_and_tower_troop_level16_v1"
            )
            source["final_tower_hp"]["opponent"]["king"] = 5_000
            source["final_tower_hp"]["opponent"]["total"] = (
                5_000
                + source["final_tower_hp"]["opponent"]["princess0"]
                + source["final_tower_hp"]["opponent"]["princess1"]
            )
            plan = compile_battle(source, native_ingest_contract=contract)
            self.assertEqual(plan.sides[1].king_tower_level, 16)
            self.assertEqual(
                plan.sides[1].king_tower_level_provenance,
                "ranked_template_cap16_and_tower_troop_level16_v1",
            )
            self.assertEqual(plan.sides[1].final_tower_hp.king, 5_000)  # type: ignore[union-attr]
            source_path = Path(directory) / "damaged-king.json"
            source_path.write_text(json.dumps(source), encoding="utf-8")
            audited = audit_one(
                {"source_path": str(source_path), "battle_tag": source["battle_tag"]},
                native_ingest_contract=contract,
            )
            self.assertTrue(audited["king_tower_levels_complete"])
            self.assertTrue(audited["final_tower_hp_complete"])
            self.assertTrue(audited["authoritative_native_full_candidate"])

            wrong_provenance = deepcopy(source)
            wrong_provenance["rounds"][0]["opponent"][0][
                "king_tower_level_provenance"
            ] = "ranked_template_cap16_and_full_king_hp_v1"
            with self.assertRaisesRegex(
                ReplayPlanError, "king_tower_level_provenance_invalid"
            ):
                compile_battle(
                    wrong_provenance, native_ingest_contract=contract
                )

    def test_damaged_king_without_level_16_tower_troop_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, contract = authoritative_fixture(Path(directory))
            source["final_tower_hp"]["opponent"]["king"] = 5_000
            source["final_tower_hp"]["opponent"]["total"] = (
                5_000
                + source["final_tower_hp"]["opponent"]["princess0"]
                + source["final_tower_hp"]["opponent"]["princess1"]
            )
            with self.assertRaisesRegex(
                ReplayPlanError, "king_tower_level_exact_evidence_missing"
            ):
                compile_battle(source, native_ingest_contract=contract)

    def test_plan_and_materialization_preserve_authoritative_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, contract = authoritative_fixture(Path(directory))
            plan = compile_battle(source, native_ingest_contract=contract)
            self.assertTrue(plan.native_replay_ready)
            self.assertEqual(plan.source_schema_version, 5)
            self.assertEqual(plan.numeric_game_mode_id, 72_000_450)
            self.assertEqual(plan.native_execution_game_mode_id, 72_000_006)
            self.assertEqual(
                plan.native_execution_game_mode_provenance,
                "frozen_native_ingest_contract_mode_map_v1",
            )
            self.assertEqual(plan.battle_index, 1_787_218_979)
            self.assertEqual(plan.sides[0].tower_troop_level, 13)
            self.assertEqual(plan.sides[1].tower_troop_level, 10)
            self.assertEqual(plan.sides[0].king_tower_level, 16)
            self.assertEqual(plan.sides[1].king_tower_level, 16)
            self.assertEqual(plan.sides[1].final_tower_hp.princess0, 0)  # type: ignore[union-attr]
            self.assertIn("source_exact_game_build_missing", plan.limitations)
            self.assertNotIn("source_king_tower_level_missing", plan.limitations)
            self.assertNotIn("source_numeric_game_mode_missing", plan.limitations)
            self.assertNotIn("source_tower_troop_level_missing", plan.limitations)

            template = json.loads(
                (PROJECT_ROOT / "examples" / "eight-card-bootstrap.json").read_text(
                    encoding="utf-8"
                )
            )
            replay, _ = materialize_replay(plan, template)
            self.assertEqual(replay["battle"]["gamemode"], 72_000_006)
            # Source Tower Troop levels 13 and 10 become native zero-based 12/9.
            self.assertEqual(replay["battle"]["deck0"]["sc"][0]["l"], 12)
            self.assertEqual(replay["battle"]["deck1"]["sc"][0]["l"], 9)
            # The level-16 source anchor is written through every native field.
            for side in range(2):
                self.assertEqual(replay["battle"][f"avatar{side}"]["expLevel"], 16)
                self.assertEqual(replay["battle"][f"avatar{side}"]["kt"], 16)
                self.assertEqual(replay["battle"]["hbd"][side]["kt"], 16)

    def test_schema5_tampering_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, contract = authoritative_fixture(Path(directory))
            mutations = {
                "mode": lambda value: value.__setitem__("numeric_game_mode_id", 72_000_007),
                "execution_mode": lambda value: value.__setitem__(
                    "native_execution_game_mode_id", 72_000_450
                ),
                "execution_mode_provenance": lambda value: value.__setitem__(
                    "native_execution_game_mode_provenance", "guessed_mode_map"
                ),
                "contract": lambda value: value["authoritative_native_contract"].__setitem__(
                    "contract_sha256", "0" * 64
                ),
                "tower_level": lambda value: value["rounds"][0]["team"][0].__setitem__(
                    "tower_troop_level", None
                ),
                "king_level": lambda value: value["rounds"][0]["team"][0].__setitem__(
                    "king_tower_level", None
                ),
                "king_level_provenance": lambda value: value["rounds"][0]["team"][0].__setitem__(
                    "king_tower_level_provenance", "guessed_from_deck"
                ),
                "tower_hp": lambda value: value["final_tower_hp"]["team"].__setitem__(
                    "total", 1
                ),
                "slot_mapping": lambda value: value["final_tower_hp"].__setitem__(
                    "slot_mapping_provenance", "guessed_left_right"
                ),
                "deck_card": lambda value: value["rounds"][0]["team"][0]["deck_cards"][0].__setitem__(
                    "level", 1
                ),
            }
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    changed = deepcopy(source)
                    mutate(changed)
                    with self.assertRaises(ReplayPlanError):
                        compile_battle(changed, native_ingest_contract=contract)

    def test_all_frozen_source_modes_execute_as_uncapped_standard_1v1(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, contract = authoritative_fixture(Path(directory))
            template = json.loads(
                (PROJECT_ROOT / "examples" / "eight-card-bootstrap.json").read_text(
                    encoding="utf-8"
                )
            )
            for source_mode in (72_000_006, 72_000_450, 72_000_464):
                with self.subTest(source_mode=source_mode):
                    changed = deepcopy(source)
                    changed["numeric_game_mode_id"] = source_mode
                    plan = compile_battle(
                        changed, native_ingest_contract=contract
                    )
                    self.assertEqual(plan.numeric_game_mode_id, source_mode)
                    self.assertEqual(
                        plan.native_execution_game_mode_id, 72_000_006
                    )
                    replay, _ = materialize_replay(plan, template)
                    self.assertEqual(replay["battle"]["gamemode"], 72_000_006)

    def test_terminal_hp_compares_princess_slots_as_multiset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source, contract = authoritative_fixture(Path(directory))
            plan = compile_battle(source, native_ingest_contract=contract)
            expected = _source_final_tower_hp(plan)
            self.assertIsNotNone(expected)
            episode = {
                "crown_towers": [
                    {"side": 0, "type": "King Tower", "hp": 7728},
                    {"side": 0, "type": "Princess Tower", "x": 3500, "hp": 2500},
                    {"side": 0, "type": "Princess Tower", "x": 14500, "hp": 3052},
                    {"side": 1, "type": "King Tower", "hp": 7728},
                    {"side": 1, "type": "Princess Tower", "x": 3500, "hp": 2200},
                    {"side": 1, "type": "Princess Tower", "x": 14500, "hp": 0},
                ]
            }
            observed = _observed_final_tower_hp(episode)
            self.assertIsNotNone(observed)
            self.assertTrue(_tower_hp_matches(expected, observed))  # type: ignore[arg-type]
            self.assertEqual(
                observed[0]["slot_mapping_provenance"],  # type: ignore[index]
                "native_princess_lanes_compared_as_multiset",
            )

    def test_auditor_and_generator_accept_schema5_while_schema3_stays_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, contract = authoritative_fixture(root)
            source_path = root / "battle.json"
            source_path.write_text(json.dumps(source), encoding="utf-8")
            audited = audit_one(
                {"source_path": str(source_path), "battle_tag": source["battle_tag"]},
                native_ingest_contract=contract,
            )
            self.assertTrue(audited["authoritative_native_full_candidate"])
            self.assertTrue(audited["schema5_authoritative_contract_verified"])
            candidate = {
                key: audited[key] for key in (
                    "battle_tag", "source_path", "source_sha256",
                    "source_schema_version", "duration_ticks", "deployment_actions",
                    "ability_count_reported", "ability_events_observed", "ability_log_tier",
                    "coordinate_tier", "eligibility_tier",
                    "authoritative_native_full_candidate",
                    "schema5_authoritative_contract_verified",
                    "tower_troop_levels_complete", "king_tower_levels_complete",
                    "final_tower_hp_complete",
                    "numeric_game_mode_id", "native_execution_game_mode_id",
                    "native_execution_game_mode_provenance", "battle_index",
                )
            }
            _validate_candidate_row(candidate, 1)

            legacy = dict(candidate)
            legacy["source_schema_version"] = 3
            for key in (
                "schema5_authoritative_contract_verified",
                "tower_troop_levels_complete", "king_tower_levels_complete",
                "final_tower_hp_complete",
                "numeric_game_mode_id", "native_execution_game_mode_id",
                "native_execution_game_mode_provenance", "battle_index",
            ):
                legacy.pop(key, None)
            _validate_candidate_row(legacy, 2)


if __name__ == "__main__":
    unittest.main()
