from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from expert_v1.freeze_schema5_manifest import freeze


ROOT = Path(__file__).resolve().parents[1]


class FreezeSchema5ManifestTests(unittest.TestCase):
    def _fixture(self, directory: Path) -> tuple[Path, Path, Path]:
        root = directory / "authoritative"
        battle_path = root / "raw" / "battles" / "SC" / "SCHEMA5FIXTURE.json"
        battle_path.parent.mkdir(parents=True)
        value = json.loads((
            ROOT / "tests" / "fixtures" / "expert_schema5_authoritative.json"
        ).read_text(encoding="utf-8"))
        value["authoritative_native_contract"] = {
            "game_version": "15.535.29",
            "contract_sha256": "a" * 64,
            "contract_file_sha256": "b" * 64,
        }
        value["deck_metadata"]["source_list_url"] = (
            "https://royaleapi.com/player/TEAM/battles"
        )
        value["team_tags"] = ["TEAM"]
        value["opponent_tags"] = ["OPPO"]
        battle_path.write_text(json.dumps(value), encoding="utf-8")
        index = root / "index.jsonl"
        index.write_text(json.dumps({
            "kind": "authoritative_battle",
            "schema_version": 5,
            "battle_tag": "SCHEMA5FIXTURE",
            "saved_path": str(battle_path),
        }) + "\n", encoding="utf-8")
        database = directory / "progress.sqlite3"
        connection = sqlite3.connect(database)
        connection.execute(
            "CREATE TABLE authoritative_results("
            "battle_tag TEXT,status TEXT,tier TEXT,saved_path TEXT,"
            "contract_sha256 TEXT)"
        )
        connection.execute(
            "INSERT INTO authoritative_results VALUES(?,?,?,?,?)",
            (
                "SCHEMA5FIXTURE", "accepted", "native_static_v2",
                str(battle_path), "a" * 64,
            ),
        )
        connection.commit()
        connection.close()
        return database, root, battle_path

    def test_freeze_is_content_addressed_and_production_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            database, root, _ = self._fixture(base)
            output = base / "manifest.jsonl"
            result = freeze(
                db_path=database,
                authoritative_root=root,
                output=output,
                target=1,
                allow_incomplete=False,
            )
            self.assertTrue(result["production_ready"])
            self.assertEqual(result["accepted_battles"], 1)
            row = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(row["source_schema_version"], 5)
            self.assertEqual(row["source_numeric_game_mode_id"], 72000450)
            self.assertEqual(row["native_execution_game_mode_id"], 72000006)
            self.assertEqual(len(row["source_sha256"]), 64)
            self.assertEqual(row["player_tags"], sorted(row["player_tags"]))
            self.assertEqual(len(row["source_group"]), 64)

    def test_file_tampering_is_detected_against_indexed_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            database, root, battle_path = self._fixture(base)
            value = json.loads(battle_path.read_text(encoding="utf-8"))
            value["authoritative_native_contract"]["contract_sha256"] = "c" * 64
            battle_path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "contract eligibility"):
                freeze(
                    db_path=database,
                    authoritative_root=root,
                    output=base / "manifest.jsonl",
                    target=1,
                    allow_incomplete=False,
                )


if __name__ == "__main__":
    unittest.main()
