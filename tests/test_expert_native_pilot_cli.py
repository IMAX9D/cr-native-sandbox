from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.pilot_expert_native import _append_jsonl, _load_fixed_selection


class ExpertNativePilotCliTest(unittest.TestCase):
    def test_fixed_selection_is_loaded_without_rescanning(self) -> None:
        value = {
            "battle_tag": "BATTLE",
            "source_path": "C:/immutable/source.json",
            "source_sha256": "a" * 64,
            "source_schema_version": 3,
            "team_crowns": 1,
            "opponent_crowns": 0,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "selection.jsonl"
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                _append_jsonl(handle, value)
            tasks, summary = _load_fixed_selection(path, expected_episodes=1)
        self.assertEqual(tasks[0].battle_tag, "BATTLE")
        self.assertEqual(tasks[0].source_sha256, "a" * 64)
        self.assertEqual(summary["kind"], "fixed_prior_selection_v1")
        self.assertEqual(
            summary["source_selection_sha256"],
            hashlib.sha256(
                (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
                .encode("utf-8")
            ).hexdigest(),
        )

    def test_fixed_selection_count_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "selection.jsonl"
            path.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "contains 0 tasks"):
                _load_fixed_selection(path, expected_episodes=1)


if __name__ == "__main__":
    unittest.main()
