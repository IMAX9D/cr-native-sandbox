from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_expert_selfplay_stage1_stream import (
    Journal,
    parse_ports,
    prune_payloads,
    split_ports,
    validate_update,
)


class Stage1StreamTests(unittest.TestCase):
    def test_explicit_ports_split_into_equal_lanes(self) -> None:
        ports = parse_ports("39031-39078")
        groups = split_ports(ports, 6)
        self.assertEqual(len(groups), 6)
        self.assertTrue(all(len(group) == 8 for group in groups))
        self.assertEqual([port for group in groups for port in group], ports)

    def test_pruning_touches_only_old_committed_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            completed = []
            for index in range(4):
                collection = root / "collections" / f"c{index}"
                shard = collection / "rollouts" / "shard-1"
                shard.mkdir(parents=True)
                (shard / "rollout.pt").write_bytes(b"rollout")
                (collection / "collection-result.json").write_text(json.dumps({
                    "status": "collected", "ledger_state": "CLOSED",
                    "shard": str(shard),
                }), encoding="utf-8")
                update = root / "updates" / f"u{index}"
                checkpoint_dir = update / "checkpoints"
                checkpoint_dir.mkdir(parents=True)
                checkpoint = checkpoint_dir / f"checkpoint-{index}.pt"
                checkpoint.write_bytes(b"checkpoint")
                (update / "result.json").write_text(json.dumps({
                    "status": "completed", "actor_unchanged": True,
                    "ledger_states": ["COMMITTED"], "metrics": {"loss": 1.0},
                    "fresh_validation_before_update": {"loss": 1.0},
                    "checkpoint": str(checkpoint),
                }), encoding="utf-8")
                completed.append({
                    "update": index + 1,
                    "collection_run": str(collection),
                    "update_run": str(update),
                })

            prune_payloads(
                run_root=root,
                completed=completed,
                retain_rollouts=2,
                retain_checkpoints=3,
                journal=Journal(root),
            )

            self.assertFalse((root / "collections/c0/rollouts/shard-1/rollout.pt").exists())
            self.assertFalse((root / "collections/c1/rollouts/shard-1/rollout.pt").exists())
            self.assertTrue((root / "collections/c2/rollouts/shard-1/rollout.pt").exists())
            self.assertFalse((root / "updates/u0/checkpoints/checkpoint-0.pt").exists())
            self.assertTrue((root / "updates/u1/checkpoints/checkpoint-1.pt").exists())
            result, checkpoint = validate_update(root / "updates/u3")
            self.assertEqual(result["status"], "completed")
            self.assertTrue(checkpoint.is_file())


if __name__ == "__main__":
    unittest.main()
