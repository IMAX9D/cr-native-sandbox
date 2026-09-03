from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

from scripts.run_expert_selfplay_stage2_loop import run
from scripts.start_expert_selfplay_stage2 import _canary_continuation


COLLECTOR = r'''import argparse,json
from pathlib import Path
p=argparse.ArgumentParser()
for name in ("checkpoint","opponent-checkpoint","expert-manifest","ports","host","run-dir","learner-deck","opponent-deck-root","episodes","updates","policy-version","curriculum-stage","opponent-policy-id","step-ticks","max-decisions","timeout","seed","device","cpu-threads"):
 p.add_argument("--"+name)
p.add_argument("--collect-only",action="store_true")
a=p.parse_args();r=Path(a.run_dir);s=r/"rollouts"/"shard-1";s.mkdir(parents=True)
(s/"rollout.pt").write_bytes(b"rollout")
(r/"collection-result.json").write_text(json.dumps({"status":"collected","ledger_state":"CLOSED","shard":str(s)}))
'''


TRAINER = r'''import argparse,json,torch
from pathlib import Path
p=argparse.ArgumentParser()
for name in ("base-checkpoint","continuation-checkpoint","expert-manifest","run-dir","device","cpu-threads","ppo-epochs","chunk-batch-size","retain-checkpoints"):
 p.add_argument("--"+name)
p.add_argument("--shard",action="append")
a=p.parse_args();c=torch.load(a.continuation_checkpoint,weights_only=False);pre=int(c.get("policy_version",0));glob=int(c["global_update"])+1
r=Path(a.run_dir);(r/"checkpoints").mkdir(parents=True);(r/"exports").mkdir()
ck=r/"checkpoints"/f"checkpoint-{glob}.pt";ex=r/"exports"/f"actor-{pre+1}.pt"
torch.save({"kind":"cr_native_expert_selfplay_stage2_checkpoint_v1","global_update":glob,"policy_version":pre+1},ck);torch.save({"kind":"export"},ex)
(r/"result.json").write_text(json.dumps({"status":"completed","guard":{"action":"accept"},"ledger_states":["COMMITTED"]*len(a.shard),"metrics":{"loss":1.0,"approx_update_kl":0.0},"checkpoint":str(ck),"behavior_export":str(ex),"policy_version":pre+1,"global_update":glob,"retry_attempt":0}))
'''


class Stage2LoopTests(unittest.TestCase):
    def test_formal_entry_requires_three_completed_canary_updates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.pt"
            behavior = root / "behavior.pt"
            checkpoint.write_bytes(b"checkpoint")
            behavior.write_bytes(b"behavior")
            progress = {
                "status": "completed",
                "completion_reason": "requested_updates_committed",
                "completed_updates": [{"metrics": {"loss": 1.0}}] * 3,
                "latest_checkpoint": str(checkpoint),
                "latest_behavior_export": str(behavior),
            }
            (root / "progress.json").write_text(json.dumps(progress), encoding="utf-8")
            (root / "local-rtx3080-attestation.json").write_text(json.dumps({
                "status": "passed",
                "device": "NVIDIA GeForce RTX 3080",
                "weights_sha256": hashlib.sha256(behavior.read_bytes()).hexdigest(),
                "finite": True,
            }), encoding="utf-8")
            self.assertEqual(
                _canary_continuation(root),
                (checkpoint.resolve(), behavior.resolve()),
            )
            progress["completed_updates"] = progress["completed_updates"][:2]
            (root / "progress.json").write_text(json.dumps(progress), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "three"):
                _canary_continuation(root)

    def test_one_command_chains_strict_policy_versions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collect = root / "collect.py"
            train = root / "train.py"
            collect.write_text(COLLECTOR, encoding="utf-8")
            train.write_text(TRAINER, encoding="utf-8")
            base = root / "base.pt"
            opponent = root / "opponent.pt"
            manifest = root / "manifest.json"
            deck = root / "deck.json"
            pool = root / "pool"
            pool.mkdir()
            for path in (base, opponent, manifest, deck):
                path.write_bytes(b"fixture")
            continuation = root / "stage1.pt"
            torch.save({
                "kind": "cr_native_expert_selfplay_checkpoint_v1",
                "global_update": 52,
            }, continuation)
            args = argparse.Namespace(
                base_checkpoint=base,
                base_opponent_checkpoint=opponent,
                initial_continuation=continuation,
                expert_manifest=manifest,
                ports="39031-39034",
                collectors=2,
                updates=2,
                run_root=root / "run",
                learner_deck=deck,
                opponent_deck_root=pool,
                host="127.0.0.1",
                step_ticks=4,
                max_decisions=100,
                timeout=2.0,
                seed=9,
                device="cpu",
                collector_cpu_threads=1,
                trainer_cpu_threads=1,
                ppo_epochs=1,
                chunk_batch_size=1,
                retain_checkpoints=3,
                retain_rollout_updates=2,
                retain_artifact_updates=3,
                minimum_free_gb=0.0,
                python=sys.executable,
                collect_script=collect,
                train_script=train,
            )

            result = run(args)

            self.assertEqual(result["status"], "completed")
            self.assertEqual(
                [row["policy_version"] for row in result["completed_updates"]],
                [1, 2],
            )
            self.assertEqual(
                [row["global_update"] for row in result["completed_updates"]],
                [53, 54],
            )
            self.assertIsNone(result["active_update"])


if __name__ == "__main__":
    unittest.main()
