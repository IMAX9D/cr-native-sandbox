from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from policy_v1.data import Windows, collate, prepare
from policy_v1.loss import bc_loss
from policy_v1.model import Policy, PolicyConfig
from policy_v1.smoke import create_fixture


class PolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = create_fixture(self.root / "data")
        prepare(self.data, self.root / "cache", allow_smoke=True, verify_hashes=True)
        self.ds = Windows(
            self.data,
            self.root / "cache",
            "train",
            targets=4,
            frame_window=8,
            event_window=8,
        )
        self.model = Policy(
            PolicyConfig(
                12, 4, width=32, heads=4, layers=2, frame_window=8, event_window=8
            )
        ).eval()

    def tearDown(self):
        self.temp.cleanup()

    def test_event_rotation_pairing_and_exclusion(self):
        r = self.ds.records[0]
        with np.load(self.root / "cache" / r["events"]) as f:
            ev = f["events"]
            off = f["offsets"]
            a, b = ev[off[0] : off[1]], ev[off[1] : off[2]]
            np.testing.assert_array_equal(a[:, 0], b[:, 0])
            np.testing.assert_array_equal(a[:, 1], b[:, 1] ^ 1)
            np.testing.assert_array_equal(a[:, 4], 575 - b[:, 4])
        first = self.ds[0]
        self.assertTrue((first["event_ticks"] < first["frame_ticks"][-1]).all())
        # A newly deployed opposing unit must not reveal its source player's hand.
        self.assertNotIn("opponent_hand", first)

    @torch.no_grad()
    def test_no_future_frame_or_event_leakage(self):
        b = collate([self.ds[2]])
        # Inject a future event into the context deliberately to test model masks,
        # independent of the loader's exclusion at the last frame.
        for k in (
            "event_ticks",
            "event_side",
            "event_kind",
            "event_card",
            "event_ability",
            "event_position",
            "event_mask",
        ):
            b[k] = torch.zeros(
                (1, 2), dtype=torch.bool if k == "event_mask" else torch.long
            )
        b["event_mask"][:] = True
        b["event_ticks"][0] = torch.tensor([101, 110])
        b["event_card"][:] = 1
        baseline = self.model(b)
        changed = {k: v.clone() for k, v in b.items()}
        changed["event_card"][0, 1] = 3
        changed["entity_tokens"][:, 6:] = 4
        changed["public_scalars"][:, 6:, 1] = 0.1
        other = self.model(changed)
        for key in ("timing", "card", "position"):
            torch.testing.assert_close(
                baseline[key][:, :6], other[key][:, :6], rtol=1e-5, atol=1e-6
            )
        # An event at the same tick is also hidden from that tick's decision.
        changed = {k: v.clone() for k, v in b.items()}
        changed["event_card"][0, 1] = 4
        same = self.model(changed)
        mask = b["frame_ticks"] <= 110
        torch.testing.assert_close(
            baseline["timing"][mask], same["timing"][mask], rtol=1e-5, atol=1e-6
        )

    @torch.no_grad()
    def test_entity_order_invariance_and_identity_position_sensitivity(self):
        b = collate([self.ds[1]])
        base = self.model(b)
        perm = {k: v.clone() for k, v in b.items()}
        for k in (
            "entity_tokens",
            "entity_positions",
            "entity_relations",
            "entity_numeric",
            "entity_mask",
        ):
            perm[k] = perm[k].flip(2)
        result = self.model(perm)
        torch.testing.assert_close(base["card"], result["card"], rtol=1e-5, atol=1e-6)
        perm["entity_positions"] = perm["entity_positions"].flip(2)
        self.assertGreater(
            float((base["card"] - self.model(perm)["card"]).abs().max()), 1e-6
        )

    @torch.no_grad()
    def test_empty_scene_events_padding_and_batch_invariance(self):
        item = self.ds[0]
        for k in (
            "entity_tokens",
            "entity_positions",
            "entity_relations",
            "entity_numeric",
            "entity_mask",
        ):
            item[k].zero_()
        for k in list(item):
            if k.startswith("event_"):
                item[k] = item[k][:0]
        one = self.model(collate([item]))
        many = self.model(collate([item, self.ds[3]]))
        self.assertTrue(torch.isfinite(many["timing"]).all())
        torch.testing.assert_close(
            one["timing"][0],
            many["timing"][0, : len(item["frame_ticks"])],
            rtol=1e-5,
            atol=1e-6,
        )

    def test_loss_all_heads_backward_and_label_masks(self):
        batch = collate([self.ds[0], self.ds[len(self.ds) // 2]])
        out = self.model(batch)
        loss, metrics = bc_loss(out, batch)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(metrics["card_count"], 0)
        self.assertGreater(metrics["ability_count"], 0)
        for name, p in self.model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
        empty = {k: v.clone() for k, v in batch.items()}
        empty["loss_mask"].zero_()
        zero, stats = bc_loss(self.model(empty), empty)
        self.assertEqual(float(zero), 0)
        self.assertEqual(stats["timing_count"], 0)
        bad = {k: v.clone() for k, v in batch.items()}
        bad["card_mask"].zero_()
        with self.assertRaisesRegex(ValueError, "illegal supervised card"):
            bc_loss(out, bad)

    def test_source_or_cache_mutation_rejected(self):
        r = self.ds.records[0]
        with (self.root / "cache" / r["events"]).open("ab") as f:
            f.write(b"changed")
        with self.assertRaisesRegex(ValueError, "cache changed"):
            Windows(self.data, self.root / "cache", "train")

    def test_saturated_time_feature_keeps_event_order(self):
        for split in ("train", "validation"):
            p = self.data / "shards" / (split + "-00000") / "public_scalars.npy"
            values = np.load(p)
            values[:, 0] = np.tile(np.minimum(np.arange(5990, 6014), 6000) / 6000.0, 2)
            np.save(p, values)
        prepare(self.data, self.root / "late-cache", allow_smoke=True)
        ds = Windows(
            self.data, self.root / "late-cache", "train", targets=4, frame_window=8
        )
        item = ds[5]
        self.assertEqual(int(item["frame_ticks"][-1]), 6013)
        self.assertTrue((item["frame_ticks"].diff() == 1).all())

    def test_training_split_leakage_rejected(self):
        manifest = json.loads((self.data / "manifest.json").read_text())
        manifest["splits"]["validation"] = manifest["splits"]["train"]
        (self.data / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "battle leakage"):
            prepare(self.data, self.root / "bad-cache", allow_smoke=True)


if __name__ == "__main__":
    unittest.main()
