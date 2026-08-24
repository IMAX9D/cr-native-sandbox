from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from selfplay_v2.migrate import ACTOR_PREFIXES, BACKBONE_PREFIXES, initialize_model
from selfplay_v2.model import ContinuousRatePolicyValueNet
from training.model import RecurrentPolicyValueNet
from training.run_contract import CHECKPOINT_KIND, state_dict_digest


class V2MigrationTests(unittest.TestCase):
    def test_backbone_only_never_copies_actor(self):
        torch.manual_seed(1)
        parent = RecurrentPolicyValueNet(hidden_size=16)
        checkpoint = {
            "kind": CHECKPOINT_KIND,
            "model": parent.state_dict(),
            "current_model_digest": state_dict_digest(parent.state_dict()),
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "p010.pt"
            torch.save(checkpoint, path)
            torch.manual_seed(2)
            model = ContinuousRatePolicyValueNet(
                hidden_size=16, lambda_initial=0.2
            )
            actor_before = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
                if name.startswith(ACTOR_PREFIXES)
            }
            report = initialize_model(
                model, mode="backbone_only", parent_checkpoint=path
            )
            self.assertTrue(report["copied_tensors"])
            self.assertEqual(report["actor_tensors_copied"], [])
            for name, value in model.state_dict().items():
                if name.startswith(BACKBONE_PREFIXES):
                    self.assertTrue(torch.equal(value.cpu(), parent.state_dict()[name]))
                if name.startswith(ACTOR_PREFIXES):
                    self.assertTrue(torch.equal(value, actor_before[name]))

    def test_scratch_refuses_parent(self):
        model = ContinuousRatePolicyValueNet(hidden_size=16)
        with self.assertRaises(ValueError):
            initialize_model(
                model,
                mode="scratch",
                parent_checkpoint=Path("unexpected.pt"),
            )


if __name__ == "__main__":
    unittest.main()
