import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import torch

from policy_v1.data import prepare, Windows, collate
from policy_v1.model import Policy, PolicyConfig
from policy_v1.loss import bc_loss
from policy_v1.smoke import create_fixture
from policy_v1.train import parser, run, load_checkpoint


class TrainingTests(unittest.TestCase):
    def test_resume_matches_uninterrupted_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = create_fixture(root / "data")
            prepare(data, root / "cache", allow_smoke=True)

            def train(name, steps, resume=False):
                argv = [
                    "--data",
                    str(data),
                    "--cache",
                    str(root / "cache"),
                    "--run-dir",
                    str(root / name),
                    "--allow-smoke",
                    "--device",
                    "cpu",
                    "--width",
                    "16",
                    "--heads",
                    "2",
                    "--layers",
                    "1",
                    "--frame-window",
                    "4",
                    "--event-window",
                    "4",
                    "--targets",
                    "4",
                    "--batch-size",
                    "2",
                    "--workers",
                    "0",
                    "--max-steps",
                    str(steps),
                    "--eval-batches",
                    "2",
                    "--cpu-threads",
                    "1",
                ]
                if resume:
                    argv += ["--resume", str(root / name / "last.pt")]
                with contextlib.redirect_stdout(io.StringIO()):
                    run(parser().parse_args(argv))
                return load_checkpoint(root / name / "last.pt")

            full = train("full", 4)
            train("resumed", 2)
            resumed = train("resumed", 4, True)
            self.assertEqual(resumed["step"], 4)
            for key in full["model"]:
                torch.testing.assert_close(
                    full["model"][key], resumed["model"][key], rtol=0, atol=0
                )
            self.assertEqual(full["next_batch"], resumed["next_batch"])

    def test_tiny_batch_can_learn_deployments_and_skills(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = create_fixture(root / "data")
            prepare(data, root / "cache", allow_smoke=True)
            ds = Windows(
                data, root / "cache", "train", targets=4, frame_window=4, event_window=4
            )
            batch = collate([ds[0], ds[len(ds) // 2]])
            torch.set_num_threads(1)
            torch.manual_seed(2)
            model = Policy(
                PolicyConfig(
                    12, 4, width=16, heads=2, layers=1, frame_window=4, event_window=4
                )
            )
            opt = torch.optim.Adam(model.parameters(), lr=0.003)
            initial = float(bc_loss(model(batch), batch)[0].detach())
            for _ in range(30):
                opt.zero_grad()
                loss, _ = bc_loss(model(batch), batch)
                loss.backward()
                opt.step()
            final = float(bc_loss(model(batch), batch)[0].detach())
            self.assertLess(final, initial * 0.4)


if __name__ == "__main__":
    unittest.main()
