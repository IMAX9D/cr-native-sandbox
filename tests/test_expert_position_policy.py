from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from expert_v1.training_v1.dataset import NativeExpertSequenceDataset, collate_sequences
from expert_v1.training_v1.losses import behaviour_cloning_loss
from expert_v1.training_v1.model import ExpertPolicyConfig, RecurrentExpertPolicy
from expert_v1.training_v1.smoke_data import create_smoke_dataset
from expert_v1.training_v1 import train
from expert_v1.training_v1.fork_position_run import create_position_fork, evaluate_checkpoint
from expert_v1.training_v1.fork_run import assert_tensors_equal
from scripts.experiment_expert_position_stability import forward_position_variant


class StablePositionPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset_root = create_smoke_dataset(self.root / "dataset")
        self.dataset = NativeExpertSequenceDataset(self.dataset_root, split="train", sequence_length=8, burn_in=2)
        d = self.dataset.manifest["dimensions"]
        self.config = ExpertPolicyConfig(grid_channels=d["grid_channels"], public_scalar_size=d["public_scalar_size"],
            card_vocab_size=d["card_vocab_size"], ability_vocab_size=d["ability_vocab_size"],
            max_ability_slots=d["max_ability_slots"], hidden_size=16, card_embedding_size=8, spatial_size=8)
        self.batch = collate_sequences([self.dataset[0], self.dataset[1]])
        torch.manual_seed(42)

    def tearDown(self):
        self.dataset.close()
        self.temp.cleanup()

    def test_legacy_config_and_signature_stay_unchanged(self):
        legacy = self.config.to_dict()
        self.assertNotIn("position_head_fp32", legacy)
        self.assertNotIn("position_logit_softcap", legacy)
        self.assertEqual(ExpertPolicyConfig(**legacy).to_dict(), legacy)
        args = train.build_parser().parse_args([])
        first, payload = train._run_signature(args, dataset_manifest_sha256="x", observation_mode=self.config.observation_mode)
        args.stop_at_step = 100
        args.stop_after_epoch = 1
        self.assertEqual(first, train._run_signature(args, dataset_manifest_sha256="x", observation_mode=self.config.observation_mode)[0])
        self.assertNotIn("position_head_fp32", payload)
        args.position_head_fp32 = True
        args.position_logit_softcap = 20.
        self.assertNotEqual(first, train._run_signature(args, dataset_manifest_sha256="x", observation_mode=self.config.observation_mode)[0])

    def test_invalid_policy_configuration_rejected(self):
        for cap in (0., -1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                replace(self.config, position_head_fp32=True, position_logit_softcap=cap)
        with self.assertRaises(ValueError):
            replace(self.config, position_logit_softcap=20.)

    def test_integrated_sparse_matches_experiment_and_dense_gradients(self):
        reference = RecurrentExpertPolicy(self.config)
        sparse = deepcopy(reference)
        dense = deepcopy(reference)
        stable_config = replace(self.config, position_head_fp32=True, position_logit_softcap=20.)
        sparse.config = dense.config = stable_config
        ref_out = forward_position_variant(reference, self.batch, "fp32_softcap20")
        sparse_out = sparse.forward_batch(self.batch, supervised_positions=True)
        dense_out = dense.forward_batch(self.batch)
        selected = self.batch["loss_mask"] & self.batch["position_label_mask"]
        b, t = selected.nonzero(as_tuple=True)
        cards = self.batch["card_slot"][b, t]
        for output in (sparse_out, dense_out):
            torch.testing.assert_close(output.position_logits[b, t, cards], ref_out.position_logits[b, t, cards], atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(output.card_logits, ref_out.card_logits, atol=0, rtol=0)
            torch.testing.assert_close(output.ability_position_logits, ref_out.ability_position_logits, atol=0, rtol=0)
        self.assertLessEqual(float(dense_out.position_logits.detach().abs().max()), 20.)
        losses = [behaviour_cloning_loss(o, self.batch, stable_config)[0] for o in (ref_out, sparse_out, dense_out)]
        for loss in losses:
            loss.backward()
        for name, parameter in reference.named_parameters():
            for model in (sparse, dense):
                other = dict(model.named_parameters())[name]
                self.assertEqual(parameter.grad is None, other.grad is None, name)
                if parameter.grad is not None:
                    torch.testing.assert_close(parameter.grad, other.grad, atol=5e-6, rtol=2e-4, msg=name)

    def test_inference_is_label_blind_and_half_weights_supported(self):
        config = replace(self.config, position_head_fp32=True, position_logit_softcap=20.)
        model = RecurrentExpertPolicy(config)
        first = model.forward_batch(self.batch).position_logits
        changed = dict(self.batch)
        changed["position_label_mask"] = ~self.batch["position_label_mask"]
        changed["card_slot"] = (self.batch["card_slot"] + 1) % 4
        second = model.forward_batch(changed).position_logits
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        model.half()
        q = model._query_fp32(torch.randn(2, self.config.hidden_size + self.config.card_embedding_size).half())
        c = model._cells_fp32(torch.randn(2, self.config.spatial_size, 32, 18).half())
        self.assertEqual(q.dtype, torch.float32)
        self.assertEqual(c.dtype, torch.float32)

    def test_control_identity_idempotency_and_scheduling(self):
        root = self.root / "run"
        request = {"request_id": "test-1", "expected_run_id": "run", "at_step": 5,
                   "stop_after_save": True, "export_fp16": True}
        train._atomic_json(root / "control/checkpoint-request.json", request)
        self.assertIsNone(train._pending_checkpoint_request(root, "run", 4))
        read = train._pending_checkpoint_request(root, "run", 5)
        self.assertEqual(read["request_id"], "test-1")
        train._atomic_json(root / "control/checkpoint-response.json", {"status": "saved", "request_id": "test-1", "request_sha256": read["request_sha256"]})
        self.assertIsNone(train._pending_checkpoint_request(root, "run", 6))
        request.update(request_id="test-2", expected_run_id="wrong")
        train._atomic_json(root / "control/checkpoint-request.json", request)
        self.assertIsNone(train._pending_checkpoint_request(root, "run", 6))
        self.assertEqual(json.loads((root / "control/checkpoint-response.json").read_text())["status"], "rejected")

    def test_mid_epoch_pause_resume_preserves_updates(self):
        command = ["--smoke", "--resume", "--dataset-root", str(self.dataset_root), "--output-root", str(self.root / "runs"),
            "--epochs", "2", "--batch-size", "2", "--sequence-length", "16", "--burn-in", "4",
            "--workers", "0", "--integrity-workers", "1", "--max-eval-batches", "1",
            "--hidden-size", "16", "--card-embedding-size", "8", "--spatial-size", "8", "--seed", "123",
            "--device", "cpu", "--position-head-fp32", "--position-logit-softcap", "20"]
        with redirect_stdout(io.StringIO()):
            baseline = train.run(train.build_parser().parse_args(command + ["--run-id", "baseline"]))
            resumed = train.run(train.build_parser().parse_args(command + ["--run-id", "resumed", "--stop-at-step", "2"]))
            paused = torch.load(resumed / "checkpoints/latest.pt", weights_only=False)
            self.assertEqual(paused["global_step"], 2)
            self.assertTrue(paused["model_config"]["position_head_fp32"])
            self.assertTrue((resumed / "checkpoints/manual/stop-at-2/weights-2-fp16.pt").is_file())
            train.run(train.build_parser().parse_args(command + ["--run-id", "resumed", "--resume"]))
        a = torch.load(baseline / "checkpoints/latest.pt", weights_only=False)
        b = torch.load(resumed / "checkpoints/latest.pt", weights_only=False)
        self.assertEqual(a["global_step"], b["global_step"])
        for name in a["model_state"]:
            torch.testing.assert_close(a["model_state"][name], b["model_state"][name], atol=1e-7, rtol=1e-6, msg=name)
        self.assertTrue(torch.equal(a["rng"]["train_loader_generator"], b["rng"]["train_loader_generator"]))

    def test_epoch_pause_follows_validation_and_exports_and_resumes(self):
        command = ["--smoke", "--resume", "--dataset-root", str(self.dataset_root),
            "--output-root", str(self.root / "runs"), "--epochs", "2", "--batch-size", "2",
            "--sequence-length", "16", "--burn-in", "4", "--workers", "0",
            "--integrity-workers", "1", "--hidden-size", "16", "--card-embedding-size", "8",
            "--spatial-size", "8", "--seed", "123", "--device", "cpu",
            "--position-head-fp32", "--position-logit-softcap", "20"]
        with redirect_stdout(io.StringIO()):
            baseline = train.run(train.build_parser().parse_args(command + ["--run-id", "full"]))
            root = train.run(train.build_parser().parse_args(command + ["--run-id", "boundary", "--stop-after-epoch", "1"]))
            saved = torch.load(root / "checkpoints/latest.pt", weights_only=False)
            self.assertTrue(saved["epoch_complete"])
            self.assertEqual(saved["epoch"], 1)
            self.assertTrue(torch.isfinite(torch.tensor(saved["validation_metrics"]["loss"])))
            self.assertTrue((root / "checkpoints/epochs/epoch-001.pt").is_file())
            self.assertTrue((root / "exports/epochs/epoch-001-fp16.pt").is_file())
            self.assertFalse((root / "checkpoints/epochs/epoch-002.pt").exists())
            progress = json.loads((root / "training-progress.json").read_text())
            self.assertEqual(progress["reason"], "stop_after_epoch")
            train.run(train.build_parser().parse_args(command + ["--run-id", "boundary", "--stop-after-epoch", "1"]))
            self.assertEqual(torch.load(root / "checkpoints/latest.pt", weights_only=False)["global_step"], saved["global_step"])
            train.run(train.build_parser().parse_args(command + ["--run-id", "boundary"]))
        a = torch.load(baseline / "checkpoints/latest.pt", weights_only=False)
        b = torch.load(root / "checkpoints/latest.pt", weights_only=False)
        self.assertEqual(a["global_step"], b["global_step"])
        for name in a["model_state"]:
            torch.testing.assert_close(a["model_state"][name], b["model_state"][name], atol=1e-7, rtol=1e-6)
        self.assertTrue(torch.equal(a["rng"]["train_loader_generator"], b["rng"]["train_loader_generator"]))

    def test_policy_fork_preserves_state_and_rebuilds_validation(self):
        command = ["--smoke", "--resume", "--dataset-root", str(self.dataset_root),
            "--output-root", str(self.root / "runs"), "--run-id", "source", "--epochs", "2",
            "--batch-size", "2", "--sequence-length", "16", "--burn-in", "4", "--workers", "0",
            "--integrity-workers", "1", "--hidden-size", "16", "--card-embedding-size", "8",
            "--spatial-size", "8", "--seed", "321", "--device", "cpu", "--stop-at-step", "2"]
        with redirect_stdout(io.StringIO()):
            source = train.run(train.build_parser().parse_args(command))
            receipt = create_position_fork(source, source / "checkpoints/latest.pt", self.root / "runs", "stable", 2)
            stable = Path(receipt["run_root"])
            self.assertFalse((stable / "checkpoints/best.pt").exists())
            result = evaluate_checkpoint(stable, initialize=True, device_name="cpu")
        self.assertTrue(result["full_validation"])
        self.assertEqual(result["global_step"], 2)
        original = torch.load(source / "checkpoints/latest.pt", weights_only=False)
        current = torch.load(stable / "checkpoints/latest.pt", weights_only=False)
        cfg = {**original["model_config"], "position_head_fp32": True, "position_logit_softcap": 20.}
        assert_tensors_equal(original, current, expected_model_config=cfg)
        self.assertEqual(current["best_validation_loss"], result["validation"]["loss"])
        train._load_certified_checkpoints(stable, torch.device("cpu"))
        with self.assertRaisesRegex(RuntimeError, "already installed"):
            evaluate_checkpoint(stable, initialize=True, device_name="cpu")


if __name__ == "__main__":
    unittest.main()
