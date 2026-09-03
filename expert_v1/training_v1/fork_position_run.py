"""Audited opt-in migration to FP32/softcap position scoring, with fresh validation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import torch

from . import train
from .fork_run import args_from_manifest, assert_tensors_equal
from .model import ExpertPolicyConfig, RecurrentExpertPolicy, configure_position_precision
from .schema import sha256_file


def create_position_fork(source_run: Path, checkpoint_path: Path, output_root: Path,
                         run_id: str, expected_step: int, softcap: float = 20.) -> dict:
    manifest = json.loads((source_run / "manifest.json").read_text(encoding="utf-8-sig"))
    args = args_from_manifest(manifest)
    signature, _ = train._run_signature(args, dataset_manifest_sha256=manifest["dataset_manifest_sha256"],
                                       observation_mode=manifest["training"]["observation_mode"])
    identity, _ = train._optimizer_identity(args, run_id=args.run_id, model_config=manifest["model"])
    if signature != manifest["run_signature_sha256"] or identity != manifest["optimizer"]["identity_sha256"]:
        raise RuntimeError("source runtime contract does not match its authenticated manifest")
    source = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    train._certify_checkpoint(checkpoint_path, source, dataset_manifest_sha256=manifest["dataset_manifest_sha256"],
        run_signature_sha256=signature, model_config=manifest["model"], run_id=args.run_id,
        optimizer_identity_sha256=identity)
    if source["global_step"] != expected_step:
        raise RuntimeError("source step differs from the requested migration point")
    if Path(run_id).name != run_id or "/" in run_id or "\\" in run_id or run_id in (".", "..", args.run_id):
        raise ValueError("a distinct single-component run id is required")
    if manifest["model"].get("position_head_fp32", False):
        raise ValueError("source already uses an FP32 head; this migration only enables the policy once")
    config = ExpertPolicyConfig(**{**manifest["model"], "position_head_fp32": True,
                                  "position_logit_softcap": float(softcap)})
    args.run_id = run_id
    args.position_head_fp32 = True
    args.position_logit_softcap = float(softcap)
    args.output_root = output_root.resolve()
    new_signature, settings = train._run_signature(args, dataset_manifest_sha256=manifest["dataset_manifest_sha256"],
                                                  observation_mode=manifest["training"]["observation_mode"])
    changes = {key for key in settings if settings[key] != manifest["training"].get(key)}
    if changes != {"position_head_fp32", "position_logit_softcap"}:
        raise RuntimeError(f"unexpected training-setting changes: {changes}")
    new_identity, optimizer_contract = train._optimizer_identity(args, run_id=run_id, model_config=config.to_dict())
    destination = args.output_root / run_id
    destination.mkdir(parents=True, exist_ok=False)
    lineage = {"source_run": str(source_run.resolve()), "source_checkpoint": str(checkpoint_path.resolve()),
        "source_checkpoint_sha256": sha256_file(checkpoint_path), "source_step": expected_step,
        "changed_training_fields": sorted(changes), "weight_tensors_preserved": True,
        "optimizer_moments_preserved": True, "rng_preserved": True,
        "validation_policy": "recompute initial best on full validation under the new scoring policy"}
    new_manifest = {**manifest, "run_id": run_id, "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": config.to_dict(), "training": settings, "run_signature_sha256": new_signature,
        "optimizer": {**optimizer_contract, "identity_sha256": new_identity}, "fork_lineage": lineage}
    train._atomic_json(destination / "manifest.json", new_manifest)
    target = {**source, "run_id": run_id, "model_config": config.to_dict(), "run_signature_sha256": new_signature,
        "optimizer_identity_sha256": new_identity, "checkpoint_role": "latest", "is_best": False,
        "best_validation_loss": float("inf"), "epochs_without_improvement": 0,
        "training_metrics": {}, "validation_metrics": {}, "position_policy_origin": lineage}
    train._atomic_torch(destination / "checkpoints/latest.pt", target)
    restored = torch.load(destination / "checkpoints/latest.pt", map_location="cpu", weights_only=False, mmap=True)
    assert_tensors_equal(source, restored, expected_model_config=config.to_dict())
    train._certify_checkpoint(destination / "checkpoints/latest.pt", restored,
        dataset_manifest_sha256=manifest["dataset_manifest_sha256"], run_signature_sha256=new_signature,
        model_config=config.to_dict(), run_id=run_id, optimizer_identity_sha256=new_identity)
    train._append_jsonl(destination / "events.jsonl", {"event": "position_policy_fork_created", **lineage,
        "run_id": run_id, "global_step": expected_step})
    train._atomic_json(destination / "training-progress.json", {"kind": "cr_expert_training_progress_v1",
        "status": "validation_baseline_required", "epoch": source["epoch"], "epochs": args.epochs,
        "batch": source["batch_in_epoch"], "batches": source["batches_in_epoch"], "global_step": expected_step,
        "updated_utc": datetime.now(timezone.utc).isoformat()})
    receipt = {"ok": True, "run_root": str(destination), "run_id": run_id, "resume_step": expected_step,
               "model_config": config.to_dict(), "state_verified": True, "baseline_ready": False, **lineage}
    train._atomic_json(destination / "fork-receipt.json", receipt)
    return receipt


def evaluate_checkpoint(run_root: Path, *, initialize: bool, device_name: str = "cuda") -> dict:
    manifest = json.loads((run_root / "manifest.json").read_text())
    args = args_from_manifest(manifest)
    if args.max_eval_batches:
        raise ValueError("policy validation baseline requires untruncated full validation")
    with train.TrainingInstanceLock(run_root.parent / ".expert-training-v1.lock", run_id=manifest["run_id"] + "-validation"):
        latest = run_root / "checkpoints/latest.pt"
        checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
        train._certify_checkpoint(latest, checkpoint, dataset_manifest_sha256=manifest["dataset_manifest_sha256"],
            run_signature_sha256=manifest["run_signature_sha256"], model_config=manifest["model"],
            run_id=manifest["run_id"], optimizer_identity_sha256=manifest["optimizer"]["identity_sha256"])
        if initialize and (run_root / "checkpoints/best.pt").exists():
            raise RuntimeError("initial validation already installed; do not reset the best baseline")
        if sha256_file(Path(manifest["dataset_root"]) / "manifest.json") != manifest["dataset_manifest_sha256"]:
            raise RuntimeError("dataset manifest changed")
        device = torch.device(device_name)
        model = RecurrentExpertPolicy(ExpertPolicyConfig(**manifest["model"]))
        configure_position_precision(model.config)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.to(device)
        normalizer = train.DatasetPrecomputedNormalizer(model.config.public_scalar_size)
        normalizer.load_state_dict(checkpoint["normalizer_state"])
        loader = train._loader(Path(manifest["dataset_root"]), args.validation_split, args, shuffle=False)
        total = len(loader)
        began = time.perf_counter()
        def progress(done: int, partial: dict[str, float]):
            if done % 20 and done != total:
                return
            value = {"phase": "full_validation", "global_step": checkpoint["global_step"], "completed": done,
                "total": total, "percent": done * 100 / total, "elapsed_seconds": time.perf_counter() - began,
                "partial_loss": partial["loss"]}
            train._atomic_json(run_root / "validation-progress.json", value)
            print(json.dumps(value), flush=True)
        metrics = train._evaluate(model, loader, device, maximum_batches=0, normalizer=normalizer,
                                  precision=getattr(args, "precision", "fp32"), progress_callback=progress)
        if any(not math.isfinite(float(v)) for v in metrics.values()):
            raise FloatingPointError("nonfinite full validation metrics")
        score = float(metrics["loss"])
        improved = initialize or score < float(checkpoint["best_validation_loss"]) - args.minimum_delta
        checkpoint["validation_metrics"] = metrics
        checkpoint["is_best"] = improved
        if improved:
            checkpoint["best_validation_loss"] = score
            checkpoint["epochs_without_improvement"] = 0
            train._atomic_torch(run_root / "checkpoints/best.pt", {**checkpoint, "checkpoint_role": "best"})
        train._atomic_torch(latest, {**checkpoint, "checkpoint_role": "latest"})
        result = {"event": "checkpoint_validation_complete", "run_id": manifest["run_id"],
            "global_step": checkpoint["global_step"], "initial_baseline": initialize, "full_validation": True,
            "metric_aggregation_version": train.MetricAccumulator.VERSION,
            "validation": metrics, "best_updated": improved, "validation_batches": total,
            "validation_windows": len(loader.dataset), "wall_seconds": time.perf_counter() - began}
        train._atomic_json(run_root / f"validation-step-{checkpoint['global_step']}.json", result)
        if initialize:
            train._atomic_json(run_root / "initial-validation.json", result)
            train._atomic_json(run_root / "training-progress.json", {"kind": "cr_expert_training_progress_v1",
                "status": "ready_to_resume", "epoch": checkpoint["epoch"], "epochs": args.epochs,
                "batch": checkpoint["batch_in_epoch"], "batches": checkpoint["batches_in_epoch"],
                "global_step": checkpoint["global_step"], "learning_rate": args.learning_rate,
                "updated_utc": datetime.now(timezone.utc).isoformat()})
        train._append_jsonl(run_root / "events.jsonl", result)
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--source-run", type=Path, required=True)
    create.add_argument("--checkpoint", type=Path, required=True)
    create.add_argument("--output-root", type=Path, required=True)
    create.add_argument("--run-id", required=True)
    create.add_argument("--expected-step", type=int, required=True)
    create.add_argument("--softcap", type=float, default=20.)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--run-root", type=Path, required=True)
    evaluate.add_argument("--initialize", action="store_true")
    evaluate.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command == "create":
        result = create_position_fork(args.source_run, args.checkpoint, args.output_root, args.run_id, args.expected_step, args.softcap)
    else:
        result = evaluate_checkpoint(args.run_root, initialize=args.initialize, device_name=args.device)
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
