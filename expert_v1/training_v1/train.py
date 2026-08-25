"""One-command recurrent expert behaviour-cloning trainer."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import NativeExpertSequenceDataset, collate_sequences
from .losses import MetricAccumulator, behaviour_cloning_loss
from .model import ExpertPolicyConfig, RecurrentExpertPolicy
from .schema import read_manifest, sha256_file, validate_shard
from .smoke_data import create_smoke_dataset


CHECKPOINT_KIND = "cr_native_expert_bc_checkpoint_v1"


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_torch(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(dict(value), temporary)
    temporary.replace(path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=device.type == "cuda") for key, value in batch.items()}


def _loader(
    root: Path,
    split: str,
    args: argparse.Namespace,
    *,
    shuffle: bool,
) -> DataLoader[dict[str, torch.Tensor]]:
    dataset = NativeExpertSequenceDataset(
        root,
        split=split,
        sequence_length=args.sequence_length,
        burn_in=args.burn_in,
        validate=True,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        collate_fn=collate_sequences,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
        drop_last=False,
    )


def _evaluate(
    model: RecurrentExpertPolicy,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    maximum_batches: int,
) -> dict[str, float]:
    model.eval()
    accumulator = MetricAccumulator()
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if maximum_batches and index >= maximum_batches:
                break
            batch = _move(batch, device)
            output = model.forward_batch(batch)
            _loss, metrics = behaviour_cloning_loss(output, batch, model.config)
            accumulator.add(metrics)
    return accumulator.result()


def run(args: argparse.Namespace) -> Path:
    _seed(args.seed)
    dataset_root = args.dataset_root.resolve()
    if args.smoke:
        create_smoke_dataset(dataset_root, replace=True)
    manifest = read_manifest(dataset_root)
    if not args.smoke:
        if manifest.get("production_ready") is not True:
            raise RuntimeError("dataset is not marked production_ready")
        if manifest.get("native_replay_validated") is not True:
            raise RuntimeError("dataset lacks native replay validation")
        if (manifest.get("split_contract") or {}).get("player_holdout_test") is not True:
            raise RuntimeError("production dataset lacks a player-holdout test split")
        expected_source = args.expected_source_manifest.resolve()
        if not expected_source.is_file():
            raise RuntimeError(f"active accepted source manifest is missing: {expected_source}")
        source_contract = manifest.get("source_manifest") or {}
        compiled_source = Path(str(source_contract.get("path", ""))).resolve()
        if compiled_source != expected_source:
            raise RuntimeError(
                f"compiled dataset uses stale/wrong source manifest: {compiled_source}"
            )
        live_source_digest = sha256_file(expected_source)
        if source_contract.get("sha256") != live_source_digest:
            raise RuntimeError(
                "active accepted source manifest changed after dataset compilation"
            )
        gates = manifest.get("quality_gates") or {}
        required_zero = (
            "split_collisions",
            "forbidden_actor_features",
            "nonfinite_features",
            "expert_label_mask_violations",
            "native_action_rejections",
            "terminal_mismatches",
        )
        failures = {
            name: gates.get(name)
            for name in required_zero
            if gates.get(name) != 0
        }
        if failures:
            raise RuntimeError(f"dataset quality gates are not clean: {failures}")
        provenance = manifest.get("state_provenance") or {}
        if "terminal_validation_unknown" not in gates:
            raise RuntimeError("dataset does not report terminal_validation_unknown")
        if not all(
            key in provenance
            for key in ("authoritative_rows", "native_generated_unanchored_rows")
        ):
            raise RuntimeError("dataset does not declare state provenance counts")
        if (
            int(provenance.get("authoritative_rows", 0))
            + int(provenance.get("native_generated_unanchored_rows", 0))
            <= 0
        ):
            raise RuntimeError("dataset contains no state-conditioned training rows")
        unanchored_rows = int(provenance.get("native_generated_unanchored_rows", 0))
        terminal_unknown = int(gates.get("terminal_validation_unknown", 0))
        if (unanchored_rows or terminal_unknown) and not args.allow_unanchored_native_states:
            raise RuntimeError(
                "dataset contains unanchored libg-generated states; pass the explicit "
                "--allow-unanchored-native-states acknowledgement to train an approximate "
                "state-conditioned model"
            )
    shard_summary: dict[str, dict[str, int]] = {}
    for split, shards in manifest["splits"].items():
        for relative in shards:
            shard_summary[f"{split}:{relative}"] = validate_shard(
                dataset_root / relative, manifest
            )

    dimensions = manifest["dimensions"]
    config = ExpertPolicyConfig(
        grid_channels=int(dimensions["grid_channels"]),
        public_scalar_size=int(dimensions["public_scalar_size"]),
        card_vocab_size=int(dimensions["card_vocab_size"]),
        ability_vocab_size=int(dimensions["ability_vocab_size"]),
        max_ability_slots=int(dimensions["max_ability_slots"]),
        card_embedding_size=args.card_embedding_size,
        hidden_size=args.hidden_size,
        lambda_max=args.lambda_max,
        lambda_initial=args.lambda_initial,
    )
    device = _device(args.device)
    model = RecurrentExpertPolicy(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    train_loader = _loader(dataset_root, "train", args, shuffle=True)
    validation_loader = _loader(dataset_root, "validation", args, shuffle=False)
    test_loader = _loader(dataset_root, "test", args, shuffle=False)
    if not len(train_loader.dataset) or not len(validation_loader.dataset) or not len(test_loader.dataset):
        raise RuntimeError("train/validation/test must all contain at least one sequence window")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"expert-v1-{stamp}"
    run_root = args.output_root.resolve() / run_id
    if run_root.exists():
        raise FileExistsError(run_root)
    (run_root / "checkpoints").mkdir(parents=True)
    events = run_root / "events.jsonl"
    run_manifest = {
        "schema_version": 1,
        "kind": "cr_native_expert_bc_run_v1",
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "dataset_manifest_sha256": sha256_file(dataset_root / "manifest.json"),
        "source_manifest": manifest.get("source_manifest"),
        "dataset_shards": shard_summary,
        "model": config.to_dict(),
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "burn_in": args.burn_in,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip": args.gradient_clip,
            "early_stopping_patience": args.early_stopping_patience,
            "minimum_delta": args.minimum_delta,
            "seed": args.seed,
        },
        "semantics": {
            "algorithm": "supervised_behaviour_cloning",
            "reward": None,
            "ppo": False,
            "actor_information": "public_only_v1",
            "action_heads": ["timing_hazard", "action_kind", "hand_slot", "position", "ability", "ability_position"],
            "allow_unanchored_native_states": bool(args.allow_unanchored_native_states),
            "state_provenance": manifest.get("state_provenance", {}),
        },
        "device": str(device),
    }
    _atomic_json(run_root / "manifest.json", run_manifest)
    best_validation = float("inf")
    epochs_without_improvement = 0
    updates = 0
    completed_epoch = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        completed_epoch = epoch
        model.train()
        accumulator = MetricAccumulator()
        epoch_started = time.perf_counter()
        examples = 0
        for batch_index, batch in enumerate(train_loader):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            batch = _move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model.forward_batch(batch)
            loss, metrics = behaviour_cloning_loss(output, batch, config)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("expert BC loss became non-finite")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError("expert BC gradient became non-finite")
            optimizer.step()
            metrics["gradient_norm"] = float(gradient_norm.detach().item())
            accumulator.add(metrics)
            examples += int(batch["loss_mask"].sum().item())
            updates += 1
        training_metrics = accumulator.result()
        validation_metrics = _evaluate(
            model, validation_loader, device, maximum_batches=args.max_eval_batches
        )
        epoch_event = {
            "event": "epoch_complete",
            "epoch": epoch,
            "updates": updates,
            "examples": examples,
            "wall_seconds": time.perf_counter() - epoch_started,
            "training": training_metrics,
            "validation": validation_metrics,
        }
        _append_jsonl(events, epoch_event)
        checkpoint = {
            "kind": CHECKPOINT_KIND,
            "schema_version": 1,
            "epoch": epoch,
            "updates": updates,
            "model_config": config.to_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "dataset_manifest_sha256": run_manifest["dataset_manifest_sha256"],
            "training_metrics": training_metrics,
            "validation_metrics": validation_metrics,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        _atomic_torch(run_root / "checkpoints" / "latest.pt", checkpoint)
        validation_loss = float(validation_metrics.get("loss", float("inf")))
        if validation_loss < best_validation - args.minimum_delta:
            best_validation = validation_loss
            epochs_without_improvement = 0
            _atomic_torch(run_root / "checkpoints" / "best.pt", checkpoint)
        else:
            epochs_without_improvement += 1
        print(json.dumps(epoch_event, ensure_ascii=False), flush=True)
        if epochs_without_improvement >= args.early_stopping_patience:
            _append_jsonl(events, {
                "event": "early_stopping",
                "epoch": epoch,
                "best_validation_loss": best_validation,
                "patience": args.early_stopping_patience,
            })
            break
    best_path = run_root / "checkpoints" / "best.pt"
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"])
    test_metrics = _evaluate(model, test_loader, device, maximum_batches=args.max_eval_batches)
    final = {
        "event": "run_complete",
        "epochs_requested": args.epochs,
        "epochs_completed": completed_epoch,
        "updates": updates,
        "wall_seconds": time.perf_counter() - started,
        "best_validation_loss": best_validation,
        "test": test_metrics,
        "checkpoint": str(best_path),
    }
    _append_jsonl(events, final)
    _atomic_json(run_root / "result.json", final)
    print(json.dumps(final, ensure_ascii=False), flush=True)
    return run_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"D:\AI_data\cr-native-core\expert-v1\compiled\native-bc-v1"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"D:\AI_data\cr-native-core\expert-v1\runs"),
    )
    parser.add_argument(
        "--expected-source-manifest",
        type=Path,
        default=Path(
            r"D:\AI_data\cr-native-core\expert-v1\training-dataset"
            r"\version-window-20260804\accepted.jsonl"
        ),
    )
    parser.add_argument("--run-id")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--burn-in", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--card-embedding-size", type=int, default=64)
    parser.add_argument("--lambda-max", type=float, default=20.0)
    parser.add_argument("--lambda-initial", type=float, default=0.30)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--minimum-delta", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--allow-unanchored-native-states",
        action="store_true",
        help="explicitly accept scene/label drift risk when source replays lack state anchors",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.early_stopping_patience <= 0:
        raise ValueError("epochs, batch size and early stopping patience must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
