"""One-command recurrent expert behaviour-cloning trainer."""

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import time
from typing import Any, Mapping, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from .dataset import NativeExpertSequenceDataset, collate_sequences
from .losses import MetricAccumulator, behaviour_cloning_loss
from .model import ExpertPolicyConfig, RecurrentExpertPolicy, configure_position_precision
from .schema import (
    OBSERVATION_NATIVE,
    OBSERVATION_SEQUENCE,
    read_manifest,
    sha256_file,
    validate_shard,
    verify_dataset_integrity,
)
from .smoke_data import create_smoke_dataset


CHECKPOINT_KIND = "cr_native_expert_bc_checkpoint_v1"
CHECKPOINT_SCHEMA_VERSION = 4
RUN_KIND = "cr_native_expert_bc_run_v1"
RUN_SCHEMA_VERSION = 3
CHECKPOINT_REQUIRED_FIELDS = {
    "epoch",
    "step",
    "global_step",
    "model_config",
    "model_state",
    "optimizer_state",
    "scheduler_state",
    "normalizer_state",
    "rng",
    "best_validation_loss",
    "epochs_without_improvement",
    "dataset_manifest_sha256",
    "run_signature_sha256",
    "run_id",
    "optimizer_identity_sha256",
    "checkpoint_role",
    "epoch_complete",
    "batch_in_epoch",
    "batches_in_epoch",
    "epoch_start_train_generator_state",
}


class TrainingInstanceLock(AbstractContextManager["TrainingInstanceLock"]):
    """OS-backed single-instance lock released automatically after a crash."""

    def __init__(self, path: Path, *, run_id: str) -> None:
        self.path = path
        self.run_id = run_id
        self._handle: Any = None

    def __enter__(self) -> "TrainingInstanceLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+b")
        self._handle.seek(0, os.SEEK_END)
        if self._handle.tell() == 0:
            self._handle.write(b"\0")
            self._handle.flush()
        self._handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as error:
            self._handle.close()
            self._handle = None
            raise RuntimeError(
                f"another expert training process already owns {self.path}"
            ) from error
        metadata = {
            "pid": os.getpid(),
            "run_id": self.run_id,
            "acquired_utc": datetime.now(timezone.utc).isoformat(),
        }
        self._handle.seek(1)
        self._handle.truncate()
        self._handle.write(json.dumps(metadata, ensure_ascii=False).encode("utf-8"))
        self._handle.flush()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self._handle is None:
            return
        self._handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None


class DatasetPrecomputedNormalizer:
    """Explicit identity normalizer for already-normalized compiled features.

    Keeping this as a stateful runtime component makes checkpoint semantics
    complete without silently changing the established feature contract.
    """

    KIND = "dataset_precomputed_identity_v1"

    def __init__(self, public_scalar_size: int) -> None:
        self.public_scalar_size = int(public_scalar_size)

    def normalize_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if int(batch["public_scalars"].shape[-1]) != self.public_scalar_size:
            raise RuntimeError("normalizer/public scalar dimension mismatch")
        return batch

    def state_dict(self) -> dict[str, Any]:
        return {"kind": self.KIND, "public_scalar_size": self.public_scalar_size}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if dict(state) != self.state_dict():
            raise RuntimeError(f"checkpoint normalizer state is incompatible: {state}")


class LiveTrainingWindow:
    """Telemetry only: actual minibatch means, without changing the loss."""

    FIELDS = ("loss", "loss_position", "loss_card", "gradient_norm")

    def __init__(self) -> None:
        self.count = 0
        self.sums = {name: 0.0 for name in self.FIELDS}
        self.loss_max = 0.0
        self.gradient_max = 0.0
        self.spikes10 = 0
        self.spikes20 = 0

    def add(self, metrics: Mapping[str, float]) -> None:
        self.count += 1
        for name in self.FIELDS:
            self.sums[name] += float(metrics.get(name, 0.0))
        loss = float(metrics["loss"])
        self.loss_max = max(self.loss_max, loss)
        self.gradient_max = max(self.gradient_max, float(metrics["gradient_norm"]))
        self.spikes10 += int(loss > 10.0)
        self.spikes20 += int(loss > 20.0)

    def summary(self) -> dict[str, int | float]:
        return {
            "window_batches": self.count,
            **{f"{name}_window_mean": value / max(1, self.count) for name, value in self.sums.items()},
            "loss_window_max": self.loss_max,
            "gradient_norm_window_max": self.gradient_max,
            "loss_window_gt10": self.spikes10,
            "loss_window_gt20": self.spikes20,
        }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_torch(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        torch.save(dict(value), handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(
        destination.name + f".{os.getpid()}.tmp"
    )
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def _save_rolling_latest(
    checkpoints: Path, checkpoint: Mapping[str, Any]
) -> None:
    latest = checkpoints / "latest.pt"
    previous_1 = checkpoints / "previous-1.pt"
    previous_2 = checkpoints / "previous-2.pt"
    if previous_1.is_file():
        _atomic_copy(previous_1, previous_2)
    if latest.is_file():
        _atomic_copy(latest, previous_1)
    _atomic_torch(
        latest, {**dict(checkpoint), "checkpoint_role": "latest"}
    )


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _pending_checkpoint_request(run_root: Path, run_id: str, step: int) -> dict[str, Any] | None:
    """Read-only command polling; malformed requests never change training settings."""
    request_path = run_root / "control" / "checkpoint-request.json"
    response_path = run_root / "control" / "checkpoint-response.json"
    if not request_path.is_file():
        return None
    raw = request_path.read_bytes() if request_path.stat().st_size <= 8192 else b"oversized-request"
    digest = hashlib.sha256(raw).hexdigest()
    try:
        response = json.loads(response_path.read_text()) if response_path.is_file() else {}
        if response.get("request_sha256") == digest:
            return None
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("request must be an object")
        request_id = value.get("request_id")
        if not isinstance(request_id, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,96}", request_id) is None:
            raise ValueError("invalid checkpoint request id")
        if response.get("request_id") == request_id and response.get("status") == "saved":
            return None
        if value.get("expected_run_id") != run_id:
            raise ValueError("checkpoint request run identity mismatch")
        allowed = {"request_id", "expected_run_id", "at_step", "stop_after_save", "preserve", "export_fp16", "reason"}
        if set(value) - allowed:
            raise ValueError("unsupported checkpoint control fields")
        at_step = value.get("at_step", 0)
        if not isinstance(at_step, int) or isinstance(at_step, bool) or at_step < 0:
            raise ValueError("at_step must be a nonnegative integer")
        for name, default in (("stop_after_save", False), ("preserve", True), ("export_fp16", True)):
            value.setdefault(name, default)
            if not isinstance(value[name], bool):
                raise ValueError(f"{name} must be boolean")
        if step < at_step:
            return None
        return {**value, "request_sha256": digest}
    except (ValueError, TypeError, UnicodeError) as error:
        _atomic_json(response_path, {"status": "rejected", "request_sha256": digest,
            "error": str(error), "global_step": step, "updated_utc": datetime.now(timezone.utc).isoformat()})
        _append_jsonl(run_root / "events.jsonl", {"event": "checkpoint_request_rejected",
            "request_sha256": digest, "global_step": step, "error": str(error)})
        return None


def _finish_checkpoint_request(run_root: Path, request: Mapping[str, Any], checkpoint: Mapping[str, Any],
                               model: RecurrentExpertPolicy) -> dict[str, Any]:
    step = int(checkpoint["global_step"])
    latest = run_root / "checkpoints" / "latest.pt"
    snapshot_root = run_root / "checkpoints" / "manual" / str(request["request_id"])
    response = {"status": "saved", "request_id": request["request_id"],
        "request_sha256": request.get("request_sha256"), "global_step": step,
        "checkpoint": str(latest.resolve()), "stop_after_save": bool(request.get("stop_after_save", False)),
        "updated_utc": datetime.now(timezone.utc).isoformat()}
    if request.get("preserve", True) or request.get("export_fp16", True):
        snapshot_root.mkdir(parents=True, exist_ok=True)
    if request.get("preserve", True):
        snapshot = snapshot_root / f"checkpoint-{step}.pt"
        if snapshot.exists():
            if sha256_file(snapshot) != sha256_file(latest):
                raise RuntimeError("conflicting existing manual checkpoint; preserving both states")
        else:
            _atomic_copy(latest, snapshot)
        response["preserved_checkpoint"] = str(snapshot.resolve())
        response["checkpoint_sha256"] = sha256_file(snapshot)
        best = run_root / "checkpoints" / "best.pt"
        if best.exists() and not (snapshot_root / "best.pt").exists():
            _atomic_copy(best, snapshot_root / "best.pt")
        if (snapshot_root / "best.pt").exists():
            response["preserved_best_checkpoint"] = str((snapshot_root / "best.pt").resolve())
        manifest = run_root / "manifest.json"
        if manifest.exists() and not (snapshot_root / "source-run-manifest.json").exists():
            _atomic_copy(manifest, snapshot_root / "source-run-manifest.json")
    if request.get("export_fp16", True):
        export = snapshot_root / f"weights-{step}-fp16.pt"
        if not export.exists():
            _atomic_torch(export, _inference_weights_payload(epoch=int(checkpoint["epoch"]), global_step=step,
                model=model, dataset_manifest_sha256=str(checkpoint["dataset_manifest_sha256"]),
                run_signature_sha256=str(checkpoint["run_signature_sha256"]), run_id=str(checkpoint["run_id"])))
        response["inference_export"] = str(export.resolve())
        response["inference_export_sha256"] = sha256_file(export)
    _atomic_json(run_root / "control" / "checkpoint-response.json", response)
    _append_jsonl(run_root / "events.jsonl", {"event": "checkpoint_requested_saved", **response})
    return response


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _capture_rng(train_generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "train_loader_generator": train_generator.get_state(),
    }


def _restore_rng(state: Mapping[str, Any], train_generator: torch.Generator) -> None:
    required = {
        "python",
        "numpy",
        "torch",
        "cuda",
        "train_loader_generator",
    }
    missing = sorted(required - set(state))
    if missing:
        raise RuntimeError(f"checkpoint RNG state is incomplete: {missing}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    cuda_state = state["cuda"]
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        # ``torch.load(..., map_location=device)`` also relocates the saved
        # CUDA RNG byte tensors.  Linux requires CPU ByteTensors here, while
        # the original Windows runtime accepted the relocated values.  Moving
        # the exact bytes back to CPU preserves the RNG stream across hosts.
        torch.cuda.set_rng_state_all(
            [
                item.detach().cpu().to(dtype=torch.uint8).contiguous()
                for item in cuda_state
            ]
        )
    train_generator.set_state(state["train_loader_generator"].cpu())


def _run_signature(
    args: argparse.Namespace,
    *,
    dataset_manifest_sha256: str,
    observation_mode: str,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "observation_mode": observation_mode,
        "trainer_contract": "expert_bc_resume_v2",
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "scheduler": "constant_lr_v1",
        "normalizer": DatasetPrecomputedNormalizer.KIND,
        # Preserve the user's CLI contract in the stable id. The resolved
        # device is checked separately in run manifest compatibility so an
        # unavailable GPU cannot silently create a fresh CPU run.
        "device_request": args.device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "burn_in": args.burn_in,
        "workers": args.workers,
        "train_split": args.train_split,
        "validation_split": args.validation_split,
        "test_split": args.test_split,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "hidden_size": args.hidden_size,
        "card_embedding_size": args.card_embedding_size,
        "lambda_max": args.lambda_max,
        "lambda_initial": args.lambda_initial,
        "early_stopping_patience": args.early_stopping_patience,
        "minimum_delta": args.minimum_delta,
        "seed": args.seed,
        "max_train_batches": args.max_train_batches,
        "max_eval_batches": args.max_eval_batches,
        "allow_unanchored_native_states": bool(args.allow_unanchored_native_states),
    }
    # Keep the legacy/default signature byte-for-byte compatible while binding
    # cloud-only topology and numerical changes when they are requested.
    spatial_size = int(getattr(args, "spatial_size", 64))
    precision = str(getattr(args, "precision", "fp32"))
    prefetch_factor = int(getattr(args, "prefetch_factor", 2))
    fused_adamw = bool(getattr(args, "fused_adamw", False))
    if spatial_size != 64:
        payload["spatial_size"] = spatial_size
    if precision != "fp32":
        payload["precision"] = precision
    if prefetch_factor != 2:
        payload["prefetch_factor"] = prefetch_factor
    if fused_adamw:
        payload["fused_adamw"] = True
    if bool(getattr(args, "position_head_fp32", False)):
        payload["position_head_fp32"] = True
    if getattr(args, "position_logit_softcap", None) is not None:
        payload["position_logit_softcap"] = float(args.position_logit_softcap)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def _optimizer_identity(
    args: argparse.Namespace,
    *,
    run_id: str,
    model_config: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Return the immutable optimizer/run identity stored in every checkpoint.

    ``optimizer_state`` alone is not an identity: it can be copied between two
    otherwise compatible runs.  This contract binds the optimizer algorithm
    and hyperparameters to the exact run and model topology before any resume
    artifact is accepted.
    """

    payload = {
        "kind": "expert_bc_adamw_optimizer_identity_v1",
        "run_id": str(run_id),
        "algorithm": "torch.optim.AdamW",
        "torch_version": str(torch.__version__),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "amsgrad": False,
        "maximize": False,
        "capturable": False,
        "differentiable": False,
        "foreach": None,
        "fused": True if bool(getattr(args, "fused_adamw", False)) else None,
        "scheduler": "constant_lr_v1",
        "model_config": dict(model_config),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest(), payload


def _stable_run_id(
    *,
    observation_mode: str,
    dataset_manifest_sha256: str,
    run_signature_sha256: str,
) -> str:
    mode = "sequence" if observation_mode == OBSERVATION_SEQUENCE else "native"
    return (
        f"expert-{mode}-v1-{dataset_manifest_sha256[:12]}-"
        f"{run_signature_sha256[:10]}"
    )


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _move(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    normalizer: DatasetPrecomputedNormalizer,
) -> dict[str, torch.Tensor]:
    moved = {
        key: value.to(device, non_blocking=device.type == "cuda")
        for key, value in batch.items()
    }
    return normalizer.normalize_batch(moved)


def _loader(
    root: Path,
    split: str,
    args: argparse.Namespace,
    *,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader[dict[str, torch.Tensor]]:
    dataset = NativeExpertSequenceDataset(
        root,
        split=split,
        sequence_length=args.sequence_length,
        burn_in=args.burn_in,
        # run() has already hash-verified and semantically validated every
        # shard once. Repeating the full scans for three DataLoaders only
        # burns startup I/O and cannot improve the admission decision.
        validate=False,
    )
    sampler = RandomSampler(dataset, generator=generator) if shuffle else None
    worker_generator = torch.Generator().manual_seed(
        args.seed + {"train": 101, "validation": 202, "test": 303}[split]
    )
    loader_options: dict[str, Any] = {}
    if args.workers > 0:
        loader_options["prefetch_factor"] = int(
            getattr(args, "prefetch_factor", 2)
        )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.workers,
        collate_fn=collate_sequences,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
        drop_last=False,
        generator=worker_generator,
        **loader_options,
    )


def _evaluate(
    model: RecurrentExpertPolicy,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    maximum_batches: int,
    normalizer: DatasetPrecomputedNormalizer,
    precision: str = "fp32",
    progress_callback: Callable[[int, dict[str, float]], None] | None = None,
) -> dict[str, float]:
    model.eval()
    accumulator = MetricAccumulator()
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if maximum_batches and index >= maximum_batches:
                break
            batch = _move(batch, device, normalizer)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=precision == "bf16",
            ):
                output = model.forward_batch(batch, supervised_positions=model.config.position_head_fp32)
                _loss, metrics = behaviour_cloning_loss(output, batch, model.config)
            if not bool(torch.isfinite(_loss)):
                raise FloatingPointError("expert BC validation loss became non-finite")
            accumulator.add(metrics)
            if progress_callback is not None:
                progress_callback(index + 1, accumulator.result())
    return accumulator.result()


def _validate_training_admission(
    args: argparse.Namespace,
    dataset_root: Path,
    manifest: Mapping[str, Any],
    observation_mode: str,
) -> None:
    if args.smoke:
        if getattr(args, "allow_nonproduction_smoke", False):
            if (
                manifest.get("production_ready") is not False
                or manifest.get("smoke_only") is not True
                or manifest.get("smoke_reason")
                != "smoke_deficits_preserved_v1"
                or ((manifest.get("token_coverage") or {}).get("gate") or {}).get(
                    "admitted"
                )
                is not False
            ):
                raise RuntimeError(
                    "non-production smoke requires an authenticated "
                    "smoke_deficits_preserved_v1 dataset"
                )
        elif manifest.get("production_ready") is False and manifest.get("smoke_only") is True:
            raise RuntimeError(
                "smoke-only compiled dataset requires --allow-nonproduction-smoke"
            )
        return
    if manifest.get("production_ready") is not True:
        raise RuntimeError("dataset is not marked production_ready")
    if observation_mode == OBSERVATION_NATIVE:
        if manifest.get("native_replay_validated") is not True:
            raise RuntimeError("dataset lacks native replay validation")
    elif observation_mode == OBSERVATION_SEQUENCE:
        if manifest.get("native_replay_validated") is not False:
            raise RuntimeError("sequence-only dataset must not claim native replay validation")
    else:
        raise RuntimeError(f"unsupported observation mode: {observation_mode}")
    if (manifest.get("split_contract") or {}).get("player_holdout_test") is not True:
        raise RuntimeError("production dataset lacks a player-holdout test split")
    expected_source = args.expected_source_manifest.resolve()
    if not expected_source.is_file():
        raise RuntimeError(f"active accepted source manifest is missing: {expected_source}")
    source_contract = manifest.get("source_manifest") or {}
    compiled_source = Path(str(source_contract.get("path", ""))).resolve()
    if compiled_source != expected_source:
        raise RuntimeError(f"compiled dataset uses stale/wrong source manifest: {compiled_source}")
    if source_contract.get("sha256") != sha256_file(expected_source):
        raise RuntimeError("active accepted source manifest changed after dataset compilation")
    gates = manifest.get("quality_gates") or {}
    required_zero = [
        "split_collisions",
        "forbidden_actor_features",
        "nonfinite_features",
        "expert_label_mask_violations",
    ]
    if observation_mode == OBSERVATION_NATIVE:
        required_zero.extend(("native_action_rejections", "terminal_mismatches"))
    else:
        required_zero.extend(("fabricated_native_grid_rows", "player_holdout_leaks"))
    failures = {name: gates.get(name) for name in required_zero if gates.get(name) != 0}
    if failures:
        raise RuntimeError(f"dataset quality gates are not clean: {failures}")
    provenance = manifest.get("state_provenance") or {}
    if observation_mode == OBSERVATION_SEQUENCE:
        if provenance.get("mode") != "sequence_only":
            raise RuntimeError("sequence-only state provenance is not explicit")
        if int(provenance.get("sequence_only_rows", 0)) <= 0:
            raise RuntimeError("sequence-only dataset has no training rows")
        if int(provenance.get("native_grid_rows", -1)) != 0:
            raise RuntimeError("sequence-only dataset contains/claims native grid rows")
    else:
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


def _checkpoint_payload(
    *,
    epoch: int,
    global_step: int,
    model: RecurrentExpertPolicy,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    normalizer: DatasetPrecomputedNormalizer,
    train_generator: torch.Generator,
    dataset_manifest_sha256: str,
    run_signature_sha256: str,
    run_id: str,
    optimizer_identity_sha256: str,
    best_validation_loss: float,
    epochs_without_improvement: int,
    is_best: bool,
    training_metrics: Mapping[str, float],
    validation_metrics: Mapping[str, float],
    epoch_complete: bool,
    batch_in_epoch: int,
    batches_in_epoch: int,
    epoch_start_train_generator_state: torch.Tensor,
    epoch_sampler_needs_tail: bool = False,
) -> dict[str, Any]:
    return {
        "kind": CHECKPOINT_KIND,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "epoch": int(epoch),
        "step": int(global_step),
        "global_step": int(global_step),
        "updates": int(global_step),
        "model_config": model.config.to_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "normalizer_state": normalizer.state_dict(),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "run_signature_sha256": run_signature_sha256,
        "run_id": run_id,
        "optimizer_identity_sha256": optimizer_identity_sha256,
        "best_validation_loss": float(best_validation_loss),
        "epochs_without_improvement": int(epochs_without_improvement),
        "is_best": bool(is_best),
        "training_metrics": dict(training_metrics),
        "validation_metrics": dict(validation_metrics),
        "rng": _capture_rng(train_generator),
        "epoch_complete": bool(epoch_complete),
        "batch_in_epoch": int(batch_in_epoch),
        "batches_in_epoch": int(batches_in_epoch),
        "epoch_start_train_generator_state": (
            epoch_start_train_generator_state.detach().cpu().clone()
        ),
        "epoch_sampler_needs_tail": bool(epoch_sampler_needs_tail) if not epoch_complete else False,
    }


def _inference_weights_payload(
    *,
    epoch: int,
    global_step: int,
    model: RecurrentExpertPolicy,
    dataset_manifest_sha256: str,
    run_signature_sha256: str,
    run_id: str,
) -> dict[str, Any]:
    model_state = {
        name: (
            value.detach().cpu().to(dtype=torch.float16).contiguous()
            if value.is_floating_point()
            else value.detach().cpu().contiguous()
        )
        for name, value in model.state_dict().items()
    }
    return {
        "kind": "cr_native_expert_inference_weights_v1",
        "schema_version": 1,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "run_id": run_id,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "run_signature_sha256": run_signature_sha256,
        "model_config": model.config.to_dict(),
        "storage_dtype": "float16",
        "model_state": model_state,
    }


def _certify_checkpoint(
    path: Path,
    value: Any,
    *,
    dataset_manifest_sha256: str | None = None,
    run_signature_sha256: str | None = None,
    model_config: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    optimizer_identity_sha256: str | None = None,
) -> dict[str, Any]:
    """Authenticate one checkpoint without mutating any runtime state."""

    if not isinstance(value, dict):
        raise RuntimeError(f"invalid checkpoint payload: {path}")
    if value.get("kind") != CHECKPOINT_KIND:
        raise RuntimeError(f"unexpected checkpoint kind: {path}")
    if int(value.get("schema_version", -1)) != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(
            f"checkpoint is not fully resumable schema v{CHECKPOINT_SCHEMA_VERSION}: {path}"
        )
    missing = sorted(CHECKPOINT_REQUIRED_FIELDS - set(value))
    if missing:
        raise RuntimeError(f"checkpoint is incomplete: {path}: {missing}")
    expected = {
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "run_signature_sha256": run_signature_sha256,
        "run_id": run_id,
        "optimizer_identity_sha256": optimizer_identity_sha256,
    }
    for field, wanted in expected.items():
        if wanted is not None and value.get(field) != wanted:
            raise RuntimeError(f"checkpoint {field} mismatch: {path}")
    if model_config is not None and value.get("model_config") != dict(model_config):
        raise RuntimeError(f"checkpoint model configuration mismatch: {path}")
    expected_role = "best" if path.name == "best.pt" else "latest"
    if value.get("checkpoint_role") != expected_role:
        raise RuntimeError(f"checkpoint role mismatch: {path}")
    if not isinstance(value.get("model_state"), Mapping):
        raise RuntimeError(f"checkpoint model state is malformed: {path}")
    if "epoch_sampler_needs_tail" in value and not isinstance(value["epoch_sampler_needs_tail"], bool):
        raise RuntimeError(f"checkpoint sampler continuation flag is malformed: {path}")
    optimizer_state = value.get("optimizer_state")
    if (
        not isinstance(optimizer_state, Mapping)
        or not isinstance(optimizer_state.get("state"), Mapping)
        or not isinstance(optimizer_state.get("param_groups"), list)
        or not optimizer_state["param_groups"]
    ):
        raise RuntimeError(f"checkpoint optimizer state is malformed: {path}")
    epoch = int(value.get("epoch", -1))
    step = int(value.get("step", -1))
    global_step = int(value.get("global_step", -1))
    if epoch < 0 or step < 0 or global_step < 0 or step != global_step:
        raise RuntimeError(f"checkpoint progress counters are invalid: {path}")
    batch_in_epoch = int(value.get("batch_in_epoch", -1))
    batches_in_epoch = int(value.get("batches_in_epoch", -1))
    if (
        not isinstance(value.get("epoch_complete"), bool)
        or batch_in_epoch < 0
        or batches_in_epoch <= 0
        or batch_in_epoch > batches_in_epoch
        or not isinstance(value.get("epoch_start_train_generator_state"), torch.Tensor)
    ):
        raise RuntimeError(f"checkpoint intra-epoch state is invalid: {path}")
    return value


def _load_certified_checkpoints(
    run_root: Path,
    device: torch.device,
    *,
    dataset_manifest_sha256: str | None = None,
    run_signature_sha256: str | None = None,
    model_config: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    optimizer_identity_sha256: str | None = None,
) -> dict[str, tuple[Path, dict[str, Any]]]:
    """Load and independently certify latest.pt and every present best.pt."""

    candidates = [run_root / "checkpoints" / "latest.pt"]
    best = run_root / "checkpoints" / "best.pt"
    if best.is_file():
        candidates.append(best)
    loaded: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in candidates:
        if not path.is_file():
            if path.name == "latest.pt":
                raise RuntimeError(f"resume checkpoint is missing: {path}")
            continue
        value = torch.load(path, map_location=device, weights_only=False)
        certified = _certify_checkpoint(
            path,
            value,
            dataset_manifest_sha256=dataset_manifest_sha256,
            run_signature_sha256=run_signature_sha256,
            model_config=model_config,
            run_id=run_id,
            optimizer_identity_sha256=optimizer_identity_sha256,
        )
        loaded[path.name] = (path, certified)
    if "best.pt" in loaded:
        latest_value = loaded["latest.pt"][1]
        best_value = loaded["best.pt"][1]
        if best_value.get("is_best") is not True:
            raise RuntimeError("best.pt is not marked as a best checkpoint")
        if (
            int(best_value["epoch"]) > int(latest_value["epoch"])
            or int(best_value["global_step"]) > int(latest_value["global_step"])
            or float(best_value["best_validation_loss"])
            != float(latest_value["best_validation_loss"])
        ):
            raise RuntimeError("latest.pt/best.pt progress identity mismatch")
    return loaded


def _load_resume_checkpoint(
    run_root: Path,
    device: torch.device,
    *,
    dataset_manifest_sha256: str | None = None,
    run_signature_sha256: str | None = None,
    model_config: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    optimizer_identity_sha256: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    loaded = _load_certified_checkpoints(
        run_root,
        device,
        dataset_manifest_sha256=dataset_manifest_sha256,
        run_signature_sha256=run_signature_sha256,
        model_config=model_config,
        run_id=run_id,
        optimizer_identity_sha256=optimizer_identity_sha256,
    )
    return max(
        loaded.values(),
        key=lambda item: (
            int(item[1].get("epoch", -1)),
            int(item[1].get("global_step", -1)),
            item[0].name == "latest.pt",
        ),
    )


def _restore_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    model: RecurrentExpertPolicy,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    normalizer: DatasetPrecomputedNormalizer,
    train_generator: torch.Generator,
    dataset_manifest_sha256: str,
    run_signature_sha256: str,
    run_id: str,
    optimizer_identity_sha256: str,
) -> tuple[int, int, float, int, int, torch.Tensor | None]:
    missing = sorted(CHECKPOINT_REQUIRED_FIELDS - set(checkpoint))
    if missing:
        raise RuntimeError(f"checkpoint is incomplete: {missing}")
    if checkpoint.get("dataset_manifest_sha256") != dataset_manifest_sha256:
        raise RuntimeError("checkpoint dataset manifest SHA-256 mismatch")
    if checkpoint.get("run_signature_sha256") != run_signature_sha256:
        raise RuntimeError("checkpoint training signature mismatch")
    if checkpoint.get("run_id") != run_id:
        raise RuntimeError("checkpoint run identity mismatch")
    if checkpoint.get("optimizer_identity_sha256") != optimizer_identity_sha256:
        raise RuntimeError("checkpoint optimizer identity mismatch")
    if checkpoint["model_config"] != model.config.to_dict():
        raise RuntimeError("checkpoint model configuration mismatch")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    normalizer.load_state_dict(checkpoint["normalizer_state"])
    _restore_rng(checkpoint["rng"], train_generator)
    epoch = int(checkpoint["epoch"])
    global_step = int(checkpoint["global_step"])
    if int(checkpoint["step"]) != global_step:
        raise RuntimeError("checkpoint step/global_step counters disagree")
    best_validation_loss = float(checkpoint["best_validation_loss"])
    epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
    if epoch < 0 or global_step < 0 or epochs_without_improvement < 0:
        raise RuntimeError("checkpoint progress counters are invalid")
    epoch_complete = bool(checkpoint["epoch_complete"])
    resume_batch = 0 if epoch_complete else int(checkpoint["batch_in_epoch"])
    completed_epoch = epoch if epoch_complete else epoch - 1
    epoch_start_state = (
        None
        if epoch_complete
        else checkpoint["epoch_start_train_generator_state"].cpu().clone()
    )
    if completed_epoch < 0:
        completed_epoch = 0
    return (
        completed_epoch,
        global_step,
        best_validation_loss,
        epochs_without_improvement,
        resume_batch,
        epoch_start_state,
    )


def run(args: argparse.Namespace) -> Path:
    dataset_root = args.dataset_root.resolve()
    if args.smoke and (not args.resume or not (dataset_root / "manifest.json").is_file()):
        create_smoke_dataset(dataset_root, replace=True)
    preliminary_manifest = read_manifest(dataset_root)
    preliminary_digest = sha256_file(dataset_root / "manifest.json")
    observation_mode = str(preliminary_manifest.get("observation_mode") or OBSERVATION_NATIVE)
    device = _device(args.device)
    precision = str(getattr(args, "precision", "fp32"))
    if precision == "bf16" and device.type != "cuda":
        raise RuntimeError("BF16 expert training requires CUDA")
    run_signature_sha256, signature_payload = _run_signature(
        args,
        dataset_manifest_sha256=preliminary_digest,
        observation_mode=observation_mode,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or (
        _stable_run_id(
            observation_mode=observation_mode,
            dataset_manifest_sha256=preliminary_digest,
            run_signature_sha256=run_signature_sha256,
        )
        if args.resume
        else f"expert-v1-{stamp}"
    )
    output_root = args.output_root.resolve()
    run_root = output_root / run_id
    with TrainingInstanceLock(output_root / ".expert-training-v1.lock", run_id=run_id):
        trusted_manifest_text = os.environ.get(
            "CR_EXPERT_TRUST_INTEGRITY_MANIFEST", ""
        ).strip()
        trust_existing_integrity = (
            os.environ.get("CR_EXPERT_TRUST_EXISTING_INTEGRITY") == "1"
            or bool(trusted_manifest_text)
        )
        if trust_existing_integrity:
            if trusted_manifest_text:
                existing_manifest_path = Path(trusted_manifest_text).resolve()
            else:
                if not args.resume:
                    raise RuntimeError(
                        "existing integrity may only be trusted while resuming"
                    )
                existing_manifest_path = run_root / "manifest.json"
            if not existing_manifest_path.is_file():
                raise RuntimeError(
                    "trusted integrity requires an existing authenticated run manifest"
                )
            existing_run_manifest = json.loads(
                existing_manifest_path.read_text(encoding="utf-8-sig")
            )
            integrity = dict(existing_run_manifest.get("dataset_integrity") or {})
            shard_files = preliminary_manifest.get("shard_file_sha256") or {}
            if (
                integrity.get("manifest_sha256") != preliminary_digest
                or int(integrity.get("shard_files", -1)) != len(shard_files)
            ):
                raise RuntimeError(
                    "existing integrity receipt does not match the active dataset"
                )
            missing_file = next(
                (
                    relative
                    for relative in shard_files
                    if not (dataset_root / relative).is_file()
                ),
                None,
            )
            if missing_file is not None:
                raise RuntimeError(
                    f"trusted resume dataset file is missing: {missing_file}"
                )
            manifest = preliminary_manifest
        else:
            manifest, integrity = verify_dataset_integrity(
                dataset_root, workers=args.integrity_workers
            )
        actual_observation_mode = str(manifest.get("observation_mode") or OBSERVATION_NATIVE)
        actual_signature, actual_signature_payload = _run_signature(
            args,
            dataset_manifest_sha256=integrity["manifest_sha256"],
            observation_mode=actual_observation_mode,
        )
        if (
            actual_observation_mode != observation_mode
            or integrity["manifest_sha256"] != preliminary_digest
            or actual_signature != run_signature_sha256
        ):
            raise RuntimeError("dataset changed while training startup validation was running")
        signature_payload = actual_signature_payload
        _validate_training_admission(args, dataset_root, manifest, observation_mode)

        if trust_existing_integrity:
            cached_shards = existing_run_manifest.get("dataset_shards") or {}
            expected_shard_keys = {
                f"{split}:{relative}"
                for split, shards in manifest["splits"].items()
                for relative in shards
            }
            if set(cached_shards) != expected_shard_keys:
                raise RuntimeError(
                    "existing shard validation receipt does not cover the active dataset"
                )
            shard_summary = {
                str(key): {
                    "sequences": int(value["sequences"]),
                    "rows": int(value["rows"]),
                }
                for key, value in cached_shards.items()
            }
        else:
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
            entity_numeric_size=int(dimensions.get("entity_numeric_size", 3)),
            card_embedding_size=args.card_embedding_size,
            spatial_size=int(getattr(args, "spatial_size", 64)),
            hidden_size=args.hidden_size,
            lambda_max=args.lambda_max,
            lambda_initial=args.lambda_initial,
            observation_mode=observation_mode,
            position_head_fp32=bool(getattr(args, "position_head_fp32", False)),
            position_logit_softcap=getattr(args, "position_logit_softcap", None),
        )
        configure_position_precision(config)
        optimizer_identity_sha256, optimizer_contract = _optimizer_identity(
            args,
            run_id=run_id,
            model_config=config.to_dict(),
        )
        run_manifest = {
            "schema_version": RUN_SCHEMA_VERSION,
            "kind": RUN_KIND,
            "run_id": run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_root": str(dataset_root),
            "dataset_manifest_sha256": integrity["manifest_sha256"],
            "dataset_integrity": integrity,
            "source_manifest": manifest.get("source_manifest"),
            "dataset_shards": shard_summary,
            "model": config.to_dict(),
            "optimizer": {
                **optimizer_contract,
                "identity_sha256": optimizer_identity_sha256,
            },
            "training": signature_payload,
            "run_signature_sha256": run_signature_sha256,
            "semantics": {
                "algorithm": "supervised_behaviour_cloning",
                "reward": None,
                "ppo": False,
                "actor_information": "public_only_v1",
                "action_heads": (
                    ["timing_hazard", "hand_slot", "position"]
                    if observation_mode == OBSERVATION_SEQUENCE
                    else [
                        "timing_hazard",
                        "action_kind",
                        "hand_slot",
                        "position",
                        "ability",
                        "ability_position",
                    ]
                ),
                "observation_mode": observation_mode,
                "allow_unanchored_native_states": bool(args.allow_unanchored_native_states),
                "state_provenance": manifest.get("state_provenance", {}),
                "scheduler": "constant_lr_v1",
                "normalizer": DatasetPrecomputedNormalizer.KIND,
            },
            "device": str(device),
        }

        continuing = run_root.exists()
        if continuing and not args.resume:
            raise FileExistsError(run_root)
        if continuing:
            existing_path = run_root / "manifest.json"
            if not existing_path.is_file():
                raise RuntimeError(f"resume run manifest is missing: {existing_path}")
            existing = json.loads(existing_path.read_text(encoding="utf-8-sig"))
            for key in (
                "schema_version",
                "kind",
                "run_id",
                "dataset_root",
                "dataset_manifest_sha256",
                "model",
                "optimizer",
                "training",
                "run_signature_sha256",
                "device",
            ):
                if existing.get(key) != run_manifest.get(key):
                    raise RuntimeError(f"resume run manifest mismatch: {key}")
            result_path = run_root / "result.json"
            if result_path.is_file():
                if not (run_root / "checkpoints" / "best.pt").is_file():
                    raise RuntimeError("completed run is missing its best checkpoint")
                completed_checkpoints = _load_certified_checkpoints(
                    run_root,
                    device,
                    dataset_manifest_sha256=integrity["manifest_sha256"],
                    run_signature_sha256=run_signature_sha256,
                    model_config=config.to_dict(),
                    run_id=run_id,
                    optimizer_identity_sha256=optimizer_identity_sha256,
                )
                latest_path, latest_checkpoint = completed_checkpoints["latest.pt"]
                best_path, best_checkpoint = completed_checkpoints["best.pt"]
                completed_result = json.loads(result_path.read_text(encoding="utf-8"))
                if (
                    completed_result.get("dataset_manifest_sha256")
                    != integrity["manifest_sha256"]
                    or completed_result.get("run_signature_sha256")
                    != run_signature_sha256
                    or completed_result.get("run_id") != run_id
                    or completed_result.get("optimizer_identity_sha256")
                    != optimizer_identity_sha256
                ):
                    raise RuntimeError("completed run result does not match this run")
                expected_best_reference = {
                    "path": str(best_path.resolve()),
                    "sha256": sha256_file(best_path),
                    "epoch": int(best_checkpoint["epoch"]),
                    "global_step": int(best_checkpoint["global_step"]),
                    "best_validation_loss": float(
                        best_checkpoint["best_validation_loss"]
                    ),
                }
                expected_latest_reference = {
                    "path": str(latest_path.resolve()),
                    "sha256": sha256_file(latest_path),
                    "epoch": int(latest_checkpoint["epoch"]),
                    "global_step": int(latest_checkpoint["global_step"]),
                }
                if (
                    completed_result.get("checkpoint") != str(best_path.resolve())
                    or completed_result.get("best_checkpoint")
                    != expected_best_reference
                    or completed_result.get("latest_checkpoint")
                    != expected_latest_reference
                    or int(completed_result.get("epochs_completed", -1))
                    != int(latest_checkpoint["epoch"])
                    or int(completed_result.get("global_step", -1))
                    != int(latest_checkpoint["global_step"])
                    or float(completed_result.get("best_validation_loss", float("nan")))
                    != float(best_checkpoint["best_validation_loss"])
                ):
                    raise RuntimeError(
                        "completed run result does not reference the authenticated "
                        "best/latest checkpoints"
                    )
                print(result_path.read_text(encoding="utf-8").strip(), flush=True)
                return run_root
        else:
            (run_root / "checkpoints").mkdir(parents=True)
            _atomic_json(run_root / "manifest.json", run_manifest)

        _seed(args.seed)
        model = RecurrentExpertPolicy(config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=args.weight_decay,
            amsgrad=False,
            foreach=None,
            maximize=False,
            capturable=False,
            differentiable=False,
            fused=True if bool(getattr(args, "fused_adamw", False)) else None,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _epoch: 1.0)
        normalizer = DatasetPrecomputedNormalizer(config.public_scalar_size)
        train_generator = torch.Generator().manual_seed(args.seed)
        train_loader = _loader(
            dataset_root,
            args.train_split,
            args,
            shuffle=True,
            generator=train_generator,
        )
        validation_loader = _loader(
            dataset_root, args.validation_split, args, shuffle=False
        )
        test_loader = _loader(
            dataset_root, args.test_split, args, shuffle=False
        )
        if not all(
            len(loader.dataset) for loader in (train_loader, validation_loader, test_loader)
        ):
            raise RuntimeError(
                "train/validation/test must all contain at least one sequence window"
            )

        events = run_root / "events.jsonl"
        best_validation = float("inf")
        epochs_without_improvement = 0
        updates = 0
        completed_epoch = 0
        resume_batch_in_epoch = 0
        resume_epoch_start_generator_state: torch.Tensor | None = None
        resume_sampler_needs_tail = False
        latest_exists = (run_root / "checkpoints" / "latest.pt").is_file()
        if continuing and latest_exists:
            checkpoint_path, checkpoint = _load_resume_checkpoint(
                run_root,
                device,
                dataset_manifest_sha256=integrity["manifest_sha256"],
                run_signature_sha256=run_signature_sha256,
                model_config=config.to_dict(),
                run_id=run_id,
                optimizer_identity_sha256=optimizer_identity_sha256,
            )
            (
                completed_epoch,
                updates,
                best_validation,
                epochs_without_improvement,
                resume_batch_in_epoch,
                resume_epoch_start_generator_state,
            ) = _restore_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                normalizer=normalizer,
                train_generator=train_generator,
                dataset_manifest_sha256=integrity["manifest_sha256"],
                run_signature_sha256=run_signature_sha256,
                run_id=run_id,
                optimizer_identity_sha256=optimizer_identity_sha256,
            )
            # Legacy resumed runs used an explicit suffix sampler. Missing
            # metadata preserves that historical behavior. New checkpoints
            # track RandomSampler's otherwise-empty final randperm consumption.
            resume_sampler_needs_tail = bool(checkpoint.get("epoch_sampler_needs_tail", False))
            latest_path = run_root / "checkpoints" / "latest.pt"
            if checkpoint_path != latest_path:
                _atomic_torch(latest_path, checkpoint)
            _append_jsonl(
                events,
                {
                    "event": "run_resumed",
                    "epoch": completed_epoch,
                    "global_step": updates,
                    "checkpoint": str(checkpoint_path),
                    "resumed_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
        elif continuing:
            checkpoint_files = list((run_root / "checkpoints").glob("*.pt"))
            if checkpoint_files or (events.is_file() and events.stat().st_size):
                raise RuntimeError(
                    "run has partial progress but no complete latest checkpoint; "
                    "preserving evidence"
                )
            _append_jsonl(
                events,
                {
                    "event": "run_resumed_before_first_checkpoint",
                    "epoch": 0,
                    "global_step": 0,
                    "resumed_utc": datetime.now(timezone.utc).isoformat(),
                },
            )

        started = time.perf_counter()
        progress_path = run_root / "training-progress.json"
        train_batches_total = len(train_loader)
        if args.max_train_batches:
            train_batches_total = min(
                train_batches_total, int(args.max_train_batches)
            )
        total_training_steps = train_batches_total * args.epochs
        stop_at_step = int(getattr(args, "stop_at_step", 0))
        stop_after_epoch = int(getattr(args, "stop_after_epoch", 0))
        if stop_after_epoch and completed_epoch >= stop_after_epoch:
            _atomic_json(progress_path, {"kind": "cr_expert_training_progress_v1", "status": "paused",
                "global_step": updates, "epoch": completed_epoch, "epochs": args.epochs,
                "reason": "stop_after_epoch", "updated_utc": datetime.now(timezone.utc).isoformat()})
            return run_root
        if stop_at_step and updates >= stop_at_step:
            _atomic_json(progress_path, {"kind": "cr_expert_training_progress_v1", "status": "paused",
                "global_step": updates, "epoch": completed_epoch, "epochs": args.epochs,
                "reason": "stop_at_step_already_reached", "updated_utc": datetime.now(timezone.utc).isoformat()})
            return run_root
        last_saved_percent = min(
            100, (updates * 100) // max(total_training_steps, 1)
        )
        should_train = epochs_without_improvement < args.early_stopping_patience
        if should_train:
            for epoch in range(completed_epoch + 1, args.epochs + 1):
                if (
                    resume_batch_in_epoch > 0
                    and resume_epoch_start_generator_state is not None
                    and epoch == completed_epoch + 1
                ):
                    train_generator.set_state(
                        resume_epoch_start_generator_state.cpu()
                    )
                epoch_start_generator_state = train_generator.get_state().cpu().clone()
                epoch_loader = train_loader
                batch_offset = 0
                sampler_needs_tail = (
                    not args.max_train_batches or args.max_train_batches >= len(train_loader)
                )
                explicit_resume_loader = False
                if (
                    resume_batch_in_epoch > 0
                    and epoch == completed_epoch + 1
                ):
                    # Recreate the exact epoch permutation from its checkpointed
                    # start state, then hand only the unseen suffix to workers.
                    # Iterating and discarding already-trained batches performs
                    # all dataset I/O again and can leave the GPU idle for hours.
                    permutation = torch.randperm(
                        len(train_loader.dataset), generator=train_generator
                    )
                    sample_offset = min(
                        resume_batch_in_epoch * args.batch_size,
                        len(permutation),
                    )
                    remaining_indices = permutation[sample_offset:].tolist()
                    resume_loader_options: dict[str, Any] = {}
                    if args.workers > 0:
                        resume_loader_options["prefetch_factor"] = int(
                            getattr(args, "prefetch_factor", 2)
                        )
                    epoch_loader = DataLoader(
                        train_loader.dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        sampler=remaining_indices,
                        num_workers=args.workers,
                        collate_fn=collate_sequences,
                        pin_memory=torch.cuda.is_available(),
                        persistent_workers=args.workers > 0,
                        drop_last=False,
                        generator=torch.Generator().manual_seed(args.seed + 101),
                        **resume_loader_options,
                    )
                    batch_offset = resume_batch_in_epoch
                    sampler_needs_tail = resume_sampler_needs_tail
                    explicit_resume_loader = True
                model.train()
                accumulator = MetricAccumulator()
                live_window = LiveTrainingWindow()
                epoch_started = time.perf_counter()
                examples = 0
                _atomic_json(progress_path, {
                    "kind": "cr_expert_training_progress_v1",
                    "status": "training",
                    "epoch": epoch,
                    "epochs": args.epochs,
                    "batch": batch_offset,
                    "batches": train_batches_total,
                    "global_step": updates,
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                })
                for local_batch_index, batch in enumerate(epoch_loader):
                    batch_index = batch_offset + local_batch_index
                    if args.max_train_batches and batch_index >= args.max_train_batches:
                        break
                    batch = _move(batch, device, normalizer)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=precision == "bf16",
                    ):
                        output = model.forward_batch(batch, supervised_positions=config.position_head_fp32)
                        loss, metrics = behaviour_cloning_loss(output, batch, config)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("expert BC loss became non-finite")
                    loss.backward()
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.gradient_clip
                    )
                    if not bool(torch.isfinite(gradient_norm)):
                        raise FloatingPointError("expert BC gradient became non-finite")
                    optimizer.step()
                    metrics["gradient_norm"] = float(gradient_norm.detach().item())
                    live_window.add(metrics)
                    accumulator.add(metrics)
                    examples += int(batch["loss_mask"].sum().item())
                    updates += 1
                    completed_batch = batch_index + 1
                    completed_percent = min(
                        100,
                        (updates * 100) // max(total_training_steps, 1),
                    )
                    checkpoint_request = _pending_checkpoint_request(run_root, run_id, updates)
                    if stop_at_step and updates >= stop_at_step:
                        checkpoint_request = {"request_id": f"stop-at-{stop_at_step}", "expected_run_id": run_id,
                            "stop_after_save": True, "preserve": True, "export_fp16": True,
                            "reason": "bounded_validation_target"}
                    if completed_percent > last_saved_percent or checkpoint_request is not None:
                        rolling_checkpoint = _checkpoint_payload(
                            epoch=epoch,
                            global_step=updates,
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            normalizer=normalizer,
                            train_generator=train_generator,
                            dataset_manifest_sha256=integrity["manifest_sha256"],
                            run_signature_sha256=run_signature_sha256,
                            run_id=run_id,
                            optimizer_identity_sha256=optimizer_identity_sha256,
                            best_validation_loss=best_validation,
                            epochs_without_improvement=epochs_without_improvement,
                            is_best=False,
                            training_metrics=accumulator.result(),
                            validation_metrics={},
                            epoch_complete=False,
                            batch_in_epoch=completed_batch,
                            batches_in_epoch=train_batches_total,
                            epoch_start_train_generator_state=(
                                epoch_start_generator_state
                            ),
                            epoch_sampler_needs_tail=sampler_needs_tail,
                        )
                        _save_rolling_latest(
                            run_root / "checkpoints", rolling_checkpoint
                        )
                        last_saved_percent = completed_percent
                        if checkpoint_request is not None:
                            response = _finish_checkpoint_request(run_root, checkpoint_request, rolling_checkpoint, model)
                            if checkpoint_request.get("stop_after_save", False):
                                _atomic_json(progress_path, {"kind": "cr_expert_training_progress_v1", "status": "paused",
                                    "epoch": epoch, "epochs": args.epochs, "batch": completed_batch,
                                    "batches": train_batches_total, "global_step": updates,
                                    "loss": float(loss.detach().item()), **live_window.summary(),
                                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                                    "checkpoint": response, "updated_utc": datetime.now(timezone.utc).isoformat()})
                                _append_jsonl(events, {"event": "run_paused", "global_step": updates,
                                    "request_id": checkpoint_request["request_id"], "reason": checkpoint_request.get("reason", "user_request")})
                                return run_root
                    if (
                        completed_batch % 100 == 0
                        or completed_batch == train_batches_total
                    ):
                        _atomic_json(progress_path, {
                            "kind": "cr_expert_training_progress_v1",
                            "status": "training",
                            "epoch": epoch,
                            "epochs": args.epochs,
                            "batch": completed_batch,
                            "batches": train_batches_total,
                            "global_step": updates,
                            "loss": float(loss.detach().item()),
                            **live_window.summary(),
                            "learning_rate": float(optimizer.param_groups[0]["lr"]),
                            "position_logit_absmax": float(output.position_logits.detach().abs().amax().item()),
                            "updated_utc": datetime.now(timezone.utc).isoformat(),
                        })
                        live_window = LiveTrainingWindow()
                if explicit_resume_loader and sampler_needs_tail:
                    # RandomSampler consumes a second permutation even when
                    # its final slice is empty; a fixed-index suffix does not.
                    # Match the uninterrupted sampler state without changing
                    # any batch that is actually trained.
                    torch.randperm(len(train_loader.dataset), generator=train_generator)
                resume_batch_in_epoch = 0
                resume_epoch_start_generator_state = None
                resume_sampler_needs_tail = False
                _atomic_json(progress_path, {
                    "kind": "cr_expert_training_progress_v1",
                    "status": "validation",
                    "epoch": epoch,
                    "epochs": args.epochs,
                    "batch": train_batches_total,
                    "batches": train_batches_total,
                    "global_step": updates,
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                })
                training_metrics = accumulator.result()
                validation_metrics = _evaluate(
                    model,
                    validation_loader,
                    device,
                    maximum_batches=args.max_eval_batches,
                    normalizer=normalizer,
                    precision=precision,
                )
                validation_loss = float(validation_metrics.get("loss", float("inf")))
                improved = validation_loss < best_validation - args.minimum_delta
                if improved:
                    best_validation = validation_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                scheduler.step()
                completed_epoch = epoch
                epoch_event = {
                    "event": "epoch_complete",
                    "epoch": epoch,
                    "global_step": updates,
                    "updates": updates,
                    "examples": examples,
                    "wall_seconds": time.perf_counter() - epoch_started,
                    "training": training_metrics,
                    "validation": validation_metrics,
                }
                checkpoint = _checkpoint_payload(
                    epoch=epoch,
                    global_step=updates,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    normalizer=normalizer,
                    train_generator=train_generator,
                    dataset_manifest_sha256=integrity["manifest_sha256"],
                    run_signature_sha256=run_signature_sha256,
                    run_id=run_id,
                    optimizer_identity_sha256=optimizer_identity_sha256,
                    best_validation_loss=best_validation,
                    epochs_without_improvement=epochs_without_improvement,
                    is_best=improved,
                    training_metrics=training_metrics,
                    validation_metrics=validation_metrics,
                    epoch_complete=True,
                    batch_in_epoch=train_batches_total,
                    batches_in_epoch=train_batches_total,
                    epoch_start_train_generator_state=(
                        epoch_start_generator_state
                    ),
                )
                if improved:
                    _atomic_torch(
                        run_root / "checkpoints" / "best.pt",
                        {**checkpoint, "checkpoint_role": "best"},
                    )
                _atomic_torch(
                    run_root / "checkpoints" / "latest.pt",
                    {**checkpoint, "checkpoint_role": "latest"},
                )
                epoch_checkpoint_dir = run_root / "checkpoints" / "epochs"
                epoch_checkpoint_dir.mkdir(parents=True, exist_ok=True)
                _atomic_torch(
                    epoch_checkpoint_dir / f"epoch-{epoch:03d}.pt",
                    {**checkpoint, "checkpoint_role": "epoch"},
                )
                inference_export_dir = run_root / "exports" / "epochs"
                inference_export_dir.mkdir(parents=True, exist_ok=True)
                _atomic_torch(
                    inference_export_dir / f"epoch-{epoch:03d}-fp16.pt",
                    _inference_weights_payload(
                        epoch=epoch,
                        global_step=updates,
                        model=model,
                        dataset_manifest_sha256=integrity["manifest_sha256"],
                        run_signature_sha256=run_signature_sha256,
                        run_id=run_id,
                    ),
                )
                _append_jsonl(events, epoch_event)
                print(json.dumps(epoch_event, ensure_ascii=False), flush=True)
                if stop_after_epoch and epoch >= stop_after_epoch:
                    # All epoch artifacts and full validation precede this runtime-only stop.
                    _append_jsonl(events, {"event": "run_paused", "reason": "stop_after_epoch",
                        "epoch": epoch, "global_step": updates})
                    _atomic_json(progress_path, {"kind": "cr_expert_training_progress_v1", "status": "paused",
                        "epoch": epoch, "epochs": args.epochs, "batch": train_batches_total,
                        "batches": train_batches_total, "global_step": updates,
                        "reason": "stop_after_epoch", "validation": validation_metrics,
                        "updated_utc": datetime.now(timezone.utc).isoformat()})
                    return run_root
                if epochs_without_improvement >= args.early_stopping_patience:
                    _append_jsonl(
                        events,
                        {
                            "event": "early_stopping",
                            "epoch": epoch,
                            "best_validation_loss": best_validation,
                            "patience": args.early_stopping_patience,
                        },
                    )
                    break

        best_path = run_root / "checkpoints" / "best.pt"
        if not best_path.is_file():
            raise RuntimeError("training has no complete best checkpoint")
        final_checkpoints = _load_certified_checkpoints(
            run_root,
            device,
            dataset_manifest_sha256=integrity["manifest_sha256"],
            run_signature_sha256=run_signature_sha256,
            model_config=config.to_dict(),
            run_id=run_id,
            optimizer_identity_sha256=optimizer_identity_sha256,
        )
        latest_path, latest_checkpoint = final_checkpoints["latest.pt"]
        best_path, best_checkpoint = final_checkpoints["best.pt"]
        model.load_state_dict(best_checkpoint["model_state"], strict=True)
        test_metrics = _evaluate(
            model,
            test_loader,
            device,
            maximum_batches=args.max_eval_batches,
            normalizer=normalizer,
            precision=precision,
        )
        session_wall_seconds = time.perf_counter() - started
        final = {
            "event": "run_complete",
            "epochs_requested": args.epochs,
            "epochs_completed": completed_epoch,
            "global_step": updates,
            "step": updates,
            "updates": updates,
            "wall_seconds": session_wall_seconds,
            "session_wall_seconds": session_wall_seconds,
            "best_validation_loss": best_validation,
            "test": test_metrics,
            "checkpoint": str(best_path),
            "best_checkpoint": {
                "path": str(best_path.resolve()),
                "sha256": sha256_file(best_path),
                "epoch": int(best_checkpoint["epoch"]),
                "global_step": int(best_checkpoint["global_step"]),
                "best_validation_loss": float(
                    best_checkpoint["best_validation_loss"]
                ),
            },
            "latest_checkpoint": {
                "path": str(latest_path.resolve()),
                "sha256": sha256_file(latest_path),
                "epoch": int(latest_checkpoint["epoch"]),
                "global_step": int(latest_checkpoint["global_step"]),
            },
            "dataset_manifest_sha256": integrity["manifest_sha256"],
            "run_signature_sha256": run_signature_sha256,
            "run_id": run_id,
            "optimizer_identity_sha256": optimizer_identity_sha256,
        }
        _append_jsonl(events, final)
        _atomic_json(run_root / "result.json", final)
        _atomic_json(progress_path, {
            "kind": "cr_expert_training_progress_v1",
            "status": "completed",
            "epoch": completed_epoch,
            "epochs": args.epochs,
            "batch": train_batches_total,
            "batches": train_batches_total,
            "global_step": updates,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        })
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
            r"\version-window-20260804\accepted-cycle-clean.jsonl"
        ),
    )
    parser.add_argument("--run-id")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "resume the stable run from its latest complete checkpoint; when --run-id "
            "is omitted a deterministic dataset/configuration run id is used"
        ),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--burn-in", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--test-split", default="test")
    parser.add_argument(
        "--integrity-workers",
        type=int,
        default=0,
        help="parallel dataset checksum workers (0 = automatic)",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--card-embedding-size", type=int, default=64)
    parser.add_argument("--spatial-size", type=int, default=64)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--fused-adamw", action="store_true")
    parser.add_argument("--position-head-fp32", action="store_true")
    parser.add_argument("--position-logit-softcap", type=float)
    parser.add_argument("--stop-at-step", type=int, default=0,
                        help="runtime-only global step limit: save complete state and pause; excluded from model signature")
    parser.add_argument("--stop-after-epoch", type=int, default=0,
                        help="runtime-only pause after validation and all epoch artifacts; excluded from model signature")
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
        "--allow-nonproduction-smoke",
        action="store_true",
        help=(
            "allow only a compiler-authenticated smoke_deficits_preserved_v1 "
            "dataset during --smoke; formal training remains forbidden"
        ),
    )
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
    if args.workers < 0 or args.integrity_workers < 0:
        raise ValueError("workers and integrity workers must be non-negative")
    if args.prefetch_factor <= 0 or args.spatial_size <= 0:
        raise ValueError("prefetch factor and spatial size must be positive")
    if args.stop_at_step < 0:
        raise ValueError("stop-at-step must be nonnegative")
    if args.stop_after_epoch < 0 or args.stop_after_epoch > args.epochs:
        raise ValueError("stop-after-epoch must be zero or within the configured epochs")
    split_names = (args.train_split, args.validation_split, args.test_split)
    if set(split_names) != {"train", "validation", "test"}:
        raise ValueError(
            "train/validation/test split arguments must be one permutation"
        )
    if args.allow_nonproduction_smoke and not args.smoke:
        raise ValueError("--allow-nonproduction-smoke requires --smoke")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
