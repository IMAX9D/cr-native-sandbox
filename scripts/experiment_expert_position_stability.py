"""Isolated, reproducible position-head experiments; never changes a live trainer.

CPU screening reuses one BF16 backbone forward for all head alternatives and
only recomputes supervised position rows. Its gradients are HEAD-ONLY probes,
not the global gradient of the full training network. GPU confirmation is a
separate stage. All input data remain read-only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed, wait
import gc
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time

try:
    import resource
except ImportError:  # Functional unit tests also run on the Windows workspace.
    resource = None

import torch
from torch.nn import functional as F

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from expert_v1.training_v1.dataset import NativeExpertSequenceDataset, collate_sequences
from expert_v1.training_v1.losses import behaviour_cloning_loss, _safe_mask
from expert_v1.training_v1.losses import MetricAccumulator
from expert_v1.training_v1.model import ExpertPolicyConfig, RecurrentExpertPolicy
from expert_v1.training_v1.schema import ARENA_COLUMNS
from expert_v1.training_v1.train import _atomic_json, _atomic_torch, _append_jsonl
from expert_v1.training_v1.train import DatasetPrecomputedNormalizer, _move, _restore_rng, _capture_rng

VARIANTS = ("bf16_original", "fp32_raw", "fp32_softcap10", "fp32_softcap20",
            "fp32_qknorm16", "fp32_qknorm32")
_DATASETS: dict[str, NativeExpertSequenceDataset] = {}


def configure_threads(count: int) -> None:
    torch.set_num_threads(count)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)


def peak_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2 if resource else 0.


def checkpoint_model(path: Path) -> tuple[RecurrentExpertPolicy, dict]:
    value = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    model = RecurrentExpertPolicy(ExpertPolicyConfig(**value["model_config"]))
    model.load_state_dict(value["model_state"], strict=True)
    return model.eval(), value


def transform_scores(logits: torch.Tensor, variant: str) -> torch.Tensor:
    if variant in ("fp32_raw", "bf16_original"):
        return logits
    if variant.startswith("fp32_softcap"):
        cap = float(variant.removeprefix("fp32_softcap"))
        centered = logits.float() - logits.float().mean(-1, keepdim=True)
        return cap * torch.tanh(centered / cap)
    raise ValueError(variant)


def position_summary(logits: torch.Tensor, labels: torch.Tensor,
                     legal: torch.Tensor, weights: torch.Tensor) -> tuple[dict, torch.Tensor]:
    safe = _safe_mask(legal)
    scores = logits.float().masked_fill(~safe, torch.finfo(torch.float32).min)
    nll = F.cross_entropy(scores, labels, reduction="none")
    valid = torch.isfinite(nll)
    denominator = weights[valid].sum()
    loss = ((nll[valid] * weights[valid]).sum() / denominator
            if bool(valid.any()) and float(denominator.detach()) > 0
            else logits.sum() * 0)
    prediction = scores.argmax(-1)
    row_error = (prediction // ARENA_COLUMNS - labels // ARENA_COLUMNS).float()
    col_error = (prediction % ARENA_COLUMNS - labels % ARENA_COLUMNS).float()
    distance = torch.sqrt(row_error.square() + col_error.square())
    spans = scores.amax(-1) - logits.float().masked_fill(~safe, torch.inf).amin(-1)
    summary = {
        "position_loss": float(loss.detach()), "labels": int(labels.numel()),
        "label_weight": float(denominator.detach()),
        "nll_weighted_sum": float((nll[valid] * weights[valid]).sum().detach()),
        "label_nll_max": float(nll.max().detach()),
        "label_nll_p95": float(torch.quantile(nll.detach(), .95)),
        "label_nll_gt20": int((nll > 20).sum()),
        "position_mean_cell_error": float(distance.mean()),
        "position_within_1_cell": float((distance <= 1).float().mean()),
        "legal_target_failures": int((~safe.gather(1, labels[:, None]).squeeze(1)).sum()),
        "legal_logit_span_max": float(spans.max().detach()),
        "finite": bool(torch.isfinite(nll).all() and torch.isfinite(loss)),
    }
    return summary, loss


def capture_position_inputs(model: RecurrentExpertPolicy, batch: dict,
                            *, gradients: bool = False):
    selected = batch["loss_mask"] & batch["position_label_mask"]
    rows = selected.nonzero(as_tuple=False)
    if not len(rows):
        return model.forward_batch(batch), None
    b, t = rows.unbind(1)
    cards = batch["card_slot"][b, t].clamp(0, 3)
    flat = b * selected.shape[1] + t
    captured = {"batch_rows": b, "time_rows": t, "cards": cards,
                "labels": batch["position"][b, t],
                "legal": batch["position_mask"][b, t],
                "weights": batch["sample_weight"][b, t].clamp_min(0)}

    def query_hook(_module, args):
        value = args[0][b, t, cards]
        captured["query_input"] = value if gradients else value.detach()

    def cell_hook(_module, args):
        value = args[0].index_select(0, flat)
        captured["cell_input"] = value if gradients else value.detach()

    handles = [model.position_query.register_forward_pre_hook(query_hook),
               model.cell_features.register_forward_pre_hook(cell_hook)]
    try:
        output = model.forward_batch(batch)
    finally:
        for handle in handles:
            handle.remove()
    captured["original_logits"] = output.position_logits[b, t, cards]
    return output, captured


def fp32_head(model: RecurrentExpertPolicy, captured: dict, variant: str):
    device_type = captured["query_input"].device.type
    with torch.autocast(device_type=device_type, enabled=False):
        query = model.position_query(captured["query_input"].float())
        cells = model.cell_features(captured["cell_input"].float()).flatten(2).transpose(1, 2)
        if variant.startswith("fp32_qknorm"):
            scale = float(variant.removeprefix("fp32_qknorm"))
            query = F.normalize(query, dim=-1, eps=1e-6)
            cells = F.normalize(cells, dim=-1, eps=1e-6)
            return torch.einsum("ne,npe->np", query, cells) * scale
        scores = torch.einsum("ne,npe->np", query, cells) / math.sqrt(cells.shape[-1])
        return transform_scores(scores, variant)


def diagnostic_batch(model: RecurrentExpertPolicy, batch: dict, *, gradient_probe: bool):
    began = time.perf_counter()
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output, captured = capture_position_inputs(model, batch)
        _, base_metrics = behaviour_cloning_loss(output, batch, model.config)
    backbone_seconds = time.perf_counter() - began
    if captured is None:
        return {"no_position_labels": True, "backbone_seconds": backbone_seconds,
                "baseline_metrics": base_metrics, "variants": {}}
    results = {}
    with torch.no_grad():
        baseline, _ = position_summary(captured["original_logits"], captured["labels"],
                                       captured["legal"], captured["weights"])
    baseline["mean_batch_total_loss"] = base_metrics["loss"]
    baseline["position_loss_parity_error"] = abs(baseline["position_loss"] - base_metrics["loss_position"])
    results["bf16_original"] = baseline
    del output
    for variant in VARIANTS[1:]:
        model.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(gradient_probe):
            logits = fp32_head(model, captured, variant)
            summary, loss = position_summary(logits, captured["labels"], captured["legal"], captured["weights"])
            summary["mean_batch_total_loss"] = (base_metrics["loss"] - base_metrics["loss_position"]
                                                 + summary["position_loss"])
            if gradient_probe:
                loss.backward()
                params = list(model.position_query.parameters()) + list(model.cell_features.parameters())
                norm2 = sum(float(p.grad.detach().float().square().sum())
                            for p in params if p.grad is not None)
                summary["head_only_gradient_norm"] = math.sqrt(norm2)
        results[variant] = summary
        del logits, loss
    model.zero_grad(set_to_none=True)
    return {"backbone_seconds": backbone_seconds, "elapsed_seconds": time.perf_counter() - began,
            "baseline_metrics": base_metrics, "variants": results,
            "supervised_rows": len(captured["labels"]), "input_steps": int(batch["loss_mask"].numel())}


def forward_position_variant(model: RecurrentExpertPolicy, batch: dict, variant: str):
    """Experimental training forward, preserving the official loss/masks.

    FP32 work is restricted to supervised card/position rows. Unsupervised
    rows have zero derivative in the official loss. Other action heads,
    including ability positions, remain on their original computation path.
    """
    if variant in ("bf16_original", "position_lr_quarter"):
        return model.forward_batch(batch)
    output, captured = capture_position_inputs(model, batch, gradients=torch.is_grad_enabled())
    if captured is None:
        # Retain the ordinary zero-gradient graph and optimizer decay behavior.
        return output
    captured.pop("original_logits")
    logits = fp32_head(model, captured, variant)
    positions = torch.zeros_like(output.position_logits, dtype=torch.float32)
    positions[captured["batch_rows"], captured["time_rows"], captured["cards"]] = logits
    return output._replace(position_logits=positions)


def gpu_evaluate(model, normalizer, fixtures, variant, output, phase):
    model.eval()
    accumulators = {"stress": MetricAccumulator(), "validation": MetricAccumulator()}
    rows = []
    for i, fixture in enumerate(fixtures):
        batch = _move(torch.load(fixture["path"], weights_only=True, map_location="cpu"),
                      torch.device("cuda"), normalizer)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = forward_position_variant(model, batch, variant)
            loss, metrics = behaviour_cloning_loss(prediction, batch, model.config)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite evaluation loss")
        accumulators[fixture["cohort"]].add(metrics)
        rows.append({"fixture": fixture["name"], "cohort": fixture["cohort"], **metrics})
        _atomic_json(output / "experiment-progress.json", {"phase": phase, "completed": i + 1,
            "total": len(fixtures), "percent": 100 * (i + 1) / len(fixtures)})
        del batch, prediction, loss
    return {"cohorts": {k: v.result() for k, v in accumulators.items()}, "batches": rows}


def gpu_trial(args):
    configure_threads(args.threads)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for confirmation")
    occupants = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
    others = [int(line.strip()) for line in occupants.splitlines() if line.strip().isdigit()
              and int(line.strip()) != os.getpid()]
    if others:
        raise RuntimeError(f"GPU still occupied; leave the live trainer untouched: {others}")
    args.output.mkdir(parents=True, exist_ok=False)
    fixtures = json.loads(args.fixtures.read_text())
    model, source = checkpoint_model(args.checkpoint)
    manifest_sha = hashlib.sha256((args.dataset_root / "manifest.json").read_bytes()).hexdigest()
    if source["dataset_manifest_sha256"] != manifest_sha or fixtures["dataset_manifest_sha256"] != manifest_sha:
        raise RuntimeError("dataset identity mismatch")
    dataset = NativeExpertSequenceDataset(args.dataset_root, split="validation", sequence_length=128,
                                         burn_in=32, validate=False)
    if math.ceil(len(dataset) / args.batch_size) != source["batches_in_epoch"]:
        raise RuntimeError("trial must retain the source batch size")
    generator = torch.Generator().set_state(source["epoch_start_train_generator_state"].cpu())
    permutation = torch.randperm(len(dataset), generator=generator)
    offset = int(source["batch_in_epoch"]) * args.batch_size
    indices = permutation[offset:offset + args.updates * args.batch_size].tolist()
    if len(indices) != args.updates * args.batch_size:
        raise RuntimeError("trial must stay within the source epoch")
    options = {"prefetch_factor": 2, "persistent_workers": True} if args.workers else {}
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=indices,
        num_workers=args.workers, pin_memory=True, collate_fn=collate_sequences,
        generator=torch.Generator().manual_seed(20260830 + 101), **options)
    del permutation
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model.to(device)
    normalizer = DatasetPrecomputedNormalizer(model.config.public_scalar_size)
    normalizer.load_state_dict(source["normalizer_state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, fused=True)
    optimizer.load_state_dict(source["optimizer_state"])
    for group in optimizer.param_groups:
        group["lr"] = args.learning_rate
        group["initial_lr"] = args.learning_rate
    if args.variant == "position_lr_quarter":
        original_group = optimizer.param_groups[0]
        head = {p for name, p in model.named_parameters()
                if name.startswith(("position_query.", "cell_features."))}
        optimizer.param_groups = [
            {**original_group, "params": [p for p in original_group["params"] if p not in head]},
            {**original_group, "params": [p for p in original_group["params"] if p in head],
             "lr": args.learning_rate / 4, "initial_lr": args.learning_rate / 4},
        ]
    rng_generator = torch.Generator()
    _restore_rng(source["rng"], rng_generator)
    metadata = {"kind": "cr_expert_position_gpu_trial_v1", "source_checkpoint": str(args.checkpoint),
        "source_step": source["global_step"], "source_run": source["run_id"], "variant": args.variant,
        "updates": args.updates, "batch_size": args.batch_size, "precision": "bf16_backbone",
        "head_precision": "bf16" if args.variant in ("bf16_original", "position_lr_quarter") else "fp32",
        "learning_rates": [g["lr"] for g in optimizer.param_groups], "gradient_clip": 1.0,
        "dataset_manifest_sha256": manifest_sha, "parameters": sum(p.numel() for p in model.parameters()),
        "tf32": False, "formal_run_modified": False, "promotion_requires_explicit_variant_support": True,
        "pid": os.getpid(), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    _atomic_json(args.output / "manifest.json", metadata)
    print(json.dumps({"event": "gpu_trial_started", **metadata}), flush=True)
    writer = None
    if args.tensorboard_root:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(args.tensorboard_root / args.output.name))
        writer.add_text("experiment/configuration", json.dumps(metadata, indent=2), source["global_step"])
    before = gpu_evaluate(model, normalizer, fixtures["fixtures"], args.variant, args.output, "validation_before")
    _atomic_json(args.output / "before.json", before)
    if writer:
        for cohort, metrics in before["cohorts"].items():
            for key in ("loss", "loss_position", "position_mean_cell_error", "position_within_1_cell"):
                writer.add_scalar(f"experiment/{cohort}/{key}", metrics[key], source["global_step"])
        writer.flush()
    # Evaluation is deterministic here; restore all RNGs for matched training.
    _restore_rng(source["rng"], rng_generator)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    began = time.perf_counter()
    rows = []
    try:
        for u, batch in enumerate(loader, 1):
            batch = _move(batch, device, normalizer)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                prediction = forward_position_variant(model, batch, args.variant)
                loss, metrics = behaviour_cloning_loss(prediction, batch, model.config)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite training loss")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not bool(torch.isfinite(gradient)):
                raise FloatingPointError("nonfinite training gradient")
            optimizer.step()
            row = {"trial_update": u, "source_global_step": int(source["global_step"]) + u,
                   **metrics, "gradient_norm": float(gradient), "elapsed_seconds": time.perf_counter() - began}
            rows.append(row)
            _append_jsonl(args.output / "training.jsonl", row)
            if u % 10 == 0 or u == args.updates:
                window = rows[-10:]
                progress = {"phase": "gpu_short_training", "completed": u, "total": args.updates,
                    "percent": 100 * u / args.updates, "source_global_step": row["source_global_step"],
                    "window_loss": sum(x["loss"] for x in window) / len(window),
                    "window_max_loss": max(x["loss"] for x in window),
                    "window_gradient": sum(x["gradient_norm"] for x in window) / len(window),
                    "elapsed_seconds": row["elapsed_seconds"], "updates_per_second": u / row["elapsed_seconds"]}
                _atomic_json(args.output / "experiment-progress.json", progress)
                if writer:
                    step = row["source_global_step"]
                    writer.add_scalar("experiment/loss", progress["window_loss"], step)
                    writer.add_scalar("experiment/loss_max", progress["window_max_loss"], step)
                    writer.add_scalar("experiment/gradient_norm_preclip", progress["window_gradient"], step)
                    writer.add_scalar("experiment/loss_position", sum(x["loss_position"] for x in window) / len(window), step)
                    writer.add_scalar("experiment/updates_per_second", progress["updates_per_second"], step)
                    writer.flush()
                print(json.dumps(progress), flush=True)
            del prediction, loss, batch
        torch.cuda.synchronize()
        training_seconds = time.perf_counter() - began
        after = gpu_evaluate(model, normalizer, fixtures["fixtures"], args.variant, args.output, "validation_after")
        _atomic_json(args.output / "after.json", after)
        if writer:
            for cohort, metrics in after["cohorts"].items():
                for key in ("loss", "loss_position", "position_mean_cell_error", "position_within_1_cell"):
                    writer.add_scalar(f"experiment/{cohort}/{key}", metrics[key], int(source["global_step"]) + args.updates)
            writer.flush()
        checkpoint = {**metadata, "model_config": model.config.to_dict(), "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "normalizer_state": normalizer.state_dict(),
            "global_step": int(source["global_step"]) + args.updates, "trial_updates": args.updates,
            "rng": _capture_rng(rng_generator), "epoch_start_train_generator_state": source["epoch_start_train_generator_state"]}
        _atomic_torch(args.output / "trial-checkpoint.pt", checkpoint)
        result = {**metadata, "training_seconds": training_seconds,
            "updates_per_second": args.updates / training_seconds,
            "peak_cuda_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "training_mean_loss": sum(r["loss"] for r in rows) / len(rows),
            "training_max_loss": max(r["loss"] for r in rows),
            "training_loss_gt10": sum(r["loss"] > 10 for r in rows),
            "training_mean_gradient": sum(r["gradient_norm"] for r in rows) / len(rows),
            "training_max_gradient": max(r["gradient_norm"] for r in rows),
            "before": before["cohorts"], "after": after["cohorts"],
            "checkpoint": str(args.output / "trial-checkpoint.pt"),
            "limitation": "short isolated pilot, not proof of long-run stability or full validation"}
        _atomic_json(args.output / "result.json", result)
        _atomic_json(args.output / "experiment-progress.json", {"phase": "completed", "completed": args.updates,
            "total": args.updates, "percent": 100})
        print(json.dumps({"event": "gpu_trial_complete", "result": str(args.output / "result.json")}), flush=True)
    except BaseException as error:
        _atomic_json(args.output / "failure.json", {"error_type": type(error).__name__, "error": str(error),
            "updates_completed": len(rows), "variant": args.variant})
        raise
    finally:
        if writer:
            writer.close()


def pack_initializer(dataset_root: str):
    configure_threads(1)
    for split in ("train", "validation"):
        _DATASETS[split] = NativeExpertSequenceDataset(Path(dataset_root), split=split,
            sequence_length=128, burn_in=32, validate=False, maximum_open_shards=4)


def pack_fixture(task: dict) -> dict:
    began = time.perf_counter()
    dataset = _DATASETS[task["split"]]
    batch = collate_sequences([dataset[i] for i in task["indices"]])
    output = Path(task["path"])
    _atomic_torch(output, batch)
    return {**task, "bytes": output.stat().st_size, "seconds": time.perf_counter() - began,
            "supervised_positions": int((batch["loss_mask"] & batch["position_label_mask"]).sum()),
            "tensor_shapes": {k: list(v.shape) for k, v in batch.items()}}


def prepare(args):
    configure_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "fixtures.json").exists():
        raise FileExistsError("fixture manifest already exists; use it without repacking")
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    train_data = NativeExpertSequenceDataset(args.dataset_root, split="validation", validate=False)
    validation_data = NativeExpertSequenceDataset(args.dataset_root, split="train", validate=False)
    generator = torch.Generator().set_state(source["epoch_start_train_generator_state"].cpu())
    permutation = torch.randperm(len(train_data), generator=generator)
    batches_per_epoch = math.ceil(len(train_data) / args.batch_size)
    tasks = []
    for step in args.steps:
        batch_index = step - (int(source["epoch"]) - 1) * batches_per_epoch - 1
        start = batch_index * args.batch_size
        assert 0 <= start < len(permutation)
        tasks.append({"name": f"stress-step{step}", "cohort": "stress", "source_step": step,
            "split": "validation", "indices": permutation[start:start + args.batch_size].tolist(),
            "path": str((args.output / "fixtures" / f"stress-step{step}.pt").resolve())})
    val_indices = torch.randperm(len(validation_data), generator=torch.Generator().manual_seed(args.seed))
    for i in range(args.validation_batches):
        tasks.append({"name": f"validation-{i:03d}", "cohort": "validation", "source_step": None,
            "split": "train", "indices": val_indices[i * args.batch_size:(i + 1) * args.batch_size].tolist(),
            "path": str((args.output / "fixtures" / f"validation-{i:03d}.pt").resolve())})
    metadata = {"kind": "cr_expert_position_fixtures_v1", "dataset_root": str(args.dataset_root.resolve()),
        "dataset_manifest_sha256": hashlib.sha256((args.dataset_root / "manifest.json").read_bytes()).hexdigest(),
        "permutation_checkpoint": str(args.checkpoint.resolve()), "permutation_source_step": source["global_step"],
        "permutation_method": "same explicit epoch-start randperm reconstruction used by resumed trainer",
        "batch_size": args.batch_size, "seed": args.seed, "sequence_length": 128, "burn_in": 32,
        "training_split": "validation", "validation_split": "train", "dataset_modified": False}
    train_data.close(); validation_data.close()
    del train_data, validation_data, permutation, val_indices, source
    began = time.perf_counter()
    ready = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                             initializer=pack_initializer, initargs=(str(args.dataset_root),)) as pool:
        jobs = [pool.submit(pack_fixture, task) for task in tasks]
        for future in as_completed(jobs):
            ready.append(future.result())
            progress = {"phase": "pack_fixed_inputs", "completed": len(ready), "total": len(tasks),
                        "percent": len(ready) * 100 / len(tasks), "elapsed_seconds": time.perf_counter() - began}
            _atomic_json(args.output / "experiment-progress.json", progress)
            print(json.dumps(progress), flush=True)
    metadata["fixtures"] = sorted(ready, key=lambda x: x["name"])
    _atomic_json(args.output / "fixtures.json", metadata)
    print(json.dumps({"event": "fixtures_ready", "count": len(ready),
                      "bytes": sum(t["bytes"] for t in ready)}), flush=True)


def probe_case(alias: str, checkpoint_path: str, manifest_path: str, output_root: str, threads: int):
    configure_threads(threads)
    root = Path(output_root) / alias
    root.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(Path(manifest_path).read_text())
    model, checkpoint = checkpoint_model(Path(checkpoint_path))
    assert checkpoint["dataset_manifest_sha256"] == manifest["dataset_manifest_sha256"]
    identity = {"alias": alias, "checkpoint": checkpoint_path, "checkpoint_step": checkpoint["global_step"],
                "checkpoint_run": checkpoint["run_id"], "threads": threads, "pid": os.getpid()}
    del checkpoint
    began = time.perf_counter()
    records = []
    for index, fixture in enumerate(manifest["fixtures"]):
        batch = torch.load(fixture["path"], map_location="cpu", weights_only=True)
        gradient_probe = fixture["source_step"] in (133637, 137237, 138937)
        result = diagnostic_batch(model, batch, gradient_probe=gradient_probe)
        record = {**identity, "fixture": fixture["name"], "cohort": fixture["cohort"],
                  "source_step": fixture["source_step"], "gradient_scope": "position_head_only" if gradient_probe else "none",
                  **result}
        _append_jsonl(root / "records.jsonl", record)
        records.append(record)
        progress = {"alias": alias, "completed": index + 1, "total": len(manifest["fixtures"]),
                    "elapsed_seconds": time.perf_counter() - began,
                    "max_rss_gib": peak_rss_gib(),
                    "last_fixture": fixture["name"]}
        _atomic_json(root / "progress.json", progress)
        del batch
    _atomic_json(root / "result.json", {**identity, "records": records,
        "elapsed_seconds": time.perf_counter() - began,
        "limitations": ["CPU BF16 backbone, not CUDA equivalence proof",
                        "head-only gradients, not full-network gradients", "fixed pilot subset, not full validation"]})
    return str(root / "result.json")


def summarize(paths: list[str]):
    groups = {}
    for path in paths:
        case = json.loads(Path(path).read_text())
        for row in case["records"]:
            for variant, metrics in row["variants"].items():
                groups.setdefault((case["alias"], row["cohort"], variant), []).append(metrics)
    rows = []
    for (alias, cohort, variant), values in sorted(groups.items()):
        labels = sum(v["labels"] for v in values)
        weight = sum(v["label_weight"] for v in values)
        measured_gradients = [v["head_only_gradient_norm"] for v in values if "head_only_gradient_norm" in v]
        rows.append({"case": alias, "cohort": cohort, "variant": variant, "batches": len(values),
            "labels": labels, "mean_batch_loss": sum(v["mean_batch_total_loss"] for v in values) / len(values),
            "mean_batch_position_loss": sum(v["position_loss"] for v in values) / len(values),
            "weighted_label_nll": sum(v["nll_weighted_sum"] for v in values) / max(weight, 1e-12),
            "worst_batch_loss": max(v["mean_batch_total_loss"] for v in values),
            "worst_label_nll": max(v["label_nll_max"] for v in values),
            "label_nll_gt20": sum(v["label_nll_gt20"] for v in values),
            "position_mean_cell_error": sum(v["position_mean_cell_error"] * v["labels"] for v in values) / max(labels, 1),
            "position_within_1_cell": sum(v["position_within_1_cell"] * v["labels"] for v in values) / max(labels, 1),
            "head_gradient_max": max(measured_gradients) if measured_gradients else None,
            "all_finite": all(v["finite"] for v in values)})
    return rows


def probe(args):
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [v.split("=", 1) for v in args.checkpoint]
    manifest = json.loads(args.fixtures.read_text())
    total = len(cases) * len(manifest["fixtures"])
    last = -1
    paths = []
    began = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                             max_tasks_per_child=1) as pool:
        pending = {pool.submit(probe_case, alias, path, str(args.fixtures), str(args.output), args.threads)
                   for alias, path in cases}
        while pending:
            done, pending = wait(pending, timeout=1)
            for future in done:
                paths.append(future.result())
            statuses = []
            for alias, _path in cases:
                p = args.output / alias / "progress.json"
                if p.exists():
                    statuses.append(json.loads(p.read_text()))
            completed = sum(s["completed"] for s in statuses)
            if completed != last:
                progress = {"phase": "cpu_head_screen", "completed": completed, "total": total,
                    "percent": completed * 100 / total, "workers": args.workers, "threads_per_worker": args.threads,
                    "elapsed_seconds": time.perf_counter() - began, "cases": statuses}
                _atomic_json(args.output / "experiment-progress.json", progress)
                print(json.dumps(progress), flush=True)
                last = completed
    summary = {"kind": "cr_expert_position_cpu_screen_v1", "fixtures": str(args.fixtures),
               "results": paths, "summary": summarize(paths), "elapsed_seconds": time.perf_counter() - began}
    _atomic_json(args.output / "summary.json", summary)
    print(json.dumps({"event": "cpu_screen_complete", "summary": str(args.output / "summary.json")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("prepare")
    pack.add_argument("--dataset-root", type=Path, required=True)
    pack.add_argument("--checkpoint", type=Path, required=True)
    pack.add_argument("--output", type=Path, required=True)
    pack.add_argument("--batch-size", type=int, default=32)
    pack.add_argument("--workers", type=int, default=4)
    pack.add_argument("--validation-batches", type=int, default=8)
    pack.add_argument("--seed", type=int, default=20260831)
    pack.add_argument("--steps", type=int, nargs="+", default=[108337,113737,123837,131537,133637,135137,136637,137237,138937])
    screen = commands.add_parser("probe")
    screen.add_argument("--fixtures", type=Path, required=True)
    screen.add_argument("--checkpoint", action="append", required=True, help="alias=/absolute/checkpoint.pt")
    screen.add_argument("--output", type=Path, required=True)
    screen.add_argument("--workers", type=int, default=2)
    screen.add_argument("--threads", type=int, default=6)
    trial = commands.add_parser("gpu-trial")
    trial.add_argument("--dataset-root", type=Path, required=True)
    trial.add_argument("--checkpoint", type=Path, required=True)
    trial.add_argument("--fixtures", type=Path, required=True)
    trial.add_argument("--output", type=Path, required=True)
    trial.add_argument("--variant", choices=(*VARIANTS, "position_lr_quarter"), required=True)
    trial.add_argument("--updates", type=int, default=150)
    trial.add_argument("--batch-size", type=int, default=32)
    trial.add_argument("--learning-rate", type=float, default=1e-4)
    trial.add_argument("--workers", type=int, default=4)
    trial.add_argument("--threads", type=int, default=4)
    trial.add_argument("--tensorboard-root", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "probe":
        probe(args)
    else:
        gpu_trial(args)


if __name__ == "__main__":
    main()
