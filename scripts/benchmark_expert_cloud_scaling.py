"""Benchmark larger expert-policy configurations on a compiled native dataset.

This is a training hot-path benchmark, not a model-quality experiment.  It
uses real sequence windows, masks, losses, backward passes and AdamW updates
without creating or modifying a formal training run.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch
from torch.utils.data import DataLoader, RandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from expert_v1.training_v1.dataset import (
    NativeExpertSequenceDataset,
    collate_sequences,
)
from expert_v1.training_v1.losses import behaviour_cloning_loss
from expert_v1.training_v1.model import ExpertPolicyConfig, RecurrentExpertPolicy
from expert_v1.training_v1.schema import OBSERVATION_NATIVE, read_manifest
from expert_v1.training_v1.train import DatasetPrecomputedNormalizer


DEFAULT_SPECS = (
    "h768-b64,768,128,192,64",
    "h1024-b64,1024,192,256,64",
    "h1536-b32,1536,256,384,32",
)


def parse_spec(value: str) -> tuple[str, int, int, int, int]:
    fields = value.split(",")
    if len(fields) != 5:
        raise argparse.ArgumentTypeError(
            "spec must be NAME,HIDDEN,CARD_EMBEDDING,SPATIAL,BATCH"
        )
    name = fields[0].strip()
    numbers = tuple(int(field) for field in fields[1:])
    if not name or any(number <= 0 for number in numbers):
        raise argparse.ArgumentTypeError("spec name and dimensions must be positive")
    return (name, *numbers)


def move_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    normalizer: DatasetPrecomputedNormalizer,
) -> dict[str, torch.Tensor]:
    moved = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    return normalizer.normalize_batch(moved)


def make_loader(
    dataset: NativeExpertSequenceDataset,
    *,
    batch_size: int,
    workers: int,
    prefetch: int,
    seed: int,
) -> DataLoader[dict[str, torch.Tensor]]:
    options: dict[str, Any] = {}
    if workers > 0:
        options["prefetch_factor"] = prefetch
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=RandomSampler(dataset, generator=torch.Generator().manual_seed(seed)),
        num_workers=workers,
        collate_fn=collate_sequences,
        pin_memory=True,
        persistent_workers=workers > 0,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed + 1),
        **options,
    )


def benchmark_spec(
    *,
    name: str,
    hidden_size: int,
    card_embedding_size: int,
    spatial_size: int,
    batch_size: int,
    manifest: dict[str, Any],
    dataset: NativeExpertSequenceDataset,
    workers: int,
    prefetch: int,
    warmup_steps: int,
    measured_steps: int,
    seed: int,
) -> dict[str, Any]:
    device = torch.device("cuda")
    dimensions = manifest["dimensions"]
    config = ExpertPolicyConfig(
        grid_channels=int(dimensions["grid_channels"]),
        public_scalar_size=int(dimensions["public_scalar_size"]),
        card_vocab_size=int(dimensions["card_vocab_size"]),
        ability_vocab_size=int(dimensions["ability_vocab_size"]),
        max_ability_slots=int(dimensions["max_ability_slots"]),
        entity_numeric_size=int(dimensions.get("entity_numeric_size", 3)),
        card_embedding_size=card_embedding_size,
        spatial_size=spatial_size,
        hidden_size=hidden_size,
        lambda_max=20.0,
        lambda_initial=0.3,
        observation_mode=OBSERVATION_NATIVE,
    )
    model = RecurrentExpertPolicy(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-4, fused=True
    )
    normalizer = DatasetPrecomputedNormalizer(config.public_scalar_size)
    loader = make_loader(
        dataset,
        batch_size=batch_size,
        workers=workers,
        prefetch=prefetch,
        seed=seed,
    )
    iterator = iter(loader)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model.train()

    measured_rows = 0
    measured_loss = 0.0
    elapsed = 0.0
    try:
        for step in range(warmup_steps + measured_steps):
            if step == warmup_steps:
                torch.cuda.synchronize(device)
                started = time.perf_counter()
            batch = move_batch(next(iterator), device, normalizer)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model.forward_batch(batch)
                loss, _metrics = behaviour_cloning_loss(output, batch, config)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("benchmark loss became non-finite")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError("benchmark gradient became non-finite")
            optimizer.step()
            if step >= warmup_steps:
                measured_rows += int(batch["loss_mask"].sum().item())
                measured_loss += float(loss.detach().item())
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        return {
            "name": name,
            "status": "ok",
            "hidden_size": hidden_size,
            "card_embedding_size": card_embedding_size,
            "spatial_size": spatial_size,
            "batch_size": batch_size,
            "parameters": parameters,
            "weights_fp32_mib": parameters * 4 / (1024**2),
            "warmup_steps": warmup_steps,
            "measured_steps": measured_steps,
            "elapsed_seconds": elapsed,
            "batches_per_second": measured_steps / elapsed,
            "sequence_windows_per_second": measured_steps * batch_size / elapsed,
            "valid_native_rows_per_second": measured_rows / elapsed,
            "mean_loss": measured_loss / measured_steps,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024**2),
        }
    except torch.OutOfMemoryError as error:
        return {
            "name": name,
            "status": "oom",
            "hidden_size": hidden_size,
            "card_embedding_size": card_embedding_size,
            "spatial_size": spatial_size,
            "batch_size": batch_size,
            "parameters": parameters,
            "error": str(error),
        }
    finally:
        del iterator, loader, optimizer, model
        gc.collect()
        torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--burn-in", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--measured-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--spec", action="append", type=parse_spec)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    manifest = read_manifest(args.dataset_root.resolve())
    dataset = NativeExpertSequenceDataset(
        args.dataset_root.resolve(),
        split=args.split,
        sequence_length=args.sequence_length,
        burn_in=args.burn_in,
        validate=False,
    )
    specs = args.spec or [parse_spec(value) for value in DEFAULT_SPECS]
    results: list[dict[str, Any]] = []
    for spec in specs:
        result = benchmark_spec(
            name=spec[0],
            hidden_size=spec[1],
            card_embedding_size=spec[2],
            spatial_size=spec[3],
            batch_size=spec[4],
            manifest=manifest,
            dataset=dataset,
            workers=args.workers,
            prefetch=args.prefetch,
            warmup_steps=args.warmup_steps,
            measured_steps=args.measured_steps,
            seed=args.seed,
        )
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    payload = {
        "kind": "cr_expert_cloud_scaling_sweep_v1",
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "results": results,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
