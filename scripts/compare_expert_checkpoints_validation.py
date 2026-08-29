"""Compare expert checkpoints on one deterministic validation subset."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch
from torch.utils.data import DataLoader, RandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from expert_v1.training_v1.dataset import NativeExpertSequenceDataset, collate_sequences
from expert_v1.training_v1.model import ExpertPolicyConfig, RecurrentExpertPolicy
from expert_v1.training_v1.train import DatasetPrecomputedNormalizer, _evaluate


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
        sampler=RandomSampler(dataset, generator=torch.Generator().manual_seed(seed)),
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_sequences,
        pin_memory=True,
        persistent_workers=workers > 0,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed + 1),
        **options,
    )


def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    dataset: NativeExpertSequenceDataset,
    batch_size: int,
    workers: int,
    prefetch: int,
    batches: int,
    seed: int,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ExpertPolicyConfig(**checkpoint["model_config"])
    model = RecurrentExpertPolicy(config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    normalizer = DatasetPrecomputedNormalizer(config.public_scalar_size)
    normalizer.load_state_dict(checkpoint["normalizer_state"])
    identity = {
        "path": str(checkpoint_path.resolve()),
        "global_step": int(checkpoint["global_step"]),
        "epoch": int(checkpoint["epoch"]),
        "batch_in_epoch": int(checkpoint["batch_in_epoch"]),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    del checkpoint
    gc.collect()
    device = torch.device("cuda")
    model.to(device)
    loader = make_loader(
        dataset,
        batch_size=batch_size,
        workers=workers,
        prefetch=prefetch,
        seed=seed,
    )
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    metrics = _evaluate(
        model,
        loader,
        device,
        maximum_batches=batches,
        normalizer=normalizer,
        precision="bf16",
    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if any(not math.isfinite(float(value)) for value in metrics.values()):
        raise FloatingPointError(f"non-finite validation metric: {checkpoint_path}")
    result = {
        **identity,
        "validation_batches": batches,
        "elapsed_seconds": elapsed,
        "batches_per_second": batches / elapsed,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        "metrics": metrics,
    }
    del loader, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch", type=int, default=4)
    parser.add_argument("--batches", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    dataset = NativeExpertSequenceDataset(
        args.dataset_root.resolve(),
        split=args.split,
        sequence_length=128,
        burn_in=32,
        validate=False,
    )
    results = [
        evaluate_checkpoint(
            checkpoint,
            dataset=dataset,
            batch_size=args.batch_size,
            workers=args.workers,
            prefetch=args.prefetch,
            batches=args.batches,
            seed=args.seed,
        )
        for checkpoint in args.checkpoint
    ]
    comparisons: list[dict[str, Any]] = []
    if results:
        baseline = results[0]
        for result in results[1:]:
            comparisons.append(
                {
                    "from_step": baseline["global_step"],
                    "to_step": result["global_step"],
                    "metric_delta": {
                        key: float(result["metrics"][key])
                        - float(baseline["metrics"][key])
                        for key in baseline["metrics"]
                        if key in result["metrics"]
                    },
                }
            )
    payload = {
        "kind": "cr_expert_checkpoint_validation_comparison_v1",
        "device": torch.cuda.get_device_name(0),
        "split": args.split,
        "seed": args.seed,
        "results": results,
        "comparisons": comparisons,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
