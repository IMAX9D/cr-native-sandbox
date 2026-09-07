"""Standalone single-device pipeline benchmark; never saves model weights."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler

from .data import Windows, collate
from .loss import bc_loss
from .model import Policy, PolicyConfig
from .timing import StageTimer, timed_batches
from .train import move, optimizer_update, seed_all


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "cache"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    for name, default in (
        ("width", 256),
        ("layers", 3),
        ("heads", 8),
        ("frame-window", 128),
        ("event-window", 128),
        ("targets", 32),
        ("batch-size", 32),
        ("workers", 2),
        ("cpu-threads", 4),
        ("steps", 50),
        ("warmup", 5),
        ("log-every", 10),
        ("seed", 42),
    ):
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument("--allow-smoke", action="store_true")
    return p


def run(args, *, model_factory=Policy, config_factory=None, dataset_factory=Windows, loss_fn=bc_loss, collate_fn=collate):
    if (
        min(args.steps, args.batch_size, args.targets, args.cpu_threads, args.log_every)
        < 1
    ):
        raise ValueError(
            "steps, batch size, targets, threads and log interval must be positive"
        )
    if min(args.workers, args.warmup) < 0:
        raise ValueError("workers and warmup must be non-negative")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cpu" and args.precision != "fp32":
        raise ValueError("CPU benchmark requires fp32")
    torch.set_num_threads(args.cpu_threads)
    seed_all(args.seed)
    dataset = dataset_factory(
        args.data,
        args.cache,
        args.split,
        targets=args.targets,
        frame_window=args.frame_window,
        event_window=args.event_window,
    )
    if dataset.index["smoke_only"] and not args.allow_smoke:
        raise ValueError("synthetic fixture requires --allow-smoke")
    if config_factory is not None:
        config = config_factory(args, dataset.index["dimensions"])
    else:
        config = PolicyConfig(
            **{
                key: dataset.index["dimensions"][key]
                for key in (
                    "card_vocab_size",
                    "ability_vocab_size",
                    "public_scalar_size",
                    "entity_numeric_size",
                    "grid_channels",
                )
            },
            width=args.width,
            layers=args.layers,
            heads=args.heads,
            frame_window=args.frame_window,
            event_window=args.event_window,
        )
    model = model_factory(config).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = (
        torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")
        if hasattr(torch.amp, "GradScaler")
        else torch.cuda.amp.GradScaler(enabled=args.precision == "fp16")
    )
    sampler = DistributedSampler(
        dataset, num_replicas=1, rank=0, shuffle=True, seed=args.seed
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed),
        persistent_workers=args.workers > 0,
    )
    print(
        json.dumps(
            {
                "phase": "benchmark_start",
                "device": str(device),
                "parameters": sum(p.numel() for p in model.parameters()),
                "config": asdict(config),
                "precision": args.precision,
                "batch_size": args.batch_size,
                "workers": args.workers,
                "requested_measured_batches": args.steps,
                "warmup_batches": args.warmup,
            }
        ),
        flush=True,
    )
    timer = StageTimer(device, enabled=True, warmup=args.warmup)
    updates = overflows = consecutive = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for attempt, (batch, wait) in enumerate(timed_batches(loader), 1):
        timer.begin(wait, batch["frame_ticks"].shape[0])
        with timer.stage("host_to_device"):
            b = move(batch, device)
        with timer.stage("forward"):
            with (
                nullcontext()
                if args.precision == "fp32"
                else torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16 if args.precision == "fp16" else torch.bfloat16,
                )
            ):
                output = model(b)
        with timer.stage("loss"):
            loss, _ = loss_fn(output, b)
        with timer.stage("backward"):
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
        with timer.stage("optimizer"):
            updated, _, _ = optimizer_update(model, optimizer, scaler, 1.0)
        consecutive = 0 if updated else consecutive + 1
        if timer.active:
            updates += int(updated)
            overflows += int(not updated)
        if consecutive >= 32:
            raise FloatingPointError("32 consecutive FP16 overflows")
        if timer.active and timer.batches % args.log_every == 0:
            print(
                json.dumps({"phase": "benchmark_progress", **timer.report()}),
                flush=True,
            )
        if attempt >= args.warmup + args.steps:
            break
    report = timer.report()
    if report is None:
        raise ValueError(
            "dataset exhausted before measured batches; reduce warmup/batch size"
        )
    report.update(
        phase="benchmark",
        successful_updates=updates,
        skipped_overflows=overflows,
        peak_cuda_mb=(
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else 0
        ),
    )
    print(json.dumps(report), flush=True)
    return report


def main():
    run(parser().parse_args())


if __name__ == "__main__":
    main()
