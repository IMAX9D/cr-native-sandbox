"""Standalone single-GPU / torchrun DDP offline trainer."""

from __future__ import annotations
import argparse
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .data import Windows, collate
from .loss import bc_loss, summarize
from .model import Policy, PolicyConfig


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def load_checkpoint(path):
    # Checkpoints contain optimizer/RNG state. Load only your own trusted files.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "cache"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--run-dir", "--run", dest="run", type=Path, required=True)
    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split", default="validation")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--frame-window", type=int, default=128)
    p.add_argument("--event-window", type=int, default=128)
    p.add_argument("--targets", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=4, help="per GPU")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument(
        "--eval-batches", type=int, default=100, help="0 means full held-out split"
    )
    p.add_argument("--resume", type=Path)
    p.add_argument("--evaluate-only", action="store_true")
    p.add_argument(
        "--allow-smoke", action="store_true", help="only for synthetic fixture"
    )
    return p


def run(args):
    if (
        min(
            args.epochs,
            args.batch_size,
            args.targets,
            args.cpu_threads,
            args.log_every,
            args.save_every,
        )
        < 1
    ):
        raise ValueError(
            "positive epochs/batch/targets/threads/log/save intervals required"
        )
    if min(args.workers, args.max_steps, args.eval_batches) < 0:
        raise ValueError("negative argument")
    if args.train_split == args.val_split:
        raise ValueError("training and validation splits must differ")
    torch.set_num_threads(args.cpu_threads)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world > 1
    device = (
        torch.device("cuda", local)
        if args.device != "cpu" and torch.cuda.is_available()
        else torch.device("cpu")
    )
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError("CUDA not available")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if args.precision != "fp32" and device.type != "cuda":
        raise ValueError("mixed precision requires CUDA; CPU smoke uses fp32")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 unsupported on this GPU")
    if distributed:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    seed_all(args.seed + rank)
    train = Windows(
        args.data,
        args.cache,
        args.train_split,
        targets=args.targets,
        frame_window=args.frame_window,
        event_window=args.event_window,
    )
    valid = Windows(
        args.data,
        args.cache,
        args.val_split,
        targets=args.targets,
        frame_window=args.frame_window,
        event_window=args.event_window,
    )
    if not len(train) or not len(valid):
        raise ValueError("empty training/validation split")
    if train.index["smoke_only"] and not args.allow_smoke:
        raise ValueError("synthetic smoke requires --allow-smoke")
    train_tags = {t for r in train.records for t in r["battle_tags"]}
    val_tags = {t for r in valid.records for t in r["battle_tags"]}
    if train_tags & val_tags:
        raise ValueError("training/validation battle overlap")
    dims = train.index["dimensions"]
    config = PolicyConfig(
        card_vocab_size=dims["card_vocab_size"],
        ability_vocab_size=dims["ability_vocab_size"],
        public_scalar_size=dims["public_scalar_size"],
        entity_numeric_size=dims["entity_numeric_size"],
        grid_channels=dims["grid_channels"],
        width=args.width,
        layers=args.layers,
        heads=args.heads,
        frame_window=args.frame_window,
        event_window=args.event_window,
    )
    contract = {
        "manifest_sha256": train.index["manifest_sha256"],
        "train_split": args.train_split,
        "val_split": args.val_split,
        "targets": args.targets,
        "batch_size_per_rank": args.batch_size,
        "world_size": world,
        "precision": args.precision,
        "seed": args.seed,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "event_contract": train.index["event_contract"],
    }
    model = Policy(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = (
        torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")
        if hasattr(torch.amp, "GradScaler")
        else torch.cuda.amp.GradScaler(enabled=args.precision == "fp16")
    )
    epoch = 0
    cursor = 0
    step = 0
    best = float("inf")
    pending_rng = None
    if args.resume:
        saved = load_checkpoint(args.resume)
        if saved["config"] != asdict(config) or saved["contract"] != contract:
            raise ValueError(
                "checkpoint model/data/training contract differs; use the original arguments"
            )
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        epoch, cursor, step, best = (
            saved["epoch"],
            saved["next_batch"],
            saved["step"],
            saved["best_val"],
        )
        pending_rng = saved["rng"][rank]
    if args.evaluate_only and not args.resume:
        raise ValueError("--evaluate-only requires --resume")
    if rank == 0:
        args.run.mkdir(parents=True, exist_ok=True)
        if (args.run / "last.pt").exists() and not args.resume:
            raise FileExistsError("run exists; use --resume or a new run")
        print(
            json.dumps(
                {
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "device": str(device),
                    "world_size": world,
                    "train_windows": len(train),
                    "validation_windows": len(valid),
                    "config": asdict(config),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        (args.run / "config.json").write_text(
            json.dumps(
                {
                    "model": asdict(config),
                    "contract": contract,
                    "torch": torch.__version__,
                    "arguments": {
                        k: str(v) if isinstance(v, Path) else v
                        for k, v in vars(args).items()
                    },
                },
                indent=2,
            )
        )
    if distributed:
        dist.barrier()
        parallel = DistributedDataParallel(
            model, device_ids=[local] if device.type == "cuda" else None
        )
    else:
        parallel = model
    sampler = DistributedSampler(
        train,
        num_replicas=world,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=False,
    )
    generator = torch.Generator().manual_seed(args.seed + rank)
    loader = DataLoader(
        train,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
        generator=generator,
        persistent_workers=args.workers > 0,
    )
    # Exact validation partition; no duplicated tail samples under DDP.
    validation = DataLoader(
        valid,
        batch_size=args.batch_size,
        sampler=list(range(rank, len(valid), world)),
        num_workers=args.workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16

    def autocast():
        return (
            nullcontext()
            if args.precision == "fp32"
            else torch.autocast("cuda", dtype=dtype)
        )

    def record(payload):
        if rank == 0:
            print(json.dumps(payload, ensure_ascii=False), flush=True)
            with (args.run / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(payload) + "\n")

    def save(name):
        local_rng = rng_state()
        states = [None] * world
        if distributed:
            dist.all_gather_object(states, local_rng)
        else:
            states = [local_rng]
        if rank == 0:
            out = {
                "config": asdict(config),
                "contract": contract,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "next_batch": cursor,
                "step": step,
                "best_val": best,
                "rng": states,
            }
            temporary = args.run / (name + ".partial")
            torch.save(out, temporary)
            temporary.replace(args.run / name)

    def evaluate():
        model.eval()
        stats = defaultdict(float)
        with torch.no_grad():
            for vi, batch in enumerate(validation):
                if args.eval_batches and vi >= args.eval_batches:
                    break
                b = move(batch, device)
                with autocast():
                    output = model(b)
                _, s = bc_loss(output, b)
                for k, v in s.items():
                    stats[k] += v
        if distributed:
            gathered = [None] * world
            dist.all_gather_object(gathered, dict(stats))
            stats = defaultdict(float)
            for part in gathered:
                for k, v in part.items():
                    stats[k] += v
        metrics = summarize(stats)
        record(
            {
                "phase": "validation",
                "step": step,
                "epoch": epoch,
                "max_batches_per_rank": args.eval_batches,
                **metrics,
            }
        )
        return metrics["loss"]

    if args.evaluate_only:
        evaluate()
        if distributed:
            dist.destroy_process_group()
        return
    start_time = time.monotonic()
    while epoch < args.epochs and (not args.max_steps or step < args.max_steps):
        sampler.set_epoch(epoch)
        parallel.train()
        accumulated = defaultdict(float)
        for bi, batch in enumerate(loader):
            if bi < cursor:
                continue
            if pending_rng is not None:
                restore_rng(pending_rng)
                pending_rng = None
            b = move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                output = parallel(b)
            loss, stats = bc_loss(output, b, distributed=distributed)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip, error_if_nonfinite=True
            )
            scaler.step(optimizer)
            scaler.update()
            step += 1
            cursor = bi + 1
            for k, v in stats.items():
                accumulated[k] += v
            if step % args.log_every == 0:
                if distributed:
                    parts = [None] * world
                    dist.all_gather_object(parts, dict(accumulated))
                    accumulated = defaultdict(float)
                    for part in parts:
                        for k, v in part.items():
                            accumulated[k] += v
                record(
                    {
                        "phase": "train",
                        "step": step,
                        "epoch": epoch,
                        "elapsed_seconds": time.monotonic() - start_time,
                        "peak_cuda_mb": (
                            torch.cuda.max_memory_allocated() / 2**20
                            if device.type == "cuda"
                            else 0
                        ),
                        **summarize(accumulated),
                    }
                )
                accumulated.clear()
            if step % args.save_every == 0:
                save("last.pt")
            if args.max_steps and step >= args.max_steps:
                break
        if cursor >= len(loader):
            epoch += 1
            cursor = 0
        value = evaluate()
        if value < best:
            best = value
            save("best.pt")
        save("last.pt")
    if distributed:
        dist.destroy_process_group()


def main():
    run(parser().parse_args())


if __name__ == "__main__":
    main()
