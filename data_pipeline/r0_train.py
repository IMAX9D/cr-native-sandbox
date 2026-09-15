"""Continuous R0 Elite imitation training over whatever usable battles exist.

Consumes the compiler's own battle-level partitions, carries the recurrent state
across the shards of one lane so the same battle keeps its memory, and reports the
four production numbers: processed battles, quarantined battles, real optimizer
steps, train/validation loss. Checkpoints are written atomically and the run is
resumable from its cursor.

Configuration follows the production plan: default R0 Elite (72 entity features +
6 damage statistics), AdamW at 1e-4, small batches with TBPTT gradient
accumulation, gradient clipping at 1.0, and no field is dropped.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
import tempfile

import torch

from training_r0.catalog import CardVocabulary
from training_r0.config import ModelConfig, Temperatures
from training_r0.il_dataset import load_sequence
from training_r0.il_learning import update_imitation_sequence
from training_r0.model import R0Policy
from training_r0.session import weights_hash
from .prepare_600k import atomic_json, digest
from .r0_admission import current_contract, require_batch


def usable_battles(dataset_roots: list, ledgers: list) -> tuple[dict, dict]:
    """Map battle -> compiled dataset dir for every 'usable' ledger entry."""
    usable, counts = {}, {"processed": 0, "quarantined": 0, "usable": 0, "skipped": 0, "captured": 0}
    for ledger in ledgers:
        path = Path(ledger)
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            status = row.get("status")
            if status == "usable":
                counts["usable"] += 1
            elif status == "quarantined":
                counts["quarantined"] += 1
            elif status == "captured":
                counts["captured"] += 1
            else:
                counts["skipped"] += 1
            if status != "usable":
                continue
            counts["processed"] += 1
            for root in dataset_roots:
                candidate = Path(root) / row["battle_tag"]
                if (candidate / "manifest.json").is_file():
                    usable.setdefault(row["battle_tag"], candidate)
    return usable, counts


def lane_shards(root: Path, manifest: dict, partition: str) -> dict:
    """Group shards into per-lane ordered lists so one battle keeps its memory."""
    lanes = {}
    for shard in manifest["shards"]:
        if partition != "all" and shard.get("partition") != partition:
            continue
        lane = (shard["battle_uid"], shard["side"])
        lanes.setdefault(lane, []).append(shard)
    for lane in lanes:
        lanes[lane].sort(key=lambda s: s["first_tick"])
    return lanes


def lane_sequences(root: Path, shards: list):
    for shard in shards:
        yield load_sequence((root / shard["path"]).resolve(), expected_sha256=shard["sha256"])


def evaluate(model, root: Path, partitions=("val",), tbptt_steps=16, max_shards=None):
    """Forward-only loss on held-out battles, carrying lane state like training."""
    contract = current_contract()
    total = 0.0
    decisions = 0
    shards_used = 0
    model.eval()
    with torch.no_grad():
        for dataset in root:
            manifest = json.loads((Path(dataset) / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("contract_sha256") != contract["sha256"]:
                continue
            for partition in partitions:
                for lane, shards in lane_shards(Path(dataset), manifest, partition).items():
                    state = None
                    for sequence in lane_sequences(Path(dataset), shards):
                        if max_shards is not None and shards_used >= max_shards:
                            model.train()
                            return (total / decisions if decisions else float('nan')), decisions, shards_used
                        device = next(model.parameters()).device
                        if not sequence.starts_episode and state is None:
                            break
                        state = state.to(device) if state is not None else None
                        count = int(sequence.label_known.sum())
                        if count == 0:
                            continue
                        for t, observation in enumerate(sequence.observations):
                            batch = observation.to(device)
                            require_batch(batch, model)
                            known = sequence.label_known[t].to(device)
                            output = model(batch, state, forced=sequence.actions[t].to(device),
                                           temperatures=Temperatures())
                            state = output.next_state
                            if bool(known.any()):
                                total += float(-torch.where(known, output.logp, 0.).sum())
                                decisions += int(known.sum())
                        shards_used += 1
                        state = state.detach()
    model.train()
    # None means "no held-out decision was available", which is different from a
    # loss of zero and must not be written as a number.
    return ((total / decisions) if decisions else None), decisions, shards_used


def save_checkpoint(path: Path, payload: dict) -> str:
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=path.name + ".partial-", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return digest(path)


def train(args) -> dict:
    roots = [Path(r) for r in args.dataset]
    battles, counts = usable_battles(roots, args.ledger)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    contract = current_contract()
    config = ModelConfig()
    if config.allow_incomplete_capture:
        raise ValueError("diagnostic R0 cannot produce production weights")
    device = args.device
    model = R0Policy(CardVocabulary.from_native(), config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    step = 0
    epoch = 0
    cursor = None
    best_val = None
    history = []
    if args.resume and (output / "latest.pt").is_file():
        saved = torch.load(output / "latest.pt", map_location=device, weights_only=True)
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        step = int(saved.get("step", 0))
        epoch = int(saved.get("epoch", 0))
        cursor = saved.get("cursor")
        best_val = saved.get("best_val")
        if best_val is not None and not math.isfinite(best_val):
            best_val = None
        history = []
        for entry in saved.get("history", []):
            history.append({key: (None if isinstance(value, float) and not math.isfinite(value) else value)
                            for key, value in entry.items()})
        print(f"resumed from {output/'latest.pt'} at step {step}, epoch {epoch}", flush=True)

    lanes = []
    for tag, dataset in sorted(battles.items()):
        manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("contract_sha256") != contract["sha256"]:
            print(f"  {tag}: compiled under another contract, skipped", flush=True)
            continue
        for lane, shards in lane_shards(dataset, manifest, "train").items():
            lanes.append((tag, dataset, lane, shards))
    if not lanes:
        raise ValueError("no usable training lanes; nothing to train on")
    print(f"training lanes: {len(lanes)} from {len(battles)} usable battles", flush=True)

    started = time.perf_counter()
    stop_reason = "epochs_completed"
    for epoch_index in range(epoch, args.epochs):
        epoch = epoch_index
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_shards = 0
        for lane_index, (tag, dataset, lane, shards) in enumerate(lanes):
            if cursor is not None and (epoch_index, lane_index) < (cursor["epoch"], cursor["lane"]):
                continue
            state = None
            lanes_done = 0
            for shard_index, sequence in enumerate(lane_sequences(dataset, shards)):
                if shards[shard_index]["starts_episode"] is False and state is None:
                    # A mid-battle shard without its predecessor cannot carry memory.
                    break
                try:
                    metrics, state = update_imitation_sequence(model, optimizer, sequence,
                                                               initial_state=state, tbptt_steps=args.tbptt_steps)
                except (ValueError, FloatingPointError, RuntimeError) as error:
                    message = str(error)
                    if "out of memory" in message.lower():
                        torch.cuda.empty_cache()
                        stop_reason = f"cuda_oom:{message[:120]}"
                        raise
                    print(f"  {tag} side {lane[1]} shard {shard_index}: {type(error).__name__}: {message[:160]}",
                          flush=True)
                    break
                step += int(metrics.get("optimizer_steps", 1))
                epoch_steps += int(metrics.get("optimizer_steps", 1))
                epoch_loss += float(metrics["loss"])
                epoch_shards += 1
                lanes_done += 1
                if not math.isfinite(metrics["loss"]) or not math.isfinite(metrics["gradient_norm"]):
                    stop_reason = "nonfinite_metrics"
                    raise FloatingPointError("nonfinite training metrics")
                if args.max_steps and step >= args.max_steps:
                    stop_reason = "max_steps"
                    break
                if args.checkpoint_every and step % args.checkpoint_every == 0:
                    payload = dict(kind="r0-elite-training-checkpoint.v1", model_config=asdict(config),
                                   model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                                   epoch=epoch_index, cursor=dict(epoch=epoch_index, lane=lane_index),
                                   best_val=best_val, history=history, contract_sha256=contract["sha256"],
                                   weights_sha256=weights_hash(model), trained_battles=sorted(battles))
                    save_checkpoint(output / "latest.pt", payload)
            if args.max_steps and step >= args.max_steps:
                break
        train_nll, train_decisions, _ = evaluate(model, [d for _, d, _, _ in lanes], partitions=("train",),
                                                 tbptt_steps=args.tbptt_steps, max_shards=args.train_eval_shards)
        val_loss, val_decisions, val_shards = evaluate(model, [d for _, d, _, _ in lanes],
                                                       partitions=("val",), tbptt_steps=args.tbptt_steps,
                                                       max_shards=args.val_shards)
        entry = dict(epoch=epoch_index, step=step,
                     train_objective=(epoch_loss / epoch_shards) if epoch_shards else None,
                     train_loss=train_nll, val_loss=val_loss,
                     train_decisions=train_decisions,
                     val_status=('measured' if val_loss is not None else 'no_validation_battle'),
                     train_shards=epoch_shards, val_shards=val_shards,
                     val_decisions=val_decisions, seconds=round(time.perf_counter() - started, 1))
        # train_loss/val_loss are both per-decision negative log-likelihood of the
        # same objective, so the pair is directly comparable.
        history.append(entry)
        payload = dict(kind="r0-elite-training-checkpoint.v1", model_config=asdict(config),
                       model=model.state_dict(), optimizer=optimizer.state_dict(), step=step, epoch=epoch_index,
                       cursor=dict(epoch=epoch_index + 1, lane=0), best_val=best_val, history=history,
                       contract_sha256=contract["sha256"], weights_sha256=weights_hash(model),
                       trained_battles=sorted(battles))
        save_checkpoint(output / f"epoch-{epoch_index:03d}.pt", payload)
        save_checkpoint(output / "latest.pt", payload)
        if val_loss is not None and math.isfinite(val_loss) and (best_val is None or val_loss < best_val):
            best_val = val_loss
            save_checkpoint(output / "best.pt", dict(payload, best_val=best_val))
        cursor = dict(epoch=epoch_index + 1, lane=0)
        atomic_json(output / "progress.json", dict(kind="r0-elite-training-progress.v1", **counts,
                                                   train_steps=step, epochs_completed=epoch_index + 1,
                                                   train_loss=entry["train_loss"], val_loss=val_loss,
                                                   best_val=best_val, history=history,
                                                   dataset_roots=[str(r) for r in roots],
                                                   contract_sha256=contract["sha256"],
                                                   checkpoint=dict(path="latest.pt", sha256=digest(output / "latest.pt")),
                                                   best_checkpoint=dict(path="best.pt",
                                                                        sha256=digest(output / "best.pt"))
                                                   if (output / "best.pt").is_file() else None,
                                                   stop_reason=stop_reason if args.max_steps and step >= args.max_steps else "running"))
        print(json.dumps(entry, ensure_ascii=False), flush=True)
        if args.max_steps and step >= args.max_steps:
            break
    progress = json.loads((output / "progress.json").read_text(encoding="utf-8")) if (output / "progress.json").is_file() else {}
    progress.update(stop_reason=stop_reason, train_steps=step,
                    seconds=round(time.perf_counter() - started, 1))
    atomic_json(output / "progress.json", progress)
    return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, nargs='+', required=True)
    parser.add_argument('--ledger', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--tbptt-steps', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--max-steps', type=int, default=0)
    parser.add_argument('--checkpoint-every', type=int, default=10)
    parser.add_argument('--val-shards', type=int, default=20)
    parser.add_argument('--train-eval-shards', type=int, default=20,
                        help='bounded train-partition shards used for a comparable per-decision train NLL')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    print(json.dumps(train(args), ensure_ascii=False, indent=2, default=str))


if __name__ == '__main__':
    main()
