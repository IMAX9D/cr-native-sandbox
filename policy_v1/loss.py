"""Conditional BC; WAIT exposure is retained and masked labels never train."""

from __future__ import annotations
import torch
import torch.distributed as dist
from torch.nn import functional as F


def bc_loss(output, b, *, distributed=False):
    base = b["loss_mask"] & b["frame_mask"]
    weights = b["sample_weight"].float()
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("non-finite or negative sample weights")
    losses = {}
    stats = {}

    def add(name, values, valid, correct=None):
        numerator = (values * weights[valid]).sum()
        denominator = weights[valid].sum()
        normalizer = denominator.detach().clone()
        world = dist.get_world_size() if distributed else 1
        if distributed:
            dist.all_reduce(normalizer)
        # DDP averages gradients; normalize by the GLOBAL weighted count.
        losses[name] = numerator * world / normalizer.clamp_min(1)
        stats[name + "_sum"] = float(numerator.detach())
        stats[name + "_weight"] = float(denominator.detach())
        stats[name + "_count"] = int(valid.sum())
        if correct is not None:
            stats[name + "_correct"] = int(correct.sum())

    mask = base & b["timing_label_mask"]
    timing = F.binary_cross_entropy_with_logits(
        output["timing"][mask].float(), b["play_now"][mask].float(), reduction="none"
    )
    add("timing", timing, mask, (output["timing"][mask] > 0) == b["play_now"][mask])
    for name, label, labelmask, legal in [
        ("kind", "action_kind", "kind_label_mask", "action_kind_mask"),
        ("card", "card_slot", "card_label_mask", "card_mask"),
        ("ability", "ability_slot", "ability_label_mask", "ability_mask"),
        ("position", "position", "position_label_mask", "position_mask"),
        (
            "ability_position",
            "ability_position",
            "ability_position_label_mask",
            "ability_position_mask",
        ),
    ]:
        mask = base & b[labelmask]
        logits = output[name]
        if name in ("position", "ability_position"):
            slots = b["card_slot" if name == "position" else "ability_slot"].clamp_min(
                0
            )
            idx = slots[..., None, None].expand(*slots.shape, 1, 576)
            logits = logits.gather(2, idx).squeeze(2)
        logits = logits[mask].float()
        labels = b[label][mask]
        legal_mask = b[legal][mask]
        if len(labels):
            if (labels < 0).any() or (labels >= logits.shape[-1]).any():
                raise ValueError("invalid " + name + " label")
            if not legal_mask.gather(-1, labels[:, None]).all():
                raise ValueError("illegal supervised " + name + " action")
        logits = logits.masked_fill(~legal_mask, -1e9)
        values = (
            F.cross_entropy(logits, labels, reduction="none")
            if len(labels)
            else logits.sum(-1)
        )
        add(name, values, mask, logits.argmax(-1) == labels)
    total = sum(losses.values())
    if not torch.isfinite(total):
        raise FloatingPointError("non-finite BC loss")
    return total, stats


def summarize(stats):
    out = {}
    for name in ("timing", "kind", "card", "ability", "position", "ability_position"):
        count = stats.get(name + "_count", 0)
        weight = stats.get(name + "_weight", 0)
        out[name + "_loss"] = stats.get(name + "_sum", 0) / max(weight, 1)
        out[name + "_accuracy"] = stats.get(name + "_correct", 0) / max(count, 1)
        out[name + "_count"] = count
    out["loss"] = sum(
        out[n + "_loss"]
        for n in ("timing", "kind", "card", "ability", "position", "ability_position")
    )
    return out
