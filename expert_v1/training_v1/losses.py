"""Conditional behaviour-cloning objective and offline metrics."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Mapping

import torch
from torch import Tensor
import torch.nn.functional as F

from .model import ExpertPolicyOutput, ExpertPolicyConfig, masked_logits


def _weighted_mean(values: Tensor, mask: Tensor, weights: Tensor) -> Tensor:
    selected = mask & torch.isfinite(values)
    denominator = weights[selected].sum()
    if not bool(selected.any()) or float(denominator.detach().item()) <= 0:
        return values.sum() * 0.0
    return (values[selected] * weights[selected]).sum() / denominator


def _safe_mask(mask: Tensor) -> Tensor:
    safe = mask.clone()
    empty = ~safe.any(dim=-1)
    if bool(empty.any()):
        safe[empty] = False
        safe[..., 0] |= empty
    return safe


def behaviour_cloning_loss(
    output: ExpertPolicyOutput,
    batch: Mapping[str, Tensor],
    config: ExpertPolicyConfig,
    *,
    coefficients: Mapping[str, float] | None = None,
) -> tuple[Tensor, dict[str, float]]:
    coefficients = {
        "timing": 1.0,
        "kind": 0.5,
        "card": 1.0,
        "position": 1.0,
        "ability": 1.0,
        "ability_position": 1.0,
        **(coefficients or {}),
    }
    base = batch["loss_mask"]
    weights = batch["sample_weight"].clamp_min(0)

    exposure = batch["timing_exposure_ticks"].clamp_min(1) * config.native_tick_seconds
    rate = torch.sigmoid(output.rate_logits) * config.lambda_max
    log_no_play = -(rate * exposure)
    play_probability = -torch.expm1(log_no_play)
    log_play = torch.log(play_probability.clamp_min(torch.finfo(rate.dtype).tiny))
    timing_nll = -torch.where(batch["play_now"], log_play, log_no_play)
    timing_mask = base & batch["timing_label_mask"]
    timing_loss = _weighted_mean(timing_nll, timing_mask, weights)

    kind_logits = masked_logits(output.action_kind_logits, _safe_mask(batch["action_kind_mask"]))
    kind_nll = F.cross_entropy(
        kind_logits.flatten(0, 1), batch["action_kind"].flatten(), reduction="none", ignore_index=-100
    ).reshape_as(batch["action_kind"])
    kind_mask = base & batch["kind_label_mask"]
    kind_loss = _weighted_mean(kind_nll, kind_mask, weights)

    card_logits = masked_logits(output.card_logits, _safe_mask(batch["card_mask"]))
    card_nll = F.cross_entropy(
        card_logits.flatten(0, 1), batch["card_slot"].flatten(), reduction="none", ignore_index=-100
    ).reshape_as(batch["card_slot"])
    card_mask = base & batch["card_label_mask"]
    card_loss = _weighted_mean(card_nll, card_mask, weights)

    selected_card = batch["card_slot"].clamp(0, 3)
    gather_position = selected_card[..., None, None].expand(*selected_card.shape, 1, output.position_logits.shape[-1])
    position_logits = output.position_logits.gather(2, gather_position).squeeze(2)
    position_logits = masked_logits(position_logits, _safe_mask(batch["position_mask"]))
    position_nll = F.cross_entropy(
        position_logits.flatten(0, 1), batch["position"].flatten(), reduction="none", ignore_index=-100
    ).reshape_as(batch["position"])
    position_mask = base & batch["position_label_mask"]
    position_loss = _weighted_mean(position_nll, position_mask, weights)

    ability_logits = masked_logits(output.ability_logits, _safe_mask(batch["ability_mask"]))
    ability_nll = F.cross_entropy(
        ability_logits.flatten(0, 1), batch["ability_slot"].flatten(), reduction="none", ignore_index=-100
    ).reshape_as(batch["ability_slot"])
    ability_mask = base & batch["ability_label_mask"]
    ability_loss = _weighted_mean(ability_nll, ability_mask, weights)

    ability_slot = batch["ability_slot"].clamp(0, config.max_ability_slots - 1)
    gather_ability_position = ability_slot[..., None, None].expand(
        *ability_slot.shape, 1, output.ability_position_logits.shape[-1]
    )
    ability_position_logits = output.ability_position_logits.gather(
        2, gather_ability_position
    ).squeeze(2)
    ability_position_logits = masked_logits(
        ability_position_logits, _safe_mask(batch["ability_position_mask"])
    )
    ability_position_nll = F.cross_entropy(
        ability_position_logits.flatten(0, 1),
        batch["ability_position"].flatten(),
        reduction="none",
        ignore_index=-100,
    ).reshape_as(batch["ability_position"])
    ability_position_mask = base & batch["ability_position_label_mask"]
    ability_position_loss = _weighted_mean(
        ability_position_nll, ability_position_mask, weights
    )

    losses = {
        "timing": timing_loss,
        "kind": kind_loss,
        "card": card_loss,
        "position": position_loss,
        "ability": ability_loss,
        "ability_position": ability_position_loss,
    }
    total = sum(coefficients[name] * value for name, value in losses.items())
    metrics: dict[str, float] = {
        "loss": float(total.detach().item()),
        **{f"loss_{name}": float(value.detach().item()) for name, value in losses.items()},
        "lambda_mean": float(rate[timing_mask].mean().detach().item()) if bool(timing_mask.any()) else 0.0,
        "play_probability_mean": float(play_probability[timing_mask].mean().detach().item()) if bool(timing_mask.any()) else 0.0,
    }

    with torch.no_grad():
        for name, logits, labels, mask in (
            ("kind", kind_logits, batch["action_kind"], kind_mask),
            ("card", card_logits, batch["card_slot"], card_mask),
            ("ability", ability_logits, batch["ability_slot"], ability_mask),
        ):
            metrics[f"{name}_count"] = float(mask.sum().item())
            metrics[f"{name}_top1"] = (
                float((logits.argmax(-1)[mask] == labels[mask]).float().mean().item())
                if bool(mask.any()) else 0.0
            )
            if name == "card":
                top3 = logits.topk(k=min(3, logits.shape[-1]), dim=-1).indices
                metrics["card_top3"] = (
                    float((top3[mask] == labels[mask, None]).any(-1).float().mean().item())
                    if bool(mask.any()) else 0.0
                )
        metrics["timing_count"] = float(timing_mask.sum().item())
        metrics["timing_brier"] = (
            float(((play_probability[timing_mask] - batch["play_now"][timing_mask].float()) ** 2).mean().item())
            if bool(timing_mask.any()) else 0.0
        )
        metrics["position_count"] = float(position_mask.sum().item())
        if bool(position_mask.any()):
            predicted = position_logits.argmax(-1)[position_mask]
            target = batch["position"][position_mask]
            prow, pcol = predicted.div(18, rounding_mode="floor"), predicted.remainder(18)
            trow, tcol = target.div(18, rounding_mode="floor"), target.remainder(18)
            distance = torch.sqrt((prow - trow).float().square() + (pcol - tcol).float().square())
            metrics["position_mean_cell_error"] = float(distance.mean().item())
            metrics["position_within_1_cell"] = float((distance <= 1.0).float().mean().item())
        else:
            metrics["position_mean_cell_error"] = 0.0
            metrics["position_within_1_cell"] = 0.0
    return total, metrics


class MetricAccumulator:
    def __init__(self) -> None:
        self.weighted: dict[str, float] = defaultdict(float)
        self.weights: dict[str, float] = defaultdict(float)

    def add(self, metrics: Mapping[str, float]) -> None:
        base = max(1.0, float(metrics.get("timing_count", 1.0)))
        for key, value in metrics.items():
            if key.endswith("_count"):
                self.weighted[key] += float(value)
                self.weights[key] = 1.0
                continue
            if key.startswith("card_"):
                weight = max(1.0, float(metrics.get("card_count", 0.0)))
            elif key.startswith("position_"):
                weight = max(1.0, float(metrics.get("position_count", 0.0)))
            elif key.startswith("ability_"):
                weight = max(1.0, float(metrics.get("ability_count", 0.0)))
            elif key.startswith("kind_"):
                weight = max(1.0, float(metrics.get("kind_count", 0.0)))
            else:
                weight = base
            self.weighted[key] += float(value) * weight
            self.weights[key] += weight

    def result(self) -> dict[str, float]:
        return {
            key: value / max(self.weights.get(key, 1.0), 1e-12)
            for key, value in self.weighted.items()
        }
