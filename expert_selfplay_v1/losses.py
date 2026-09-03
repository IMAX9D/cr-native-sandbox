"""Separated Critic warm-up and auxiliary objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .critic import CriticOutput


@dataclass(frozen=True)
class CriticTargets:
    returns: Tensor
    wdl_class: Tensor
    crown_difference: Tensor
    tower_hp_difference: Tensor
    future_damage: Tensor
    loss_mask: Tensor


@dataclass(frozen=True)
class CriticLoss:
    total: Tensor
    value: Tensor
    wdl: Tensor
    crown: Tensor
    tower_hp: Tensor
    future_damage: Tensor


def critic_loss(
    output: CriticOutput,
    targets: CriticTargets,
    *,
    value_coefficient: float = 1.0,
    auxiliary_coefficient: float = 0.10,
) -> CriticLoss:
    mask = targets.loss_mask.bool()
    if mask.shape != output.values.shape or not bool(mask.any()):
        raise ValueError("Critic loss requires a non-empty matching mask")
    value = F.smooth_l1_loss(output.values[mask], targets.returns[mask])
    wdl = F.cross_entropy(output.wdl_logits[mask], targets.wdl_class[mask].long())
    crown = F.smooth_l1_loss(
        output.crown_difference[mask], targets.crown_difference[mask]
    )
    tower_hp = F.smooth_l1_loss(
        output.tower_hp_difference[mask], targets.tower_hp_difference[mask]
    )
    future_damage = F.smooth_l1_loss(
        output.future_damage[mask], targets.future_damage[mask]
    )
    auxiliary = wdl + crown + tower_hp + future_damage
    total = value_coefficient * value + auxiliary_coefficient * auxiliary
    return CriticLoss(total, value, wdl, crown, tower_hp, future_damage)


def explained_variance(values: Tensor, returns: Tensor, mask: Tensor) -> Tensor:
    selected_values = values[mask.bool()].float()
    selected_returns = returns[mask.bool()].float()
    variance = torch.var(selected_returns, unbiased=False)
    if float(variance) <= 1e-12:
        return variance.new_zeros(())
    return 1.0 - torch.var(
        selected_returns - selected_values, unbiased=False
    ) / variance
