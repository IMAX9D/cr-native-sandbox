"""Unified-ratio PPO objective for marked-hazard expert actions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class ExpertPpoLoss:
    total: Tensor
    policy: Tensor
    value: Tensor
    entropy: Tensor
    bc_kl: Tensor
    approx_update_kl: Tensor
    clip_fraction: Tensor


def recurrent_ppo_loss(
    *,
    new_log_prob: Tensor,
    old_log_prob: Tensor,
    advantages: Tensor,
    values: Tensor,
    returns: Tensor,
    joint_entropy: Tensor,
    bc_kl: Tensor,
    loss_mask: Tensor,
    clip_epsilon: float = 0.10,
    value_coefficient: float = 0.50,
    entropy_coefficient: float = 0.01,
    bc_kl_coefficient: float = 1.0,
) -> ExpertPpoLoss:
    shape = new_log_prob.shape
    tensors = (
        old_log_prob, advantages, values, returns, joint_entropy, bc_kl, loss_mask
    )
    if any(value.shape != shape for value in tensors):
        raise ValueError("PPO tensors must share one shape")
    mask = loss_mask.bool()
    if not bool(mask.any()):
        raise ValueError("PPO loss mask is empty")
    log_ratio = new_log_prob[mask].float() - old_log_prob[mask].float()
    ratio = torch.exp(log_ratio)
    normalized_advantage = advantages[mask].float()
    unclipped = ratio * normalized_advantage
    clipped = torch.clamp(
        ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon
    ) * normalized_advantage
    policy = -torch.minimum(unclipped, clipped).mean()
    value = F.smooth_l1_loss(values[mask].float(), returns[mask].float())
    entropy = joint_entropy[mask].float().mean()
    mean_bc_kl = bc_kl[mask].float().mean()
    total = (
        policy + value_coefficient * value
        - entropy_coefficient * entropy
        + bc_kl_coefficient * mean_bc_kl
    )
    approx_update_kl = ((ratio - 1.0) - log_ratio).mean()
    clip_fraction = ((ratio - 1.0).abs() > clip_epsilon).float().mean()
    return ExpertPpoLoss(
        total, policy, value, entropy, mean_bc_kl,
        approx_update_kl, clip_fraction,
    )
