"""Fail-closed PPO update admission and one-shot retry policy."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Mapping


@dataclass(frozen=True)
class UpdateGuardConfig:
    target_kl: float = 0.008
    hard_kl: float = 0.015
    target_clip_fraction: float = 0.25
    hard_clip_fraction: float = 0.40
    maximum_rate_ratio: float = 2.0
    minimum_rate_ratio: float = 0.50


@dataclass(frozen=True)
class UpdateGuardDecision:
    action: Literal["accept", "retry", "halt"]
    reasons: tuple[str, ...]
    actor_lr_multiplier: float
    ppo_epochs: int | None
    bc_kl_multiplier: float


def evaluate_update(
    metrics: Mapping[str, float],
    *,
    retry_attempt: int,
    config: UpdateGuardConfig = UpdateGuardConfig(),
) -> UpdateGuardDecision:
    required = (
        "loss", "approx_update_kl", "clip_fraction",
        "rate_mean_before", "rate_mean_after",
    )
    reasons: list[str] = []
    for name in required:
        value = float(metrics.get(name, math.nan))
        if not math.isfinite(value):
            reasons.append(f"{name}:nonfinite")
    if reasons:
        return UpdateGuardDecision("halt", tuple(reasons), 0.0, None, 0.0)
    kl = float(metrics["approx_update_kl"])
    clip = float(metrics["clip_fraction"])
    if kl > config.hard_kl:
        reasons.append("kl:hard_limit")
    elif kl > config.target_kl:
        reasons.append("kl:target_exceeded")
    if clip > config.hard_clip_fraction:
        reasons.append("clip_fraction:hard_limit")
    elif clip > config.target_clip_fraction:
        reasons.append("clip_fraction:target_exceeded")
    before = float(metrics["rate_mean_before"])
    after = float(metrics["rate_mean_after"])
    ratio = after / max(before, 1e-8)
    if ratio > config.maximum_rate_ratio or ratio < config.minimum_rate_ratio:
        reasons.append("rate_distribution:shift")
    if not reasons:
        return UpdateGuardDecision("accept", (), 1.0, None, 1.0)
    if retry_attempt == 0:
        return UpdateGuardDecision("retry", tuple(reasons), 0.5, 1, 2.0)
    return UpdateGuardDecision("halt", tuple(reasons), 0.0, None, 0.0)
