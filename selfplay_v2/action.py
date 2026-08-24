"""Numerically stable continuous-time action timing primitives."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


NATIVE_TICK_SECONDS = 0.05
DEFAULT_LAMBDA_MAX = 20.0


@dataclass(frozen=True)
class RateDistribution:
    rate: Tensor
    play_probability: Tensor
    log_play: Tensor
    log_no_play: Tensor
    entropy: Tensor


def initial_rate_bias(lambda_initial: float, lambda_max: float) -> float:
    if not 0.0 < lambda_initial < lambda_max:
        raise ValueError("lambda_initial must be in (0, lambda_max)")
    probability = lambda_initial / lambda_max
    return math.log(probability / (1.0 - probability))


def rate_distribution(
    rate_logits: Tensor,
    *,
    lambda_max: float = DEFAULT_LAMBDA_MAX,
    tick_seconds: float = NATIVE_TICK_SECONDS,
) -> RateDistribution:
    if lambda_max <= 0.0 or tick_seconds <= 0.0:
        raise ValueError("lambda_max and tick_seconds must be positive")
    rate = torch.sigmoid(rate_logits) * lambda_max
    exposure = rate * tick_seconds
    log_no_play = -exposure
    probability = -torch.expm1(-exposure)
    tiny = torch.finfo(rate_logits.dtype).tiny
    log_play = torch.log(probability.clamp_min(tiny))
    entropy = -(
        probability * log_play
        + torch.exp(log_no_play) * log_no_play
    )
    return RateDistribution(
        rate=rate,
        play_probability=probability,
        log_play=log_play,
        log_no_play=log_no_play,
        entropy=entropy,
    )


def timing_log_probability(
    distribution: RateDistribution,
    *,
    play_now: Tensor,
    timing_valid: Tensor,
) -> Tensor:
    value = torch.where(
        play_now,
        distribution.log_play,
        distribution.log_no_play,
    )
    return torch.where(timing_valid, value, torch.zeros_like(value))


def sample_play_now(
    distribution: RateDistribution,
    *,
    timing_valid: Tensor,
    deterministic_uniform: Tensor | None = None,
) -> Tensor:
    if deterministic_uniform is None:
        uniform = torch.rand_like(distribution.play_probability)
    else:
        if deterministic_uniform.shape != distribution.play_probability.shape:
            raise ValueError("deterministic_uniform shape mismatch")
        uniform = deterministic_uniform
    return timing_valid & (uniform < distribution.play_probability)
