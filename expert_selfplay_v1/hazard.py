"""Numerically stable marked-hazard policy mathematics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class HazardLogProb:
    total: Tensor
    timing: Tensor
    mark: Tensor
    event_probability: Tensor


def lambda_from_logits(rate_logits: Tensor, lambda_max: float) -> Tensor:
    if lambda_max <= 0:
        raise ValueError("lambda_max must be positive")
    return torch.sigmoid(rate_logits.float()) * float(lambda_max)


def _terms(
    lambda_per_second: Tensor,
    delta_ticks: Tensor,
    can_act: Tensor,
    *,
    tick_hz: float,
) -> tuple[Tensor, Tensor, Tensor]:
    if tick_hz <= 0:
        raise ValueError("tick_hz must be positive")
    if lambda_per_second.shape != delta_ticks.shape or can_act.shape != delta_ticks.shape:
        raise ValueError("hazard tensors must share one shape")
    if bool((delta_ticks < 1).any()):
        raise ValueError("delta_ticks must be positive")
    if bool((lambda_per_second < 0).any()):
        raise ValueError("lambda must be nonnegative")
    exposure = lambda_per_second.float() * delta_ticks.float() / float(tick_hz)
    exposure = torch.where(can_act.bool(), exposure, torch.zeros_like(exposure))
    event_probability = -torch.expm1(-exposure)
    log_wait = -exposure
    # A zero-rate event is impossible and correctly receives -inf.
    log_event = torch.log(-torch.expm1(-exposure))
    return event_probability, log_wait, log_event


def marked_hazard_log_prob(
    *,
    lambda_per_second: Tensor,
    delta_ticks: Tensor,
    event_happened: Tensor,
    can_act: Tensor,
    mark_log_prob: Tensor,
    tick_hz: float = 20.0,
) -> HazardLogProb:
    if event_happened.shape != delta_ticks.shape or mark_log_prob.shape != delta_ticks.shape:
        raise ValueError("event/mark tensors must match delta_ticks")
    event = event_happened.bool()
    legal = can_act.bool()
    if bool((event & ~legal).any()):
        raise ValueError("an action event cannot occur when can_act is false")
    probability, log_wait, log_event = _terms(
        lambda_per_second, delta_ticks, legal, tick_hz=tick_hz
    )
    timing = torch.where(event, log_event, log_wait)
    timing = torch.where(legal, timing, torch.zeros_like(timing))
    mark = torch.where(event, mark_log_prob.float(), torch.zeros_like(mark_log_prob.float()))
    return HazardLogProb(
        total=timing + mark,
        timing=timing,
        mark=mark,
        event_probability=probability,
    )


def marked_hazard_entropy(
    *,
    lambda_per_second: Tensor,
    delta_ticks: Tensor,
    can_act: Tensor,
    mark_entropy: Tensor,
    tick_hz: float = 20.0,
) -> Tensor:
    probability, _log_wait, _log_event = _terms(
        lambda_per_second, delta_ticks, can_act.bool(), tick_hz=tick_hz
    )
    epsilon = torch.finfo(probability.dtype).eps
    p = probability.clamp(epsilon, 1.0 - epsilon)
    timing_entropy = -(p * torch.log(p) + (1.0 - p) * torch.log1p(-p))
    total = timing_entropy + probability * mark_entropy.float()
    return torch.where(can_act.bool(), total, torch.zeros_like(total))


def marked_hazard_kl(
    *,
    source_lambda: Tensor,
    target_lambda: Tensor,
    delta_ticks: Tensor,
    can_act: Tensor,
    mark_kl: Tensor,
    tick_hz: float = 20.0,
) -> Tensor:
    source, _a, _b = _terms(source_lambda, delta_ticks, can_act.bool(), tick_hz=tick_hz)
    target, _c, _d = _terms(target_lambda, delta_ticks, can_act.bool(), tick_hz=tick_hz)
    epsilon = torch.finfo(source.dtype).eps
    p = source.clamp(epsilon, 1.0 - epsilon)
    q = target.clamp(epsilon, 1.0 - epsilon)
    timing = p * (torch.log(p) - torch.log(q))
    timing += (1.0 - p) * (torch.log1p(-p) - torch.log1p(-q))
    total = timing + source * mark_kl.float()
    return torch.where(can_act.bool(), total, torch.zeros_like(total))
