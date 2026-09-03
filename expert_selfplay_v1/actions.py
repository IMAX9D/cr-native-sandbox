"""Hierarchical marked actions for the expert Actor heads."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from expert_v1.training_v1.model import ExpertPolicyConfig, ExpertPolicyOutput
from .hazard import (
    HazardLogProb,
    lambda_from_logits,
    marked_hazard_entropy,
    marked_hazard_kl,
    marked_hazard_log_prob,
)


@dataclass(frozen=True)
class ExpertActionMasks:
    action_kind: Tensor
    cards: Tensor
    positions: Tensor
    abilities: Tensor
    ability_positions: Tensor
    ability_requires_target: Tensor


@dataclass(frozen=True)
class RecordedExpertAction:
    event_happened: Tensor
    action_kind: Tensor
    card_slot: Tensor
    position: Tensor
    ability_slot: Tensor
    ability_position: Tensor
    ability_requires_target: Tensor


@dataclass(frozen=True)
class EvaluatedExpertAction:
    log_prob: HazardLogProb
    entropy: Tensor
    mark_log_prob: Tensor
    mark_entropy: Tensor


def _distribution(logits: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    if logits.shape != mask.shape:
        raise ValueError("categorical logits/mask shapes differ")
    valid = mask.bool()
    if bool((~valid.any(dim=-1)).any()):
        raise ValueError("categorical action layer is all-invalid")
    masked = logits.float().masked_fill(~valid, -torch.inf)
    log_probability = torch.log_softmax(masked, dim=-1)
    probability = torch.softmax(masked, dim=-1)
    entropy = -(probability * torch.where(valid, log_probability, torch.zeros_like(log_probability))).sum(-1)
    return log_probability, probability, entropy


def _categorical_kl(
    source_logits: Tensor, target_logits: Tensor, mask: Tensor
) -> tuple[Tensor, Tensor]:
    source_logp, source_probability, _entropy = _distribution(source_logits, mask)
    target_logp, _target_probability, _target_entropy = _distribution(target_logits, mask)
    value = source_probability * torch.where(
        mask.bool(), source_logp - target_logp, torch.zeros_like(source_logp)
    )
    return value.sum(dim=-1), source_probability


def _selected(
    log_probability: Tensor,
    index: Tensor,
    mask: Tensor,
    *,
    active: Tensor,
) -> Tensor:
    safe = index.long().clamp(0, log_probability.shape[-1] - 1)
    chosen_valid = mask.gather(-1, safe.unsqueeze(-1)).squeeze(-1).bool()
    if bool((active.bool() & ~chosen_valid).any()):
        raise ValueError("recorded action is invalid under its pre-action mask")
    selected = log_probability.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return torch.where(active.bool(), selected, torch.zeros_like(selected))


def _conditional_mask(mask: Tensor, parent_valid: Tensor) -> Tensor:
    result = mask.bool().clone()
    if result.shape[:-1] != parent_valid.shape:
        raise ValueError("conditional mask parent shape differs")
    missing = parent_valid.bool() & ~result.any(dim=-1)
    if bool(missing.any()):
        raise ValueError("valid parent action has an all-invalid child layer")
    inactive = ~parent_valid.bool()
    if bool(inactive.any()):
        result[inactive] = False
        result[..., 0] |= inactive
    return result


def evaluate_expert_action(
    *,
    output: ExpertPolicyOutput,
    config: ExpertPolicyConfig,
    masks: ExpertActionMasks,
    action: RecordedExpertAction,
    delta_ticks: Tensor,
) -> EvaluatedExpertAction:
    prefix = output.rate_logits.shape
    tensors = (
        action.event_happened, action.action_kind, action.card_slot,
        action.position, action.ability_slot, action.ability_position,
        action.ability_requires_target, delta_ticks,
    )
    if any(value.shape != prefix for value in tensors):
        raise ValueError("recorded action tensors must match Actor [batch,time]")
    can_act = masks.action_kind.any(dim=-1)
    action_kind_mask = _conditional_mask(masks.action_kind, can_act)
    kind_logp, kind_p, kind_entropy = _distribution(
        output.action_kind_logits, action_kind_mask
    )
    card_mask = _conditional_mask(masks.cards, masks.action_kind[..., 0])
    ability_mask = _conditional_mask(masks.abilities, masks.action_kind[..., 1])
    card_logp, card_p, card_entropy = _distribution(output.card_logits, card_mask)
    ability_logp, ability_p, ability_entropy = _distribution(
        output.ability_logits, ability_mask
    )

    card_position_entropy = []
    card_position_logp = []
    for slot in range(output.card_logits.shape[-1]):
        position_mask = _conditional_mask(
            masks.positions[..., slot, :], masks.cards[..., slot]
        )
        logp, _p, entropy = _distribution(
            output.position_logits[..., slot, :], position_mask
        )
        card_position_logp.append(logp)
        card_position_entropy.append(entropy)
    card_position_logp_t = torch.stack(card_position_logp, dim=-2)
    card_position_entropy_t = torch.stack(card_position_entropy, dim=-1)

    ability_position_entropy = []
    ability_position_logp = []
    for slot in range(output.ability_logits.shape[-1]):
        position_parent = masks.abilities[..., slot] & masks.ability_requires_target[..., slot]
        position_mask = _conditional_mask(
            masks.ability_positions[..., slot, :], position_parent
        )
        logp, _p, entropy = _distribution(
            output.ability_position_logits[..., slot, :],
            position_mask,
        )
        ability_position_logp.append(logp)
        ability_position_entropy.append(entropy)
    ability_position_logp_t = torch.stack(ability_position_logp, dim=-2)
    ability_position_entropy_t = torch.stack(ability_position_entropy, dim=-1)

    event = action.event_happened.bool()
    normal = action.action_kind.long() == 0
    ability = ~normal
    kind_selected = _selected(
        kind_logp, action.action_kind, action_kind_mask, active=event
    )
    card_selected = _selected(
        card_logp, action.card_slot, card_mask, active=event & normal
    )
    card_slot = action.card_slot.long().clamp(0, output.card_logits.shape[-1] - 1)
    selected_card_position_logp = card_position_logp_t.gather(
        -2,
        card_slot.unsqueeze(-1).unsqueeze(-1).expand(*prefix, 1, card_position_logp_t.shape[-1]),
    ).squeeze(-2)
    safe_position_masks = torch.stack([
        _conditional_mask(masks.positions[..., slot, :], masks.cards[..., slot])
        for slot in range(masks.cards.shape[-1])
    ], dim=-2)
    selected_card_position_mask = safe_position_masks.gather(
        -2,
        card_slot.unsqueeze(-1).unsqueeze(-1).expand(*prefix, 1, masks.positions.shape[-1]),
    ).squeeze(-2)
    position_selected = _selected(
        selected_card_position_logp,
        action.position,
        selected_card_position_mask,
        active=event & normal,
    )

    ability_selected = _selected(
        ability_logp, action.ability_slot, ability_mask, active=event & ability
    )
    ability_slot = action.ability_slot.long().clamp(0, output.ability_logits.shape[-1] - 1)
    selected_ability_position_logp = ability_position_logp_t.gather(
        -2,
        ability_slot.unsqueeze(-1).unsqueeze(-1).expand(
            *prefix, 1, ability_position_logp_t.shape[-1]
        ),
    ).squeeze(-2)
    safe_ability_position_masks = torch.stack([
        _conditional_mask(
            masks.ability_positions[..., slot, :],
            masks.abilities[..., slot] & masks.ability_requires_target[..., slot],
        )
        for slot in range(masks.abilities.shape[-1])
    ], dim=-2)
    selected_ability_position_mask = safe_ability_position_masks.gather(
        -2,
        ability_slot.unsqueeze(-1).unsqueeze(-1).expand(
            *prefix, 1, masks.ability_positions.shape[-1]
        ),
    ).squeeze(-2)
    ability_position_selected = _selected(
        selected_ability_position_logp,
        action.ability_position,
        selected_ability_position_mask,
        active=event & ability & action.ability_requires_target.bool(),
    )
    ability_position_selected = torch.where(
        action.ability_requires_target.bool(), ability_position_selected,
        torch.zeros_like(ability_position_selected),
    )
    normal_mark = kind_selected + card_selected + position_selected
    ability_mark = kind_selected + ability_selected + ability_position_selected
    mark_log_prob = torch.where(normal, normal_mark, ability_mark)
    mark_log_prob = torch.where(event, mark_log_prob, torch.zeros_like(mark_log_prob))

    normal_entropy = card_entropy + (card_p * card_position_entropy_t).sum(-1)
    targeted_ability_entropy = ability_entropy + (
        ability_p * masks.ability_requires_target.float() * ability_position_entropy_t
    ).sum(-1)
    mark_entropy = kind_entropy + kind_p[..., 0] * normal_entropy
    mark_entropy = mark_entropy + kind_p[..., 1] * targeted_ability_entropy
    hazard = marked_hazard_log_prob(
        lambda_per_second=lambda_from_logits(output.rate_logits, config.lambda_max),
        delta_ticks=delta_ticks,
        event_happened=event,
        can_act=can_act,
        mark_log_prob=mark_log_prob,
        tick_hz=1.0 / config.native_tick_seconds,
    )
    entropy = marked_hazard_entropy(
        lambda_per_second=lambda_from_logits(output.rate_logits, config.lambda_max),
        delta_ticks=delta_ticks,
        can_act=can_act,
        mark_entropy=mark_entropy,
        tick_hz=1.0 / config.native_tick_seconds,
    )
    return EvaluatedExpertAction(hazard, entropy, mark_log_prob, mark_entropy)


def expert_policy_kl(
    *,
    source: ExpertPolicyOutput,
    target: ExpertPolicyOutput,
    config: ExpertPolicyConfig,
    masks: ExpertActionMasks,
    delta_ticks: Tensor,
) -> Tensor:
    can_act = masks.action_kind.any(dim=-1)
    action_kind_mask = _conditional_mask(masks.action_kind, can_act)
    kind_kl, kind_probability = _categorical_kl(
        source.action_kind_logits, target.action_kind_logits, action_kind_mask
    )
    card_mask = _conditional_mask(masks.cards, masks.action_kind[..., 0])
    card_kl, card_probability = _categorical_kl(
        source.card_logits, target.card_logits, card_mask
    )
    card_position_kl = []
    for slot in range(source.card_logits.shape[-1]):
        position_mask = _conditional_mask(
            masks.positions[..., slot, :], masks.cards[..., slot]
        )
        value, _probability = _categorical_kl(
            source.position_logits[..., slot, :],
            target.position_logits[..., slot, :],
            position_mask,
        )
        card_position_kl.append(value)
    card_position = torch.stack(card_position_kl, dim=-1)
    normal_kl = card_kl + (card_probability * card_position).sum(dim=-1)

    ability_mask = _conditional_mask(masks.abilities, masks.action_kind[..., 1])
    ability_kl, ability_probability = _categorical_kl(
        source.ability_logits, target.ability_logits, ability_mask
    )
    ability_position_kl = []
    for slot in range(source.ability_logits.shape[-1]):
        position_mask = _conditional_mask(
            masks.ability_positions[..., slot, :],
            masks.abilities[..., slot] & masks.ability_requires_target[..., slot],
        )
        value, _probability = _categorical_kl(
            source.ability_position_logits[..., slot, :],
            target.ability_position_logits[..., slot, :],
            position_mask,
        )
        ability_position_kl.append(value)
    ability_position = torch.stack(ability_position_kl, dim=-1)
    ability_kl = ability_kl + (
        ability_probability
        * masks.ability_requires_target.float()
        * ability_position
    ).sum(dim=-1)
    mark_kl = kind_kl + kind_probability[..., 0] * normal_kl
    mark_kl = mark_kl + kind_probability[..., 1] * ability_kl
    return marked_hazard_kl(
        source_lambda=lambda_from_logits(source.rate_logits, config.lambda_max),
        target_lambda=lambda_from_logits(target.rate_logits, config.lambda_max),
        delta_ticks=delta_ticks,
        can_act=can_act,
        mark_kl=mark_kl,
        tick_hz=1.0 / config.native_tick_seconds,
    )
