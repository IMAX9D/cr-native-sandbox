"""Named parameter-stage contracts for safe expert PPO unfreezing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from .critic import ExpertActorCritic


StageName = Literal["stage1_critic", "stage2_reaction", "stage2_heads", "stage3_full", "stage4_expand"]


@dataclass(frozen=True)
class ParameterGroupSpec:
    name: str
    prefixes: tuple[str, ...]
    learning_rate: float
    weight_decay: float = 0.0


CRITIC_SPEC = ParameterGroupSpec("critic", ("critic.",), 3e-4, 1e-4)
ACTOR_SPECS = {
    "entity": ParameterGroupSpec(
        "actor_entity",
        ("actor_adapter.actor.entity_relation_embedding.",
         "actor_adapter.actor.entity_encoder."),
        1.0e-6,
    ),
    "spatial": ParameterGroupSpec(
        "actor_spatial",
        ("actor_adapter.actor.spatial.", "actor_adapter.actor.cell_features."),
        7.5e-7,
    ),
    "recurrent": ParameterGroupSpec(
        "actor_recurrent", ("actor_adapter.actor.recurrent.",), 3.0e-7
    ),
    "timing": ParameterGroupSpec(
        "actor_timing", ("actor_adapter.actor.rate_head.",), 2.0e-6
    ),
    "action_heads": ParameterGroupSpec(
        "actor_action_heads",
        (
            "actor_adapter.actor.action_kind_head.",
            "actor_adapter.actor.card_query.",
            "actor_adapter.actor.card_key.",
            "actor_adapter.actor.card_bias.",
            "actor_adapter.actor.position_query.",
            "actor_adapter.actor.ability_query.",
            "actor_adapter.actor.ability_key.",
            "actor_adapter.actor.ability_bias.",
            "actor_adapter.actor.ability_position_query.",
        ),
        2.0e-7,
    ),
    "embeddings": ParameterGroupSpec(
        "actor_embeddings",
        (
            "actor_adapter.actor.card_embedding.",
            "actor_adapter.actor.ability_embedding.",
            "actor_adapter.actor.card_context.",
            "actor_adapter.actor.scalar.",
        ),
        1.0e-7,
    ),
}


def stage_specs(stage: StageName) -> list[ParameterGroupSpec]:
    if stage == "stage1_critic":
        return [CRITIC_SPEC]
    if stage == "stage2_reaction":
        return [
            ParameterGroupSpec("critic", ("critic.",), 1e-4, 1e-4),
            ACTOR_SPECS["entity"], ACTOR_SPECS["spatial"],
            ACTOR_SPECS["recurrent"], ACTOR_SPECS["timing"],
        ]
    if stage == "stage2_heads":
        return [*stage_specs("stage2_reaction"), ACTOR_SPECS["action_heads"]]
    if stage in ("stage3_full", "stage4_expand"):
        factor = 0.65 if stage == "stage4_expand" else 1.0
        return [
            ParameterGroupSpec("critic", ("critic.",), 5e-5, 1e-4),
            ParameterGroupSpec("actor_entity", ACTOR_SPECS["entity"].prefixes, 7.5e-7 * factor),
            ParameterGroupSpec("actor_spatial", ACTOR_SPECS["spatial"].prefixes, 7.5e-7 * factor),
            ParameterGroupSpec("actor_recurrent", ACTOR_SPECS["recurrent"].prefixes, 3e-7 * factor),
            ParameterGroupSpec("actor_timing", ACTOR_SPECS["timing"].prefixes, 1e-6 * factor),
            ParameterGroupSpec("actor_action_heads", ACTOR_SPECS["action_heads"].prefixes, 5e-7 * factor),
            ParameterGroupSpec("actor_embeddings", ACTOR_SPECS["embeddings"].prefixes, 1e-7 * factor),
        ]
    raise ValueError(f"unknown curriculum stage: {stage}")


def configure_stage(model: ExpertActorCritic, stage: StageName) -> dict[str, object]:
    specs = stage_specs(stage)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    assigned: dict[str, str] = {}
    trainable = 0
    for name, parameter in model.named_parameters():
        matches = [spec for spec in specs if name.startswith(spec.prefixes)]
        if len(matches) > 1:
            raise RuntimeError(f"parameter assigned to multiple groups: {name}")
        if matches:
            parameter.requires_grad_(True)
            assigned[name] = matches[0].name
            trainable += parameter.numel()
    if not assigned:
        raise RuntimeError("curriculum stage produced no trainable parameters")
    return {
        "stage": stage,
        "trainable_parameters": trainable,
        "trainable_names": sorted(assigned),
        "parameter_groups": {name: group for name, group in sorted(assigned.items())},
    }


def build_optimizer(
    model: ExpertActorCritic,
    stage: StageName,
) -> tuple[torch.optim.AdamW, dict[str, str]]:
    report = configure_stage(model, stage)
    mapping = dict(report["parameter_groups"])
    groups = []
    for spec in stage_specs(stage):
        parameters = [
            parameter for name, parameter in model.named_parameters()
            if parameter.requires_grad and mapping.get(name) == spec.name
        ]
        if parameters:
            groups.append({
                "params": parameters,
                "lr": spec.learning_rate,
                "weight_decay": spec.weight_decay,
                "group_name": spec.name,
            })
    optimizer = torch.optim.AdamW(groups, eps=1e-5)
    return optimizer, mapping
