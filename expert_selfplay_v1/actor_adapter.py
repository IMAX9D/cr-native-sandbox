"""Non-destructive access to expert Actor features for PPO wrappers."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from expert_v1.training_v1.model import (
    ExpertPolicyOutput,
    ExpertPolicyWithFeatures,
    RecurrentExpertPolicy,
)


def actor_state_digest(actor: RecurrentExpertPolicy) -> str:
    digest = hashlib.sha256()
    for name, value in actor.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class ExpertActorAdapter(nn.Module):
    def __init__(self, actor: RecurrentExpertPolicy) -> None:
        super().__init__()
        self.actor = actor

    def forward_with_features(self, **values: Any) -> ExpertPolicyWithFeatures:
        return self.actor.forward_with_features(**values)

    def export_actor_state(self) -> OrderedDict[str, Tensor]:
        return OrderedDict(
            (name, value.detach().cpu().clone())
            for name, value in self.actor.state_dict().items()
        )


def assert_actor_equivalence(
    actor: RecurrentExpertPolicy,
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    before_keys = tuple(actor.state_dict().keys())
    before_digest = actor_state_digest(actor)
    direct: ExpertPolicyOutput = actor.forward_sequence(**dict(inputs))
    featured = actor.forward_with_features(**dict(inputs))
    for name in ExpertPolicyOutput._fields:
        left = getattr(direct, name)
        right = getattr(featured.output, name)
        if isinstance(left, tuple):
            if any(not torch.equal(a, b) for a, b in zip(left, right, strict=True)):
                raise RuntimeError(f"Actor output changed through adapter: {name}")
        elif not torch.equal(left, right):
            raise RuntimeError(f"Actor output changed through adapter: {name}")
    after_keys = tuple(actor.state_dict().keys())
    after_digest = actor_state_digest(actor)
    if before_keys != after_keys or before_digest != after_digest:
        raise RuntimeError("Actor adapter changed the expert state_dict contract")
    if featured.pre_head_latent.shape[:2] != direct.rate_logits.shape:
        raise RuntimeError("captured Actor latent has the wrong prefix shape")
    return {
        "state_dict_keys": len(before_keys),
        "actor_sha256": before_digest,
        "latent_shape": list(featured.pre_head_latent.shape),
    }
