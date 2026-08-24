"""Strict v0.1 P010 to v0.2 initialization boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from training.run_contract import CHECKPOINT_KIND as V1_CHECKPOINT_KIND
from training.run_contract import state_dict_digest
from .model import ContinuousRatePolicyValueNet


BACKBONE_PREFIXES = (
    "spatial.",
    "public_scalar.",
    "recurrent.",
    "privileged.",
    "value_head.",
)
ACTOR_PREFIXES = (
    "rate_head.",
    "card_head.",
    "position_map.",
    "position_context.",
)


def _is_backbone(name: str) -> bool:
    return name.startswith(BACKBONE_PREFIXES)


def initialize_model(
    model: ContinuousRatePolicyValueNet,
    *,
    mode: str,
    parent_checkpoint: Path | None = None,
) -> dict[str, Any]:
    if mode not in ("scratch", "backbone_only"):
        raise ValueError("initialization mode must be scratch or backbone_only")
    before_actor = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name.startswith(ACTOR_PREFIXES)
    }
    if mode == "scratch":
        if parent_checkpoint is not None:
            raise ValueError("scratch initialization cannot accept a parent")
        return {
            "mode": mode,
            "parent_checkpoint": None,
            "parent_model_digest": None,
            "copied_tensors": [],
            "actor_tensors_copied": [],
            "model_digest": state_dict_digest(model.state_dict()),
        }
    if parent_checkpoint is None or not parent_checkpoint.is_file():
        raise FileNotFoundError("backbone_only requires a P010 checkpoint")
    checkpoint = torch.load(
        parent_checkpoint, map_location="cpu", weights_only=False
    )
    if checkpoint.get("kind") != V1_CHECKPOINT_KIND:
        raise RuntimeError("warm-start parent is not a v0.1 checkpoint")
    parent: Mapping[str, torch.Tensor] = checkpoint["model"]
    current = model.state_dict()
    copied: list[str] = []
    with torch.no_grad():
        for name, value in current.items():
            if not _is_backbone(name):
                continue
            source = parent.get(name)
            if source is None or source.shape != value.shape:
                raise RuntimeError(f"backbone migration mismatch: {name}")
            value.copy_(source.to(device=value.device, dtype=value.dtype))
            copied.append(name)
    model.load_state_dict(current)
    changed_actor = [
        name
        for name, before in before_actor.items()
        if not torch.equal(model.state_dict()[name].detach().cpu(), before)
    ]
    if changed_actor:
        raise RuntimeError(
            "backbone migration modified v0.2 actor tensors: "
            + ",".join(changed_actor)
        )
    parent_digest = state_dict_digest(parent)
    expected_digest = checkpoint.get("current_model_digest")
    if expected_digest is not None and str(expected_digest) != parent_digest:
        raise RuntimeError("P010 model digest mismatch")
    return {
        "mode": mode,
        "parent_checkpoint": str(parent_checkpoint.resolve()),
        "parent_model_digest": parent_digest,
        "copied_tensors": copied,
        "actor_tensors_copied": changed_actor,
        "model_digest": state_dict_digest(model.state_dict()),
    }
