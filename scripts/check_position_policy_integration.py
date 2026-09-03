"""Compare integrated position scoring with the isolated experiment on one fixed batch."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import torch

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from expert_v1.training_v1.model import ExpertPolicyConfig, RecurrentExpertPolicy, configure_position_precision
from expert_v1.training_v1.losses import behaviour_cloning_loss
from expert_v1.training_v1.train import _atomic_json
from scripts.experiment_expert_position_stability import forward_position_variant


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    legacy = replace(ExpertPolicyConfig(**checkpoint["model_config"]), position_head_fp32=False, position_logit_softcap=None)
    stable = replace(legacy, position_head_fp32=True, position_logit_softcap=20.)
    configure_position_precision(stable)
    raw = torch.load(args.fixture, map_location="cpu", weights_only=True)
    selected = (raw["loss_mask"] & raw["position_label_mask"]).any(1).nonzero().flatten()[:2]
    if len(selected) != 2:
        raise ValueError("fixture needs two actor windows with supervised positions")
    device = torch.device(args.device)
    batch = {key: value.index_select(0, selected).to(device) for key, value in raw.items()}
    model = RecurrentExpertPolicy(legacy).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.train()
    with torch.autocast(device.type, dtype=torch.bfloat16):
        output = forward_position_variant(model, batch, "fp32_softcap20")
        loss, _ = behaviour_cloning_loss(output, batch, stable)
    loss.backward()
    reference_loss = float(loss.detach())
    reference_grad = {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters() if p.grad is not None}
    b, t = (batch["loss_mask"] & batch["position_label_mask"]).nonzero(as_tuple=True)
    cards = batch["card_slot"][b, t]
    reference_positions = output.position_logits[b, t, cards].detach().cpu()
    reference_cards = output.card_logits.detach().cpu()
    del output, loss
    model.zero_grad(set_to_none=True)
    model.config = stable
    with torch.autocast(device.type, dtype=torch.bfloat16):
        output = model.forward_batch(batch, supervised_positions=True)
        loss, _ = behaviour_cloning_loss(output, batch, stable)
    loss.backward()
    numerator = denominator = maximum = 0.
    for name, parameter in model.named_parameters():
        if (parameter.grad is not None) != (name in reference_grad):
            raise RuntimeError(f"gradient presence changed: {name}")
        if parameter.grad is not None:
            grad = parameter.grad.detach().cpu()
            delta = grad - reference_grad[name]
            numerator += float(delta.square().sum())
            denominator += float(reference_grad[name].square().sum())
            maximum = max(maximum, float(delta.abs().max()))
    result = {"source_step": checkpoint["global_step"], "device": str(device),
        "loss_reference": reference_loss, "loss_integrated": float(loss.detach()),
        "position_max_abs_error": float((output.position_logits[b, t, cards].detach().cpu() - reference_positions).abs().max()),
        "card_max_abs_error": float((output.card_logits.detach().cpu() - reference_cards).abs().max()),
        "gradient_relative_l2_error": (numerator / max(denominator, 1e-30)) ** .5,
        "gradient_max_abs_error": maximum, "position_labels": len(b),
        "parameters": sum(p.numel() for p in model.parameters())}
    if (abs(result["loss_reference"] - result["loss_integrated"]) >= 1e-5
            or result["position_max_abs_error"] >= 1e-4
            or result["card_max_abs_error"] != 0
            or result["gradient_relative_l2_error"] >= .01):
        _atomic_json(args.output, {"ok": False, **result})
        raise RuntimeError(f"integrated policy diverges from experiment: {result}")
    _atomic_json(args.output, {"ok": True, **result})
    print(json.dumps({"event": "gpu_integration_parity_passed", **result}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
