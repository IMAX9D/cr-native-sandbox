"""Create the required local RTX-3080 attestation for a Stage-2 Canary export."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import torch

from expert_v1.training_v1.model import ExpertPolicyConfig, RecurrentExpertPolicy, configure_position_precision
from expert_selfplay_v1.stage2_training import EXPERT_INFERENCE_KIND, _state_digest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inputs(config: ExpertPolicyConfig, device: torch.device) -> dict[str, object]:
    floating = torch.float16
    return {
        "grid": torch.zeros(1, 1, config.grid_channels, 32, 18, device=device, dtype=floating),
        "public_scalars": torch.zeros(1, 1, config.public_scalar_size, device=device, dtype=floating),
        "own_deck_tokens": torch.ones(1, 1, 8, device=device, dtype=torch.long),
        "hand_tokens": torch.ones(1, 1, 4, device=device, dtype=torch.long),
        "next_card_token": torch.ones(1, 1, device=device, dtype=torch.long),
        "revealed_enemy_tokens": torch.zeros(1, 1, 8, device=device, dtype=torch.long),
        "ability_tokens": torch.zeros(
            1, 1, config.max_ability_slots, device=device, dtype=torch.long
        ),
        "delta_ticks": torch.full((1, 1), 4.0, device=device, dtype=floating),
        "entity_tokens": torch.ones(1, 1, 1, device=device, dtype=torch.long),
        "entity_positions": torch.zeros(1, 1, 1, device=device, dtype=torch.long),
        "entity_relations": torch.zeros(1, 1, 1, device=device, dtype=torch.long),
        "entity_numeric": torch.ones(
            1, 1, 1, config.entity_numeric_size, device=device, dtype=floating
        ),
        "entity_mask": torch.ones(1, 1, 1, device=device, dtype=torch.bool),
    }


def atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        Path(temporary).unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--expert-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("local CUDA GPU is required")
    device_name = torch.cuda.get_device_name(0)
    if "RTX 3080" not in device_name:
        raise RuntimeError(f"formal local gate requires RTX 3080, got {device_name}")
    weights = args.weights.resolve(strict=True)
    manifest = args.expert_manifest.resolve(strict=True)
    payload = torch.load(weights, map_location="cpu", weights_only=False, mmap=True)
    if payload.get("kind") != EXPERT_INFERENCE_KIND:
        raise RuntimeError("unsupported Stage-2 Actor export")
    if payload.get("dataset_manifest_sha256") != sha256_file(manifest):
        raise RuntimeError("Stage-2 Actor export/expert manifest mismatch")
    if _state_digest(payload["model_state"]) != payload.get("actor_state_sha256"):
        raise RuntimeError("Stage-2 Actor export state hash mismatch")
    config = ExpertPolicyConfig(**dict(payload["model_config"]))
    configure_position_precision(config)
    model = RecurrentExpertPolicy(config)
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device="cuda", dtype=torch.float16).eval()
    batch = inputs(config, torch.device("cuda"))
    hidden = tuple(
        value.to(dtype=torch.float16)
        for value in model.initial_hidden(1, device="cuda")
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(args.warmup):
            output = model.forward_sequence(**batch, hidden=hidden)
            hidden = output.hidden
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(args.steps):
            output = model.forward_sequence(**batch, hidden=hidden)
            hidden = output.hidden
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    tensors = (
        output.rate_logits, output.action_kind_logits, output.card_logits,
        output.position_logits, output.ability_logits,
        output.ability_position_logits, *output.hidden,
    )
    if any(not bool(torch.isfinite(value).all()) for value in tensors):
        raise FloatingPointError("local Stage-2 Actor emitted NaN/Inf")
    result: dict[str, object] = {
        "kind": "cr_native_stage2_local_rtx3080_attestation_v1",
        "status": "passed",
        "device": device_name,
        "weights": str(weights),
        "weights_sha256": sha256_file(weights),
        "actor_state_sha256": payload["actor_state_sha256"],
        "policy_version": int(payload["selfplay_policy_version"]),
        "global_update": int(payload["selfplay_global_update"]),
        "parameters": sum(value.numel() for value in model.parameters()),
        "steps": args.steps,
        "milliseconds_per_decision": elapsed * 1000 / args.steps,
        "decisions_per_second": args.steps / elapsed,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "finite": True,
    }
    atomic_json(args.output.resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
