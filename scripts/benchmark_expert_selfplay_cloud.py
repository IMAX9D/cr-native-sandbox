"""Bounded RTX benchmark for the 177M expert Actor and independent Critic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from expert_selfplay_v1.critic import PrivilegedCritic, PrivilegedCriticConfig
from expert_v1.training_v1.model import ExpertPolicyConfig, RecurrentExpertPolicy


def actor_inputs(config: ExpertPolicyConfig, batch: int, device: torch.device):
    entities = 12
    return {
        "grid": torch.zeros(batch, 1, config.grid_channels, 32, 18, device=device, dtype=torch.float16),
        "public_scalars": torch.zeros(batch, 1, config.public_scalar_size, device=device, dtype=torch.float16),
        "own_deck_tokens": torch.arange(1, 9, device=device).view(1, 1, 8).expand(batch, -1, -1),
        "hand_tokens": torch.arange(1, 5, device=device).view(1, 1, 4).expand(batch, -1, -1),
        "next_card_token": torch.full((batch, 1), 5, device=device, dtype=torch.long),
        "revealed_enemy_tokens": torch.zeros(batch, 1, 8, device=device, dtype=torch.long),
        "ability_tokens": torch.zeros(batch, 1, config.max_ability_slots, device=device, dtype=torch.long),
        "delta_ticks": torch.ones(batch, 1, device=device, dtype=torch.float16),
        "entity_tokens": torch.randint(1, config.card_vocab_size, (batch, 1, entities), device=device),
        "entity_positions": torch.randint(0, 576, (batch, 1, entities), device=device),
        "entity_relations": torch.randint(0, 2, (batch, 1, entities), device=device),
        "entity_numeric": torch.rand(batch, 1, entities, 3, device=device, dtype=torch.float16),
        "entity_mask": torch.ones(batch, 1, entities, device=device, dtype=torch.bool),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requires one visible GPU")
    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ExpertPolicyConfig(**checkpoint["model_config"])
    actor = RecurrentExpertPolicy(config)
    actor.load_state_dict(checkpoint["model_state"], strict=True)
    actor.eval().to(device=device, dtype=torch.float16)
    actor_rows = []
    with torch.inference_mode():
        for batch in (1, 2, 4, 8, 16, 32):
            inputs = actor_inputs(config, batch, device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            for _ in range(3):
                output = actor.forward_sequence(**inputs)
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(args.iterations):
                output = actor.forward_sequence(**inputs)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if not torch.isfinite(output.rate_logits).all():
                raise FloatingPointError("Actor benchmark output is non-finite")
            actor_rows.append({
                "batch": batch,
                "milliseconds_per_batch": elapsed * 1000 / args.iterations,
                "decisions_per_second": batch * args.iterations / elapsed,
                "peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
            })
            del inputs, output

    critic = PrivilegedCritic(PrivilegedCriticConfig(
        actor_latent_size=config.hidden_size,
        card_vocab_size=config.card_vocab_size,
        public_grid_channels=config.grid_channels,
        entity_numeric_size=config.entity_numeric_size,
        scalar_size=32,
    )).to(device)
    critic.train()
    critic_rows = []
    for batch in (1, 2, 4, 8):
        steps, entities, private = 16, 24, 24
        values = {
            "actor_latent": torch.randn(batch, steps, config.hidden_size, device=device),
            "grid": torch.randn(batch, steps, config.grid_channels, 32, 18, device=device),
            "entity_tokens": torch.randint(1, config.card_vocab_size, (batch, steps, entities), device=device),
            "entity_positions": torch.randint(0, 576, (batch, steps, entities), device=device),
            "entity_relations": torch.randint(0, 2, (batch, steps, entities), device=device),
            "entity_numeric": torch.rand(batch, steps, entities, config.entity_numeric_size, device=device),
            "entity_mask": torch.ones(batch, steps, entities, device=device, dtype=torch.bool),
            "private_card_tokens": torch.randint(1, config.card_vocab_size, (batch, steps, private), device=device),
            "private_card_owners": torch.randint(0, 2, (batch, steps, private), device=device),
            "private_card_slots": torch.arange(private, device=device).view(1, 1, -1).expand(batch, steps, -1),
            "private_card_mask": torch.ones(batch, steps, private, device=device, dtype=torch.bool),
            "scalars": torch.randn(batch, steps, 32, device=device),
        }
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = critic(**values)
            loss = output.values.square().mean()
        loss.backward()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        critic.zero_grad(set_to_none=True)
        critic_rows.append({
            "batch": batch,
            "steps": steps,
            "milliseconds_forward_backward": elapsed * 1000,
            "tokens_per_second": batch * steps / elapsed,
            "peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
        })
        del values, output, loss

    result = {
        "kind": "cr_native_expert_selfplay_cloud_benchmark_v1",
        "gpu": torch.cuda.get_device_name(0),
        "actor_parameters": sum(value.numel() for value in actor.parameters()),
        "critic_parameters": sum(value.numel() for value in critic.parameters()),
        "actor_fp16": actor_rows,
        "critic_bf16_train": critic_rows,
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
