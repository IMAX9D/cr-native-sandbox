"""Measure one-tick expert-policy inference on the deployment GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from expert_v1.training_v1.dataset import NativeExpertSequenceDataset, collate_sequences
from expert_v1.training_v1.model import ExpertPolicyConfig, RecurrentExpertPolicy
from expert_v1.training_v1.schema import OBSERVATION_NATIVE, read_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--card-embedding-size", type=int)
    parser.add_argument("--spatial-size", type=int)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    root = args.dataset_root.resolve()
    manifest = read_manifest(root)
    dimensions = manifest["dimensions"]
    dataset = NativeExpertSequenceDataset(
        root,
        split=args.split,
        sequence_length=128,
        burn_in=32,
        validate=False,
    )
    one_tick = {key: value[:1] for key, value in dataset[0].items()}
    batch = collate_sequences([one_tick])
    device = torch.device("cuda")
    batch = {key: value.to(device) for key, value in batch.items()}
    weights_payload = None
    if args.weights is not None:
        weights_payload = torch.load(args.weights, map_location="cpu", weights_only=False)
        if weights_payload.get("kind") != "cr_native_expert_inference_weights_v1":
            raise RuntimeError("unsupported inference weights artifact")
        config = ExpertPolicyConfig(**weights_payload["model_config"])
    else:
        if any(
            value is None
            for value in (
                args.hidden_size,
                args.card_embedding_size,
                args.spatial_size,
            )
        ):
            raise ValueError("model dimensions are required when --weights is omitted")
        config = ExpertPolicyConfig(
            grid_channels=int(dimensions["grid_channels"]),
            public_scalar_size=int(dimensions["public_scalar_size"]),
            card_vocab_size=int(dimensions["card_vocab_size"]),
            ability_vocab_size=int(dimensions["ability_vocab_size"]),
            max_ability_slots=int(dimensions["max_ability_slots"]),
            entity_numeric_size=int(dimensions.get("entity_numeric_size", 3)),
            hidden_size=int(args.hidden_size),
            card_embedding_size=int(args.card_embedding_size),
            spatial_size=int(args.spatial_size),
            lambda_max=20.0,
            lambda_initial=0.3,
            observation_mode=OBSERVATION_NATIVE,
        )
    model = RecurrentExpertPolicy(config)
    if weights_payload is not None:
        model.load_state_dict(weights_payload["model_state"], strict=True)
    model.to(device).eval()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(args.warmup):
            with torch.autocast("cuda", dtype=torch.float16):
                model.forward_batch(batch)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(args.steps):
            with torch.autocast("cuda", dtype=torch.float16):
                model.forward_batch(batch)
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    result = {
        "kind": "cr_expert_local_inference_gate_v1",
        "device": torch.cuda.get_device_name(0),
        "weights": str(args.weights.resolve()) if args.weights is not None else None,
        "global_step": (
            int(weights_payload["global_step"])
            if weights_payload is not None
            else None
        ),
        "parameters": parameters,
        "weights_fp32_mib": parameters * 4 / (1024**2),
        "steps": args.steps,
        "milliseconds_per_tick": elapsed * 1000.0 / args.steps,
        "ticks_per_second": args.steps / elapsed,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024**2),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
