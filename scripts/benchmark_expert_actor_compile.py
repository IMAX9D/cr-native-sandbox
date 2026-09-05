"""Compare eager and torch.compile latency for the production Expert Actor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import Tensor

from expert_v1.training_v1.model import (
    ExpertPolicyConfig,
    RecurrentExpertPolicy,
    configure_position_precision,
)
from scripts.benchmark_expert_selfplay_cloud import actor_inputs


def _latency(
    function: Callable[..., Any],
    inputs: dict[str, Tensor],
    *,
    iterations: int,
) -> tuple[float, Any]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = None
    with torch.inference_mode():
        for _ in range(iterations):
            output = function(**inputs)
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0 / iterations, output


def _max_output_delta(left: Any, right: Any) -> float:
    result = 0.0
    for left_value, right_value in zip(left, right, strict=True):
        left_rows = left_value if isinstance(left_value, tuple) else (left_value,)
        right_rows = right_value if isinstance(right_value, tuple) else (right_value,)
        for left_tensor, right_tensor in zip(left_rows, right_rows, strict=True):
            result = max(
                result,
                float((left_tensor.float() - right_tensor.float()).abs().max()),
            )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch < 1 or args.iterations < 1:
        raise ValueError("benchmark sizes must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requires one visible GPU")

    device = torch.device("cuda")
    checkpoint = torch.load(
        args.checkpoint.resolve(strict=True),
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    config = ExpertPolicyConfig(**dict(checkpoint["model_config"]))
    configure_position_precision(config)
    actor = RecurrentExpertPolicy(config)
    actor.load_state_dict(checkpoint["model_state"], strict=True)
    actor.eval().to(device=device, dtype=torch.float16)
    inputs = actor_inputs(config, args.batch, device)

    eager = actor.forward_sequence
    with torch.inference_mode():
        for _ in range(5):
            eager_output = eager(**inputs)
    eager_ms, eager_output = _latency(
        eager, inputs, iterations=args.iterations
    )

    compiled = torch.compile(
        actor.forward_sequence,
        backend="inductor",
        mode="reduce-overhead",
        fullgraph=False,
        dynamic=False,
    )
    compile_started = time.perf_counter()
    with torch.inference_mode():
        compiled_output = compiled(**inputs)
    torch.cuda.synchronize()
    compilation_seconds = time.perf_counter() - compile_started
    with torch.inference_mode():
        for _ in range(4):
            compiled_output = compiled(**inputs)
    compiled_ms, compiled_output = _latency(
        compiled, inputs, iterations=args.iterations
    )
    max_delta = _max_output_delta(eager_output, compiled_output)
    if not all(
        bool(torch.isfinite(value).all())
        for value in compiled_output
        if isinstance(value, Tensor)
    ):
        raise FloatingPointError("compiled Actor produced NaN/Inf")

    result = {
        "kind": "cr_native_expert_actor_compile_benchmark_v1",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "batch": args.batch,
        "iterations": args.iterations,
        "parameters": sum(value.numel() for value in actor.parameters()),
        "eager_ms": eager_ms,
        "compiled_ms": compiled_ms,
        "speedup": eager_ms / compiled_ms,
        "compilation_seconds": compilation_seconds,
        "max_output_abs_delta": max_delta,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
