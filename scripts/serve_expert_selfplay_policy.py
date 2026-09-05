"""Serve one shared CUDA policy service for multi-process collectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from expert_selfplay_v1.batched_policy import BatchedPolicyService
from expert_selfplay_v1.remote_policy import RemotePolicyServer
from scripts.run_expert_selfplay_v1 import load_base


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--opponent-checkpoint", type=Path, required=True)
    parser.add_argument("--expert-manifest", type=Path, required=True)
    parser.add_argument("--address", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=20905000)
    parser.add_argument("--microbatch-ms", type=float, default=2.0)
    parser.add_argument("--max-actor-rows", type=int, default=256)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    loaded = load_base(args.checkpoint, args.expert_manifest, device=device)
    opponent = load_base(
        args.opponent_checkpoint, args.expert_manifest, device=device
    )
    service = BatchedPolicyService(device=device, seed=args.seed)
    service.register_actor(
        loaded.actor, actor_sha256=loaded.actor_sha256, verify_content=True
    )
    if opponent.actor_sha256 != loaded.actor_sha256:
        service.register_actor(
            opponent.actor,
            actor_sha256=opponent.actor_sha256,
            verify_content=True,
        )
    server = RemotePolicyServer(
        service,
        args.address,
        microbatch_seconds=args.microbatch_ms / 1000.0,
        max_actor_rows=args.max_actor_rows,
    )
    metrics = server.serve_forever()
    if args.result is not None:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(
            json.dumps(metrics, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(metrics, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
