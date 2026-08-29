"""Export a resumable expert checkpoint as compact FP16 inference weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from expert_v1.training_v1.model import ExpertPolicyConfig, RecurrentExpertPolicy
from expert_v1.training_v1.train import _inference_weights_payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ExpertPolicyConfig(**checkpoint["model_config"])
    model = RecurrentExpertPolicy(config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    payload = _inference_weights_payload(
        epoch=int(checkpoint["epoch"]),
        global_step=int(checkpoint["global_step"]),
        model=model,
        dataset_manifest_sha256=str(checkpoint["dataset_manifest_sha256"]),
        run_signature_sha256=str(checkpoint["run_signature_sha256"]),
        run_id=str(checkpoint["run_id"]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(args.output)
    result = {
        "kind": payload["kind"],
        "checkpoint": str(args.checkpoint.resolve()),
        "output": str(args.output.resolve()),
        "global_step": payload["global_step"],
        "parameters": sum(value.numel() for value in payload["model_state"].values()),
        "bytes": args.output.stat().st_size,
        "sha256": sha256_file(args.output),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
