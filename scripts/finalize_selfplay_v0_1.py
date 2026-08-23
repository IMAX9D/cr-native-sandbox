"""Finalize an already-trained v0.1 run after resumable evaluation completes."""

from __future__ import annotations

from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from native_core.worker import MultiAvdWorkerPool
from training.run_contract import CHECKPOINT_KIND
from training.schema import RunStore


def _validate(run_root: Path, target: int) -> dict[str, Any]:
    checkpoint_path = run_root / "checkpoints" / "latest.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("kind") != CHECKPOINT_KIND:
        raise RuntimeError("latest checkpoint kind mismatch")
    if int(checkpoint["native_ticks"]) < target:
        raise RuntimeError("training target was not reached")
    if not all(torch.isfinite(value).all() for value in checkpoint["model"].values()):
        raise RuntimeError("model contains NaN/Inf")
    if not all(np.isfinite(float(value)) for value in checkpoint["metrics"].values()):
        raise RuntimeError("metrics contain NaN/Inf")
    if not checkpoint.get("optimizer") or not checkpoint.get("rng_state"):
        raise RuntimeError("checkpoint is not resumable")
    required_recovery = list(range(250_000, target + 1, 250_000))
    required_candidates = [0, *range(500_000, target + 1, 500_000)]
    for value in required_recovery:
        if not (run_root / "checkpoints" / f"recovery-{value:09d}.pt").is_file():
            raise RuntimeError(f"missing recovery checkpoint {value}")
    for value in required_candidates:
        if not (
            run_root / "evaluations" / "candidates"
            / f"P{value // 100_000:03d}.pt"
        ).is_file():
            raise RuntimeError(f"missing evaluation candidate {value}")
    return {
        "latest_checkpoint": str(checkpoint_path),
        "native_ticks": int(checkpoint["native_ticks"]),
        "agent_steps": int(checkpoint["agent_steps"]),
        "episodes": int(checkpoint["completed_episodes"]),
        "iteration": int(checkpoint["iteration"]),
        "current_model_digest": checkpoint["current_model_digest"],
        "initial_model_digest": checkpoint["initial_model_digest"],
        "metrics": checkpoint["metrics"],
        "behavior": checkpoint["behavior"],
        "required_recovery_checkpoints": required_recovery,
        "required_evaluation_candidates": required_candidates,
        "passed": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--target-native-ticks", type=int, default=1_000_000)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(r"D:\AI_data\runtime\venv\Scripts\python.exe"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_root = args.run_root.resolve()
    manifest = json.loads(
        (run_root / "manifest.json").read_text(encoding="utf-8-sig")
    )
    data_root = Path(manifest["data_root"]).resolve()
    supervisor_root = data_root / "supervisor-logs" / run_root.name
    resource_path = supervisor_root / "training-resource-summary.json"
    evaluation_path = (
        run_root / "evaluations" / "official-v0.1" / "evaluation-summary.json"
    )
    if not resource_path.is_file() or not evaluation_path.is_file():
        raise RuntimeError("resource or evaluation summary is incomplete")
    resources = json.loads(resource_path.read_text(encoding="utf-8-sig"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8-sig"))
    if evaluation.get("passed_integrity") is not True:
        raise RuntimeError("evaluation suite failed integrity")
    initial_stage_path = supervisor_root / "stage-summary.json"
    initial_stage = (
        json.loads(initial_stage_path.read_text(encoding="utf-8-sig"))
        if initial_stage_path.is_file() else {}
    )
    training = _validate(run_root, args.target_native_ticks)
    overall = resources.get("overall", {})
    warnings: list[str] = []
    minimum_ram = overall.get("system_ram_available_gb", {}).get("min")
    if minimum_ram is not None and float(minimum_ram) < 1.0:
        warnings.append(
            f"minimum available system RAM was {float(minimum_ram):.2f} GiB"
        )
    maximum_guest_swap = overall.get("guest_swap_used_mb_total", {}).get("max", 0.0)
    if float(maximum_guest_swap) > 0.0:
        raise RuntimeError("Android guest swap was used during training")
    summary = {
        "schema_version": 1,
        "kind": "selfplay_v0_1_guarded_stage",
        "run_id": run_root.name,
        "target_native_ticks": args.target_native_ticks,
        "started_utc": initial_stage.get("started_utc"),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "training_configuration": {
            "avds": 2,
            "workers": 8,
            "workers_per_avd": 4,
            "native_tick_hz": 20,
            "decision_frequency_hz": 20,
            "reward": "tower_hp_potential_v1",
        },
        "training": training,
        "resources": resources,
        "resource_warnings": warnings,
        "evaluation": evaluation,
        "recovery": {
            "evaluation_resumed": True,
            "training_was_not_resumed": True,
        },
        "passed": True,
    }
    RunStore._atomic_json(supervisor_root / "stage-summary.json", summary)
    RunStore._atomic_json(run_root / "stage-summary.json", summary)
    report = subprocess.run(
        [
            str(args.python),
            str(PROJECT_ROOT / "scripts" / "generate_selfplay_report.py"),
            "--run-root", str(run_root),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=(
            subprocess.CREATE_NO_WINDOW
            if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        ),
        check=False,
    )
    print(report.stdout, end="", flush=True)
    if report.returncode:
        raise RuntimeError("report generation failed")
    summary["report"] = {
        "repository": str(
            PROJECT_ROOT / "docs" / "SELFPLAY_V0_1_TRAINING_REPORT.zh-CN.md"
        ),
        "run_copy": str(
            run_root / "reports" / "SELFPLAY_V0_1_TRAINING_REPORT.zh-CN.md"
        ),
    }
    RunStore._atomic_json(supervisor_root / "stage-summary.json", summary)
    RunStore._atomic_json(run_root / "stage-summary.json", summary)
    try:
        MultiAvdWorkerPool(avds=4, workers_per_avd=4).stop(keep_vms=False)
    finally:
        print(json.dumps({
            "run": str(run_root),
            "native_ticks": training["native_ticks"],
            "evaluation_matchups": (
                len(evaluation.get("pair_summaries", {}))
                + len(evaluation.get("random_legal_summaries", {}))
            ),
            "passed": True,
            "training_continued": False,
        }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
