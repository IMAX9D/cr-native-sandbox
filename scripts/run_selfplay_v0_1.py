"""Run one guarded Self-Play v0.1 stage, then execute the official evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
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
from training.resource_monitor import TrainingResourceMonitor
from training.run_contract import CHECKPOINT_KIND
from training.schema import RunStore


DEFAULT_PYTHON = Path(r"D:\AI_data\runtime\venv\Scripts\python.exe")
DEFAULT_DATA_ROOT = Path(r"D:\AI_data\cr-native-core\selfplay-v0.1")
DEFAULT_ADB = Path(r"D:\Codex\toolchains\android-sdk\platform-tools\adb.exe")


def _run_id(target: int) -> str:
    stage = {1_000_000: "stage-a", 5_000_000: "stage-b", 10_000_000: "stage-c"}.get(
        target, f"target-{target}"
    )
    return (
        f"selfplay-v0.1-{stage}-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )


def _run_root_from_resume(path: Path) -> Path:
    current = path.resolve().parent
    while current.parent != current:
        if (current / "manifest.json").is_file():
            return current
        current = current.parent
    raise RuntimeError("could not locate run root for resume checkpoint")


def _stream_process(
    command: list[str],
    *,
    monitor: TrainingResourceMonitor,
    log_path: Path,
) -> tuple[int, list[dict[str, Any]]]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    events: list[dict[str, Any]] = []
    flags = (
        subprocess.CREATE_NO_WINDOW
        if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
    )
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=flags,
    )
    assert process.stdout is not None
    with log_path.open("w", encoding="utf-8") as log:
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            events.append(value)
            if value.get("event") == "training_phase":
                monitor.set_phase(str(value["phase"]))
            if len(monitor.samples) >= 3:
                recent = monitor.samples[-3:]
                if all(
                    float(item.get("system_ram_available_gb", 999.0)) < 0.10
                    for item in recent
                ):
                    process.terminate()
                    process.wait(timeout=30)
                    raise RuntimeError(
                        "available system RAM stayed below 0.10 GiB"
                    )
                if any(
                    float(item.get("guest_swap_used_mb_total", 0.0)) > 0.0
                    for item in monitor.samples
                ):
                    process.terminate()
                    process.wait(timeout=30)
                    raise RuntimeError("Android guest swap became active")
                swap_values = [
                    float(item.get("system_swap_used_gb", 0.0))
                    for item in monitor.samples
                ]
                if max(swap_values) - swap_values[0] > 1.0:
                    process.terminate()
                    process.wait(timeout=30)
                    raise RuntimeError(
                        "system pagefile usage grew by more than 1 GiB"
                    )
    return process.wait(), events


def _validate_stage(run_root: Path, target: int) -> dict[str, Any]:
    latest_path = run_root / "checkpoints" / "latest.pt"
    checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
    if checkpoint.get("kind") != CHECKPOINT_KIND:
        raise RuntimeError("latest checkpoint contract mismatch")
    native_ticks = int(checkpoint["native_ticks"])
    if native_ticks < target:
        raise RuntimeError(f"training stopped at {native_ticks} before target {target}")
    if not all(torch.isfinite(value).all() for value in checkpoint["model"].values()):
        raise RuntimeError("latest model contains NaN or Inf")
    if not checkpoint.get("optimizer") or not checkpoint.get("rng_state"):
        raise RuntimeError("latest checkpoint cannot resume optimizer/RNG state")
    required_recovery = list(range(250_000, target + 1, 250_000))
    required_candidates = [0, *range(500_000, target + 1, 500_000)]
    missing_recovery = [
        value for value in required_recovery
        if not (run_root / "checkpoints" / f"recovery-{value:09d}.pt").is_file()
    ]
    missing_candidates = [
        value for value in required_candidates
        if not (
            run_root / "evaluations" / "candidates"
            / f"P{value // 100_000:03d}.pt"
        ).is_file()
    ]
    if missing_recovery or missing_candidates:
        raise RuntimeError(
            f"missing milestones: recovery={missing_recovery}, candidates={missing_candidates}"
        )
    metrics = dict(checkpoint.get("metrics", {}))
    finite_metrics = all(
        np.isfinite(float(value)) for value in metrics.values()
    )
    if not finite_metrics:
        raise RuntimeError("latest PPO metrics contain NaN or Inf")
    return {
        "latest_checkpoint": str(latest_path),
        "native_ticks": native_ticks,
        "agent_steps": int(checkpoint["agent_steps"]),
        "episodes": int(checkpoint["completed_episodes"]),
        "iteration": int(checkpoint["iteration"]),
        "current_model_digest": checkpoint["current_model_digest"],
        "initial_model_digest": checkpoint["initial_model_digest"],
        "metrics": metrics,
        "behavior": checkpoint.get("behavior", {}),
        "required_recovery_checkpoints": required_recovery,
        "required_evaluation_candidates": required_candidates,
        "passed": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-native-ticks", type=int, default=1_000_000)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--run-id")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--paired-seeds", type=int, default=16)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--keep-vms", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.target_native_ticks < 1:
        raise ValueError("target-native-ticks must be positive")
    if not args.python.is_file():
        raise FileNotFoundError(f"Python runtime not found: {args.python}")
    git_status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True,
        stdout=subprocess.PIPE, check=True,
        creationflags=(
            subprocess.CREATE_NO_WINDOW
            if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        ),
    ).stdout.strip()
    if git_status and not args.allow_dirty:
        raise RuntimeError("formal training requires a clean committed worktree")
    if args.resume:
        run_root = _run_root_from_resume(args.resume)
        data_root = run_root.parent.parent
        if data_root.resolve() != args.data_root.resolve():
            raise RuntimeError("resume run is outside --data-root")
        run_id = run_root.name
        if args.run_id and args.run_id != run_id:
            raise RuntimeError("run-id does not match resume run")
    else:
        run_id = args.run_id or _run_id(args.target_native_ticks)
        run_root = args.data_root.resolve() / "runs" / run_id
        if run_root.exists():
            raise FileExistsError(f"formal run already exists: {run_root}")

    active_pool = MultiAvdWorkerPool(avds=2, workers_per_avd=4)
    recycle_pool = MultiAvdWorkerPool(avds=4, workers_per_avd=4)
    supervisor_root = args.data_root.resolve() / "supervisor-logs" / run_id
    supervisor_root.mkdir(parents=True, exist_ok=True)
    stage_summary: dict[str, Any] = {
        "schema_version": 1,
        "kind": "selfplay_v0_1_guarded_stage",
        "run_id": run_id,
        "target_native_ticks": args.target_native_ticks,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(args.data_root.resolve()),
        "training_configuration": {
            "avds": 2,
            "workers": 8,
            "workers_per_avd": 4,
            "native_tick_hz": 20,
            "decision_frequency_hz": 20,
            "reward": "tower_hp_potential_v1",
        },
    }
    RunStore._atomic_json(supervisor_root / "stage-summary.json", stage_summary)
    try:
        recycle_pool.stop(keep_vms=False)
        workers_ready = active_pool.ensure_ready(configure_direct=True)
        RunStore._atomic_json(supervisor_root / "workers-ready.json", workers_ready)
        monitor = TrainingResourceMonitor(
            adb=DEFAULT_ADB,
            serials=[pool.config.serial for pool in active_pool.pools],
            emulator_ports=[
                pool.config.emulator_port for pool in active_pool.pools
            ],
            workers_per_avd=4,
        )
        command = [
            str(args.python), "-m", "training.train",
            "--iterations", "1000000",
            "--target-native-ticks", str(args.target_native_ticks),
            "--episodes-per-iteration", "8",
            "--workers", "8",
            "--avds", "2",
            "--workers-per-avd", "4",
            "--seed", str(args.seed),
            "--max-ticks", "7200",
            "--device", "cuda",
            "--reward", "tower_hp_potential_v1",
            "--transport", "direct",
            "--data-root", str(args.data_root.resolve()),
            "--run-id", run_id,
            "--skip-worker-start",
            "--emit-phase-events",
        ]
        if args.resume:
            command.extend(["--resume", str(args.resume.resolve())])
        monitor.start()
        try:
            exit_code, events = _stream_process(
                command,
                monitor=monitor,
                log_path=supervisor_root / "training.log",
            )
        finally:
            monitor.stop()
        RunStore._atomic_json(
            supervisor_root / "training-resource-samples.json", monitor.samples
        )
        resource_summary = monitor.summary()
        RunStore._atomic_json(
            supervisor_root / "training-resource-summary.json", resource_summary
        )
        if exit_code:
            raise RuntimeError(f"training process exited with code {exit_code}")
        stage_summary["training"] = _validate_stage(
            run_root, args.target_native_ticks
        )
        stage_summary["resources"] = resource_summary
        stage_summary["worker_status_after_training"] = active_pool.status()
        sampling_resources = resource_summary.get("phases", {}).get("sampling", {})
        stage_summary["resource_warnings"] = []
        available_min = sampling_resources.get(
            "system_ram_available_gb", {}
        ).get("min")
        if available_min is not None and float(available_min) < 1.0:
            stage_summary["resource_warnings"].append(
                f"minimum available system RAM was {float(available_min):.2f} GiB"
            )
        guest_swap = sampling_resources.get(
            "guest_swap_used_mb_total", {}
        ).get("max", 0.0)
        if float(guest_swap) > 0.0:
            raise RuntimeError("Android guest swap was used during formal training")

        if not args.skip_evaluation:
            evaluation_command = [
                str(args.python), str(
                    PROJECT_ROOT / "scripts" / "evaluate_selfplay_v0_1.py"
                ),
                "--run-root", str(run_root),
                "--paired-seeds", str(args.paired_seeds),
                "--workers", "8",
                "--avds", "2",
                "--workers-per-avd", "4",
                "--device", "cuda",
                "--skip-worker-start",
                "--keep-vms",
                "--resume",
            ]
            evaluation = subprocess.run(
                evaluation_command,
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
            (supervisor_root / "evaluation.log").write_text(
                evaluation.stdout, encoding="utf-8"
            )
            print(evaluation.stdout, end="", flush=True)
            if evaluation.returncode:
                raise RuntimeError(
                    f"evaluation process exited with code {evaluation.returncode}"
                )
            evaluation_summary_path = (
                run_root / "evaluations" / "official-v0.1"
                / "evaluation-summary.json"
            )
            stage_summary["evaluation"] = json.loads(
                evaluation_summary_path.read_text(encoding="utf-8-sig")
            )
        stage_summary["completed_utc"] = datetime.now(timezone.utc).isoformat()
        stage_summary["passed"] = True
        RunStore._atomic_json(supervisor_root / "stage-summary.json", stage_summary)
        RunStore._atomic_json(run_root / "stage-summary.json", stage_summary)
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
            raise RuntimeError(
                f"training report generation failed with code {report.returncode}"
            )
        stage_summary["report"] = {
            "repository": str(
                PROJECT_ROOT / "docs" / "SELFPLAY_V0_1_TRAINING_REPORT.zh-CN.md"
            ),
            "run_copy": str(
                run_root / "reports" / "SELFPLAY_V0_1_TRAINING_REPORT.zh-CN.md"
            ),
        }
        RunStore._atomic_json(supervisor_root / "stage-summary.json", stage_summary)
        RunStore._atomic_json(run_root / "stage-summary.json", stage_summary)
        print(json.dumps({
            "run": str(run_root),
            "native_ticks": stage_summary["training"]["native_ticks"],
            "evaluation_complete": "evaluation" in stage_summary,
            "passed": True,
        }, ensure_ascii=False), flush=True)
        return 0
    except Exception as error:
        stage_summary["completed_utc"] = datetime.now(timezone.utc).isoformat()
        stage_summary["passed"] = False
        stage_summary["error"] = str(error)
        RunStore._atomic_json(supervisor_root / "stage-summary.json", stage_summary)
        if run_root.exists():
            RunStore._atomic_json(run_root / "stage-summary.json", stage_summary)
        raise
    finally:
        if not args.keep_vms:
            try:
                active_pool.stop(keep_vms=False)
            except Exception as error:
                print(f"worker cleanup warning: {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
