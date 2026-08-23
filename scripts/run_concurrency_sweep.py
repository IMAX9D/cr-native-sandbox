"""Run complete Recurrent PPO scaling tiers across 1..4 Android AVDs."""

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

from native_core.worker import MultiAvdWorkerPool, WorkerError
from training.resource_monitor import TrainingResourceMonitor


DEFAULT_PYTHON = Path(r"D:\AI_data\runtime\venv\Scripts\python.exe")
DEFAULT_TRAINING_ROOT = Path(r"D:\AI_data\cr-native-core\training")
DEFAULT_SWEEP_ROOT = Path(r"D:\AI_data\cr-native-core\scaling-sweeps")
DEFAULT_ADB = Path(r"D:\Codex\toolchains\android-sdk\platform-tools\adb.exe")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _validate_run(run_root: Path, expected_episodes: int) -> dict[str, Any]:
    checkpoint_path = run_root / "checkpoints" / "latest.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    episodes = checkpoint.get("episode_summaries", [])
    seeds = [int(item["seed"]) for item in episodes]
    episode_failures = sum(
        not bool(item.get("terminated")) or bool(item.get("truncated"))
        for item in episodes
    )
    trajectory_files = sorted((run_root / "trajectories").glob("*.npz"))
    trajectory_failures: list[dict[str, Any]] = []
    steps = 0
    for path in trajectory_files:
        with np.load(path) as arrays:
            count = len(arrays["rewards"])
            steps += count
            finite = all(
                np.isfinite(arrays[name]).all()
                for name in (
                    "grid", "scalars", "privileged", "log_probabilities",
                    "values", "rewards",
                )
            )
            cards = arrays["cards"]
            positions = arrays["positions"]
            rows = np.arange(count)
            legal_cards = bool(arrays["card_masks"][rows, cards].all())
            playing = cards > 0
            legal_positions = bool(
                arrays["position_masks"][
                    rows[playing], cards[playing] - 1, positions[playing]
                ].all()
            ) if playing.any() else True
            last_done = bool(arrays["dones"][-1]) if count else False
            if not (finite and legal_cards and legal_positions and last_done):
                trajectory_failures.append({
                    "file": path.name,
                    "finite": bool(finite),
                    "legal_cards": legal_cards,
                    "legal_positions": legal_positions,
                    "last_done": last_done,
                })
    model_finite = all(
        torch.isfinite(value).all().item()
        for value in checkpoint["model"].values()
    )
    passed = (
        len(episodes) == expected_episodes
        and len(set(seeds)) == expected_episodes
        and episode_failures == 0
        and len(trajectory_files) == expected_episodes * 2
        and not trajectory_failures
        and model_finite
        and bool(checkpoint.get("optimizer"))
    )
    return {
        "passed": passed,
        "episodes": len(episodes),
        "unique_seeds": len(set(seeds)),
        "episode_failures": episode_failures,
        "episode_failure_rate": (
            episode_failures / expected_episodes if expected_episodes else 0.0
        ),
        "trajectory_files": len(trajectory_files),
        "trajectory_agent_steps": steps,
        "trajectory_failures": trajectory_failures,
        "model_finite": model_finite,
        "optimizer_present": bool(checkpoint.get("optimizer")),
    }


def _tier_result(
    *,
    avds: int,
    workers: int,
    run_id: str,
    event: dict[str, Any],
    resources: dict[str, Any],
    validation: dict[str, Any],
    worker_status: dict[str, Any],
) -> dict[str, Any]:
    metrics = event["metrics"]
    profile = event["sampling_profile"]
    environment_wall = (
        float(metrics["iteration_wall_seconds"])
        - float(metrics["learner_wall_seconds"])
    )
    environment_steps = float(metrics["environment_steps"])
    episodes = validation["episodes"]
    service_states = [
        state
        for instance in worker_status["instances"]
        for state in instance["services"]
    ]
    worker_failures = sum(not bool(value) for value in service_states)
    return {
        "avds": avds,
        "workers": workers,
        "policy_batch_size_expected": workers * 2,
        "run_id": run_id,
        "environment_steps": environment_steps,
        "environment_steps_per_second": float(
            metrics["environment_steps_per_second"]
        ),
        "training_steps_per_second": (
            environment_steps / float(metrics["iteration_wall_seconds"])
        ),
        "policy_decisions_per_second": (
            float(profile["policy_decisions"]) / environment_wall
        ),
        "episodes_per_hour": (
            episodes * 3600.0 / float(metrics["iteration_wall_seconds"])
        ),
        "environment_wall_seconds": environment_wall,
        "ppo_update_seconds": float(metrics["learner_wall_seconds"]),
        "iteration_wall_seconds": float(metrics["iteration_wall_seconds"]),
        "inference_seconds": float(profile["vector_inference_seconds"]),
        "inference_ms_per_round": (
            float(profile["vector_inference_seconds"])
            * 1000.0 / float(profile["vector_round_count"])
        ),
        "native_transition_wall_seconds": float(
            profile["vector_transition_wall_seconds"]
        ),
        "native_transition_ms_per_round": (
            float(profile["vector_transition_wall_seconds"])
            * 1000.0 / float(profile["vector_round_count"])
        ),
        "rpc_p50_ms": float(profile["rpc_latency_p50_ms"]),
        "rpc_p95_ms": float(profile["rpc_latency_p95_ms"]),
        "rpc_p99_ms": float(profile["rpc_latency_p99_ms"]),
        "rpc_failure_rate": float(profile["rpc_failure_rate"]),
        "worker_failures": worker_failures,
        "worker_failure_rate": (
            worker_failures / len(service_states) if service_states else 1.0
        ),
        "episode_failure_rate": validation["episode_failure_rate"],
        "policy_batch_size_mean": float(profile["policy_batch_size_mean"]),
        "policy_batch_size_max": int(profile["policy_batch_size_max"]),
        "active_workers_mean": float(profile["active_workers_mean"]),
        "barrier": {
            key: float(value)
            for key, value in profile.items()
            if key.startswith("worker_transition_")
        },
        "resources": resources,
        "validation": validation,
        "worker_status": worker_status,
        "barrier_profile": event["barrier_profile"],
    }


def _stop_reason(previous: dict[str, Any], current: dict[str, Any]) -> str | None:
    gain = (
        current["training_steps_per_second"]
        / previous["training_steps_per_second"] - 1.0
    )
    if current["worker_failure_rate"] or current["episode_failure_rate"]:
        return "worker or episode failures appeared"
    if current["rpc_failure_rate"]:
        return "RPC failures appeared"
    if gain < 0.20:
        return f"training throughput gain was only {gain * 100.0:.1f}% (<20%)"
    if current["episodes_per_hour"] <= previous["episodes_per_hour"]:
        return "episodes/hour stopped increasing"
    if (
        current["rpc_p95_ms"] > previous["rpc_p95_ms"] * 1.75
        and current["rpc_p99_ms"] > previous["rpc_p99_ms"] * 1.75
    ):
        return "RPC p95 and p99 both increased by more than 75%"
    sampling = current["resources"].get("phases", {}).get("sampling", {})
    available = sampling.get("system_ram_available_gb", {}).get("min", 999.0)
    swap = sampling.get("guest_swap_used_mb_total", {}).get("max", 0.0)
    if available < 2.0 or swap > 0.0:
        return "RAM safety margin was exhausted"
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiers", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--max-ticks", type=int, default=7200)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--training-root", type=Path, default=DEFAULT_TRAINING_ROOT)
    parser.add_argument("--sweep-root", type=Path, default=DEFAULT_SWEEP_ROOT)
    parser.add_argument("--sweep-id")
    parser.add_argument("--no-auto-stop", action="store_true")
    parser.add_argument("--keep-vms", action="store_true")
    parser.add_argument("--resume-sweep", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    tiers = sorted(set(args.tiers))
    if not tiers or tiers[0] < 1 or tiers[-1] > 4:
        raise ValueError("tiers must be selected from 1..4")
    sweep_id = args.sweep_id or (
        "native-scaling-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    output_root = args.sweep_root / sweep_id
    summary_path = output_root / "sweep-summary.json"
    if args.resume_sweep:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.setdefault("setup_failures", []).append({
            "utc": summary.get("completed_utc"),
            "error": summary.pop("error", None),
        })
        summary.pop("completed_utc", None)
        summary["stop_reason"] = None
    else:
        output_root.mkdir(parents=True, exist_ok=False)
        summary = {
            "schema_version": 1,
            "kind": "native_training_concurrency_scaling_sweep",
            "sweep_id": sweep_id,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "constraints": {
                "workers_per_avd": 4,
                "cores_per_avd": 4,
                "transport": "direct",
                "native_tick_hz": 20,
                "reward": "potential",
                "cuda_graph": True,
                "network_and_ppo_unchanged": True,
            },
            "tiers_requested": tiers,
            "tiers": [],
            "stop_reason": None,
        }
    _atomic_json(output_root / "sweep-summary.json", summary)
    active_pool: MultiAvdWorkerPool | None = None
    try:
        for avds in tiers:
            if any(int(item["avds"]) == avds for item in summary["tiers"]):
                continue
            workers = avds * 4
            print(f"[scaling] preparing {avds} AVD / {workers} Worker", flush=True)
            active_pool = MultiAvdWorkerPool(avds=avds, workers_per_avd=4)
            ready = active_pool.ensure_ready(configure_direct=True)
            _atomic_json(output_root / f"tier-{avds}-workers-ready.json", ready)
            run_id = f"{sweep_id}-{avds}avd-{workers}worker"
            command = [
                str(args.python), "-m", "training.train",
                "--iterations", "1",
                "--episodes-per-iteration", str(workers),
                "--workers", str(workers),
                "--avds", str(avds),
                "--workers-per-avd", "4",
                "--max-ticks", str(args.max_ticks),
                "--seed", str(args.seed),
                "--run-id", run_id,
                "--data-root", str(args.training_root),
                "--transport", "direct",
                "--skip-worker-start",
                "--emit-phase-events",
            ]
            monitor = TrainingResourceMonitor(
                adb=DEFAULT_ADB,
                serials=[pool.config.serial for pool in active_pool.pools],
                emulator_ports=[
                    pool.config.emulator_port for pool in active_pool.pools
                ],
                workers_per_avd=4,
            )
            iteration_event: dict[str, Any] | None = None
            output_lines: list[str] = []
            monitor.start()
            try:
                process = subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=os.environ.copy(),
                    creationflags=(
                        subprocess.CREATE_NO_WINDOW
                        if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
                    ),
                )
                assert process.stdout is not None
                for line in process.stdout:
                    text = line.rstrip()
                    output_lines.append(text)
                    print(text, flush=True)
                    try:
                        value = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    if value.get("event") == "training_phase":
                        monitor.set_phase(str(value["phase"]))
                    elif value.get("event") == "iteration_complete":
                        iteration_event = value
                exit_code = process.wait()
            finally:
                monitor.stop()
            (output_root / f"tier-{avds}-training.log").write_text(
                "\n".join(output_lines) + "\n", encoding="utf-8"
            )
            _atomic_json(
                output_root / f"tier-{avds}-resource-samples.json",
                monitor.samples,
            )
            resource_summary = monitor.summary()
            _atomic_json(
                output_root / f"tier-{avds}-resource-summary.json",
                resource_summary,
            )
            if exit_code or iteration_event is None:
                raise RuntimeError(
                    f"tier {avds} training failed with exit code {exit_code}"
                )
            run_root = args.training_root / "runs" / run_id
            validation = _validate_run(run_root, workers)
            worker_status = active_pool.status()
            result = _tier_result(
                avds=avds,
                workers=workers,
                run_id=run_id,
                event=iteration_event,
                resources=resource_summary,
                validation=validation,
                worker_status=worker_status,
            )
            _atomic_json(output_root / f"tier-{avds}-result.json", result)
            summary["tiers"].append(result)
            if not validation["passed"]:
                summary["stop_reason"] = "training data validation failed"
            elif len(summary["tiers"]) >= 2 and not args.no_auto_stop:
                summary["stop_reason"] = _stop_reason(
                    summary["tiers"][-2], summary["tiers"][-1]
                )
            _atomic_json(output_root / "sweep-summary.json", summary)
            if summary["stop_reason"]:
                print(f"[scaling] stop: {summary['stop_reason']}", flush=True)
                break
        summary["completed_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(output_root / "sweep-summary.json", summary)
        print(json.dumps({
            "sweep": str(output_root),
            "tiers_completed": len(summary["tiers"]),
            "stop_reason": summary["stop_reason"],
        }, ensure_ascii=False), flush=True)
        return 0
    except (OSError, RuntimeError, WorkerError) as error:
        summary["error"] = str(error)
        summary["completed_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(output_root / "sweep-summary.json", summary)
        raise
    finally:
        if active_pool is not None and not args.keep_vms:
            try:
                active_pool.stop(keep_vms=False)
            except Exception as error:
                print(f"[scaling] cleanup warning: {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
