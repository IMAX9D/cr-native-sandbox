"""Guarded 5M v0.2 Scratch training with the live browser dashboard."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any
import webbrowser

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from native_core.worker import MultiAvdWorkerPool
from scripts.run_training_dashboard import DashboardState
from selfplay_v2 import CHECKPOINT_KIND
from training.dashboard import TrainingDashboardServer
from training.resource_monitor import TrainingResourceMonitor
from training.schema import RunStore


DEFAULT_PYTHON = Path(r"D:\AI_data\runtime\venv\Scripts\python.exe")
DEFAULT_DATA_ROOT = Path(r"D:\AI_data\cr-native-core\selfplay-v0.2")
DEFAULT_ADB = Path(r"D:\Codex\toolchains\android-sdk\platform-tools\adb.exe")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    RunStore._atomic_json(path, value)


def _worker(
    *,
    args: argparse.Namespace,
    state: DashboardState,
    output_root: Path,
) -> None:
    pool = MultiAvdWorkerPool(avds=2, workers_per_avd=4)
    recycle = MultiAvdWorkerPool(avds=4, workers_per_avd=4)
    monitor: TrainingResourceMonitor | None = None
    process: subprocess.Popen[str] | None = None
    watchdog_stop = threading.Event()
    watchdog_errors: list[str] = []
    try:
        state.set_phase("startup", "关闭旧Worker")
        recycle.stop(keep_vms=False)
        state.set_phase("startup", "冷启动 2 AVD / 8 Worker")
        ready = pool.ensure_ready(configure_direct=True)
        _atomic_json(output_root / "workers-ready.json", ready)
        monitor = TrainingResourceMonitor(
            adb=DEFAULT_ADB,
            serials=[item.config.serial for item in pool.pools],
            emulator_ports=[item.config.emulator_port for item in pool.pools],
            workers_per_avd=4,
        )
        state.attach_monitor(monitor)
        monitor.start()
        command = [
            str(args.python), "-m", "selfplay_v2.train",
            "--target-native-ticks", str(args.target_native_ticks),
            "--iterations", "1000000",
            "--episodes-per-iteration", "8",
            "--workers", "8",
            "--avds", "2",
            "--workers-per-avd", "4",
            "--initialization", "scratch",
            "--lambda-initial", "0.30",
            "--lambda-max", "20.0",
            "--seed", "20260824",
            "--max-ticks", "7200",
            "--device", "cuda",
            "--transport", "direct",
            "--data-root", str(args.data_root),
            "--run-id", args.run_id,
            "--skip-worker-start",
            "--emit-phase-events",
        ]
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
        state.training_pid = process.pid

        def watchdog() -> None:
            baseline_swap: float | None = None
            guest_swap_high_samples = 0
            while not watchdog_stop.wait(1.0):
                if process is None or process.poll() is not None or monitor is None:
                    return
                if not monitor.samples:
                    continue
                samples = monitor.samples
                latest = samples[-1]
                if baseline_swap is None:
                    baseline_swap = float(latest.get("system_swap_used_gb", 0.0))
                if len(samples) >= 3 and all(
                    float(item.get("system_ram_available_gb", 999.0)) < 0.10
                    for item in samples[-3:]
                ):
                    watchdog_errors.append("系统可用内存连续3秒低于0.10 GiB")
                guest_swap = float(latest.get("guest_swap_used_mb_total", 0.0))
                guest_swap_high_samples = (
                    guest_swap_high_samples + 1 if guest_swap > 64.0 else 0
                )
                if guest_swap_high_samples >= 3:
                    watchdog_errors.append("Android guest swap连续3次超过64 MiB")
                system_swap = float(latest.get("system_swap_used_gb", 0.0))
                if baseline_swap is not None and system_swap - baseline_swap > 1.0:
                    watchdog_errors.append("系统页面文件用量增长超过1 GiB")
                if watchdog_errors:
                    process.terminate()
                    return

        watchdog_thread = threading.Thread(
            target=watchdog, name="v2-resource-watchdog", daemon=True
        )
        watchdog_thread.start()
        state.set_phase("sampling", "v0.2原生对局采样")
        assert process.stdout is not None
        with (output_root / "training.log").open("w", encoding="utf-8") as log:
            for line in process.stdout:
                log.write(line)
                log.flush()
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if value.get("event") == "training_phase":
                    phase = str(value["phase"])
                    labels = {
                        "sampling": "v0.2原生对局采样",
                        "learner": "v0.2 Recurrent PPO更新",
                        "finalize": "健康检查与保存v0.2 Checkpoint",
                    }
                    state.set_phase(phase, labels.get(phase, phase))
                elif value.get("event") == "iteration_complete":
                    state.update_from_iteration(value)
        exit_code = process.wait()
        watchdog_stop.set()
        watchdog_thread.join(timeout=5)
        if watchdog_errors:
            raise RuntimeError(watchdog_errors[0])
        if exit_code:
            raise RuntimeError(f"v0.2 training exited with code {exit_code}")
        checkpoint_path = state.run_root / "checkpoints" / "latest.pt"
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if (
            checkpoint.get("kind") != CHECKPOINT_KIND
            or int(checkpoint["native_ticks"]) < args.target_native_ticks
        ):
            raise RuntimeError("v0.2 training exited before certified 5M target")
        state.finish()
    except Exception as error:
        state.finish(error=str(error))
    finally:
        watchdog_stop.set()
        if monitor is not None:
            monitor.stop()
            _atomic_json(output_root / "resource-samples.json", {
                "samples": monitor.samples,
            })
            _atomic_json(output_root / "resource-summary.json", monitor.summary())
        try:
            pool.stop(keep_vms=False)
        except Exception as error:
            with state.lock:
                state.warnings.append(f"Worker清理警告：{error}")
        summary = state.snapshot()
        summary.update({
            "schema_version": 1,
            "kind": "selfplay_v2_5m_supervisor_summary",
            "initialization": "scratch",
            "lambda_initial": 0.30,
            "lambda_max": 20.0,
            "reward_changed_from_v1": False,
            "error": state.error,
        })
        _atomic_json(output_root / "dashboard-summary.json", summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--target-native-ticks", type=int, default=5_000_000)
    parser.add_argument("--run-id")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open-browser", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.python.is_file():
        raise FileNotFoundError(f"Python runtime missing: {args.python}")
    revision = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
        creationflags=(
            subprocess.CREATE_NO_WINDOW
            if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        ),
    ).stdout.strip()
    if revision:
        raise RuntimeError("formal v0.2 training requires a clean worktree")
    args.data_root = args.data_root.resolve()
    args.run_id = args.run_id or (
        "selfplay-v0.2-scratch-5m-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    run_root = args.data_root / "runs" / args.run_id
    if run_root.exists():
        raise FileExistsError(f"v0.2 run already exists: {run_root}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = args.data_root / "dashboard" / f"5m-{stamp}"
    output_root.mkdir(parents=True, exist_ok=False)
    state = DashboardState(
        run_root=run_root,
        output_root=output_root,
        target_native_ticks=args.target_native_ticks,
        checkpoint_kind=CHECKPOINT_KIND,
        display_title="Self-Play v0.2 · Scratch → P050",
        display_subtitle=(
            "原生20Hz · 连续行动率λ · 2 AVD / 8 Worker · Reward保持v0.1"
        ),
    )
    server = TrainingDashboardServer(
        host=args.host,
        port=args.port,
        snapshot=state.snapshot,
        request_shutdown=state.request_shutdown,
    )
    server.start()
    host, port = server.address
    url = f"http://{host}:{port}/"
    _atomic_json(output_root / "launcher.json", {
        "schema_version": 1,
        "kind": "selfplay_v2_dashboard_launcher",
        "created_utc": state.started_utc,
        "url": url,
        "pid": os.getpid(),
        "run_id": args.run_id,
        "run_root": str(run_root),
        "target_native_ticks": args.target_native_ticks,
        "initialization": "scratch",
        "lambda_initial": 0.30,
    })
    thread = threading.Thread(
        target=_worker,
        kwargs={"args": args, "state": state, "output_root": output_root},
        name="selfplay-v2-5m",
        daemon=True,
    )
    thread.start()
    if not args.no_open_browser:
        webbrowser.open(url, new=1)
    try:
        state.shutdown_event.wait()
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
