"""Resume frozen P010 self-play to a bounded target with a live web dashboard."""

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

import psutil
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from native_core.worker import MultiAvdWorkerPool
from training.dashboard import TrainingDashboardServer
from training.resource_monitor import TrainingResourceMonitor
from training.run_contract import CHECKPOINT_KIND
from training.schema import RunStore


DEFAULT_PYTHON = Path(r"D:\AI_data\runtime\venv\Scripts\python.exe")
DEFAULT_RUNTIME_ROOT = Path(
    r"D:\AI_data\worktrees\CR-Native-Core-selfplay-a24e0ba-lf"
)
DEFAULT_DATA_ROOT = Path(r"D:\AI_data\cr-native-core\selfplay-v0.1")
DEFAULT_RUN_ID = "selfplay-v0.1-stage-a-20260823T141402Z"
DEFAULT_ADB = Path(
    r"D:\Codex\toolchains\android-sdk\platform-tools\adb.exe"
)
CARD_IDS = (
    26000000, 26000001, 26000003, 26000010,
    26000014, 26000021, 27000000, 28000001,
)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    RunStore._atomic_json(path, value)


class DashboardState:
    def __init__(
        self,
        *,
        run_root: Path,
        output_root: Path,
        target_native_ticks: int,
        checkpoint_kind: str = CHECKPOINT_KIND,
        display_title: str = "Self-Play v0.1 · P010 → P020",
        display_subtitle: str = (
            "冻结原生20Hz · 2 AVD / 8 Worker · Recurrent PPO · 塔血势函数"
        ),
    ) -> None:
        self.run_root = run_root
        self.output_root = output_root
        self.target_native_ticks = target_native_ticks
        self.checkpoint_kind = checkpoint_kind
        self.display_title = display_title
        self.display_subtitle = display_subtitle
        self.events_path = run_root / "logs" / "events.jsonl"
        self.lock = threading.RLock()
        self.phase = "initializing"
        self.phase_label = "准备冻结运行时"
        self.status = "running"
        self.running = True
        self.error: str | None = None
        self.warnings: list[str] = []
        self.started_monotonic = time.perf_counter()
        self.started_utc = datetime.now(timezone.utc).isoformat()
        self.training_started_monotonic: float | None = None
        self.completed_utc: str | None = None
        self.training_pid: int | None = None
        self.monitor: TrainingResourceMonitor | None = None
        self.history: list[dict[str, Any]] = []
        self.resource_history: list[dict[str, Any]] = []
        self.latest: dict[str, Any] = {}
        self.native_ticks = 0
        self.episodes = 0
        self.shutdown_event = threading.Event()
        self._event_offset = 0
        self._event_partial = ""
        self._load_checkpoint()
        self._refresh_events()

    def _load_checkpoint(self) -> None:
        path = self.run_root / "checkpoints" / "latest.pt"
        if not path.is_file():
            return
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("kind") != self.checkpoint_kind:
            raise RuntimeError("latest checkpoint contract mismatch")
        self.native_ticks = int(checkpoint["native_ticks"])
        self.episodes = int(checkpoint["completed_episodes"])
        self.latest = {
            "iteration": int(checkpoint["iteration"]),
            "native_ticks": self.native_ticks,
            "episodes": self.episodes,
            "metrics": dict(checkpoint.get("metrics", {})),
            "behavior": dict(checkpoint.get("behavior", {})),
            "sampling_profile": dict(checkpoint.get("sampling_profile", {})),
            "model_digest": checkpoint.get("current_model_digest"),
        }

    def _refresh_events(self) -> None:
        if not self.events_path.is_file():
            return
        with self.events_path.open("r", encoding="utf-8") as stream:
            stream.seek(self._event_offset)
            chunk = stream.read()
            self._event_offset = stream.tell()
        if not chunk:
            return
        text = self._event_partial + chunk
        lines = text.splitlines(keepends=True)
        self._event_partial = ""
        if lines and not lines[-1].endswith(("\n", "\r")):
            self._event_partial = lines.pop()
        by_iteration = {int(item["iteration"]): item for item in self.history}
        for raw in lines:
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if value.get("event") != "iteration_complete":
                continue
            iteration = int(value["iteration"])
            by_iteration[iteration] = {
                "iteration": iteration,
                "native_ticks": int(value["native_ticks"]),
                "episodes": int(value["episodes"]),
                "metrics": dict(value.get("metrics", {})),
                "behavior": dict(value.get("behavior", {})),
            }
        self.history = [by_iteration[key] for key in sorted(by_iteration)]

    def set_phase(self, phase: str, label: str) -> None:
        with self.lock:
            self.phase = phase
            self.phase_label = label
            if self.monitor is not None:
                self.monitor.set_phase(phase)

    def update_from_iteration(self, event: dict[str, Any]) -> None:
        with self.lock:
            self.native_ticks = int(event["native_ticks"])
            self.episodes = int(event["episodes"])
            self.latest = {
                "iteration": int(event["iteration"]),
                "native_ticks": self.native_ticks,
                "episodes": self.episodes,
                "metrics": dict(event.get("metrics", {})),
                "behavior": dict(event.get("behavior", {})),
                "sampling_profile": dict(event.get("sampling_profile", {})),
            }
            self._refresh_events()

    def attach_monitor(self, monitor: TrainingResourceMonitor) -> None:
        with self.lock:
            self.monitor = monitor

    def current_resource(self) -> dict[str, Any]:
        if self.monitor is not None and self.monitor.samples:
            sample = dict(self.monitor.samples[-1])
            self.resource_history = [
                {
                    key: value for key, value in item.items()
                    if key in (
                        "monotonic_seconds", "host_cpu_percent",
                        "gpu_utilization_percent", "system_ram_available_gb",
                    )
                }
                for item in self.monitor.samples[-120:]
            ]
            return sample
        memory = psutil.virtual_memory()
        return {
            "host_cpu_percent": psutil.cpu_percent(None),
            "system_ram_available_gb": memory.available / (1024.0 ** 3),
            "guest_worker_count": 0.0,
        }

    def finish(self, *, error: str | None = None) -> None:
        with self.lock:
            self.running = False
            self.completed_utc = datetime.now(timezone.utc).isoformat()
            self.error = error
            if error:
                self.status = "error"
                self.phase = "error"
                self.phase_label = "训练已停止：发生错误"
            else:
                self.status = "complete"
                self.phase = "complete"
                self.phase_label = "2M训练完成 · Worker已关闭"
            try:
                self._load_checkpoint()
                self._refresh_events()
            except Exception as load_error:
                if not self.error:
                    self.error = str(load_error)
                    self.status = "error"

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            self._refresh_events()
            resource = self.current_resource()
            progress = self.native_ticks / max(1, self.target_native_ticks)
            metrics = self.latest.get("metrics", {})
            steps_per_second = float(
                metrics.get("training_steps_per_second", 0.0) or 0.0
            )
            remaining = max(0, self.target_native_ticks - self.native_ticks)
            eta = remaining / steps_per_second if steps_per_second > 0 else None
            warnings = list(self.warnings)
            available = resource.get("system_ram_available_gb")
            if available is not None and float(available) < 1.0:
                warnings.append(
                    f"系统可用内存仅 {float(available):.2f} GiB"
                )
            rejection_rate = float(
                self.latest.get("behavior", {}).get(
                    "native_action_rejection_rate", 0.0
                ) or 0.0
            )
            if rejection_rate:
                warnings.append(f"原生动作拒绝率 {rejection_rate:.4%}")
            if self.error:
                warnings.append(self.error)
            return {
                "schema_version": 1,
                "kind": "selfplay_training_dashboard_state",
                "run_id": self.run_root.name,
                "display_title": self.display_title,
                "display_subtitle": self.display_subtitle,
                "run_root": str(self.run_root),
                "output_root": str(self.output_root),
                "status": self.status,
                "running": self.running,
                "phase": self.phase,
                "phase_label": self.phase_label,
                "started_utc": self.started_utc,
                "completed_utc": self.completed_utc,
                "training_pid": self.training_pid,
                "native_ticks": self.native_ticks,
                "target_native_ticks": self.target_native_ticks,
                "progress": min(1.0, progress),
                "eta_seconds": eta,
                "episodes": self.episodes,
                "latest": self.latest,
                "history": self.history,
                "resource": resource,
                "resource_history": self.resource_history,
                "warnings": warnings,
            }

    def request_shutdown(self) -> tuple[bool, str]:
        with self.lock:
            if self.running:
                return False, "训练仍在运行；为避免误停，本面板只能在训练结束后关闭。"
            self.shutdown_event.set()
            return True, "面板正在关闭。"


def _validate_frozen_runtime(
    runtime_root: Path,
    checkpoint: dict[str, Any],
    python: Path,
) -> None:
    if not (runtime_root / ".git").exists():
        raise FileNotFoundError(f"frozen runtime worktree is missing: {runtime_root}")
    flags = (
        subprocess.CREATE_NO_WINDOW
        if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=runtime_root,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
        creationflags=flags,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=runtime_root,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
        creationflags=flags,
    ).stdout.strip()
    expected = str(checkpoint["config"].get("source_revision", ""))
    if revision != expected or dirty:
        raise RuntimeError(
            f"frozen runtime mismatch: revision={revision}, expected={expected}, "
            f"dirty={bool(dirty)}"
        )
    implementation = subprocess.run(
        [
            str(python),
            "-c",
            "from training.train import _implementation_digest; "
            "print(_implementation_digest())",
        ],
        cwd=runtime_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        creationflags=flags,
    ).stdout.strip()
    expected_implementation = str(
        checkpoint["config"].get("implementation_digest", "")
    )
    if implementation != expected_implementation:
        raise RuntimeError(
            "frozen runtime byte digest mismatch: "
            f"runtime={implementation}, expected={expected_implementation}"
        )


def _training_worker(
    *,
    args: argparse.Namespace,
    state: DashboardState,
    output_root: Path,
) -> None:
    pool = MultiAvdWorkerPool(avds=2, workers_per_avd=4)
    recycle_pool = MultiAvdWorkerPool(avds=4, workers_per_avd=4)
    monitor: TrainingResourceMonitor | None = None
    process: subprocess.Popen[str] | None = None
    watchdog_stop = threading.Event()
    watchdog_error: list[str] = []
    try:
        state.set_phase("startup", "关闭旧Worker")
        recycle_pool.stop(keep_vms=False)
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
        state.training_started_monotonic = time.perf_counter()
        command = [
            str(args.python), "-m", "training.train",
            "--iterations", "1000000",
            "--target-native-ticks", str(args.target_native_ticks),
            "--episodes-per-iteration", "8",
            "--workers", "8",
            "--avds", "2",
            "--workers-per-avd", "4",
            "--seed", "1",
            "--max-ticks", "7200",
            "--device", "cuda",
            "--reward", "tower_hp_potential_v1",
            "--transport", "direct",
            "--data-root", str(args.data_root),
            "--run-id", args.run_id,
            "--resume", str(args.resume),
            "--skip-worker-start",
            "--emit-phase-events",
        ]
        flags = (
            subprocess.CREATE_NO_WINDOW
            if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        )
        process = subprocess.Popen(
            command,
            cwd=args.runtime_root,
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
            while not watchdog_stop.wait(1.0):
                if process is None or process.poll() is not None or monitor is None:
                    return
                samples = monitor.samples
                if not samples:
                    continue
                if baseline_swap is None:
                    baseline_swap = float(samples[0].get("system_swap_used_gb", 0.0))
                if len(samples) >= 3 and all(
                    float(item.get("system_ram_available_gb", 999.0)) < 0.10
                    for item in samples[-3:]
                ):
                    watchdog_error.append("系统可用内存连续3秒低于0.10 GiB")
                if float(samples[-1].get("guest_swap_used_mb_total", 0.0)) > 0.0:
                    watchdog_error.append("Android guest swap 已启用")
                current_swap = float(samples[-1].get("system_swap_used_gb", 0.0))
                if baseline_swap is not None and current_swap - baseline_swap > 1.0:
                    watchdog_error.append("系统页面文件用量增长超过1 GiB")
                if watchdog_error:
                    process.terminate()
                    return

        watchdog_thread = threading.Thread(
            target=watchdog, name="training-memory-watchdog", daemon=True
        )
        watchdog_thread.start()
        state.set_phase("sampling", "原生对局采样")
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
                        "sampling": "原生对局采样",
                        "learner": "Recurrent PPO 更新",
                        "finalize": "健康检查与保存 Checkpoint",
                    }
                    state.set_phase(phase, labels.get(phase, phase))
                elif value.get("event") == "iteration_complete":
                    state.update_from_iteration(value)
        exit_code = process.wait()
        watchdog_stop.set()
        watchdog_thread.join(timeout=5)
        if watchdog_error:
            raise RuntimeError(watchdog_error[0])
        if exit_code:
            raise RuntimeError(f"training process exited with code {exit_code}")
        checkpoint = torch.load(
            args.resume, map_location="cpu", weights_only=False
        )
        if int(checkpoint["native_ticks"]) < args.target_native_ticks:
            raise RuntimeError("training process exited before the 2M target")
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
        except Exception as cleanup_error:
            with state.lock:
                state.warnings.append(f"Worker清理警告：{cleanup_error}")
        summary = state.snapshot()
        summary["runtime_root"] = str(args.runtime_root)
        summary["resume_checkpoint"] = str(args.resume)
        summary["training_semantics_changed"] = False
        _atomic_json(output_root / "dashboard-summary.json", summary)
        marker = {
            "schema_version": 1,
            "kind": "selfplay_v0_1_resume_to_2m",
            "authorized_utc": state.started_utc,
            "completed_utc": state.completed_utc,
            "source_runtime": str(args.runtime_root),
            "resume_checkpoint": str(args.resume),
            "target_native_ticks": args.target_native_ticks,
            "status": state.status,
            "error": state.error,
            "dashboard_output": str(output_root),
            "training_semantics_changed": False,
        }
        _atomic_json(state.run_root / "RESUME_TO_2M.json", marker)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--target-native-ticks", type=int, default=2_000_000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open-browser", action="store_true")
    parser.add_argument("--preview", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.runtime_root = args.runtime_root.resolve()
    args.data_root = args.data_root.resolve()
    run_root = args.data_root / "runs" / args.run_id
    args.resume = run_root / "checkpoints" / "latest.pt"
    if not args.python.is_file() or not args.resume.is_file():
        raise FileNotFoundError("Python runtime or latest checkpoint is missing")
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    if checkpoint.get("kind") != CHECKPOINT_KIND:
        raise RuntimeError("resume checkpoint kind mismatch")
    if int(checkpoint["native_ticks"]) >= args.target_native_ticks:
        raise RuntimeError("checkpoint has already reached the requested target")
    _validate_frozen_runtime(args.runtime_root, checkpoint, args.python)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = args.data_root / "dashboard" / f"resume-2m-{stamp}"
    output_root.mkdir(parents=True, exist_ok=False)
    state = DashboardState(
        run_root=run_root,
        output_root=output_root,
        target_native_ticks=args.target_native_ticks,
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
        "kind": "training_dashboard_launcher",
        "created_utc": state.started_utc,
        "url": url,
        "pid": os.getpid(),
        "run_root": str(run_root),
        "runtime_root": str(args.runtime_root),
        "target_native_ticks": args.target_native_ticks,
    })
    if args.preview:
        state.running = False
        state.status = "complete"
        state.phase = "preview"
        state.phase_label = "面板预览 · 未启动训练"
    else:
        worker = threading.Thread(
            target=_training_worker,
            kwargs={"args": args, "state": state, "output_root": output_root},
            name="selfplay-resume-2m",
            daemon=True,
        )
        worker.start()
    if not args.no_open_browser:
        webbrowser.open(url, new=1)
    try:
        state.shutdown_event.wait()
    except KeyboardInterrupt:
        if state.running:
            raise RuntimeError("refusing to hide a running training process")
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
