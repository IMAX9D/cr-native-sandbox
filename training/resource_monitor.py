"""Low-overhead host, GPU and Android resource sampling for scaling runs."""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import subprocess
import threading
import time
from typing import Any, Iterable

import psutil
import pynvml


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _statistics(values: Iterable[float]) -> dict[str, float]:
    samples = [float(value) for value in values]
    if not samples:
        return {}
    return {
        "mean": statistics.fmean(samples),
        "p50": _percentile(samples, 0.50),
        "p95": _percentile(samples, 0.95),
        "p99": _percentile(samples, 0.99),
        "min": min(samples),
        "max": max(samples),
    }


class TrainingResourceMonitor:
    """Sample resources without touching the policy/environment process."""

    def __init__(
        self,
        *,
        adb: Path,
        serials: list[str],
        emulator_ports: list[int],
        workers_per_avd: int,
        cores_per_avd: int = 4,
        interval_seconds: float = 1.0,
        guest_interval_seconds: float = 5.0,
    ) -> None:
        self.adb = adb
        self.serials = serials
        self.emulator_ports = emulator_ports
        self.workers_per_avd = workers_per_avd
        self.cores_per_avd = cores_per_avd
        self.interval_seconds = interval_seconds
        self.guest_interval_seconds = guest_interval_seconds
        self.phase = "startup"
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._qemu: dict[str, psutil.Process] = {}
        self._last_guest: dict[str, dict[str, Any]] = {}
        self._nvml_handle: Any = None

    def set_phase(self, phase: str) -> None:
        self.phase = phase

    def _find_qemu_processes(self) -> None:
        wanted = {str(port): f"emulator-{port}" for port in self.emulator_ports}
        for process in psutil.process_iter(("name", "cmdline")):
            try:
                if "qemu-system" not in (process.info["name"] or ""):
                    continue
                command = process.info["cmdline"] or []
                for index, value in enumerate(command[:-1]):
                    if value == "-port" and command[index + 1] in wanted:
                        self._qemu[wanted[command[index + 1]]] = process
                        process.cpu_percent(None)
            except (psutil.Error, OSError):
                continue

    def _guest_snapshot(self, serial: str) -> dict[str, Any]:
        command = [
            str(self.adb), "-s", serial, "shell",
            "cat /proc/meminfo; echo __CR_PS__; ps -A -o PID,RSS,ARGS",
        ]
        result = subprocess.run(
            command,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
            creationflags=(
                subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW")
                else 0
            ),
            check=False,
        )
        if result.returncode:
            return {"error": (result.stderr or result.stdout).strip()}
        memory: dict[str, int] = {}
        worker_rss_kb: list[int] = []
        in_processes = False
        for line in result.stdout.splitlines():
            if line.strip() == "__CR_PS__":
                in_processes = True
                continue
            if not in_processes:
                if ":" in line:
                    name, raw = line.split(":", 1)
                    fields = raw.strip().split()
                    if fields and fields[0].isdigit():
                        memory[name] = int(fields[0])
                continue
            if "royale.nativehost.JniHost" not in line:
                continue
            fields = line.strip().split(None, 2)
            if len(fields) >= 2 and fields[1].isdigit():
                worker_rss_kb.append(int(fields[1]))
        return {
            "mem_total_mb": memory.get("MemTotal", 0) / 1024.0,
            "mem_available_mb": memory.get("MemAvailable", 0) / 1024.0,
            "swap_used_mb": (
                memory.get("SwapTotal", 0) - memory.get("SwapFree", 0)
            ) / 1024.0,
            "worker_rss_mb": [value / 1024.0 for value in worker_rss_kb],
            "worker_count": len(worker_rss_kb),
        }

    def _sample(self, *, refresh_guest: bool) -> dict[str, Any]:
        host_memory = psutil.virtual_memory()
        sample: dict[str, Any] = {
            "monotonic_seconds": time.perf_counter(),
            "phase": self.phase,
            "host_cpu_percent": psutil.cpu_percent(None),
            "system_ram_used_gb": host_memory.used / (1024.0 ** 3),
            "system_ram_available_gb": host_memory.available / (1024.0 ** 3),
        }
        if self._nvml_handle is not None:
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
            gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            sample.update({
                "gpu_utilization_percent": float(utilization.gpu),
                "gpu_memory_utilization_percent": float(utilization.memory),
                "gpu_vram_used_mb": gpu_memory.used / (1024.0 ** 2),
                "gpu_vram_total_mb": gpu_memory.total / (1024.0 ** 2),
            })

        qemu: dict[str, dict[str, float]] = {}
        qemu_cpu_total = 0.0
        qemu_rss_total_mb = 0.0
        for serial, process in self._qemu.items():
            try:
                cpu = float(process.cpu_percent(None))
                rss = process.memory_info().rss / (1024.0 ** 2)
            except (psutil.Error, OSError):
                continue
            qemu[serial] = {"cpu_percent": cpu, "rss_mb": rss}
            qemu_cpu_total += cpu
            qemu_rss_total_mb += rss
        sample["qemu"] = qemu
        sample["qemu_cpu_percent_total"] = qemu_cpu_total
        sample["qemu_rss_mb_total"] = qemu_rss_total_mb
        allocated = max(1, len(self.serials) * self.cores_per_avd)
        sample["avd_allocated_cpu_utilization_percent"] = (
            qemu_cpu_total / allocated
        )

        if refresh_guest:
            for serial in self.serials:
                try:
                    self._last_guest[serial] = self._guest_snapshot(serial)
                except (OSError, subprocess.SubprocessError) as error:
                    self._last_guest[serial] = {"error": str(error)}
        sample["guest"] = json.loads(json.dumps(self._last_guest))
        valid_guest = [
            value for value in self._last_guest.values()
            if "error" not in value
        ]
        worker_rss = [
            rss
            for value in valid_guest
            for rss in value.get("worker_rss_mb", [])
        ]
        sample["guest_worker_count"] = float(len(worker_rss))
        sample["guest_worker_rss_mb_total"] = float(sum(worker_rss))
        sample["guest_worker_rss_mb_mean"] = (
            statistics.fmean(worker_rss) if worker_rss else 0.0
        )
        available = [value["mem_available_mb"] for value in valid_guest]
        sample["guest_mem_available_mb_min"] = min(available) if available else 0.0
        sample["guest_swap_used_mb_total"] = float(
            sum(value["swap_used_mb"] for value in valid_guest)
        )
        return sample

    def _run(self) -> None:
        next_guest = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            self.samples.append(self._sample(refresh_guest=now >= next_guest))
            if now >= next_guest:
                next_guest = now + self.guest_interval_seconds
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        self._find_qemu_processes()
        psutil.cpu_percent(None)
        try:
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except pynvml.NVMLError:
            self._nvml_handle = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.interval_seconds * 2.0))
        if self._nvml_handle is not None:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass

    def summary(self) -> dict[str, Any]:
        scalar_keys = (
            "host_cpu_percent",
            "system_ram_used_gb",
            "system_ram_available_gb",
            "gpu_utilization_percent",
            "gpu_memory_utilization_percent",
            "gpu_vram_used_mb",
            "qemu_cpu_percent_total",
            "qemu_rss_mb_total",
            "avd_allocated_cpu_utilization_percent",
            "guest_worker_count",
            "guest_worker_rss_mb_total",
            "guest_worker_rss_mb_mean",
            "guest_mem_available_mb_min",
            "guest_swap_used_mb_total",
        )
        phases = sorted({str(sample["phase"]) for sample in self.samples})
        result: dict[str, Any] = {
            "sample_count": len(self.samples),
            "interval_seconds": self.interval_seconds,
            "overall": {},
            "phases": {},
        }
        for key in scalar_keys:
            result["overall"][key] = _statistics(
                sample[key] for sample in self.samples if key in sample
            )
        for phase in phases:
            selected = [sample for sample in self.samples if sample["phase"] == phase]
            result["phases"][phase] = {
                key: _statistics(
                    sample[key] for sample in selected if key in sample
                )
                for key in scalar_keys
            }
        return result
