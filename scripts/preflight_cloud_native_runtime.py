"""Read-only admission check for co-located native AVD workers and GPU learner."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess


def _read(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _cgroup_limit() -> int | None:
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        value = _read(path).strip()
        if value and value != "max":
            try:
                return int(value)
            except ValueError:
                pass
    return None


def _nvidia() -> dict[str, object]:
    command = shutil.which("nvidia-smi")
    if not command:
        return {"available": False, "gpus": []}
    result = subprocess.run(
        [command, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=False,
    )
    rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    return {"available": result.returncode == 0 and bool(rows), "gpus": rows}


def main() -> int:
    cpu = _read("/proc/cpuinfo").lower()
    virtualization_flag = " vmx " in f" {cpu} " or " svm " in f" {cpu} "
    kvm_exists = Path("/dev/kvm").exists()
    kvm_rw = os.access("/dev/kvm", os.R_OK | os.W_OK) if kvm_exists else False
    memory_kib = 0
    for line in _read("/proc/meminfo").splitlines():
        if line.startswith("MemTotal:"):
            memory_kib = int(line.split()[1])
            break
    disk = shutil.disk_usage("/root/autodl-tmp" if Path("/root/autodl-tmp").exists() else "/")
    gpu = _nvidia()
    checks = {
        "linux": os.name == "posix",
        "virtualization_flag": virtualization_flag,
        "dev_kvm_exists": kvm_exists,
        "dev_kvm_read_write": kvm_rw,
        "memory_gib": memory_kib / 1024 / 1024,
        "cgroup_memory_limit_gib": (
            _cgroup_limit() / 1024**3 if _cgroup_limit() is not None else None
        ),
        "data_disk_free_gib": disk.free / 1024**3,
        "nvidia": gpu,
    }
    avd_admitted = bool(
        checks["linux"]
        and virtualization_flag
        and kvm_exists
        and kvm_rw
        and checks["memory_gib"] >= 32
        and checks["data_disk_free_gib"] >= 80
    )
    learner_admitted = bool(gpu["available"] and checks["memory_gib"] >= 32)
    result = {
        "kind": "cr_native_cloud_runtime_preflight_v1",
        "checks": checks,
        "avd_workers_admitted": avd_admitted,
        "gpu_learner_admitted": learner_admitted,
        "co_located_training_admitted": avd_admitted and learner_admitted,
        "failure_policy": "fail_closed_no_software_emulator_fallback",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["co_located_training_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
