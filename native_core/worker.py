"""Lifecycle manager for the no-window Android x86_64 libg workers.

Android is only the ABI/process container.  Battles run in Surface-free
``app_process`` services and reset in-process between episodes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

from .client import request


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _path_from_env(name: str) -> Path | None:
    """Resolve a required path from the environment, with no personal fallback.

    Returns ``None`` when the variable is unset; the CLI fails closed with a
    clear message instead of silently using a developer-specific path.
    """
    value = os.environ.get(name)
    return Path(value) if value else None


DEFAULT_SDK = _path_from_env("CR_SANDBOX_ANDROID_SDK")
DEFAULT_JDK = _path_from_env("CR_SANDBOX_JDK")
DEFAULT_DATA = _path_from_env("CR_SANDBOX_DATA")
DEFAULT_APKS = _path_from_env("CR_SANDBOX_APKS")
DEFAULT_AVD_NAMES = (
    "royale_worker_api31",
    "royale_worker_api31_b",
    "royale_worker_api31_c",
    "royale_worker_api31_d",
)


class WorkerError(RuntimeError):
    pass


def _creation_flags() -> int:
    return subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def _powershell() -> str:
    return shutil.which("pwsh.exe") or shutil.which("powershell.exe") or "powershell.exe"


@dataclass(frozen=True)
class WorkerConfig:
    sdk_root: Path | None = DEFAULT_SDK
    jdk_root: Path | None = DEFAULT_JDK
    data_root: Path | None = DEFAULT_DATA
    apk_root: Path | None = DEFAULT_APKS
    avd_name: str = "royale_worker_api31"
    emulator_port: int = 5554
    service_base_port: int = 37031
    direct_base_port: int = 38031
    cores: int = 4
    memory_mb: int = 4096

    @property
    def adb(self) -> Path:
        return self.sdk_root / "platform-tools" / "adb.exe"

    @property
    def emulator(self) -> Path:
        return self.sdk_root / "emulator" / "emulator.exe"

    @property
    def serial(self) -> str:
        return f"emulator-{self.emulator_port}"

    @property
    def avd_home(self) -> Path | None:
        return _path_from_env("CR_SANDBOX_AVD_HOME")

    @property
    def logs(self) -> Path:
        return self.data_root / "android" / "logs" / self.serial


class HeadlessWorkerPool:
    def __init__(self, config: WorkerConfig = WorkerConfig()) -> None:
        self.config = config

    def _require(self, *paths: Path | None) -> None:
        missing = [
            str(path) for path in paths if path is None or not path.is_file()
        ]
        if missing:
            raise WorkerError("missing worker input: " + ", ".join(missing))

    def _env(self) -> dict[str, str]:
        value = os.environ.copy()
        value["ANDROID_SDK_ROOT"] = str(self.config.sdk_root)
        value["ANDROID_AVD_HOME"] = str(self.config.avd_home)
        value["JAVA_HOME"] = str(self.config.jdk_root)
        return value

    def _adb(self, *args: str, timeout: float = 30.0, check: bool = True) -> str:
        self._require(self.config.adb)
        result = subprocess.run(
            [str(self.config.adb), "-s", self.config.serial, *args],
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            creationflags=_creation_flags(),
            check=False,
        )
        if check and result.returncode:
            raise WorkerError((result.stderr or result.stdout).strip())
        return result.stdout

    def vm_ready(self) -> bool:
        try:
            return (
                self._adb("get-state", timeout=2).strip() == "device"
                and self._adb("shell", "getprop", "sys.boot_completed", timeout=2).strip()
                == "1"
            )
        except Exception:
            return False

    def start_vm(self, timeout: float = 150.0) -> dict[str, Any]:
        if self.vm_ready():
            return {"ready": True, "started": False, "serial": self.config.serial}
        self._require(self.config.emulator, self.config.adb)
        self.config.logs.mkdir(parents=True, exist_ok=True)
        self.config.avd_home.mkdir(parents=True, exist_ok=True)
        stdout_path = self.config.logs / "emulator.log"
        stderr_path = self.config.logs / "emulator-error.log"
        command = [
            str(self.config.emulator), "-avd", self.config.avd_name,
            "-port", str(self.config.emulator_port), "-no-window", "-no-audio",
            "-no-boot-anim", "-no-snapshot", "-gpu", "swiftshader_indirect",
            "-accel", "on", "-memory", str(self.config.memory_mb),
            "-cores", str(self.config.cores),
        ]
        with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
            process = subprocess.Popen(
                command, cwd=self.config.data_root, env=self._env(),
                stdout=stdout, stderr=stderr, creationflags=_creation_flags(),
            )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                detail = stderr_path.read_text(encoding="utf-8", errors="replace")[-3000:]
                raise WorkerError(f"emulator exited {process.returncode}: {detail}")
            if self.vm_ready():
                self._adb("root", timeout=15, check=False)
                time.sleep(1)
                return {
                    "ready": True, "started": True, "serial": self.config.serial,
                    "launcher_pid": process.pid,
                }
            time.sleep(1)
        raise WorkerError("headless Android worker boot timed out")

    def ensure_package(self) -> dict[str, Any]:
        paths = self._adb(
            "shell", "pm", "path", "com.supercell.clashroyale", check=False
        )
        if all(name in paths for name in (
            "base.apk", "split_config.x86_64.apk", "split_install_time_asset_pack.apk"
        )):
            return {"complete": True, "installed": False}
        apks = sorted(
            self.config.apk_root.glob("*.apk"),
            key=lambda path: (path.name != "base.apk", path.name),
        )
        if len(apks) < 5:
            raise WorkerError(f"complete APK set missing under {self.config.apk_root}")
        output = self._adb(
            "install-multiple", "-r", "-t", *(str(path) for path in apks),
            timeout=600, check=False,
        )
        if "Success" not in output:
            raise WorkerError("APK installation failed: " + output.strip())
        return {"complete": True, "installed": True}

    def service_ready(self, slot: int) -> bool:
        try:
            return bool(request(
                {"op": "ping"}, port=self.config.service_base_port + slot, timeout=1
            ).get("ok"))
        except Exception:
            return False

    def configure_direct_ports(self, workers: int) -> dict[str, Any]:
        """Map host TCP ports straight through Emulator NAT, bypassing ADB proxy."""
        if workers < 1 or workers > 8:
            raise ValueError("workers must be in 1..8")
        mappings = []
        for slot in range(workers):
            host_port = self.config.direct_base_port + slot
            guest_port = self.config.service_base_port + slot
            self._adb(
                "emu", "redir", "del", f"tcp:{host_port}",
                timeout=5, check=False,
            )
            output = self._adb(
                "emu", "redir", "add", f"tcp:{host_port}:{guest_port}",
                timeout=5, check=False,
            )
            if "OK" not in output:
                raise WorkerError(
                    f"emulator redirection {host_port}->{guest_port} failed: "
                    + output.strip()
                )
            try:
                ready = bool(
                    request({"op": "ping"}, port=host_port, timeout=2).get("ok")
                )
            except Exception as error:
                raise WorkerError(
                    f"direct worker transport did not answer on {host_port}: "
                    + str(error)
                ) from error
            if not ready:
                raise WorkerError(
                    f"direct worker transport did not answer on {host_port}"
                )
            mappings.append({
                "slot": slot,
                "host_port": host_port,
                "guest_port": guest_port,
                "ready": True,
            })
        return {"kind": "emulator_tcp_redir", "mappings": mappings}

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _service_artifacts_current(self, slot: int) -> bool:
        local = {
            "lifecycle-probe.jar": PROJECT_ROOT / "artifacts" / "lifecycle-probe.jar",
            "libnative_host_bridge.so": PROJECT_ROOT / "artifacts" / "libnative_core_probe.so",
        }
        if any(not path.is_file() for path in local.values()):
            return False
        remote_root = f"/data/local/tmp/cr-native-direct-{slot}"
        for name, path in local.items():
            output = self._adb(
                "shell", "sha256sum", f"{remote_root}/{name}",
                timeout=5, check=False,
            ).strip()
            if not output or output.split()[0].lower() != self._sha256(path):
                return False
        return True

    def start_service(self, slot: int) -> dict[str, Any]:
        port = self.config.service_base_port + slot
        self._adb("forward", f"tcp:{port}", f"tcp:{port}", check=False)
        if self.service_ready(slot) and self._service_artifacts_current(slot):
            replay = json.loads(
                (PROJECT_ROOT / "examples" / "eight-card-bootstrap.json").read_text(
                    encoding="utf-8-sig"
                )
            )
            state = request(
                {"op": "reset", "replay": replay}, port=port, timeout=30
            )["state"]
            return self._attest(slot, state, started=False)
        command = [
            _powershell(), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
            str(PROJECT_ROOT / "scripts" / "start_direct_service.ps1"),
            "-Adb", str(self.config.adb), "-Serial", self.config.serial,
            "-Port", str(port), "-Slot", str(slot), "-DataRoot", str(self.config.data_root),
            "-BootstrapReplayJson", str(PROJECT_ROOT / "examples" / "eight-card-bootstrap.json"),
        ]
        result = subprocess.run(
            command, cwd=PROJECT_ROOT, text=True, encoding="utf-8", errors="replace",
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=420,
            creationflags=_creation_flags(), check=False,
        )
        if result.returncode:
            raise WorkerError((result.stderr or result.stdout).strip())
        state = request({"op": "observe"}, port=port, timeout=5)["state"]
        return self._attest(slot, state, started=True)

    def _attest(self, slot: int, state: dict[str, Any], *, started: bool) -> dict[str, Any]:
        episode = state.get("episode", {})
        towers = episode.get("crown_towers", [])
        maxima = sorted(int(item.get("max_hp", -1)) for item in towers)
        if not (
            state.get("coherent") is True and state.get("entity_count") == 6
            and state.get("state_hash_scope") == "public-observe-v6"
            and maxima == [3052, 3052, 3052, 3052, 4824, 4824]
        ):
            raise WorkerError(f"slot {slot} failed native opening attestation")
        return {
            "slot": slot, "port": self.config.service_base_port + slot,
            "ready": True, "started": started, "tick": state["tick"],
            "state_hash": state.get("state_hash"), "tower_max_hp": maxima,
        }

    def ensure_ready(
        self, workers: int, *, configure_direct: bool = True
    ) -> dict[str, Any]:
        if workers < 1 or workers > 8:
            raise ValueError("workers must be in 1..8")
        self.config.data_root.mkdir(parents=True, exist_ok=True)
        vm = self.start_vm()
        package = self.ensure_package()
        # DataTables/libg cold initialization is intentionally serialized.
        services = [self.start_service(slot) for slot in range(workers)]
        result = {
            "vm": vm,
            "package": package,
            "services": services,
        }
        if configure_direct:
            result["direct_transport"] = self.configure_direct_ports(workers)
        return result

    def stop(self, workers: int, *, keep_vm: bool = True) -> dict[str, Any]:
        services = []
        for slot in range(workers):
            port = self.config.service_base_port + slot
            try:
                request({"op": "shutdown"}, port=port, timeout=2)
            except Exception:
                pass
            script = PROJECT_ROOT / "scripts" / "stop_direct_service.ps1"
            subprocess.run(
                [_powershell(), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                 str(script), "-Adb", str(self.config.adb), "-Serial", self.config.serial,
                 "-Port", str(port), "-Slot", str(slot)],
                cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                creationflags=_creation_flags(), check=False,
            )
            services.append({"slot": slot, "port": port, "stopped": not self.service_ready(slot)})
        vm_stopped = False
        if not keep_vm and self.vm_ready():
            self._adb("emu", "kill", timeout=10, check=False)
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and self.vm_ready():
                time.sleep(0.5)
            vm_stopped = not self.vm_ready()
        return {"services": services, "vm_stopped": vm_stopped}


class MultiAvdWorkerPool:
    """Horizontal AVD pool with four isolated libg Workers per VM."""

    def __init__(
        self,
        *,
        avds: int,
        workers_per_avd: int = 4,
        avd_names: tuple[str, ...] = DEFAULT_AVD_NAMES,
        emulator_base_port: int = 5554,
        service_base_port: int = 37031,
        service_port_stride: int = 100,
        direct_base_port: int = 38031,
        cores_per_avd: int = 4,
        memory_mb_per_avd: int = 4096,
    ) -> None:
        if avds < 1 or avds > len(avd_names):
            raise ValueError(f"avds must be in 1..{len(avd_names)}")
        if workers_per_avd < 1 or workers_per_avd > 4:
            raise ValueError("workers_per_avd must be in 1..4")
        self.avds = avds
        self.workers_per_avd = workers_per_avd
        self.pools = [
            HeadlessWorkerPool(WorkerConfig(
                avd_name=avd_names[index],
                emulator_port=emulator_base_port + 2 * index,
                service_base_port=service_base_port + service_port_stride * index,
                direct_base_port=(
                    direct_base_port + workers_per_avd * index
                ),
                cores=cores_per_avd,
                memory_mb=memory_mb_per_avd,
            ))
            for index in range(avds)
        ]

    @property
    def workers(self) -> int:
        return self.avds * self.workers_per_avd

    def environment_ports(self, transport: str) -> list[int]:
        if transport not in ("direct", "adb"):
            raise ValueError("transport must be direct or adb")
        ports: list[int] = []
        for pool in self.pools:
            base = (
                pool.config.direct_base_port
                if transport == "direct"
                else pool.config.service_base_port
            )
            ports.extend(base + slot for slot in range(self.workers_per_avd))
        return ports

    def ensure_ready(self, *, configure_direct: bool = True) -> dict[str, Any]:
        instances = []
        for avd_index, pool in enumerate(self.pools):
            avd_ini = pool.config.avd_home / f"{pool.config.avd_name}.ini"
            if not avd_ini.is_file():
                raise WorkerError(f"AVD is not provisioned: {avd_ini}")
            state = pool.ensure_ready(
                self.workers_per_avd,
                configure_direct=configure_direct,
            )
            state["avd_index"] = avd_index
            state["avd_name"] = pool.config.avd_name
            state["serial"] = pool.config.serial
            instances.append(state)
        return {
            "avds": self.avds,
            "workers_per_avd": self.workers_per_avd,
            "workers": self.workers,
            "instances": instances,
        }

    def configure_direct_ports(self) -> dict[str, Any]:
        return {
            "avds": [
                {
                    "avd_index": index,
                    "serial": pool.config.serial,
                    **pool.configure_direct_ports(self.workers_per_avd),
                }
                for index, pool in enumerate(self.pools)
            ]
        }

    def status(self) -> dict[str, Any]:
        return {
            "instances": [
                {
                    "avd_index": index,
                    "avd_name": pool.config.avd_name,
                    "serial": pool.config.serial,
                    "vm_ready": pool.vm_ready(),
                    "services": [
                        pool.service_ready(slot)
                        for slot in range(self.workers_per_avd)
                    ],
                }
                for index, pool in enumerate(self.pools)
            ]
        }

    def stop(self, *, keep_vms: bool = True) -> dict[str, Any]:
        return {
            "instances": [
                {
                    "avd_index": index,
                    "serial": pool.config.serial,
                    **pool.stop(
                        self.workers_per_avd,
                        keep_vm=keep_vms,
                    ),
                }
                for index, pool in enumerate(self.pools)
            ]
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "status", "stop"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--base-port", type=int, default=37031)
    parser.add_argument("--sdk-root", type=Path, default=DEFAULT_SDK)
    parser.add_argument("--jdk-root", type=Path, default=DEFAULT_JDK)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--apk-root", type=Path, default=DEFAULT_APKS)
    parser.add_argument("--avd-name", default="royale_worker_api31")
    parser.add_argument("--emulator-port", type=int, default=5554)
    parser.add_argument("--direct-base-port", type=int, default=38031)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--memory-mb", type=int, default=4096)
    parser.add_argument("--transport", choices=("direct", "adb"), default="direct")
    parser.add_argument("--stop-vm", action="store_true")
    args = parser.parse_args()
    undefined = [
        name
        for name, value in {
            "CR_SANDBOX_ANDROID_SDK": args.sdk_root,
            "CR_SANDBOX_JDK": args.jdk_root,
            "CR_SANDBOX_DATA": args.data_root,
            "CR_SANDBOX_APKS": args.apk_root,
        }.items()
        if value is None
    ]
    if not os.environ.get("CR_SANDBOX_AVD_HOME"):
        undefined.append("CR_SANDBOX_AVD_HOME")
    if undefined:
        print(
            "error: missing environment variable(s): " + ", ".join(undefined),
            file=sys.stderr,
        )
        print(
            "Copy runtime.env.example.ps1 to runtime.env.ps1, edit the paths "
            "for this machine, and dot-source it (`. .\\runtime.env.ps1`) "
            "before running the worker.",
            file=sys.stderr,
        )
        return 2
    pool = HeadlessWorkerPool(WorkerConfig(
        sdk_root=args.sdk_root,
        jdk_root=args.jdk_root,
        data_root=args.data_root,
        apk_root=args.apk_root,
        avd_name=args.avd_name,
        emulator_port=args.emulator_port,
        service_base_port=args.base_port,
        direct_base_port=args.direct_base_port,
        cores=args.cores,
        memory_mb=args.memory_mb,
    ))
    if args.action == "start":
        value = pool.ensure_ready(
            args.workers, configure_direct=args.transport == "direct"
        )
    elif args.action == "stop":
        value = pool.stop(args.workers, keep_vm=not args.stop_vm)
    else:
        value = {
            "vm_ready": pool.vm_ready(),
            "services": [pool.service_ready(i) for i in range(args.workers)],
        }
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
