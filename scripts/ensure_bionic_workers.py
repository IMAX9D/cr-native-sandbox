"""Idempotently launch the headless Bionic libg Worker fleet on Linux."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from native_core.client import request as native_request


BOOT_JARS = (
    "/apex/com.android.art/javalib/core-oj.jar",
    "/apex/com.android.art/javalib/core-libart.jar",
    "/apex/com.android.art/javalib/okhttp.jar",
    "/apex/com.android.art/javalib/bouncycastle.jar",
    "/apex/com.android.art/javalib/apache-xml.jar",
    "/system/framework/framework.jar",
    "/system/framework/framework-graphics.jar",
    "/system/framework/ext.jar",
    "/system/framework/telephony-common.jar",
    "/system/framework/voip-common.jar",
    "/system/framework/ims-common.jar",
    "/apex/com.android.i18n/javalib/core-icu4j.jar",
    "/apex/com.android.appsearch/javalib/framework-appsearch.jar",
    "/apex/com.android.conscrypt/javalib/conscrypt.jar",
    "/apex/com.android.ipsec/javalib/android.net.ipsec.ike.jar",
    "/apex/com.android.media/javalib/updatable-media.jar",
    "/apex/com.android.mediaprovider/javalib/framework-mediaprovider.jar",
    "/apex/com.android.os.statsd/javalib/framework-statsd.jar",
    "/apex/com.android.permission/javalib/framework-permission.jar",
    "/apex/com.android.permission/javalib/framework-permission-s.jar",
    "/apex/com.android.scheduling/javalib/framework-scheduling.jar",
    "/apex/com.android.sdkext/javalib/framework-sdkextensions.jar",
    "/apex/com.android.tethering/javalib/framework-connectivity.jar",
    "/apex/com.android.tethering/javalib/framework-tethering.jar",
    "/apex/com.android.wifi/javalib/framework-wifi.jar",
)


def ready(port: int, timeout: float = 0.25) -> bool:
    try:
        response = native_request(
            {"op": "status"}, host="127.0.0.1", port=port,
            timeout=max(timeout, 0.5),
        )
        state = response.get("state") if isinstance(response, Mapping) else None
        read_ok = state.get("read_ok") if isinstance(state, Mapping) else None
        return bool(
            response.get("ok")
            and isinstance(state, Mapping)
            and int(state.get("current_state_type", -1)) == 4
            and int(state.get("tick", -1)) >= 0
            and state.get("battle") not in (None, "0x0")
            and state.get("replay_data") not in (None, "0x0")
            and isinstance(read_ok, Mapping)
            and all(
                bool(read_ok.get(name, False))
                for name in ("root", "context", "manager_fields", "battle", "tick")
            )
        )
    except (OSError, ConnectionError, TypeError, ValueError):
        return False


def port_bindable(port: int) -> bool:
    """Return whether a replacement server can bind after a Worker exits."""
    candidate = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        candidate.bind(("0.0.0.0", port))
        return True
    except OSError:
        return False
    finally:
        candidate.close()


def wait_port_bindable(port: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if port_bindable(port):
            return True
        time.sleep(0.25)
    return port_bindable(port)


def command(
    direct: Path,
    port: int,
    *,
    execution_mode: str = "interpreter",
) -> list[str]:
    if execution_mode not in ("interpreter", "jit"):
        raise ValueError("execution_mode must be interpreter or jit")
    boot = ":".join(BOOT_JARS)
    classpath = ":".join((
        "/system/framework/android.test.base.jar",
        "/system/framework/android.test.mock.jar",
        str(direct / "lifecycle-probe.jar"),
        str(direct / "base.apk"),
    ))
    vm_mode = ["-Xint"] if execution_mode == "interpreter" else []
    return [
        "/apex/com.android.runtime/bin/linker64",
        "/apex/com.android.art/bin/dalvikvm64",
        *vm_mode,
        f"-Xbootclasspath:{boot}",
        f"-Xbootclasspath-locations:{boot}",
        f"-Djava.library.path={direct}",
        "-cp",
        classpath,
        "royale.nativehost.JniHost",
        str(direct),
        "serve-direct",
        str(port),
    ]


def environment(direct: Path) -> dict[str, str]:
    value = dict(os.environ)
    value.update({
        "ANDROID_ART_ROOT": "/apex/com.android.art",
        "ANDROID_DATA": "/data",
        "ANDROID_I18N_ROOT": "/apex/com.android.i18n",
        "ANDROID_ROOT": "/system",
        "ANDROID_RUNTIME_ROOT": "/apex/com.android.runtime",
        "ANDROID_TZDATA_ROOT": "/apex/com.android.tzdata",
        "CR_BINDERLESS_ANDROID": "1",
        "CR_BINDERLESS_NATIVE_CONFIG_GETTERS": "1",
        "CR_BINDERLESS_NATIVE_CONFIG_POSTPROCESS": "1",
        "CR_BINDERLESS_PRELOAD_CORE_DATA": "0",
        "CR_BINDERLESS_DEFER_NULL_DEPENDENCIES": "0",
        "CR_BINDERLESS_DEFER_OPTIONAL_CLIENT_GLOBALS": "0",
        "CR_NATIVE_LOADING_INITIAL_SETTLE_MS": "0",
        "CR_NATIVE_LOADING_MAX_FRAMES": "10000",
        "CR_NATIVE_LOADING_TIMEOUT_MS": "30000",
        "CR_NATIVE_LOADING_SLEEP_MS": "5",
        "HOME": "/root",
        "LD_LIBRARY_PATH": str(direct),
        "TMPDIR": str(direct / "cache"),
    })
    return value


def ensure_android_properties(runtime_root: Path) -> Path:
    """Restore Android's property area, which lives on a fresh /dev tmpfs."""
    target = Path("/dev/__properties__")
    serial = target / "properties_serial"
    if serial.is_file():
        return serial
    archive = runtime_root.parent / "android-properties.tar.gz"
    if not archive.is_file():
        raise RuntimeError(f"Android property archive is missing: {archive}")
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["tar", "-xzf", str(archive), "-C", str(target)],
        check=True,
        stdin=subprocess.DEVNULL,
    )
    if not serial.is_file():
        raise RuntimeError(f"Android property bootstrap failed: {serial}")
    return serial


def ensure(args: argparse.Namespace) -> dict[str, object]:
    if args.count < 1 or not 1 <= args.base_port <= 65536 - args.count:
        raise ValueError("Worker port range must fit in 1..65535")
    runtime_root = args.runtime_root.resolve(strict=True)
    property_serial = ensure_android_properties(runtime_root)
    working_directory = runtime_root / "worker0" / "assets"
    if not working_directory.is_dir():
        raise RuntimeError(f"Bionic Worker asset directory is missing: {working_directory}")
    log_root = runtime_root / "worker-logs"
    log_root.mkdir(exist_ok=True)
    required_system = [
        Path("/apex/com.android.runtime/bin/linker64"),
        Path("/apex/com.android.art/bin/dalvikvm64"),
        *(Path(value) for value in BOOT_JARS),
    ]
    missing_system = [str(path) for path in required_system if not path.exists()]
    if missing_system:
        raise RuntimeError(f"Bionic system image is incomplete: {missing_system}")

    slot_offset = int(getattr(args, "slot_offset", 0))
    execution_mode = str(getattr(args, "execution_mode", "interpreter"))
    if slot_offset < 0 or execution_mode not in ("interpreter", "jit"):
        raise ValueError("invalid Worker slot offset/execution mode")
    launched = []
    for local_slot in range(args.count):
        slot = slot_offset + local_slot
        port = args.base_port + local_slot
        if ready(port):
            continue
        direct = Path(f"/data/local/tmp/cr-native-direct-{slot}")
        required = (
            direct / "lifecycle-probe.jar",
            direct / "base.apk",
            direct / "libnative_host_bridge.so",
            direct / "libg.so",
            direct / "assets",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError(f"Worker {slot} runtime is incomplete: {missing}")
        if not wait_port_bindable(port, args.ready_timeout):
            raise RuntimeError(
                f"Worker port {port} did not become bindable after a prior exit"
            )
        (direct / "cache").mkdir(parents=True, exist_ok=True)
        log_path = log_root / f"worker-{slot:02d}-port-{port}.log"
        log = log_path.open("ab", buffering=0)
        process = subprocess.Popen(
            command(direct, port, execution_mode=execution_mode),
            cwd=working_directory,
            env=environment(direct),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        launched.append({"slot": slot, "port": port, "pid": process.pid})

    deadline = time.monotonic() + args.ready_timeout
    pending = list(range(args.base_port, args.base_port + args.count))
    while pending and time.monotonic() < deadline:
        pending = [port for port in pending if not ready(port)]
        if pending:
            time.sleep(0.25)
    if not pending:
        # A bare TCP accept is insufficient: a stale process can disappear
        # immediately after the first probe.  Require two semantic status
        # passes before advertising the fleet to collectors.
        for _ in range(2):
            time.sleep(0.5)
            pending = [
                port for port in range(args.base_port, args.base_port + args.count)
                if not ready(port)
            ]
            if pending:
                break
    if pending:
        tails = {}
        for port in pending:
            slot = slot_offset + port - args.base_port
            path = log_root / f"worker-{slot:02d}-port-{port}.log"
            if path.is_file():
                tails[str(port)] = path.read_text(
                    encoding="utf-8", errors="replace"
                )[-4000:]
        raise RuntimeError(f"Workers did not become ready: {pending}; tails={tails}")
    return {
        "ready": args.count,
        "base_port": args.base_port,
        "slot_offset": slot_offset,
        "execution_mode": execution_mode,
        "launched": launched,
        "reused": args.count - len(launched),
        "runtime_root": str(runtime_root),
        "property_serial": str(property_serial),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path("/root/autodl-tmp/expert-selfplay-v1/bionic-runtime"),
    )
    parser.add_argument("--base-port", type=int, default=39031)
    parser.add_argument("--count", type=int, default=72)
    parser.add_argument("--slot-offset", type=int, default=0)
    parser.add_argument(
        "--execution-mode",
        choices=("interpreter", "jit"),
        default="interpreter",
    )
    parser.add_argument("--ready-timeout", type=float, default=120.0)
    args = parser.parse_args()
    if args.count < 1 or args.ready_timeout <= 0:
        raise ValueError("Worker count and timeout must be positive")
    print(ensure(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
