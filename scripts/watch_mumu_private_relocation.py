"""Wait for the next live battle and capture the relocated player path."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time


CLI = Path(r"C:\Program Files\Netease\MuMu\nx_main\mumu-cli.exe")
OUTPUT = Path(
    r"D:\AI_data\cr-native-core\runtime-160402002-arm64"
    r"\cycle-relocation-live.json"
)


def run(*args: str) -> str:
    result = subprocess.run(
        [str(CLI), *args], capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=10,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    return result.stdout.strip()


def main() -> int:
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        pid = run("sh", "-v", "1", "-c", "pidof com.supercell.clashroyale")
        if pid.isdigit():
            raw = run(
                "sh", "-v", "1", "-c",
                f"/data/local/tmp/mumu-cycle-vector-scan {pid}",
            )
            try:
                payload = json.loads(raw)
            except ValueError:
                payload = None
            if isinstance(payload, dict) and payload.get("pairs"):
                OUTPUT.parent.mkdir(parents=True, exist_ok=True)
                temporary = OUTPUT.with_suffix(".json.tmp")
                temporary.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                temporary.replace(OUTPUT)
                print(json.dumps(payload, ensure_ascii=False), flush=True)
                return 0
        time.sleep(0.35)
    print("timeout waiting for live player relocation", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
