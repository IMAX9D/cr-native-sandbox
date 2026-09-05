"""Managed NVIDIA MPS lifecycle for multi-process Actor inference."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any


class ManagedMPS:
    """Start one private MPS control daemon and always stop it on exit."""

    def __init__(self, *, enabled: bool, root: Path | str) -> None:
        self.enabled = bool(enabled)
        self.root = Path(root).resolve()
        self.pipe_directory = self.root / "pipe"
        self.log_directory = self.root / "log"
        self.control = shutil.which("nvidia-cuda-mps-control")
        self.started = False
        self.environment = os.environ.copy()

    def start(self) -> dict[str, str]:
        if not self.enabled:
            return dict(self.environment)
        if self.control is None:
            raise RuntimeError("nvidia-cuda-mps-control is unavailable")
        self.pipe_directory.mkdir(parents=True, exist_ok=True)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        self.environment.update({
            "CUDA_MPS_PIPE_DIRECTORY": str(self.pipe_directory),
            "CUDA_MPS_LOG_DIRECTORY": str(self.log_directory),
        })
        started = subprocess.run(
            [self.control, "-d"],
            env=self.environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=20,
        )
        if started.returncode != 0:
            raise RuntimeError(
                "NVIDIA MPS failed to start: " + started.stdout.strip()
            )
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if any(self.pipe_directory.iterdir()):
                self.started = True
                return dict(self.environment)
            time.sleep(0.05)
        self.stop()
        raise RuntimeError("NVIDIA MPS control pipe did not become ready")

    def stop(self) -> dict[str, Any]:
        if not self.started or self.control is None:
            return {"requested": False, "returncode": None, "output": ""}
        stopped = subprocess.run(
            [self.control],
            input="quit\n",
            env=self.environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=20,
        )
        self.started = False
        return {
            "requested": True,
            "returncode": int(stopped.returncode),
            "output": stopped.stdout.strip(),
        }

    def __enter__(self) -> dict[str, str]:
        return self.start()

    def __exit__(self, *_error: object) -> None:
        self.stop()


__all__ = ["ManagedMPS"]
