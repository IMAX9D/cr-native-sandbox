"""Foreground, read-only observer for a single explicitly selected expert run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time


def identity(pid: int) -> tuple[str, str] | None:
    try:
        values = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return values[0], values[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    launch = json.loads((root / "launch.json").read_text())
    pid = int(launch["trainer_pid"])
    first = identity(pid)
    start = int(launch["resume_step"])
    target = int(launch.get("target_step", 0))
    last = None
    last_change = time.monotonic()
    previous_time = None
    previous_step = None
    print(json.dumps({"event": "foreground_monitor_started", "run": root.name, "pid": pid,
                      "start_step": start, "target_step": target, "scheduled_task": False}), flush=True)
    while True:
        progress = json.loads((root / "training-progress.json").read_text())
        now = identity(pid)
        alive = now is not None and now[0] not in ("Z", "X") and first is not None and now[1] == first[1]
        signature = (progress.get("global_step"), progress.get("status"), progress.get("updated_utc"))
        if signature != last:
            last = signature
            last_change = time.monotonic()
            step = int(progress.get("global_step", 0))
            stamp = datetime.fromisoformat(progress["updated_utc"])
            rate = None
            if previous_time is not None and previous_step is not None and step > previous_step:
                elapsed = (stamp - previous_time).total_seconds()
                rate = (step - previous_step) / elapsed if elapsed > 0 else None
            previous_time, previous_step = stamp, step
            warnings = ["nonfinite:" + key for key, value in progress.items()
                        if isinstance(value, float) and not math.isfinite(value)]
            if progress.get("loss_window_gt10", 0):
                warnings.append("loss_over_10")
            if progress.get("loss_window_gt20", 0):
                warnings.append("loss_over_20")
            if progress.get("learning_rate", launch["learning_rate"]) != launch["learning_rate"]:
                warnings.append("learning_rate_changed")
            events = dict(line.split() for line in Path("/sys/fs/cgroup/memory.events").read_text().splitlines())
            if int(events.get("oom", 0)) or int(events.get("oom_kill", 0)):
                warnings.append("oom_event")
            print(json.dumps({"event": "progress", "utc": datetime.now(timezone.utc).isoformat(),
                "step": step, "added_steps": step - start, "target": target,
                "continuation_percent": (step - start) * 100 / (target - start) if target > start else None,
                "status": progress.get("status"), "window_n": progress.get("window_batches"),
                "loss_mean": progress.get("loss_window_mean"), "loss_last": progress.get("loss"),
                "loss_max": progress.get("loss_window_max"), "grad_mean": progress.get("gradient_norm_window_mean"),
                "grad_max": progress.get("gradient_norm_window_max"), "gt10": progress.get("loss_window_gt10", 0),
                "gt20": progress.get("loss_window_gt20", 0), "rate": rate, "warnings": warnings}), flush=True)
        elif time.monotonic() - last_change > 180:
            print(json.dumps({"event": "no_progress_180s", "status": progress.get("status"),
                              "step": progress.get("global_step")}), flush=True)
            last_change = time.monotonic()
        if not alive:
            expected = progress.get("status") == "paused" and (not target or progress.get("global_step", 0) >= target)
            print(json.dumps({"event": "target_paused" if expected else "trainer_exited",
                              "progress": progress}), flush=True)
            return 0 if expected else 1
        time.sleep(10)


if __name__ == "__main__":
    raise SystemExit(main())
