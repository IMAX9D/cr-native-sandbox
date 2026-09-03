"""Mirror the lightweight expert-training receipts into TensorBoard events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Mapping

from torch.utils.tensorboard import SummaryWriter


def add_numeric_group(
    writer: SummaryWriter,
    prefix: str,
    values: Mapping[str, Any],
    step: int,
) -> None:
    for name, value in values.items():
        if isinstance(value, (int, float)):
            writer.add_scalar(f"{prefix}/{name}", float(value), step)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--cache-status", type=Path)
    args = parser.parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.log_dir), flush_secs=2)
    last_progress_step = -1
    events_offset = 0
    try:
        while True:
            if args.progress.is_file():
                try:
                    progress = json.loads(args.progress.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    progress = {}
                step = int(progress.get("global_step", -1))
                if step >= 0 and step != last_progress_step:
                    if isinstance(progress.get("loss"), (int, float)):
                        writer.add_scalar("train/loss_live", float(progress["loss"]), step)
                    for source, tag in {
                        "loss_window_mean": "train/loss",
                        "loss_window_max": "train/loss_window_max",
                        "loss_position_window_mean": "train/loss_position_window_mean",
                        "loss_card_window_mean": "train/loss_card_window_mean",
                        "gradient_norm_window_mean": "train/gradient_norm_window_mean",
                        "gradient_norm_window_max": "train/gradient_norm_window_max",
                        "loss_window_gt10": "train/loss_window_gt10",
                        "loss_window_gt20": "train/loss_window_gt20",
                        "window_batches": "train/window_batches",
                        "learning_rate": "train/learning_rate",
                        "position_logit_absmax": "train/position_logit_absmax",
                    }.items():
                        if isinstance(progress.get(source), (int, float)):
                            writer.add_scalar(tag, float(progress[source]), step)
                    epoch = int(progress.get("epoch", 0))
                    batch = int(progress.get("batch", 0))
                    batches = int(progress.get("batches", 0))
                    epochs = int(progress.get("epochs", 0))
                    if batches > 0 and epochs > 0:
                        fraction = ((max(epoch, 1) - 1) + batch / batches) / epochs
                        writer.add_scalar("progress/percent", fraction * 100.0, step)
                    writer.add_scalar("progress/epoch", epoch, step)
                    writer.add_scalar("progress/batch", batch, step)
                    writer.add_text("run/status", str(progress.get("status", "")), step)
                    if args.cache_status is not None:
                        try:
                            cache = json.loads(args.cache_status.read_text())
                            pid = int(cache["pid"])
                            process = Path(f"/proc/{pid}")
                            locked_gib = 0.0
                            if (process / "status").is_file():
                                for line in (process / "status").read_text().splitlines():
                                    if line.startswith("VmLck:"):
                                        locked_gib = int(line.split()[1]) / (1024 ** 2)
                            writer.add_scalar("cache/locked_GiB", locked_gib, step)
                            writer.add_scalar("cache/budget_GiB", float(cache["budget_gib"]), step)
                            writer.add_scalar("cache/reserve_GiB", float(cache["reserve_gib"]), step)
                            used = int(Path("/sys/fs/cgroup/memory.current").read_text())
                            writer.add_scalar("cache/container_used_GiB", used / (1024 ** 3), step)
                        except (OSError, ValueError, KeyError):
                            pass
                    last_progress_step = step
                    writer.flush()
            if args.events.is_file():
                try:
                    with args.events.open("r", encoding="utf-8") as handle:
                        handle.seek(events_offset)
                        for line in handle:
                            try:
                                event = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            step = int(event.get("global_step", 0))
                            if event.get("event") == "checkpoint_validation_complete" and event.get("full_validation") is True:
                                validation = event.get("validation") or {}
                                add_numeric_group(writer, "checkpoint/validation", validation, step)
                                if isinstance(validation.get("loss"), (int, float)):
                                    writer.add_scalar("val/loss", float(validation["loss"]), step)
                            if event.get("event") == "epoch_complete":
                                add_numeric_group(
                                    writer,
                                    "epoch/train",
                                    event.get("training") or {},
                                    step,
                                )
                                add_numeric_group(
                                    writer,
                                    "epoch/validation",
                                    event.get("validation") or {},
                                    step,
                                )
                                validation_loss = (event.get("validation") or {}).get("loss")
                                if isinstance(validation_loss, (int, float)):
                                    # Discoverable alias for COMPLETE epoch validation.
                                    # Small-subset diagnostics must use a separate tag.
                                    writer.add_scalar("val/loss", float(validation_loss), step)
                                writer.add_scalar(
                                    "epoch/wall_seconds",
                                    float(event.get("wall_seconds", 0.0)),
                                    step,
                                )
                        events_offset = handle.tell()
                    writer.flush()
                except OSError:
                    pass
            time.sleep(max(0.5, args.interval))
    except KeyboardInterrupt:
        return 0
    finally:
        writer.close()


if __name__ == "__main__":
    raise SystemExit(main())
