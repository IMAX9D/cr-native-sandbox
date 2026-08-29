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
