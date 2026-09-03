"""Ask a running expert trainer to save at the next safe optimizer boundary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import uuid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--at-step", type=int, default=0)
    parser.add_argument("--stop-after-save", action="store_true")
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--no-preserve", action="store_true")
    parser.add_argument("--replace-pending", action="store_true")
    args = parser.parse_args()
    if args.at_step < 0:
        raise ValueError("at-step must be nonnegative")
    root = args.run_root.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8-sig"))
    if manifest.get("run_id") != root.name or "optimizer" not in manifest:
        raise ValueError("not an authenticated expert run directory")
    control = root / "control"
    control.mkdir(exist_ok=True)
    target = control / "checkpoint-request.json"
    response = control / "checkpoint-response.json"
    if target.exists() and not args.replace_pending:
        previous = json.loads(target.read_text())
        ack = json.loads(response.read_text()) if response.exists() else {}
        if previous.get("request_id") != ack.get("request_id") or ack.get("status") != "saved":
            raise RuntimeError("a pending request already exists; use --replace-pending explicitly")
    value = {"request_id": uuid.uuid4().hex, "expected_run_id": manifest["run_id"], "at_step": args.at_step,
             "stop_after_save": args.stop_after_save, "preserve": not args.no_preserve,
             "export_fp16": not args.no_export, "reason": "user_requested_checkpoint"}
    temporary = target.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(json.dumps({"requested": value, "response_path": str(response)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
