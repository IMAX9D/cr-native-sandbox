"""Build a prune manifest for downloaded battles older than a Unix cutoff."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path, PureWindowsPath
import re
from typing import Any
from urllib.parse import parse_qs, urlparse

try:
    import orjson
except ImportError:  # pragma: no cover
    orjson = None


BEFORE_RE = re.compile(r"[?&]before=(\d{10,13})")


def loads(raw: bytes) -> dict[str, Any]:
    value = orjson.loads(raw) if orjson is not None else json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError("JSON root is not an object")
    return value


def dump_line(value: Any) -> bytes:
    encoded = orjson.dumps(value) if orjson is not None else json.dumps(
        value, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return encoded + b"\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset = args.dataset_root.resolve(strict=True)
    index = (dataset / "data" / "index.jsonl").resolve(strict=True)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with index.open("rb") as source, output.open("wb") as target:
        for raw in source:
            item = loads(raw)
            if item.get("kind") != "battle":
                continue
            url = str(item.get("url") or "")
            match = BEFORE_RE.search(url)
            if not match:
                continue
            timestamp = int(match.group(1))
            if timestamp > 10_000_000_000:
                timestamp //= 1000
            if timestamp >= args.min_timestamp:
                continue
            saved = PureWindowsPath(str(item.get("saved_path") or ""))
            parts = [part.lower() for part in saved.parts]
            try:
                raw_index = parts.index("battles")
            except ValueError as error:
                raise ValueError(f"unexpected saved_path: {saved}") from error
            relative = str(PureWindowsPath(*saved.parts[raw_index + 1 :]))
            tag = (parse_qs(urlparse(url).query).get("tag") or [""])[0]
            target.write(dump_line({
                "schema_version": 1,
                "battle_tag": tag,
                "path": relative,
                "reasons": ["before_native_balance_window"],
                "version_timestamp": timestamp,
                "min_timestamp": args.min_timestamp,
            }))
            count += 1
    summary = {
        "schema_version": 1,
        "kind": "expert_age_prune_manifest_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset),
        "min_timestamp": args.min_timestamp,
        "records": count,
        "output": str(output),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--min-timestamp", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
