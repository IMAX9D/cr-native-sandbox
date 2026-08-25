"""Split an expert union manifest by a balance-patch timestamp window."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
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
    encoded = (
        orjson.dumps(value) if orjson is not None
        else json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    return encoded + b"\n"


def timestamp_seconds(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result // 1000 if result > 10_000_000_000 else result


def load_index(path: Path) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    with path.open("rb") as handle:
        for raw in handle:
            item = loads(raw)
            if item.get("kind") != "battle":
                continue
            url = str(item.get("url") or "")
            tag = (parse_qs(urlparse(url).query).get("tag") or [""])[0]
            if not tag:
                tag = str(item.get("battle_tag") or "")
            match = BEFORE_RE.search(url)
            cursor = timestamp_seconds(match.group(1)) if match else None
            if tag:
                result[tag] = cursor
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    union = args.union_manifest.resolve(strict=True)
    indexes: dict[str, dict[str, int | None]] = {}
    for binding in args.batch_index:
        batch, separator, raw_path = binding.partition("=")
        if not separator or not batch or not raw_path:
            raise ValueError("--batch-index must be BATCH=PATH")
        indexes[batch] = load_index(Path(raw_path).resolve(strict=True))
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    accepted_path = output / "accepted.jsonl"
    rejected_path = output / "rejected-old.jsonl"
    counters: Counter[str] = Counter()
    with (
        union.open("rb") as source,
        accepted_path.open("wb") as accepted,
        rejected_path.open("wb") as rejected,
    ):
        for raw in source:
            record = loads(raw)
            tag = str(record.get("battle_tag") or "")
            batch = str(record.get("batch") or "")
            source_value = loads(Path(str(record["source_path"])).read_bytes())
            exact = timestamp_seconds(source_value.get("timestamp"))
            cursor = indexes.get(batch, {}).get(tag)
            if exact is not None:
                timestamp = exact
                quality = "exact"
            elif cursor is not None:
                timestamp = cursor
                quality = "list_cursor_upper_bound"
            else:
                timestamp = None
                quality = "recent_first_page"
            enriched = {
                **record,
                "version_timestamp": timestamp,
                "version_timestamp_quality": quality,
                "min_timestamp": args.min_timestamp,
            }
            if timestamp is not None and timestamp < args.min_timestamp:
                rejected.write(dump_line(enriched))
                counters["rejected_old"] += 1
                counters[f"rejected_schema_{record.get('schema_version', 1)}"] += 1
            else:
                accepted.write(dump_line(enriched))
                counters["accepted"] += 1
                counters[f"accepted_schema_{record.get('schema_version', 1)}"] += 1
                counters[f"accepted_quality_{quality}"] += 1
    summary = {
        "schema_version": 1,
        "kind": "expert_version_window_summary_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "union_manifest": str(union),
        "min_timestamp": args.min_timestamp,
        "min_timestamp_utc": datetime.fromtimestamp(
            args.min_timestamp, tz=timezone.utc
        ).isoformat(),
        **counters,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--union-manifest", type=Path, required=True)
    parser.add_argument("--batch-index", action="append", required=True)
    parser.add_argument("--min-timestamp", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    result = run(parser.parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
