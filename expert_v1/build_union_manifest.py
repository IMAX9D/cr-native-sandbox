"""Build a battle-tag-deduplicated virtual dataset across source batches."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

try:
    import orjson
except ImportError:  # pragma: no cover
    orjson = None


def loads(raw: bytes) -> dict[str, Any]:
    value = orjson.loads(raw) if orjson is not None else json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError("JSON root is not an object")
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    selected: dict[str, dict[str, Any]] = {}
    input_counts: Counter[str] = Counter()
    duplicate_tags: set[str] = set()
    upgrades = 0
    for priority, batch_arg in enumerate(args.batch):
        batch = batch_arg.resolve(strict=True)
        battle_root = (batch / "raw" / "battles").resolve(strict=True)
        for path in battle_root.rglob("*.json"):
            value = loads(path.read_bytes())
            tag = str(value.get("battle_tag") or "").strip()
            if not tag:
                raise ValueError(f"missing battle_tag: {path}")
            schema_version = int(value.get("schema_version") or 1)
            input_counts[batch.name] += 1
            candidate = {
                "battle_tag": tag,
                "source_path": str(path),
                "batch": batch.name,
                "schema_version": schema_version,
                "priority": priority,
            }
            current = selected.get(tag)
            if current is None:
                selected[tag] = candidate
                continue
            duplicate_tags.add(tag)
            if (schema_version, priority) > (
                int(current["schema_version"]), int(current["priority"])
            ):
                if schema_version > int(current["schema_version"]):
                    upgrades += 1
                selected[tag] = candidate

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        for tag in sorted(selected):
            record = dict(selected[tag])
            record.pop("priority", None)
            encoded = (
                orjson.dumps(record) if orjson is not None
                else json.dumps(record, ensure_ascii=False).encode("utf-8")
            )
            handle.write(encoded + b"\n")
    schema_counts = Counter(
        int(record["schema_version"]) for record in selected.values()
    )
    summary = {
        "schema_version": 1,
        "kind": "expert_union_manifest_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output": str(output),
        "input_batches": dict(input_counts),
        "input_battles": sum(input_counts.values()),
        "duplicate_battle_tags": len(duplicate_tags),
        "metadata_upgrades_selected": upgrades,
        "unique_battles": len(selected),
        "remaining_to_target": max(0, args.target_count - len(selected)),
        "selected_schema_versions": dict(sorted(schema_counts.items())),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=100_000)
    return parser


def main() -> int:
    summary = run(build_parser().parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
