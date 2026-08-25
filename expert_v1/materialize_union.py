"""Materialize a union manifest as one deduplicated battle directory."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
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
    union_path = args.union_manifest.resolve(strict=True)
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    building = output.with_name(output.name + ".building")
    if building.exists():
        raise FileExistsError(f"staging output already exists: {building}")

    records: list[dict[str, Any]] = []
    tags: set[str] = set()
    for line_number, raw in enumerate(union_path.open("rb"), 1):
        if not raw.strip():
            continue
        record = loads(raw)
        tag = str(record.get("battle_tag") or "").strip()
        source = Path(str(record.get("source_path") or "")).resolve(strict=True)
        if not tag or source.suffix.lower() != ".json":
            raise ValueError(f"invalid union row {line_number}")
        if tag in tags:
            raise ValueError(f"duplicate union battle tag: {tag}")
        source_value = loads(source.read_bytes())
        if str(source_value.get("battle_tag") or "") != tag:
            raise ValueError(f"source battle tag mismatch: {source}")
        tags.add(tag)
        records.append({**record, "source": source})
    if len(records) != args.expected_count:
        raise RuntimeError(
            f"expected {args.expected_count} union records, got {len(records)}"
        )

    linked = 0
    copied = 0
    schema_counts: Counter[int] = Counter()
    try:
        battle_root = building / "raw" / "battles"
        battle_root.mkdir(parents=True)
        index_path = building / "index.jsonl"
        with index_path.open("wb") as index:
            for record in records:
                tag = str(record["battle_tag"])
                source = record["source"]
                destination = battle_root / tag[:2].lower() / f"{tag}.json"
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source, destination)
                    linked += 1
                except OSError:
                    shutil.copy2(source, destination)
                    copied += 1
                schema_version = int(record.get("schema_version") or 1)
                schema_counts[schema_version] += 1
                row = {
                    "battle_tag": tag,
                    "kind": "battle",
                    "saved_path": str(destination.relative_to(building)),
                    "source_path": str(source),
                    "source_batch": record.get("batch"),
                    "schema_version": schema_version,
                }
                encoded = (
                    orjson.dumps(row) if orjson is not None
                    else json.dumps(row, ensure_ascii=False).encode("utf-8")
                )
                index.write(encoded + b"\n")
        summary = {
            "schema_version": 1,
            "kind": "expert_materialized_union_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "union_manifest": str(union_path),
            "battle_count": len(records),
            "hardlinked_files": linked,
            "copied_files": copied,
            "selected_schema_versions": dict(sorted(schema_counts.items())),
        }
        (building / "SOURCE_MANIFEST.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(building, output)
        return summary
    except Exception:
        if building.exists():
            shutil.rmtree(building)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--union-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    result = run(parser.parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
