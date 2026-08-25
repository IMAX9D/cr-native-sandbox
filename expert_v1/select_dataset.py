"""Copy an audited battle selection into an immutable source batch."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized(value: str) -> str:
    return str(PureWindowsPath(value)).lower()


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = args.dataset_root.resolve(strict=True)
    battle_root = (dataset_root / "data" / "raw" / "battles").resolve(strict=True)
    index_path = (dataset_root / "data" / "index.jsonl").resolve(strict=True)
    selection_path = args.selection_manifest.resolve(strict=True)
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    building = output.with_name(output.name + ".building")
    if building.exists():
        raise FileExistsError(f"staging output already exists: {building}")

    selected: list[tuple[Path, str, dict[str, Any]]] = []
    seen_targets: set[Path] = set()
    saved_paths: set[str] = set()
    for line_number, raw in enumerate(selection_path.open("rb"), 1):
        if not raw.strip():
            continue
        record = loads(raw)
        relative = Path(str(record.get("path") or ""))
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe selection path at line {line_number}: {relative}")
        source = (battle_root / relative).resolve(strict=True)
        if not source.is_relative_to(battle_root) or source.suffix.lower() != ".json":
            raise ValueError(f"selection target escaped battle root: {source}")
        if source in seen_targets:
            raise ValueError(f"duplicate selection target: {source}")
        seen_targets.add(source)
        dataset_path = str(source.relative_to(dataset_root))
        saved_paths.add(normalized(dataset_path))
        selected.append((source, str(relative), record))
    if len(selected) != args.expected_count:
        raise RuntimeError(
            f"expected {args.expected_count} selected files, got {len(selected)}"
        )

    index_rows: list[bytes] = []
    with index_path.open("rb") as handle:
        for raw in handle:
            try:
                item = loads(raw)
                if (
                    item.get("kind") == "battle"
                    and normalized(str(item.get("saved_path") or "")) in saved_paths
                ):
                    index_rows.append(raw)
            except Exception:
                continue
    if len(index_rows) != len(selected):
        raise RuntimeError(
            f"index rows do not match selection ({len(index_rows)} != {len(selected)})"
        )

    try:
        (building / "raw" / "battles").mkdir(parents=True)
        total_bytes = 0
        for source, relative, _ in selected:
            destination = building / "raw" / "battles" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            total_bytes += destination.stat().st_size
        output_index = building / "index.jsonl"
        with output_index.open("wb") as handle:
            handle.writelines(index_rows)
        shutil.copy2(selection_path, building / "selection.jsonl")
        summary = {
            "schema_version": 1,
            "kind": "expert_source_batch_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_dataset": str(dataset_root),
            "selection_manifest": str(selection_path),
            "battle_count": len(selected),
            "battle_bytes": total_bytes,
            "index_sha256": sha256(output_index),
            "source_files_preserved": True,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    return parser


def main() -> int:
    summary = run(build_parser().parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
