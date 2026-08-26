"""Freeze a target-sized expert corpus from a clean base plus live accepts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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


def dump(value: Any) -> bytes:
    encoded = orjson.dumps(value) if orjson is not None else json.dumps(
        value, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return encoded + b"\n"


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as handle:
        return [loads(raw) for raw in handle if raw.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    base_path = args.base_manifest.resolve(strict=True)
    live_path = args.live_manifest.resolve(strict=True)
    base = read_rows(base_path)
    live = read_rows(live_path)
    base_tags = {str(row.get("battle_tag") or "") for row in base}
    if "" in base_tags or len(base_tags) != len(base):
        raise RuntimeError("base manifest is not unique and complete")
    unique_live: list[dict[str, Any]] = []
    seen = set(base_tags)
    for row in live:
        tag = str(row.get("battle_tag") or "")
        if not tag or tag in seen:
            continue
        seen.add(tag)
        unique_live.append(row)
    required = args.target - len(base)
    if required < 0:
        raise RuntimeError("base manifest already exceeds target")
    if len(unique_live) < required:
        raise RuntimeError(
            f"target not reached: need {required} live rows, have {len(unique_live)}"
        )
    selected = unique_live[:required]
    overflow = unique_live[required:]

    batch = args.output_batch.resolve()
    if batch.exists():
        raise FileExistsError(batch)
    building = batch.with_name(batch.name + ".building")
    if building.exists():
        raise FileExistsError(building)
    final_path = args.output_manifest.resolve()
    final_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        battle_root = building / "raw" / "battles"
        battle_root.mkdir(parents=True)
        index_path = building / "index.jsonl"
        frozen_live: list[dict[str, Any]] = []
        linked = copied = 0
        with index_path.open("wb") as index:
            for row in selected:
                tag = str(row["battle_tag"])
                source = Path(str(row["source_path"])).resolve(strict=True)
                value = loads(source.read_bytes())
                if str(value.get("battle_tag") or "") != tag:
                    raise RuntimeError(f"source tag mismatch: {source}")
                destination = battle_root / tag[:2].lower() / f"{tag}.json"
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source, destination)
                    linked += 1
                except OSError:
                    shutil.copy2(source, destination)
                    copied += 1
                frozen = {
                    **row,
                    # The staging directory is atomically renamed below. Persist
                    # the final batch path, never the transient `.building` path.
                    "source_path": str(
                        (batch / destination.relative_to(building)).resolve()
                    ),
                    "batch": batch.name,
                }
                frozen_live.append(frozen)
                index.write(dump({
                    "kind": "battle", "battle_tag": tag,
                    "saved_path": str(destination.relative_to(building)),
                    "source_download_path": str(source),
                    "schema_version": int(value.get("schema_version") or 0),
                }))
        batch_manifest = {
            "schema_version": 1,
            "kind": "expert_source_batch_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "battle_count": len(frozen_live),
            "hardlinked_files": linked,
            "copied_files": copied,
            "source_live_manifest": str(live_path),
        }
        (building / "SOURCE_MANIFEST.json").write_text(
            json.dumps(batch_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(building, batch)
        temp_final = final_path.with_suffix(final_path.suffix + ".tmp")
        with temp_final.open("wb") as handle:
            for row in base:
                handle.write(dump(row))
            for row in frozen_live:
                handle.write(dump(row))
        temp_final.replace(final_path)
        overflow_path = final_path.with_name(final_path.stem + "-overflow.jsonl")
        with overflow_path.open("wb") as handle:
            for row in overflow:
                handle.write(dump(row))
        result = {
            "schema_version": 1,
            "kind": "expert_corpus_finalization_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "base_count": len(base),
            "selected_live_count": len(frozen_live),
            "overflow_count": len(overflow),
            "final_count": len(base) + len(frozen_live),
            "target": args.target,
            "final_manifest": str(final_path),
            "final_manifest_sha256": sha256(final_path),
            "output_batch": str(batch),
        }
        final_path.with_suffix(".summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return result
    except Exception:
        if building.exists():
            shutil.rmtree(building)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--live-manifest", type=Path, required=True)
    parser.add_argument("--output-batch", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--target", type=int, default=100_000)
    result = run(parser.parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
