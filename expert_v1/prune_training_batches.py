"""Permanently remove rejected battle tags from materialized source batches."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

try:
    import orjson
except ImportError:  # pragma: no cover
    orjson = None


def loads(raw: bytes) -> dict[str, Any]:
    value = orjson.loads(raw) if orjson is not None else json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError("JSON root is not an object")
    return value


def tag_from_index(item: dict[str, Any]) -> str:
    tag = str(item.get("battle_tag") or "")
    if tag:
        return tag
    return (parse_qs(urlparse(str(item.get("url") or "")).query).get("tag") or [""])[0]


def rewrite_jsonl(path: Path, rejected: set[str]) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    temp = path.with_suffix(path.suffix + ".pruning")
    kept = removed = 0
    with path.open("rb") as source, temp.open("wb") as target:
        for raw in source:
            try:
                item = loads(raw)
                tag = tag_from_index(item)
            except Exception:
                tag = ""
            if tag and tag in rejected:
                removed += 1
            else:
                target.write(raw)
                kept += 1
    temp.replace(path)
    return kept, removed


def run(args: argparse.Namespace) -> dict[str, Any]:
    training_root = args.training_root.resolve(strict=True)
    rejected: set[str] = set()
    with args.rejected_manifest.resolve(strict=True).open("rb") as handle:
        for raw in handle:
            tag = str(loads(raw).get("battle_tag") or "")
            if tag:
                rejected.add(tag)
    if len(rejected) != args.expected_tags:
        raise RuntimeError(
            f"expected {args.expected_tags} rejected tags, got {len(rejected)}"
        )

    results: dict[str, Any] = {}
    for raw_batch in args.batch:
        batch = raw_batch.resolve(strict=True)
        if not batch.is_relative_to(training_root):
            raise ValueError(f"batch escapes training root: {batch}")
        battle_root = (batch / "raw" / "battles").resolve(strict=True)
        targets: list[Path] = []
        for path in battle_root.rglob("*.json"):
            item = loads(path.read_bytes())
            if str(item.get("battle_tag") or "") in rejected:
                target = path.resolve(strict=True)
                if not target.is_relative_to(battle_root):
                    raise ValueError(f"target escapes battle root: {target}")
                targets.append(target)
        if args.apply:
            for target in targets:
                target.unlink()
            index_kept, index_removed = rewrite_jsonl(batch / "index.jsonl", rejected)
            selection_kept, selection_removed = rewrite_jsonl(
                batch / "selection.jsonl", rejected
            )
            manifest_path = batch / "SOURCE_MANIFEST.json"
            manifest = loads(manifest_path.read_bytes()) if manifest_path.exists() else {}
            manifest["active_battle_count"] = sum(
                1 for _ in battle_root.rglob("*.json")
            )
            manifest["version_prune"] = {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "removed_files": len(targets),
                "rejected_manifest": str(args.rejected_manifest.resolve(strict=True)),
            }
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        else:
            index_kept = index_removed = selection_kept = selection_removed = 0
        results[batch.name] = {
            "target_files": len(targets),
            "applied": bool(args.apply),
            "index_kept": index_kept,
            "index_removed": index_removed,
            "selection_kept": selection_kept,
            "selection_removed": selection_removed,
        }
    return {
        "schema_version": 1,
        "kind": "expert_training_batch_prune_v1",
        "rejected_tags": len(rejected),
        "batches": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--rejected-manifest", type=Path, required=True)
    parser.add_argument("--expected-tags", type=int, required=True)
    parser.add_argument("--batch", type=Path, action="append", required=True)
    parser.add_argument("--apply", action="store_true")
    result = run(parser.parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
