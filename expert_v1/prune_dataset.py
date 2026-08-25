"""Safely remove dataset battles listed by an expert audit rejection manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import tempfile
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


def normalized_saved_path(value: str) -> str:
    return str(PureWindowsPath(value)).lower()


def collect_targets(
    rejection_manifest: Path, battle_root: Path, dataset_root: Path
) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    seen: set[Path] = set()
    resolved_battle_root = battle_root.resolve(strict=True)
    for line_number, raw in enumerate(rejection_manifest.open("rb"), 1):
        if not raw.strip():
            continue
        record = loads(raw)
        relative = Path(str(record.get("path") or ""))
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe path at manifest line {line_number}: {relative}")
        target = (resolved_battle_root / relative).resolve(strict=True)
        if not target.is_relative_to(resolved_battle_root):
            raise ValueError(f"target escapes battle root: {target}")
        if target.suffix.lower() != ".json" or not target.is_file():
            raise ValueError(f"target is not a battle JSON file: {target}")
        if target in seen:
            raise ValueError(f"duplicate target: {target}")
        seen.add(target)
        targets.append({
            "target": target,
            "dataset_path": str(target.relative_to(dataset_root.resolve(strict=True))),
            "audit_path": str(relative),
            "battle_tag": record.get("battle_tag"),
            "reasons": record.get("reasons", []),
        })
    return targets


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = args.dataset_root.resolve(strict=True)
    battle_root = (dataset_root / "data" / "raw" / "battles").resolve(strict=True)
    index_path = (dataset_root / "data" / "index.jsonl").resolve(strict=True)
    rejection_manifest = args.rejection_manifest.resolve(strict=True)
    targets = collect_targets(rejection_manifest, battle_root, dataset_root)
    if len(targets) != args.expected_count:
        raise RuntimeError(
            f"refusing prune: expected {args.expected_count} targets, got {len(targets)}"
        )

    delete_paths = {
        normalized_saved_path(item["dataset_path"])
        for item in targets
    }
    index_before_sha256 = sha256(index_path)
    removed_index_rows = 0
    retained_index_rows = 0
    fd, temp_name = tempfile.mkstemp(
        prefix="index-prune-", suffix=".jsonl", dir=index_path.parent
    )
    os.close(fd)
    temp_index = Path(temp_name)
    try:
        with index_path.open("rb") as source, temp_index.open("wb") as destination:
            for raw in source:
                remove = False
                try:
                    item = loads(raw)
                    saved_path = str(item.get("saved_path") or "")
                    remove = (
                        item.get("kind") == "battle"
                        and normalized_saved_path(saved_path) in delete_paths
                    )
                except Exception:
                    pass
                if remove:
                    removed_index_rows += 1
                else:
                    destination.write(raw)
                    retained_index_rows += 1

        if removed_index_rows != len(targets):
            raise RuntimeError(
                "refusing prune: index rows do not match targets "
                f"({removed_index_rows} != {len(targets)})"
            )

        summary = {
            "schema_version": 1,
            "kind": "expert_dataset_prune_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_root": str(dataset_root),
            "rejection_manifest": str(rejection_manifest),
            "apply": bool(args.apply),
            "target_files": len(targets),
            "target_bytes": sum(item["target"].stat().st_size for item in targets),
            "removed_index_rows": removed_index_rows,
            "retained_index_rows": retained_index_rows,
            "index_before_sha256": index_before_sha256,
        }
        if not args.apply:
            return summary

        args.audit_root.mkdir(parents=True, exist_ok=True)
        detail_path = args.audit_root / "deleted-files.jsonl"
        with detail_path.open("wb") as detail:
            for item in targets:
                record = {
                    "dataset_path": item["dataset_path"],
                    "battle_tag": item["battle_tag"],
                    "reasons": item["reasons"],
                    "bytes": item["target"].stat().st_size,
                    "sha256": sha256(item["target"]),
                }
                encoded = (
                    orjson.dumps(record) if orjson is not None
                    else json.dumps(record, ensure_ascii=False).encode("utf-8")
                )
                detail.write(encoded + b"\n")

        for item in targets:
            item["target"].unlink()
        os.replace(temp_index, index_path)
        summary["index_after_sha256"] = sha256(index_path)
        summary["deleted_files_manifest"] = str(detail_path)
        summary_path = args.audit_root / "summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return summary
    finally:
        temp_index.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path(r"D:\皇室战争数据集"))
    parser.add_argument(
        "--rejection-manifest", type=Path,
        default=Path(
            r"D:\AI_data\cr-native-core\expert-v1\dataset-audit\rejected.jsonl"
        ),
    )
    parser.add_argument(
        "--audit-root", type=Path,
        default=Path(r"D:\AI_data\cr-native-core\expert-v1\dataset-prune"),
    )
    parser.add_argument("--expected-count", type=int, default=4359)
    parser.add_argument("--apply", action="store_true")
    return parser


def main() -> int:
    summary = run(build_parser().parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
