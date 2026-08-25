"""Recover terminal crown anchors retained in crawler source indexes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

try:
    import orjson
except ImportError:  # pragma: no cover
    orjson = None


def loads(raw: bytes) -> dict[str, Any]:
    value = orjson.loads(raw) if orjson is not None else json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError("JSON line root is not an object")
    return value


def batch_root(source_path: Path) -> Path:
    """Return the batch containing ``raw/battles/<shard>/<file>.json``."""
    for parent in source_path.parents:
        if parent.name == "raw":
            return parent.parent
    raise ValueError(f"source path is not inside a raw directory: {source_path}")


def index_terminal_anchors(index_path: Path) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    with index_path.open("rb") as source:
        for raw in source:
            row = loads(raw)
            if row.get("kind") != "battle":
                continue
            query = parse_qs(urlsplit(str(row.get("url") or "")).query)
            tag = str((query.get("tag") or [""])[0])
            if not tag:
                continue
            try:
                crowns = (
                    int(query["team_crowns"][0]),
                    int(query["opponent_crowns"][0]),
                )
            except (KeyError, IndexError, TypeError, ValueError):
                continue
            if any(not 0 <= value <= 3 for value in crowns):
                continue
            previous = result.get(tag)
            if previous is not None and previous != crowns:
                raise ValueError(
                    f"conflicting terminal crowns for {tag}: {previous} vs {crowns}"
                )
            result[tag] = crowns
    return result


def _index_terminal_anchors_by_file(
    index_path: Path,
) -> dict[str, tuple[str, tuple[int, int]]]:
    result: dict[str, tuple[str, tuple[int, int]]] = {}
    with index_path.open("rb") as source:
        for raw in source:
            row = loads(raw)
            if row.get("kind") != "battle":
                continue
            saved_name = Path(str(row.get("saved_path") or "").replace("\\", "/")).name
            query = parse_qs(urlsplit(str(row.get("url") or "")).query)
            tag = str((query.get("tag") or [""])[0])
            if not saved_name or not tag:
                continue
            try:
                crowns = (
                    int(query["team_crowns"][0]),
                    int(query["opponent_crowns"][0]),
                )
            except (KeyError, IndexError, TypeError, ValueError):
                continue
            key = saved_name.lower()
            previous = result.get(key)
            value = (tag, crowns)
            if previous is not None and previous != value:
                raise ValueError(
                    f"conflicting source index entries for {saved_name}"
                )
            result[key] = value
    return result


def manifest_terminal_anchors(
    manifest_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, tuple[int, int]]]:
    """Join every accepted manifest record to its batch source index."""
    records: list[dict[str, Any]] = []
    roots: set[Path] = set()
    with manifest_path.open("rb") as source:
        for raw in source:
            row = loads(raw)
            path = Path(str(row.get("source_path") or ""))
            records.append(row)
            roots.add(batch_root(path))
    by_root: dict[Path, dict[str, tuple[str, tuple[int, int]]]] = {}
    for root in sorted(roots):
        index = root / "index.jsonl"
        if not index.is_file():
            raise FileNotFoundError(f"source batch index is missing: {index}")
        by_root[root] = _index_terminal_anchors_by_file(index)
    selected: dict[str, tuple[int, int]] = {}
    missing: list[str] = []
    for row in records:
        tag = str(row.get("battle_tag") or "")
        path = Path(str(row["source_path"]))
        root = batch_root(path)
        indexed = by_root[root].get(path.name.lower())
        if indexed is None or indexed[0] != tag:
            missing.append(tag)
            continue
        selected[tag] = indexed[1]
    missing.sort()
    if missing:
        raise ValueError(
            f"terminal crown anchors missing for {len(missing)} accepted battles; "
            f"first={missing[:5]}"
        )
    return records, selected
