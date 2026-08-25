"""Join player identities and terminal crowns from crawler source indexes."""

from __future__ import annotations

import argparse
import hashlib
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


def dump_line(value: Any) -> bytes:
    encoded = orjson.dumps(value) if orjson is not None else json.dumps(
        value, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return encoded + b"\n"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_index(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open("rb") as handle:
        for raw in handle:
            item = loads(raw)
            if item.get("kind") != "battle":
                continue
            query = parse_qs(urlparse(str(item.get("url") or "")).query)
            tag = (query.get("tag") or [""])[0]
            if not tag:
                continue
            def integer(name: str) -> int | None:
                try:
                    return int((query.get(name) or [None])[0])
                except (TypeError, ValueError):
                    return None
            result[tag] = {
                "team_tags": [x for x in (query.get("team_tags") or [""])[0].split(",") if x],
                "opponent_tags": [x for x in (query.get("opponent_tags") or [""])[0].split(",") if x],
                "team_crowns": integer("team_crowns"),
                "opponent_crowns": integer("opponent_crowns"),
            }
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = args.manifest.resolve(strict=True)
    indexes: dict[str, dict[str, dict[str, Any]]] = {}
    for binding in args.batch_index:
        batch, separator, raw_path = binding.partition("=")
        if not separator:
            raise ValueError("--batch-index must be BATCH=PATH")
        indexes[batch] = load_index(Path(raw_path).resolve(strict=True))
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = missing = 0
    with manifest.open("rb") as source, output.open("wb") as target:
        for raw in source:
            item = loads(raw)
            joined = indexes.get(str(item.get("batch")), {}).get(str(item.get("battle_tag")))
            if joined is None:
                missing += 1
                continue
            team = joined["team_crowns"]
            opponent = joined["opponent_crowns"]
            if team is None or opponent is None:
                missing += 1
                continue
            outcome = 1 if team > opponent else -1 if team < opponent else 0
            target.write(dump_line({
                **item,
                **joined,
                "team_outcome": outcome,
                "terminal_provenance": "crawler_index_query",
            }))
            rows += 1
    if missing:
        raise RuntimeError(f"missing terminal metadata for {missing} records")
    return {
        "records": rows,
        "missing": missing,
        "output": str(output),
        "output_sha256": sha256(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch-index", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
