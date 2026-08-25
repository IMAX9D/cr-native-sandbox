"""Require both player-side eight-card cycles to be valid for training."""

from __future__ import annotations

import argparse
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


def dump(value: Any) -> bytes:
    encoded = orjson.dumps(value) if orjson is not None else json.dumps(
        value, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return encoded + b"\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    invalid: set[str] = set()
    for path in args.invalid_sides:
        with path.resolve(strict=True).open("rb") as handle:
            for raw in handle:
                invalid.add(str(loads(raw).get("battle_tag") or ""))
    source = args.manifest.resolve(strict=True)
    accepted_path = args.output.resolve()
    rejected_path = accepted_path.with_name(accepted_path.stem + "-rejected.jsonl")
    accepted_path.parent.mkdir(parents=True, exist_ok=True)
    accepted = rejected = 0
    with (
        source.open("rb") as handle,
        accepted_path.open("wb") as good,
        rejected_path.open("wb") as bad,
    ):
        for raw in handle:
            item = loads(raw)
            if str(item.get("battle_tag") or "") in invalid:
                bad.write(dump({**item, "cycle_rejection": "one_or_both_sides_invalid"}))
                rejected += 1
            else:
                good.write(raw if raw.endswith(b"\n") else raw + b"\n")
                accepted += 1
    return {
        "accepted": accepted,
        "rejected": rejected,
        "invalid_tags": len(invalid),
        "output": str(accepted_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--invalid-sides", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
