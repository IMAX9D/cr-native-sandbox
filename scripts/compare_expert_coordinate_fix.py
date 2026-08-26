#!/usr/bin/env python3
"""Compare two fixed-selection pilots across a coordinate-ingestion change."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


FAILURE_CODES = re.compile(r"codes_\[([^]]+)\]")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _indexed(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in _jsonl(path):
        tag = str(row["battle_tag"])
        if tag in result:
            raise RuntimeError(f"duplicate battle tag: {tag}")
        result[tag] = row
    return result


def _codes(row: Mapping[str, Any]) -> list[int]:
    rejection = row.get("first_rejection")
    if isinstance(rejection, Mapping) and isinstance(
        rejection.get("result_codes"), list
    ):
        return [int(value) for value in rejection["result_codes"]]
    match = FAILURE_CODES.search(str(row.get("failure") or ""))
    if match is None:
        return []
    return [int(value.strip()) for value in match.group(1).split(",")]


def _category(row: Mapping[str, Any]) -> str:
    if row.get("teacher_forced_success") is True:
        return "success"
    codes = _codes(row)
    for code in (13, 4, 3):
        if code in codes:
            return f"code{code}"
    failure = str(row.get("failure") or "")
    if failure.startswith("native_terminal_before_"):
        return "terminal_before"
    return "other"


def _source_data_i(row: Mapping[str, Any]) -> int | None:
    source = json.loads(
        Path(str(row["source_path"])).read_text(encoding="utf-8-sig")
    )
    events = source.get("card_plays") or source.get("ability_plays") or []
    flags = {
        int(event["data_i"])
        for event in events
        if isinstance(event, Mapping) and event.get("data_i") in (0, 1)
    }
    if not flags:
        return None
    if len(flags) != 1:
        raise RuntimeError(
            f"mixed data_i flags in source battle {row['battle_tag']}: {flags}"
        )
    return next(iter(flags))


def compare(before_root: Path, after_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    before_root = before_root.resolve(strict=True)
    after_root = after_root.resolve(strict=True)
    before_selection = before_root / "selection.jsonl"
    after_selection = after_root / "selection.jsonl"
    if before_selection.read_bytes() != after_selection.read_bytes():
        raise RuntimeError("pilot selections are not byte-identical")
    before = _indexed(before_root / "results.jsonl")
    after = _indexed(after_root / "results.jsonl")
    if set(before) != set(after):
        raise RuntimeError("pilot result tag sets differ")

    transitions: Counter[str] = Counter()
    after_by_data_i: Counter[str] = Counter()
    per_tag: list[dict[str, Any]] = []
    for tag in sorted(before):
        first = before[tag]
        second = after[tag]
        if first.get("source_sha256") != second.get("source_sha256"):
            raise RuntimeError(f"source SHA changed for {tag}")
        flag = _source_data_i(second)
        first_category = _category(first)
        second_category = _category(second)
        transition = f"{first_category}->{second_category}"
        transitions[transition] += 1
        after_by_data_i[f"data_i={flag}:{second_category}"] += 1
        per_tag.append({
            "battle_tag": tag,
            "data_i": flag,
            "transition": transition,
            "source_sha256": second.get("source_sha256"),
            "before": {
                "category": first_category,
                "failure": first.get("failure"),
                "accepted_deployment_actions": first.get(
                    "accepted_deployment_actions"
                ),
            },
            "after": {
                "category": second_category,
                "failure": second.get("failure"),
                "accepted_deployment_actions": second.get(
                    "accepted_deployment_actions"
                ),
            },
        })

    before_categories = Counter(_category(row) for row in before.values())
    after_categories = Counter(_category(row) for row in after.values())
    summary = {
        "schema_version": 1,
        "kind": "expert_native_coordinate_fix_comparison_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "before_root": str(before_root),
        "after_root": str(after_root),
        "semantic_delta": (
            "native coordinates compiled from RoyaleAPI x_raw/y_raw/data_i: "
            "data_i=0 rotates 18000x32000; data_i=1 is identity"
        ),
        "selection_sha256": _sha256(before_selection),
        "selection_bytes_equal": True,
        "battle_count": len(per_tag),
        "all_source_sha256_equal": True,
        "category_counts": {
            "before": dict(sorted(before_categories.items())),
            "after": dict(sorted(after_categories.items())),
        },
        "transition_counts": dict(sorted(transitions.items())),
        "after_by_data_i": dict(sorted(after_by_data_i.items())),
        "before_results_sha256": _sha256(before_root / "results.jsonl"),
        "after_results_sha256": _sha256(after_root / "results.jsonl"),
        "all_previous_successes_preserved": transitions["success->success"]
        == before_categories["success"],
    }
    return summary, per_tag


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before_root", type=Path)
    parser.add_argument("after_root", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    summary, per_tag = compare(args.before_root, args.after_root)
    with (output_root / "per-tag.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for row in per_tag:
            handle.write(json.dumps(
                row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ) + "\n")
    summary["per_tag_sha256"] = _sha256(output_root / "per-tag.jsonl")
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
