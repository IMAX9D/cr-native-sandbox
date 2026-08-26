"""Compare two fixed-selection native action execution-phase pilots."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from expert_v1.native_pilot import sha256_file


DEFAULT_A = Path(
    r"D:\AI_data\cr-native-core\expert-v1"
    r"\native-teacher-forced-pilot-100-seed-dynamic-v7"
)
DEFAULT_B = Path(
    r"D:\AI_data\cr-native-core\expert-v1"
    r"\native-teacher-forced-pilot-100-action-phase-plus1-v8"
)
DEFAULT_OUTPUT = Path(
    r"D:\AI_data\cr-native-core\expert-v1"
    r"\native-action-phase-ab-v7-v8"
)
FAILURE_CODES = re.compile(r"codes_\[([^]]+)\]")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _indexed(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in _jsonl(path):
        tag = str(row["battle_tag"])
        if tag in result:
            raise RuntimeError(f"duplicate result tag: {tag}")
        result[tag] = row
    return result


def failure_codes(row: Mapping[str, Any]) -> list[int]:
    rejection = row.get("first_rejection")
    if isinstance(rejection, Mapping) and isinstance(
        rejection.get("result_codes"), list
    ):
        return [int(value) for value in rejection["result_codes"]]
    match = FAILURE_CODES.search(str(row.get("failure") or ""))
    if match is None:
        return []
    return [int(value.strip()) for value in match.group(1).split(",")]


def category(row: Mapping[str, Any]) -> str:
    if row.get("teacher_forced_success") is True:
        return "success"
    codes = failure_codes(row)
    if 13 in codes:
        return "code13"
    if 4 in codes:
        return "code4"
    failure = str(row.get("failure") or "")
    if failure.startswith("native_terminal_before_source_tick_") or failure.startswith(
        "native_terminal_before_execution_tick_"
    ):
        return "terminal_before"
    return "other"


def _rejection_ticks(row: Mapping[str, Any], *, default_offset: int) -> dict[str, Any]:
    rejection = row.get("first_rejection")
    if not isinstance(rejection, Mapping):
        return {
            "source_tick": None,
            "execution_tick": None,
            "execution_tick_offset": default_offset,
        }
    execution_tick = rejection.get("execution_tick", rejection.get("tick"))
    source_tick = rejection.get("source_tick")
    if source_tick is None and execution_tick is not None:
        source_tick = int(execution_tick) - default_offset
    return {
        "source_tick": None if source_tick is None else int(source_tick),
        "execution_tick": (
            None if execution_tick is None else int(execution_tick)
        ),
        "execution_tick_offset": int(
            rejection.get("execution_tick_offset", default_offset)
        ),
    }


def compact_result(
    row: Mapping[str, Any], *, default_offset: int
) -> dict[str, Any]:
    rejection = row.get("first_rejection")
    events = (
        rejection.get("events", []) if isinstance(rejection, Mapping) else []
    )
    return {
        "category": category(row),
        "teacher_forced_success": bool(row.get("teacher_forced_success")),
        "failure": row.get("failure"),
        "failure_codes": failure_codes(row),
        **_rejection_ticks(row, default_offset=default_offset),
        "rejected_cards": sorted({
            str(event["base_token"])
            for event in events
            if isinstance(event, Mapping) and event.get("base_token") is not None
        }),
        "chosen_seed": row.get("chosen_seed", row.get("seed")),
        "accepted_deployment_actions": int(
            row.get("accepted_deployment_actions") or 0
        ),
        "source_deployment_actions": int(
            row.get("source_deployment_actions") or 0
        ),
        "stored_tick_count": int(row.get("stored_tick_count") or 0),
        "terminal_status": row.get("terminal_status"),
        "source_crowns": row.get("source_crowns"),
        "observed_crowns": row.get("observed_crowns"),
        "logical_training_state_sha256": row.get(
            "logical_training_state_sha256"
        ),
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def compare(
    a_root: Path,
    b_root: Path,
    *,
    expected_tags: int = 100,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    a_root = a_root.resolve(strict=True)
    b_root = b_root.resolve(strict=True)
    a_selection = a_root / "selection.jsonl"
    b_selection = b_root / "selection.jsonl"
    if a_selection.read_bytes() != b_selection.read_bytes():
        raise RuntimeError("A/B selections are not byte-identical")
    selection_sha256 = sha256_file(a_selection)
    a = _indexed(a_root / "results.jsonl")
    b = _indexed(b_root / "results.jsonl")
    if set(a) != set(b):
        raise RuntimeError("A/B result Tag sets differ")
    if len(a) != expected_tags:
        raise RuntimeError(f"expected {expected_tags} Tags, got {len(a)}")
    selection_tags = {
        str(row["battle_tag"]) for row in _jsonl(a_selection)
    }
    if selection_tags != set(a):
        raise RuntimeError("A/B results do not cover the exact fixed selection")
    a_summary = _json(a_root / "summary.json")
    b_summary = _json(b_root / "summary.json")
    a_offset = int(a_summary.get("configuration", {}).get(
        "action_execution_tick_offset", 0
    ))
    b_offset = int(b_summary.get("configuration", {}).get(
        "action_execution_tick_offset", 0
    ))
    if (a_offset, b_offset) != (0, 1):
        raise RuntimeError(f"expected phase offsets 0/1, got {a_offset}/{b_offset}")
    declared_selection_sha = b_summary.get("selection", {}).get(
        "source_selection_sha256"
    )
    if declared_selection_sha != selection_sha256:
        raise RuntimeError(
            "B summary fixed-selection SHA does not match emitted selection"
        )

    per_tag: list[dict[str, Any]] = []
    transitions: Counter[str] = Counter()
    for tag in sorted(a):
        if a[tag].get("source_sha256") != b[tag].get("source_sha256"):
            raise RuntimeError(f"source SHA changed for {tag}")
        av = compact_result(a[tag], default_offset=a_offset)
        bv = compact_result(b[tag], default_offset=b_offset)
        transition = f"{av['category']}->{bv['category']}"
        transitions[transition] += 1
        per_tag.append({
            "battle_tag": tag,
            "source_sha256": a[tag].get("source_sha256"),
            "source_sha256_equal": True,
            "chosen_seed_equal": av["chosen_seed"] == bv["chosen_seed"],
            "logical_training_state_sha256_equal": (
                av["logical_training_state_sha256"]
                == bv["logical_training_state_sha256"]
            ),
            "terminal_status_equal": av["terminal_status"] == bv["terminal_status"],
            "transition": transition,
            "a": av,
            "b": bv,
        })

    seed_mismatches = [
        item["battle_tag"] for item in per_tag
        if not item["chosen_seed_equal"]
    ]
    if seed_mismatches:
        raise RuntimeError(f"chosen seed changed across A/B: {seed_mismatches}")

    a_categories = Counter(item["a"]["category"] for item in per_tag)
    b_categories = Counter(item["b"]["category"] for item in per_tag)
    shared_success = [
        item for item in per_tag
        if item["a"]["category"] == item["b"]["category"] == "success"
    ]
    a_terminal = Counter(str(item["a"]["terminal_status"]) for item in per_tag)
    b_terminal = Counter(str(item["b"]["terminal_status"]) for item in per_tag)
    summary = {
        "schema_version": 1,
        "kind": "expert_native_action_phase_ab_v1",
        "created_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "a_root": str(a_root),
        "b_root": str(b_root),
        "a_execution_tick_offset": a_offset,
        "b_execution_tick_offset": b_offset,
        "source_labels_unchanged": True,
        "selection_sha256": selection_sha256,
        "selection_bytes_equal": True,
        "tag_count": len(per_tag),
        "all_source_sha256_equal": True,
        "chosen_seed_equal_count": sum(
            item["chosen_seed_equal"] for item in per_tag
        ),
        "category_counts": {
            "a": dict(sorted(a_categories.items())),
            "b": dict(sorted(b_categories.items())),
        },
        "transition_counts": dict(sorted(transitions.items())),
        "shared_success": {
            "count": len(shared_success),
            "terminal_status_equal_count": sum(
                item["terminal_status_equal"] for item in shared_success
            ),
            "logical_training_state_sha256_equal_count": sum(
                item["logical_training_state_sha256_equal"]
                for item in shared_success
            ),
        },
        "terminal_status_counts": {
            "a": dict(sorted(a_terminal.items())),
            "b": dict(sorted(b_terminal.items())),
        },
        "throughput": {
            "a": {
                **a_summary["throughput"],
                "stored_ticks_per_second": a_summary["tick_trace"][
                    "ticks_per_wall_second"
                ],
            },
            "b": {
                **b_summary["throughput"],
                "stored_ticks_per_second": b_summary["tick_trace"][
                    "ticks_per_wall_second"
                ],
            },
        },
        "result_sha256": {
            "a": sha256_file(a_root / "results.jsonl"),
            "b": sha256_file(b_root / "results.jsonl"),
        },
        "conclusion": {
            "phase_semantics": "unknown",
            "production_default_offset": 0,
            "promote_offset_1": False,
            "reason": (
                "+1 removes six initial code13 failures but only three become "
                "complete successes; three move to later code4 and five code13 "
                "remain; no source per-Tick state truth selects the changed hashes"
            ),
        },
    }
    return summary, per_tag


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-root", type=Path, default=DEFAULT_A)
    parser.add_argument("--b-root", type=Path, default=DEFAULT_B)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-tags", type=int, default=100)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    summary, per_tag = compare(
        args.a_root, args.b_root, expected_tags=args.expected_tags
    )
    with (output_root / "per-tag.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for value in per_tag:
            handle.write(json.dumps(
                value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ) + "\n")
    summary["per_tag_sha256"] = sha256_file(output_root / "per-tag.jsonl")
    _atomic_json(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
