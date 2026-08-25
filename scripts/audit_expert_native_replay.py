"""Benchmark and audit expert artifacts for native replay eligibility."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import orjson
except ImportError:  # pragma: no cover
    orjson = None

from expert_v1.native_replay_plan import ReplayPlanError, compile_battle
from expert_v1.source_terminal_anchors import manifest_terminal_anchors


def loads(raw: bytes) -> dict[str, Any]:
    value = orjson.loads(raw) if orjson is not None else json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError("JSON root is not an object")
    return value


def audit_one(
    item: tuple[Path, tuple[int, int] | None]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    path, terminal_crowns = item
    try:
        plan = compile_battle(
            loads(path.read_bytes()), terminal_crowns=terminal_crowns
        )
        return ({
            "battle_tag": plan.battle_tag,
            "source_schema_version": plan.source_schema_version,
            "tier": plan.replay_tier,
            "native_replay_ready": plan.native_replay_ready,
            "state_provenance": plan.state_provenance,
            "action_provenance": plan.action_provenance,
            "hand_provenance": plan.hand_provenance,
            "ability_provenance": plan.ability_provenance,
            "terminal_provenance": plan.terminal_provenance,
            "terminal_crowns": plan.terminal_crowns,
            "duration_ticks": plan.duration_ticks,
            "actions": len(plan.actions),
            "missing_abilities": sum(
                side.missing_ability_event_count for side in plan.sides
            ),
            "compatible_initial_states": [
                side.cycle.compatible_initial_state_count for side in plan.sides
            ],
            "limitations": list(plan.limitations),
        }, None)
    except Exception as error:
        return (None, {
            "path": str(path),
            "error_type": type(error).__name__,
            "error": str(error),
            "expected_rejection": isinstance(error, ReplayPlanError),
        })


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_items(
    args: argparse.Namespace,
) -> list[tuple[Path, tuple[int, int] | None]]:
    if args.manifest is not None:
        records, anchors = manifest_terminal_anchors(args.manifest)
        if args.limit:
            records = records[:args.limit]
        return [
            (
                Path(str(row["source_path"])),
                anchors[str(row["battle_tag"])],
            )
            for row in records
        ]
    assert args.input_root is not None
    files = sorted(args.input_root.rglob("*.json"))
    if args.limit:
        files = files[:args.limit]
    return [(path, None) for path in files]


def run(args: argparse.Namespace) -> dict[str, Any]:
    items = source_items(args)
    if not items:
        raise FileNotFoundError("no source battle JSON found")
    started = time.perf_counter()
    counters: Counter[str] = Counter()
    limitations: Counter[str] = Counter()
    rejections: Counter[str] = Counter()
    ready_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for accepted, rejected in executor.map(audit_one, items):
            counters["files"] += 1
            if accepted is not None:
                counters["compiled"] += 1
                counters[f"schema_v{accepted['source_schema_version']}"] += 1
                counters[f"tier_{accepted['tier']}"] += 1
                counters["actions"] += int(accepted["actions"])
                counters["native_ticks"] += int(accepted["duration_ticks"])
                counters["missing_abilities"] += int(accepted["missing_abilities"])
                for reason in accepted["limitations"]:
                    limitations[str(reason)] += 1
                if accepted["native_replay_ready"]:
                    counters["native_replay_ready"] += 1
                    ready_rows.append(accepted)
            else:
                assert rejected is not None
                counters["rejected"] += 1
                rejections[str(rejected["error"])] += 1
                if len(rejected_rows) < args.max_rejected_examples:
                    rejected_rows.append(rejected)
    elapsed = time.perf_counter() - started
    summary = {
        "schema_version": 1,
        "kind": "expert_native_replay_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": (
            str(args.input_root.resolve()) if args.input_root is not None else None
        ),
        "source_manifest": (
            str(args.manifest.resolve()) if args.manifest is not None else None
        ),
        "source_manifest_sha256": (
            sha256(args.manifest) if args.manifest is not None else None
        ),
        "terminal_anchor_count": sum(
            1 for _, crowns in items if crowns is not None
        ),
        "workers": args.workers,
        "elapsed_seconds": elapsed,
        "files_per_second": counters["files"] / elapsed,
        "actions_per_second": counters["actions"] / elapsed,
        "source_native_ticks_per_second": counters["native_ticks"] / elapsed,
        **counters,
        "limitation_counts": dict(limitations.most_common()),
        "top_rejections": dict(rejections.most_common(50)),
        "original_state_exact": 0,
        "original_state_exact_reason": (
            "source lacks libg build, RNG seed and per-tick state anchors"
        ),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_root / "native-replay-ready.jsonl").open("w", encoding="utf-8") as out:
        for row in ready_rows:
            out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    with (args.output_root / "rejected-examples.jsonl").open("w", encoding="utf-8") as out:
        for row in rejected_rows:
            out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--input-root", type=Path)
    sources.add_argument("--manifest", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-rejected-examples", type=int, default=1000)
    args = parser.parse_args()
    if not 1 <= args.workers <= 64:
        raise ValueError("workers must be in 1..64")
    print(json.dumps(run(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
