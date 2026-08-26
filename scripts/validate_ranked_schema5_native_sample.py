"""Run a fixed Schema5 sample through frozen libg and preserve every result."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from expert_v1.native_replay_plan import compile_battle  # noqa: E402
from expert_v1.native_replay_runner import execute_plan, load_template  # noqa: E402
from native_core.env import NativeRoyaleEnv  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=38031)
    parser.add_argument("--maximum-seeds", type=int, default=4096)
    args = parser.parse_args()
    selection = args.selection.resolve(strict=True)
    rows = [
        json.loads(line) for line in selection.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not rows or len({row["battle_tag"] for row in rows}) != len(rows):
        raise ValueError("selection must contain unique non-empty battles")
    template = load_template(PROJECT_ROOT / "examples" / "eight-card-bootstrap.json")
    results: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    with NativeRoyaleEnv(port=args.port) as env:
        for row in rows:
            source_path = Path(str(row["source_path"])).resolve(strict=True)
            if _sha(source_path) != str(row["source_sha256"]):
                raise RuntimeError(f"source SHA changed: {source_path}")
            source = json.loads(source_path.read_text(encoding="utf-8"))
            plan = compile_battle(
                source,
                terminal_crowns=(
                    int(row["team_crowns"]), int(row["opponent_crowns"])
                ),
            )
            try:
                replay = execute_plan(
                    env,
                    plan,
                    template,
                    capture_decisions=False,
                    maximum_seeds_to_test=args.maximum_seeds,
                    action_execution_tick_offset=1,
                )
                result = replay.json()
                failure = str(result.get("failure") or "")
            except Exception as error:  # preserve each fail-closed sample
                result = {}
                failure = f"{type(error).__name__}: {error}"
            failures[failure or "accepted"] += 1
            record = {
                "battle_tag": plan.battle_tag,
                "source_path": str(source_path),
                "source_sha256": row["source_sha256"],
                "source_mode": plan.numeric_game_mode_id,
                "execution_mode": plan.native_execution_game_mode_id,
                "king_tower_levels": [side.king_tower_level for side in plan.sides],
                "source_actions": len(plan.actions) + len(plan.ability_events),
                "source_abilities": len(plan.ability_events),
                "accepted": bool(result.get("accepted")),
                "teacher_forced_success": bool(result.get("teacher_forced_success")),
                "failure": failure or None,
                "chosen_seed": result.get("chosen_seed"),
                "seeds_tested": result.get("seeds_tested"),
                "accepted_actions": result.get("accepted_actions"),
                "terminal_diagnostic_status": result.get("terminal_diagnostic_status"),
                "terminal_tower_hp_match": result.get("terminal_tower_hp_match"),
                "native_ticks_advanced": result.get("native_ticks_advanced"),
                "wall_seconds": result.get("wall_seconds"),
            }
            results.append(record)
            _atomic_json(args.output_root / "results" / f"{plan.battle_tag}.json", {
                "schema_version": 1,
                "kind": "ranked_schema5_native_sample_result_v1",
                "record": record,
                "plan": plan.json(),
                "native_result": result,
            })
    summary = {
        "schema_version": 1,
        "kind": "ranked_schema5_native_sample_summary_v1",
        "selection": str(selection),
        "selection_sha256": _sha(selection),
        "episodes": len(results),
        "teacher_forced_successes": sum(
            bool(row["teacher_forced_success"]) for row in results
        ),
        "accepted_native_prefixes": sum(bool(row["accepted"]) for row in results),
        "source_mode_counts": dict(Counter(str(row["source_mode"]) for row in results)),
        "execution_mode_counts": dict(Counter(str(row["execution_mode"]) for row in results)),
        "failure_counts": dict(failures),
        "results": results,
    }
    _atomic_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["teacher_forced_successes"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
