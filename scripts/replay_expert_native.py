"""Replay one expert JSON through libg using the gap-batched native path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import orjson
except ImportError:  # pragma: no cover
    orjson = None

from expert_v1.native_replay_plan import compile_battle
from expert_v1.native_replay_runner import (
    calibrated_players,
    execute_plan,
    load_template,
)
from native_core.env import NativeRoyaleEnv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--port", type=int, default=38031)
    parser.add_argument(
        "--template", type=Path,
        default=PROJECT_ROOT / "examples" / "eight-card-bootstrap.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-non-candidate", action="store_true")
    parser.add_argument("--no-decision-records", action="store_true")
    parser.add_argument("--team-crowns", type=int)
    parser.add_argument("--opponent-crowns", type=int)
    args = parser.parse_args()
    raw = args.source.read_bytes()
    value = orjson.loads(raw) if orjson is not None else json.loads(raw)
    if (args.team_crowns is None) != (args.opponent_crowns is None):
        raise ValueError("team/opponent crowns must be supplied together")
    terminal_crowns = (
        None if args.team_crowns is None
        else (args.team_crowns, args.opponent_crowns)
    )
    plan = compile_battle(value, terminal_crowns=terminal_crowns)
    if not plan.native_replay_ready and not args.allow_non_candidate:
        raise RuntimeError(
            f"plan tier {plan.replay_tier!r} is not native-replay-ready; "
            "use --allow-non-candidate only for an explicitly synthetic audit"
        )
    template = load_template(args.template)
    with NativeRoyaleEnv(port=args.port) as env:
        calibration = calibrated_players(env, template)
        result = execute_plan(
            env, plan, template, calibration,
            capture_decisions=not args.no_decision_records,
        )
    output = {
        "source": str(args.source.resolve()),
        "plan": plan.json(),
        "result": result.json(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "battle_tag": plan.battle_tag,
        "tier": plan.replay_tier,
        "accepted": result.accepted,
        "failure": result.failure,
        "native_ticks_advanced": result.native_ticks_advanced,
        "wall_seconds": result.wall_seconds,
    }, ensure_ascii=False, indent=2))
    return 0 if result.accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
