"""Create a libg replay JSON with arbitrary base-form decks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from native_core.decks import build_replay, parse_deck_text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck0", required=True, help="eight comma-separated ids/names")
    parser.add_argument("--deck1", required=True, help="eight comma-separated ids/names")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--level", type=int, default=11)
    parser.add_argument(
        "--template", type=Path,
        default=PROJECT_ROOT / "examples" / "eight-card-bootstrap.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    template = json.loads(args.template.read_text(encoding="utf-8-sig"))
    replay = build_replay(
        template,
        parse_deck_text(args.deck0),
        parse_deck_text(args.deck1),
        seed=args.seed,
        level=args.level,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(replay, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
