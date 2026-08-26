"""Generate the authoritative expert per-Tick dataset with original libg.so."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from expert_v1.native_dataset_generator import (  # noqa: E402
    prepare_run,
    run_generation,
)
from expert_v1.native_replay_plan import DEFAULT_NATIVE_SEED  # noqa: E402
from expert_v1.tick_store_v1.work_queue import TickStoreWorkQueue  # noqa: E402


DEFAULT_CANDIDATES = Path(
    r"D:\AI_data\cr-native-core\expert-v1\native-eligibility-v1"
    r"\queues\authoritative-native-full.jsonl"
)
DEFAULT_OUTPUT = Path(
    r"D:\AI_data\cr-native-core\expert-v1"
    r"\native-authoritative-ticks-v1"
)
DEFAULT_TEMPLATE = PROJECT_ROOT / "examples" / "eight-card-bootstrap.json"
DEFAULT_PORTS = [38031, 38032, 38033, 38034]


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--queue", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--selection-seed", default="authoritative-native-full-v1"
    )
    parser.add_argument(
        "--deployment-zero-quota", type=int,
        help="exact number of source-reports-zero battles in a limited run",
    )
    parser.add_argument(
        "--ability-exact-quota", type=int,
        help="exact number of exact-Tick ability-positive battles in a limited run",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_NATIVE_SEED)
    parser.add_argument("--maximum-seeds", type=int, default=4096)
    parser.add_argument("--trace-batch-steps", type=int, default=64)
    parser.add_argument("--episodes-per-shard", type=int, default=256)


def _prepare(args: argparse.Namespace) -> int:
    tasks, selection, queue, contract = prepare_run(
        candidate_queue=args.queue,
        output_root=args.output_root,
        template_path=args.template,
        limit=args.limit,
        selection_seed=args.selection_seed,
        deployment_zero_quota=args.deployment_zero_quota,
        ability_exact_quota=args.ability_exact_quota,
        seed=args.seed,
        maximum_seeds_to_test=args.maximum_seeds,
        trace_batch_steps=args.trace_batch_steps,
        episodes_per_shard=args.episodes_per_shard,
    )
    with TickStoreWorkQueue(queue) as work_queue:
        counts = work_queue.counts()
    print(json.dumps({
        "prepared": True,
        "selected_battles": len(tasks),
        "selection_manifest": str(selection),
        "work_queue": str(queue),
        "queue_counts": counts,
        "run_contract": contract,
    }, ensure_ascii=False, indent=2))
    return 0


def _run(args: argparse.Namespace) -> int:
    summary = run_generation(
        candidate_queue=args.queue,
        output_root=args.output_root,
        template_path=args.template,
        ports=args.ports,
        workers=args.workers,
        limit=args.limit,
        selection_seed=args.selection_seed,
        deployment_zero_quota=args.deployment_zero_quota,
        ability_exact_quota=args.ability_exact_quota,
        seed=args.seed,
        maximum_seeds_to_test=args.maximum_seeds,
        trace_batch_steps=args.trace_batch_steps,
        episodes_per_shard=args.episodes_per_shard,
        lease_seconds=args.lease_seconds,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["publication_ready"] else 2


def _status(args: argparse.Namespace) -> int:
    output = args.output_root.resolve()
    queue_path = output / "work-queue.sqlite3"
    value: dict[str, object] = {
        "output_root": str(output),
        "prepared": queue_path.exists(),
        "summary": None,
        "manifest": None,
    }
    if queue_path.exists():
        with TickStoreWorkQueue(queue_path) as queue:
            value["queue_counts"] = queue.counts()
    for name in ("summary", "manifest"):
        path = output / f"{name}.json"
        if path.exists():
            value[name] = json.loads(path.read_text(encoding="utf-8-sig"))
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser(
        "prepare", help="freeze selection and initialize the resume queue"
    )
    _common(prepare_parser)
    prepare_parser.set_defaults(function=_prepare)

    run_parser = commands.add_parser(
        "run", help="resume or start authoritative native generation"
    )
    _common(run_parser)
    run_parser.add_argument("--workers", type=int, default=4)
    run_parser.add_argument(
        "--ports", type=int, nargs="+", default=DEFAULT_PORTS
    )
    run_parser.add_argument("--lease-seconds", type=float, default=900.0)
    run_parser.set_defaults(function=_run)

    status_parser = commands.add_parser("status")
    status_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    status_parser.set_defaults(function=_status)
    args = parser.parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
