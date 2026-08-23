"""Build the fixed-seed v0.1 cross-play matrix and RandomLegal baselines."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import random
import re
import sys
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from native_core.env import NativeRoyaleEnv
from native_core.worker import MultiAvdWorkerPool
from training.baselines import RandomLegalPolicy
from training.evaluate import evaluate_pair, load_neural_policy
from training.schema import RunStore


DEFAULT_REPLAY = PROJECT_ROOT / "examples" / "eight-card-bootstrap.json"
LABEL_PATTERN = re.compile(r"^P(\d{3})$")


def _candidate_key(path: Path) -> int:
    match = LABEL_PATTERN.match(path.stem)
    if not match:
        raise ValueError(f"invalid candidate label: {path.stem}")
    return int(match.group(1))


def _reset_policy_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _release() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--paired-seeds", type=int, default=16)
    parser.add_argument("--seed-start", type=int, default=900_000)
    parser.add_argument("--policy-seed", type=int, default=202_608_23)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--avds", type=int, default=2)
    parser.add_argument("--workers-per-avd", type=int, default=4)
    parser.add_argument("--max-ticks", type=int, default=7200)
    parser.add_argument("--base-port", type=int, default=37031)
    parser.add_argument("--direct-base-port", type=int, default=38031)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--skip-worker-start", action="store_true")
    parser.add_argument("--keep-vms", action="store_true")
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def _resume_matchup(
    path: Path,
    *,
    seeds: list[int],
    policy_seed: int,
    candidate_path: Path,
    opponent_path: Path | None,
    random_seed: int | None = None,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    expected_candidate = str(candidate_path.resolve())
    actual_opponent = value.get("opponent", {})
    valid_opponent = (
        str(actual_opponent.get("path")) == str(opponent_path.resolve())
        if opponent_path is not None
        else (
            actual_opponent.get("kind") == "random_legal_v1"
            and int(actual_opponent.get("seed", -1)) == int(random_seed)
        )
    )
    if not (
        value.get("kind") == "native_paired_side_swapped_evaluation"
        and value.get("environment_seeds") == seeds
        and int(value.get("policy_rng_seed", -1)) == policy_seed
        and str(value.get("candidate", {}).get("path")) == expected_candidate
        and valid_opponent
        and value.get("summary", {}).get("passed_integrity") is True
    ):
        raise RuntimeError(f"existing matchup cannot be safely resumed: {path}")
    return dict(value["summary"])


def main() -> int:
    args = build_parser().parse_args()
    run_root = args.run_root.resolve()
    manifest_path = run_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("kind") != "native_eight_card_selfplay_run":
        raise RuntimeError("not a native v0.1 training run")
    if args.paired_seeds < 1:
        raise ValueError("paired-seeds must be positive")
    if args.workers != args.avds * args.workers_per_avd:
        raise ValueError("workers must equal avds * workers-per-avd")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    candidates = sorted(
        (run_root / "evaluations" / "candidates").glob("P[0-9][0-9][0-9].pt"),
        key=_candidate_key,
    )
    if len(candidates) < 2:
        raise RuntimeError("evaluation requires P000 and at least one trained candidate")
    labels = [path.stem for path in candidates]
    seeds = list(range(args.seed_start, args.seed_start + args.paired_seeds))
    output_root = run_root / "evaluations" / "official-v0.1"
    matchups_root = output_root / "matchups"
    RunStore._atomic_json(output_root / "seed-set-v1.json", {
        "schema_version": 1,
        "kind": "paired_side_swapped_evaluation_seed_set",
        "environment_seeds": seeds,
        "policy_seed_base": args.policy_seed,
        "games_per_matchup": len(seeds) * 2,
    })
    replay = json.loads(args.replay.read_text(encoding="utf-8-sig"))
    pool = MultiAvdWorkerPool(
        avds=args.avds,
        workers_per_avd=args.workers_per_avd,
        service_base_port=args.base_port,
        direct_base_port=args.direct_base_port,
    )
    matrix = {
        row: {column: (0.5 if row == column else None) for column in labels}
        for row in labels
    }
    pair_summaries: dict[str, Any] = {}
    random_summaries: dict[str, Any] = {}
    matchup_index = 0
    try:
        if not args.skip_worker_start:
            pool.ensure_ready(configure_direct=True)
        else:
            pool.configure_direct_ports()
        envs = [
            NativeRoyaleEnv(port=port, timeout=30)
            for port in pool.environment_ports("direct")
        ]
        cuda_graph = not args.disable_cuda_graph
        for candidate_index in range(1, len(candidates)):
            for opponent_index in range(candidate_index):
                candidate_path = candidates[candidate_index]
                opponent_path = candidates[opponent_index]
                policy_seed = args.policy_seed + matchup_index
                matchup_index += 1
                name = f"{candidate_path.stem}-vs-{opponent_path.stem}"
                matchup_path = matchups_root / f"{name}.json"
                existing = _resume_matchup(
                    matchup_path,
                    seeds=seeds,
                    policy_seed=policy_seed,
                    candidate_path=candidate_path,
                    opponent_path=opponent_path,
                ) if args.resume else None
                if existing is not None:
                    pair_summaries[name] = existing
                    matrix[candidate_path.stem][opponent_path.stem] = existing[
                        "score_rate"
                    ]
                    matrix[opponent_path.stem][candidate_path.stem] = (
                        1.0 - existing["score_rate"]
                    )
                    print(json.dumps({
                        "event": "evaluation_matchup_resumed",
                        "matchup": name,
                    }), flush=True)
                    continue
                candidate, candidate_meta = load_neural_policy(
                    candidate_path, device=device, cuda_graph=cuda_graph
                )
                opponent, opponent_meta = load_neural_policy(
                    opponent_path, device=device, cuda_graph=cuda_graph
                )
                _reset_policy_rng(policy_seed)
                summary, records = evaluate_pair(
                    envs=envs,
                    candidate=candidate,
                    opponent=opponent,
                    replay=replay,
                    seeds=seeds,
                    device=device,
                    max_ticks=args.max_ticks,
                )
                value = {
                    "schema_version": 1,
                    "kind": "native_paired_side_swapped_evaluation",
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "candidate": candidate_meta,
                    "opponent": opponent_meta,
                    "policy_rng_seed": policy_seed,
                    "environment_seeds": seeds,
                    "summary": summary,
                    "matches": records,
                }
                RunStore._atomic_json(matchup_path, value)
                pair_summaries[name] = summary
                matrix[candidate_path.stem][opponent_path.stem] = summary[
                    "score_rate"
                ]
                matrix[opponent_path.stem][candidate_path.stem] = (
                    1.0 - summary["score_rate"]
                )
                del candidate, opponent
                _release()

        for candidate_path in candidates:
            policy_seed = args.policy_seed + matchup_index
            matchup_index += 1
            name = f"{candidate_path.stem}-vs-RandomLegal"
            matchup_path = matchups_root / f"{name}.json"
            existing = _resume_matchup(
                matchup_path,
                seeds=seeds,
                policy_seed=policy_seed,
                candidate_path=candidate_path,
                opponent_path=None,
                random_seed=policy_seed + 1,
            ) if args.resume else None
            if existing is not None:
                random_summaries[candidate_path.stem] = existing
                print(json.dumps({
                    "event": "evaluation_matchup_resumed",
                    "matchup": name,
                }), flush=True)
                continue
            candidate, candidate_meta = load_neural_policy(
                candidate_path, device=device, cuda_graph=cuda_graph
            )
            _reset_policy_rng(policy_seed)
            random_policy = RandomLegalPolicy(policy_seed + 1)
            summary, records = evaluate_pair(
                envs=envs,
                candidate=candidate,
                opponent=random_policy,
                replay=replay,
                seeds=seeds,
                device=device,
                max_ticks=args.max_ticks,
            )
            value = {
                "schema_version": 1,
                "kind": "native_paired_side_swapped_evaluation",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "candidate": candidate_meta,
                "opponent": {
                    "kind": "random_legal_v1",
                    "seed": policy_seed + 1,
                    "legality": "current_hand+elixir+native_deployment_mask",
                },
                "policy_rng_seed": policy_seed,
                "environment_seeds": seeds,
                "summary": summary,
                "matches": records,
            }
            RunStore._atomic_json(matchup_path, value)
            random_summaries[candidate_path.stem] = summary
            del candidate, random_policy
            _release()

        integrity = all(
            value["passed_integrity"]
            for value in [*pair_summaries.values(), *random_summaries.values()]
        )
        final_label = labels[-1]
        previous_label = labels[-2]
        summary = {
            "schema_version": 1,
            "kind": "selfplay_v0_1_evaluation_suite",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "run_root": str(run_root),
            "candidate_labels": labels,
            "paired_seed_count": len(seeds),
            "games_per_matchup": len(seeds) * 2,
            "cross_play_score_matrix": matrix,
            "pair_summaries": pair_summaries,
            "random_legal_summaries": random_summaries,
            "final_candidate": final_label,
            "required_comparisons": {
                "vs_initial": pair_summaries.get(f"{final_label}-vs-P000"),
                "vs_previous": pair_summaries.get(
                    f"{final_label}-vs-{previous_label}"
                ),
                "vs_random_legal": random_summaries.get(final_label),
            },
            "passed_integrity": integrity,
        }
        RunStore._atomic_json(output_root / "evaluation-summary.json", summary)
        print(json.dumps({
            "evaluation_summary": str(
                (output_root / "evaluation-summary.json").resolve()
            ),
            "candidates": labels,
            "matchups": len(pair_summaries) + len(random_summaries),
            "passed_integrity": integrity,
        }, ensure_ascii=False), flush=True)
        if not integrity:
            raise RuntimeError("one or more evaluation matchups failed integrity")
        return 0
    finally:
        if not args.keep_vms:
            pool.stop(keep_vms=False)


if __name__ == "__main__":
    raise SystemExit(main())
