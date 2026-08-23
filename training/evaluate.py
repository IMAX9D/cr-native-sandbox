"""Paired-seed, side-swapped evaluation on the original native core."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch

from native_core.env import NativeRoyaleEnv
from native_core.worker import MultiAvdWorkerPool

from .baselines import RandomLegalPolicy
from .model import RecurrentPolicyValueNet
from .run_contract import CHECKPOINT_KIND, state_dict_digest
from .schema import RunStore
from .vector_rollout import VectorNativeSelfPlayCollector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPLAY = PROJECT_ROOT / "examples" / "eight-card-bootstrap.json"


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    RunStore._atomic_json(path, value)


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 0.0
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _mean_interval(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(values))
    if len(values) == 1:
        return mean, 0.0, 1.0
    margin = 1.959963984540054 * float(np.std(values, ddof=1)) / math.sqrt(
        len(values)
    )
    return mean, max(0.0, mean - margin), min(1.0, mean + margin)


def summarize_evaluation(records: list[dict[str, Any]]) -> dict[str, Any]:
    games = len(records)
    wins = sum(item["outcome"] == "win" for item in records)
    losses = sum(item["outcome"] == "loss" for item in records)
    draws = sum(item["outcome"] == "draw" for item in records)
    win_low, win_high = wilson_interval(wins, games)
    paired_scores: dict[int, list[float]] = {}
    for item in records:
        paired_scores.setdefault(int(item["seed"]), []).append(float(item["score"]))
    incomplete = sorted(
        seed for seed, values in paired_scores.items() if len(values) != 2
    )
    pair_values = [float(np.mean(values)) for values in paired_scores.values()]
    score_mean, score_low, score_high = _mean_interval(pair_values)
    normal_terminals = sum(bool(item["terminated"]) for item in records)
    rejection_count = sum(int(item["native_action_rejections"]) for item in records)
    rejection_codes: dict[str, int] = {}
    for item in records:
        for code, count in item.get(
            "native_action_rejection_codes", {}
        ).items():
            rejection_codes[str(code)] = (
                rejection_codes.get(str(code), 0) + int(count)
            )
    return {
        "games": games,
        "paired_seed_count": len(paired_scores),
        "incomplete_paired_seeds": incomplete,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / games if games else 0.0,
        "win_rate_95ci": [win_low, win_high],
        "score_rate": score_mean,
        "paired_score_rate_95ci": [score_low, score_high],
        "average_crown_difference": float(np.mean([
            item["crown_difference"] for item in records
        ])) if records else 0.0,
        "average_tower_hp_difference": float(np.mean([
            item["tower_hp_difference"] for item in records
        ])) if records else 0.0,
        "average_match_ticks": float(np.mean([
            item["match_ticks"] for item in records
        ])) if records else 0.0,
        "average_match_seconds": float(np.mean([
            item["match_ticks"] * 0.05 for item in records
        ])) if records else 0.0,
        "normal_terminal_rate": normal_terminals / games if games else 0.0,
        "native_action_rejections": rejection_count,
        "native_action_rejection_codes": rejection_codes,
        "passed_integrity": (
            games > 0
            and normal_terminals == games
            and rejection_count == 0
            and not incomplete
        ),
    }


def load_neural_policy(
    checkpoint_path: Path,
    *,
    device: torch.device,
    cuda_graph: bool,
) -> tuple[RecurrentPolicyValueNet, dict[str, Any]]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if checkpoint.get("kind") != CHECKPOINT_KIND:
        raise RuntimeError(f"checkpoint kind mismatch: {checkpoint_path}")
    model = RecurrentPolicyValueNet().to(device)
    model.load_state_dict(checkpoint["model"])
    model.enable_cuda_graph_inference(cuda_graph and device.type == "cuda")
    model.eval()
    digest = state_dict_digest(checkpoint["model"])
    expected = checkpoint.get("current_model_digest")
    if expected is not None and str(expected) != digest:
        raise RuntimeError(f"checkpoint model digest mismatch: {checkpoint_path}")
    return model, {
        "kind": "checkpoint",
        "path": str(checkpoint_path.resolve()),
        "native_ticks": int(checkpoint.get("native_ticks", 0)),
        "iteration": int(checkpoint.get("iteration", 0)),
        "model_digest": digest,
    }


def evaluate_pair(
    *,
    envs: list[NativeRoyaleEnv],
    candidate: Any,
    opponent: Any,
    replay: Mapping[str, Any],
    seeds: list[int],
    device: torch.device,
    max_ticks: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for candidate_side in (0, 1):
        for offset in range(0, len(seeds), len(envs)):
            wave = seeds[offset : offset + len(envs)]
            policies = {
                candidate_side: candidate,
                1 - candidate_side: opponent,
            }
            collector = VectorNativeSelfPlayCollector(
                envs[: len(wave)],
                candidate,
                replay,
                device=device,
                reward_mode="terminal",
                max_ticks=max_ticks,
                policies_by_side=policies,
            )
            results = collector.collect(wave)
            for result in results:
                if result.winner is None:
                    outcome = "draw"
                    score = 0.5
                elif int(result.winner) == candidate_side:
                    outcome = "win"
                    score = 1.0
                else:
                    outcome = "loss"
                    score = 0.0
                direction = 1 if candidate_side == 0 else -1
                records.append({
                    "seed": result.seed,
                    "candidate_side": candidate_side,
                    "winner": result.winner,
                    "outcome": outcome,
                    "score": score,
                    "terminated": result.terminated,
                    "truncated": result.truncated,
                    "match_ticks": int(result.behavior.get("match_ticks", result.tick)),
                    "crown_difference": direction * int(
                        result.behavior.get("crown_difference_side0", 0)
                    ),
                    "tower_hp_difference": direction * int(
                        result.behavior.get("tower_hp_difference_side0", 0)
                    ),
                    "native_action_rejections": int(
                        result.behavior.get("native_rejection_count", 0)
                    ),
                    "native_action_rejection_codes": dict(
                        result.behavior.get("native_rejection_codes", {})
                    ),
                    "state_hash": result.state_hash,
                })
    return summarize_evaluation(records), records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--opponent", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paired-seeds", type=int, default=32)
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
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.paired_seeds < 1:
        raise ValueError("paired-seeds must be positive")
    if args.workers != args.avds * args.workers_per_avd:
        raise ValueError("workers must equal avds * workers-per-avd")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    random.seed(args.policy_seed)
    np.random.seed(args.policy_seed)
    torch.manual_seed(args.policy_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.policy_seed)
    cuda_graph = not args.disable_cuda_graph
    candidate, candidate_meta = load_neural_policy(
        args.candidate, device=device, cuda_graph=cuda_graph
    )
    if args.opponent:
        opponent, opponent_meta = load_neural_policy(
            args.opponent, device=device, cuda_graph=cuda_graph
        )
    else:
        opponent = RandomLegalPolicy(args.policy_seed + 1)
        opponent_meta = {
            "kind": "random_legal_v1",
            "seed": args.policy_seed + 1,
            "legality": "current_hand+elixir+native_deployment_mask",
        }
    random.seed(args.policy_seed)
    np.random.seed(args.policy_seed)
    torch.manual_seed(args.policy_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.policy_seed)
    replay = json.loads(args.replay.read_text(encoding="utf-8-sig"))
    seeds = list(range(args.seed_start, args.seed_start + args.paired_seeds))
    pool = MultiAvdWorkerPool(
        avds=args.avds,
        workers_per_avd=args.workers_per_avd,
        service_base_port=args.base_port,
        direct_base_port=args.direct_base_port,
    )
    try:
        if not args.skip_worker_start:
            pool.ensure_ready(configure_direct=True)
        else:
            pool.configure_direct_ports()
        envs = [
            NativeRoyaleEnv(port=port, timeout=30)
            for port in pool.environment_ports("direct")
        ]
        summary, records = evaluate_pair(
            envs=envs,
            candidate=candidate,
            opponent=opponent,
            replay=replay,
            seeds=seeds,
            device=device,
            max_ticks=args.max_ticks,
        )
        output = {
            "schema_version": 1,
            "kind": "native_paired_side_swapped_evaluation",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "candidate": candidate_meta,
            "opponent": opponent_meta,
            "policy_rng_seed": args.policy_seed,
            "environment_seeds": seeds,
            "native_tick_hz": 20,
            "summary": summary,
            "matches": records,
        }
        _atomic_json(args.output, output)
        print(json.dumps({
            "evaluation": str(args.output.resolve()),
            **summary,
        }, ensure_ascii=False), flush=True)
        if not summary["passed_integrity"]:
            raise RuntimeError("evaluation integrity checks failed")
        return 0
    finally:
        if not args.keep_vms:
            pool.stop(keep_vms=False)


if __name__ == "__main__":
    raise SystemExit(main())
