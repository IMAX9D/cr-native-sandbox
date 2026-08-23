"""Persistent-worker recurrent PPO self-play on the original native core."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time
from typing import Any
import uuid

import numpy as np
import torch

from native_core.client import JsonLineClient
from native_core.env import NativeRoyaleEnv
from native_core.worker import HeadlessWorkerPool, WorkerConfig

from .model import RecurrentPolicyValueNet
from .ppo import PPOConfig, PPOTrainer
from .rollout import EpisodeResult, save_episode
from .schema import RunStore
from .vector_rollout import VectorNativeSelfPlayCollector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPLAY = PROJECT_ROOT / "examples" / "eight-card-bootstrap.json"
DEFAULT_DATA_ROOT = Path(r"D:\AI_data\cr-native-core\training")


def _atomic_torch_save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1000000)
    parser.add_argument("--episodes-per-iteration", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-ticks", type=int, default=7200)
    parser.add_argument("--base-port", type=int, default=37031)
    parser.add_argument("--direct-base-port", type=int, default=38031)
    parser.add_argument("--transport", choices=("direct", "adb"), default="direct")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--reward", choices=("terminal", "potential"), default="potential")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-id")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--skip-worker-start", action="store_true")
    parser.add_argument("--profile-native", action="store_true")
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.smoke:
        args.iterations = 1
        args.episodes_per_iteration = 1
        args.workers = 1
        args.max_ticks = 128
        args.run_id = args.run_id or "smoke-" + datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        ) + "-" + uuid.uuid4().hex[:8]
    if min(args.iterations, args.episodes_per_iteration, args.workers) < 1:
        raise ValueError("iterations, episodes-per-iteration and workers must be positive")
    if args.workers > 8:
        raise ValueError("workers must be in 1..8")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    replay = json.loads(args.replay.read_text(encoding="utf-8-sig"))
    ppo_config = PPOConfig()
    config = {
        "schema_version": 1,
        "algorithm": "persistent_native_recurrent_ppo",
        "iterations": args.iterations,
        "episodes_per_iteration": args.episodes_per_iteration,
        "workers": args.workers,
        "seed": args.seed,
        "max_ticks": args.max_ticks,
        "base_port": args.base_port,
        "environment_base_port": (
            args.direct_base_port if args.transport == "direct" else args.base_port
        ),
        "transport": args.transport,
        "device": str(device),
        "torch": torch.__version__,
        "reward": args.reward,
        "ppo": asdict(ppo_config),
        "episode_reset": "native_battle_game_state_4_to_4_in_process",
        "truth_source": "surface_free_original_libg_15.535.29_x86_64",
        "action_legality": "native_validate_deployment_18x32",
        "profile_native": args.profile_native,
        "cuda_graph_inference": (
            args.device.startswith("cuda") and not args.disable_cuda_graph
        ),
    }
    paths = RunStore(args.data_root).create(config, run_id=args.run_id)
    events = paths.logs / "events.jsonl"
    _append_jsonl(events, {
        "event": "run_start", "utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
    })

    pool = HeadlessWorkerPool(WorkerConfig(
        service_base_port=args.base_port,
        direct_base_port=args.direct_base_port,
    ))
    if not args.skip_worker_start:
        worker_state = pool.ensure_ready(
            args.workers, configure_direct=args.transport == "direct"
        )
        _append_jsonl(events, {"event": "workers_ready", "state": worker_state})
    elif args.transport == "direct":
        direct_state = pool.configure_direct_ports(args.workers)
        _append_jsonl(events, {"event": "direct_transport_ready", "state": direct_state})

    model = RecurrentPolicyValueNet().to(device)
    model.enable_cuda_graph_inference(
        device.type == "cuda" and not args.disable_cuda_graph
    )
    trainer = PPOTrainer(model, device=device, config=ppo_config)
    native_ticks = 0
    completed_episodes = 0
    starting_iteration = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer"])
        native_ticks = int(checkpoint.get("native_ticks", 0))
        completed_episodes = int(checkpoint.get("completed_episodes", 0))
        starting_iteration = int(checkpoint.get("iteration", 0))

    next_seed = args.seed
    environment_base_port = (
        args.direct_base_port if args.transport == "direct" else args.base_port
    )
    envs = [
        NativeRoyaleEnv(
            port=environment_base_port + slot,
            timeout=30,
            profile_native=args.profile_native,
        )
        for slot in range(args.workers)
    ]
    for offset in range(1, args.iterations + 1):
        iteration = starting_iteration + offset
        pending = list(range(args.episodes_per_iteration))
        results: list[EpisodeResult] = []
        rpc_total_latency: list[float] = []
        rpc_receive_latency: list[float] = []
        iteration_started = time.perf_counter()
        while pending:
            wave = pending[: args.workers]
            del pending[: len(wave)]
            model.eval()
            seeds = list(range(next_seed, next_seed + len(wave)))
            next_seed += len(wave)
            collector = VectorNativeSelfPlayCollector(
                envs[: len(wave)], model, replay, device=device,
                reward_mode=args.reward, max_ticks=args.max_ticks,
            )
            collected = collector.collect(seeds)
            rpc_total_latency.extend(collector.rpc_latency_samples["total"])
            rpc_receive_latency.extend(collector.rpc_latency_samples["receive"])
            for result in collected:
                save_episode(paths.trajectories, result, full_debug=args.smoke)
                results.append(result)
                _append_jsonl(events, {"event": "episode_complete", **result.summary()})
                native_ticks += sum(len(item.rewards) for item in result.trajectories) // 2
                completed_episodes += 1

        trajectories = [item for result in results for item in result.trajectories]
        sampling_profile: dict[str, float] = {}
        for result in results:
            for key, value in result.profile.items():
                sampling_profile[key] = sampling_profile.get(key, 0.0) + float(value)
        sampling_profile.update(JsonLineClient.latency_summary(
            rpc_total_latency,
            rpc_receive_latency,
            attempts=sampling_profile.get("rpc_attempts", 0.0),
            failures=sampling_profile.get("rpc_failures", 0.0),
        ))
        update_started = time.perf_counter()
        metrics = trainer.update(trajectories)
        metrics["learner_wall_seconds"] = time.perf_counter() - update_started
        metrics["iteration_wall_seconds"] = time.perf_counter() - iteration_started
        metrics["environment_steps"] = float(
            sum(len(item.rewards) for item in trajectories) // 2
        )
        metrics["environment_steps_per_second"] = (
            metrics["environment_steps"]
            / max(1e-9, metrics["iteration_wall_seconds"] - metrics["learner_wall_seconds"])
        )
        checkpoint = {
            "schema_version": 1,
            "kind": "native_eight_card_recurrent_ppo_checkpoint",
            "iteration": iteration,
            "native_ticks": native_ticks,
            "completed_episodes": completed_episodes,
            "model": model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "config": config,
            "metrics": metrics,
            "episode_summaries": [item.summary() for item in results],
            "sampling_profile": sampling_profile,
        }
        numbered = paths.checkpoints / f"checkpoint-{iteration:06d}.pt"
        _atomic_torch_save(numbered, checkpoint)
        _atomic_torch_save(paths.checkpoints / "latest.pt", checkpoint)
        if args.smoke:
            loaded = torch.load(
                paths.checkpoints / "latest.pt", map_location="cpu", weights_only=False
            )
            if not (
                loaded.get("kind") == "native_eight_card_recurrent_ppo_checkpoint"
                and int(loaded.get("iteration", -1)) == iteration
                and int(loaded.get("completed_episodes", 0)) >= 1
                and loaded.get("model")
                and loaded.get("optimizer")
            ):
                raise RuntimeError("smoke checkpoint reload/contract validation failed")
        event = {
            "event": "iteration_complete", "iteration": iteration,
            "native_ticks": native_ticks, "episodes": completed_episodes,
            "metrics": metrics, "checkpoint": str(numbered),
            "sampling_profile": sampling_profile,
        }
        _append_jsonl(events, event)
        RunStore._atomic_json(args.data_root / "latest_run.json", {
            "run": str(paths.root), "checkpoint": str(paths.checkpoints / "latest.pt"),
            "native_ticks": native_ticks, "episodes": completed_episodes,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        })
        print(json.dumps(event, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
