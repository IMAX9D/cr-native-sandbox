"""Persistent native Self-Play v0.2 continuous-rate recurrent PPO."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
from typing import Any, Mapping
import uuid

import numpy as np
import torch

from native_core.client import JsonLineClient
from native_core.env import NativeRoyaleEnv
from native_core.worker import MultiAvdWorkerPool
from training.ppo import PPOConfig
from training.rollout import EpisodeResult, save_episode
from training.run_contract import (
    assert_healthy,
    capture_rng_state,
    clone_state_dict,
    model_digest,
    model_distance,
    restore_rng_state,
)
from training.schema import PotentialReward, RunStore
from training.vector_rollout import summarize_barrier

from . import CHECKPOINT_KIND, SCHEMA_VERSION
from .migrate import initialize_model
from .model import ContinuousRatePolicyValueNet
from .ppo import ContinuousRatePPOTrainer
from .rollout import aggregate_timed_behavior
from .vector_rollout import ContinuousRateVectorCollector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPLAY = PROJECT_ROOT / "examples" / "eight-card-bootstrap.json"
DEFAULT_DATA_ROOT = Path(r"D:\AI_data\cr-native-core\selfplay-v0.2")
DEFAULT_PARENT = Path(
    r"D:\AI_data\cr-native-core\selfplay-v0.1\runs"
    r"\selfplay-v0.1-stage-a-20260823T141402Z"
    r"\evaluations\candidates\P010.pt"
)


def _atomic_torch_save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(
            value, ensure_ascii=False, separators=(",", ":")
        ) + "\n")
        stream.flush()


def _atomic_npz(path: Path, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def _crossed(before: int, after: int, interval: int) -> list[int]:
    first = (before // interval + 1) * interval
    return list(range(first, after + 1, interval)) if after >= first else []


def _candidate_name(native_ticks: int) -> str:
    return f"P{native_ticks // 100_000:03d}"


def _emit_phase(enabled: bool, phase: str, iteration: int) -> None:
    if enabled:
        print(json.dumps({
            "event": "training_phase",
            "phase": phase,
            "iteration": iteration,
        }), flush=True)


def _git_state() -> tuple[str, bool]:
    flags = (
        subprocess.CREATE_NO_WINDOW
        if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        creationflags=flags,
        check=False,
    ).stdout.strip() or "unknown"
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        creationflags=flags,
        check=False,
    ).stdout.strip())
    return revision, dirty


def _implementation_digest() -> str:
    digest = hashlib.sha256()
    roots = (
        PROJECT_ROOT / "selfplay_v2",
        PROJECT_ROOT / "training",
        PROJECT_ROOT / "native_core",
        PROJECT_ROOT / "android_probe" / "native",
        PROJECT_ROOT / "android_probe" / "java",
    )
    files = sorted(
        path
        for root in roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".py", ".cpp", ".h", ".java"}
    )
    for path in files:
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def _semantic_digest(config: Mapping[str, Any]) -> str:
    names = (
        "algorithm", "workers", "avds", "workers_per_avd", "max_ticks",
        "environment_ports", "transport", "device", "reward_contract",
        "ppo", "network_schema", "action_schema", "observation_schema",
        "rate_contract", "initialization", "native_tick_hz",
        "decision_frequency_hz", "implementation_digest",
    )
    frozen = {name: config.get(name) for name in names}
    return hashlib.sha256(json.dumps(
        frozen, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def _build_checkpoint(
    *,
    model: ContinuousRatePolicyValueNet,
    trainer: ContinuousRatePPOTrainer,
    iteration: int,
    native_ticks: int,
    agent_steps: int,
    episodes: int,
    next_seed: int,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    initial_digest: str,
    metrics: Mapping[str, Any] | None = None,
    behavior: Mapping[str, Any] | None = None,
    episode_summaries: list[Mapping[str, Any]] | None = None,
    sampling_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": CHECKPOINT_KIND,
        "iteration": int(iteration),
        "native_ticks": int(native_ticks),
        "agent_steps": int(agent_steps),
        "completed_episodes": int(episodes),
        "next_seed": int(next_seed),
        "model": clone_state_dict(model),
        "optimizer": deepcopy(trainer.optimizer.state_dict()),
        "scheduler": None,
        "rng_state": capture_rng_state(),
        "config": dict(config),
        "semantic_digest": _semantic_digest(config),
        "run_manifest": dict(manifest),
        "initial_model_digest": initial_digest,
        "current_model_digest": model_digest(model),
        "metrics": dict(metrics or {}),
        "behavior": dict(behavior or {}),
        "episode_summaries": [dict(value) for value in (episode_summaries or [])],
        "sampling_profile": dict(sampling_profile or {}),
    }


def _restore_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    model: ContinuousRatePolicyValueNet,
    trainer: ContinuousRatePPOTrainer,
    semantic_digest: str,
) -> dict[str, int]:
    if checkpoint.get("kind") != CHECKPOINT_KIND:
        raise RuntimeError("v0.2 checkpoint kind mismatch")
    if str(checkpoint.get("semantic_digest")) != semantic_digest:
        raise RuntimeError("v0.2 frozen training semantics mismatch")
    model.load_state_dict(checkpoint["model"])
    trainer.optimizer.load_state_dict(checkpoint["optimizer"])
    restore_rng_state(dict(checkpoint["rng_state"]))
    return {
        "iteration": int(checkpoint["iteration"]),
        "native_ticks": int(checkpoint["native_ticks"]),
        "agent_steps": int(checkpoint["agent_steps"]),
        "episodes": int(checkpoint["completed_episodes"]),
        "next_seed": int(checkpoint["next_seed"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-native-ticks", type=int, default=5_000_000)
    parser.add_argument("--iterations", type=int, default=1_000_000)
    parser.add_argument("--episodes-per-iteration", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--avds", type=int, default=2)
    parser.add_argument("--workers-per-avd", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--max-ticks", type=int, default=7200)
    parser.add_argument("--base-port", type=int, default=37031)
    parser.add_argument("--direct-base-port", type=int, default=38031)
    parser.add_argument("--transport", choices=("direct", "adb"), default="direct")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--initialization", choices=("scratch", "backbone_only"), default="scratch")
    parser.add_argument("--parent-checkpoint", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--lambda-max", type=float, default=20.0)
    parser.add_argument("--lambda-initial", type=float, default=0.20)
    parser.add_argument("--checkpoint-interval", type=int, default=250_000)
    parser.add_argument("--candidate-interval", type=int, default=500_000)
    parser.add_argument("--run-id")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--skip-worker-start", action="store_true")
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--emit-phase-events", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.smoke:
        args.iterations = 1
        args.target_native_ticks = 128
        args.episodes_per_iteration = 1
        args.workers = args.avds = args.workers_per_avd = 1
        args.max_ticks = 128
        args.run_id = args.run_id or (
            "v2-smoke-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-" + uuid.uuid4().hex[:8]
        )
    if args.episodes_per_iteration is None:
        args.episodes_per_iteration = args.workers
    if args.workers != args.avds * args.workers_per_avd:
        raise ValueError("workers must equal avds * workers-per-avd")
    if args.workers_per_avd > 4 or min(
        args.target_native_ticks, args.iterations, args.workers,
        args.episodes_per_iteration,
    ) < 1:
        raise ValueError("invalid training dimensions")
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
    source_revision, source_dirty = _git_state()
    ppo_config = PPOConfig()
    ports = [
        args.direct_base_port + worker
        if args.transport == "direct"
        else args.base_port
        + (worker // args.workers_per_avd) * 100
        + worker % args.workers_per_avd
        for worker in range(args.workers)
    ]
    model = ContinuousRatePolicyValueNet(
        lambda_max=args.lambda_max,
        lambda_initial=args.lambda_initial,
    ).to(device)
    parent = (
        args.parent_checkpoint.resolve()
        if args.initialization == "backbone_only" else None
    )
    initialization_report = initialize_model(
        model,
        mode=args.initialization,
        parent_checkpoint=parent,
    )
    model.enable_cuda_graph_inference(
        device.type == "cuda" and not args.disable_cuda_graph
    )
    config = {
        "schema_version": SCHEMA_VERSION,
        "algorithm": "continuous_rate_native_recurrent_ppo_v2",
        "initial_target_native_ticks": args.target_native_ticks,
        "iterations": args.iterations,
        "episodes_per_iteration": args.episodes_per_iteration,
        "workers": args.workers,
        "avds": args.avds,
        "workers_per_avd": args.workers_per_avd,
        "seed": args.seed,
        "max_ticks": args.max_ticks,
        "base_port": args.base_port,
        "environment_ports": ports,
        "transport": args.transport,
        "device": str(device),
        "torch": torch.__version__,
        "source_revision": source_revision,
        "source_dirty": source_dirty,
        "implementation_digest": _implementation_digest(),
        "replay_path": str(args.replay.resolve()),
        "replay_digest": hashlib.sha256(args.replay.read_bytes()).hexdigest(),
        "reward": "tower_hp_potential_v1",
        "reward_contract": {
            "schema": PotentialReward.schema_version,
            "terminal": "win=+1,draw=0,loss=-1",
            "gamma": ppo_config.gamma,
            "shaping_scale": 0.20,
            "potential": "normalized_total_crown_tower_hp_fraction_difference",
            "excluded": [
                "elixir", "kills", "river_crossing", "unit_damage",
                "board_value", "card_usage", "position_prior",
            ],
        },
        "ppo": asdict(ppo_config),
        "network_schema": "continuous_rate_recurrent_policy_value_v1",
        "action_schema": "rate_then_card_then_position_v1",
        "observation_schema": "compact_train_v1_unchanged",
        "rate_contract": {
            "parameterization": "lambda_max_times_sigmoid_v1",
            "native_tick_seconds": 0.05,
            "lambda_max": args.lambda_max,
            "lambda_initial": args.lambda_initial,
            "forced_no_play_log_probability": 0.0,
            "forced_no_play_actor_gradient": False,
        },
        "initialization": initialization_report,
        "native_tick_hz": 20,
        "decision_frequency_hz": 20,
        "checkpoint_interval_native_ticks": args.checkpoint_interval,
        "candidate_interval_native_ticks": args.candidate_interval,
        "truth_source": "surface_free_original_libg_15.535.29_x86_64",
        "cuda_graph_inference": (
            device.type == "cuda" and not args.disable_cuda_graph
        ),
    }
    requested_semantics = _semantic_digest(config)
    store = RunStore(args.data_root)
    resume_checkpoint: dict[str, Any] | None = None
    if args.resume:
        resume_path = args.resume.resolve()
        run_root = resume_path.parent
        expected_runs = (args.data_root.resolve() / "runs").resolve()
        while run_root != expected_runs and not (run_root / "manifest.json").is_file():
            run_root = run_root.parent
        if run_root.parent != expected_runs:
            raise RuntimeError("resume checkpoint is outside v0.2 data root")
        if args.run_id and args.run_id != run_root.name:
            raise RuntimeError("run-id does not match resume checkpoint")
        paths, manifest = store.open(run_root.name)
        stored_config = dict(manifest["config"])
        if _semantic_digest(stored_config) != requested_semantics:
            raise RuntimeError("resume command changes frozen v0.2 semantics")
        config = stored_config
        resume_checkpoint = torch.load(
            resume_path, map_location="cpu", weights_only=False
        )
    else:
        paths = store.create(config, run_id=args.run_id)
        manifest = json.loads(
            (paths.root / "manifest.json").read_text(encoding="utf-8-sig")
        )
    events = paths.logs / "events.jsonl"
    _append_jsonl(events, {
        "event": "run_resume" if resume_checkpoint else "run_start",
        "utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "requested_target_native_ticks": args.target_native_ticks,
    })
    trainer = ContinuousRatePPOTrainer(
        model, device=device, config=ppo_config
    )
    initial_state = clone_state_dict(model)
    initial_digest = model_digest(model)
    native_ticks = agent_steps = episodes = starting_iteration = 0
    next_seed = args.seed
    initial_path = paths.evaluations / "candidates" / "P000.pt"
    if resume_checkpoint:
        initial_checkpoint = torch.load(
            initial_path, map_location="cpu", weights_only=False
        )
        initial_state = {
            name: value.detach().cpu().clone()
            for name, value in initial_checkpoint["model"].items()
        }
        initial_digest = str(initial_checkpoint["current_model_digest"])
        restored = _restore_checkpoint(
            resume_checkpoint,
            model=model,
            trainer=trainer,
            semantic_digest=requested_semantics,
        )
        starting_iteration = restored["iteration"]
        native_ticks = restored["native_ticks"]
        agent_steps = restored["agent_steps"]
        episodes = restored["episodes"]
        next_seed = restored["next_seed"]
    else:
        checkpoint = _build_checkpoint(
            model=model,
            trainer=trainer,
            iteration=0,
            native_ticks=0,
            agent_steps=0,
            episodes=0,
            next_seed=next_seed,
            config=config,
            manifest=manifest,
            initial_digest=initial_digest,
        )
        _atomic_torch_save(paths.checkpoints / "initial.pt", checkpoint)
        _atomic_torch_save(paths.checkpoints / "latest.pt", checkpoint)
        _atomic_torch_save(initial_path, checkpoint)

    pool = MultiAvdWorkerPool(
        avds=args.avds,
        workers_per_avd=args.workers_per_avd,
        service_base_port=args.base_port,
        direct_base_port=args.direct_base_port,
    )
    if not args.skip_worker_start:
        ready = pool.ensure_ready(configure_direct=args.transport == "direct")
        _append_jsonl(events, {"event": "workers_ready", "state": ready})
    elif args.transport == "direct":
        pool.configure_direct_ports()
    if ports != pool.environment_ports(args.transport):
        raise RuntimeError("v0.2 environment port contract mismatch")
    envs = [NativeRoyaleEnv(port=port, timeout=30) for port in ports]
    iteration_offset = 0
    process_started = time.perf_counter()
    while (
        iteration_offset < args.iterations
        and native_ticks < args.target_native_ticks
    ):
        iteration_offset += 1
        iteration = starting_iteration + iteration_offset
        ticks_before = native_ticks
        results: list[EpisodeResult] = []
        rpc_total: list[float] = []
        rpc_receive: list[float] = []
        barrier_rows: list[tuple[int, int, int, float, float, float]] = []
        iteration_started = time.perf_counter()
        _emit_phase(args.emit_phase_events, "sampling", iteration)
        pending = args.episodes_per_iteration
        while pending:
            wave_size = min(args.workers, pending)
            seeds = list(range(next_seed, next_seed + wave_size))
            next_seed += wave_size
            pending -= wave_size
            collector = ContinuousRateVectorCollector(
                envs[:wave_size],
                model,
                replay,
                device=device,
                max_ticks=args.max_ticks,
            )
            collected = collector.collect(seeds)
            rpc_total.extend(collector.rpc_latency_samples["total"])
            rpc_receive.extend(collector.rpc_latency_samples["receive"])
            barrier_rows.extend(collector.barrier_rows)
            for result in collected:
                save_episode(paths.trajectories, result, full_debug=args.smoke)
                results.append(result)
                _append_jsonl(events, {
                    "event": "episode_complete", **result.summary()
                })
                steps = sum(
                    len(trajectory.rewards) for trajectory in result.trajectories
                )
                native_ticks += steps // 2
                agent_steps += steps
                episodes += 1
        trajectories = [
            trajectory for result in results for trajectory in result.trajectories
        ]
        sampling_profile: dict[str, float] = {}
        for result in results:
            for key, value in result.profile.items():
                sampling_profile[key] = (
                    sampling_profile.get(key, 0.0) + float(value)
                )
        sampling_profile.update(JsonLineClient.latency_summary(
            rpc_total,
            rpc_receive,
            attempts=sampling_profile.get("rpc_attempts", 0.0),
            failures=sampling_profile.get("rpc_failures", 0.0),
        ))
        sampling_profile.update(summarize_barrier(barrier_rows))
        behavior, histogram = aggregate_timed_behavior(results)
        _atomic_npz(
            paths.evaluations / f"behavior-{iteration:06d}.npz",
            schema_version=np.asarray(1, dtype=np.int32),
            deployment_histogram=histogram,
        )
        _emit_phase(args.emit_phase_events, "learner", iteration)
        update_started = time.perf_counter()
        metrics = trainer.update(trajectories)
        metrics["learner_wall_seconds"] = time.perf_counter() - update_started
        _emit_phase(args.emit_phase_events, "finalize", iteration)
        metrics["iteration_wall_seconds"] = time.perf_counter() - iteration_started
        metrics["environment_steps"] = float(
            sum(len(item.rewards) for item in trajectories) // 2
        )
        metrics["agent_steps"] = float(
            sum(len(item.rewards) for item in trajectories)
        )
        environment_wall = max(
            1e-9,
            metrics["iteration_wall_seconds"] - metrics["learner_wall_seconds"],
        )
        metrics["environment_steps_per_second"] = (
            metrics["environment_steps"] / environment_wall
        )
        metrics["training_steps_per_second"] = (
            metrics["environment_steps"]
            / max(1e-9, metrics["iteration_wall_seconds"])
        )
        metrics["policy_decisions_per_second"] = (
            float(sampling_profile.get("policy_decisions", 0.0))
            / environment_wall
        )
        metrics["episodes_per_hour"] = (
            len(results) * 3600.0
            / max(1e-9, metrics["iteration_wall_seconds"])
        )
        metrics["cumulative_native_ticks"] = float(native_ticks)
        metrics["cumulative_agent_steps"] = float(agent_steps)
        metrics["cumulative_episodes"] = float(episodes)
        metrics["process_wall_seconds"] = time.perf_counter() - process_started
        metrics.update(model_distance(model, initial_state))
        assert_healthy(
            model=model,
            metrics=metrics,
            behavior=behavior,
            sampling_profile=sampling_profile,
            require_normal_terminal=not args.smoke,
        )
        checkpoint = _build_checkpoint(
            model=model,
            trainer=trainer,
            iteration=iteration,
            native_ticks=native_ticks,
            agent_steps=agent_steps,
            episodes=episodes,
            next_seed=next_seed,
            config=config,
            manifest=manifest,
            initial_digest=initial_digest,
            metrics=metrics,
            behavior=behavior,
            episode_summaries=[result.summary() for result in results],
            sampling_profile=sampling_profile,
        )
        _atomic_torch_save(paths.checkpoints / "latest.pt", checkpoint)
        recovery_paths: list[str] = []
        for threshold in _crossed(
            ticks_before, native_ticks, args.checkpoint_interval
        ):
            target = paths.checkpoints / f"recovery-{threshold:09d}.pt"
            _atomic_torch_save(target, checkpoint)
            recovery_paths.append(str(target))
        candidate_paths: list[str] = []
        for threshold in _crossed(
            ticks_before, native_ticks, args.candidate_interval
        ):
            target = paths.evaluations / "candidates" / (
                _candidate_name(threshold) + ".pt"
            )
            _atomic_torch_save(target, checkpoint)
            candidate_paths.append(str(target))
        event = {
            "event": "iteration_complete",
            "iteration": iteration,
            "native_ticks": native_ticks,
            "agent_steps": agent_steps,
            "episodes": episodes,
            "metrics": metrics,
            "behavior": behavior,
            "sampling_profile": sampling_profile,
            "checkpoint": str(paths.checkpoints / "latest.pt"),
            "recovery_checkpoints": recovery_paths,
            "evaluation_candidates": candidate_paths,
        }
        _append_jsonl(events, event)
        RunStore._atomic_json(args.data_root / "latest_run.json", {
            "run": str(paths.root),
            "checkpoint": str(paths.checkpoints / "latest.pt"),
            "native_ticks": native_ticks,
            "agent_steps": agent_steps,
            "episodes": episodes,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        })
        print(json.dumps(event, ensure_ascii=False), flush=True)
    _append_jsonl(events, {
        "event": "run_complete",
        "utc": datetime.now(timezone.utc).isoformat(),
        "native_ticks": native_ticks,
        "agent_steps": agent_steps,
        "episodes": episodes,
        "target_native_ticks": args.target_native_ticks,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
