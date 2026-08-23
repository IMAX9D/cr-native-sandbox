"""Persistent-worker recurrent PPO self-play on the original native core."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
from typing import Any
import uuid

import numpy as np
import torch

from native_core.client import JsonLineClient
from native_core.env import NativeRoyaleEnv
from native_core.worker import MultiAvdWorkerPool

from .model import RecurrentPolicyValueNet
from .ppo import PPOConfig, PPOTrainer
from .rollout import EpisodeResult, save_episode
from .run_contract import (
    aggregate_behavior,
    assert_healthy,
    build_checkpoint,
    clone_state_dict,
    model_digest,
    model_distance,
    restore_checkpoint,
    semantic_digest,
)
from .schema import PotentialReward, RunStore
from .vector_rollout import VectorNativeSelfPlayCollector, summarize_barrier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPLAY = PROJECT_ROOT / "examples" / "eight-card-bootstrap.json"
DEFAULT_DATA_ROOT = Path(r"D:\AI_data\cr-native-core\selfplay-v0.1")


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


def _atomic_barrier_save(
    path: Path,
    rows: list[tuple[int, int, int, float, float, float]],
    waves: list[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(rows, dtype=np.float64).reshape(-1, 6)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema_version=np.asarray(1, dtype=np.int32),
            wave=np.asarray(waves, dtype=np.int32),
            round_index=values[:, 0].astype(np.int64),
            active_workers=values[:, 1].astype(np.int16),
            policy_batch_size=values[:, 2].astype(np.int16),
            fastest_seconds=values[:, 3],
            median_seconds=values[:, 4],
            slowest_seconds=values[:, 5],
        )
    temporary.replace(path)


def _atomic_behavior_save(path: Path, histogram: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema_version=np.asarray(1, dtype=np.int32),
            deployment_histogram=histogram,
        )
    temporary.replace(path)


def _crossed_thresholds(before: int, after: int, interval: int) -> list[int]:
    if interval <= 0 or after <= before:
        return []
    first = (before // interval + 1) * interval
    return list(range(first, after + 1, interval))


def _candidate_name(native_ticks: int) -> str:
    if native_ticks % 100_000:
        return f"T{native_ticks:09d}"
    return f"P{native_ticks // 100_000:03d}"


def _implementation_digest() -> str:
    digest = hashlib.sha256()
    roots = (
        PROJECT_ROOT / "training",
        PROJECT_ROOT / "native_core",
        PROJECT_ROOT / "android_probe" / "native",
        PROJECT_ROOT / "android_probe" / "src",
    )
    files = sorted(
        path
        for root in roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".py", ".java", ".cpp", ".h"}
    )
    for path in files:
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_state() -> tuple[str, bool]:
    flags = (
        subprocess.CREATE_NO_WINDOW
        if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        creationflags=flags, check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        creationflags=flags, check=False,
    )
    return revision.stdout.strip() or "unknown", bool(status.stdout.strip())


def _emit_phase(enabled: bool, phase: str, iteration: int) -> None:
    if enabled:
        print(json.dumps({
            "event": "training_phase",
            "phase": phase,
            "iteration": iteration,
        }), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1000000)
    parser.add_argument("--target-native-ticks", type=int)
    parser.add_argument("--checkpoint-interval", type=int, default=250_000)
    parser.add_argument("--candidate-interval", type=int, default=500_000)
    parser.add_argument("--episodes-per-iteration", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--avds", type=int, default=1)
    parser.add_argument("--workers-per-avd", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-ticks", type=int, default=7200)
    parser.add_argument("--base-port", type=int, default=37031)
    parser.add_argument("--direct-base-port", type=int, default=38031)
    parser.add_argument("--transport", choices=("direct", "adb"), default="direct")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument(
        "--reward",
        choices=("terminal", "potential", "tower_hp_potential_v1"),
        default="tower_hp_potential_v1",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-id")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--skip-worker-start", action="store_true")
    parser.add_argument("--profile-native", action="store_true")
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--emit-phase-events", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.smoke:
        args.iterations = 1
        args.episodes_per_iteration = 1
        args.workers = 1
        args.avds = 1
        args.workers_per_avd = 1
        args.max_ticks = 128
        args.target_native_ticks = None
        args.run_id = args.run_id or "smoke-" + datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        ) + "-" + uuid.uuid4().hex[:8]
    if args.episodes_per_iteration is None:
        args.episodes_per_iteration = args.workers
    if args.workers_per_avd is None:
        if args.workers % args.avds:
            raise ValueError("workers must divide evenly across avds")
        args.workers_per_avd = args.workers // args.avds
    if min(
        args.iterations,
        args.episodes_per_iteration,
        args.workers,
        args.avds,
        args.workers_per_avd,
    ) < 1:
        raise ValueError("iterations, episodes-per-iteration and workers must be positive")
    if args.target_native_ticks is not None and args.target_native_ticks < 1:
        raise ValueError("target-native-ticks must be positive")
    if args.checkpoint_interval < 1 or args.candidate_interval < 1:
        raise ValueError("checkpoint and candidate intervals must be positive")
    if args.workers != args.avds * args.workers_per_avd:
        raise ValueError("workers must equal avds * workers-per-avd")
    if args.workers_per_avd > 4:
        raise ValueError("each AVD may host at most 4 workers")
    if args.workers > 32:
        raise ValueError("workers must be in 1..32")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    if args.reward == "potential":
        args.reward = "tower_hp_potential_v1"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    replay = json.loads(args.replay.read_text(encoding="utf-8-sig"))
    source_revision, source_dirty = _git_state()
    implementation_digest = _implementation_digest()
    replay_digest = hashlib.sha256(args.replay.read_bytes()).hexdigest()
    ppo_config = PPOConfig()
    environment_ports = [
        (
            args.direct_base_port + worker
            if args.transport == "direct"
            else args.base_port
            + (worker // args.workers_per_avd) * 100
            + worker % args.workers_per_avd
        )
        for worker in range(args.workers)
    ]
    config = {
        "schema_version": 2,
        "algorithm": "persistent_native_recurrent_ppo",
        "iterations": args.iterations,
        "initial_target_native_ticks": args.target_native_ticks,
        "checkpoint_interval_native_ticks": args.checkpoint_interval,
        "candidate_interval_native_ticks": args.candidate_interval,
        "episodes_per_iteration": args.episodes_per_iteration,
        "workers": args.workers,
        "avds": args.avds,
        "workers_per_avd": args.workers_per_avd,
        "seed": args.seed,
        "max_ticks": args.max_ticks,
        "base_port": args.base_port,
        "environment_ports": environment_ports,
        "transport": args.transport,
        "device": str(device),
        "torch": torch.__version__,
        "source_revision": source_revision,
        "source_dirty": source_dirty,
        "implementation_digest": implementation_digest,
        "replay_path": str(args.replay.resolve()),
        "replay_digest": replay_digest,
        "reward": args.reward,
        "reward_contract": {
            "schema": PotentialReward.schema_version,
            "terminal": "win=+1,draw=0,loss=-1",
            "gamma": ppo_config.gamma,
            "shaping_scale": 0.20,
            "potential": "normalized_total_crown_tower_hp_fraction_difference",
            "excluded": [
                "elixir", "kills", "river_crossing", "unit_damage", "board_value"
            ],
            "terminal_state_potential": 0.0,
        },
        "ppo": asdict(ppo_config),
        "recurrent_training": {
            "hidden_reset": "zero_each_episode",
            "side_state": "independent",
            "burn_in": ppo_config.burn_in,
            "train_length": ppo_config.train_length,
            "bptt": True,
        },
        "episode_reset": "native_battle_game_state_4_to_4_in_process",
        "truth_source": "surface_free_original_libg_15.535.29_x86_64",
        "action_legality": "native_validate_deployment_18x32",
        "observation_schema": "compact_train_v1",
        "network_schema": "recurrent_policy_value_v1",
        "native_tick_hz": 20,
        "decision_frequency_hz": 20,
        "profile_native": args.profile_native,
        "cuda_graph_inference": (
            args.device.startswith("cuda") and not args.disable_cuda_graph
        ),
    }
    requested_semantics = semantic_digest(config)
    store = RunStore(args.data_root)
    resume_checkpoint: dict[str, Any] | None = None
    if args.resume:
        resume_path = args.resume.resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        expected_runs_root = (args.data_root.resolve() / "runs").resolve()
        run_root = resume_path.parent
        while run_root != expected_runs_root and not (
            run_root / "manifest.json"
        ).is_file():
            run_root = run_root.parent
        if run_root.parent != expected_runs_root:
            raise RuntimeError("resume checkpoint is outside the selected data root")
        if args.run_id is not None and args.run_id != run_root.name:
            raise RuntimeError("run-id does not match the resume checkpoint run")
        paths, run_manifest = store.open(run_root.name)
        stored_config = dict(run_manifest["config"])
        if semantic_digest(stored_config) != requested_semantics:
            raise RuntimeError("resume command changes frozen training semantics")
        config = stored_config
        resume_checkpoint = torch.load(
            resume_path, map_location="cpu", weights_only=False
        )
    else:
        paths = store.create(config, run_id=args.run_id)
        run_manifest = json.loads(
            (paths.root / "manifest.json").read_text(encoding="utf-8-sig")
        )

    events = paths.logs / "events.jsonl"
    _append_jsonl(events, {
        "event": "run_resume" if resume_checkpoint is not None else "run_start",
        "utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "requested_target_native_ticks": args.target_native_ticks,
        "resume": str(args.resume.resolve()) if args.resume else None,
    })

    model = RecurrentPolicyValueNet().to(device)
    model.enable_cuda_graph_inference(
        device.type == "cuda" and not args.disable_cuda_graph
    )
    trainer = PPOTrainer(model, device=device, config=ppo_config)
    initial_state = clone_state_dict(model)
    initial_digest = model_digest(model)

    pool = MultiAvdWorkerPool(
        avds=args.avds,
        workers_per_avd=args.workers_per_avd,
        service_base_port=args.base_port,
        direct_base_port=args.direct_base_port,
    )
    if not args.skip_worker_start:
        worker_state = pool.ensure_ready(
            configure_direct=args.transport == "direct"
        )
        _append_jsonl(events, {"event": "workers_ready", "state": worker_state})
    elif args.transport == "direct":
        direct_state = pool.configure_direct_ports()
        _append_jsonl(events, {"event": "direct_transport_ready", "state": direct_state})

    native_ticks = 0
    agent_steps = 0
    completed_episodes = 0
    starting_iteration = 0
    next_seed = args.seed
    low_entropy_iterations = 0
    initial_path = paths.evaluations / "candidates" / "P000.pt"
    if resume_checkpoint is not None:
        if not initial_path.is_file():
            raise RuntimeError("formal run is missing the P000 initial candidate")
        initial_checkpoint = torch.load(
            initial_path, map_location="cpu", weights_only=False
        )
        initial_state = {
            name: value.detach().cpu().clone()
            for name, value in initial_checkpoint["model"].items()
        }
        initial_digest = str(initial_checkpoint["current_model_digest"])
        restored = restore_checkpoint(
            resume_checkpoint,
            model=model,
            optimizer=trainer.optimizer,
            expected_semantic_digest=requested_semantics,
        )
        native_ticks = restored["native_ticks"]
        agent_steps = restored["agent_steps"]
        completed_episodes = restored["completed_episodes"]
        starting_iteration = restored["iteration"]
        next_seed = restored["next_seed"]
        low_entropy_iterations = int(
            resume_checkpoint.get("metrics", {}).get(
                "low_entropy_consecutive_iterations", 0
            )
        )
        if str(resume_checkpoint.get("initial_model_digest")) != initial_digest:
            raise RuntimeError("resume checkpoint refers to a different P000 model")
    else:
        initial_checkpoint = build_checkpoint(
            model=model,
            optimizer=trainer.optimizer,
            iteration=0,
            native_ticks=0,
            agent_steps=0,
            completed_episodes=0,
            next_seed=next_seed,
            config=config,
            run_manifest=run_manifest,
            initial_model_digest=initial_digest,
        )
        _atomic_torch_save(paths.checkpoints / "initial.pt", initial_checkpoint)
        _atomic_torch_save(initial_path, initial_checkpoint)
        _atomic_torch_save(paths.checkpoints / "latest.pt", initial_checkpoint)

    if environment_ports != pool.environment_ports(args.transport):
        raise RuntimeError("manifest environment ports do not match Worker pool")
    envs = [
        NativeRoyaleEnv(
            port=port,
            timeout=30,
            profile_native=args.profile_native,
        )
        for port in environment_ports
    ]
    offset = 0
    process_started = time.perf_counter()
    while offset < args.iterations and (
        args.target_native_ticks is None or native_ticks < args.target_native_ticks
    ):
        offset += 1
        iteration = starting_iteration + offset
        ticks_before_iteration = native_ticks
        pending = list(range(args.episodes_per_iteration))
        results: list[EpisodeResult] = []
        rpc_total_latency: list[float] = []
        rpc_receive_latency: list[float] = []
        barrier_rows: list[tuple[int, int, int, float, float, float]] = []
        barrier_waves: list[int] = []
        wave_index = 0
        iteration_started = time.perf_counter()
        _emit_phase(args.emit_phase_events, "sampling", iteration)
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
            barrier_rows.extend(collector.barrier_rows)
            barrier_waves.extend([wave_index] * len(collector.barrier_rows))
            wave_index += 1
            for result in collected:
                save_episode(paths.trajectories, result, full_debug=args.smoke)
                results.append(result)
                _append_jsonl(events, {"event": "episode_complete", **result.summary()})
                native_ticks += sum(len(item.rewards) for item in result.trajectories) // 2
                agent_steps += sum(len(item.rewards) for item in result.trajectories)
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
        sampling_profile.update(summarize_barrier(barrier_rows))
        barrier_path = paths.evaluations / f"barrier-{iteration:06d}.npz"
        _atomic_barrier_save(barrier_path, barrier_rows, barrier_waves)
        behavior, deployment_histogram = aggregate_behavior(results)
        behavior_path = paths.evaluations / f"behavior-{iteration:06d}.npz"
        _atomic_behavior_save(behavior_path, deployment_histogram)
        _emit_phase(args.emit_phase_events, "learner", iteration)
        update_started = time.perf_counter()
        metrics = trainer.update(trajectories)
        metrics["learner_wall_seconds"] = time.perf_counter() - update_started
        _emit_phase(args.emit_phase_events, "finalize", iteration)
        metrics["iteration_wall_seconds"] = time.perf_counter() - iteration_started
        metrics["environment_steps"] = float(
            sum(len(item.rewards) for item in trajectories) // 2
        )
        metrics["agent_steps"] = float(sum(len(item.rewards) for item in trajectories))
        metrics["environment_steps_per_second"] = (
            metrics["environment_steps"]
            / max(1e-9, metrics["iteration_wall_seconds"] - metrics["learner_wall_seconds"])
        )
        metrics["training_steps_per_second"] = (
            metrics["environment_steps"]
            / max(1e-9, metrics["iteration_wall_seconds"])
        )
        metrics["policy_decisions_per_second"] = (
            float(sampling_profile.get("policy_decisions", 0.0))
            / max(
                1e-9,
                metrics["iteration_wall_seconds"] - metrics["learner_wall_seconds"],
            )
        )
        metrics["episodes_per_hour"] = (
            len(results) * 3600.0 / max(1e-9, metrics["iteration_wall_seconds"])
        )
        metrics["cumulative_native_ticks"] = float(native_ticks)
        metrics["cumulative_agent_steps"] = float(agent_steps)
        metrics["cumulative_episodes"] = float(completed_episodes)
        metrics["process_wall_seconds"] = time.perf_counter() - process_started
        metrics.update(model_distance(model, initial_state))
        if float(metrics.get("entropy", 0.0)) < 0.05:
            low_entropy_iterations += 1
        else:
            low_entropy_iterations = 0
        metrics["low_entropy_consecutive_iterations"] = float(
            low_entropy_iterations
        )
        try:
            assert_healthy(
                model=model,
                metrics=metrics,
                behavior=behavior,
                sampling_profile=sampling_profile,
                require_normal_terminal=not args.smoke,
            )
            if low_entropy_iterations >= 5:
                raise RuntimeError("policy entropy collapsed for five iterations")
        except RuntimeError as error:
            failure_checkpoint = build_checkpoint(
                model=model,
                optimizer=trainer.optimizer,
                iteration=iteration,
                native_ticks=native_ticks,
                agent_steps=agent_steps,
                completed_episodes=completed_episodes,
                next_seed=next_seed,
                config=config,
                run_manifest=run_manifest,
                initial_model_digest=initial_digest,
                metrics=metrics,
                behavior=behavior,
                episode_summaries=(item.summary() for item in results),
                sampling_profile=sampling_profile,
                barrier_profile=str(barrier_path),
            )
            failed_path = paths.checkpoints / f"failed-{iteration:06d}.pt"
            _atomic_torch_save(failed_path, failure_checkpoint)
            _append_jsonl(events, {
                "event": "run_aborted",
                "utc": datetime.now(timezone.utc).isoformat(),
                "iteration": iteration,
                "native_ticks": native_ticks,
                "reason": str(error),
                "checkpoint": str(failed_path),
            })
            raise

        checkpoint = build_checkpoint(
            model=model,
            optimizer=trainer.optimizer,
            iteration=iteration,
            native_ticks=native_ticks,
            agent_steps=agent_steps,
            completed_episodes=completed_episodes,
            next_seed=next_seed,
            config=config,
            run_manifest=run_manifest,
            initial_model_digest=initial_digest,
            metrics=metrics,
            behavior=behavior,
            episode_summaries=(item.summary() for item in results),
            sampling_profile=sampling_profile,
            barrier_profile=str(barrier_path),
        )
        _atomic_torch_save(paths.checkpoints / "latest.pt", checkpoint)
        recovery_paths: list[str] = []
        for threshold in _crossed_thresholds(
            ticks_before_iteration, native_ticks, args.checkpoint_interval
        ):
            target = paths.checkpoints / f"recovery-{threshold:09d}.pt"
            _atomic_torch_save(target, checkpoint)
            recovery_paths.append(str(target))
        candidate_paths: list[str] = []
        for threshold in _crossed_thresholds(
            ticks_before_iteration, native_ticks, args.candidate_interval
        ):
            target = (
                paths.evaluations / "candidates"
                / f"{_candidate_name(threshold)}.pt"
            )
            _atomic_torch_save(target, checkpoint)
            candidate_paths.append(str(target))
        if args.smoke:
            loaded = torch.load(
                paths.checkpoints / "latest.pt", map_location="cpu", weights_only=False
            )
            if not (
                loaded.get("kind") == "native_eight_card_recurrent_ppo_checkpoint"
                and int(loaded.get("iteration", -1)) == iteration
                and int(loaded.get("completed_episodes", 0)) >= 1
                and int(loaded.get("agent_steps", 0)) >= 2
                and loaded.get("model")
                and loaded.get("optimizer")
                and loaded.get("rng_state")
            ):
                raise RuntimeError("smoke checkpoint reload/contract validation failed")
        event = {
            "event": "iteration_complete", "iteration": iteration,
            "native_ticks": native_ticks, "episodes": completed_episodes,
            "agent_steps": agent_steps,
            "metrics": metrics,
            "behavior": behavior,
            "checkpoint": str(paths.checkpoints / "latest.pt"),
            "recovery_checkpoints": recovery_paths,
            "evaluation_candidates": candidate_paths,
            "sampling_profile": sampling_profile,
            "barrier_profile": str(barrier_path),
            "behavior_profile": str(behavior_path),
        }
        _append_jsonl(events, event)
        RunStore._atomic_json(args.data_root / "latest_run.json", {
            "run": str(paths.root), "checkpoint": str(paths.checkpoints / "latest.pt"),
            "native_ticks": native_ticks,
            "agent_steps": agent_steps,
            "episodes": completed_episodes,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        })
        print(json.dumps(event, ensure_ascii=False), flush=True)
    _append_jsonl(events, {
        "event": "run_complete",
        "utc": datetime.now(timezone.utc).isoformat(),
        "native_ticks": native_ticks,
        "agent_steps": agent_steps,
        "episodes": completed_episodes,
        "target_native_ticks": args.target_native_ticks,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
