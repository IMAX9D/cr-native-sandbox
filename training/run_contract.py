"""Frozen v0.1 run-state, behavior metrics and fail-closed health checks."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import random
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from .rollout import EpisodeResult
from .schema import CARD_IDS, CARD_INDEX


CHECKPOINT_KIND = "native_eight_card_recurrent_ppo_checkpoint"
CHECKPOINT_SCHEMA_VERSION = 2


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def model_digest(model: torch.nn.Module) -> str:
    return state_dict_digest(model.state_dict())


def state_dict_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def semantic_digest(config: Mapping[str, Any]) -> str:
    """Digest only fields that may not change within one formal run."""
    names = (
        "algorithm", "workers", "avds", "workers_per_avd", "max_ticks",
        "environment_ports", "transport", "device", "reward",
        "reward_contract", "ppo", "episode_reset", "truth_source",
        "action_legality", "observation_schema", "network_schema",
        "native_tick_hz", "decision_frequency_hz", "cuda_graph_inference",
        "implementation_digest", "replay_digest",
    )
    frozen = {name: config.get(name) for name in names}
    encoded = json.dumps(
        frozen, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    native_ticks: int,
    agent_steps: int,
    completed_episodes: int,
    next_seed: int,
    config: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    initial_model_digest: str,
    metrics: Mapping[str, Any] | None = None,
    behavior: Mapping[str, Any] | None = None,
    episode_summaries: Iterable[Mapping[str, Any]] = (),
    sampling_profile: Mapping[str, Any] | None = None,
    barrier_profile: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "kind": CHECKPOINT_KIND,
        "iteration": int(iteration),
        "native_ticks": int(native_ticks),
        "agent_steps": int(agent_steps),
        "completed_episodes": int(completed_episodes),
        "next_seed": int(next_seed),
        "model": clone_state_dict(model),
        "optimizer": deepcopy(optimizer.state_dict()),
        "scheduler": None,
        "rng_state": capture_rng_state(),
        "config": dict(config),
        "semantic_digest": semantic_digest(config),
        "run_manifest": dict(run_manifest),
        "initial_model_digest": initial_model_digest,
        "current_model_digest": model_digest(model),
        "metrics": dict(metrics or {}),
        "behavior": dict(behavior or {}),
        "episode_summaries": [dict(value) for value in episode_summaries],
        "sampling_profile": dict(sampling_profile or {}),
        "barrier_profile": barrier_profile,
    }


def restore_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_semantic_digest: str | None = None,
) -> dict[str, int]:
    required = {
        "model", "optimizer", "rng_state", "native_ticks", "agent_steps",
        "completed_episodes", "next_seed", "iteration", "semantic_digest",
    }
    missing = sorted(required.difference(checkpoint))
    if checkpoint.get("kind") != CHECKPOINT_KIND or missing:
        raise RuntimeError(f"invalid checkpoint contract; missing={missing}")
    actual_semantics = str(checkpoint["semantic_digest"])
    if (
        expected_semantic_digest is not None
        and actual_semantics != expected_semantic_digest
    ):
        raise RuntimeError("checkpoint training semantics do not match this run")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    restore_rng_state(dict(checkpoint["rng_state"]))
    return {
        "iteration": int(checkpoint["iteration"]),
        "native_ticks": int(checkpoint["native_ticks"]),
        "agent_steps": int(checkpoint["agent_steps"]),
        "completed_episodes": int(checkpoint["completed_episodes"]),
        "next_seed": int(checkpoint["next_seed"]),
    }


def model_distance(
    model: torch.nn.Module,
    reference: dict[str, torch.Tensor],
) -> dict[str, float]:
    squared_delta = 0.0
    squared_reference = 0.0
    maximum_delta = 0.0
    with torch.no_grad():
        for name, value in model.state_dict().items():
            current = value.detach().float().cpu()
            baseline = reference[name].detach().float().cpu()
            delta = current - baseline
            squared_delta += float(torch.sum(delta * delta))
            squared_reference += float(torch.sum(baseline * baseline))
            maximum_delta = max(maximum_delta, float(torch.max(torch.abs(delta))))
    l2 = squared_delta ** 0.5
    reference_l2 = squared_reference ** 0.5
    return {
        "parameter_delta_l2": l2,
        "parameter_relative_delta_l2": l2 / max(reference_l2, 1e-12),
        "parameter_delta_abs_max": maximum_delta,
    }


def aggregate_behavior(
    results: Iterable[EpisodeResult],
) -> tuple[dict[str, Any], np.ndarray]:
    episodes = list(results)
    histogram = np.zeros((len(CARD_IDS), 32, 18), dtype=np.int64)
    decisions = waits = elixir_samples = leak_steps = 0
    elixir_weighted = 0.0
    rejection_count = 0
    attempts: dict[str, int] = {}
    plays: dict[str, int] = {}
    match_ticks: list[int] = []
    crown_differences: list[int] = []
    tower_hp_differences: list[int] = []
    damage = [0, 0]
    draws = 0
    terminated = 0
    truncated = 0
    for result in episodes:
        behavior = result.behavior
        decisions += int(behavior.get("decision_count", 0))
        waits += int(behavior.get("wait_count", 0))
        samples = int(behavior.get("elixir_sample_count", 0))
        elixir_samples += samples
        elixir_weighted += float(behavior.get("average_elixir", 0.0)) * samples
        leak_steps += int(behavior.get("elixir_leak_steps", 0))
        rejection_count += int(behavior.get("native_rejection_count", 0))
        for key, value in behavior.get("card_attempts", {}).items():
            attempts[key] = attempts.get(key, 0) + int(value)
        for key, value in behavior.get("card_plays", {}).items():
            plays[key] = plays.get(key, 0) + int(value)
        for card_id, position in behavior.get("deployment_positions", []):
            card_index = CARD_INDEX.get(int(card_id))
            if card_index is None:
                continue
            row, column = divmod(int(position), 18)
            if 0 <= row < 32 and 0 <= column < 18:
                histogram[card_index, row, column] += 1
        match_ticks.append(int(behavior.get("match_ticks", result.tick)))
        crown_differences.append(int(behavior.get("crown_difference_side0", 0)))
        tower_hp_differences.append(
            int(behavior.get("tower_hp_difference_side0", 0))
        )
        inflicted = behavior.get("tower_damage_inflicted", [0, 0])
        damage[0] += int(inflicted[0])
        damage[1] += int(inflicted[1])
        draws += result.winner is None
        terminated += bool(result.terminated)
        truncated += bool(result.truncated)
    episode_count = len(episodes)
    action_attempts = sum(attempts.values())
    return {
        "episodes": episode_count,
        "terminated": terminated,
        "truncated": truncated,
        "episode_failure_rate": (
            (episode_count - terminated) / episode_count
            if episode_count else 1.0
        ),
        "average_episode_ticks": float(np.mean(match_ticks)) if match_ticks else 0.0,
        "draw_rate": draws / episode_count if episode_count else 0.0,
        "decision_count": decisions,
        "wait_count": waits,
        "wait_ratio": waits / decisions if decisions else 0.0,
        "card_attempts": attempts,
        "card_plays": plays,
        "card_usage_rate": {
            str(card_id): plays.get(str(card_id), 0) / max(1, sum(plays.values()))
            for card_id in CARD_IDS
        },
        "average_elixir": elixir_weighted / elixir_samples if elixir_samples else 0.0,
        "elixir_leak_ratio": leak_steps / elixir_samples if elixir_samples else 0.0,
        "native_action_rejections": rejection_count,
        "native_action_rejection_rate": (
            rejection_count / action_attempts if action_attempts else 0.0
        ),
        "average_crown_difference_side0": (
            float(np.mean(crown_differences)) if crown_differences else 0.0
        ),
        "average_abs_crown_difference": (
            float(np.mean(np.abs(crown_differences))) if crown_differences else 0.0
        ),
        "average_tower_hp_difference_side0": (
            float(np.mean(tower_hp_differences)) if tower_hp_differences else 0.0
        ),
        "average_tower_damage_by_side": [
            value / episode_count if episode_count else 0.0 for value in damage
        ],
    }, histogram


def assert_healthy(
    *,
    model: torch.nn.Module,
    metrics: dict[str, float],
    behavior: dict[str, Any],
    sampling_profile: dict[str, float],
    require_normal_terminal: bool = True,
) -> None:
    numeric = [float(value) for value in metrics.values()]
    if not all(np.isfinite(numeric)):
        raise RuntimeError("non-finite PPO metric")
    for name, value in model.state_dict().items():
        if not torch.isfinite(value).all():
            raise RuntimeError(f"non-finite model tensor: {name}")
    if float(metrics.get("value_abs_max", 0.0)) > 10.0:
        raise RuntimeError("critic value magnitude exceeded safety bound")
    if float(metrics.get("gradient_norm", 0.0)) > 100.0:
        raise RuntimeError("gradient norm exceeded safety bound")
    if (
        require_normal_terminal
        and float(behavior.get("episode_failure_rate", 0.0)) > 0.0
    ):
        raise RuntimeError("episode failed to reach a normal terminal")
    if float(sampling_profile.get("rpc_failure_rate", 0.0)) > 0.0:
        raise RuntimeError("RPC failure detected")
    if float(behavior.get("native_action_rejection_rate", 0.0)) > 0.01:
        raise RuntimeError("native action rejection rate exceeded 1%")
