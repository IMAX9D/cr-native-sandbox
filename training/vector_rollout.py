"""Cross-Worker synchronous collector with one global policy batch per Tick."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import time
from typing import Any, Mapping

import numpy as np
import torch

from native_core.client import JsonLineClient
from native_core.env import NativeRoyaleEnv

from .model import RecurrentPolicyValueNet, SampledAction
from .rollout import AgentTrajectory, EpisodeResult
from .schema import (
    ActionMaskCache, ObservationEncoder, PotentialReward, build_action_masks,
)


@dataclass
class _Cursor:
    env: NativeRoyaleEnv
    seed: int
    state: dict[str, Any]
    trajectories: tuple[AgentTrajectory, AgentTrajectory]
    hidden: dict[int, tuple[torch.Tensor, torch.Tensor]]
    native_masks: dict[tuple[int, int], list[str]] = field(default_factory=dict)
    mask_cache: ActionMaskCache = field(default_factory=ActionMaskCache)
    public_actions: dict[int, dict[str, int] | None] = field(
        default_factory=lambda: {0: None, 1: None}
    )
    action_log: list[dict[str, Any]] = field(default_factory=list)
    action_count: int = 0
    started: float = field(default_factory=time.perf_counter)
    last_hash: str | None = None
    episode: dict[str, Any] | None = None
    done: bool = False
    profile: dict[str, float] = field(default_factory=dict)

    def record(self, name: str, started_at: float) -> None:
        self.profile[name] = self.profile.get(name, 0.0) + (
            time.perf_counter() - started_at
        )


class VectorNativeSelfPlayCollector:
    """Synchronize N independent libg battles around one global inference."""

    def __init__(
        self,
        envs: list[NativeRoyaleEnv],
        policy: RecurrentPolicyValueNet,
        replay: Mapping[str, Any],
        *,
        device: torch.device,
        reward_mode: str = "terminal",
        max_ticks: int = 7200,
    ) -> None:
        if not envs:
            raise ValueError("vector collector requires at least one environment")
        if reward_mode not in ("terminal", "potential"):
            raise ValueError("reward_mode must be terminal or potential")
        self.envs = envs
        self.policy = policy
        self.replay = json.loads(json.dumps(replay))
        self.device = device
        self.reward_mode = reward_mode
        self.max_ticks = max_ticks
        self.encoder = ObservationEncoder()
        self.potential_reward = PotentialReward(gamma=0.99995)
        self.vector_profile: dict[str, float] = {}
        self._cuda_graph_stats_before = dict(policy.cuda_graph_stats)
        self.rpc_latency_samples: dict[str, list[float]] = {
            "total": [],
            "receive": [],
        }

    def _record_vector(self, name: str, started_at: float) -> None:
        self.vector_profile[name] = self.vector_profile.get(name, 0.0) + (
            time.perf_counter() - started_at
        )

    @staticmethod
    def _canonical_positions(mask: np.ndarray, side: int) -> np.ndarray:
        value = mask.reshape(4, 32, 18)
        if side == 1:
            value = value[:, ::-1, ::-1]
        return np.ascontiguousarray(value.reshape(4, 32 * 18))

    @staticmethod
    def _absolute_cell(position: int, side: int) -> tuple[int, int]:
        row, column = divmod(position, 18)
        if side == 1:
            row, column = 31 - row, 17 - column
        return column * 1000 + 500, row * 1000 + 500

    @staticmethod
    def _timed_reset(
        env: NativeRoyaleEnv, replay: Mapping[str, Any]
    ) -> tuple[dict[str, Any], float]:
        started = time.perf_counter()
        return env.reset(replay, warmup_steps=100), time.perf_counter() - started

    @staticmethod
    def _timed_transition(
        env: NativeRoyaleEnv, actions: list[Mapping[str, int]]
    ) -> tuple[dict[str, Any], float]:
        started = time.perf_counter()
        result = env.joint_training_transition(actions, steps=1)
        return result, time.perf_counter() - started

    def _prepare_masks(self, cursor: _Cursor) -> None:
        for player in cursor.state.get("players", []):
            side = int(player["side"])
            for raw_index in player["hand_deck_indices"]:
                deck_index = int(raw_index)
                if deck_index < 0 or (side, deck_index) in cursor.native_masks:
                    continue
                result = cursor.env.probe_grid(side=side, deck_index=deck_index)
                cursor.native_masks[(side, deck_index)] = [
                    str(row) for row in result["rows"]
                ]

    def _start(self, seeds: list[int]) -> list[_Cursor]:
        if len(seeds) != len(self.envs):
            raise ValueError("one seed is required per vector environment")
        replays = []
        for seed in seeds:
            value = json.loads(json.dumps(self.replay))
            value["rndSeed"] = seed
            replays.append(value)
        for env in self.envs:
            env.reset_rpc_profile()
        with ThreadPoolExecutor(max_workers=len(self.envs)) as executor:
            futures = [
                executor.submit(self._timed_reset, env, replay)
                for env, replay in zip(self.envs, replays, strict=True)
            ]
            reset_results = [future.result() for future in futures]
        cursors = []
        for env, seed, (state, reset_seconds) in zip(
            self.envs, seeds, reset_results, strict=True
        ):
            cursors.append(_Cursor(
                env=env,
                seed=seed,
                state=state,
                trajectories=(
                    AgentTrajectory(side=0, seed=seed),
                    AgentTrajectory(side=1, seed=seed),
                ),
                hidden={
                    side: self.policy.initial_hidden(1, device=self.device)
                    for side in (0, 1)
                },
                last_hash=state.get("state_hash"),
                profile={"reset_seconds": reset_seconds},
            ))
        return cursors

    def collect(self, seeds: list[int]) -> list[EpisodeResult]:
        cursors = self._start(seeds)
        self.policy.eval()
        with ThreadPoolExecutor(max_workers=len(cursors)) as executor:
            while True:
                active = [
                    cursor for cursor in cursors
                    if not cursor.done and int(cursor.state["tick"]) < self.max_ticks
                ]
                if not active:
                    break
                round_started = time.perf_counter()
                for cursor in active:
                    cursor.profile["decision_steps"] = (
                        cursor.profile.get("decision_steps", 0.0) + 1.0
                    )
                    started = time.perf_counter()
                    self._prepare_masks(cursor)
                    cursor.record("mask_probe_seconds", started)

                encoded_rows: list[tuple[np.ndarray, ...]] = []
                row_owners: list[tuple[_Cursor, int, list[int]]] = []
                for cursor in active:
                    for side in (0, 1):
                        started = time.perf_counter()
                        grid, scalars = self.encoder.encode(
                            cursor.state,
                            side=side,
                            public_actions=cursor.public_actions,
                        )
                        privileged = self.encoder.privileged(
                            cursor.state, side=side
                        )
                        cursor.record("encoding_seconds", started)
                        started = time.perf_counter()
                        card_mask, position_masks, hand = build_action_masks(
                            cursor.state,
                            side=side,
                            native_masks=cursor.native_masks,
                            decks=cursor.env.decks,
                            cache=cursor.mask_cache,
                        )
                        position_masks = self._canonical_positions(
                            position_masks, side
                        )
                        cursor.record("mask_build_seconds", started)
                        h_before = cursor.hidden[side]
                        encoded_rows.append((
                            grid, scalars, privileged, card_mask, position_masks,
                            h_before[0][0, 0].detach().cpu().numpy().copy(),
                            h_before[1][0, 0].detach().cpu().numpy().copy(),
                        ))
                        row_owners.append((cursor, side, hand))

                inference_started = time.perf_counter()
                hidden_batch = (
                    torch.cat(
                        [owner[0].hidden[owner[1]][0] for owner in row_owners],
                        dim=1,
                    ),
                    torch.cat(
                        [owner[0].hidden[owner[1]][1] for owner in row_owners],
                        dim=1,
                    ),
                )
                sampled = self.policy.sample_batch(
                    torch.from_numpy(np.stack([row[0] for row in encoded_rows])).to(self.device),
                    torch.from_numpy(np.stack([row[1] for row in encoded_rows])).to(self.device),
                    torch.from_numpy(np.stack([row[2] for row in encoded_rows])).to(self.device),
                    torch.from_numpy(np.stack([row[3] for row in encoded_rows])).to(self.device),
                    torch.from_numpy(np.stack([row[4] for row in encoded_rows])).to(self.device),
                    hidden_batch,
                )
                self._record_vector("vector_inference_seconds", inference_started)
                self.vector_profile["policy_decisions"] = (
                    self.vector_profile.get("policy_decisions", 0.0)
                    + float(len(sampled))
                )

                per_cursor: dict[int, dict[str, Any]] = {
                    id(cursor): {
                        "samples": {}, "encoded": {}, "chosen": [],
                        "next_public": {0: None, 1: None},
                    }
                    for cursor in active
                }
                for sample, encoded, (cursor, side, hand) in zip(
                    sampled, encoded_rows, row_owners, strict=True
                ):
                    cursor.hidden[side] = sample.hidden
                    values = per_cursor[id(cursor)]
                    values["samples"][side] = sample
                    values["encoded"][side] = encoded
                    if sample.card > 0:
                        deck_index = hand[sample.card - 1]
                        card_id = int(cursor.env.decks[side][deck_index]["card_id"])
                        x, y = self._absolute_cell(sample.position, side)
                        values["chosen"].append({
                            "side": side, "deck_index": deck_index, "x": x, "y": y,
                        })
                        values["next_public"][side] = {
                            "card_id": card_id, "x": x, "y": y,
                        }

                transition_wall_started = time.perf_counter()
                futures = {
                    id(cursor): executor.submit(
                        self._timed_transition,
                        cursor.env,
                        per_cursor[id(cursor)]["chosen"],
                    )
                    for cursor in active
                }
                transitions = {
                    key: future.result() for key, future in futures.items()
                }
                self._record_vector(
                    "vector_transition_wall_seconds", transition_wall_started
                )

                for cursor in active:
                    values = per_cursor[id(cursor)]
                    transition, transition_seconds = transitions[id(cursor)]
                    cursor.profile["transition_seconds"] = (
                        cursor.profile.get("transition_seconds", 0.0)
                        + transition_seconds
                    )
                    native_timing = transition.get("timing_v1")
                    if isinstance(native_timing, Mapping):
                        for timing_name, nanoseconds in native_timing.items():
                            profile_name = f"host_{timing_name[:-3]}_seconds"
                            cursor.profile[profile_name] = (
                                cursor.profile.get(profile_name, 0.0)
                                + float(nanoseconds) / 1_000_000_000.0
                            )
                    native_action = transition["joint_action"]
                    accepted_sides = {
                        int(item["side"])
                        for item in native_action["actions"]
                        if bool(item["result"].get("accepted", False))
                    }
                    rejected = [
                        item for item in native_action["actions"]
                        if not bool(item["result"].get("accepted", False))
                    ]
                    unexpected = [
                        item for item in rejected
                        if int(item["result"].get("result_code", -1)) not in (3, 4)
                    ]
                    if unexpected and int(cursor.state["tick"]) >= 100:
                        raise RuntimeError(
                            "action selected from legal mask was rejected: "
                            + json.dumps(unexpected, ensure_ascii=False)
                        )
                    for side in (0, 1):
                        if side not in accepted_sides:
                            values["next_public"][side] = None
                    cursor.action_count += len(accepted_sides)
                    episode = transition["step"]["episode"]
                    done = bool(episode["terminated"] or episode["truncated"])
                    next_state = None if done else transition["state"]
                    started = time.perf_counter()
                    terminal_rewards = episode.get(
                        "rewards_by_side", {0: 0.0, 1: 0.0}
                    )
                    if self.reward_mode == "potential":
                        rewards = self.potential_reward.transition(
                            cursor.state,
                            next_state,
                            terminal_rewards=terminal_rewards,
                            done=done,
                        )
                    else:
                        rewards = {
                            0: float(terminal_rewards.get(0, 0.0)),
                            1: float(terminal_rewards.get(1, 0.0)),
                        }
                    cursor.record("reward_seconds", started)
                    started = time.perf_counter()
                    for side in (0, 1):
                        row = values["encoded"][side]
                        sample: SampledAction = values["samples"][side]
                        trajectory = cursor.trajectories[side]
                        trajectory.grid.append(row[0])
                        trajectory.scalars.append(row[1])
                        trajectory.privileged.append(row[2])
                        trajectory.card_masks.append(row[3])
                        trajectory.position_masks.append(row[4])
                        trajectory.cards.append(sample.card)
                        trajectory.positions.append(sample.position)
                        trajectory.log_probabilities.append(sample.log_probability)
                        trajectory.values.append(sample.value)
                        trajectory.rewards.append(float(rewards[side]))
                        trajectory.dones.append(done)
                        trajectory.hidden_h.append(row[5])
                        trajectory.hidden_c.append(row[6])
                    cursor.action_log.append({
                        "tick": int(cursor.state["tick"]),
                        "actions": [dict(action) for action in values["chosen"]],
                        "native": native_action,
                        "state_hash": cursor.state.get("state_hash"),
                    })
                    cursor.record("trajectory_seconds", started)
                    if done:
                        cursor.done = True
                        cursor.episode = episode
                    else:
                        assert next_state is not None
                        cursor.state = next_state
                        cursor.public_actions = values["next_public"]
                        cursor.last_hash = cursor.state.get("state_hash")
                self._record_vector("vector_round_seconds", round_started)

        results = []
        for cursor in cursors:
            episode = cursor.episode or cursor.state["episode"]
            truncated = not bool(episode.get("terminated", False))
            if truncated and cursor.trajectories[0].dones:
                cursor.trajectories[0].dones[-1] = True
                cursor.trajectories[1].dones[-1] = True
            cursor.profile.update(cursor.env.rpc_profile)
            results.append(EpisodeResult(
                seed=cursor.seed,
                tick=int(episode.get("terminal_tick", cursor.state["tick"])),
                winner=episode.get("winner"),
                outcome=str(
                    episode.get("outcome", "truncated" if truncated else "ongoing")
                ),
                terminated=bool(episode.get("terminated", False)),
                truncated=truncated,
                wall_seconds=time.perf_counter() - cursor.started,
                actions=cursor.action_count,
                trajectories=cursor.trajectories,
                action_log=cursor.action_log,
                state_hash=cursor.last_hash,
                profile=cursor.profile,
            ))
        if results:
            total_latency: list[float] = []
            receive_latency: list[float] = []
            attempts = 0.0
            failures = 0.0
            for env in self.envs:
                samples = env.rpc_latency_samples()
                total_latency.extend(samples["total"])
                receive_latency.extend(samples["receive"])
                attempts += env.rpc_profile.get("rpc_attempts", 0.0)
                failures += env.rpc_profile.get("rpc_failures", 0.0)
            self.rpc_latency_samples = {
                "total": total_latency,
                "receive": receive_latency,
            }
            results[0].profile.update(JsonLineClient.latency_summary(
                total_latency,
                receive_latency,
                attempts=attempts,
                failures=failures,
            ))
            for name, value in self.policy.cuda_graph_stats.items():
                self.vector_profile[f"cuda_graph_{name}"] = (
                    value - self._cuda_graph_stats_before.get(name, 0.0)
                )
            results[0].profile.update(self.vector_profile)
        return results
