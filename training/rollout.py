"""Synchronous, same-tick self-play collection from one native battle."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch

from native_core.env import NativeRoyaleEnv

from .model import RecurrentPolicyValueNet
from .schema import ObservationEncoder, PotentialReward, build_action_masks


@dataclass
class AgentTrajectory:
    side: int
    seed: int
    grid: list[np.ndarray] = field(default_factory=list)
    scalars: list[np.ndarray] = field(default_factory=list)
    privileged: list[np.ndarray] = field(default_factory=list)
    card_masks: list[np.ndarray] = field(default_factory=list)
    position_masks: list[np.ndarray] = field(default_factory=list)
    cards: list[int] = field(default_factory=list)
    positions: list[int] = field(default_factory=list)
    log_probabilities: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    hidden_h: list[np.ndarray] = field(default_factory=list)
    hidden_c: list[np.ndarray] = field(default_factory=list)

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "grid": np.stack(self.grid),
            "scalars": np.stack(self.scalars),
            "privileged": np.stack(self.privileged),
            "card_masks": np.stack(self.card_masks),
            "position_masks": np.stack(self.position_masks),
            "cards": np.asarray(self.cards, dtype=np.int64),
            "positions": np.asarray(self.positions, dtype=np.int64),
            "log_probabilities": np.asarray(self.log_probabilities, dtype=np.float32),
            "values": np.asarray(self.values, dtype=np.float32),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
            "dones": np.asarray(self.dones, dtype=np.bool_),
            "hidden_h": np.stack(self.hidden_h),
            "hidden_c": np.stack(self.hidden_c),
        }


@dataclass
class EpisodeResult:
    seed: int
    tick: int
    winner: int | None
    outcome: str
    terminated: bool
    truncated: bool
    wall_seconds: float
    actions: int
    trajectories: tuple[AgentTrajectory, AgentTrajectory]
    action_log: list[dict[str, Any]]
    state_hash: str | None
    profile: dict[str, float]

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "native_selfplay_episode",
            "seed": self.seed,
            "tick": self.tick,
            "winner": self.winner,
            "outcome": self.outcome,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "wall_seconds": self.wall_seconds,
            "actions": self.actions,
            "state_hash": self.state_hash,
            "profile": self.profile,
        }


class NativeSelfPlayCollector:
    """Both policies decide from one observation, then submit one joint RPC."""

    def __init__(
        self,
        env: NativeRoyaleEnv,
        policy: RecurrentPolicyValueNet,
        replay: Mapping[str, Any],
        *,
        device: torch.device,
        reward_mode: str = "terminal",
        max_ticks: int = 7200,
    ) -> None:
        if reward_mode not in ("terminal", "potential"):
            raise ValueError("reward_mode must be terminal or potential")
        self.env = env
        self.policy = policy
        self.replay = json.loads(json.dumps(replay))
        self.device = device
        self.reward_mode = reward_mode
        self.max_ticks = max_ticks
        self.encoder = ObservationEncoder()
        self.potential_reward = PotentialReward(gamma=0.99995)
        self.native_masks: dict[tuple[int, int], list[str]] = {}

    def _prepare_native_masks(self, state: Mapping[str, Any]) -> None:
        for player in state.get("players", []):
            side = int(player["side"])
            for deck_index in player["hand_deck_indices"]:
                deck_index = int(deck_index)
                if deck_index < 0:
                    continue
                if (side, deck_index) in self.native_masks:
                    continue
                result = self.env.probe_grid(side=side, deck_index=deck_index)
                rows = [str(row) for row in result["rows"]]
                self.native_masks[(side, deck_index)] = rows

    @staticmethod
    def _canonical_positions(mask: np.ndarray, side: int) -> np.ndarray:
        shaped = mask.reshape(4, 32, 18)
        if side == 1:
            shaped = shaped[:, ::-1, ::-1]
        return np.ascontiguousarray(shaped.reshape(4, 32 * 18))

    @staticmethod
    def _absolute_cell(position: int, side: int) -> tuple[int, int]:
        row, column = divmod(position, 18)
        if side == 1:
            row, column = 31 - row, 17 - column
        return column * 1000 + 500, row * 1000 + 500

    def collect(self, seed: int) -> EpisodeResult:
        profile: dict[str, float] = {}
        def record_time(name: str, started_at: float) -> None:
            profile[name] = profile.get(name, 0.0) + (
                time.perf_counter() - started_at
            )

        replay = json.loads(json.dumps(self.replay))
        replay["rndSeed"] = seed
        # One persistent app_process owns libg. Every episode uses libg's
        # native BattleGameState 4->4 replacement; no Android or process boot.
        stage_started = time.perf_counter()
        state = self.env.reset(replay, warmup_steps=100)
        record_time("reset_seconds", stage_started)
        self.policy.eval()
        trajectories = (
            AgentTrajectory(side=0, seed=seed),
            AgentTrajectory(side=1, seed=seed),
        )
        hidden = {
            side: self.policy.initial_hidden(1, device=self.device)
            for side in (0, 1)
        }
        public_actions: dict[int, dict[str, int] | None] = {0: None, 1: None}
        action_log: list[dict[str, Any]] = []
        action_count = 0
        started = time.perf_counter()
        last_hash = state.get("state_hash")
        while int(state["tick"]) < self.max_ticks:
            profile["decision_steps"] = profile.get("decision_steps", 0.0) + 1.0
            stage_started = time.perf_counter()
            self._prepare_native_masks(state)
            record_time("mask_probe_seconds", stage_started)
            samples: dict[int, Any] = {}
            chosen: list[dict[str, int]] = []
            encoded: dict[int, tuple[np.ndarray, ...]] = {}
            hands: dict[int, list[int]] = {}
            next_public: dict[int, dict[str, int] | None] = {0: None, 1: None}
            for side in (0, 1):
                stage_started = time.perf_counter()
                grid, scalars = self.encoder.encode(
                    state, side=side, public_actions=public_actions
                )
                privileged = self.encoder.privileged(state, side=side)
                record_time("encoding_seconds", stage_started)
                stage_started = time.perf_counter()
                card_mask, position_masks, hand = build_action_masks(
                    state,
                    side=side,
                    native_masks=self.native_masks,
                    decks=self.env.decks,
                )
                position_masks = self._canonical_positions(position_masks, side)
                record_time("mask_build_seconds", stage_started)
                h_before = hidden[side]
                hands[side] = hand
                encoded[side] = (
                    grid,
                    scalars,
                    privileged,
                    card_mask,
                    position_masks,
                    h_before[0][0, 0].detach().cpu().numpy().copy(),
                    h_before[1][0, 0].detach().cpu().numpy().copy(),
                )
            hidden_batch = (
                torch.cat((hidden[0][0], hidden[1][0]), dim=1),
                torch.cat((hidden[0][1], hidden[1][1]), dim=1),
            )
            stage_started = time.perf_counter()
            sampled = self.policy.sample_batch(
                torch.from_numpy(
                    np.stack((encoded[0][0], encoded[1][0]))
                ).to(self.device),
                torch.from_numpy(
                    np.stack((encoded[0][1], encoded[1][1]))
                ).to(self.device),
                torch.from_numpy(
                    np.stack((encoded[0][2], encoded[1][2]))
                ).to(self.device),
                torch.from_numpy(
                    np.stack((encoded[0][3], encoded[1][3]))
                ).to(self.device),
                torch.from_numpy(
                    np.stack((encoded[0][4], encoded[1][4]))
                ).to(self.device),
                hidden_batch,
            )
            record_time("inference_seconds", stage_started)
            for side, sample in enumerate(sampled):
                hidden[side] = sample.hidden
                samples[side] = sample
                if sample.card > 0:
                    hand_index = sample.card - 1
                    deck_index = hands[side][hand_index]
                    card_id = int(self.env.decks[side][deck_index]["card_id"])
                    x, y = self._absolute_cell(sample.position, side)
                    action = {
                        "side": side,
                        "deck_index": deck_index,
                        "x": x,
                        "y": y,
                    }
                    chosen.append(action)
                    next_public[side] = {
                        "card_id": card_id,
                        "x": x,
                        "y": y,
                    }
            stage_started = time.perf_counter()
            transition = self.env.joint_training_transition(chosen, steps=1)
            record_time("transition_seconds", stage_started)
            native_action = transition["joint_action"]
            accepted_sides = {
                int(item["side"])
                for item in native_action["actions"]
                if bool(item["result"].get("accepted", False))
            }
            rejected = [
                item
                for item in native_action["actions"]
                if not bool(item["result"].get("accepted", False))
            ]
            unexpected_rejections = [
                item
                for item in rejected
                if int(item["result"].get("result_code", -1)) not in (3, 4)
            ]
            if unexpected_rejections and int(state["tick"]) >= 100:
                raise RuntimeError(
                    "action selected from legal mask was rejected: "
                    + json.dumps(unexpected_rejections, ensure_ascii=False)
                )
            for side in (0, 1):
                if side not in accepted_sides:
                    next_public[side] = None
            action_count += len(accepted_sides)
            native_step = transition["step"]
            episode = native_step["episode"]
            done = bool(episode["terminated"] or episode["truncated"])
            next_state = None if done else transition["state"]
            terminal_rewards = episode.get("rewards_by_side", {0: 0.0, 1: 0.0})
            stage_started = time.perf_counter()
            if self.reward_mode == "potential":
                rewards = self.potential_reward.transition(
                    state,
                    next_state,
                    terminal_rewards=terminal_rewards,
                    done=done,
                )
            else:
                rewards = {
                    0: float(terminal_rewards.get(0, 0.0)),
                    1: float(terminal_rewards.get(1, 0.0)),
                }
            record_time("reward_seconds", stage_started)
            stage_started = time.perf_counter()
            for side in (0, 1):
                record = trajectories[side]
                grid, scalars, privileged, card_mask, position_masks, h, c = encoded[side]
                sample = samples[side]
                record.grid.append(grid)
                record.scalars.append(scalars)
                record.privileged.append(privileged)
                record.card_masks.append(card_mask)
                record.position_masks.append(position_masks)
                record.cards.append(sample.card)
                record.positions.append(sample.position)
                record.log_probabilities.append(sample.log_probability)
                record.values.append(sample.value)
                record.rewards.append(float(rewards[side]))
                record.dones.append(done)
                record.hidden_h.append(h)
                record.hidden_c.append(c)
            action_log.append(
                {
                    "tick": int(state["tick"]),
                    "actions": [dict(action) for action in chosen],
                    "native": native_action,
                    "state_hash": state.get("state_hash"),
                }
            )
            record_time("trajectory_seconds", stage_started)
            if done:
                last_hash = state.get("state_hash")
                break
            assert next_state is not None
            state = next_state
            public_actions = next_public
            last_hash = state.get("state_hash")
        else:
            episode = state["episode"]
            done = False

        truncated = not bool(episode.get("terminated", False))
        if truncated and trajectories[0].dones:
            trajectories[0].dones[-1] = True
            trajectories[1].dones[-1] = True
        for key, value in self.env.rpc_profile.items():
            profile[key] = value
        return EpisodeResult(
            seed=seed,
            tick=int(episode.get("terminal_tick", state["tick"])),
            winner=episode.get("winner"),
            outcome=str(episode.get("outcome", "truncated" if truncated else "ongoing")),
            terminated=bool(episode.get("terminated", False)),
            truncated=truncated,
            wall_seconds=time.perf_counter() - started,
            actions=action_count,
            trajectories=trajectories,
            action_log=action_log,
            state_hash=str(last_hash) if last_hash is not None else None,
            profile=profile,
        )


def save_episode(path: Path, result: EpisodeResult, *, full_debug: bool = False) -> None:
    """Persist compact tensors and replayable action metadata."""
    path.mkdir(parents=True, exist_ok=True)
    for trajectory in result.trajectories:
        target = path / f"seed-{result.seed}-side-{trajectory.side}.npz"
        temporary = target.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **trajectory.arrays())
        temporary.replace(target)
    summary = result.summary()
    if full_debug:
        summary["action_log"] = result.action_log
    target = path / f"seed-{result.seed}.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
