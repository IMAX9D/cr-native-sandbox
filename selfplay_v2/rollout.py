"""v0.2 trajectories and opportunity-normalized behavior telemetry."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable

import numpy as np

from training.rollout import EpisodeResult, summarize_episode_behavior
from training.run_contract import aggregate_behavior
from training.schema import CARD_COSTS, CARD_IDS
from .model import SampledTimedAction


@dataclass
class TimedAgentTrajectory:
    side: int
    seed: int
    grid: list[np.ndarray] = field(default_factory=list)
    scalars: list[np.ndarray] = field(default_factory=list)
    privileged: list[np.ndarray] = field(default_factory=list)
    card_masks: list[np.ndarray] = field(default_factory=list)
    position_masks: list[np.ndarray] = field(default_factory=list)
    cards: list[int] = field(default_factory=list)
    positions: list[int] = field(default_factory=list)
    timing_valids: list[bool] = field(default_factory=list)
    rates: list[float] = field(default_factory=list)
    play_probabilities: list[float] = field(default_factory=list)
    policy_entropies: list[float] = field(default_factory=list)
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
            "timing_valids": np.asarray(self.timing_valids, dtype=np.bool_),
            "rates": np.asarray(self.rates, dtype=np.float32),
            "play_probabilities": np.asarray(
                self.play_probabilities, dtype=np.float32
            ),
            "policy_entropies": np.asarray(
                self.policy_entropies, dtype=np.float32
            ),
            "log_probabilities": np.asarray(
                self.log_probabilities, dtype=np.float32
            ),
            "values": np.asarray(self.values, dtype=np.float32),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
            "dones": np.asarray(self.dones, dtype=np.bool_),
            "hidden_h": np.stack(self.hidden_h),
            "hidden_c": np.stack(self.hidden_c),
        }


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
    }


class EpisodeTelemetry:
    def __init__(self) -> None:
        self.cards: dict[int, dict[str, Any]] = {
            card_id: {
                "ticks_in_hand": 0,
                "legal_ticks": 0,
                "affordable_ticks": 0,
                "playable_ticks": 0,
                "selected_count": 0,
                "elixir_at_play": [],
                "hand_hold_durations": [],
                "position_entropy": [],
                "position_normalized_entropy": [],
                "position_effective_cells": [],
                "position_top1_mass": [],
                "position_top5_mass": [],
            }
            for card_id in CARD_IDS
        }
        self.rates: list[float] = []
        self.valid_rates: list[float] = []
        self.play_probabilities: list[float] = []
        self.play_probabilities_by_elixir: dict[int, list[float]] = {
            value: [] for value in range(11)
        }
        self.play_events = 0
        self.timing_valid_ticks = 0
        self.no_play_streak = {0: 0, 1: 0}
        self.no_play_durations: list[int] = []
        self.hand_streak: dict[tuple[int, int], int] = {
            (side, card_id): 0 for side in (0, 1) for card_id in CARD_IDS
        }

    def record_opportunity(
        self,
        *,
        side: int,
        slot_card_ids: list[int],
        legal: np.ndarray,
        affordable: np.ndarray,
        playable: np.ndarray,
    ) -> None:
        in_hand = set(slot_card_ids)
        for card_id in CARD_IDS:
            key = (side, card_id)
            if card_id in in_hand:
                self.hand_streak[key] += 1
            elif self.hand_streak[key]:
                self.cards[card_id]["hand_hold_durations"].append(
                    self.hand_streak[key]
                )
                self.hand_streak[key] = 0
        for slot, card_id in enumerate(slot_card_ids):
            value = self.cards[card_id]
            value["ticks_in_hand"] += 1
            value["legal_ticks"] += int(bool(legal[slot]))
            value["affordable_ticks"] += int(bool(affordable[slot]))
            value["playable_ticks"] += int(bool(playable[slot]))

    def record_sample(
        self,
        *,
        side: int,
        slot_card_ids: list[int],
        elixir: int,
        sample: SampledTimedAction,
    ) -> None:
        self.rates.append(sample.rate)
        self.play_probabilities.append(sample.play_probability)
        self.play_probabilities_by_elixir[max(0, min(10, elixir))].append(
            sample.play_probability
        )
        if sample.timing_valid:
            self.timing_valid_ticks += 1
            self.valid_rates.append(sample.rate)
        if sample.play_now:
            self.play_events += 1
            if self.no_play_streak[side]:
                self.no_play_durations.append(self.no_play_streak[side])
            self.no_play_streak[side] = 0
            card_id = slot_card_ids[sample.card - 1]
            self.cards[card_id]["selected_count"] += 1
            self.cards[card_id]["elixir_at_play"].append(elixir)
        else:
            self.no_play_streak[side] += 1
        for slot, card_id in enumerate(slot_card_ids):
            diagnostic = sample.position_diagnostics[slot]
            if diagnostic is None:
                continue
            value = self.cards[card_id]
            value["position_entropy"].append(diagnostic["entropy"])
            value["position_normalized_entropy"].append(
                diagnostic["normalized_entropy"]
            )
            value["position_effective_cells"].append(
                diagnostic["effective_cells"]
            )
            value["position_top1_mass"].append(diagnostic["top1_mass"])
            value["position_top5_mass"].append(diagnostic["top5_mass"])

    def summary(self, *, native_ticks: int) -> dict[str, Any]:
        for side in (0, 1):
            if self.no_play_streak[side]:
                self.no_play_durations.append(self.no_play_streak[side])
            for card_id in CARD_IDS:
                streak = self.hand_streak[(side, card_id)]
                if streak:
                    self.cards[card_id]["hand_hold_durations"].append(streak)
        card_summary: dict[str, Any] = {}
        for card_id, raw in self.cards.items():
            in_hand = int(raw["ticks_in_hand"])
            legal = int(raw["legal_ticks"])
            affordable = int(raw["affordable_ticks"])
            playable = int(raw["playable_ticks"])
            selected = int(raw["selected_count"])
            card_summary[str(card_id)] = {
                "ticks_in_hand": in_hand,
                "legal_ticks": legal,
                "affordable_ticks": affordable,
                "playable_ticks": playable,
                "selected_count": selected,
                "legal_given_in_hand": legal / in_hand if in_hand else 0.0,
                "affordable_given_in_hand": (
                    affordable / in_hand if in_hand else 0.0
                ),
                "playable_given_in_hand": (
                    playable / in_hand if in_hand else 0.0
                ),
                "selected_given_playable": (
                    selected / playable if playable else 0.0
                ),
                "average_elixir_at_play": (
                    float(np.mean(raw["elixir_at_play"]))
                    if raw["elixir_at_play"] else 0.0
                ),
                "average_hand_hold_ticks": (
                    float(np.mean(raw["hand_hold_durations"]))
                    if raw["hand_hold_durations"] else 0.0
                ),
                "position_entropy": _percentiles(raw["position_entropy"]),
                "position_normalized_entropy": _percentiles(
                    raw["position_normalized_entropy"]
                ),
                "position_effective_cells": _percentiles(
                    raw["position_effective_cells"]
                ),
                "position_top1_mass": _percentiles(
                    raw["position_top1_mass"]
                ),
                "position_top5_mass": _percentiles(
                    raw["position_top5_mass"]
                ),
            }
        duration_seconds = max(1, native_ticks) * 0.05 * 2.0
        return {
            "cards": card_summary,
            "rate": _percentiles(self.rates),
            "valid_rate": _percentiles(self.valid_rates),
            "play_probability": _percentiles(self.play_probabilities),
            "play_probability_by_elixir": {
                str(elixir): _percentiles(values)
                for elixir, values in self.play_probabilities_by_elixir.items()
            },
            "play_events": self.play_events,
            "play_events_per_side_second": self.play_events / duration_seconds,
            "timing_valid_ticks": self.timing_valid_ticks,
            "no_play_duration_ticks": _percentiles(
                [float(value) for value in self.no_play_durations]
            ),
        }


def summarize_timed_episode(
    result: EpisodeResult,
    telemetry: EpisodeTelemetry,
) -> dict[str, Any]:
    base = summarize_episode_behavior(
        result.trajectories,
        result.action_log,
        result.terminal_episode,
    )
    base["timing_v2"] = telemetry.summary(
        native_ticks=len(result.trajectories[0].cards)
    )
    return base


def aggregate_timed_behavior(
    results: Iterable[EpisodeResult],
) -> tuple[dict[str, Any], np.ndarray]:
    episodes = list(results)
    behavior, histogram = aggregate_behavior(episodes)
    timing_values = [
        item.behavior.get("timing_v2", {}) for item in episodes
    ]
    total_decisions = sum(
        sum(len(trajectory.cards) for trajectory in item.trajectories)
        for item in episodes
    )
    total_valid = sum(
        int(value.get("timing_valid_ticks", 0)) for value in timing_values
    )
    play_events = sum(
        int(value.get("play_events", 0)) for value in timing_values
    )

    def weighted_metric(
        section: str, metric: str, weights: list[int]
    ) -> float:
        denominator = sum(weights)
        if not denominator:
            return 0.0
        return sum(
            float(value.get(section, {}).get(metric, 0.0)) * weight
            for value, weight in zip(timing_values, weights, strict=True)
        ) / denominator

    decision_weights = [
        sum(len(trajectory.cards) for trajectory in item.trajectories)
        for item in episodes
    ]
    valid_weights = [
        int(value.get("timing_valid_ticks", 0)) for value in timing_values
    ]
    card_summary: dict[str, Any] = {}
    for card_id in CARD_IDS:
        key = str(card_id)
        rows = [value.get("cards", {}).get(key, {}) for value in timing_values]
        counts = {
            name: sum(int(row.get(name, 0)) for row in rows)
            for name in (
                "ticks_in_hand", "legal_ticks", "affordable_ticks",
                "playable_ticks", "selected_count",
            )
        }
        in_hand = counts["ticks_in_hand"]
        playable = counts["playable_ticks"]
        selected = counts["selected_count"]

        def selected_weighted(name: str) -> float:
            return (
                sum(
                    float(row.get(name, 0.0))
                    * int(row.get("selected_count", 0))
                    for row in rows
                ) / selected
                if selected else 0.0
            )

        position_weight = [int(row.get("playable_ticks", 0)) for row in rows]
        position_denominator = sum(position_weight)

        def position_mean(name: str) -> float:
            if not position_denominator:
                return 0.0
            return sum(
                float(row.get(name, {}).get("mean", 0.0)) * weight
                for row, weight in zip(rows, position_weight, strict=True)
            ) / position_denominator

        card_summary[key] = {
            **counts,
            "legal_given_in_hand": (
                counts["legal_ticks"] / in_hand if in_hand else 0.0
            ),
            "affordable_given_in_hand": (
                counts["affordable_ticks"] / in_hand if in_hand else 0.0
            ),
            "playable_given_in_hand": (
                playable / in_hand if in_hand else 0.0
            ),
            "selected_given_playable": (
                selected / playable if playable else 0.0
            ),
            "average_elixir_at_play": selected_weighted(
                "average_elixir_at_play"
            ),
            "position_normalized_entropy_mean": position_mean(
                "position_normalized_entropy"
            ),
            "position_effective_cells_mean": position_mean(
                "position_effective_cells"
            ),
            "position_top1_mass_mean": position_mean("position_top1_mass"),
            "position_top5_mass_mean": position_mean("position_top5_mass"),
        }
    behavior["timing_v2"] = {
        "cards": card_summary,
        "rate_mean": weighted_metric("rate", "mean", decision_weights),
        "rate_p50": weighted_metric("rate", "p50", decision_weights),
        "rate_p95": weighted_metric("rate", "p95", decision_weights),
        "valid_rate_mean": weighted_metric(
            "valid_rate", "mean", valid_weights
        ),
        "play_probability_mean": weighted_metric(
            "play_probability", "mean", decision_weights
        ),
        "play_events": play_events,
        "timing_valid_ticks": total_valid,
        "play_events_per_side_second": (
            play_events / max(1.0, total_decisions * 0.05)
        ),
        "decision_count": total_decisions,
    }
    return behavior, histogram
