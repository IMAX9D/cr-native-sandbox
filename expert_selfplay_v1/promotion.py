"""Fixed-panel win-rate/Elo candidate admission without training-data leakage."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from .league import LeagueState, MatchupStats


def wilson_score_interval(points: float, games: int, *, z: float = 1.96) -> tuple[float, float]:
    if games < 1 or not 0.0 <= points <= games:
        raise ValueError("invalid scored game sample")
    probability = points / games
    denominator = 1.0 + z * z / games
    center = (probability + z * z / (2.0 * games)) / denominator
    margin = z * math.sqrt(
        probability * (1.0 - probability) / games + z * z / (4.0 * games * games)
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def fitted_candidate_elo(
    panel: Mapping[str, MatchupStats], opponent_ratings: Mapping[str, float]
) -> float:
    if not panel:
        raise ValueError("Elo fit requires a fixed evaluation panel")
    scored = sum(value.wins + 0.5 * value.draws for value in panel.values())
    games = sum(value.games for value in panel.values())
    if games < 1 or any(name not in opponent_ratings for name in panel):
        raise ValueError("Elo panel/rating coverage is incomplete")

    def expected(rating: float) -> float:
        return sum(
            value.games / (1.0 + 10.0 ** ((opponent_ratings[name] - rating) / 400.0))
            for name, value in panel.items()
        )

    lower, upper = -4000.0, 5000.0
    for _ in range(96):
        middle = (lower + upper) / 2.0
        if expected(middle) < scored:
            lower = middle
        else:
            upper = middle
    return (lower + upper) / 2.0


@dataclass(frozen=True)
class PromotionCriteria:
    champion_min_games: int = 512
    champion_score_lower_bound: float = 0.50
    base_min_games: int = 256
    base_score_lower_bound: float = 0.50
    history_min_games: int = 128
    history_score_floor: float = 0.42
    minimum_elo_gain: float = 0.0


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    candidate_elo: float
    reasons: tuple[str, ...]
    intervals: dict[str, tuple[float, float]]


def decide_promotion(
    *,
    candidate_id: str,
    league: LeagueState,
    panel: Mapping[str, MatchupStats],
    criteria: PromotionCriteria = PromotionCriteria(),
) -> PromotionDecision:
    if candidate_id in {league.base_policy_id, league.champion_policy_id}:
        raise ValueError("candidate must be a new immutable policy")
    required = {league.champion_policy_id, league.base_policy_id}
    if not required.issubset(panel):
        raise ValueError("promotion panel lacks champion/base")
    ratings = dict(league.eval_elo)
    ratings.setdefault(league.base_policy_id, 1000.0)
    ratings.setdefault(league.champion_policy_id, ratings[league.base_policy_id])
    for opponent in panel:
        ratings.setdefault(opponent, 1000.0)
    candidate_elo = fitted_candidate_elo(panel, ratings)
    reasons: list[str] = []
    intervals: dict[str, tuple[float, float]] = {}
    for opponent, stats in panel.items():
        if stats.games < 1:
            reasons.append(f"{opponent}:no_games")
            continue
        interval = wilson_score_interval(stats.wins + 0.5 * stats.draws, stats.games)
        intervals[opponent] = interval
        if opponent == league.champion_policy_id:
            if stats.games < criteria.champion_min_games:
                reasons.append("champion:insufficient_games")
            if interval[0] < criteria.champion_score_lower_bound:
                reasons.append("champion:score_lower_bound")
        elif opponent == league.base_policy_id:
            if stats.games < criteria.base_min_games:
                reasons.append("base:insufficient_games")
            if interval[0] < criteria.base_score_lower_bound:
                reasons.append("base:score_lower_bound")
        else:
            if stats.games < criteria.history_min_games:
                reasons.append(f"{opponent}:insufficient_games")
            if stats.smoothed_score < criteria.history_score_floor:
                reasons.append(f"{opponent}:regression")
    champion_elo = ratings[league.champion_policy_id]
    if candidate_elo < champion_elo + criteria.minimum_elo_gain:
        reasons.append("elo:insufficient_gain")
    return PromotionDecision(
        promoted=not reasons,
        candidate_elo=candidate_elo,
        reasons=tuple(reasons),
        intervals=intervals,
    )


def apply_promotion(
    league: LeagueState,
    *,
    candidate_id: str,
    decision: PromotionDecision,
) -> None:
    if not decision.promoted:
        if candidate_id not in league.failed_candidate_ids:
            league.failed_candidate_ids.append(candidate_id)
        return
    previous = league.champion_policy_id
    if previous != league.base_policy_id and previous not in league.active_history_ids:
        league.active_history_ids.append(previous)
    league.champion_policy_id = candidate_id
    league.eval_elo[candidate_id] = decision.candidate_elo
