"""Immutable policy records and exact-quota opponent scheduling."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import random
from typing import Literal, Mapping


PolicyRole = Literal["base", "champion", "historical", "candidate"]


@dataclass(frozen=True)
class PolicyRecord:
    policy_id: str
    actor_fp16_sha256: str
    actor_master_sha256: str | None
    parent_policy_id: str | None
    created_update: int
    promoted_round: int | None
    curriculum_stage: str
    role: PolicyRole
    encoder_schema_hash: str
    action_schema_hash: str
    frozen: bool = True

    def validate(self) -> None:
        if not self.policy_id or self.created_update < 0 or not self.frozen:
            raise ValueError("policy record is not immutable")
        for name in ("actor_fp16_sha256", "encoder_schema_hash", "action_schema_hash"):
            value = str(getattr(self, name))
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"invalid {name}")
        if self.actor_master_sha256 is not None:
            value = self.actor_master_sha256
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError("invalid actor_master_sha256")


@dataclass
class MatchupStats:
    wins: int = 0
    draws: int = 0
    losses: int = 0
    crown_diff_sum: float = 0.0
    tower_hp_diff_sum: float = 0.0
    defense_metric_sum: float = 0.0

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def smoothed_score(self) -> float:
        return (self.wins + 0.5 * self.draws + 2.0) / (self.games + 4.0)

    def add(self, result: float, *, crown_diff: float = 0.0,
            tower_hp_diff: float = 0.0, defense_metric: float = 0.0) -> None:
        if result > 0:
            self.wins += 1
        elif result < 0:
            self.losses += 1
        else:
            self.draws += 1
        self.crown_diff_sum += float(crown_diff)
        self.tower_hp_diff_sum += float(tower_hp_diff)
        self.defense_metric_sum += float(defense_metric)


@dataclass
class LeagueState:
    base_policy_id: str
    champion_policy_id: str
    active_history_ids: list[str] = field(default_factory=list)
    archived_policy_ids: list[str] = field(default_factory=list)
    failed_candidate_ids: list[str] = field(default_factory=list)
    eval_elo: dict[str, float] = field(default_factory=dict)
    training_matchups: dict[str, MatchupStats] = field(default_factory=dict)
    formal_eval_matchups: dict[str, MatchupStats] = field(default_factory=dict)

    @staticmethod
    def matchup_key(candidate_id: str, opponent_id: str, deck_bucket: str = "all") -> str:
        return f"{candidate_id}|{opponent_id}|{deck_bucket}"

    def score(self, candidate_id: str, opponent_id: str, deck_bucket: str = "all") -> float:
        key = self.matchup_key(candidate_id, opponent_id, deck_bucket)
        return self.training_matchups.get(key, MatchupStats()).smoothed_score

    def active_history(self) -> list[str]:
        return list(dict.fromkeys(
            value for value in self.active_history_ids
            if value not in {self.base_policy_id, self.champion_policy_id}
        ))

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["training_matchups"] = {
            key: asdict(stats) for key, stats in self.training_matchups.items()
        }
        value["formal_eval_matchups"] = {
            key: asdict(stats) for key, stats in self.formal_eval_matchups.items()
        }
        return value


@dataclass(frozen=True)
class OpponentAssignment:
    episode_index: int
    learner_side: int
    category: Literal["champion", "history", "base"]
    policy_id: str
    history_mode: Literal["close", "hard", "uniform"] | None = None


def exact_quotas(total: int, weights: Mapping[str, float]) -> dict[str, int]:
    if total < 1 or not weights or any(value < 0 for value in weights.values()):
        raise ValueError("invalid quota request")
    scale = sum(weights.values())
    if scale <= 0:
        raise ValueError("quota weights sum to zero")
    raw = {key: total * value / scale for key, value in weights.items()}
    result = {key: int(math.floor(value)) for key, value in raw.items()}
    remaining = total - sum(result.values())
    order = sorted(weights, key=lambda key: (raw[key] - result[key], key), reverse=True)
    for key in order[:remaining]:
        result[key] += 1
    return result


class OpponentScheduler:
    def __init__(self, league: LeagueState) -> None:
        self.league = league

    @staticmethod
    def _weighted_pick(
        rng: random.Random,
        candidates: list[str],
        weights: list[float],
        counts: dict[str, int],
        cap: int,
    ) -> str:
        available = [
            (candidate, weight) for candidate, weight in zip(candidates, weights, strict=True)
            if counts.get(candidate, 0) < cap
        ]
        if not available:
            available = list(zip(candidates, weights, strict=True))
        names, values = zip(*available, strict=True)
        selected = rng.choices(names, weights=values, k=1)[0]
        counts[selected] = counts.get(selected, 0) + 1
        return selected

    def _history_assignments(
        self,
        *,
        count: int,
        candidate_id: str,
        rng: random.Random,
        deck_bucket: str,
    ) -> list[tuple[str, str]]:
        history = self.league.active_history()
        if count == 0:
            return []
        if not history:
            raise ValueError("history quota requested with an empty active history")
        modes = exact_quotas(count, {"close": 0.50, "hard": 0.25, "uniform": 0.25})
        # A strict 25% cap is possible once at least four distinct opponents exist.
        cap = max(math.ceil(count * 0.25), math.ceil(count / len(history)))
        counts: dict[str, int] = {}
        result: list[tuple[str, str]] = []
        scores = [self.league.score(candidate_id, opponent, deck_bucket) for opponent in history]
        for mode in ("close", "hard", "uniform"):
            if mode == "close":
                weights = [4.0 * score * (1.0 - score) + 1e-3 for score in scores]
            elif mode == "hard":
                weights = [(1.0 - score) ** 2 + 1e-3 for score in scores]
            else:
                weights = [1.0] * len(history)
            for _ in range(modes[mode]):
                selected = self._weighted_pick(rng, history, weights, counts, cap)
                result.append((selected, mode))
        rng.shuffle(result)
        return result

    def build_batch(
        self,
        *,
        episode_count: int,
        candidate_id: str,
        seed: int,
        deck_bucket: str = "all",
    ) -> list[OpponentAssignment]:
        rng = random.Random(seed)
        history = self.league.active_history()
        weights = (
            {"champion": 0.40, "history": 0.40, "base": 0.20}
            if history else {"champion": 0.40, "base": 0.60}
        )
        quotas = exact_quotas(episode_count, weights)
        rows: list[tuple[str, str, str | None]] = []
        rows.extend(
            ("champion", self.league.champion_policy_id, None)
            for _ in range(quotas.get("champion", 0))
        )
        rows.extend(
            ("base", self.league.base_policy_id, None)
            for _ in range(quotas.get("base", 0))
        )
        rows.extend(
            ("history", policy_id, mode)
            for policy_id, mode in self._history_assignments(
                count=quotas.get("history", 0),
                candidate_id=candidate_id,
                rng=rng,
                deck_bucket=deck_bucket,
            )
        )
        rng.shuffle(rows)
        # Exact or one-game side balance for odd batches, independent of category.
        sides = [index % 2 for index in range(episode_count)]
        rng.shuffle(sides)
        return [
            OpponentAssignment(
                episode_index=index,
                learner_side=sides[index],
                category=category,  # type: ignore[arg-type]
                policy_id=policy_id,
                history_mode=mode,  # type: ignore[arg-type]
            )
            for index, (category, policy_id, mode) in enumerate(rows)
        ]
