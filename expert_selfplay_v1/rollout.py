"""Single-learner immutable episode records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping


@dataclass(frozen=True)
class EpisodeHeader:
    episode_id: str
    batch_id: str
    seed: int
    learner_side: int
    behavior_policy_version: int
    behavior_actor_sha256: str
    opponent_policy_id: str
    opponent_actor_sha256: str
    learner_deck_sha256: str
    opponent_deck_sha256: str
    curriculum_stage: str
    initial_hidden_sha256: str

    def validate(self) -> None:
        if not self.episode_id or not self.batch_id or self.seed < 0:
            raise ValueError("invalid episode identity")
        if self.learner_side not in (0, 1) or self.behavior_policy_version < 0:
            raise ValueError("invalid learner/policy version")
        for name, value in asdict(self).items():
            if name.endswith("sha256") and (
                len(str(value)) != 64
                or any(character not in "0123456789abcdef" for character in str(value))
            ):
                raise ValueError(f"invalid {name}")


@dataclass(frozen=True)
class DecisionRecord:
    tick: int
    delta_ticks: int
    side: int
    event_happened: bool
    action_kind: int
    card_slot: int
    position: int
    ability_slot: int
    ability_position: int
    old_logp_total: float
    old_logp_timing: float
    old_logp_action_type: float
    old_logp_slot: float
    old_logp_position: float
    reward_damage_dealt: float
    reward_damage_received: float
    reward_towers_dealt: float
    reward_towers_received: float
    reward_terminal: float
    reward_total: float
    value: float
    terminated: bool
    truncated: bool
    native_entity_count: int
    encoded_entity_count: int

    def validate(self, learner_side: int) -> None:
        if self.side != learner_side:
            raise ValueError("opponent trajectory cannot enter learner PPO")
        if self.tick < 0 or self.delta_ticks < 1:
            raise ValueError("invalid decision time")
        if self.terminated and self.truncated:
            raise ValueError("transition cannot be terminated and truncated")
        if self.native_entity_count > 0 and self.encoded_entity_count == 0:
            raise ValueError("native nonempty observation was encoded as empty")
        reward_sum = (
            self.reward_damage_dealt + self.reward_damage_received
            + self.reward_towers_dealt + self.reward_towers_received
            + self.reward_terminal
        )
        if abs(reward_sum - self.reward_total) > 1e-6:
            raise ValueError("reward components do not sum to reward_total")


class LearnerEpisodeBuffer:
    def __init__(self, header: EpisodeHeader) -> None:
        header.validate()
        self.header = header
        self.decisions: list[DecisionRecord] = []

    def append(self, value: DecisionRecord) -> None:
        value.validate(self.header.learner_side)
        if self.decisions and value.tick <= self.decisions[-1].tick:
            raise ValueError("decision ticks must be strictly increasing")
        if self.decisions and (self.decisions[-1].terminated or self.decisions[-1].truncated):
            raise ValueError("cannot append after episode end")
        self.decisions.append(value)

    def freeze(self) -> dict[str, Any]:
        if not self.decisions:
            raise ValueError("cannot freeze an empty learner episode")
        if not (self.decisions[-1].terminated or self.decisions[-1].truncated):
            raise ValueError("learner episode is incomplete")
        payload: dict[str, Any] = {
            "schema_version": 1,
            "kind": "cr_native_expert_selfplay_learner_episode_v1",
            "header": asdict(self.header),
            "decisions": [asdict(value) for value in self.decisions],
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        payload["content_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return payload
