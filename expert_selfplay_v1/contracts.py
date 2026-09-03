"""Fail-closed rollout and encoded-entity contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np


class EntityInputContractError(RuntimeError):
    pass


class EntityInputGuard:
    def __init__(self, *, position_count: int = 32 * 18) -> None:
        self.position_count = int(position_count)
        self.frames = 0
        self.native_nonempty_frames = 0
        self.encoded_nonempty_frames = 0
        self.native_entities = 0
        self.encoded_entities = 0
        self.unique_tokens: set[int] = set()
        self.unique_positions: set[int] = set()

    def observe(
        self,
        *,
        native_eligible_entities: int,
        entity_tokens: Sequence[int] | np.ndarray,
        entity_positions: Sequence[int] | np.ndarray,
        entity_mask: Sequence[bool] | np.ndarray,
    ) -> None:
        tokens = np.asarray(entity_tokens)
        positions = np.asarray(entity_positions)
        mask = np.asarray(entity_mask, dtype=np.bool_)
        if not (tokens.shape == positions.shape == mask.shape):
            raise EntityInputContractError("entity token/position/mask shapes differ")
        if native_eligible_entities < 0:
            raise EntityInputContractError("native eligible entity count is negative")
        visible_tokens = tokens[mask].astype(np.int64, copy=False)
        visible_positions = positions[mask].astype(np.int64, copy=False)
        if visible_tokens.size and np.any(visible_tokens <= 0):
            raise EntityInputContractError("visible native entity has PAD/unknown token")
        if visible_positions.size and (
            np.any(visible_positions < 0) or np.any(visible_positions >= self.position_count)
        ):
            raise EntityInputContractError("visible native entity position is out of range")
        if native_eligible_entities > 0 and visible_tokens.size == 0:
            raise EntityInputContractError(
                "native state is nonempty but encoded entity input is empty"
            )
        self.frames += 1
        self.native_nonempty_frames += int(native_eligible_entities > 0)
        self.encoded_nonempty_frames += int(visible_tokens.size > 0)
        self.native_entities += int(native_eligible_entities)
        self.encoded_entities += int(visible_tokens.size)
        self.unique_tokens.update(int(value) for value in visible_tokens)
        self.unique_positions.update(int(value) for value in visible_positions)

    def summary(self) -> dict[str, int | float]:
        return {
            "frames": self.frames,
            "native_nonempty_frames": self.native_nonempty_frames,
            "encoded_nonempty_frames": self.encoded_nonempty_frames,
            "native_nonempty_rate": self.native_nonempty_frames / max(1, self.frames),
            "encoded_nonempty_rate": self.encoded_nonempty_frames / max(1, self.frames),
            "native_entities": self.native_entities,
            "encoded_entities": self.encoded_entities,
            "unique_entity_tokens": len(self.unique_tokens),
            "unique_entity_positions": len(self.unique_positions),
        }


@dataclass(frozen=True)
class BatchManifest:
    run_id: str
    batch_id: str
    policy_version: int
    behavior_actor_sha256: str
    encoder_schema_sha256: str
    action_schema_sha256: str
    reward_schema_sha256: str
    native_lib_sha256: str
    episode_count: int

    def validate(self) -> None:
        if not self.run_id or not self.batch_id or self.policy_version < 0:
            raise ValueError("invalid rollout batch identity")
        if self.episode_count < 1:
            raise ValueError("rollout batch must contain episodes")
        for name, value in asdict(self).items():
            if name.endswith("sha256") and (
                len(str(value)) != 64
                or any(character not in "0123456789abcdef" for character in str(value))
            ):
                raise ValueError(f"invalid {name}")

    def digest(self) -> str:
        self.validate()
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_schema_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
