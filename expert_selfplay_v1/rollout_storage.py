"""Fail-closed storage primitives for learner-only recurrent rollouts.

The native workers collect a complete episode before anything in this module is
eligible for PPO.  This file deliberately keeps three boundaries explicit:

* :class:`CriticObservationAdapter` is the only adapter that turns privileged
  (training-only) state into tensors accepted by :class:`PrivilegedCritic`.
* :class:`LearnerEpisodeChunker` computes variable-time GAE over the complete
  learner trajectory, then emits 16-decision burn-in / 64-decision loss chunks.
* :class:`ImmutableRolloutShardWriter` publishes a JSON manifest and a torch
  payload together via an atomic directory rename and records the committed
  content hash in :class:`RolloutLedger` when one is supplied.

Opponent decisions, incomplete games, time-truncated games, non-finite values,
and manifest/policy mismatches all fail closed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .contracts import BatchManifest
from .critic import PrivilegedCriticConfig
from .gae import variable_time_gae
from .ledger import RolloutLedger
from .rollout import DecisionRecord, EpisodeHeader, LearnerEpisodeBuffer


SHARD_SCHEMA_VERSION = 1
SHARD_KIND = "cr_native_expert_selfplay_rollout_shard_v1"
CHUNK_KIND = "cr_native_expert_selfplay_recurrent_chunk_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_SHARD_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_numpy(value: Any, *, name: str) -> np.ndarray:
    if isinstance(value, Tensor):
        if value.device.type != "cpu":
            value = value.detach().cpu()
        else:
            value = value.detach()
        try:
            return value.numpy()
        except TypeError as error:
            raise ValueError(f"{name} cannot be converted to a numpy array") from error
    try:
        return np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not array-like") from error


def _require_finite(value: np.ndarray, *, name: str) -> None:
    if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN/Inf")


@dataclass(frozen=True)
class CriticPrivateObservation:
    """One model-decision state, including Critic-only private information.

    Card arrays are already flattened into the Critic's private-card slots.  A
    worker may use those slots for both decks, hands, next cards, cycles, and
    abilities; ``private_card_owners`` and ``private_card_slots`` preserve the
    owner and semantic slot identity.
    """

    grid: Any
    entity_tokens: Any
    entity_positions: Any
    entity_relations: Any
    entity_numeric: Any
    entity_mask: Any
    private_card_tokens: Any
    private_card_owners: Any
    private_card_slots: Any
    private_card_mask: Any
    scalars: Any

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CriticPrivateObservation":
        missing = [name for name in cls.__dataclass_fields__ if name not in value]
        if missing:
            raise ValueError(f"Critic observation is missing fields: {missing}")
        return cls(**{name: value[name] for name in cls.__dataclass_fields__})


class CriticObservationAdapter:
    """Pad and validate privileged observations for ``PrivilegedCritic``.

    The returned tensors always have ``[batch=1, decisions, ...]`` prefixes.
    Ragged entity/private-card axes are padded to at least one element because
    the Critic's empty-sequence fallback addresses slot zero.
    """

    def __init__(
        self,
        config: PrivilegedCriticConfig,
        *,
        grid_height: int = 32,
        grid_width: int = 18,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = config
        self.grid_height = int(grid_height)
        self.grid_width = int(grid_width)
        self.device = torch.device(device)
        if self.grid_height < 1 or self.grid_width < 1:
            raise ValueError("Critic grid dimensions must be positive")

    def _validated(self, value: CriticPrivateObservation) -> dict[str, np.ndarray]:
        arrays = {
            name: _as_numpy(getattr(value, name), name=name)
            for name in value.__dataclass_fields__
        }
        expected_grid = (
            self.config.public_grid_channels,
            self.grid_height,
            self.grid_width,
        )
        if arrays["grid"].shape != expected_grid:
            raise ValueError(
                f"grid shape {arrays['grid'].shape} does not match {expected_grid}"
            )
        if arrays["scalars"].shape != (self.config.scalar_size,):
            raise ValueError("Critic scalar vector has the wrong shape")

        entity_count = arrays["entity_tokens"].size
        for name in (
            "entity_tokens", "entity_positions", "entity_relations", "entity_mask"
        ):
            if arrays[name].shape != (entity_count,):
                raise ValueError("Critic entity arrays have inconsistent shapes")
        if arrays["entity_numeric"].shape != (
            entity_count,
            self.config.entity_numeric_size,
        ):
            raise ValueError("Critic entity numeric array has the wrong shape")

        integer_fields = (
            "entity_tokens", "entity_positions", "entity_relations",
            "private_card_tokens", "private_card_owners", "private_card_slots",
        )
        for name in integer_fields:
            if not np.issubdtype(arrays[name].dtype, np.integer):
                raise TypeError(f"{name} must contain integers")

        private_count = arrays["private_card_tokens"].size
        for name in (
            "private_card_tokens", "private_card_owners", "private_card_slots",
            "private_card_mask",
        ):
            if arrays[name].shape != (private_count,):
                raise ValueError("Critic private-card arrays have inconsistent shapes")
        if private_count > self.config.private_slot_count:
            raise ValueError("Critic private-card state exceeds configured slots")

        for name in ("grid", "entity_numeric", "scalars"):
            _require_finite(arrays[name], name=name)

        entity_mask = arrays["entity_mask"].astype(np.bool_, copy=False)
        if arrays["entity_tokens"].size and (
            np.any(arrays["entity_tokens"] < 0)
            or np.any(arrays["entity_tokens"] >= self.config.card_vocab_size)
        ):
            raise ValueError("Critic entity token is out of vocabulary")
        visible_entity_tokens = arrays["entity_tokens"][entity_mask]
        visible_entity_positions = arrays["entity_positions"][entity_mask]
        visible_relations = arrays["entity_relations"][entity_mask]
        if visible_entity_tokens.size and (
            np.any(visible_entity_tokens <= 0)
            or np.any(visible_entity_tokens >= self.config.card_vocab_size)
        ):
            raise ValueError("visible Critic entity token is out of vocabulary")
        if visible_entity_positions.size and (
            np.any(visible_entity_positions < 0)
            or np.any(visible_entity_positions >= self.config.position_count)
        ):
            raise ValueError("visible Critic entity position is out of range")
        if visible_relations.size and np.any((visible_relations < 0) | (visible_relations > 1)):
            raise ValueError("visible Critic entity relation must be 0 or 1")

        private_mask = arrays["private_card_mask"].astype(np.bool_, copy=False)
        if arrays["private_card_tokens"].size and (
            np.any(arrays["private_card_tokens"] < 0)
            or np.any(arrays["private_card_tokens"] >= self.config.card_vocab_size)
        ):
            raise ValueError("Critic private-card token is out of vocabulary")
        visible_private_tokens = arrays["private_card_tokens"][private_mask]
        visible_private_owners = arrays["private_card_owners"][private_mask]
        visible_private_slots = arrays["private_card_slots"][private_mask]
        if visible_private_tokens.size and (
            np.any(visible_private_tokens <= 0)
            or np.any(visible_private_tokens >= self.config.card_vocab_size)
        ):
            raise ValueError("visible Critic private-card token is out of vocabulary")
        if visible_private_owners.size and np.any(
            (visible_private_owners < 0) | (visible_private_owners > 1)
        ):
            raise ValueError("visible Critic private-card owner must be 0 or 1")
        if visible_private_slots.size and np.any(
            (visible_private_slots < 0)
            | (visible_private_slots >= self.config.private_slot_count)
        ):
            raise ValueError("visible Critic private-card slot is out of range")
        return arrays

    def encode(
        self, value: CriticPrivateObservation | Mapping[str, Any]
    ) -> dict[str, Tensor]:
        return self.encode_sequence([value])

    def encode_sequence(
        self,
        values: Sequence[CriticPrivateObservation | Mapping[str, Any]],
    ) -> dict[str, Tensor]:
        if not values:
            raise ValueError("Critic observation sequence cannot be empty")
        normalized = [
            self._validated(
                value if isinstance(value, CriticPrivateObservation)
                else CriticPrivateObservation.from_mapping(value)
            )
            for value in values
        ]
        steps = len(normalized)
        max_entities = max(1, *(row["entity_tokens"].size for row in normalized))
        max_private = max(1, *(row["private_card_tokens"].size for row in normalized))

        grid = np.stack([row["grid"] for row in normalized]).astype(np.float32)
        scalars = np.stack([row["scalars"] for row in normalized]).astype(np.float32)
        entity_tokens = np.zeros((steps, max_entities), dtype=np.int64)
        entity_positions = np.zeros((steps, max_entities), dtype=np.int64)
        entity_relations = np.zeros((steps, max_entities), dtype=np.int64)
        entity_numeric = np.zeros(
            (steps, max_entities, self.config.entity_numeric_size), dtype=np.float32
        )
        entity_mask = np.zeros((steps, max_entities), dtype=np.bool_)
        private_tokens = np.zeros((steps, max_private), dtype=np.int64)
        private_owners = np.zeros((steps, max_private), dtype=np.int64)
        private_slots = np.zeros((steps, max_private), dtype=np.int64)
        private_mask = np.zeros((steps, max_private), dtype=np.bool_)

        for index, row in enumerate(normalized):
            entity_count = row["entity_tokens"].size
            private_count = row["private_card_tokens"].size
            entity_tokens[index, :entity_count] = row["entity_tokens"]
            entity_positions[index, :entity_count] = row["entity_positions"]
            entity_relations[index, :entity_count] = row["entity_relations"]
            entity_numeric[index, :entity_count] = row["entity_numeric"]
            entity_mask[index, :entity_count] = row["entity_mask"]
            private_tokens[index, :private_count] = row["private_card_tokens"]
            private_owners[index, :private_count] = row["private_card_owners"]
            private_slots[index, :private_count] = row["private_card_slots"]
            private_mask[index, :private_count] = row["private_card_mask"]

        result = {
            "grid": torch.from_numpy(grid).unsqueeze(0),
            "entity_tokens": torch.from_numpy(entity_tokens).unsqueeze(0),
            "entity_positions": torch.from_numpy(entity_positions).unsqueeze(0),
            "entity_relations": torch.from_numpy(entity_relations).unsqueeze(0),
            "entity_numeric": torch.from_numpy(entity_numeric).unsqueeze(0),
            "entity_mask": torch.from_numpy(entity_mask).unsqueeze(0),
            "private_card_tokens": torch.from_numpy(private_tokens).unsqueeze(0),
            "private_card_owners": torch.from_numpy(private_owners).unsqueeze(0),
            "private_card_slots": torch.from_numpy(private_slots).unsqueeze(0),
            "private_card_mask": torch.from_numpy(private_mask).unsqueeze(0),
            "scalars": torch.from_numpy(scalars).unsqueeze(0),
        }
        result = {name: tensor.to(self.device) for name, tensor in result.items()}
        validate_critic_inputs(result, self.config)
        return result


def validate_critic_inputs(
    values: Mapping[str, Tensor],
    config: PrivilegedCriticConfig,
    *,
    actor_latent: Tensor | None = None,
) -> tuple[int, int]:
    """Validate the exact tensor contract used by ``PrivilegedCritic.forward``."""

    required = {
        "grid", "entity_tokens", "entity_positions", "entity_relations",
        "entity_numeric", "entity_mask", "private_card_tokens",
        "private_card_owners", "private_card_slots", "private_card_mask", "scalars",
    }
    missing = sorted(required.difference(values))
    if missing:
        raise ValueError(f"Critic tensor batch is missing fields: {missing}")
    if any(not isinstance(values[name], Tensor) for name in required):
        raise TypeError("all Critic inputs must be torch tensors")
    grid = values["grid"]
    if grid.ndim != 5 or grid.shape[2:] != (config.public_grid_channels, 32, 18):
        raise ValueError("Critic grid tensor has the wrong shape")
    batch, steps = int(grid.shape[0]), int(grid.shape[1])
    prefix = (batch, steps)
    if values["scalars"].shape != (*prefix, config.scalar_size):
        raise ValueError("Critic scalar tensor has the wrong shape")
    entity_count = values["entity_tokens"].shape[-1]
    if entity_count < 1:
        raise ValueError("Critic entity tensor must retain one padding slot")
    for name in ("entity_tokens", "entity_positions", "entity_relations", "entity_mask"):
        if values[name].shape != (*prefix, entity_count):
            raise ValueError("Critic entity tensor prefixes differ")
    if values["entity_numeric"].shape != (
        *prefix, entity_count, config.entity_numeric_size
    ):
        raise ValueError("Critic entity numeric tensor has the wrong shape")
    private_count = values["private_card_tokens"].shape[-1]
    if not 1 <= private_count <= config.private_slot_count:
        raise ValueError("Critic private-card tensor has an invalid slot count")
    for name in (
        "private_card_tokens", "private_card_owners", "private_card_slots",
        "private_card_mask",
    ):
        if values[name].shape != (*prefix, private_count):
            raise ValueError("Critic private-card tensor prefixes differ")

    for name in ("grid", "entity_numeric", "scalars"):
        if not values[name].is_floating_point():
            raise TypeError(f"{name} must be floating point")
        if not bool(torch.isfinite(values[name]).all()):
            raise ValueError(f"{name} contains NaN/Inf")
    for name in (
        "entity_tokens", "entity_positions", "entity_relations",
        "private_card_tokens", "private_card_owners", "private_card_slots",
    ):
        if values[name].dtype != torch.long:
            raise TypeError(f"{name} must use torch.long")
    for name in ("entity_mask", "private_card_mask"):
        if values[name].dtype != torch.bool:
            raise TypeError(f"{name} must use torch.bool")
    if bool((values["entity_tokens"] < 0).any()) or bool(
        (values["entity_tokens"] >= config.card_vocab_size).any()
    ):
        raise ValueError("Critic entity token tensor is out of vocabulary")
    if bool((values["private_card_tokens"] < 0).any()) or bool(
        (values["private_card_tokens"] >= config.card_vocab_size).any()
    ):
        raise ValueError("Critic private-card token tensor is out of vocabulary")
    if actor_latent is not None:
        if actor_latent.shape != (*prefix, config.actor_latent_size):
            raise ValueError("Actor latent does not match the Critic prefix/width")
        if not actor_latent.is_floating_point() or not bool(
            torch.isfinite(actor_latent).all()
        ):
            raise ValueError("Actor latent must be finite floating point")
    return prefix


@dataclass(frozen=True)
class RecurrentChunkConfig:
    burn_in: int = 16
    unroll: int = 64
    gamma_per_tick: float = 0.99995
    gae_lambda_per_tick: float = 0.995

    def validate(self) -> None:
        if self.burn_in != 16 or self.unroll != 64:
            raise ValueError("v1 rollout contract requires burn_in=16 and unroll=64")
        if not 0.0 < self.gamma_per_tick <= 1.0:
            raise ValueError("gamma_per_tick must be in (0,1]")
        if not 0.0 < self.gae_lambda_per_tick <= 1.0:
            raise ValueError("gae_lambda_per_tick must be in (0,1]")


def _episode_without_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): item for key, item in value.items() if key != "content_sha256"}


def _validate_complete_episode(
    value: Mapping[str, Any] | LearnerEpisodeBuffer,
) -> tuple[dict[str, Any], EpisodeHeader, list[DecisionRecord]]:
    frozen = value.freeze() if isinstance(value, LearnerEpisodeBuffer) else dict(value)
    if frozen.get("schema_version") != 1 or frozen.get("kind") != (
        "cr_native_expert_selfplay_learner_episode_v1"
    ):
        raise ValueError("unsupported learner episode schema")
    expected_digest = frozen.get("content_sha256")
    if not isinstance(expected_digest, str) or not _SHA256.fullmatch(expected_digest):
        raise ValueError("learner episode has no valid content hash")
    actual_digest = _sha256_bytes(_canonical_json(_episode_without_digest(frozen)))
    if actual_digest != expected_digest:
        raise ValueError("learner episode content hash mismatch")
    try:
        header = EpisodeHeader(**dict(frozen["header"]))
    except (KeyError, TypeError) as error:
        raise ValueError("invalid learner episode header") from error
    header.validate()
    raw_decisions = frozen.get("decisions")
    if not isinstance(raw_decisions, list) or not raw_decisions:
        raise ValueError("learner episode is empty")
    decisions: list[DecisionRecord] = []
    previous_tick = -1
    for index, raw in enumerate(raw_decisions):
        try:
            decision = DecisionRecord(**dict(raw))
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid learner decision {index}") from error
        decision.validate(header.learner_side)
        if decision.tick <= previous_tick:
            raise ValueError("decision ticks must be strictly increasing")
        if decision.truncated:
            raise ValueError("time-truncated episode cannot become a PPO rollout")
        if decision.terminated != (index == len(raw_decisions) - 1):
            raise ValueError("only the final learner decision may be terminal")
        values = (
            decision.old_logp_total, decision.old_logp_timing,
            decision.old_logp_action_type, decision.old_logp_slot,
            decision.old_logp_position, decision.reward_damage_dealt,
            decision.reward_damage_received, decision.reward_towers_dealt,
            decision.reward_towers_received, decision.reward_terminal,
            decision.reward_total, decision.value,
        )
        if not all(math.isfinite(item) for item in values):
            raise ValueError("learner decision contains NaN/Inf")
        previous_tick = decision.tick
        decisions.append(decision)
    return frozen, header, decisions


class LearnerEpisodeChunker:
    """Turn complete learner episodes into recurrent PPO chunks."""

    def __init__(self, config: RecurrentChunkConfig | None = None) -> None:
        self.config = config or RecurrentChunkConfig()
        self.config.validate()

    def chunk(
        self,
        episode: Mapping[str, Any] | LearnerEpisodeBuffer,
        *,
        step_payloads: Sequence[Mapping[str, Any]] | None = None,
        validate_step_payloads: bool = True,
    ) -> list[dict[str, Any]]:
        frozen, header, decisions = _validate_complete_episode(episode)
        if step_payloads is not None and len(step_payloads) != len(decisions):
            raise ValueError("step payload count does not match learner decisions")
        if step_payloads is not None and validate_step_payloads:
            # Hashing the payload now also rejects non-finite and unsupported values.
            _semantic_digest(list(step_payloads))

        rewards = np.asarray([row.reward_total for row in decisions], dtype=np.float32)
        values = np.asarray([row.value for row in decisions], dtype=np.float32)
        terminated = np.asarray([row.terminated for row in decisions], dtype=np.bool_)
        delta_ticks = np.asarray([row.delta_ticks for row in decisions], dtype=np.int64)
        advantages, returns = variable_time_gae(
            rewards,
            values,
            terminated,
            delta_ticks,
            bootstrap_value=0.0,
            gamma_per_tick=self.config.gamma_per_tick,
            gae_lambda_per_tick=self.config.gae_lambda_per_tick,
        )
        if not np.isfinite(advantages).all() or not np.isfinite(returns).all():
            raise ValueError("GAE produced NaN/Inf")

        chunks: list[dict[str, Any]] = []
        for loss_start in range(0, len(decisions), self.config.unroll):
            loss_end = min(len(decisions), loss_start + self.config.unroll)
            sequence_start = max(0, loss_start - self.config.burn_in)
            sequence_end = loss_end
            burn_count = loss_start - sequence_start
            sequence_decisions = decisions[sequence_start:sequence_end]
            loss_mask = torch.zeros(len(sequence_decisions), dtype=torch.bool)
            loss_mask[burn_count:] = True
            chunk: dict[str, Any] = {
                "schema_version": SHARD_SCHEMA_VERSION,
                "kind": CHUNK_KIND,
                "episode_id": header.episode_id,
                "episode_content_sha256": frozen["content_sha256"],
                "header": asdict(header),
                "chunk_index": len(chunks),
                "sequence_start": sequence_start,
                "sequence_end": sequence_end,
                "loss_start": loss_start,
                "loss_end": loss_end,
                "burn_in": burn_count,
                "unroll": loss_end - loss_start,
                "initial_hidden_sha256": header.initial_hidden_sha256,
                "decisions": [asdict(row) for row in sequence_decisions],
                "advantages": torch.from_numpy(
                    advantages[sequence_start:sequence_end].copy()
                ),
                "returns": torch.from_numpy(returns[sequence_start:sequence_end].copy()),
                "loss_mask": loss_mask,
            }
            if step_payloads is not None:
                # Exact recurrent state is only required at the beginning of
                # a chunk.  Keeping a hidden pair on every decision inflated
                # native rollouts and retained the same state repeatedly in
                # overlapping burn-in windows.
                selected_payloads: list[dict[str, Any]] = []
                for offset, raw_payload in enumerate(
                    step_payloads[sequence_start:sequence_end]
                ):
                    payload = dict(raw_payload)
                    actor_inputs = payload.get("actor_inputs")
                    if isinstance(actor_inputs, Mapping):
                        actor_inputs = dict(actor_inputs)
                        if offset:
                            actor_inputs.pop("hidden", None)
                        payload["actor_inputs"] = actor_inputs
                    selected_payloads.append(payload)
                chunk["step_payloads"] = selected_payloads
            chunks.append(chunk)
        return chunks


def _canonical_semantic(value: Any) -> Any:
    """Canonical, finite-only description used to hash a torch payload."""

    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, Tensor):
        tensor = value.detach().cpu()
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError("torch shard contains NaN/Inf")
        # dtype reinterpretation requires a real trailing dimension.  Online
        # rollout metadata intentionally contains scalar (0-dim) tensors for
        # log-probability components and values, so flatten before viewing the
        # underlying bytes.  This also works for dtypes (for example bfloat16)
        # that NumPy cannot represent directly.
        if tensor.numel() == 0:
            raw = b""
        else:
            byte_source = tensor.reshape(-1).clone(
                memory_format=torch.contiguous_format
            )
            raw = byte_source.view(torch.uint8).numpy().tobytes()
        return {
            "__tensor__": True,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "sha256": _sha256_bytes(raw),
        }
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        _require_finite(array, name="numpy shard array")
        return {
            "__ndarray__": True,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": _sha256_bytes(array.tobytes()),
        }
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("torch shard mapping keys must be strings")
            result[key] = _canonical_semantic(item)
        return {key: result[key] for key in sorted(result)}
    if isinstance(value, (list, tuple)):
        return [_canonical_semantic(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("torch shard contains NaN/Inf")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, bytes):
        return {"__bytes__": base64.b64encode(value).decode("ascii")}
    raise TypeError(f"unsupported rollout shard value: {type(value).__name__}")


def _semantic_digest(value: Any) -> str:
    canonical = _canonical_semantic(value)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _fsync_file(path: Path) -> None:
    # Windows requires a writable file handle for FlushFileBuffers/os.fsync.
    with path.open("r+b") as source:
        os.fsync(source.fileno())


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        # Windows does not fsync directory handles; the atomic rename still holds.
        pass
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class ShardWriteResult:
    shard_uuid: str
    directory: Path
    manifest_path: Path
    torch_path: Path
    content_sha256: str
    payload_sha256: str
    torch_sha256: str
    created: bool
    ledger_recorded: bool | None


class ImmutableRolloutShardWriter:
    """Atomically publish immutable learner-only rollout shards."""

    def __init__(
        self,
        root: Path | str,
        batch_manifest: BatchManifest,
        *,
        ledger: RolloutLedger | None = None,
        chunker: LearnerEpisodeChunker | None = None,
    ) -> None:
        batch_manifest.validate()
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.batch_manifest = batch_manifest
        self.ledger = ledger
        self.chunker = chunker or LearnerEpisodeChunker()

    def _bind_episode(self, header: EpisodeHeader) -> None:
        manifest = self.batch_manifest
        if header.batch_id != manifest.batch_id:
            raise ValueError("episode batch_id does not match BatchManifest")
        if header.behavior_policy_version != manifest.policy_version:
            raise ValueError("episode policy version does not match BatchManifest")
        if header.behavior_actor_sha256 != manifest.behavior_actor_sha256:
            raise ValueError("episode Actor hash does not match BatchManifest")

    @staticmethod
    def _load_existing(
        directory: Path,
        *,
        expected_payload_sha256: str,
        expected_manifest_digest: str,
    ) -> ShardWriteResult:
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError("immutable shard path exists without manifest.json")
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        if metadata.get("payload_sha256") != expected_payload_sha256:
            raise RuntimeError("duplicate shard UUID has conflicting payload")
        if metadata.get("batch_manifest_sha256") != expected_manifest_digest:
            raise RuntimeError("duplicate shard UUID has conflicting BatchManifest")
        verified = verify_rollout_shard(directory)
        return ShardWriteResult(
            shard_uuid=str(metadata["shard_uuid"]),
            directory=directory,
            manifest_path=manifest_path,
            torch_path=directory / "rollout.pt",
            content_sha256=str(metadata["content_sha256"]),
            payload_sha256=str(metadata["payload_sha256"]),
            torch_sha256=str(metadata["torch_sha256"]),
            created=False,
            ledger_recorded=None,
        )

    def write(
        self,
        shard_uuid: str,
        episodes: Sequence[Mapping[str, Any] | LearnerEpisodeBuffer],
        *,
        step_payloads_by_episode: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    ) -> ShardWriteResult:
        if not _SAFE_SHARD_ID.fullmatch(shard_uuid):
            raise ValueError("unsafe rollout shard UUID")
        if not episodes:
            raise ValueError("rollout shard cannot be empty")
        if len(episodes) > self.batch_manifest.episode_count:
            raise ValueError("shard episode count exceeds BatchManifest batch size")

        episode_rows: list[dict[str, Any]] = []
        seen_episode_ids: set[str] = set()
        for episode in episodes:
            frozen, header, _decisions = _validate_complete_episode(episode)
            self._bind_episode(header)
            if header.episode_id in seen_episode_ids:
                raise ValueError("rollout shard contains duplicate episode IDs")
            seen_episode_ids.add(header.episode_id)
            sidecars = (
                None if step_payloads_by_episode is None
                else step_payloads_by_episode.get(header.episode_id)
            )
            # The complete shard semantic digest below validates every selected
            # sidecar tensor.  Avoid hashing the same data once here and then a
            # second time after it has been arranged into recurrent chunks.
            chunks = self.chunker.chunk(
                frozen,
                step_payloads=sidecars,
                validate_step_payloads=False,
            )
            episode_rows.append({
                "episode_id": header.episode_id,
                "episode_content_sha256": frozen["content_sha256"],
                "header": asdict(header),
                "decision_count": len(frozen["decisions"]),
                "chunks": chunks,
            })
        if step_payloads_by_episode is not None:
            unknown = set(step_payloads_by_episode).difference(seen_episode_ids)
            if unknown:
                raise ValueError(f"step payloads reference unknown episodes: {sorted(unknown)}")

        manifest_digest = self.batch_manifest.digest()
        payload: dict[str, Any] = {
            "schema_version": SHARD_SCHEMA_VERSION,
            "kind": SHARD_KIND,
            "shard_uuid": shard_uuid,
            "batch_manifest_sha256": manifest_digest,
            "episodes": episode_rows,
        }
        payload_digest = _semantic_digest(payload)
        final_directory = self.root / shard_uuid
        if final_directory.exists():
            result = self._load_existing(
                final_directory,
                expected_payload_sha256=payload_digest,
                expected_manifest_digest=manifest_digest,
            )
            ledger_recorded = None
            if self.ledger is not None:
                ledger_recorded = self.ledger.record_shard(
                    self.batch_manifest.batch_id,
                    shard_uuid=shard_uuid,
                    content_sha256=result.content_sha256,
                )
            return ShardWriteResult(**{
                **result.__dict__, "ledger_recorded": ledger_recorded
            })

        temporary = Path(tempfile.mkdtemp(prefix=f".{shard_uuid}.", dir=self.root))
        try:
            torch_path = temporary / "rollout.pt"
            torch.save(payload, torch_path)
            _fsync_file(torch_path)
            torch_digest = _sha256_file(torch_path)
            metadata_without_digest: dict[str, Any] = {
                "schema_version": SHARD_SCHEMA_VERSION,
                "kind": SHARD_KIND,
                "shard_uuid": shard_uuid,
                "batch_manifest": asdict(self.batch_manifest),
                "batch_manifest_sha256": manifest_digest,
                "episode_ids": sorted(seen_episode_ids),
                "episode_count": len(episode_rows),
                "decision_count": sum(row["decision_count"] for row in episode_rows),
                "chunk_count": sum(len(row["chunks"]) for row in episode_rows),
                "payload_sha256": payload_digest,
                "torch_sha256": torch_digest,
                "torch_file": "rollout.pt",
            }
            content_digest = _sha256_bytes(_canonical_json(metadata_without_digest))
            metadata = {**metadata_without_digest, "content_sha256": content_digest}
            manifest_path = temporary / "manifest.json"
            with manifest_path.open("x", encoding="utf-8", newline="\n") as output:
                json.dump(metadata, output, sort_keys=True, separators=(",", ":"))
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            _fsync_directory(temporary)
            try:
                temporary.rename(final_directory)
            except OSError:
                if not final_directory.exists():
                    raise
                result = self._load_existing(
                    final_directory,
                    expected_payload_sha256=payload_digest,
                    expected_manifest_digest=manifest_digest,
                )
                return result
            _fsync_directory(self.root)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

        ledger_recorded: bool | None = None
        if self.ledger is not None:
            ledger_recorded = self.ledger.record_shard(
                self.batch_manifest.batch_id,
                shard_uuid=shard_uuid,
                content_sha256=content_digest,
            )
        return ShardWriteResult(
            shard_uuid=shard_uuid,
            directory=final_directory,
            manifest_path=final_directory / "manifest.json",
            torch_path=final_directory / "rollout.pt",
            content_sha256=content_digest,
            payload_sha256=payload_digest,
            torch_sha256=torch_digest,
            created=True,
            ledger_recorded=ledger_recorded,
        )


def verify_rollout_shard(
    directory: Path | str,
    *,
    expected_batch_manifest: BatchManifest | None = None,
    return_payload: bool = False,
    mmap: bool = False,
    verify_semantic_digest: bool = True,
    known_torch_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify hashes, manifest binding, and learner-only terminal coverage."""

    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    torch_path = directory / "rollout.pt"
    if not manifest_path.is_file() or not torch_path.is_file():
        raise ValueError("rollout shard is missing JSON or torch payload")
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    stored_content_digest = metadata.get("content_sha256")
    if not isinstance(stored_content_digest, str) or not _SHA256.fullmatch(
        stored_content_digest
    ):
        raise ValueError("rollout shard has an invalid content hash")
    metadata_without_digest = {
        key: value for key, value in metadata.items() if key != "content_sha256"
    }
    if _sha256_bytes(_canonical_json(metadata_without_digest)) != stored_content_digest:
        raise ValueError("rollout shard JSON content hash mismatch")
    if known_torch_sha256 is None:
        actual_torch_sha256 = _sha256_file(torch_path)
    else:
        if not _SHA256.fullmatch(known_torch_sha256):
            raise ValueError("known rollout torch hash is invalid")
        actual_torch_sha256 = known_torch_sha256
    if actual_torch_sha256 != metadata.get("torch_sha256"):
        raise ValueError("rollout shard torch hash mismatch")
    try:
        manifest = BatchManifest(**dict(metadata["batch_manifest"]))
    except (KeyError, TypeError) as error:
        raise ValueError("rollout shard BatchManifest is invalid") from error
    if manifest.digest() != metadata.get("batch_manifest_sha256"):
        raise ValueError("rollout shard BatchManifest digest mismatch")
    if expected_batch_manifest is not None and (
        expected_batch_manifest.digest() != manifest.digest()
    ):
        raise ValueError("rollout shard is bound to another BatchManifest")

    payload = torch.load(
        torch_path, map_location="cpu", weights_only=False, mmap=mmap
    )
    if verify_semantic_digest and (
        _semantic_digest(payload) != metadata.get("payload_sha256")
    ):
        raise ValueError("rollout shard semantic payload hash mismatch")
    if payload.get("batch_manifest_sha256") != manifest.digest():
        raise ValueError("torch payload is not bound to its BatchManifest")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("rollout shard payload has no episodes")
    episode_ids = [str(episode.get("episode_id", "")) for episode in episodes]
    if (
        len(set(episode_ids)) != len(episode_ids)
        or metadata.get("episode_count") != len(episodes)
        or metadata.get("episode_ids") != sorted(episode_ids)
        or len(episodes) > manifest.episode_count
    ):
        raise ValueError("rollout shard episode coverage differs from its manifest")
    for episode in episodes:
        header = EpisodeHeader(**dict(episode["header"]))
        header.validate()
        if header.episode_id != episode["episode_id"]:
            raise ValueError("stored episode identity differs from its header")
        if header.batch_id != manifest.batch_id:
            raise ValueError("stored episode has the wrong batch_id")
        if header.behavior_policy_version != manifest.policy_version:
            raise ValueError("stored episode has the wrong policy version")
        if header.behavior_actor_sha256 != manifest.behavior_actor_sha256:
            raise ValueError("stored episode has the wrong Actor hash")
        chunks = episode.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("stored episode has no recurrent chunks")
        terminal_count = 0
        expected_loss_start = 0
        reconstructed_decisions: list[dict[str, Any]] = []
        for chunk in chunks:
            if chunk.get("kind") != CHUNK_KIND:
                raise ValueError("stored recurrent chunk has the wrong kind")
            if chunk.get("loss_start") != expected_loss_start:
                raise ValueError("stored recurrent chunks have a coverage gap")
            expected_loss_start = int(chunk["loss_end"])
            if not 0 <= int(chunk["burn_in"]) <= 16:
                raise ValueError("stored recurrent chunk has invalid burn-in")
            if not 1 <= int(chunk["unroll"]) <= 64:
                raise ValueError("stored recurrent chunk has invalid unroll")
            decisions = chunk.get("decisions", [])
            mask = chunk.get("loss_mask")
            if not isinstance(mask, Tensor) or mask.dtype != torch.bool:
                raise ValueError("stored recurrent chunk has invalid loss mask")
            if len(decisions) != mask.numel():
                raise ValueError("stored recurrent chunk decision/mask lengths differ")
            for decision, is_loss in zip(decisions, mask.tolist()):
                if int(decision["side"]) != header.learner_side:
                    raise ValueError("opponent trajectory entered a stored PPO chunk")
                if bool(is_loss) and bool(decision["terminated"]):
                    terminal_count += 1
                if bool(is_loss):
                    reconstructed_decisions.append(dict(decision))
                if bool(decision["truncated"]):
                    raise ValueError("truncated trajectory entered a stored PPO chunk")
            for name in ("advantages", "returns"):
                value = chunk.get(name)
                if not isinstance(value, Tensor) or not bool(torch.isfinite(value).all()):
                    raise ValueError(f"stored recurrent chunk {name} is invalid")
        if expected_loss_start != int(episode["decision_count"]):
            raise ValueError("stored recurrent chunks do not cover the complete episode")
        if terminal_count != 1:
            raise ValueError("stored episode does not contain exactly one terminal target")
        reconstructed = {
            "schema_version": 1,
            "kind": "cr_native_expert_selfplay_learner_episode_v1",
            "header": asdict(header),
            "decisions": reconstructed_decisions,
            "content_sha256": episode["episode_content_sha256"],
        }
        _validate_complete_episode(reconstructed)
    if return_payload:
        metadata = dict(metadata)
        metadata["_payload"] = payload
    return metadata


# Concise aliases for coordinator/learner call sites.
AtomicRolloutShardWriter = ImmutableRolloutShardWriter
RecurrentEpisodeChunker = LearnerEpisodeChunker
