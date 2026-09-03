"""Stage-1 Critic warm-up for expert-initialized self-play.

The Actor is deliberately treated as immutable data in this module.  Every
recurrent chunk is replayed through ``forward_with_features`` to obtain the
pre-head recurrent latent, but that replay happens with gradients disabled and
the latent is detached before it reaches the privileged Critic.

``LearnerEpisodeChunker`` supplies ``returns`` and ``loss_mask``.  Its optional
``step_payloads`` sidecar must contain ``actor_inputs`` and ``critic_inputs``
for every decision.  Auxiliary targets can be stored either in each sidecar as
``critic_targets`` or once on the chunk as ``critic_targets``.  The five heads
are always optimized together; silently omitting an auxiliary target is not
allowed.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
from hashlib import sha256
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from expert_v1.training_v1.model import ExpertPolicyConfig, RecurrentExpertPolicy

from . import CHECKPOINT_KIND
from .actor_adapter import actor_state_digest
from .critic import CriticOutput, ExpertActorCritic, PrivilegedCritic, PrivilegedCriticConfig
from .losses import CriticTargets, critic_loss, explained_variance


CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CriticTrainingConfig:
    """Numerical contract for one Stage-1 Critic update."""

    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    adam_epsilon: float = 1.0e-5
    max_grad_norm: float = 1.0
    value_coefficient: float = 1.0
    auxiliary_coefficient: float = 0.10
    use_bf16_autocast: bool = True
    chunk_batch_size: int = 8
    retain_checkpoints: int = 3

    def validate(self) -> None:
        positive = {
            "learning_rate": self.learning_rate,
            "adam_epsilon": self.adam_epsilon,
            "max_grad_norm": self.max_grad_norm,
            "value_coefficient": self.value_coefficient,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if not math.isfinite(self.auxiliary_coefficient) or self.auxiliary_coefficient < 0:
            raise ValueError("auxiliary_coefficient must be finite and non-negative")
        if self.retain_checkpoints < 1:
            raise ValueError("retain_checkpoints must be positive")
        if self.chunk_batch_size < 1:
            raise ValueError("chunk_batch_size must be positive")


# Ranks after removing an optional batch prefix while retaining the singleton
# online time axis.  This is the same public shape convention as PolicyRequest.
_ACTOR_INPUT_RANKS = {
    "grid": 4,
    "public_scalars": 2,
    "own_deck_tokens": 2,
    "hand_tokens": 2,
    "next_card_token": 1,
    "revealed_enemy_tokens": 2,
    "ability_tokens": 2,
    "delta_ticks": 1,
    "entity_tokens": 2,
    "entity_positions": 2,
    "entity_relations": 2,
    "entity_numeric": 3,
    "entity_mask": 2,
    "previous_event_card_token": 1,
    "previous_event_side": 1,
    "previous_event_position": 1,
}
_ACTOR_RAGGED = {
    "entity_tokens",
    "entity_positions",
    "entity_relations",
    "entity_numeric",
    "entity_mask",
}
_CRITIC_INPUT_RANKS = {
    "grid": 4,
    "entity_tokens": 2,
    "entity_positions": 2,
    "entity_relations": 2,
    "entity_numeric": 3,
    "entity_mask": 2,
    "private_card_tokens": 2,
    "private_card_owners": 2,
    "private_card_slots": 2,
    "private_card_mask": 2,
    "scalars": 2,
}
_CRITIC_ENTITY_RAGGED = {
    "entity_tokens",
    "entity_positions",
    "entity_relations",
    "entity_numeric",
    "entity_mask",
}
_CRITIC_PRIVATE_RAGGED = {
    "private_card_tokens",
    "private_card_owners",
    "private_card_slots",
    "private_card_mask",
}
_TARGET_NAMES = (
    "wdl_class",
    "crown_difference",
    "tower_hp_difference",
    "future_damage",
)


def _module_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    for value in module.parameters():
        return value.device, value.dtype
    for value in module.buffers():
        return value.device, value.dtype
    return torch.device("cpu"), torch.float32


def _tensor(value: Any, *, name: str) -> Tensor:
    if isinstance(value, Tensor):
        result = value.detach()
    else:
        try:
            result = torch.as_tensor(value)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} must be tensor-like") from error
    if result.is_floating_point() and not bool(torch.isfinite(result).all()):
        raise FloatingPointError(f"{name} contains NaN/Inf")
    return result


def _one_online_step(value: Any, *, name: str, rank: int) -> Tensor:
    result = _tensor(value, name=name)
    # Accept [batch=1,time=1,...], [time=1,...], or a value with the
    # singleton time axis omitted.
    if result.ndim == rank + 1:
        if result.shape[0] != 1:
            raise ValueError(f"{name} contains more than one batch row")
        result = result.squeeze(0)
    if result.ndim == rank - 1:
        result = result.unsqueeze(0)
    if result.ndim != rank or result.shape[0] != 1:
        raise ValueError(f"{name} must contain exactly one online time step")
    return result


def _pad_axis_one(values: Sequence[Tensor], *, name: str) -> list[Tensor]:
    maximum = max(1, *(int(value.shape[1]) for value in values))
    result: list[Tensor] = []
    for value in values:
        if value.shape[1] == maximum:
            result.append(value)
            continue
        shape = list(value.shape)
        shape[1] = maximum - int(value.shape[1])
        result.append(torch.cat((value, value.new_zeros(shape)), dim=1))
    return result


def _collate_inputs(
    rows: Sequence[Mapping[str, Any]],
    *,
    ranks: Mapping[str, int],
    ragged_groups: Sequence[set[str]],
    device: torch.device,
    floating_dtype: torch.dtype,
    kind: str,
) -> dict[str, Tensor | None]:
    if not rows:
        raise ValueError(f"{kind} sequence cannot be empty")
    key_set = set(rows[0])
    if any(set(row) != key_set for row in rows[1:]):
        raise ValueError(f"{kind} steps have different input fields")
    unknown = key_set.difference(ranks)
    if unknown:
        raise ValueError(f"unknown {kind} input fields: {sorted(unknown)}")
    output: dict[str, Tensor | None] = {}
    for name in key_set:
        raw = [row[name] for row in rows]
        if all(value is None for value in raw):
            output[name] = None
            continue
        if any(value is None for value in raw):
            raise ValueError(f"{kind} input {name} is None for only part of a chunk")
        values = [
            _one_online_step(value, name=f"{kind}.{name}", rank=ranks[name])
            for value in raw
        ]
        for group in ragged_groups:
            if name in group:
                values = _pad_axis_one(values, name=name)
                break
        shapes = {tuple(value.shape[1:]) for value in values}
        if len(shapes) != 1:
            raise ValueError(f"{kind} input {name} has incompatible step shapes")
        sequence = torch.cat(values, dim=0).unsqueeze(0)
        if sequence.is_floating_point():
            sequence = sequence.to(device=device, dtype=floating_dtype)
        else:
            sequence = sequence.to(device=device)
        if name.endswith("mask"):
            sequence = sequence.bool()
        output[name] = sequence
    return output


def _initial_hidden(
    chunk: Mapping[str, Any],
    actor_rows: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor] | None:
    value = chunk.get("initial_hidden")
    if value is None and actor_rows and "hidden" in actor_rows[0]:
        value = actor_rows[0]["hidden"]
    if value is None:
        return None
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("initial_hidden must be an (h, c) pair")
    result = tuple(
        _tensor(item, name=f"initial_hidden[{index}]").to(device=device, dtype=dtype)
        for index, item in enumerate(value)
    )
    if result[0].ndim != 3 or result[0].shape != result[1].shape:
        raise ValueError("initial_hidden tensors have incompatible shapes")
    return result  # type: ignore[return-value]


def _target_sequence(chunk: Mapping[str, Any], payloads: Sequence[Mapping[str, Any]]) -> dict[str, Tensor]:
    chunk_targets = chunk.get("critic_targets")
    if chunk_targets is not None:
        if not isinstance(chunk_targets, Mapping):
            raise TypeError("chunk critic_targets must be a mapping")
        missing = sorted(set(_TARGET_NAMES).difference(chunk_targets))
        if missing:
            raise ValueError(f"chunk critic_targets is missing fields: {missing}")
        return {
            name: _tensor(chunk_targets[name], name=f"critic_targets.{name}")
            for name in _TARGET_NAMES
        }

    columns: dict[str, list[Tensor]] = {name: [] for name in _TARGET_NAMES}
    for index, payload in enumerate(payloads):
        targets = payload.get("critic_targets", payload.get("targets"))
        if not isinstance(targets, Mapping):
            raise ValueError(f"step_payloads[{index}] has no critic_targets")
        missing = sorted(set(_TARGET_NAMES).difference(targets))
        if missing:
            raise ValueError(
                f"step_payloads[{index}].critic_targets is missing fields: {missing}"
            )
        for name in _TARGET_NAMES:
            columns[name].append(_tensor(targets[name], name=f"critic_targets.{name}"))
    result: dict[str, Tensor] = {}
    for name, values in columns.items():
        if name == "future_damage":
            normalized = [value.reshape(-1) for value in values]
            if any(value.numel() != 2 for value in normalized):
                raise ValueError("future_damage target must have two values per step")
            result[name] = torch.stack(normalized)
        else:
            if any(value.numel() != 1 for value in values):
                raise ValueError(f"{name} target must be scalar per step")
            result[name] = torch.stack([value.reshape(()) for value in values])
    return result


def _critic_targets(
    chunk: Mapping[str, Any],
    payloads: Sequence[Mapping[str, Any]],
    *,
    steps: int,
    device: torch.device,
) -> CriticTargets:
    returns = _tensor(chunk.get("returns"), name="chunk.returns").float().reshape(-1)
    mask = _tensor(chunk.get("loss_mask"), name="chunk.loss_mask").bool().reshape(-1)
    if returns.numel() != steps or mask.numel() != steps:
        raise ValueError("chunk returns/loss_mask length differs from step_payloads")
    if not bool(mask.any()):
        raise ValueError("chunk loss_mask is empty")
    auxiliary = _target_sequence(chunk, payloads)
    for name, value in auxiliary.items():
        expected = (steps, 2) if name == "future_damage" else (steps,)
        if tuple(value.shape) != expected:
            raise ValueError(f"{name} target shape {tuple(value.shape)} != {expected}")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"{name} target contains NaN/Inf")
    wdl = auxiliary["wdl_class"]
    if wdl.is_floating_point() and not bool(torch.equal(wdl, wdl.round())):
        raise ValueError("wdl_class target must contain integer classes")
    wdl = wdl.long()
    if bool(((wdl < 0) | (wdl > 2)).any()):
        raise ValueError("wdl_class target must be in [0, 2]")
    return CriticTargets(
        returns=returns.unsqueeze(0).to(device),
        wdl_class=wdl.unsqueeze(0).to(device),
        crown_difference=auxiliary["crown_difference"].float().unsqueeze(0).to(device),
        tower_hp_difference=auxiliary["tower_hp_difference"].float().unsqueeze(0).to(device),
        future_damage=auxiliary["future_damage"].float().unsqueeze(0).to(device),
        loss_mask=mask.unsqueeze(0).to(device),
    )


def _float_output(value: CriticOutput) -> CriticOutput:
    return CriticOutput(*(item.float() for item in value))


def _batch_tensors(values: Sequence[Tensor], *, name: str) -> Tensor:
    """Join prepared [1,T,...] chunks, padding only the ragged token axis."""

    if not values or any(value.ndim != values[0].ndim for value in values):
        raise ValueError(f"{name} prepared chunks have incompatible ranks")
    rank = values[0].ndim
    if rank < 2 or any(value.shape[0] != 1 for value in values):
        raise ValueError(f"{name} prepared chunks must have singleton batch axes")
    maxima = [max(int(value.shape[axis]) for value in values) for axis in range(rank)]
    for axis in range(1, rank):
        sizes = {int(value.shape[axis]) for value in values}
        if len(sizes) > 1 and axis != 2:
            raise ValueError(
                f"{name} differs on non-ragged prepared axis {axis}: {sorted(sizes)}"
            )
    padded: list[Tensor] = []
    for value in values:
        if list(value.shape) == maxima:
            padded.append(value)
            continue
        # torch.nn.functional.pad specifies dimensions from last to first.
        padding: list[int] = []
        for axis in reversed(range(rank)):
            padding.extend((0, maxima[axis] - int(value.shape[axis])))
        padded.append(torch.nn.functional.pad(value, padding))
    return torch.cat(padded, dim=0)


def _batch_input_mappings(
    values: Sequence[Mapping[str, Tensor | tuple[Tensor, Tensor] | None]],
    *,
    kind: str,
) -> dict[str, Tensor | tuple[Tensor, Tensor] | None]:
    if not values:
        raise ValueError(f"{kind} prepared batch cannot be empty")
    keys = set(values[0])
    if any(set(value) != keys for value in values[1:]):
        raise ValueError(f"{kind} prepared chunks have different fields")
    result: dict[str, Tensor | tuple[Tensor, Tensor] | None] = {}
    for name in keys:
        column = [value[name] for value in values]
        if all(value is None for value in column):
            result[name] = None
        elif name == "hidden":
            if any(
                not isinstance(value, (tuple, list)) or len(value) != 2
                for value in column
            ):
                raise ValueError("prepared hidden state is only partially present")
            result[name] = tuple(
                torch.cat([value[index] for value in column], dim=1)  # type: ignore[index]
                for index in range(2)
            )  # type: ignore[assignment]
        else:
            if any(not isinstance(value, Tensor) for value in column):
                raise TypeError(f"{kind}.{name} is not consistently tensor-valued")
            result[name] = _batch_tensors(
                column, name=f"{kind}.{name}"  # type: ignore[arg-type]
            )
    return result


def _batch_targets(values: Sequence[CriticTargets]) -> CriticTargets:
    if not values:
        raise ValueError("Critic target batch cannot be empty")
    return CriticTargets(*(
        torch.cat([getattr(value, name) for value in values], dim=0)
        for name in (
            "returns", "wdl_class", "crown_difference",
            "tower_hp_difference", "future_damage", "loss_mask",
        )
    ))


def _clone_state(module: nn.Module, *, fp32: bool = False) -> OrderedDict[str, Tensor]:
    result: OrderedDict[str, Tensor] = OrderedDict()
    for name, value in module.state_dict().items():
        copy = value.detach().cpu()
        if fp32 and copy.is_floating_point():
            copy = copy.float()
        result[name] = copy.clone()
    return result


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(value: Mapping[str, Any]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    # Checkpoints may be opened with a CUDA map location by external callers;
    # PyTorch's CPU and CUDA generators both require CPU ByteTensor state.
    torch.set_rng_state(value["torch_cpu"].detach().cpu())
    if torch.cuda.is_available() and value.get("torch_cuda"):
        torch.cuda.set_rng_state_all([
            item.detach().cpu() for item in value["torch_cuda"]
        ])


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_expert_actor(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
    inference_dtype: torch.dtype | None = None,
) -> tuple[RecurrentExpertPolicy, dict[str, Any]]:
    """Load a formal expert checkpoint and return its authenticated source reference."""

    path = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("expert checkpoint must contain a mapping")
    model_config = payload.get("model_config")
    model_state = payload.get("model_state")
    if not isinstance(model_config, Mapping) or not isinstance(model_state, Mapping):
        raise ValueError("expert checkpoint is missing model_config/model_state")
    actor = RecurrentExpertPolicy(ExpertPolicyConfig(**dict(model_config)))
    actor.load_state_dict(model_state, strict=True)
    if any(
        value.is_floating_point() and not bool(torch.isfinite(value).all())
        for value in actor.state_dict().values()
    ):
        raise FloatingPointError("expert Actor checkpoint contains NaN/Inf")
    actor = actor.to(device=device)
    if inference_dtype is not None:
        actor = actor.to(dtype=inference_dtype)
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    reference = {
        "path": str(path),
        "file_sha256": _file_sha256(path),
        "actor_sha256": actor_state_digest(actor),
        "source_global_step": payload.get("global_step"),
    }
    return actor, reference


class Stage1CriticTrainer:
    """A fail-closed, Actor-immutable Stage-1 Critic trainer."""

    def __init__(
        self,
        model: ExpertActorCritic,
        *,
        config: CriticTrainingConfig | None = None,
        device: torch.device | str | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        actor_source_reference: Mapping[str, Any] | str | Path | None = None,
        run_config: Mapping[str, Any] | None = None,
        global_update: int = 0,
    ) -> None:
        self.config = config or CriticTrainingConfig()
        self.config.validate()
        inferred_device, _ = _module_device_dtype(model)
        self.device = torch.device(device) if device is not None else inferred_device
        self.model = model.to(self.device)
        self.model.actor.eval()
        self.model.critic.train()
        for parameter in self.model.actor.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        for parameter in self.model.critic.parameters():
            parameter.requires_grad_(True)
        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.critic.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            eps=self.config.adam_epsilon,
        )
        if not self.optimizer.param_groups:
            raise ValueError("Critic optimizer has no parameter groups")
        actor_parameter_ids = {id(value) for value in self.model.actor.parameters()}
        if any(
            id(parameter) in actor_parameter_ids
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        ):
            raise ValueError("Stage-1 optimizer must not contain Actor parameters")
        if global_update < 0:
            raise ValueError("global_update must be non-negative")
        self.global_update = int(global_update)
        self.run_config = dict(run_config or {})
        self.actor_source_reference = self._normalize_source_reference(
            actor_source_reference
        )
        self.actor_sha256 = actor_state_digest(self.model.actor)

    @staticmethod
    def _normalize_source_reference(
        value: Mapping[str, Any] | str | Path | None,
    ) -> dict[str, Any] | None:
        if value is None:
            return None
        if isinstance(value, Mapping):
            return dict(value)
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise ValueError("Actor source reference does not exist")
        return {"path": str(path), "file_sha256": _file_sha256(path)}

    @classmethod
    def from_expert_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        critic_config: PrivilegedCriticConfig | None = None,
        critic_scalar_size: int = 32,
        config: CriticTrainingConfig | None = None,
        device: torch.device | str = "cpu",
        actor_inference_dtype: torch.dtype | None = None,
        run_config: Mapping[str, Any] | None = None,
    ) -> "Stage1CriticTrainer":
        actor, reference = load_expert_actor(
            checkpoint_path, device=device, inference_dtype=actor_inference_dtype
        )
        critic_config = critic_config or PrivilegedCriticConfig(
            actor_latent_size=actor.config.hidden_size,
            card_vocab_size=actor.config.card_vocab_size,
            scalar_size=critic_scalar_size,
        )
        critic = PrivilegedCritic(critic_config).to(device)
        return cls(
            ExpertActorCritic(actor, critic),
            config=config,
            device=device,
            actor_source_reference=reference,
            run_config=run_config,
        )

    def _prepare_chunk(
        self, chunk: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Tensor], CriticTargets]:
        payloads = chunk.get("step_payloads")
        if not isinstance(payloads, Sequence) or isinstance(payloads, (str, bytes)) or not payloads:
            raise ValueError("Critic training chunk has no step_payloads")
        if any(not isinstance(payload, Mapping) for payload in payloads):
            raise TypeError("every step_payload must be a mapping")
        actor_rows: list[Mapping[str, Any]] = []
        critic_rows: list[Mapping[str, Any]] = []
        for index, payload in enumerate(payloads):
            actor_inputs = payload.get("actor_inputs")
            critic_inputs = payload.get("critic_inputs")
            if not isinstance(actor_inputs, Mapping) or not isinstance(critic_inputs, Mapping):
                raise ValueError(
                    f"step_payloads[{index}] must contain actor_inputs and critic_inputs"
                )
            actor_rows.append(dict(actor_inputs))
            critic_rows.append(dict(critic_inputs))

        hidden = _initial_hidden(
            chunk,
            actor_rows,
            device=self.device,
            dtype=_module_device_dtype(self.model.actor)[1],
        )
        clean_actor_rows = [
            {name: value for name, value in row.items() if name != "hidden"}
            for row in actor_rows
        ]
        actor_inputs = _collate_inputs(
            clean_actor_rows,
            ranks=_ACTOR_INPUT_RANKS,
            ragged_groups=(_ACTOR_RAGGED,),
            device=self.device,
            floating_dtype=_module_device_dtype(self.model.actor)[1],
            kind="Actor",
        )
        actor_inputs["hidden"] = hidden
        critic_inputs = _collate_inputs(
            critic_rows,
            ranks=_CRITIC_INPUT_RANKS,
            ragged_groups=(_CRITIC_ENTITY_RAGGED, _CRITIC_PRIVATE_RAGGED),
            device=self.device,
            floating_dtype=torch.float32,
            kind="Critic",
        )
        if any(value is None for value in critic_inputs.values()):
            raise ValueError("Critic inputs cannot contain None")
        typed_critic = {name: value for name, value in critic_inputs.items() if value is not None}
        targets = _critic_targets(
            chunk, payloads, steps=len(payloads), device=self.device
        )
        return actor_inputs, typed_critic, targets

    def _chunk_batches(
        self, chunks: Sequence[Mapping[str, Any]]
    ) -> tuple[int, list[list[tuple[Mapping[str, Any], int]]]]:
        counts = [
            int(_tensor(chunk.get("loss_mask"), name="chunk.loss_mask")
                .bool().sum().item())
            for chunk in chunks
        ]
        total_steps = sum(counts)
        if total_steps < 1:
            raise ValueError("Stage-1 batch has no unmasked loss steps")
        by_length: dict[int, list[tuple[Mapping[str, Any], int]]] = {}
        for chunk, count in zip(chunks, counts, strict=True):
            payloads = chunk.get("step_payloads")
            length = len(payloads) if isinstance(payloads, Sequence) else -1
            by_length.setdefault(length, []).append((chunk, count))
        batches: list[list[tuple[Mapping[str, Any], int]]] = []
        for length in sorted(by_length, reverse=True):
            rows = by_length[length]
            for start in range(0, len(rows), self.config.chunk_batch_size):
                batches.append(rows[start:start + self.config.chunk_batch_size])
        return total_steps, batches

    def evaluate_chunks(
        self, chunks: Sequence[Mapping[str, Any]]
    ) -> dict[str, float | int]:
        """Evaluate the current Critic on fresh chunks without updating it."""

        if not chunks:
            raise ValueError("Stage-1 evaluation requires at least one chunk")
        total_steps, chunk_batches = self._chunk_batches(chunks)
        accumulator = {
            "loss": 0.0,
            "value_loss": 0.0,
            "wdl_loss": 0.0,
            "crown_loss": 0.0,
            "tower_hp_loss": 0.0,
            "future_damage_loss": 0.0,
            "explained_variance": 0.0,
        }
        ev_values: list[Tensor] = []
        ev_returns: list[Tensor] = []
        self.model.actor.eval()
        self.model.critic.eval()
        autocast_enabled = self.config.use_bf16_autocast and self.device.type in {
            "cpu", "cuda"
        }
        try:
            with torch.no_grad():
                for chunk_batch in chunk_batches:
                    prepared = [self._prepare_chunk(chunk) for chunk, _ in chunk_batch]
                    actor_inputs = _batch_input_mappings(
                        [row[0] for row in prepared], kind="Actor"
                    )
                    critic_inputs = _batch_input_mappings(
                        [row[1] for row in prepared], kind="Critic"
                    )
                    targets = _batch_targets([row[2] for row in prepared])
                    count = sum(row[1] for row in chunk_batch)
                    featured = self.model.actor.forward_with_features(**actor_inputs)
                    actor_latent = featured.pre_head_latent.detach()
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=torch.bfloat16,
                        enabled=autocast_enabled,
                    ):
                        output = self.model.critic(
                            actor_latent=actor_latent, **critic_inputs
                        )
                    output = _float_output(output)
                    if any(not bool(torch.isfinite(value).all()) for value in output):
                        raise FloatingPointError("Critic evaluation emitted NaN/Inf")
                    losses = critic_loss(
                        output,
                        targets,
                        value_coefficient=self.config.value_coefficient,
                        auxiliary_coefficient=self.config.auxiliary_coefficient,
                    )
                    values = (
                        losses.total,
                        losses.value,
                        losses.wdl,
                        losses.crown,
                        losses.tower_hp,
                        losses.future_damage,
                    )
                    if any(not bool(torch.isfinite(value)) for value in values):
                        raise FloatingPointError("Critic evaluation loss contains NaN/Inf")
                    weight = count / total_steps
                    for name, value in zip(
                        (
                            "loss", "value_loss", "wdl_loss", "crown_loss",
                            "tower_hp_loss", "future_damage_loss",
                        ),
                        values,
                        strict=True,
                    ):
                        accumulator[name] += float(value) * weight
                    mask = targets.loss_mask.bool()
                    ev_values.append(output.values[mask].detach().cpu())
                    ev_returns.append(targets.returns[mask].detach().cpu())
        finally:
            self.model.critic.train()
        joined_values = torch.cat(ev_values)
        joined_returns = torch.cat(ev_returns)
        accumulator["explained_variance"] = float(
            explained_variance(
                joined_values,
                joined_returns,
                torch.ones_like(joined_returns, dtype=torch.bool),
            )
        )
        return {
            **accumulator,
            "loss_steps": total_steps,
            "chunks": len(chunks),
            "chunk_batches": len(chunk_batches),
            "chunk_batch_size": self.config.chunk_batch_size,
            "bf16_autocast": int(autocast_enabled),
            "explained_variance_global": 1,
        }

    def train_update(self, chunks: Sequence[Mapping[str, Any]]) -> dict[str, float | int | str]:
        """Run exactly one optimizer update over one or more recurrent chunks."""

        if not chunks:
            raise ValueError("Stage-1 update requires at least one chunk")
        before_hash = actor_state_digest(self.model.actor)
        if before_hash != self.actor_sha256:
            raise RuntimeError("Actor changed before Stage-1 Critic update")
        if any(parameter.grad is not None for parameter in self.model.actor.parameters()):
            raise RuntimeError("Actor carried a gradient into Stage-1 Critic update")

        # Count masks on CPU, then prepare/train one chunk at a time.  Keeping
        # every recurrent chunk resident on the GPU made real multi-episode
        # shards scale memory with the entire rollout before the first
        # backward pass.
        total_steps, chunk_batches = self._chunk_batches(chunks)
        accumulator = {
            "loss": 0.0,
            "value_loss": 0.0,
            "wdl_loss": 0.0,
            "crown_loss": 0.0,
            "tower_hp_loss": 0.0,
            "future_damage_loss": 0.0,
            "explained_variance": 0.0,
        }
        ev_values: list[Tensor] = []
        ev_returns: list[Tensor] = []
        self.optimizer.zero_grad(set_to_none=True)
        self.model.actor.eval()
        self.model.critic.train()
        autocast_enabled = self.config.use_bf16_autocast and self.device.type in {
            "cpu", "cuda"
        }
        for chunk_batch in chunk_batches:
            prepared = [self._prepare_chunk(chunk) for chunk, _ in chunk_batch]
            actor_inputs = _batch_input_mappings(
                [row[0] for row in prepared], kind="Actor"
            )
            critic_inputs = _batch_input_mappings(
                [row[1] for row in prepared], kind="Critic"
            )
            targets = _batch_targets([row[2] for row in prepared])
            count = sum(row[1] for row in chunk_batch)
            # no_grad, rather than inference_mode, yields an ordinary detached
            # tensor that autograd may safely save as a Critic layer input.
            with torch.no_grad():
                featured = self.model.actor.forward_with_features(**actor_inputs)
                actor_latent = featured.pre_head_latent.detach()
            if not bool(torch.isfinite(actor_latent).all()):
                raise FloatingPointError("Actor pre_head_latent contains NaN/Inf")
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                output = self.model.critic(
                    actor_latent=actor_latent, **critic_inputs
                )
            output = _float_output(output)
            if any(not bool(torch.isfinite(value).all()) for value in output):
                raise FloatingPointError("Critic emitted NaN/Inf")
            losses = critic_loss(
                output,
                targets,
                value_coefficient=self.config.value_coefficient,
                auxiliary_coefficient=self.config.auxiliary_coefficient,
            )
            components = (
                losses.total,
                losses.value,
                losses.wdl,
                losses.crown,
                losses.tower_hp,
                losses.future_damage,
            )
            if any(not bool(torch.isfinite(value)) for value in components):
                raise FloatingPointError("Critic loss contains NaN/Inf")
            weight = count / total_steps
            (losses.total * weight).backward()
            values = (
                losses.total,
                losses.value,
                losses.wdl,
                losses.crown,
                losses.tower_hp,
                losses.future_damage,
            )
            for name, value in zip(
                (
                    "loss", "value_loss", "wdl_loss", "crown_loss",
                    "tower_hp_loss", "future_damage_loss",
                ),
                values,
                strict=True,
            ):
                accumulator[name] += float(value.detach()) * weight
            mask = targets.loss_mask.bool()
            ev_values.append(output.values.detach()[mask].cpu())
            ev_returns.append(targets.returns.detach()[mask].cpu())

        joined_values = torch.cat(ev_values)
        joined_returns = torch.cat(ev_returns)
        accumulator["explained_variance"] = float(
            explained_variance(
                joined_values,
                joined_returns,
                torch.ones_like(joined_returns, dtype=torch.bool),
            )
        )

        critic_parameters = [
            parameter for parameter in self.model.critic.parameters() if parameter.requires_grad
        ]
        gradients = [parameter.grad for parameter in critic_parameters if parameter.grad is not None]
        if not gradients:
            raise RuntimeError("Stage-1 Critic update produced no gradients")
        if any(not bool(torch.isfinite(value).all()) for value in gradients):
            raise FloatingPointError("Critic gradient contains NaN/Inf")
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            critic_parameters, self.config.max_grad_norm, error_if_nonfinite=True
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("Critic gradient norm is NaN/Inf")
        if any(
            parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
            for parameter in critic_parameters
        ):
            raise FloatingPointError("clipped Critic gradient contains NaN/Inf")
        self.optimizer.step()
        if any(
            value.is_floating_point() and not bool(torch.isfinite(value).all())
            for value in self.model.critic.state_dict().values()
        ):
            raise FloatingPointError("Critic parameter became NaN/Inf")
        if any(parameter.grad is not None for parameter in self.model.actor.parameters()):
            raise RuntimeError("Critic update leaked a gradient into the Actor")
        after_hash = actor_state_digest(self.model.actor)
        if after_hash != before_hash:
            raise RuntimeError("Stage-1 Critic update changed the Actor hash")

        self.global_update += 1
        metrics: dict[str, float | int | str] = {
            **accumulator,
            "gradient_norm": float(gradient_norm.detach()),
            "gradient_clip": float(self.config.max_grad_norm),
            "loss_steps": total_steps,
            "chunks": len(chunks),
            "chunk_batches": len(chunk_batches),
            "chunk_batch_size": self.config.chunk_batch_size,
            "global_update": self.global_update,
            "actor_sha256": after_hash,
            "bf16_autocast": int(autocast_enabled),
            "explained_variance_global": 1,
        }
        numeric = [float(value) for value in metrics.values() if isinstance(value, (int, float))]
        if not all(math.isfinite(value) for value in numeric):
            raise FloatingPointError("Stage-1 metrics contain NaN/Inf")
        return metrics

    def checkpoint_bundle(
        self, metrics: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        current_hash = actor_state_digest(self.model.actor)
        if current_hash != self.actor_sha256:
            raise RuntimeError("Actor changed before checkpoint publication")
        inference_state = _clone_state(self.model.actor, fp32=False)
        # A source reference avoids a second 177M FP32 copy.  Without one, the
        # checkpoint is self-contained and carries the canonical FP32 master.
        actor_fp32_master = (
            None
            if self.actor_source_reference is not None
            else _clone_state(self.model.actor, fp32=True)
        )
        actor_config = getattr(self.model.actor, "config", None)
        critic_config = getattr(self.model.critic, "config", None)
        config = {
            "stage": "stage1_critic",
            "trainer": asdict(self.config),
            "run": dict(self.run_config),
            "actor": (
                actor_config.to_dict()
                if actor_config is not None and hasattr(actor_config, "to_dict")
                else None
            ),
            "critic": (
                asdict(critic_config)
                if critic_config is not None and hasattr(critic_config, "__dataclass_fields__")
                else None
            ),
        }
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "kind": CHECKPOINT_KIND,
            "stage": "stage1_critic",
            "global_update": self.global_update,
            "actor_sha256": current_hash,
            "actor_source_reference": (
                dict(self.actor_source_reference)
                if self.actor_source_reference is not None
                else None
            ),
            "actor_fp32_master": actor_fp32_master,
            "actor_inference_state": inference_state,
            "critic": _clone_state(self.model.critic),
            "optimizer": self.optimizer.state_dict(),
            "rng": _capture_rng_state(),
            "config": config,
            "metrics": dict(metrics or {}),
        }

    def save_checkpoint(
        self,
        checkpoint_directory: str | Path,
        metrics: Mapping[str, Any] | None = None,
    ) -> Path:
        """Atomically publish a restorable bundle and retain three versions."""

        root = Path(checkpoint_directory)
        root.mkdir(parents=True, exist_ok=True)
        final = root / f"checkpoint-{self.global_update:012d}.pt"
        bundle = self.checkpoint_bundle(metrics)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{final.name}.", suffix=".tmp", dir=root
        )
        try:
            with os.fdopen(descriptor, "wb") as output:
                torch.save(bundle, output)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_name, final)
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            Path(temporary_name).unlink(missing_ok=True)
            raise

        latest = root / "latest.pt"
        descriptor, latest_temporary = tempfile.mkstemp(
            prefix=".latest.pt.", suffix=".tmp", dir=root
        )
        os.close(descriptor)
        try:
            Path(latest_temporary).unlink(missing_ok=True)
            try:
                os.link(final, latest_temporary)
            except OSError:
                shutil.copyfile(final, latest_temporary)
            os.replace(latest_temporary, latest)
        except BaseException:
            Path(latest_temporary).unlink(missing_ok=True)
            raise

        versions = sorted(root.glob("checkpoint-*.pt"))
        for stale in versions[:-self.config.retain_checkpoints]:
            stale.unlink()
        return final

    def restore_checkpoint(
        self,
        checkpoint_path: str | Path,
        *,
        restore_rng: bool = True,
    ) -> dict[str, Any]:
        # Deserialize on CPU so RNG state retains its required ByteTensor
        # device and the full optimizer bundle does not transiently duplicate
        # itself in VRAM. load_state_dict moves model/optimizer tensors to the
        # parameter devices below.
        value = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        required = {
            "kind", "schema_version", "global_update", "actor_sha256",
            "actor_inference_state", "critic", "optimizer", "rng", "config", "metrics",
        }
        if not isinstance(value, Mapping) or value.get("kind") != CHECKPOINT_KIND:
            raise RuntimeError("invalid expert self-play checkpoint kind")
        missing = sorted(required.difference(value))
        if value.get("schema_version") != CHECKPOINT_SCHEMA_VERSION or missing:
            raise RuntimeError(f"incomplete Stage-1 checkpoint; missing={missing}")
        if value["actor_sha256"] != actor_state_digest(self.model.actor):
            raise RuntimeError("checkpoint references a different Actor")
        self.model.critic.load_state_dict(value["critic"], strict=True)
        self.optimizer.load_state_dict(value["optimizer"])
        self.global_update = int(value["global_update"])
        if restore_rng:
            _restore_rng_state(value["rng"])
        return dict(value["metrics"])


def train_stage1_update(
    model: ExpertActorCritic,
    chunks: Sequence[Mapping[str, Any]],
    **trainer_kwargs: Any,
) -> tuple[Stage1CriticTrainer, dict[str, float | int | str]]:
    """Convenience smoke path: construct a trainer and execute one update."""

    trainer = Stage1CriticTrainer(model, **trainer_kwargs)
    return trainer, trainer.train_update(chunks)
