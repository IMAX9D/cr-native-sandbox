"""Content-addressed batched inference for the recurrent expert Actor.

The native workers deliberately do not own a copy of the Actor.  They send one
public observation and its *pre-action* legal masks to this service.  Requests
that use the same Actor content hash are collated and dispatched by exactly one
``forward_sequence`` call.  Recurrent state is retained per
``(actor_sha256, worker_id, side)`` and can be reset at episode boundaries.

This module samples the complete marked-hazard action: wait versus event,
normal-card versus ability, slot, and (when required) target cell.  Returned
log-probability components are the behavior-policy values needed by PPO; their
sum is the one joint log probability and no conditional term is counted for a
wait or an inactive branch.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from typing import Hashable, Mapping, Sequence

import torch
from torch import Tensor, nn

from expert_v1.training_v1.model import (
    ExpertPolicyConfig,
    ExpertPolicyOutput,
    RecurrentExpertPolicy,
)

from .actions import ExpertActionMasks
from .actor_adapter import actor_state_digest
from .hazard import lambda_from_logits


_SHA256_LENGTH = 64


def _valid_sha256(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class PolicyRequest:
    """One online Actor decision.

    ``actor_inputs`` describes one time step.  Values may either omit the time
    dimension (for example ``public_scalars: [P]``) or retain a singleton time
    dimension (``[1, P]``).  A leading singleton batch dimension is accepted as
    well.  Entity rows may have different lengths across requests; the service
    pads them while preserving ``entity_mask``.

    ``delta_ticks`` is both the elapsed interval supplied to the recurrent
    Actor and the hazard exposure window.  It is returned unchanged with the
    sampled action, making variable-time rollouts explicit.
    """

    worker_id: Hashable
    side: int
    actor_sha256: str
    actor_inputs: Mapping[str, Tensor | None]
    masks: ExpertActionMasks
    delta_ticks: int = 1
    reset_hidden: bool = False

    def validate_identity(self) -> None:
        if not _valid_sha256(self.actor_sha256):
            raise ValueError("actor_sha256 must be a lowercase SHA-256 digest")
        if self.side not in (0, 1):
            raise ValueError("side must be 0 or 1")
        try:
            hash(self.worker_id)
        except TypeError as error:
            raise ValueError("worker_id must be hashable") from error
        if int(self.delta_ticks) != self.delta_ticks or self.delta_ticks < 1:
            raise ValueError("delta_ticks must be a positive integer")


# A descriptive alias for callers that prefer the service name in type hints.
BatchedPolicyRequest = PolicyRequest


@dataclass(frozen=True)
class SampledPolicyAction:
    worker_id: Hashable
    side: int
    actor_sha256: str
    delta_ticks: int
    event_happened: bool
    action_kind: int
    card_slot: int
    position: int
    ability_slot: int
    ability_position: int
    ability_requires_target: bool
    lambda_per_second: float
    event_probability: float
    logp_total: float
    logp_timing: float
    logp_action_type: float
    logp_slot: float
    logp_position: float
    logp_mark: float

    # Rollout records use the ``old_logp_*`` spelling.  Properties avoid a
    # second, potentially divergent copy of those values.
    @property
    def old_logp_total(self) -> float:
        return self.logp_total

    @property
    def old_logp_timing(self) -> float:
        return self.logp_timing

    @property
    def old_logp_action_type(self) -> float:
        return self.logp_action_type

    @property
    def old_logp_slot(self) -> float:
        return self.logp_slot

    @property
    def old_logp_position(self) -> float:
        return self.logp_position


# Ranks after removing batch but retaining one online time dimension.
_INPUT_RANKS = {
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

_ENTITY_KEYS = {
    "entity_tokens",
    "entity_positions",
    "entity_relations",
    "entity_numeric",
    "entity_mask",
}

_MASK_RANKS = {
    "action_kind": 1,
    "cards": 1,
    "positions": 2,
    "abilities": 1,
    "ability_positions": 2,
    "ability_requires_target": 1,
}


def _online_tensor(value: Tensor, *, name: str, rank: int) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"Actor input {name} must be a Tensor or None")
    result = value
    # Accept a caller-owned singleton batch prefix.
    if result.ndim == rank + 1:
        if result.shape[0] != 1:
            raise ValueError(f"Actor input {name} contains more than one batch row")
        result = result.squeeze(0)
    if result.ndim == rank - 1:
        result = result.unsqueeze(0)
    if result.ndim != rank or result.shape[0] != 1:
        raise ValueError(
            f"Actor input {name} must contain exactly one online time step"
        )
    return result


def _mask_tensor(value: Tensor, *, name: str, rank: int) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"action mask {name} must be a Tensor")
    result = value
    while result.ndim > rank and result.shape[0] == 1:
        result = result.squeeze(0)
    if result.ndim != rank:
        raise ValueError(f"action mask {name} has the wrong rank")
    return result.bool()


def _module_device_dtype(
    actor: nn.Module, fallback: torch.device
) -> tuple[torch.device, torch.dtype]:
    for value in actor.parameters():
        return value.device, value.dtype
    for value in actor.buffers():
        return value.device, value.dtype
    return fallback, torch.float32


def _pad_entities(values: Sequence[Tensor], *, name: str) -> Tensor:
    maximum = max(int(value.shape[1]) for value in values)
    padded: list[Tensor] = []
    for value in values:
        if value.shape[1] == maximum:
            padded.append(value)
            continue
        shape = list(value.shape)
        shape[1] = maximum - value.shape[1]
        fill = torch.zeros(shape, dtype=value.dtype, device=value.device)
        padded.append(torch.cat((value, fill), dim=1))
    result = torch.stack(padded, dim=0)
    if name == "entity_mask":
        result = result.bool()
    return result


def _masked_choice(
    logits: Tensor,
    mask: Tensor,
    *,
    deterministic: bool,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Return sampled indices and their finite categorical log probabilities."""

    if logits.shape != mask.shape:
        raise ValueError("categorical logits/mask shapes differ")
    legal = mask.bool()
    if bool((~legal.any(dim=-1)).any()):
        raise ValueError("categorical action layer is all-invalid")
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("Actor emitted non-finite categorical logits")
    masked = logits.float().masked_fill(~legal, -torch.inf)
    logp = torch.log_softmax(masked, dim=-1)
    if deterministic:
        index = masked.argmax(dim=-1)
    else:
        probability = torch.softmax(masked, dim=-1)
        index = torch.multinomial(
            probability, 1, replacement=True, generator=generator
        ).squeeze(-1)
    selected = logp.gather(-1, index.unsqueeze(-1)).squeeze(-1)
    if not bool(torch.isfinite(selected).all()):
        raise FloatingPointError("sampled categorical log probability is non-finite")
    return index, selected


class BatchedPolicyService:
    """Synchronous micro-batch service for one or more immutable Actors.

    The caller chooses the micro-batch boundary (normally all native sides that
    became ready during the same scheduler turn).  Different content hashes are
    isolated; every non-empty hash group incurs one and only one Actor forward.
    """

    def __init__(
        self,
        *,
        device: torch.device | str | None = None,
        deterministic: bool = False,
        seed: int = 0,
        deterministic_event_threshold: float = 0.5,
    ) -> None:
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if not 0.0 <= deterministic_event_threshold <= 1.0:
            raise ValueError("deterministic_event_threshold must be in [0, 1]")
        self.deterministic = bool(deterministic)
        self.deterministic_event_threshold = float(deterministic_event_threshold)
        self._actors: dict[str, RecurrentExpertPolicy] = {}
        self._hidden: dict[tuple[str, Hashable, int], tuple[Tensor, Tensor]] = {}
        self._pre_action_hidden: dict[
            tuple[str, Hashable, int], tuple[Tensor, Tensor]
        ] = {}
        self._generator = torch.Generator(device=self.device.type)
        self._generator.manual_seed(int(seed))
        self.forward_calls = 0

    @property
    def registered_actor_hashes(self) -> tuple[str, ...]:
        return tuple(self._actors)

    @property
    def recurrent_state_count(self) -> int:
        return len(self._hidden)

    def register_actor(
        self,
        actor: RecurrentExpertPolicy,
        *,
        actor_sha256: str | None = None,
        verify_content: bool = False,
    ) -> str:
        """Register one behavior Actor and return its content address.

        A checkpoint/artifact hash can be supplied by the league.  Set
        ``verify_content`` only when that hash was defined using
        :func:`actor_state_digest`; file hashes intentionally need not equal the
        in-memory state digest.
        """

        if not isinstance(actor, nn.Module):
            raise TypeError("actor must be a torch module")
        actor.to(self.device)
        actor.eval()
        digest = actor_state_digest(actor) if actor_sha256 is None or verify_content else None
        content_hash = digest if actor_sha256 is None else actor_sha256
        assert content_hash is not None
        if not _valid_sha256(content_hash):
            raise ValueError("actor_sha256 must be a lowercase SHA-256 digest")
        if verify_content and digest != content_hash:
            raise ValueError("registered Actor content does not match actor_sha256")
        previous = self._actors.get(content_hash)
        if previous is not None and previous is not actor:
            raise ValueError("a different Actor is already registered under this hash")
        self._actors[content_hash] = actor
        return content_hash

    def unregister_actor(self, actor_sha256: str) -> None:
        self._actors.pop(actor_sha256, None)
        self.reset_hidden(actor_sha256=actor_sha256)

    def reset_hidden(
        self,
        *,
        worker_id: Hashable | None = None,
        side: int | None = None,
        actor_sha256: str | None = None,
    ) -> int:
        """Discard matching recurrent states and return the number removed."""

        if side is not None and side not in (0, 1):
            raise ValueError("side must be 0 or 1")
        keys = [
            key
            for key in self._hidden
            if (actor_sha256 is None or key[0] == actor_sha256)
            and (worker_id is None or key[1] == worker_id)
            and (side is None or key[2] == side)
        ]
        for key in keys:
            del self._hidden[key]
            self._pre_action_hidden.pop(key, None)
        return len(keys)

    # Episode-oriented spelling used by rollout coordinators.
    def reset_episode(self, worker_id: Hashable) -> int:
        return self.reset_hidden(worker_id=worker_id)

    def last_pre_action_hidden(
        self, *, actor_sha256: str, worker_id: Hashable, side: int
    ) -> tuple[Tensor, Tensor]:
        """Return the exact recurrent state used by the most recent decision."""

        key = (actor_sha256, worker_id, side)
        hidden = self._pre_action_hidden.get(key)
        if hidden is None:
            raise KeyError(f"no pre-action recurrent state recorded for {key!r}")
        return tuple(value.detach().cpu().contiguous().clone() for value in hidden)  # type: ignore[return-value]

    def _batch_inputs(
        self,
        requests: Sequence[PolicyRequest],
        *,
        device: torch.device,
        floating_dtype: torch.dtype,
    ) -> dict[str, Tensor | None]:
        key_sets = [set(request.actor_inputs) for request in requests]
        if any(keys != key_sets[0] for keys in key_sets[1:]):
            raise ValueError("requests in an Actor batch have different input fields")
        unknown = key_sets[0] - set(_INPUT_RANKS)
        if unknown:
            raise ValueError(f"unknown Actor input fields: {sorted(unknown)}")
        output: dict[str, Tensor | None] = {}
        for name in key_sets[0]:
            raw_values = [request.actor_inputs[name] for request in requests]
            if all(value is None for value in raw_values):
                output[name] = None
                continue
            if any(value is None for value in raw_values):
                raise ValueError(f"Actor input {name} is None for only part of a batch")
            tensors = [
                _online_tensor(value, name=name, rank=_INPUT_RANKS[name])
                for value in raw_values
                if value is not None
            ]
            moved: list[Tensor] = []
            for value in tensors:
                if value.is_floating_point():
                    value = value.to(device=device, dtype=floating_dtype)
                else:
                    value = value.to(device=device)
                moved.append(value)
            if name in _ENTITY_KEYS:
                output[name] = _pad_entities(moved, name=name)
            else:
                shapes = {tuple(value.shape) for value in moved}
                if len(shapes) != 1:
                    raise ValueError(f"Actor input {name} has incompatible shapes")
                output[name] = torch.stack(moved, dim=0)

        requested_delta = torch.tensor(
            [request.delta_ticks for request in requests],
            device=device,
            dtype=floating_dtype,
        ).unsqueeze(-1)
        existing = output.get("delta_ticks")
        if existing is not None and not bool(
            torch.equal(existing.float(), requested_delta.float())
        ):
            raise ValueError("actor_inputs delta_ticks differs from request delta_ticks")
        output["delta_ticks"] = requested_delta
        return output

    @staticmethod
    def _batch_masks(requests: Sequence[PolicyRequest], device: torch.device) -> ExpertActionMasks:
        fields: dict[str, Tensor] = {}
        for name, rank in _MASK_RANKS.items():
            values = [
                _mask_tensor(getattr(request.masks, name), name=name, rank=rank)
                for request in requests
            ]
            shapes = {tuple(value.shape) for value in values}
            if len(shapes) != 1:
                raise ValueError(f"action mask {name} has incompatible shapes")
            fields[name] = torch.stack(values, dim=0).to(device=device)
        return ExpertActionMasks(**fields)

    @staticmethod
    def _validate_output(output: ExpertPolicyOutput, batch_size: int) -> None:
        tensors = (
            output.rate_logits,
            output.action_kind_logits,
            output.card_logits,
            output.position_logits,
            output.ability_logits,
            output.ability_position_logits,
        )
        for value in tensors:
            if value.shape[0] != batch_size or value.shape[1] != 1:
                raise ValueError("Actor output must preserve [batch, one-time-step]")
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError("Actor emitted NaN or Inf")
        hidden = output.hidden
        if len(hidden) != 2 or hidden[0].shape[1] != batch_size:
            raise ValueError("Actor returned malformed recurrent hidden state")
        if not bool(torch.isfinite(hidden[0]).all() and torch.isfinite(hidden[1]).all()):
            raise FloatingPointError("Actor emitted non-finite recurrent state")

    @staticmethod
    def _validate_masks(masks: ExpertActionMasks, output: ExpertPolicyOutput) -> None:
        kinds = masks.action_kind
        cards = masks.cards
        positions = masks.positions
        abilities = masks.abilities
        ability_positions = masks.ability_positions
        targeted = masks.ability_requires_target
        expected = {
            "action_kind": output.action_kind_logits[:, 0].shape,
            "cards": output.card_logits[:, 0].shape,
            "positions": output.position_logits[:, 0].shape,
            "abilities": output.ability_logits[:, 0].shape,
            "ability_positions": output.ability_position_logits[:, 0].shape,
            "ability_requires_target": output.ability_logits[:, 0].shape,
        }
        actual = {
            "action_kind": kinds.shape,
            "cards": cards.shape,
            "positions": positions.shape,
            "abilities": abilities.shape,
            "ability_positions": ability_positions.shape,
            "ability_requires_target": targeted.shape,
        }
        for name in expected:
            if actual[name] != expected[name]:
                raise ValueError(f"action mask {name} does not match Actor output")
        if bool((kinds[:, 0] & ~cards.any(dim=-1)).any()):
            raise ValueError("normal action kind is legal but every card is illegal")
        if bool((cards & ~positions.any(dim=-1)).any()):
            raise ValueError("a legal card has no legal placement cell")
        if bool((kinds[:, 1] & ~abilities.any(dim=-1)).any()):
            raise ValueError("ability action kind is legal but every ability is illegal")
        target_missing = abilities & targeted & ~ability_positions.any(dim=-1)
        if bool(target_missing.any()):
            raise ValueError("a legal targeted ability has no legal target cell")

    def _hidden_batch(
        self,
        actor_sha256: str,
        actor: RecurrentExpertPolicy,
        requests: Sequence[PolicyRequest],
        device: torch.device,
        floating_dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        rows: list[tuple[Tensor, Tensor]] = []
        for request in requests:
            key = (actor_sha256, request.worker_id, request.side)
            if request.reset_hidden:
                self._hidden.pop(key, None)
            hidden = self._hidden.get(key)
            if hidden is None:
                hidden = actor.initial_hidden(1, device=device)
            rows.append((
                hidden[0].to(device=device, dtype=floating_dtype),
                hidden[1].to(device=device, dtype=floating_dtype),
            ))
        return (
            torch.cat([value[0] for value in rows], dim=1),
            torch.cat([value[1] for value in rows], dim=1),
        )

    def _store_hidden(
        self,
        actor_sha256: str,
        requests: Sequence[PolicyRequest],
        hidden: tuple[Tensor, Tensor],
    ) -> None:
        for index, request in enumerate(requests):
            key = (actor_sha256, request.worker_id, request.side)
            self._hidden[key] = (
                hidden[0][:, index : index + 1].detach(),
                hidden[1][:, index : index + 1].detach(),
            )

    def _sample_group(
        self,
        actor_sha256: str,
        indexed_requests: Sequence[tuple[int, PolicyRequest]],
        *,
        deterministic: bool,
    ) -> list[tuple[int, SampledPolicyAction]]:
        actor = self._actors.get(actor_sha256)
        if actor is None:
            raise KeyError(f"unregistered Actor content hash: {actor_sha256}")
        requests = [request for _index, request in indexed_requests]
        device, floating_dtype = _module_device_dtype(actor, self.device)
        inputs = self._batch_inputs(
            requests, device=device, floating_dtype=floating_dtype
        )
        masks = self._batch_masks(requests, device)
        hidden = self._hidden_batch(
            actor_sha256, actor, requests, device, floating_dtype
        )
        for index, request in enumerate(requests):
            key = (actor_sha256, request.worker_id, request.side)
            self._pre_action_hidden[key] = (
                hidden[0][:, index:index + 1].detach(),
                hidden[1][:, index:index + 1].detach(),
            )
        with torch.inference_mode():
            output = actor.forward_sequence(**inputs, hidden=hidden)
        self.forward_calls += 1
        self._validate_output(output, len(requests))
        self._store_hidden(actor_sha256, requests, output.hidden)

        # Remove the singleton online time dimension.
        rate_logits = output.rate_logits[:, 0].float()
        kind_logits = output.action_kind_logits[:, 0].float()
        card_logits = output.card_logits[:, 0].float()
        position_logits = output.position_logits[:, 0].float()
        ability_logits = output.ability_logits[:, 0].float()
        ability_position_logits = output.ability_position_logits[:, 0].float()
        self._validate_masks(masks, output)

        can_act = masks.action_kind.any(dim=-1)
        delta = torch.tensor(
            [request.delta_ticks for request in requests],
            dtype=torch.float32,
            device=device,
        )
        config: ExpertPolicyConfig = actor.config
        rate = lambda_from_logits(rate_logits, config.lambda_max)
        exposure = rate * delta * float(config.native_tick_seconds)
        exposure = torch.where(can_act, exposure, torch.zeros_like(exposure))
        event_probability = -torch.expm1(-exposure)
        if not bool(torch.isfinite(rate).all() and torch.isfinite(event_probability).all()):
            raise FloatingPointError("non-finite marked-hazard rate/probability")
        if deterministic:
            event = can_act & (
                event_probability >= self.deterministic_event_threshold
            )
        else:
            draw = torch.rand(
                event_probability.shape, device=device, generator=self._generator
            )
            event = can_act & (draw < event_probability)

        # Allocate safe default branch values.  We sample only active subsets so
        # an all-invalid inactive conditional layer is never evaluated.
        batch = len(requests)
        action_kind = torch.zeros(batch, dtype=torch.long, device=device)
        card_slot = torch.zeros_like(action_kind)
        position = torch.zeros_like(action_kind)
        ability_slot = torch.zeros_like(action_kind)
        ability_position = torch.zeros_like(action_kind)
        ability_requires_target = torch.zeros(batch, dtype=torch.bool, device=device)
        logp_action_type = torch.zeros(batch, dtype=torch.float32, device=device)
        logp_slot = torch.zeros_like(logp_action_type)
        logp_position = torch.zeros_like(logp_action_type)

        active = event.nonzero(as_tuple=True)[0]
        if active.numel():
            selected, selected_logp = _masked_choice(
                kind_logits.index_select(0, active),
                masks.action_kind.index_select(0, active),
                deterministic=deterministic,
                generator=self._generator,
            )
            action_kind[active] = selected
            logp_action_type[active] = selected_logp

        normal = (event & (action_kind == 0)).nonzero(as_tuple=True)[0]
        if normal.numel():
            selected, selected_logp = _masked_choice(
                card_logits.index_select(0, normal),
                masks.cards.index_select(0, normal),
                deterministic=deterministic,
                generator=self._generator,
            )
            card_slot[normal] = selected
            logp_slot[normal] = selected_logp
            chosen_logits = position_logits[normal, selected]
            chosen_masks = masks.positions[normal, selected]
            selected_position, selected_position_logp = _masked_choice(
                chosen_logits,
                chosen_masks,
                deterministic=deterministic,
                generator=self._generator,
            )
            position[normal] = selected_position
            logp_position[normal] = selected_position_logp

        ability = (event & (action_kind == 1)).nonzero(as_tuple=True)[0]
        if ability.numel():
            selected, selected_logp = _masked_choice(
                ability_logits.index_select(0, ability),
                masks.abilities.index_select(0, ability),
                deterministic=deterministic,
                generator=self._generator,
            )
            ability_slot[ability] = selected
            logp_slot[ability] = selected_logp
            needs_target = masks.ability_requires_target[ability, selected]
            ability_requires_target[ability] = needs_target
            target_rows = ability[needs_target]
            target_slots = selected[needs_target]
            if target_rows.numel():
                chosen_logits = ability_position_logits[target_rows, target_slots]
                chosen_masks = masks.ability_positions[target_rows, target_slots]
                selected_position, selected_position_logp = _masked_choice(
                    chosen_logits,
                    chosen_masks,
                    deterministic=deterministic,
                    generator=self._generator,
                )
                ability_position[target_rows] = selected_position
                logp_position[target_rows] = selected_position_logp

        logp_mark = logp_action_type + logp_slot + logp_position
        # Stable timing terms.  Impossible events are never sampled, and a
        # probability-one wait is likewise never sampled, so selected terms are
        # finite without clamping or changing the behavior distribution.
        logp_timing = torch.where(
            event,
            torch.log(-torch.expm1(-exposure)),
            -exposure,
        )
        logp_timing = torch.where(can_act, logp_timing, torch.zeros_like(logp_timing))
        logp_total = logp_timing + logp_mark
        finite = torch.stack(
            (
                rate,
                event_probability,
                logp_timing,
                logp_action_type,
                logp_slot,
                logp_position,
                logp_mark,
                logp_total,
            ),
            dim=-1,
        )
        if not bool(torch.isfinite(finite).all()):
            raise FloatingPointError("sampled marked-hazard action is non-finite")

        answer: list[tuple[int, SampledPolicyAction]] = []
        for row, (original_index, request) in enumerate(indexed_requests):
            answer.append((original_index, SampledPolicyAction(
                worker_id=request.worker_id,
                side=request.side,
                actor_sha256=actor_sha256,
                delta_ticks=request.delta_ticks,
                event_happened=bool(event[row].item()),
                action_kind=int(action_kind[row].item()),
                card_slot=int(card_slot[row].item()),
                position=int(position[row].item()),
                ability_slot=int(ability_slot[row].item()),
                ability_position=int(ability_position[row].item()),
                ability_requires_target=bool(ability_requires_target[row].item()),
                lambda_per_second=float(rate[row].item()),
                event_probability=float(event_probability[row].item()),
                logp_total=float(logp_total[row].item()),
                logp_timing=float(logp_timing[row].item()),
                logp_action_type=float(logp_action_type[row].item()),
                logp_slot=float(logp_slot[row].item()),
                logp_position=float(logp_position[row].item()),
                logp_mark=float(logp_mark[row].item()),
            )))
        return answer

    def act(
        self,
        requests: Sequence[PolicyRequest],
        *,
        deterministic: bool | None = None,
    ) -> list[SampledPolicyAction]:
        """Evaluate and sample a micro-batch, preserving request order."""

        rows = list(requests)
        if not rows:
            return []
        identities: set[tuple[str, Hashable, int]] = set()
        grouped: OrderedDict[str, list[tuple[int, PolicyRequest]]] = OrderedDict()
        for index, request in enumerate(rows):
            if not isinstance(request, PolicyRequest):
                raise TypeError("requests must contain PolicyRequest values")
            request.validate_identity()
            identity = (request.actor_sha256, request.worker_id, request.side)
            if identity in identities:
                raise ValueError(
                    "one micro-batch cannot contain two sequential decisions "
                    "for the same Actor/worker/side"
                )
            identities.add(identity)
            grouped.setdefault(request.actor_sha256, []).append((index, request))
        sampled: list[SampledPolicyAction | None] = [None] * len(rows)
        mode = self.deterministic if deterministic is None else bool(deterministic)
        for actor_sha256, group in grouped.items():
            for index, action in self._sample_group(
                actor_sha256, group, deterministic=mode
            ):
                sampled[index] = action
        if any(value is None for value in sampled):
            raise RuntimeError("internal batched-policy result loss")
        return [value for value in sampled if value is not None]

    # Common coordinator spellings.
    infer_batch = act
    sample_batch = act


__all__ = [
    "BatchedPolicyRequest",
    "BatchedPolicyService",
    "PolicyRequest",
    "SampledPolicyAction",
]
