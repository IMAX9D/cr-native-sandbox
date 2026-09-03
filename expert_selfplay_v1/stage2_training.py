"""Stage-2 recurrent PPO for the expert-initialized marked-hazard Actor.

The behavior Actor is version locked by the rollout manifest.  Values and GAE
are recomputed from the admitted continuation checkpoint, while old joint log
probabilities remain the exact FP16 behavior values recorded by the native
collector.  Only the named Stage-2 reaction groups are trainable.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from expert_v1.training_v1.model import (
    ExpertPolicyConfig,
    ExpertPolicyOutput,
    RecurrentExpertPolicy,
    configure_position_precision,
)

from .actions import (
    ExpertActionMasks,
    RecordedExpertAction,
    evaluate_expert_action,
    expert_policy_kl,
)
from .actor_adapter import actor_state_digest
from .contracts import BatchManifest
from .critic import ExpertActorCritic, PrivilegedCritic, PrivilegedCriticConfig
from .critic_training import (
    _ACTOR_INPUT_RANKS,
    _ACTOR_RAGGED,
    _CRITIC_ENTITY_RAGGED,
    _CRITIC_INPUT_RANKS,
    _batch_input_mappings,
    _batch_targets,
    _batch_tensors,
    _capture_rng_state,
    _clone_state,
    _collate_inputs,
    _critic_targets,
    _initial_hidden,
    _restore_rng_state,
)
from .hazard import lambda_from_logits
from .losses import critic_loss
from .ppo import recurrent_ppo_loss
from .rollout import DecisionRecord, EpisodeHeader, LearnerEpisodeBuffer
from .rollout_storage import LearnerEpisodeChunker, verify_rollout_shard
from .stages import configure_stage, stage_specs
from .update_guard import UpdateGuardDecision, evaluate_update


STAGE1_KIND = "cr_native_expert_selfplay_checkpoint_v1"
STAGE2_KIND = "cr_native_expert_selfplay_stage2_checkpoint_v1"
EXPERT_INFERENCE_KIND = "cr_native_expert_inference_weights_v1"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Stage2TrainingConfig:
    ppo_epochs: int = 2
    clip_epsilon: float = 0.10
    value_coefficient: float = 0.50
    entropy_coefficient: float = 0.01
    bc_kl_coefficient: float = 1.0
    bc_kl_soft_limit: float = 0.03
    critic_auxiliary_coefficient: float = 0.10
    actor_grad_clip: float = 0.50
    critic_grad_clip: float = 1.00
    chunk_batch_size: int = 2
    retain_checkpoints: int = 3

    def validate(self) -> None:
        positive = {
            "ppo_epochs": self.ppo_epochs,
            "clip_epsilon": self.clip_epsilon,
            "value_coefficient": self.value_coefficient,
            "bc_kl_coefficient": self.bc_kl_coefficient,
            "bc_kl_soft_limit": self.bc_kl_soft_limit,
            "actor_grad_clip": self.actor_grad_clip,
            "critic_grad_clip": self.critic_grad_clip,
            "chunk_batch_size": self.chunk_batch_size,
            "retain_checkpoints": self.retain_checkpoints,
        }
        for name, value in positive.items():
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value in (
            ("entropy_coefficient", self.entropy_coefficient),
            ("critic_auxiliary_coefficient", self.critic_auxiliary_coefficient),
        ):
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.clip_epsilon >= 1:
            raise ValueError("clip_epsilon must be below one")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_digest(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(str(name).encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _half_state(actor: RecurrentExpertPolicy) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().to(dtype=torch.float16).contiguous().clone()
        for name, value in actor.state_dict().items()
    }


def _float_state(value: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {
        str(name): tensor.detach().cpu().float().contiguous().clone()
        for name, tensor in value.items()
    }


def _finite(value: Any) -> bool:
    if isinstance(value, Tensor):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return True


def _load_inference(path: Path) -> tuple[dict[str, Any], ExpertPolicyConfig]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping) or value.get("kind") != EXPERT_INFERENCE_KIND:
        raise RuntimeError("base checkpoint is not an expert inference artifact")
    if not isinstance(value.get("model_config"), Mapping) or not isinstance(
        value.get("model_state"), Mapping
    ):
        raise RuntimeError("base inference artifact is incomplete")
    config = ExpertPolicyConfig(**dict(value["model_config"]))
    configure_position_precision(config)
    return dict(value), config


def _optimizer_parameter_names(
    optimizer: torch.optim.Optimizer,
    named_parameters: Mapping[str, Tensor],
) -> list[list[str]]:
    names = {id(parameter): name for name, parameter in named_parameters.items()}
    result: list[list[str]] = []
    for group in optimizer.param_groups:
        row = []
        for parameter in group["params"]:
            if id(parameter) not in names:
                raise RuntimeError("optimizer contains an unnamed parameter")
            row.append(names[id(parameter)])
        result.append(row)
    return result


def _cpu_optimizer_state(value: Mapping[str, Any]) -> dict[str, Any]:
    def convert(item: Any) -> Any:
        if isinstance(item, Tensor):
            return item.detach().cpu().clone()
        if isinstance(item, Mapping):
            return {key: convert(child) for key, child in item.items()}
        if isinstance(item, list):
            return [convert(child) for child in item]
        if isinstance(item, tuple):
            return tuple(convert(child) for child in item)
        return deepcopy(item)

    return convert(value)


def _atomic_torch(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            torch.save(dict(value), output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        Path(temporary).unlink(missing_ok=True)
        raise


def _clone_module_state(module: torch.nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _restore_module_state(module: torch.nn.Module, value: Mapping[str, Tensor]) -> None:
    module.load_state_dict(value, strict=True)


class Stage2PPOTrainer:
    def __init__(
        self,
        *,
        base_inference_checkpoint: Path,
        continuation_checkpoint: Path,
        expert_manifest: Path,
        device: torch.device | str,
        config: Stage2TrainingConfig | None = None,
    ) -> None:
        self.config = config or Stage2TrainingConfig()
        self.config.validate()
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.base_path = base_inference_checkpoint.resolve(strict=True)
        self.continuation_path = continuation_checkpoint.resolve(strict=True)
        self.manifest_path = expert_manifest.resolve(strict=True)
        self.base_payload, actor_config = _load_inference(self.base_path)
        if str(self.base_payload.get("dataset_manifest_sha256", "")) != _file_sha256(
            self.manifest_path
        ):
            raise RuntimeError("base Actor is not bound to the supplied expert manifest")

        continuation = torch.load(
            self.continuation_path, map_location="cpu", weights_only=False, mmap=True
        )
        if not isinstance(continuation, Mapping) or continuation.get("kind") not in {
            STAGE1_KIND, STAGE2_KIND
        }:
            raise RuntimeError("unsupported Stage-2 continuation checkpoint")
        self.continuation_kind = str(continuation["kind"])
        self.global_update = int(continuation["global_update"])
        self.policy_version = int(continuation.get("policy_version", 0))

        if self.continuation_kind == STAGE1_KIND:
            behavior_state = continuation["actor_inference_state"]
            master_state = _float_state(behavior_state)
            expected_behavior_hash = str(continuation["actor_sha256"])
            critic_optimizer_state = continuation["optimizer"]
            actor_optimizer_state = None
            actor_optimizer_names = None
        else:
            behavior_state = continuation["actor_behavior_fp16"]
            master_state = continuation["actor_master"]
            expected_behavior_hash = str(continuation["behavior_actor_sha256"])
            critic_optimizer_state = continuation["critic_optimizer"]
            actor_optimizer_state = continuation["actor_optimizer"]
            actor_optimizer_names = continuation["actor_optimizer_parameter_names"]
        if _state_digest(behavior_state) != expected_behavior_hash:
            raise RuntimeError("continuation behavior Actor hash is invalid")

        actor = RecurrentExpertPolicy(actor_config)
        actor.load_state_dict(master_state, strict=True)
        actor.to(device=self.device, dtype=torch.float32)
        bc_actor = RecurrentExpertPolicy(actor_config)
        bc_actor.load_state_dict(_float_state(self.base_payload["model_state"]), strict=True)
        bc_actor.to(device=self.device, dtype=torch.float32).eval()
        for parameter in bc_actor.parameters():
            parameter.requires_grad_(False)
        self.bc_actor = bc_actor
        self.bc_behavior_sha256 = _state_digest({
            name: value.detach().cpu().to(torch.float16)
            for name, value in self.base_payload["model_state"].items()
        })

        critic_config_value = continuation.get("config", {}).get("critic")
        if not isinstance(critic_config_value, Mapping):
            raise RuntimeError("continuation checkpoint has no Critic configuration")
        critic_config = PrivilegedCriticConfig(**dict(critic_config_value))
        critic = PrivilegedCritic(critic_config)
        critic.load_state_dict(continuation["critic"], strict=True)
        critic.to(self.device)
        self.model = ExpertActorCritic(actor, critic).to(self.device)

        report = configure_stage(self.model, "stage2_reaction")
        mapping = dict(report["parameter_groups"])
        named = dict(self.model.named_parameters())
        actor_groups = []
        for spec in stage_specs("stage2_reaction"):
            if spec.name == "critic":
                continue
            parameters = [
                parameter for name, parameter in named.items()
                if parameter.requires_grad and mapping.get(name) == spec.name
            ]
            if parameters:
                actor_groups.append({
                    "params": parameters,
                    "lr": spec.learning_rate,
                    "weight_decay": spec.weight_decay,
                    "group_name": spec.name,
                })
        if not actor_groups:
            raise RuntimeError("Stage-2 produced no trainable Actor parameter groups")
        self.actor_optimizer = torch.optim.AdamW(actor_groups, eps=1e-5)
        self.critic_optimizer = torch.optim.AdamW(
            self.model.critic.parameters(), lr=1e-4, weight_decay=1e-4, eps=1e-5
        )
        self.critic_optimizer.load_state_dict(critic_optimizer_state)
        for group in self.critic_optimizer.param_groups:
            group["lr"] = 1e-4
            group["weight_decay"] = 1e-4
        self.actor_optimizer_parameter_names = _optimizer_parameter_names(
            self.actor_optimizer, named
        )
        if actor_optimizer_state is not None:
            if actor_optimizer_names != self.actor_optimizer_parameter_names:
                raise RuntimeError("Actor optimizer parameter-name mapping changed")
            self.actor_optimizer.load_state_dict(actor_optimizer_state)
        self.critic_optimizer_parameter_names = _optimizer_parameter_names(
            self.critic_optimizer, named
        )
        recorded_critic_names = continuation.get("critic_optimizer_parameter_names")
        if recorded_critic_names is not None and recorded_critic_names != self.critic_optimizer_parameter_names:
            raise RuntimeError("Critic optimizer parameter-name mapping changed")
        _restore_rng_state(continuation["rng"])
        self.behavior_actor_sha256 = _state_digest(_half_state(self.model.actor))
        if self.behavior_actor_sha256 != expected_behavior_hash:
            raise RuntimeError("FP32 master does not round-trip to continuation behavior Actor")
        self.master_actor_sha256 = actor_state_digest(self.model.actor)
        self.stage_report = report
        self.base_checkpoint_sha256 = _file_sha256(self.base_path)
        self.continuation_sha256 = _file_sha256(self.continuation_path)

    def _extract_episode(
        self, stored: Mapping[str, Any]
    ) -> tuple[EpisodeHeader, list[DecisionRecord], list[dict[str, Any]]]:
        header = EpisodeHeader(**dict(stored["header"]))
        header.validate()
        decisions: list[DecisionRecord] = []
        payloads: list[dict[str, Any]] = []
        expected_start = 0
        for chunk in stored.get("chunks", []):
            if int(chunk["loss_start"]) != expected_start:
                raise RuntimeError("stored episode chunks have a coverage gap")
            mask = chunk["loss_mask"].bool().tolist()
            for raw, payload, selected in zip(
                chunk["decisions"], chunk["step_payloads"], mask, strict=True
            ):
                if selected:
                    decisions.append(DecisionRecord(**dict(raw)))
                    payloads.append(dict(payload))
            expected_start = int(chunk["loss_end"])
        if expected_start != int(stored["decision_count"]) or len(decisions) != len(payloads):
            raise RuntimeError("stored episode reconstruction is incomplete")
        for index, payload in enumerate(payloads):
            actor_inputs = payload.get("actor_inputs")
            hidden = actor_inputs.get("hidden") if isinstance(actor_inputs, Mapping) else None
            if not isinstance(hidden, (tuple, list)) or len(hidden) != 2:
                raise RuntimeError(
                    f"Stage-2 rollout lacks exact pre-action hidden at decision {index}"
                )
        return header, decisions, payloads

    def _full_episode_inputs(
        self, payloads: Sequence[Mapping[str, Any]]
    ) -> tuple[dict[str, Any], dict[str, Tensor]]:
        actor_rows = [dict(payload["actor_inputs"]) for payload in payloads]
        clean_rows = [
            {name: value for name, value in row.items() if name != "hidden"}
            for row in actor_rows
        ]
        actor_inputs = _collate_inputs(
            clean_rows,
            ranks=_ACTOR_INPUT_RANKS,
            ragged_groups=(_ACTOR_RAGGED,),
            device=self.device,
            floating_dtype=torch.float32,
            kind="Actor",
        )
        actor_inputs["hidden"] = None
        critic_inputs = _collate_inputs(
            [dict(payload["critic_inputs"]) for payload in payloads],
            ranks=_CRITIC_INPUT_RANKS,
            ragged_groups=(_CRITIC_ENTITY_RAGGED,),
            device=self.device,
            floating_dtype=torch.float32,
            kind="Critic",
        )
        if any(value is None for value in critic_inputs.values()):
            raise RuntimeError("Critic rollout input is partially absent")
        return actor_inputs, {
            name: value for name, value in critic_inputs.items() if value is not None
        }

    @staticmethod
    def _slice_bc_output(output: ExpertPolicyOutput, index: int) -> dict[str, Tensor]:
        return {
            name: getattr(output, name)[0, index].detach().cpu().to(torch.float16).clone()
            for name in (
                "rate_logits", "action_kind_logits", "card_logits",
                "position_logits", "ability_logits", "ability_position_logits",
            )
        }

    def prepare_rollout(
        self, shard_directory: Path
    ) -> tuple[list[dict[str, Any]], BatchManifest, dict[str, Any]]:
        shard_directory = shard_directory.resolve(strict=True)
        verified = verify_rollout_shard(
            shard_directory, return_payload=True, mmap=True
        )
        payload = verified.pop("_payload")
        batch_manifest = BatchManifest(**dict(verified["batch_manifest"]))
        batch_manifest.validate()
        if batch_manifest.behavior_actor_sha256 != self.behavior_actor_sha256:
            raise RuntimeError("rollout behavior Actor differs from continuation Actor")
        if batch_manifest.policy_version != self.policy_version:
            raise RuntimeError("rollout policy version is stale or from the future")
        episodes = payload.get("episodes")
        if not isinstance(episodes, list) or len(episodes) != batch_manifest.episode_count:
            raise RuntimeError("rollout episode count differs from its BatchManifest")

        output_chunks: list[dict[str, Any]] = []
        prepared_decisions = 0
        self.model.eval()
        self.bc_actor.eval()
        with torch.no_grad():
            for stored in episodes:
                header, decisions, step_payloads = self._extract_episode(stored)
                if header.curriculum_stage != "stage2_reaction":
                    raise RuntimeError("Stage-2 trainer rejected a non-Stage-2 episode")
                actor_inputs, critic_inputs = self._full_episode_inputs(step_payloads)
                current = self.model.actor.forward_with_features(**actor_inputs)
                critic = self.model.critic(
                    actor_latent=current.pre_head_latent.detach(), **critic_inputs
                )
                bc_output = self.bc_actor.forward_sequence(**actor_inputs)
                if not _finite((current.output, critic, bc_output)):
                    raise FloatingPointError("rollout value/BC preprocessing emitted NaN/Inf")
                rebuilt = LearnerEpisodeBuffer(header)
                enriched_payloads: list[dict[str, Any]] = []
                for index, (decision, step_payload) in enumerate(
                    zip(decisions, step_payloads, strict=True)
                ):
                    rebuilt.append(replace(
                        decision, value=float(critic.values[0, index].float().cpu())
                    ))
                    enriched = dict(step_payload)
                    enriched["bc_output"] = self._slice_bc_output(bc_output, index)
                    enriched_payloads.append(enriched)
                frozen = rebuilt.freeze()
                output_chunks.extend(
                    LearnerEpisodeChunker().chunk(
                        frozen, step_payloads=enriched_payloads
                    )
                )
                prepared_decisions += len(decisions)
        self.model.train()
        self.bc_actor.eval()
        if not output_chunks:
            raise RuntimeError("Stage-2 admitted rollout produced no recurrent chunks")
        return output_chunks, batch_manifest, {
            "episodes": len(episodes),
            "decisions": prepared_decisions,
            "chunks": len(output_chunks),
            "shard_content_sha256": verified["content_sha256"],
        }

    @staticmethod
    def _stack_rows(
        rows: Sequence[Mapping[str, Any]], name: str, *, device: torch.device,
        floating_dtype: torch.dtype | None = None,
    ) -> Tensor:
        values = [torch.as_tensor(row[name]) for row in rows]
        shapes = {tuple(value.shape) for value in values}
        if len(shapes) != 1:
            raise RuntimeError(f"Stage-2 row {name} has incompatible shapes")
        result = torch.stack(values, dim=0).unsqueeze(0).to(device)
        if floating_dtype is not None and result.is_floating_point():
            result = result.to(floating_dtype)
        return result

    def _prepare_chunk(self, chunk: Mapping[str, Any]) -> dict[str, Any]:
        payloads = chunk.get("step_payloads")
        if not isinstance(payloads, Sequence) or not payloads:
            raise RuntimeError("Stage-2 chunk has no step payloads")
        actor_rows = [dict(payload["actor_inputs"]) for payload in payloads]
        hidden = _initial_hidden(
            chunk, actor_rows, device=self.device, dtype=torch.float32
        )
        if hidden is None:
            raise RuntimeError("Stage-2 chunk has no exact recurrent hidden anchor")
        actor_inputs = _collate_inputs(
            [
                {name: value for name, value in row.items() if name != "hidden"}
                for row in actor_rows
            ],
            ranks=_ACTOR_INPUT_RANKS,
            ragged_groups=(_ACTOR_RAGGED,),
            device=self.device,
            floating_dtype=torch.float32,
            kind="Actor",
        )
        actor_inputs["hidden"] = hidden
        critic_inputs = _collate_inputs(
            [dict(payload["critic_inputs"]) for payload in payloads],
            ranks=_CRITIC_INPUT_RANKS,
            ragged_groups=(_CRITIC_ENTITY_RAGGED,),
            device=self.device,
            floating_dtype=torch.float32,
            kind="Critic",
        )
        if any(value is None for value in critic_inputs.values()):
            raise RuntimeError("Stage-2 Critic input is partially absent")
        mask_rows = [dict(payload["action_masks"]) for payload in payloads]
        action_rows = [dict(payload["recorded_action"]) for payload in payloads]
        bc_rows = [dict(payload["bc_output"]) for payload in payloads]
        masks = ExpertActionMasks(**{
            name: self._stack_rows(mask_rows, name, device=self.device).bool()
            for name in ExpertActionMasks.__dataclass_fields__
        })
        action = RecordedExpertAction(**{
            name: self._stack_rows(action_rows, name, device=self.device)
            for name in RecordedExpertAction.__dataclass_fields__
        })
        bc_output = ExpertPolicyOutput(
            *(self._stack_rows(
                bc_rows, name, device=self.device, floating_dtype=torch.float32
            ) for name in (
                "rate_logits", "action_kind_logits", "card_logits",
                "position_logits", "ability_logits", "ability_position_logits",
            )),
            (torch.empty(0, device=self.device), torch.empty(0, device=self.device)),
        )
        decisions = chunk["decisions"]
        old_log_prob = torch.tensor(
            [[float(value["old_logp_total"]) for value in decisions]],
            device=self.device,
        )
        advantages = torch.as_tensor(chunk["advantages"], device=self.device).float().unsqueeze(0)
        returns = torch.as_tensor(chunk["returns"], device=self.device).float().unsqueeze(0)
        loss_mask = torch.as_tensor(chunk["loss_mask"], device=self.device).bool().unsqueeze(0)
        targets = _critic_targets(
            chunk, payloads, steps=len(payloads), device=self.device
        )
        return {
            "actor_inputs": actor_inputs,
            "critic_inputs": {
                name: value for name, value in critic_inputs.items() if value is not None
            },
            "masks": masks,
            "action": action,
            "bc_output": bc_output,
            "old_log_prob": old_log_prob,
            "advantages": advantages,
            "returns": returns,
            "loss_mask": loss_mask,
            "targets": targets,
        }

    @staticmethod
    def _batch_dataclass(values: Sequence[Any], type_: type) -> Any:
        return type_(**{
            name: _batch_tensors(
                [getattr(value, name) for value in values], name=f"Stage2.{name}"
            )
            for name in type_.__dataclass_fields__
        })

    def _combine(self, chunks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        rows = [self._prepare_chunk(chunk) for chunk in chunks]
        bc_names = (
            "rate_logits", "action_kind_logits", "card_logits",
            "position_logits", "ability_logits", "ability_position_logits",
        )
        bc = ExpertPolicyOutput(
            *(_batch_tensors(
                [getattr(row["bc_output"], name) for row in rows],
                name=f"Stage2.bc.{name}",
            ) for name in bc_names),
            (torch.empty(0, device=self.device), torch.empty(0, device=self.device)),
        )
        return {
            "actor_inputs": _batch_input_mappings(
                [row["actor_inputs"] for row in rows], kind="Actor"
            ),
            "critic_inputs": _batch_input_mappings(
                [row["critic_inputs"] for row in rows], kind="Critic"
            ),
            "masks": self._batch_dataclass([row["masks"] for row in rows], ExpertActionMasks),
            "action": self._batch_dataclass([row["action"] for row in rows], RecordedExpertAction),
            "bc_output": bc,
            "old_log_prob": _batch_tensors([row["old_log_prob"] for row in rows], name="old_logp"),
            "advantages": _batch_tensors([row["advantages"] for row in rows], name="advantages"),
            "returns": _batch_tensors([row["returns"] for row in rows], name="returns"),
            "loss_mask": _batch_tensors([row["loss_mask"] for row in rows], name="loss_mask").bool(),
            "targets": _batch_targets([row["targets"] for row in rows]),
        }

    def _chunk_batches(
        self, chunks: Sequence[Mapping[str, Any]]
    ) -> list[list[Mapping[str, Any]]]:
        groups: dict[int, list[Mapping[str, Any]]] = {}
        for chunk in chunks:
            groups.setdefault(len(chunk["step_payloads"]), []).append(chunk)
        result = []
        for length in sorted(groups, reverse=True):
            rows = groups[length]
            for start in range(0, len(rows), self.config.chunk_batch_size):
                result.append(rows[start:start + self.config.chunk_batch_size])
        return result

    def _advantage_statistics(
        self, chunks: Sequence[Mapping[str, Any]]
    ) -> tuple[float, float]:
        values = []
        for chunk in chunks:
            mask = torch.as_tensor(chunk["loss_mask"]).bool()
            values.append(torch.as_tensor(chunk["advantages"]).float()[mask])
        joined = torch.cat(values)
        return float(joined.mean()), float(joined.std(unbiased=False).clamp_min(1e-6))

    def _policy_metrics(
        self,
        batches: Sequence[Sequence[Mapping[str, Any]]],
        *,
        clip_epsilon: float,
    ) -> dict[str, float]:
        log_ratios: list[Tensor] = []
        bc_values: list[Tensor] = []
        entropies: list[Tensor] = []
        rates: list[Tensor] = []
        event_probabilities: list[Tensor] = []
        actual_events: list[Tensor] = []
        self.model.eval()
        with torch.no_grad():
            for chunk_batch in batches:
                batch = self._combine(chunk_batch)
                featured = self.model.actor.forward_with_features(**batch["actor_inputs"])
                evaluated = evaluate_expert_action(
                    output=featured.output,
                    config=self.model.actor.config,
                    masks=batch["masks"],
                    action=batch["action"],
                    delta_ticks=batch["actor_inputs"]["delta_ticks"],
                )
                bc_kl = expert_policy_kl(
                    source=featured.output,
                    target=batch["bc_output"],
                    config=self.model.actor.config,
                    masks=batch["masks"],
                    delta_ticks=batch["actor_inputs"]["delta_ticks"],
                )
                mask = batch["loss_mask"]
                log_ratios.append(
                    (evaluated.log_prob.total - batch["old_log_prob"])[mask].float().cpu()
                )
                bc_values.append(bc_kl[mask].float().cpu())
                entropies.append(evaluated.entropy[mask].float().cpu())
                rates.append(lambda_from_logits(
                    featured.output.rate_logits, self.model.actor.config.lambda_max
                )[mask].float().cpu())
                event_probabilities.append(
                    evaluated.log_prob.event_probability[mask].float().cpu()
                )
                actual_events.append(batch["action"].event_happened[mask].float().cpu())
        self.model.train()
        log_ratio = torch.cat(log_ratios)
        ratio = torch.exp(log_ratio)
        return {
            "approx_update_kl": float(((ratio - 1.0) - log_ratio).mean()),
            "clip_fraction": float(((ratio - 1.0).abs() > clip_epsilon).float().mean()),
            "bc_kl": float(torch.cat(bc_values).mean()),
            "joint_entropy": float(torch.cat(entropies).mean()),
            "rate_mean": float(torch.cat(rates).mean()),
            "event_probability_mean": float(torch.cat(event_probabilities).mean()),
            "actual_event_rate": float(torch.cat(actual_events).mean()),
            "evaluated_steps": int(log_ratio.numel()),
        }

    def _snapshot(self) -> dict[str, Any]:
        return {
            "actor": _clone_module_state(self.model.actor),
            "critic": _clone_module_state(self.model.critic),
            "actor_optimizer": _cpu_optimizer_state(self.actor_optimizer.state_dict()),
            "critic_optimizer": _cpu_optimizer_state(self.critic_optimizer.state_dict()),
            "rng": _capture_rng_state(),
        }

    def _restore_snapshot(self, value: Mapping[str, Any]) -> None:
        _restore_module_state(self.model.actor, value["actor"])
        _restore_module_state(self.model.critic, value["critic"])
        self.actor_optimizer.load_state_dict(value["actor_optimizer"])
        self.critic_optimizer.load_state_dict(value["critic_optimizer"])
        _restore_rng_state(value["rng"])

    def _run_attempt(
        self,
        chunks: Sequence[Mapping[str, Any]],
        *,
        ppo_epochs: int,
        actor_lr_multiplier: float,
        bc_kl_coefficient: float,
    ) -> tuple[dict[str, float], dict[str, float]]:
        specs_by_name = {
            spec.name: spec
            for spec in stage_specs("stage2_reaction")
            if spec.name != "critic"
        }
        for group in self.actor_optimizer.param_groups:
            group_name = str(group.get("group_name", ""))
            if group_name not in specs_by_name:
                raise RuntimeError(f"unknown Stage-2 Actor optimizer group: {group_name}")
            group["lr"] = (
                specs_by_name[group_name].learning_rate * actor_lr_multiplier
            )
        batches = self._chunk_batches(chunks)
        before = self._policy_metrics(batches, clip_epsilon=self.config.clip_epsilon)
        if (
            abs(before["approx_update_kl"]) > 1e-4
            or before["clip_fraction"] > 0.001
        ):
            raise RuntimeError(
                "FP32 master does not reproduce the recorded FP16 behavior policy: "
                f"kl={before['approx_update_kl']:.6g} "
                f"clip_fraction={before['clip_fraction']:.6g}"
            )
        advantage_mean, advantage_std = self._advantage_statistics(chunks)
        accumulators = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "bc_kl_loss": 0.0,
            "critic_auxiliary_loss": 0.0,
        }
        minibatches = 0
        actor_grad_norm_max = 0.0
        critic_grad_norm_max = 0.0
        self.model.train()
        for _epoch in range(ppo_epochs):
            order = list(range(len(batches)))
            random.shuffle(order)
            for batch_index in order:
                batch = self._combine(batches[batch_index])
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                featured = self.model.actor.forward_with_features(**batch["actor_inputs"])
                evaluated = evaluate_expert_action(
                    output=featured.output,
                    config=self.model.actor.config,
                    masks=batch["masks"],
                    action=batch["action"],
                    delta_ticks=batch["actor_inputs"]["delta_ticks"],
                )
                bc_kl = expert_policy_kl(
                    source=featured.output,
                    target=batch["bc_output"],
                    config=self.model.actor.config,
                    masks=batch["masks"],
                    delta_ticks=batch["actor_inputs"]["delta_ticks"],
                )
                critic_output = self.model.critic(
                    actor_latent=featured.pre_head_latent.detach(),
                    **batch["critic_inputs"],
                )
                normalized_advantages = (
                    batch["advantages"] - advantage_mean
                ) / advantage_std
                ppo = recurrent_ppo_loss(
                    new_log_prob=evaluated.log_prob.total,
                    old_log_prob=batch["old_log_prob"],
                    advantages=normalized_advantages,
                    values=critic_output.values,
                    returns=batch["returns"],
                    joint_entropy=evaluated.entropy,
                    bc_kl=bc_kl,
                    loss_mask=batch["loss_mask"],
                    clip_epsilon=self.config.clip_epsilon,
                    value_coefficient=self.config.value_coefficient,
                    entropy_coefficient=self.config.entropy_coefficient,
                    bc_kl_coefficient=bc_kl_coefficient,
                )
                auxiliary = critic_loss(
                    critic_output,
                    batch["targets"],
                    value_coefficient=0.0,
                    auxiliary_coefficient=self.config.critic_auxiliary_coefficient,
                )
                total = ppo.total + auxiliary.total
                if not bool(torch.isfinite(total)):
                    raise FloatingPointError("Stage-2 PPO loss is NaN/Inf")
                total.backward()
                actor_parameters = [
                    parameter for parameter in self.model.actor.parameters()
                    if parameter.requires_grad
                ]
                critic_parameters = [
                    parameter for parameter in self.model.critic.parameters()
                    if parameter.requires_grad
                ]
                actor_norm = torch.nn.utils.clip_grad_norm_(
                    actor_parameters, self.config.actor_grad_clip,
                    error_if_nonfinite=True,
                )
                critic_norm = torch.nn.utils.clip_grad_norm_(
                    critic_parameters, self.config.critic_grad_clip,
                    error_if_nonfinite=True,
                )
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                actor_grad_norm_max = max(actor_grad_norm_max, float(actor_norm))
                critic_grad_norm_max = max(critic_grad_norm_max, float(critic_norm))
                for name, value in (
                    ("loss", total),
                    ("policy_loss", ppo.policy),
                    ("value_loss", ppo.value),
                    ("entropy", ppo.entropy),
                    ("bc_kl_loss", ppo.bc_kl),
                    ("critic_auxiliary_loss", auxiliary.total),
                ):
                    accumulators[name] += float(value.detach())
                minibatches += 1
        if minibatches < 1:
            raise RuntimeError("Stage-2 PPO produced no minibatches")
        after = self._policy_metrics(batches, clip_epsilon=self.config.clip_epsilon)
        metrics = {
            name: value / minibatches for name, value in accumulators.items()
        }
        metrics.update({
            "approx_update_kl": after["approx_update_kl"],
            "clip_fraction": after["clip_fraction"],
            "bc_kl": after["bc_kl"],
            "joint_entropy": after["joint_entropy"],
            "rate_mean_before": before["rate_mean"],
            "behavior_recompute_kl_before": before["approx_update_kl"],
            "behavior_recompute_clip_fraction_before": before["clip_fraction"],
            "rate_mean_after": after["rate_mean"],
            "event_probability_before": before["event_probability_mean"],
            "event_probability_after": after["event_probability_mean"],
            "actual_event_rate": after["actual_event_rate"],
            "actor_grad_norm_max": actor_grad_norm_max,
            "critic_grad_norm_max": critic_grad_norm_max,
            "evaluated_steps": after["evaluated_steps"],
            "ppo_epochs": ppo_epochs,
            "minibatches": minibatches,
            "actor_lr_multiplier": actor_lr_multiplier,
            "bc_kl_coefficient": bc_kl_coefficient,
        })
        if not all(math.isfinite(float(value)) for value in metrics.values()):
            raise FloatingPointError("Stage-2 PPO metrics contain NaN/Inf")
        return metrics, before

    def train_update(
        self, chunks: Sequence[Mapping[str, Any]]
    ) -> tuple[dict[str, float], UpdateGuardDecision, int]:
        if not chunks:
            raise ValueError("Stage-2 update requires recurrent chunks")
        snapshot = self._snapshot()
        for attempt in range(2):
            self._restore_snapshot(snapshot)
            metrics, _before = self._run_attempt(
                chunks,
                ppo_epochs=(self.config.ppo_epochs if attempt == 0 else 1),
                actor_lr_multiplier=(1.0 if attempt == 0 else 0.5),
                bc_kl_coefficient=(
                    self.config.bc_kl_coefficient if attempt == 0
                    else self.config.bc_kl_coefficient * 2.0
                ),
            )
            decision = evaluate_update(metrics, retry_attempt=attempt)
            if metrics["bc_kl"] > self.config.bc_kl_soft_limit:
                reasons = (*decision.reasons, "bc_kl:soft_limit")
                decision = UpdateGuardDecision(
                    "retry" if attempt == 0 else "halt",
                    reasons,
                    0.5 if attempt == 0 else 0.0,
                    1 if attempt == 0 else None,
                    2.0 if attempt == 0 else 0.0,
                )
            if decision.action == "accept":
                if not _finite((self.model.actor.state_dict(), self.model.critic.state_dict())):
                    raise FloatingPointError("Stage-2 accepted non-finite parameters")
                self.global_update += 1
                self.policy_version += 1
                return metrics, decision, attempt
            if decision.action == "halt":
                self._restore_snapshot(snapshot)
                raise RuntimeError(
                    f"Stage-2 update guard halted after retry: {decision.reasons}"
                )
        raise AssertionError("Stage-2 retry loop exhausted")

    def save(
        self,
        directory: Path,
        *,
        metrics: Mapping[str, Any],
        guard: UpdateGuardDecision,
        retry_attempt: int,
        rollout: Mapping[str, Any],
    ) -> tuple[Path, Path]:
        directory = directory.resolve()
        checkpoints = directory / "checkpoints"
        exports = directory / "exports"
        checkpoints.mkdir(parents=True, exist_ok=True)
        exports.mkdir(parents=True, exist_ok=True)
        behavior_state = _half_state(self.model.actor)
        behavior_sha = _state_digest(behavior_state)
        master_sha = actor_state_digest(self.model.actor)
        bundle = {
            "kind": STAGE2_KIND,
            "schema_version": SCHEMA_VERSION,
            "stage": "stage2_reaction",
            "global_update": self.global_update,
            "policy_version": self.policy_version,
            "base_inference_checkpoint": str(self.base_path),
            "base_inference_sha256": self.base_checkpoint_sha256,
            "continuation_checkpoint": str(self.continuation_path),
            "continuation_sha256": self.continuation_sha256,
            "actor_master": _clone_state(self.model.actor, fp32=True),
            "master_actor_sha256": master_sha,
            "actor_behavior_fp16": behavior_state,
            "behavior_actor_sha256": behavior_sha,
            "frozen_bc_behavior_sha256": self.bc_behavior_sha256,
            "critic": _clone_state(self.model.critic),
            "actor_optimizer": _cpu_optimizer_state(self.actor_optimizer.state_dict()),
            "critic_optimizer": _cpu_optimizer_state(self.critic_optimizer.state_dict()),
            "actor_optimizer_parameter_names": self.actor_optimizer_parameter_names,
            "critic_optimizer_parameter_names": self.critic_optimizer_parameter_names,
            "rng": _capture_rng_state(),
            "config": {
                "trainer": asdict(self.config),
                "actor": self.model.actor.config.to_dict(),
                "critic": asdict(self.model.critic.config),
                "stage_report": self.stage_report,
            },
            "metrics": dict(metrics),
            "update_guard": asdict(guard),
            "retry_attempt": retry_attempt,
            "rollout": dict(rollout),
        }
        if not _finite(bundle):
            raise FloatingPointError("Stage-2 checkpoint bundle contains NaN/Inf")
        checkpoint = checkpoints / f"checkpoint-{self.global_update:012d}.pt"
        _atomic_torch(checkpoint, bundle)
        latest = checkpoints / "latest.pt"
        temporary_latest = checkpoints / f".latest.{os.getpid()}.tmp"
        temporary_latest.unlink(missing_ok=True)
        try:
            os.link(checkpoint, temporary_latest)
        except OSError:
            shutil.copyfile(checkpoint, temporary_latest)
        os.replace(temporary_latest, latest)

        export = {
            "kind": EXPERT_INFERENCE_KIND,
            "dataset_manifest_sha256": str(self.base_payload["dataset_manifest_sha256"]),
            "model_config": self.model.actor.config.to_dict(),
            "model_state": behavior_state,
            "global_step": int(self.base_payload.get("global_step", -1)),
            "run_id": f"stage2-policy-v{self.policy_version:06d}",
            "selfplay_policy_version": self.policy_version,
            "selfplay_global_update": self.global_update,
            "actor_state_sha256": behavior_sha,
        }
        export_path = exports / f"actor-policy-{self.policy_version:06d}-fp16.pt"
        _atomic_torch(export_path, export)

        for stale in sorted(checkpoints.glob("checkpoint-*.pt"))[:-self.config.retain_checkpoints]:
            stale.unlink()
        for stale in sorted(exports.glob("actor-policy-*-fp16.pt"))[:-self.config.retain_checkpoints]:
            stale.unlink()
        self.behavior_actor_sha256 = behavior_sha
        self.master_actor_sha256 = master_sha
        return checkpoint, export_path
