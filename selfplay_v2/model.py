"""20 Hz recurrent actor/critic with a bounded action-rate timing head."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from training.schema import GRID_CHANNELS, PRIVILEGED_SIZE, SCALAR_SIZE
from .action import (
    DEFAULT_LAMBDA_MAX,
    NATIVE_TICK_SECONDS,
    initial_rate_bias,
    rate_distribution,
    sample_play_now,
    timing_log_probability,
)


class TimedPolicyOutput(NamedTuple):
    rate_logits: Tensor
    card_logits: Tensor
    position_logits: Tensor
    values: Tensor
    hidden: tuple[Tensor, Tensor]


@dataclass(frozen=True)
class SampledTimedAction:
    card: int
    position: int
    play_now: bool
    timing_valid: bool
    log_probability: float
    value: float
    rate: float
    play_probability: float
    entropy: float
    hidden: tuple[Tensor, Tensor]
    position_diagnostics: tuple[dict[str, float] | None, ...]


def _masked_logits(logits: Tensor, mask: Tensor) -> Tensor:
    if logits.shape != mask.shape:
        raise ValueError(f"mask shape {mask.shape} != logits shape {logits.shape}")
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


def _safe_card_mask(mask: Tensor, timing_valid: Tensor) -> Tensor:
    safe = mask.clone()
    invalid = ~timing_valid
    if invalid.any():
        safe[invalid] = False
        safe[invalid, 0] = True
    return safe


def _safe_position_masks(masks: Tensor) -> Tensor:
    safe = masks.clone()
    empty = ~safe.any(dim=-1)
    if empty.any():
        safe[empty] = False
        safe[..., 0] |= empty
    return safe


def _position_distributions(
    logits: Tensor, masks: Tensor
) -> tuple[Categorical, Tensor]:
    safe = _safe_position_masks(masks)
    distribution = Categorical(logits=_masked_logits(logits, safe))
    return distribution, distribution.entropy()


class _CudaGraphStep:
    def __init__(
        self,
        model: "ContinuousRatePolicyValueNet",
        grid: Tensor,
        scalars: Tensor,
        privileged: Tensor,
        hidden: tuple[Tensor, Tensor],
    ) -> None:
        self.grid = grid.clone()
        self.scalars = scalars.clone()
        self.privileged = privileged.clone()
        self.hidden = (hidden[0].clone(), hidden[1].clone())
        warmup = torch.cuda.Stream(device=grid.device)
        warmup.wait_stream(torch.cuda.current_stream(grid.device))
        with torch.cuda.stream(warmup):
            for _ in range(3):
                model.forward_step(
                    self.grid, self.scalars, self.privileged, self.hidden
                )
        torch.cuda.current_stream(grid.device).wait_stream(warmup)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = model.forward_step(
                self.grid, self.scalars, self.privileged, self.hidden
            )

    def run(
        self,
        grid: Tensor,
        scalars: Tensor,
        privileged: Tensor,
        hidden: tuple[Tensor, Tensor],
    ) -> TimedPolicyOutput:
        self.grid.copy_(grid)
        self.scalars.copy_(scalars)
        self.privileged.copy_(privileged)
        self.hidden[0].copy_(hidden[0])
        self.hidden[1].copy_(hidden[1])
        self.graph.replay()
        return self.output


class ContinuousRatePolicyValueNet(nn.Module):
    def __init__(
        self,
        hidden_size: int = 256,
        *,
        lambda_max: float = DEFAULT_LAMBDA_MAX,
        lambda_initial: float = 0.20,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.lambda_max = float(lambda_max)
        self.lambda_initial = float(lambda_initial)
        self.spatial = nn.Sequential(
            nn.Conv2d(GRID_CHANNELS, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.SiLU(),
        )
        self.position_map = nn.Conv2d(64, 4, 1)
        self.public_scalar = nn.Sequential(
            nn.Linear(SCALAR_SIZE, 96),
            nn.SiLU(),
            nn.Linear(96, 64),
            nn.SiLU(),
        )
        self.recurrent = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.rate_head = nn.Linear(hidden_size, 1)
        self.card_head = nn.Linear(hidden_size, 4)
        self.position_context = nn.Linear(hidden_size, 4)
        self.privileged = nn.Sequential(
            nn.Linear(PRIVILEGED_SIZE, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size + 64, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )
        self.reset_rate_prior(lambda_initial)
        self._cuda_graph_inference = False
        self._cuda_graph_steps: dict[
            tuple[int, int | None, torch.dtype], _CudaGraphStep
        ] = {}
        self.cuda_graph_stats: dict[str, float] = {
            "captures": 0.0,
            "capture_seconds": 0.0,
            "replays": 0.0,
        }

    def reset_rate_prior(self, lambda_initial: float) -> None:
        self.lambda_initial = float(lambda_initial)
        with torch.no_grad():
            self.rate_head.weight.zero_()
            self.rate_head.bias.fill_(
                initial_rate_bias(self.lambda_initial, self.lambda_max)
            )

    def enable_cuda_graph_inference(self, enabled: bool = True) -> None:
        if not enabled:
            self._cuda_graph_steps.clear()
        self._cuda_graph_inference = enabled

    def initial_hidden(
        self, batch_size: int, *, device: torch.device | str
    ) -> tuple[Tensor, Tensor]:
        shape = (1, batch_size, self.hidden_size)
        return (
            torch.zeros(shape, device=device),
            torch.zeros(shape, device=device),
        )

    def forward_sequence(
        self,
        grid: Tensor,
        scalars: Tensor,
        privileged: Tensor,
        hidden: tuple[Tensor, Tensor] | None = None,
    ) -> TimedPolicyOutput:
        if grid.ndim != 5:
            raise ValueError("grid must have shape [batch,time,channels,32,18]")
        batch, steps = grid.shape[:2]
        spatial = self.spatial(grid.flatten(0, 1))
        pooled = spatial.mean(dim=(-2, -1)).reshape(batch, steps, 64)
        public = self.public_scalar(scalars)
        recurrent, next_hidden = self.recurrent(
            torch.cat((pooled, public), dim=-1), hidden
        )
        rate_logits = self.rate_head(recurrent).squeeze(-1)
        card_logits = self.card_head(recurrent)
        position = self.position_map(spatial).reshape(
            batch, steps, 4, 32 * 18
        )
        position = position + self.position_context(recurrent).unsqueeze(-1)
        private = self.privileged(privileged)
        values = self.value_head(
            torch.cat((recurrent, private), dim=-1)
        ).squeeze(-1)
        return TimedPolicyOutput(
            rate_logits, card_logits, position, values, next_hidden
        )

    def forward_step(
        self,
        grid: Tensor,
        scalars: Tensor,
        privileged: Tensor,
        hidden: tuple[Tensor, Tensor] | None = None,
    ) -> TimedPolicyOutput:
        output = self.forward_sequence(
            grid.unsqueeze(1),
            scalars.unsqueeze(1),
            privileged.unsqueeze(1),
            hidden,
        )
        return TimedPolicyOutput(
            output.rate_logits[:, 0],
            output.card_logits[:, 0],
            output.position_logits[:, 0],
            output.values[:, 0],
            output.hidden,
        )

    def _inference_step(
        self,
        grid: Tensor,
        scalars: Tensor,
        privileged: Tensor,
        hidden: tuple[Tensor, Tensor] | None,
    ) -> tuple[TimedPolicyOutput, bool]:
        if not (
            self._cuda_graph_inference
            and grid.is_cuda
            and hidden is not None
        ):
            return self.forward_step(grid, scalars, privileged, hidden), False
        key = (grid.shape[0], grid.device.index, grid.dtype)
        runner = self._cuda_graph_steps.get(key)
        if runner is None:
            started = time.perf_counter()
            runner = _CudaGraphStep(self, grid, scalars, privileged, hidden)
            self._cuda_graph_steps[key] = runner
            self.cuda_graph_stats["captures"] += 1.0
            self.cuda_graph_stats["capture_seconds"] += time.perf_counter() - started
        self.cuda_graph_stats["replays"] += 1.0
        return runner.run(grid, scalars, privileged, hidden), True

    @torch.inference_mode()
    def sample_batch(
        self,
        grid: Tensor,
        scalars: Tensor,
        privileged: Tensor,
        card_masks: Tensor,
        position_masks: Tensor,
        hidden: tuple[Tensor, Tensor] | None,
        *,
        deterministic: bool = False,
        collect_position_diagnostics: bool = True,
    ) -> list[SampledTimedAction]:
        if card_masks.shape != (grid.shape[0], 4):
            raise ValueError("card_masks must have shape [batch,4]")
        output, graph_backed = self._inference_step(
            grid, scalars, privileged, hidden
        )
        stable_hidden = (
            (output.hidden[0].clone(), output.hidden[1].clone())
            if graph_backed else output.hidden
        )
        timing_valid = card_masks.any(dim=-1)
        timing = rate_distribution(
            output.rate_logits,
            lambda_max=self.lambda_max,
            tick_seconds=NATIVE_TICK_SECONDS,
        )
        if deterministic:
            play_now = timing_valid & (timing.play_probability >= 0.5)
        else:
            play_now = sample_play_now(timing, timing_valid=timing_valid)
        timing_log_prob = timing_log_probability(
            timing, play_now=play_now, timing_valid=timing_valid
        )
        safe_cards = _safe_card_mask(card_masks, timing_valid)
        card_distribution = Categorical(
            logits=_masked_logits(output.card_logits, safe_cards)
        )
        card_indices = (
            torch.argmax(card_distribution.logits, dim=-1)
            if deterministic else card_distribution.sample()
        )
        card_log_prob = card_distribution.log_prob(card_indices)
        all_positions, position_entropies = _position_distributions(
            output.position_logits, position_masks
        )
        card_entropy = card_distribution.entropy()
        conditional_entropy = card_entropy + (
            card_distribution.probs * position_entropies
        ).sum(dim=-1)
        total_entropy = torch.where(
            timing_valid,
            timing.entropy + timing.play_probability * conditional_entropy,
            torch.zeros_like(timing.entropy),
        )
        results: list[SampledTimedAction] = []
        for index in range(grid.shape[0]):
            card_index = int(card_indices[index].item())
            position = 0
            log_probability = timing_log_prob[index]
            if bool(play_now[index]):
                position_distribution = Categorical(
                    logits=_masked_logits(
                        output.position_logits[index, card_index],
                        position_masks[index, card_index],
                    )
                )
                position_tensor = (
                    torch.argmax(position_distribution.logits, dim=-1)
                    if deterministic else position_distribution.sample()
                )
                position = int(position_tensor.item())
                log_probability = (
                    log_probability
                    + card_log_prob[index]
                    + position_distribution.log_prob(position_tensor)
                )
            diagnostics: list[dict[str, float] | None] = []
            for slot in range(4):
                if (
                    not collect_position_diagnostics
                    or not bool(card_masks[index, slot])
                ):
                    diagnostics.append(None)
                    continue
                probabilities = all_positions.probs[index, slot]
                entropy = float(position_entropies[index, slot].item())
                legal_count = int(position_masks[index, slot].sum().item())
                top_count = min(5, legal_count)
                top_values = torch.topk(probabilities, top_count).values
                diagnostics.append({
                    "legal_cells": float(legal_count),
                    "entropy": entropy,
                    "normalized_entropy": (
                        entropy / math.log(legal_count)
                        if legal_count > 1 else 0.0
                    ),
                    "effective_cells": math.exp(entropy),
                    "top1_mass": float(top_values[0].item()),
                    "top5_mass": float(top_values.sum().item()),
                })
            results.append(SampledTimedAction(
                card=card_index + 1 if bool(play_now[index]) else 0,
                position=position,
                play_now=bool(play_now[index]),
                timing_valid=bool(timing_valid[index]),
                log_probability=float(log_probability.item()),
                value=float(output.values[index].item()),
                rate=float(timing.rate[index].item()),
                play_probability=float(timing.play_probability[index].item()),
                entropy=float(total_entropy[index].item()),
                hidden=(
                    stable_hidden[0][:, index : index + 1].detach(),
                    stable_hidden[1][:, index : index + 1].detach(),
                ),
                position_diagnostics=tuple(diagnostics),
            ))
        return results

    def evaluate_actions(
        self,
        grid: Tensor,
        scalars: Tensor,
        privileged: Tensor,
        card_masks: Tensor,
        position_masks: Tensor,
        cards: Tensor,
        positions: Tensor,
        hidden: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, tuple[Tensor, Tensor]]:
        output = self.forward_sequence(grid, scalars, privileged, hidden)
        timing_valid = card_masks.any(dim=-1)
        play_now = cards > 0
        timing = rate_distribution(
            output.rate_logits,
            lambda_max=self.lambda_max,
            tick_seconds=NATIVE_TICK_SECONDS,
        )
        log_probability = timing_log_probability(
            timing, play_now=play_now, timing_valid=timing_valid
        )
        safe_cards = _safe_card_mask(card_masks, timing_valid)
        card_distribution = Categorical(
            logits=_masked_logits(output.card_logits, safe_cards)
        )
        card_indices = (cards - 1).clamp(min=0)
        card_log_probability = card_distribution.log_prob(card_indices)
        selected_index = card_indices.unsqueeze(-1).unsqueeze(-1).expand(
            *cards.shape, 1, 32 * 18
        )
        selected_logits = output.position_logits.gather(
            2, selected_index
        ).squeeze(2)
        selected_masks = position_masks.gather(2, selected_index).squeeze(2)
        safe_selected = selected_masks.clone()
        empty = ~safe_selected.any(dim=-1)
        if empty.any():
            safe_selected[empty] = False
            safe_selected[empty, 0] = True
        position_distribution = Categorical(
            logits=_masked_logits(selected_logits, safe_selected)
        )
        position_log_probability = position_distribution.log_prob(positions)
        log_probability = log_probability + torch.where(
            play_now,
            card_log_probability + position_log_probability,
            torch.zeros_like(log_probability),
        )
        _all_position_distribution, position_entropies = _position_distributions(
            output.position_logits, position_masks
        )
        conditional_entropy = card_distribution.entropy() + (
            card_distribution.probs * position_entropies
        ).sum(dim=-1)
        entropy = torch.where(
            timing_valid,
            timing.entropy + timing.play_probability * conditional_entropy,
            torch.zeros_like(timing.entropy),
        )
        return log_probability, entropy, output.values, output.hidden
