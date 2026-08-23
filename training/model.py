"""Small recurrent actor / privileged-critic network for the eight-card game."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from .schema import GRID_CHANNELS, PRIVILEGED_SIZE, SCALAR_SIZE


class PolicyOutput(NamedTuple):
    card_logits: Tensor
    position_logits: Tensor
    values: Tensor
    hidden: tuple[Tensor, Tensor]


@dataclass(frozen=True)
class SampledAction:
    card: int
    position: int
    log_probability: float
    value: float
    hidden: tuple[Tensor, Tensor]


class _CudaGraphStep:
    """Shape-specialized CUDA Graph for the pure recurrent network forward."""

    def __init__(
        self,
        model: "RecurrentPolicyValueNet",
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
    ) -> PolicyOutput:
        self.grid.copy_(grid)
        self.scalars.copy_(scalars)
        self.privileged.copy_(privileged)
        self.hidden[0].copy_(hidden[0])
        self.hidden[1].copy_(hidden[1])
        self.graph.replay()
        return self.output


def _masked_logits(logits: Tensor, mask: Tensor) -> Tensor:
    if logits.shape != mask.shape:
        raise ValueError(f"mask shape {mask.shape} != logits shape {logits.shape}")
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


class RecurrentPolicyValueNet(nn.Module):
    """Actor sees public tensors; only the value head sees privileged data."""

    def __init__(self, hidden_size: int = 256) -> None:
        super().__init__()
        self.hidden_size = hidden_size
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
            nn.Linear(SCALAR_SIZE, 96), nn.SiLU(), nn.Linear(96, 64), nn.SiLU()
        )
        self.recurrent = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.card_head = nn.Linear(hidden_size, 5)
        self.position_context = nn.Linear(hidden_size, 4)
        self.privileged = nn.Sequential(
            nn.Linear(PRIVILEGED_SIZE, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU()
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size + 64, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )
        self._cuda_graph_inference = False
        self._cuda_graph_steps: dict[
            tuple[int, int | None, torch.dtype], _CudaGraphStep
        ] = {}
        self.cuda_graph_stats: dict[str, float] = {
            "captures": 0.0,
            "capture_seconds": 0.0,
            "replays": 0.0,
        }

    def enable_cuda_graph_inference(self, enabled: bool = True) -> None:
        if not enabled:
            self._cuda_graph_steps.clear()
        self._cuda_graph_inference = enabled

    def _inference_step(
        self,
        grid: Tensor,
        scalars: Tensor,
        privileged: Tensor,
        hidden: tuple[Tensor, Tensor] | None,
    ) -> tuple[PolicyOutput, bool]:
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
            runner = _CudaGraphStep(
                self, grid, scalars, privileged, hidden
            )
            self._cuda_graph_steps[key] = runner
            self.cuda_graph_stats["captures"] += 1.0
            self.cuda_graph_stats["capture_seconds"] += (
                time.perf_counter() - started
            )
        self.cuda_graph_stats["replays"] += 1.0
        return runner.run(grid, scalars, privileged, hidden), True

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
    ) -> PolicyOutput:
        if grid.ndim != 5:
            raise ValueError("grid must have shape [batch,time,channels,32,18]")
        batch, steps = grid.shape[:2]
        spatial = self.spatial(grid.flatten(0, 1))
        pooled = spatial.mean(dim=(-2, -1)).reshape(batch, steps, 64)
        public = self.public_scalar(scalars)
        recurrent, next_hidden = self.recurrent(
            torch.cat((pooled, public), dim=-1), hidden
        )
        card_logits = self.card_head(recurrent)
        position = self.position_map(spatial).reshape(batch, steps, 4, 32 * 18)
        position = position + self.position_context(recurrent).unsqueeze(-1)
        critic_private = self.privileged(privileged)
        values = self.value_head(
            torch.cat((recurrent, critic_private), dim=-1)
        ).squeeze(-1)
        return PolicyOutput(card_logits, position, values, next_hidden)

    def forward_step(
        self,
        grid: Tensor,
        scalars: Tensor,
        privileged: Tensor,
        hidden: tuple[Tensor, Tensor] | None = None,
    ) -> PolicyOutput:
        output = self.forward_sequence(
            grid.unsqueeze(1),
            scalars.unsqueeze(1),
            privileged.unsqueeze(1),
            hidden,
        )
        return PolicyOutput(
            output.card_logits[:, 0],
            output.position_logits[:, 0],
            output.values[:, 0],
            output.hidden,
        )

    @torch.inference_mode()
    def sample_batch(
        self,
        grid: Tensor,
        scalars: Tensor,
        privileged: Tensor,
        card_mask: Tensor,
        position_masks: Tensor,
        hidden: tuple[Tensor, Tensor] | None,
        *,
        deterministic: bool = False,
    ) -> list[SampledAction]:
        output, graph_backed = self._inference_step(
            grid, scalars, privileged, hidden
        )
        stable_hidden = (
            (output.hidden[0].clone(), output.hidden[1].clone())
            if graph_backed else output.hidden
        )
        card_distribution = Categorical(
            logits=_masked_logits(output.card_logits, card_mask)
        )
        card_tensor = (
            torch.argmax(card_distribution.logits, dim=-1)
            if deterministic
            else card_distribution.sample()
        )
        card_log_probability = card_distribution.log_prob(card_tensor)
        results: list[SampledAction] = []
        for index in range(grid.shape[0]):
            card = int(card_tensor[index].item())
            log_probability = card_log_probability[index]
            position = 0
            if card > 0:
                logits = output.position_logits[index, card - 1]
                mask = position_masks[index, card - 1]
                position_distribution = Categorical(
                    logits=_masked_logits(logits, mask)
                )
                position_value = (
                    torch.argmax(position_distribution.logits, dim=-1)
                    if deterministic
                    else position_distribution.sample()
                )
                position = int(position_value.item())
                log_probability = log_probability + position_distribution.log_prob(
                    position_value
                )
            results.append(
                SampledAction(
                    card=card,
                    position=position,
                    log_probability=float(log_probability.item()),
                    value=float(output.values[index].item()),
                    hidden=(
                        stable_hidden[0][:, index : index + 1].detach(),
                        stable_hidden[1][:, index : index + 1].detach(),
                    ),
                )
            )
        return results

    @torch.inference_mode()
    def sample(
        self,
        grid: Tensor,
        scalars: Tensor,
        privileged: Tensor,
        card_mask: Tensor,
        position_masks: Tensor,
        hidden: tuple[Tensor, Tensor] | None,
        *,
        deterministic: bool = False,
    ) -> SampledAction:
        results = self.sample_batch(
            grid,
            scalars,
            privileged,
            card_mask,
            position_masks,
            hidden,
            deterministic=deterministic,
        )
        if len(results) != 1:
            raise ValueError("sample requires batch size 1; use sample_batch")
        return results[0]

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
        card_distribution = Categorical(
            logits=_masked_logits(output.card_logits, card_masks)
        )
        log_probability = card_distribution.log_prob(cards)
        entropy = card_distribution.entropy()
        playing = cards > 0
        if playing.any():
            selected_index = (cards - 1).clamp_min(0).unsqueeze(-1).unsqueeze(-1)
            selected_index = selected_index.expand(*cards.shape, 1, 32 * 18)
            selected_logits = output.position_logits.gather(2, selected_index).squeeze(2)
            selected_masks = position_masks.gather(2, selected_index).squeeze(2)
            position_distribution = Categorical(
                logits=_masked_logits(selected_logits, selected_masks)
            )
            position_log_probability = position_distribution.log_prob(positions)
            position_entropy = position_distribution.entropy()
            log_probability = log_probability + torch.where(
                playing, position_log_probability, torch.zeros_like(log_probability)
            )
            entropy = entropy + torch.where(
                playing, position_entropy, torch.zeros_like(entropy)
            )
        return log_probability, entropy, output.values, output.hidden
