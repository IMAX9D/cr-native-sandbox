"""Strict legal baselines for action-rate coverage and evaluation."""

from __future__ import annotations

import math

import numpy as np
import torch

from .action import NATIVE_TICK_SECONDS
from .model import SampledTimedAction


class RandomRateLegalPolicy:
    hidden_size = 1
    cuda_graph_stats: dict[str, float] = {}

    def __init__(self, *, rate: float, seed: int) -> None:
        if rate <= 0.0:
            raise ValueError("random rate must be positive")
        self.rate = float(rate)
        self.play_probability = 1.0 - math.exp(
            -self.rate * NATIVE_TICK_SECONDS
        )
        self.generator = np.random.default_rng(int(seed))

    def eval(self) -> "RandomRateLegalPolicy":
        return self

    def initial_hidden(
        self, batch_size: int, *, device: torch.device | str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (1, batch_size, 1)
        return torch.zeros(shape, device=device), torch.zeros(shape, device=device)

    @torch.inference_mode()
    def sample_batch(
        self,
        grid: torch.Tensor,
        scalars: torch.Tensor,
        privileged: torch.Tensor,
        card_masks: torch.Tensor,
        position_masks: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None,
        *,
        deterministic: bool = False,
    ) -> list[SampledTimedAction]:
        del scalars, privileged
        if deterministic:
            raise ValueError("RandomRateLegalPolicy has no deterministic mode")
        if hidden is None:
            hidden = self.initial_hidden(grid.shape[0], device=grid.device)
        cards = card_masks.detach().cpu().numpy().astype(np.bool_, copy=False)
        positions = position_masks.detach().cpu().numpy().astype(np.bool_, copy=False)
        results: list[SampledTimedAction] = []
        for index in range(grid.shape[0]):
            legal_cards = np.flatnonzero(cards[index])
            timing_valid = bool(len(legal_cards))
            play_now = bool(
                timing_valid
                and self.generator.random() < self.play_probability
            )
            log_probability = 0.0
            card = 0
            position = 0
            diagnostics: list[dict[str, float] | None] = []
            conditional_entropy = 0.0
            if timing_valid:
                timing_entropy = -(
                    self.play_probability * math.log(self.play_probability)
                    + (1.0 - self.play_probability)
                    * math.log(1.0 - self.play_probability)
                )
                position_entropies = []
                for slot in range(4):
                    count = int(positions[index, slot].sum())
                    if not cards[index, slot]:
                        diagnostics.append(None)
                        continue
                    entropy = math.log(count)
                    position_entropies.append(entropy)
                    diagnostics.append({
                        "legal_cells": float(count),
                        "entropy": entropy,
                        "normalized_entropy": 1.0 if count > 1 else 0.0,
                        "effective_cells": float(count),
                        "top1_mass": 1.0 / count,
                        "top5_mass": min(5, count) / count,
                    })
                conditional_entropy = math.log(len(legal_cards)) + float(
                    np.mean(position_entropies)
                )
                log_probability = math.log(
                    self.play_probability if play_now
                    else 1.0 - self.play_probability
                )
            else:
                timing_entropy = 0.0
                diagnostics = [None, None, None, None]
            if play_now:
                card_index = int(self.generator.choice(legal_cards))
                legal_positions = np.flatnonzero(positions[index, card_index])
                position = int(self.generator.choice(legal_positions))
                card = card_index + 1
                log_probability -= math.log(len(legal_cards))
                log_probability -= math.log(len(legal_positions))
            results.append(SampledTimedAction(
                card=card,
                position=position,
                play_now=play_now,
                timing_valid=timing_valid,
                log_probability=log_probability,
                value=0.0,
                rate=self.rate,
                play_probability=self.play_probability,
                entropy=(
                    timing_entropy
                    + self.play_probability * conditional_entropy
                ),
                hidden=(
                    hidden[0][:, index : index + 1].detach().clone(),
                    hidden[1][:, index : index + 1].detach().clone(),
                ),
                position_diagnostics=tuple(diagnostics),
            ))
        return results
