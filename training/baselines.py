"""Auditable non-learning policies used as v0.1 evaluation baselines."""

from __future__ import annotations

import math

import numpy as np
import torch

from .model import SampledAction


class RandomLegalPolicy:
    """Uniform legal high-level action, then uniform legal deployment cell."""

    hidden_size = 1
    cuda_graph_stats: dict[str, float] = {}

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self.generator = np.random.default_rng(self.seed)

    def eval(self) -> "RandomLegalPolicy":
        return self

    def initial_hidden(
        self, batch_size: int, *, device: torch.device | str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (1, batch_size, self.hidden_size)
        return (
            torch.zeros(shape, device=device),
            torch.zeros(shape, device=device),
        )

    @torch.inference_mode()
    def sample_batch(
        self,
        grid: torch.Tensor,
        scalars: torch.Tensor,
        privileged: torch.Tensor,
        card_mask: torch.Tensor,
        position_masks: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None,
        *,
        deterministic: bool = False,
    ) -> list[SampledAction]:
        del scalars, privileged
        if deterministic:
            raise ValueError("RandomLegalPolicy has no deterministic mode")
        if card_mask.shape != (grid.shape[0], 5):
            raise ValueError("random baseline card mask shape mismatch")
        if position_masks.shape != (grid.shape[0], 4, 32 * 18):
            raise ValueError("random baseline position mask shape mismatch")
        if hidden is None:
            hidden = self.initial_hidden(grid.shape[0], device=grid.device)
        cards = card_mask.detach().cpu().numpy().astype(np.bool_, copy=False)
        positions = (
            position_masks.detach().cpu().numpy().astype(np.bool_, copy=False)
        )
        results: list[SampledAction] = []
        for index in range(grid.shape[0]):
            legal_cards = np.flatnonzero(cards[index])
            if len(legal_cards) == 0 or legal_cards[0] != 0:
                raise RuntimeError("WAIT must be present in every legal card mask")
            card = int(self.generator.choice(legal_cards))
            position = 0
            log_probability = -math.log(len(legal_cards))
            if card > 0:
                legal_positions = np.flatnonzero(positions[index, card - 1])
                if len(legal_positions) == 0:
                    raise RuntimeError("playable card has no legal deployment cell")
                position = int(self.generator.choice(legal_positions))
                log_probability -= math.log(len(legal_positions))
            results.append(SampledAction(
                card=card,
                position=position,
                log_probability=log_probability,
                value=0.0,
                hidden=(
                    hidden[0][:, index : index + 1].detach().clone(),
                    hidden[1][:, index : index + 1].detach().clone(),
                ),
            ))
        return results
