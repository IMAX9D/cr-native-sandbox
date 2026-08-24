"""Recurrent PPO for the continuous action-rate hierarchical policy."""

from __future__ import annotations

from dataclasses import asdict
import random
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor

from training.ppo import PPOConfig, generalized_advantage_estimate
from .model import ContinuousRatePolicyValueNet
from .rollout import TimedAgentTrajectory


class ContinuousRatePPOTrainer:
    def __init__(
        self,
        model: ContinuousRatePolicyValueNet,
        *,
        device: torch.device,
        config: PPOConfig = PPOConfig(),
    ) -> None:
        self.model = model
        self.device = device
        self.config = config
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, eps=1e-5
        )

    def _chunks(
        self, trajectories: Iterable[TimedAgentTrajectory]
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        chunks: list[dict[str, Any]] = []
        all_advantages: list[np.ndarray] = []
        prepared: list[tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]] = []
        for trajectory in trajectories:
            arrays = trajectory.arrays()
            advantages, returns = generalized_advantage_estimate(
                arrays["rewards"],
                arrays["values"],
                arrays["dones"],
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
            )
            all_advantages.append(advantages)
            prepared.append((arrays, advantages, returns))
        if not prepared:
            raise ValueError("PPO update requires at least one trajectory")
        concatenated = np.concatenate(all_advantages)
        mean = float(concatenated.mean())
        standard_deviation = float(concatenated.std()) + 1e-8
        for arrays, advantages, returns in prepared:
            normalized = (advantages - mean) / standard_deviation
            length = len(advantages)
            for train_start in range(0, length, self.config.train_length):
                sequence_start = max(0, train_start - self.config.burn_in)
                sequence_end = min(length, train_start + self.config.train_length)
                chunks.append({
                    "arrays": arrays,
                    "advantages": normalized,
                    "returns": returns,
                    "start": sequence_start,
                    "end": sequence_end,
                    "loss_start": train_start - sequence_start,
                })
        values_all = np.concatenate([item[0]["values"] for item in prepared])
        returns_all = np.concatenate([item[2] for item in prepared])
        return_variance = float(np.var(returns_all))
        explained_variance = (
            1.0 - float(np.var(returns_all - values_all)) / return_variance
            if return_variance > 1e-12 else 0.0
        )
        rates_all = np.concatenate([item[0]["rates"] for item in prepared])
        probabilities_all = np.concatenate([
            item[0]["play_probabilities"] for item in prepared
        ])
        valid_all = np.concatenate([
            item[0]["timing_valids"] for item in prepared
        ])
        cards_all = np.concatenate([item[0]["cards"] for item in prepared])
        return chunks, {
            "explained_variance": explained_variance,
            "value_mean": float(np.mean(values_all)),
            "value_abs_max": float(np.max(np.abs(values_all))),
            "return_mean": float(np.mean(returns_all)),
            "return_std": float(np.std(returns_all)),
            "rate_mean": float(np.mean(rates_all)),
            "rate_valid_mean": (
                float(np.mean(rates_all[valid_all])) if valid_all.any() else 0.0
            ),
            "play_probability_mean": float(np.mean(probabilities_all)),
            "play_event_rate_per_decision": float(np.mean(cards_all > 0)),
            "timing_choice_valid_rate": float(np.mean(valid_all)),
        }

    @staticmethod
    def _pad(array: np.ndarray, length: int, *, value: float = 0.0) -> np.ndarray:
        if len(array) == length:
            return array
        result = np.full((length, *array.shape[1:]), value, dtype=array.dtype)
        result[: len(array)] = array
        return result

    def _batch(
        self, chunks: list[dict[str, Any]]
    ) -> dict[str, Tensor | tuple[Tensor, Tensor]]:
        length = max(item["end"] - item["start"] for item in chunks)
        keys = (
            "grid", "scalars", "privileged", "card_masks", "position_masks",
            "cards", "positions", "log_probabilities", "values",
        )
        values: dict[str, list[np.ndarray]] = {
            key: [] for key in (*keys, "advantages", "returns", "loss_mask")
        }
        hidden_h: list[np.ndarray] = []
        hidden_c: list[np.ndarray] = []
        for item in chunks:
            start, end = item["start"], item["end"]
            arrays = item["arrays"]
            for key in keys:
                values[key].append(self._pad(arrays[key][start:end], length))
            values["advantages"].append(
                self._pad(item["advantages"][start:end], length)
            )
            values["returns"].append(
                self._pad(item["returns"][start:end], length)
            )
            mask = np.zeros(length, dtype=np.bool_)
            mask[item["loss_start"] : end - start] = True
            values["loss_mask"].append(mask)
            hidden_h.append(arrays["hidden_h"][start])
            hidden_c.append(arrays["hidden_c"][start])
        batch: dict[str, Any] = {}
        booleans = {"card_masks", "position_masks", "loss_mask"}
        integers = {"cards", "positions"}
        for key, items in values.items():
            tensor = torch.from_numpy(np.stack(items))
            if key in booleans:
                tensor = tensor.bool()
            elif key in integers:
                tensor = tensor.long()
            else:
                tensor = tensor.float()
            batch[key] = tensor.to(self.device)
        batch["hidden"] = (
            torch.from_numpy(np.stack(hidden_h)).float().unsqueeze(0).to(self.device),
            torch.from_numpy(np.stack(hidden_c)).float().unsqueeze(0).to(self.device),
        )
        return batch

    def update(
        self, trajectories: Iterable[TimedAgentTrajectory]
    ) -> dict[str, float]:
        chunks, diagnostics = self._chunks(trajectories)
        self.model.train()
        metrics: dict[str, list[float]] = {
            "loss": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
            "gradient_norm": [],
        }
        for _epoch in range(self.config.epochs):
            random.shuffle(chunks)
            for offset in range(0, len(chunks), self.config.chunk_batch_size):
                batch = self._batch(
                    chunks[offset : offset + self.config.chunk_batch_size]
                )
                new_log_probability, entropy, values, _hidden = (
                    self.model.evaluate_actions(
                        batch["grid"],
                        batch["scalars"],
                        batch["privileged"],
                        batch["card_masks"],
                        batch["position_masks"],
                        batch["cards"],
                        batch["positions"],
                        batch["hidden"],
                    )
                )
                mask = batch["loss_mask"]
                log_ratio = (
                    new_log_probability[mask]
                    - batch["log_probabilities"][mask]
                )
                ratio = torch.exp(log_ratio)
                advantages = batch["advantages"][mask]
                unclipped = ratio * advantages
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.config.clip,
                    1.0 + self.config.clip,
                ) * advantages
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = torch.nn.functional.mse_loss(
                    values[mask], batch["returns"][mask]
                )
                mean_entropy = entropy[mask].mean()
                loss = (
                    policy_loss
                    + self.config.value_coefficient * value_loss
                    - self.config.entropy_coefficient * mean_entropy
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip
                )
                self.optimizer.step()
                with torch.no_grad():
                    approx_kl = ((torch.exp(log_ratio) - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        (ratio - 1.0).abs() > self.config.clip
                    ).float().mean()
                for key, value in (
                    ("loss", loss),
                    ("policy_loss", policy_loss),
                    ("value_loss", value_loss),
                    ("entropy", mean_entropy),
                    ("approx_kl", approx_kl),
                    ("clip_fraction", clip_fraction),
                    ("gradient_norm", gradient_norm),
                ):
                    metrics[key].append(float(value.detach().cpu()))
        result = {key: float(np.mean(value)) for key, value in metrics.items()}
        result.update(diagnostics)
        result["learning_rate"] = float(self.optimizer.param_groups[0]["lr"])
        return result

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.config)
