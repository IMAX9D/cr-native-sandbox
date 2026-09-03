"""Variable-native-time generalized advantage estimation."""

from __future__ import annotations

import numpy as np


def variable_time_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    terminated: np.ndarray,
    delta_ticks: np.ndarray,
    *,
    bootstrap_value: float,
    gamma_per_tick: float = 0.99995,
    gae_lambda_per_tick: float = 0.995,
) -> tuple[np.ndarray, np.ndarray]:
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    terminated = np.asarray(terminated, dtype=np.bool_)
    delta_ticks = np.asarray(delta_ticks, dtype=np.int64)
    if not (rewards.shape == values.shape == terminated.shape == delta_ticks.shape):
        raise ValueError("GAE arrays must share one shape")
    if rewards.ndim != 1 or not len(rewards):
        raise ValueError("GAE requires one non-empty trajectory")
    if np.any(delta_ticks < 1):
        raise ValueError("delta_ticks must be positive")
    if not 0.0 < gamma_per_tick <= 1.0 or not 0.0 < gae_lambda_per_tick <= 1.0:
        raise ValueError("per-tick discount factors must be in (0,1]")
    advantages = np.zeros_like(rewards)
    gae = 0.0
    next_value = float(bootstrap_value)
    for index in range(len(rewards) - 1, -1, -1):
        continuation = 0.0 if terminated[index] else 1.0
        gamma = gamma_per_tick ** int(delta_ticks[index])
        trace = gae_lambda_per_tick ** int(delta_ticks[index])
        delta = rewards[index] + gamma * continuation * next_value - values[index]
        gae = delta + gamma * trace * continuation * gae
        advantages[index] = gae
        next_value = float(values[index])
    return advantages, (advantages + values).astype(np.float32)


def discount_interval_rewards(
    per_tick_rewards: np.ndarray, *, gamma_per_tick: float = 0.99995
) -> float:
    values = np.asarray(per_tick_rewards, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("interval rewards must be a non-empty vector")
    powers = np.power(float(gamma_per_tick), np.arange(len(values), dtype=np.float64))
    return float(np.dot(values, powers))
