"""Bounded CPU cache for immutable recurrent PPO minibatch inputs."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import fields, is_dataclass
import time
from typing import Any, Callable, Hashable

import torch


def map_tensors(value: Any, function: Callable[[torch.Tensor], torch.Tensor]) -> Any:
    if isinstance(value, torch.Tensor):
        return function(value)
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(**{
            field.name: map_tensors(getattr(value, field.name), function)
            for field in fields(value)
        })
    if isinstance(value, dict):
        return {key: map_tensors(item, function) for key, item in value.items()}
    if isinstance(value, tuple):
        items = [map_tensors(item, function) for item in value]
        return type(value)(*items) if hasattr(value, "_fields") else tuple(items)
    if isinstance(value, list):
        return [map_tensors(item, function) for item in value]
    return value


def tensor_bytes(value: Any) -> int:
    total = 0

    def count(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal total
        total += tensor.numel() * tensor.element_size()
        return tensor

    map_tensors(value, count)
    return total


class PreparedBatchCache:
    def __init__(self, maximum_bytes: int, *, pin_memory: bool = False) -> None:
        if maximum_bytes < 0:
            raise ValueError("prepared minibatch cache size cannot be negative")
        self.maximum_bytes = int(maximum_bytes)
        self.pin_memory = bool(pin_memory)
        self.rows: OrderedDict[Hashable, tuple[Any, int]] = OrderedDict()
        self.bytes = 0
        self.peak_bytes = 0
        self.hits = 0
        self.misses = 0
        self.build_seconds = 0.0

    def get(self, key: Hashable, build: Callable[[], Any]) -> Any:
        existing = self.rows.pop(key, None)
        if existing is not None:
            self.rows[key] = existing
            self.hits += 1
            return existing[0]
        self.misses += 1
        started = time.perf_counter()
        value = build()
        size = tensor_bytes(value)
        if self.maximum_bytes and size <= self.maximum_bytes:
            while self.bytes + size > self.maximum_bytes:
                _key, (_old, length) = self.rows.popitem(last=False)
                self.bytes -= length
            if self.pin_memory:
                value = map_tensors(value, lambda tensor: tensor.pin_memory())
            self.rows[key] = (value, size)
            self.bytes += size
            self.peak_bytes = max(self.peak_bytes, self.bytes)
        self.build_seconds += time.perf_counter() - started
        return value

    def clear(self) -> None:
        self.rows.clear()
        self.bytes = 0

    def metrics(self) -> dict[str, float | int]:
        return {
            "prepared_cache_hits": self.hits,
            "prepared_cache_misses": self.misses,
            "prepared_cache_peak_bytes": self.peak_bytes,
            "prepared_batch_build_seconds": self.build_seconds,
        }


def batch_to_device(value: Any, device: torch.device) -> Any:
    if device.type == "cpu":
        return value
    return map_tensors(value, lambda tensor: tensor.to(device, non_blocking=True))
