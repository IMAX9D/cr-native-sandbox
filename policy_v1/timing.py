"""Opt-in synchronized wall timings; never enabled during normal training."""

from contextlib import contextmanager
from collections import defaultdict
import time

import torch


STAGES = ("data_wait", "host_to_device", "forward", "loss", "backward", "optimizer")


def timed_batches(loader):
    # First wait includes iterator/worker startup. Later waits measure only
    # next(), not the preceding model work, logging, or checkpoint save.
    started = time.perf_counter()
    iterator = iter(loader)
    while True:
        try:
            batch = next(iterator)
        except StopIteration:
            return
        yield batch, time.perf_counter() - started
        started = time.perf_counter()


class StageTimer:
    def __init__(self, device, *, enabled=False, warmup=5):
        if warmup < 0:
            raise ValueError("profile warmup must be non-negative")
        self.device = device
        self.enabled = enabled
        self.warmup = warmup
        self.seen = 0
        self.active = False
        self.clear()

    def clear(self):
        self.seconds = defaultdict(float)
        self.batches = 0
        self.windows = 0

    def begin(self, wait_seconds, windows):
        self.seen += 1
        self.active = self.enabled and self.seen > self.warmup
        if self.active:
            self.seconds["data_wait"] += wait_seconds
            self.batches += 1
            self.windows += windows

    def synchronize(self):
        if self.enabled and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def stage(self, name):
        if not self.enabled:
            yield
            return
        self.synchronize()
        started = time.perf_counter()
        try:
            yield
        finally:
            self.synchronize()
            if self.active:
                self.seconds[name] += time.perf_counter() - started

    def report(self):
        if not self.batches:
            return None
        total = sum(self.seconds.values())
        return {
            "timed_batches": self.batches,
            "timed_windows": self.windows,
            "warmup_batches": self.warmup,
            "stage_ms_per_batch": {
                name: self.seconds[name] * 1000 / self.batches for name in STAGES
            },
            "data_wait_percent": 100 * self.seconds["data_wait"] / max(total, 1e-12),
            "windows_per_timed_second": self.windows / max(total, 1e-12),
            "timing_mode": (
                "cuda_synchronized_wall" if self.device.type == "cuda" else "cpu_wall"
            ),
        }
