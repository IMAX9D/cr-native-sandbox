"""Bounded independent collector groups sharing one remote policy process.

Each group owns disjoint native workers and a private encoder/client. A slow
transition blocks its small group, not every worker. Completed results still
pass the original complete-episode validation and return in scheduled order.
This is not asynchronous PPO: behavior policy identities and shard closure
remain unchanged for the entire collection wave.
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from typing import Any, Callable, Sequence


class ReadyGroupCollector:
    def __init__(self, factory: Callable[[int, threading.Event], Any], *,
                 group_size: int, max_groups: int = 8):
        if group_size < 1 or max_groups < 1:
            raise ValueError("ready group limits must be positive")
        self.factory, self.group_size, self.max_groups = factory, group_size, max_groups
        self.last_profile: dict[str, float] = {}

    def collect_batch(self, specs: Sequence[Any], *, value_fn=None,
                      rolling_replacement: bool = False):
        rows = list(specs)
        if not rows:
            self.last_profile = {}
            return []
        by_worker: OrderedDict[Any, list[Any]] = OrderedDict()
        env_owners: dict[int, Any] = {}
        expected = []
        for spec in rows:
            identity = spec.header.episode_id
            expected.append(identity)
            previous = env_owners.setdefault(id(spec.env), spec.worker_id)
            if previous != spec.worker_id:
                raise ValueError("one native environment has multiple worker identities")
            queue = by_worker.setdefault(spec.worker_id, [])
            if queue and (queue[0].env is not spec.env or not rolling_replacement):
                raise ValueError("duplicate worker or changed native environment")
            queue.append(spec)
        if len(set(expected)) != len(expected):
            raise ValueError("duplicate scheduled episode identity")
        workers = list(by_worker.values())
        groups = [sum(workers[i:i+self.group_size], [])
                  for i in range(0, len(workers), self.group_size)]
        if len(groups) > self.max_groups:
            raise ValueError("ready groups exceed configured concurrency limit")
        cancelled = threading.Event()
        started = time.perf_counter()

        def run(index, group):
            collector = self.factory(index, cancelled)
            try:
                result = collector.collect_batch(group, value_fn=value_fn,
                                                  rolling_replacement=rolling_replacement)
                return result, dict(collector.last_profile)
            finally:
                # Factory must provide a group-private remote client, never the
                # shared CUDA service or another group's client.
                close = getattr(collector.policy_service, "close", None)
                if callable(close):
                    close()

        profiles, results = [], {}
        with ThreadPoolExecutor(max_workers=len(groups), thread_name_prefix="ready-group") as pool:
            futures = [pool.submit(run, i, group) for i, group in enumerate(groups)]
            try:
                for future in as_completed(futures):
                    completed, profile = future.result()
                    profiles.append(profile)
                    for episode in completed:
                        if episode.episode_id in results:
                            raise ValueError("ready groups returned duplicate episodes")
                        results[episode.episode_id] = episode
            except BaseException:
                cancelled.set()
                for future in futures:
                    future.cancel()
                raise
        if set(results) != set(expected):
            raise ValueError("ready groups returned incomplete or unexpected episodes")
        self.last_profile = {
            key: sum(float(profile.get(key, 0)) for profile in profiles)
            for key in {key for profile in profiles for key in profile}
        }
        self.last_profile.update(
            sum_group_seconds=self.last_profile.get("total_seconds", 0.0),
            total_seconds=time.perf_counter() - started,
            ready_group_count=float(len(groups)),
            ready_group_size=float(self.group_size),
        )
        return [results[identity] for identity in expected]
