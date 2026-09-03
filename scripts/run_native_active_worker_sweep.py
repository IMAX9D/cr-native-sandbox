"""Measure real native Tick throughput across an explicit Worker port set.

This is deliberately a runtime benchmark, not a training entry point.  Every
counted Tick is advanced by ``joint_training_transition_v1`` and validated
against the state returned by libg.  It never starts, stops, or replaces a
Worker process.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from native_core.client import JsonLineClient
from native_core.env import NativeRoyaleEnv
from training.schema import ActionMaskCache, build_action_masks


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_integer_set(value: str, *, label: str) -> list[int]:
    """Parse ``1,3-5`` while preserving order and rejecting duplicates."""
    result: list[int] = []
    seen: set[int] = set()
    for token in value.replace("\n", ",").split(","):
        token = token.strip()
        if not token or token.startswith("#"):
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            start, stop = int(left), int(right)
            if stop < start:
                raise ValueError(f"descending {label} range: {token}")
            values: Iterable[int] = range(start, stop + 1)
        else:
            values = (int(token),)
        for item in values:
            if item <= 0:
                raise ValueError(f"{label} values must be positive")
            if item in seen:
                raise ValueError(f"duplicate {label}: {item}")
            seen.add(item)
            result.append(item)
    if not result:
        raise ValueError(f"at least one {label} is required")
    return result


def load_ports(direct: str | None, source: Path | None) -> list[int]:
    if bool(direct) == bool(source):
        raise ValueError("specify exactly one of --ports or --ports-file")
    if direct:
        ports = parse_integer_set(direct, label="port")
    else:
        assert source is not None
        raw = source.read_text(encoding="utf-8-sig")
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, list):
            ports = parse_integer_set(
                ",".join(str(item) for item in decoded), label="port"
            )
        elif isinstance(decoded, Mapping) and isinstance(decoded.get("ports"), list):
            ports = parse_integer_set(
                ",".join(str(item) for item in decoded["ports"]), label="port"
            )
        else:
            uncommented = ",".join(
                line.split("#", 1)[0] for line in raw.splitlines()
            )
            ports = parse_integer_set(uncommented, label="port")
    if any(port > 65535 for port in ports):
        raise ValueError("ports must be in 1..65535")
    return ports


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _read_number(path: Path) -> int | None:
    try:
        raw = path.read_text(encoding="ascii").strip()
        return None if raw == "max" else int(raw)
    except (OSError, ValueError):
        return None


def _read_key_values(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        for line in path.read_text(encoding="ascii").splitlines():
            key, value = line.split(None, 1)
            result[key] = int(value)
    except (OSError, ValueError):
        return {}
    return result


def cgroup_root() -> Path | None:
    root = Path("/sys/fs/cgroup")
    if not (root / "cgroup.controllers").is_file():
        return None
    try:
        unified = next(
            line.split("::", 1)[1]
            for line in Path("/proc/self/cgroup").read_text(
                encoding="ascii"
            ).splitlines()
            if line.startswith("0::")
        )
        current = root / unified.lstrip("/")
        if (current / "cgroup.procs").is_file():
            return current
    except (OSError, StopIteration):
        pass
    if (root / "cgroup.procs").is_file():
        return root
    return None


def discover_worker_pids(ports: Sequence[int]) -> list[int]:
    try:
        import psutil  # type: ignore[import-not-found]

        wanted = set(ports)
        return sorted({
            int(item.pid)
            for item in psutil.net_connections(kind="tcp")
            if item.pid is not None
            and item.status == psutil.CONN_LISTEN
            and item.laddr
            and int(item.laddr.port) in wanted
        })
    except (ImportError, OSError, PermissionError):
        return []


class ResourceSampler:
    """Low-rate cgroup, host, Worker-process and GPU telemetry sampler."""

    def __init__(
        self, *, ports: Sequence[int], interval: float, worker_pids: Sequence[int]
    ) -> None:
        self.interval = interval
        self.root = cgroup_root()
        self.worker_pids = list(worker_pids) or discover_worker_pids(ports)
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._previous_cpu: tuple[float, int] | None = None
        self.memory_events_before = self._memory_events()

    def _memory_events(self) -> dict[str, int]:
        return _read_key_values(self.root / "memory.events") if self.root else {}

    def _sample_processes(self) -> dict[str, Any]:
        try:
            import psutil  # type: ignore[import-not-found]

            rss = threads = alive = 0
            cpu = 0.0
            for pid in self.worker_pids:
                try:
                    process = psutil.Process(pid)
                    rss += int(process.memory_info().rss)
                    threads += int(process.num_threads())
                    cpu += float(process.cpu_percent(interval=None))
                    alive += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return {
                "worker_processes_alive": alive,
                "worker_rss_bytes": rss,
                "worker_threads": threads,
                "worker_cpu_percent": cpu,
            }
        except ImportError:
            return {}

    @staticmethod
    def _sample_gpu() -> dict[str, Any]:
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,"
                    "power.draw,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
                check=True,
            )
            rows = []
            for line in completed.stdout.splitlines():
                values = [float(item.strip()) for item in line.split(",")]
                if len(values) == 5:
                    rows.append(values)
            if not rows:
                return {}
            return {
                "gpu_utilization_percent": max(row[0] for row in rows),
                "gpu_memory_used_mib": sum(row[1] for row in rows),
                "gpu_memory_total_mib": sum(row[2] for row in rows),
                "gpu_power_watts": sum(row[3] for row in rows),
                "gpu_temperature_c": max(row[4] for row in rows),
            }
        except (OSError, subprocess.SubprocessError, ValueError):
            return {}

    def sample(self) -> dict[str, Any]:
        now = time.time()
        value: dict[str, Any] = {"timestamp": now}
        if self.root:
            memory_current = _read_number(self.root / "memory.current")
            memory_max = _read_number(self.root / "memory.max")
            pids_current = _read_number(self.root / "pids.current")
            pids_max = _read_number(self.root / "pids.max")
            cpu_stat = _read_key_values(self.root / "cpu.stat")
            cpu_capacity: float | None = None
            try:
                quota_raw, period_raw = (self.root / "cpu.max").read_text(
                    encoding="ascii"
                ).split()
                if quota_raw != "max":
                    cpu_capacity = int(quota_raw) / int(period_raw)
                    value["cgroup_cpu_capacity_cores"] = cpu_capacity
            except (OSError, ValueError, ZeroDivisionError):
                pass
            if memory_current is not None:
                value["cgroup_memory_current_bytes"] = memory_current
            if memory_max is not None and memory_current is not None:
                value["cgroup_memory_max_bytes"] = memory_max
                value["cgroup_memory_fraction"] = memory_current / memory_max
            if pids_current is not None:
                value["cgroup_pids_current"] = pids_current
            if pids_max is not None and pids_current is not None:
                value["cgroup_pids_max"] = pids_max
                value["cgroup_pids_fraction"] = pids_current / pids_max
            usage = cpu_stat.get("usage_usec")
            if usage is not None and self._previous_cpu is not None:
                then, previous = self._previous_cpu
                elapsed = max(1e-9, now - then)
                value["cgroup_cpu_cores"] = (
                    (usage - previous) / 1_000_000.0 / elapsed
                )
                if cpu_capacity:
                    value["cgroup_cpu_utilization_percent"] = (
                        value["cgroup_cpu_cores"] / cpu_capacity * 100.0
                    )
            if usage is not None:
                self._previous_cpu = (now, usage)
            for key, count in self._memory_events().items():
                value[f"memory_event_{key}"] = count
        try:
            load1, load5, load15 = os.getloadavg()
            value.update({"load1": load1, "load5": load5, "load15": load15})
        except OSError:
            pass
        value.update(self._sample_processes())
        value.update(self._sample_gpu())
        self.samples.append(value)
        return value

    def start(self) -> None:
        self.sample()

        def run() -> None:
            while not self._stop.wait(self.interval):
                self.sample()

        self._thread = threading.Thread(target=run, name="resource-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval + 1.0))
        self.sample()

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "sample_count": len(self.samples),
            "worker_pids": self.worker_pids,
            "memory_events_delta": {
                key: value - self.memory_events_before.get(key, 0)
                for key, value in self._memory_events().items()
            },
        }
        keys = {key for row in self.samples for key in row if key != "timestamp"}
        for key in sorted(keys):
            values = [float(row[key]) for row in self.samples if key in row]
            if values:
                result[f"{key}_mean"] = statistics.fmean(values)
                result[f"{key}_peak"] = max(values)
        return result


@dataclass
class WorkerSlot:
    env: NativeRoyaleEnv
    replay: dict[str, Any]
    seed: int
    state: dict[str, Any]
    masks: dict[tuple[int, int], list[str]] = field(default_factory=dict)
    mask_cache: ActionMaskCache = field(default_factory=ActionMaskCache)
    transitions: int = 0
    advanced_ticks: int = 0
    terminals: int = 0
    resets: int = 0
    actions_attempted: int = 0
    actions_accepted: int = 0
    actions_rejected: int = 0
    unexpected_rejections: int = 0
    result_codes: dict[int, int] = field(default_factory=dict)
    native_timing_ns: dict[str, int] = field(default_factory=dict)

    def seeded_replay(self) -> dict[str, Any]:
        value = json.loads(json.dumps(self.replay))
        value["rndSeed"] = self.seed
        return value

    def restart(self) -> None:
        self.seed += 1
        self.state = self.env.restart(self.seeded_replay(), warmup_steps=100)
        self.resets += 1


def deterministic_legal_actions(slot: WorkerSlot) -> list[dict[str, int]]:
    actions: list[dict[str, int]] = []
    for player in slot.state.get("players", []):
        side = int(player["side"])
        for raw_index in player.get("hand_deck_indices", [])[:4]:
            deck_index = int(raw_index)
            if deck_index < 0 or (side, deck_index) in slot.masks:
                continue
            response = slot.env.probe_grid(side=side, deck_index=deck_index)
            slot.masks[(side, deck_index)] = [str(row) for row in response["rows"]]
        card_mask, position_masks, hand = build_action_masks(
            slot.state,
            side=side,
            native_masks=slot.masks,
            decks=slot.env.decks,
            cache=slot.mask_cache,
        )
        legal_cards = [index for index in range(1, 5) if bool(card_mask[index])]
        if not legal_cards:
            continue
        card_choice = legal_cards[0]
        hand_index = card_choice - 1
        positions = position_masks[hand_index].nonzero()[0]
        if not len(positions):
            raise RuntimeError("legal card has an empty position mask")
        row, column = divmod(int(positions[0]), 18)
        actions.append({
            "type": "play",
            "side": side,
            "deck_index": int(hand[hand_index]),
            "x": column * 1000 + 500,
            "y": row * 1000 + 500,
        })
    return actions


def validate_transition(
    slot: WorkerSlot,
    transition: Mapping[str, Any],
    *,
    requested_steps: int,
) -> tuple[int, bool]:
    episode = transition.get("step", {}).get("episode")
    if not isinstance(episode, Mapping):
        raise RuntimeError("transition is missing terminal metadata")
    done = bool(episode.get("terminated") or episode.get("truncated"))
    previous_tick = int(slot.state["tick"])
    if done:
        final_tick = episode.get("terminal_tick")
        if not isinstance(final_tick, int) or isinstance(final_tick, bool):
            raise RuntimeError("terminal transition has no integer terminal_tick")
        if episode.get("outcome") in (None, "ongoing"):
            raise RuntimeError("terminal transition has an ongoing outcome")
        if not isinstance(episode.get("crowns"), list):
            raise RuntimeError("terminal transition has no crown vector")
        if not isinstance(episode.get("rewards"), list):
            raise RuntimeError("terminal transition has no reward vector")
        current_tick = final_tick
    else:
        state = transition.get("state")
        if not isinstance(state, Mapping):
            raise RuntimeError("nonterminal transition is missing next state")
        current_tick = int(state["tick"])
    advanced = current_tick - previous_tick
    if advanced < 0:
        raise RuntimeError(
            f"native Tick regression: {previous_tick}->{current_tick}"
        )
    # A terminal can be reported one RPC after the last observable battle
    # frame.  In that case terminal_tick legitimately equals the slot's last
    # observed tick: no Tick is counted, and the caller immediately recycles
    # the episode.  A zero-delta nonterminal transition is still a real stall.
    if advanced == 0 and not done:
        raise RuntimeError(
            f"native Tick freeze: {previous_tick}->{current_tick}"
        )
    if advanced > requested_steps:
        raise RuntimeError(
            f"native Tick over-advance: requested {requested_steps}, got {advanced}"
        )
    return advanced, done


def one_transition(
    slot: WorkerSlot, *, mode: str, decision_ticks: int
) -> tuple[float, bool]:
    actions = deterministic_legal_actions(slot) if mode == "deterministic-legal" else []
    started = time.perf_counter()
    transition = slot.env.joint_training_transition(actions, steps=decision_ticks)
    latency = time.perf_counter() - started
    advanced, done = validate_transition(
        slot, transition, requested_steps=decision_ticks
    )
    native_actions = transition["joint_action"].get("actions", [])
    timing = transition.get("timing_v1")
    if isinstance(timing, Mapping):
        for key, raw in timing.items():
            if isinstance(raw, int) and not isinstance(raw, bool):
                slot.native_timing_ns[str(key)] = (
                    slot.native_timing_ns.get(str(key), 0) + raw
                )
    slot.actions_attempted += len(native_actions)
    for item in native_actions:
        result = item.get("result", {})
        accepted = bool(result.get("accepted", False))
        code = int(result.get("result_code", -1))
        slot.result_codes[code] = slot.result_codes.get(code, 0) + 1
        if accepted:
            slot.actions_accepted += 1
        else:
            slot.actions_rejected += 1
            if code not in (3, 4):
                slot.unexpected_rejections += 1
    slot.transitions += 1
    slot.advanced_ticks += advanced
    if done:
        slot.terminals += 1
        slot.restart()
    else:
        slot.state = dict(transition["state"])
    return latency, done


def aggregate_rpc(slots: Sequence[WorkerSlot]) -> dict[str, float]:
    total: list[float] = []
    receive: list[float] = []
    attempts = failures = 0.0
    result: dict[str, float] = {}
    for slot in slots:
        samples = slot.env.rpc_latency_samples()
        total.extend(samples.get("total", []))
        receive.extend(samples.get("receive", []))
        attempts += slot.env.rpc_profile.get("rpc_attempts", 0.0)
        failures += slot.env.rpc_profile.get("rpc_failures", 0.0)
        for key, value in slot.env.rpc_profile.items():
            result[key] = result.get(key, 0.0) + float(value)
    result.update(JsonLineClient.latency_summary(
        total, receive, attempts=attempts, failures=failures
    ))
    return result


def safety_reasons(
    *,
    slots: Sequence[WorkerSlot],
    resources: ResourceSampler,
    memory_fraction: float,
    pids_fraction: float,
    max_unexpected_rejections: int,
    max_rpc_p99_ms: float,
) -> list[str]:
    reasons: list[str] = []
    latest = resources.samples[-1] if resources.samples else resources.sample()
    if latest.get("cgroup_memory_fraction", 0.0) >= memory_fraction:
        reasons.append("cgroup memory safety threshold reached")
    if latest.get("cgroup_pids_fraction", 0.0) >= pids_fraction:
        reasons.append("cgroup PID safety threshold reached")
    events = resources._memory_events()
    for key in ("oom", "oom_kill"):
        if events.get(key, 0) > resources.memory_events_before.get(key, 0):
            reasons.append(f"cgroup memory.events {key} increased")
    unexpected = sum(slot.unexpected_rejections for slot in slots)
    if unexpected > max_unexpected_rejections:
        reasons.append("unexpected native action rejection threshold exceeded")
    if max_rpc_p99_ms > 0:
        p99 = aggregate_rpc(slots).get("rpc_latency_p99_ms", 0.0)
        if p99 > max_rpc_p99_ms:
            reasons.append("RPC p99 latency threshold exceeded")
    return reasons


def make_slots(
    *,
    ports: Sequence[int],
    host: str,
    timeout: float,
    replay: Mapping[str, Any],
    seed_base: int,
) -> list[WorkerSlot]:
    envs = [
        NativeRoyaleEnv(
            host=host, port=port, timeout=timeout, profile_native=True
        )
        for port in ports
    ]
    try:
        with ThreadPoolExecutor(max_workers=len(envs)) as executor:
            futures = []
            replays = []
            for index, env in enumerate(envs):
                value = json.loads(json.dumps(replay))
                value["rndSeed"] = seed_base + index
                replays.append(value)
                futures.append(executor.submit(env.reset, value, warmup_steps=100))
            states = [future.result() for future in futures]
        return [
            WorkerSlot(
                env=env,
                replay=dict(replay),
                seed=seed_base + index,
                state=state,
            )
            for index, (env, state) in enumerate(zip(envs, states, strict=True))
        ]
    except BaseException:
        for env in envs:
            env.close()
        raise


def close_slots(slots: Sequence[WorkerSlot]) -> None:
    for slot in slots:
        slot.env.close()


def run_phase(
    slots: Sequence[WorkerSlot],
    *,
    seconds: float,
    mode: str,
    decision_ticks: int,
    output_interval: float,
    progress: Callable[[float], None] | None = None,
    gate: Callable[[], Sequence[str]] | None = None,
) -> tuple[list[float], list[str]]:
    latencies: list[float] = []
    started = time.perf_counter()
    deadline = started + seconds
    next_output = started + output_interval
    next_gate = started + min(1.0, max(0.1, output_interval))
    reasons: list[str] = []
    with ThreadPoolExecutor(max_workers=len(slots)) as executor:
        # Keep exactly one request in flight per Worker.  A round barrier makes
        # every fast Worker wait for the slowest RPC and badly understates the
        # useful capacity at larger tiers.
        pending: dict[Future[tuple[float, bool]], WorkerSlot] = {
            executor.submit(
                one_transition,
                slot,
                mode=mode,
                decision_ticks=decision_ticks,
            ): slot
            for slot in slots
        }
        while pending:
            now = time.perf_counter()
            completed, _ = wait(
                pending,
                timeout=max(0.0, min(0.25, deadline - now)),
                return_when=FIRST_COMPLETED,
            )
            for future in completed:
                slot = pending.pop(future)
                if future.cancelled():
                    continue
                latency, _done = future.result()
                latencies.append(latency)
                if not reasons and time.perf_counter() < deadline:
                    pending[executor.submit(
                        one_transition,
                        slot,
                        mode=mode,
                        decision_ticks=decision_ticks,
                    )] = slot
            now = time.perf_counter()
            if gate is not None and now >= next_gate:
                reasons = list(gate())
                next_gate = now + 1.0
            if progress is not None and now >= next_output:
                progress(now - started)
                next_output = now + output_interval
            if reasons or now >= deadline:
                # Do not enqueue more work.  At most one bounded RPC per Worker
                # remains, and the executor drains those before returning.
                for future in pending:
                    future.cancel()
    return latencies, reasons


def summarize_tier(
    slots: Sequence[WorkerSlot],
    latencies: Sequence[float],
    *,
    elapsed: float,
    resources: ResourceSampler,
    reasons: Sequence[str],
) -> dict[str, Any]:
    ticks = sum(slot.advanced_ticks for slot in slots)
    worker_tick_rates = [
        slot.advanced_ticks / elapsed if elapsed else 0.0
        for slot in slots
    ]
    actions_attempted = sum(slot.actions_attempted for slot in slots)
    actions_accepted = sum(slot.actions_accepted for slot in slots)
    result_codes: dict[str, int] = {}
    native_timing_ns: dict[str, int] = {}
    for slot in slots:
        for code, count in slot.result_codes.items():
            result_codes[str(code)] = result_codes.get(str(code), 0) + count
        for key, value in slot.native_timing_ns.items():
            native_timing_ns[key] = native_timing_ns.get(key, 0) + value
    return {
        "status": "gated" if reasons else "complete",
        "safety_reasons": list(reasons),
        "workers": len(slots),
        "wall_seconds": elapsed,
        "transitions": sum(slot.transitions for slot in slots),
        "advanced_native_ticks": ticks,
        "native_ticks_per_second": ticks / elapsed if elapsed else 0.0,
        "worker_ticks_per_second_mean": (
            statistics.fmean(worker_tick_rates) if worker_tick_rates else 0.0
        ),
        "worker_ticks_per_second_p05": percentile(worker_tick_rates, 0.05),
        "worker_ticks_per_second_p50": percentile(worker_tick_rates, 0.50),
        "worker_ticks_per_second_p95": percentile(worker_tick_rates, 0.95),
        "completed_terminals": sum(slot.terminals for slot in slots),
        "successful_resets": sum(slot.resets for slot in slots),
        "actions_attempted": actions_attempted,
        "actions_accepted": actions_accepted,
        "actions_rejected": sum(slot.actions_rejected for slot in slots),
        "unexpected_rejections": sum(
            slot.unexpected_rejections for slot in slots
        ),
        "action_acceptance_rate": (
            actions_accepted / actions_attempted if actions_attempted else 0.0
        ),
        "native_result_codes": result_codes,
        "native_timing_seconds": {
            key: value / 1_000_000_000.0
            for key, value in sorted(native_timing_ns.items())
        },
        "transition_latency_p50_ms": percentile(latencies, 0.50) * 1000.0,
        "transition_latency_p95_ms": percentile(latencies, 0.95) * 1000.0,
        "transition_latency_p99_ms": percentile(latencies, 0.99) * 1000.0,
        "transition_latency_max_ms": max(latencies, default=0.0) * 1000.0,
        "rpc": aggregate_rpc(slots),
        "resources": resources.summary(),
    }


def recommended_tier(rows: Sequence[Mapping[str, Any]]) -> int | None:
    complete = [row for row in rows if row.get("status") == "complete"]
    if not complete:
        return None
    best = int(complete[0]["workers"])
    previous = float(complete[0]["native_ticks_per_second"])
    for row in complete[1:]:
        current = float(row["native_ticks_per_second"])
        gain = (current / previous - 1.0) if previous > 0 else math.inf
        if gain < 0.05:
            break
        best = int(row["workers"])
        previous = current
    return best


def workload_name(action_mode: str) -> str:
    names = {
        "deterministic-legal": "deterministic_legal_native_actions_v1",
        "wait": "wait_native_tick_v1",
    }
    try:
        return names[action_mode]
    except KeyError as error:
        raise ValueError(f"unknown action mode: {action_mode}") from error


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Active real-libg Worker scaling sweep (no process management)."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ports", help="comma/range list, e.g. 38031-38062")
    source.add_argument("--ports-file", type=Path, help="JSON list/object or text")
    parser.add_argument("--tiers", default="32,48,64,96,128")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--replay", type=Path, default=PROJECT_ROOT / "examples" / "eight-card-bootstrap.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-seconds", type=float, default=20.0)
    parser.add_argument("--measure-seconds", type=float, default=180.0)
    parser.add_argument("--decision-ticks", type=int, default=1)
    parser.add_argument("--action-mode", choices=("deterministic-legal", "wait"), default="deterministic-legal")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=120_000)
    parser.add_argument("--resource-interval", type=float, default=2.0)
    parser.add_argument("--output-interval", type=float, default=10.0)
    parser.add_argument("--worker-pids", default="")
    parser.add_argument("--max-memory-fraction", type=float, default=0.80)
    parser.add_argument("--max-pids-fraction", type=float, default=0.80)
    parser.add_argument("--max-unexpected-rejections", type=int, default=0)
    parser.add_argument("--max-rpc-p99-ms", type=float, default=0.0)
    args = parser.parse_args()

    ports = load_ports(args.ports, args.ports_file)
    tiers = parse_integer_set(args.tiers, label="tier")
    if tiers != sorted(tiers):
        raise ValueError("tiers must be strictly increasing")
    if tiers[-1] > len(ports):
        raise ValueError(
            f"largest tier needs {tiers[-1]} ports, only {len(ports)} supplied"
        )
    if args.warmup_seconds < 0 or args.measure_seconds <= 0:
        raise ValueError("warmup must be nonnegative and measure must be positive")
    if args.decision_ticks <= 0:
        raise ValueError("decision-ticks must be positive")
    if not 0 < args.max_memory_fraction <= 1 or not 0 < args.max_pids_fraction <= 1:
        raise ValueError("resource safety fractions must be in (0, 1]")
    worker_pids = (
        parse_integer_set(args.worker_pids, label="worker PID")
        if args.worker_pids else []
    )
    replay = json.loads(args.replay.read_text(encoding="utf-8-sig"))
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "native_active_worker_sweep_v1",
        "workload": workload_name(args.action_mode),
        "status": "running",
        "started_utc": utc_now(),
        "updated_utc": utc_now(),
        "config": {
            "host": args.host,
            "ports": ports,
            "tiers": tiers,
            "warmup_seconds": args.warmup_seconds,
            "measure_seconds": args.measure_seconds,
            "decision_ticks": args.decision_ticks,
            "action_mode": args.action_mode,
            "profile_native": True,
            "replay": str(args.replay.resolve()),
        },
        "tiers": [],
    }
    atomic_json(args.output, report)
    try:
        for tier_index, tier in enumerate(tiers):
            slots = make_slots(
                ports=ports[:tier], host=args.host, timeout=args.timeout,
                replay=replay, seed_base=args.seed + tier_index * 1_000_000,
            )
            try:
                if args.warmup_seconds:
                    run_phase(
                        slots, seconds=args.warmup_seconds, mode=args.action_mode,
                        decision_ticks=args.decision_ticks,
                        output_interval=args.output_interval,
                    )
                # Measurement begins at a clean episode and with clean counters.
                with ThreadPoolExecutor(max_workers=tier) as executor:
                    futures = [executor.submit(slot.restart) for slot in slots]
                    for future in futures:
                        future.result()
                for slot in slots:
                    slot.transitions = slot.advanced_ticks = slot.terminals = 0
                    slot.resets = slot.actions_attempted = slot.actions_accepted = 0
                    slot.actions_rejected = slot.unexpected_rejections = 0
                    slot.result_codes.clear()
                    slot.native_timing_ns.clear()
                    slot.env.reset_rpc_profile()
                resources = ResourceSampler(
                    ports=ports[:tier], interval=args.resource_interval,
                    worker_pids=worker_pids,
                )
                resources.start()
                measured_at = time.perf_counter()

                def gate() -> list[str]:
                    return safety_reasons(
                        slots=slots, resources=resources,
                        memory_fraction=args.max_memory_fraction,
                        pids_fraction=args.max_pids_fraction,
                        max_unexpected_rejections=args.max_unexpected_rejections,
                        max_rpc_p99_ms=args.max_rpc_p99_ms,
                    )

                def save_progress(elapsed: float) -> None:
                    report["active_tier"] = {
                        "workers": tier,
                        "elapsed_seconds": elapsed,
                        "advanced_native_ticks": sum(
                            slot.advanced_ticks for slot in slots
                        ),
                    }
                    report["updated_utc"] = utc_now()
                    atomic_json(args.output, report)

                latencies: list[float] = []
                reasons: list[str] = []
                try:
                    latencies, reasons = run_phase(
                        slots, seconds=args.measure_seconds,
                        mode=args.action_mode,
                        decision_ticks=args.decision_ticks,
                        output_interval=args.output_interval,
                        progress=save_progress, gate=gate,
                    )
                except BaseException as error:
                    reasons = [f"{type(error).__name__}: {error}"]
                    elapsed = time.perf_counter() - measured_at
                    resources.stop()
                    row = summarize_tier(
                        slots, latencies, elapsed=elapsed,
                        resources=resources, reasons=reasons,
                    )
                    row["status"] = "failed"
                    report["tiers"].append(row)
                    report.pop("active_tier", None)
                    report["updated_utc"] = utc_now()
                    atomic_json(args.output, report)
                    raise
                else:
                    elapsed = time.perf_counter() - measured_at
                    resources.stop()
                    row = summarize_tier(
                        slots, latencies, elapsed=elapsed,
                        resources=resources, reasons=reasons,
                    )
                report["tiers"].append(row)
                report.pop("active_tier", None)
                report["recommended_workers"] = recommended_tier(report["tiers"])
                report["updated_utc"] = utc_now()
                atomic_json(args.output, report)
                if reasons:
                    break
            finally:
                close_slots(slots)
        report["status"] = (
            "gated" if report["tiers"] and report["tiers"][-1]["status"] == "gated"
            else "complete"
        )
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        report["error"] = "KeyboardInterrupt"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report.pop("active_tier", None)
        report["finished_utc"] = utc_now()
        report["updated_utc"] = utc_now()
        atomic_json(args.output, report)


if __name__ == "__main__":
    main()
