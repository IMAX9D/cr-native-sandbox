"""Confirmed public events are never fabricated from supervision placeholders."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class ConfirmedEvent:
    command_id: str
    event_tick: int
    observed_tick: int
    side: int
    kind: str = 'unknown'
    card_id: int | None = None
    native_target: tuple[int, int] | None = None

    def __post_init__(self):
        if not self.command_id or self.side not in (0, 1) or self.kind not in ('unknown', 'play', 'ability'):
            raise ValueError('invalid confirmed event identity')
        if any(type(t) is not int for t in (self.event_tick, self.observed_tick)) or not 0 <= self.event_tick <= self.observed_tick:
            raise ValueError('event cannot become known before it happened')
        if self.native_target is not None:
            x, y = self.native_target
            if not 0 <= x <= 18000 or not 0 <= y <= 32000:
                raise ValueError('event target outside arena')


class EventHistory:
    def __init__(self, episode_uid: str, capacity: int = 32, max_commands: int = 4096):
        if not episode_uid or min(capacity, max_commands) <= 0:
            raise ValueError('history needs a bounded episode identity')
        self.episode_uid = episode_uid
        self.events = deque(maxlen=capacity)
        self._seen: dict[str, ConfirmedEvent] = {}
        self.max_commands = max_commands
        self.last_observed_tick = -1

    def record(self, event: ConfirmedEvent) -> bool:
        previous = self._seen.get(event.command_id)
        if previous is not None:
            if previous != event:
                raise ValueError('conflicting duplicate command receipt')
            return False
        if len(self._seen) >= self.max_commands:
            raise OverflowError('episode command ledger capacity exceeded')
        if event.observed_tick < self.last_observed_tick:
            raise ValueError('confirmed history must arrive in observation order')
        self.last_observed_tick = event.observed_tick
        self._seen[event.command_id] = event
        self.events.append(event)
        return True

    def snapshot(self, observation_tick: int) -> tuple[ConfirmedEvent, ...]:
        return tuple(event for event in self.events if event.observed_tick <= observation_tick)


@dataclass(frozen=True)
class LabelValidity:
    action_occurred_known: bool
    candidate_known: bool = False
    target_known: bool = False
    delay_known: bool = False
    value_known: bool = False

    def __post_init__(self):
        if any(type(value) is not bool for value in (self.action_occurred_known,self.candidate_known,self.target_known,self.delay_known,self.value_known)):
            raise ValueError('label validity must use explicit booleans')
        if (self.target_known or self.delay_known) and not self.candidate_known:
            raise ValueError('conditional labels need a candidate; use a marginal/partial loss instead')


def trusted_window(decision_tick: int, first_execution_divergence: int | None,
                   *, window_ticks: int = 5, discard_tail_ticks: int = 200) -> bool:
    if decision_tick < 0 or window_ticks <= 0 or discard_tail_ticks < 0:
        raise ValueError('invalid trust window')
    return first_execution_divergence is None or decision_tick + window_ticks <= first_execution_divergence - discard_tail_ticks
