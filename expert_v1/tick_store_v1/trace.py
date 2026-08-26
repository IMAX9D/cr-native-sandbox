"""Bridge compact native Tick traces into the binary Tick store."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .schema import TickState, TickStoreContractError, normalize_native_state


TRACE_KIND = "libg_native_train_tick_trace_v1"
TRACE_ENCODING = "compact-train-v1"


@dataclass(slots=True)
class TickTraceAccumulator:
    """Collect consecutive native states across ≤64-Tick RPC batches.

    A command may mutate hand/elixir at the same boundary Tick between two
    batches.  Consequently, a following batch's ``initial_frame`` must have
    the same Tick as the last stored pre-action state, but it is deliberately
    not required to be byte-identical and is not stored twice.  Every advanced
    complete frame must then increment exactly one native Tick.
    """

    states: list[TickState] = field(default_factory=list)
    batches: int = 0
    complete_frames: int = 0
    incomplete_terminal_frames: int = 0
    terminal_episode: dict[str, Any] | None = None

    def start(self, state: Mapping[str, Any]) -> TickState:
        """Record the pre-action state at the episode's first boundary."""
        normalized = normalize_native_state(state)
        if self.states:
            raise TickStoreContractError("compact Tick accumulator already started")
        self.states.append(normalized)
        self.complete_frames += 1
        return normalized

    def extend(self, trace: Mapping[str, Any]) -> int:
        if (
            trace.get("schema_version") != 1
            or trace.get("trace_schema_version") != 1
            or trace.get("kind") != TRACE_KIND
            or trace.get("encoding") != TRACE_ENCODING
            or trace.get("fixed_dt") != 0.05
        ):
            raise TickStoreContractError("unsupported compact Tick trace")
        frames = trace.get("frames")
        initial = trace.get("initial_frame")
        stepped = trace.get("stepped")
        if (
            not isinstance(initial, Mapping)
            or not isinstance(frames, list)
            or not isinstance(stepped, int)
            or isinstance(stepped, bool)
            or len(frames) != stepped
            or trace.get("final_frame_index") != stepped
        ):
            raise TickStoreContractError("compact Tick trace frame count mismatch")
        if initial.get("observation_complete") is not True:
            raise TickStoreContractError("compact Tick trace initial state is incomplete")
        initial_state = normalize_native_state(initial["state"])
        if not self.states:
            self.start(initial["state"])
        elif initial_state.tick != self.states[-1].tick:
            raise TickStoreContractError(
                "compact Tick trace boundary mismatch: "
                f"{self.states[-1].tick}->{initial_state.tick}"
            )

        appended = 0
        for index, raw_frame in enumerate(frames, start=1):
            if (
                not isinstance(raw_frame, Mapping)
                or raw_frame.get("frame_index") != index
                or raw_frame.get("advanced_steps") != index
                or not isinstance(raw_frame.get("state"), Mapping)
                or "step" in raw_frame
            ):
                raise TickStoreContractError("compact Tick trace frame contract mismatch")
            if raw_frame.get("observation_complete") is not True:
                if index != stepped or trace.get("terminal") is not True:
                    raise TickStoreContractError(
                        "only the terminal compact Tick frame may be incomplete"
                    )
                episode = raw_frame["state"].get("episode")
                if not isinstance(episode, Mapping) or not episode.get(
                    "terminated", False
                ):
                    raise TickStoreContractError(
                        "incomplete compact terminal frame lacks terminal episode"
                    )
                self.terminal_episode = dict(episode)
                self.incomplete_terminal_frames += 1
                continue
            state = normalize_native_state(raw_frame["state"])
            expected_tick = self.states[-1].tick + 1
            if state.tick != expected_tick:
                raise TickStoreContractError(
                    f"compact Tick trace is not 20 Hz consecutive: "
                    f"expected {expected_tick}, got {state.tick}"
                )
            self.states.append(state)
            self.complete_frames += 1
            appended += 1
            episode = raw_frame["state"].get("episode")
            if isinstance(episode, Mapping) and episode.get("terminated", False):
                self.terminal_episode = dict(episode)

        self.batches += 1
        final_state = frames[-1]["state"] if frames else initial["state"]
        if trace.get("final_tick") != int(final_state["tick"]):
            raise TickStoreContractError("compact Tick trace final tick mismatch")
        return appended

    @property
    def tick_start(self) -> int | None:
        return None if not self.states else self.states[0].tick

    @property
    def tick_stop(self) -> int | None:
        return None if not self.states else self.states[-1].tick

    def metadata(self) -> dict[str, Any]:
        return {
            "trace_kind": TRACE_KIND,
            "trace_encoding": TRACE_ENCODING,
            "native_tick_hz": 20,
            "trace_batches": self.batches,
            "complete_frames": self.complete_frames,
            "incomplete_terminal_frames": self.incomplete_terminal_frames,
            "tick_start": self.tick_start,
            "tick_stop": self.tick_stop,
        }
