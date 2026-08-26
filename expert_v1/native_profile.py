"""Versioned production semantics for RoyaleAPI native teacher forcing.

Source marker labels are immutable observations.  The profile controls only
the native command-consumption boundary used when replaying those labels.
"""

from __future__ import annotations

from typing import Any


ROYALEAPI_NATIVE_TEACHER_FORCED_PROFILE_NAME = (
    "royaleapi_native_teacher_forced"
)
ROYALEAPI_NATIVE_TEACHER_FORCED_PROFILE_VERSION = 1
ROYALEAPI_NATIVE_TEACHER_FORCED_ACTION_EXECUTION_TICK_OFFSET = 1


def validate_action_execution_tick_offset(offset: int) -> int:
    """Accept the production boundary or the explicit historical diagnostic."""
    value = int(offset)
    if value not in (0, 1):
        raise ValueError("action execution Tick offset must be exactly 0 or 1")
    return value


def action_tick_provenance(offset: int) -> str:
    value = validate_action_execution_tick_offset(offset)
    return (
        "source marker is immutable RoyaleAPI time_raw T; native execution "
        f"boundary is source_tick+{value}; source label unchanged"
    )


def native_teacher_forced_profile(offset: int) -> dict[str, Any]:
    """Return self-describing metadata for summaries and Tick Stores."""
    effective = validate_action_execution_tick_offset(offset)
    return {
        "name": ROYALEAPI_NATIVE_TEACHER_FORCED_PROFILE_NAME,
        "version": ROYALEAPI_NATIVE_TEACHER_FORCED_PROFILE_VERSION,
        "default_action_execution_tick_offset": (
            ROYALEAPI_NATIVE_TEACHER_FORCED_ACTION_EXECUTION_TICK_OFFSET
        ),
        "effective_action_execution_tick_offset": effective,
        "source_marker_tick_immutable": True,
        "source_marker": "royaleapi_time_raw_T",
        "native_execution_boundary": f"source_tick+{effective}",
        "diagnostic_override": (
            effective
            != ROYALEAPI_NATIVE_TEACHER_FORCED_ACTION_EXECUTION_TICK_OFFSET
        ),
    }
