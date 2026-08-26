"""Gym-style facade for the persistent, Surface-free native ``libg`` service."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

try:
    from .card_catalog import catalog as live_card_catalog, observed_card
except ImportError:  # direct ``python native_core/env.py`` consumers
    from card_catalog import (  # type: ignore[no-redef]
        catalog as live_card_catalog,
        observed_card,
    )

try:
    from .client import (
        MAX_TRACE_RESPONSE_BYTES,
        MAX_TRACE_STEPS,
        MIN_TRACE_RESPONSE_BYTES,
        TRACE_SCHEMA_VERSION,
        JsonLineClient,
    )
except ImportError:  # direct ``python native_core/env.py`` consumers
    from client import (  # type: ignore[no-redef]
        MAX_TRACE_RESPONSE_BYTES,
        MAX_TRACE_STEPS,
        MIN_TRACE_RESPONSE_BYTES,
        TRACE_SCHEMA_VERSION,
        JsonLineClient,
    )


CARD_NAMES = {
    card_id: str(value["display_name"])
    for card_id, value in live_card_catalog().items()
}

ABILITY_STATE_NAMES = {
    0: "unknown",
    1: "absent",
    2: "ready",
    3: "on_cooldown",
    4: "all_charges_consumed",
    5: "limited_availability",
    6: "disabled",
    7: "not_enough_elixir",
    8: "temporarily_unavailable",
    9: "deploying",
    10: "pending",
    11: "casting",
    12: "not_yet_available",
}


class NativeHostError(RuntimeError):
    """The native service rejected an operation."""


class NativeRoyaleEnv:
    """Small synchronous ``reset / observe / act / step`` training API."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 37031,
        timeout: float = 15.0,
        profile_native: bool = False,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.profile_native = profile_native
        self.replay: dict[str, Any] | None = None
        self.decks: list[list[dict[str, int]]] = [[], []]
        self.accounts: list[tuple[int, int]] = [(1, 1), (2, 2)]
        self.last_reset_attempts = 0
        self.last_episode: dict[str, Any] | None = None
        self.rpc_profile: dict[str, float] = {}
        self.client = JsonLineClient(
            host=host, port=port, timeout=timeout, profile=self.rpc_profile
        )

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.client.request(payload)
        if not response.get("ok"):
            raise NativeHostError(
                f"{response.get('error_type', 'native error')}: "
                f"{response.get('error', response)}"
            )
        return response

    def close(self) -> None:
        self.client.close()

    def reset_rpc_profile(self) -> None:
        self.client.reset_profile()

    def rpc_latency_samples(self) -> dict[str, list[float]]:
        return self.client.latency_samples()

    def __enter__(self) -> "NativeRoyaleEnv":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @staticmethod
    def read_replay(source: Path | Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(source, Path):
            value = json.loads(source.read_text(encoding="utf-8-sig"))
        else:
            value = deepcopy(dict(source))
        if not isinstance(value, dict) or not isinstance(value.get("battle"), dict):
            raise TypeError("native replay must be an object containing battle")
        return value

    def reset(
        self,
        replay: Path | Mapping[str, Any],
        *,
        warmup_steps: int = 100,
    ) -> dict[str, Any]:
        value = self.read_replay(replay)
        self._configure_replay(value)
        self.last_reset_attempts = 1
        response = self._request({"op": "reset", "replay": value})
        state = self._enrich_state(response["state"])
        current_tick = int(state["tick"])
        if warmup_steps > current_tick:
            self.step(warmup_steps - current_tick)
        return self.observe()

    def attach(
        self,
        replay: Path | Mapping[str, Any],
        *,
        warmup_steps: int = 100,
    ) -> dict[str, Any]:
        """Adopt a fresh worker's bootstrap battle without reloading libg."""
        value = self.read_replay(replay)
        self._configure_replay(value)
        self.last_reset_attempts = 0
        if warmup_steps:
            self.step(warmup_steps)
        return self.observe()

    def restart(
        self,
        replay: Path | Mapping[str, Any],
        *,
        warmup_steps: int = 0,
    ) -> dict[str, Any]:
        """Alias for the production in-process 4->4 battle replacement."""
        value = self.read_replay(replay)
        self._configure_replay(value)
        self.last_reset_attempts = 0
        response = self._request({"op": "reset", "replay": value})
        current_tick = int(response["state"]["tick"])
        if warmup_steps > current_tick:
            self.step(warmup_steps - current_tick)
        return self.observe()

    def _configure_replay(self, value: Mapping[str, Any]) -> None:
        battle = value["battle"]
        self.decks = []
        self.accounts = []
        for side in range(2):
            spells = battle[f"deck{side}"]["sp"]
            self.decks.append(
                [
                    {"card_id": int(item["d"]), "level": int(item["l"]) + 1}
                    | {"form_flags": int(item.get("el", 0))}
                    for item in spells
                ]
            )
            avatar = battle[f"avatar{side}"]
            self.accounts.append(
                (int(avatar["accountID.hi"]), int(avatar["accountID.lo"]))
            )
        self.replay = deepcopy(dict(value))
        self.last_episode = None

    def observe(self) -> dict[str, Any]:
        state = self._request({"op": "observe"})["state"]
        return self._enrich_state(state)

    def observe_train(self) -> dict[str, Any]:
        state = self._request({"op": "observe_train_v1"})["state"]
        return self._enrich_training_state(state)

    @staticmethod
    def _validate_training_state(state: Mapping[str, Any]) -> None:
        """Validate the raw compact observation without expanding it.

        The batched Tick-store path intentionally keeps this transport form:
        enriching every frame with names and duplicated hand objects costs
        memory and CPU but adds no native state.
        """
        if (
            state.get("schema_version") != 1
            or state.get("kind") != "libg_native_train_state_v1"
            or state.get("coherent") is not True
            or not isinstance(state.get("tick"), int)
            or isinstance(state.get("tick"), bool)
            or not isinstance(state.get("entities"), list)
            or not isinstance(state.get("players"), list)
            or len(state["players"]) != 2
            or not isinstance(state.get("episode"), Mapping)
        ):
            raise NativeHostError("compact training observation contract mismatch")

    def _enrich_training_state(self, state: Mapping[str, Any]) -> dict[str, Any]:
        self._validate_training_state(state)
        return self._enrich_state(state)

    def _enrich_state(self, state: Mapping[str, Any]) -> dict[str, Any]:
        result = deepcopy(dict(state))
        result["elapsed_seconds"] = round(int(result["tick"]) * 0.05, 3)
        if isinstance(result.get("episode"), Mapping):
            result["episode"] = self._enrich_episode(result["episode"])
            self.last_episode = deepcopy(result["episode"])
        for player in result.get("players", []):
            side = int(player["side"])
            if isinstance(player.get("elixir_raw"), int):
                player["elixir_exact"] = player["elixir_raw"] / 10_000.0
            hand: list[dict[str, Any]] = []
            for hand_index, deck_index in enumerate(player["hand_deck_indices"]):
                if deck_index < 0 or side >= len(self.decks):
                    continue
                card = self.decks[side][deck_index]
                card_id = card["card_id"]
                hand.append(
                    {
                        "hand_index": hand_index,
                        "deck_index": deck_index,
                        "card_id": card_id,
                        "level": card["level"],
                        "form_flags": int(card.get("form_flags", 0)),
                        "has_evolution": bool(int(card.get("form_flags", 0)) & 1),
                        "has_hero": bool(int(card.get("form_flags", 0)) & 2),
                        "name": CARD_NAMES.get(card_id, str(card_id)),
                    }
                )
            player["hand"] = hand
        for entity in result.get("entities", []):
            if isinstance(entity.get("category"), int):
                # libg's 5,000,000-series generation key is stable for the
                # life of an entity and is the public handle accepted by the
                # native ability command. Raw process pointers stay private.
                entity["entity_id"] = int(entity["category"])
            if isinstance(entity.get("ability_state_code"), int):
                entity["ability_state_name"] = ABILITY_STATE_NAMES.get(
                    int(entity["ability_state_code"]), "unknown_native_state"
                )
            native_card_id = int(entity.get("card_id", -1))
            if native_card_id < 0:
                continue
            identity = observed_card(native_card_id)
            entity["native_card_id"] = native_card_id
            entity.update(identity)
            entity["name"] = CARD_NAMES.get(
                int(identity["base_card_id"]), str(identity["form_name"])
            )
        return result

    @staticmethod
    def _enrich_episode(episode: Mapping[str, Any]) -> dict[str, Any]:
        result = deepcopy(dict(episode))
        rewards = result.get("rewards", [0.0, 0.0])
        crowns = result.get("crowns", [0, 0])
        if isinstance(rewards, list) and len(rewards) == 2:
            result["rewards_by_side"] = {
                0: float(rewards[0]),
                1: float(rewards[1]),
            }
        if isinstance(crowns, list) and len(crowns) == 2:
            result["crowns_by_side"] = {
                0: int(crowns[0]),
                1: int(crowns[1]),
            }
        return result

    def act(self, *, side: int, deck_index: int, x: int, y: int) -> dict[str, Any]:
        account_hi, account_lo = self.accounts[side]
        return self._request(
            {
                "op": "act",
                "action": {
                    "type": "play",
                    "side": side,
                    "deck_index": deck_index,
                    "x": x,
                    "y": y,
                    "account_hi": account_hi,
                    "account_lo": account_lo,
                    "dry_run": False,
                },
            }
        )["result"]

    def use_ability(self, *, side: int, entity_id: int) -> dict[str, Any]:
        """Press the authoritative native ability button for a live entity."""
        account_hi, account_lo = self.accounts[side]
        return self._request(
            {
                "op": "ability",
                "action": {
                    "type": "ability",
                    "side": side,
                    "entity_id": int(entity_id),
                    "account_hi": account_hi,
                    "account_lo": account_lo,
                },
            }
        )["result"]

    def _joint_payload(
        self, actions: list[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        seen: set[int] = set()
        for value in actions:
            side = int(value["side"])
            if side not in (0, 1) or side in seen:
                raise ValueError("joint actions require unique sides 0 and/or 1")
            seen.add(side)
            account_hi, account_lo = self.accounts[side]
            action_type = str(value.get("type", "play"))
            if action_type == "ability":
                payload.append({
                    "type": "ability",
                    "side": side,
                    "entity_id": int(value["entity_id"]),
                    "account_hi": account_hi,
                    "account_lo": account_lo,
                })
            elif action_type == "play":
                payload.append({
                    "type": "play",
                    "side": side,
                    "deck_index": int(value["deck_index"]),
                    "x": int(value["x"]),
                    "y": int(value["y"]),
                    "account_hi": account_hi,
                    "account_lo": account_lo,
                    "dry_run": False,
                })
            else:
                raise ValueError(f"unknown native action type: {action_type}")
        return payload

    def joint_act(self, actions: list[Mapping[str, Any]]) -> dict[str, Any]:
        """Submit both decisions in one RPC and a fixed side-0/side-1 order."""
        payload = self._joint_payload(actions)
        return self._request({"op": "joint_act", "actions": payload})["result"]

    def joint_transition(
        self, actions: list[Mapping[str, Any]], *, steps: int = 1
    ) -> dict[str, Any]:
        """Joint action, native step and next observation in one loopback RPC."""
        payload = self._joint_payload(actions)
        result = self._request(
            {"op": "joint_transition", "actions": payload, "steps": steps}
        )["result"]
        native_step = result["step"]
        if isinstance(native_step.get("episode"), Mapping):
            native_step["episode"] = self._enrich_episode(native_step["episode"])
            self.last_episode = deepcopy(native_step["episode"])
        if isinstance(result.get("state"), Mapping):
            result["state"] = self._enrich_state(result["state"])
        return result

    def joint_training_transition(
        self, actions: list[Mapping[str, Any]], *, steps: int = 1
    ) -> dict[str, Any]:
        """One compact RPC for joint actions, one native Tick and next state."""
        payload = self._joint_payload(actions)
        raw = self._request(
            {
                "op": "joint_training_transition_v1",
                "actions": payload,
                "steps": steps,
                "profile_native": self.profile_native,
            }
        )["result"]
        episode = raw.get("episode")
        if not isinstance(episode, Mapping):
            raise NativeHostError("compact transition episode is missing")
        enriched_episode = self._enrich_episode(episode)
        self.last_episode = deepcopy(enriched_episode)
        result: dict[str, Any] = {
            "joint_action": raw["joint_action"],
            "step": {"episode": enriched_episode},
        }
        if isinstance(raw.get("timing_v1"), Mapping):
            result["timing_v1"] = dict(raw["timing_v1"])
        if isinstance(raw.get("state"), Mapping):
            result["state"] = self._enrich_training_state(raw["state"])
        elif not (
            enriched_episode.get("terminated")
            or enriched_episode.get("truncated")
        ):
            raise NativeHostError("compact transition next state is missing")
        return result

    def probe(
        self, *, side: int, deck_index: int, x: int, y: int
    ) -> dict[str, Any]:
        """Ask libg whether a hand card may use this point without mutating state."""
        account_hi, account_lo = self.accounts[side]
        return self._request(
            {
                "op": "act",
                "action": {
                    "side": side,
                    "deck_index": deck_index,
                    "x": x,
                    "y": y,
                    "account_hi": account_hi,
                    "account_lo": account_lo,
                    "dry_run": True,
                },
            }
        )["result"]

    def probe_grid(self, *, side: int, deck_index: int) -> dict[str, Any]:
        """Return libg's 18x32 deploy mask for one current hand card."""
        account_hi, account_lo = self.accounts[side]
        return self._request(
            {
                "op": "probe_grid",
                "action": {
                    "side": side,
                    "deck_index": deck_index,
                    "account_hi": account_hi,
                    "account_lo": account_lo,
                },
            }
        )["result"]

    def step(self, steps: int = 1) -> dict[str, Any]:
        result = self._request({"op": "step", "steps": steps})["result"]
        if isinstance(result.get("episode"), Mapping):
            result["episode"] = self._enrich_episode(result["episode"])
            self.last_episode = deepcopy(result["episode"])
        return result

    def trace(
        self,
        steps: int = 1,
        *,
        trace_schema_version: int = TRACE_SCHEMA_VERSION,
        max_response_bytes: int = MAX_TRACE_RESPONSE_BYTES,
    ) -> dict[str, Any]:
        """Advance and return an initial frame plus every native Tick.

        Version 1 uses lossless full observations.  The strict guards are part
        of the truth-oracle contract: callers cannot silently accept a future
        encoding or request an unbounded JNI/JSON response.
        """
        if not 1 <= steps <= MAX_TRACE_STEPS:
            raise ValueError("trace steps must be in 1..64")
        if trace_schema_version != TRACE_SCHEMA_VERSION:
            raise ValueError("trace_schema_version must be 1")
        if not (
            MIN_TRACE_RESPONSE_BYTES
            <= max_response_bytes
            <= MAX_TRACE_RESPONSE_BYTES
        ):
            raise ValueError(
                "max_response_bytes must be in 65536..33554432"
            )
        raw = self._request(
            {
                "op": "step_trace",
                "steps": steps,
                "trace_schema_version": trace_schema_version,
                "max_response_bytes": max_response_bytes,
            }
        )["result"]
        return self._decode_trace_result(
            raw,
            steps=steps,
            max_response_bytes=max_response_bytes,
        )

    def trace_train(
        self,
        steps: int = 1,
        *,
        trace_schema_version: int = TRACE_SCHEMA_VERSION,
        max_response_bytes: int = MAX_TRACE_RESPONSE_BYTES,
        allow_nonterminal_freeze: bool = False,
    ) -> dict[str, Any]:
        """Advance up to 64 native 20 Hz Ticks in one compact RPC.

        Frames stay in raw ``observe_train_v1`` form so the Tick-store can
        normalize and delta-encode them without first allocating GUI/debug
        enrichments.  The initial frame is the pre-transition boundary and
        each complete frame is one authoritative 50 ms transition later.

        ``allow_nonterminal_freeze`` is only for a post-action duration fence:
        it accepts a final incomplete suffix whose logic Tick never advances.
        Never enable it while advancing toward a future expert action.
        """
        if not 1 <= steps <= MAX_TRACE_STEPS:
            raise ValueError("trace steps must be in 1..64")
        if trace_schema_version != TRACE_SCHEMA_VERSION:
            raise ValueError("trace_schema_version must be 1")
        if not (
            MIN_TRACE_RESPONSE_BYTES
            <= max_response_bytes
            <= MAX_TRACE_RESPONSE_BYTES
        ):
            raise ValueError(
                "max_response_bytes must be in 65536..33554432"
            )
        raw = self._request(
            {
                "op": "step_train_trace_v1",
                "steps": steps,
                "trace_schema_version": trace_schema_version,
                "max_response_bytes": max_response_bytes,
            }
        )["result"]
        return self._decode_train_trace_result(
            raw,
            steps=steps,
            max_response_bytes=max_response_bytes,
            allow_nonterminal_freeze=allow_nonterminal_freeze,
        )

    def _decode_train_trace_result(
        self,
        raw: Any,
        *,
        steps: int,
        max_response_bytes: int,
        allow_nonterminal_freeze: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise NativeHostError("compact Tick trace result must be an object")
        result = deepcopy(dict(raw))
        if (
            result.get("schema_version") != 1
            or result.get("trace_schema_version") != TRACE_SCHEMA_VERSION
            or result.get("kind") != "libg_native_train_tick_trace_v1"
            or result.get("encoding") != "compact-train-v1"
            or result.get("fixed_dt") != 0.05
            or result.get("requested_steps") != steps
            or result.get("max_response_bytes") != max_response_bytes
        ):
            raise NativeHostError("compact Tick trace protocol/version mismatch")
        stepped = result.get("stepped")
        frames = result.get("frames")
        initial = result.get("initial_frame")
        if (
            not isinstance(stepped, int)
            or isinstance(stepped, bool)
            or not 0 <= stepped <= steps
            or not isinstance(frames, list)
            or len(frames) != stepped
            or not isinstance(initial, Mapping)
            or result.get("final_frame_index") != stepped
        ):
            raise NativeHostError("compact Tick trace frame count mismatch")

        previous_tick: int | None = None
        incomplete_suffix_started = False
        for index, frame in enumerate([initial, *frames]):
            if (
                not isinstance(frame, Mapping)
                or frame.get("frame_index") != index
                or frame.get("advanced_steps") != index
                or not isinstance(frame.get("observation_complete"), bool)
                or not isinstance(frame.get("state"), Mapping)
                or "step" in frame
            ):
                raise NativeHostError("compact Tick trace frame contract mismatch")
            state = frame["state"]
            if frame["observation_complete"]:
                if incomplete_suffix_started:
                    raise NativeHostError(
                        "compact Tick trace resumed after a frozen suffix"
                    )
                self._validate_training_state(state)
                tick = int(state["tick"])
                if previous_tick is not None and tick != previous_tick + 1:
                    raise NativeHostError(
                        f"compact Tick trace is not consecutive: "
                        f"{previous_tick}->{tick}"
                    )
                previous_tick = tick
            else:
                terminal_frame = (
                    result.get("terminal") is True
                    and previous_tick is not None
                    and int(state.get("tick", -1)) == previous_tick
                )
                nonterminal_freeze = (
                    allow_nonterminal_freeze
                    and result.get("terminal") is False
                    and previous_tick is not None
                    and int(state.get("tick", -1)) == previous_tick
                )
                if not terminal_frame and not nonterminal_freeze:
                    raise NativeHostError(
                        "incomplete compact observation is only valid at terminal: "
                        f"index={index}, stepped={stepped}, "
                        f"state_tick={state.get('tick')}, previous_tick={previous_tick}, "
                        f"terminal={result.get('terminal')}, "
                        f"allow_nonterminal_freeze={allow_nonterminal_freeze}"
                    )
                if nonterminal_freeze:
                    result["nonterminal_freeze"] = True
                if terminal_frame or nonterminal_freeze:
                    incomplete_suffix_started = True

        if result.get("initial_tick") != int(initial["state"]["tick"]):
            raise NativeHostError("compact Tick trace initial tick mismatch")
        final = frames[-1] if frames else initial
        if result.get("final_tick") != int(final["state"]["tick"]):
            raise NativeHostError("compact Tick trace final tick mismatch")
        final_episode = final["state"].get("episode")
        if result.get("terminal") is True and (
            not isinstance(final_episode, Mapping)
            or not final_episode.get("terminated", False)
        ):
            raise NativeHostError("compact Tick trace terminal frame is missing")
        if isinstance(final_episode, Mapping):
            self.last_episode = self._enrich_episode(final_episode)
        return result

    def joint_transition_trace(
        self,
        actions: list[Mapping[str, int]],
        *,
        steps: int = 1,
        trace_schema_version: int = TRACE_SCHEMA_VERSION,
        max_response_bytes: int = MAX_TRACE_RESPONSE_BYTES,
    ) -> dict[str, Any]:
        """Apply canonical joint actions, then collect a batch Tick trace."""
        if not 1 <= steps <= MAX_TRACE_STEPS:
            raise ValueError("trace steps must be in 1..64")
        if trace_schema_version != TRACE_SCHEMA_VERSION:
            raise ValueError("trace_schema_version must be 1")
        if not (
            MIN_TRACE_RESPONSE_BYTES
            <= max_response_bytes
            <= MAX_TRACE_RESPONSE_BYTES
        ):
            raise ValueError(
                "max_response_bytes must be in 65536..33554432"
            )
        payload: list[dict[str, int | bool]] = []
        seen: set[int] = set()
        for value in actions:
            side = int(value["side"])
            if side not in (0, 1) or side in seen:
                raise ValueError(
                    "joint actions require unique sides 0 and/or 1"
                )
            seen.add(side)
            account_hi, account_lo = self.accounts[side]
            payload.append(
                {
                    "side": side,
                    "deck_index": int(value["deck_index"]),
                    "x": int(value["x"]),
                    "y": int(value["y"]),
                    "account_hi": account_hi,
                    "account_lo": account_lo,
                    "dry_run": False,
                }
            )
        raw = self._request(
            {
                "op": "joint_transition_trace",
                "actions": payload,
                "steps": steps,
                "trace_schema_version": trace_schema_version,
                "max_response_bytes": max_response_bytes,
            }
        )["result"]
        if not isinstance(raw, Mapping):
            raise NativeHostError(
                "joint_transition_trace result must be an object"
            )
        trace = self._decode_trace_result(
            raw.get("trace"),
            steps=steps,
            max_response_bytes=max_response_bytes,
        )
        result = deepcopy(dict(raw))
        result["trace"] = trace
        episode = result.get("episode")
        if not isinstance(episode, Mapping):
            raise NativeHostError(
                "joint_transition_trace episode must be an object"
            )
        result["episode"] = self._enrich_episode(episode)
        final_frame = (
            trace["frames"][-1]
            if trace["frames"]
            else trace["initial_frame"]
        )
        if result["episode"] != final_frame["state"].get("episode"):
            raise NativeHostError(
                "joint_transition_trace episode does not match final frame"
            )
        self.last_episode = deepcopy(result["episode"])
        return result

    def _decode_trace_result(
        self,
        raw: Any,
        *,
        steps: int,
        max_response_bytes: int,
    ) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise NativeHostError("step_trace result must be an object")
        result = deepcopy(dict(raw))
        if (
            result.get("schema_version") != 1
            or result.get("trace_schema_version") != TRACE_SCHEMA_VERSION
            or result.get("kind") != "libg_native_tick_trace"
            or result.get("encoding") != "full-v1"
            or result.get("requested_steps") != steps
            or result.get("max_response_bytes") != max_response_bytes
        ):
            raise NativeHostError("step_trace protocol/version mismatch")
        stepped = result.get("stepped")
        frames = result.get("frames")
        initial = result.get("initial_frame")
        if (
            not isinstance(stepped, int)
            or isinstance(stepped, bool)
            or not 0 <= stepped <= steps
            or not isinstance(frames, list)
            or len(frames) != stepped
            or not isinstance(initial, Mapping)
        ):
            raise NativeHostError("step_trace frame count mismatch")
        result["initial_frame"] = self._enrich_trace_frame(
            initial, expected_index=0, expected_advanced_steps=0
        )
        enriched_frames: list[dict[str, Any]] = []
        for index, frame in enumerate(frames, start=1):
            enriched_frames.append(
                self._enrich_trace_frame(
                    frame,
                    expected_index=index,
                    expected_advanced_steps=index,
                )
            )
        result["frames"] = enriched_frames
        if result.get("final_frame_index") != stepped:
            raise NativeHostError("step_trace final frame index mismatch")
        final_frame = enriched_frames[-1] if enriched_frames else result["initial_frame"]
        final_episode = final_frame["state"].get("episode")
        if result.get("terminal") is True:
            if not isinstance(final_episode, Mapping) or not final_episode.get(
                "terminated", False
            ):
                raise NativeHostError("step_trace terminal frame is missing")
        if isinstance(final_episode, Mapping):
            self.last_episode = deepcopy(dict(final_episode))
        return result

    def _enrich_trace_frame(
        self,
        frame: Mapping[str, Any],
        *,
        expected_index: int,
        expected_advanced_steps: int,
    ) -> dict[str, Any]:
        if (
            frame.get("frame_index") != expected_index
            or frame.get("advanced_steps") != expected_advanced_steps
            or not isinstance(frame.get("observation_complete"), bool)
            or not isinstance(frame.get("state"), Mapping)
        ):
            raise NativeHostError("step_trace frame contract mismatch")
        state = frame["state"]
        if (
            not isinstance(state.get("tick"), int)
            or isinstance(state.get("tick"), bool)
            or not isinstance(state.get("entities"), list)
            or not isinstance(state.get("players"), list)
            or not isinstance(state.get("coherent"), bool)
            or not isinstance(state.get("state_hash"), str)
            or state.get("state_hash_scope") != "public-observe-v6"
            or state.get("state_hash_certificate") is not False
        ):
            raise NativeHostError("step_trace observation contract mismatch")
        for player in state["players"]:
            if (
                not isinstance(player, Mapping)
                or not isinstance(player.get("cycle_deck_indices"), list)
                or not isinstance(player.get("next_deck_index"), int)
                or not isinstance(player.get("refill_timer"), int)
            ):
                raise NativeHostError("step_trace player cycle is missing")
        rich_integer_fields = (
            "target_previous_x",
            "target_previous_y",
            "movement_direction_x",
            "movement_direction_y",
            "collision_accumulator_x",
            "collision_accumulator_y",
            "collision_count",
            "avoidance_offset",
            "path_segment_direction_x",
            "path_segment_direction_y",
            "path_node_consumed",
        )
        nullable_integer_fields = (
            "pending_damage",
            "event_timer_ms",
            "attack_progress_ms",
            "attack_load_timer_ms",
        )
        for entity in state["entities"]:
            if (
                not isinstance(entity, Mapping)
                or not isinstance(entity.get("category"), int)
                or isinstance(entity.get("category"), bool)
                or not isinstance(entity.get("generation_key"), int)
                or isinstance(entity.get("generation_key"), bool)
                or not isinstance(entity.get("creation_ordinal"), int)
                or isinstance(entity.get("creation_ordinal"), bool)
                or entity["generation_key"] != entity["category"]
                or entity["creation_ordinal"] != entity["category"] - 5_000_000
                or entity["creation_ordinal"] < 0
                or any(
                not isinstance(entity.get(field), int)
                or isinstance(entity.get(field), bool)
                for field in rich_integer_fields
                )
            ):
                raise NativeHostError("step_trace rich entity state is missing")
            if any(
                entity.get(field) is not None
                and (
                    not isinstance(entity.get(field), int)
                    or isinstance(entity.get(field), bool)
                )
                for field in nullable_integer_fields
            ):
                raise NativeHostError("step_trace rich entity timer is invalid")
            path_nodes = entity.get("path_nodes")
            if path_nodes is not None and (
                not isinstance(path_nodes, list)
                or len(path_nodes) > 115
                or any(
                    not isinstance(node, int) or isinstance(node, bool)
                    for node in path_nodes
                )
            ):
                raise NativeHostError("step_trace native path state is invalid")
        generation_keys = [
            int(entity["generation_key"]) for entity in state["entities"]
        ]
        if len(generation_keys) != len(set(generation_keys)):
            raise NativeHostError("step_trace Character generation key is duplicated")
        projectiles = state.get("projectiles")
        if not isinstance(projectiles, list):
            raise NativeHostError("step_trace native projectile state is missing")
        for projectile in projectiles:
            if (
                not isinstance(projectile, Mapping)
                or projectile.get("vtable_rva") != "0x1969b38"
                or any(
                    not isinstance(projectile.get(field), int)
                    or isinstance(projectile.get(field), bool)
                    for field in (
                        "generation_key",
                        "side",
                        "x",
                        "y",
                        "x2",
                        "y2",
                        "card_id",
                        "target_x",
                        "target_y",
                    )
                )
            ):
                raise NativeHostError("step_trace native projectile is invalid")
        result = deepcopy(dict(frame))
        result["state"] = self._enrich_state(state)
        if isinstance(result.get("step"), Mapping):
            native_step = deepcopy(dict(result["step"]))
            if isinstance(native_step.get("episode"), Mapping):
                native_step["episode"] = self._enrich_episode(
                    native_step["episode"]
                )
            result["step"] = native_step
        return result

    def step_episode(
        self,
        action: Mapping[str, int] | None,
        *,
        steps: int = 1,
    ) -> tuple[dict[str, Any], dict[int, float], bool, bool, dict[str, Any]]:
        """Apply one action and return a multi-agent Gym-style transition."""
        action_result = None
        if action is not None:
            action_result = self.act(
                side=int(action["side"]),
                deck_index=int(action["deck_index"]),
                x=int(action["x"]),
                y=int(action["y"]),
            )
        native_step = self.step(steps)
        episode = native_step["episode"]
        terminated = bool(episode["terminated"])
        truncated = bool(episode["truncated"])
        rewards = dict(episode["rewards_by_side"])
        if terminated or truncated:
            observation = {
                "schema_version": 1,
                "kind": "libg_native_terminal_state",
                "tick": int(episode["terminal_tick"]),
                "elapsed_seconds": round(
                    int(episode["terminal_tick"]) * 0.05, 3
                ),
                "entities": [],
                "players": [],
                "episode": deepcopy(episode),
                "final_observation_complete": False,
            }
        else:
            observation = self.observe()
        info = {
            "native_step": native_step,
            "native_action": action_result,
            "episode": deepcopy(episode),
        }
        return observation, rewards, terminated, truncated, info

    def export_terminal(
        self,
        path: Path,
        episode: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Write a stable JSON artifact for a completed native episode."""
        value = deepcopy(dict(episode or self.last_episode or {}))
        if not value.get("terminated") and not value.get("truncated"):
            raise ValueError("native episode has not reached a terminal state")
        artifact = {
            "schema_version": 1,
            "kind": "libg_native_terminal_export",
            "episode": value,
            "decks": deepcopy(self.decks),
            "accounts": [list(account) for account in self.accounts],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return artifact

    def transition(
        self,
        action: Mapping[str, int] | None,
        *,
        steps: int = 1,
    ) -> dict[str, Any]:
        if action is not None:
            self.act(
                side=int(action["side"]),
                deck_index=int(action["deck_index"]),
                x=int(action["x"]),
                y=int(action["y"]),
            )
        self.step(steps)
        return self.observe()
