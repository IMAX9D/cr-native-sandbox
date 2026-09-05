"""Local multi-process policy batching over an authenticated Unix socket."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from multiprocessing.connection import Client, Connection, Listener
from pathlib import Path
import queue
import threading
import time
import traceback
from typing import Any, Hashable, Sequence

import numpy as np
import torch
from torch import Tensor

from .actions import ExpertActionMasks
from .batched_policy import (
    BatchedPolicyService,
    PolicyRequest,
    SampledPolicyAction,
)


PROTOCOL_KIND = "cr_native_remote_policy_v1"
DEFAULT_AUTHKEY = b"cr-native-policy-v1"


class RemotePolicyError(RuntimeError):
    pass


@dataclass
class _Pending:
    operation: str
    payload: dict[str, Any]
    event: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: dict[str, str] | None = None


def _request_to_wire(request: PolicyRequest) -> dict[str, Any]:
    return {
        "worker_id": request.worker_id,
        "side": request.side,
        "actor_sha256": request.actor_sha256,
        "actor_inputs": {
            name: (
                None
                if value is None
                else value.detach().cpu().contiguous().numpy()
            )
            for name, value in request.actor_inputs.items()
        },
        "masks": {
            name: getattr(request.masks, name)
            .detach().cpu().contiguous().numpy()
            for name in request.masks.__dataclass_fields__
        },
        "delta_ticks": request.delta_ticks,
        "reset_hidden": request.reset_hidden,
    }


def _request_from_wire(value: Any) -> PolicyRequest:
    if not isinstance(value, dict):
        raise TypeError("remote policy request row must be an object")
    actor_inputs = value.get("actor_inputs")
    masks = value.get("masks")
    if not isinstance(actor_inputs, dict) or not isinstance(masks, dict):
        raise TypeError("remote policy request tensors are missing")
    return PolicyRequest(
        worker_id=value["worker_id"],
        side=int(value["side"]),
        actor_sha256=str(value["actor_sha256"]),
        actor_inputs={
            str(name): (
                None if item is None else torch.from_numpy(np.asarray(item))
            )
            for name, item in actor_inputs.items()
        },
        masks=ExpertActionMasks(**{
            str(name): torch.from_numpy(np.asarray(item))
            for name, item in masks.items()
        }),
        delta_ticks=int(value["delta_ticks"]),
        reset_hidden=bool(value["reset_hidden"]),
    )


class RemotePolicyClient:
    """Duck-typed policy service used by one CPU collector process."""

    def __init__(
        self,
        address: str | Path,
        *,
        authkey: bytes = DEFAULT_AUTHKEY,
        expected_actor_hashes: Sequence[str] = (),
    ) -> None:
        self.address = str(address)
        self._connection = Client(
            self.address, family="AF_UNIX", authkey=authkey
        )
        self._lock = threading.Lock()
        self._hidden: dict[
            tuple[str, Hashable, int], tuple[Tensor, Tensor]
        ] = {}
        self.forward_calls = 0
        status = self._request("status")
        hashes = tuple(str(value) for value in status["actor_hashes"])
        expected = tuple(expected_actor_hashes)
        if expected and set(hashes) != set(expected):
            self.close()
            raise RemotePolicyError(
                f"remote Actor hashes differ: {hashes!r} != {expected!r}"
            )
        self._actor_hashes = hashes

    @property
    def registered_actor_hashes(self) -> tuple[str, ...]:
        return self._actor_hashes

    def _request(self, operation: str, **payload: Any) -> Any:
        message = {
            "kind": PROTOCOL_KIND,
            "operation": operation,
            **payload,
        }
        with self._lock:
            self._connection.send(message)
            response = self._connection.recv()
        if not isinstance(response, dict) or response.get("kind") != PROTOCOL_KIND:
            raise RemotePolicyError("remote policy returned an invalid envelope")
        if response.get("ok") is not True:
            raise RemotePolicyError(
                f"{response.get('error_type', 'RemotePolicyError')}: "
                f"{response.get('error', response)}"
            )
        return response.get("result")

    def reset_episode(self, worker_id: Hashable) -> int:
        removed = int(self._request("reset", worker_id=worker_id))
        for key in [key for key in self._hidden if key[1] == worker_id]:
            del self._hidden[key]
        return removed

    def act(
        self,
        requests: Sequence[PolicyRequest],
        *,
        deterministic: bool | None = None,
    ) -> list[SampledPolicyAction]:
        rows = list(requests)
        result = self._request(
            "act",
            requests=[_request_to_wire(request) for request in rows],
            deterministic=deterministic,
        )
        actions = list(result["actions"])
        hidden = [
            tuple(
                torch.from_numpy(np.asarray(item)).contiguous().clone()
                for item in state
            )
            for state in result["pre_action_hidden"]
        ]
        if len(actions) != len(rows) or len(hidden) != len(rows):
            raise RemotePolicyError("remote policy dropped an action or hidden state")
        if not all(isinstance(value, SampledPolicyAction) for value in actions):
            raise RemotePolicyError("remote policy returned an invalid action type")
        for action, state in zip(actions, hidden, strict=True):
            if not (
                isinstance(state, tuple)
                and len(state) == 2
                and all(isinstance(value, Tensor) for value in state)
            ):
                raise RemotePolicyError("remote policy returned invalid hidden state")
            self._hidden[(action.actor_sha256, action.worker_id, action.side)] = state
        self.forward_calls += len({request.actor_sha256 for request in rows})
        return actions

    def last_pre_action_hidden(
        self, *, actor_sha256: str, worker_id: Hashable, side: int
    ) -> tuple[Tensor, Tensor]:
        key = (actor_sha256, worker_id, side)
        hidden = self._hidden.get(key)
        if hidden is None:
            raise KeyError(f"no remote pre-action recurrent state for {key!r}")
        return tuple(value.contiguous().clone() for value in hidden)  # type: ignore[return-value]

    def last_pre_action_hidden_batch(
        self, actions: Sequence[SampledPolicyAction]
    ) -> list[tuple[Tensor, Tensor]]:
        return [
            self.last_pre_action_hidden(
                actor_sha256=action.actor_sha256,
                worker_id=action.worker_id,
                side=action.side,
            )
            for action in actions
        ]

    def server_metrics(self) -> dict[str, Any]:
        return dict(self._request("status")["metrics"])

    def shutdown_server(self) -> dict[str, Any]:
        return dict(self._request("shutdown"))

    def close(self) -> None:
        connection = getattr(self, "_connection", None)
        if connection is None:
            return
        try:
            self._request("close_client")
        except (EOFError, OSError, RemotePolicyError):
            pass
        finally:
            connection.close()
            self._connection = None  # type: ignore[assignment]

    def __enter__(self) -> "RemotePolicyClient":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class RemotePolicyServer:
    """Serialize recurrent state while dynamically batching client turns."""

    def __init__(
        self,
        service: BatchedPolicyService,
        address: str | Path,
        *,
        authkey: bytes = DEFAULT_AUTHKEY,
        microbatch_seconds: float = 0.002,
        max_actor_rows: int = 256,
    ) -> None:
        if microbatch_seconds < 0 or max_actor_rows < 1:
            raise ValueError("remote policy batching limits are invalid")
        self.service = service
        self.address = Path(address)
        self.authkey = authkey
        self.microbatch_seconds = float(microbatch_seconds)
        self.max_actor_rows = int(max_actor_rows)
        self._queue: queue.Queue[_Pending] = queue.Queue()
        self._stop = threading.Event()
        self._listener: Listener | None = None
        self.metrics: dict[str, float] = {
            "client_act_calls": 0.0,
            "microbatches": 0.0,
            "actor_rows": 0.0,
            "policy_seconds": 0.0,
            "max_microbatch_rows": 0.0,
        }

    def _handler(self, connection: Connection) -> None:
        try:
            while not self._stop.is_set():
                try:
                    message = connection.recv()
                except EOFError:
                    break
                if not isinstance(message, dict) or message.get("kind") != PROTOCOL_KIND:
                    connection.send({
                        "kind": PROTOCOL_KIND,
                        "ok": False,
                        "error_type": "ValueError",
                        "error": "invalid remote policy request envelope",
                    })
                    continue
                pending = _Pending(
                    str(message.get("operation", "")), dict(message)
                )
                self._queue.put(pending)
                pending.event.wait()
                if pending.error is None:
                    response = {
                        "kind": PROTOCOL_KIND,
                        "ok": True,
                        "result": pending.result,
                    }
                else:
                    response = {
                        "kind": PROTOCOL_KIND,
                        "ok": False,
                        **pending.error,
                    }
                connection.send(response)
                if pending.operation in ("close_client", "shutdown"):
                    break
        except (BrokenPipeError, ConnectionResetError, EOFError, OSError):
            pass
        finally:
            connection.close()

    def _accept(self) -> None:
        assert self._listener is not None
        while not self._stop.is_set():
            try:
                connection = self._listener.accept()
            except (OSError, EOFError):
                break
            threading.Thread(
                target=self._handler, args=(connection,), daemon=True
            ).start()

    @staticmethod
    def _fail(pending: _Pending, error: BaseException) -> None:
        pending.error = {
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": "".join(traceback.format_exception(error)),
        }
        pending.event.set()

    def _act(self, pending_rows: list[_Pending]) -> None:
        flattened: list[PolicyRequest] = []
        lengths = []
        for pending in pending_rows:
            requests = [
                _request_from_wire(value)
                for value in pending.payload.get("requests", ())
            ]
            flattened.extend(requests)
            lengths.append(len(requests))
        modes = {pending.payload.get("deterministic") for pending in pending_rows}
        if len(modes) != 1:
            raise ValueError("one remote microbatch cannot mix sampling modes")
        started = time.perf_counter()
        actions = self.service.act(
            flattened, deterministic=next(iter(modes))
        )
        hidden = self.service.last_pre_action_hidden_batch(actions)
        elapsed = time.perf_counter() - started
        self.metrics["client_act_calls"] += float(len(pending_rows))
        self.metrics["microbatches"] += 1.0
        self.metrics["actor_rows"] += float(len(flattened))
        self.metrics["policy_seconds"] += elapsed
        self.metrics["max_microbatch_rows"] = max(
            self.metrics["max_microbatch_rows"], float(len(flattened))
        )
        cursor = 0
        for pending, length in zip(pending_rows, lengths, strict=True):
            pending.result = {
                "actions": actions[cursor:cursor + length],
                "pre_action_hidden": [
                    tuple(value.contiguous().numpy() for value in state)
                    for state in hidden[cursor:cursor + length]
                ],
            }
            cursor += length
            pending.event.set()

    def _status(self) -> dict[str, Any]:
        metrics = dict(self.metrics)
        metrics["service_forward_calls"] = float(self.service.forward_calls)
        if metrics["microbatches"]:
            metrics["mean_microbatch_rows"] = (
                metrics["actor_rows"] / metrics["microbatches"]
            )
        else:
            metrics["mean_microbatch_rows"] = 0.0
        return {
            "actor_hashes": list(self.service.registered_actor_hashes),
            "metrics": metrics,
        }

    def serve_forever(self) -> dict[str, Any]:
        self.address.parent.mkdir(parents=True, exist_ok=True)
        self.address.unlink(missing_ok=True)
        self._listener = Listener(
            str(self.address), family="AF_UNIX", authkey=self.authkey
        )
        threading.Thread(target=self._accept, daemon=True).start()
        backlog: deque[_Pending] = deque()
        try:
            while not self._stop.is_set():
                pending = backlog.popleft() if backlog else self._queue.get()
                try:
                    if pending.operation == "act":
                        batch = [pending]
                        rows = len(pending.payload.get("requests", ()))
                        mode = pending.payload.get("deterministic")
                        deadline = time.perf_counter() + self.microbatch_seconds
                        while rows < self.max_actor_rows:
                            remaining = deadline - time.perf_counter()
                            if remaining <= 0:
                                break
                            try:
                                candidate = self._queue.get(timeout=remaining)
                            except queue.Empty:
                                break
                            candidate_rows = len(candidate.payload.get("requests", ()))
                            if (
                                candidate.operation == "act"
                                and candidate.payload.get("deterministic") == mode
                                and rows + candidate_rows <= self.max_actor_rows
                            ):
                                batch.append(candidate)
                                rows += candidate_rows
                            else:
                                backlog.append(candidate)
                                break
                        try:
                            self._act(batch)
                        except BaseException as error:
                            for item in batch:
                                self._fail(item, error)
                    elif pending.operation == "reset":
                        pending.result = self.service.reset_episode(
                            pending.payload.get("worker_id")
                        )
                        pending.event.set()
                    elif pending.operation == "status":
                        pending.result = self._status()
                        pending.event.set()
                    elif pending.operation == "close_client":
                        pending.result = True
                        pending.event.set()
                    elif pending.operation == "shutdown":
                        pending.result = self._status()["metrics"]
                        pending.event.set()
                        self._stop.set()
                    else:
                        raise ValueError(
                            f"unknown remote policy operation: {pending.operation!r}"
                        )
                except BaseException as error:
                    self._fail(pending, error)
        finally:
            self._stop.set()
            if self._listener is not None:
                self._listener.close()
            self.address.unlink(missing_ok=True)
        return self._status()["metrics"]


__all__ = [
    "DEFAULT_AUTHKEY",
    "PROTOCOL_KIND",
    "RemotePolicyClient",
    "RemotePolicyError",
    "RemotePolicyServer",
]
