"""Local multi-process policy batching over an authenticated Unix socket."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from multiprocessing.connection import Client, Connection, Listener
from multiprocessing.reduction import ForkingPickler
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
    PolicyIdentity,
    SampledPolicyAction,
)
from .policy_columns import MAX_TENSOR_BYTES, decode_columns, encode_columns


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
    queued_at: float = field(default_factory=time.perf_counter)
    delivered: threading.Event = field(default_factory=threading.Event)


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
        "capture_pre_action_hidden": request.capture_pre_action_hidden,
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
        capture_pre_action_hidden=bool(value.get("capture_pre_action_hidden", True)),
    )


class RemotePolicyClient:
    """Duck-typed policy service used by one CPU collector process."""

    def __init__(
        self,
        address: str | Path,
        *,
        authkey: bytes = DEFAULT_AUTHKEY,
        expected_actor_hashes: Sequence[str] = (),
        wire_format: str = "rows-v1",
        connection_family: str = "AF_UNIX",
    ) -> None:
        if wire_format not in ("rows-v1", "columns-v2"):
            raise ValueError("unknown policy wire format")
        if connection_family not in ("AF_UNIX", "AF_PIPE"):
            raise ValueError("policy IPC must use a local socket or named pipe")
        self.address = str(address)
        self.wire_format = wire_format
        self._connection = Client(
            self.address, family=connection_family, authkey=authkey
        )
        self._lock = threading.Lock()
        self._hidden: dict[
            tuple[str, Hashable, int], tuple[Tensor, Tensor]
        ] = {}
        self.forward_calls = 0
        status = self._request("status")
        if wire_format == "columns-v2" and wire_format not in status.get("wire_formats", ()):
            self.close()
            raise RemotePolicyError("server does not support columnar requests; no action was sent")
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
        if getattr(self, "wire_format", "rows-v1") == "columns-v2":
            result = self._request("act_columns_v2", packet=encode_columns(rows), deterministic=deterministic)
        else:
            result = self._request(
                "act", requests=[_request_to_wire(request) for request in rows], deterministic=deterministic,
            )
        actions = list(result["actions"])
        hidden = [
            None if state is None else
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
        for request, action, state in zip(rows, actions, hidden, strict=True):
            key = (action.actor_sha256, action.worker_id, action.side)
            if key != (request.actor_sha256, request.worker_id, request.side):
                raise RemotePolicyError("remote policy reordered action identities")
            if state is None:
                self._hidden.pop(key, None)
                if request.capture_pre_action_hidden:
                    raise RemotePolicyError("remote policy dropped a requested recurrent anchor")
                continue
            if not (
                isinstance(state, tuple)
                and len(state) == 2
                and all(isinstance(value, Tensor) for value in state)
            ):
                raise RemotePolicyError("remote policy returned invalid hidden state")
            self._hidden[key] = state
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
        max_pending_requests: int = 64,
        connection_family: str = "AF_UNIX",
    ) -> None:
        if microbatch_seconds < 0 or max_actor_rows < 1 or max_pending_requests < 1:
            raise ValueError("remote policy batching limits are invalid")
        self.service = service
        if connection_family not in ("AF_UNIX", "AF_PIPE"):
            raise ValueError("policy IPC must use a local socket or named pipe")
        self.connection_family = connection_family
        self.address = Path(address)
        self.authkey = authkey
        self.microbatch_seconds = float(microbatch_seconds)
        self.max_actor_rows = int(max_actor_rows)
        self._queue: queue.Queue[_Pending] = queue.Queue(maxsize=max_pending_requests)
        self._stop = threading.Event()
        self.ready_event = threading.Event()
        self._listener: Listener | None = None
        self._io_lock = threading.Lock()
        self._io_metrics = {"request_wire_bytes": 0.0, "response_wire_bytes": 0.0,
                            "request_unpickle_seconds": 0.0, "response_pickle_seconds": 0.0}
        self.metrics: dict[str, float] = {
            "client_act_calls": 0.0,
            "microbatches": 0.0,
            "actor_rows": 0.0,
            "policy_seconds": 0.0,
            "max_microbatch_rows": 0.0,
            "hidden_rows_transferred": 0.0,
            "hidden_bytes_transferred": 0.0,
            "columnar_client_act_calls": 0.0,
            "request_schema_decode_seconds": 0.0,
            "inference_seconds": 0.0,
            "recurrent_export_seconds": 0.0,
            "queue_wait_seconds": 0.0,
        }

    def _handler(self, connection: Connection) -> None:
        try:
            while not self._stop.is_set():
                try:
                    wire = connection.recv_bytes(MAX_TENSOR_BYTES + 1024 * 1024)
                except EOFError:
                    break
                decoded_at = time.perf_counter()
                message = ForkingPickler.loads(wire)
                with self._io_lock:
                    self._io_metrics["request_wire_bytes"] += len(wire)
                    self._io_metrics["request_unpickle_seconds"] += time.perf_counter() - decoded_at
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
                while not self._stop.is_set():
                    try:
                        self._queue.put(pending, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                else:
                    break
                while not pending.event.wait(timeout=0.1):
                    if self._stop.is_set():
                        break
                if not pending.event.is_set():
                    break
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
                encoded_at = time.perf_counter()
                wire = ForkingPickler.dumps(response)
                with self._io_lock:
                    self._io_metrics["response_pickle_seconds"] += time.perf_counter() - encoded_at
                    self._io_metrics["response_wire_bytes"] += len(wire)
                try:
                    connection.send_bytes(wire)
                finally:
                    pending.delivered.set()
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
        decode_started = time.perf_counter()
        columnar = pending_rows[0].operation == "act_columns_v2"
        if any((pending.operation == "act_columns_v2") != columnar for pending in pending_rows):
            raise ValueError("one microbatch cannot mix wire formats")
        if sum(self._row_count(pending) for pending in pending_rows) > self.max_actor_rows:
            raise ValueError("policy microbatch exceeds row capacity")
        flattened: list[PolicyRequest | PolicyIdentity] = []
        tensor_batches = []
        lengths = []
        for pending in pending_rows:
            if columnar:
                batches, requests = decode_columns(pending.payload.get("packet"), offset=len(flattened))
                tensor_batches.extend(batches)
            else:
                requests = [_request_from_wire(value) for value in pending.payload.get("requests", ())]
            flattened.extend(requests)
            lengths.append(len(requests))
        if len(flattened) > self.max_actor_rows:
            raise ValueError("policy microbatch exceeds row capacity")
        modes = {pending.payload.get("deterministic") for pending in pending_rows}
        if len(modes) != 1:
            raise ValueError("one remote microbatch cannot mix sampling modes")
        started = time.perf_counter()
        self.metrics["request_schema_decode_seconds"] += started - decode_started
        self.metrics["queue_wait_seconds"] += sum(max(0.0, decode_started - p.queued_at) for p in pending_rows)
        if columnar:
            actions = self.service.act_tensor_batches(tensor_batches, deterministic=next(iter(modes)))
            self.metrics["columnar_client_act_calls"] += len(pending_rows)
        else:
            actions = self.service.act(flattened, deterministic=next(iter(modes)))
        export_started = time.perf_counter()
        self.metrics["inference_seconds"] += export_started - started
        if len(actions) != len(flattened):
            raise ValueError("policy service dropped an action")
        captures = [i for i, request in enumerate(flattened) if request.capture_pre_action_hidden]
        captured = self.service.last_pre_action_hidden_batch([actions[i] for i in captures])
        if len(captured) != len(captures):
            raise ValueError("policy service dropped a recurrent anchor")
        hidden: list[Any] = [None] * len(actions)
        for index, state in zip(captures, captured, strict=True):
            hidden[index] = tuple(value.contiguous().numpy() for value in state)
        self.metrics["hidden_rows_transferred"] += len(captures)
        self.metrics["hidden_bytes_transferred"] += sum(
            value.nbytes for state in hidden if state is not None for value in state
        )
        elapsed = time.perf_counter() - started
        self.metrics["recurrent_export_seconds"] += time.perf_counter() - export_started
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
                "pre_action_hidden": hidden[cursor:cursor + length],
            }
            cursor += length
            pending.event.set()

    def _status(self) -> dict[str, Any]:
        metrics = dict(self.metrics)
        with self._io_lock:
            metrics.update(self._io_metrics)
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
            "wire_formats": ["rows-v1"] + (["columns-v2"] if callable(getattr(self.service, "act_tensor_batches", None)) else []),
            "configuration": {
                "dense_sampling": bool(getattr(self.service, "dense_sampling", False)),
                "compile_actors": bool(getattr(self.service, "compile_actors", False)),
                "collate_before_transfer": bool(getattr(self.service, "collate_before_transfer", False)),
                "microbatch_seconds": self.microbatch_seconds,
                "max_actor_rows": self.max_actor_rows,
                "max_pending_requests": self._queue.maxsize,
            },
        }

    def serve_forever(self) -> dict[str, Any]:
        if self.connection_family == "AF_UNIX":
            self.address.parent.mkdir(parents=True, exist_ok=True)
            if self.address.exists():
                raise RemotePolicyError("policy socket already exists; verify and stop its owner first")
        self._listener = Listener(
            str(self.address), family=self.connection_family, authkey=self.authkey
        )
        if self.connection_family == "AF_UNIX":
            self.address.chmod(0o600)
        self.ready_event.set()
        threading.Thread(target=self._accept, daemon=True).start()
        backlog: deque[_Pending] = deque()
        try:
            while not self._stop.is_set():
                pending = backlog.popleft() if backlog else self._queue.get()
                try:
                    if pending.operation in ("act", "act_columns_v2"):
                        batch = [pending]
                        rows = self._row_count(pending)
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
                            candidate_rows = self._row_count(candidate)
                            if (
                                candidate.operation == pending.operation
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
                        pending.delivered.wait(timeout=2.0)
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
            if self.connection_family == "AF_UNIX":
                self.address.unlink(missing_ok=True)
        return self._status()["metrics"]

    @staticmethod
    def _row_count(pending: _Pending) -> int:
        if pending.operation == "act_columns_v2":
            packet = pending.payload.get("packet")
            value = packet.get("row_count") if isinstance(packet, dict) else None
            return value if type(value) is int and value >= 0 else 0
        requests = pending.payload.get("requests", ())
        return len(requests) if isinstance(requests, (list, tuple)) else 0


__all__ = [
    "DEFAULT_AUTHKEY",
    "PROTOCOL_KIND",
    "RemotePolicyClient",
    "RemotePolicyError",
    "RemotePolicyServer",
]
