"""Local JSON-line client for the isolated libg service."""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from pathlib import Path
from typing import Any, MutableMapping


MAX_RESPONSE_BYTES = 64 * 1024 * 1024
MAX_REQUEST_BYTES = 32 * 1024 * 1024
TRACE_SCHEMA_VERSION = 1
MAX_TRACE_STEPS = 64
MIN_TRACE_RESPONSE_BYTES = 64 * 1024
MAX_TRACE_RESPONSE_BYTES = 32 * 1024 * 1024
IDEMPOTENT_OPS = {"ping", "status", "observe", "probe_grid"}


class JsonLineClient:
    """One serialized persistent connection to a single native Worker.

    Mutating requests are never replayed after an ambiguous I/O failure.
    Read-only requests may reconnect once. The next explicit request always
    opens a fresh connection after a failure or Worker replacement.
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 37031,
        timeout: float = 10.0,
        profile: MutableMapping[str, float] | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.profile = profile
        self._socket: socket.socket | None = None
        self._reader: Any = None
        self._lock = threading.Lock()

    def close(self) -> None:
        reader, connection = self._reader, self._socket
        self._reader = None
        self._socket = None
        if reader is not None:
            try:
                reader.close()
            except OSError:
                pass
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass

    def _connect(self) -> float:
        started = time.perf_counter()
        connection = socket.create_connection(
            (self.host, self.port), timeout=self.timeout
        )
        connection.settimeout(self.timeout)
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._socket = connection
        self._reader = connection.makefile("rb", buffering=64 * 1024)
        return time.perf_counter() - started

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        encoded_started = time.perf_counter()
        encoded = json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8") + b"\n"
        serialized = time.perf_counter()
        if len(encoded) > MAX_REQUEST_BYTES:
            raise ValueError("native host request exceeds safety limit")
        operation = str(payload.get("op", ""))
        with self._lock:
            for attempt in range(2):
                connect_seconds = 0.0
                try:
                    if self._socket is None:
                        connect_seconds = self._connect()
                    assert self._socket is not None and self._reader is not None
                    sent_started = time.perf_counter()
                    self._socket.sendall(encoded)
                    sent = time.perf_counter()
                    raw = self._reader.readline(MAX_RESPONSE_BYTES + 2)
                    received = time.perf_counter()
                    if not raw:
                        raise ConnectionError("native host closed without a response")
                    if len(raw) > MAX_RESPONSE_BYTES + 1 or not raw.endswith(b"\n"):
                        raise ValueError("native host response exceeds safety limit")
                    response_bytes = len(raw) - 1
                    response = json.loads(raw[:-1].decode("utf-8"))
                    parsed = time.perf_counter()
                    if not isinstance(response, dict):
                        raise TypeError("native host response root must be an object")
                    if self.profile is not None:
                        values = {
                            "rpc_calls": 1.0,
                            "rpc_request_bytes": float(len(encoded)),
                            "rpc_response_bytes": float(response_bytes),
                            "rpc_serialize_seconds": serialized - encoded_started,
                            "rpc_connect_seconds": connect_seconds,
                            "rpc_send_seconds": sent - sent_started,
                            "rpc_receive_seconds": received - sent,
                            "rpc_parse_seconds": parsed - received,
                            "rpc_total_seconds": parsed - encoded_started,
                            "rpc_reconnects": float(attempt),
                        }
                        for key, value in values.items():
                            self.profile[key] = self.profile.get(key, 0.0) + value
                    if operation == "shutdown":
                        self.close()
                    return response
                except (OSError, ConnectionError, ValueError, json.JSONDecodeError):
                    self.close()
                    if attempt == 0 and operation in IDEMPOTENT_OPS:
                        continue
                    raise
        raise AssertionError("persistent request retry loop exhausted")

    def __enter__(self) -> "JsonLineClient":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def request(
    payload: dict[str, Any],
    *,
    host: str = "127.0.0.1",
    port: int = 37031,
    timeout: float = 10.0,
    profile: MutableMapping[str, float] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    ) + b"\n"
    serialized = time.perf_counter()
    if len(encoded) > MAX_REQUEST_BYTES:
        raise ValueError("native host request exceeds safety limit")
    connect_started = time.perf_counter()
    with socket.create_connection((host, port), timeout=timeout) as connection:
        connected = time.perf_counter()
        connection.sendall(encoded)
        sent = time.perf_counter()
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = connection.recv(64 * 1024)
            if not chunk:
                break
            newline = chunk.find(b"\n")
            if newline >= 0:
                total += newline
                if total > MAX_RESPONSE_BYTES:
                    raise ValueError("native host response exceeds safety limit")
                chunks.append(chunk[:newline])
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_RESPONSE_BYTES:
                raise ValueError("native host response exceeds safety limit")
    received = time.perf_counter()
    if not chunks:
        raise ConnectionError("native host closed without a response")
    response = json.loads(b"".join(chunks).decode("utf-8"))
    parsed = time.perf_counter()
    if not isinstance(response, dict):
        raise TypeError("native host response root must be an object")
    if profile is not None:
        values = {
            "rpc_calls": 1.0,
            "rpc_request_bytes": float(len(encoded)),
            "rpc_response_bytes": float(total),
            "rpc_serialize_seconds": serialized - started,
            "rpc_connect_seconds": connected - connect_started,
            "rpc_send_seconds": sent - connected,
            "rpc_receive_seconds": received - sent,
            "rpc_parse_seconds": parsed - received,
            "rpc_total_seconds": parsed - started,
        }
        for key, value in values.items():
            profile[key] = profile.get(key, 0.0) + value
    return response


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=37031)
    parser.add_argument("--timeout", type=float, default=10.0)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("ping")
    commands.add_parser("status")
    commands.add_parser("observe")
    commands.add_parser("shutdown")
    step = commands.add_parser("step")
    step.add_argument("steps", nargs="?", type=int, default=1)
    trace = commands.add_parser("trace")
    trace.add_argument("steps", nargs="?", type=int, default=1)
    trace.add_argument(
        "--trace-schema-version", type=int, default=TRACE_SCHEMA_VERSION
    )
    trace.add_argument(
        "--max-response-bytes", type=int, default=MAX_TRACE_RESPONSE_BYTES
    )
    load = commands.add_parser("load-replay")
    load.add_argument("path", type=Path)
    act = commands.add_parser("act")
    act.add_argument("path", type=Path)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    if arguments.command in {"load-replay", "act"}:
        replay = json.loads(arguments.path.read_text(encoding="utf-8-sig"))
        if not isinstance(replay, dict):
            raise TypeError(f"{arguments.command} JSON root must be an object")
        if arguments.command == "load-replay":
            payload: dict[str, Any] = {"op": "load_replay", "replay": replay}
        else:
            payload = {"op": "act", "action": replay}
    elif arguments.command == "step":
        payload = {"op": "step", "steps": arguments.steps}
    elif arguments.command == "trace":
        if not 1 <= arguments.steps <= MAX_TRACE_STEPS:
            raise ValueError("trace steps must be in 1..64")
        if arguments.trace_schema_version != TRACE_SCHEMA_VERSION:
            raise ValueError("trace_schema_version must be 1")
        if not (
            MIN_TRACE_RESPONSE_BYTES
            <= arguments.max_response_bytes
            <= MAX_TRACE_RESPONSE_BYTES
        ):
            raise ValueError(
                "max_response_bytes must be in 65536..33554432"
            )
        payload = {
            "op": "step_trace",
            "steps": arguments.steps,
            "trace_schema_version": arguments.trace_schema_version,
            "max_response_bytes": arguments.max_response_bytes,
        }
    else:
        payload = {"op": arguments.command}
    response = request(
        payload,
        host=arguments.host,
        port=arguments.port,
        timeout=arguments.timeout,
    )
    print(json.dumps(response, ensure_ascii=False, indent=2))
    return 0 if response.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
