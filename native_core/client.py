"""Local JSON-line client for the isolated libg service."""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path
from typing import Any


MAX_RESPONSE_BYTES = 64 * 1024 * 1024
MAX_REQUEST_BYTES = 32 * 1024 * 1024
TRACE_SCHEMA_VERSION = 1
MAX_TRACE_STEPS = 64
MIN_TRACE_RESPONSE_BYTES = 64 * 1024
MAX_TRACE_RESPONSE_BYTES = 32 * 1024 * 1024


def request(
    payload: dict[str, Any],
    *,
    host: str = "127.0.0.1",
    port: int = 37031,
    timeout: float = 10.0,
) -> dict[str, Any]:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    ) + b"\n"
    if len(encoded) > MAX_REQUEST_BYTES:
        raise ValueError("native host request exceeds safety limit")
    with socket.create_connection((host, port), timeout=timeout) as connection:
        connection.sendall(encoded)
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
    if not chunks:
        raise ConnectionError("native host closed without a response")
    response = json.loads(b"".join(chunks).decode("utf-8"))
    if not isinstance(response, dict):
        raise TypeError("native host response root must be an object")
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
