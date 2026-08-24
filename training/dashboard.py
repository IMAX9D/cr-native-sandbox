"""Loopback-only HTTP surface for live native self-play telemetry."""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from typing import Any, Callable
from urllib.parse import urlparse


class TrainingDashboardServer:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        snapshot: Callable[[], dict[str, Any]],
        request_shutdown: Callable[[], tuple[bool, str]],
    ) -> None:
        html_path = Path(__file__).with_name("dashboard.html")
        self.html = html_path.read_bytes()
        self.snapshot = snapshot
        self.request_shutdown = request_shutdown
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args: object) -> None:
                return

            def _send(
                self,
                status: HTTPStatus,
                body: bytes,
                content_type: str,
            ) -> None:
                self.send_response(status.value)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                if path in ("/", "/index.html"):
                    self._send(
                        HTTPStatus.OK,
                        owner.html,
                        "text/html; charset=utf-8",
                    )
                    return
                if path == "/api/state":
                    payload = json.dumps(
                        owner.snapshot(),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    self._send(
                        HTTPStatus.OK,
                        payload,
                        "application/json; charset=utf-8",
                    )
                    return
                self._send(
                    HTTPStatus.NOT_FOUND,
                    b'{"error":"not found"}',
                    "application/json; charset=utf-8",
                )

            def do_POST(self) -> None:  # noqa: N802
                if urlparse(self.path).path != "/api/shutdown":
                    self._send(
                        HTTPStatus.NOT_FOUND,
                        b'{"error":"not found"}',
                        "application/json; charset=utf-8",
                    )
                    return
                accepted, message = owner.request_shutdown()
                body = json.dumps(
                    {"accepted": accepted, "message": message},
                    ensure_ascii=False,
                ).encode("utf-8")
                self._send(
                    HTTPStatus.OK if accepted else HTTPStatus.CONFLICT,
                    body,
                    "application/json; charset=utf-8",
                )

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.thread = threading.Thread(
            target=self.httpd.serve_forever,
            name="training-dashboard-http",
            daemon=True,
        )

    @property
    def address(self) -> tuple[str, int]:
        host, port = self.httpd.server_address[:2]
        return str(host), int(port)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)
