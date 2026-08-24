from __future__ import annotations

import json
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from training.dashboard import TrainingDashboardServer


class TrainingDashboardTests(unittest.TestCase):
    def setUp(self):
        self.running = True

        def request_shutdown():
            if self.running:
                return False, "training running"
            return True, "closing"

        self.server = TrainingDashboardServer(
            host="127.0.0.1",
            port=0,
            snapshot=lambda: {
                "status": "running",
                "native_ticks": 1_033_302,
            },
            request_shutdown=request_shutdown,
        )
        self.server.start()
        host, port = self.server.address
        self.base = f"http://{host}:{port}"

    def tearDown(self):
        self.server.stop()

    def test_serves_dashboard_and_json_without_cache(self):
        with urlopen(self.base + "/", timeout=2) as response:
            html = response.read().decode("utf-8")
            self.assertIn("Self-Play v0.1", html)
            self.assertEqual(response.headers["Cache-Control"], "no-store")
        with urlopen(self.base + "/api/state", timeout=2) as response:
            state = json.loads(response.read())
            self.assertEqual(state["native_ticks"], 1_033_302)

    def test_dashboard_cannot_hide_running_training(self):
        request = Request(self.base + "/api/shutdown", method="POST")
        with self.assertRaises(HTTPError) as captured:
            urlopen(request, timeout=2)
        self.assertEqual(captured.exception.code, 409)
        self.running = False
        with urlopen(request, timeout=2) as response:
            value = json.loads(response.read())
            self.assertTrue(value["accepted"])


if __name__ == "__main__":
    unittest.main()
