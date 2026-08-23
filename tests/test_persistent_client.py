from __future__ import annotations

import json
import socket
import threading
import unittest

from native_core.client import JsonLineClient


class _Server:
    def __init__(self, handler):
        self.handler = handler
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(8)
        self.port = self.listener.getsockname()[1]
        self.accepts = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        self.handler(self)

    def accept(self):
        connection, _address = self.listener.accept()
        self.accepts += 1
        return connection

    def close(self):
        self.listener.close()
        self.thread.join(timeout=2)


class PersistentClientTests(unittest.TestCase):
    def test_reuses_one_connection_and_serializes_threads(self):
        received = []

        def handler(server):
            with server.accept() as connection, connection.makefile("rwb") as stream:
                for _ in range(40):
                    request = json.loads(stream.readline())
                    received.append(request["id"])
                    stream.write(json.dumps({"ok": True, "id": request["id"]}).encode() + b"\n")
                    stream.flush()

        server = _Server(handler)
        client = JsonLineClient(port=server.port, timeout=2)
        results = []
        lock = threading.Lock()

        def call(value):
            response = client.request({"op": "ping", "id": value})
            with lock:
                results.append(response["id"])

        threads = [threading.Thread(target=call, args=(index,)) for index in range(40)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        client.close()
        server.close()
        self.assertEqual(sorted(results), list(range(40)))
        self.assertEqual(sorted(received), list(range(40)))
        self.assertEqual(server.accepts, 1)

    def test_idempotent_request_reconnects_once(self):
        def handler(server):
            for index in range(2):
                with server.accept() as connection, connection.makefile("rwb") as stream:
                    request = json.loads(stream.readline())
                    stream.write(json.dumps({"ok": True, "index": index, "op": request["op"]}).encode() + b"\n")
                    stream.flush()

        server = _Server(handler)
        profile = {}
        client = JsonLineClient(port=server.port, timeout=2, profile=profile)
        self.assertEqual(client.request({"op": "ping"})["index"], 0)
        self.assertEqual(client.request({"op": "ping"})["index"], 1)
        client.close()
        server.close()
        self.assertEqual(server.accepts, 2)
        self.assertEqual(profile["rpc_reconnects"], 1.0)

    def test_mutating_request_is_not_replayed_after_ambiguous_close(self):
        mutations = []

        def handler(server):
            with server.accept() as connection, connection.makefile("rwb") as stream:
                request = json.loads(stream.readline())
                mutations.append(request["op"])
                # Deliberately close after applying the mutation, before reply.

        server = _Server(handler)
        client = JsonLineClient(port=server.port, timeout=2)
        with self.assertRaises(ConnectionError):
            client.request({"op": "reset", "replay": {}})
        client.close()
        server.close()
        self.assertEqual(mutations, ["reset"])
        self.assertEqual(server.accepts, 1)


if __name__ == "__main__":
    unittest.main()
