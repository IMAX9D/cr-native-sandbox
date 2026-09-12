from dataclasses import fields, replace
from pathlib import Path
import unittest

import numpy as np
import torch

from expert_selfplay_v1.actions import ExpertActionMasks
from expert_selfplay_v1.batched_policy import PolicyRequest, SampledPolicyAction
from expert_selfplay_v1.remote_policy import (
    RemotePolicyClient, RemotePolicyError, RemotePolicyServer,
    _Pending, _request_from_wire, _request_to_wire,
)


def request(capture=True):
    masks = ExpertActionMasks(**{
        field.name: torch.ones((1,), dtype=torch.bool)
        for field in fields(ExpertActionMasks)
    })
    return PolicyRequest(7, 0, 'a' * 64, {'public_scalars': torch.ones(1)},
                         masks, capture_pre_action_hidden=capture)


class Service:
    registered_actor_hashes = ('a' * 64,)
    forward_calls = 0

    def __init__(self):
        self.captured = []

    def act(self, requests, deterministic=None):
        self.forward_calls += 1
        result = []
        for req in requests:
            values = {field.name: 0 for field in fields(SampledPolicyAction)}
            values.update(worker_id=req.worker_id, side=req.side,
                          actor_sha256=req.actor_sha256, delta_ticks=req.delta_ticks)
            result.append(SampledPolicyAction(**values))
        return result

    def last_pre_action_hidden_batch(self, actions):
        self.captured.extend(actions)
        return [(torch.full((1, 1, 4), float(self.forward_calls)),
                 torch.full((1, 1, 4), float(self.forward_calls))) for _ in actions]


class RemotePolicyTransferTests(unittest.TestCase):
    def setUp(self):
        self.service = Service()
        self.server = RemotePolicyServer(self.service, Path('unused.sock'), max_pending_requests=3)

    def execute(self, rows):
        pending = _Pending('act', {'requests': [_request_to_wire(r) for r in rows],
                                   'deterministic': True})
        self.server._act([pending])
        self.assertTrue(pending.event.is_set())
        return pending.result

    def test_wire_defaults_preserve_legacy_capture(self):
        wire = _request_to_wire(request(False))
        self.assertFalse(_request_from_wire(wire).capture_pre_action_hidden)
        wire.pop('capture_pre_action_hidden')
        self.assertTrue(_request_from_wire(wire).capture_pre_action_hidden)

    def test_sparse_capture_does_not_skip_policy_dynamics(self):
        result = self.execute([request(False), replace(request(True), worker_id=8)])
        self.assertEqual(len(result['actions']), 2)
        self.assertIsNone(result['pre_action_hidden'][0])
        np.testing.assert_equal(result['pre_action_hidden'][1][0], np.ones((1, 1, 4)))
        self.assertEqual(len(self.service.captured), 1)
        self.assertEqual(self.server.metrics['actor_rows'], 2)
        self.assertEqual(self.server.metrics['hidden_rows_transferred'], 1)
        self.assertEqual(self.server.metrics['hidden_bytes_transferred'], 32)
        self.assertEqual(self.server._queue.maxsize, 3)

    def test_client_clears_stale_anchor_and_checks_missing_capture(self):
        client = RemotePolicyClient.__new__(RemotePolicyClient)
        client._hidden = {}
        client.forward_calls = 0
        client._request = lambda _op, **payload: self.execute(
            [_request_from_wire(row) for row in payload['requests']])
        client.act([request(True)])
        torch.testing.assert_close(client.last_pre_action_hidden(
            actor_sha256='a'*64, worker_id=7, side=0)[0], torch.ones(1, 1, 4))
        client.act([request(False)])
        with self.assertRaises(KeyError):
            client.last_pre_action_hidden(actor_sha256='a'*64, worker_id=7, side=0)
        client._request = lambda *_a, **_kw: self.execute([request(False)])
        with self.assertRaisesRegex(RemotePolicyError, 'requested recurrent anchor'):
            client.act([request(True)])

    def test_reordered_actions_are_rejected(self):
        client = RemotePolicyClient.__new__(RemotePolicyClient)
        client._hidden, client.forward_calls = {}, 0
        client._request = lambda *_a, **_kw: self.execute([replace(request(), worker_id=8)])
        with self.assertRaisesRegex(RemotePolicyError, 'reordered'):
            client.act([request()])

    def test_invalid_queue_limit_is_rejected(self):
        with self.assertRaises(ValueError):
            RemotePolicyServer(self.service, 'unused.sock', max_pending_requests=0)


if __name__ == '__main__':
    unittest.main()
