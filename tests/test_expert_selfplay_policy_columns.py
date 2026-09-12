from dataclasses import replace
from multiprocessing.reduction import ForkingPickler
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import threading
import unittest
import uuid
from unittest.mock import patch

import numpy as np
import torch

from expert_selfplay_v1.batched_policy import BatchedPolicyService
from expert_selfplay_v1.policy_columns import MAX_ROWS, decode_columns, encode_columns
from expert_selfplay_v1.remote_policy import RemotePolicyClient, RemotePolicyServer, _Pending
from expert_v1.training_v1.model import ExpertPolicyConfig, RecurrentExpertPolicy
from scripts.benchmark_policy_columns import fixture
from test_expert_selfplay_batched_policy import CountingActor, request


def make_rows(step=0):
    rows = []
    for i, digest in enumerate(('a'*64, 'b'*64, 'a'*64, 'b'*64)):
        n = 1 + (step + i) % 5
        row = request(i, digest, scalar=(-1 if i % 2 else 1), reset=step == 0,
                      extra_inputs={'entity_tokens':torch.arange(1,n+1),
                                    'entity_positions':torch.arange(n),
                                    'entity_relations':torch.zeros(n,dtype=torch.long),
                                    'entity_numeric':torch.ones(n,3),
                                    'entity_mask':torch.ones(n,dtype=torch.bool)})
        rows.append(replace(row, side=i%2, capture_pre_action_hidden=step%3==0))
    return rows


def service(seed=123, deterministic=True):
    result = BatchedPolicyService(device='cpu', seed=seed, deterministic=deterministic)
    for digest in ('a'*64,'b'*64):
        result.register_actor(CountingActor(),actor_sha256=digest)
    return result


def _server_process(address, family, ready_pipe, authkey):
    torch.set_num_threads(1)
    server = RemotePolicyServer(service(),address,connection_family=family,authkey=authkey)
    def notify():
        ready_pipe.send({'ready': server.ready_event.wait(10)})
    threading.Thread(target=notify,daemon=True).start()
    try:
        metrics=server.serve_forever()
        ready_pipe.send({'metrics':metrics})
    finally:
        ready_pipe.close()


class PolicyColumnsTests(unittest.TestCase):
    def test_real_network_forward_probabilities_and_memory_match(self):
        torch.set_num_threads(1)
        torch.manual_seed(123)
        config=ExpertPolicyConfig(grid_channels=8,public_scalar_size=64,card_vocab_size=512,
                                 ability_vocab_size=8,max_ability_slots=2,card_embedding_size=8,
                                 spatial_size=8,hidden_size=16,lambda_initial=10.0)
        first,second=RecurrentExpertPolicy(config),RecurrentExpertPolicy(config)
        second.load_state_dict(first.state_dict())
        a=BatchedPolicyService(device='cpu',deterministic=True)
        b=BatchedPolicyService(device='cpu',deterministic=True)
        a.register_actor(first,actor_sha256='a'*64)
        b.register_actor(second,actor_sha256='a'*64)
        rows=[replace(row,actor_sha256='a'*64) for row in fixture(0,3)]
        for _ in range(3):
            expected=a.act(rows)
            batches,_=decode_columns(encode_columns(rows))
            actual=b.act_tensor_batches(batches)
            self.assertEqual(expected,actual)
            for left,right in zip(a.last_pre_action_hidden_batch(expected),
                                  b.last_pre_action_hidden_batch(actual),strict=True):
                for h1,h2 in zip(left,right,strict=True):torch.testing.assert_close(h1,h2)

    def test_identical_stochastic_actions_and_recurrent_states_across_turns(self):
        baseline, columnar = service(deterministic=False), service(deterministic=False)
        for step in range(12):
            rows=make_rows(step)
            packet=ForkingPickler.loads(ForkingPickler.dumps(encode_columns(rows)))
            batches, metadata=decode_columns(packet)
            expected=baseline.act(rows)
            actual=columnar.act_tensor_batches(batches)
            self.assertEqual(expected, actual)
            self.assertEqual([r.worker_id for r in metadata],[r.worker_id for r in rows])
            for first,second in zip(baseline.last_pre_action_hidden_batch(expected),
                                    columnar.last_pre_action_hidden_batch(actual),strict=True):
                for a,b in zip(first,second,strict=True): torch.testing.assert_close(a,b)
            if step==5:
                self.assertEqual(baseline.reset_episode(2),columnar.reset_episode(2))

    def test_coalesced_packets_preserve_original_mixed_actor_order(self):
        rows=make_rows()
        first,_=decode_columns(encode_columns(rows[:2]))
        second,_=decode_columns(encode_columns(rows[2:]),offset=2)
        self.assertEqual(service().act(rows),service().act_tensor_batches(first+second))

    def test_no_row_tensor_reconstruction_on_fast_inference_path(self):
        instance=service()
        batches,_=decode_columns(encode_columns(make_rows()))
        with patch.object(instance,'_batch_inputs',side_effect=AssertionError('row reconstruction')), \
                patch.object(instance,'_batch_masks',side_effect=AssertionError('row masks')):
            self.assertEqual(len(instance.act_tensor_batches(batches)),4)

    def test_bad_packets_rejected_before_inference(self):
        for mutate in (
            lambda p:p.update(row_count=MAX_ROWS+1),
            lambda p:p.update(tensor_bytes=0),
            lambda p:p['groups'][0]['indices'].__setitem__(0,99),
            lambda p:p['groups'][0].update(actor_sha256='c'*64),
            lambda p:p['groups'][0]['masks'].update(cards=np.ones((2,4),dtype=np.float32)),
            lambda p:p['rows'][0].update(capture_pre_action_hidden='false'),
        ):
            packet=encode_columns(make_rows())
            mutate(packet)
            with self.assertRaises((ValueError,TypeError)): decode_columns(packet)

    def test_duplicate_rows_and_partial_none_rejected(self):
        rows=make_rows()
        with self.assertRaises(ValueError):encode_columns([rows[0],rows[0]])
        inputs=dict(rows[2].actor_inputs);inputs['entity_mask']=None
        rows[2]=replace(rows[2],actor_inputs=inputs)
        with self.assertRaises(ValueError):encode_columns(rows)

    def test_byte_size_and_array_count_are_columnar(self):
        packet=encode_columns(make_rows())
        arrays=[v for g in packet['groups'] for columns in (g['actor_inputs'],g['masks'])
                for v in columns.values() if v is not None]
        self.assertEqual(packet['tensor_bytes'],sum(a.nbytes for a in arrays))
        self.assertEqual(len(arrays),2*(7+6))
        self.assertEqual(decode_columns(encode_columns([])),([],[]))

    def test_delta_mismatch_rejected_without_advancing_memory(self):
        instance=service()
        packet=encode_columns(make_rows())
        packet['groups'][0]['actor_inputs']['delta_ticks'][0,0]=8
        batches,_=decode_columns(packet)
        with self.assertRaisesRegex(ValueError,'delta_ticks'):instance.act_tensor_batches(batches)
        self.assertEqual(instance.recurrent_state_count,0)

    def test_invalid_card_capacity_fails_before_forward(self):
        instance=service()
        packet=encode_columns(make_rows())
        group=packet['groups'][0]
        group['masks']['cards']=np.ones((2,3),dtype=np.bool_)
        group['masks']['positions']=np.ones((2,3,576),dtype=np.bool_)
        packet['tensor_bytes']=sum(v.nbytes for g in packet['groups']
            for columns in (g['actor_inputs'],g['masks']) for v in columns.values() if v is not None)
        batches,_=decode_columns(packet)
        with self.assertRaisesRegex(ValueError,'dimensions'):instance.act_tensor_batches(batches)
        self.assertEqual(instance.forward_calls,0)
        self.assertEqual(instance.recurrent_state_count,0)

    def test_remote_coalescing_and_capacity_gate(self):
        rows=make_rows()
        server=RemotePolicyServer(service(),'unused')
        pending=[_Pending('act_columns_v2',{'packet':encode_columns(part),'deterministic':True})
                 for part in (rows[:2],rows[2:])]
        server._act(pending)
        self.assertEqual([a for p in pending for a in p.result['actions']],service().act(rows))
        self.assertEqual(server.metrics['columnar_client_act_calls'],2)
        limited=RemotePolicyServer(service(),'unused',max_actor_rows=2)
        with self.assertRaisesRegex(ValueError,'capacity'):limited._act(pending)
        self.assertEqual(limited.service.forward_calls,0)

    def test_real_separate_process_transport_and_graceful_shutdown(self):
        context=mp.get_context('spawn')
        authkey=os.urandom(32)
        with tempfile.TemporaryDirectory(prefix='cr-policy-columns-') as temp:
            family='AF_PIPE' if os.name=='nt' else 'AF_UNIX'
            address=(r'\\.\pipe\cr-policy-'+uuid.uuid4().hex if os.name=='nt' else str(Path(temp)/'policy.sock'))
            parent,child=context.Pipe()
            process=context.Process(target=_server_process,args=(address,family,child,authkey))
            process.start();child.close()
            try:
                self.assertTrue(parent.poll(15),'server did not become ready')
                self.assertTrue(parent.recv()['ready'])
                with RemotePolicyClient(address,connection_family=family,authkey=authkey,
                                        wire_format='columns-v2') as client:
                    expected=service()
                    for step in range(3):self.assertEqual(client.act(make_rows(step)),expected.act(make_rows(step)))
                    metrics=client.server_metrics()
                    self.assertEqual(metrics['columnar_client_act_calls'],3)
                    self.assertGreater(metrics['request_wire_bytes'],0)
                    client.shutdown_server()
                process.join(5)
                self.assertEqual(process.exitcode,0)
            finally:
                if process.is_alive():process.terminate();process.join(5)
                parent.close()


if __name__=='__main__':unittest.main()
