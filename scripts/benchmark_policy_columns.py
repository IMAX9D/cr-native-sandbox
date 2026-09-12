"""CPU-only IPC/assembly benchmark; NOT an expert-model or games/day benchmark.

Uses native-shaped synthetic observations and a tiny deterministic Actor in a
separate server process. Both formats keep the same masks, hidden transitions,
message counts and warmup. Real GPU/libg throughput requires a separate test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import statistics
import sys
import tempfile
import threading
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
from expert_v1.training_v1.model import ExpertPolicyConfig, ExpertPolicyOutput
from expert_selfplay_v1.actions import ExpertActionMasks
from expert_selfplay_v1.batched_policy import BatchedPolicyService, PolicyRequest
from expert_selfplay_v1.remote_policy import RemotePolicyClient, RemotePolicyServer, _request_to_wire
from expert_selfplay_v1.policy_columns import encode_columns
from multiprocessing.reduction import ForkingPickler


class TinyBenchmarkActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = ExpertPolicyConfig(grid_channels=8, public_scalar_size=64,
            card_vocab_size=512, ability_vocab_size=8, max_ability_slots=2,
            hidden_size=8, card_embedding_size=8, spatial_size=8)

    def initial_hidden(self, batch_size, *, device):
        return (torch.zeros(1,batch_size,8,device=device), torch.zeros(1,batch_size,8,device=device))

    def forward_sequence(self, *, public_scalars, hidden, **unused):
        b,t=public_scalars.shape[:2]
        return ExpertPolicyOutput(torch.ones(b,t), torch.zeros(b,t,2), torch.zeros(b,t,4),
            torch.zeros(b,t,4,576), torch.zeros(b,t,2), torch.zeros(b,t,2,576),
            (hidden[0]+1, hidden[1]+1))


def fixture(client_id, count):
    rows=[]
    generator=torch.Generator().manual_seed(1234+client_id)
    for i in range(count):
        n=8+i%17
        inputs={
            'grid':torch.rand(1,8,32,18,generator=generator),
            'public_scalars':torch.rand(1,64,generator=generator),
            'own_deck_tokens':torch.arange(1,9).unsqueeze(0),
            'hand_tokens':torch.arange(1,5).unsqueeze(0),
            'next_card_token':torch.tensor([5]),
            'revealed_enemy_tokens':torch.arange(1,9).unsqueeze(0),
            'ability_tokens':torch.tensor([[1,2]]),
            'delta_ticks':torch.tensor([5.]),
            'entity_tokens':torch.arange(1,n+1).unsqueeze(0),
            'entity_positions':torch.arange(n).unsqueeze(0),
            'entity_relations':torch.zeros(1,n,dtype=torch.long),
            'entity_numeric':torch.ones(1,n,3),
            'entity_mask':torch.ones(1,n,dtype=torch.bool),
        }
        masks=ExpertActionMasks(torch.tensor([True,False]),torch.ones(4,dtype=torch.bool),
            torch.ones(4,576,dtype=torch.bool),torch.zeros(2,dtype=torch.bool),
            torch.zeros(2,576,dtype=torch.bool),torch.zeros(2,dtype=torch.bool))
        rows.append(PolicyRequest((client_id,i),i%2,('a' if i%2 else 'b')*64,inputs,masks,
                                  delta_ticks=5,capture_pre_action_hidden=False))
    return rows


def serve(address, family, key, notifications, microbatch_ms):
    torch.set_num_threads(1)
    service=BatchedPolicyService(device='cpu',deterministic=True)
    for digest in ('a'*64,'b'*64): service.register_actor(TinyBenchmarkActor(),actor_sha256=digest)
    server=RemotePolicyServer(service,address,connection_family=family,authkey=key,
                              microbatch_seconds=microbatch_ms/1000,max_actor_rows=256)
    def ready(): notifications.put({'type':'server-ready','ready':server.ready_event.wait(15)})
    threading.Thread(target=ready,daemon=True).start()
    server.serve_forever()


def collect(address, family, key, format_name, client_id, rows, warmup, iterations, barrier, notifications):
    torch.set_num_threads(1)
    try:
        requests=fixture(client_id,rows)
        with RemotePolicyClient(address,connection_family=family,authkey=key,wire_format=format_name) as client:
            for _ in range(warmup): client.act(requests)
            notifications.put({'type':'client-ready','client':client_id})
            barrier.wait(timeout=30)
            latencies=[]
            for _ in range(iterations):
                started=time.perf_counter()
                actions=client.act(requests)
                latencies.append(time.perf_counter()-started)
                if len(actions)!=rows:raise RuntimeError('lost action')
            notifications.put({'type':'client-done','client':client_id,'latencies':latencies})
    except BaseException:
        notifications.put({'type':'error','traceback':traceback.format_exc()})
        raise


def trial(args, format_name):
    context=mp.get_context('spawn')
    family='AF_PIPE' if os.name=='nt' else 'AF_UNIX'
    key=os.urandom(32)
    processes=[]
    with tempfile.TemporaryDirectory(prefix='cr-columnar-bench-') as tmp:
        address=(r'\\.\pipe\cr-columns-'+uuid.uuid4().hex if os.name=='nt' else str(Path(tmp)/'policy.sock'))
        events=context.Queue()
        barrier=context.Barrier(args.clients+1)
        server=context.Process(target=serve,args=(address,family,key,events,args.microbatch_ms));processes.append(server)
        server.start()
        monitor=None
        try:
            first=events.get(timeout=25)
            if first!={'type':'server-ready','ready':True}:raise RuntimeError(first)
            monitor=RemotePolicyClient(address,connection_family=family,authkey=key)
            for i in range(args.clients):
                p=context.Process(target=collect,args=(address,family,key,format_name,i,args.rows,
                                                       args.warmup,args.iterations,barrier,events))
                processes.append(p);p.start()
            for _ in range(args.clients):
                event=events.get(timeout=30)
                if event['type']!='client-ready':raise RuntimeError(event)
            before=monitor.server_metrics()
            started=time.perf_counter();barrier.wait(timeout=30)
            results=[]
            for _ in range(args.clients):
                event=events.get(timeout=60)
                if event['type']!='client-done':raise RuntimeError(event)
                results.append(event)
            elapsed=time.perf_counter()-started
            after=monitor.server_metrics()
            monitor.shutdown_server();monitor.close();monitor=None
            for p in processes:
                p.join(5)
                if p.exitcode!=0:raise RuntimeError(f'benchmark child exit {p.exitcode}')
            latencies=[v for r in results for v in r['latencies']]
            return {'wire_format':format_name,'wall_seconds':elapsed,
                    'requests':args.clients*args.iterations,
                    'actor_rows':args.clients*args.iterations*args.rows,
                    'actor_rows_per_second':args.clients*args.iterations*args.rows/elapsed,
                    'rpc_p50_ms':float(np.percentile(latencies,50))*1000,
                    'rpc_p95_ms':float(np.percentile(latencies,95))*1000,
                    'server_delta':{k:after[k]-before.get(k,0) for k in after
                                    if k not in ('mean_microbatch_rows','max_microbatch_rows')}}
        finally:
            if monitor is not None:
                # Do not block cleanup on a failed peer's close handshake.
                monitor._connection.close()
            for p in processes:
                if p.is_alive():p.terminate()
            for p in processes:p.join(5)
            events.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows',type=int,default=12)
    parser.add_argument('--clients',type=int,default=4)
    parser.add_argument('--iterations',type=int,default=60)
    parser.add_argument('--warmup',type=int,default=10)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--microbatch-ms',type=float,default=0,
                        help='0 isolates IPC/assembly; avoids platform timer/coalescing confounds')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not (1<=args.rows<=64 and 1<=args.clients<=4 and args.rows*args.clients<=256
            and 1<=args.iterations<=1000 and 0<=args.warmup<=100 and 1<=args.repeats<=10
            and 0<=args.microbatch_ms<=10):
        raise ValueError('benchmark limits are invalid')
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(1)
    rows=fixture(0,args.rows)
    legacy={'requests':[_request_to_wire(r) for r in rows]}
    columnar={'packet':encode_columns(rows)}
    result={'kind':'cpu_policy_ipc_assembly_benchmark_v2','expert_model':False,'libg':False,
            'gpu_used':False,'config':vars(args)|{'output':str(args.output)},
            'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
                'expert_selfplay_v1/policy_columns.py','expert_selfplay_v1/remote_policy.py',
                'expert_selfplay_v1/batched_policy.py','scripts/benchmark_policy_columns.py')},
            'python_version':sys.version,'torch_version':torch.__version__,
            'serialized_request_bytes':{'rows-v1':len(ForkingPickler.dumps(legacy)),
                                        'columns-v2':len(ForkingPickler.dumps(columnar))},'trials':[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for repeat in range(args.repeats):
        order=('rows-v1','columns-v2') if repeat%2==0 else ('columns-v2','rows-v1')
        for mode in order:
            row=trial(args,mode);row['repeat']=repeat
            result['trials'].append(row)
            args.output.write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps({k:row[k] for k in ('wire_format','repeat','actor_rows_per_second','rpc_p95_ms')}),flush=True)
    result['median_actor_rows_per_second']={mode:statistics.median(r['actor_rows_per_second'] for r in result['trials'] if r['wire_format']==mode)
                                            for mode in ('rows-v1','columns-v2')}
    a,b=(result['median_actor_rows_per_second'][mode] for mode in ('rows-v1','columns-v2'))
    result['local_proxy_ratio']=b/a
    result['completed']=True
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'median':result['median_actor_rows_per_second'],'ratio':b/a,
                      'not_a_cloud_training_result':True}),flush=True)


if __name__=='__main__':main()
