"""Round3 opt-in variants, importing round2 without changing its files."""
import os,sys,json,atexit
from pathlib import Path
ROOT=Path(__file__).resolve().parent
variant=os.environ.get('CR_FULL_VARIANT','control')
sys.path.insert(0,'/root/autodl-tmp/resident-opt-20260912')
os.environ['CR_OPT_VARIANT']='dense'
import run_variant as prior
sys.path.insert(0,str(ROOT))
import torch
import policy_test

original_load=policy_test.load_release
def load(*args,**kwargs):
    model,encoder,identity=original_load(*args,**kwargs)
    if variant in ('vector','all','graph','buffers','compile','binary','compact','compact-binary','graph-compact'):
        from vector_encoder import NativeObservationEncoder
        contract=json.loads(Path(args[1] if len(args)>1 else kwargs['contract_path']).read_text())
        candidate=NativeObservationEncoder.from_manifest(contract['encoder'])
        if os.environ.get('CR_FULL_VERIFY')=='1':
            original_encoder=candidate.encode_batch
            def checked(frames,**kw):
                a=encoder.encode_batch(frames,**kw);b=original_encoder(frames,**kw)
                for key in a:assert torch.equal(a[key],b[key]),'encoder mismatch '+key
                assert a.ability_entity_keys==b.ability_entity_keys
                assert torch.equal(a.ability_mask,b.ability_mask)
                return b
            candidate.encode_batch=checked
        encoder_out=candidate
    else:encoder_out=encoder
    if variant in ('graph','buffers','all','graph-compact'):
        from cuda_graph import GraphRunner
        graph=GraphRunner(model,mode='buffers' if variant=='buffers' else 'graph')
        reference=model.forward_stream
        def forward(b,h=None,reset=None):
            out,new=graph(b,h,reset)
            if os.environ.get('CR_FULL_VERIFY')=='1':
                expected,eh=reference(b,h,reset=reset)
                for k in out:torch.testing.assert_close(out[k],expected[k],atol=1e-4,rtol=2e-4)
                for x,y in zip(new,eh):torch.testing.assert_close(x,y,atol=1e-4,rtol=2e-4)
            return out,new
        model.forward_stream=forward
        atexit.register(lambda:print('GRAPH_STATS '+json.dumps(dict(counts=dict(graph.stats),failures=graph.failures)),flush=True))
    if variant=='compile':
        from cuda_graph import compute
        # CPU guards remain in eager reference; compile is an isolated benchmark candidate.
        original=model.forward_stream
        compiled=torch.compile(lambda b,h:compute(model,b,h),dynamic=True,mode='reduce-overhead')
        def forward(b,h=None,reset=None):
            if reset is not None or h is None or model.training:return original(b,h,reset=reset)
            if b['frame_mask'].shape[1]!=1 or 'loss_mask' in b:raise ValueError('compiled online path requires all-valid T1')
            shape=(*b['frame_mask'].shape,2,model.config.history_length)
            if any(tuple(b[k].shape)!=shape for k in ('history_card','history_position','history_age','history_mask','history_known')):raise ValueError('history shape mismatch')
            if not bool(b['frame_mask'].all()) or not bool(torch.isfinite(b['history_age']).all()) or bool((b['history_age']<0).any()):raise ValueError('invalid online inputs')
            out,new=compiled(b,h)
            if os.environ.get('CR_FULL_VERIFY')=='1':
                expected,eh=original(b,h)
                for k in out:torch.testing.assert_close(out[k],expected[k],atol=1e-4,rtol=2e-4)
                for x,y in zip(new,eh):torch.testing.assert_close(x,y,atol=1e-4,rtol=2e-4)
            return out,new
        model.forward_stream=forward
    return model,encoder_out,identity
policy_test.load_release=load
if variant in ('binary','compact-binary'):
    from binary_client import BinaryClient
    policy_test.JsonLineClient=BinaryClient
if variant in ('compact','compact-binary','graph-compact'):
    BaseClient=policy_test.JsonLineClient
    class CompactClient(BaseClient):
        def request(self,payload):
            if not getattr(self,'_compact_certified',False):
                identity=super().request({'op':'runtime_identity_v1'})['identity']
                if identity.get('resident_train_schema_v1') is not True:raise ValueError('host does not support requested training observation schema')
                self._compact_certified=True
            if payload.get('op')=='resident_batch':payload={**payload,'observation_schema':'train-v1'}
            result=super().request(payload)
            if payload.get('op')=='resident_batch' and any(r['state'].get('kind')!='libg_native_train_state_v1' for r in result['results']):raise ValueError('unexpected observation schema')
            return result
    policy_test.JsonLineClient=CompactClient
if variant!='control':
    from owned_enrich import enrich
    old=policy_test.ResidentEnv._enrich_state
    def checked_enrich(self,state):
        value=enrich(self,state)
        if os.environ.get('CR_FULL_VERIFY')=='1':assert value==old(self,state),'enrichment changed'
        return value
    policy_test.ResidentEnv._enrich_state=checked_enrich

if __name__=='__main__':policy_test.main()
