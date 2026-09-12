"""Bounded shape-bucket CUDA graphs; explicit eager fallback and original guards."""
import collections, types
import torch
from hokoff_model.model import Policy

def compute(model,b,state):
    x=model.encode(b);rec,h=model.lstm(x,state);out=Policy.heads(model,rec,b)
    if model.config.spatial_skip_channels:
        B,T=b['frame_mask'].shape;grid=b['grid'].reshape(B*T,model.config.grid_channels,32,18)
        if model.config.spatial_type_dim:grid=torch.cat((grid,model.spatial_type_grid(b)[:,0].to(grid)),1)
        context=out['context'].reshape(B*T,-1);hand=model.cards(b['hand_tokens'].reshape(B*T,4))
        correction=model.position_skip(grid,context,hand)
        out['position']=(out['position'].reshape(B*T,4,576)+correction.to(out['position'].dtype)).reshape(B,T,4,576)
    return out,h

class GraphRunner:
    def __init__(self,model,max_graphs=8,mode='graph'):
        self.model=model;self.max_graphs=max_graphs;self.mode=mode;self.cache={};self.fallback=set()
        self.stats=collections.Counter();self.failures=[]
        self.reference=model.forward_stream
    def __call__(self,b,state=None,reset=None):
        if self.model.training or b['frame_mask'].shape[1]!=1 or 'loss_mask' in b or reset is not None:
            self.stats['general_fallback']+=1;return self.reference(b,state,reset=reset)
        if not bool(b['frame_mask'].all()):raise ValueError('online graph refuses padding')
        expected=(*b['frame_mask'].shape,2,self.model.config.history_length)
        for key in ('history_card','history_position','history_age','history_mask','history_known'):
            if tuple(b[key].shape)!=expected:raise ValueError('history shape mismatch')
        if not bool(torch.isfinite(b['history_age']).all()) or bool((b['history_age']<0).any()):raise ValueError('invalid history age')
        if state is None:state=self.model.initial_hidden(b['frame_mask'].shape[0],device='cuda')
        n=b['entity_tokens'].shape[2];bucket=max(8,1<<(max(1,n)-1).bit_length())
        if bucket>2048:raise ValueError('entity capacity exceeded')
        key=(b['frame_mask'].shape[0],bucket,tuple((k,v.dtype,tuple(v.shape[3:]) if k.startswith('entity_') else tuple(v.shape[1:])) for k,v in b.items()))
        if key in self.fallback or (key not in self.cache and len(self.cache)>=self.max_graphs):
            self.stats['capacity_fallback']+=1;return self.reference(b,state)
        if key not in self.cache:
            inputs={}
            for k,v in b.items():
                shape=list(v.shape)
                if k.startswith('entity_'):shape[2]=bucket
                inputs[k]=torch.zeros(shape,device=v.device,dtype=v.dtype)
            h=tuple(torch.empty_like(v) for v in state)
            item=dict(inputs=inputs,state=h,graph=None)
            self._copy(item,b,state)
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            try:
                with torch.cuda.stream(stream),torch.inference_mode():
                    for _ in range(3):compute(self.model,inputs,h)
                torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
                if self.mode=='graph':
                    g=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g,stream=stream),torch.inference_mode():out,nh=compute(self.model,inputs,h)
                    item.update(graph=g,output=out,new_hidden=nh)
                self.cache[key]=item;self.stats['builds']+=1
            except Exception as e:
                self.fallback.add(key);self.failures.append(type(e).__name__+': '+str(e)[:250]);self.stats['capture_fallback']+=1
                return self.reference(b,state)
        item=self.cache[key];self._copy(item,b,state)
        if item['graph'] is not None:
            item['graph'].replay();self.stats['replays']+=1
            # Hidden must be owned: different agents must never share graph output storage.
            return item['output'],tuple(v.clone() for v in item['new_hidden'])
        self.stats['buffer_calls']+=1
        return compute(self.model,item['inputs'],item['state'])
    @staticmethod
    def _copy(item,b,state):
        for key,dst in item['inputs'].items():
            src=b[key]
            if key.startswith('entity_'):
                dst.zero_();dst[:,:,:src.shape[2]].copy_(src)
            else:dst.copy_(src)
        for dst,src in zip(item['state'],state):dst.copy_(src)
