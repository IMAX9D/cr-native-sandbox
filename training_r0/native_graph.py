"""Complete typed native-definition graph; learned semantics with gated IDs."""
from functools import lru_cache
from pathlib import Path
import hashlib,json,math
import torch
from torch import nn
from native_core.card_catalog import catalog,form_index


@lru_cache(maxsize=1)
def native_definitions():
    path=Path(__file__).with_name('data')/'native_definitions.json'
    data=path.read_bytes();raw=json.loads(data)
    if raw.get('schema')!='r0-native-definition-graph.v1' or raw.get('runtime_sha256')!='fa6704b83cb9c5b8eecb7b56c9671b834d636a3a6d9ac446e698e1262dc246ba':
        raise ValueError('native definition graph schema/runtime mismatch')
    return raw,hashlib.sha256(data).hexdigest()


@lru_cache(maxsize=1)
def native_node_ids():return {key:i for i,key in enumerate(native_definitions()[0]['node_names'])}


def node_id(name):return native_node_ids().get(name,1) if name is not None else 1


class NativeGraphEncoder(nn.Module):
    def __init__(self,width,layers):
        super().__init__();raw,_=native_definitions();d=width//2;small=max(8,width//8)
        self.node_count=len(raw['node_names'])
        self.field=nn.Embedding(len(raw['field_names']),small,padding_idx=0)
        self.string=nn.Embedding(len(raw['string_values']),small,padding_idx=0)
        self.value_type=nn.Embedding(4,small)
        self.attribute=nn.Sequential(nn.Linear(3*small+3,d),nn.SiLU(),nn.Linear(d,d),nn.LayerNorm(d))
        self.attribute_score=nn.Linear(d,1)
        self.node_identity=nn.Embedding(self.node_count,d,padding_idx=0)
        self.identity_gate=nn.Linear(d,1)
        nn.init.zeros_(self.identity_gate.weight);nn.init.constant_(self.identity_gate.bias,-2.)
        self.node_norm=nn.LayerNorm(d)
        self.edge_projection=nn.Linear(small+1,d)
        self.forward_messages=nn.ModuleList(nn.Linear(d,d) for _ in range(layers))
        self.reverse_messages=nn.ModuleList(nn.Linear(d,d) for _ in range(layers))
        self.norms=nn.ModuleList(nn.LayerNorm(d) for _ in range(layers))
        self.output=nn.Linear(d,width)
        attrs=raw['attributes']
        self.register_buffer('attr_node',torch.tensor([a[0] for a in attrs]))
        self.register_buffer('attr_field',torch.tensor([a[1] for a in attrs]))
        self.register_buffer('attr_type',torch.tensor([a[2] for a in attrs]))
        self.register_buffer('attr_string',torch.tensor([a[3] for a in attrs]))
        self.register_buffer('attr_numeric',torch.tensor([[math.copysign(math.log1p(abs(a[4])),a[4])/16,float(a[2] in (0,1)),a[5]] for a in attrs]))
        edges=torch.tensor(raw['edges'],dtype=torch.long)
        self.register_buffer('edge_source',edges[:,0]);self.register_buffer('edge_target',edges[:,1])
        self.register_buffer('edge_field',edges[:,2]);self.register_buffer('edge_conflict',edges[:,3:].float())

    def forward(self):
        attrs=self.attribute(torch.cat((self.field(self.attr_field),self.string(self.attr_string),self.value_type(self.attr_type),self.attr_numeric),-1))
        scores=self.attribute_score(attrs).squeeze(-1)
        maximum=scores.new_full((self.node_count,),-torch.inf).scatter_reduce(0,self.attr_node,scores,reduce='amax',include_self=True)
        weights=(scores-maximum[self.attr_node]).exp()
        sums=weights.new_zeros(self.node_count).scatter_add(0,self.attr_node,weights)
        nodes=attrs.new_zeros(self.node_count,attrs.shape[-1]).index_add(0,self.attr_node,attrs*(weights/sums[self.attr_node].clamp_min(1e-12))[:,None])
        nodes=self.node_norm(nodes+torch.sigmoid(self.identity_gate(nodes))*self.node_identity.weight)
        edge=self.edge_projection(torch.cat((self.field(self.edge_field),self.edge_conflict),-1))
        count=nodes.new_zeros(self.node_count).index_add(0,self.edge_source,torch.ones_like(self.edge_source,dtype=nodes.dtype)).index_add(0,self.edge_target,torch.ones_like(self.edge_target,dtype=nodes.dtype)).clamp_min(1).sqrt()[:,None]
        for forward,reverse,norm in zip(self.forward_messages,self.reverse_messages,self.norms):
            messages=torch.zeros_like(nodes).index_add(0,self.edge_source,forward(nodes[self.edge_target]+edge)).index_add(0,self.edge_target,reverse(nodes[self.edge_source]+edge))
            nodes=norm(nodes+torch.nn.functional.silu(messages/count))
        return self.output(nodes)*(torch.arange(self.node_count,device=nodes.device)!=0)[:,None]


def card_roots(vocabulary):
    raw,_=native_definitions();lookup=native_node_ids();rows=catalog();forms=form_index();result=[]
    for card in (None,None,*vocabulary.native_ids):
        if card is None:result.append([]);continue
        row=rows.get(vocabulary.base(card),{});form=forms.get(card)
        name=row.get('internal_name') if form is None else form['form_name']
        keys=[f'{kind}.{name}' for kind in ('SPELL_CHARACTER','SPELL_BUILDING','SPELL_OTHER','SPELL_HERO')]
        result.append([lookup[key] for key in keys if key in lookup])
    size=max(1,max(map(len,result)));indices=torch.zeros(len(result),size,dtype=torch.long);mask=torch.zeros_like(indices,dtype=torch.bool)
    for i,values in enumerate(result):
        if values:indices[i,:len(values)]=torch.tensor(values);mask[i,:len(values)]=True
    return indices,mask
