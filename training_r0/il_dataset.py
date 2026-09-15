"""Non-pickle, lossless tensor sequence shards for the current R0 interfaces."""
from dataclasses import dataclass,fields,is_dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import torch

from .actions import ActionSequence
from .observation import PublicBatch
from .semantic_tensorizer import SemanticBatch
from .elite_tensorizer import EliteBatch
from .config import DECISION_TICKS

TYPES={c.__name__:c for c in (PublicBatch,SemanticBatch,EliteBatch,ActionSequence)}


@dataclass(frozen=True)
class ILSequence:
    observations:tuple[PublicBatch,...]
    actions:tuple[ActionSequence,...]
    label_known:torch.Tensor
    starts_episode:bool
    source_sha256:str

    def validate(self):
        if not self.observations or len(self.observations)!=len(self.actions):raise ValueError('empty/mismatched IL sequence')
        b=self.observations[0].batch_size;t=len(self.observations)
        if self.label_known.dtype!=torch.bool or self.label_known.shape!=(t,b):raise ValueError('invalid IL label mask')
        if len(self.source_sha256)!=64 or any(c not in '0123456789abcdef' for c in self.source_sha256) or type(self.starts_episode) is not bool:raise ValueError('missing source/episode identity')
        first=self.observations[0]
        if self.starts_episode and any(tick!=0 for tick in first.ticks):raise ValueError('episode-start sequence must begin at Tick 0')
        for i,(obs,action) in enumerate(zip(self.observations,self.actions)):
            if obs.batch_size!=b or obs.schema_hash!=first.schema_hash or obs.episode_uids!=first.episode_uids or obs.sides!=first.sides:
                raise ValueError('IL lane identity/schema changed midsequence')
            if obs.ticks!=tuple(tick+i*DECISION_TICKS for tick in first.ticks):raise ValueError('IL sequence is not temporally contiguous')
            action.validate(b)


def _encode(value,prefix,tensors):
    if isinstance(value,torch.Tensor):
        if value.dtype==torch.bfloat16:raise ValueError('store original float32 inputs, not a reduced-precision cache')
        array=value.detach().cpu().numpy()
        if array.dtype.kind not in 'bifu' or (array.dtype.kind=='f' and not np.isfinite(array).all()):raise ValueError('nonfinite/unsupported tensor')
        tensors[prefix]=array;return {'tensor':prefix}
    if is_dataclass(value):
        if type(value).__name__ not in TYPES:raise ValueError('unsupported serialized class')
        return {'class':type(value).__name__,'fields':{f.name:_encode(getattr(value,f.name),prefix+'.'+f.name,tensors) for f in fields(value)}}
    if isinstance(value,tuple):return {'tuple':[_encode(v,prefix+f'.{i}',tensors) for i,v in enumerate(value)]}
    if value is None or isinstance(value,(str,int,float,bool)):return value
    raise TypeError(type(value))


def _decode(tree,tensors):
    if not isinstance(tree,dict):return tree
    if 'tensor' in tree:return torch.from_numpy(tensors[tree['tensor']].copy())
    if 'tuple' in tree:return tuple(_decode(x,tensors) for x in tree['tuple'])
    if tree.get('class') not in TYPES:raise ValueError('unrecognized serialized class')
    cls=TYPES[tree['class']]
    if set(tree['fields'])!={f.name for f in fields(cls)}:raise ValueError('serialized fields do not match current R0')
    return cls(**{k:_decode(v,tensors) for k,v in tree['fields'].items()})


def save_sequence(sequence,path):
    sequence.validate();path=Path(path)
    if path.exists():raise FileExistsError(path)
    trees=[];by_name={}
    for observation,action in zip(sequence.observations,sequence.actions):
        tensors={};tree=_encode((observation,action),'sample',tensors);trees.append(tree)
        for name,array in tensors.items():by_name.setdefault(name,[]).append(array)
    arrays={};layout={}
    for index,(name,values) in enumerate(sorted(by_name.items())):
        if len(values)!=len(trees) or len({v.dtype.str for v in values})!=1:raise ValueError('tensor schema changes inside sequence')
        key='column_'+str(index)
        arrays[key]=np.concatenate([v.reshape(-1) for v in values])
        shapes=[list(v.shape) for v in values];sizes=[v.size for v in values]
        layout[name]=dict(key=key,shapes=shapes,offsets=np.cumsum([0,*sizes]).tolist())
    metadata=dict(schema='r0-il-sequence.v1',trees=trees,layout=layout,starts_episode=sequence.starts_episode,source_sha256=sequence.source_sha256)
    arrays['metadata']=np.frombuffer(json.dumps(metadata,separators=(',',':')).encode(),dtype=np.uint8)
    arrays['label_known']=sequence.label_known.cpu().numpy()
    path.parent.mkdir(parents=True,exist_ok=True)
    # Publish only a complete shard; a crash never leaves a final-looking NPZ.
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='wb',prefix=path.name+'.partial-',dir=path.parent,delete=False) as f:
            temporary=Path(f.name);np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
        os.link(temporary,path)  # Atomic and refuses to replace an existing user artifact.
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sequence(path,*,expected_sha256):
    path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected_sha256:raise ValueError('IL shard hash mismatch')
    observations=[];actions=[]
    with np.load(path,allow_pickle=False) as archive:
        meta=json.loads(archive['metadata'].tobytes())
        if meta.get('schema')!='r0-il-sequence.v1':raise ValueError('unsupported IL shard')
        columns={entry['key']:archive[entry['key']] for entry in meta['layout'].values()}
        for i,tree in enumerate(meta['trees']):
            tensors={}
            for name,entry in meta['layout'].items():
                start,end=entry['offsets'][i:i+2];shape=entry['shapes'][i];flat=columns[entry['key']]
                if not 0<=start<=end<=flat.size or int(np.prod(shape))!=end-start:raise ValueError('IL tensor extent mismatch')
                tensors[name]=flat[start:end].reshape(shape)
            obs,action=_decode(tree,tensors);observations.append(obs);actions.append(action)
        sequence=ILSequence(tuple(observations),tuple(actions),torch.from_numpy(archive['label_known'].copy()),meta['starts_episode'],meta['source_sha256'])
    sequence.validate();return sequence
