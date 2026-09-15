"""Tensorize public supplements without passing opaque routing IDs to a model."""
from dataclasses import dataclass, fields

import torch
from torch import Tensor

from .semantics import (DYNAMIC_SCALES, EFFECT_KINDS, ENTITY_KINDS, GROUP_KINDS,
    RELATION_KINDS, PublicSemantics,PLANE_NAMES)
from .elite_tensorizer import EliteBatch,encode_elite


@dataclass(frozen=True)
class SemanticBatch:
    dynamic: Tensor
    kinds: Tensor
    geometry_kind: Tensor  # 0 unknown, 1 point, 2 circle, 3 rectangle
    radii: Tensor  # tiles
    half_extents: Tensor  # tiles
    membership: Tensor  # [B,G,N]
    group_kinds: Tensor
    group_mask: Tensor
    entity_groups: Tensor
    effect_types: Tensor
    effect_sources: Tensor
    effect_targets: Tensor
    effect_features: Tensor
    effect_mask: Tensor
    edge_sources: Tensor
    edge_targets: Tensor
    edge_types: Tensor
    edge_mask: Tensor
    enemy_tokens: Tensor
    enemy_mask: Tensor
    command_tokens: Tensor
    command_kinds: Tensor
    command_features: Tensor
    command_pending: Tensor
    command_mask: Tensor
    candidate_sources: Tensor
    planes: Tensor
    elite: EliteBatch | None = None

    def to(self, device):
        return SemanticBatch(**{f.name:getattr(self,f.name).to(device) if getattr(self,f.name) is not None else None for f in fields(self)})


def encode_semantics(frames, vocabulary, config, tensors) -> SemanticBatch:
    b,n=tensors['entity_tokens'].shape
    infos=[]
    for frame in frames:
        s=frame.semantics or PublicSemantics(frame.episode_uid,frame.view.actor_side,frame.view.tick)
        if (s.episode_uid,s.actor_side,s.tick)!=(frame.episode_uid,frame.view.actor_side,frame.view.tick):
            raise ValueError('semantic frame belongs to another episode/side/tick')
        entities=list(frame.view.entities)+list(frame.view.towers)
        keys={e.key:j for j,e in enumerate(entities)}
        if len(keys)!=len(entities): raise ValueError('ambiguous entity/tower routing key')
        details={e.key:e for e in s.entities}
        if any(key not in keys for key in details): raise ValueError('semantic detail for absent entity')
        groups,indices,kinds={},[],[]
        for j,e in enumerate(frame.view.entities):
            detail=details.get(e.key)
            # Unidentified deployment membership stays singleton; no invented causality.
            key=(e.relation,detail.group_kind,detail.group_id) if detail and detail.group_id is not None else ('singleton',j)
            if key not in groups:
                groups[key]=len(groups)
                kinds.append(GROUP_KINDS.index(detail.group_kind) if detail and detail.group_id is not None else GROUP_KINDS.index('singleton'))
            indices.append(groups[key])
        revealed=set(s.revealed_enemy_cards)
        revealed.update(e.card_id for e in frame.events if e.side!=frame.view.actor_side and e.card_id is not None)
        revealed.update(e.card_id for e in frame.view.entities if e.relation==1 and e.card_id>0 and vocabulary.token(e.card_id)>1)
        revealed=sorted({vocabulary.base(card) for card in revealed})
        infos.append((s,entities,keys,details,indices,kinds,revealed))
    g=max(1,max(len(x[5]) for x in infos)); e=max(1,max(len(x[0].effects) for x in infos))
    r=max(1,max(len(x[0].relations) for x in infos)); k=max(1,max(len(x[6]) for x in infos))
    a=max(1,max(len(x[0].previous)+len(x[0].pending) for x in infos))
    if g>config.max_groups or e>config.max_effects or r>config.max_edges or k>config.max_revealed_cards or any(len(x[0].pending)>config.max_pending for x in infos):
        raise OverflowError('rich semantic capacity exceeded; no truncation')
    def z(*shape):return torch.zeros(shape)
    def l(*shape):return torch.zeros(shape,dtype=torch.long)
    def m(*shape):return torch.zeros(shape,dtype=torch.bool)
    result=SemanticBatch(dynamic=z(b,n,2*len(DYNAMIC_SCALES)),kinds=l(b,n),geometry_kind=l(b,n),
        radii=z(b,n),half_extents=z(b,n,2),membership=m(b,g,n),group_kinds=l(b,g),group_mask=m(b,g),
        entity_groups=torch.full((b,n),-1,dtype=torch.long),effect_types=l(b,e),effect_sources=l(b,e),
        effect_targets=l(b,e),effect_features=z(b,e,8),effect_mask=m(b,e),edge_sources=l(b,r),
        edge_targets=l(b,r),edge_types=l(b,r),edge_mask=m(b,r),enemy_tokens=l(b,k),enemy_mask=m(b,k),
        command_tokens=l(b,a),command_kinds=l(b,a),command_features=z(b,a,9),command_pending=m(b,a),
        command_mask=m(b,a),candidate_sources=torch.full_like(tensors['candidate_tokens'],-1),planes=z(b,len(PLANE_NAMES),32,18))
    result.planes[:,14:18]=tensors['grid']
    for i,(s,entities,keys,details,indices,kinds,revealed) in enumerate(infos):
        for j,entity in enumerate(entities):
            is_tower=j>=len(frames[i].view.entities)
            detail=details.get(entity.key)
            kind=('king' if entity.role==0 else 'princess') if is_tower else detail.kind if detail else 'unknown'
            if is_tower and detail and detail.kind not in ('unknown',kind):
                raise ValueError('tower kind contradicts public native state')
            result.kinds[i,j]=ENTITY_KINDS.index(kind)
            if not is_tower:
                result.membership[i,indices[j],j]=True
                result.entity_groups[i,j]=indices[j]
            if detail:
                values=dict(detail.values)
                for f,(name,scale) in enumerate(DYNAMIC_SCALES.items()):
                    if name in values:
                        result.dynamic[i,j,2*f]=values[name]/scale
                        result.dynamic[i,j,2*f+1]=1
                if detail.radius_tiles is not None:
                    result.radii[i,j]=detail.radius_tiles
                    result.geometry_kind[i,j]=2 if detail.radius_tiles>0 else 1
                if detail.half_extent_tiles is not None:
                    result.half_extents[i,j]=torch.tensor(detail.half_extent_tiles)
                    result.geometry_kind[i,j]=3 if any(detail.half_extent_tiles) else 1
            rel=int(tensors['entity_relations'][i,j])
            # Keep off-board flight entities in tokens/relations/groups, but do
            # not paint their centres onto the opposite edge via negative
            # indexing or clamp them onto an in-arena cell.
            position=tensors['entity_positions'][i,j]
            if not bool(((position>=0)&(position<=1)).all()):continue
            col=min(17,int(tensors['entity_positions'][i,j,0]*18)); row=min(31,int(tensors['entity_positions'][i,j,1]*32))
            values=dict(detail.values) if detail else {}
            base=rel*7
            if kind in ('building','king','princess'):
                if detail and detail.half_extent_tiles is not None and any(detail.half_extent_tiles):
                    cx,cy=entity.x/1000,entity.y/1000;hx,hy=detail.half_extent_tiles
                    yy,xx=torch.meshgrid(torch.arange(32)+.5,torch.arange(18)+.5,indexing='ij')
                    footprint=(xx>=cx-hx)&(xx<cx+hx)&(yy>=cy-hy)&(yy<cy+hy)
                    result.planes[i,base+2]+=footprint
                else:result.planes[i,base+2,row,col]+=1
            elif kind=='projectile':result.planes[i,base+3,row,col]+=1
            elif 'flying' in values:result.planes[i,base+int(values['flying']>0),row,col]+=1
            if entity.hp>=0:result.planes[i,base+4,row,col]+=entity.hp/10000
            result.planes[i,base+5,row,col]+=values.get('shield',0.)/10000
            if detail and detail.threat is not None:result.planes[i,base+6,row,col]+=detail.threat/5000
            result.planes[i,18+rel,row,col]+=float(kind=='area')
        result.group_kinds[i,:len(kinds)]=torch.tensor(kinds,dtype=torch.long)
        result.group_mask[i,:len(kinds)]=True
        for j,ef in enumerate(s.effects):
            if ef.target_key not in keys: raise ValueError('effect target absent from public frame')
            result.effect_types[i,j]=EFFECT_KINDS.index(ef.kind)
            result.effect_sources[i,j]=vocabulary.token(ef.source_card)
            result.effect_targets[i,j]=keys[ef.target_key]
            result.effect_mask[i,j]=True
            result.effect_features[i,j,0]=(s.tick-ef.observed_tick)/1200
            result.effect_features[i,j,1]=float(ef.kind!='unknown')
            for col,(v,scale) in enumerate(((ef.magnitude,5000.),(ef.remaining_ms,30000.),(ef.stacks,16.))):
                if v is not None:result.effect_features[i,j,2+2*col:4+2*col]=torch.tensor([v/scale,1.])
        for j,edge in enumerate(s.relations):
            if edge.source_key not in keys or edge.target_key not in keys: raise ValueError('relation references absent entity')
            result.edge_sources[i,j]=keys[edge.source_key];result.edge_targets[i,j]=keys[edge.target_key]
            result.edge_types[i,j]=RELATION_KINDS.index(edge.kind);result.edge_mask[i,j]=True
        result.enemy_tokens[i,:len(revealed)]=torch.tensor([vocabulary.token(card) for card in revealed],dtype=torch.long)
        result.enemy_mask[i,:len(revealed)]=True
        for j,cmd in enumerate((*s.previous,*s.pending)):
            result.command_tokens[i,j]=vocabulary.token(cmd.card_id)
            result.command_kinds[i,j]=('unknown','play','ability').index(cmd.kind)
            result.command_pending[i,j]=j>=len(s.previous);result.command_mask[i,j]=True
            result.command_features[i,j]=torch.tensor([(s.tick-cmd.issued_tick)/1200,(s.tick-cmd.observed_tick)/1200,
                float(cmd.card_id is not None),*(cmd.target or (0.,0.)),float(cmd.target is not None),
                (cmd.cost or 0.)/10,float(cmd.cost is not None),float(cmd.kind!='unknown')])
        for j,candidate in enumerate(frames[i].candidates):
            if candidate.kind=='ability':
                if candidate.source_entity not in keys: raise ValueError('candidate semantic source absent')
                result.candidate_sources[i,j]=keys[candidate.source_entity]
    from dataclasses import replace
    return replace(result,elite=encode_elite(frames,infos,result,tensors,vocabulary,config))
