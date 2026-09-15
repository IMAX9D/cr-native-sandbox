"""Rich semantic, hierarchical and spatial encoders inspired by FirstLight V4.

Independent local implementation. Schema and weights are not reference-compatible.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from .semantics import (STATIC_SCALES,DYNAMIC_SCALES,EFFECT_KINDS,ENTITY_KINDS,
    GROUP_KINDS,RELATION_KINDS,PLANE_NAMES)
from .native_graph import card_roots


def mlp(source,target):
    return nn.Sequential(nn.Linear(source,target),nn.SiLU(),nn.LayerNorm(target))


def masked_mean(values,mask):
    return (values*mask[...,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)


class CardSemanticEncoder(nn.Module):
    def __init__(self,vocabulary,catalog,width,enabled=True):
        super().__init__()
        self.enabled=enabled
        self.identity=nn.Embedding(vocabulary.size,width,padding_idx=0)
        self.static=mlp(2*len(STATIC_SCALES),width)
        self.operation_kind=nn.Embedding(len(EFFECT_KINDS),width)
        self.operation=mlp(2*width+7,width)
        self.norm=nn.LayerNorm(width)
        self.id_gate=nn.Linear(width,1)
        nn.init.zeros_(self.id_gate.weight);nn.init.constant_(self.id_gate.bias,-2.)
        roots,root_mask=card_roots(vocabulary)
        self.register_buffer('native_roots',roots);self.register_buffer('native_root_mask',root_mask)
        profiles={p.card_id:p for p in catalog.profiles}
        slots=max(1,max((len(p.operations) for p in profiles.values()),default=0))
        features=torch.zeros(vocabulary.size,2*len(STATIC_SCALES))
        op_types=torch.zeros(vocabulary.size,slots,dtype=torch.long)
        op_cards=torch.ones_like(op_types)
        op_values=torch.zeros(vocabulary.size,slots,7)
        op_mask=torch.zeros(vocabulary.size,slots,dtype=torch.bool)
        for card,profile in profiles.items():
            token=vocabulary.token(card)
            if token<2:raise ValueError('mechanic profile outside vocabulary')
            values=dict(profile.values)
            for j,(name,scale) in enumerate(STATIC_SCALES.items()):
                if name in values:features[token,2*j:2*j+2]=torch.tensor([values[name]/scale,1.])
            for j,op in enumerate(profile.operations):
                op_types[token,j]=EFFECT_KINDS.index(op.kind);op_cards[token,j]=vocabulary.token(op.produced_card)
                for k,(value,scale) in enumerate(((op.magnitude,5000.),(op.duration_ms,30000.),(op.radius_tiles,5.))):
                    if value is not None:op_values[token,j,2*k:2*k+2]=torch.tensor([value/scale,1.])
                op_values[token,j,6]=float(op.produced_card is not None)
                op_mask[token,j]=True
        for name,value in dict(features=features,op_types=op_types,op_cards=op_cards,op_values=op_values,op_mask=op_mask).items():
            self.register_buffer(name,value)

    def memory(self,native_memory=None):
        identity=self.identity.weight
        if not self.enabled:return identity
        operations=self.operation(torch.cat((self.operation_kind(self.op_types),self.identity(self.op_cards),self.op_values),-1))
        pooled=(operations*self.op_mask[...,None]).sum(1)/self.op_mask.sum(1,keepdim=True).clamp_min(1)
        semantics=self.static(self.features)+pooled
        if native_memory is not None:
            semantics=semantics+(native_memory[self.native_roots]*self.native_root_mask[:,:,None]).sum(1)/self.native_root_mask.sum(1,keepdim=True).clamp_min(1)
        result=self.norm(semantics+torch.sigmoid(self.id_gate(semantics))*identity)
        return result*(torch.arange(len(result),device=result.device)!=0)[:,None]

    def forward(self,tokens):
        return self.memory()[tokens]


class ResidualSpatialBlock(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.layers=nn.Sequential(nn.Conv2d(channels,channels,3,padding=1),nn.SiLU(),nn.Conv2d(channels,channels,3,padding=1))
    def forward(self,x):return torch.nn.functional.silu(x+self.layers(x))


class SpatialEncoder(nn.Module):
    def __init__(self,config):
        super().__init__()
        c=config
        self.scatter_value=nn.Linear(c.width,c.learned_scatter_channels)
        self.input=nn.Conv2d(len(PLANE_NAMES)+c.learned_scatter_channels+3,c.spatial_channels,3,padding=1)
        self.blocks=nn.Sequential(*(ResidualSpatialBlock(c.spatial_channels) for _ in range(c.spatial_blocks)))
        self.summary=mlp(2*c.spatial_channels,c.width//2)
        y,x=torch.meshgrid(torch.linspace(-1,1,32),torch.linspace(-1,1,18),indexing='ij')
        self.register_buffer('coordinates',torch.stack((x,y,torch.sign(y)))[None])
        cy,cx=torch.meshgrid(torch.arange(32)+.5,torch.arange(18)+.5,indexing='ij')
        self.register_buffer('centers',torch.stack((cx,cy),-1).reshape(576,2))

    def scatter(self,values,positions,mask,geometry_kind,radii,half_extents):
        """Bilinear points; circle/rectangle footprints. All paths differentiable."""
        b,n,d=values.shape
        xy=positions*positions.new_tensor([18.,32.])
        xy=torch.minimum(xy.clamp_min(0),xy.new_tensor([17.,31.]))
        floor=xy.floor().long();fraction=xy-floor
        point=mask & (geometry_kind<2) & ((positions>=0)&(positions<=1)).all(-1)
        out=values.new_zeros(b,576,d)
        for dx,dy in ((0,0),(1,0),(0,1),(1,1)):
            x=(floor[...,0]+dx).clamp_max(17);y=(floor[...,1]+dy).clamp_max(31)
            weight=(fraction[...,0] if dx else 1-fraction[...,0])*(fraction[...,1] if dy else 1-fraction[...,1])
            out=out.scatter_add(1,(y*18+x)[...,None].expand(-1,-1,d),values*(weight*point)[...,None])
        # Bound temporary geometry by one board per entity, not channels per entity.
        delta=self.centers[None,None]-positions[:,:,None]*positions.new_tensor([18.,32.])
        circle=delta.square().sum(-1)<=radii[:,:,None].square()
        rectangle=(delta.abs()<=half_extents[:,:,None]).all(-1)
        coverage=((circle & (geometry_kind==2)[:,:,None]) | (rectangle & (geometry_kind==3)[:,:,None])) & mask[:,:,None]
        out=out+torch.einsum('bnk,bnd->bkd',coverage.to(values.dtype),values)
        return out.transpose(1,2).reshape(b,d,32,18)

    def forward(self,entities,batch):
        s=batch.semantic
        scattered=self.scatter(self.scatter_value(entities),batch.entity_positions,batch.entity_mask,s.geometry_kind,s.radii,s.half_extents)
        features=self.blocks(torch.nn.functional.silu(self.input(torch.cat((s.planes,scattered,self.coordinates.expand(batch.batch_size,-1,-1,-1)),1))))
        summary=self.summary(torch.cat((features.mean((2,3)),features.amax((2,3))),-1))
        return features,summary


class RichEntityEncoder(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=config;d=config.width
        self.dynamic=mlp(2*len(DYNAMIC_SCALES),d)
        self.kind=nn.Embedding(len(ENTITY_KINDS),d)
        self.geometry=mlp(7,d)
        self.effect_kind=nn.Embedding(len(EFFECT_KINDS),d)
        self.effect=mlp(2*d+8,d)
        self.norm=nn.LayerNorm(d)
        self.local_queries=nn.Parameter(torch.randn(config.local_pool_slots,d)/math.sqrt(d))
        self.group_type=nn.Embedding(len(GROUP_KINDS),d)
        self.group_stats=mlp(10,d)
        self.group_merge=mlp((config.local_pool_slots+2)*d,d)
        self.card_runtime=mlp(4,d)
        self.command_kind=nn.Embedding(3,d)
        self.command_encoder=mlp(2*d+9,d)
        self.pending_kind=nn.Embedding(2,d)
        self.command_pool=mlp(2*d,d//2)

    def entities(self,base,batch,card_memory):
        s=batch.semantic
        geometry=torch.cat((s.radii[:,:,None]/5,s.half_extents/5,torch.nn.functional.one_hot(s.geometry_kind,4)),-1).to(base.dtype)
        output=base+self.dynamic(s.dynamic)+self.kind(s.kinds)+self.geometry(geometry)
        if self.config.use_effects:
            effects=self.effect(torch.cat((self.effect_kind(s.effect_types),card_memory[s.effect_sources],s.effect_features),-1))*s.effect_mask[...,None]
            aggregates=torch.zeros_like(base).scatter_add(1,s.effect_targets[...,None].expand_as(effects),effects)
            output=output+aggregates
        return self.norm(output)*batch.entity_mask[...,None]

    def groups(self,entities,batch):
        s=batch.semantic
        membership=s.membership
        if not self.config.use_groups:
            membership=torch.eye(entities.shape[1],dtype=torch.bool,device=entities.device)[None].expand(batch.batch_size,-1,-1)
            membership=membership & batch.entity_mask[:,:,None] & (batch.entity_features[:,:,6]>0)[:,:,None]
        mask=membership.any(-1)
        score=torch.einsum('sd,bnd->bsn',self.local_queries,entities)/math.sqrt(entities.shape[-1])
        score=score[:,None].expand(-1,membership.shape[1],-1,-1).masked_fill(~membership[:,:,None],-torch.inf)
        score=torch.where(mask[:,:,None,None],score,torch.zeros_like(score))
        weights=score.softmax(-1)*membership[:,:,None]
        slots=torch.einsum('bgsn,bnd->bgsd',weights,entities).flatten(2)
        count=membership.sum(-1,keepdim=True).clamp_min(1)
        pos=torch.einsum('bgn,bnd->bgd',membership.to(entities.dtype),batch.entity_positions)/count
        hp=batch.entity_features[:,:,2]
        mean_hp=torch.einsum('bgn,bn->bg',membership.to(entities.dtype),hp)/count.squeeze(-1)
        half=torch.where((s.geometry_kind==2)[:,:,None],s.radii[:,:,None].expand(-1,-1,2),s.half_extents)
        half=half/half.new_tensor([18.,32.])
        extent_min=(batch.entity_positions-half)[:,None].masked_fill(~membership[:,:,:,None],torch.inf).amin(2)
        extent_max=(batch.entity_positions+half)[:,None].masked_fill(~membership[:,:,:,None],-torch.inf).amax(2)
        extent_min=torch.where(mask[:,:,None],extent_min,0.);extent_max=torch.where(mask[:,:,None],extent_max,0.)
        variance=torch.einsum('bgn,bgnd->bgd',membership.to(entities.dtype),(batch.entity_positions[:,None]-pos[:,:,None]).square())/count
        own=torch.einsum('bgn,bn->bg',membership.to(entities.dtype),batch.entity_relations.to(entities.dtype))/count.squeeze(-1)
        stats=torch.cat((pos,extent_max-extent_min,variance,count/16,mean_hp[:,:,None],own[:,:,None],mask[:,:,None]),-1)
        kinds=s.group_kinds if self.config.use_groups else torch.full_like(mask,5,dtype=torch.long)
        groups=self.group_merge(torch.cat((slots,self.group_stats(stats),self.group_type(kinds)),-1))*mask[:,:,None]
        return groups,mask,pos,torch.cat((extent_min,extent_max),-1),membership

    def commands(self,batch,card_memory):
        s=batch.semantic
        if not self.config.use_command_memory:
            return card_memory.new_zeros(batch.batch_size,self.config.width//2)
        features=self.command_encoder(torch.cat((card_memory[s.command_tokens],self.command_kind(s.command_kinds),s.command_features),-1))+self.pending_kind(s.command_pending.long())
        return self.command_pool(torch.cat((masked_mean(features,s.command_mask & ~s.command_pending),masked_mean(features,s.command_mask & s.command_pending)),-1))


def spatial_pairs(positions,extents,spatial_mask):
    delta=positions[:,:,None]-positions[:,None,:]
    distance=delta.square().sum(-1,keepdim=True).clamp_min(1e-12).sqrt()
    overlap=(torch.minimum(extents[:,:,None,2:],extents[:,None,:,2:])-torch.maximum(extents[:,:,None,:2],extents[:,None,:,:2])).clamp_min(0)
    both=spatial_mask[:,:,None]&spatial_mask[:,None,:]
    return torch.cat((delta,delta.abs(),distance,overlap,both[:,:,:,None]),-1)*both[:,:,:,None]


class RelationBlock(nn.Module):
    def __init__(self,width,heads):
        super().__init__();self.heads=heads
        self.norm1=nn.LayerNorm(width);self.norm2=nn.LayerNorm(width)
        self.attention=nn.MultiheadAttention(width,heads,dropout=0,batch_first=True)
        self.spatial_bias=nn.Sequential(nn.Linear(8,32),nn.SiLU(),nn.Linear(32,heads))
        self.edge_bias=nn.Embedding(len(RELATION_KINDS),heads,padding_idx=0)
        self.ffn=nn.Sequential(nn.Linear(width,4*width),nn.SiLU(),nn.Linear(4*width,width))

    def forward(self,tokens,mask,pairs,edges=None):
        b,n,_=tokens.shape
        bias=self.spatial_bias(pairs)
        if edges is not None:
            sources,targets,kinds,valid=edges
            indices=torch.arange(b,device=tokens.device)[:,None]*n*n+sources*n+targets
            edge_values=self.edge_bias(kinds)*valid[:,:,None]
            added=bias.new_zeros(b*n*n,self.heads).index_add(0,indices.flatten(),edge_values.reshape(-1,self.heads))
            bias=bias+added.reshape_as(bias)
        bias=bias.permute(0,3,1,2).masked_fill(~mask[:,None,None,:],-torch.inf).reshape(b*self.heads,n,n)
        normalized=self.norm1(tokens)
        attended,_=self.attention(normalized,normalized,normalized,attn_mask=bias,need_weights=False)
        tokens=tokens+attended
        return (tokens+self.ffn(self.norm2(tokens)))*mask[:,:,None]
