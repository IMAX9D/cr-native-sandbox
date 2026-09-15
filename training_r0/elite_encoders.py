"""Full feature-bank encoders and structured event/command/candidate conditioning."""
import math
import torch
from torch import nn
from .feature_contract import CHILD_FEATURES,TOWER_FEATURES,GROUP_FEATURES,CARD_FEATURES,MATCH_FEATURES,COMBAT_FEATURES,COMBAT_KINDS,ABILITY_FEATURES
from .lifecycle import LIFECYCLE_FEATURE_NAMES


def mlp(source,middle,target):return nn.Sequential(nn.Linear(source,middle),nn.SiLU(),nn.Linear(middle,target),nn.LayerNorm(target))


class LearnedSetPool(nn.Module):
    def __init__(self,width,out,slots=2):
        super().__init__();self.query=nn.Parameter(torch.randn(slots,width)*.02)
        self.output=mlp(slots*width,out,out)
    def forward(self,items,mask):
        logits=torch.einsum('bnd,sd->bsn',items,self.query)/math.sqrt(items.shape[-1])
        logits=logits.masked_fill(~mask[:,None],-torch.inf)
        logits=torch.where(mask.any(-1)[:,None,None],logits,torch.zeros_like(logits))
        weights=logits.softmax(-1)*mask[:,None]
        return self.output(torch.einsum('bsn,bnd->bsd',weights,items).flatten(1))


class EliteEncoders(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=config;d=config.width;h=d//2
        self.child=mlp(2*len(CHILD_FEATURES),d,d)
        self.lifecycle=mlp(len(LIFECYCLE_FEATURE_NAMES),d,d)
        self.archetype=nn.Linear(d,d)
        self.archetype_gate=nn.Linear(2*d,1)
        nn.init.zeros_(self.archetype_gate.weight);nn.init.constant_(self.archetype_gate.bias,-2.)
        self.tower=mlp(2*len(TOWER_FEATURES),d,d)
        self.tower_troop=nn.Embedding(5,d)
        self.effect_native=nn.Linear(d,d)
        self.effect_gate=nn.Linear(d,1)
        nn.init.zeros_(self.effect_gate.weight);nn.init.constant_(self.effect_gate.bias,-2.)
        self.group=mlp(2*len(GROUP_FEATURES),h,d)
        self.card_runtime=mlp(2*len(CARD_FEATURES),h,d)
        self.match=mlp(2*len(MATCH_FEATURES),h,h)
        self.event_kind=nn.Embedding(len(COMBAT_KINDS),h)
        self.event_owner=nn.Embedding(2,h)
        self.event_card=nn.Linear(d,h)
        self.event_form=nn.Embedding(5,h)
        self.event_source=nn.Linear(d,h)
        self.event_target=nn.Linear(d,h)
        self.event_numeric=mlp(2*len(COMBAT_FEATURES),h,h)
        self.event_norm=nn.LayerNorm(h)
        self.event_pool=LearnedSetPool(h,h,2)
        self.event_merge=mlp(2*h,h,h)
        self.command_visible=nn.Linear(d,h)
        self.command_effective=nn.Linear(d,h)
        self.command_form=nn.Embedding(5,h)
        self.command_ability=nn.Linear(d,h)
        self.command_offset=nn.Embedding(6,32)
        self.command_step=nn.Embedding(3,32)
        self.command_extra=mlp(7,h,h)
        self.command_target=mlp(3,h,h)
        self.command_detail=mlp(5*h+64,h,h)
        self.previous_pool=LearnedSetPool(h,h)
        self.pending_pool=LearnedSetPool(h,h)
        self.previous_count=nn.Embedding(3,h)
        self.commands_merge=mlp(3*h,h,h)
        self.candidate_effective=nn.Linear(d,d)
        self.candidate_ability=mlp(d+2*len(ABILITY_FEATURES),d,d)

    def entities(self,base,batch,native_memory):
        e=batch.semantic.elite
        child=self.child(e.child)+self.lifecycle(e.lifecycle)*e.lifecycle_mask[:,:,None]
        archetype=self.archetype(native_memory[e.archetypes])
        enriched=base+child+torch.sigmoid(self.archetype_gate(torch.cat((base,child),-1)))*archetype
        if self.config.use_effects:
            effects=self.effect_native(native_memory[e.effect_nodes])
            effects=effects*torch.sigmoid(self.effect_gate(effects))*batch.semantic.effect_mask[:,:,None]
            enriched=enriched+torch.zeros_like(enriched).scatter_add(1,batch.semantic.effect_targets[:,:,None].expand_as(effects),effects)
        towers=self.tower(e.tower)+self.tower_troop(e.tower_troops)
        tower_mask=(batch.entity_features[:,:,6]==0)&batch.entity_mask
        return (enriched+torch.where(tower_mask[:,:,None],towers,torch.zeros_like(towers)))*batch.entity_mask[:,:,None]

    def events(self,batch,card_memory,tokens,entity_to_token,previous_summary):
        e=batch.semantic.elite
        def gather(indices):
            mapped=entity_to_token.gather(1,indices.clamp_min(0))
            return tokens.gather(1,mapped[:,:,None].expand(-1,-1,tokens.shape[-1]))*(indices>=0)[:,:,None]
        values=self.event_norm(self.event_kind(e.combat_types)+self.event_owner(e.combat_sides)+self.event_card(card_memory[e.combat_cards])+
            self.event_form(e.combat_forms)+self.event_source(gather(e.combat_sources))+self.event_target(gather(e.combat_targets))+self.event_numeric(e.combat_features))
        return self.event_merge(torch.cat((previous_summary,self.event_pool(values,e.combat_mask)),-1))

    def commands(self,batch,card_memory,native_memory,summary):
        s=batch.semantic;e=s.elite
        values=self.command_detail(torch.cat((self.command_visible(card_memory[s.command_tokens]),
            self.command_effective(card_memory[e.command_effective])+self.command_form(e.command_forms),self.command_ability(native_memory[e.command_abilities]),
            self.command_extra(e.command_extra),self.command_target(s.command_features[:,:,3:6]),
            self.command_offset(e.command_offsets),self.command_step(e.command_steps)),-1))
        previous=s.command_mask&~s.command_pending;pending=s.command_mask&s.command_pending
        return self.commands_merge(torch.cat((summary,self.previous_pool(values,previous)+self.previous_count(previous.sum(-1).clamp_max(2)),self.pending_pool(values,pending)),-1))


class LogitHead(nn.Sequential):
    def __init__(self,source,width,outputs):super().__init__(nn.Linear(source,width),nn.SiLU(),nn.Linear(width,outputs))
    @property
    def weight(self):return self[-1].weight
    @property
    def bias(self):return self[-1].bias
