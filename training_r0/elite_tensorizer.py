"""Full FirstLight feature bank with separate value/known tensors and coverage.

No anonymous filler dimensions: names are frozen from the reference contract.
Only observed values and justified local derivations are marked known.
"""
from dataclasses import dataclass,fields
import torch
from torch import Tensor
from .feature_contract import CHILD_FEATURES,TOWER_FEATURES,GROUP_FEATURES,CARD_FEATURES,MATCH_FEATURES,COMBAT_FEATURES,COMBAT_KINDS,ABILITY_FEATURES
from .native_graph import node_id
from .lifecycle import LIFECYCLE_FEATURE_NAMES

REQUIRED_CAPTURE=('entity_runtime','archetypes','tower_runtime','groups','combat_events','card_runtime','match_runtime','commands','effects','relations','geometry','lifecycle')


@dataclass(frozen=True)
class EliteBatch:
    child:Tensor
    tower:Tensor
    groups:Tensor
    own_cards:Tensor
    enemy_cards:Tensor
    match:Tensor
    archetypes:Tensor
    tower_troops:Tensor
    combat_types:Tensor
    combat_features:Tensor
    combat_sources:Tensor
    combat_targets:Tensor
    combat_cards:Tensor
    combat_forms:Tensor
    combat_sides:Tensor
    combat_mask:Tensor
    command_effective:Tensor
    command_forms:Tensor
    command_abilities:Tensor
    command_extra:Tensor
    command_offsets:Tensor
    command_steps:Tensor
    candidate_effective:Tensor
    candidate_abilities:Tensor
    candidate_ability_features:Tensor
    capture_complete:Tensor
    lifecycle:Tensor
    effect_nodes:Tensor
    lifecycle_mask:Tensor  # Data admission/routing only; not a learnable feature.

    def to(self,device):return EliteBatch(**{f.name:getattr(self,f.name).to(device) for f in fields(self)})


def fill(destination,names,values):
    for j,key in enumerate(names):
        if key in values:destination[2*j:2*j+2]=torch.tensor([values[key],1.])


def encode_elite(frames,infos,sem,tensors,vocabulary,config):
    b,n=tensors['entity_tokens'].shape;g=sem.group_mask.shape[1];k=sem.enemy_tokens.shape[1];a=sem.command_mask.shape[1];c=tensors['candidate_tokens'].shape[1]
    events=max(1,max(len(s.combat_events) for s,*_ in infos))
    if events>config.max_history:raise OverflowError('combat event capacity exceeded; do not discard events')
    z=lambda *shape:torch.zeros(shape)
    l=lambda *shape:torch.zeros(shape,dtype=torch.long)
    m=lambda *shape:torch.zeros(shape,dtype=torch.bool)
    result=EliteBatch(child=z(b,n,2*len(CHILD_FEATURES)),tower=z(b,n,2*len(TOWER_FEATURES)),groups=z(b,g,2*len(GROUP_FEATURES)),
        own_cards=z(b,8,2*len(CARD_FEATURES)),enemy_cards=z(b,k,2*len(CARD_FEATURES)),match=z(b,2*len(MATCH_FEATURES)),
        archetypes=torch.ones(b,n,dtype=torch.long),tower_troops=l(b,n),combat_types=l(b,events),combat_features=z(b,events,2*len(COMBAT_FEATURES)),
        combat_sources=torch.full((b,events),-1,dtype=torch.long),combat_targets=torch.full((b,events),-1,dtype=torch.long),
        combat_cards=l(b,events),combat_forms=torch.full((b,events),4,dtype=torch.long),combat_sides=l(b,events),combat_mask=m(b,events),
        command_effective=l(b,a),command_forms=torch.full((b,a),4,dtype=torch.long),command_abilities=torch.ones(b,a,dtype=torch.long),
        command_extra=z(b,a,7),command_offsets=l(b,a),command_steps=l(b,a),candidate_effective=l(b,c),candidate_abilities=torch.ones(b,c,dtype=torch.long),
        candidate_ability_features=z(b,c,2*len(ABILITY_FEATURES)),capture_complete=m(b),lifecycle=z(b,n,len(LIFECYCLE_FEATURE_NAMES)),effect_nodes=torch.ones_like(sem.effect_types),lifecycle_mask=m(b,n))
    dynamic_map={'velocity_x':('velocity_x_over_10',10.),'velocity_y':('velocity_y_over_10',10.),'shield':('shield_over_10000',10000.),
        'age_ms':('age_over_60000ms',60000.),'attack_cooldown_ms':('attack_cooldown_remaining_over_5000ms',5000.),
        'attack_phase_ms':('attack_phase_remaining_over_5000ms',5000.),'deployment_ms':('deployment_remaining_over_5000ms',5000.),
        'projectile_damage':('projectile_damage_over_5000',5000.),'targetable':('visibility_targetable',1.),'attack_windup':('attack_windup',1.),
        'attack_charging':('attack_charging',1.),'damage_multiplier':('attack_damage_multiplier_over_5',5.)}
    for i,(s,entities,keys,details,indices,kinds,revealed) in enumerate(infos):
        result.capture_complete[i]=set(REQUIRED_CAPTURE)<=set(s.capture_capabilities) and all(e.key in details for e in entities)
        for j,effect in enumerate(s.effects):result.effect_nodes[i,j]=node_id(effect.effect_name)
        for j,entity in enumerate(entities):
            is_tower=j>=len(frames[i].view.entities);detail=details.get(entity.key);values=dict(detail.values) if detail else {}
            child={};tower={}
            if entity.max_hp>0 and entity.hp>=0:
                child.update(hitpoints_ratio=entity.hp/entity.max_hp,hitpoints_over_10000=entity.hp/10000,max_hitpoints_over_10000=entity.max_hp/10000)
                tower.update(hitpoints_ratio=entity.hp/entity.max_hp,hitpoints_over_5000=entity.hp/5000,max_hitpoints_over_5000=entity.max_hp/5000)
            child.update(x_over_board_width=entity.x/18000,y_over_board_height=entity.y/32000)
            if not is_tower and entity.relation==0:child['is_ability_source']=float(entity.own_ability_slot>0)
            for name,(target,scale) in dynamic_map.items():
                if name in values:child[target]=values[name]/scale
            if detail:
                if not is_tower and (node_id(detail.archetype)<=1 or detail.group_id is None or detail.kind=='unknown'):
                    result.capture_complete[i]=False
                if detail.damage is not None:
                    damage=detail.damage
                    if (damage.episode_uid,damage.as_of_tick)!=(s.episode_uid,s.tick):raise ValueError('future/stale lifecycle summary or wrong episode')
                    if not damage.damage_complete and not config.allow_incomplete_capture:raise ValueError('incomplete damage ledger cannot become a production training input')
                    result.lifecycle_mask[i,j]=damage.damage_complete
                    for f,value in enumerate(damage.values):
                        result.lifecycle[i,j,f]=value/10000.
                result.archetypes[i,j]=node_id(detail.archetype)
                result.tower_troops[i,j]=('unknown','princess','cannoneer','dagger_duchess','royal_chef').index(detail.tower_troop)
                if detail.radius_tiles is not None:child['projectile_radius_over_5_tiles']=detail.radius_tiles/5
                if 'shield' in values and entity.max_hp>0:tower['shield_over_max_hitpoints']=values['shield']/entity.max_hp
                if 'king_activated' in values:tower['king_activated']=values['king_activated']
                child.update(detail.child_features);tower.update(detail.tower_features)
            fill(result.child[i,j],CHILD_FEATURES,child);fill(result.tower[i,j],TOWER_FEATURES,tower)
        for group in range(len(kinds)):
            js=sem.membership[i,group].nonzero().flatten().tolist();es=[entities[j] for j in js]
            hp=[e.hp for e in es if e.hp>=0];ratio=[e.hp/e.max_hp for e in es if e.max_hp>0 and e.hp>=0]
            xy=tensors['entity_positions'][i,js];variance=xy.var(0,unbiased=False)
            stats=dict(member_count_over_16=len(es)/16,dropped_child_count_over_16=0.,
                x_span_tiles=float((xy[:,0].max()-xy[:,0].min())*18),y_span_tiles=float((xy[:,1].max()-xy[:,1].min())*32),
                x_variance_over_board_width_squared=float(variance[0]),y_variance_over_board_height_squared=float(variance[1]),
                xy_covariance_over_board_area=float(((xy[:,0]-xy[:,0].mean())*(xy[:,1]-xy[:,1].mean())).mean()))
            if len(hp)==len(es):stats['summed_hitpoints_over_10000']=sum(hp)/10000
            if len(ratio)==len(es):stats.update(mean_hitpoints_ratio=sum(ratio)/len(ratio),min_hitpoints_ratio=min(ratio),max_hitpoints_ratio=max(ratio))
            ds=[dict(details[e.key].values) if e.key in details else {} for e in es]
            if all('shield' in d for d in ds):stats['summed_shield_over_10000']=sum(d['shield'] for d in ds)/10000
            if all('velocity_x' in d and 'velocity_y' in d for d in ds):
                velocity=torch.tensor([[d['velocity_x'],d['velocity_y']] for d in ds]);speed=velocity.square().sum(-1).sqrt()
                stats.update(mean_velocity_x_over_10=float(velocity[:,0].mean()/10),mean_velocity_y_over_10=float(velocity[:,1].mean()/10),
                    mean_speed_over_10=float(speed.mean()/10),velocity_dispersion_over_100=float(velocity.var(0,unbiased=False).sum()/100))
            if all(e.relation==0 for e in es):stats['has_ability_source']=float(any(e.own_ability_slot>0 for e in es))
            if 'relations' in s.capture_capabilities:
                members={e.key for e in es};stats['has_visible_target']=float(any(r.source_key in members and r.kind=='targets' for r in s.relations))
                stats['has_causal_parent']=float(any(r.source_key in members and r.kind=='spawned_by' for r in s.relations))
            if all(e.key in details and details[e.key].kind!='unknown' for e in es):
                projectiles=[d for e,d in zip(es,ds) if details[e.key].kind=='projectile']
                damage=[d['projectile_damage'] for d in projectiles if 'projectile_damage' in d]
                stats.update(projectile_count_over_16=len(projectiles)/16,projectile_damage_coverage=len(damage)/max(1,len(projectiles)),known_projectile_damage_sum_over_5000=sum(damage)/5000)
            first=es[0];detail=details.get(first.key)
            for supplied in s.group_runtime:
                if detail and (supplied.relation,supplied.group_id)==(first.relation,detail.group_id):stats.update(supplied.features)
            fill(result.groups[i,group],GROUP_FEATURES,stats)
        runtime={(row.relation,vocabulary.base(row.card_id)):dict(row.features) for row in s.card_runtime}
        for j,card in enumerate(frames[i].own_deck):
            token=vocabulary.token(card)
            values=dict(in_hand=float(bool((tensors['hand_tokens'][i]==token).any())),is_next_card=float(tensors['next_tokens'][i]==token),revealed=1.)
            values.update(runtime.get((0,vocabulary.base(card)),{}));fill(result.own_cards[i,j],CARD_FEATURES,values)
        for j,card in enumerate(revealed):
            values={'revealed':1.};extra=runtime.get((1,card),{})
            if any(key in extra for key in ('in_hand','is_next_card','ability_cooldown_remaining_over_30000ms')):raise ValueError('private enemy card runtime forbidden')
            values.update(extra);fill(result.enemy_cards[i,j],CARD_FEATURES,values)
        view=frames[i].view
        match=dict(own_elixir_over_10=view.own_player.elixir_raw/100000,own_crowns_over_3=view.episode.own_crowns/3,enemy_crowns_over_3=view.episode.enemy_crowns/3,
            visible_entity_count_over_capacity=len(view.entities)/config.max_entities,current_event_group_count_over_capacity=len(s.combat_events)/config.max_history,
            group_overflow_over_capacity=0.,child_overflow_over_capacity=0.,event_overflow_over_capacity=0.,candidate_count_over_capacity=len(frames[i].candidates)/config.max_candidates,actor_is_true_red=float(view.actor_side==1))
        match.update(s.match_features);fill(result.match[i],MATCH_FEATURES,match)
        for j,event in enumerate(s.combat_events):
            result.combat_types[i,j]=COMBAT_KINDS.index(event.kind);result.combat_mask[i,j]=True
            result.combat_cards[i,j]=vocabulary.token(event.source_card);result.combat_forms[i,j]=event.source_form if event.source_form is not None else 4
            result.combat_sources[i,j]=keys.get(event.source_key,-1);result.combat_targets[i,j]=keys.get(event.target_key,-1)
            result.combat_sides[i,j]=int(event.side!=view.actor_side)
            values=dict(latest_age_over_decision_ticks=(s.tick-event.observed_tick)/5,has_source=float(event.source_key is not None),has_target=float(event.target_key is not None),source_form_known=float(event.source_form is not None))
            values.update(event.features);fill(result.combat_features[i,j],COMBAT_FEATURES,values)
        for j,cmd in enumerate((*s.previous,*s.pending)):
            result.command_effective[i,j]=vocabulary.token(cmd.effective_card)
            result.command_forms[i,j]=4 if cmd.effective_form is None else cmd.effective_form
            result.command_abilities[i,j]=node_id(cmd.ability_name)
            result.command_extra[i,j]=torch.tensor([*(cmd.source_position or (0.,0.)),float(cmd.source_position is not None),
                float(cmd.effective_card is not None),float(cmd.effective_form is not None),float(cmd.ability_name is not None),float(cmd.delay_offset_bin is not None)])
            result.command_offsets[i,j]=0 if cmd.delay_offset_bin is None else cmd.delay_offset_bin+1
            result.command_steps[i,j]=0 if cmd.sequence_index is None else cmd.sequence_index+1
        for j,candidate in enumerate(frames[i].candidates):
            result.candidate_effective[i,j]=vocabulary.token(candidate.effective_card_id)
            result.candidate_abilities[i,j]=node_id(candidate.ability_name)
            fill(result.candidate_ability_features[i,j],ABILITY_FEATURES,dict(candidate.ability_features))
    return result
