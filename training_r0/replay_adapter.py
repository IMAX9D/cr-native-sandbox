"""Truthful replay-frame adapter and per-field coverage audit for elite-v3.

Missing native mechanisms remain unknown and block production admission. This
adapter never turns a timestamp, HP difference or a capability name into proof.
"""
from __future__ import annotations
from collections import Counter,deque
from dataclasses import replace
from pathlib import Path
import gzip,json,io
from contextlib import contextmanager

from .native_projection import normalize_native_state
from expert_v1.tick_store_v1.deployment_masks import normalize_native_probe,derive_deployment_rows
from .catalog import CardVocabulary
from .config import ModelConfig
from .observation import ObservationBuilder,Candidate
from .semantics import PublicSemantics,EntityDetail,PublicEffect,PublicRelation,CombatEvent,OwnCommand,CardRuntime
from .history import EventHistory,ConfirmedEvent
from .lifecycle import DamageTracker,DamageEvent
from .feature_contract import CHILD_FEATURES,TOWER_FEATURES,GROUP_FEATURES,CARD_FEATURES,MATCH_FEATURES,COMBAT_FEATURES
from .native_graph import native_node_ids
from .replay_facts import state_objects,actor_velocity,projectile_nominal_features,effect_kind,phase_features,nominal_geometry,definition_facts,special_features


class ReplayFeatureAssembler:
    def __init__(self,replay,episode_uid,*,config=None,capture_identity=None):
        self.uid=episode_uid;self.decks=[tuple(dict(card_id=int(c['d']),level=int(c['l'])+1,form_flags=int(c.get('el',0))) for c in replay['battle'][f'deck{s}']['sp']) for s in (0,1)]
        self.replay=replay
        self.capture_identity=dict(capture_identity or {})
        timeline_path=Path(__file__).with_name('data')/'match_timelines.json'
        self.timelines=json.loads(timeline_path.read_text('utf-8')) if timeline_path.is_file() else {}
        self.vocabulary=CardVocabulary.from_native();self.builder=ObservationBuilder(self.vocabulary,config or ModelConfig(allow_incomplete_capture=True))
        self.damage=DamageTracker(episode_uid,continuous_damage_capture=True)
        self.history=EventHistory(episode_uid);self.commands=[];self.last=None;self.pointers={};self.seen={};self.birth_commands={}
        self.coverage=Counter();self.opportunities=Counter();self.issues=Counter();self.samples=0
        self.last_before=None;self.last_telemetry_counts=(0,0);self.telemetry_epoch=None
        self.registrations={};self.native_commands={};self.last_lifecycle_sequence=0;self.lifecycle_available=False
        self.event_window=deque();self.event_ids=set();self.last_inputs={};self.last_capability_reasons={}
        self.capture_failures=set();self.last_damage_sequence=0;self.last_public_objects={}
        self.tower_native={};self.tower_details={};self.last_details={};self.capability_counts=Counter()

    def _public_event(self,event_id,kind,observed_tick,side,source=None,target=None,amount=None,position=None):
        if event_id in self.event_ids:return
        if side not in (0,1):
            self.capture_failures.add('combat_event_owner_unknown');return
        self.event_ids.add(event_id)
        self.event_window.append(dict(id=event_id,kind=kind,tick=observed_tick,side=side,
            source=source,target=target,amount=amount,position=position))

    def command(self,record):
        request=record['request'];response=record['response'].get('result',{})
        actions=request.get('actions',[request.get('action',{})]);results=response.get('actions',[{'result':response}])
        for action,result in zip(actions,results):
            receipt=result.get('result',{})
            if not receipt.get('accepted'):continue
            side=int(action['side']);card=self.decks[side][action['deck_index']]['card_id'] if action.get('type','play')=='play' else None
            sequence=int(receipt.get('native_command_sequence',0))
            self.commands.append(dict(id=f'native-command-{sequence}' if sequence else f'command-{len(self.commands)}',tick=int(receipt.get('tick',record['tick'])),side=side,card_id=card,kind=action.get('type','play'),action=action,receipt=receipt))
            command=self.commands[-1]
            if sequence:self.native_commands[sequence]=command
            target=(action['x'],action['y']) if command['kind']=='play' else None
            self.history.record(ConfirmedEvent(command['id'],command['tick'],command['tick'],side,command['kind'],card,target))

    @staticmethod
    def tower_key(tower,side):
        relation=int(tower['side']!=side)
        if tower['type']=='king':return relation*3
        lane=0 if tower['lane']=='left' else 1
        if side==1:lane=1-lane
        return relation*3+1+lane

    @staticmethod
    def _tower_troop(archetype):
        # Only independently named native tower definitions are classified.
        # King towers are not secretly a selected princess-tower troop.
        return {'BUILDING.PrincessTower':'princess','BUILDING.TowerPrincess':'princess',
            'BUILDING.TowerCannoneer':'cannoneer','BUILDING.TowerDaggerDuchess':'dagger_duchess',
            'BUILDING.TowerRoyalChef':'royal_chef'}.get(archetype,'unknown')

    def _match_features(self,state):
        if self.capture_identity.get('libg_sha256')!=self.timelines.get('runtime_sha256'):return {}
        mode=self.timelines.get('modes',{}).get(str(self.replay['battle']['gamemode']))
        timeline=self.timelines.get('timelines',{}).get(mode['timeline']) if mode else None
        if not timeline:return {}
        tick=state['tick'];sections=timeline['sections'];duration=sum(s['ticks'] for s in sections)
        result={'remaining_time_over_300000ms':max(0,duration-tick)*50/300000.,'producer_entity_overflow_over_capacity':0.}
        elapsed=0
        for section in sections:
            if elapsed<=tick<elapsed+section['ticks']:
                result['overtime_or_sudden_death']=float(section['phase']=='overtime')
                result['tiebreak']=0.  # Still within a resource-declared gameplay section.
                break
            elapsed+=section['ticks']
        elapsed=0
        for rate in timeline['rates']:
            if elapsed<=tick<elapsed+rate['ticks']:
                result['elixir_multiplier_over_3']=rate['visible']/30.
                break
            elapsed+=rate['ticks']
        # No queued local commands exist in synchronous replay collection; do
        # not include future expert commands from the offline supervision plan.
        result['reserved_elixir_over_10']=0.
        return result

    def _card_runtime(self,state,side,tick):
        revealed={self.vocabulary.base(c['card_id']) for c in self.commands if c['side']!=side and c['card_id'] and c['tick']<=tick}
        unknown=max(0,8-len(revealed))/8.
        rows=[]
        for i,card in enumerate(self.decks[side]):
            base=self.vocabulary.base(card['card_id'])
            values={'unknown_opponent_card_fraction':unknown,'evolution_state_known':0.,'ability_state_known':0.}
            carriers=[e for e in state['entities'] if e['side']==side and e.get('ability_slot',0)>0 and self.vocabulary.base(e.get('card_id',-1))==base]
            # If multiple native carriers expose conflicting states, retain the
            # candidate-specific ability state, not an invented card-level one.
            if len(carriers)==1:
                e=carriers[0]
                if e.get('ability_state_code',-1)>=0:
                    values.update(ability_state_known=1.,ability_available=float(e['ability_available']))
                    if e.get('ability_cooldown_remaining_ms',-1)>=0:values['ability_cooldown_remaining_over_30000ms']=e['ability_cooldown_remaining_ms']/30000.
                    if e.get('ability_charges_remaining',-1)>=0:values['ability_charges_over_4']=e['ability_charges_remaining']/4.
            rows.append(CardRuntime(card['card_id'],0,tuple(values.items())))
        for card in sorted(revealed):rows.append(CardRuntime(card,1,(('revealed',1.),('unknown_opponent_card_fraction',unknown))))
        return rows

    def _capabilities(self,frame,objects,details,rows,side):
        """Domain evidence, never a claim that all optional fields are known.

        Frame checks are necessary, not a dataset admission certificate. The
        compiler must additionally validate source, ruleset, splits and labels.
        """
        reasons={};state=frame['state'];telemetry=frame['telemetry'];episode=state['episode']
        identity=self.capture_identity.get('libg_sha256')==self.timelines.get('runtime_sha256') and bool(self.timelines.get('runtime_sha256'))
        coherent=frame['details'].get('valid') is True and state.get('coherent') is True and set(objects)<=set(details)
        facts={key:details.get(key,{}) for key in objects}
        effect_classified=state.get('effects_classified') is True
        if not effect_classified and identity:
            # x86-v1 state writer originally recognized projectiles but called
            # native AEOs "unclassified". Recover classification only with the
            # independently checked generation family, vtable and data family.
            projectile_keys={p['generation_key'] for p in state.get('projectiles',())}
            unmatched=[e for e in state.get('effects',()) if e['category'] not in projectile_keys]
            area_proofs=[e for e in unmatched if 3000000<=e['category']<4000000 and
                e.get('vtable_rva')=='0x19691f8' and details.get(e['category'],{}).get('native_data_id',-1)//1000000==22]
            effect_classified=bool(unmatched) and len(area_proofs)==len(unmatched)==state.get('unclassified_effect_count')
        def certify(name,condition,reason):
            if not condition:reasons[name]=reason
            return bool(condition)
        # Make an unknown geometry attributable: which archetypes actually lacked
        # it, instead of only that the domain failed.
        geometry_offenders=sorted({(row.archetype or f'kind:{row.kind}') for row in rows
                                   if row.radius_tiles is None and row.kind not in ('projectile','area')})
        if geometry_offenders:self.issues['geometry_unknown:'+','.join(geometry_offenders[:4])]+=1
        capabilities=[]
        specials=all(d.get('special_runtime',{}).get('valid') is True for d in facts.values())
        phases=all((d.get('phase_runtime',{}).get('applicable') is False) or
            (d.get('phase_runtime',{}).get('valid') is True and d['phase_runtime'].get('hooks_installed') is True and not d['phase_runtime'].get('overflow')) for d in facts.values())
        checks=[
            ('entity_runtime',coherent and specials and phases,'entity/phase/special runtime not fully inspected'),
            ('archetypes',identity and all(row.archetype is not None for row in rows),'unresolved native definition or runtime identity'),
            ('tower_runtime',episode.get('tower_snapshot_complete') is True and phases and all(self._tower_troop(self.archetype(details[k]))!='unknown' for k in self.tower_native if k in objects and self.tower_native[k]['type']!='king'),'tower snapshot/runtime/troop unknown'),
            ('groups',all(row.group_id and row.group_kind!='unknown' for row in rows),'unresolved group identity'),
            ('combat_events',telemetry.get('enabled') is True and self.lifecycle_available and not self.capture_failures,'event stream missing or discontinuous'),
            ('card_runtime',len(state.get('players',()))==2 and all(len(p.get('hand_deck_indices',()))==4 for p in state['players']) and
                {m['deck_index'] for m in frame['masks'] if m['side']==side}=={index for p in state['players'] if p['side']==side for index in p['hand_deck_indices'] if index>=0},'current own hand/candidate observations incomplete'),
            ('match_runtime',bool(self._match_features(state)) and episode.get('result_source')=='native_logic_terminal_and_crown_tower_entities','native episode or pinned timeline missing'),
            ('commands',self.capture_identity.get('capture_from_tick_zero') is True and self.damage.capture_start_tick==0,'command stream did not start at reset'),
            ('effects',effect_classified and all(d.get('effects_valid') is True or d.get('component_count',0)<4 for d in facts.values()),'effect list classification unavailable'),
            ('relations',coherent and self.lifecycle_available and not self.capture_failures,'current target/parent stream incomplete'),
            # Point/line projectile definitions do not necessarily expose a
            # circular radius. Preserve their explicit unknown extent; units
            # and towers still require measured/static collision geometry.
            ('geometry',identity and all(row.radius_tiles is not None or row.kind in ('projectile','area') for row in rows),'physical entity geometry unknown'),
            ('lifecycle',self.lifecycle_available and not self.capture_failures and all(row.damage is None or row.damage.damage_complete for row in rows),'native lifecycle/attribution ledger incomplete'),
        ]
        for name,condition,reason in checks:
            if certify(name,condition,reason):capabilities.append(name)
        return capabilities,reasons

    def archetype(self,detail,family=None):
        name=detail['native_name'];ids=native_node_ids();table=detail['native_data_id']//1000000
        prefix={34:'CHARACTER',35:'BUILDING',10:'PROJECTILE',11:'BUFF',22:'AEO'}.get(table)
        if prefix and f'{prefix}.{name}' in ids:return f'{prefix}.{name}'
        if f'EXT.{name}' in ids:return f'EXT.{name}'
        matches=[key for key in ids if key.endswith('.'+name)] if name else []
        if len(matches)>1 and family:
            # The same definition name can exist in several families (for example
            # CHARACTER.GoblinMachine and SPELL_CHARACTER.GoblinMachine). Walk the
            # family preference in order and take the first family that leaves
            # exactly one candidate, so ambiguity is resolved by evidence about
            # what the entity is, not by list order.
            preferred={'troop':('CHARACTER.','SPELL_CHARACTER.','EXT.'),
                       'building':('BUILDING.','CHARACTER.','EXT.'),
                       'projectile':('PROJECTILE.','EXT.'),
                       'area':('AEO.','BUFF.','ACTION.','EXT.'),
                       'tower':('BUILDING.','CHARACTER.','EXT.')}.get(family,())
            for prefix in preferred:
                narrowed=[key for key in matches if key.startswith(prefix)]
                if len(narrowed)==1:return narrowed[0]
        return matches[0] if len(matches)==1 else None

    def ingest(self,frame):
        state=frame['state'];tick=state['tick'];telemetry=frame['telemetry']
        duplicate_tick=self.last is not None and tick==self.last['state']['tick']
        # A post-command resample of an already emitted decision Tick cannot
        # retroactively enter that input. It first becomes public at T+1.
        event_observed_tick=tick+1 if duplicate_tick else tick
        if self.last is not None and tick not in (self.last['state']['tick'],self.last['state']['tick']+1):raise ValueError('replay adapter requires every Tick')
        if state.get('coherent') is not True or frame['details']['tick']!=tick:raise ValueError('incoherent adapter frame')
        details={e['key']:e for e in frame['details']['objects']}
        all_objects=state_objects(state)
        self.pointers={e['id']:key for key,e in all_objects.items()}
        if self.last is None:self.damage.capture_start_tick=tick
        if frame.get('capture_quality_flags'):self.capture_failures.add('raw_capture_quality_flags')
        lifecycle=telemetry.get('lifecycle')
        if lifecycle is not None:
            self.lifecycle_available=True
            if lifecycle.get('gap') or lifecycle.get('rejected'):
                self.issues['lifecycle_capture_incomplete']+=1;self.damage.mark_capture_gap(tick,tick);self.capture_failures.add('lifecycle_gap')
            for event in lifecycle.get('events',()):
                sequence=int(event['sequence'])
                if sequence<=self.last_lifecycle_sequence:continue
                if sequence!=self.last_lifecycle_sequence+1:self.capture_failures.add('lifecycle_sequence_gap')
                self.last_lifecycle_sequence=sequence
                if event['operation']==3:
                    # The hook confirms actual healing, but a TLS source candidate
                    # is not yet proof of the healer. Keep source unknown.
                    self._public_event(f'life:{telemetry["epoch"]}:{sequence}','heal',event_observed_tick,event['side'],
                        target=event['key'],amount=event['heal_amount'])
                if event['operation']!=1:continue
                key=event['key']
                if key in self.registrations:continue  # pending->live registration, not a second birth
                self.registrations[key]=event
                if key in self.damage.births:continue
                parent=event.get('parent_key')
                if parent not in self.damage.births:parent=None
                if parent is not None and self.damage.births[parent].side!=event['side']:
                    # A causal action context may be an enemy attacker rather
                    # than the owner of a death-spawn. Preserve the raw receipt,
                    # reject its lineage and fail capture admission; do not
                    # crash the rest of the diagnostic replay or forge a link.
                    self.issues['native_registration_parent_owner_mismatch']+=1
                    self.capture_failures.add('native_parent_owner_mismatch')
                    self.damage.mark_capture_gap(tick,tick);parent=None
                if parent is None:
                    # Some effects carry no parent slot at all. The native spawn
                    # routine that instantiated them did receive the creating
                    # entity, which the host records as the creation context, so
                    # the lineage comes from that context and not from proximity,
                    # the same Tick, or a name match.
                    creator=event.get('context_key')
                    if event.get('context_caller') and creator in self.damage.births:
                        if self.damage.births[creator].side==event['side']:
                            parent=creator;self.issues['native_creation_context_parents']+=1
                        else:
                            # This channel is supplementary: the entity's own parent
                            # slot stays authoritative. A cross-side candidate is
                            # discarded and recorded rather than failing the battle.
                            self.issues['native_creation_context_owner_mismatch']+=1
                command=self.native_commands.get(event.get('command_sequence'))
                born=event['birth_tick'] if event.get('birth_tick',-1)>=0 else event['tick']
                entity_kind='effect' if key<4000000 else 'projectile' if key<5000000 else 'unit'
                if event['side'] in (0,1):
                    self.damage.spawn(key,event['side'],observed_tick=tick,birth_tick=born,parent_key=parent,
                        deployment_id=None if parent is not None else command['id'] if command else None,kind=entity_kind)
        # Terminal tower snapshots retain their old address. Native allocators
        # may reuse it for a new unit/projectile; never classify by pointer alone.
        tower_by_pointer={t['id']:t for t in state['episode']['crown_towers'] if any(
            e['id']==t['id'] and e.get('kind') in (12,13) and e['side']==t['side']
            and e['x']==t['x'] and e['y']==t['y'] for e in state['entities'])}
        for key,e in all_objects.items():
            if e['id'] in tower_by_pointer:
                self.tower_native[key]=tower_by_pointer[e['id']]
                if key in details:self.tower_details[key]=details[key]
        for key,e in sorted(all_objects.items(),reverse=True):
            if key in self.seen:continue
            d=details.get(key,{})
            created=d.get('created_tick_candidate',-1) if key>=5000000 else -1
            if e['id'] in tower_by_pointer:created=0
            birth_tick=created if 0<=created<=tick else None
            matching=[c for c in self.commands if c['side']==e['side'] and c['card_id'] is not None and self.vocabulary.base(e.get('card_id',-1))==self.vocabulary.base(c['card_id']) and c['tick']==birth_tick]
            registration=self.registrations.get(key,{})
            command=self.native_commands.get(registration.get('command_sequence')) or (matching[0] if len(matching)==1 else None)
            parent=d.get('source_key') if 4000000<=key<5000000 else None
            if parent not in self.damage.births:parent=None
            if parent is not None and self.damage.births[parent].side!=e['side']:
                self.issues['projectile_parent_owner_changed']+=1;parent=None
            if key not in self.damage.births:
                self.damage.spawn(key,e['side'],observed_tick=tick,birth_tick=birth_tick,
                    deployment_id=command['id'] if command else None,parent_key=parent,kind='effect' if key<4000000 else 'projectile' if key<5000000 else 'unit')
            self.seen[key]=dict(e);self.birth_commands[key]=command
        now_counts=(telemetry['rejected'],telemetry['unattributed'])
        counter_growth=any(now>old for now,old in zip(now_counts,self.last_telemetry_counts))
        if self.telemetry_epoch is not None and telemetry['epoch']!=self.telemetry_epoch:counter_growth=True
        if counter_growth or telemetry['gap']:
            self.damage.mark_capture_gap(tick,tick);self.issues['damage_capture_not_complete']+=1
            self.capture_failures.add('damage_capture_gap')
        self.last_telemetry_counts=now_counts;self.telemetry_epoch=telemetry['epoch']
        for event in telemetry['events']:
            sequence=int(event['sequence'])
            if sequence<=self.last_damage_sequence:continue
            if sequence!=self.last_damage_sequence+1:self.capture_failures.add('damage_sequence_gap')
            self.last_damage_sequence=sequence
            candidate_source=event.get('attacker_key',event['source_key'])
            source=candidate_source if candidate_source in self.damage.births else None
            if source is None:self.issues['damage_source_not_seen']+=1
            birth=self.damage.births.get(source)
            nonunit=birth is not None and self.damage.nonunit_source_confirmed(source)
            if nonunit:self.issues['nonunit_source_damage_events']+=1
            self.damage.record(DamageEvent(f"{telemetry['epoch']}:{event['sequence']}",event['tick'],tick,source,
                'tower' if event.get('target_is_tower',event['target_kind'] in (12,13)) else 'unit',float(event['effective_damage']),nonunit))
            event_id=f"damage:{telemetry['epoch']}:{sequence}"
            target=event['target_key'];target_object=all_objects.get(target,self.last_public_objects.get(target))
            position=(target_object['x'],target_object['y']) if target_object else None
            self._public_event(event_id,'damage',event_observed_tick,event['source_side'],source,target,event['effective_damage'],position)
            if event.get('lethal') and target in self.damage.births and target not in self.damage.deaths:
                self.damage.mark_dead(target,death_tick=event['tick'],observed_tick=tick)
                self._public_event(event_id+':death','death',event_observed_tick,event['target_side'],target=target,position=position)
        # Public object-set changes are exact spawn/despawn observations. A
        # disappearance is NOT a death or projectile impact without a receipt.
        if self.last is not None and not duplicate_tick:
            for key in all_objects.keys()-self.last_public_objects.keys():
                e=all_objects[key];kind='area_create' if key<4000000 else 'projectile_spawn' if key<5000000 else 'spawn'
                self._public_event(f'appear:{key}:{tick}',kind,tick,e['side'],target=key,position=(e['x'],e['y']))
            for key in self.last_public_objects.keys()-all_objects.keys():
                e=self.last_public_objects[key]
                self._public_event(f'disappear:{key}:{tick}','despawn',tick,e['side'],target=key,position=(e['x'],e['y']))
            old_pointers={e['id']:key for key,e in self.last_public_objects.items()}
            for key in all_objects.keys() & self.last_public_objects.keys():
                e=all_objects[key];old=self.last_public_objects[key];d=details.get(key,{})
                target=self.pointers.get(e.get('target'));previous_target=old_pointers.get(old.get('target'))
                if target!=previous_target:
                    kind='target_lose' if target is None else 'target_acquire' if previous_target is None else 'target_change'
                    self._public_event(f'target:{key}:{tick}',kind,tick,e['side'],key,target,position=(e['x'],e['y']))
                phase=d.get('phase_runtime',{});old_phase=self.last_details.get(key,{}).get('phase_runtime',{})
                if phase.get('valid') and old_phase.get('valid'):
                    for sequence,kind in [('start_sequence','attack_start'),('release_sequence','attack_release')]:
                        if phase.get(sequence,0)>old_phase.get(sequence,0):
                            self._public_event(f'{kind}:{key}:{phase[sequence]}',kind,tick,e['side'],key,target,position=(e['x'],e['y']))
                    if phase.get('sequence_index')!=old_phase.get('sequence_index'):
                        self._public_event(f'attack-sequence:{key}:{tick}','attack_sequence_change',tick,e['side'],key,target)
                old_detail=self.last_details.get(key,{})
                if d.get('native_data_id')!=old_detail.get('native_data_id'):
                    self._public_event(f'transform:{key}:{tick}','transform',tick,e['side'],key,key,position=(e['x'],e['y']))
                if d.get('effects_valid') and old_detail.get('effects_valid'):
                    before=Counter(effect['name'] for effect in old_detail.get('active_effects',()))
                    after=Counter(effect['name'] for effect in d.get('active_effects',()))
                    for name in before.keys()|after.keys():
                        if before[name]==after[name]:continue
                        kind='effect_apply' if not before[name] else 'effect_remove' if not after[name] else 'effect_stack'
                        self._public_event(f'effect:{key}:{name}:{tick}',kind,tick,e['side'],target=key,position=(e['x'],e['y']))
                    hidden=d.get('invisible_count_candidate',-1);old_hidden=old_detail.get('invisible_count_candidate',-1)
                    if hidden>=0 and old_hidden>=0 and bool(hidden)!=bool(old_hidden):
                        self._public_event(f'visibility:{key}:{tick}','visibility_change',tick,e['side'],key,key)
        if not duplicate_tick:
            self.last_public_objects=all_objects
            self.last_details=details
        while self.event_window and self.event_window[0]['tick']<=tick-5:
            self.event_ids.discard(self.event_window.popleft()['id'])
        if self.last is not None and tick!=self.last['state']['tick']:self.last_before=self.last
        self.last=frame
        if tick%5 or duplicate_tick:return []
        self.last_inputs={};self.last_capability_reasons={}
        outputs=[]
        for side in (0,1):
            try:outputs.append(self._actor(frame,all_objects,details,tower_by_pointer,side))
            except (ValueError,KeyError,OverflowError) as error:
                if not self.builder.config.allow_incomplete_capture:raise
                self.issues[f'actor_rejected:{type(error).__name__}:{str(error)[:100]}']+=1
        return outputs

    def _actor(self,frame,objects,details,tower_by_pointer,side):
        state=frame['state'];tick=state['tick']
        raw_entities=[dict(e) for e in objects.values() if e['id'] not in tower_by_pointer]
        raw={**state,'kind':'libg_native_train_state_v1','entities':raw_entities,'entity_count':len(raw_entities)}
        normalized=normalize_native_state(raw)
        key_map={key:self.tower_key(tower,side) for key,tower in self.tower_native.items()}
        key_map.update({key:self.tower_key(tower_by_pointer[e['id']],side) if e['id'] in tower_by_pointer else key for key,e in objects.items()})
        rows=[];relations=[];effects=[]
        previous_state=(self.last_before or {}).get('state',{})
        previous_objects=state_objects(previous_state)
        for key,e in objects.items():
            d=details.get(key,{})
            if not d:
                self.issues['entity_details_missing']+=1
                continue
            values={};child={};tower={};mapped=key_map[key]
            is_tower=e['id'] in tower_by_pointer
            kind=tower_by_pointer[e['id']]['type'] if is_tower else 'area' if key<4000000 else 'projectile' if key<5000000 else 'building' if d['native_data_id']//1000000==35 else 'troop'
            phase_values,phase_child=phase_features(d,tick)
            values.update(phase_values);child.update(phase_child)
            child.update(special_features(d))
            if d['builtin_shield']>=0:values['shield']=float(d['builtin_shield'])
            if 0<=d['created_tick_candidate']<=tick:values['age_ms']=(tick-d['created_tick_candidate'])*50.
            previous=previous_objects.get(key)
            values.update(actor_velocity(e,previous,tick=tick,previous_tick=previous_state.get('tick'),side=side))
            target=self.pointers.get(e.get('target'))
            if target in key_map:relations.append(PublicRelation(mapped,key_map[target],'targets',tick))
            child['has_visible_target']=float(target in key_map)
            if is_tower:tower['has_visible_target']=float(target in key_map)
            if is_tower:
                # Both banks share these named phase features; writing only the
                # child bank silently hid the already captured tower behaviour.
                tower.update({name:value for name,value in child.items() if name in TOWER_FEATURES})
                for value_name,feature_name in [('attack_cooldown_ms','attack_cooldown_remaining_over_5000ms'),('attack_phase_ms','attack_phase_remaining_over_5000ms')]:
                    if value_name in values:tower[feature_name]=values[value_name]/5000.
            if 4000000<=key<5000000:
                nominal=projectile_nominal_features(self.archetype(d,'projectile'),include_damage_basis=True)
                if 'projectile_damage' in values:nominal.pop('projectile_damage_over_5000',None)
                elif 'projectile_damage_over_5000' in nominal:values['projectile_damage']=nominal['projectile_damage_over_5000']*5000.
                for name,value in nominal.items():child.setdefault(name,value)
                child['projectile_radius_known']=float('projectile_radius_over_5_tiles' in child)
                child['projectile_damage_known']=float('projectile_damage' in values or 'projectile_damage_over_5000' in child)
                child.setdefault('projectile_in_flight',1.)
                child.update(projectile_target_x_over_board_width=(17999-e['target_x'] if side else e['target_x'])/18000,
                    projectile_target_y_over_board_height=(31999-e['target_y'] if side else e['target_y'])/32000,projectile_target_position_known=1.)
            command=self.birth_commands.get(key)
            damage=replace(self.damage.snapshot(key,tick),unit_key=mapped) if kind not in ('projectile','area') else None
            if d['effects_valid']:
                invisible=d.get('invisible_count_candidate',-1)
                if isinstance(invisible,int) and invisible>=0:child['visibility_hidden_or_burrowed']=float(invisible>0)
                for j,buff in enumerate(d['active_effects']):
                    effects.append(PublicEffect(mapped,effect_kind(buff['name']),tick,remaining_ms=float(buff['remaining_ms']) if buff['remaining_ms']>=0 else None,effect_name='BUFF.'+buff['name']))
            elif d['component_count']>=4:self.issues['effects_not_classified']+=1
            archetype=self.archetype(d,kind)
            if archetype is None:self.issues['archetype_unresolved:'+d['native_name']]+=1
            prototype=definition_facts().get(archetype,{})
            flying=prototype.get('FlyingHeight')
            if isinstance(flying,(int,float)):values['flying']=float(flying>0)
            birth=self.damage.births.get(key)
            group=birth.deployment_id if birth and birth.deployment_id else command['id'] if command else f'entity-{key}'
            group_kind='deployment' if (birth and birth.deployment_id) or command else 'singleton'
            if birth and birth.parent_key in key_map:relations.append(PublicRelation(mapped,key_map[birth.parent_key],'spawned_by',tick))
            tower_troop=self._tower_troop(archetype) if is_tower else 'unknown'
            rows.append(EntityDetail(mapped,kind,group,group_kind,
                radius_tiles=nominal_geometry(archetype),
                values=tuple(values.items()),archetype=archetype,child_features=tuple(child.items()),tower_features=tuple(tower.items()),
                tower_troop=tower_troop,damage=damage))
        # The public crown-tower snapshot deliberately persists destroyed
        # towers after the native object leaves the active collection. Preserve
        # their identity and confirmed terminal state, not stale attack/buff
        # timers copied from the last live object.
        existing={row.key for row in rows}
        for native_key,prior in self.tower_native.items():
            mapped=self.tower_key(prior,side)
            if mapped in existing:continue
            terminal=next((t for t in state['episode']['crown_towers'] if
                (t['side'],t['type'],t['lane'])==(prior['side'],prior['type'],prior['lane'])),None)
            static=self.tower_details.get(native_key)
            if not terminal or not terminal.get('destroyed') or terminal['hp']!=0 or not static:continue
            archetype=self.archetype(static,'tower')
            rows.append(EntityDetail(mapped,terminal['type'],f'entity-{native_key}','singleton',
                radius_tiles=nominal_geometry(archetype),archetype=archetype,
                tower_features=(('active',0.),('has_visible_target',0.)),tower_troop=self._tower_troop(archetype),
                damage=replace(self.damage.snapshot(native_key,tick),unit_key=mapped)))
        player=normalized.players[side];candidates=[]
        for mask in frame['masks']:
            if mask['side']!=side or mask['deck_index'] not in player.hand:continue
            index=mask['deck_index'];slot=player.hand.index(index);card=self.decks[side][index]
            probe=normalize_native_probe(mask['result']);grid=derive_deployment_rows(probe,normalized,side=side,card_id=card['card_id'])
            flat=tuple(ch=='1' for row in grid for ch in row)
            if side:flat=tuple(reversed(flat))
            candidates.append(Candidate(index,card['card_id'],'play',probe['card_cost_raw']/10000.,True,'grid',flat,
                hand_slot=slot,form_flags=None,first_only=card['card_id']==28000006,effective_card_id=probe['resolved_data_id']))
        for entity in normalized.entities:
            if entity.side==side and entity.ability_slot>0:
                if entity.ability_mana_cost<0:raise ValueError('native ability cost unavailable')
                candidates.append(Candidate(10000000+entity.key,entity.card_id,'ability',float(entity.ability_mana_cost),bool(entity.ability_available),'none',(False,)*576,
                    source_entity=entity.key,form_flags=None))
        events=[]
        for event in self.event_window:
            features={'event_count_over_16':1/16.,'span_over_decision_ticks':0.,'amount_coverage':float(event['amount'] is not None),
                'position_coverage':float(event['position'] is not None)}
            if event['amount'] is not None:features.update(amount_sum_over_5000=event['amount']/5000.,max_amount_over_5000=event['amount']/5000.)
            if event['position'] is not None:
                x,y=event['position'];features.update(mean_x_over_board_width=(17999-x if side else x)/18000.,mean_y_over_board_height=(31999-y if side else y)/32000.)
            source=event['source'];target=event['target'];card=self.seen.get(source,{}).get('card_id')
            events.append(CombatEvent(event['id'],event['kind'],event['tick'],event['side'],
                key_map.get(source,source),key_map.get(target,target),card if card and card>0 else None,features=tuple(features.items())))
        previous=[]
        for command in [c for c in self.commands if c['side']==side and c['tick']<=tick][-2:]:
            target=None
            if command['kind']=='play':
                x,y=command['action']['x'],command['action']['y'];target=((17999-x if side else x)/18000,(31999-y if side else y)/32000)
            previous.append(OwnCommand(command['id'],command['tick'],command['tick'],command['kind'],command['card_id'],target))
        card_runtime=self._card_runtime(state,side,tick)
        capabilities,reasons=self._capabilities(frame,objects,details,rows,side)
        self.last_capability_reasons[side]=reasons
        self.capability_counts.update(capabilities)
        # The strict tensorizer refuses a whole batch that contains one sample
        # whose capture streams are incomplete, so such a sample can never carry
        # training signal. Record exactly why it is unusable: the compiler
        # excludes those samples and reports the reason instead of silently
        # dropping data or shipping a dataset the model will reject.
        unresolved=sorted({row.archetype or f'kind:{row.kind}' for row in rows
                           if row.archetype is None or row.group_id is None or row.kind=='unknown'})
        if reasons or unresolved:
            label=','.join(sorted(reasons)) if reasons else 'unresolved_entity'
            if unresolved:label+=f'+unresolved:{",".join(unresolved[:3])}'
            self.issues['incomplete_capture:'+label]+=1
        semantic=PublicSemantics(self.uid,side,tick,entities=tuple(rows),effects=tuple(effects),relations=tuple(relations),previous=tuple(previous),combat_events=tuple(events),
            card_runtime=tuple(card_runtime),match_features=tuple(self._match_features(state).items()),source='native-elite-details.x86-v1',capture_capabilities=tuple(capabilities))
        frame_input=self.builder.from_native(raw,episode_uid=self.uid,actor_side=side,own_deck=self.decks[side],candidates=candidates,history=self.history,semantics=semantic)
        self.last_inputs[side]=frame_input
        batch=self.builder.batch([frame_input]);self.samples+=1
        for name,names,tensor,mask in [('child',CHILD_FEATURES,batch.semantic.elite.child,batch.entity_mask),('tower',TOWER_FEATURES,batch.semantic.elite.tower,batch.entity_mask & (batch.entity_features[:,:,6]==0)),
            ('group',GROUP_FEATURES,batch.semantic.elite.groups,batch.semantic.group_mask)]:
            if name=='child':mask=mask & (batch.entity_features[:,:,6]>0)
            for j,field in enumerate(names):
                self.coverage[name+'.'+field]+=int((tensor[:,:,2*j+1].bool()&mask).sum());self.opportunities[name+'.'+field]+=int(mask.sum())
        self.coverage['damage.valid_entities']+=int(batch.semantic.elite.lifecycle_mask.sum())
        return batch

    def report(self):
        return dict(schema='elite-replay-field-coverage.v1',actor_samples=self.samples,training_ready=False,issues=dict(self.issues),
            known_counts=dict(self.coverage),opportunities=dict(self.opportunities),capability_sample_counts=dict(self.capability_counts),
            last_capability_failures=self.last_capability_reasons,persistent_capture_failures=sorted(self.capture_failures),
            missing_child_fields=[name for name in CHILD_FEATURES if not self.coverage['child.'+name]],
            note='Unknown mechanisms are not marked absent. Production remains blocked until per-build telemetry coverage is certified.')


@contextmanager
def raw_capture_lines(path):
    if str(path).endswith('.zst'):
        import zstandard
        with Path(path).open('rb') as source,zstandard.ZstdDecompressor().stream_reader(source) as raw,io.TextIOWrapper(raw,encoding='utf-8') as text:
            yield text
    else:
        with gzip.open(path,'rt',encoding='utf-8') as text:yield text


def audit_capture(path:Path,output:Path):
    assembler=None
    with raw_capture_lines(path) as lines:
        for line in lines:
            row=json.loads(line)
            if row['kind']=='episode':assembler=ReplayFeatureAssembler(row['replay'],path.parent.name,capture_identity=row.get('identity'))
            elif row['kind']=='command_receipt':assembler.command(row)
            elif row['kind']=='frame':assembler.ingest(row)
    report=assembler.report();output.write_text(json.dumps(report,ensure_ascii=True,indent=2)+'\n',encoding='utf-8');return report
