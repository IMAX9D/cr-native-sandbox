"""Conservative facts derived from public raw captures and pinned definitions."""
from functools import lru_cache
import json
import math
from pathlib import Path


def state_objects(state):
    objects={e['category']:e for e in state.get('entities',())}
    objects.update({e['generation_key']:{**e,'category':e['generation_key'],'hp':-1,'max_hp':-1,'level':-1,'kind':0} for e in state.get('projectiles',())})
    objects.update({e['category']:{**e,'hp':-1,'max_hp':-1,'level':-1} for e in state.get('effects',()) if 3000000<=e['category']<4000000})
    return objects


def actor_velocity(current,previous,*,tick,previous_tick,side):
    if previous is None or previous_tick is None or tick<=previous_tick:return {}
    if current.get('id')!=previous.get('id'):return {}
    scale=(-1 if side else 1)*20./(1000*(tick-previous_tick))
    return dict(velocity_x=(current['x']-previous['x'])*scale,velocity_y=(current['y']-previous['y'])*scale)


@lru_cache(maxsize=1)
def definition_facts():
    raw=json.loads((Path(__file__).with_name('data')/'native_definitions.json').read_text('utf-8'))
    if raw['runtime_sha256']!='fa6704b83cb9c5b8eecb7b56c9671b834d636a3a6d9ac446e698e1262dc246ba':
        raise ValueError('native definition facts runtime mismatch')
    out={}
    for node,field,kind,string,number,conflict in raw['attributes']:
        if conflict:
            out.setdefault(raw['node_names'][node],{}).setdefault('__conflicts__',set()).add(raw['field_names'][field]);continue
        out.setdefault(raw['node_names'][node],{})[raw['field_names'][field]]=raw['string_values'][string] if kind==2 else number
    return out


def effective_definition(archetype,seen=()):
    if archetype in seen:raise ValueError('native definition inheritance cycle')
    if archetype not in definition_facts() and archetype and '.' in archetype:
        family,name=archetype.split('.',1);extension='EXT.'+name
        declared_base=definition_facts().get(extension,{}).get('Base','')
        # EXT creates a named native row in the explicitly declared base table.
        if isinstance(declared_base,str) and declared_base.startswith(family+'.'):
            return effective_definition(extension,(*seen,archetype))
    local=definition_facts().get(archetype,{})
    base=local.get('Base')
    if isinstance(base,str) and base not in definition_facts():
        family=archetype.split('.',1)[0]
        families=('AEO','AREA_EFFECT_OBJECT') if family in ('AEO','AREA_EFFECT_OBJECT') else (family,)
        matches=[f+'.'+base for f in families if f+'.'+base in definition_facts()]
        if '.' in base:
            base_family,base_name=base.split('.',1);extension='EXT.'+base_name
            declared_base=definition_facts().get(extension,{}).get('Base','')
            if isinstance(declared_base,str) and declared_base.startswith(base_family+'.'):matches=[extension]
        base=matches[0] if len(matches)==1 else None
    result=effective_definition(base,(*seen,archetype)) if isinstance(base,str) and base in definition_facts() else {}
    result.update({k:v for k,v in local.items() if k!='__conflicts__'})
    # Native TOML additive/replacement overrides are preserved as array facts.
    # Apply only the verified scalar + and = forms; unsupported operations
    # remain unknown instead of leaking an inherited stale value.
    for field,operator in local.items():
        if not field.endswith('[0]') or not isinstance(operator,str):continue
        if operator not in ('+','-','*','/','='):continue
        name=field[:-3];operand=local.get(name+'[1]');old=result.get(name)
        if operator=='=' and isinstance(operand,(int,float)):result[name]=operand
        elif operator=='+' and isinstance(old,(int,float)) and isinstance(operand,(int,float)):result[name]=old+operand
        else:result.pop(name,None)
    for key in local.get('__conflicts__',()):result.pop(key,None)
    return result


@lru_cache(maxsize=1)
def rarity_curves():
    data=json.loads((Path(__file__).with_name('data')/'rarity_curves.json').read_text('utf-8'))
    if data['runtime_sha256']!='fa6704b83cb9c5b8eecb7b56c9671b834d636a3a6d9ac446e698e1262dc246ba':raise ValueError('rarity curve runtime mismatch')
    return data['curves']


def projectile_nominal_features(archetype,*,include_damage_basis=False):
    d=effective_definition(archetype)
    result={}
    if d.get('Homing') in (0,1):result['projectile_homing']=float(d['Homing'])
    radius=d.get('Radius')
    if isinstance(radius,(int,float)) and radius>=0:result['projectile_radius_over_5_tiles']=radius/5000.
    if include_damage_basis:
        curve=rarity_curves().get(d.get('Rarity'));damage=d.get('Damage')
        if curve and 11 in curve['levels'] and isinstance(damage,(int,float)) and damage>=0:
            multiplier=curve['multipliers'][curve['levels'].index(11)]
            result['projectile_damage_over_5000']=math.trunc(damage*multiplier/100)/5000.
    # Nominal base damage is NOT silently substituted for level/buff-scaled damage.
    return result


def effect_kind(name):
    d=effective_definition('BUFF.'+name)
    if d.get('Invisible')==1:return 'invisibility'
    if d.get('SpeedMultiplier')==-100 and d.get('HitSpeedMultiplier')==-100:return 'stun'
    if isinstance(d.get('DamagePerSecond'),(int,float)) and d['DamagePerSecond']>0:return 'damage'
    if isinstance(d.get('HealPerSecond'),(int,float)) and d['HealPerSecond']>0:return 'heal'
    if isinstance(d.get('SpeedMultiplier'),(int,float)):
        if d['SpeedMultiplier']<0:return 'slow'
        if d['SpeedMultiplier']>100:return 'speed'
    return 'unknown'


def phase_features(detail,tick):
    """First pre-command observation per Tick; native events precede T by one."""
    p=detail.get('phase_runtime',{})
    if not p.get('valid') or not p.get('hooks_installed') or p.get('overflow'):return {},{}
    child={};values={}
    stage=p.get('sequence_index')
    if isinstance(stage,int) and stage>=0:child['attack_sequence_index_over_10']=stage/10.
    deploy=p.get('deploy_remaining_ms')
    if isinstance(deploy,int) and deploy>=0:
        child['deployment_in_progress']=float(deploy>0);values['deployment_ms']=float(deploy)
    phase=None
    if p.get('last_release_tick',-1)>=0 and p['last_release_tick']==tick-1:phase='release'
    elif p.get('start_sequence',0)>p.get('execute_sequence',0) and p.get('target_present'):phase='windup'
    elif p.get('load_remaining_ms',-1)>0:phase='cooldown'
    elif p.get('target_present') is False and p.get('timeline_ms')==0:phase='idle'
    if phase is not None:
        child.update(attack_windup=float(phase=='windup'),attack_release_or_channel=float(phase=='release'),attack_cooldown=float(phase=='cooldown'))
    if phase=='cooldown':values['attack_cooldown_ms']=float(p['load_remaining_ms'])
    speed=p.get('effective_movement_speed')
    if isinstance(speed,int) and speed>=0:child['movement_effective_speed_over_1000']=speed/1000.
    progress=p.get('classic_charge_progress')
    if isinstance(progress,int) and progress>=-1:
        child['classic_charge_ready']=float(progress>=10000)
        if progress>=0:child['classic_charge_progress_over_10000']=progress/10000.
    if phase=='windup' and p.get('hit_speed_ms',0)>0 and p.get('scale_output_ms',0)>0 and p.get('scale_tick')==tick-1:
        remaining=p['hit_speed_ms']-(p['timeline_ms']+p['attack_dash_time_ms'])%p['hit_speed_ms']
        # ceil(number of verified native scaled steps) * 50ms.
        values['attack_phase_ms']=float(((remaining+p['scale_output_ms']-1)//p['scale_output_ms'])*50)
    return values,child


def nominal_geometry(archetype):
    d=effective_definition(archetype)
    family=archetype.split('.',1)[0] if archetype else ''
    if family=='EXT':
        base=d.get('Base','');family=base.split('.',1)[0]
    radius=d.get('ProjectileRadius',d.get('Radius')) if family=='PROJECTILE' else d.get('Radius') if family in ('AEO','AREA_EFFECT_OBJECT') else d.get('CollisionRadius')
    return radius/1000. if isinstance(radius,(int,float)) and radius>=0 else None


def special_features(detail):
    data=detail.get('special_runtime')
    if not data or data.get('valid') is not True:return {}
    result={}
    for name in ('capture','relocation','periodic_modifier','extra_spawn','projectile'):
        row=data.get(name)
        if row and row.get('present') and row.get('valid') is not True:raise ValueError('invalid native special runtime: '+name)
    capture=data.get('capture',{})
    if 'present' in capture:result['capture_runtime_known']=float(capture['present'])
    if capture.get('present'):
        names=('capture_acquired_delay','capture_grab_pause','capture_dragging','capture_contained','capture_release_pending')
        for i,name in enumerate(names):result[name]=float(any(t['phase']==i for t in capture['targets']))
        remaining=[t['remaining_ms'] for t in capture['targets'] if t['remaining_ms']>=0]
        if remaining:result['capture_phase_budget_remaining_over_5000ms']=min(remaining)/5000.
        result['capture_action_cycle_progress']=min(capture['hit_accumulator_ms']/capture['hit_frequency_ms'],1.)
        if capture['cooldown_ms']>0:result['capture_cooldown_remaining_ratio']=capture['cooldown_remaining_ms']/capture['cooldown_ms']
    relocation=data.get('relocation',{})
    if 'present' in relocation:result['threshold_relocation_runtime_known']=float(relocation['present'])
    if relocation.get('present'):
        n=len(relocation['thresholds']);stage=relocation['stage'];index=(stage-1)//2
        result.update(relocation_waiting_threshold=float(stage%2==1 and stage<2*n+1),relocation_active=float(stage%2==0),
            relocation_exhausted=float(stage==2*n+1),relocation_burrowed=float(stage%2==0 and relocation['owner_state']==6 and relocation['remaining_ms']>0),
            relocation_remaining_ratio=relocation['remaining_ms']/relocation['duration_ms'],relocation_index_ratio=index/n)
        if index<n:result['relocation_threshold_percent']=relocation['thresholds'][index]/100.
    periodic=data.get('periodic_modifier',{})
    if 'present' in periodic:result['periodic_attack_modifier_present']=float(periodic['present'])
    if periodic.get('present'):
        result.update(periodic_attack_modifier_progress_ratio=periodic['completed']/periodic['period'],
            periodic_attack_modifier_source_death_linger=float(periodic['source_death_linger']),
            periodic_attack_modifier_linger_remaining_ratio=periodic['remaining_ms']/periodic['linger_ms'])
    extra=data.get('extra_spawn',{})
    if 'present' in extra:result['extra_spawn_accumulator_known']=float(extra['present'])
    if extra.get('present'):result['extra_spawn_accumulator_ratio']=extra['current']/extra['capacity']
    projectile=data.get('projectile',{})
    if projectile.get('present'):
        result['projectile_in_flight']=float(not projectile['terminal'])
        result['projectile_drag_stage_known']=float(projectile['drag_configured'])
        if projectile['drag_configured']:result['projectile_drag_back_active']=float(projectile['drag_stage']==1)
        if projectile['homing_known']:result['projectile_homing']=float(projectile['homing'])
    if data.get('variables_valid'):
        variables=dict(zip(data['variable_keys'],data['variable_values']))
        progress=variables.get(1846274699);decay=variables.get(531698662)
        # Pinned InfernoDragon_EV1 resources contain min(50, count+1) and
        # VARIABLE.InfernoDragon_EV1_DecayTime.DefaultValue=7000.
        result['attack_sequence_progress_known']=float(progress is not None)
        result['attack_sequence_decay_known']=float(decay is not None)
        if progress is not None:
            if not 0<=progress<=50:raise ValueError('invalid native attack progression')
            result['attack_sequence_progress_ratio']=progress/50.
        if decay is not None:
            if not 0<=decay<=7000:raise ValueError('invalid native attack decay')
            result['attack_sequence_decay_remaining_ratio']=decay/7000.
    return result
