"""Observed damage attribution, separately from hindsight deployment evaluation.

Only confirmed effective damage is accepted. Never infer a source by splitting
an HP delta across nearby attackers. Descendants retain their deployment link.
"""
from dataclasses import dataclass
import math

LIFECYCLE_FEATURE_NAMES=('unit_damage_lifetime','tower_damage_lifetime','unit_damage_1s','tower_damage_1s',
    'unit_damage_3s','tower_damage_3s')


@dataclass(frozen=True)
class UnitBirth:
    key:int
    side:int
    observed_tick:int
    birth_tick:int|None
    deployment_id:str|None
    parent_key:int|None=None
    kind:str='unit'


@dataclass(frozen=True)
class DamageEvent:
    event_id:str
    occurred_tick:int
    observed_tick:int
    source_key:int|None
    target_kind:str
    amount:float
    confirmed_nonunit_source:bool=False
    def __post_init__(self):
        if not self.event_id or self.target_kind not in ('unit','tower') or any(type(t) is not int for t in (self.occurred_tick,self.observed_tick)) or not 0<=self.occurred_tick<=self.observed_tick:
            raise ValueError('invalid damage receipt identity/time')
        if isinstance(self.amount,bool) or not isinstance(self.amount,(int,float)) or not math.isfinite(self.amount) or self.amount<0:
            raise ValueError('effective damage must be finite and nonnegative')
        if self.source_key is not None and type(self.source_key) is not int:raise ValueError('invalid source generation key')
        if type(self.confirmed_nonunit_source) is not bool:raise ValueError('source applicability must be explicit')


@dataclass(frozen=True)
class DamageSummary:
    episode_uid:str
    unit_key:int
    as_of_tick:int
    values:tuple[float,...]
    damage_complete:bool
    def __post_init__(self):
        if not self.episode_uid or type(self.unit_key) is not int or type(self.as_of_tick) is not int or self.as_of_tick<0 or len(self.values)!=len(LIFECYCLE_FEATURE_NAMES):raise ValueError('invalid lifecycle summary')
        if any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in self.values):raise ValueError('invalid lifecycle measurement')
        if type(self.damage_complete) is not bool:raise ValueError('lifecycle coverage must be explicit')


class DamageTracker:
    def __init__(self,episode_uid:str,*,continuous_damage_capture:bool=False,capture_start_tick:int=0,max_events:int=200000):
        if not episode_uid or max_events<=0 or type(continuous_damage_capture) is not bool:raise ValueError('invalid damage tracker')
        self.episode_uid=episode_uid;self.continuous=continuous_damage_capture;self.max_events=max_events
        if type(capture_start_tick) is not int or capture_start_tick<0:raise ValueError('invalid capture start')
        self.capture_start_tick=capture_start_tick;self.gaps=[]
        self.births={};self.events={};self.deaths={};self._by_source={};self._unattributed=[]

    def spawn(self,key:int,side:int,*,observed_tick:int,birth_tick:int|None=None,deployment_id:str|None=None,parent_key:int|None=None,kind:str='unit'):
        if type(key) is not int or side not in (0,1) or type(observed_tick) is not int or observed_tick<0:raise ValueError('invalid unit birth')
        if birth_tick is not None and (type(birth_tick) is not int or not 0<=birth_tick<=observed_tick):raise ValueError('invalid birth time')
        if parent_key is not None:
            parent=self.births.get(parent_key)
            if parent is None or parent.observed_tick>observed_tick:raise ValueError('parent lineage unknown at spawn')
            if parent.side!=side:raise ValueError('parent/child owner mismatch')
            if birth_tick is not None and parent.birth_tick is not None and birth_tick<parent.birth_tick:raise ValueError('child born before parent')
            if deployment_id is not None and deployment_id!=parent.deployment_id:raise ValueError('conflicting deployment lineage')
            deployment_id=parent.deployment_id
        if kind not in ('unit','projectile','effect'):raise ValueError('invalid lifecycle entity kind')
        item=UnitBirth(key,side,observed_tick,birth_tick,deployment_id,parent_key,kind)
        if key in self.births and self.births[key]!=item:raise ValueError('generation key reused; preserve lifecycle identity')
        self.births[key]=item

    def record(self,event:DamageEvent):
        old=self.events.get(event.event_id)
        if old is not None:
            if old!=event:raise ValueError('conflicting duplicate damage event')
            return False
        if len(self.events)>=self.max_events:raise OverflowError('damage ledger capacity exceeded')
        if event.source_key is not None:
            birth=self.births.get(event.source_key)
            if birth is None:raise ValueError('unregistered source; use unknown, never guess an attacker')
            if birth.birth_tick is not None and event.occurred_tick<birth.birth_tick:raise ValueError('damage before source birth')
        credited=self.credited_source(event)
        confirmed=event.confirmed_nonunit_source and self.nonunit_source_confirmed(event.source_key)
        if event.confirmed_nonunit_source and not confirmed:raise ValueError('nonunit source lacks native deployment evidence')
        # Validate the entire receipt before mutating either ledger. A rejected
        # event must not subsequently look like an accepted duplicate on retry.
        self.events[event.event_id]=event
        if credited is None:
            if not confirmed:self._unattributed.append(event)
        else:self._by_source.setdefault(credited,[]).append(event)
        return True

    def mark_dead(self,key:int,*,death_tick:int,observed_tick:int):
        if key not in self.births or not 0<=death_tick<=observed_tick:raise ValueError('invalid death receipt')
        if key in self.deaths and self.deaths[key]!=(death_tick,observed_tick):raise ValueError('conflicting death receipt')
        self.deaths[key]=(death_tick,observed_tick)

    def mark_capture_gap(self,start_tick:int,end_tick:int):
        if not 0<=start_tick<=end_tick:raise ValueError('invalid capture gap')
        self.gaps.append((start_tick,end_tick))

    def credited_source(self,event:DamageEvent):
        key=event.source_key
        while key is not None:
            birth=self.births[key]
            if birth.kind=='unit':return key
            key=birth.parent_key
        return None

    def source_chain_root(self,key:int):
        """Follow the recorded native parent chain; never invent a link."""
        birth=self.births.get(key);seen=set()
        while birth is not None and birth.parent_key is not None:
            if birth.key in seen:return None
            seen.add(birth.key);birth=self.births.get(birth.parent_key)
        return birth

    def nonunit_source_confirmed(self,key:int)->bool:
        """Native deployment evidence is read at the chain root, not the leaf.

        A spell's area object and the projectile or child effect it spawns are
        one causal source: the root carries the originating native command,
        while the child carries only the lineage. Demanding the evidence on the
        immediate object misreports an attributed spell hit as absent
        attribution, which then suspends every later lifecycle summary.
        """
        birth=self.births.get(key)
        if birth is None or birth.kind=='unit':return False
        root=self.source_chain_root(key)
        return root is not None and root.kind!='unit' and root.deployment_id is not None

    def snapshot(self,key:int,as_of_tick:int):
        birth=self.births.get(key)
        if birth is None or as_of_tick<birth.observed_tick:raise ValueError('unit was not observed at snapshot time')
        own=[e for e in self._by_source.get(key,()) if e.observed_tick<=as_of_tick]
        def damage(kind,start=None):
            return sum(e.amount for e in own if e.target_kind==kind and (start is None or e.occurred_tick>start))
        values=(damage('unit'),damage('tower'),damage('unit',as_of_tick-20),damage('tower',as_of_tick-20),
            damage('unit',as_of_tick-60),damage('tower',as_of_tick-60))
        complete=self.continuous and birth.birth_tick is not None and birth.birth_tick>=self.capture_start_tick and not any(
            e.observed_tick<=as_of_tick and e.occurred_tick>=(birth.birth_tick if birth.birth_tick is not None else birth.observed_tick)
            for e in self._unattributed)
        complete=complete and not any(end>=(birth.birth_tick if birth.birth_tick is not None else birth.observed_tick) and start<=as_of_tick for start,end in self.gaps)
        return DamageSummary(self.episode_uid,key,as_of_tick,values,complete)

    def deployment_totals(self,deployment_id:str,as_of_tick:int):
        """Observed result only, not an optimal-placement score or future label."""
        keys={b.key for b in self.births.values() if b.deployment_id==deployment_id and b.observed_tick<=as_of_tick}
        events=[e for e in self.events.values() if e.source_key in keys and e.observed_tick<=as_of_tick]
        return {'unit_damage':sum(e.amount for e in events if e.target_kind=='unit'),
            'tower_damage':sum(e.amount for e in events if e.target_kind=='tower'),'descendant_count':len(keys),
            'all_observed_units_dead':bool(keys) and all(k in self.deaths and self.deaths[k][1]<=as_of_tick for k in keys)}
