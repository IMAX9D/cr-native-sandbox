"""Versioned public semantics for the FirstLight-inspired rich R0 adaptation.

The optional runtime supplement is actor-projected, identified and explicit;
we never scrape arbitrary debug fields or infer private opponent state. Missing
values are unknown, not zero-valued facts. Native addresses are routing only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

from native_core.card_catalog import catalog, form_index
from .catalog import CardVocabulary
from .config import digest
from .feature_contract import CHILD_FEATURES,TOWER_FEATURES,GROUP_FEATURES,CARD_FEATURES,MATCH_FEATURES,COMBAT_FEATURES,COMBAT_KINDS,REFERENCE_FEATURES
from .lifecycle import DamageSummary,LIFECYCLE_FEATURE_NAMES

REFERENCE_COMMIT = '28d66cc0a5d65888515e22fdf22f11d783b65efb'
SEMANTIC_VERSION = 'r0-public-semantic.v3'
# Values are nominal metadata, never the live effect of a buff or card level.
STATIC_SCALES = dict(cost=10., is_troop=1., is_building=1., is_spell=1.,
    attacks_ground=1., attacks_air=1., targets_buildings_only=1., flying=1.,
    speed_tiles_s=10., range_tiles=12., hit_interval_ms=5000., sight_tiles=12.,
    damage=5000., max_hp=10000., radius_tiles=5., summon_count=16.,
    ability_cost=10., ability_charges=4., ability_cast_ms=5000.,
    ability_delay_ms=5000., ability_cooldown_ms=30000., evolution_cycles=4.,
    speed_native=120., flying_height_native=1000.)
DYNAMIC_SCALES = dict(velocity_x=10., velocity_y=10., shield=10000.,
    age_ms=60000., attack_cooldown_ms=5000., attack_phase_ms=5000.,
    deployment_ms=5000., projectile_damage=5000., targetable=1., flying=1.,
    attack_windup=1., attack_charging=1., damage_multiplier=5.,
    movement_speed=10., king_activated=1., tower_charge=1.)
EFFECT_KINDS = ('unknown','damage','heal','shield','stun','slow','speed',
    'damage_multiplier','summon','knockback','transform','invisibility','invulnerability')
RELATION_KINDS = ('none','targets','targeted_by','spawned_by','spawns','same_group',
    'same_card','same_side','opposes','carries','carried_by','effect_source','effect_target','supports',
    'source_of','sourced_by','targeting_modifier_of','targeting_modified_by','card_represents_group',
    'spawned_by_group','derived_from_volley','tower_troop_supports','tower_troop_supported_by','captures','captured_by')
ENTITY_KINDS = ('unknown','troop','building','projectile','area','king','princess')
GROUP_KINDS = ('unknown','deployment','spawn_wave','volley','persistent_effect','singleton')
PLANE_NAMES = tuple(f'{side}_{feature}' for side in ('own','enemy') for feature in
    ('ground_count','air_count','building_count','projectile_count','absolute_hp','shield','threat'))+(
    'own_count','enemy_count','own_hp_ratio_sum','enemy_hp_ratio_sum','own_area','enemy_area')
SEMANTIC_CONTRACT = dict(version=SEMANTIC_VERSION,static=tuple(STATIC_SCALES.items()),
    dynamic=tuple(DYNAMIC_SCALES.items()),effects=EFFECT_KINDS,relations=RELATION_KINDS,
    entity_kinds=ENTITY_KINDS,groups=GROUP_KINDS,planes=PLANE_NAMES,
    geometry='normalized_actor_positions;tile_radius_or_half_extents;bilinear_points.v1',
    command_features='issued_age,observed_age,card_known,x,y,target_known,cost,cost_known,kind_known')
SEMANTIC_CONTRACT['firstlight_features']=REFERENCE_FEATURES
SEMANTIC_CONTRACT['lifecycle']=LIFECYCLE_FEATURE_NAMES
SEMANTIC_CONTRACT['projectile_damage_basis']='native_effective_when_available_else_explicit_nominal_level11_v1'


def finite(value, name: str, minimum=None) -> float:
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    if minimum is not None and value < minimum:
        raise ValueError(f'{name} below minimum')
    return float(value)


def checked_features(items: tuple[tuple[str,float],...], scales: dict) -> None:
    if len(dict(items)) != len(items) or any(key not in scales for key,_ in items):
        raise ValueError('duplicate or unsupported semantic feature')
    for key,value in items:
        finite(value,key)
        if not key.startswith('velocity_') and value < 0:
            raise ValueError('negative semantic feature')


def reference_values(items,names):
    if len(dict(items))!=len(items) or any(key not in names for key,_ in items):
        raise ValueError('duplicate or unsupported full semantic feature')
    for key,value in items:finite(value,key)


@dataclass(frozen=True)
class MechanicOperation:
    kind: str
    magnitude: float | None = None
    duration_ms: float | None = None
    radius_tiles: float | None = None
    produced_card: int | None = None

    def __post_init__(self):
        if self.kind not in EFFECT_KINDS:
            raise ValueError('unknown mechanic operation kind')
        for name in ('magnitude','duration_ms','radius_tiles'):
            if getattr(self,name) is not None:
                finite(getattr(self,name),name,0)
        if self.produced_card is not None and (type(self.produced_card) is not int or self.produced_card <= 0):
            raise ValueError('invalid produced card')


@dataclass(frozen=True)
class CardProfile:
    card_id: int
    values: tuple[tuple[str,float],...] = ()
    operations: tuple[MechanicOperation,...] = ()

    def __post_init__(self):
        if type(self.card_id) is not int or self.card_id <= 0:
            raise ValueError('invalid profile card')
        checked_features(self.values,STATIC_SCALES)
        if len(self.operations)>32 or any(not isinstance(x,MechanicOperation) for x in self.operations):
            raise ValueError('invalid or oversized mechanic profile')


@dataclass(frozen=True)
class SemanticCatalog:
    profiles: tuple[CardProfile,...]
    source: str
    runtime_sha256: str

    def __post_init__(self):
        if not self.source or len(self.runtime_sha256)!=64 or any(c not in '0123456789abcdef' for c in self.runtime_sha256):
            raise ValueError('semantic catalog needs source/runtime identity')
        if len({x.card_id for x in self.profiles})!=len(self.profiles):
            raise ValueError('duplicate semantic profile')

    @property
    def sha256(self):
        return digest({'schema':SEMANTIC_VERSION,**asdict(self)})

    @classmethod
    def from_native(cls, vocabulary: CardVocabulary):
        rows,forms=catalog(),form_index()
        result=[]
        for card in vocabulary.native_ids:
            form=forms.get(card)
            row=rows.get(vocabulary.base(card),{})
            # Cost/type of a form is not assumed to equal its base form.
            values={}
            if form is None:
                if row.get('elixir') is not None: values['cost']=float(row['elixir'])
                if row.get('type') in ('troop','building','spell'):
                    values.update({f'is_{kind}':float(row['type']==kind) for kind in ('troop','building','spell')})
                if row.get('evolution_cycles') is not None: values['evolution_cycles']=float(row['evolution_cycles'])
            prefix='hero_ability_' if form and form['card_form']=='hero' else 'active_ability_'
            if not form or form['card_form']=='hero':
                for target,key in [('ability_cost','mana_cost'),('ability_charges','max_charges'),('ability_cast_ms','cast_time_ms'),('ability_delay_ms','trigger_delay_ms'),('ability_cooldown_ms','cooldown_ms')]:
                    if row.get(prefix+key) is not None: values[target]=float(row[prefix+key])
            result.append(CardProfile(card,tuple(sorted(values.items()))))
        runtime='fa6704b83cb9c5b8eecb7b56c9671b834d636a3a6d9ac446e698e1262dc246ba'
        source='native_core/data/live_card_catalog.json:15.535.29'
        path=Path(__file__).with_name('data')/'nominal_combat.json'
        if not path.is_file():
            raise FileNotFoundError('rich R0 nominal combat table missing; do not silently fall back to a different catalog')
        supplement=json.loads(path.read_text(encoding='utf-8'))
        if supplement.get('schema')!='r0-nominal-combat.v1' or supplement.get('runtime_sha256')!=runtime:
            raise ValueError('nominal combat table runtime/schema mismatch')
        extra={int(card):tuple(sorted(values.items())) for card,values in supplement['cards'].items()}
        for values in extra.values():checked_features(values,STATIC_SCALES)
        result=[CardProfile(p.card_id,tuple(sorted({**dict(p.values),**dict(extra.get(p.card_id,()))}.items())),p.operations) for p in result]
        source+=';nominal_combat:'+digest(supplement)
        return cls(tuple(result),source,runtime)

    @classmethod
    def read(cls, path: str | Path):
        raw=json.loads(Path(path).read_text(encoding='utf-8'))
        if raw.get('schema')!=SEMANTIC_VERSION: raise ValueError('unsupported semantic catalog schema')
        profiles=tuple(CardProfile(int(r['card_id']),tuple(sorted(r['values'].items())),
            tuple(MechanicOperation(**op) for op in r.get('operations',()))) for r in raw['profiles'])
        return cls(profiles,raw['source'],raw['runtime_sha256'])


@dataclass(frozen=True)
class EntityDetail:
    key: int
    kind: str = 'unknown'
    group_id: str | None = None
    group_kind: str = 'unknown'
    radius_tiles: float | None = None
    half_extent_tiles: tuple[float,float] | None = None
    values: tuple[tuple[str,float],...] = ()
    archetype: str | None = None
    child_features: tuple[tuple[str,float],...] = ()
    tower_features: tuple[tuple[str,float],...] = ()
    tower_troop: str = 'unknown'
    threat: float | None = None
    damage: DamageSummary | None = None

    def __post_init__(self):
        if type(self.key) is not int or self.kind not in ENTITY_KINDS or self.group_kind not in GROUP_KINDS:
            raise ValueError('invalid public entity detail')
        if self.group_id is not None and (not isinstance(self.group_id,str) or not self.group_id):
            raise ValueError('invalid opaque group identity')
        if self.radius_tiles is not None: finite(self.radius_tiles,'radius',0)
        if self.half_extent_tiles is not None:
            if len(self.half_extent_tiles)!=2: raise ValueError('extent requires x/y')
            for v in self.half_extent_tiles: finite(v,'extent',0)
        if self.radius_tiles is not None and self.half_extent_tiles is not None:
            raise ValueError('choose circle or rectangle geometry')
        checked_features(self.values,DYNAMIC_SCALES)
        reference_values(self.child_features,CHILD_FEATURES)
        reference_values(self.tower_features,TOWER_FEATURES)
        if self.archetype is not None and (not isinstance(self.archetype,str) or not self.archetype):raise ValueError('invalid archetype')
        if self.tower_troop not in ('unknown','princess','cannoneer','dagger_duchess','royal_chef'):raise ValueError('unknown tower troop')
        if self.threat is not None:finite(self.threat,'threat',0)
        if self.damage is not None and (not isinstance(self.damage,DamageSummary) or self.damage.unit_key!=self.key):raise ValueError('lifecycle belongs to another unit')


@dataclass(frozen=True)
class PublicEffect:
    target_key: int
    kind: str
    observed_tick: int
    magnitude: float | None = None
    remaining_ms: float | None = None
    stacks: float | None = None
    source_card: int | None = None
    effect_name: str | None = None

    def __post_init__(self):
        if type(self.target_key) is not int or self.kind not in EFFECT_KINDS or type(self.observed_tick) is not int or self.observed_tick<0:
            raise ValueError('invalid public effect identity/time')
        for k in ('magnitude','remaining_ms','stacks'):
            if getattr(self,k) is not None: finite(getattr(self,k),k,0)
        if self.source_card is not None and (type(self.source_card) is not int or self.source_card<=0):
            raise ValueError('invalid effect source card')
        if self.effect_name is not None and (not isinstance(self.effect_name,str) or not self.effect_name):raise ValueError('invalid effect identity')


@dataclass(frozen=True)
class PublicRelation:
    source_key: int
    target_key: int
    kind: str
    observed_tick: int

    def __post_init__(self):
        if self.kind not in RELATION_KINDS[1:] or any(type(x) is not int for x in (self.source_key,self.target_key,self.observed_tick)) or self.observed_tick<0:
            raise ValueError('invalid public relation')


@dataclass(frozen=True)
class OwnCommand:
    """Confirmed previous or submitted pending command; canonical actor target."""
    command_id: str
    issued_tick: int
    observed_tick: int
    kind: str
    card_id: int | None = None
    target: tuple[float,float] | None = None  # normalized canonical x,y
    cost: float | None = None
    effective_card: int | None = None
    effective_form: int | None = None
    ability_name: str | None = None
    source_position: tuple[float,float] | None = None
    delay_offset_bin: int | None = None
    sequence_index: int | None = None

    def __post_init__(self):
        if not self.command_id or self.kind not in ('unknown','play','ability') or any(type(t) is not int for t in (self.issued_tick,self.observed_tick)) or not 0<=self.issued_tick<=self.observed_tick:
            raise ValueError('invalid own command')
        if self.target is not None and (len(self.target)!=2 or any(not 0<=finite(x,'command target')<=1 for x in self.target)):
            raise ValueError('command target outside canonical arena')
        if self.cost is not None and not 0<=finite(self.cost,'cost')<=10: raise ValueError('invalid command cost')
        if self.card_id is not None and (type(self.card_id) is not int or self.card_id<=0):
            raise ValueError('invalid command card')
        if self.effective_card is not None and (type(self.effective_card) is not int or self.effective_card<=0):raise ValueError('invalid effective card')
        if self.effective_form is not None and self.effective_form not in (0,1,2,3):raise ValueError('invalid effective form')
        if self.delay_offset_bin is not None and self.delay_offset_bin not in range(5):raise ValueError('invalid command offset')
        if self.sequence_index is not None and self.sequence_index not in (0,1):raise ValueError('invalid micro-action index')
        if self.source_position is not None and (len(self.source_position)!=2 or any(not 0<=finite(x,'source position')<=1 for x in self.source_position)):raise ValueError('invalid command source')


@dataclass(frozen=True)
class CombatEvent:
    event_id: str
    kind: str
    observed_tick: int
    side: int
    source_key: int | None = None
    target_key: int | None = None
    source_card: int | None = None
    source_form: int | None = None
    features: tuple[tuple[str,float],...] = ()

    def __post_init__(self):
        if not self.event_id or self.kind not in COMBAT_KINDS or type(self.observed_tick) is not int or self.observed_tick<0 or self.side not in (0,1):raise ValueError('invalid combat event')
        reference_values(self.features,COMBAT_FEATURES)
        if self.source_form is not None and self.source_form not in (0,1,2,3):raise ValueError('invalid event form')


@dataclass(frozen=True)
class CardRuntime:
    card_id: int
    relation: int
    features: tuple[tuple[str,float],...] = ()
    def __post_init__(self):
        if type(self.card_id) is not int or self.card_id<=0 or self.relation not in (0,1):raise ValueError('invalid public card runtime')
        reference_values(self.features,CARD_FEATURES)


@dataclass(frozen=True)
class GroupRuntime:
    group_id:str
    relation:int
    features:tuple[tuple[str,float],...]=()
    def __post_init__(self):
        if not self.group_id or self.relation not in (0,1):raise ValueError('invalid group runtime')
        reference_values(self.features,GROUP_FEATURES)


@dataclass(frozen=True)
class PublicSemantics:
    episode_uid: str
    actor_side: int
    tick: int
    entities: tuple[EntityDetail,...] = ()
    effects: tuple[PublicEffect,...] = ()
    relations: tuple[PublicRelation,...] = ()
    previous: tuple[OwnCommand,...] = ()
    pending: tuple[OwnCommand,...] = ()
    revealed_enemy_cards: tuple[int,...] = ()
    source: str = 'explicit-public-adapter.v1'
    combat_events: tuple[CombatEvent,...] = ()
    match_features: tuple[tuple[str,float],...] = ()
    card_runtime: tuple[CardRuntime,...] = ()
    group_runtime: tuple[GroupRuntime,...] = ()
    capture_capabilities: tuple[str,...] = ()

    def __post_init__(self):
        if not self.episode_uid or self.actor_side not in (0,1) or type(self.tick) is not int or self.tick<0 or not self.source:
            raise ValueError('semantic frame needs episode/side/tick/source')
        if len({e.key for e in self.entities})!=len(self.entities): raise ValueError('duplicate entity detail')
        if len(self.previous)>2: raise ValueError('previous action has at most two micro actions')
        commands=(*self.previous,*self.pending)
        if len({c.command_id for c in commands})!=len(commands): raise ValueError('duplicate or already-confirmed pending command')
        if any(x.observed_tick>self.tick for x in (*self.effects,*self.relations,*commands,*self.combat_events)):
            raise ValueError('future semantic event')
        if len(set(self.revealed_enemy_cards))!=len(self.revealed_enemy_cards) or any(type(c) is not int or c<=0 for c in self.revealed_enemy_cards):
            raise ValueError('invalid revealed enemy cards')
        reference_values(self.match_features,MATCH_FEATURES)
        if len({e.event_id for e in self.combat_events})!=len(self.combat_events):raise ValueError('duplicate combat event')
        if len({(c.relation,c.card_id) for c in self.card_runtime})!=len(self.card_runtime):raise ValueError('duplicate card runtime')
        if len({(g.relation,g.group_id) for g in self.group_runtime})!=len(self.group_runtime):raise ValueError('duplicate group runtime')
