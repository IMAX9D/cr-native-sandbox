"""Public-only tensor adapter; placement candidates must be supplied explicitly.

Supports coherent observe_train_v1 frames. It does not invent placement masks,
targeted-ability support, or native same-offset execution semantics.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Mapping, Sequence

import torch
from torch import Tensor

from expert_v1.tick_store_v1.schema import ActorTick, actor_projection, normalize_native_state
from .catalog import CardVocabulary
from .config import CANDIDATE_FEATURES, CANDIDATE_FEATURE_NAMES, CELLS, ENTITY_FEATURES, ENTITY_FEATURE_NAMES, EVENT_FEATURES, EVENT_FEATURE_NAMES, HEIGHT, OBSERVATION_SCHEMA, PUBLIC_SCALARS, PUBLIC_SCALAR_NAMES, WIDTH, ModelConfig, digest
from .history import ConfirmedEvent, EventHistory


@dataclass(frozen=True)
class Candidate:
    uid: int
    card_id: int
    kind: str  # play or ability
    cost: float
    available: bool
    target_mode: str  # grid or none
    placement: tuple[bool, ...]
    hand_slot: int = -1
    source_entity: int = -1
    form_flags: int | None = None  # Observed form, not merely deck's enabled form.
    first_only: bool = False
    exclusion_group: int = -1
    is_building: bool = False
    footprint_half_tiles: tuple[float, float] | None = None
    placement_offset_tiles: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self):
        if type(self.card_id) is not int or self.card_id <= 0:
            raise ValueError('candidate needs a native card identity')
        if any(type(value) is not bool for value in (self.available,self.first_only,self.is_building)):
            raise ValueError('candidate flags must be explicit booleans')
        if type(self.uid) is not int or not 0 <= self.uid < 2**63:
            raise ValueError('candidate UID must be an opaque nonnegative int64')
        if self.kind not in ('play', 'ability') or self.target_mode not in ('grid', 'none'):
            raise ValueError('unknown candidate kind/target mode')
        if not math.isfinite(self.cost) or not 0 <= self.cost <= 10:
            raise ValueError('candidate cost outside 0..10')
        if len(self.placement) != CELLS or any(type(x) is not bool for x in self.placement):
            raise ValueError('explicit canonical 32x18 boolean placement mask required')
        if self.kind == 'play' and (self.hand_slot not in range(4) or self.target_mode != 'grid'):
            raise ValueError('play candidate requires a real hand slot and grid target')
        if self.kind == 'ability' and self.source_entity < 0:
            raise ValueError('ability candidate requires source entity identity')
        if self.form_flags is not None and self.form_flags not in (0, 1, 2, 3):
            raise ValueError('invalid form flag')
        if any(not math.isfinite(x) for x in self.placement_offset_tiles):
            raise ValueError('invalid placement offset')
        if self.footprint_half_tiles is not None and (not self.is_building or any(not math.isfinite(x) or x <= 0 for x in self.footprint_half_tiles)):
            raise ValueError('invalid building footprint')
        if len(self.placement_offset_tiles) != 2 or self.footprint_half_tiles is not None and len(self.footprint_half_tiles) != 2:
            raise ValueError('geometry requires x/y pairs')


@dataclass(frozen=True)
class PublicFrame:
    episode_uid: str
    view: ActorTick
    own_deck: tuple[int, ...]
    enabled_deck_forms: tuple[int, ...]
    candidates: tuple[Candidate, ...]
    events: tuple[ConfirmedEvent, ...]
    commands_allowed: bool
    native_entity_count: int
    source: str = 'native_observe_train_v1'


@dataclass(frozen=True)
class PublicBatch:
    entity_tokens: Tensor
    entity_features: Tensor
    entity_positions: Tensor
    entity_relations: Tensor
    entity_mask: Tensor
    grid: Tensor
    scalars: Tensor
    hand_tokens: Tensor
    hand_forms: Tensor
    deck_tokens: Tensor
    deck_forms: Tensor
    next_tokens: Tensor
    candidate_tokens: Tensor
    candidate_kinds: Tensor
    candidate_forms: Tensor
    candidate_features: Tensor
    candidate_uids: Tensor  # Routing only; never passed to a learnable encoder.
    candidate_mask: Tensor
    placement: Tensor
    costs: Tensor
    grid_targets: Tensor
    hand_slots: Tensor
    first_only: Tensor
    exclusion_groups: Tensor
    buildings: Tensor
    footprint_known: Tensor
    footprint_half: Tensor
    placement_offsets: Tensor
    elixir: Tensor
    event_tokens: Tensor
    event_kinds: Tensor
    event_relations: Tensor
    event_features: Tensor
    event_mask: Tensor
    episode_uids: tuple[str, ...]
    sides: tuple[int, ...]
    ticks: tuple[int, ...]
    schema_hash: str

    @property
    def batch_size(self) -> int:
        return self.scalars.shape[0]

    def to(self, device: torch.device | str) -> 'PublicBatch':
        return PublicBatch(**{field.name: getattr(self, field.name).to(device) if isinstance(getattr(self, field.name), Tensor) else getattr(self, field.name) for field in fields(self)})


class ObservationBuilder:
    def __init__(self, vocabulary: CardVocabulary, config: ModelConfig):
        self.vocabulary = vocabulary
        self.config = config
        self.schema_hash = digest({'schema': OBSERVATION_SCHEMA, 'vocabulary': vocabulary.sha256,
                                   'features': [ENTITY_FEATURE_NAMES, CANDIDATE_FEATURE_NAMES, EVENT_FEATURE_NAMES, PUBLIC_SCALAR_NAMES],
                                   'orientation':'native_actor_projection_17999_31999',
                                   'candidate_rules':'conservative_shadow.v1', 'unknown_form_code':4,
                                   'height': HEIGHT, 'width': WIDTH})

    def from_native(self, raw: Mapping, *, episode_uid: str, actor_side: int,
                    own_deck: Sequence[int | Mapping], candidates: Sequence[Candidate],
                    history: EventHistory | None = None) -> PublicFrame:
        if not episode_uid or raw.get('kind') != 'libg_native_train_state_v1' or raw.get('coherent') is not True:
            raise ValueError('R0 requires an identified coherent observe_train_v1 frame')
        if 'entities' not in raw or type(raw.get('entity_count')) is not int or raw['entity_count'] != len(raw['entities']):
            raise ValueError('native entity count differs from actual payload; no empty fallback')
        normalized = normalize_native_state(raw)
        view = actor_projection(normalized, actor_side=actor_side)
        deck, forms = [], []
        for item in own_deck:
            if isinstance(item, Mapping):
                card = int(item.get('card_id', item.get('d', -1)))
                form = int(item.get('form_flags', item.get('el', 0)))
            else:
                card, form = int(item), 0
            if self.vocabulary.token(card) == 1 or form not in (0, 1, 2, 3):
                raise ValueError('deck is outside frozen vocabulary/form schema')
            deck.append(card)
            forms.append(form)
        if len(deck) != 8 or len({self.vocabulary.base(card) for card in deck}) != 8:
            raise ValueError('own eight-card deck must be explicit and unique')
        if history is not None and history.episode_uid != episode_uid:
            raise ValueError('history belongs to a different episode')
        entity_by_id = {entity.key: entity for entity in normalized.entities}
        for item in candidates:
            if item.kind == 'play':
                deck_index = view.own_player.hand[item.hand_slot]
                if deck_index < 0 or self.vocabulary.base(item.card_id) != self.vocabulary.base(deck[deck_index]):
                    raise ValueError('candidate card does not match the actual current hand')
                if self.vocabulary.base(item.card_id) == 28000006 and not item.first_only:
                    raise ValueError('Mirror must use its certified dynamic cost and first-only contract')
            else:
                entity = entity_by_id.get(item.source_entity)
                if entity is None or entity.side != actor_side:
                    raise ValueError('ability source is not a current own entity')
                if self.vocabulary.base(item.card_id) != self.vocabulary.base(entity.card_id):
                    raise ValueError('ability candidate identity differs from source entity')
                if item.available and (not entity.ability_available or entity.ability_mana_cost < 0 or abs(item.cost - entity.ability_mana_cost) > 1e-6):
                    raise ValueError('ability availability/cost differs from native own state')
        return PublicFrame(episode_uid, view, tuple(deck), tuple(forms), tuple(candidates),
                           history.snapshot(view.tick) if history is not None else (),
                           bool(normalized.episode.commands_allowed) and not bool(view.episode.terminated), len(normalized.entities))

    def batch(self, frames: Sequence[PublicFrame]) -> PublicBatch:
        if not frames:
            raise ValueError('empty frame batch')
        b = len(frames)
        n = max(1, max(len(frame.view.entities) + len(frame.view.towers) for frame in frames))
        c = max(1, max(len(frame.candidates) for frame in frames))
        h = max(1, max(len(frame.events) for frame in frames))
        if n > self.config.max_entities or c > self.config.max_candidates or h > self.config.max_history:
            raise OverflowError('R0 capacity exceeded; expand the bucket, never silently drop data')
        def zeros(*shape, dtype=torch.float32):
            return torch.zeros(shape, dtype=dtype)
        long, boolean = torch.long, torch.bool
        tensors = dict(
            entity_tokens=zeros(b,n,dtype=long), entity_features=zeros(b,n,ENTITY_FEATURES),
            entity_positions=zeros(b,n,2), entity_relations=zeros(b,n,dtype=long), entity_mask=zeros(b,n,dtype=boolean),
            grid=zeros(b,4,HEIGHT,WIDTH), scalars=zeros(b,PUBLIC_SCALARS),
            hand_tokens=zeros(b,4,dtype=long), hand_forms=zeros(b,4,dtype=long),
            deck_tokens=zeros(b,8,dtype=long), deck_forms=zeros(b,8,dtype=long), next_tokens=zeros(b,dtype=long),
            candidate_tokens=zeros(b,c,dtype=long), candidate_kinds=zeros(b,c,dtype=long), candidate_forms=torch.full((b,c),4,dtype=long),
            candidate_features=zeros(b,c,CANDIDATE_FEATURES), candidate_uids=torch.full((b,c),-1,dtype=long),
            candidate_mask=zeros(b,c,dtype=boolean), placement=zeros(b,c,CELLS,dtype=boolean), costs=zeros(b,c),
            grid_targets=zeros(b,c,dtype=boolean), hand_slots=torch.full((b,c),-1,dtype=long), first_only=zeros(b,c,dtype=boolean),
            exclusion_groups=torch.full((b,c),-1,dtype=long), buildings=zeros(b,c,dtype=boolean),
            footprint_known=zeros(b,c,dtype=boolean), footprint_half=zeros(b,c,2), placement_offsets=zeros(b,c,2),
            elixir=zeros(b), event_tokens=zeros(b,h,dtype=long), event_kinds=zeros(b,h,dtype=long),
            event_relations=zeros(b,h,dtype=long), event_features=zeros(b,h,EVENT_FEATURES), event_mask=zeros(b,h,dtype=boolean),
        )
        vocab = self.vocabulary
        for i, frame in enumerate(frames):
            view = frame.view
            if not frame.episode_uid or frame.native_entity_count != len(view.entities):
                raise ValueError('public entity projection lost native entities')
            if len({item.uid for item in frame.candidates}) != len(frame.candidates):
                raise ValueError('candidate UIDs must be unique within a frame')
            if len(frame.own_deck) != 8 or len(frame.enabled_deck_forms) != 8:
                raise ValueError('frame deck shape mismatch')
            elixir = view.own_player.elixir_raw / 10000.0
            if not math.isfinite(elixir) or not 0 <= elixir <= 10:
                raise ValueError('own elixir outside 0..10')
            tensors['elixir'][i] = elixir
            tensors['scalars'][i] = torch.tensor([elixir/10, view.tick/6000, view.episode.own_crowns/3,
                view.episode.enemy_crowns/3, float(view.episode.terminated), float(frame.commands_allowed)])
            tensors['deck_tokens'][i] = torch.tensor([vocab.token(card) for card in frame.own_deck])
            tensors['deck_forms'][i] = torch.tensor(frame.enabled_deck_forms)
            for slot, index in enumerate(view.own_player.hand):
                if index >= 0:
                    tensors['hand_tokens'][i,slot] = vocab.token(frame.own_deck[index])
                    tensors['hand_forms'][i,slot] = frame.enabled_deck_forms[index]
            nxt = view.own_player.next_deck_index
            tensors['next_tokens'][i] = vocab.token(frame.own_deck[nxt]) if nxt >= 0 else 0
            rows = [(entity, 0, entity.relation, entity.card_id, entity.level) for entity in view.entities]
            rows += [(tower, 1 if tower.role == 0 else 2, tower.side, None, -1) for tower in view.towers]
            for j, (entity, kind, relation, card_id, level) in enumerate(rows):
                x, y = entity.x/18000, entity.y/32000
                known_hp = entity.max_hp > 0 and 0 <= entity.hp <= entity.max_hp
                if entity.max_hp > 0 and entity.hp > entity.max_hp:
                    raise ValueError('entity HP exceeds declared maximum')
                hp = entity.hp/entity.max_hp if known_hp else 0.0
                if not 0 <= x <= 1 or not 0 <= y <= 1:
                    raise ValueError('projected position outside arena')
                tensors['entity_tokens'][i,j] = vocab.token(card_id)
                tensors['entity_relations'][i,j] = relation
                tensors['entity_mask'][i,j] = True
                tensors['entity_positions'][i,j] = torch.tensor([x,y])
                tensors['entity_features'][i,j] = torch.tensor([x,y,hp,float(known_hp),max(0,level)/16,float(level>=0),float(kind==0),float(kind==1),float(kind==2),float(getattr(entity,'own_ability_available',0))])
                col, row = min(WIDTH-1,int(x*WIDTH)), min(HEIGHT-1,int(y*HEIGHT))
                tensors['grid'][i,relation,row,col] += 1
                tensors['grid'][i,relation+2,row,col] += hp
            own_entities = {item.key: item for item in view.entities if item.relation == 0}
            for j, item in enumerate(frame.candidates):
                source = own_entities.get(item.source_entity)
                source_xy = (source.x/18000,source.y/32000) if source is not None else (0.0,0.0)
                grid = item.target_mode == 'grid'
                available = frame.commands_allowed and item.available and item.cost <= elixir + 1e-6 and (not grid or any(item.placement))
                tensors['candidate_tokens'][i,j] = vocab.token(item.card_id)
                tensors['candidate_kinds'][i,j] = int(item.kind == 'ability')
                tensors['candidate_forms'][i,j] = 4 if item.form_flags is None else item.form_flags
                tensors['candidate_features'][i,j] = torch.tensor([item.cost/10, item.hand_slot/3,*source_xy,float(source is not None),float(grid),float(item.form_flags is not None),float(item.first_only)])
                tensors['candidate_uids'][i,j] = item.uid
                tensors['candidate_mask'][i,j] = available
                tensors['placement'][i,j] = torch.tensor(item.placement)
                tensors['costs'][i,j] = item.cost
                tensors['grid_targets'][i,j] = grid
                tensors['hand_slots'][i,j] = item.hand_slot
                tensors['first_only'][i,j] = item.first_only
                tensors['exclusion_groups'][i,j] = item.exclusion_group
                tensors['buildings'][i,j] = item.is_building
                tensors['footprint_known'][i,j] = item.footprint_half_tiles is not None
                tensors['footprint_half'][i,j] = torch.tensor(item.footprint_half_tiles or (0.,0.))
                tensors['placement_offsets'][i,j] = torch.tensor(item.placement_offset_tiles)
            for j, event in enumerate(frame.events):
                if event.observed_tick > view.tick:
                    raise ValueError('future event in public frame')
                target_known = event.native_target is not None
                x,y = event.native_target if target_known else (0,0)
                if target_known and view.actor_side == 1:
                    x,y = max(0,min(17999,17999-x)),max(0,min(31999,31999-y))
                tensors['event_tokens'][i,j] = vocab.token(event.card_id)
                tensors['event_kinds'][i,j] = {'unknown':1,'play':2,'ability':3}[event.kind]
                tensors['event_relations'][i,j] = int(event.side != view.actor_side)
                tensors['event_mask'][i,j] = True
                tensors['event_features'][i,j] = torch.tensor([(view.tick-event.observed_tick)/1200,float(event.card_id is not None),float(event.kind!='unknown'),x/18000 if target_known else 0,y/32000 if target_known else 0,float(target_known),1.0])
        return PublicBatch(**tensors, episode_uids=tuple(frame.episode_uid for frame in frames),
                           sides=tuple(frame.view.actor_side for frame in frames), ticks=tuple(frame.view.tick for frame in frames), schema_hash=self.schema_hash)
