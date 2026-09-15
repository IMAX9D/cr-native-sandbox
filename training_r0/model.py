"""Local R0 relation/spatial/LSTM actor-value and joint two-action decoder.

New implementation of the reference method, not a checkpoint-compatible clone.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from .actions import ActionSequence, DecisionEnvelope, ShadowLegality
from .catalog import CardVocabulary
from .config import CANDIDATE_FEATURES, ENTITY_FEATURES, EVENT_FEATURES, PUBLIC_SCALARS, ModelConfig, Temperatures, digest
from .observation import ObservationBuilder, PublicBatch
from .semantics import SemanticCatalog, REFERENCE_COMMIT
from .components import CardSemanticEncoder, RichEntityEncoder, SpatialEncoder, RelationBlock, spatial_pairs
from .native_graph import NativeGraphEncoder
from .elite_encoders import EliteEncoders,LearnedSetPool,LogitHead


@dataclass(frozen=True)
class RecurrentState:
    hidden: Tensor
    cell: Tensor

    def detach(self) -> 'RecurrentState':
        return RecurrentState(self.hidden.detach(), self.cell.detach())

    def to(self, device) -> 'RecurrentState':
        return RecurrentState(self.hidden.to(device), self.cell.to(device))


@dataclass(frozen=True)
class PolicyContext:
    policy: Tensor
    tokens: Tensor
    token_mask: Tensor
    spatial: Tensor
    candidates: Tensor
    value: Tensor
    next_state: RecurrentState


@dataclass(frozen=True)
class PolicyOutput:
    actions: ActionSequence
    logp: Tensor
    entropy: Tensor  # Sampled-path conditional regularizer, not exact tree entropy.
    value: Tensor
    next_state: RecurrentState
    logp_parts: dict[str, Tensor]
    conservative_geometry: Tensor
    decision: DecisionEnvelope


def mlp(source: int, target: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(source, target), nn.SiLU(), nn.LayerNorm(target))


def mean_masked(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(1) / weights.sum(1).clamp_min(1)


class R0Policy(nn.Module):
    def __init__(self, vocabulary: CardVocabulary, config: ModelConfig | None = None, semantic_catalog: SemanticCatalog | None = None):
        super().__init__()
        self.config = c = config or ModelConfig()
        from dataclasses import asdict
        semantic_catalog=semantic_catalog or SemanticCatalog.from_native(vocabulary)
        self.observation_schema_hash = ObservationBuilder(vocabulary, c, semantic_catalog).schema_hash
        self.model_schema_hash = digest({'model':'r0-elite-relation-spatial-lstm.v3','config':asdict(c),'vocabulary':vocabulary.sha256,'semantics':semantic_catalog.sha256,'observation':self.observation_schema_hash})
        self.reference_commit=REFERENCE_COMMIT
        self.semantic_catalog_sha256=semantic_catalog.sha256
        self.register_buffer('_contract',torch.tensor(list(bytes.fromhex(self.model_schema_hash)),dtype=torch.uint8))
        d, channels = c.width, c.spatial_channels
        self.cards = CardSemanticEncoder(vocabulary,semantic_catalog,d,c.use_mechanics)
        self.forms = nn.Embedding(5,d)
        self.relations = nn.Embedding(2,d)
        self.kinds = nn.Embedding(2,d)
        self.event_kinds = nn.Embedding(4,d,padding_idx=0)
        self.entity_encoder = mlp(2*d+ENTITY_FEATURES,d)
        self.candidate_encoder = mlp(3*d+CANDIDATE_FEATURES,d)
        self.event_encoder = mlp(3*d+EVENT_FEATURES,d)
        self.scalar_encoder = mlp(PUBLIC_SCALARS,d)
        self.hand_encoder = mlp(4*d,d)
        self.summary = mlp(5*d,d)
        self.rich = RichEntityEncoder(c)
        self.native_graph=NativeGraphEncoder(d,c.native_graph_layers)
        self.elite=EliteEncoders(c)
        self.token_kinds=nn.Embedding(5,d)
        self.relation_blocks = nn.ModuleList(RelationBlock(d,c.heads) for _ in range(c.layers))
        self.spatial = SpatialEncoder(c)
        self.event_attention=nn.MultiheadAttention(d,c.heads,dropout=0,batch_first=True)
        self.candidate_attention=nn.MultiheadAttention(d,c.heads,dropout=0,batch_first=True)
        self.candidate_source=nn.Linear(d,d)
        self.scalar_projection=mlp(d,d//2)
        self.event_projection=mlp(d,d//2)
        self.candidate_projection=mlp(d,d//2)
        self.core_input = mlp(d+5*(d//2),c.hidden)
        self.core = nn.LSTMCell(c.hidden,c.hidden)
        self.hidden_policy = nn.LayerNorm(c.hidden)
        self.scene_policy=nn.Linear(d,c.hidden)
        self.spatial_policy=nn.Linear(d//2,c.hidden)
        self.candidate_policy=nn.Linear(d//2,c.hidden)
        self.scene_value=nn.Linear(d,c.hidden)
        self.spatial_value=nn.Linear(d//2,c.hidden)
        self.scalar_value=nn.Linear(d//2,c.hidden)
        self.event_value=nn.Linear(d//2,c.hidden)
        self.policy_norm = nn.LayerNorm(c.hidden)
        self.value_norm = nn.LayerNorm(c.hidden)
        self.value_head = nn.Sequential(nn.Linear(c.hidden,d),nn.SiLU(),nn.Linear(d,1))
        self.cross_attention = nn.MultiheadAttention(c.hidden,c.heads,kdim=d,vdim=d,dropout=0,batch_first=True)
        self.decoder_norm = nn.LayerNorm(c.hidden)
        self.gate_head = LogitHead(c.hidden,d,2)
        self.decoder_initial=nn.Linear(c.hidden,c.hidden)
        self.candidate_query = nn.Linear(c.hidden,d)
        self.candidate_key=nn.Linear(d,d)
        self.location_condition=nn.Sequential(mlp(c.hidden+d,d),nn.Linear(d,d//2))
        self.location_film = nn.Linear(d//2,2*channels)
        self.location_key = nn.Conv2d(channels,d//2,1)
        self.location_query = nn.Sequential(mlp(c.hidden+d,d),nn.Linear(d,d//2))
        self.no_target_embedding=nn.Parameter(torch.randn(d//2)*.02)
        self.delay_head = nn.Sequential(mlp(c.hidden+d+d//2,d),nn.Linear(d,5))
        self.offset_embedding = nn.Embedding(5,32)
        self.action_embedding = nn.Sequential(mlp(d+d//2+32,d),nn.Linear(d,d//2))
        self.remaining_pool=LearnedSetPool(d,d//2)
        self.decoder_step=nn.Sequential(mlp(d+9,c.hidden),nn.Linear(c.hidden,c.hidden))
        self.continue_head = LogitHead(c.hidden+d+9,d,2)
        # Transparent initial prior. Trained probabilities and temperatures
        # are separate; this is not a deployment play-rate multiplier.
        nn.init.zeros_(self.gate_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias[1:2], math.log(.15/.85))
        nn.init.zeros_(self.continue_head.weight)
        nn.init.zeros_(self.continue_head.bias)
        nn.init.constant_(self.continue_head.bias[1:2],math.log(.25/.75))

    def initial_state(self, batch_size: int, *, device=None) -> RecurrentState:
        parameter = next(self.parameters())
        device = parameter.device if device is None else device
        return RecurrentState(torch.zeros(batch_size,self.config.hidden,device=device,dtype=parameter.dtype),
                              torch.zeros(batch_size,self.config.hidden,device=device,dtype=parameter.dtype))

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        stored=state_dict.get(prefix+'_contract')
        if stored is None or not torch.equal(stored.cpu(),self._contract.cpu()):
            raise RuntimeError('R0 model/observation/semantic catalog contract mismatch; legacy weights cannot be silently migrated')
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def _check_batch(self, batch: PublicBatch) -> None:
        if batch.schema_hash != self.observation_schema_hash:
            raise ValueError('model/observation vocabulary schema mismatch')
        if batch.entity_tokens.shape[1] > self.config.max_entities or batch.candidate_tokens.shape[1] > self.config.max_candidates:
            raise ValueError('observation exceeds configured capacity')
        if batch.semantic.elite is None:raise ValueError('elite feature bank missing')
        if not self.config.allow_incomplete_capture and not bool(batch.semantic.elite.capture_complete.all()):
            raise ValueError('elite capture streams incomplete; diagnostic-only models must explicitly allow incomplete capture')

    def encode(self, batch: PublicBatch, state: RecurrentState | None = None,
               *, episode_start: Tensor | None = None) -> PolicyContext:
        self._check_batch(batch)
        b = batch.batch_size
        state = self.initial_state(b) if state is None else state
        if state.hidden.shape != (b,self.config.hidden) or state.cell.shape != state.hidden.shape:
            raise ValueError('recurrent state belongs to another batch/model')
        if episode_start is not None:
            if episode_start.shape != (b,) or episode_start.dtype != torch.bool:
                raise ValueError('episode_start must be bool [B]')
            state = RecurrentState(torch.where(episode_start[:,None],0.,state.hidden), torch.where(episode_start[:,None],0.,state.cell))
        native_memory=self.native_graph()
        card_memory=self.cards.memory(native_memory if self.config.use_mechanics else None)
        entity = self.entity_encoder(torch.cat((card_memory[batch.entity_tokens],self.relations(batch.entity_relations),batch.entity_features),-1))
        entity=self.rich.entities(entity,batch,card_memory)
        entity=self.elite.entities(entity,batch,native_memory)
        candidate = self.candidate_encoder(torch.cat((card_memory[batch.candidate_tokens],self.kinds(batch.candidate_kinds),self.forms(batch.candidate_forms),batch.candidate_features),-1))
        sources=batch.semantic.candidate_sources
        candidate=candidate+self.candidate_source(entity.gather(1,sources.clamp_min(0)[:,:,None].expand(-1,-1,self.config.width)))*(sources>=0)[:,:,None]
        advanced=batch.semantic.elite
        candidate=candidate+self.elite.candidate_effective(card_memory[advanced.candidate_effective])+self.elite.candidate_ability(torch.cat((native_memory[advanced.candidate_abilities],advanced.candidate_ability_features),-1))
        # Identity/mask is routing; the UID's numerical value is never encoded.
        candidate_summary = mean_masked(candidate,batch.candidate_uids >= 0)
        events = self.event_encoder(torch.cat((card_memory[batch.event_tokens],self.event_kinds(batch.event_kinds),self.relations(batch.event_relations),batch.event_features),-1))
        event_summary = mean_masked(events,batch.event_mask)
        hand = card_memory[batch.hand_tokens] + self.forms(batch.hand_forms)
        hand = hand * (batch.hand_tokens > 0)[:,:,None]
        own_cards=card_memory[batch.deck_tokens]+self.forms(batch.deck_forms)
        deck = mean_masked(own_cards,batch.deck_tokens>0)
        own = self.hand_encoder(hand.flatten(1)) + deck + card_memory[batch.next_tokens]
        scalar = self.scalar_encoder(batch.scalars)
        summary = self.summary(torch.cat((scalar,own,candidate_summary,event_summary,mean_masked(entity,batch.entity_mask)),-1))
        groups,group_mask,group_pos,group_extent,membership=self.rich.groups(entity,batch)
        if self.config.use_groups:groups=groups+self.elite.group(advanced.groups)*group_mask[:,:,None]
        in_hand=(batch.deck_tokens[:,:,None]==batch.hand_tokens[:,None]).any(-1)
        is_next=batch.deck_tokens==batch.next_tokens[:,None]
        runtime=torch.stack((in_hand,is_next,batch.deck_tokens>0,torch.ones_like(in_hand)),-1).to(entity.dtype)
        own_cards=own_cards+self.rich.card_runtime(runtime)+self.elite.card_runtime(advanced.own_cards)+self.token_kinds.weight[1]
        enemy=card_memory[batch.semantic.enemy_tokens]+self.elite.card_runtime(advanced.enemy_cards)+self.token_kinds.weight[2]
        towers=entity+self.token_kinds.weight[3]
        tower_mask=batch.entity_mask & (batch.entity_features[:,:,6]==0)
        if bool((tower_mask.sum(-1)>self.config.max_towers).any()):raise ValueError('tower capacity exceeded')
        ranks=(tower_mask.long().cumsum(-1)-1).clamp_min(0)
        packed_towers=entity.new_zeros(b,self.config.max_towers,self.config.width).scatter_add(1,ranks[:,:,None].expand_as(towers),towers*tower_mask[:,:,None])
        packed_positions=entity.new_zeros(b,self.config.max_towers,2).scatter_add(1,ranks[:,:,None].expand(-1,-1,2),batch.entity_positions*tower_mask[:,:,None])
        packed_mask=torch.arange(self.config.max_towers,device=entity.device)[None]<tower_mask.sum(-1)[:,None]
        tokens=torch.cat((summary[:,None]+self.token_kinds.weight[0],own_cards,enemy,packed_towers,groups+self.token_kinds.weight[4]),1)
        mask=torch.cat((torch.ones(b,1,dtype=torch.bool,device=entity.device),batch.deck_tokens>0,batch.semantic.enemy_mask,packed_mask,group_mask),1)
        tower_start=1+batch.deck_tokens.shape[1]+enemy.shape[1]
        group_start=tower_start+self.config.max_towers
        positions=torch.cat((entity.new_zeros(b,tower_start,2),packed_positions,group_pos),1)
        extent=torch.cat((entity.new_zeros(b,tower_start,4),torch.cat((packed_positions,packed_positions),-1),group_extent),1)
        spatial_mask=torch.cat((torch.zeros(b,tower_start,dtype=torch.bool,device=entity.device),packed_mask,group_mask),1)
        pairs=spatial_pairs(positions,extent,spatial_mask)
        group_index=membership.long().argmax(1)
        entity_tokens=torch.where(tower_mask,ranks+tower_start,group_index+group_start)
        s=batch.semantic
        edges=(entity_tokens.gather(1,s.edge_sources),entity_tokens.gather(1,s.edge_targets),s.edge_types,s.edge_mask) if self.config.use_relation_edges else None
        for block in self.relation_blocks:
            tokens = block(tokens,mask,pairs,edges)
        scene = tokens[:,0]
        spatial,spatial_summary=self.spatial(entity,batch)
        event_update,_=self.event_attention(events,tokens,tokens,key_padding_mask=~mask,need_weights=False)
        candidate_update,_=self.candidate_attention(candidate,tokens,tokens,key_padding_mask=~mask,need_weights=False)
        candidate=candidate+candidate_update
        candidate_summary=self.candidate_projection(mean_masked(candidate,batch.candidate_uids>=0))
        event_summary=self.event_projection(mean_masked(events+event_update,batch.event_mask))
        event_summary=self.elite.events(batch,card_memory,tokens,entity_tokens,event_summary)
        scalar_summary=self.scalar_projection(scalar)+self.elite.match(advanced.match)
        command_summary=self.rich.commands(batch,card_memory)
        if self.config.use_command_memory:command_summary=self.elite.commands(batch,card_memory,native_memory,command_summary)
        core_input = self.core_input(torch.cat((scene,spatial_summary,scalar_summary,event_summary,command_summary,candidate_summary),-1))
        hidden,cell = self.core(core_input,(state.hidden,state.cell))
        context = self.policy_norm(self.hidden_policy(hidden)+self.scene_policy(scene)+self.spatial_policy(spatial_summary)+self.candidate_policy(candidate_summary))
        value_context=self.value_norm(self.hidden_policy(hidden)+self.scene_value(scene)+self.spatial_value(spatial_summary)+self.scalar_value(scalar_summary)+self.event_value(event_summary))
        return PolicyContext(context,tokens,mask,spatial,candidate,self.value_head(value_context).squeeze(-1),RecurrentState(hidden,cell))

    def _context(self, query: Tensor, context: PolicyContext) -> Tensor:
        value,_ = self.cross_attention(query[:,None],context.tokens,context.tokens,key_padding_mask=~context.token_mask,need_weights=False)
        return self.decoder_norm(query+value[:,0])

    @staticmethod
    def _distribution(logits: Tensor, mask: Tensor, temperature: float) -> Categorical:
        if logits.shape != mask.shape or mask.dtype != torch.bool:
            raise ValueError('distribution mask shape/dtype mismatch')
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError('nonfinite policy logits')
        safe = mask.clone()
        safe[:,0] |= ~safe.any(-1)  # Used only for inactive conditional branches.
        return Categorical(logits=(logits.float()/temperature).masked_fill(~safe,-torch.inf))

    @staticmethod
    def _choice(distribution: Categorical, sample: bool, generator=None) -> Tensor:
        return torch.multinomial(distribution.probs,1,generator=generator).squeeze(-1) if sample else distribution.probs.argmax(-1)

    @staticmethod
    def _selected(distribution: Categorical, index: Tensor, mask: Tensor, active: Tensor) -> Tensor:
        if bool((active & ~mask.gather(1,index[:,None])[:,0]).any()):
            raise ValueError('stored action is illegal under recorded observation/shadow state')
        return torch.where(active,distribution.log_prob(index),torch.zeros_like(distribution.logits[:,0]))

    def gate_distribution(self, batch: PublicBatch, context: PolicyContext, temperatures: Temperatures) -> Categorical:
        legal = ShadowLegality(batch).candidate_mask(0).any(-1)
        return self._distribution(self.gate_head(context.policy),torch.stack((torch.ones_like(legal),legal),-1),temperatures.gate)

    def decode(self, batch: PublicBatch, context: PolicyContext, *,
               temperatures: Temperatures = Temperatures(), sample: bool = True,
               forced: ActionSequence | None = None, generator=None) -> PolicyOutput:
        b, c = batch.candidate_uids.shape
        device = batch.elixir.device
        if forced is not None:
            forced.validate(b)
        shadow = ShadowLegality(batch)
        gate_mask = torch.stack((torch.ones(b,dtype=torch.bool,device=device),shadow.candidate_mask(0).any(-1)),-1)
        gate_dist = self.gate_distribution(batch,context,temperatures)
        gate = (forced.count>0).long() if forced is not None else self._choice(gate_dist,sample,generator)
        gate_logp = self._selected(gate_dist,gate,gate_mask,torch.ones(b,dtype=torch.bool,device=device))
        active = gate.bool()
        count = active.long()
        uids = torch.full((b,2),-1,dtype=torch.long,device=device)
        targets, offsets = uids.clone(),uids.clone()
        parts = {'gate':gate_logp}
        entropy = gate_dist.entropy()
        decoder_base=self.decoder_initial(context.policy)
        decoder = self._context(decoder_base,context)
        rows = torch.arange(b,device=device)
        for step in range(2):
            legal = shadow.candidate_mask(step)
            if bool((active & ~legal.any(-1)).any()):
                raise ValueError('active micro action has no legal candidate')
            candidate_dist = self._distribution(torch.einsum('bd,bcd->bc',self.candidate_query(decoder),self.candidate_key(context.candidates))/math.sqrt(self.config.width),legal,temperatures.action)
            if forced is None:
                selected = self._choice(candidate_dist,sample,generator)
            else:
                matches = (batch.candidate_uids == forced.candidate_uid[:,step,None]) & legal
                if bool((active & (matches.sum(-1)!=1)).any()):
                    raise ValueError('recorded candidate UID is missing, duplicated or illegal')
                selected = matches.long().argmax(-1)
            parts[f'candidate{step}'] = self._selected(candidate_dist,selected,legal,active)
            chosen = context.candidates[rows,selected]
            condition = torch.cat((decoder,chosen),-1)
            gamma,beta = self.location_film(self.location_condition(condition)).chunk(2,-1)
            keys = self.location_key(context.spatial*(1+gamma[:,:,None,None])+beta[:,:,None,None])
            logits = torch.einsum('bd,bdhw->bhw',self.location_query(condition),keys).flatten(1)/math.sqrt(self.config.width//2)
            placement = shadow.placement(selected)
            grid = active & batch.grid_targets[rows,selected]
            location_dist = self._distribution(logits,placement,temperatures.action)
            target = forced.target_cell[:,step].clamp_min(0) if forced is not None else self._choice(location_dist,sample,generator)
            if forced is not None and bool((grid & (forced.target_cell[:,step] < 0)).any()):
                raise ValueError('stored spatial action is missing its target; never fill cell zero')
            if forced is not None and bool((active & ~grid & (forced.target_cell[:,step]!=-1)).any()):
                raise ValueError('nonspatial action has a stored grid target')
            parts[f'target{step}'] = self._selected(location_dist,target,placement,grid)
            target_feature = keys.flatten(2).gather(2,target[:,None,None].expand(-1,self.config.width//2,1))[:,:,0]
            target_feature = torch.where(grid[:,None],target_feature,self.no_target_embedding[None])
            delay_mask = shadow.offset_mask(step)
            delay_dist = self._distribution(self.delay_head(torch.cat((decoder,chosen,target_feature),-1)),delay_mask,temperatures.action)
            offset = forced.offset_bin[:,step].clamp_min(0) if forced is not None else self._choice(delay_dist,sample,generator)
            parts[f'offset{step}'] = self._selected(delay_dist,offset,delay_mask,active)
            uids[:,step] = torch.where(active,batch.candidate_uids[rows,selected],-1)
            targets[:,step] = torch.where(grid,target,-1)
            offsets[:,step] = torch.where(active,offset,-1)
            entropy = entropy + torch.where(active,candidate_dist.entropy()+delay_dist.entropy(),0.) + torch.where(grid,location_dist.entropy(),0.)
            if step == 0:
                shadow.apply(active,selected,target,offset)
                action_embedding = self.action_embedding(torch.cat((chosen,target_feature,self.offset_embedding(offset)),-1))
                remaining=self.remaining_pool(context.candidates,shadow.candidate_mask(1))
                autoregressive=torch.cat((action_embedding,shadow.features(1),remaining),-1)
                cont_mask = torch.stack((torch.ones_like(active),shadow.candidate_mask(1).any(-1)),-1)
                cont_dist = self._distribution(self.continue_head(torch.cat((decoder,autoregressive),-1)),cont_mask,temperatures.continuation)
                continuation = (forced.count==2).long() if forced is not None else self._choice(cont_dist,sample,generator)
                parts['continue'] = self._selected(cont_dist,continuation,cont_mask,active)
                entropy = entropy + torch.where(active,cont_dist.entropy(),0.)
                active = active & continuation.bool()
                count = count + active.long()
                decoder = self._context(decoder_base+self.decoder_step(autoregressive),context)
        actions = ActionSequence(count,uids,targets,offsets)
        actions.validate(b)
        decision = DecisionEnvelope(actions,tuple(zip(batch.episode_uids,batch.sides,batch.ticks)),batch.schema_hash)
        return PolicyOutput(actions,torch.stack(tuple(parts.values())).sum(0),entropy,context.value,context.next_state,parts,shadow.conservative_geometry,decision)

    def forward(self, batch: PublicBatch, state: RecurrentState | None = None, *,
                episode_start: Tensor | None = None, temperatures: Temperatures = Temperatures(),
                sample: bool = True, forced: ActionSequence | None = None, generator=None) -> PolicyOutput:
        context = self.encode(batch,state,episode_start=episode_start)
        return self.decode(batch,context,temperatures=temperatures,sample=sample,forced=forced,generator=generator)
