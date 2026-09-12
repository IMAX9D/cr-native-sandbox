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
from .config import CANDIDATE_FEATURES, CELLS, ENTITY_FEATURES, EVENT_FEATURES, HEIGHT, PUBLIC_SCALARS, WIDTH, ModelConfig, Temperatures, digest
from .observation import ObservationBuilder, PublicBatch


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


class RelationBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0, batch_first=True)
        self.relation_bias = nn.Sequential(nn.Linear(5,32), nn.SiLU(), nn.Linear(32,heads))
        self.ffn = nn.Sequential(nn.Linear(width,4*width), nn.SiLU(), nn.Linear(4*width,width))

    def forward(self, tokens: Tensor, mask: Tensor, pair_features: Tensor) -> Tensor:
        b,n,_ = tokens.shape
        bias = self.relation_bias(pair_features).permute(0,3,1,2)
        bias = bias.masked_fill(~mask[:,None,None,:], -torch.inf).reshape(b*self.heads,n,n)
        normalized = self.norm1(tokens)
        attended, _ = self.attention(normalized, normalized, normalized, attn_mask=bias, need_weights=False)
        tokens = tokens + attended
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens * mask[:,:,None]


class R0Policy(nn.Module):
    def __init__(self, vocabulary: CardVocabulary, config: ModelConfig | None = None):
        super().__init__()
        self.config = c = config or ModelConfig()
        from dataclasses import asdict
        self.model_schema_hash = digest({'model':'r0-relation-spatial-lstm.v1','config':asdict(c),'vocabulary':vocabulary.sha256})
        self.observation_schema_hash = ObservationBuilder(vocabulary, c).schema_hash
        d, channels = c.width, c.spatial_channels
        self.cards = nn.Embedding(vocabulary.size, d, padding_idx=0)
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
        self.relation_blocks = nn.ModuleList(RelationBlock(d,c.heads) for _ in range(c.layers))
        self.entity_scatter = nn.Linear(d,channels)
        self.spatial = nn.Sequential(
            nn.Conv2d(channels+4,channels,3,padding=1), nn.SiLU(),
            nn.Conv2d(channels,channels,3,padding=1), nn.SiLU(),
            nn.Conv2d(channels,channels,3,padding=1), nn.SiLU(),
        )
        self.spatial_summary = nn.Sequential(nn.AdaptiveAvgPool2d((4,3)), nn.Flatten(), mlp(channels*12,d))
        self.core_input = mlp(4*d,c.hidden)
        self.core = nn.LSTMCell(c.hidden,c.hidden)
        self.hidden_policy = nn.Linear(c.hidden,d)
        self.policy_norm = nn.LayerNorm(d)
        self.value_head = nn.Sequential(nn.Linear(d,d),nn.SiLU(),nn.Linear(d,1))
        self.cross_attention = nn.MultiheadAttention(d,c.heads,dropout=0,batch_first=True)
        self.decoder_norm = nn.LayerNorm(d)
        self.gate_head = nn.Linear(d,2)
        self.candidate_query = nn.Linear(d,d)
        self.location_film = nn.Linear(2*d,2*channels)
        self.location_key = nn.Conv2d(channels,channels,1)
        self.location_query = nn.Linear(2*d,channels)
        self.delay_head = nn.Sequential(mlp(2*d+channels,d),nn.Linear(d,5))
        self.offset_embedding = nn.Embedding(5,d)
        self.action_embedding = mlp(2*d+channels,d)
        self.shadow_encoder = mlp(6,d)
        self.continue_head = nn.Linear(3*d,2)
        # Transparent initial prior. Trained probabilities and temperatures
        # are separate; this is not a deployment play-rate multiplier.
        nn.init.zeros_(self.gate_head.bias)
        nn.init.constant_(self.gate_head.bias[1:2], math.log(.15/.85))

    def initial_state(self, batch_size: int, *, device=None) -> RecurrentState:
        parameter = next(self.parameters())
        device = parameter.device if device is None else device
        return RecurrentState(torch.zeros(batch_size,self.config.hidden,device=device,dtype=parameter.dtype),
                              torch.zeros(batch_size,self.config.hidden,device=device,dtype=parameter.dtype))

    def _check_batch(self, batch: PublicBatch) -> None:
        if batch.schema_hash != self.observation_schema_hash:
            raise ValueError('model/observation vocabulary schema mismatch')
        if batch.entity_tokens.shape[1] > self.config.max_entities or batch.candidate_tokens.shape[1] > self.config.max_candidates:
            raise ValueError('observation exceeds configured capacity')

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
        entity = self.entity_encoder(torch.cat((self.cards(batch.entity_tokens),self.relations(batch.entity_relations),batch.entity_features),-1))
        candidate = self.candidate_encoder(torch.cat((self.cards(batch.candidate_tokens),self.kinds(batch.candidate_kinds),self.forms(batch.candidate_forms),batch.candidate_features),-1))
        # Identity/mask is routing; the UID's numerical value is never encoded.
        candidate_summary = mean_masked(candidate,batch.candidate_uids >= 0)
        events = self.event_encoder(torch.cat((self.cards(batch.event_tokens),self.event_kinds(batch.event_kinds),self.relations(batch.event_relations),batch.event_features),-1))
        event_summary = mean_masked(events,batch.event_mask)
        hand = self.cards(batch.hand_tokens) + self.forms(batch.hand_forms)
        hand = hand * (batch.hand_tokens > 0)[:,:,None]
        deck = mean_masked(self.cards(batch.deck_tokens)+self.forms(batch.deck_forms),batch.deck_tokens>0)
        own = self.hand_encoder(hand.flatten(1)) + deck + self.cards(batch.next_tokens)
        scalar = self.scalar_encoder(batch.scalars)
        summary = self.summary(torch.cat((scalar,own,candidate_summary,event_summary,mean_masked(entity,batch.entity_mask)),-1))
        tokens = torch.cat((summary[:,None],entity),1)
        mask = torch.cat((torch.ones(b,1,dtype=torch.bool,device=tokens.device),batch.entity_mask),1)
        positions = torch.cat((batch.entity_positions.new_zeros(b,1,2),batch.entity_positions),1)
        relations = torch.cat((torch.full((b,1),2,dtype=torch.long,device=tokens.device),batch.entity_relations),1)
        delta = positions[:,:,None]-positions[:,None,:]
        distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        same = (relations[:,:,None] == relations[:,None,:]).to(tokens.dtype)[:,:,:,None]
        pairs = torch.cat((delta,distance,same,1-same),-1)
        spatial_pair = batch.entity_mask[:,None,:] & batch.entity_mask[:,:,None]
        pair_mask = torch.zeros_like(mask[:,:,None] & mask[:,None,:])
        pair_mask[:,1:,1:] = spatial_pair
        pairs = pairs * pair_mask[:,:,:,None]
        for block in self.relation_blocks:
            tokens = block(tokens,mask,pairs)
        scene = tokens[:,0]
        features = self.entity_scatter(tokens[:,1:]) * batch.entity_mask[:,:,None]
        x = (batch.entity_positions[:,:,0]*WIDTH).long().clamp(0,WIDTH-1)
        y = (batch.entity_positions[:,:,1]*HEIGHT).long().clamp(0,HEIGHT-1)
        indices = y*WIDTH+x
        scatter = features.new_zeros(b,CELLS,self.config.spatial_channels).scatter_add(1,indices[:,:,None].expand_as(features),features)
        spatial = self.spatial(torch.cat((batch.grid,scatter.transpose(1,2).reshape(b,self.config.spatial_channels,HEIGHT,WIDTH)),1))
        spatial_summary = self.spatial_summary(spatial)
        core_input = self.core_input(torch.cat((scene,spatial_summary,candidate_summary,event_summary),-1))
        hidden,cell = self.core(core_input,(state.hidden,state.cell))
        context = self.policy_norm(self.hidden_policy(hidden)+scene+spatial_summary+candidate_summary)
        return PolicyContext(context,tokens,mask,spatial,candidate,self.value_head(context).squeeze(-1),RecurrentState(hidden,cell))

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
        decoder = self._context(context.policy,context)
        rows = torch.arange(b,device=device)
        for step in range(2):
            legal = shadow.candidate_mask(step)
            if bool((active & ~legal.any(-1)).any()):
                raise ValueError('active micro action has no legal candidate')
            candidate_dist = self._distribution(torch.einsum('bd,bcd->bc',self.candidate_query(decoder),context.candidates)/math.sqrt(self.config.width),legal,temperatures.action)
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
            gamma,beta = self.location_film(condition).chunk(2,-1)
            keys = self.location_key(context.spatial*(1+gamma[:,:,None,None])+beta[:,:,None,None])
            logits = torch.einsum('bd,bdhw->bhw',self.location_query(condition),keys).flatten(1)/math.sqrt(self.config.spatial_channels)
            placement = shadow.placement(selected)
            grid = active & batch.grid_targets[rows,selected]
            location_dist = self._distribution(logits,placement,temperatures.action)
            target = forced.target_cell[:,step].clamp_min(0) if forced is not None else self._choice(location_dist,sample,generator)
            if forced is not None and bool((grid & (forced.target_cell[:,step] < 0)).any()):
                raise ValueError('stored spatial action is missing its target; never fill cell zero')
            if forced is not None and bool((active & ~grid & (forced.target_cell[:,step]!=-1)).any()):
                raise ValueError('nonspatial action has a stored grid target')
            parts[f'target{step}'] = self._selected(location_dist,target,placement,grid)
            target_feature = keys.flatten(2).gather(2,target[:,None,None].expand(-1,self.config.spatial_channels,1))[:,:,0]
            target_feature = target_feature * grid[:,None]
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
                shadow_embedding = self.shadow_encoder(shadow.features(1))
                cont_mask = torch.stack((torch.ones_like(active),shadow.candidate_mask(1).any(-1)),-1)
                cont_dist = self._distribution(self.continue_head(torch.cat((decoder,action_embedding,shadow_embedding),-1)),cont_mask,temperatures.continuation)
                continuation = (forced.count==2).long() if forced is not None else self._choice(cont_dist,sample,generator)
                parts['continue'] = self._selected(cont_dist,continuation,cont_mask,active)
                entropy = entropy + torch.where(active,cont_dist.entropy(),0.)
                active = active & continuation.bool()
                count = count + active.long()
                decoder = self._context(context.policy+action_embedding+shadow_embedding,context)
        actions = ActionSequence(count,uids,targets,offsets)
        actions.validate(b)
        decision = DecisionEnvelope(actions,tuple(zip(batch.episode_uids,batch.sides,batch.ticks)),batch.schema_hash)
        return PolicyOutput(actions,torch.stack(tuple(parts.values())).sum(0),entropy,context.value,context.next_state,parts,shadow.conservative_geometry,decision)

    def forward(self, batch: PublicBatch, state: RecurrentState | None = None, *,
                episode_start: Tensor | None = None, temperatures: Temperatures = Temperatures(),
                sample: bool = True, forced: ActionSequence | None = None, generator=None) -> PolicyOutput:
        context = self.encode(batch,state,episode_start=episode_start)
        return self.decode(batch,context,temperatures=temperatures,sample=sample,forced=forced,generator=generator)
