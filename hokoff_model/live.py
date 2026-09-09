"""Fixed4 BC inference for the existing native Worker. No PPO updates."""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import numpy as np
import torch
from policy_v1.data import digest
from policy_v1.train import load_checkpoint
from expert_selfplay_v1.native_observation import (
    NativeObservationEncoder, NativeActorFrame, RevealedEnemyTracker,
)
from expert_v1.tick_store_v1.schema import normalize_native_state
from expert_v1.tick_store_v1.deployment_masks import (
    normalize_native_probe, derive_deployment_rows, MIRROR_CARD_ID,
)
from .fixed_data import check_mask_proof
from .train_fixed import FixedConfig, FixedPolicy


@dataclass
class LiveMasks:
    cards: np.ndarray
    positions: np.ndarray
    abilities: np.ndarray
    hand: tuple
    ability_keys: tuple
    card_cost_raw: np.ndarray


def canonical_position_to_native(position, side):
    if side not in (0,1) or not 0 <= position < 576:
        raise ValueError('invalid side/position')
    row, column = divmod(position,18)
    if side == 1: row, column = 31-row,17-column
    return column*1000+500,row*1000+500


class NativeMaskProvider:
    """The compiler's audited mask projection, with live native selection/costs."""
    def __init__(self, env):
        self.env = env
        self.probes = {}
        self.position_cache = {}

    def reset(self):
        self.probes.clear()
        self.position_cache.clear()

    def for_side(self, state, side, deck, ability_keys, ability_available):
        player = state.players[side]
        opened = state.tick >= 100 and state.episode.commands_allowed and not state.episode.terminated
        cards = np.zeros(4,bool); positions=np.zeros((4,576),bool)
        costs=np.full(4,-1,np.int64)
        for slot, deck_index in enumerate(player.hand):
            if not opened or deck_index < 0: continue
            card_id = int(deck[deck_index]['card_id'])
            key = (side,deck_index)
            probe = self.probes.get(key)
            if (probe is None or card_id == MIRROR_CARD_ID or
                    probe['selection_strategy'] == 'native_dynamic_choice'):
                probe = normalize_native_probe(self.env.probe_grid(side=side,deck_index=deck_index))
                self.probes[key] = probe
            costs[slot] = probe['card_cost_raw']
            # The audited projection only reads tower identity/geometry/aliveness.
            signature=tuple((t.side,t.role,t.lane,t.x,t.y,t.hp>0) for t in state.towers)
            cache_key=(side,deck_index,card_id,probe['resolved_data_id'],tuple(probe['rows']),signature)
            flat=self.position_cache.get(cache_key)
            if flat is None:
                rows=derive_deployment_rows(probe,state,side=side,card_id=card_id)
                mask=np.array([[c=='1' for c in row] for row in rows],dtype=bool)
                if side==1:mask=mask[::-1,::-1]
                flat=mask.reshape(-1).copy()
                self.position_cache[cache_key]=flat
            positions[slot]=flat
            cards[slot] = player.elixir_raw >= costs[slot] and positions[slot].any()
        entities = {e.key:e for e in state.entities if e.side == side}
        abilities=np.asarray(ability_available,dtype=bool).copy()
        for slot in range(len(abilities)):
            entity = entities.get(ability_keys[slot]) if slot < len(ability_keys) else None
            abilities[slot] &= bool(opened and entity is not None and entity.ability_available
                                    and entity.ability_mana_cost >= 0
                                    and player.elixir_raw >= entity.ability_mana_cost*10000)
        return LiveMasks(cards,positions,abilities,player.hand,ability_keys,costs)


def masked_argmax(logits, mask):
    mask=torch.as_tensor(mask,dtype=torch.bool,device=logits.device)
    if not mask.any(): raise ValueError('cannot select from an empty legal mask')
    if not torch.isfinite(logits).all(): raise FloatingPointError('non-finite action logits')
    return int(logits.masked_fill(~mask,-torch.inf).argmax())


def decode_action(output, side, masks, threshold):
    """Deterministic BC evaluation, not a PPO sampling/log-probability contract."""
    probability=float(output['timing'][side,0].sigmoid())
    if not math.isfinite(probability): raise FloatingPointError('non-finite timing probability')
    record=dict(side=side,timing_probability=probability,action='wait')
    if probability < threshold:
        record['reason']='below_timing_threshold'; return None,record
    legal_kind=np.array([masks.cards.any(),masks.abilities.any()])
    if not legal_kind.any():
        record['reason']='no_legal_action'; return None,record
    kind=masked_argmax(output['kind'][side,0],legal_kind)
    if kind == 0:
        slot=masked_argmax(output['card'][side,0],masks.cards)
        position=masked_argmax(output['position'][side,0,slot],masks.positions[slot])
        x,y=canonical_position_to_native(position,side)
        native=dict(type='play',side=side,deck_index=int(masks.hand[slot]),x=x,y=y)
        record.update(action='play',card_slot=slot,position=position,
                      card_cost_raw=int(masks.card_cost_raw[slot]),native=native)
    else:
        slot=masked_argmax(output['ability'][side,0],masks.abilities)
        native=dict(type='ability',side=side,entity_id=int(masks.ability_keys[slot]))
        record.update(action='ability',ability_slot=slot,native=native)
    return native,record


class FixedLiveAgent:
    def __init__(self, model, encoder, *, device='cpu', threshold=.5):
        if not math.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError('timing threshold must be in (0,1)')
        self.model=model.to(device).eval().requires_grad_(False)
        self.encoder=encoder;self.device=device;self.threshold=threshold
        self.reset()

    @classmethod
    def load(cls, checkpoint, manifest_path, *, device='cpu', threshold=.5):
        checkpoint,manifest_path=Path(checkpoint),Path(manifest_path)
        sha=digest(checkpoint);saved=load_checkpoint(checkpoint)
        if digest(checkpoint)!=sha: raise ValueError('checkpoint changed while loading')
        manifest=json.loads(manifest_path.read_text())
        check_mask_proof(manifest)
        if saved['contract']['manifest_sha256']!=digest(manifest_path):
            raise ValueError('checkpoint and native vocabulary manifest differ')
        if saved['config']['architecture']!=FixedConfig.architecture:
            raise ValueError('live adapter supports the original fixed-period BC checkpoint only')
        config=FixedConfig(**saved['config'])
        if config.decision_period!=4: raise ValueError('this runner requires fixed4')
        for key in ('card_vocab_size','ability_vocab_size','public_scalar_size','entity_numeric_size','grid_channels'):
            if getattr(config,key)!=manifest['dimensions'][key]: raise ValueError('model dimension mismatch: '+key)
        model=FixedPolicy(config);model.load_state_dict(saved['model'],strict=True)
        result=cls(model,NativeObservationEncoder.from_manifest(manifest),device=device,threshold=threshold)
        result.identity=dict(checkpoint=str(checkpoint.resolve()),checkpoint_sha256=sha,
                             checkpoint_step=saved['step'],manifest_sha256=digest(manifest_path),
                             architecture=config.architecture,decision_period=4,timing_threshold=threshold,
                             policy_mode='greedy_masked_bc',training_updates=False)
        return result

    def reset(self):
        self.hidden=None;self.last_tick=None
        self.tracker=RevealedEnemyTracker(self.encoder)

    def observations(self, state, decks):
        state=normalize_native_state(state)
        delta=0 if self.last_tick is None else state.tick-self.last_tick
        if self.last_tick is not None and delta!=self.model.config.decision_period:
            raise ValueError(f'fixed observation clock changed: {self.last_tick} -> {state.tick}; reset at new episode')
        encoded=self.encoder.encode_batch([
            NativeActorFrame(state,side,decks[side],revealed_enemy_tokens=self.tracker.tokens_for(side),delta_ticks=delta)
            for side in (0,1)])
        batch=dict(encoded)
        batch['frame_ticks']=torch.tensor(encoded.ticks,dtype=torch.long)[:,None]
        batch['prev_elapsed_ticks']=batch.pop('delta_ticks')
        batch['frame_mask']=torch.ones((2,1),dtype=torch.bool)
        return state,encoded,{k:v.to(self.device) for k,v in batch.items()}

    def decide(self, raw_state, decks, masks):
        state,encoded,batch=self.observations(raw_state,decks)
        with torch.inference_mode():
            output,self.hidden=self.model.forward_stream(batch,self.hidden)
        self.last_tick=state.tick
        actions=[];records=[]
        for side in (0,1):
            p=float(output['timing'][side,0].sigmoid())
            if not math.isfinite(p): raise FloatingPointError('non-finite timing probability')
            if p < self.threshold:
                records.append(dict(side=side,timing_probability=p,action='wait',reason='below_timing_threshold'))
                continue
            legal=masks.for_side(state,side,decks[side],encoded.ability_entity_keys[side],
                                 encoded.ability_mask[side,0].numpy())
            action,record=decode_action(output,side,legal,self.threshold)
            record.update(elixir_raw=state.players[side].elixir_raw)
            if action is not None: actions.append(action)
            records.append(record)
        return actions,records,dict(encoded_entities=list(encoded.encoded_entity_counts),
                                    native_entities=list(encoded.native_entity_counts))

    def record_accepted(self, actions, accepted_sides, decks):
        for action in actions:
            if action['side'] in accepted_sides and action['type']=='play':
                self.tracker.record_play(played_side=action['side'],card=decks[action['side']][action['deck_index']])
