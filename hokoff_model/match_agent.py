"""Release-weight adapter for controlled native human-vs-BC matches."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import torch

from expert_selfplay_v1.native_observation import NativeObservationEncoder, NativeActorFrame, RevealedEnemyTracker
from expert_v1.tick_store_v1.schema import normalize_native_state
from native_core.card_catalog import metadata
from .live import NativeMaskProvider, masked_argmax, canonical_position_to_native
from .model import Policy
from .train_fixed import FixedConfig, FixedPolicy
from .online_history import OnlinePlayHistory, PublicPlay

DEFAULT_CONTRACT = Path(__file__).with_name('match_encoder_contract.json')
LIBG_SHA = 'fa6704b83cb9c5b8eecb7b56c9671b834d636a3a6d9ac446e698e1262dc246ba'


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


class HistoryOnlinePolicy(FixedPolicy):
    """An explicit online-history path; the old BC-only guard stays unchanged."""
    def initial_hidden(self, batch_size, *, device=None):
        reference = next(self.parameters())
        return tuple(torch.zeros(1,batch_size,self.config.hidden_size,
                     device=device or reference.device,dtype=reference.dtype) for _ in range(2))

    def forward_stream(self, batch, state=None, reset=None):
        expected = (*batch['frame_mask'].shape, 2, self.config.history_length)
        for key in ('history_card','history_position','history_age','history_mask','history_known'):
            if key not in batch or tuple(batch[key].shape) != expected:
                raise ValueError('online history missing or malformed: '+key)
        if not torch.isfinite(batch['history_age']).all() or (batch['history_age'] < 0).any():
            raise ValueError('invalid online history age')
        return Policy.forward_stream(self,batch,state=state,reset=reset)


def load_release(checkpoint, contract_path=DEFAULT_CONTRACT, *, device='cpu'):
    contract_path = Path(contract_path)
    contract = json.loads(contract_path.read_text(encoding='utf-8-sig'))
    if contract.get('kind') != 'hokoff_match_encoder_contract_v1' or contract.get('libg_sha256') != LIBG_SHA:
        raise ValueError('unsupported match encoder/runtime contract')
    sha = file_sha(checkpoint)
    if sha not in contract['allowed_weights']: raise ValueError('weight hash is not bound to this encoder contract')
    saved = torch.load(checkpoint,map_location='cpu',weights_only=True)
    if saved['config'].get('architecture') != FixedConfig.architecture:
        raise ValueError('match box supports the fixed-period release only')
    config = FixedConfig(**saved['config'])
    if config.decision_period != 4 or config.history_length != 4:
        raise ValueError('unsupported decision/history contract')
    dimensions = contract['encoder']['dimensions']
    for key in ('card_vocab_size','ability_vocab_size','grid_channels','public_scalar_size','entity_numeric_size'):
        if getattr(config,key) != dimensions[key]: raise ValueError('encoder dimension mismatch: '+key)
    table = json.loads(Path(__file__).with_name('combat_features.json').read_text(encoding='utf-8'))
    if (contract['encoder']['card_vocabulary'] != table['card_vocabulary']
            or config.combat_features != table['features'] or table['source_libg_sha256'] != LIBG_SHA):
        raise ValueError('release combat table/vocabulary mismatch')
    encoder = NativeObservationEncoder.from_manifest(contract['encoder'])
    model = HistoryOnlinePolicy(config)
    model.load_state_dict(saved['model'],strict=True)
    if not all(bool(torch.isfinite(value).all()) for value in model.state_dict().values() if value.is_floating_point()):
        raise FloatingPointError('release has non-finite weights')
    model.to(device).eval().requires_grad_(False)
    if file_sha(checkpoint) != sha: raise ValueError('checkpoint changed while loading')
    identity = dict(policy_version='hokoff-fixed-bc',training_step=int(saved['step']),
                    path=str(Path(checkpoint).resolve()),model_digest=sha,checkpoint_sha256=sha,
                    encoder_contract_sha256=file_sha(contract_path),
                    source_manifest_sha256=contract['source_manifest_sha256'],
                    native_libg_sha256=LIBG_SHA,decision_ticks=4,history_length=4,
                    parameters=sum(p.numel() for p in model.parameters()),
                    mode='greedy-masked-fixed4',timing_threshold=.5)
    return model,encoder,identity


def verify_runtime(env):
    response = env.client.request({'op':'runtime_identity_v1'})
    value = response.get('identity',{})
    if not response.get('ok') or value.get('libg_sha256') != LIBG_SHA:
        raise ValueError('native libg version differs; update host or use the frozen x86_64 runtime')
    return value


class MatchAgent:
    def __init__(self, model, encoder, *, side=1, device='cpu'):
        if side not in (0,1): raise ValueError('invalid actor side')
        self.model,self.encoder,self.side,self.device = model,encoder,side,device
        self.history = OnlinePlayHistory(model.config.history_length)
        self.revealed = RevealedEnemyTracker(encoder)
        self.reset(0)

    def reset(self,tick=0):
        self.history.reset(); self.revealed.reset()
        self.hidden = self.model.initial_hidden(1,device=self.device)
        self.last_decision_tick = None
        self.observed_tick = tick
        self.decisions = 0
        self.last_audit = {}
        self.mask_provider = None

    def card_token(self, card):
        card_id = int(card['card_id']); flags = int(card.get('form_flags',0))
        if flags not in (0,1,2): raise ValueError('ambiguous combined card form')
        if flags: card_id = int(metadata(card_id)['evolution_form_id' if flags==1 else 'hero_form_id'])
        if card_id not in self.encoder.card_id_to_token: raise ValueError('card outside release vocabulary')
        return self.encoder.card_id_to_token[card_id]

    def record_transition(self,before_tick,after_tick,actions,receipt,decks,*,terminal=False):
        if before_tick != self.observed_tick or after_tick < before_tick:
            raise ValueError('unobserved native transition; refuse stale history')
        if not terminal and after_tick <= before_tick: raise ValueError('native clock did not advance')
        requested = {int(action['side']): action for action in actions}
        if len(requested) != len(actions): raise ValueError('duplicate side actions')
        results = receipt.get('actions')
        if not isinstance(results,list) or len(results) != len(actions): raise ValueError('incomplete native receipt')
        seen = set()
        confirmed = []
        for item in results:
            side = int(item['side'])
            if side not in requested or side in seen: raise ValueError('receipt identity differs')
            seen.add(side)
            result = item['result']
            if not isinstance(result.get('accepted'),bool): raise ValueError('ambiguous native acceptance')
            action = requested[side]
            if not result['accepted']: continue
            fields = ('entity_id',) if action.get('type') == 'ability' else ('deck_index','x','y')
            if any(result.get(key) != action[key] for key in fields):
                raise ValueError('native accepted a different command')
            if action.get('type','play') == 'ability': continue
            card = decks[side][int(action['deck_index'])]
            x,y = int(action['x']),int(action['y'])
            if not 0 <= x < 18000 or not 0 <= y < 32000: raise ValueError('invalid accepted position')
            confirmed.append((PublicPlay(before_tick,side,self.card_token(card),(y//1000)*18+x//1000),card))
        for event,card in confirmed:
            self.history.record(event)
            self.revealed.record_play(played_side=event.side,card=card)
        self.observed_tick = after_tick

    def decide(self,raw_state,decks,env):
        state = normalize_native_state(raw_state)
        tick = state.tick
        if tick != self.observed_tick: raise ValueError('state advanced outside controlled match')
        if tick % self.model.config.decision_period:
            return None,dict(policy_skipped=True,tick=tick)
        delta = 0 if self.last_decision_tick is None else tick-self.last_decision_tick
        if self.last_decision_tick is not None and delta != 4:
            raise ValueError('duplicate/skipped fixed4 decision')
        encoded = self.encoder.encode_batch([NativeActorFrame(state,self.side,decks[self.side],
                    revealed_enemy_tokens=self.revealed.tokens_for(self.side),delta_ticks=delta)])
        batch = dict(encoded)
        batch['frame_ticks'] = torch.tensor([[tick]],dtype=torch.long)
        batch['prev_elapsed_ticks'] = batch.pop('delta_ticks')
        batch['frame_mask'] = torch.ones(1,1,dtype=torch.bool)
        batch.update(self.history.query(self.side,tick))
        batch = {key:value.to(self.device) for key,value in batch.items()}
        with torch.inference_mode(): output,hidden = self.model.forward_stream(batch,self.hidden)
        if not all(bool(torch.isfinite(value).all()) for value in [*output.values(),*hidden]):
            raise FloatingPointError('non-finite model decision')
        self.hidden = tuple(value.detach() for value in hidden)
        self.last_decision_tick = tick; self.decisions += 1
        probability = float(output['timing'][0,0].sigmoid())
        audit = dict(tick=tick,play_probability=probability,value=0.0,
                     encoded_entities=encoded.encoded_entity_counts[0],
                     native_entities=encoded.native_entity_counts[0],
                     history_slots=int(batch['history_mask'].sum()),
                     confirmed_normal_plays=list(self.history.accepted_plays),decision_number=self.decisions)
        self.last_audit = audit
        if probability < .5: return None,{**audit,'reason':'below_timing_threshold'}
        if self.mask_provider is None: self.mask_provider = NativeMaskProvider(env)
        masks = self.mask_provider.for_side(state,self.side,decks[self.side],encoded.ability_entity_keys[0],encoded.ability_mask[0,0].cpu().numpy())
        legal_kind = [bool(masks.cards.any()),bool(masks.abilities.any())]
        if not any(legal_kind): return None,{**audit,'reason':'no_legal_action'}
        kind = masked_argmax(output['kind'][0,0],legal_kind)
        if kind == 0:
            slot = masked_argmax(output['card'][0,0],masks.cards)
            position = masked_argmax(output['position'][0,0,slot],masks.positions[slot])
            x,y = canonical_position_to_native(position,self.side)
            deck_index = int(masks.hand[slot]); card_id = int(decks[self.side][deck_index]['card_id'])
            return dict(side=self.side,deck_index=deck_index,x=x,y=y,card_id=card_id),dict(audit,card_slot=slot,position=position)
        slot = masked_argmax(output['ability'][0,0],masks.abilities)
        key = int(masks.ability_keys[slot])
        entity = next(entity for entity in state.entities if entity.key == key and entity.side == self.side)
        return dict(type='ability',side=self.side,entity_id=key,card_id=entity.card_id),dict(audit,action_type='ability',entity_id=key)
