"""Task-owned fresh decoded frames; copy only nodes enrichment actually changes."""
from copy import deepcopy
from collections.abc import Mapping
from native_core.env import CARD_NAMES,ABILITY_STATE_NAMES,observed_card

def enrich(self,state):
    # Shared nested paths/effects must remain read-only in this isolated collector.
    result=dict(state)
    result['players']=[dict(p) for p in state.get('players',[])]
    result['entities']=[dict(e) for e in state.get('entities',[])]
    result['elapsed_seconds']=round(int(result['tick'])*.05,3)
    if isinstance(result.get('episode'),Mapping):
        result['episode']=self._enrich_episode(result['episode'])
        self.last_episode=deepcopy(result['episode'])
    for p in result['players']:
        side=int(p['side'])
        if isinstance(p.get('elixir_raw'),int):p['elixir_exact']=p['elixir_raw']/10000.
        hand=[]
        for hi,di in enumerate(p['hand_deck_indices']):
            if di<0 or side>=len(self.decks):continue
            c=self.decks[side][di];cid=c['card_id'];flags=int(c.get('form_flags',0))
            hand.append(dict(hand_index=hi,deck_index=di,card_id=cid,level=c['level'],form_flags=flags,
                             has_evolution=bool(flags&1),has_hero=bool(flags&2),name=CARD_NAMES.get(cid,str(cid))))
        p['hand']=hand
    for e in result['entities']:
        if isinstance(e.get('category'),int):e['entity_id']=int(e['category'])
        if isinstance(e.get('ability_state_code'),int):e['ability_state_name']=ABILITY_STATE_NAMES.get(int(e['ability_state_code']),'unknown_native_state')
        cid=int(e.get('card_id',-1))
        if cid<0:continue
        identity=observed_card(cid);e['native_card_id']=cid;e.update(identity)
        e['name']=CARD_NAMES.get(int(identity['base_card_id']),str(identity['form_name']))
    return result
