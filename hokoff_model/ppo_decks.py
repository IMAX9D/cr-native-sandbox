"""Independent random base-form decks for both policies, within the IL vocabulary."""
from copy import deepcopy
import random
from native_core.card_catalog import catalog
from expert_selfplay_v1.native_observation import NativeObservationEncoder


# Native 150535029 dynamic MergeMaiden resolves into these observable card IDs.
# The current IL knows its selectable spell, but not these two spawned forms.
SPAWNED_FORMS={28000025:(26000104,26000105)}


def missing_spawned_forms(ids,known):
    return {str(card):[v for v in SPAWNED_FORMS.get(card,()) if v not in known]
            for card in ids if any(v not in known for v in SPAWNED_FORMS.get(card,()))}


def opponent_deck_config(requested,manifest,*,saved=None,role="opponent"):
    if saved is not None:
        mode=saved.get('mode','fixed')
        if requested is not None and requested!=mode:raise ValueError(f'resume {role} deck mode differs; start a new experiment')
        if mode=='random' and manifest is not None:
            encoder=NativeObservationEncoder.from_manifest(manifest)
            if missing_spawned_forms(saved['card_ids'],encoder.card_id_to_token):
                raise ValueError('saved random pool has unsupported spawned forms; start a new experiment')
        return deepcopy(saved)
    mode=requested or 'random'
    if mode not in ('fixed','random'):raise ValueError(f'{role} deck mode must be fixed or random')
    if mode=='fixed':return dict(mode='fixed')
    encoder=NativeObservationEncoder.from_manifest(manifest)
    rows=[r for r in catalog().values() if r.get('standard_1v1')
          and r['card_id'] in encoder.card_id_to_token
          and (not r.get('active_ability') or r['card_id'] in encoder.ability_id_to_token)]
    excluded=missing_spawned_forms([r['card_id'] for r in rows],encoder.card_id_to_token)
    rows=[r for r in rows if str(r['card_id']) not in excluded]
    ids=sorted(r['card_id'] for r in rows)
    champions=sorted(r['card_id'] for r in rows if r.get('rarity')=='Champion')
    if len(ids)<8 or len(set(ids)-set(champions))<7:raise ValueError('insufficient legal random deck pool')
    return dict(mode=mode,sampling='uniform_unique_8_max_one_champion_v1',
                card_ids=ids,champion_ids=champions,excluded_spawned_forms=excluded,forms='base',levels='fixture_slot_levels')


def episode_replay(fixture,*,seed,episode_id,learner_side,config=None,learner_config=None):
    if learner_side not in (0,1):raise ValueError('learner side must be 0/1')
    replay=deepcopy(fixture);replay['rndSeed']=seed+episode_id
    # Distinct role-based streams: changing one side's mode does not redraw the other.
    # Keep the original opponent seed unchanged for existing checkpoint continuity.
    for side,cfg,role in ((1-learner_side,config,'opponent'),(learner_side,learner_config,'learner')):
        if not cfg or cfg['mode']=='fixed':continue
        if cfg['mode']!='random' or cfg['sampling']!='uniform_unique_8_max_one_champion_v1':
            raise ValueError('unknown saved deck sampling scheme')
        ids=cfg['card_ids'];champions=set(cfg['champion_ids'])
        if len(ids)!=len(set(ids)) or len(ids)<8 or len(set(ids)-champions)<7:
            raise ValueError('invalid saved random card pool')
        rng=random.Random(f'hokoff-{role}-deck-v1:{seed}:{episode_id}')
        while True:
            selected=rng.sample(ids,8)
            if len(set(selected)&champions)<=1:break
        deck=replay['battle'][f'deck{side}'];levels=[card['l'] for card in deck['sp']]
        deck['sp']=[dict(d=card,l=level) for card,level in zip(selected,levels,strict=True)]
    return replay
