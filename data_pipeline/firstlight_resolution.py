"""Auditable FirstLight-inspired ability policy for a future capture version.

Not wired into the already-running fixed-seed raw pilot. A newest eligible
carrier is an inference, never evidence of the original player's choice.
"""
from __future__ import annotations


def select_ability(entities, *, side, allowed_card_ids=None, exact_entity_key=None):
    eligible=[];excluded=[]
    for e in entities:
        if e.get('side')!=side:continue
        key=e.get('generation_key',e.get('category'))
        if allowed_card_ids is not None and e.get('card_id') not in allowed_card_ids:continue
        if exact_entity_key is not None and key!=exact_entity_key:continue
        if not e.get('ability_slot',0):continue
        reason=None
        if e.get('ability_available') is not True:reason='native_not_available'
        elif e.get('ability_cooldown_remaining_ms')!=0:reason='cooldown_or_unknown'
        elif type(e.get('ability_charges_remaining')) is not int or e['ability_charges_remaining']<=0:reason='no_charges_or_unknown'
        if reason:excluded.append({'key':key,'reason':reason})
        else:eligible.append(e)
    evidence=dict(eligible_keys=[e.get('generation_key',e.get('category')) for e in eligible],excluded=excluded,
                  source_identity_confirmed=exact_entity_key is not None)
    if not eligible:return dict(status='unavailable',selected=None,**evidence)
    if len(eligible)==1:
        e=eligible[0]
        return dict(status='exact_source' if exact_entity_key is not None else 'unique_eligible',
                    selected=e.get('generation_key',e.get('category')),**evidence)
    # Local runtime exposes explicit creation order; do not compare pointers.
    if any(type(e.get('creation_ordinal')) is not int or e['creation_ordinal']<0 for e in eligible):
        return dict(status='ambiguous_creation_order_unknown',selected=None,**evidence)
    newest=max(e['creation_ordinal'] for e in eligible)
    winners=[e for e in eligible if e['creation_ordinal']==newest]
    if len(winners)!=1:return dict(status='ambiguous_latest_tie',selected=None,**evidence)
    e=winners[0]
    return dict(status='inferred_newest_eligible',selected=e.get('generation_key',e.get('category')),**evidence)


def rejected_window(execution_tick, *, first_decision_tick, decision_ticks=5):
    """Match FirstLight's decision < execution <= decision+period interval."""
    if type(execution_tick) is not int or type(first_decision_tick) is not int or decision_ticks<1:
        raise ValueError('invalid clock contract')
    if execution_tick<=first_decision_tick:return None
    return first_decision_tick+((execution_tick-first_decision_tick-1)//decision_ticks)*decision_ticks
