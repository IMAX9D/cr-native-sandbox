"""R0 extension preserving verified native projectiles outside the arena.

The legacy expert Tick-store remains unchanged. Live projectiles may overshoot
the board (for example an Executioner's returning axe); these are observations,
not legal deployment coordinates and must never be clamped to an edge.
"""
from dataclasses import replace
from expert_v1.tick_store_v1 import schema as legacy


def normalize_native_state(raw):
    outside=[];inside=[]
    for entity in raw.get('entities',()):
        x=legacy._integer(entity.get('x'),'entity.x');y=legacy._integer(entity.get('y'),'entity.y')
        if 0<=x<=18000 and 0<=y<=32000:inside.append(entity);continue
        key=legacy._integer(entity.get('category'),'entity.category')
        if not (4000000<=key<5000000 and entity.get('vtable_rva')=='0x1969b38' and
                -(2**31)<=x<2**31 and -(2**31)<=y<2**31):
            raise legacy.TickStoreContractError('unverified entity outside native arena')
        side=legacy._integer(entity.get('side'),'entity.side')
        if side not in (0,1):raise legacy.TickStoreContractError('invalid projectile side')
        defaults={'card_id':-1,'level':-1,'hp':-1,'max_hp':-1,'behavior_state':0,'ability_slot':0,
            'ability_state_code':-1,'ability_available':0,'ability_cooldown_remaining_ms':-1,
            'ability_charges_remaining':-1,'ability_pending_ms':-1,'ability_mana_cost':-1}
        values={name:legacy._integer(entity.get(name),name,default=default) for name,default in defaults.items()}
        if values['ability_slot'] or values['ability_available']:
            raise legacy.TickStoreContractError('projectile cannot advertise a player ability')
        outside.append(legacy.EntityState(key=key,side=side,x=x,y=y,**values))
    normalized=legacy.normalize_native_state({**raw,'entities':inside})
    entities=(*normalized.entities,*outside)
    if len({e.key for e in entities})!=len(entities):raise legacy.TickStoreContractError('duplicate entity generation key')
    return replace(normalized,entities=tuple(sorted(entities,key=lambda e:e.key)))


def actor_projection(state,*,actor_side):
    view=legacy.actor_projection(state,actor_side=actor_side)
    originals={e.key:e for e in state.entities}
    return replace(view,entities=tuple(replace(e,
        x=originals[e.key].x if actor_side==0 else 17999-originals[e.key].x,
        y=originals[e.key].y if actor_side==0 else 31999-originals[e.key].y)
        if 4000000<=e.key<5000000 else e for e in view.entities))
