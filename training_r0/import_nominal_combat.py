"""Import known nominal attributes from a hash-bound local combat table.

Does not recover absent mechanics, infer runtime buffs or convert native speed
to physical speed. The generated table records the source hash and limitations.
"""
import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_FIELDS=('ManaCost','IsBuilding','AttacksGround','AttacksAir','TargetOnlyBuildings',
    'FlyingHeight','Speed','Range','HitSpeed','SightRange')
EXPECTED_SCALES=(10,1,1,1,1,1000,120,12000,5000,12000)
RUNTIME='fa6704b83cb9c5b8eecb7b56c9671b834d636a3a6d9ac446e698e1262dc246ba'


def convert(source: Path, destination: Path):
    from .semantics import STATIC_SCALES,checked_features
    data=source.read_bytes();raw=json.loads(data)
    if raw.get('source_libg_sha256')!=RUNTIME or raw.get('schema')!='cr_nominal_static_combat_v1' or tuple(raw.get('fields',()))!=EXPECTED_FIELDS or tuple(raw.get('scales',()))!=EXPECTED_SCALES:
        raise ValueError('nominal source runtime/schema/fields/scales differ')
    if len(raw['card_vocabulary'])!=len(raw['features']):raise ValueError('nominal rows differ')
    names=('cost','is_building','attacks_ground','attacks_air','targets_buildings_only',
        'flying_height_native','speed_native','range_tiles','hit_interval_ms','sight_tiles')
    factors=(10,1,1,1,1,1000,120,12,5000,12)
    cards={}
    for token,row in zip(raw['card_vocabulary'],raw['features']):
        if len(row)!=20 or any(x not in (0,1) for x in row[10:]):raise ValueError('invalid known-bit row')
        if '@' not in token:continue
        card=str(int(token.rsplit('@',1)[1]))
        if card in cards:raise ValueError('duplicate source card')
        values={name:row[j]*factors[j] for j,name in enumerate(names) if row[10+j]}
        checked_features(tuple(values.items()),STATIC_SCALES)
        cards[card]=values
    result=dict(schema='r0-nominal-combat.v1',runtime_sha256=RUNTIME,
        source_sha256=hashlib.sha256(data).hexdigest(),source_schema=raw['schema'],
        limitations=raw.get('limitations',[]),cards=cards)
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return dict(cards=len(cards),cards_with_known_fields=sum(bool(v) for v in cards.values()),source_sha256=result['source_sha256'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('destination',type=Path)
    args=parser.parse_args();print(json.dumps(convert(args.source,args.destination)))
