"""Import exact rarity level curves from the pinned game's raw resource table."""
import argparse
import csv
import io
from pathlib import Path
from training_r0.build_native_semantics import decode
from .prepare_600k import atomic_json,digest


def build(path):
    current=None;rows={}
    for row in csv.DictReader(io.StringIO(decode(path))):
        name=row['Name']
        if name=='String':continue
        if name:
            current=dict(level_count=int(row['LevelCount']),relative_level=int(row['RelativeLevel']),multipliers=[100]);rows[name]=current
        if current is not None and row.get('PowerLevelMultiplier'):current['multipliers'].append(int(row['PowerLevelMultiplier']))
    for value in rows.values():
        value['multipliers']=value['multipliers'][:value['level_count']]
        if len(value['multipliers'])!=value['level_count']:raise ValueError('incomplete rarity level curve')
        value['levels']=list(range(value['relative_level']+1,value['relative_level']+1+value['level_count']))
    return dict(schema='r0-rarity-curves.v1',runtime_sha256='fa6704b83cb9c5b8eecb7b56c9671b834d636a3a6d9ac446e698e1262dc246ba',
                source=str(path),source_sha256=digest(path),curves=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('preserve existing curve artifact')
    atomic_json(a.output,build(a.source))
