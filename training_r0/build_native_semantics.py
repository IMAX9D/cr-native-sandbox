"""Preserve native battle definitions, typed attributes and unambiguous links.

Raw definitions are not falsely labeled resolved/effective game mechanics.
Conflicting definitions and unresolved references remain in the source graph.
"""
from pathlib import Path
import argparse,csv,hashlib,io,json,lzma,math,tomllib,re

RUNTIME='fa6704b83cb9c5b8eecb7b56c9671b834d636a3a6d9ac446e698e1262dc246ba'
KINDS={'characters':'CHARACTER','buildings':'BUILDING','projectiles':'PROJECTILE',
    'character_buffs':'BUFF','area_effect_objects':'AREA_EFFECT_OBJECT',
    'spells_characters':'SPELL_CHARACTER','spells_buildings':'SPELL_BUILDING','spells_other':'SPELL_OTHER',
    'spells_heroes':'SPELL_HERO','character_abilities':'ABILITY','spawn_groups':'SPAWN_GROUP',
    'targetresolvers':'TARGET_RESOLVER','shapes':'SHAPE','game_object_filters':'FILTER','card_forms':'CARD_FORM'}


def decode(path):
    data=path.read_bytes()
    if data[:1]==b']':data=lzma.decompress(data[:9]+b'\0'*4+data[9:],format=lzma.FORMAT_ALONE)
    return data.decode('utf-8-sig')


def flatten(value,path=''):
    if isinstance(value,dict):
        for key,item in value.items():yield from flatten(item,path+'.'+str(key) if path else str(key))
    elif isinstance(value,list):
        if not value:yield path,'empty_list',''
        for i,item in enumerate(value):yield from flatten(item,f'{path}[{i}]')
    elif value is not None and value!='':
        if isinstance(value,bool):yield path,'boolean',int(value)
        elif isinstance(value,(int,float)):
            if not math.isfinite(value):raise ValueError('nonfinite native number')
            yield path,'number',value
        else:
            if str(value).lower() in ('true','false'):yield path,'boolean',int(str(value).lower()=='true')
            else:
                try:
                    number=float(value)
                    if not math.isfinite(number):raise ValueError('nonfinite native value')
                    yield path,'number',number
                except ValueError:yield path,'string',str(value)


def build(runtime:Path,output:Path):
    if hashlib.sha256((runtime/'libg.so').read_bytes()).hexdigest()!=RUNTIME:
        raise ValueError('native semantics require frozen x86_64 runtime')
    root=runtime/'assets/csv_logic';records={};sources={};failures=[]
    def add(kind,name,row,path):
        key=f'{kind}.{name}'
        attrs=list(flatten(row))
        records.setdefault(key,[]).append(dict(source=str(path.relative_to(root)),attributes=attrs))
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix not in ('.csv','.toml'):continue
        stem=path.stem.removesuffix('_evo').removesuffix('_evolved')
        fallback=KINDS.get(stem) or KINDS.get(path.parent.name)
        try:
            if path.suffix=='.csv':
                if fallback is None:continue
                for row in csv.DictReader(io.StringIO(decode(path))):
                    if row.get('Name') not in (None,'','string'):add(fallback,row['Name'],{k:v for k,v in row.items() if k!='Name'},path)
            else:
                tables=tomllib.loads(decode(path))
                for kind,table in tables.items():
                    if not isinstance(table,dict):continue
                    if kind in {*KINDS.values(),'EXT','ACTION','ACTION_GROUP','TARGETRESOLVER','AREA','EFFECT'} or kind.isupper():
                        for name,row in table.items():
                            add(kind,name,row if isinstance(row,dict) else {'Value':row},path)
                    elif fallback:add(fallback,kind,table,path)
            sources[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
        except (ValueError,tomllib.TOMLDecodeError,csv.Error) as error:
            failures.append(dict(path=str(path.relative_to(root)),error=str(error)))
    if failures:raise ValueError(f'native definition parse failures: {failures}')
    names=sorted(records);lookup={name:i+2 for i,name in enumerate(names)}
    short={}
    for name in names:short.setdefault(name.split('.',1)[1],[]).append(name)
    field_names=sorted({field for variants in records.values() for v in variants for field,_,_ in v['attributes']})
    string_values=sorted({value for variants in records.values() for v in variants for _,kind,value in v['attributes'] if kind=='string'})
    fidx={v:i+1 for i,v in enumerate(field_names)};sidx={v:i+1 for i,v in enumerate(string_values)}
    attributes=[];edges=[];conflicts=[]
    for name,variants in records.items():
        byfield={}
        for variant in variants:
            for field,kind,value in variant['attributes']:
                byfield.setdefault(field,set()).add((kind,value))
        for field,values in sorted(byfield.items()):
            conflict=len(values)>1
            if conflict:conflicts.append([name,field,sorted(values)])
            for kind,value in sorted(values):
                attributes.append([lookup[name],fidx[field],{'number':0,'boolean':1,'string':2,'empty_list':3}[kind],
                    sidx[value] if kind=='string' else 0, value if kind in ('number','boolean') else 0,int(conflict)])
                if kind=='string':
                    targets=[value] if value in lookup else short.get(value,[])
                    leaf=re.sub(r'\[\d+\]','',field.rsplit('.',1)[-1])
                    # These native fields declare the referenced table family.
                    # Other ambiguous strings remain attributes, without guessed links.
                    families=None
                    if leaf in ('SummonCharacter','SummonCharacterSecond','SummonCharactersList','SpawnCharacter','SpawnCharacters','DeathSpawn','TransformCharacter'):
                        families=('CHARACTER.','BUILDING.','EXT.')
                    elif leaf in ('Projectile','CustomFirstProjectile','ProjectileSpecial','SpawnProjectile','OverrideProjectile'):
                        families=('PROJECTILE.',)
                    elif leaf in ('Buff','BuffOnDamage','StartingBuff','TargetBuff','ChainedBuff'):
                        families=('BUFF.',)
                    elif leaf in ('SpawnAreaEffectObject','AreaEffectObject'):
                        families=('AREA_EFFECT_OBJECT.','AEO.')
                    elif leaf.endswith('Action') or leaf.endswith('ActionGroup'):
                        families=('ACTION.','ACTION_GROUP.')
                    if families and value not in lookup:targets=[t for t in targets if t.startswith(families)]
                    if len(targets)==1:edges.append([lookup[name],lookup[targets[0]],fidx[field],int(conflict)])
    result=dict(schema='r0-native-definition-graph.v1',runtime_sha256=RUNTIME,
        node_names=['<PAD>','<UNKNOWN>',*names],field_names=['<PAD>',*field_names],string_values=['<PAD>',*string_values],
        attributes=attributes,edges=edges,conflicts=conflicts,source_files=sources,
        source_records=records,limitations=['raw definitions, not resolved effective behavior','ambiguous references remain string attributes','conflicting definitions retained with conflict bits'])
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8')
    return dict(nodes=len(names),fields=len(field_names),strings=len(string_values),attributes=len(attributes),edges=len(edges),conflicts=len(conflicts),bytes=output.stat().st_size)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('runtime',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();print(json.dumps(build(args.runtime,args.output)))
