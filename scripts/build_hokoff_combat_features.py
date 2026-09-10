import csv,io,lzma,tomllib,json,hashlib
from pathlib import Path
import argparse
parser=argparse.ArgumentParser(description='Build nominal combat inputs from frozen native assets')
parser.add_argument('--runtime-root',type=Path,default=Path.home()/'cr-data/native-workers/hokoff-ppo/port-39039')
parser.add_argument('--data',type=Path,default=Path.home()/'cr-data/expert-dataset/native-bc-v1')
parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'hokoff_model/combat_features.json')
args=parser.parse_args()
repo=Path(__file__).resolve().parents[1]
libg_sha=hashlib.sha256((args.runtime_root/'libg.so').read_bytes()).hexdigest()
assert libg_sha=='fa6704b83cb9c5b8eecb7b56c9671b834d636a3a6d9ac446e698e1262dc246ba'
root=args.runtime_root/'assets/csv_logic' 
cat=json.load(open(repo/'native_core/data/live_card_catalog.json'))
manifest=json.load(open(args.data/'manifest.json'))
assert cat['game_version']=='15.535.29'
def decode(p):
 b=p.read_bytes()
 if b[:1]==b']':b=lzma.decompress(b[:9]+b'\0'*4+b[9:],format=lzma.FORMAT_ALONE)
 return b.decode('utf-8-sig')
fields=[('ManaCost',10),('IsBuilding',1),('AttacksGround',1),('AttacksAir',1),('TargetOnlyBuildings',1),('FlyingHeight',1000),('Speed',120),('Range',12000),('HitSpeed',5000),('SightRange',12000)]
needed={k for k,_ in fields}|{'Base','SummonCharacter','SummonCharacterSecond'}
rows={};sources={};conflicts=set()
def add(kind,name,value,p):
 key=kind+'.'+name
 small={k:v for k,v in value.items() if k in needed and v not in ('',None)}
 old=rows.setdefault(key,{})
 for k,v in small.items():
  if k in old and old[k]!=v:conflicts.add((key,k))
  else:old[k]=v
 sources.setdefault(key,set()).add(str(p.relative_to(root)))
for name,kind in [('characters.csv','CHARACTER'),('buildings.csv','BUILDING'),('spells_characters.csv','SPELL_CHARACTER'),('spells_buildings.csv','SPELL_BUILDING'),('spells_other.csv','SPELL_OTHER')]:
 p=root/name
 if p.exists():
  for v in csv.DictReader(io.StringIO(decode(p))):
   if v.get('Name') not in [None,'','string']:add(kind,v['Name'],v,p)
fail=[]
for p in sorted(root.rglob('*.toml')):
 try:d=tomllib.loads(decode(p))
 except Exception as e:fail.append(str(p.relative_to(root)));continue
 for kind,table in d.items():
  if kind in ['CHARACTER','BUILDING','EXT','SPELL_CHARACTER','SPELL_BUILDING','SPELL_OTHER','SPELL_HERO'] and isinstance(table,dict):
   for name,v in table.items():
    if isinstance(v,dict):add(kind,name,v,p)
def resolved(key,seen=()):
 if key in seen or key not in rows:return {}
 row=rows[key];base=row.get('Base');out=resolved(base,seen+(key,)) if isinstance(base,str) else {}
 for k,v in row.items():
  if k!='Base':out[k]=None if (key,k) in conflicts else v
 return out
byid={int(c['card_id']):(c,'base') for c in cat['cards']}
for c in cat['cards']:
 for f in ['evolution','hero']:
  if c.get(f+'_form_id') is not None:byid[int(c[f+'_form_id'])]=(c,f)
features=[];audit=[]
for token in manifest['card_vocabulary']:
 vals={};reason='unmapped';unit=None
 if '@' in token:
  ident=int(token.rsplit('@',1)[1]);item=byid.get(ident)
  if item:
   c,form=item;vals['ManaCost']=c.get('elixir')
   name=c['internal_name'] if form=='base' else c.get(form+'_form')
   kinds=['SPELL_CHARACTER','SPELL_BUILDING','SPELL_OTHER','SPELL_HERO']
   found=[resolved(k+'.'+str(name)) for k in kinds if k+'.'+str(name) in rows]
   spell=found[0] if len(found)==1 else {}
   vals['ManaCost']=spell.get('ManaCost',c.get('elixir') if form=='base' else None)
   unit=spell.get('SummonCharacter') or (c.get('summon_character') if form=='base' else c.get('hero_character') if form=='hero' else None)
   second=spell.get('SummonCharacterSecond')
   if second and second!=unit:reason='mixed_summons';unit=None
   else:reason='unit_not_resolved'
   if unit:
    candidates=[k+'.'+unit for k in ['CHARACTER','BUILDING','EXT'] if k+'.'+unit in rows]
    if len(candidates)==1:
     vals.update(resolved(candidates[0]));reason='nominal_static_unit'
     if candidates[0].startswith('BUILDING.'):vals['IsBuilding']=True
     elif candidates[0].startswith('CHARACTER.'):vals['IsBuilding']=False
 values=[];known=[]
 for k,scale in fields:
  v=vals.get(k)
  if isinstance(v,str):
   if v.lower() in ['true','false']:v=v.lower()=='true'
   else:
    try:v=float(v)
    except ValueError:v=None
  ok=isinstance(v,(int,float)) and not isinstance(v,list) and v>=0
  values.append(float(v)/scale if ok else 0.);known.append(float(ok))
 features.append(values+known);audit.append(dict(token=token,unit=unit,reason=reason,known=sum(known)))
artifact=dict(source_libg_sha256=libg_sha,schema='cr_nominal_static_combat_v1',game_version='15.535.29',
 fields=[k for k,_ in fields],scales=[s for _,s in fields],card_vocabulary=manifest['card_vocabulary'],
 features=features,audit=audit,toml_parse_failures=fail,
 source_catalog_sha256=hashlib.sha256(Path(repo/'native_core/data/live_card_catalog.json').read_bytes()).hexdigest(),
 source_assets_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file() and p.suffix in ['.toml','.csv']},
 limitations=['nominal static attributes only, not attack target or current buffs','missing and conflicting values marked unknown','mixed summon units not assigned one guessed profile','damage/level scaling excluded'])
args.output.write_text(json.dumps(artifact,ensure_ascii=False,indent=2))
print('tokens',len(features),'unit_mapped',sum(a['reason']=='nominal_static_unit' for a in audit),'parse_failures',len(fail))
for name in ['musketeer@','giant@','knight@','goblin-gang@']:
 for i,t in enumerate(manifest['card_vocabulary']):
  if t.startswith(name):print(audit[i],features[i])
