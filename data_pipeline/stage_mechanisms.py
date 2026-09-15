"""Stage bounded, source-hash-verified mechanism examples from the original tar.

Names select test candidates only; actual native activation must be verified.
"""
import argparse
import hashlib
import json
import lzma
from pathlib import Path
import sqlite3
import tarfile
from .prepare_600k import atomic_json

TARGETS=('goblin-cage-ev1','goblin-drill-ev1','rune-giant','inferno-dragon-ev1',
         'inferno-dragon','inferno-tower','mighty-miner','electro-giant','heal-spirit',
         'skeleton-king','golden-knight','monk','archer-queen','royal-chef','dagger-duchess')


def stage(archive,catalog,output,maximum):
    output.mkdir(parents=True,exist_ok=False);sources=output/'sources';sources.mkdir()
    connection=sqlite3.connect(catalog.as_uri()+'?mode=ro',uri=True)
    records=[];covered=set();scanned=0
    with lzma.open(archive,'rb') as stream,tarfile.open(fileobj=stream,mode='r|') as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith('.json'):continue
            scanned+=1;raw=tar.extractfile(member).read();payload=json.loads(raw)
            # Earlier samples test mechanisms before adding historical-patch routing.
            forms={str(p.get('card_form') or p.get('card','')) for p in payload.get('card_plays',())}
            for p in payload.get('team',())+payload.get('opponent',()):
                if isinstance(p,dict):forms.add(str(p.get('tower_troop','')))
            hits=set(TARGETS)&forms
            if hits-covered:
                row=connection.execute('select tag,sha256,size,timestamp from battles where archive=? and member=?',(archive.name,member.name)).fetchone()
                if row is None or len(raw)!=row[2] or hashlib.sha256(raw).hexdigest()!=row[1]:raise ValueError('source catalog mismatch')
                tag,sha,size,timestamp=row
                target=sources/(tag+'.json')
                with target.open('xb') as f:f.write(raw)
                records.append(dict(battle_tag=tag,source_file=str(target.resolve()),sha256=sha,member=member.name,
                    catalog_timestamp=timestamp,candidate_mechanisms=sorted(hits)))
                covered.update(hits)
            if scanned>=maximum or len(covered)==len(TARGETS):break
    result=dict(schema='cr-source-selection.v1',selected=len(records),scanned=scanned,records=records,
        candidate_coverage=sorted(covered),missing_candidates=sorted(set(TARGETS)-covered),training_ready=False)
    atomic_json(output/'selection.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--maximum',type=int,default=5000);a=p.parse_args()
    r=stage(a.archive,a.catalog,a.output,a.maximum);print(json.dumps({k:v for k,v in r.items() if k!='records'}))
