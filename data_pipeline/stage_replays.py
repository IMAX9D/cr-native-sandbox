"""Copy selected, catalog-verified original members without unpacking the corpus."""
import argparse
import hashlib
import json
import lzma
from pathlib import Path
import sqlite3
import tarfile
from .prepare_600k import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--battle-tag',action='append',required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);sources=a.output/'sources';sources.mkdir()
    c=sqlite3.connect(a.catalog.as_uri()+'?mode=ro',uri=True);wanted={}
    for tag in a.battle_tag:
        row=c.execute('select member,sha256,size,timestamp,schema_version from battles where tag=? and archive=?',(tag,a.archive.name)).fetchone()
        if row is None:raise ValueError('battle absent from requested archive catalog')
        wanted[row[0]]=(tag,*row)
    result=[]
    with lzma.open(a.archive,'rb') as f,tarfile.open(fileobj=f,mode='r|') as tar:
        for member in tar:
            if member.name not in wanted:continue
            tag,name,sha,size,timestamp,schema=wanted.pop(member.name);raw=tar.extractfile(member).read()
            if len(raw)!=size or hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('member checksum mismatch')
            target=sources/(tag+'.json')
            with target.open('xb') as out:out.write(raw)
            result.append(dict(battle_tag=tag,member=name,sha256=sha,source_file=str(target.resolve()),catalog_timestamp=timestamp,catalog_schema=schema))
            if not wanted:break
    if wanted:raise ValueError('archive members missing')
    atomic_json(a.output/'selection.json',dict(schema='cr-source-selection.v1',selected=len(result),records=result,training_ready=False))
    print(json.dumps(dict(staged=len(result),output=str(a.output))))


if __name__=='__main__':main()
