"""Create reproducible native replay fixtures for the most common compiled decks."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--deck-report',type=Path,required=True);p.add_argument('--dataset-manifest',type=Path,required=True);p.add_argument('--card-catalog',type=Path,required=True);p.add_argument('--template',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=10);a=p.parse_args()
    report=json.loads(a.deck_report.read_text());manifest=json.loads(a.dataset_manifest.read_text(encoding='utf-8-sig'));catalog=json.loads(a.card_catalog.read_text())['cards'];template=json.loads(a.template.read_text())
    forms={}
    for card in catalog:
        forms[int(card['card_id'])]=(int(card['card_id']),0)
        if card.get('evolution_form_id'):forms[int(card['evolution_form_id'])]=(int(card['card_id']),1)
        if card.get('hero_form_id'):forms[int(card['hero_form_id'])]=(int(card['card_id']),2)
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);rows=[]
    for rank,item in enumerate(report['splits']['validation']['top_decks'][:a.count],1):
        spells=[]
        for token in item['tokens']:
            raw=int(manifest['card_vocabulary'][token].rsplit('@',1)[1]);base,flag=forms[raw];value={'d':base,'l':10}
            if flag:value['el']=flag
            spells.append(value)
        if len({x['d'] for x in spells})!=8:raise RuntimeError('top deck has duplicate base card')
        replay=json.loads(json.dumps(template));replay['rndSeed']=rank
        for side in range(2):replay['battle'][f'deck{side}']['sp']=spells
        path=out/f'deck-{rank:02d}.json';path.write_text(json.dumps(replay,indent=2)+'\n')
        rows.append({'rank':rank,'path':str(path),'tokens':item['tokens'],'cards':item['cards'],'actor_sequences':item['actor_sequences'],'fraction':item['fraction']})
    value={'kind':'expert_top_deck_native_fixtures_v1','count':len(rows),'source_report_sha256':hashlib.sha256(a.deck_report.read_bytes()).hexdigest(),'decks':rows}
    (out/'manifest.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))


if __name__=='__main__':main()
