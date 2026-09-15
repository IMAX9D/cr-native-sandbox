"""Check saved opening layouts against public card sequences, without rerunning."""
from collections import Counter
import argparse
import io
import json
from pathlib import Path
import zstandard as zstd
from .prepare_600k import atomic_json


def run(root):
    rows=[];counts=Counter()
    for directory in sorted((root/'raw-capture-v2').iterdir()):
        if directory.name.endswith('.partial') or not (directory/'report.json').exists():continue
        path=directory/'native-frames.jsonl.zst'
        if not path.exists():continue
        first=None
        with path.open('rb') as f,zstd.ZstdDecompressor().stream_reader(f) as raw,io.TextIOWrapper(raw,encoding='utf-8') as text:
            for line in text:
                r=json.loads(line)
                if r['kind']=='frame':first=r['state'];break
        if first is None:continue
        plan=json.loads((directory/'plan.json').read_text('utf-8'));sides=[]
        for p in first['players']:
            hand=set(p['hand_deck_indices']);queue=list(p['cycle_deck_indices']);failure=None
            for action in (a for a in plan['actions'] if a['side']==p['side']):
                card=action['logical_card_index']
                if card not in hand:
                    failure=action['tick'];break
                hand.remove(card);hand.add(queue.pop(0));queue.append(card)
            sides.append(dict(side=p['side'],compatible=failure is None,first_incompatible_source_tick=failure))
        counts['both_compatible' if all(x['compatible'] for x in sides) else 'at_least_one_side_incompatible']+=1
        counts['capture_start_tick_'+str(first['tick'])]+=1
        rows.append(dict(battle_tag=directory.name,capture_start_tick=first['tick'],sides=sides))
    result=dict(audited_captures=len(rows),counts=dict(counts),rows=rows,
                note='Shadow eight-card cycle check from first captured state, not proof of original hidden deal. Fixed diagnostic seed was not calibrated.')
    atomic_json(root/'initial-deal-audit.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
