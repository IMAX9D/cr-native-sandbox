"""Read-only deck/card diversity audit over one deck row per actor sequence."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json, math, os
from pathlib import Path
import numpy as np


def one(task):
    raw,split=task; p=Path(raw)
    offsets=np.load(p/'sequence_offsets.npy',mmap_mode='r',allow_pickle=False)
    decks=np.load(p/'own_deck_tokens.npy',mmap_mode='r',allow_pickle=False)[offsets[:-1]]
    deck_counts=Counter(','.join(map(str,sorted(map(int,row)))) for row in decks)
    card_counts=Counter(int(token) for row in decks for token in row if int(token)>0)
    return split,deck_counts,card_counts,len(decks)


def main():
    a=argparse.ArgumentParser();a.add_argument('--dataset-root',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--workers',type=int,default=4);args=a.parse_args()
    root=args.dataset_root.resolve();m=json.loads((root/'manifest.json').read_text(encoding='utf-8-sig'))
    tasks=[(str((root/p).resolve()),split) for split,paths in m['splits'].items() for p in paths]
    merged={split:{'decks':Counter(),'cards':Counter(),'sequences':0} for split in m['splits']}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for split,decks,cards,n in pool.map(one,tasks):
            merged[split]['decks'].update(decks);merged[split]['cards'].update(cards);merged[split]['sequences']+=n
    result={'kind':'expert_deck_diversity_audit_v1','read_only':True,'splits':{}}
    vocab=m['card_vocabulary']
    for split,x in merged.items():
        counts=np.array(list(x['decks'].values()),dtype=np.int64);total=x['sequences'];unique=len(counts)
        probabilities=counts/counts.sum();effective=float(math.exp(-(probabilities*np.log(probabilities)).sum()))
        top=x['decks'].most_common(20)
        result['splits'][split]={
            'actor_sequences':total,'unique_unordered_decks':unique,'singleton_decks':int((counts==1).sum()),
            'median_actor_sequences_per_deck':float(np.median(counts)),'effective_deck_count_entropy':effective,
            'top1_fraction':top[0][1]/total,'top10_fraction':sum(v for _,v in x['decks'].most_common(10))/total,
            'top100_fraction':sum(v for _,v in x['decks'].most_common(100))/total,
            'top_decks':[{'tokens':[int(v) for v in key.split(',')],'cards':[vocab[int(v)] for v in key.split(',')],'actor_sequences':count,'fraction':count/total} for key,count in top],
            'deck_counts': dict(x['decks']),
            'card_coverage':[{'token':token,'card':vocab[token],'actor_sequences':x['cards'].get(token,0),'fraction':x['cards'].get(token,0)/total} for token in range(1,len(vocab))],
            'cards_under_100_actor_sequences':sum(x['cards'].get(token,0)<100 for token in range(1,len(vocab))),
            'cards_under_1000_actor_sequences':sum(x['cards'].get(token,0)<1000 for token in range(1,len(vocab))),
        }
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'output':str(args.output),'splits':{k:{'actor_sequences':v['actor_sequences'],'unique_decks':v['unique_unordered_decks']} for k,v in result['splits'].items()}}))


if __name__=='__main__':main()
