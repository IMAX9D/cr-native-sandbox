"""Classify saved receipts and dry-run candidate policy without changing captures."""
import argparse
from collections import Counter
import io
import json
from pathlib import Path

import zstandard as zstd

from .prepare_600k import atomic_json
from .firstlight_resolution import select_ability


def run(root):
    totals=Counter();rows=[];proposals=Counter()
    for directory in sorted((root/'raw-capture-v2').iterdir()):
        if directory.name.endswith('.partial'):continue
        if not (directory/'report.json').exists():continue
        report=json.loads((directory/'report.json').read_text('utf-8'))
        if not (directory/'native-frames.jsonl.zst').exists():
            rows.append(dict(battle_tag=directory.name,capture_error=report.get('error')));continue
        counts=Counter();first=None;state=None;pending_ability=None;examples=[]
        with (directory/'native-frames.jsonl.zst').open('rb') as src,zstd.ZstdDecompressor().stream_reader(src) as raw,io.TextIOWrapper(raw,encoding='utf-8') as text:
            for line in text:
                r=json.loads(line)
                if r['kind']=='frame':state=r['state']
                elif r['kind']=='source_action':
                    pending_ability=r if r['source_kind']=='ability' else None
                elif r['kind']=='command_receipt':
                    result=r['response'].get('result',{})
                    if result.get('accepted') is False:
                        reason=result.get('reason') or result.get('result_reason') or 'result_code_'+str(result.get('result_code'))
                        counts[reason]+=1
                        if first is None:first=r['tick']
                elif r['kind']=='source_action_not_executed':
                    counts[r['reason']]+=1
                    if first is None:first=r['tick']
                    if state is not None:
                        candidate=select_ability(state['entities'],side=r['side'])
                        proposals[candidate['status']]+=1
                        if len(examples)<4:examples.append(dict(tick=r['tick'],proposal=candidate))
        totals.update(counts)
        rows.append(dict(battle_tag=directory.name,rejection_reasons=dict(counts),first_observed_action_divergence_tick=first,
                         note='Not a proof earlier state was identical; fixed seed and historical rules may diverge earlier.',dry_run_examples=examples))
    result=dict(audited_attempts=len(rows),rejection_reasons=dict(totals),unresolved_ability_policy_proposals=dict(proposals),
                policy_executed=False,training_ready=False,rows=rows)
    atomic_json(root/'rejection-analysis.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
