"""Audit observed transitions around deployment labels; never shift labels or fit a model."""

from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from policy_v1.data import Windows
from policy_v1.train import load_checkpoint
from .model import Config, Policy
from .diagnose_timing import collect, sequence_plan, parser as context_parser


SOURCE_NOTE = (
    'Code intends to keep the pre-action boundary state, ignore the next trace\'s '
    'duplicate same-tick initial_frame, and supervise at source_tick + execution_offset. '
    'A visible state change at label+1 is therefore compatible with correct alignment; '
    'a model reacting at +1 is not evidence to move labels forward.'
)


def summarize_deployments(record, arrays, lo, hi, radius=10, examples=8):
    """All indices are local to one actor sequence; require a full valid neighborhood."""
    ticks = np.asarray(record['ticks'])
    p = np.asarray(record['probabilities'])
    valid = np.asarray(record['valid'],dtype=bool)
    actual = np.asarray(record['labels'],dtype=bool) & valid
    labels = arrays['card_label_mask'][lo:hi].astype(bool)
    kind_valid = arrays['kind_label_mask'][lo:hi].astype(bool)
    kinds = arrays['action_kind'][lo:hi]
    deploy = actual & labels & kind_valid & (kinds == 0)
    hand = arrays['hand_tokens'][lo:hi]
    slots = arrays['card_slot'][lo:hi]
    elixir = arrays['public_scalars'][lo:hi,1]
    offsets = np.arange(-radius,radius+1)
    counts = Counter(candidate_deployments=int(deploy.sum()))
    cost_hist = Counter(); hand_hist = Counter(); jump_hist = Counter()
    curves = []; elixir_curves = []; examples_out = []
    for event in np.flatnonzero(deploy):
        begin, end = event-radius-1, event+radius+1
        if begin<0 or end>len(ticks) or not valid[begin:end].all():
            counts['excluded_boundary_or_invalid'] += 1
            continue
        if actual[begin:end].sum()!=1:
            counts['excluded_nearby_own_action'] += 1
            continue
        slot = int(slots[event])
        if not 0<=slot<4:
            raise ValueError('invalid supervised hand slot')
        counts['isolated_deployments'] += 1
        ix = event+offsets
        # Differences at offset k are state[t+k] minus state[t+k-1].
        delta_elixir = np.asarray(elixir[ix]-elixir[ix-1],dtype=np.float64)
        changed = np.any(hand[ix]!=hand[ix-1],axis=-1)
        drops = delta_elixir < -.02  # ratio drop >0.02; does not assume a card cost table
        for offset,name in ((0,'at_label'),(1,'next_tick')):
            counts[name+'_elixir_drop'] += int(drops[radius+offset])
            counts[name+'_hand_change'] += int(changed[radius+offset])
        if drops.any():
            cost_hist[str(int(offsets[np.argmin(delta_elixir)]))] += 1
        else:
            cost_hist['none'] += 1
        near_change = np.flatnonzero(changed)
        if len(near_change):
            nearest = min(near_change,key=lambda j:(abs(offsets[j]),offsets[j]))
            hand_hist[str(int(offsets[nearest]))] += 1
        else:
            hand_hist['none'] += 1
        delta_p = p[ix]-p[ix-1]
        if delta_p.max()>1e-9:
            jump_hist[str(int(offsets[np.argmax(delta_p)]))] += 1
        else:
            jump_hist['no_positive_jump'] += 1
        curves.append(p[ix].astype(np.float64))
        elixir_curves.append(delta_elixir)
        if len(examples_out)<examples:
            selected_token = int(hand[event,slot])
            states = []
            for offset,row in zip(offsets,ix):
                start,stop = map(int,arrays['entity_offsets'][lo+row:lo+row+2])
                own = arrays['entity_relations'][start:stop] == 0
                tokens = arrays['entity_tokens'][start:stop]
                states.append({
                    'offset_ticks':int(offset),'tick':int(ticks[row]),
                    'probability':float(p[row]),'elixir_ratio':float(elixir[row]),
                    'elixir_delta_ratio':float(elixir[row]-elixir[row-1]),
                    'hand_tokens':hand[row].tolist(),
                    'own_entities':int(own.sum()),
                    'own_entities_with_selected_token':int((own & (tokens==selected_token)).sum()),
                })
            examples_out.append({'label_tick':int(ticks[event]),'selected_slot':slot,
                                 'selected_token_at_label':selected_token,
                                 'note':'token read from labeled hand slot, not independently verified source-event identity',
                                 'frames':states})
    return counts,cost_hist,hand_hist,jump_hist,curves,elixir_curves,examples_out


def analyze(dataset,chosen,records,radius,example_limit):
    counts=Counter();cost=Counter();hands=Counter();jumps=Counter()
    curves=[]; elixir_curves=[]; examples=[]
    for (sh,seq),record in zip(chosen,records):
        arrays,_,_=dataset._open(sh)
        lo,hi=map(int,dataset.records[sh]['offsets'][seq:seq+2])
        if len(record['ticks']) != hi-lo:
            raise ValueError('array/model sequence length mismatch')
        result=summarize_deployments(record,arrays,lo,hi,radius,max(0,example_limit-len(examples)))
        c,e,h,j,ps,es,samples=result
        counts.update(c);cost.update(e);hands.update(h);jumps.update(j)
        curves.extend(ps);elixir_curves.extend(es)
        for sample in samples:
            sample.update(shard=dataset.records[sh]['path'],sequence_index=seq)
        examples.extend(samples)
    n=counts['isolated_deployments']
    rate_keys=('at_label_elixir_drop','next_tick_elixir_drop','at_label_hand_change','next_tick_hand_change')
    offsets=np.arange(-radius,radius+1)
    curve=[]
    if n:
        values=np.stack(curves); changes=np.stack(elixir_curves)
        for i,offset in enumerate(offsets):
            curve.append({'offset_ticks':int(offset),'offset_ms':int(offset*50),
                          'mean_probability':float(values[:,i].mean()),
                          'median_probability':float(np.median(values[:,i])),
                          'mean_change_from_t_minus_1':float((values[:,i]-values[:,radius-1]).mean()),
                          'mean_elixir_delta_ratio':float(changes[:,i].mean())})
    return {'counts':dict(counts),'transition_rates':{k:counts[k]/n if n else None for k in rate_keys},
            'largest_elixir_drop_offset_histogram':dict(cost),
            'nearest_hand_change_offset_histogram':dict(hands),
            'largest_positive_probability_jump_offset_histogram':dict(jumps),
            'event_aligned_probability_curve':curve,'examples':examples,
            'drop_threshold_ratio':.02,
            'interpretation':'State-transition correlations only, not proof of real-client alignment. '
             'Nearby own actions and incomplete neighborhoods excluded; opponent actions/deaths/refill may still confound. '
             'Spells and deployment delays may have no immediate unit-count increase.'}


def source_review(data):
    manifest=json.loads((data/'manifest.json').read_text())
    components=manifest.get('compiler',{}).get('components',{})
    repo=Path(__file__).resolve().parents[1]
    checked={}
    for key,relative in (
        ('compiler_sha256','expert_v1/compile_native_bc_dataset.py'),
        ('native_dataset_generator_sha256','expert_v1/native_dataset_generator.py'),
    ):
        path=repo/relative
        checked[relative]=(hashlib.sha256(path.read_bytes()).hexdigest()==components[key]
                           if path.is_file() and key in components else None)
    return {'intent':SOURCE_NOTE,'manifest_action_alignment':manifest.get('compiler',{}).get('action_alignment'),
            'local_file_matches_manifest':checked,
            'trace_reference':'expert_v1/tick_store_v1/trace.py: TickTraceAccumulator.start/extend',
            'scope':'Trace implementation itself is not independently pinned by these two component hashes.'}


def parser():
    p=context_parser()
    p.description=__doc__
    p.set_defaults(sequences=64)
    p.add_argument('--arm',choices=('baseline','weighted'),default='baseline')
    p.add_argument('--radius',type=int,default=10,help='ticks before/after each deployment')
    p.add_argument('--examples',type=int,default=8)
    return p


def run(args):
    if min(args.sequences,args.batch_size,args.cpu_threads,args.radius)<1 or min(args.workers,args.examples)<0:
        raise ValueError('invalid arguments')
    if args.checkpoint and args.comparison_dir:
        raise ValueError('choose checkpoint OR comparison directory')
    checkpoint=args.checkpoint
    if checkpoint is None:
        root=args.comparison_dir
        if root is None:
            dirs=[d for d in args.runs.glob('hokoff-weight-compare-*')
                  if (d/'baseline/last.pt').is_file() and (d/'weighted/last.pt').is_file()]
            if not dirs:raise FileNotFoundError('No weight comparison found; pass --checkpoint')
            root=max(dirs,key=lambda d:(d/'weighted/last.pt').stat().st_mtime_ns)
        checkpoint=root/args.arm/'last.pt'
    output=args.output or checkpoint.parent/('alignment-audit-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'.json')
    if output.exists():raise FileExistsError(output)
    saved=load_checkpoint(checkpoint);contract=saved['contract']
    if saved['config'].get('architecture')!='hokoff_cr_lstm_v1':raise ValueError('wrong architecture')
    split=contract['val_split']
    if split=='test' or split==contract['train_split']:raise ValueError('requires non-test held-out split')
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(args.cpu_threads)
    config=Config(**saved['config'])
    dataset=Windows(args.data,args.cache,split,targets=contract['targets'],frame_window=config.frame_window,event_window=1)
    if dataset.index['manifest_sha256']!=contract['manifest_sha256']:raise ValueError('manifest differs')
    if dataset.index['smoke_only'] and not args.allow_smoke:raise ValueError('synthetic data requires --allow-smoke')
    chosen,indices,owners=sequence_plan(dataset,args.sequences,args.seed)
    if not chosen:raise ValueError('empty split')
    print(json.dumps({'phase':'alignment_audit_start','checkpoint':str(checkpoint),'split':split,
                      'sequences':len(chosen),'radius_ticks':args.radius}),flush=True)
    model=Policy(config).to(device).eval();model.load_state_dict(saved['model'])
    records=collect(model,dataset,chosen,indices,owners,args,device)
    report=analyze(dataset,chosen,records,args.radius,args.examples)
    report.update(source_review=source_review(args.data),checkpoint=str(checkpoint),checkpoint_step=saved['step'],
                  split=split,sequences=len(chosen),seed=args.seed,precision='fp32',
                  selection=[{'shard':dataset.records[sh]['path'],'sequence_index':seq} for sh,seq in chosen])
    summary={'phase':'alignment_summary','report_path':str(output),
             **{k:report[k] for k in ('checkpoint','counts','transition_rates',
                  'largest_elixir_drop_offset_histogram','nearest_hand_change_offset_histogram',
                  'largest_positive_probability_jump_offset_histogram','event_aligned_probability_curve','source_review')}}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps(summary),flush=True)
    return summary


def main():
    run(parser().parse_args())


if __name__=='__main__':main()
