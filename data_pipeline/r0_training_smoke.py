"""Strict real-sequence IL proof: default R0, one update, checkpoint restoration.

This is bounded engineering acceptance, not expert-model quality evaluation and
not blanket admission of the source archive. No relaxed capture switch exists.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import tempfile
import time

import torch

from training_r0.catalog import CardVocabulary
from training_r0.config import ModelConfig, DECISION_TICKS, Temperatures
from training_r0.il_dataset import ILSequence,load_sequence
from training_r0.il_learning import update_imitation_sequence
from training_r0.model import R0Policy,RecurrentState
from training_r0.session import weights_hash
from .prepare_600k import atomic_json,digest
from .r0_admission import current_contract,require_batch


def sequence_window(root,*,side=0,steps=8,first_tick=None):
    """Read a continuous prefix plus an action-bearing training window, one lane.

    first_tick selects a later window (for example the region around an
    attribution repair) while the whole lane before it is still replayed as
    burn-in, so the recurrent state stays continuous from Tick 0.
    """
    root=Path(root).resolve();manifest=json.loads((root/'manifest.json').read_text('utf-8'))
    if manifest.get('contract_sha256')!=current_contract()['sha256']:raise ValueError('compiled contract differs from current R0; recompile')
    if manifest.get('sequence_gaps'):raise ValueError('cannot carry recurrence through missing decision observations')
    shards=sorted((s for s in manifest['shards'] if s['side']==side),key=lambda s:s['first_tick'])
    if not shards or shards[0]['first_tick']!=0:raise ValueError('strict smoke needs a Tick-0 recurrent prefix')
    prefix=[];selected=[];previous=-DECISION_TICKS;uid=None;source_sha=None;used=[]
    for shard in shards:
        path=(root/shard['path']).resolve()
        if not path.is_relative_to(root):raise ValueError('shard path outside dataset')
        sequence=load_sequence(path,expected_sha256=shard['sha256']);used.append(shard)
        if uid is None:uid=sequence.observations[0].episode_uids;source_sha=sequence.source_sha256
        if source_sha!=sequence.source_sha256:raise ValueError('source changed across chunks')
        for t,(obs,action) in enumerate(zip(sequence.observations,sequence.actions)):
            if obs.sides!=(side,) or obs.episode_uids!=uid or obs.ticks!=(previous+DECISION_TICKS,):raise ValueError('recurrent lane discontinuity')
            previous=obs.ticks[0];known=sequence.label_known[t].clone()
            if first_tick is not None and obs.ticks[0]<first_tick:prefix.append(obs);continue
            if not selected and not (bool(known[0]) and int(action.count[0])>0):prefix.append(obs);continue
            selected.append((obs,action,known))
            if len(selected)>=steps:break
        if len(selected)>=steps:break
    if not selected:raise ValueError('no known expert action target in this lane')
    target=ILSequence(tuple(x[0] for x in selected),tuple(x[1] for x in selected),torch.stack([x[2] for x in selected]),not prefix,source_sha)
    target.validate()
    return prefix,target,used


def assert_tree_equal(left,right):
    if isinstance(left,torch.Tensor):
        if not isinstance(right,torch.Tensor) or left.dtype!=right.dtype or not torch.equal(left.cpu(),right.cpu()):raise ValueError('checkpoint tensor restoration mismatch')
    elif isinstance(left,dict):
        if not isinstance(right,dict) or left.keys()!=right.keys():raise ValueError('checkpoint mapping restoration mismatch')
        for key in left:assert_tree_equal(left[key],right[key])
    elif isinstance(left,(tuple,list)):
        if type(left)!=type(right) or len(left)!=len(right):raise ValueError('checkpoint sequence restoration mismatch')
        for a,b in zip(left,right):assert_tree_equal(a,b)
    elif left!=right:raise ValueError('checkpoint value restoration mismatch')


def save_checkpoint(path,payload):
    path=Path(path);temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='wb',prefix=path.name+'.partial-',dir=path.parent,delete=False) as f:
            temporary=Path(f.name);torch.save(payload,f);f.flush();os.fsync(f.fileno())
        os.link(temporary,path)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)
    return digest(path)


def run(dataset,output,*,device='cuda',side=0,steps=8,tbptt_steps=2,learning_rate=1e-5,first_tick=None):
    output=Path(output)
    if output.exists():raise FileExistsError('use a new smoke output directory')
    if side not in (0,1) or not 1<=steps<=64 or not 1<=tbptt_steps<=steps:raise ValueError('bounded smoke requires valid side, 1..64 steps and TBPTT <= steps')
    if not math.isfinite(learning_rate) or learning_rate<=0:raise ValueError('invalid smoke learning rate')
    output.mkdir(parents=True);started=time.perf_counter()
    evidence=dict(kind='r0-real-il-training-smoke.v1',status='running',dataset=str(Path(dataset).resolve()),
        dataset_manifest_sha256=digest(Path(dataset)/'manifest.json'),whole_archive_training_ready=False,checks={})
    try:
        prefix,sequence,used=sequence_window(dataset,side=side,steps=steps,first_tick=first_tick)
        config=ModelConfig();torch.manual_seed(42)
        model=R0Policy(CardVocabulary.from_native(),config).to(device)
        model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate)
        evidence.update(model_config=asdict(config),parameter_count=sum(p.numel() for p in model.parameters()),device=str(device),
            side=side,burnin_decisions=len(prefix),training_decisions=len(sequence.observations),first_training_tick=sequence.observations[0].ticks[0],
            last_training_tick=sequence.observations[-1].ticks[0],requested_first_tick=first_tick,source_sha256=sequence.source_sha256,shards=used,
            action_targets=sum(int(a.count[0]) for a,k in zip(sequence.actions,sequence.label_known) if bool(k[0])))
        # Reject incomplete capture before any optimizer mutation.
        for observation in (*prefix,*sequence.observations):require_batch(observation.to(device),model)
        evidence['checks']['strict_batch_contract']=True
        state=model.initial_state(1)
        with torch.no_grad():
            for observation in prefix:
                state=model(observation.to(device),state,sample=False).next_state
        before=weights_hash(model)
        metrics,final_state=update_imitation_sequence(model,optimizer,sequence,initial_state=state if prefix else None,tbptt_steps=tbptt_steps)
        if not all(math.isfinite(metrics[k]) for k in ('loss','gradient_norm')):raise ValueError('nonfinite update metrics')
        after=weights_hash(model)
        if before==after:raise ValueError('optimizer did not change model weights')
        evidence['checks'].update(finite_loss_and_gradient=True,parameters_changed=True)
        payload=dict(kind='r0-real-il-smoke-checkpoint.v1',model_config=asdict(config),model=model.state_dict(),optimizer=optimizer.state_dict(),
            recurrent_state=dict(hidden=final_state.hidden,cell=final_state.cell),contract_sha256=current_contract()['sha256'],
            source_sha256=sequence.source_sha256,model_weights_sha256=after,optimizer_steps=1)
        checkpoint=output/'checkpoint.pt';checkpoint_sha=save_checkpoint(checkpoint,payload)
        saved=torch.load(checkpoint,map_location=device,weights_only=True)
        restored=R0Policy(CardVocabulary.from_native(),config).to(device)
        restored.load_state_dict(saved['model'],strict=True)
        restored_optimizer=torch.optim.AdamW(restored.parameters(),lr=learning_rate);restored_optimizer.load_state_dict(saved['optimizer'])
        if weights_hash(restored)!=after:raise ValueError('restored model weights differ')
        assert_tree_equal(optimizer.state_dict(),restored_optimizer.state_dict())
        restored_state=RecurrentState(**saved['recurrent_state'])
        assert_tree_equal(dict(hidden=final_state.hidden,cell=final_state.cell),saved['recurrent_state'])
        # Fixed observation/forced command and identical recurrent state exercise
        # restoration independently of sampled-action randomness.
        model.eval();restored.eval();obs=sequence.observations[-1].to(device);forced=sequence.actions[-1].to(device)
        with torch.no_grad():
            original=model(obs,final_state,forced=forced,temperatures=Temperatures())
            replica=restored(obs,restored_state,forced=forced,temperatures=Temperatures())
        for name in ('logp','value'):
            a,b=getattr(original,name),getattr(replica,name)
            if not torch.isfinite(a).all() or not torch.allclose(a,b,rtol=1e-6,atol=1e-7):raise ValueError('restored policy output differs')
        evidence['checks'].update(model_restored=True,optimizer_restored=True,recurrent_state_restored=True,restored_forward_matches=True)
        evidence.update(status='passed',metrics=metrics,weights_before_sha256=before,weights_after_sha256=after,
            checkpoint=dict(path=checkpoint.name,sha256=checkpoint_sha),acceptance_scope='Only referenced real sequence prefix/window; no claim of full mechanism, historical ruleset, archive coverage or policy quality.')
    except Exception as error:
        evidence.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        evidence['elapsed_seconds']=time.perf_counter()-started;atomic_json(output/'report.json',evidence)
    return evidence


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('dataset',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--side',type=int,default=0);p.add_argument('--steps',type=int,default=8)
    p.add_argument('--tbptt-steps',type=int,default=2)
    a=p.parse_args();print(json.dumps(run(a.dataset,a.output,device=a.device,side=a.side,steps=a.steps,tbptt_steps=a.tbptt_steps),indent=2))


if __name__=='__main__':main()
