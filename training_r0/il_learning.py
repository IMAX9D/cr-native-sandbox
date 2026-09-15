"""Strict recurrent imitation update using the exact R0 forced-action API."""
import torch
from .model import RecurrentState
from .config import Temperatures
from .il_dataset import ILSequence
from data_pipeline.r0_admission import require_batch


def update_imitation_sequence(model,optimizer,sequence:ILSequence,*,initial_state=None,tbptt_steps=16):
    sequence.validate()
    if model.config.allow_incomplete_capture:raise ValueError('diagnostic model cannot perform accepted R0 IL training')
    if not sequence.starts_episode and initial_state is None:raise ValueError('mid-episode sequence requires carried recurrent state')
    if tbptt_steps<1:raise ValueError('invalid TBPTT size')
    device=next(model.parameters()).device
    state=initial_state.to(device) if initial_state is not None else model.initial_state(sequence.observations[0].batch_size)
    count=int(sequence.label_known.sum())
    if count==0:raise ValueError('no accepted imitation targets')
    optimizer.zero_grad(set_to_none=True);total=0.;chunks=0
    try:
        for start in range(0,len(sequence.observations),tbptt_steps):
            terms=[]
            for t in range(start,min(start+tbptt_steps,len(sequence.observations))):
                obs=sequence.observations[t].to(device);require_batch(obs,model)
                known=sequence.label_known[t].to(device)
                output=model(obs,state,forced=sequence.actions[t].to(device),temperatures=Temperatures())
                if not torch.isfinite(output.logp[known]).all():raise ValueError('expert labels conflict with native candidate legality')
                terms.append(-torch.where(known,output.logp,0.).sum()/count)
                state=output.next_state
            loss=torch.stack(terms).sum()
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite IL loss')
            loss.backward();total+=float(loss.detach());state=state.detach();chunks+=1
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError('nonfinite updated R0 weights')
    except BaseException:
        # A failed later TBPTT chunk must not leak earlier accumulated gradients
        # into a subsequent update. An optimizer-step failure still invalidates
        # the caller's model/optimizer; this is not a transactional weight rollback.
        optimizer.zero_grad(set_to_none=True)
        raise
    return dict(loss=total,gradient_norm=float(norm),valid_decisions=count,tbptt_chunks=chunks,optimizer_steps=1),state
