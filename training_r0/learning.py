"""CPU-testable recurrent PPO minibatch and partial gate IL loss.

No dataset loop, native collector, checkpoint publisher or cloud runner here.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from .config import Temperatures, TrainingConfig
from .model import PolicyContext, R0Policy, RecurrentState
from .observation import PublicBatch
from .rollout import BehaviorIdentity, RolloutSegment, advantages


def gate_imitation_loss(model: R0Policy, observation: PublicBatch, context: PolicyContext,
                        occurred: Tensor, known: Tensor) -> Tensor:
    """A known ACT with unknown candidate is not relabelled as WAIT.

    More general candidate/target marginalization belongs to the IL compiler;
    this function deliberately does not fabricate conditional action labels.
    """
    if occurred.shape != (observation.batch_size,) or known.shape != occurred.shape or occurred.dtype != torch.bool or known.dtype != torch.bool:
        raise ValueError('gate supervision requires bool [B] labels and known mask')
    distribution = model.gate_distribution(observation,context,Temperatures())
    logp = distribution.log_prob(occurred.long())
    if bool((known & ~torch.isfinite(logp)).any()):
        raise ValueError('known action label conflicts with legal candidate set')
    return -torch.where(known,logp,0.).sum()/known.sum().clamp_min(1)


def update_minibatch(model: R0Policy, optimizer: torch.optim.Optimizer, segment: RolloutSegment,
                     *, expected_behavior: BehaviorIdentity, config: TrainingConfig = TrainingConfig()) -> dict:
    """One optimizer step AFTER all recurrent time chunks, not one per chunk."""
    segment.validate(max_steps=config.collection_steps)
    if segment.behavior != expected_behavior or model.observation_schema_hash != segment.behavior.observation_schema_hash or model.model_schema_hash != segment.behavior.model_schema_hash:
        raise ValueError('behavior/temperature/runtime/observation contract mismatch')
    device = next(model.parameters()).device
    valid = segment.valid.to(device)
    count = int(valid.sum())
    adv,returns = advantages(segment,gamma=config.gamma,gae_lambda=config.gae_lambda)
    adv,returns = adv.to(device),returns.to(device)
    selected = adv[valid]
    adv = (adv-selected.mean())/selected.std(unbiased=False).clamp_min(1e-8)
    old_logp = segment.logp.to(device)
    state = segment.initial_state.to(device)
    optimizer.zero_grad(set_to_none=True)
    sums = dict(policy=0.,value=0.,entropy=0.,kl=0.,clip_fraction=0.)
    chunks = 0
    model.train()
    try:
        for start in range(0,segment.time_steps,config.tbptt_steps):
            chunks += 1
            losses = []
            for step in range(start,min(start+config.tbptt_steps,segment.time_steps)):
                output = model(segment.observations[step].to(device),state,
                    episode_start=segment.episode_start[step].to(device),forced=segment.actions[step].to(device),
                    temperatures=segment.behavior.temperatures)
                alive = valid[step]
                state = RecurrentState(torch.where(alive[:,None],output.next_state.hidden,state.hidden),
                                       torch.where(alive[:,None],output.next_state.cell,state.cell))
                log_ratio = output.logp-old_logp[step]
                ratio = log_ratio.exp()
                if not bool(torch.isfinite(ratio[alive]).all()) or not bool(torch.isfinite(output.value[alive]).all()):
                    raise FloatingPointError('nonfinite learner output/importance ratio')
                policy = -torch.minimum(ratio*adv[step],ratio.clamp(1-config.clip_range,1+config.clip_range)*adv[step])
                value = (output.value-returns[step]).square()
                entropy = output.entropy
                kl = torch.expm1(log_ratio)-log_ratio
                clip = (ratio.sub(1).abs()>config.clip_range).float()
                loss = policy + config.value_coefficient*value - config.entropy_coefficient*entropy
                losses.append(torch.where(alive,loss,0.).sum()/count)
                for name,term in dict(policy=policy,value=value,entropy=entropy,kl=kl,clip_fraction=clip).items():
                    sums[name] += float(torch.where(alive,term.detach(),0.).sum())/count
            torch.stack(losses).sum().backward()
            state = state.detach()  # carry values, truncate gradients only
        if not all(math.isfinite(value) for value in sums.values()):
            raise FloatingPointError('nonfinite PPO metrics')
        if sums['kl'] > config.target_kl:
            optimizer.zero_grad(set_to_none=True)
            return {**sums,'updated':False,'reason':'kl_guard','valid_decisions':count,'tbptt_chunks':chunks,'optimizer_steps':0}
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(),config.max_grad_norm,error_if_nonfinite=True))
        optimizer.step()
        if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
            raise FloatingPointError('nonfinite weights after optimizer step; must not publish')
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        raise
    return {**sums,'updated':True,'gradient_norm_preclip':norm,'valid_decisions':count,'tbptt_chunks':chunks,'optimizer_steps':1}
