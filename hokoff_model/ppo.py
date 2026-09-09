"""Head-only fixed4 PPO with an immutable IL reference and rollback KL guards."""
from copy import deepcopy
from dataclasses import dataclass,asdict
import hashlib
import math
import torch
from torch import nn
from expert_selfplay_v1.ppo import recurrent_ppo_loss
from .ppo_actions import train_action_heads_only,action_heads,action_distribution,exact_kl,entropy


@dataclass
class PPOConfig:
    actor_lr: float=1e-5
    critic_lr: float=3e-4
    epochs: int=2
    batch_size: int=256
    clip_epsilon: float=.1
    entropy_coefficient: float=.001
    kl_coefficient: float=1.
    il_kl_limit: float=.02
    il_action_kl_limit: float=.03
    update_kl_limit: float=.01
    max_state_kl: float=.2
    anchor_size: int=512

    def validate(self):
        for k,v in asdict(self).items():
            if not math.isfinite(v) or v<=0:raise ValueError('positive finite PPO setting required: '+k)
        if self.clip_epsilon>=1:raise ValueError('clip epsilon must be below one')


def state_digest(model,*,frozen_only=False):
    frozen={name for name,p in model.named_parameters() if not p.requires_grad}
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        if frozen_only and name not in frozen:continue
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def take(data,indices):return {k:v[indices] for k,v in data.items()}
def to_device(data,device):return {k:v.to(device) for k,v in data.items()}


class FixedPPO:
    def __init__(self,actor,reference,config,*,device='cpu'):
        config.validate();self.config=config;self.device=device
        self.actor=actor.to(device).eval()
        self.reference=reference.to(device).eval().requires_grad_(False)
        self.parameters=train_action_heads_only(actor)
        self.critic=nn.Sequential(nn.LayerNorm(actor.config.width),nn.Linear(actor.config.width,128),
                                  nn.SiLU(),nn.Linear(128,1)).to(device)
        nn.init.zeros_(self.critic[-1].weight);nn.init.zeros_(self.critic[-1].bias)
        self.actor_optimizer=torch.optim.Adam(self.parameters,lr=config.actor_lr)
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=config.critic_lr)
        self.beta=config.kl_coefficient;self.accepted_updates=0;self.rejected_updates=0
        self.anchor=None
        self.reference_hash=state_digest(reference)
        self.frozen_hash=state_digest(actor,frozen_only=True)

    def values(self,features):return self.critic(features['context'].detach()).squeeze(-1)

    def distribution(self,model,data):return action_distribution(action_heads(model,data),data)

    @torch.no_grad()
    def guard(self,data,reference_dist,old_dist,anchor,anchor_reference):
        current=self.distribution(self.actor,data)
        il,marks=exact_kl(reference_dist,current);old,old_marks=exact_kl(old_dist,current)
        available=reference_dist['any_action']
        def active_mean(v,m):return float(v[m].mean()) if m.any() else 0.
        result=dict(il_kl=float(il.mean()),il_action_kl=active_mean(marks,available),
                    il_kl_max=float(il.max()),il_action_kl_max=float(marks.max()),
                    update_kl=float(old.mean()),update_action_kl=active_mean(old_marks,available))
        a=self.distribution(self.actor,anchor)
        ak,am=exact_kl(anchor_reference,a)
        result.update(anchor_il_kl=float(ak.mean()),anchor_il_action_kl=active_mean(am,a['any_action']),
                      anchor_il_kl_max=float(ak.max()),anchor_il_action_kl_max=float(am.max()))
        c=self.config
        limits=dict(il_kl=c.il_kl_limit,il_action_kl=c.il_action_kl_limit,
                    il_kl_max=c.max_state_kl,il_action_kl_max=c.max_state_kl,
                    update_kl=c.update_kl_limit,update_action_kl=c.update_kl_limit,
                    anchor_il_kl=c.il_kl_limit,anchor_il_action_kl=c.il_action_kl_limit,
                    anchor_il_kl_max=c.max_state_kl,anchor_il_action_kl_max=c.max_state_kl)
        failures=[key for key,limit in limits.items() if not math.isfinite(result[key]) or result[key]>limit]
        return result,failures

    def update(self,raw,*,generator):
        c=self.config;data=to_device(raw,self.device);n=len(data['context'])
        if not n:raise ValueError('empty PPO rollout')
        if self.anchor is None:
            ids=torch.randperm(n,generator=generator,device=self.device)[:c.anchor_size]
            self.anchor={k:v[ids].detach().cpu().clone() for k,v in data.items()
                         if k in ('context','hand_tokens','ability_tokens','cards','positions','abilities')}
        anchor=to_device(self.anchor,self.device)
        with torch.no_grad():
            reference_dist=self.distribution(self.reference,data)
            old_dist=self.distribution(self.actor,data)
            anchor_reference=self.distribution(self.reference,anchor)
            recomputed=old_dist['log_prob'].gather(1,data['action'][:,None]).squeeze(1)
            mismatch=float((recomputed-data['old_log_prob']).abs().max())
            if mismatch>5e-4:raise ValueError(f'behavior log probabilities cannot be reproduced: {mismatch}')
        advantages=data['advantage']
        advantages=(advantages-advantages.mean())/advantages.std(unbiased=False).clamp_min(1e-6)
        before,failed=self.guard(data,reference_dist,old_dist,anchor,anchor_reference)
        if failed:raise RuntimeError('actor already violates IL guard on new observations: '+','.join(failed))
        records=[];stopped=False
        for epoch in range(c.epochs):
            order=torch.randperm(n,generator=generator,device=self.device)
            for start in range(0,n,c.batch_size):
                ids=order[start:start+c.batch_size];batch=take(data,ids)
                dist=self.distribution(self.actor,batch)
                reference=take(reference_dist,ids)
                joint_kl,mark_kl=exact_kl(reference,dist)
                log_prob=dist['log_prob'].gather(1,batch['action'][:,None]).squeeze(1)
                loss=recurrent_ppo_loss(new_log_prob=log_prob,old_log_prob=batch['old_log_prob'],
                    advantages=advantages[ids],values=self.values(batch),returns=batch['return'],
                    joint_entropy=entropy(dist),bc_kl=joint_kl+mark_kl,
                    loss_mask=torch.ones_like(log_prob,dtype=torch.bool),clip_epsilon=c.clip_epsilon,
                    entropy_coefficient=c.entropy_coefficient,bc_kl_coefficient=self.beta)
                if not torch.isfinite(loss.total):raise FloatingPointError('non-finite PPO objective')
                self.actor_optimizer.zero_grad(set_to_none=True);self.critic_optimizer.zero_grad(set_to_none=True)
                loss.total.backward()
                actor_norm=nn.utils.clip_grad_norm_(self.parameters,.5)
                critic_norm=nn.utils.clip_grad_norm_(self.critic.parameters(),1.)
                if not torch.isfinite(actor_norm) or not torch.isfinite(critic_norm):
                    raise FloatingPointError('non-finite PPO gradients')
                # The critic has no shared trainable parameter with the Actor.
                self.critic_optimizer.step()
                snapshot=[p.detach().clone() for p in self.parameters]
                optimizer_snapshot=deepcopy(self.actor_optimizer.state_dict())
                try:
                    self.actor_optimizer.step()
                    metrics,failures=self.guard(data,reference_dist,old_dist,anchor,anchor_reference)
                except BaseException:
                    # Includes interruption or numerical failure during candidate evaluation.
                    with torch.no_grad():
                        for p,old in zip(self.parameters,snapshot,strict=True):p.copy_(old)
                    self.actor_optimizer.load_state_dict(optimizer_snapshot)
                    raise
                accepted=not failures
                if not accepted:
                    with torch.no_grad():
                        for p,old in zip(self.parameters,snapshot,strict=True):p.copy_(old)
                    self.actor_optimizer.load_state_dict(optimizer_snapshot)
                    for group in self.actor_optimizer.param_groups:group['lr']=max(1e-7,group['lr']*.5)
                    self.beta=min(100.,self.beta*2)
                    self.rejected_updates+=1;stopped=True
                else:self.accepted_updates+=1
                records.append(dict(epoch=epoch,minibatch=start//c.batch_size,accepted=accepted,
                    rejected_by=failures,policy_loss=float(loss.policy.detach()),value_loss=float(loss.value.detach()),
                    entropy=float(loss.entropy.detach()),kl_penalty=float(loss.bc_kl.detach()),
                    approximate_update_kl=float(loss.approx_update_kl.detach()),clip_fraction=float(loss.clip_fraction.detach()),
                    actor_grad_norm=float(actor_norm),critic_grad_norm=float(critic_norm),**metrics))
                if stopped:break
            if stopped:break
        final,violations=self.guard(data,reference_dist,old_dist,anchor,anchor_reference)
        if violations:raise RuntimeError('post-update/rollback guard failure: '+','.join(violations))
        if state_digest(self.reference)!=self.reference_hash:raise RuntimeError('IL reference changed')
        if state_digest(self.actor,frozen_only=True)!=self.frozen_hash:raise RuntimeError('frozen encoder/LSTM changed')
        if not stopped:
            utilization=max(final['il_kl']/c.il_kl_limit,final['il_action_kl']/c.il_action_kl_limit)
            if utilization>.75:self.beta=min(100.,self.beta*2)
            elif utilization<.25:self.beta=max(.1,self.beta*.9)
        return dict(rollout_count=n,behavior_log_prob_max_error=mismatch,
                    accepted_updates=sum(r['accepted'] for r in records),rejected_updates=sum(not r['accepted'] for r in records),
                    beta=self.beta,actor_lr=self.actor_optimizer.param_groups[0]['lr'],
                    reference_unchanged=True,encoder_lstm_unchanged=True,**final,updates=records)
