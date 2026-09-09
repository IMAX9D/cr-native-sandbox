"""Exact fixed-period action distribution: WAIT, 4x576 card moves, hero slots."""
import torch
from torch.nn import functional as F

CARD_ACTIONS=4*576
HEAD_NAMES=('timing','kind','card_score','ability_score','position_head')


def train_action_heads_only(model):
    model.requires_grad_(False)
    for name in HEAD_NAMES: getattr(model,name).requires_grad_(True)
    return [p for p in model.parameters() if p.requires_grad]


def action_heads(model,features):
    """Frozen observation/LSTM/context features, same action heads as fixed BC."""
    c=features['context'].detach()
    hand=model.cards(features['hand_tokens']).detach()
    ability=model.abilities(features['ability_tokens']).detach()
    cards=torch.cat((c[:,None,:].expand_as(hand),hand),-1)
    abilities=torch.cat((c[:,None,:].expand_as(ability),ability),-1)
    return dict(timing=model.timing(c).squeeze(-1),kind=model.kind(c),
                card=model.card_score(cards).squeeze(-1),position=model.position_head(cards),
                ability=model.ability_score(abilities).squeeze(-1))


def masked_log_softmax(logits,mask):
    mask=mask.bool()
    # An unused empty branch gets a finite sentinel distribution. The joint
    # support below removes every illegal leaf, so it can never be sampled.
    safe=mask.clone()
    safe[...,0] |= ~mask.any(-1)
    return F.log_softmax(logits.float().masked_fill(~safe,-1e9),-1)


def action_distribution(out,masks):
    positions=masks['positions'].bool()
    cards=masks['cards'].bool() & positions.any(-1)
    abilities=masks['abilities'].bool()
    kinds=torch.stack((cards.any(-1),abilities.any(-1)),-1)
    any_action=kinds.any(-1)
    kind_log=masked_log_softmax(out['kind'],kinds)
    card_log=masked_log_softmax(out['card'],cards)
    position_log=masked_log_softmax(out['position'],positions)
    ability_log=masked_log_softmax(out['ability'],abilities)
    deploy=kind_log[:,0,None,None]+card_log[:,:,None]+position_log
    skill=kind_log[:,1,None]+ability_log
    support=torch.cat(((cards[:,:,None]&positions).flatten(1),abilities),-1)
    mark=torch.cat((deploy.flatten(1),skill),-1).masked_fill(~support,-1e9)
    mark=F.log_softmax(mark,-1)
    timing=out['timing'].float()
    log_event=torch.where(any_action,F.logsigmoid(timing),torch.full_like(timing,-1e9))
    log_wait=torch.where(any_action,F.logsigmoid(-timing),torch.zeros_like(timing))
    joint=torch.cat((log_wait[:,None],log_event[:,None]+mark),-1)
    joint_support=torch.cat((torch.ones_like(any_action[:,None]),support),-1)
    joint=F.log_softmax(joint.masked_fill(~joint_support,-1e9),-1)
    if not torch.isfinite(joint).all():raise FloatingPointError('non-finite joint action distribution')
    return dict(log_prob=joint,mark_log_prob=mark,any_action=any_action)


def exact_kl(reference,current):
    """D_KL(reference || current), exactly summed over all legal actions."""
    joint=(reference['log_prob'].exp()*(reference['log_prob']-current['log_prob'])).sum(-1)
    mark=(reference['mark_log_prob'].exp()*(reference['mark_log_prob']-current['mark_log_prob'])).sum(-1)
    mark=torch.where(reference['any_action'],mark,torch.zeros_like(mark))
    return joint.clamp_min(0),mark.clamp_min(0)


def entropy(distribution):
    p=distribution['log_prob']
    return -(p.exp()*p).sum(-1)


def sample_actions(distribution,generator):
    return torch.multinomial(distribution['log_prob'].exp(),1,generator=generator).squeeze(-1)


def native_action(index,side,masks):
    from .live import canonical_position_to_native
    index=int(index)
    if index==0:return None
    if 1<=index<=CARD_ACTIONS:
        slot,position=divmod(index-1,576)
        if not masks.cards[slot] or not masks.positions[slot,position]:raise ValueError('sampled illegal deployment')
        x,y=canonical_position_to_native(position,side)
        return dict(type='play',side=side,deck_index=int(masks.hand[slot]),x=x,y=y)
    slot=index-1-CARD_ACTIONS
    if not 0<=slot<len(masks.ability_keys) or not masks.abilities[slot]:raise ValueError('sampled illegal ability')
    return dict(type='ability',side=side,entity_id=int(masks.ability_keys[slot]))
