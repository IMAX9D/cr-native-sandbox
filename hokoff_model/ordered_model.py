"""Training-only, two-operation forecast decoder with explicit STOP and tick offsets.

Tokens name card/ability *types*, not native instances. This is not an executable
macro policy: runtime must resolve instances and recheck cost/cooldown/legality.
"""
import torch
from torch import nn
from torch.nn import functional as F

STOP, DEPLOY, ABILITY = 0, 1, 2


class OrderedDecoder(nn.Module):
    def __init__(self, width, card_vocab, ability_vocab, period=4, *, conditioned=True):
        super().__init__()
        if period < 2:
            raise ValueError('two distinct source ticks require period >= 2')
        self.period, self.conditioned = period, conditioned
        self.card_embedding = nn.Embedding(card_vocab, width, padding_idx=0)
        self.ability_embedding = nn.Embedding(ability_vocab, width, padding_idx=0)
        self.position_embedding = nn.Embedding(577, width, padding_idx=576)
        self.type_embedding = nn.Embedding(3, width)
        self.tick_embedding = nn.Embedding(period, width)
        self.second_embedding = nn.Parameter(torch.zeros(width))
        self.transition = nn.GRUCell(width, width)
        self.kind = nn.Linear(width, 3)
        self.card = nn.Linear(width, card_vocab)
        self.ability = nn.Linear(width, ability_vocab)
        self.position = nn.Sequential(nn.Linear(2*width, width), nn.GELU(), nn.Linear(width, 576))
        self.tick = nn.Linear(width, period)

    def heads(self, h, target=None, min_tick=None):
        kind, card, ability = self.kind(h), self.card(h), self.ability(h)
        card = card.clone(); ability = ability.clone()
        card[:, 0] = -1e9; ability[:, 0] = -1e9
        predicted_type = kind.argmax(-1)
        predicted_token = torch.where(predicted_type == ABILITY, ability.argmax(-1), card.argmax(-1))
        # Teacher forcing within an action: position depends on the chosen card.
        position_token = card.argmax(-1) if target is None else torch.where(
            target['type'] == DEPLOY, target['token'], 0)
        position = self.position(torch.cat((h, self.card_embedding(position_token)), -1))
        tick = self.tick(h)
        if min_tick is not None:
            illegal = torch.arange(self.period, device=h.device)[None, :] < min_tick[:, None]
            tick = tick.masked_fill(illegal, -1e9)
            # When the first operation uses the last tick, the second must STOP.
            kind = kind.clone()
            kind[:, 1:] = kind[:, 1:].masked_fill((min_tick >= self.period)[:, None], -1e9)
            predicted_type = kind.argmax(-1)
            predicted_token = torch.where(predicted_type == ABILITY, ability.argmax(-1), card.argmax(-1))
        prediction = dict(type=predicted_type, token=predicted_token,
                          position=position.argmax(-1), tick=tick.argmax(-1))
        return dict(type=kind, card=card, ability=ability, position=position, tick=tick), prediction

    def forward(self, context, targets=None, *, first_action=None):
        """With targets: teacher forcing. Without targets: autonomous two-step decode.

        Gold second-action fields never condition the recurrent transition. Gold first
        action conditions step two; gold card conditions its own position distribution.
        """
        if targets is not None and first_action is not None:
            raise ValueError('teacher forcing and diagnostic prefix are mutually exclusive')
        first_target = None if targets is None else {k: targets[k][:, 0] for k in ('type','token','position','tick')}
        first, predicted_first = self.heads(context, first_target)
        previous = predicted_first if first_target is None else first_target
        if first_action is not None:
            previous = first_action
            predicted_first = first_action  # explicit oracle-prefix diagnostic only
        active = previous['type'] != STOP
        card_token = torch.where(previous['type'] == DEPLOY, previous['token'], 0)
        ability_token = torch.where(previous['type'] == ABILITY, previous['token'], 0)
        position = torch.where(previous['type'] == DEPLOY, previous['position'], 576)
        action = (self.type_embedding(previous['type']) + self.card_embedding(card_token)
                  + self.ability_embedding(ability_token) + self.position_embedding(position)
                  + self.tick_embedding(previous['tick'])) * active[:, None]
        x = self.second_embedding.expand_as(context)
        if self.conditioned:
            x = x + action
        second_context = self.transition(x, context)
        second_target = None if targets is None else {k: targets[k][:, 1] for k in ('type','token','position','tick')}
        # Ordering is a structural constraint, also present in the ablation.
        second, predicted_second = self.heads(second_context, second_target,
                                              torch.where(active, previous['tick']+1, 0))
        if targets is None:
            predicted_second['type'] = torch.where(active, predicted_second['type'], STOP)
        return [first, second], {k: torch.stack((predicted_first[k], predicted_second[k]), 1)
                                for k in predicted_first}


def ordered_loss(outputs, targets):
    losses = {}
    for name in ('type', 'card', 'ability', 'position', 'tick'):
        values = []
        for step, output in enumerate(outputs):
            valid = targets['mask'][:, step].clone()
            kind = targets['type'][:, step]
            if name in ('card', 'position'): valid &= kind == DEPLOY
            elif name == 'ability': valid &= kind == ABILITY
            elif name == 'tick': valid &= kind != STOP
            label = targets['token' if name in ('card','ability') else name][:, step]
            values.append(F.cross_entropy(output[name][valid].float(), label[valid], reduction='none'))
        joined = torch.cat(values)
        losses[name] = joined.mean() if joined.numel() else joined.sum()
    total = sum(losses.values())
    if not torch.isfinite(total):
        raise FloatingPointError('non-finite ordered loss')
    return total, {k: float(v.detach()) for k, v in losses.items()}


def sequence_metrics(prediction, target):
    """Free-running metrics; skill positions and tokens after STOP are ignored."""
    valid = target['mask'][:, 0]
    actual_active = (target['type'] != STOP) & target['mask']
    predicted_active = prediction['type'] != STOP
    actual_count = actual_active.sum(-1)
    predicted_count = predicted_active.sum(-1)
    type_ok = ((prediction['type'] == target['type']) | ~target['mask']).all(-1)
    token_ok = ((prediction['token'] == target['token']) | ~actual_active).all(-1)
    position_ok = ((prediction['position'] == target['position']) | (target['type'] != DEPLOY)).all(-1)
    tick_ok = ((prediction['tick'] == target['tick']) | ~actual_active).all(-1)
    exact = type_ok & token_ok & position_ok & (actual_count == predicted_count)
    def mean(x, mask=valid):
        return float(x[mask].float().mean()) if mask.any() else None
    out = dict(count=int(valid.sum()), count_accuracy=mean(actual_count == predicted_count),
               sequence_accuracy=mean(exact), sequence_with_tick_accuracy=mean(exact & tick_ok))
    for n in range(3):
        m = valid & (actual_count == n)
        out.update({f'count_{n}': int(m.sum()), f'predicted_count_{n}': int((valid & (predicted_count == n)).sum()), f'count_{n}_recall': mean(predicted_count == n, m),
                    f'sequence_{n}_accuracy': mean(exact, m)})
    two = valid & (predicted_count == 2)
    out['two_action_precision'] = mean(actual_count == 2, two)
    active = actual_active & valid[:, None]
    deploy = (target['type'] == DEPLOY) & active
    ability = (target['type'] == ABILITY) & active
    for name, mask, eq in [('token',active,prediction['token']==target['token']),
                          ('position',deploy,prediction['position']==target['position']),
                          ('ability',ability,prediction['token']==target['token']),
                          ('tick',active,prediction['tick']==target['tick'])]:
        out[name+'_count'] = int(mask.sum())
        out[name+'_accuracy'] = float(eq[mask].float().mean()) if mask.any() else None
    return out


def continuation_metrics(prediction, target):
    """Score only step two under an explicitly supplied correct first action."""
    valid = (target['type'][:,0] != STOP) & target['mask'][:,1]
    actual = target['type'][:,1] != STOP
    predicted = prediction['type'][:,1] != STOP
    kind_ok = prediction['type'][:,1] == target['type'][:,1]
    token_ok = prediction['token'][:,1] == target['token'][:,1]
    position_ok = ((prediction['position'][:,1] == target['position'][:,1]) |
                   (target['type'][:,1] != DEPLOY))
    exact = kind_ok & (token_ok | ~actual) & (position_ok | ~actual)
    def mean(value, mask):
        return float(value[mask].float().mean()) if mask.any() else None
    single, double = valid & ~actual, valid & actual
    return dict(count=int(valid.sum()), stop_count=int(single.sum()), two_action_count=int(double.sum()),
                stop_after_one_accuracy=mean(~predicted,single), second_action_recall=mean(predicted,double),
                second_action_exact_accuracy=mean(exact,double),
                second_action_with_tick_accuracy=mean(exact & (prediction['tick'][:,1]==target['tick'][:,1]),double),
                continuation_accuracy=mean(exact,valid))
