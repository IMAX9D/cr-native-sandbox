"""Current-action BC plus censored, action-conditional re-observation delay CE."""
import math
import torch
import torch.distributed as dist
from torch.nn import functional as F

from .metrics import bc_loss as action_loss, summarize as action_summary


def bc_loss(output, b, *, distributed=False, timing_positive_weight=1.0, delay_weight=1.0,
            delay_short_weight=1.0):
    if not math.isfinite(delay_short_weight) or delay_short_weight <= 0:
        raise ValueError('delay_short_weight must be finite and positive')
    if 'timing_target' in b:
        raise ValueError('future-horizon timing labels cannot be mixed with current-action decisions')
    loss, stats = action_loss(output, b, distributed=distributed, timing_positive_weight=timing_positive_weight)
    base = b['frame_mask'] & b['loss_mask'] & b['timing_label_mask']
    # Unknown conditional identity still permits timing supervision, never invent a delay branch.
    known_mode = ~b['play_now'] | (b['kind_label_mask'] & ((b['action_kind'] == 0) | (b['action_kind'] == 1)))
    mask = base & b['delay_label_mask'] & known_mode
    mode = torch.where(b['play_now'], b['action_kind']+1, 0).clamp(0, 2)
    K = output['delay'].shape[-1]
    logits = output['delay'].gather(2, mode[..., None, None].expand(*mode.shape, 1, K)).squeeze(2)[mask].float()
    labels = b['delay_target_ticks'][mask]-1
    if ((labels < 0) | (labels >= K)).any():
        raise ValueError('known delay target outside 1..K')
    values = F.cross_entropy(logits, labels, reduction='none') if len(labels) else logits.sum(-1)
    short = labels < K-1
    sample_weights = b['sample_weight'][mask].float()
    weights = sample_weights * torch.where(short, delay_short_weight, 1.0)
    numerator, denominator = (values*weights).sum(), weights.sum()
    normalizer = denominator.detach().clone()
    world = dist.get_world_size() if distributed else 1
    if distributed:
        dist.all_reduce(normalizer)
    loss = loss+delay_weight*numerator*world/normalizer.clamp_min(1)
    with torch.no_grad():
        pred = logits.argmax(-1)
        predict_short = pred < K-1
        stats.update(delay_sum=float(numerator.detach()), delay_weight=float(denominator), delay_count=len(labels),
                     delay_scaled_sum=float(numerator.detach())*delay_weight,
                     delay_correct=int((pred == labels).sum()),
                     delay_abs_error_ticks=int((pred-labels).abs().sum()),
                     delay_late_count=int((pred > labels).sum()),
                     delay_max_target_count=int((labels == K-1).sum()),
                     delay_short_count=int(short.sum()), delay_short_exact=int(((pred == labels) & short).sum()),
                     delay_short_tp=int((short & predict_short).sum()),
                     delay_short_fp=int((~short & predict_short).sum()),
                     delay_short_fn=int((short & ~predict_short).sum()),
                     delay_short_abs_error_ticks=int((pred[short]-labels[short]).abs().sum()),
                     delay_short_late_count=int((pred[short] > labels[short]).sum()),
                     delay_censored_count=int((base & ~b['delay_label_mask']).sum()),
                     delay_unknown_mode_count=int((base & b['delay_label_mask'] & ~known_mode).sum()))
        short_probability = logits.softmax(-1)[:, :-1].sum(-1)
        stats.update(delay_unweighted_sum=float((values*sample_weights).sum()),
                     delay_unweighted_weight=float(sample_weights.sum()),
                     delay_short_loss_sum=float(values[short].sum()),
                     delay_max_loss_sum=float(values[~short].sum()),
                     delay_short_probability_sum=float(short_probability[short].sum()),
                     delay_max_short_probability_sum=float(short_probability[~short].sum()),
                     delay_predicted_ticks_sum=int((pred+1).sum()))
        for i, (pred_count, target_count) in enumerate(zip(
                torch.bincount(pred, minlength=K).tolist(),
                torch.bincount(labels, minlength=K).tolist()), start=1):
            stats[f'delay_predicted_{i}_count'] = pred_count
            stats[f'delay_target_{i}_count'] = target_count
    if not torch.isfinite(loss):
        raise FloatingPointError('non-finite decision loss')
    return loss, stats


def summarize(stats):
    result = action_summary(stats)
    n, s = stats.get('delay_count', 0), stats.get('delay_short_count', 0)
    w = max(stats.get('delay_weight', 0), 1)
    tp, fp, fn = (stats.get(k, 0) for k in ('delay_short_tp', 'delay_short_fp', 'delay_short_fn'))
    result.update(delay_loss=stats.get('delay_sum', 0)/w, delay_count=n,
                  delay_accuracy=stats.get('delay_correct', 0)/max(n, 1),
                  delay_mae_ticks=stats.get('delay_abs_error_ticks', 0)/max(n, 1),
                  delay_late_rate=stats.get('delay_late_count', 0)/max(n, 1),
                  delay_always_max_accuracy=stats.get('delay_max_target_count', 0)/max(n, 1),
                  delay_short_count=s, delay_short_accuracy=stats.get('delay_short_exact', 0)/max(s, 1),
                  delay_short_precision=tp/max(tp+fp, 1), delay_short_recall=tp/max(tp+fn, 1),
                  delay_short_mae_ticks=stats.get('delay_short_abs_error_ticks', 0)/max(s, 1),
                  delay_short_late_rate=stats.get('delay_short_late_count', 0)/max(s, 1),
                  delay_censored_count=stats.get('delay_censored_count', 0),
                  delay_unknown_mode_count=stats.get('delay_unknown_mode_count', 0))
    result['loss'] += stats.get('delay_scaled_sum', 0)/w
    result.update(
        delay_unweighted_loss=stats.get('delay_unweighted_sum', 0)/max(stats.get('delay_unweighted_weight', 0), 1),
        delay_short_loss=stats.get('delay_short_loss_sum', 0)/max(s, 1),
        delay_max_loss=stats.get('delay_max_loss_sum', 0)/max(n-s, 1),
        delay_predicted_short_rate=(tp+fp)/max(n, 1),
        delay_actual_short_rate=s/max(n, 1),
        delay_short_probability_on_short=stats.get('delay_short_probability_sum', 0)/max(s, 1),
        delay_short_probability_on_max=stats.get('delay_max_short_probability_sum', 0)/max(n-s, 1),
        delay_predicted_mean_ticks=stats.get('delay_predicted_ticks_sum', 0)/max(n, 1),
    )
    for key, count in stats.items():
        if key.startswith(('delay_predicted_', 'delay_target_')) and key.endswith('_count'):
            result[key[:-6]+'_rate'] = count/max(n, 1)
    return result
