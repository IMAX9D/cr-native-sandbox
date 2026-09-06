"""Unweighted action metrics with an optional timing positive-class weight."""
import torch
from policy_v1.loss import bc_loss as original_loss, summarize as original_summary


def bc_loss(output, b, *, distributed=False, timing_positive_weight=1.0):
    if 'timing_target' in b:
        b = dict(b, play_now=b['timing_target'], timing_label_mask=b['timing_target_mask'])
    loss, stats = original_loss(output,b,distributed=distributed, timing_positive_weight=timing_positive_weight)
    with torch.no_grad():
        valid = b['frame_mask'] & b['loss_mask'] & b['timing_label_mask']
        actual = b['play_now'][valid].bool()
        predicted = output['timing'][valid] > 0
        stats.update(action_tp=int((actual & predicted).sum()),
                     action_fp=int((~actual & predicted).sum()),
                     action_fn=int((actual & ~predicted).sum()))
    return loss,stats


def summarize(stats, *, timing_horizon_ticks=0):
    result = original_summary(stats)
    tp,fp,fn = (stats.get(k,0) for k in ('action_tp','action_fp','action_fn'))
    n = stats.get('timing_count',0)
    result.update(action_precision=tp/max(tp+fp,1), action_recall=tp/max(tp+fn,1),
                  predicted_action_rate=(tp+fp)/max(n,1), actual_action_rate=(tp+fn)/max(n,1),
                  action_tp=tp,action_fp=fp,action_fn=fn)
    if timing_horizon_ticks:
        for old,new in (
            ('action_precision','forecast_precision'), ('action_recall','forecast_recall'),
            ('predicted_action_rate','predicted_forecast_positive_rate'),
            ('actual_action_rate','forecast_positive_rate'),
            ('action_tp','forecast_tp'), ('action_fp','forecast_fp'), ('action_fn','forecast_fn'),
        ):
            result[new] = result.pop(old)
        result['timing_horizon_ticks'] = timing_horizon_ticks
    return result
