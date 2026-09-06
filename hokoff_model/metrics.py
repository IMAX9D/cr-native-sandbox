"""Additional unweighted action counts; unchanged BC optimization objective."""
import torch
from policy_v1.loss import bc_loss as original_loss, summarize as original_summary


def bc_loss(output, b, *, distributed=False):
    loss, stats = original_loss(output,b,distributed=distributed)
    with torch.no_grad():
        valid = b['frame_mask'] & b['loss_mask'] & b['timing_label_mask']
        actual = b['play_now'][valid].bool()
        predicted = output['timing'][valid] > 0
        stats.update(action_tp=int((actual & predicted).sum()),
                     action_fp=int((~actual & predicted).sum()),
                     action_fn=int((actual & ~predicted).sum()))
    return loss,stats


def summarize(stats):
    result = original_summary(stats)
    tp,fp,fn = (stats.get(k,0) for k in ('action_tp','action_fp','action_fn'))
    n = stats.get('timing_count',0)
    result.update(action_precision=tp/max(tp+fp,1), action_recall=tp/max(tp+fn,1),
                  predicted_action_rate=(tp+fp)/max(n,1), actual_action_rate=(tp+fn)/max(n,1),
                  action_tp=tp,action_fp=fp,action_fn=fn)
    return result
