"""Pure ranking and task-gradient statistics for offline fixed-policy diagnostics."""
import numpy as np


def timing_diagnostics(probability, actual):
    probability = np.asarray(probability, dtype=np.float64)
    actual = np.asarray(actual)
    if (probability.ndim != 1 or actual.shape != probability.shape or not len(actual)
            or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1))
            or not np.isin(actual, [0, 1]).all()):
        raise ValueError('expected nonempty finite probabilities and binary labels')
    actual = actual.astype(bool); n = len(actual); positives = int(actual.sum())
    order = np.argsort(-probability, kind='stable')
    scores, labels = probability[order], actual[order]
    # Equal scores form one threshold group; never reward their input order.
    ends = np.r_[np.flatnonzero(np.diff(scores) != 0), n-1]
    tp = np.cumsum(labels)[ends]; predicted = ends+1
    precision = tp/predicted; recall = tp/max(positives, 1)
    ap = float(np.sum(precision*np.diff(np.r_[0, tp]))/positives) if positives else None
    base = positives/n

    def point(count, hits):
        return dict(predicted_count=int(count), true_positive=int(hits),
                    precision=float(hits/max(count, 1)), recall=float(hits/max(positives, 1)),
                    predicted_action_rate=float(count/n))

    thresholds = {}
    for threshold in (.01, .02, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9):
        selected = probability > threshold
        thresholds[str(threshold)] = point(int(selected.sum()), int((selected & actual).sum()))
    budgets = {}
    for name, rate in [('actual_rate', base), ('1_percent', .01), ('2_percent', .02),
                       ('5_percent', .05), ('10_percent', .1), ('20_percent', .2)]:
        budget = int(np.floor(n*rate+1e-9))
        j = int(np.searchsorted(predicted, budget, side='right'))-1
        count, hits = (int(predicted[j]), int(tp[j])) if j >= 0 else (0, 0)
        budgets[name] = dict(requested_action_rate=rate, budget_count=budget,
            threshold_inclusive=float(scores[ends[j]]) if j >= 0 else None,
            **point(count, hits))
    result = dict(average_precision=ap, positive_rate=base,
        ap_over_positive_rate=ap/base if positives else None,
        points=n, positive_count=positives, thresholds=thresholds, action_budgets=budgets,
        budget_tie_policy='include_complete_score_groups_only_without_exceeding_budget',
        mean_probability_on_actions=float(probability[actual].mean()) if positives else None,
        mean_probability_on_waits=float(probability[~actual].mean()) if positives < n else None,
        max_probability=float(probability.max()))
    curve = dict(threshold=scores[ends], precision=precision, recall=recall,
                 predicted_action_rate=predicted/n, true_positive=tp, predicted_count=predicted)
    return result, curve


def gradient_geometry(gram, names):
    """Gram matrix of raw per-task gradients, before clipping/Adam preconditioning."""
    gram = np.asarray(gram, dtype=np.float64)
    if gram.shape != (len(names), len(names)) or not np.isfinite(gram).all():
        raise ValueError('invalid task gradient Gram matrix')
    norms = np.sqrt(np.maximum(np.diag(gram), 0)); norm_sum = float(norms.sum())
    tasks = {name: dict(norm=float(norms[i]), norm_fraction=float(norms[i]/norm_sum) if norm_sum else None)
             for i, name in enumerate(names)}
    pairs = {}
    for i, left in enumerate(names):
        for j in range(i+1, len(names)):
            denom = norms[i]*norms[j]
            cosine = float(np.clip(gram[i,j]/denom, -1, 1)) if denom > 0 else None
            pairs[left+'__'+names[j]] = cosine
    return dict(tasks=tasks, pair_cosines=pairs,
        cancellation_ratio=float(np.sqrt(max(float(gram.sum()), 0))/norm_sum) if norm_sum else None)


def summarize_gradient_batches(batches, groups, names):
    def summary(values):
        values = np.asarray([v for v in values if v is not None], dtype=np.float64)
        return dict(count=len(values), mean=float(values.mean()) if len(values) else None,
                    median=float(np.median(values)) if len(values) else None,
                    p10=float(np.quantile(values,.1)) if len(values) else None,
                    p90=float(np.quantile(values,.9)) if len(values) else None)
    result = {}
    for group in groups:
        records = [b['groups'][group] for b in batches]
        tasks = {name: dict(norm=summary([r['tasks'][name]['norm'] for r in records]),
            norm_fraction=summary([r['tasks'][name]['norm_fraction'] for r in records])) for name in names}
        pairs = {}
        for key in records[0]['pair_cosines']:
            values = [r['pair_cosines'][key] for r in records]
            known = [v for v in values if v is not None]
            pairs[key] = dict(**summary(values),
                negative_cosine_rate=sum(v < 0 for v in known)/len(known) if known else None)
        result[group] = dict(tasks=tasks, pairs=pairs,
            cancellation_ratio=summary([r['cancellation_ratio'] for r in records]))
    return result
