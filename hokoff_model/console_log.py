"""Aligned terminal logs; metrics.jsonl retains its original precision and schema."""
import math


PRIORITY = (
    'phase', 'step', 'epoch',
    'action_precision', 'action_recall',
    'card_accuracy', 'position_accuracy',
    'delay_short_precision', 'delay_short_recall', 'delay_short_accuracy',
    'delay_short_mae_ticks', 'delay_short_late_rate',
    'delay_predicted_short_rate', 'delay_actual_short_rate', 'delay_predicted_mean_ticks',
    'delay_accuracy', 'delay_always_max_accuracy',
    'loss', 'timing_loss', 'card_loss', 'position_loss', 'delay_loss',
)
FOOTER = ('elapsed_seconds', 'peak_cuda_mb')


def format_console(payload):
    def flatten(mapping, prefix=''):
        for key, value in mapping.items():
            name = prefix+key
            if isinstance(value, dict):
                yield from flatten(value, name+'.')
            else:
                yield name, value

    def value_text(key, value):
        if isinstance(value, float) and math.isfinite(value):
            if key.endswith(('_count', '_tp', '_fp', '_fn')) and value.is_integer():
                return str(int(value))
            if value != 0 and abs(value) < .0001:
                return format(value, '.3g')
            return format(value, '.4f')
        return str(value)

    phase = payload.get('phase', 'setup')
    title = str(phase)
    if 'step' in payload:
        title += ' | step '+format(int(payload['step']), ',')
    if 'epoch' in payload:
        title += ' | epoch '+str(payload['epoch'])
    fields = dict(flatten(payload))
    order = [k for k in PRIORITY if k in fields and k not in ('phase', 'step', 'epoch')]
    order.extend(k for k in fields if k not in PRIORITY and k not in FOOTER)
    tail = [k for k in FOOTER if k in fields]
    # Original property names, aligned at the colon; no translated metric labels.
    pad = max(32, max((len(k) for k in order+tail), default=0))
    width = max(80, pad+24, len(title)+4)
    lines = ['#'*width, title.center(width), '']
    lines.extend(f'{k:>{pad}}: {value_text(k, fields[k])}' for k in order)
    lines.append('-'*width)
    lines.extend(f'{k:>{pad}}: {value_text(k, fields[k])}' for k in tail)
    return '\n'.join(lines)+'\n'
