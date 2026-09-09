"""Add timing-only nonlinear capacity while preserving the starting policy exactly."""
from dataclasses import dataclass,asdict
from torch import nn
from policy_v1.train import load_checkpoint
from .train_fixed import FixedConfig,FixedPolicy


@dataclass
class CapacityConfig(FixedConfig):
    architecture: str = 'hokoff_cr_lstm_fixed_timing_residual_v1'
    timing_hidden_size: int = 256

    def __post_init__(self):
        super().__post_init__()
        if self.timing_hidden_size < 1: raise ValueError('positive timing hidden size required')


class CapacityPolicy(FixedPolicy):
    def __init__(self,config):
        super().__init__(config)
        self.timing_residual=nn.Sequential(nn.Linear(config.width,config.timing_hidden_size),
            nn.GELU(),nn.Linear(config.timing_hidden_size,1))
        nn.init.zeros_(self.timing_residual[-1].weight)
        nn.init.zeros_(self.timing_residual[-1].bias)

    def heads(self,recurrent,b):
        out=super().heads(recurrent,b)
        out['timing']=out['timing']+self.timing_residual(out['context']).squeeze(-1)
        return out


SUPPORTED_ARCHITECTURES=(FixedConfig.architecture,CapacityConfig.architecture)


def policy_from_config(config):
    architecture=config.get('architecture')
    if architecture==FixedConfig.architecture:
        config=FixedConfig(**config);return config,FixedPolicy(config)
    if architecture==CapacityConfig.architecture:
        config=CapacityConfig(**config);return config,CapacityPolicy(config)
    raise ValueError('requires a fixed-period or timing-capacity checkpoint')


def initialize_from_source(config,*,checkpoint):
    saved=load_checkpoint(checkpoint)
    if saved['config'].get('architecture')!=FixedConfig.architecture:
        raise ValueError('capacity comparison starts from the original fixed-policy architecture')
    before=dict(saved['config']);after=asdict(config)
    for key in ('architecture','timing_hidden_size'):
        before.pop(key,None);after.pop(key,None)
    if before!=after:raise ValueError('source and target model dimensions/period differ')
    _,model=policy_from_config(asdict(config))
    missing,unexpected=model.load_state_dict(saved['model'],strict=False)
    allowed=[k for k in model.state_dict() if k.startswith('timing_residual.')]
    if sorted(missing)!=sorted(allowed) or unexpected:raise ValueError('unexpected source model keys')
    # Unlike the generic fixed-policy warm start, preserve the trained timing head.
    return model
