"""Learner-side segment contracts; physical slots never alias logical episodes."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .actions import ActionSequence
from .config import Temperatures
from .model import RecurrentState
from .observation import PublicBatch


@dataclass(frozen=True)
class BehaviorIdentity:
    policy_id: str
    version: int
    weights_sha256: str
    runtime_sha256: str
    observation_schema_hash: str
    model_schema_hash: str
    temperatures: Temperatures = Temperatures()

    def __post_init__(self):
        if not self.policy_id or type(self.version) is not int or self.version < 0:
            raise ValueError('invalid behavior identity')
        for value in (self.weights_sha256,self.runtime_sha256,self.observation_schema_hash,self.model_schema_hash):
            if len(value) != 64 or any(char not in '0123456789abcdef' for char in value):
                raise ValueError('behavior identity requires full frozen hashes')


@dataclass(frozen=True)
class LaneIdentity:
    episode_uid: str
    side: int
    role: str
    designated_learner_side: int | None = None

    def __post_init__(self):
        if not self.episode_uid or self.side not in (0,1) or self.role not in ('current_current','current_frozen','specialist'):
            raise ValueError('invalid logical learner lane')
        if self.role != 'current_current' and self.designated_learner_side not in (0,1):
            raise ValueError('frozen/specialist role needs a designated learner side')

    @property
    def trainable(self) -> bool:
        return self.role == 'current_current' or self.side == self.designated_learner_side


@dataclass(frozen=True)
class RolloutSegment:
    observations: tuple[PublicBatch, ...]
    actions: tuple[ActionSequence, ...]
    lanes: tuple[LaneIdentity, ...]
    behavior: BehaviorIdentity
    initial_state: RecurrentState
    hidden_origin_version: int
    logp: Tensor
    values: Tensor
    next_values: Tensor
    rewards: Tensor
    elapsed_ticks: Tensor
    valid: Tensor
    episode_start: Tensor
    terminated: Tensor
    truncated: Tensor
    bootstrap_known: Tensor

    @property
    def time_steps(self) -> int:
        return len(self.observations)

    def validate(self, *, max_steps: int = 160) -> None:
        t,b = self.time_steps,len(self.lanes)
        if not 0 < t <= max_steps or b < 1 or len(self.actions) != t:
            raise ValueError('invalid segment dimensions')
        if any(not lane.trainable for lane in self.lanes):
            raise ValueError('frozen opponent observations cannot become learner PPO data')
        identities = [(lane.episode_uid,lane.side) for lane in self.lanes]
        if len(set(identities)) != b:
            raise ValueError('duplicate logical episode-side lane')
        for field in ('valid','episode_start','terminated','truncated','bootstrap_known'):
            value = getattr(self,field)
            if value.shape != (t,b) or value.dtype != torch.bool:
                raise ValueError(f'{field} must be bool [T,B]')
        for field in ('logp','values','next_values','rewards','elapsed_ticks'):
            value = getattr(self,field)
            if value.shape != (t,b) or not value.is_floating_point() or value.requires_grad or not bool(torch.isfinite(value).all()):
                raise ValueError(f'{field} must be finite detached float [T,B]')
        if not bool(self.valid.any()) or bool((self.valid[1:] & ~self.valid[:-1]).any()):
            raise ValueError('each lane must be a contiguous valid prefix')
        if bool(self.episode_start[1:].any()):
            raise ValueError('new episode requires a new logical lane, not a mid-segment RNN reset')
        if bool((self.terminated & self.truncated).any()):
            raise ValueError('terminal and truncation are different outcomes')
        if bool(((self.terminated | self.truncated) & ~self.valid).any()):
            raise ValueError('padding cannot terminate or truncate a game')
        if bool(((self.terminated | self.truncated)[:-1] & self.valid[1:]).any()):
            raise ValueError('samples continue past terminal/truncation in the same lane')
        if bool((self.valid & ~self.terminated & ~self.bootstrap_known).any()):
            raise ValueError('nonterminal transitions need a known value bootstrap')
        if bool((self.valid & (self.elapsed_ticks <= 0)).any()):
            raise ValueError('valid transitions need actual positive elapsed ticks')
        if bool((self.valid & (self.elapsed_ticks != self.elapsed_ticks.round())).any()):
            raise ValueError('elapsed native ticks must be integral, not silently rounded')
        if not 0 <= self.hidden_origin_version <= self.behavior.version:
            raise ValueError('hidden-state provenance is missing or from the future')
        for state in (self.initial_state.hidden,self.initial_state.cell):
            if state.ndim != 2 or state.shape[0] != b or state.requires_grad or not bool(torch.isfinite(state).all()):
                raise ValueError('missing/invalid actual pre-action recurrent state')
        if self.initial_state.hidden.shape != self.initial_state.cell.shape:
            raise ValueError('hidden/cell shape mismatch')
        starts = self.episode_start[0]
        if bool((self.initial_state.hidden[starts] != 0).any()) or bool((self.initial_state.cell[starts] != 0).any()):
            raise ValueError('new episodes must start with a reset state')
        for step,(observation,action) in enumerate(zip(self.observations,self.actions)):
            if observation.schema_hash != self.behavior.observation_schema_hash or observation.batch_size != b:
                raise ValueError('rollout observation schema/batch changed')
            if list(zip(observation.episode_uids,observation.sides)) != identities:
                raise ValueError('episode-side identity changed inside logical lane')
            action.validate(b)
            if step:
                for lane in range(b):
                    if bool(self.valid[step,lane]) and observation.ticks[lane] != self.observations[step-1].ticks[lane] + int(self.elapsed_ticks[step-1,lane]):
                        raise ValueError('observation chronology differs from actual elapsed ticks')


@torch.no_grad()
def advantages(segment: RolloutSegment, *, gamma: float, gae_lambda: float) -> tuple[Tensor,Tensor]:
    """Variable native-time GAE; chunk ends bootstrap, game terminals do not."""
    if not 0 < gamma <= 1 or not 0 < gae_lambda <= 1:
        raise ValueError('invalid GAE coefficients')
    discount = gamma ** (segment.elapsed_ticks/5)
    trace = gae_lambda ** (segment.elapsed_ticks/5)
    bootstrap = torch.where(segment.terminated,torch.zeros_like(segment.next_values),segment.next_values)
    delta = segment.rewards + discount*bootstrap - segment.values
    result = torch.zeros_like(segment.values)
    running = torch.zeros_like(segment.values[0])
    for step in reversed(range(segment.time_steps)):
        continues = ~segment.terminated[step] & ~segment.truncated[step]
        if step+1 < segment.time_steps:
            continues &= segment.valid[step+1]
        else:
            continues &= False
        running = torch.where(segment.valid[step],delta[step]+discount[step]*trace[step]*continues*running,0.)
        result[step] = running
    return result,result+segment.values
