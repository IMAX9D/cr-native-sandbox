"""One frozen inference snapshot; RNN state keyed by logical episode and side."""
from __future__ import annotations

from copy import deepcopy
import hashlib

import torch

from .config import DECISION_TICKS, Temperatures
from .model import PolicyOutput, R0Policy, RecurrentState
from .observation import PublicBatch


def weights_hash(model: R0Policy) -> str:
    result = hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        result.update(key.encode())
        result.update(str((value.dtype,tuple(value.shape))).encode())
        result.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


class FrozenPolicySession:
    """No RPC transport or policy hot-swap. One copy per policy, not per game.

    Snapshot ownership prevents a learner optimizer from mutating an in-flight
    behavior model. Segment-boundary state migration is a later runner concern.
    """
    def __init__(self, model: R0Policy, *, temperatures: Temperatures = Temperatures(), max_sessions: int = 512):
        if max_sessions <= 0:
            raise ValueError('session capacity must be positive')
        self._model = deepcopy(model).eval()
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)
        self.temperatures = temperatures
        self.weights_sha256 = weights_hash(self._model)
        self.max_sessions = max_sessions
        self._states: dict[tuple[str,int], RecurrentState] = {}
        self._ticks: dict[tuple[str,int], int] = {}

    @property
    def state_count(self) -> int:
        return len(self._states)

    @torch.no_grad()
    def act(self, observation: PublicBatch, *, sample: bool = True, generator=None) -> PolicyOutput:
        observation = observation.to(next(self._model.parameters()).device)
        keys = list(zip(observation.episode_uids,observation.sides))
        if len(set(keys)) != len(keys):
            raise ValueError('duplicate episode-side in inference batch')
        if len(self._states) + sum(key not in self._states for key in keys) > self.max_sessions:
            raise OverflowError('release completed episodes before admitting more sessions')
        states = []
        for key,tick in zip(keys,observation.ticks):
            if key in self._ticks and tick != self._ticks[key] + DECISION_TICKS:
                raise ValueError('duplicate/skipped inference tick; do not silently advance hidden twice')
            states.append(self._states[key] if key in self._states else self._model.initial_state(1))
        initial = RecurrentState(torch.cat([state.hidden for state in states]),torch.cat([state.cell for state in states]))
        output = self._model(observation,initial,temperatures=self.temperatures,sample=sample,generator=generator)
        for index,(key,tick) in enumerate(zip(keys,observation.ticks)):
            self._states[key] = RecurrentState(output.next_state.hidden[index:index+1].detach().clone(),
                                                output.next_state.cell[index:index+1].detach().clone())
            self._ticks[key] = tick
        return output

    def release_episode(self, episode_uid: str) -> None:
        for key in tuple(self._states):
            if key[0] == episode_uid:
                del self._states[key]
                del self._ticks[key]
