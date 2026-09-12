"""Explicit local reward control, not a reconstruction of missing upstream code."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TowerTotals:
    own: float
    enemy: float

    def __post_init__(self):
        if any(not math.isfinite(value) or value < 0 for value in (self.own,self.enemy)):
            raise ValueError('tower totals must be measured finite nonnegative values')


@dataclass(frozen=True)
class RewardControl:
    initial: TowerTotals
    gamma_per_decision: float = 0.9997
    shaping_beta: float = 0.05

    def __post_init__(self):
        if min(self.initial.own,self.initial.enemy) <= 0:
            raise ValueError('initial tower totals must be fixed and positive')
        if not 0 < self.gamma_per_decision <= 1 or not math.isfinite(self.shaping_beta) or self.shaping_beta < 0:
            raise ValueError('invalid shaping configuration')

    def potential(self, current: TowerTotals) -> float:
        return current.own/self.initial.own-current.enemy/self.initial.enemy

    def transition(self, before: TowerTotals, after: TowerTotals, *, elapsed_ticks: int,
                   terminated: bool, outcome: int | None = None) -> dict[str,float]:
        if type(elapsed_ticks) is not int or elapsed_ticks <= 0:
            raise ValueError('reward needs actual positive native elapsed ticks')
        if type(terminated) is not bool or terminated and (type(outcome) is not int or outcome not in (-1,0,1)) or not terminated and outcome is not None:
            raise ValueError('only a normal terminal may supply W/D/L outcome')
        terminal = float(outcome) if terminated else 0.0
        next_potential = 0.0 if terminated else self.potential(after)
        shaping = self.shaping_beta*(self.gamma_per_decision**(elapsed_ticks/5)*next_potential-self.potential(before))
        return {'terminal':terminal,'tower_shaping':shaping,'overflow':0.0,'total':terminal+shaping}
