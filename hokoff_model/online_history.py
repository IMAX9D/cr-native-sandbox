"""Strict-past public play history from acknowledged native commands.

Matches HistoryIndex.query: normal plays only, newest first, own/enemy order,
actor-relative cells, age in seconds. Never reconstruct from future labels.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class PublicPlay:
    tick: int
    side: int
    token: int
    absolute_cell: int
    known: bool = True


class OnlinePlayHistory:
    def __init__(self, length=4):
        if not 1 <= length <= 16: raise ValueError('invalid history length')
        self.length = length
        self.reset()

    def reset(self):
        self.events = [[], []]
        self.last_tick = [-1, -1]
        self.last_event = [None, None]
        self.accepted_plays = [0, 0]

    def record(self, event: PublicPlay):
        if event.side not in (0, 1) or event.tick < 0:
            raise ValueError('invalid event identity')
        if event.known and (event.token <= 0 or not 0 <= event.absolute_cell < 576):
            raise ValueError('known play needs a valid token and cell')
        side = event.side
        if event.tick < self.last_tick[side]: raise ValueError('history clock regressed')
        if event.tick == self.last_tick[side]:
            if event == self.last_event[side]: return False
            raise ValueError('ambiguous second command for the same side/tick')
        self.events[side].append(event)
        # Retain the preceding full window while the newest same-tick event
        # is still excluded by strict-past queries.
        self.events[side] = self.events[side][-(self.length+1):]
        self.last_tick[side], self.last_event[side] = event.tick, event
        self.accepted_plays[side] += 1
        return True

    def query(self, actor_side, tick):
        if actor_side not in (0, 1) or tick < 0: raise ValueError('invalid observation')
        result = {key: torch.zeros(1,1,2,self.length,dtype=(torch.bool if key in
                  ('history_mask','history_known') else torch.float32 if key == 'history_age' else torch.long))
                  for key in ('history_card','history_position','history_age','history_mask','history_known')}
        for relation, side in enumerate((actor_side, 1-actor_side)):
            selected = [event for event in self.events[side] if event.tick < tick][-self.length:]
            for index,event in enumerate(reversed(selected)):
                result['history_mask'][0,0,relation,index] = True
                result['history_known'][0,0,relation,index] = event.known
                result['history_age'][0,0,relation,index] = (tick-event.tick)/20.0
                if event.known:
                    result['history_card'][0,0,relation,index] = event.token
                    result['history_position'][0,0,relation,index] = (
                        event.absolute_cell if actor_side == 0 else 575-event.absolute_cell)
        return result
