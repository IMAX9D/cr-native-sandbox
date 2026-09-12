"""Identical bounded strict-past history, assembled in NumPy rather than scalar Torch ops."""
import numpy as np
import torch
from hokoff_model.online_history import OnlinePlayHistory


class FastHistory(OnlinePlayHistory):
    def query(self, actor_side, tick):
        if actor_side not in (0, 1) or tick < 0:
            raise ValueError('invalid observation')
        shape = (1, 1, 2, self.length)
        values = {key: np.zeros(shape, dtype=(np.bool_ if key in ('history_mask', 'history_known')
                  else np.float32 if key == 'history_age' else np.int64))
                  for key in ('history_card', 'history_position', 'history_age', 'history_mask', 'history_known')}
        for relation, side in enumerate((actor_side, 1-actor_side)):
            selected = [event for event in self.events[side] if event.tick < tick][-self.length:]
            for index, event in enumerate(reversed(selected)):
                at = (0, 0, relation, index)
                values['history_mask'][at] = True
                values['history_known'][at] = event.known
                values['history_age'][at] = (tick-event.tick)/20.0
                if event.known:
                    values['history_card'][at] = event.token
                    values['history_position'][at] = event.absolute_cell if actor_side == 0 else 575-event.absolute_cell
        return {key: torch.from_numpy(value) for key, value in values.items()}
