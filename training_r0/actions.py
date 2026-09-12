"""Two-micro-action packets, conservative shadow legality and disabled-by-default planning."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .config import CELLS, HEIGHT, OFFSET_TICKS, WIDTH
from .observation import PublicBatch, PublicFrame


@dataclass(frozen=True)
class ActionSequence:
    count: Tensor
    candidate_uid: Tensor
    target_cell: Tensor
    offset_bin: Tensor

    def validate(self, batch_size: int) -> None:
        if self.count.shape != (batch_size,) or self.count.dtype != torch.long:
            raise ValueError('micro-action count must be int64 [B]')
        if bool(((self.count < 0) | (self.count > 2)).any()):
            raise ValueError('at most two micro actions')
        for value in (self.candidate_uid, self.target_cell, self.offset_bin):
            if value.shape != (batch_size, 2) or value.dtype != torch.long or value.device != self.count.device:
                raise ValueError('action field must be int64 [B,2] on the same device')
        active = torch.arange(2, device=self.count.device)[None] < self.count[:, None]
        if any(bool((value[~active] != -1).any()) for value in (self.candidate_uid,self.target_cell,self.offset_bin)):
            raise ValueError('inactive micro actions require -1 sentinels')
        if bool((self.candidate_uid[active] < 0).any()) or bool(((self.offset_bin[active] < 0) | (self.offset_bin[active] >= len(OFFSET_TICKS))).any()):
            raise ValueError('invalid active candidate/delay')
        if bool((active[:,1] & (self.offset_bin[:,1] < self.offset_bin[:,0])).any()):
            raise ValueError('offsets must be nondecreasing; equal offsets are representable')
        if bool(((self.target_cell[active] < -1) | (self.target_cell[active] >= CELLS)).any()):
            raise ValueError('invalid target cell')

    def to(self, device) -> 'ActionSequence':
        return ActionSequence(*(value.to(device) for value in (self.count,self.candidate_uid,self.target_cell,self.offset_bin)))


@dataclass(frozen=True)
class DecisionEnvelope:
    actions: ActionSequence
    frame_keys: tuple[tuple[str,int,int], ...]
    observation_schema_hash: str

    def validate(self) -> None:
        self.actions.validate(len(self.frame_keys))
        if any(not uid or side not in (0,1) or type(tick) is not int or tick < 0 for uid,side,tick in self.frame_keys):
            raise ValueError('decision needs exact episode/side/tick identity')


class ShadowLegality:
    def __init__(self, batch: PublicBatch):
        self.batch = batch
        self.remaining = batch.elixir.clone()
        self.used = torch.zeros_like(batch.candidate_mask)
        self.used_ability = torch.zeros(batch.batch_size, dtype=torch.bool, device=batch.elixir.device)
        self.previous_offset = torch.zeros(batch.batch_size, dtype=torch.long, device=batch.elixir.device)
        self._placement = batch.placement.clone()
        self.conservative_geometry = torch.zeros_like(self.used_ability)

    def candidate_mask(self, step: int) -> Tensor:
        mask = self.batch.candidate_mask & ~self.used & (self.batch.costs <= self.remaining[:, None] + 1e-6)
        if step:
            mask &= ~self.batch.first_only
            mask &= ~(self.used_ability[:, None] & (self.batch.candidate_kinds == 1))
        return mask & (~self.batch.grid_targets | self._placement.any(-1))

    def placement(self, candidate: Tensor) -> Tensor:
        return self._placement.gather(1, candidate[:, None, None].expand(-1, 1, CELLS))[:,0]

    def offset_mask(self, step: int) -> Tensor:
        bins = torch.arange(len(OFFSET_TICKS), device=self.previous_offset.device)
        return torch.ones(self.batch.batch_size, len(OFFSET_TICKS), dtype=torch.bool, device=bins.device) if step == 0 else bins[None] >= self.previous_offset[:,None]

    def features(self, step: int) -> Tensor:
        return torch.stack((self.remaining/10, self.used.float().mean(-1), self.used_ability.float(),
                            self.previous_offset.float()/4, self.candidate_mask(step).float().mean(-1),
                            self.conservative_geometry.float()), dim=-1)

    def apply(self, active: Tensor, candidate: Tensor, target: Tensor, offset: Tensor) -> None:
        batch = self.batch
        rows = torch.arange(batch.batch_size, device=active.device)
        cost = batch.costs[rows, candidate]
        self.remaining = self.remaining - torch.where(active, cost, torch.zeros_like(cost))
        selected = torch.nn.functional.one_hot(candidate, batch.candidate_mask.shape[1]).bool() & active[:,None]
        chosen_slot = batch.hand_slots[rows, candidate]
        same_hand = (chosen_slot[:,None] >= 0) & (batch.hand_slots == chosen_slot[:,None]) & (batch.candidate_kinds == 0)
        chosen_group = batch.exclusion_groups[rows,candidate]
        same_group = (chosen_group[:,None] >= 0) & (batch.exclusion_groups == chosen_group[:,None])
        self.used |= selected | ((same_hand | same_group) & active[:,None])
        self.used_ability |= active & (batch.candidate_kinds[rows,candidate] == 1)
        self.previous_offset = torch.where(active, offset, self.previous_offset)
        building = active & batch.buildings[rows,candidate]
        if not bool(building.any()):
            return
        chosen_known = batch.footprint_known[rows,candidate]
        uncertain_pair = building[:,None] & batch.buildings & (~chosen_known[:,None] | ~batch.footprint_known)
        self.conservative_geometry |= uncertain_pair.any(-1)
        # Unknown geometry never becomes an invented zero-size footprint.
        self._placement &= ~uncertain_pair[:,:,None]
        cells = torch.arange(CELLS, device=active.device, dtype=batch.elixir.dtype)
        xy = torch.stack((cells.remainder(WIDTH)+.5, torch.div(cells,WIDTH,rounding_mode='floor')+.5), -1)
        centers = xy[None,None] + batch.placement_offsets[:,:,None,:]
        chosen_center = xy[target.clamp_min(0)] + batch.placement_offsets[rows,candidate]
        extent = batch.footprint_half + batch.footprint_half[rows,candidate][:,None]
        overlap = ((centers - chosen_center[:,None,None]).abs() < extent[:,:,None]).all(-1)
        block = overlap & building[:,None,None] & batch.buildings[:,:,None] & chosen_known[:,None,None] & batch.footprint_known[:,:,None]
        self._placement &= ~block


@dataclass(frozen=True)
class TimingCertificate:
    runtime_sha256: str
    observation_schema_hash: str
    first_submit_delta_ticks: int
    equal_offset_order_verified: bool = False
    targeted_abilities_verified: bool = False

    def __post_init__(self):
        if any(len(value) != 64 or any(char not in '0123456789abcdef' for char in value) for value in (self.runtime_sha256,self.observation_schema_hash)):
            raise ValueError('certificate requires explicit runtime/observation hashes')
        if type(self.first_submit_delta_ticks) is not int or self.first_submit_delta_ticks < 0:
            raise ValueError('native submit origin must be calibrated explicitly')
        if type(self.equal_offset_order_verified) is not bool or type(self.targeted_abilities_verified) is not bool:
            raise ValueError('certificate capability flags must be explicit booleans')


def command_plan(frame: PublicFrame, decision: DecisionEnvelope, row: int, *,
                 certificate: TimingCertificate | None, runtime_sha256: str,
                 observation_schema_hash: str) -> list[dict]:
    """Return a plan only. No RPC or native process execution exists here."""
    decision.validate()
    if not 0 <= row < len(decision.frame_keys) or decision.frame_keys[row] != (frame.episode_uid,frame.view.actor_side,frame.view.tick):
        raise ValueError('stale or wrong-episode decision; do not reschedule old actions')
    if decision.observation_schema_hash != observation_schema_hash:
        raise ValueError('decision observation schema mismatch')
    actions = decision.actions
    count = int(actions.count[row])
    if not count:
        return []
    if certificate is None or certificate.runtime_sha256 != runtime_sha256 or certificate.observation_schema_hash != observation_schema_hash:
        raise ValueError('native timing is not certified for this runtime/observation schema')
    offsets = actions.offset_bin[row,:count].tolist()
    if count == 2 and offsets[0] == offsets[1] and not certificate.equal_offset_order_verified:
        raise ValueError('same-offset command ordering not certified; never shift it silently')
    by_uid = {candidate.uid: candidate for candidate in frame.candidates}
    result = []
    remaining = frame.view.own_player.elixir_raw/10000
    used_uids,used_slots,used_groups = set(),set(),set()
    used_ability = False
    placed = []
    for order in range(count):
        uid = int(actions.candidate_uid[row,order])
        if uid not in by_uid:
            raise ValueError('action references another frame candidate')
        candidate = by_uid[uid]
        if not frame.commands_allowed or not candidate.available or candidate.cost > remaining + 1e-6:
            raise ValueError('command plan is unavailable or exceeds elixir')
        if uid in used_uids or (candidate.hand_slot >= 0 and candidate.hand_slot in used_slots) or (candidate.exclusion_group >= 0 and candidate.exclusion_group in used_groups):
            raise ValueError('command plan repeats a candidate/hand/exclusion group')
        if order and candidate.first_only or candidate.kind == 'ability' and used_ability:
            raise ValueError('candidate cannot be used at this micro-action position')
        cell = int(actions.target_cell[row,order])
        target = None
        if candidate.target_mode == 'grid':
            if cell < 0 or not candidate.placement[cell]:
                raise ValueError('native plan has an invalid target')
            if candidate.kind == 'ability' and not certificate.targeted_abilities_verified:
                raise ValueError('targeted ability command is not certified')
            x, y = (cell % WIDTH)*1000+500, (cell//WIDTH)*1000+500
            target = (x,y) if frame.view.actor_side == 0 else (17999-x,31999-y)
            if candidate.is_building:
                center = (cell%WIDTH+.5+candidate.placement_offset_tiles[0], cell//WIDTH+.5+candidate.placement_offset_tiles[1])
                for previous,previous_center in placed:
                    if candidate.footprint_half_tiles is None or previous.footprint_half_tiles is None:
                        raise ValueError('two-building geometry is not known')
                    if all(abs(center[axis]-previous_center[axis]) < candidate.footprint_half_tiles[axis]+previous.footprint_half_tiles[axis] for axis in (0,1)):
                        raise ValueError('planned building footprints overlap')
                placed.append((candidate,center))
        elif cell != -1:
            raise ValueError('non-spatial action must not acquire a target')
        result.append(dict(episode_uid=frame.episode_uid, side=frame.view.actor_side,
            candidate_uid=uid, kind=candidate.kind, hand_slot=candidate.hand_slot,
            source_entity=candidate.source_entity, native_target=target,
            submit_tick=frame.view.tick+certificate.first_submit_delta_ticks+OFFSET_TICKS[offsets[order]], sequence_order=order))
        remaining -= candidate.cost
        used_uids.add(uid)
        if candidate.hand_slot >= 0:
            used_slots.add(candidate.hand_slot)
        if candidate.exclusion_group >= 0:
            used_groups.add(candidate.exclusion_group)
        used_ability |= candidate.kind == 'ability'
    return result
