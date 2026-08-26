"""Memory-mapped recurrent windows over compiled native replay shards."""

from __future__ import annotations

from bisect import bisect_right
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import (
    OBSERVATION_SEQUENCE,
    load_shard_arrays,
    read_manifest,
    unpack_position_masks,
    validate_shard,
)


TOKEN_FIELDS = (
    "own_deck_tokens",
    "hand_tokens",
    "next_card_token",
    "revealed_enemy_tokens",
    "ability_tokens",
)
MASK_FIELDS = (
    "card_mask",
    "action_kind_mask",
    "ability_mask",
    "timing_label_mask",
    "kind_label_mask",
    "card_label_mask",
    "position_label_mask",
    "ability_label_mask",
    "ability_position_label_mask",
    "play_now",
)
LABEL_FIELDS = (
    "action_kind",
    "card_slot",
    "position",
    "ability_slot",
    "ability_position",
)
ENTITY_WINDOW_FIELDS = (
    "entity_tokens",
    "entity_positions",
    "entity_relations",
    "entity_numeric",
    "entity_mask",
)


class NativeExpertSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    """Return target windows with a recurrent burn-in prefix.

    Windows may be shuffled safely: every non-initial window carries the
    preceding ``burn_in`` observations while its loss mask excludes them.
    """

    def __init__(
        self,
        root: Path,
        *,
        split: str,
        sequence_length: int = 128,
        burn_in: int = 32,
        validate: bool = True,
    ) -> None:
        if sequence_length <= 0 or burn_in < 0:
            raise ValueError("sequence_length must be positive and burn_in non-negative")
        self.root = root.resolve()
        self.manifest = read_manifest(self.root)
        if split not in self.manifest["splits"]:
            raise KeyError(f"unknown dataset split: {split}")
        self.split = split
        self.sequence_length = int(sequence_length)
        self.burn_in = int(burn_in)
        self.shards = [(self.root / relative).resolve() for relative in self.manifest["splits"][split]]
        self._arrays: dict[int, dict[str, np.ndarray]] = {}
        self._offsets: list[np.ndarray] = []
        self._sequence_window_prefix: list[np.ndarray] = []
        self._shard_window_prefix = [0]
        for shard in self.shards:
            if validate:
                validate_shard(shard, self.manifest)
            offsets = np.load(shard / "sequence_offsets.npy", mmap_mode="r")
            lengths = np.diff(offsets)
            windows = np.asarray(
                [math.ceil(int(length) / self.sequence_length) for length in lengths],
                dtype=np.int64,
            )
            prefix = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(windows)))
            self._offsets.append(offsets)
            self._sequence_window_prefix.append(prefix)
            self._shard_window_prefix.append(self._shard_window_prefix[-1] + int(prefix[-1]))

    def __len__(self) -> int:
        return self._shard_window_prefix[-1]

    def _open(self, shard_index: int) -> dict[str, np.ndarray]:
        arrays = self._arrays.get(shard_index)
        if arrays is None:
            arrays = load_shard_arrays(self.shards[shard_index], self.manifest)
            self._arrays[shard_index] = arrays
        return arrays

    def _locate(self, index: int) -> tuple[int, int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect_right(self._shard_window_prefix, index) - 1
        local = index - self._shard_window_prefix[shard_index]
        sequence_prefix = self._sequence_window_prefix[shard_index]
        sequence_index = int(np.searchsorted(sequence_prefix, local, side="right") - 1)
        window_index = int(local - sequence_prefix[sequence_index])
        return shard_index, sequence_index, window_index

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        shard_index, sequence_index, window_index = self._locate(index)
        arrays = self._open(shard_index)
        offsets = self._offsets[shard_index]
        sequence_start = int(offsets[sequence_index])
        sequence_stop = int(offsets[sequence_index + 1])
        target_start = sequence_start + window_index * self.sequence_length
        target_stop = min(sequence_stop, target_start + self.sequence_length)
        read_start = max(sequence_start, target_start - self.burn_in)
        sl = slice(read_start, target_stop)
        count = target_stop - read_start
        burn = target_start - read_start

        item: dict[str, torch.Tensor] = {
            "public_scalars": torch.from_numpy(np.asarray(arrays["public_scalars"][sl]).copy()).float(),
            "delta_ticks": torch.from_numpy(np.asarray(arrays["delta_ticks"][sl]).copy()).float(),
            "timing_exposure_ticks": torch.from_numpy(
                np.asarray(arrays["timing_exposure_ticks"][sl]).copy()
            ).float(),
            "sample_weight": torch.from_numpy(np.asarray(arrays["sample_weight"][sl]).copy()).float(),
        }
        sequence_only = (
            str(self.manifest.get("observation_mode")) == OBSERVATION_SEQUENCE
        )
        if sequence_only:
            for name in (
                "own_deck_tokens",
                "hand_tokens",
                "next_card_token",
                "revealed_enemy_tokens",
                "previous_event_card_token",
                "previous_event_side",
                "previous_event_position",
                "card_slot",
                "position",
            ):
                item[name] = torch.from_numpy(np.asarray(arrays[name][sl]).copy()).long()
            for name in (
                "card_mask",
                "timing_label_mask",
                "card_label_mask",
                "position_label_mask",
                "play_now",
            ):
                item[name] = torch.from_numpy(np.asarray(arrays[name][sl]).copy()).bool()
        else:
            item["grid"] = torch.from_numpy(
                np.asarray(arrays["grid"][sl]).copy()
            ).float().div_(255.0)
            item["position_mask"] = torch.from_numpy(
                unpack_position_masks(np.asarray(arrays["selected_position_mask_packed"][sl]))
            )
            item["ability_position_mask"] = torch.from_numpy(
                unpack_position_masks(np.asarray(arrays["ability_position_mask_packed"][sl]))
            )
            for name in TOKEN_FIELDS + LABEL_FIELDS:
                item[name] = torch.from_numpy(np.asarray(arrays[name][sl]).copy()).long()
            for name in MASK_FIELDS:
                item[name] = torch.from_numpy(np.asarray(arrays[name][sl]).copy()).bool()
            entity_offsets = arrays["entity_offsets"]
            starts = np.asarray(
                entity_offsets[read_start : target_stop + 1], dtype=np.int64
            )
            counts = np.diff(starts)
            maximum_entities = max(1, int(counts.max(initial=0)))
            entity_tokens = torch.zeros((count, maximum_entities), dtype=torch.long)
            entity_positions = torch.zeros((count, maximum_entities), dtype=torch.long)
            entity_relations = torch.zeros((count, maximum_entities), dtype=torch.long)
            entity_numeric = torch.zeros(
                (
                    count,
                    maximum_entities,
                    int(self.manifest["dimensions"]["entity_numeric_size"]),
                ),
                dtype=torch.float32,
            )
            entity_mask = torch.zeros((count, maximum_entities), dtype=torch.bool)
            for local_row, (start, stop) in enumerate(zip(starts, starts[1:])):
                length = int(stop - start)
                if not length:
                    continue
                entity_tokens[local_row, :length] = torch.from_numpy(
                    np.asarray(arrays["entity_tokens"][start:stop]).copy()
                ).long()
                entity_positions[local_row, :length] = torch.from_numpy(
                    np.asarray(arrays["entity_positions"][start:stop]).copy()
                ).long()
                entity_relations[local_row, :length] = torch.from_numpy(
                    np.asarray(arrays["entity_relations"][start:stop]).copy()
                ).long()
                entity_numeric[local_row, :length] = torch.from_numpy(
                    np.asarray(arrays["entity_numeric"][start:stop]).copy()
                ).float()
                entity_mask[local_row, :length] = True
            item.update(
                entity_tokens=entity_tokens,
                entity_positions=entity_positions,
                entity_relations=entity_relations,
                entity_numeric=entity_numeric,
                entity_mask=entity_mask,
            )
        loss_mask = torch.ones(count, dtype=torch.bool)
        loss_mask[:burn] = False
        item["loss_mask"] = loss_mask
        return item


def collate_sequences(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not items:
        raise ValueError("cannot collate an empty batch")
    maximum = max(int(item["loss_mask"].shape[0]) for item in items)
    maximum_entities = max(
        (int(item["entity_tokens"].shape[1]) for item in items if "entity_tokens" in item),
        default=0,
    )
    output: dict[str, list[torch.Tensor]] = {key: [] for key in items[0]}
    for item in items:
        steps = int(item["loss_mask"].shape[0])
        for key, value in item.items():
            if key in ENTITY_WINDOW_FIELDS:
                shape = (maximum, maximum_entities) + tuple(value.shape[2:])
            else:
                shape = (maximum,) + tuple(value.shape[1:])
            if key in LABEL_FIELDS:
                padded = torch.full(shape, -100, dtype=value.dtype)
            else:
                padded = torch.zeros(shape, dtype=value.dtype)
            if key in ENTITY_WINDOW_FIELDS:
                padded[:steps, : value.shape[1]] = value
            else:
                padded[:steps] = value
            output[key].append(padded)
    return {key: torch.stack(values) for key, values in output.items()}
