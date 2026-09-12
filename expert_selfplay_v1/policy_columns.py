"""Versioned columnar policy requests for local authenticated IPC.

Numerical fields are transmitted once per Actor batch, not once per row.
The server keeps these columns batched all the way to inference. This is not
shared memory and does not make pickle safe for untrusted network peers.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Sequence

import numpy as np
import torch

from .actions import ExpertActionMasks
from .batched_policy import (
    PolicyIdentity, PolicyRequest, PolicyTensorBatch,
    _ENTITY_KEYS, _INPUT_RANKS, _MASK_RANKS, _mask_tensor, _online_tensor, _pad_entities,
)

COLUMNS_KIND = "cr_native_policy_columns_v2"
MAX_ROWS = 256
MAX_ENTITY_SLOTS = 2048
MAX_TENSOR_BYTES = 64 * 1024 * 1024
_DTYPES = {np.dtype(value) for value in (
    'bool', 'int8', 'int16', 'int32', 'int64', 'uint8', 'float16', 'float32', 'float64',
)}


def _metadata(row: PolicyRequest) -> dict[str, Any]:
    row.validate_identity()
    return {'worker_id':row.worker_id, 'side':int(row.side), 'actor_sha256':row.actor_sha256,
            'delta_ticks':int(row.delta_ticks), 'reset_hidden':bool(row.reset_hidden),
            'capture_pre_action_hidden':bool(row.capture_pre_action_hidden)}


def encode_columns(requests: Sequence[PolicyRequest]) -> dict[str, Any]:
    rows = list(requests)
    if len(rows) > MAX_ROWS:
        raise ValueError("columnar request exceeds row capacity")
    groups: OrderedDict[str, list[tuple[int, PolicyRequest]]] = OrderedDict()
    metadata = [_metadata(row) for row in rows]
    seen = set()
    for index, row in enumerate(rows):
        identity = (row.actor_sha256, row.worker_id, row.side)
        if identity in seen:
            raise ValueError("duplicate sequential Actor identity in columnar request")
        seen.add(identity)
        groups.setdefault(row.actor_sha256, []).append((index, row))
    packed, total_bytes = [], 0
    for digest, indexed in groups.items():
        selected = [row for _, row in indexed]
        keys = set(selected[0].actor_inputs)
        if keys - set(_INPUT_RANKS) or any(set(row.actor_inputs) != keys for row in selected):
            raise ValueError("Actor input fields differ or are unknown")
        inputs, masks = {}, {}

        def store(name, values, *, entity=False):
            nonlocal total_bytes
            if len({value.dtype for value in values}) != 1:
                raise ValueError(f"column dtype differs between rows: {name}")
            if entity:
                if len({(value.shape[0], *value.shape[2:]) for value in values}) != 1:
                    raise ValueError(f"entity column feature shape differs: {name}")
                count = max(value.shape[1] for value in values)
                if count > MAX_ENTITY_SLOTS:
                    raise ValueError("entity column exceeds capacity")
                shape = list(values[0].shape)
                shape[1] = count
            else:
                if len({tuple(value.shape) for value in values}) != 1:
                    raise ValueError(f"column shape differs between rows: {name}")
                shape = values[0].shape
            size = len(values) * int(np.prod(shape)) * values[0].element_size()
            if total_bytes + size > MAX_TENSOR_BYTES:
                raise ValueError("columnar request exceeds tensor byte capacity")
            result = _pad_entities(values, name=name) if entity else torch.stack(values)
            array = result.contiguous().numpy()
            total_bytes += array.nbytes
            if array.dtype not in _DTYPES:
                raise TypeError(f"unsupported column dtype: {array.dtype}")
            return array

        for name in sorted(keys):
            raw = [row.actor_inputs[name] for row in selected]
            if all(value is None for value in raw):
                inputs[name] = None
                continue
            if any(value is None for value in raw):
                raise ValueError("input is None for only part of a batch")
            tensors = [_online_tensor(value, name=name, rank=_INPUT_RANKS[name]).detach().cpu() for value in raw]
            inputs[name] = store(name, tensors, entity=name in _ENTITY_KEYS)
        for name, rank in _MASK_RANKS.items():
            tensors = [_mask_tensor(getattr(row.masks, name), name=name, rank=rank).detach().cpu() for row in selected]
            masks[name] = store(name, tensors)
        packed.append({'actor_sha256': digest, 'indices': [index for index, _ in indexed],
                       'actor_inputs': inputs, 'masks': masks})
    return {'kind': COLUMNS_KIND, 'row_count': len(rows), 'rows': metadata,
            'groups': packed, 'tensor_bytes': total_bytes}


def decode_columns(packet: Any, *, offset: int = 0) -> tuple[list[PolicyTensorBatch], list[PolicyIdentity]]:
    if not isinstance(packet, dict) or packet.get('kind') != COLUMNS_KIND:
        raise ValueError("unsupported columnar request schema")
    count = packet.get('row_count')
    metadata, groups = packet.get('rows'), packet.get('groups')
    if (type(count) is not int or not 0 <= count <= MAX_ROWS or not isinstance(metadata, list)
            or len(metadata) != count or not isinstance(groups, list) or len(groups) > count):
        raise ValueError("invalid columnar request row count")
    identities = []
    for value in metadata:
        if not isinstance(value, dict) or set(value) != set(PolicyIdentity.__dataclass_fields__):
            raise ValueError("invalid policy row metadata")
        row = PolicyIdentity(**value)
        row.validate_identity()
        if type(row.reset_hidden) is not bool or type(row.capture_pre_action_hidden) is not bool:
            raise ValueError("invalid policy row flags")
        identities.append(row)
    seen_indices, seen_hashes, batches = set(), set(), []
    total_bytes = 0
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("invalid Actor column group")
        indices, digest = group.get('indices'), group.get('actor_sha256')
        if (not isinstance(indices, list) or not indices or len(indices) > count
                or digest in seen_hashes):
            raise ValueError("invalid or duplicate Actor column group")
        seen_hashes.add(digest)
        for index in indices:
            if type(index) is not int or not 0 <= index < count or index in seen_indices:
                raise ValueError("duplicate or out-of-range column index")
            seen_indices.add(index)
            if identities[index].actor_sha256 != digest:
                raise ValueError("Actor hash does not match row identity")
        inputs, masks = group.get('actor_inputs'), group.get('masks')
        if (not isinstance(inputs, dict) or set(inputs) - set(_INPUT_RANKS)
                or not isinstance(masks, dict) or set(masks) != set(_MASK_RANKS)):
            raise ValueError("invalid Actor input or mask columns")

        def tensor(value, *, name, rank, mask=False):
            nonlocal total_bytes
            if (not isinstance(value, np.ndarray) or value.dtype not in _DTYPES
                    or not value.dtype.isnative or not value.flags.c_contiguous or not value.flags.writeable
                    or value.ndim != rank + 1 or value.shape[0] != len(indices)):
                raise ValueError(f"invalid numerical column: {name}")
            if not mask and value.shape[1] != 1:
                raise ValueError("column must contain exactly one online time step")
            if name in _ENTITY_KEYS and value.shape[2] > MAX_ENTITY_SLOTS:
                raise ValueError("entity column exceeds capacity")
            if mask and value.dtype != np.bool_:
                raise ValueError("mask column must be boolean")
            total_bytes += value.nbytes
            if total_bytes > MAX_TENSOR_BYTES:
                raise ValueError("columnar request exceeds tensor byte capacity")
            return torch.from_numpy(value)

        batched_inputs = {name: None if value is None else tensor(value, name=name, rank=_INPUT_RANKS[name])
                          for name, value in inputs.items()}
        batched_masks = ExpertActionMasks(**{name: tensor(value, name=name, rank=_MASK_RANKS[name], mask=True)
                                           for name, value in masks.items()})
        batches.append(PolicyTensorBatch(tuple(offset + index for index in indices),
                       tuple(identities[index] for index in indices), batched_inputs, batched_masks))
    if seen_indices != set(range(count)) or packet.get('tensor_bytes') != total_bytes:
        raise ValueError("columnar packet dropped rows or has inconsistent byte count")
    if len({(r.actor_sha256, r.worker_id, r.side) for r in identities}) != count:
        raise ValueError("duplicate sequential Actor identity")
    return batches, identities
