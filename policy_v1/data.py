"""Read existing native-bc-v1 arrays and build public events without libg.

The cache is separate from the immutable source dataset. Only executed,
label-valid actions are converted to events; private opponent fields never
leave preparation. Coordinate rotation is 575-cell for the 18x32 grid.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


SCALARS = [
    "tick_fraction",
    "own_elixir_ratio",
    "commands_allowed",
    "terminated",
    "own_crowns_ratio",
    "enemy_crowns_ratio",
    "own_king_hp_ratio",
    "own_left_hp_ratio",
    "own_right_hp_ratio",
    "enemy_king_hp_ratio",
    "enemy_left_hp_ratio",
    "enemy_right_hp_ratio",
    "own_visible_entity_count_log",
    "enemy_visible_entity_count_log",
    "own_visible_entity_hp_log",
    "enemy_visible_entity_hp_log",
]
ROW_TOKENS = [
    "own_deck_tokens",
    "hand_tokens",
    "next_card_token",
    "revealed_enemy_tokens",
    "ability_tokens",
]
LABELS = ["action_kind", "card_slot", "position", "ability_slot", "ability_position"]
MASKS = [
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
]
FIELDS = (
    ROW_TOKENS
    + LABELS
    + MASKS
    + [
        "public_scalars",
        "delta_ticks",
        "timing_exposure_ticks",
        "sample_weight",
        "sequence_offsets",
        "entity_offsets",
        "entity_tokens",
        "entity_positions",
        "entity_relations",
        "entity_numeric",
        "grid_offsets",
        "grid_indices",
        "grid_values",
        "selected_position_mask_rows",
        "selected_position_mask_packed",
        "ability_position_mask_rows",
        "ability_position_mask_packed",
        "replay_extent",
    ]
)
ENTITY_FIELDS = [
    "entity_tokens",
    "entity_positions",
    "entity_relations",
    "entity_numeric",
    "entity_mask",
]
EVENT_FIELDS = [
    "event_ticks",
    "event_side",
    "event_kind",
    "event_card",
    "event_ability",
    "event_position",
    "event_mask",
]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def within(root, relative):
    result = (root / relative).resolve()
    if root.resolve() not in result.parents:
        raise ValueError("dataset path escapes root: " + str(relative))
    return result


def close_arrays(arrays):
    for a in arrays.values():
        if getattr(a, "_mmap", None) is not None:
            a._mmap.close()


def open_arrays(shard):
    # Explicit whitelist: no newly added private arrays become input features.
    return {
        k: np.load(shard / (k + ".npy"), mmap_mode="r", allow_pickle=False)
        for k in FIELDS
    }


def prepare(root, cache, *, splits=None, verify_hashes=False, allow_smoke=False):
    root, cache = Path(root).resolve(), Path(cache).resolve()
    if root == cache or root in cache.parents:
        raise ValueError("cache must be outside the source dataset")
    if (cache / "index.json").exists():
        raise FileExistsError("cache already prepared; use a new directory")
    m = json.loads((root / "manifest.json").read_text())
    if (
        m.get("kind") != "cr_native_expert_bc_dataset_v3"
        or m.get("observation_mode") != "native_state_v1"
    ):
        raise ValueError(
            "requires compiled native_state_v1 dataset, not sequence-only data"
        )
    if m.get("actor_information") != "public_only_v1":
        raise ValueError("requires public-only actor data")
    if not m.get("production_ready") and not (allow_smoke and m.get("smoke_only")):
        raise ValueError("non-production data: only synthetic smoke may opt in")
    fs = m["feature_schema"]
    if fs.get("public_scalars") != SCALARS or fs.get("entity_numeric") != [
        "level_ratio",
        "hp_ratio",
        "log_max_hp",
    ]:
        raise ValueError("unsupported scalar/entity schema")
    if int(m["dimensions"]["grid_channels"]) != 8:
        raise ValueError("requires eight-channel native grid")
    splits = list(splits or m["splits"])
    cache.mkdir(parents=True, exist_ok=True)
    index = {
        "version": 1,
        "manifest_sha256": digest(root / "manifest.json"),
        "dimensions": m["dimensions"],
        "smoke_only": bool(m.get("smoke_only")),
        "verified_source_hashes": verify_hashes,
        "splits": {},
        "event_contract": "valid_executed_labels_strictly_before_query_tick_v1",
    }
    split_battles = {}
    for split in splits:
        records, battles = [], set()
        for si, relative in enumerate(m["splits"][split]):
            shard = within(root, relative)
            metadata = json.loads((shard / "shard.json").read_text())
            arrays = open_arrays(shard)
            try:
                if verify_hashes:
                    for name in FIELDS:
                        expected = metadata["file_sha256"][name + ".npy"]
                        if digest(shard / (name + ".npy")) != expected:
                            raise ValueError("checksum mismatch: " + str(shard / name))
                offsets = np.asarray(arrays["sequence_offsets"]).copy()
                identities = metadata["sequence_identity"]
                if (
                    len(offsets) != len(identities) + 1
                    or offsets[0] != 0
                    or np.any(np.diff(offsets) <= 0)
                ):
                    raise ValueError("invalid sequence offsets")
                if len(identities) % 2:
                    raise ValueError("missing paired actor sequence")
                events, event_offsets = [], [0]
                for a in range(0, len(identities), 2):
                    left, right = identities[a : a + 2]
                    if (
                        left["actor_side"] != 0
                        or right["actor_side"] != 1
                        or left["battle_tag"] != right["battle_tag"]
                    ):
                        raise ValueError("invalid actor pairing")
                    if left["battle_tag"] in battles:
                        raise ValueError("duplicate battle within split")
                    battles.add(left["battle_tag"])
                    pair_events, pair_ticks = [], []
                    for side in (0, 1):
                        lo, hi = map(int, offsets[a + side : a + side + 2])
                        recorded_ticks = np.rint(
                            arrays["public_scalars"][lo:hi, 0] * 6000
                        ).astype(np.int64)
                        # tick_fraction saturates at 1 after tick 6000. Recover
                        # the clock from the initial tick and contiguous rows,
                        # rather than making all post-horizon events simultaneous.
                        ticks = recorded_ticks[0] + np.arange(hi - lo)
                        if (
                            not 0 <= recorded_ticks[0] < 6000
                            or not np.array_equal(
                                np.minimum(ticks, 6000), recorded_ticks
                            )
                            or not np.all(arrays["delta_ticks"][lo:hi] == 1)
                        ):
                            raise ValueError(
                                "requires contiguous 20Hz frames with unambiguous absolute ticks"
                            )
                        pair_ticks.append(ticks)
                        if not np.all(arrays["timing_exposure_ticks"][lo:hi] == 1):
                            raise ValueError("V1 timing uses one-tick Bernoulli labels")
                        for local in np.flatnonzero(
                            arrays["play_now"][lo:hi]
                            & arrays["timing_label_mask"][lo:hi]
                        ):
                            row = lo + int(local)
                            if not arrays["kind_label_mask"][row]:
                                raise ValueError("executed action lacks kind label")
                            kind = int(arrays["action_kind"][row])
                            if kind == 0:
                                if (
                                    not arrays["card_label_mask"][row]
                                    or not arrays["position_label_mask"][row]
                                ):
                                    raise ValueError(
                                        "deployment lacks complete public event fields"
                                    )
                                slot = int(arrays["card_slot"][row])
                                if not 0 <= slot < 4:
                                    raise ValueError("invalid card slot")
                                token = int(arrays["hand_tokens"][row, slot])
                                pos = int(arrays["position"][row])
                            elif kind == 1:
                                if not arrays["ability_label_mask"][row]:
                                    raise ValueError("skill lacks slot label")
                                slot = int(arrays["ability_slot"][row])
                                if not 0 <= slot < arrays["ability_tokens"].shape[1]:
                                    raise ValueError("invalid ability slot")
                                token = int(arrays["ability_tokens"][row, slot])
                                pos = (
                                    int(arrays["ability_position"][row])
                                    if arrays["ability_position_label_mask"][row]
                                    else 576
                                )
                            else:
                                raise ValueError("invalid action kind")
                            if token <= 0 or not 0 <= pos <= 576:
                                raise ValueError("invalid public event token/position")
                            # Source actor-local coordinate -> side 0 coordinate.
                            if side == 1 and pos < 576:
                                pos = 575 - pos
                            pair_events.append(
                                [int(ticks[local]), side, kind, token, pos]
                            )
                    if not np.array_equal(*pair_ticks):
                        raise ValueError("paired actors have different timestamps")
                    pair_events.sort()
                    for actor_side in (0, 1):
                        for tick, side, kind, token, pos in pair_events:
                            local_pos = (
                                575 - pos if actor_side == 1 and pos < 576 else pos
                            )
                            events.append(
                                [tick, side ^ actor_side, kind, token, local_pos]
                            )
                        event_offsets.append(len(events))
                name = "%s-%05d-events.npz" % (split, si)
                np.savez(
                    cache / name,
                    events=np.asarray(events, dtype=np.int64).reshape(-1, 5),
                    offsets=np.asarray(event_offsets, dtype=np.int64),
                )
                records.append(
                    {
                        "path": relative,
                        "offsets": offsets.tolist(),
                        "events": name,
                        "events_sha256": digest(cache / name),
                        "metadata_sha256": digest(shard / "shard.json"),
                        "battle_tags": [x["battle_tag"] for x in identities[::2]],
                    }
                )
            finally:
                close_arrays(arrays)
            if (si + 1) % 100 == 0:
                print("prepared", split, si + 1, "shards", flush=True)
        for previous, tags in split_battles.items():
            if tags & battles:
                raise ValueError(
                    "battle leakage across splits: %s / %s" % (previous, split)
                )
        split_battles[split] = battles
        index["splits"][split] = records
    temporary = cache / "index.json.partial"
    temporary.write_text(json.dumps(index, ensure_ascii=False))
    temporary.replace(cache / "index.json")
    return index


def dense_masks(a, prefix, start, stop):
    out = np.zeros((stop - start, 576), dtype=bool)
    rows = a[prefix + "_rows"]
    lo, hi = np.searchsorted(rows, [start, stop])
    if hi > lo:
        out[np.asarray(rows[lo:hi]) - start] = np.unpackbits(
            a[prefix + "_packed"][lo:hi], axis=-1, bitorder="little"
        )[:, :576]
    return out


class Windows(Dataset):
    def __init__(
        self,
        root,
        cache,
        split,
        *,
        targets=32,
        frame_window=128,
        event_window=128,
        max_open=2,
    ):
        self.root, self.cache = Path(root).resolve(), Path(cache).resolve()
        self.index = json.loads((self.cache / "index.json").read_text())
        if digest(self.root / "manifest.json") != self.index["manifest_sha256"]:
            raise ValueError("source manifest changed since preparation")
        if self.index.get("version") != 1:
            raise ValueError("unsupported prepared cache")
        if min(targets, frame_window, event_window, max_open) < 1:
            raise ValueError("window sizes must be positive")
        self.records = self.index["splits"][split]
        self.targets, self.frame_window, self.event_window, self.max_open = (
            targets,
            frame_window,
            event_window,
            max_open,
        )
        self.prefix = [0]
        self.sequence_prefix = []
        self.opened = OrderedDict()
        for record in self.records:
            if (
                digest(within(self.root, record["path"]) / "shard.json")
                != record["metadata_sha256"]
            ):
                raise ValueError("source shard metadata changed")
            if digest(within(self.cache, record["events"])) != record["events_sha256"]:
                raise ValueError("prepared event cache changed")
            lengths = np.diff(record["offsets"])
            p = np.r_[0, np.cumsum((lengths + targets - 1) // targets)]
            self.sequence_prefix.append(p)
            self.prefix.append(self.prefix[-1] + int(p[-1]))

    def __len__(self):
        return self.prefix[-1]

    def _open(self, i):
        if i not in self.opened:
            r = self.records[i]
            a = open_arrays(within(self.root, r["path"]))
            with np.load(within(self.cache, r["events"]), allow_pickle=False) as f:
                ev, eo = f["events"].copy(), f["offsets"].copy()
            self.opened[i] = (a, ev, eo)
            while len(self.opened) > self.max_open:
                _, (old, _, _) = self.opened.popitem(last=False)
                close_arrays(old)
        self.opened.move_to_end(i)
        return self.opened[i]

    def __getstate__(self):
        state = dict(self.__dict__)
        state["opened"] = OrderedDict()
        return state

    def __getitem__(self, i):
        if not 0 <= i < len(self):
            raise IndexError(i)
        sh = bisect_right(self.prefix, i) - 1
        local = i - self.prefix[sh]
        p = self.sequence_prefix[sh]
        seq = int(np.searchsorted(p, local, side="right") - 1)
        offset = self.records[sh]["offsets"]
        begin, end = offset[seq : seq + 2]
        target = begin + (local - int(p[seq])) * self.targets
        start = max(begin, target - self.frame_window + 1)
        stop = min(end, target + self.targets)
        a, events, eo = self._open(sh)
        sl = slice(start, stop)
        T = stop - start
        b = {}
        for k in ROW_TOKENS + LABELS:
            b[k] = torch.tensor(np.asarray(a[k][sl]).copy(), dtype=torch.long)
        for k in MASKS:
            b[k] = torch.tensor(np.asarray(a[k][sl]).copy(), dtype=torch.bool)
        for k in ["public_scalars", "sample_weight"]:
            b[k] = torch.tensor(np.asarray(a[k][sl]).copy(), dtype=torch.float32)
        initial_tick = int(np.rint(a["public_scalars"][begin, 0] * 6000))
        ticks = initial_tick + np.arange(start - begin, stop - begin)
        b["frame_ticks"] = torch.tensor(ticks)
        b["frame_mask"] = torch.ones(T, dtype=torch.bool)
        b["loss_mask"] = torch.arange(T) >= target - start
        b["position_mask"] = torch.tensor(
            dense_masks(a, "selected_position_mask", start, stop)
        )
        b["ability_position_mask"] = torch.tensor(
            dense_masks(a, "ability_position_mask", start, stop)
        )
        grid = np.zeros((T, 8 * 576), dtype=np.float32)
        for t in range(T):
            lo, hi = a["grid_offsets"][start + t : start + t + 2]
            grid[t, a["grid_indices"][lo:hi]] = a["grid_values"][lo:hi] / 255.0
        b["grid"] = torch.tensor(grid.reshape(T, 8, 32, 18))
        offsets = a["entity_offsets"][start : stop + 1]
        N = max(1, int(np.diff(offsets).max()))
        for k in ENTITY_FIELDS:
            shape = (
                (T, N, a["entity_numeric"].shape[-1])
                if k == "entity_numeric"
                else (T, N)
            )
            b[k] = torch.zeros(
                shape,
                dtype=(
                    torch.float32
                    if k == "entity_numeric"
                    else torch.bool if k == "entity_mask" else torch.long
                ),
            )
        for t, (lo, hi) in enumerate(zip(offsets, offsets[1:])):
            n = int(hi - lo)
            for k in ENTITY_FIELDS[:-1]:
                b[k][t, :n] = torch.tensor(np.asarray(a[k][lo:hi]).copy())
            b["entity_mask"][t, :n] = True
        ev = events[eo[seq] : eo[seq + 1]]
        lo = max(
            0, int(np.searchsorted(ev[:, 0], ticks[0], side="left")) - self.event_window
        )
        hi = int(np.searchsorted(ev[:, 0], ticks[-1], side="left"))
        ev = ev[lo:hi]
        b["event_ticks"] = torch.tensor(ev[:, 0])
        b["event_side"] = torch.tensor(ev[:, 1])
        b["event_kind"] = torch.tensor(ev[:, 2])
        b["event_card"] = torch.tensor(np.where(ev[:, 2] == 0, ev[:, 3], 0))
        b["event_ability"] = torch.tensor(np.where(ev[:, 2] == 1, ev[:, 3], 0))
        b["event_position"] = torch.tensor(ev[:, 4])
        b["event_mask"] = torch.ones(len(ev), dtype=torch.bool)
        return b


def collate(items):
    T = max(len(b["frame_ticks"]) for b in items)
    N = max(b["entity_tokens"].shape[1] for b in items)
    E = max(1, max(len(b["event_ticks"]) for b in items))
    output = {}
    for k in items[0]:
        values = []
        for b in items:
            v = b[k]
            length = E if k in EVENT_FIELDS else T
            shape = (
                (length, N) + tuple(v.shape[2:])
                if k in ENTITY_FIELDS
                else (length,) + tuple(v.shape[1:])
            )
            pad = torch.full(shape, -100 if k in LABELS else 0, dtype=v.dtype)
            if k in ENTITY_FIELDS:
                pad[: len(v), : v.shape[1]] = v
            else:
                pad[: len(v)] = v
            values.append(pad)
        output[k] = torch.stack(values)
    return output


def main():
    p = argparse.ArgumentParser(
        description="Prepare public action history from paired compiled native BC arrays (no runtime)"
    )
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--splits", nargs="+")
    p.add_argument("--verify-hashes", action="store_true")
    args = p.parse_args()
    index = prepare(
        args.data, args.cache, splits=args.splits, verify_hashes=args.verify_hashes
    )
    print(json.dumps({s: len(v) for s, v in index["splits"].items()}))


if __name__ == "__main__":
    main()
