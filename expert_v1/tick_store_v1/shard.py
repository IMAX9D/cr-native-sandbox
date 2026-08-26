"""Crash-recoverable append-only shards and actor-safe training reads."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import mmap
import os
from pathlib import Path
import struct
from typing import Any, Iterable, Iterator, Mapping
import zlib

from .codec import EpisodeReader, encode_episode
from .schema import ActorTick, TickState, actor_projection


FRAME_MAGIC = b"EPS1"
FRAME_HEADER = struct.Struct("<4sQIQII")
SHARD_KIND = "cr_native_tick_shard_v1"
STORE_KIND = "cr_native_tick_store_v1"


class ShardCorruptionError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tag_hash(tag: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(tag.encode("utf-8"), digest_size=8).digest(), "little"
    )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _scan_frames(path: Path, *, truncate_invalid_tail: bool) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    mode = "r+b" if truncate_invalid_tail else "rb"
    with path.open(mode) as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        offset = 0
        while offset < size:
            handle.seek(offset)
            raw_header = handle.read(FRAME_HEADER.size)
            if len(raw_header) != FRAME_HEADER.size:
                break
            magic, payload_size, payload_crc, tag_hash, ticks, _reserved = FRAME_HEADER.unpack(
                raw_header
            )
            if magic != FRAME_MAGIC or payload_size <= 0 or ticks <= 0:
                break
            payload = handle.read(payload_size)
            if len(payload) != payload_size or zlib.crc32(payload) != payload_crc:
                break
            try:
                reader = EpisodeReader(payload)
                tag = str(reader.metadata["battle_tag"])
            except Exception as error:
                if truncate_invalid_tail:
                    break
                raise ShardCorruptionError(
                    f"invalid episode at byte {offset}: {error}"
                ) from error
            if _tag_hash(tag) != tag_hash or reader.tick_count != ticks:
                if truncate_invalid_tail:
                    break
                raise ShardCorruptionError(f"frame metadata mismatch at byte {offset}")
            frame_size = FRAME_HEADER.size + payload_size
            entries.append(
                {
                    "battle_tag": tag,
                    "offset": offset,
                    "frame_size": frame_size,
                    "payload_size": payload_size,
                    "ticks": ticks,
                    "tick_start": int(reader.metadata["tick_start"]),
                    "tick_stop": int(reader.metadata["tick_stop"]),
                    "payload_sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
            offset += frame_size
        if offset != size:
            if not truncate_invalid_tail:
                raise ShardCorruptionError(
                    f"invalid/truncated tail starts at {offset} of {size}"
                )
            handle.truncate(offset)
            handle.flush()
            os.fsync(handle.fileno())
    tags = [entry["battle_tag"] for entry in entries]
    if len(tags) != len(set(tags)):
        raise ShardCorruptionError("duplicate battle tag inside one shard")
    return entries


class AppendOnlyShardWriter:
    """One writer owns one shard; different workers never share a data file."""

    def __init__(
        self,
        root: Path,
        name: str,
        *,
        anchor_interval: int = 256,
        compression_level: int = 1,
        fsync_each_episode: bool = True,
    ) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        if not name or any(character in name for character in "\\/:"):
            raise ValueError("shard name must be a simple filename stem")
        self.name = name
        self.anchor_interval = anchor_interval
        self.compression_level = compression_level
        self.fsync_each_episode = fsync_each_episode
        self.partial_path = self.root / f"{name}.crts.partial"
        self.final_path = self.root / f"{name}.crts"
        self.index_path = self.root / f"{name}.index.jsonl"
        self.manifest_path = self.root / f"{name}.manifest.json"
        if self.final_path.exists():
            raise FileExistsError(f"shard is already finalized: {self.final_path}")
        self.partial_path.touch(exist_ok=True)
        self.entries = _scan_frames(self.partial_path, truncate_invalid_tail=True)
        self._tags = {str(entry["battle_tag"]) for entry in self.entries}
        self._handle = self.partial_path.open("ab", buffering=0)

    @property
    def episode_count(self) -> int:
        return len(self.entries)

    @property
    def tick_count(self) -> int:
        return sum(int(entry["ticks"]) for entry in self.entries)

    def append(
        self,
        battle_tag: str,
        states: Iterable[TickState],
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        if battle_tag in self._tags:
            return next(
                entry for entry in self.entries if entry["battle_tag"] == battle_tag
            )
        blob, stats = encode_episode(
            states,
            {**dict(metadata), "battle_tag": battle_tag},
            anchor_interval=self.anchor_interval,
            compression_level=self.compression_level,
        )
        offset = self._handle.tell()
        header = FRAME_HEADER.pack(
            FRAME_MAGIC,
            len(blob),
            zlib.crc32(blob),
            _tag_hash(battle_tag),
            stats["ticks"],
            0,
        )
        self._handle.write(header)
        self._handle.write(blob)
        self._handle.flush()
        if self.fsync_each_episode:
            os.fsync(self._handle.fileno())
        entry = {
            "battle_tag": battle_tag,
            "offset": offset,
            "frame_size": len(header) + len(blob),
            "payload_size": len(blob),
            "ticks": stats["ticks"],
            "tick_start": int(metadata.get("tick_start", -1)),
            "tick_stop": int(metadata.get("tick_stop", -1)),
            "payload_sha256": hashlib.sha256(blob).hexdigest(),
            "raw_delta_bytes": stats["raw_delta_bytes"],
            "chunks": stats["chunks"],
        }
        # The source of truth is the checksummed data frame.  On restart the
        # index is rebuilt from it, so a crash between data fsync and DB commit
        # is safe and idempotent.
        reader = EpisodeReader(blob)
        entry["tick_start"] = int(reader.metadata["tick_start"])
        entry["tick_stop"] = int(reader.metadata["tick_stop"])
        self.entries.append(entry)
        self._tags.add(battle_tag)
        return dict(entry)

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()

    def finalize(self) -> dict[str, Any]:
        self.close()
        entries = _scan_frames(self.partial_path, truncate_invalid_tail=False)
        index_temporary = self.index_path.with_name(
            self.index_path.name + f".{os.getpid()}.tmp"
        )
        with index_temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for entry in entries:
                handle.write(
                    json.dumps(entry, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        index_temporary.replace(self.index_path)
        os.replace(self.partial_path, self.final_path)
        manifest = {
            "schema_version": 1,
            "kind": SHARD_KIND,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "name": self.name,
            "data_file": self.final_path.name,
            "index_file": self.index_path.name,
            "episode_count": len(entries),
            "tick_count": sum(int(entry["ticks"]) for entry in entries),
            "anchor_interval": self.anchor_interval,
            "compression": f"zlib-level-{self.compression_level}",
            "data_sha256": sha256_file(self.final_path),
            "index_sha256": sha256_file(self.index_path),
            "bytes": self.final_path.stat().st_size,
        }
        _atomic_json(self.manifest_path, manifest)
        return manifest

    def __enter__(self) -> "AppendOnlyShardWriter":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class WorkerShardSink:
    """Rotating private shard sink suitable for asynchronous workers."""

    def __init__(
        self,
        root: Path,
        worker_id: str,
        *,
        episodes_per_shard: int = 256,
        anchor_interval: int = 256,
        compression_level: int = 1,
    ) -> None:
        if episodes_per_shard <= 0:
            raise ValueError("episodes_per_shard must be positive")
        self.root = root.resolve()
        self.worker_id = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in worker_id
        )
        self.episodes_per_shard = episodes_per_shard
        self.anchor_interval = anchor_interval
        self.compression_level = compression_level
        self.finalized: list[dict[str, Any]] = []
        existing = sorted(self.root.glob(f"{self.worker_id}-*.manifest.json"))
        self.shard_index = len(existing)
        partial = sorted(self.root.glob(f"{self.worker_id}-*.crts.partial"))
        if len(partial) > 1:
            raise ShardCorruptionError("worker has multiple partial shards")
        if partial:
            stem = partial[0].name.removesuffix(".crts.partial")
            self.shard_index = int(stem.rsplit("-", 1)[1])
        self.writer = self._open()

    def _open(self) -> AppendOnlyShardWriter:
        return AppendOnlyShardWriter(
            self.root,
            f"{self.worker_id}-{self.shard_index:05d}",
            anchor_interval=self.anchor_interval,
            compression_level=self.compression_level,
        )

    def append(
        self,
        battle_tag: str,
        states: Iterable[TickState],
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        entry = self.writer.append(battle_tag, states, metadata)
        entry["shard"] = self.writer.name
        if self.writer.episode_count >= self.episodes_per_shard:
            self.finalized.append(self.writer.finalize())
            self.shard_index += 1
            self.writer = self._open()
        return entry

    def finalize(self) -> list[dict[str, Any]]:
        if self.writer.episode_count:
            self.finalized.append(self.writer.finalize())
        else:
            self.writer.close()
            self.writer.partial_path.unlink(missing_ok=True)
        return list(self.finalized)


class ShardReader:
    """Memory-mapped episode and actor-window reads for DataLoader workers."""

    def __init__(self, data_path: Path, index_path: Path | None = None) -> None:
        self.data_path = data_path.resolve(strict=True)
        self.index_path = (
            index_path.resolve(strict=True)
            if index_path is not None
            else self.data_path.with_name(
                self.data_path.name.removesuffix(".crts") + ".index.jsonl"
            ).resolve(strict=True)
        )
        self.entries: dict[str, dict[str, Any]] = {}
        with self.index_path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    tag = str(value["battle_tag"])
                    if tag in self.entries:
                        raise ShardCorruptionError(f"duplicate index tag: {tag}")
                    self.entries[tag] = value
        self._handle = self.data_path.open("rb")
        self._map = mmap.mmap(self._handle.fileno(), 0, access=mmap.ACCESS_READ)

    def episode(self, battle_tag: str) -> EpisodeReader:
        entry = self.entries[battle_tag]
        offset = int(entry["offset"])
        header = self._map[offset : offset + FRAME_HEADER.size]
        magic, payload_size, payload_crc, tag_hash, ticks, _reserved = FRAME_HEADER.unpack(
            header
        )
        if magic != FRAME_MAGIC or tag_hash != _tag_hash(battle_tag):
            raise ShardCorruptionError("indexed frame header mismatch")
        start = offset + FRAME_HEADER.size
        payload = self._map[start : start + payload_size]
        if zlib.crc32(payload) != payload_crc:
            raise ShardCorruptionError("indexed frame checksum mismatch")
        reader = EpisodeReader(payload)
        if reader.tick_count != ticks:
            raise ShardCorruptionError("indexed Tick count mismatch")
        return reader

    def actor_ticks(self, battle_tag: str, *, actor_side: int) -> Iterator[ActorTick]:
        for state in self.episode(battle_tag).iter_ticks():
            yield actor_projection(state, actor_side=actor_side)

    def actor_windows(
        self,
        battle_tag: str,
        *,
        actor_side: int,
        length: int = 128,
        burn_in: int = 32,
    ) -> Iterator[tuple[tuple[ActorTick, ...], int]]:
        """Yield overlapping recurrent windows and the burn-in prefix length."""
        if length <= 0 or burn_in < 0:
            raise ValueError("length must be positive and burn_in non-negative")
        states = list(self.episode(battle_tag).iter_ticks())
        for target_start in range(0, len(states), length):
            read_start = max(0, target_start - burn_in)
            stop = min(len(states), target_start + length)
            yield (
                tuple(
                    actor_projection(state, actor_side=actor_side)
                    for state in states[read_start:stop]
                ),
                target_start - read_start,
            )

    def close(self) -> None:
        self._map.close()
        self._handle.close()

    def __enter__(self) -> "ShardReader":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def build_store_manifest(
    root: Path,
    *,
    source_manifest: Path,
    expected_episodes: int,
    expected_ticks: int | None = None,
) -> dict[str, Any]:
    """Hash every immutable shard and atomically publish the global store."""
    root = root.resolve(strict=True)
    manifests = []
    for path in sorted(root.glob("*.manifest.json")):
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if value.get("kind") == SHARD_KIND:
            data = root / value["data_file"]
            index = root / value["index_file"]
            if sha256_file(data) != value["data_sha256"]:
                raise ShardCorruptionError(f"data hash mismatch: {data}")
            if sha256_file(index) != value["index_sha256"]:
                raise ShardCorruptionError(f"index hash mismatch: {index}")
            manifests.append(value)
    episodes = sum(int(value["episode_count"]) for value in manifests)
    ticks = sum(int(value["tick_count"]) for value in manifests)
    if episodes != expected_episodes:
        raise RuntimeError(f"expected {expected_episodes} episodes, found {episodes}")
    if expected_ticks is not None and ticks != expected_ticks:
        raise RuntimeError(f"expected {expected_ticks} Ticks, found {ticks}")
    source_manifest = source_manifest.resolve(strict=True)
    digest_input = {
        value["name"]: {
            "data": value["data_sha256"], "index": value["index_sha256"]
        }
        for value in manifests
    }
    content_digest = hashlib.sha256(
        json.dumps(digest_input, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "kind": STORE_KIND,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": {
            "path": str(source_manifest),
            "sha256": sha256_file(source_manifest),
        },
        "episode_count": episodes,
        "tick_count": ticks,
        "tick_hz": 20,
        "every_native_tick_present": True,
        "shards": manifests,
        "content_sha256": content_digest,
        "total_bytes": sum(int(value["bytes"]) for value in manifests),
    }
    _atomic_json(root / "manifest.json", manifest)
    (root / "manifest.sha256").write_text(
        f"{sha256_file(root / 'manifest.json')}  manifest.json\n", encoding="ascii"
    )
    return manifest

