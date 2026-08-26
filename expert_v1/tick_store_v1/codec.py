"""Deterministic binary anchor+delta codec for every native 20 Hz Tick."""

from __future__ import annotations

from dataclasses import replace
import bisect
import json
import struct
from typing import Any, Iterable, Mapping
import zlib

from .schema import (
    ENTITY_FIELDS,
    EPISODE_FIELDS,
    PLAYER_FIELDS,
    TOWER_FIELDS,
    EntityState,
    EpisodeState,
    PlayerPrivate,
    TickState,
    TowerState,
    require_consecutive,
)


EPISODE_MAGIC = b"CRTEPV1\0"
CHUNK_MAGIC = b"TCK1"
EPISODE_HEADER = struct.Struct("<8sHHIII")
CHUNK_HEADER = struct.Struct("<4sIHHIII")
CODEC_NONE = 0
CODEC_ZLIB = 1


class TickCodecError(ValueError):
    pass


def _uvarint(value: int) -> bytes:
    if value < 0:
        raise TickCodecError("unsigned varint cannot encode a negative value")
    output = bytearray()
    while value >= 0x80:
        output.append((value & 0x7F) | 0x80)
        value >>= 7
    output.append(value)
    return bytes(output)


def _svarint(value: int) -> bytes:
    return _uvarint((value << 1) ^ (value >> 63))


class _Cursor:
    def __init__(self, data: bytes | memoryview) -> None:
        self.data = memoryview(data)
        self.offset = 0

    def take(self, length: int) -> memoryview:
        stop = self.offset + length
        if length < 0 or stop > len(self.data):
            raise TickCodecError("truncated binary Tick payload")
        result = self.data[self.offset:stop]
        self.offset = stop
        return result

    def uvarint(self) -> int:
        value = 0
        shift = 0
        for _ in range(10):
            byte = int(self.take(1)[0])
            value |= (byte & 0x7F) << shift
            if not byte & 0x80:
                return value
            shift += 7
        raise TickCodecError("varint exceeds 64 bits")

    def svarint(self) -> int:
        value = self.uvarint()
        return (value >> 1) ^ -(value & 1)

    def finish(self) -> None:
        if self.offset != len(self.data):
            raise TickCodecError("unexpected trailing Tick payload bytes")


def _encode_values(values: Iterable[int]) -> bytes:
    return b"".join(_svarint(int(value)) for value in values)


def _decode_values(cursor: _Cursor, count: int) -> tuple[int, ...]:
    return tuple(cursor.svarint() for _ in range(count))


def _encode_player(player: PlayerPrivate) -> bytes:
    return _encode_values((player.side, *player.values()))


def _decode_player(cursor: _Cursor) -> PlayerPrivate:
    side, elixir, h0, h1, h2, h3, next_index = _decode_values(cursor, 7)
    return PlayerPrivate(side, elixir, (h0, h1, h2, h3), next_index)


def _encode_tower(tower: TowerState) -> bytes:
    return _encode_values((tower.key, *tower.values()))


def _decode_tower(cursor: _Cursor) -> TowerState:
    key, *values = _decode_values(cursor, 1 + len(TOWER_FIELDS))
    return TowerState(key, *values)


def _encode_entity(entity: EntityState) -> bytes:
    return _encode_values((entity.key, *entity.values()))


def _decode_entity(cursor: _Cursor) -> EntityState:
    key, *values = _decode_values(cursor, 1 + len(ENTITY_FIELDS))
    return EntityState(key, *values)


def _encode_episode_state(episode: EpisodeState) -> bytes:
    return _encode_values(episode.values())


def _decode_episode_state(cursor: _Cursor) -> EpisodeState:
    return EpisodeState(*_decode_values(cursor, len(EPISODE_FIELDS)))


def encode_anchor(state: TickState) -> bytes:
    output = bytearray(_svarint(state.tick))
    output.extend(_uvarint(len(state.players)))
    for player in state.players:
        output.extend(_encode_player(player))
    output.extend(_uvarint(len(state.towers)))
    for tower in state.towers:
        output.extend(_encode_tower(tower))
    output.extend(_uvarint(len(state.entities)))
    for entity in state.entities:
        output.extend(_encode_entity(entity))
    output.extend(_encode_episode_state(state.episode))
    return bytes(output)


def decode_anchor(data: bytes | memoryview) -> TickState:
    cursor = _Cursor(data)
    tick = cursor.svarint()
    players = tuple(_decode_player(cursor) for _ in range(cursor.uvarint()))
    towers = tuple(_decode_tower(cursor) for _ in range(cursor.uvarint()))
    entities = tuple(_decode_entity(cursor) for _ in range(cursor.uvarint()))
    episode = _decode_episode_state(cursor)
    cursor.finish()
    if len(players) != 2:
        raise TickCodecError("anchor does not contain exactly two players")
    return TickState(
        tick,
        (players[0], players[1]),
        tuple(sorted(towers, key=lambda item: item.key)),
        tuple(sorted(entities, key=lambda item: item.key)),
        episode,
    )


def _changed(previous: tuple[int, ...], current: tuple[int, ...]) -> tuple[int, list[int]]:
    mask = 0
    values: list[int] = []
    for index, (before, after) in enumerate(zip(previous, current, strict=True)):
        if before != after:
            mask |= 1 << index
            values.append(after)
    return mask, values


def _apply(values: tuple[int, ...], mask: int, cursor: _Cursor) -> tuple[int, ...]:
    output = list(values)
    if mask >> len(values):
        raise TickCodecError("delta field mask exceeds schema")
    for index in range(len(values)):
        if mask & (1 << index):
            output[index] = cursor.svarint()
    return tuple(output)


def _encode_keyed_delta(
    previous: Mapping[int, Any],
    current: Mapping[int, Any],
    encode_full: Any,
) -> bytes:
    output = bytearray()
    spawned = sorted(set(current) - set(previous))
    removed = sorted(set(previous) - set(current))
    updated: list[tuple[int, int, list[int]]] = []
    for key in sorted(set(previous) & set(current)):
        mask, values = _changed(previous[key].values(), current[key].values())
        if mask:
            updated.append((key, mask, values))
    output.extend(_uvarint(len(spawned)))
    for key in spawned:
        output.extend(encode_full(current[key]))
    output.extend(_uvarint(len(removed)))
    for key in removed:
        output.extend(_svarint(key))
    output.extend(_uvarint(len(updated)))
    for key, mask, values in updated:
        output.extend(_svarint(key))
        output.extend(_uvarint(mask))
        output.extend(_encode_values(values))
    return bytes(output)


def _decode_keyed_delta(
    cursor: _Cursor,
    previous: Mapping[int, Any],
    decode_full: Any,
    constructor: Any,
) -> dict[int, Any]:
    output = dict(previous)
    for _ in range(cursor.uvarint()):
        value = decode_full(cursor)
        if value.key in output:
            raise TickCodecError("spawned key already exists")
        output[value.key] = value
    for _ in range(cursor.uvarint()):
        key = cursor.svarint()
        if key not in output:
            raise TickCodecError("removed key does not exist")
        del output[key]
    for _ in range(cursor.uvarint()):
        key = cursor.svarint()
        mask = cursor.uvarint()
        if key not in output:
            raise TickCodecError("updated key does not exist")
        values = _apply(output[key].values(), mask, cursor)
        output[key] = constructor(key, *values)
    return output


def encode_delta(previous: TickState, current: TickState) -> bytes:
    if current.tick != previous.tick + 1:
        raise TickCodecError("delta Tick must immediately follow previous Tick")
    output = bytearray()
    for before, after in zip(previous.players, current.players, strict=True):
        if before.side != after.side:
            raise TickCodecError("player side changed inside episode")
        mask, values = _changed(before.values(), after.values())
        output.extend(_uvarint(mask))
        output.extend(_encode_values(values))
    output.extend(
        _encode_keyed_delta(
            {item.key: item for item in previous.towers},
            {item.key: item for item in current.towers},
            _encode_tower,
        )
    )
    output.extend(
        _encode_keyed_delta(
            {item.key: item for item in previous.entities},
            {item.key: item for item in current.entities},
            _encode_entity,
        )
    )
    mask, values = _changed(previous.episode.values(), current.episode.values())
    output.extend(_uvarint(mask))
    output.extend(_encode_values(values))
    return bytes(output)


def decode_delta(previous: TickState, data: bytes | memoryview) -> TickState:
    cursor = _Cursor(data)
    players: list[PlayerPrivate] = []
    for player in previous.players:
        values = _apply(player.values(), cursor.uvarint(), cursor)
        players.append(PlayerPrivate(player.side, values[0], values[1:5], values[5]))
    towers = _decode_keyed_delta(
        cursor,
        {item.key: item for item in previous.towers},
        _decode_tower,
        TowerState,
    )
    entities = _decode_keyed_delta(
        cursor,
        {item.key: item for item in previous.entities},
        _decode_entity,
        EntityState,
    )
    episode_values = _apply(previous.episode.values(), cursor.uvarint(), cursor)
    cursor.finish()
    return TickState(
        previous.tick + 1,
        (players[0], players[1]),
        tuple(sorted(towers.values(), key=lambda item: item.key)),
        tuple(sorted(entities.values(), key=lambda item: item.key)),
        EpisodeState(*episode_values),
    )


def _encode_chunk(states: list[TickState], compression_level: int) -> tuple[bytes, int]:
    raw = bytearray()
    anchor = encode_anchor(states[0])
    raw.extend(_uvarint(len(anchor)))
    raw.extend(anchor)
    for previous, current in zip(states, states[1:]):
        delta = encode_delta(previous, current)
        raw.extend(_uvarint(len(delta)))
        raw.extend(delta)
    raw_bytes = bytes(raw)
    compressed = zlib.compress(raw_bytes, level=compression_level)
    if len(compressed) >= len(raw_bytes):
        codec, body = CODEC_NONE, raw_bytes
    else:
        codec, body = CODEC_ZLIB, compressed
    header = CHUNK_HEADER.pack(
        CHUNK_MAGIC,
        states[0].tick,
        len(states),
        codec,
        len(raw_bytes),
        len(body),
        zlib.crc32(raw_bytes),
    )
    return header + body, len(raw_bytes)


def encode_episode(
    states: Iterable[TickState],
    metadata: Mapping[str, Any],
    *,
    anchor_interval: int = 256,
    compression_level: int = 1,
) -> tuple[bytes, dict[str, int]]:
    frames = require_consecutive(states)
    if not 16 <= anchor_interval <= 4096:
        raise ValueError("anchor_interval must be in 16..4096")
    if not 0 <= compression_level <= 9:
        raise ValueError("compression_level must be in 0..9")
    metadata_value = {
        **dict(metadata),
        "tick_start": frames[0].tick,
        "tick_stop": frames[-1].tick + 1,
        "tick_count": len(frames),
        "anchor_interval": anchor_interval,
        "store_schema_version": 1,
    }
    metadata_bytes = json.dumps(
        metadata_value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    chunks: list[bytes] = []
    raw_bytes = 0
    for start in range(0, len(frames), anchor_interval):
        chunk, raw_size = _encode_chunk(
            frames[start : start + anchor_interval], compression_level
        )
        chunks.append(chunk)
        raw_bytes += raw_size
    header = EPISODE_HEADER.pack(
        EPISODE_MAGIC,
        1,
        anchor_interval,
        len(metadata_bytes),
        len(chunks),
        len(frames),
    )
    blob = header + metadata_bytes + b"".join(chunks)
    return blob, {
        "ticks": len(frames),
        "chunks": len(chunks),
        "raw_delta_bytes": raw_bytes,
        "stored_bytes": len(blob),
    }


class EpisodeReader:
    """Random/sequential access bounded by one periodic anchor chunk."""

    def __init__(self, blob: bytes | memoryview) -> None:
        self.blob = memoryview(blob)
        if len(self.blob) < EPISODE_HEADER.size:
            raise TickCodecError("truncated episode header")
        magic, version, interval, metadata_size, chunks, ticks = EPISODE_HEADER.unpack(
            self.blob[: EPISODE_HEADER.size]
        )
        if magic != EPISODE_MAGIC or version != 1:
            raise TickCodecError("unsupported episode blob")
        self.anchor_interval = interval
        self.tick_count = ticks
        offset = EPISODE_HEADER.size
        stop = offset + metadata_size
        if stop > len(self.blob):
            raise TickCodecError("truncated episode metadata")
        self.metadata = json.loads(bytes(self.blob[offset:stop]))
        offset = stop
        self.chunks: list[tuple[int, int, int, int, int, int]] = []
        for _ in range(chunks):
            header_stop = offset + CHUNK_HEADER.size
            if header_stop > len(self.blob):
                raise TickCodecError("truncated chunk header")
            chunk = CHUNK_HEADER.unpack(self.blob[offset:header_stop])
            magic, start_tick, count, codec, raw_size, stored_size, crc = chunk
            if magic != CHUNK_MAGIC or count <= 0 or codec not in (CODEC_NONE, CODEC_ZLIB):
                raise TickCodecError("invalid Tick chunk header")
            body_offset = header_stop
            offset = body_offset + stored_size
            if offset > len(self.blob):
                raise TickCodecError("truncated Tick chunk")
            self.chunks.append((start_tick, count, body_offset, stored_size, raw_size, (codec << 32) | crc))
        if offset != len(self.blob):
            raise TickCodecError("unexpected bytes after final Tick chunk")
        if sum(item[1] for item in self.chunks) != self.tick_count:
            raise TickCodecError("episode Tick count does not match chunks")
        self._starts = [item[0] for item in self.chunks]

    def _decode_chunk(self, index: int) -> list[TickState]:
        start, count, offset, stored, raw_size, codec_crc = self.chunks[index]
        codec, crc = codec_crc >> 32, codec_crc & 0xFFFFFFFF
        body = bytes(self.blob[offset : offset + stored])
        raw = zlib.decompress(body) if codec == CODEC_ZLIB else body
        if len(raw) != raw_size or zlib.crc32(raw) != crc:
            raise TickCodecError("Tick chunk checksum mismatch")
        cursor = _Cursor(raw)
        anchor = decode_anchor(cursor.take(cursor.uvarint()))
        if anchor.tick != start:
            raise TickCodecError("anchor Tick disagrees with chunk header")
        states = [anchor]
        while len(states) < count:
            states.append(decode_delta(states[-1], cursor.take(cursor.uvarint())))
        cursor.finish()
        return states

    def iter_ticks(self) -> Iterable[TickState]:
        for index in range(len(self.chunks)):
            yield from self._decode_chunk(index)

    def read_tick(self, tick: int) -> TickState:
        index = bisect.bisect_right(self._starts, tick) - 1
        if index < 0:
            raise KeyError(tick)
        states = self._decode_chunk(index)
        offset = tick - states[0].tick
        if offset < 0 or offset >= len(states):
            raise KeyError(tick)
        return states[offset]

