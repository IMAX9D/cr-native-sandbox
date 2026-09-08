"""Version-pinned read-only MuMu transport and battle-clock admission.

No torch, no model, no emulator launch, no game commands or screen input.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
import shutil

VERSION_CODE = 160402002
LIBG_SHA256 = 'd2c8efcfe8f77e21d6a5e219999e938a42daefd7ab32f546375fb91c01f9ca97'
MANAGER_RVA = 0x1A569A8
ROOT_CONTEXT_OFFSET = 0x28
REMOTE_READER = '/data/local/tmp/mumu-live-reader-v2'
DEFAULT_ADB = Path(os.environ.get('CR_MUMU_ADB') or shutil.which('adb') or r'C:\Program Files\Netease\MuMu\nx_device\12.0\shell\adb.exe')
DEFAULT_SERIAL = '127.0.0.1:16416'
DEFAULT_READER = Path(__file__).resolve().parents[1] / 'artifacts/mumu-live/mumu-live-reader-v2-x86_64'
DEFAULT_LOG_ROOT = Path(os.environ.get('CR_MUMU_LOG_ROOT') or str(Path(__file__).resolve().parents[1] / 'artifacts/mumu-live-expert'))
NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0


def adb_run(adb: Path, serial: str, *args: str, timeout=20, check=True) -> str:
    result = subprocess.run([str(adb), '-s', serial, *args], capture_output=True,
        text=True, encoding='utf-8', errors='replace', timeout=timeout, creationflags=NO_WINDOW)
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or 'ADB failed')
    return result.stdout.strip()


def root_command(command: str) -> str:
    return 'su -c ' + shlex.quote(command)


def root_run(adb: Path, serial: str, command: str, **kwargs) -> str:
    return adb_run(adb, serial, 'shell', root_command(command), **kwargs)


def verify_runtime(adb: Path, serial: str) -> dict:
    if adb_run(adb, serial, 'get-state', timeout=5) != 'device':
        raise RuntimeError('指定已有设备不可用；不会启动其它安卓设备')
    package = adb_run(adb, serial, 'shell', 'dumpsys package com.supercell.clashroyale')
    version = re.search(r'\bversionCode=(\d+)\b', package)
    if not version or int(version[1]) != VERSION_CODE or 'primaryCpuAbi=arm64-v8a' not in package:
        raise RuntimeError('未认证的游戏版本/ABI；停止而不是套用旧偏移')
    pid_text = adb_run(adb, serial, 'shell', 'pidof com.supercell.clashroyale')
    if not pid_text.isdigit():
        raise RuntimeError('游戏没有运行或存在多个主进程；请手动打开已有设备上的游戏')
    pid = int(pid_text)
    maps = root_run(adb, serial, f'cat /proc/{pid}/maps')
    paths = {line.split()[-1] for line in maps.splitlines() if '/lib/arm64/libg.so' in line}
    if len(paths) != 1:
        raise RuntimeError('找不到唯一 ARM64 libg 映射')
    path = next(iter(paths))
    actual = root_run(adb, serial, 'sha256sum ' + shlex.quote(path)).split()[0]
    if actual != LIBG_SHA256:
        raise RuntimeError(f'libg SHA-256 不符：{actual}；禁止扫描/控制未知版本')
    stat = root_run(adb, serial, f'cat /proc/{pid}/stat')
    start_ticks = stat.rsplit(')', 1)[1].split()[19]
    fingerprint = json.loads(root_run(adb, serial, 'cat /data/user/0/com.supercell.clashroyale/update/fingerprint.json'))
    return {'package': 'com.supercell.clashroyale', 'version_code': VERSION_CODE,
        'abi': 'arm64-v8a', 'libg_sha256': actual, 'pid': pid, 'process_start_ticks': start_ticks,
        'serial': serial, 'resource_version': fingerprint.get('version'),
        'resource_fingerprint': fingerprint.get('sha'), 'memory_access': 'read_only'}


def install_reader(adb: Path, serial: str, reader: Path) -> str:
    if not reader.is_file():
        raise FileNotFoundError(f'{reader}; 请先运行 scripts/build_mumu_live_private.ps1')
    expected = hashlib.sha256(reader.read_bytes()).hexdigest()
    remote = root_run(adb, serial, f'sha256sum {REMOTE_READER}', check=False).split()
    if remote and remote[0] == expected:
        return expected
    # Do not overwrite a binary currently used by another observer/controller.
    running = root_run(adb, serial, 'pidof mumu-live-reader-v2', check=False)
    if running.strip():
        raise RuntimeError('已有不同版本的只读采样器运行中，请先关闭该观察入口')
    adb_run(adb, serial, 'push', str(reader), REMOTE_READER)
    adb_run(adb, serial, 'shell', 'chmod', '755', REMOTE_READER)
    if root_run(adb, serial, f'sha256sum {REMOTE_READER}').split()[0] != expected:
        raise RuntimeError('采样器上传 SHA-256 不一致')
    return expected


def start_reader(adb: Path, serial: str, pid: int, *, interval_ms=50, max_frames=0):
    if type(pid) is not int or pid <= 0 or not 20 <= interval_ms <= 5000 or max_frames < 0:
        raise ValueError('invalid reader arguments')
    command = f'{REMOTE_READER} {pid} {interval_ms} {hex(MANAGER_RVA)} {hex(ROOT_CONTEXT_OFFSET)} --unified {max_frames}'
    return subprocess.Popen([str(adb), '-s', serial, 'shell', root_command(command)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8',
        errors='replace', bufsize=1, creationflags=NO_WINDOW)


def stop_owned_reader(adb: Path, serial: str, reader_pid: int | None, game_pid: int) -> None:
    if type(reader_pid) is not int or reader_pid <= 0 or reader_pid == game_pid:
        return
    cmdline = root_run(adb, serial, f'cat /proc/{reader_pid}/cmdline', check=False)
    parts = cmdline.split('\0')
    if len(parts) >= 2 and parts[0] == REMOTE_READER and parts[1] == str(game_pid):
        root_run(adb, serial, f'kill -TERM {reader_pid}', check=False)


def visible_sides(frame: dict) -> list[int]:
    result = []
    for player in frame.get('players', []):
        hand, nxt = player.get('hand_deck_indices', []), player.get('next_deck_index')
        if len(hand) == 4 and all(type(v) is int and -1 <= v < 8 for v in hand):
            visible = [v for v in hand if v >= 0]
            if visible and len(set(visible)) == len(visible) and type(nxt) is int and 0 <= nxt < 8:
                result.append(player.get('side'))
    return sorted(side for side in result if side in (0, 1))


class BattleClockGuard:
    """Structure validity is not liveness; a paused/replayed frame cannot act."""
    def __init__(self, stale_seconds=1.0):
        self.stale_seconds = stale_seconds
        self.identity = None
        self.tick = -1
        self.last_progress = None
        self.advances = 0
        self.epoch = 0

    def observe(self, frame: dict | None, now=None) -> dict:
        now = time.monotonic() if now is None else now
        result = {'can_control': False, 'status': 'unresolved', 'epoch': self.epoch,
                  'clock_advances': self.advances, 'local_side': None}
        if not frame or frame.get('schema_version') != 2 or not frame.get('battle_active'):
            return result
        if not frame.get('coherent'):
            return {**result, 'status': 'incoherent'}
        chain = frame.get('chain') or {}
        if chain.get('player_state_path') != [0xA8] or not chain.get('battle') or chain.get('battle') == '0x0':
            return {**result, 'status': 'unverified_chain'}
        tick = frame.get('game_tick')
        if type(tick) is not int or tick < 0:
            return {**result, 'status': 'invalid_tick'}
        entities = frame.get('entities')
        if not isinstance(entities, list) or not entities or len(entities) != frame.get('decoded_entity_count'):
            return {**result, 'status': 'empty_or_incomplete_scene'}
        living_kings = {entity.get('side') for entity in entities
            if entity.get('card_id') == -1 and 6500 <= entity.get('x', -1) <= 11500
            and entity.get('hp', -1) > 0 and entity.get('max_hp', -1) >= entity.get('hp', -1)}
        if living_kings != {0, 1}:
            return {**result, 'status': 'terminal_or_unverified_towers'}
        identity = (frame.get('pid'), chain.get('root'), chain.get('context'), chain.get('battle'), chain.get('player_state'))
        if identity != self.identity or tick < self.tick:
            self.identity, self.tick, self.last_progress = identity, tick, now
            self.advances = 0
            self.epoch += 1
        elif tick > self.tick:
            self.tick, self.last_progress = tick, now
            self.advances += 1
        sides = visible_sides(frame)
        status = 'warming_up'
        if self.last_progress is not None and now - self.last_progress > self.stale_seconds:
            status = 'paused_or_stalled'
        elif len(sides) != 1:
            status = 'both_hands_visible_readonly' if len(sides) == 2 else 'local_side_unknown'
        elif self.advances >= 2:
            status = 'live_candidate'
        return {'can_control': status == 'live_candidate', 'status': status,
                'epoch': self.epoch, 'clock_advances': self.advances,
                'local_side': sides[0] if len(sides) == 1 else None,
                'visible_sides': sides, 'tick': tick}
