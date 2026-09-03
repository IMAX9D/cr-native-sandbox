"""Auto-detect a live MuMu friendly battle and let an expert policy control it.

The live game process is observed read-only through the already verified
``/proc/<pid>/mem`` samplers.  Actions are sent as ordinary Android touch
gestures; this module never patches libg and never calls its command executor
inside the online client.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from .gui import CARD_COSTS
from .human_vs_ai import (
    DEFAULT_EXPERT_DATASET,
    HumanVsAiGui,
    _deck_token_id,
    _load_policy,
    _seed_everything,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = Path(
    r"D:\AI_data\cr-native-core\expert-v1\downloaded\lr-ab-20260831"
    r"\candidate-lr5e-5-step157674-fp16.pt"
)
DEFAULT_DECK = PROJECT_ROOT / "examples" / "user-selected-heavy-control.json"
DEFAULT_ENTITY_HELPER = Path(
    r"D:\AI_data\worktrees\cr_re-formal\native\bin"
    r"\cr-live-sampler-x86_64"
)
DEFAULT_PRIVATE_HELPER = (
    PROJECT_ROOT / "artifacts" / "mumu-live" / "mumu-live-private-x86_64"
)
DEFAULT_MUMU_CLI = Path(r"C:\Program Files\Netease\MuMu\nx_main\mumu-cli.exe")
DEFAULT_ADB = Path(
    r"C:\Program Files\Netease\MuMu\nx_device\12.0\shell\adb.exe"
)
DEFAULT_SERIAL = "127.0.0.1:16416"
EXPECTED_VERSION_CODE = 160402002
MANAGER_GLOBAL_RVA = 0x1A569A8
ROOT_CONTEXT_OFFSET = 0x28
TERMINAL_GAME_TICK = 6150
REMOTE_ENTITY = "/data/local/tmp/cr-live-sampler-expert"
REMOTE_PRIVATE = "/data/local/tmp/mumu-live-private-expert"
LOG_ROOT = Path(r"D:\AI_data\cr-native-core\mumu-live-expert")
STATUS_PATH = LOG_ROOT / "controller-status.json"
NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


class LiveControllerError(RuntimeError):
    pass


@dataclass(frozen=True)
class ScreenLayout:
    width: int
    height: int
    viewport_left: float
    viewport_width: float
    arena_left: float
    arena_right: float
    arena_top: float
    arena_bottom: float
    hand_y: float
    hand_x: tuple[float, float, float, float]

    @classmethod
    def from_size(cls, width: int, height: int) -> "ScreenLayout":
        if width <= 0 or height <= 0:
            raise ValueError("invalid Android display size")
        # Royale is portrait.  If Android reports a landscape framebuffer,
        # use the centered 9:16 game viewport and leave side bars untouched.
        viewport_width = float(width)
        viewport_left = 0.0
        if width / height > 0.8:
            viewport_width = min(float(width), float(height) * 9.0 / 16.0)
            viewport_left = (float(width) - viewport_width) / 2.0
        return cls(
            width=width,
            height=height,
            viewport_left=viewport_left,
            viewport_width=viewport_width,
            arena_left=viewport_left + viewport_width * 0.055,
            arena_right=viewport_left + viewport_width * 0.945,
            arena_top=height * 0.105,
            arena_bottom=height * 0.790,
            hand_y=height * 0.912,
            hand_x=tuple(
                viewport_left + viewport_width * value
                for value in (0.275, 0.435, 0.595, 0.755)
            ),
        )

    def deployment_point(self, canonical_position: int) -> tuple[int, int]:
        row, column = divmod(int(canonical_position), 18)
        x_fraction = (column + 0.5) / 18.0
        y_fraction = (row + 0.5) / 32.0
        x = self.arena_left + x_fraction * (self.arena_right - self.arena_left)
        y = self.arena_bottom - y_fraction * (self.arena_bottom - self.arena_top)
        return round(x), round(y)

    def hand_point(self, slot: int) -> tuple[int, int]:
        if slot not in range(4):
            raise ValueError("hand slot must be 0..3")
        return round(self.hand_x[slot]), round(self.hand_y)


class JsonLineStream:
    def __init__(self, process: subprocess.Popen[str], event: str) -> None:
        self.process = process
        self.event = event
        self.latest: dict[str, Any] | None = None
        self.updated = 0.0
        self.stderr: list[str] = []
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_thread.start()

    def _read(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if value.get("event") != self.event:
                continue
            with self._lock:
                self.latest = value
                self.updated = time.monotonic()

    def _read_stderr(self) -> None:
        assert self.process.stderr is not None
        for line in self.process.stderr:
            text = line.strip()
            if text:
                self.stderr.append(text)
                del self.stderr[:-20]

    def snapshot(self) -> tuple[dict[str, Any] | None, float]:
        with self._lock:
            return self.latest, self.updated

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()


def _run(args: list[str], *, timeout: float = 30, check: bool = True) -> str:
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        creationflags=NO_WINDOW,
    )
    if check and result.returncode != 0:
        raise LiveControllerError(
            result.stderr.strip() or result.stdout.strip() or f"command failed: {args[0]}"
        )
    return result.stdout.strip()


def _mumu_info(cli: Path, vmindex: int) -> dict[str, Any]:
    return json.loads(_run([str(cli), "info", "-v", str(vmindex)]))


def _ensure_mumu(cli: Path, vmindex: int) -> None:
    info = _mumu_info(cli, vmindex)
    if not bool(info.get("is_android_started")):
        _run(
            [str(cli), "control", "-v", str(vmindex), "-ver", "12", "launch"]
        )
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            time.sleep(2)
            info = _mumu_info(cli, vmindex)
            if bool(info.get("is_android_started")):
                break
        else:
            raise LiveControllerError("MuMu Android 12 did not finish booting")
    _run(
        [str(cli), "control", "-v", str(vmindex), "-ver", "12", "show_window"],
        check=False,
    )


def _adb(adb: Path, serial: str, *args: str, timeout: float = 30,
         check: bool = True) -> str:
    return _run([str(adb), "-s", serial, *args], timeout=timeout, check=check)


def _connect_adb(adb: Path, serial: str) -> None:
    _run([str(adb), "connect", serial], timeout=15, check=False)
    state = _adb(adb, serial, "get-state", timeout=10)
    if state != "device":
        raise LiveControllerError(f"MuMu ADB is not ready: {state or 'unavailable'}")


def _root_shell(cli: Path, vmindex: int, command: str, *, timeout: float = 30,
                check: bool = True) -> str:
    return _run(
        [str(cli), "sh", "-v", str(vmindex), "-c", command],
        timeout=timeout,
        check=check,
    )


def _ensure_game(cli: Path, adb: Path, serial: str, vmindex: int) -> int:
    pid_text = _root_shell(
        cli, vmindex, "pidof com.supercell.clashroyale", check=False
    ).strip()
    if not pid_text:
        _adb(
            adb,
            serial,
            "shell",
            "monkey",
            "-p",
            "com.supercell.clashroyale",
            "-c",
            "android.intent.category.LAUNCHER",
            "1",
            timeout=20,
        )
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            time.sleep(2)
            pid_text = _root_shell(
                cli, vmindex, "pidof com.supercell.clashroyale", check=False
            ).strip()
            if pid_text:
                break
    values = re.findall(r"\d+", pid_text)
    if not values:
        raise LiveControllerError("Clash Royale process did not start")
    return int(values[0])


def _verify_game_runtime(cli: Path, vmindex: int, pid: int) -> None:
    package = _root_shell(
        cli,
        vmindex,
        "dumpsys package com.supercell.clashroyale | grep versionCode | head -1",
    )
    match = re.search(r"versionCode=(\d+)", package)
    if not match or int(match.group(1)) != EXPECTED_VERSION_CODE:
        actual = match.group(1) if match else "unknown"
        raise LiveControllerError(
            "Clash Royale runtime guard rejected versionCode "
            f"{actual}; expected {EXPECTED_VERSION_CODE}"
        )
    maps = _root_shell(
        cli,
        vmindex,
        f"cat /proc/{pid}/maps | grep '/lib/arm64/libg.so' | head -1",
    )
    if "/lib/arm64/libg.so" not in maps:
        raise LiveControllerError(
            "Clash Royale runtime guard requires the verified ARM64 libg mapping"
        )


def _push_helpers(
    cli: Path,
    adb: Path,
    serial: str,
    vmindex: int,
    entity_helper: Path,
    private_helper: Path,
) -> None:
    for local, remote in (
        (entity_helper, REMOTE_ENTITY),
        (private_helper, REMOTE_PRIVATE),
    ):
        if not local.is_file():
            raise FileNotFoundError(local)
        _adb(adb, serial, "push", str(local), remote, timeout=60)
        _root_shell(cli, vmindex, f"chmod 755 {remote}")


def _stop_remote_helpers(cli: Path, vmindex: int) -> None:
    for process_name in (
        "cr-live-sampler-expert",
        "mumu-live-private-expert",
    ):
        _root_shell(
            cli,
            vmindex,
            f"for p in $(pidof {process_name} 2>/dev/null); "
            'do kill -9 "$p" 2>/dev/null; done',
            timeout=10,
            check=False,
        )


def _start_root_stream(
    cli: Path, vmindex: int, command: str, event: str
) -> JsonLineStream:
    process = subprocess.Popen(
        [str(cli), "sh", "-v", str(vmindex), "-c", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=NO_WINDOW,
    )
    return JsonLineStream(process, event)


def _load_decks(path: Path) -> list[list[dict[str, int]]]:
    replay = json.loads(path.read_text(encoding="utf-8-sig"))
    result: list[list[dict[str, int]]] = []
    for side in range(2):
        spells = replay["battle"][f"deck{side}"]["sp"]
        if len(spells) != 8:
            raise LiveControllerError("configured live deck must contain eight cards")
        result.append(
            [
                {
                    "card_id": int(item["d"]),
                    "level": int(item.get("l", 10)) + 1,
                    "form_flags": int(item.get("el", 0)),
                }
                for item in spells
            ]
        )
    return result


def _tower_rows(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    towers: list[dict[str, Any]] = []
    for item in entities:
        if int(item.get("card_id", -2)) != -1:
            continue
        hp = int(item.get("hp", -1))
        max_hp = int(item.get("max_hp", -1))
        if hp < 0 or max_hp <= 0:
            continue
        x = int(item.get("x", -1))
        y = int(item.get("y", -1))
        side = int(item.get("side", -1))
        if side not in (0, 1) or not 0 <= x <= 18000 or not 0 <= y <= 32000:
            continue
        role = "king" if 6500 <= x <= 11500 else "princess"
        lane = None if role == "king" else ("left" if x < 9000 else "right")
        towers.append(
            {
                "side": side,
                "type": role,
                "lane": lane,
                "x": x,
                "y": y,
                "hp": hp,
                "max_hp": max_hp,
            }
        )
    # A coherent battle has exactly six crown towers; tolerate destroyed towers
    # remaining in the registry, but never duplicate one logical slot.
    unique: dict[tuple[int, str, str | None], dict[str, Any]] = {}
    for tower in towers:
        key = (int(tower["side"]), str(tower["type"]), tower["lane"])
        current = unique.get(key)
        if current is None or int(tower["max_hp"]) > int(current["max_hp"]):
            unique[key] = tower
    return list(unique.values())


def _native_state(
    entity_frame: dict[str, Any], private_frame: dict[str, Any]
) -> dict[str, Any]:
    entities: list[dict[str, Any]] = []
    for raw in entity_frame.get("entities", []):
        if not isinstance(raw, dict):
            continue
        entities.append(
            {
                "category": int(raw.get("category", -1)),
                "side": int(raw.get("side", -1)),
                "x": int(raw.get("x", -1)),
                "y": int(raw.get("y", -1)),
                "card_id": int(raw.get("card_id", -1)),
                "level": int(raw.get("level", -1)),
                "hp": int(raw.get("hp", -1)),
                "max_hp": int(raw.get("max_hp", -1)),
                "behavior_state": int(raw.get("behavior_state_raw", 0) or 0),
                "ability_slot": int(raw.get("ability_slot", 0) or 0),
                "ability_state_code": int(raw.get("ability_state_code", -1) or -1),
                "ability_available": int(bool(raw.get("ability_available", False))),
                "ability_cooldown_remaining_ms": int(
                    raw.get("ability_cooldown_remaining_ms", -1) or -1
                ),
                "ability_charges_remaining": int(
                    raw.get("ability_charges_remaining", -1) or -1
                ),
                "ability_pending_ms": int(raw.get("ability_pending_ms", -1) or -1),
                "ability_mana_cost": int(raw.get("ability_mana_cost", -1) or -1),
            }
        )
    towers = _tower_rows(entities)
    players: list[dict[str, Any]] = []
    for raw in private_frame.get("players", []):
        item = dict(raw)
        hand = [int(value) for value in item.get("hand_deck_indices", [])]
        # The online client deliberately hides the remote hand and cycle.
        # TickStore validation still requires a syntactically valid next index,
        # while actor_projection discards every opponent-private field.
        if hand == [-1, -1, -1, -1] and int(item.get("next_deck_index", -1)) < 0:
            item["next_deck_index"] = 0
        players.append(item)
    return {
        "kind": "libg_native_state",
        "coherent": bool(entity_frame.get("coherent", True))
        and bool(private_frame.get("coherent", True)),
        "tick": int(entity_frame["game_tick"]),
        "players": players,
        "entities": entities,
        "episode": {
            "commands_allowed": 1,
            "command_gate_code": 0,
            "native_phase": {},
            "terminated": 0,
            "crowns": [0, 0],
            "crown_towers": towers,
        },
    }


def _position_masks(decks: list[list[dict[str, int]]], side: int,
                    hand: list[int], elixir_raw: int,
                    state: dict[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray]:
    card_mask = np.zeros(4, dtype=np.bool_)
    positions = np.zeros((4, 32 * 18), dtype=np.bool_)
    for slot, deck_index in enumerate(hand):
        if deck_index not in range(8):
            continue
        card_id = int(decks[side][deck_index]["card_id"])
        cost = int(CARD_COSTS.get(card_id, 10))
        if elixir_raw < cost * 10_000:
            continue
        card_mask[slot] = True
        table = card_id // 1_000_000
        if table == 28:  # spells, including targeted and global spells
            positions[slot, :] = True
        else:  # troops/buildings: canonical own half
            positions[slot, : 16 * 18] = True
            if state is not None:
                for entity in state.get("entities", []):
                    if (
                        int(entity.get("side", -1)) != side
                        or int(entity.get("card_id", -2)) != -1
                        or int(entity.get("max_hp", -1)) <= 0
                    ):
                        continue
                    tower_x = int(entity.get("x", -1))
                    tower_y = int(entity.get("y", -1))
                    if side == 1:
                        tower_x = 17_999 - tower_x
                        tower_y = 31_999 - tower_y
                    for row in range(16):
                        cell_y = row * 1000 + 500
                        if abs(cell_y - tower_y) >= 3000:
                            continue
                        for column in range(18):
                            cell_x = column * 1000 + 500
                            if abs(cell_x - tower_x) < 3000:
                                positions[slot, row * 18 + column] = False
    return card_mask, positions


def _screen_size(adb: Path, serial: str) -> tuple[int, int]:
    output = _adb(adb, serial, "shell", "wm", "size")
    matches = re.findall(r"(\d+)x(\d+)", output)
    if not matches:
        raise LiveControllerError(f"cannot parse Android screen size: {output}")
    width, height = matches[-1]
    return int(width), int(height)


class MuMuExpertController:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.cli = args.mumu_cli.resolve()
        self.adb = args.adb.resolve()
        self.serial = args.serial
        self.decks = _load_decks(args.deck.resolve())
        selected_device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if args.device == "auto" else args.device
        self.device = torch.device(selected_device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise LiveControllerError("CUDA was requested but is unavailable")
        self.model, self.model_meta = _load_policy(
            args.checkpoint.resolve(),
            device=self.device,
            cuda_graph=self.device.type == "cuda",
            expert_dataset_root=args.expert_dataset_root.resolve(),
        )
        self.policy = SimpleNamespace(
            state=None,
            AI_SIDE=args.local_side,
            TICK_SECONDS=0.05,
            model=self.model,
            device=self.device,
            ai_hidden=self.model.initial_hidden(1, device=self.device),
            expert_card_id_to_token=dict(self.model_meta["card_id_to_token"]),
            expert_ability_id_to_token=dict(self.model_meta["ability_id_to_token"]),
            expert_revealed_enemy_tokens=[],
            expert_generator=torch.Generator(device=self.device).manual_seed(
                args.policy_seed
            ),
            expert_choice_mode=args.choice_mode,
            expert_play_rate_scale=args.play_rate_scale,
            env=SimpleNamespace(decks=self.decks),
        )
        self.entity_stream: JsonLineStream | None = None
        self.private_stream: JsonLineStream | None = None
        self.layout: ScreenLayout | None = None
        self.log_file: Path | None = None
        self.in_battle = False
        self.last_tick = -1
        self.active_streak = 0
        self.inactive_streak = 0
        self.pending: dict[str, Any] | None = None
        self.battle_number = 0
        self.last_status = ""
        self.lifecycle = "starting"
        self.last_event = "starting"
        self.last_message = "正在启动"
        self.last_action: dict[str, Any] | None = None
        self.live_player: dict[str, Any] | None = None
        self.live_deck: list[dict[str, int]] | None = None
        self.local_side = args.local_side
        self.event_counts = {
            "touch_sent": 0,
            "touch_retry": 0,
            "touch_accepted": 0,
            "touch_not_confirmed": 0,
            "inference_error": 0,
        }
        self._last_status_publish = 0.0

    def _status_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "controller_pid": os.getpid(),
            "lifecycle": self.lifecycle,
            "heartbeat_utc": datetime.now(timezone.utc).isoformat(),
            "heartbeat_unix": time.time(),
            "in_battle": self.in_battle,
            "battle_number": self.battle_number,
            "tick": self.last_tick if self.last_tick >= 0 else None,
            "active_streak": self.active_streak,
            "inactive_streak": self.inactive_streak,
            "last_event": self.last_event,
            "last_message": self.last_message,
            "last_action": self.last_action,
            "live_player": self.live_player,
            "live_deck": self.live_deck,
            "event_counts": dict(self.event_counts),
            "checkpoint": str(self.args.checkpoint.resolve()),
            "model_digest": self.model_meta.get("model_digest"),
            "model_step": self.model_meta.get("training_step"),
            "deck": str(self.args.deck.resolve()),
            "local_side": self.local_side,
            "play_rate_scale": self.args.play_rate_scale,
            "dry_run": self.args.dry_run,
            "log_file": str(self.log_file) if self.log_file else None,
            "screen": (
                [self.layout.width, self.layout.height]
                if self.layout is not None else None
            ),
            "serial": self.serial,
        }

    def _publish_status(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_status_publish < 0.2:
            return
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            self._status_payload(), ensure_ascii=False, indent=2
        ) + "\n"
        for attempt in range(20):
            try:
                # Windows readers can deny FILE_SHARE_DELETE, making atomic
                # os.replace fail while the GUI polls.  Updating the same file
                # does not require delete sharing; the GUI already treats a
                # transient partial JSON read as one skipped refresh.
                STATUS_PATH.write_text(payload, encoding="utf-8")
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.01)
        self._last_status_publish = now

    def log(self, event: str, **fields: Any) -> None:
        self.last_event = event
        self.last_message = str(fields.get("error") or fields.get("message") or event)
        if event in self.event_counts:
            self.event_counts[event] += 1
        payload = {
            "time": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **fields,
        }
        line = json.dumps(payload, ensure_ascii=False)
        print(line, flush=True)
        if self.log_file is not None:
            with self.log_file.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        self._publish_status(force=True)

    def _start_streams(self, pid: int) -> None:
        entity_command = (
            f"{REMOTE_ENTITY} {pid} 50 {hex(MANAGER_GLOBAL_RVA)} "
            f"{hex(ROOT_CONTEXT_OFFSET)} 0"
        )
        private_command = (
            f"{REMOTE_PRIVATE} {pid} 50 {hex(MANAGER_GLOBAL_RVA)} "
            f"{hex(ROOT_CONTEXT_OFFSET)}"
        )
        self.entity_stream = _start_root_stream(
            self.cli, self.args.vmindex, entity_command, "cr_live_snapshot"
        )
        self.private_stream = _start_root_stream(
            self.cli, self.args.vmindex, private_command, "mumu_live_private"
        )

    def prepare(self) -> None:
        for required in (self.cli, self.adb, self.args.checkpoint, self.args.deck,
                         self.args.entity_helper, self.args.private_helper):
            if not Path(required).is_file():
                raise FileNotFoundError(required)
        _ensure_mumu(self.cli, self.args.vmindex)
        info = _mumu_info(self.cli, self.args.vmindex)
        adb_host = str(info.get("adb_host_ip") or "127.0.0.1")
        adb_port = int(info.get("adb_port") or 0)
        if adb_port > 0:
            self.serial = f"{adb_host}:{adb_port}"
        _connect_adb(self.adb, self.serial)
        pid = _ensure_game(
            self.cli, self.adb, self.serial, self.args.vmindex
        )
        _verify_game_runtime(self.cli, self.args.vmindex, pid)
        _stop_remote_helpers(self.cli, self.args.vmindex)
        _push_helpers(
            self.cli,
            self.adb,
            self.serial,
            self.args.vmindex,
            self.args.entity_helper.resolve(),
            self.args.private_helper.resolve(),
        )
        width, height = _screen_size(self.adb, self.serial)
        self.layout = ScreenLayout.from_size(width, height)
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.log_file = LOG_ROOT / f"mumu-expert-{stamp}.jsonl"
        self._start_streams(pid)
        self.log(
            "controller_ready",
            pid=pid,
            checkpoint=str(self.args.checkpoint.resolve()),
            model_digest=self.model_meta["model_digest"],
            deck=str(self.args.deck.resolve()),
            local_side=self.args.local_side,
            screen=[width, height],
            dry_run=self.args.dry_run,
        )
        self.lifecycle = "waiting"
        self._publish_status(force=True)

    def _fresh_frames(self) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        assert self.entity_stream and self.private_stream
        entity, entity_time = self.entity_stream.snapshot()
        private, private_time = self.private_stream.snapshot()
        now = time.monotonic()
        if now - entity_time > 1.0 or now - private_time > 1.0:
            return None, None
        return entity, private

    def _stable_active(self, entity: dict[str, Any] | None,
                       private: dict[str, Any] | None) -> bool:
        entity_tick = entity.get("game_tick") if entity else None
        private_tick = private.get("game_tick") if private else None
        return bool(
            entity
            and private
            and entity.get("battle_active")
            and private.get("battle_active")
            and entity.get("coherent", True)
            and private.get("coherent", True)
            and isinstance(entity_tick, int)
            and isinstance(private_tick, int)
            and 0 <= int(entity_tick) < TERMINAL_GAME_TICK
            and 0 <= int(private_tick) < TERMINAL_GAME_TICK
            and abs(int(entity_tick) - int(private_tick)) <= 2
            and len(private.get("players", [])) == 2
            and self._visible_local_side(private) is not None
        )

    @staticmethod
    def _visible_local_side(private: dict[str, Any]) -> int | None:
        candidates: list[int] = []
        for player in private.get("players", []):
            hand = [int(value) for value in player.get("hand_deck_indices", [])]
            visible = [value for value in hand if value in range(8)]
            next_index = int(player.get("next_deck_index", -1))
            if len(hand) == 4 and visible and next_index in range(8):
                candidates.append(int(player.get("side", -1)))
        return candidates[0] if len(candidates) == 1 else None

    def _select_local_side(self, private: dict[str, Any]) -> bool:
        side = self._visible_local_side(private)
        if side is None:
            return False
        self.local_side = side
        self.policy.AI_SIDE = side
        return True

    def _bind_visible_deck(self, player: dict[str, Any]) -> bool:
        card_ids = [int(value) for value in player.get("deck_card_ids", [])]
        form_flags = [int(value) for value in player.get("deck_form_flags", [])]
        if (
            len(card_ids) != 8
            or len(form_flags) != 8
            or any(value < 25_000_000 or value > 29_999_999 for value in card_ids)
            or any(value not in (0, 1, 2) for value in form_flags)
        ):
            return False
        deck = [
            {
                "card_id": card_id,
                "level": 11,
                "form_flags": form,
            }
            for card_id, form in zip(card_ids, form_flags)
        ]
        for card in deck:
            token_id = _deck_token_id(card)
            if token_id not in self.policy.expert_card_id_to_token:
                raise LiveControllerError(
                    f"live deck token {token_id} is absent from the trained vocabulary"
                )
        self.decks[self.local_side] = deck
        self.live_deck = deck
        return True

    def _reset_policy(self) -> None:
        self.policy.ai_hidden = self.model.initial_hidden(1, device=self.device)
        self.policy.expert_revealed_enemy_tokens = []
        self.pending = None
        self.last_tick = -1

    def _begin_battle(self, tick: int, private: dict[str, Any]) -> None:
        if not self._select_local_side(private):
            return
        self.in_battle = True
        self.lifecycle = "controlling"
        self.battle_number += 1
        self.inactive_streak = 0
        self._reset_policy()
        self.log(
            "battle_detected",
            battle=self.battle_number,
            tick=tick,
            local_side=self.local_side,
        )

    def _end_battle(self) -> None:
        self.log("battle_released", battle=self.battle_number, last_tick=self.last_tick)
        self.in_battle = False
        self.lifecycle = "waiting"
        self.active_streak = 0
        self.inactive_streak = 0
        self._reset_policy()

    def _touch(self, slot: int, position: int) -> None:
        assert self.layout is not None
        start = self.layout.hand_point(slot)
        target = self.layout.deployment_point(position)
        if self.args.dry_run:
            return
        _adb(
            self.adb,
            self.serial,
            "shell",
            "input",
            "tap",
            str(start[0]),
            str(start[1]),
            timeout=5,
        )
        time.sleep(0.05)
        _adb(
            self.adb,
            self.serial,
            "shell",
            "input",
            "tap",
            str(target[0]),
            str(target[1]),
            timeout=5,
        )

    def _retry_touch(self, slot: int, position: int) -> None:
        assert self.layout is not None
        start = self.layout.hand_point(slot)
        target = self.layout.deployment_point(position)
        if self.args.dry_run:
            return
        _adb(
            self.adb,
            self.serial,
            "shell",
            "input",
            "tap",
            str(start[0]),
            str(start[1]),
            timeout=5,
        )
        time.sleep(0.05)
        _adb(
            self.adb,
            self.serial,
            "shell",
            "input",
            "tap",
            str(target[0]),
            str(target[1]),
            timeout=5,
        )

    def _update_pending(self, hand: list[int], now: float) -> bool:
        if self.pending is None:
            return False
        if hand != self.pending["hand_before"]:
            self.log(
                "touch_accepted",
                battle=self.battle_number,
                tick=self.pending["tick"],
                slot=self.pending["slot"],
                latency_ms=round((now - self.pending["sent_at"]) * 1000, 1),
            )
            self.pending = None
            return False
        if now - self.pending["sent_at"] >= 1.8:
            if not self.args.dry_run and not self.pending.get("retried", False):
                self._retry_touch(
                    int(self.pending["slot"]), int(self.pending["position"])
                )
                self.pending["retried"] = True
                self.pending["sent_at"] = time.monotonic()
                self.log(
                    "touch_retry",
                    battle=self.battle_number,
                    tick=self.pending["tick"],
                    slot=self.pending["slot"],
                    position=self.pending["position"],
                )
                return True
            self.log(
                "touch_not_confirmed",
                battle=self.battle_number,
                tick=self.pending["tick"],
                slot=self.pending["slot"],
            )
            self.pending = None
            return False
        return True

    def _reveal_enemy(self, state: dict[str, Any]) -> None:
        mapping = self.policy.expert_card_id_to_token
        for entity in state.get("entities", []):
            if int(entity.get("side", -1)) == self.local_side:
                continue
            token = mapping.get(int(entity.get("card_id", -1)))
            if token and token not in self.policy.expert_revealed_enemy_tokens:
                self.policy.expert_revealed_enemy_tokens.append(token)

    def _step(self, entity: dict[str, Any], private: dict[str, Any]) -> None:
        entity_tick = int(entity["game_tick"])
        if entity_tick <= self.last_tick:
            return
        if not self._select_local_side(private):
            self.last_tick = entity_tick
            return
        state = _native_state(entity, private)
        players = {int(item["side"]): item for item in state["players"]}
        player = players[self.local_side]
        if not self._bind_visible_deck(player):
            self.last_tick = entity_tick
            return
        hand = [int(value) for value in player["hand_deck_indices"]]
        self.live_player = {
            "elixir_raw": int(player["elixir_raw"]),
            "elixir": round(int(player["elixir_raw"]) / 10_000.0, 2),
            "hand_deck_indices": hand,
            "next_deck_index": int(player["next_deck_index"]),
            "refill_timer": int(player.get("refill_timer", 0)),
        }
        now = time.monotonic()
        pending = self._update_pending(hand, now)
        self._reveal_enemy(state)
        self.policy.state = state
        card_mask, positions = _position_masks(
            self.decks,
            self.local_side,
            hand,
            int(player["elixir_raw"]),
            state,
        )
        if pending:
            card_mask[:] = False
            positions[:] = False
        card, position, meta = HumanVsAiGui._sample_expert(
            self.policy,
            visible_hand=hand,
            card_mask=card_mask,
            position_masks=positions,
        )
        self.last_tick = entity_tick
        if card == 0:
            return
        slot = int(card) - 1
        deck_index = hand[slot]
        if deck_index not in range(8):
            self.log("invalid_model_slot", tick=entity_tick, slot=slot, hand=hand)
            return
        self._touch(slot, int(position))
        if not self.args.dry_run:
            self.pending = {
                "tick": entity_tick,
                "slot": slot,
                "deck_index": deck_index,
                "position": int(position),
                "hand_before": list(hand),
                "sent_at": time.monotonic(),
                "retried": False,
            }
        self.last_action = {
            "tick": entity_tick,
            "slot": slot,
            "deck_index": deck_index,
            "card_id": int(self.decks[self.local_side][deck_index]["card_id"]),
            "position": int(position),
            "play_probability": meta.get("play_probability"),
            "sent": not self.args.dry_run,
        }
        self.log(
            "touch_sent" if not self.args.dry_run else "touch_dry_run",
            battle=self.battle_number,
            tick=entity_tick,
            slot=slot,
            deck_index=deck_index,
            card_id=int(self.decks[self.local_side][deck_index]["card_id"]),
            position=int(position),
            play_probability=meta.get("play_probability"),
        )

    def run(self) -> None:
        self.prepare()
        assert self.entity_stream and self.private_stream
        try:
            while True:
                if self.entity_stream.process.poll() is not None:
                    raise LiveControllerError(
                        "entity sampler exited: " + " | ".join(self.entity_stream.stderr)
                    )
                if self.private_stream.process.poll() is not None:
                    raise LiveControllerError(
                        "private sampler exited: " + " | ".join(self.private_stream.stderr)
                    )
                entity, private = self._fresh_frames()
                active = self._stable_active(entity, private)
                if not self.in_battle:
                    self.active_streak = self.active_streak + 1 if active else 0
                    if self.active_streak >= 3:
                        assert entity is not None and private is not None
                        self._begin_battle(int(entity["game_tick"]), private)
                    elif self.last_status != "waiting":
                        self.last_status = "waiting"
                        self.log("waiting_for_friendly_battle")
                else:
                    if not active:
                        self.inactive_streak += 1
                        if self.inactive_streak >= 100:
                            self._end_battle()
                    else:
                        self.inactive_streak = 0
                        assert entity is not None and private is not None
                        try:
                            self._step(entity, private)
                        except Exception as exc:
                            self.last_tick = int(entity["game_tick"])
                            self.pending = None
                            self.log(
                                "inference_error",
                                battle=self.battle_number,
                                tick=entity.get("game_tick"),
                                error=f"{type(exc).__name__}: {exc}",
                            )
                            # Fail closed: keep observing, do not touch the screen.
                time.sleep(0.01)
                self._publish_status()
        except Exception as exc:
            self.lifecycle = "error"
            self.last_event = "controller_error"
            self.last_message = f"{type(exc).__name__}: {exc}"
            self._publish_status(force=True)
            raise
        finally:
            _stop_remote_helpers(self.cli, self.args.vmindex)
            self.entity_stream.stop()
            self.private_stream.stop()
            if self.lifecycle != "error":
                self.lifecycle = "stopped"
                self.last_message = "接管控制器已停止"
                self._publish_status(force=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--expert-dataset-root", type=Path, default=DEFAULT_EXPERT_DATASET)
    parser.add_argument("--deck", type=Path, default=DEFAULT_DECK)
    parser.add_argument("--entity-helper", type=Path, default=DEFAULT_ENTITY_HELPER)
    parser.add_argument("--private-helper", type=Path, default=DEFAULT_PRIVATE_HELPER)
    parser.add_argument("--mumu-cli", type=Path, default=DEFAULT_MUMU_CLI)
    parser.add_argument("--adb", type=Path, default=DEFAULT_ADB)
    parser.add_argument("--serial", default=DEFAULT_SERIAL)
    parser.add_argument("--vmindex", type=int, default=1)
    parser.add_argument("--local-side", type=int, choices=(0, 1), default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--policy-seed", type=int, default=20260901)
    parser.add_argument("--choice-mode", choices=("sample", "greedy-placement"), default="sample")
    parser.add_argument("--play-rate-scale", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not math.isfinite(args.play_rate_scale) or not 0 < args.play_rate_scale <= 10:
        raise SystemExit("--play-rate-scale must be in (0, 10]")
    _seed_everything(args.policy_seed)
    try:
        MuMuExpertController(args).run()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
