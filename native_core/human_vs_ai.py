"""One real-time human-vs-P010 match on the original native battle core."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any
import uuid

import numpy as np
import torch

from .env import CARD_NAMES, NativeRoyaleEnv
from .gui import CARD_COSTS, NativeCoreGui
from .worker import HeadlessWorkerPool, WorkerConfig
from training.evaluate import load_neural_policy
from training.schema import (
    ActionMaskCache,
    ObservationEncoder,
    build_action_masks,
)


DEFAULT_CHECKPOINT = Path(
    r"D:\AI_data\cr-native-core\selfplay-v0.1\runs"
    r"\selfplay-v0.1-stage-a-20260823T141402Z"
    r"\evaluations\candidates\P010.pt"
)
DEFAULT_REPLAY = Path("examples/eight-card-bootstrap.json")
SESSION_ROOT = Path(r"D:\AI_data\cr-native-core\human-vs-ai")


class HumanVsAiGui(NativeCoreGui):
    HUMAN_SIDE = 0
    AI_SIDE = 1
    TICK_SECONDS = 0.05

    def __init__(
        self,
        root: tk.Tk,
        env: NativeRoyaleEnv,
        replay: Path,
        *,
        checkpoint: Path,
        model: Any,
        model_meta: dict[str, Any],
        device: torch.device,
        policy_seed: int,
        autostart: bool = True,
    ) -> None:
        self.checkpoint = checkpoint.resolve()
        self.model = model
        self.model_meta = model_meta
        self.device = device
        self.policy_seed = int(policy_seed)
        self.autostart = autostart
        self.encoder = ObservationEncoder()
        self.mask_cache = ActionMaskCache()
        self.native_masks: dict[tuple[int, int], list[str]] = {}
        self.ai_hidden = self.model.initial_hidden(1, device=device)
        self.public_actions: dict[int, dict[str, int] | None] = {
            0: None, 1: None,
        }
        self.pending_human_action: dict[str, int] | None = None
        self.running = False
        self.loop_generation = 0
        self.next_deadline = 0.0
        self.terminal_announced = False
        self.session_path: Path | None = None
        self.ai_last_value = 0.0
        self.ai_last_action = "WAIT"
        self.human_plays = 0
        self.ai_plays = 0
        self.unexpected_rejections = 0
        super().__init__(root, env, replay)
        self.selected_side.set(self.HUMAN_SIDE)
        self.auto.set(False)
        root.title("Clash Royale · 人类（蓝）vs P010（红）")
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._lock_debug_controls()

    def _lock_debug_controls(self) -> None:
        stack = list(self.root.winfo_children())
        while stack:
            widget = stack.pop()
            stack.extend(widget.winfo_children())
            try:
                text = str(widget.cget("text"))
            except tk.TclError:
                continue
            if (
                text.startswith("+")
                or text in {"自动", "蓝方", "红方"}
            ):
                try:
                    widget.state(["disabled"])
                except (AttributeError, tk.TclError):
                    widget.configure(state="disabled")

    def _change_side(self) -> None:
        self.selected_side.set(self.HUMAN_SIDE)
        self.selected_deck.set(-1)
        self.deployment_mask = None
        self.raw_deployment_mask = None
        self.last_deploy_marker = None
        self.render()

    def _reset_native_battle(self) -> None:
        self.loop_generation += 1
        generation = self.loop_generation
        self.running = False
        super()._reset_native_battle()
        self.selected_side.set(self.HUMAN_SIDE)
        self.ai_hidden = self.model.initial_hidden(1, device=self.device)
        self.mask_cache = ActionMaskCache()
        self.native_masks = {}
        self.public_actions = {0: None, 1: None}
        self.pending_human_action = None
        self.terminal_announced = False
        self.session_path = None
        self.ai_last_value = 0.0
        self.ai_last_action = "WAIT"
        self.human_plays = 0
        self.ai_plays = 0
        self.unexpected_rejections = 0
        self.action_log = []
        self.running = self.autostart
        self.next_deadline = time.perf_counter() + self.TICK_SECONDS
        if self.autostart:
            self.root.after(1, lambda: self._game_loop(generation))

    def _refresh_cards(self) -> None:
        if self.state is None:
            return
        player = next(
            item for item in self.state.get("players", [])
            if int(item["side"]) == self.HUMAN_SIDE
        )
        available = set(int(value) for value in player["hand_deck_indices"])
        elixir = int(player["elixir"])
        commands_allowed = bool(
            self.state.get("episode", {}).get("commands_allowed", True)
        )
        for index, button in enumerate(self.card_buttons):
            card_id = int(self.env.decks[self.HUMAN_SIDE][index]["card_id"])
            name = CARD_NAMES.get(card_id, str(card_id))
            cost = CARD_COSTS[card_id]
            playable = (
                index in available
                and elixir >= cost
                and commands_allowed
                and self.running
            )
            marker = "●" if index in available else "○"
            button.configure(
                text=f"{marker} {index}: {name} ({cost})",
                state="normal" if playable else "disabled",
            )

    def deploy(self, event: tk.Event[tk.Canvas]) -> None:
        if not self.running or self.state is None:
            self.status.set("本局尚未运行或已经结束")
            return
        deck_index = int(self.selected_deck.get())
        if deck_index < 0:
            self.status.set("请先选择一张当前可用手牌")
            return
        player = next(
            item for item in self.state["players"]
            if int(item["side"]) == self.HUMAN_SIDE
        )
        card_id = int(self.env.decks[self.HUMAN_SIDE][deck_index]["card_id"])
        if (
            deck_index not in player["hand_deck_indices"]
            or int(player["elixir"]) < CARD_COSTS[card_id]
            or not bool(self.state["episode"].get("commands_allowed", True))
        ):
            self.status.set("该牌当前不可用：手牌、圣水或原生命令门不满足")
            self.render()
            return
        left, top, arena_width, arena_height = self._arena_geometry()
        if not (
            left <= event.x < left + arena_width
            and top <= event.y < top + arena_height
        ):
            self.status.set("点击位置在竞技场外")
            return
        x = max(0, min(
            self.ARENA_WIDTH - 1,
            round((event.x - left) / arena_width * self.ARENA_WIDTH),
        ))
        y = max(0, min(
            self.ARENA_HEIGHT - 1,
            round((1 - (event.y - top) / arena_height) * self.ARENA_HEIGHT),
        ))
        row, column = min(31, y // 1000), min(17, x // 1000)
        if (
            self.deployment_mask is not None
            and self.deployment_mask[row][column] != "1"
        ):
            self.last_deploy_marker = (x, y, False)
            self.render()
            self.status.set("该地块不在当前最终部署 Mask 内")
            return
        try:
            probe = self.env.probe(
                side=self.HUMAN_SIDE,
                deck_index=deck_index,
                x=x,
                y=y,
            )
            if not bool(probe.get("placement_valid", False)):
                self.last_deploy_marker = (x, y, False)
                self.render()
                self.status.set(
                    "落点被 libg 拒绝："
                    f"code={probe.get('result_code')} "
                    f"{probe.get('placement_reason', probe.get('reason', ''))}"
                )
                return
            self.pending_human_action = {
                "side": self.HUMAN_SIDE,
                "deck_index": deck_index,
                "x": x,
                "y": y,
                "card_id": card_id,
            }
            self.last_deploy_marker = (x, y, True)
            self.selected_deck.set(-1)
            self.deployment_mask = None
            self.raw_deployment_mask = None
            self.render()
            self.status.set(
                f"已提交 {CARD_NAMES[card_id]}，将在下一个原生 Tick 执行"
            )
        except Exception as error:
            self._stop_with_error(error)

    @staticmethod
    def _canonical_positions(mask: np.ndarray, side: int) -> np.ndarray:
        values = mask.reshape(4, 32, 18)
        if side == 1:
            values = values[:, ::-1, ::-1]
        return np.ascontiguousarray(values.reshape(4, 32 * 18))

    @staticmethod
    def _absolute_cell(position: int, side: int) -> tuple[int, int]:
        row, column = divmod(position, 18)
        if side == 1:
            row, column = 31 - row, 17 - column
        return column * 1000 + 500, row * 1000 + 500

    def _prepare_ai_masks(self) -> None:
        assert self.state is not None
        player = next(
            item for item in self.state["players"]
            if int(item["side"]) == self.AI_SIDE
        )
        for raw_index in player["hand_deck_indices"]:
            deck_index = int(raw_index)
            key = (self.AI_SIDE, deck_index)
            if deck_index < 0 or key in self.native_masks:
                continue
            result = self.env.probe_grid(
                side=self.AI_SIDE, deck_index=deck_index
            )
            self.native_masks[key] = [str(row) for row in result["rows"]]

    def _sample_ai(self) -> tuple[dict[str, int] | None, dict[str, Any]]:
        assert self.state is not None
        self._prepare_ai_masks()
        grid, scalars = self.encoder.encode(
            self.state,
            side=self.AI_SIDE,
            public_actions=self.public_actions,
        )
        privileged = self.encoder.privileged(self.state, side=self.AI_SIDE)
        card_mask, position_masks, hand = build_action_masks(
            self.state,
            side=self.AI_SIDE,
            native_masks=self.native_masks,
            decks=self.env.decks,
            cache=self.mask_cache,
        )
        position_masks = self._canonical_positions(
            position_masks, self.AI_SIDE
        )
        sample = self.model.sample(
            torch.from_numpy(grid).unsqueeze(0).to(self.device),
            torch.from_numpy(scalars).unsqueeze(0).to(self.device),
            torch.from_numpy(privileged).unsqueeze(0).to(self.device),
            torch.from_numpy(card_mask).unsqueeze(0).to(self.device),
            torch.from_numpy(position_masks).unsqueeze(0).to(self.device),
            self.ai_hidden,
        )
        self.ai_hidden = sample.hidden
        self.ai_last_value = sample.value
        if sample.card == 0:
            self.ai_last_action = "WAIT"
            return None, {
                "card": 0,
                "position": 0,
                "log_probability": sample.log_probability,
                "value": sample.value,
            }
        deck_index = int(hand[sample.card - 1])
        card_id = int(self.env.decks[self.AI_SIDE][deck_index]["card_id"])
        x, y = self._absolute_cell(sample.position, self.AI_SIDE)
        self.ai_last_action = CARD_NAMES.get(card_id, str(card_id))
        return {
            "side": self.AI_SIDE,
            "deck_index": deck_index,
            "x": x,
            "y": y,
            "card_id": card_id,
        }, {
            "card": sample.card,
            "position": sample.position,
            "log_probability": sample.log_probability,
            "value": sample.value,
        }

    def _advance_one_tick(self) -> bool:
        assert self.state is not None
        tick_before = int(self.state["tick"])
        human_action = self.pending_human_action
        self.pending_human_action = None
        ai_action, ai_sample = self._sample_ai()
        actions = [
            {
                key: value for key, value in action.items()
                if key in {"side", "deck_index", "x", "y"}
            }
            for action in (human_action, ai_action)
            if action is not None
        ]
        transition = self.env.joint_transition(actions, steps=1)
        native = transition["joint_action"]
        accepted: set[int] = set()
        results_by_side: dict[int, dict[str, Any]] = {}
        for item in native.get("actions", []):
            side = int(item["side"])
            result = dict(item["result"])
            results_by_side[side] = result
            if bool(result.get("accepted", False)):
                accepted.add(side)
            else:
                self.unexpected_rejections += 1
        if ai_action is not None and self.AI_SIDE not in accepted:
            raise RuntimeError(
                "P010 selected an action rejected by libg: "
                + json.dumps(results_by_side.get(self.AI_SIDE), ensure_ascii=False)
            )
        if human_action is not None and self.HUMAN_SIDE not in accepted:
            self.status.set(
                "你提交的动作被原生核心拒绝："
                + json.dumps(results_by_side.get(self.HUMAN_SIDE), ensure_ascii=False)
            )
        next_public: dict[int, dict[str, int] | None] = {0: None, 1: None}
        if human_action is not None and self.HUMAN_SIDE in accepted:
            self.human_plays += 1
            next_public[self.HUMAN_SIDE] = {
                "card_id": int(human_action["card_id"]),
                "x": int(human_action["x"]),
                "y": int(human_action["y"]),
            }
        if ai_action is not None and self.AI_SIDE in accepted:
            self.ai_plays += 1
            next_public[self.AI_SIDE] = {
                "card_id": int(ai_action["card_id"]),
                "x": int(ai_action["x"]),
                "y": int(ai_action["y"]),
            }
        self.public_actions = next_public
        episode = transition["step"]["episode"]
        done = bool(episode.get("terminated") or episode.get("truncated"))
        self.action_log.append({
            "tick": tick_before,
            "human_action": human_action,
            "ai_action": ai_action,
            "ai_sample": ai_sample,
            "native": native,
            "episode": episode if done else None,
        })
        if done:
            terminal_state = deepcopy(self.state)
            terminal_state["tick"] = int(episode.get("terminal_tick", tick_before))
            terminal_state["elapsed_seconds"] = round(
                int(terminal_state["tick"]) * 0.05, 3
            )
            terminal_state["episode"] = episode
            self.state = terminal_state
        else:
            self.state = transition["state"]
        self.render()
        return done

    def _game_loop(self, generation: int) -> None:
        if generation != self.loop_generation or not self.running:
            return
        try:
            done = self._advance_one_tick()
            if done:
                self.running = False
                self._announce_terminal()
                return
        except Exception as error:
            self._stop_with_error(error)
            return
        self.next_deadline += self.TICK_SECONDS
        now = time.perf_counter()
        if self.next_deadline < now:
            self.next_deadline = now
        delay_ms = max(1, round((self.next_deadline - now) * 1000))
        self.root.after(delay_ms, lambda: self._game_loop(generation))

    def render(self) -> None:
        super().render()
        if self.state is None or self.state.get("episode", {}).get("terminated"):
            return
        queued = (
            CARD_NAMES.get(int(self.pending_human_action["card_id"]), "?")
            if self.pending_human_action else "无"
        )
        self.status.set(
            self.status.get()
            + f"  |  你=蓝  P010=红"
            + f"  |  AI上次={self.ai_last_action} V={self.ai_last_value:+.3f}"
            + f"  |  待执行={queued}"
        )

    def _session_payload(self, *, partial: bool) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "native_human_vs_ai_match",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "partial": partial,
            "human_side": self.HUMAN_SIDE,
            "ai_side": self.AI_SIDE,
            "checkpoint": str(self.checkpoint),
            "model": self.model_meta,
            "policy_seed": self.policy_seed,
            "battle_seed": int(self.seed.get()),
            "native_tick_hz": 20,
            "policy_updated": False,
            "human_plays": self.human_plays,
            "ai_plays": self.ai_plays,
            "unexpected_rejections": self.unexpected_rejections,
            "state": self.state,
            "episode": (self.state or {}).get("episode", {}),
            "actions": self.action_log,
        }

    def save_session(self, *, partial: bool) -> Path:
        if self.session_path is not None and not partial:
            return self.session_path
        SESSION_ROOT.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        target = SESSION_ROOT / f"human-vs-p010-{stamp}-{uuid.uuid4().hex[:8]}.json"
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(
            json.dumps(
                self._session_payload(partial=partial),
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
        self.session_path = target
        return target

    def _announce_terminal(self) -> None:
        if self.terminal_announced:
            return
        self.terminal_announced = True
        path = self.save_session(partial=False)
        episode = (self.state or {}).get("episode", {})
        winner = episode.get("winner")
        outcome = "平局" if winner is None else (
            "你获胜" if int(winner) == self.HUMAN_SIDE else "P010 获胜"
        )
        messagebox.showinfo(
            "人机对战结束",
            f"{outcome}\n皇冠 {episode.get('crowns', [0, 0])}\n\n已保存：{path}",
        )

    def _stop_with_error(self, error: Exception) -> None:
        self.running = False
        self.loop_generation += 1
        self._error(error)

    def close(self) -> None:
        self.running = False
        self.loop_generation += 1
        if self.state is not None and self.session_path is None:
            try:
                self.save_session(partial=not bool(
                    self.state.get("episode", {}).get("terminated")
                ))
            except OSError:
                pass
        self.env.close()
        self.root.destroy()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=37031)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--policy-seed", type=int, default=20260824)
    parser.add_argument("--battle-seed", type=int, default=20260824)
    parser.add_argument("--keep-worker", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"P010 checkpoint not found: {args.checkpoint}")
    _seed_everything(args.policy_seed)
    model, model_meta = load_neural_policy(
        args.checkpoint.resolve(),
        device=device,
        cuda_graph=device.type == "cuda",
    )
    root = tk.Tk()
    env = NativeRoyaleEnv(host=args.host, port=args.port, timeout=30)
    gui = HumanVsAiGui(
        root,
        env,
        args.replay.resolve(),
        checkpoint=args.checkpoint,
        model=model,
        model_meta=model_meta,
        device=device,
        policy_seed=args.policy_seed,
        autostart=not args.smoke,
    )
    gui.seed.set(args.battle_seed)
    try:
        if args.smoke:
            root.withdraw()
            gui._reset_native_battle()
            initial_tick = int(gui.state["tick"])
            for _ in range(5):
                if gui._advance_one_tick():
                    break
            result = {
                "ok": True,
                "initial_tick": initial_tick,
                "final_tick": int(gui.state["tick"]),
                "ai_hidden_nonzero": bool(
                    torch.count_nonzero(gui.ai_hidden[0]).item()
                ),
                "unexpected_rejections": gui.unexpected_rejections,
                "model_digest": model_meta["model_digest"],
            }
            if (
                result["initial_tick"] != 100
                or result["final_tick"] != 105
                or not result["ai_hidden_nonzero"]
                or result["unexpected_rejections"] != 0
            ):
                raise RuntimeError(f"human-vs-AI smoke failed: {result}")
            print(json.dumps(result, ensure_ascii=False))
            gui.close()
            return 0
        root.mainloop()
        return 0
    finally:
        if not args.keep_worker:
            try:
                HeadlessWorkerPool(
                    WorkerConfig(service_base_port=args.port)
                ).stop(1, keep_vm=False)
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
