"""Interactive logic-acceptance GUI for the original native ``libg`` core."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk
import uuid

try:
    from .env import CARD_NAMES, NativeHostError, NativeRoyaleEnv
    from .card_catalog import catalog as live_card_catalog
except ImportError:  # direct ``python native_core/gui.py`` execution
    from env import CARD_NAMES, NativeHostError, NativeRoyaleEnv
    from card_catalog import catalog as live_card_catalog

from training.schema import deployment_mask


CARD_COSTS = {
    card_id: int(value["elixir"])
    for card_id, value in live_card_catalog().items()
    if value.get("elixir") is not None
}

TERMINAL_RECYCLE_MARKER = "native terminal is latched"


def requires_host_recycle(error: BaseException) -> bool:
    return TERMINAL_RECYCLE_MARKER in str(error)


class NativeCoreGui:
    WIDTH = 450
    HEIGHT = 800
    ARENA_WIDTH = 18000
    ARENA_HEIGHT = 32000
    BRIDGE_CENTERS_X = (3500, 14500)
    RIVER_MIN_Y = 15000
    RIVER_MAX_Y = 17000
    POCKET_DEPTH_CELLS = 5
    LANE_SPLIT_COLUMN = 9

    def __init__(self, root: tk.Tk, env: NativeRoyaleEnv, replay: Path) -> None:
        self.root = root
        self.env = env
        self.replay = replay
        self.state: dict[str, object] | None = None
        self.selected_side = tk.IntVar(value=0)
        self.selected_deck = tk.IntVar(value=-1)
        self.auto = tk.BooleanVar(value=False)
        self.auto_steps = tk.IntVar(value=5)
        self.seed = tk.IntVar(value=1)
        self.show_raw_mask = tk.BooleanVar(value=False)
        self.show_targets = tk.BooleanVar(value=True)
        self.show_paths = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="连接原生服务…")
        self.coordinates = tk.StringVar(value="坐标 —")
        self.selection_detail = tk.StringVar(value="选择实体查看原生字段")
        self.card_buttons: list[ttk.Button] = []
        self.last_deploy_marker: tuple[int, int, bool] | None = None
        self.deployment_mask: list[str] | None = None
        self.raw_deployment_mask: list[str] | None = None
        self.action_log: list[dict[str, object]] = []
        self.replay_template = self.env.read_replay(replay)

        root.title("Clash Royale 原生核心 · 游戏逻辑验收")
        root.geometry("1180x920")
        root.minsize(960, 760)
        root.protocol("WM_DELETE_WINDOW", root.destroy)
        controls = ttk.Frame(root, padding=8)
        controls.pack(fill="x")
        ttk.Button(controls, text="重置", command=self.reset).pack(side="left")
        for count in (1, 20, 200):
            ttk.Button(
                controls,
                text=f"+{count} tick",
                command=lambda value=count: self.step(value),
            ).pack(side="left", padx=(6, 0))
        ttk.Checkbutton(
            controls, text="自动", variable=self.auto, command=self._auto_loop
        ).pack(side="left", padx=(12, 0))
        ttk.Spinbox(
            controls, from_=1, to=1000, textvariable=self.auto_steps, width=6
        ).pack(side="left", padx=(4, 0))
        ttk.Label(controls, text="种子").pack(side="left", padx=(12, 3))
        ttk.Spinbox(
            controls, from_=1, to=2147483647, textvariable=self.seed, width=9
        ).pack(side="left")
        ttk.Button(controls, text="导出快照", command=self.export_snapshot).pack(
            side="left", padx=(8, 0)
        )
        ttk.Radiobutton(
            controls, text="蓝方", value=0, variable=self.selected_side,
            command=self._change_side,
        ).pack(side="right")
        ttk.Radiobutton(
            controls, text="红方", value=1, variable=self.selected_side,
            command=self._change_side,
        ).pack(side="right")

        body = ttk.Frame(root, padding=(8, 0, 8, 0))
        body.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(
            body,
            width=self.WIDTH,
            height=self.HEIGHT,
            background="#102031",
            highlightthickness=0,
        )
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.deploy)
        self.canvas.bind("<Configure>", lambda _event: self.render())
        self.canvas.bind("<Motion>", self._track_coordinates)

        side_panel = ttk.Frame(body, padding=(10, 0, 0, 0))
        side_panel.pack(side="right", fill="y")
        ttk.Label(side_panel, text="手牌 / 卡组").pack(anchor="w")
        for deck_index in range(8):
            button = ttk.Button(
                side_panel,
                text=str(deck_index),
                command=lambda value=deck_index: self._select_card(value),
                width=20,
            )
            button.pack(fill="x", pady=(5, 0))
            self.card_buttons.append(button)
        ttk.Label(
            side_panel,
            text="选卡后点击战场部署\n坐标与合法性由 libg 判定",
            foreground="#666666",
            justify="left",
        ).pack(anchor="w", pady=(16, 0))

        ttk.Separator(side_panel).pack(fill="x", pady=12)
        ttk.Checkbutton(
            side_panel, text="显示原始 libg 掩码", variable=self.show_raw_mask,
            command=self._refresh_selected_mask,
        ).pack(anchor="w")
        ttk.Checkbutton(
            side_panel, text="绘制当前目标", variable=self.show_targets,
            command=self.render,
        ).pack(anchor="w")
        ttk.Checkbutton(
            side_panel, text="绘制原生路径节点", variable=self.show_paths,
            command=self.render,
        ).pack(anchor="w")
        ttk.Label(side_panel, textvariable=self.coordinates).pack(anchor="w", pady=(6, 0))

        ttk.Label(side_panel, text="原生实体").pack(anchor="w", pady=(12, 3))
        self.entity_tree = ttk.Treeview(
            side_panel,
            columns=("side", "card", "hp", "xy", "state"),
            show="headings", height=13,
        )
        for column, title, width in (
            ("side", "方", 34), ("card", "实体", 92), ("hp", "HP", 78),
            ("xy", "坐标", 88), ("state", "状态", 42),
        ):
            self.entity_tree.heading(column, text=title)
            self.entity_tree.column(column, width=width, stretch=False, anchor="center")
        self.entity_tree.pack(fill="both", expand=True)
        self.entity_tree.bind("<<TreeviewSelect>>", self._entity_selected)
        ttk.Label(
            side_panel, textvariable=self.selection_detail, wraplength=340,
            justify="left",
        ).pack(fill="x", pady=(6, 0))

        ttk.Label(root, textvariable=self.status, padding=8).pack(fill="x")
        root.after(50, self.attach)

    def attach(self) -> None:
        try:
            self._reset_native_battle()
        except Exception as error:
            self._error(error)

    def reset(self) -> None:
        try:
            self.auto.set(False)
            self._reset_native_battle()
        except Exception as error:
            self._error(error)

    def _reset_native_battle(self) -> None:
        replay = deepcopy(self.replay_template)
        replay["rndSeed"] = int(self.seed.get())
        self.status.set("原生 BattleGameState 4→4 重置中…")
        self.root.update_idletasks()
        self.state = self.env.reset(replay, warmup_steps=100)
        self.selected_deck.set(-1)
        self.deployment_mask = None
        self.raw_deployment_mask = None
        self.last_deploy_marker = None
        self.action_log = []
        self.render()

    def step(self, count: int) -> None:
        try:
            self.env.step(count)
            self.state = self.env.observe()
            self.render()
        except Exception as error:
            self.auto.set(False)
            self._error(error)

    def deploy(self, event: tk.Event[tk.Canvas]) -> None:
        deck_index = self.selected_deck.get()
        if deck_index < 0:
            self.status.set("请先选择一张当前手牌")
            return
        geometry = self._arena_geometry()
        left, top, arena_width, arena_height = geometry
        if not (
            left <= event.x < left + arena_width
            and top <= event.y < top + arena_height
        ):
            self.status.set("点击位置在竞技场外")
            return
        x = max(
            0,
            min(
                self.ARENA_WIDTH - 1,
                round((event.x - left) / arena_width * self.ARENA_WIDTH),
            ),
        )
        y = max(
            0,
            min(
                self.ARENA_HEIGHT - 1,
                round(
                    (1 - (event.y - top) / arena_height)
                    * self.ARENA_HEIGHT
                ),
            ),
        )
        try:
            column = min(17, x // 1000)
            row = min(31, y // 1000)
            if (
                self.deployment_mask is not None
                and self.deployment_mask[row][column] != "1"
            ):
                self.last_deploy_marker = (x, y, False)
                self.render()
                self.status.set(
                    f"该地块已从当前部署层移除：({x}, {y})"
                )
                return
            probe = self.env.probe(
                side=self.selected_side.get(),
                deck_index=deck_index,
                x=x,
                y=y,
            )
            placement_valid = bool(probe.get("placement_valid", False))
            self.last_deploy_marker = (x, y, placement_valid)
            if not placement_valid:
                self.render()
                self.status.set(
                    f"落点被 libg 拒绝：({x}, {y}) "
                    f"code={probe['result_code']} "
                    f"{probe.get('placement_reason', probe.get('reason', ''))}"
                )
                return
            result = self.env.act(
                side=self.selected_side.get(), deck_index=deck_index, x=x, y=y
            )
            if not result["accepted"]:
                self.render()
                self.status.set(
                    f"原生拒绝：code={result['result_code']} "
                    f"{result.get('reason', '')}"
                )
                return
            self.action_log.append(
                {
                    "tick": int((self.state or {}).get("tick", -1)),
                    "side": self.selected_side.get(),
                    "deck_index": deck_index,
                    "card_id": int(self.env.decks[self.selected_side.get()][deck_index]["card_id"]),
                    "x": x,
                    "y": y,
                    "native_result": result,
                }
            )
            self.env.step(1)
            self.state = self.env.observe()
            self.selected_deck.set(-1)
            self.deployment_mask = None
            self.render()
        except Exception as error:
            self._error(error)

    def _auto_loop(self) -> None:
        if not self.auto.get():
            return
        self.step(max(1, self.auto_steps.get()))
        if self.auto.get():
            self.root.after(30, self._auto_loop)

    def _refresh_cards(self) -> None:
        if self.state is None:
            return
        side = self.selected_side.get()
        players = self.state.get("players", [])
        player = next((item for item in players if item["side"] == side), None)
        available = set(player["hand_deck_indices"]) if player else set()
        deck = self.env.decks[side]
        for index, button in enumerate(self.card_buttons):
            card_id = deck[index]["card_id"]
            name = CARD_NAMES.get(card_id, str(card_id))
            cost = CARD_COSTS.get(card_id, "?")
            marker = "●" if index in available else "○"
            button.configure(
                text=f"{marker} {index}: {name} ({cost})",
                state="normal" if index in available else "disabled",
            )

    def _change_side(self) -> None:
        self.selected_deck.set(-1)
        self.deployment_mask = None
        self.raw_deployment_mask = None
        self.last_deploy_marker = None
        self.render()

    def _select_card(self, deck_index: int) -> None:
        self.selected_deck.set(deck_index)
        try:
            grid = self.env.probe_grid(
                side=self.selected_side.get(), deck_index=deck_index
            )
            rows = grid.get("rows", [])
            native_mask = (
                [str(row) for row in rows]
                if len(rows) == 32
                else None
            )
            self.raw_deployment_mask = native_mask
            card_id = self.env.decks[
                self.selected_side.get()
            ][deck_index]["card_id"]
            self.deployment_mask = self._current_deployment_mask(native_mask, card_id)
            self.render()
            name = CARD_NAMES.get(
                card_id,
                str(deck_index),
            )
            valid_cells = (
                sum(row.count("1") for row in self.deployment_mask)
                if self.deployment_mask is not None
                else grid.get("valid_cells", "?")
            )
            self.status.set(
                f"已选择 {name}  |  {'原始 libg' if self.show_raw_mask.get() else '训练最终'}"
                f"可部署格 {valid_cells}"
            )
        except Exception as error:
            self.deployment_mask = None
            self._error(error)

    def _refresh_selected_mask(self) -> None:
        deck_index = self.selected_deck.get()
        if deck_index >= 0:
            self._select_card(deck_index)
        else:
            self.render()

    def _track_coordinates(self, event: tk.Event[tk.Canvas]) -> None:
        left, top, arena_width, arena_height = self._arena_geometry()
        if not (
            left <= event.x < left + arena_width
            and top <= event.y < top + arena_height
        ):
            self.coordinates.set("坐标 —")
            return
        x = max(0, min(17999, int((event.x - left) / arena_width * 18000)))
        y = max(0, min(31999, int((1 - (event.y - top) / arena_height) * 32000)))
        self.coordinates.set(f"坐标 ({x}, {y})  格 ({x // 1000}, {y // 1000})")

    def _entity_selected(self, _event: object = None) -> None:
        if self.state is None:
            return
        selected = self.entity_tree.selection()
        if not selected:
            return
        entity = next(
            (item for item in self.state.get("entities", []) if str(item["id"]) == selected[0]),
            None,
        )
        if entity is None:
            return
        fields = (
            "id", "card_id", "kind", "category", "generation_key", "side",
            "x", "y", "hp", "max_hp", "behavior_state", "target",
            "attack_progress_ms", "attack_load_timer_ms", "event_timer_ms",
            "movement_direction_x", "movement_direction_y", "collision_count",
            "collision_accumulator_x", "collision_accumulator_y",
            "avoidance_offset", "path_segment_direction_x",
            "path_segment_direction_y", "path_node_consumed",
        )
        self.selection_detail.set("  ".join(
            f"{field}={entity.get(field)}" for field in fields
            if entity.get(field) is not None
        ))

    def export_snapshot(self) -> None:
        if self.state is None:
            return
        root = Path(r"D:\AI_data\cr-native-core\gui-sessions")
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        target = root / f"logic-{stamp}-{uuid.uuid4().hex[:8]}.json"
        payload = {
            "schema_version": 1,
            "kind": "native_logic_gui_snapshot",
            "seed": int(self.seed.get()),
            "state": self.state,
            "actions": self.action_log,
        }
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
        self.status.set(f"已导出 {target}")

    def _current_deployment_mask(
        self, native_mask: list[str] | None, card_id: int
    ) -> list[str] | None:
        """Return either the raw validator or the exact training action mask."""
        if native_mask is None:
            return None
        if self.show_raw_mask.get():
            return list(native_mask)
        return deployment_mask(
            native_mask,
            self.state or {},
            side=self.selected_side.get(),
            card_id=card_id,
        )

    @classmethod
    def _crown_tower_role(cls, entity: dict[str, object]) -> str | None:
        """Identify a crown tower independently of its mutable native kind."""
        if entity.get("card_id") != -1:
            return None
        category = int(entity.get("category", -1))
        category_slot = category - 5_000_000
        if 0 <= category_slot <= 5:
            return "king" if category_slot % 3 == 0 else "princess"
        x = int(entity.get("x", -1))
        if x == cls.ARENA_WIDTH // 2:
            return "king"
        if 0 <= x < cls.ARENA_WIDTH:
            return "princess"
        return None

    def _arena_geometry(self) -> tuple[float, float, float, float]:
        """Fit the 18x32 logical arena without stretching or shifting it."""
        canvas_width = max(1.0, float(self.canvas.winfo_width()))
        canvas_height = max(1.0, float(self.canvas.winfo_height()))
        arena_width = min(
            canvas_width,
            canvas_height * self.ARENA_WIDTH / self.ARENA_HEIGHT,
        )
        arena_height = arena_width * self.ARENA_HEIGHT / self.ARENA_WIDTH
        return (
            (canvas_width - arena_width) / 2.0,
            (canvas_height - arena_height) / 2.0,
            arena_width,
            arena_height,
        )

    @staticmethod
    def _match_clock_info(tick: int, terminated: bool) -> tuple[str, str, int]:
        """Frozen standard-1v1 schedule, independently measured in libg."""
        if tick < 3600:
            seconds = math.ceil(max(0, 3600 - tick) / 20)
            phase = "常规时间"
        elif tick < 6000:
            seconds = math.ceil(max(0, 6000 - tick) / 20)
            phase = "加时"
        else:
            seconds = 0
            phase = "终局" if terminated else "决胜结算"
        multiplier = 1 if tick < 2400 else 2 if tick < 4800 else 3
        return f"{seconds // 60}:{seconds % 60:02d}", phase, multiplier

    @classmethod
    def _native_to_canvas(
        cls,
        x: int,
        y: int,
        geometry: tuple[float, float, float, float],
    ) -> tuple[float, float]:
        left, top, arena_width, arena_height = geometry
        return (
            left + x / cls.ARENA_WIDTH * arena_width,
            top + (1 - y / cls.ARENA_HEIGHT) * arena_height,
        )

    def render(self) -> None:
        if self.state is None:
            return
        canvas = self.canvas
        canvas.delete("all")
        geometry = self._arena_geometry()
        left, top, arena_width, arena_height = geometry
        right = left + arena_width
        bottom = top + arena_height
        cell_width = arena_width / 18
        cell_height = arena_height / 32
        canvas.create_rectangle(
            left, top, right, bottom,
            fill="#102234", outline="#36516a", width=2,
        )

        # Deployment legality is an overlay, not arena geometry. Keeping the
        # complete base grid prevents a side/card mask from warping the board.
        if self.deployment_mask is not None:
            for row, mask in enumerate(self.deployment_mask):
                for column in range(18):
                    if len(mask) != 18 or mask[column] != "1":
                        continue
                    cell_left = left + column * cell_width
                    cell_right = cell_left + cell_width
                    cell_top = top + (31 - row) * cell_height
                    cell_bottom = cell_top + cell_height
                    canvas.create_rectangle(
                        cell_left, cell_top, cell_right, cell_bottom,
                        outline="", fill="#142c40",
                    )
        for column in range(19):
            x = left + column * cell_width
            canvas.create_line(x, top, x, bottom, fill="#274158")
        for row in range(33):
            y = top + row * cell_height
            canvas.create_line(left, y, right, y, fill="#274158")

        river_top = self._native_to_canvas(
            0, self.RIVER_MAX_Y, geometry
        )[1]
        river_bottom = self._native_to_canvas(
            0, self.RIVER_MIN_Y, geometry
        )[1]
        canvas.create_rectangle(
            left, river_top, right, river_bottom,
            fill="#174b68", outline="#277698",
        )
        for native_bridge_x in self.BRIDGE_CENTERS_X:
            bridge_x = self._native_to_canvas(
                native_bridge_x, 16000, geometry
            )[0]
            canvas.create_rectangle(
                bridge_x - cell_width,
                river_top,
                bridge_x + cell_width,
                river_bottom,
                fill="#826d4f", outline="#b89a70",
            )

        entities_by_id = {
            str(entity.get("id")): entity for entity in self.state["entities"]
        }
        if self.show_targets.get():
            for entity in self.state["entities"]:
                target = entities_by_id.get(str(entity.get("target")))
                if target is None:
                    continue
                source_x, source_y = self._native_to_canvas(
                    int(entity["x"]), int(entity["y"]), geometry
                )
                target_x, target_y = self._native_to_canvas(
                    int(target["x"]), int(target["y"]), geometry
                )
                canvas.create_line(
                    source_x, source_y, target_x, target_y,
                    fill="#ffd34d", width=1, dash=(4, 3), arrow=tk.LAST,
                )
        if self.show_paths.get():
            for entity in self.state["entities"]:
                nodes = entity.get("path_nodes")
                if not isinstance(nodes, list) or not nodes:
                    continue
                points: list[float] = []
                source = self._native_to_canvas(
                    int(entity["x"]), int(entity["y"]), geometry
                )
                points.extend(source)
                for node in nodes:
                    if isinstance(node, dict) and "x" in node and "y" in node:
                        points.extend(self._native_to_canvas(
                            int(node["x"]), int(node["y"]), geometry
                        ))
                if len(points) >= 4:
                    canvas.create_line(*points, fill="#73e0ff", width=2)

        for entity in self.state["entities"]:
            x, y = self._native_to_canvas(
                entity["x"], entity["y"], geometry
            )
            blue = entity["side"] == 0
            color = "#35bff3" if blue else "#ff607d"
            card_id = entity["card_id"]
            tower_role = self._crown_tower_role(entity)
            crown_tower = tower_role is not None
            if crown_tower:
                radius_units = 2000 if tower_role == "king" else 1500
                radius_x = radius_units / 1000 * cell_width
                radius_y = radius_units / 1000 * cell_height
                canvas.create_rectangle(
                    x - radius_x, y - radius_y,
                    x + radius_x, y + radius_y,
                    fill="#19364b", outline=color, width=2,
                )
                canvas.create_oval(
                    x - radius_x, y - radius_y,
                    x + radius_x, y + radius_y,
                    outline="white", dash=(3, 2),
                )
            else:
                canvas.create_oval(
                    x - 10, y - 10, x + 10, y + 10,
                    fill=color, outline="white",
                )
            tower_name = (
                "King Tower" if tower_role == "king" else "Princess Tower"
            )
            label = CARD_NAMES.get(
                card_id,
                tower_name if card_id == -1 else str(card_id),
            )
            label_offset = (
                (2000 if tower_role == "king" else 1500)
                / 1000 * cell_height + 12
                if crown_tower
                else 18
            )
            canvas.create_text(
                x, y - label_offset,
                text=f"{label}\n{entity['hp']}/{entity['max_hp']}",
                fill="white", font=("Segoe UI", 8), justify="center",
            )

        selected = self.entity_tree.selection()
        children = self.entity_tree.get_children()
        if children:
            self.entity_tree.delete(*children)
        for entity in sorted(
            self.state["entities"],
            key=lambda item: (int(item.get("side", -1)), int(item.get("creation_ordinal", 0))),
        ):
            card_id = int(entity.get("card_id", -1))
            tower_role = self._crown_tower_role(entity)
            label = CARD_NAMES.get(
                card_id,
                "King" if tower_role == "king" else "Princess" if tower_role else str(card_id),
            )
            entity_id = str(entity["id"])
            self.entity_tree.insert(
                "", "end", iid=entity_id,
                values=(
                    "蓝" if int(entity.get("side", -1)) == 0 else "红",
                    label,
                    f"{entity.get('hp', 0)}/{entity.get('max_hp', 0)}",
                    f"{entity.get('x', 0)},{entity.get('y', 0)}",
                    entity.get("behavior_state", ""),
                ),
            )
        for entity_id in selected:
            if self.entity_tree.exists(entity_id):
                self.entity_tree.selection_add(entity_id)

        episode = self.state.get("episode", {})
        clock, phase, multiplier = self._match_clock_info(
            int(self.state["tick"]), bool(episode.get("terminated"))
        )
        canvas_width = max(1, self.canvas.winfo_width())
        canvas.create_text(
            canvas_width - 14, 14, text=clock, anchor="ne",
            fill="white", font=("Segoe UI", 26, "bold"), tags=("match_clock",),
        )
        canvas.create_text(
            canvas_width - 14, 52,
            text=f"{phase}  ·  ×{multiplier} 圣水",
            anchor="ne", fill="#a9c9df", font=("Segoe UI", 10),
            tags=("match_clock",),
        )

        if self.last_deploy_marker is not None:
            marker_x, marker_y, valid = self.last_deploy_marker
            x, y = self._native_to_canvas(marker_x, marker_y, geometry)
            color = "#51e884" if valid else "#ff355e"
            canvas.create_oval(
                x - 9, y - 9, x + 9, y + 9,
                outline=color, width=3,
            )
            if not valid:
                canvas.create_line(x - 6, y - 6, x + 6, y + 6, fill=color, width=3)
                canvas.create_line(x + 6, y - 6, x - 6, y + 6, fill=color, width=3)

        players = self.state.get("players", [])
        elixir = "  ".join(
            f"{'蓝' if p['side'] == 0 else '红'}:{p['elixir']}"
            for p in players
        )
        if episode.get("terminated"):
            winner = episode.get("winner")
            outcome = "平局" if winner is None else f"{'蓝' if winner == 0 else '红'}方胜"
            crowns = episode.get("crowns", [0, 0])
            self.status.set(
                f"终局 Tick {episode['terminal_tick']}  |  {outcome}"
                f"  |  皇冠 {crowns[0]}:{crowns[1]}"
                f"  |  奖励 {episode.get('rewards', [0, 0])}"
            )
            self.auto.set(False)
            self._refresh_cards()
            return
        self.status.set(
            f"Tick {self.state['tick']}  |  {self.state['elapsed_seconds']:.2f}s"
            f"  |  圣水 {elixir}  |  实体 {self.state['entity_count']}"
            f"  |  hash {self.state.get('state_hash', '—')}"
            f"  |  RNG {self.state.get('rng_state', '—')}"
        )
        self._refresh_cards()

    def _error(self, error: Exception) -> None:
        self.status.set(str(error))
        if not isinstance(error, NativeHostError):
            messagebox.showerror("原生内核错误", str(error))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay",
        type=Path,
        default=Path("examples/eight-card-bootstrap.json"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=37031)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    root = tk.Tk()
    env = NativeRoyaleEnv(host=arguments.host, port=arguments.port)
    gui = NativeCoreGui(root, env, arguments.replay.resolve())
    if arguments.smoke:
        root.withdraw()
        gui._reset_native_battle()
        player = next(item for item in gui.state["players"] if item["side"] == 0)
        gui._select_card(int(player["hand_deck_indices"][0]))
        root.update_idletasks()
        result = {
            "ok": True,
            "tick": gui.state["tick"],
            "entities": gui.state["entity_count"],
            "canvas_items": len(gui.canvas.find_all()),
            "entity_rows": len(gui.entity_tree.get_children()),
            "valid_cells": sum(row.count("1") for row in gui.deployment_mask or []),
            "clock": gui._match_clock_info(int(gui.state["tick"]), False),
            "clock_items": len(gui.canvas.find_withtag("match_clock")),
        }
        if (
            result["tick"] != 100 or result["entities"] != 6
            or result["clock"] != ("2:55", "常规时间", 1)
            or result["clock_items"] != 2
        ):
            raise RuntimeError(f"GUI smoke opening mismatch: {result}")
        print(json.dumps(result, ensure_ascii=False))
        root.destroy()
        return 0
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
