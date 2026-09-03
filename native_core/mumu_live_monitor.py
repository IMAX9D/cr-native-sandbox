"""Small non-technical monitor for the MuMu expert controller."""

from __future__ import annotations

from datetime import datetime
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = Path(
    r"D:\AI_data\cr-native-core\mumu-live-expert\controller-status.json"
)
RUNTIME_ROOT = Path(r"D:\AI_data\runtime")
MUMU_CLI = Path(r"C:\Program Files\Netease\MuMu\nx_main\mumu-cli.exe")
NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

CARD_ZH = {
    26000010: "小骷髅",
    26000014: "火枪手",
    26000021: "野猪骑士",
    26000030: "冰雪精灵",
    26000038: "冰人",
    26000009: "戈仑石人",
    26000037: "地狱飞龙",
    26000043: "野蛮人精锐",
    26000050: "皇家幽灵",
    26000054: "加农炮战车",
    26000084: "电击精灵",
    28000012: "龙卷风",
    28000015: "野蛮人滚筒",
    27000000: "加农炮",
    28000000: "火球",
    28000011: "滚木",
}
EVENT_ZH = {
    "controller_ready": "控制器就绪",
    "waiting_for_friendly_battle": "等待友谊战",
    "battle_detected": "检测到对局，开始接管",
    "touch_sent": "模型执行下牌",
    "touch_retry": "下牌补发",
    "touch_accepted": "游戏已确认下牌",
    "touch_not_confirmed": "下牌未获确认",
    "battle_released": "对局结束，释放接管",
    "inference_error": "推理异常（已停止触屏）",
    "controller_error": "控制器异常",
}


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            process_query_limited_information, False, pid
        )
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _format_time(value: str | None) -> str:
    if not value:
        return "--"
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone().strftime("%H:%M:%S")
    except ValueError:
        return str(value)


def _load_deck(path: str | None, side: int) -> list[str]:
    if not path:
        return ["未知"] * 8
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        spells = raw["battle"][f"deck{side}"]["sp"]
    except (OSError, ValueError, KeyError, TypeError):
        return ["未知"] * 8
    result: list[str] = []
    for item in spells:
        card_id = int(item.get("d", -1))
        name = CARD_ZH.get(card_id, str(card_id))
        form = int(item.get("el", 0))
        prefix = "觉醒 " if form == 1 else "英雄 " if form == 2 else ""
        result.append(prefix + name)
    return (result + ["未知"] * 8)[:8]


def _load_live_deck(raw: Any) -> list[str] | None:
    if not isinstance(raw, list) or len(raw) != 8:
        return None
    result: list[str] = []
    try:
        for item in raw:
            card_id = int(item["card_id"])
            form = int(item.get("form_flags", 0))
            prefix = "觉醒 " if form == 1 else "英雄 " if form == 2 else ""
            result.append(prefix + CARD_ZH.get(card_id, str(card_id)))
    except (KeyError, TypeError, ValueError):
        return None
    return result


class Monitor(tk.Tk):
    COLORS = {
        "starting": ("正在启动", "#eab308"),
        "waiting": ("等待友谊战", "#3b82f6"),
        "controlling": ("AI 正在接管", "#22c55e"),
        "error": ("异常 · 已停止触屏", "#ef4444"),
        "stopped": ("已停止", "#94a3b8"),
        "offline": ("控制器未运行", "#94a3b8"),
        "stale": ("状态已失联", "#ef4444"),
    }

    def __init__(self) -> None:
        super().__init__()
        self.title("CR Expert · MuMu 友谊战监控")
        self.geometry("980x680")
        self.minsize(880, 580)
        self.configure(bg="#0f172a")
        self.status: dict[str, Any] | None = None
        self.deck: list[str] = ["未知"] * 8
        self.log_path: Path | None = None
        self.log_signature: tuple[str, int, float] | None = None
        self._build()
        self.after(100, self._refresh)

    def _build(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Root.TFrame", background="#0f172a")
        style.configure("Card.TFrame", background="#1e293b")
        style.configure("Title.TLabel", background="#0f172a", foreground="white", font=("Microsoft YaHei UI", 19, "bold"))
        style.configure("Sub.TLabel", background="#0f172a", foreground="#94a3b8", font=("Microsoft YaHei UI", 10))
        style.configure("Metric.TLabel", background="#1e293b", foreground="#60a5fa", font=("Microsoft YaHei UI", 18, "bold"))
        style.configure("MetricName.TLabel", background="#1e293b", foreground="#cbd5e1", font=("Microsoft YaHei UI", 10))
        style.configure("Hand.TLabel", background="#1e293b", foreground="white", font=("Microsoft YaHei UI", 11, "bold"), anchor="center")
        style.configure("TButton", font=("Microsoft YaHei UI", 10), padding=8)
        style.configure("Treeview", background="#111827", foreground="#e2e8f0", fieldbackground="#111827", rowheight=28, font=("Microsoft YaHei UI", 9))
        style.configure("Treeview.Heading", background="#334155", foreground="white", font=("Microsoft YaHei UI", 9, "bold"))

        root = ttk.Frame(self, style="Root.TFrame", padding=20)
        root.pack(fill="both", expand=True)
        header = ttk.Frame(root, style="Root.TFrame")
        header.pack(fill="x")
        ttk.Label(header, text="CR Expert · MuMu 实战监控", style="Title.TLabel").pack(side="left")
        self.status_label = tk.Label(header, text="● 检查中", bg="#0f172a", fg="#eab308", font=("Microsoft YaHei UI", 13, "bold"))
        self.status_label.pack(side="right")
        self.detail = ttk.Label(root, text="正在读取控制器状态……", style="Sub.TLabel")
        self.detail.pack(fill="x", pady=(4, 16))

        metrics = ttk.Frame(root, style="Root.TFrame")
        metrics.pack(fill="x")
        self.metric_values: dict[str, ttk.Label] = {}
        for key, label in (("battle", "对局"), ("tick", "原生 Tick"), ("elixir", "圣水"), ("actions", "已执行 / 已确认")):
            box = ttk.Frame(metrics, style="Card.TFrame", padding=12)
            box.pack(side="left", fill="x", expand=True, padx=(0, 8))
            ttk.Label(box, text=label, style="MetricName.TLabel").pack(anchor="w")
            value = ttk.Label(box, text="--", style="Metric.TLabel")
            value.pack(anchor="w", pady=(4, 0))
            self.metric_values[key] = value

        hand_box = ttk.Frame(root, style="Card.TFrame", padding=12)
        hand_box.pack(fill="x", pady=(14, 14))
        ttk.Label(hand_box, text="当前四张手牌", style="MetricName.TLabel").pack(anchor="w", pady=(0, 8))
        cards = ttk.Frame(hand_box, style="Card.TFrame")
        cards.pack(fill="x")
        self.hand_labels: list[ttk.Label] = []
        for index in range(4):
            label = ttk.Label(cards, text=f"槽位 {index + 1}\n--", style="Hand.TLabel", padding=12)
            label.pack(side="left", fill="x", expand=True, padx=(0, 8 if index < 3 else 0))
            self.hand_labels.append(label)

        action_box = ttk.Frame(root, style="Card.TFrame", padding=12)
        action_box.pack(fill="x", pady=(0, 14))
        ttk.Label(action_box, text="模型最近一次决策", style="MetricName.TLabel").pack(anchor="w")
        self.action_label = ttk.Label(action_box, text="尚未下牌", style="Hand.TLabel")
        self.action_label.pack(fill="x", pady=(6, 0))

        events_box = ttk.Frame(root, style="Root.TFrame")
        events_box.pack(fill="both", expand=True)
        self.events = ttk.Treeview(events_box, columns=("time", "event", "detail"), show="headings", height=8)
        self.events.heading("time", text="时间")
        self.events.heading("event", text="事件")
        self.events.heading("detail", text="详情")
        self.events.column("time", width=90, anchor="center", stretch=False)
        self.events.column("event", width=190, anchor="w", stretch=False)
        self.events.column("detail", width=620, anchor="w")
        scroll = ttk.Scrollbar(events_box, orient="vertical", command=self.events.yview)
        self.events.configure(yscrollcommand=scroll.set)
        self.events.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        buttons = ttk.Frame(root, style="Root.TFrame")
        buttons.pack(fill="x", pady=(14, 0))
        ttk.Button(buttons, text="显示 MuMu", command=self._show_mumu).pack(side="left")
        ttk.Button(buttons, text="打开日志", command=self._open_log).pack(side="left", padx=8)
        self.start_button = ttk.Button(buttons, text="启动接管", command=self._start_controller)
        self.start_button.pack(side="right")
        self.stop_button = ttk.Button(buttons, text="停止接管", command=self._stop_controller)
        self.stop_button.pack(side="right", padx=8)

    def _current_lifecycle(self, status: dict[str, Any] | None) -> str:
        if not status:
            return "offline"
        pid = int(status.get("controller_pid") or 0)
        if not _pid_alive(pid):
            return "offline"
        heartbeat = float(status.get("heartbeat_unix") or 0)
        if time.time() - heartbeat > 3.0:
            return "stale"
        return str(status.get("lifecycle") or "starting")

    def _refresh(self) -> None:
        status = _read_json(STATUS_PATH)
        self.status = status
        lifecycle = self._current_lifecycle(status)
        title, color = self.COLORS.get(lifecycle, (lifecycle, "#eab308"))
        self.status_label.configure(text=f"● {title}", fg=color)
        live = bool(status and lifecycle not in ("offline", "stale", "stopped"))
        self.start_button.configure(state="disabled" if live else "normal")
        self.stop_button.configure(state="normal" if live else "disabled")
        if status:
            self.deck = (
                _load_live_deck(status.get("live_deck"))
                or _load_deck(status.get("deck"), int(status.get("local_side", 1)))
            )
            self.metric_values["battle"].configure(text=str(status.get("battle_number", 0)))
            self.metric_values["tick"].configure(text=str(status.get("tick") if status.get("tick") is not None else "--"))
            player = status.get("live_player") or {}
            self.metric_values["elixir"].configure(text=str(player.get("elixir", "--")))
            counts = status.get("event_counts") or {}
            self.metric_values["actions"].configure(text=f"{counts.get('touch_sent', 0)} / {counts.get('touch_accepted', 0)}")
            self.detail.configure(text=f"模型 step {status.get('model_step', '--')} · {status.get('play_rate_scale', 1)}x · 最后心跳 {_format_time(status.get('heartbeat_utc'))}")
            hand = player.get("hand_deck_indices") or [-1, -1, -1, -1]
            for index, label in enumerate(self.hand_labels):
                deck_index = int(hand[index]) if index < len(hand) else -1
                card = self.deck[deck_index] if deck_index in range(8) else "空"
                label.configure(text=f"槽位 {index + 1}\n{card}")
            action = status.get("last_action")
            if action:
                deck_index = int(action.get("deck_index", -1))
                card = self.deck[deck_index] if deck_index in range(8) else str(action.get("card_id", "?"))
                probability = action.get("play_probability")
                probability_text = f"{float(probability) * 100:.1f}%" if probability is not None else "--"
                self.action_label.configure(text=f"Tick {action.get('tick')} · {card} · 手牌槽 {int(action.get('slot', 0)) + 1} · 本 Tick 行动概率 {probability_text}")
            log_value = status.get("log_file")
            self.log_path = Path(log_value) if log_value else None
            self._refresh_events()
        else:
            self.detail.configure(text="未找到控制器状态；点击“启动接管”即可启动。")
        self.after(350, self._refresh)

    def _refresh_events(self) -> None:
        path = self.log_path
        if not path or not path.is_file():
            return
        stat = path.stat()
        signature = (str(path), stat.st_size, stat.st_mtime)
        if signature == self.log_signature:
            return
        self.log_signature = signature
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-120:]
        except OSError:
            return
        rows: list[tuple[str, str, str]] = []
        for line in lines:
            try:
                value = json.loads(line)
            except ValueError:
                continue
            event = str(value.get("event", ""))
            detail_fields = []
            for key, label in (("tick", "Tick"), ("slot", "槽位"), ("card_id", "卡牌"), ("position", "格子"), ("latency_ms", "确认延迟"), ("error", "异常")):
                if key in value:
                    detail_fields.append(f"{label}={value[key]}")
            rows.append((_format_time(value.get("time")), EVENT_ZH.get(event, event), "  ".join(detail_fields)))
        self.events.delete(*self.events.get_children())
        for row in rows:
            self.events.insert("", "end", values=row)
        children = self.events.get_children()
        if children:
            self.events.see(children[-1])

    def _start_controller(self) -> None:
        RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = RUNTIME_ROOT / f"mumu-expert-{stamp}.out.log"
        err_path = RUNTIME_ROOT / f"mumu-expert-{stamp}.err.log"
        out_handle = out_path.open("w", encoding="utf-8")
        err_handle = err_path.open("w", encoding="utf-8")
        subprocess.Popen(
            [sys.executable, "-m", "native_core.mumu_live_controller"],
            cwd=PROJECT_ROOT,
            stdout=out_handle,
            stderr=err_handle,
            creationflags=NO_WINDOW,
        )
        out_handle.close()
        err_handle.close()
        self.detail.configure(text="正在启动接管控制器……")

    def _stop_controller(self) -> None:
        status = self.status or {}
        pid = int(status.get("controller_pid") or 0)
        if not _pid_alive(pid):
            return
        if not messagebox.askyesno("停止接管", "确定停止 AI 接管吗？MuMu 和游戏不会关闭。"):
            return
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            creationflags=NO_WINDOW,
        )

    def _show_mumu(self) -> None:
        subprocess.run(
            [str(MUMU_CLI), "control", "-v", "1", "-ver", "12", "show_window"],
            capture_output=True,
            creationflags=NO_WINDOW,
        )

    def _open_log(self) -> None:
        path = self.log_path
        if path and path.exists():
            os.startfile(path)
        else:
            os.startfile(STATUS_PATH.parent)


def main() -> int:
    Monitor().mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
