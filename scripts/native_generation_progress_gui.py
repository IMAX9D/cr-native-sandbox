"""Minimal read-only GUI for the authoritative native Tick generation queue."""
from __future__ import annotations

import ctypes
from datetime import datetime
import json
import os
from pathlib import Path
import sqlite3
import time
import tkinter as tk
from tkinter import ttk


DATA_ROOT = Path(
    r"D:\AI_data\cr-native-core\expert-v1"
    r"\one-click-schema5-v3-current-frontier-v5"
)
STATE = DATA_ROOT / "control" / "state.json"
QUEUE = DATA_ROOT / "native-authoritative-ticks-v1" / "work-queue.sqlite3"
LOGS = DATA_ROOT / "logs"
TARGET = 100_000


def available_memory_gib() -> float | None:
    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return status.ullAvailPhys / 1024**3


def read_state() -> dict:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def queue_snapshot() -> dict:
    if not QUEUE.is_file():
        return {"counts": {}, "latest": None}
    uri = f"file:{QUEUE.as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=1)
    try:
        counts = {
            str(status): int(count)
            for status, count in connection.execute(
                "SELECT status,COUNT(*) FROM tasks GROUP BY status"
            )
        }
        latest = connection.execute(
            "SELECT MAX(updated_at) FROM tasks "
            "WHERE status IN ('done','failed')"
        ).fetchone()[0]
        return {"counts": counts, "latest": latest}
    finally:
        connection.close()


def duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--"
    hours = seconds / 3600
    if hours >= 24:
        return f"{hours / 24:.1f}天"
    if hours >= 1:
        return f"{hours:.1f}小时"
    return f"{seconds / 60:.0f}分钟"


class NativeProgress(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("CR 原生 Tick 生成进度")
        self.geometry("620x310")
        self.resizable(False, False)
        self.configure(bg="#111827")
        self.last_finished: int | None = None
        self.last_sample_time: float | None = None
        self.smoothed_rate: float | None = None
        self.vars = {name: tk.StringVar(value="--") for name in (
            "status", "progress", "counts", "rate", "eta", "stage", "memory",
        )}
        self._build()
        self.after(100, self.refresh)

    def _build(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#111827")
        style.configure("TLabel", background="#111827", foreground="#d1d5db",
                        font=("Microsoft YaHei UI", 10))
        style.configure("Title.TLabel", foreground="#f9fafb",
                        font=("Microsoft YaHei UI", 18, "bold"))
        style.configure("Status.TLabel", foreground="#60a5fa",
                        font=("Microsoft YaHei UI", 11, "bold"))
        style.configure("Green.Horizontal.TProgressbar", troughcolor="#374151",
                        background="#22c55e")

        root = ttk.Frame(self, padding=18)
        root.pack(fill="both", expand=True)
        top = ttk.Frame(root)
        top.pack(fill="x")
        ttk.Label(top, text="原生 Tick 生成", style="Title.TLabel").pack(side="left")
        ttk.Label(top, textvariable=self.vars["status"], style="Status.TLabel").pack(side="right")

        ttk.Label(root, textvariable=self.vars["progress"],
                  font=("Microsoft YaHei UI", 15, "bold")).pack(anchor="w", pady=(24, 5))
        self.bar = ttk.Progressbar(root, maximum=TARGET,
                                  style="Green.Horizontal.TProgressbar")
        self.bar.pack(fill="x", pady=(0, 14))
        for name in ("counts", "rate", "eta", "stage", "memory"):
            ttk.Label(root, textvariable=self.vars[name]).pack(anchor="w", pady=2)

        buttons = ttk.Frame(root)
        buttons.pack(fill="x", pady=(14, 0))
        ttk.Button(buttons, text="立即刷新", command=self.refresh).pack(side="left")
        ttk.Button(buttons, text="打开日志", command=lambda: os.startfile(LOGS)).pack(side="right")
        ttk.Button(buttons, text="打开数据", command=lambda: os.startfile(DATA_ROOT)).pack(side="right", padx=8)

    def refresh(self) -> None:
        try:
            now = time.time()
            snapshot = queue_snapshot()
            counts = snapshot["counts"]
            done = counts.get("done", 0)
            failed = counts.get("failed", 0)
            leased = counts.get("leased", 0)
            pending = counts.get("pending", 0)
            finished = done + failed
            if self.last_finished is not None and self.last_sample_time is not None:
                interval = max(0.001, now - self.last_sample_time)
                instant = max(0, finished - self.last_finished) / interval
                self.smoothed_rate = (
                    instant if self.smoothed_rate is None
                    else self.smoothed_rate * 0.65 + instant * 0.35
                )
            self.last_finished, self.last_sample_time = finished, now
            rate = self.smoothed_rate
            eta = (TARGET - finished) / rate if rate and rate > 0 else None
            state = read_state()
            stage = str(state.get("active_stage") or "空闲/阶段切换")
            error = state.get("last_error")
            latest = snapshot.get("latest")
            fresh = latest is not None and now - float(latest) < 15
            running = stage == "generate_native_ticks" and leased > 0 and fresh and not error

            self.vars["status"].set("● 运行中" if running else "● 检查中/已停止")
            self.vars["progress"].set(
                f"{finished:,} / {TARGET:,}  ({finished / TARGET * 100:.2f}%)"
            )
            self.vars["counts"].set(
                f"Full {done:,}　Prefix候选/失败 {failed:,}　处理中 {leased}　待处理 {pending:,}"
            )
            self.vars["rate"].set(
                "实时速度：预热中" if rate is None else f"实时速度：{rate:.3f} 场/秒"
            )
            self.vars["eta"].set(f"预计剩余：{duration(eta)}")
            self.vars["stage"].set(
                f"阶段：{stage}" + (f"　错误：{error}" if error else "")
            )
            memory = available_memory_gib()
            updated = "--" if latest is None else datetime.fromtimestamp(latest).strftime("%H:%M:%S")
            self.vars["memory"].set(
                f"最后完成：{updated}　可用内存：{memory:.1f} GiB" if memory is not None
                else f"最后完成：{updated}"
            )
            self.bar["value"] = min(TARGET, finished)
        except Exception as error:  # GUI must remain available during WAL rotations.
            self.vars["status"].set("● 读取失败")
            self.vars["stage"].set(f"读取错误：{type(error).__name__}: {error}")
        self.after(2000, self.refresh)


def main() -> None:
    NativeProgress().mainloop()


if __name__ == "__main__":
    main()
