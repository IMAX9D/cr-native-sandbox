"""Tiny live progress window for the expert dataset compile pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import psutil


DATA_ROOT = Path(
    r"D:\AI_data\cr-native-core\expert-v1"
    r"\one-click-schema5-v3-current-frontier-v5"
)
COMPILED = DATA_ROOT / "compiled" / "native-bc-v1"
STATE_PATH = DATA_ROOT / "control" / "state.json"
FORMAL_RUN = DATA_ROOT / "runs" / "expert-v1-schema5-v3-100k"
PHASE_NAMES = {
    "input_authentication": "认证输入回执",
    "result_index": "读取原生结果",
    "source_load": "读取冻结源对局",
    "episode_join": "绑定原生帧与动作标签",
    "capacity_preflight": "容量采样",
    "plan_complete": "编译计划完成",
    "shard_compile": "生成训练分片",
    "finalize": "最终校验",
    "plan_live_validation": "逐场校验实时输入",
    "plan_structure_validation": "逐场校验计划结构",
    "token_result_load": "读取 Token 证据",
    "token_episode_validation": "逐场校验 Token 标签",
    "token_coverage_finalize": "汇总 Token 覆盖",
    "final_shard_validation": "逐个校验最终分片",
    "fast_token_shard_aggregate": "汇总最终 Token 标签",
    "complete": "编译完成",
}


def compiler_process() -> psutil.Process | None:
    candidates: list[psutil.Process] = []
    for process in psutil.process_iter(["cmdline", "create_time"]):
        try:
            command = " ".join(process.info.get("cmdline") or [])
            if "expert_v1.compile_native_bc_dataset" in command:
                candidates.append(process)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return max(candidates, key=lambda value: value.create_time(), default=None)


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


class ProgressWindow:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("专家训练数据编译进度")
        self.root.geometry("640x250")
        self.root.resizable(False, False)
        frame = ttk.Frame(self.root, padding=20)
        frame.pack(fill="both", expand=True)
        self.stage = ttk.Label(frame, text="读取状态…", font=("Microsoft YaHei", 15, "bold"))
        self.stage.pack(anchor="w")
        self.value = tk.DoubleVar(value=0)
        self.bar = ttk.Progressbar(frame, variable=self.value, maximum=100, length=590)
        self.bar.pack(fill="x", pady=(18, 8))
        self.percent = ttk.Label(frame, text="0%", font=("Segoe UI", 22, "bold"))
        self.percent.pack(anchor="w")
        self.detail = ttk.Label(frame, text="", font=("Microsoft YaHei", 10), justify="left")
        self.detail.pack(anchor="w", pady=(8, 0))
        self.note = ttk.Label(
            frame,
            text="只显示编译器写出的真实计数，不使用时间或 CPU 估算。",
            foreground="#666666",
        )
        self.note.pack(anchor="w", pady=(8, 0))
        self.refresh()

    def refresh(self) -> None:
        plan_path = COMPILED / "compile-plan.json"
        result_path = COMPILED / "compile-result.json"
        progress_path = COMPILED / "compile-progress.json"
        process = compiler_process()
        memory = 0.0
        cpu = 0.0
        if process is not None:
            try:
                cpu = sum(process.cpu_times()[:2])
                memory = process.memory_info().rss / 1024**3
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                process = None
        progress = load_json(progress_path)
        state = load_json(STATE_PATH)
        active_stage = str(state.get("active_stage") or "")
        if active_stage == "formal_training":
            events_path = FORMAL_RUN / "events.jsonl"
            training_progress = load_json(
                FORMAL_RUN / "training-progress.json"
            )
            latest_epoch = 0
            global_step = 0
            loss_text = ""
            if events_path.is_file():
                try:
                    for line in events_path.read_text(
                        encoding="utf-8-sig"
                    ).splitlines():
                        event = json.loads(line)
                        if event.get("event") == "epoch_complete":
                            latest_epoch = int(event.get("epoch", latest_epoch))
                            global_step = int(event.get("global_step", global_step))
                            training = event.get("training") or {}
                            validation = event.get("validation") or {}
                            loss_text = (
                                f"　train loss={float(training.get('loss', 0)):.6f}"
                                f"　val loss={float(validation.get('loss', 0)):.6f}"
                            )
                except Exception:
                    pass
            percent = latest_epoch * 100.0 / 20
            stage = "正式专家训练"
            detail = (
                f"Epoch：{latest_epoch} / 20（精确）　global step：{global_step:,}"
                f"{loss_text}\nPython 内存：{memory:.1f} GiB"
            )
            if latest_epoch == 0:
                stage = "正式训练：准备数据/首个 Epoch"
            if training_progress.get("kind") == "cr_expert_training_progress_v1":
                epoch = int(training_progress.get("epoch", 0))
                epochs = int(training_progress.get("epochs", 20))
                batch = int(training_progress.get("batch", 0))
                batches = int(training_progress.get("batches", 0))
                global_step = int(training_progress.get("global_step", 0))
                batch_fraction = 0.0 if batches <= 0 else batch / batches
                percent = ((max(epoch, 1) - 1) + batch_fraction) * 100.0 / max(epochs, 1)
                stage = "正式专家训练"
                detail = (
                    f"Epoch：{epoch} / {epochs}　Batch：{batch:,} / {batches:,}（精确）\n"
                    f"global step：{global_step:,}　状态：{training_progress.get('status', '')}"
                )
                if "loss" in training_progress:
                    detail += f"　loss={float(training_progress['loss']):.6f}"
        elif result_path.is_file():
            percent = 100.0
            stage = "编译完成"
            detail = f"结果：{result_path}"
        elif progress.get("kind") == "cr_expert_compile_progress_v1":
            current = int(progress.get("current", 0))
            total = int(progress.get("total", 0))
            phase = str(progress.get("phase") or "")
            if phase == "shard_compile" and total > 0:
                # Worker output directories are atomically renamed only after
                # every array and shard.json is fsynced.  Counting shard.json
                # therefore gives an exact, crash-safe result even if the
                # parent process is temporarily late consuming Future events.
                disk_completed = sum(
                    1 for _ in (COMPILED / "shards").glob("*/shard.json")
                )
                current = max(current, disk_completed)
            percent = 0.0 if total == 0 else current * 100.0 / total
            stage = PHASE_NAMES.get(phase, phase or "编译处理中")
            detail = (
                f"当前阶段：{current:,} / {total:,}（精确）\n"
                f"{progress.get('detail', '')}　Python 内存：{memory:.1f} GiB"
            )
        else:
            percent = 0.0
            stage = "等待编译器发布首个精确计数"
            detail = f"Python 内存：{memory:.1f} GiB"
            if process is None:
                stage = "编译进程未运行"
        self.value.set(percent)
        self.percent.configure(text=f"{percent:.1f}%")
        self.stage.configure(text=stage)
        self.detail.configure(text=detail)
        self.root.after(1000, self.refresh)

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    ProgressWindow().run()
