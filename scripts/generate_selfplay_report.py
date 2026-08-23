"""Generate the evidence-backed Self-Play v0.1 stage report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "Unknown"
    return f"{float(value):.{digits}f}"


def _percent(value: Any, digits: int = 2) -> str:
    if value is None:
        return "Unknown"
    return f"{float(value) * 100.0:.{digits}f}%"


def _comparison(summary: dict[str, Any] | None) -> str:
    if not summary:
        return "Unknown（本阶段未生成该对局）"
    interval = summary["paired_score_rate_95ci"]
    return (
        f"W/L/D={summary['wins']}/{summary['losses']}/{summary['draws']}，"
        f"score={_percent(summary['score_rate'])}，"
        f"paired 95% CI=[{_percent(interval[0])}, {_percent(interval[1])}]"
    )


def _learning_verdict(summary: dict[str, Any] | None) -> str:
    if not summary:
        return "Unknown：缺少 Final vs Initial 正式评估。"
    low, high = summary["paired_score_rate_95ci"]
    if float(low) > 0.5:
        return "有统计显著的正向学习信号：Final 对初始化的 paired 95% CI 下界高于 50%。"
    if float(high) < 0.5:
        return "出现统计显著退化：Final 对初始化的 paired 95% CI 上界低于 50%。"
    return "尚不能证明强于初始化：置信区间仍跨过 50%，需要更多样本或继续训练。"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "docs" / "SELFPLAY_V0_1_TRAINING_REPORT.zh-CN.md",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_root = args.run_root.resolve()
    manifest = json.loads(
        (run_root / "manifest.json").read_text(encoding="utf-8-sig")
    )
    checkpoint = torch.load(
        run_root / "checkpoints" / "latest.pt",
        map_location="cpu",
        weights_only=False,
    )
    stage = json.loads(
        (run_root / "stage-summary.json").read_text(encoding="utf-8-sig")
    )
    evaluation_path = (
        run_root / "evaluations" / "official-v0.1" / "evaluation-summary.json"
    )
    evaluation = (
        json.loads(evaluation_path.read_text(encoding="utf-8-sig"))
        if evaluation_path.is_file() else {}
    )
    config = manifest["config"]
    metrics = checkpoint.get("metrics", {})
    behavior = checkpoint.get("behavior", {})
    resources = stage.get("resources", {})
    overall = resources.get("overall", {})
    required = evaluation.get("required_comparisons", {})
    final_initial = required.get("vs_initial")
    final_previous = required.get("vs_previous")
    final_random = required.get("vs_random_legal")
    matrix = evaluation.get("cross_play_score_matrix", {})

    lines = [
        "# Self-Play v0.1 训练报告",
        "",
        f"> Run：`{run_root}`  ",
        f"> 状态：{'通过阶段验收' if stage.get('passed') else '阶段失败'}  ",
        f"> 生成依据：原生轨迹、Checkpoint、资源采样与固定种子评估文件。",
        "",
        "## 1. 冻结训练语义",
        "",
        f"- 原生核心：`{config.get('truth_source')}`；",
        f"- 20 Hz 原生 Tick，策略每 Tick 可决策；",
        f"- 并发：{config.get('avds')} AVD / {config.get('workers')} Worker；",
        f"- Reward：`{config.get('reward')}`；",
        f"- 实现摘要：`{config.get('implementation_digest')}`；",
        f"- 初始化模型：`{checkpoint.get('initial_model_digest')}`；",
        f"- 当前模型：`{checkpoint.get('current_model_digest')}`。",
        "",
        "Reward 冻结为：",
        "",
        "```text",
        "r_t = terminal(+1/0/-1) + 0.2 * (gamma * Phi(next) - Phi(current))",
        "gamma = 0.99995",
        "Phi = 双方归一化皇冠塔剩余总 HP 之差",
        "terminal absorbing Phi = 0",
        "```",
        "",
        "明确排除圣水、击杀、过河、单位伤害和场面价值。",
        "",
        "## 2. 训练规模与吞吐",
        "",
        "| 指标 | 实测 |",
        "| --- | ---: |",
        f"| Native ticks | {int(checkpoint.get('native_ticks', 0)):,} |",
        f"| Agent steps | {int(checkpoint.get('agent_steps', 0)):,} |",
        f"| Episodes | {int(checkpoint.get('completed_episodes', 0)):,} |",
        f"| 最近一轮 Env steps/s | {_number(metrics.get('environment_steps_per_second'), 2)} |",
        f"| 最近一轮含 PPO steps/s | {_number(metrics.get('training_steps_per_second'), 2)} |",
        f"| 最近一轮 Episodes/hour | {_number(metrics.get('episodes_per_hour'), 2)} |",
        f"| 最近一轮墙钟 | {_number(metrics.get('iteration_wall_seconds'), 2)} s |",
        "",
        "## 3. PPO 健康状态",
        "",
        "| 指标 | 最近一轮 |",
        "| --- | ---: |",
        f"| Policy loss | {_number(metrics.get('policy_loss'), 6)} |",
        f"| Value loss | {_number(metrics.get('value_loss'), 6)} |",
        f"| Entropy | {_number(metrics.get('entropy'), 6)} |",
        f"| Approx KL | {_number(metrics.get('approx_kl'), 8)} |",
        f"| Clip fraction | {_number(metrics.get('clip_fraction'), 6)} |",
        f"| Explained variance | {_number(metrics.get('explained_variance'), 6)} |",
        f"| Gradient norm | {_number(metrics.get('gradient_norm'), 6)} |",
        f"| Value abs max | {_number(metrics.get('value_abs_max'), 6)} |",
        f"| 参数相对 L2 变化 | {_number(metrics.get('parameter_relative_delta_l2'), 6)} |",
        "",
        "所有指标及模型参数必须 finite；训练器对 NaN/Inf、连续熵坍缩、Value/梯度爆炸、"
        "RPC/终局/原生拒绝异常采用 fail-closed。",
        "",
        "## 4. 行为统计",
        "",
        f"- WAIT 比例：{_percent(behavior.get('wait_ratio'))}；",
        f"- 平均圣水：{_number(behavior.get('average_elixir'), 3)}；",
        f"- 圣水溢出比例：{_percent(behavior.get('elixir_leak_ratio'))}；",
        f"- 平局率：{_percent(behavior.get('draw_rate'))}；",
        f"- 平均对局长度：{_number(behavior.get('average_episode_ticks'), 1)} ticks；",
        f"- 原生动作拒绝：{behavior.get('native_action_rejections', 'Unknown')}；",
        f"- 八卡使用率：`{json.dumps(behavior.get('card_usage_rate', {}), ensure_ascii=False)}`。",
        "",
        "各卡 32×18 部署热图保存在 Run 的 `evaluations/behavior-*.npz`。",
        "",
        "## 5. 工程稳定性与资源",
        "",
        f"- Episode failure rate：{_percent(behavior.get('episode_failure_rate'))}；",
        f"- RPC failure rate：{_percent(checkpoint.get('sampling_profile', {}).get('rpc_failure_rate'))}；",
        f"- 最低可用系统 RAM：{_number(overall.get('system_ram_available_gb', {}).get('min'), 3)} GiB；",
        f"- 最大 GPU VRAM：{_number(overall.get('gpu_vram_used_mb', {}).get('max'), 1)} MiB；",
        f"- Android guest swap 最大值：{_number(overall.get('guest_swap_used_mb_total', {}).get('max'), 1)} MiB；",
        "- Checkpoint 包含模型、优化器、RNG、next seed 和完整 Manifest；P000、恢复点和候选点均独立保留。",
        "",
        "## 6. 固定种子正式评估",
        "",
        f"- Final vs Initial：{_comparison(final_initial)}；",
        f"- Final vs Previous：{_comparison(final_previous)}；",
        f"- Final vs RandomLegal：{_comparison(final_random)}。",
        "",
    ]
    if matrix:
        labels = list(matrix)
        lines.extend([
            "### Cross-Play score matrix",
            "",
            "行策略对列策略的 score rate（胜=1、平=0.5、负=0）：",
            "",
            "| Candidate | " + " | ".join(labels) + " |",
            "| --- | " + " | ".join("---:" for _ in labels) + " |",
        ])
        for row in labels:
            lines.append(
                f"| {row} | " + " | ".join(
                    _percent(matrix[row].get(column)) for column in labels
                ) + " |"
            )
        lines.append("")
    lines.extend([
        "## 7. 当前结论",
        "",
        f"1. Recurrent PPO 是否产生学习：{_learning_verdict(final_initial)}",
        "2. 本 Run 不能回答“纯终局 Reward 是否足够”，因为冻结方案已包含塔血势函数；只能评估当前组合。",
        f"3. 是否强于 RandomLegal：{_comparison(final_random)}。",
        "4. 是否策略坍缩：依据 Entropy、WAIT/卡牌使用率和 Cross-Play 综合判断；单个 PPO loss 不能下结论。",
        "5. 是否循环克制/遗忘：只有多个候选的 Cross-Play 矩阵出现稳定非传递关系时才能确认。",
        "6. RNN 是否利用历史：Unknown；需要无状态/打乱 hidden 的专项 ablation，不能由训练成功反推。",
        "7. 是否继续下一阶段：只有阶段健康、评估完整且无明显退化时才继续；否则先分析，不机械增加 ticks。",
        "",
        "## 8. 证据位置",
        "",
        f"- Manifest：`{run_root / 'manifest.json'}`；",
        f"- Latest checkpoint：`{run_root / 'checkpoints' / 'latest.pt'}`；",
        f"- 训练事件：`{run_root / 'logs' / 'events.jsonl'}`；",
        f"- 评估：`{evaluation_path if evaluation_path.is_file() else 'Unknown'}`；",
        f"- 阶段总表：`{run_root / 'stage-summary.json'}`。",
        "",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    data_report = run_root / "reports" / args.output.name
    data_report.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.output, data_report)
    print(json.dumps({
        "report": str(args.output.resolve()),
        "data_report": str(data_report.resolve()),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
