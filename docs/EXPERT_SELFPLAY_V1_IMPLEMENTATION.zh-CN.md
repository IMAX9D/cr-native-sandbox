# Expert Self-Play v1 实现与运行状态

## 已固化的算法边界

- 保留原 177M 专家 Actor 的输入、参数名和推理输出。
- `forward_with_features` 只额外暴露策略头前的 recurrent latent。
- 连续时间 marked-hazard 使用稳定的 `log(-expm1(-x))`；模型预测
  `lambda/秒`，不使用固定 60 Hz 强制动作。
- 等待、普通牌、技能、手牌槽和格子组成一个联合 log-prob；PPO 只使用
  一个联合 ratio。
- 联合熵和冻结 BC Actor 的 KL 均按行动事件概率加权。
- GAE 使用 `gamma_per_tick ** delta_ticks` 与
  `gae_lambda_per_tick ** delta_ticks`。
- 每局只训练 learner side；对手 Actor 整局冻结。
- 原生实体非空而编码为空、合法动作被原生拒绝、NaN/Inf 都会 hard fail。
- Reward 固定为敌塔伤害 `+0.001`、己塔伤害 `-0.0012`、摧塔 `±5`、
  胜负 `±10`；Stage 2 不修改 Reward。

## 当前卡组与 Runtime

固定 learner 卡组：野猪骑士、滚木、火球、火枪手、觉醒冰雪精灵、
骷髅兵、觉醒加农炮、英雄/精英冰人。塔楼为普通公主塔；对手卡组从
高频卡组库抽取。

Linux 云端使用无界面 Bionic `libg.so` Runtime。`ensure_bionic_workers.py`
会在实例重启后幂等恢复 Worker，不依赖临时 SSH shell 函数。

当前 Stage-2 主循环的对手权重仍固定为冻结专家 BASE；高频卡组是随机抽取的，
不等于已实现历史权重轮换。40% 最新 / 40% 历史 / 20% 专家与 Elo 晋级模块
尚未接入此主循环。

## Stage 1 已完成

- Critic 参数：11,001,608。
- Actor 全程冻结且 SHA256 不变。
- 流式 producer-consumer 采集完成 25 次有效更新。
- 最终 `global_update=52`。
- 最后三次独立新数据全局 EV：`0.2617 / 0.2784 / 0.3189`。
- 最终训练 EV：`0.2822`。
- Critic、Optimizer、Python/NumPy/Torch CPU/CUDA RNG 均已保存并验证有限。
- 最终 checkpoint：
  `formal-runs/stage1-hog26-stream-formal-v1/updates/update-00000025/checkpoints/checkpoint-000000000052.pt`。

旧 Stage 1 rollout 没有精确 LSTM pre-action hidden，因此只可用于 Critic；
Stage 2 admission 会明确拒绝它，不能被误当作 Actor PPO 数据。

## Stage 2 reaction 已实现

新采集合同在每个 recurrent chunk 起点保存实际使用的 LSTM pre-action
`hidden/cell`，默认 burn-in 16、unroll 64 对应决策索引 0、48、112……。
其余决策保留观测、动作、时间间隔和 old log-prob，不重复保存整份 hidden；
学习器从精确起始状态执行 burn-in。这与将全部 hidden 设为零不同。
云端 learner 会：

1. 验证 behavior FP16 Actor hash 与 policy version；
2. 从当前 Critic 重算 value 和 variable-time GAE；
3. 用已记录的 FP16 `old_logp_total` 计算 PPO ratio；
4. 从完整 episode 零 hidden 重放冻结 BC Actor，生成正确 recurrent BC KL；
5. 仅解冻 Entity、Spatial、LSTM 和 Timing；
6. 对 Critic latent 边界继续 `detach`；
7. 执行两轮 PPO，并监控 KL、clip fraction、BC-KL、lambda、熵与梯度；
8. Guard 失败时只允许一次“Actor LR ×0.5、PPO epoch=1、BC-KL ×2”重试；
9. 第二次失败不发布权重并停止；
10. 接受后同时发布 FP32 master checkpoint 与 canonical FP16 Actor。

实际 177M Stage 2 参数：

| 组 | 可训练参数 | LR |
|---|---:|---:|
| LSTM | 161,292,288 | `3e-7` |
| Spatial | 797,984 | `7.5e-7` |
| Entity | 199,952 | `1e-6` |
| Timing | 6,145 | `2e-6` |
| Critic | 11,001,608 | `1e-4` |

卡牌、位置、技能动作头共 14,788,452 参数继续冻结。

## 一键入口

### 准备文件

本仓库不提供模型或训练产物；干净克隆不能直接开始正式训练。需要：

- 已按 [Linux Bionic 部署说明](https://github.com/IMAX9D/cr-native-linux-bionic)
  安装的匹配 Runtime，以及对应数量的 Worker 槽位；
- 原 177M 专家 FP16 Actor 与 expert manifest；
- 已完成的 Stage-1 checkpoint，或已完整提交的 Stage-2 运行目录及其最新产物；
- learner 卡组（仓库提供 `examples/hog-2.6-evo-hero.json`）与自己的对手卡组目录。

默认的 `models/`、`formal-runs/`、`top-deck-presets-v1/` 是私有输入/输出位置，
不是随 Git 下载的资源。使用参数指向自己的文件。所有命令在仓库根目录执行。

### 首次 Canary 与正式启动

云端 Canary 示例：

```bash
python scripts/start_expert_selfplay_stage2.py --mode canary \
  --base-checkpoint /path/to/expert-fp16.pt \
  --expert-manifest /path/to/manifest.json \
  --stage1-checkpoint /path/to/stage1-checkpoint.pt \
  --opponent-deck-root /path/to/deck-presets \
  --runtime-root /path/to/bionic-runtime \
  --run-root /path/to/new-canary-run
```

默认 `throughput` 配置自动执行：

```text
恢复 96 个 Worker（默认端口 19031–19126）
→ 6 个 collector × 16 场，同时准备已发布分片
→ 同版本 2 波共 192 场全部完成并关闭批次
→ 两遍 Guarded recurrent PPO
→ 保存 FP32 checkpoint + FP16 Actor
→ 回收本版本 CUDA 进程 / 重建 MPS，复用原生 Worker
→ 下一策略版本
```

只重叠同版本内的采集与教师/value/GAE 准备；PPO 更新仍须等待该批完整关闭。
它不是无版本约束的异步 Actor–Learner。

| 参数 | throughput（默认，已短测） | conservative |
| --- | --- | --- |
| 原生 Worker / collector | 96 / 6 | 48 / 6 |
| 每次更新场数 | 2 波 × 96 = 192 | 1 波 × 48 = 48 |
| 决策窗口 | 12 Tick / 600 ms | 4 Tick / 200 ms |
| Java 宿主 / MPS | JIT / 开启 | 解释器 / 关闭 |
| 训练精度 / fused AdamW | BF16 / 开启 | FP32 / 关闭 |
| chunk batch / padding | 32 / 80 | 8 / 不填充 |
| CPU 输入缓存上限 | 8 GiB | 4 GiB |
| 准备窗口 × batch | 128 × 2 | 256 × 3 |
| 准备重叠 / CUDA 跨版本隔离 | 开启 / 开启 | 关闭 / 关闭 |

这不是针对任意机器自动测出的最优配置。BF16/MPS 需要相应 Linux NVIDIA 环境。
改变决策窗口会改变策略响应分辨率；改变 batch 会改变更新频率，仍需对战评估。

Canary 完成后，将最新 FP16 Actor 下载到本地 RTX 3080：

```powershell
python scripts/attest_stage2_local_inference.py `
  --weights <latest-fp16.pt> `
  --expert-manifest <manifest.json> `
  --output <canary-root>/local-rtx3080-attestation.json
```

只有 attestation 的设备、有限数和权重 SHA256 与 Canary 最新导出完全匹配，
formal 入口才接受启动：

```bash
python scripts/start_expert_selfplay_stage2.py \
  --mode formal \
  --canary-root <completed-canary-root> \
  --base-checkpoint /path/to/expert-fp16.pt \
  --expert-manifest /path/to/manifest.json \
  --opponent-deck-root /path/to/deck-presets \
  --runtime-root /path/to/bionic-runtime
```

### 已完成 Stage-2 的续训

已经通过完整提交的 Stage-2 运行可用 `--resume-run`，而不是重复从 Stage 1 起步：

```bash
python scripts/start_expert_selfplay_stage2.py --mode formal \
  --resume-run /path/to/completed-stage2-run \
  --base-checkpoint /path/to/original-expert-fp16.pt \
  --expert-manifest /path/to/manifest.json \
  --opponent-deck-root /path/to/deck-presets \
  --runtime-root /path/to/bionic-runtime \
  --profile throughput --updates 100
```

`--base-checkpoint` 始终是冻结 BC 教师/BASE 对手，不是新的 behavior 导出。
入口从完成状态的 `progress.json` 读取与最后一次提交匹配的 checkpoint/behavior。
搬迁运行目录后，需确保其中记录的产物绝对路径仍有效；不能只复制 JSON。

formal 默认运行 100 个严格策略版本更新；每个版本必须先关闭采样批次再更新，
不接受 `v-1` rollout。旧 rollout 和 checkpoint 按已提交状态精确清理，磁盘
剩余低于 25 GiB 时停止并保留现场。

### 监测与停机

- 根目录 `progress.json`：已提交的更新、实际场数/决策数、耗时、最新产物。
- `events.jsonl`：批次关闭、Guard、提交、GPU 回收与失败证据。
- 隔离模式详细进度位于 `cycle-NNNNNN/progress.json`；采集和 learner 日志
  位于对应 cycle 的更新子目录，不能只看外层更新计数。
- 完成/失败只停止本次训练流程，**普通入口不会给云实例关机，也不保证停止计费**。
  原生 Worker 可能保留供下一次使用。关机需显式操作云控制台，或使用
  [有界效率试验的显式停机选项](STAGE2_EFFICIENCY_20260904.zh-CN.md#运行与成本控制)。

## Checkpoint 与失败恢复

Stage 2 checkpoint 包含 FP32 Actor master、canonical FP16 behavior Actor、
Critic、Actor/Critic Optimizer、参数名映射、Python/NumPy/Torch CPU/CUDA RNG、
policy version、global update、rollout hash、Guard 指标和重试记录。

每次更新前保留 `pre-update/checkpoint.pt`。未通过 Guard 时不会 commit rollout、
不会发布 FP16 Actor、不会让 Worker 切换到失败版本。

## 验证记录与限制

- 历史 Expert Self-Play/Stage 1/Stage 2/Worker/一键入口组合：78 项通过。
- Stage 2 小模型 CPU 端到端更新与 checkpoint 恢复：通过。
- 本地 RTX 3080 CUDA update smoke：通过。
- 不带 exact hidden 的 Stage 2 rollout：按预期拒绝。
- 两轮 fake formal 策略版本 `0→1→2`、global update `52→53→54`：通过。

后续云端已推进到 Stage-2；最新效率验收连续完成 policy `19→20→21`，
384 场、102,326 个有效决策，两次均首次通过 Guard，并保存完整 checkpoint 与导出。
本轮没有重跑本地 RTX 3080 GPU attestation，不能把旧验证当成最新导出的验证。

完整更新短测折算约 6.56 万场/天，尚无 24 小时稳定性或 20 万场/天结论。
本次整理的本地回归结果见 [发布说明](RELEASE_20260905.zh-CN.md)，
云端各阶段计时与失败试验见 [效率报告](STAGE2_EFFICIENCY_20260904.zh-CN.md)。
