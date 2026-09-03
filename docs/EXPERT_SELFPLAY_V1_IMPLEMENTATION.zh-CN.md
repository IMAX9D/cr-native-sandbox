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

新采集合同会在每个 learner 决策保存实际使用的 LSTM pre-action
`hidden/cell`。云端 learner 会：

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

云端 Canary：

```bash
cd /root/autodl-tmp/expert-selfplay-v1
python scripts/start_expert_selfplay_stage2.py --mode canary
```

该命令自动执行：

```text
恢复 48 个 Worker
→ 3 个 collector × 16 场并行采集
→ 关闭 48 场 policy-version batch
→ Guarded recurrent PPO
→ 发布 FP32 checkpoint + FP16 Actor
→ 串联下一策略版本
```

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
  --canary-root <completed-canary-root>
```

formal 默认运行 100 个严格策略版本更新；每个版本必须先关闭采样批次再更新，
不接受 `v-1` rollout。旧 rollout 和 checkpoint 按已提交状态精确清理，磁盘
剩余低于 25 GiB 时停止并保留现场。

## Checkpoint 与失败恢复

Stage 2 checkpoint 包含 FP32 Actor master、canonical FP16 behavior Actor、
Critic、Actor/Critic Optimizer、参数名映射、Python/NumPy/Torch CPU/CUDA RNG、
policy version、global update、rollout hash、Guard 指标和重试记录。

每次更新前保留 `pre-update/checkpoint.pt`。未通过 Guard 时不会 commit rollout、
不会发布 FP16 Actor、不会让 Worker 切换到失败版本。

## 当前验证

- Expert Self-Play/Stage 1/Stage 2/Worker/一键入口组合：78 项通过。
- Stage 2 小模型 CPU 端到端更新与 checkpoint 恢复：通过。
- 本地 RTX 3080 CUDA update smoke：通过。
- 不带 exact hidden 的 Stage 2 rollout：按预期拒绝。
- 两轮 fake formal 策略版本 `0→1→2`、global update `52→53→54`：通过。

尚待真实 RTX 5090 完成三轮 177M Canary；通过后再执行本地 RTX 3080
attestation，并由 formal 一键入口启动长跑。
