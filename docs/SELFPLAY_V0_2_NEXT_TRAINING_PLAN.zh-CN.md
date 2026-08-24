# Self-Play v0.2 下一步训练方案

状态：规划完成，尚未实现、尚未训练。

## 1. 当前基线

### 沙盒

- 原版 `libg.so 15.535.29 x86_64`；
- 固定八卡、标准1v1、无觉醒/精英化；
- 原生移动、寻路、攻击、碰撞、伤害、圣水、手牌和终局；
- 20 Hz原生Tick；
- 2 AVD / 8 Worker；
- 原生18×32部署Validator；
- Persistent TCP、Compact Transition、Global Batch、CUDA Graph。

### v0.1模型

- CNN + public scalar + LSTM；
- `WAIT + 当前4张手牌` 五类Categorical Head；
- 每张牌576格Position Head；
- Privileged Critic；
- Recurrent PPO；
- Reward只有终局和塔血势函数。

已完成：

```text
1,033,302 native ticks
2,066,604 agent steps
224/224 正常终局
0 RPC failure
0 Worker failure
0 native action rejection
```

## 2. 已确认问题

20 Hz下每Tick重复采样 `WAIT + CARD`，使低费牌刚合法就快速被打出。

最后8局：

```text
巨人：在手 78,155 Tick，可下 14 Tick
火枪：在手 77,732 Tick，可下 13 Tick
野猪：在手 77,652 Tick，可下 22 Tick
平均圣水约 1
```

所以当前主要问题是动作时间语义，不应先修改Reward。

## 3. v0.2模型

每Tick仍然观察并更新LSTM：

```text
rate head λ(s)
      ↓
PLAY / NO-PLAY
      ↓ PLAY
card head
      ↓
position head
```

```text
λ = λ_max · sigmoid(z)
p_play = 1 - exp(-λ · 0.05)
```

策略每个Tick都保留主动行动权，不需要等待敌方先下牌，也没有不可取消的WAIT。

## 4. 第一阶段随机探索

测试：

```text
λ0 = 0.10 / 0.20 / 0.30 / 0.50 次/秒
```

每档使用固定环境seed，随机选择合法卡牌和合法落点，但：

- 不训练Actor；
- 不作为正式PPO on-policy数据；
- 默认不训练Critic；
- 只用于沙盒覆盖、Observation分布、卡牌机会和λ0选择。

选择λ0时检查：

- PLAY/sec；
- 高费牌获得的可下机会；
- 10费溢出；
- 整局无动作；
- 原生拒绝；
- 平均局长。

## 5. 初始化A/B

### Run A：Scratch主线

全部网络全新初始化，只给rate head写入选定λ0先验。

### Run B：Backbone Ablation

只迁移P010的：

```text
CNN
public scalar encoder
LSTM
privileged critic encoder
value head
```

rate/card/position Actor Head全部重新初始化，不迁移旧Optimizer。

两条Run使用完全相同的训练和评估seed，分别训练100k Tick，不能混合数据。

## 6. Opportunity-normalized Stats

每张卡记录：

```text
ticks_in_hand
legal_ticks
affordable_ticks
playable_ticks
selected_count
legal / in_hand
affordable / in_hand
playable / in_hand
selected / playable
elixir_at_play
```

这能区分经济、落点合法性和策略选择问题。

## 7. Position Head指标

保持原生576格，不加入桥头/塔后/中置等人工宏动作。

监控：

```text
H(position | legal mask)
H_normalized = H / log(valid_cells)
effective_cells = exp(H)
top1 / top5 probability mass
部署热图
```

## 8. 训练阶段

### Stage 0A：数学/迁移测试

- rate概率与数值稳定；
- 分层Log Probability；
- 精确条件Entropy；
- forced no-play不训练rate head；
- Scratch和Backbone迁移边界；
- Reward/GAE与v0.1一致。

### Stage 0B：无训练λ0 Sweep

选择初始行动率，不产生Actor梯度。

### Stage 1：10k Tick Smoke

```text
1 AVD / 1 Worker
```

验收工程、Checkpoint、RNG、主动出牌和机会指标。

### Stage 2：100k Tick A/B

```text
1 AVD / 4 Worker
Scratch vs Backbone-only
```

重点判断高费牌是否真正获得机会，以及λ是否坍缩到0或饱和。

### Stage 3：500k Tick学习门

只选Stage 2更可靠的一条Run：

```text
2 AVD / 8 Worker
```

评估：

- v0.2 vs P010；
- v0.2 vs P000；
- v0.2 vs RandomRateLegal；
- v0.2 Checkpoint Cross-Play。

不通过则停止，不直接训练2M/5M。

## 9. Opponent Pool

训练期间保存历史Checkpoint，但第一条v0.2曲线仍用Current-vs-Current，避免同时
改变两个变量。

500k行动率版本通过后，另开 `v0.2.1-opponent-pool` A/B：

```text
50% current
25% latest historical
25% uniform older historical
```

历史策略冻结，只训练current控制的一方，双方交换side。第一版不使用PFSP。

## 10. Reward边界

继续使用：

```text
terminal + 0.2 · tower HP potential shaping
```

暂不增加：

- 圣水Reward；
- 卡牌伤害；
- 击杀/过河；
- 卡牌使用率；
- 人工位置偏好。

先单独验证动作时间改造，再决定下一变量。
