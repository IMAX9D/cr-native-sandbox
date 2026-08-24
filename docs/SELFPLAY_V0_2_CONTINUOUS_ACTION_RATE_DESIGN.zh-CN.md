# Self-Play v0.2：20 Hz连续行动率设计

状态：**修订设计冻结候选，尚未实现、尚未训练**

替代：已否决的承诺式 `WAIT(N)` 设计

基线：Self-Play v0.1 / P010 / `1,033,302 native ticks`

## 1. 设计原则

一个皇室战争策略必须始终能够主动决定：

- 初始手牌不好时主动过牌；
- 为了试探对方卡组先下牌；
- 逼迫对手交出防守牌；
- 骗出关键卡后立刻转换进攻；
- 主动桥头施压；
- 为高费组合攒费；
- 对手不下牌时仍能发起行动；
- 任意Tick改变原计划。

因此v0.2不允许策略通过一个长WAIT动作放弃未来数秒的控制权，也不把“敌方下牌”
作为重新决策的前提。

冻结原则：

```text
原生Battle Core：20 Hz，每Tick固定0.05 s
策略Observation：每个原生Tick
策略Hidden更新：每个原生Tick
策略行动机会：每个原生Tick
Reward：v0.1终局 + 塔血势函数，完全不变
```

只改变“现在出牌”的概率参数化，使它按真实时间而不是按采样次数定义。

## 2. v0.1低费偏置

v0.1直接在每Tick采样：

```text
WAIT + 4个手牌槽
```

低费牌一旦合法，之后每秒有20次被随机选中的机会。若单Tick出牌概率为 `p`，
连续等待 `N` Tick的概率为：

```text
(1-p)^N
```

即使单次概率不大，等待数秒攒到4–5费的概率也会指数下降。

最后8局的机会归一化证据：

| 卡牌 | 在手Tick | 真正可下Tick | 打出 |
| --- | ---: | ---: | ---: |
| 巨人 | 78,155 | 14 | 2 |
| 火枪手 | 77,732 | 13 | 3 |
| 野猪骑士 | 77,652 | 22 | 3 |
| 骑士 | 19,560 | 433 | 180 |
| 加农炮 | 19,420 | 443 | 181 |

高费牌并非一直“可下但没选”，而是低费牌在圣水刚够时被连续抽样迅速消耗，
导致平均圣水长期约为1。

## 3. 核心方案：连续时间行动率

网络每个Tick输出一个行动率：

```text
λ(s) = λ_max · sigmoid(rate_logit(s))
单位：PLAY事件 / 原生秒
```

固定：

```text
Δt = 0.05 s
```

本Tick出牌概率：

```text
p_play(s, Δt) = 1 - exp(-λ(s) · Δt)
```

然后：

```text
play_now ~ Bernoulli(p_play)
```

如果本Tick决定出牌，再条件选择：

```text
card_slot ~ π_card(current hand, mask)
position  ~ π_position(card_slot, native mask)
```

没有任何可下手牌时，`play_now=false`为强制结果；该Tick仍更新Observation、RNN和
Value，但不把“无法行动”错误训练成策略主动等待。

## 4. 为什么它保留完整博弈空间

### 4.1 不依赖敌方动作

策略每Tick重新计算 `λ(s)`：

- 对手一直不出牌，策略仍可主动提高λ并过牌、试探或进攻；
- 对手刚出牌，下一Tick状态改变，策略可立刻提高λ进行反应；
- 自己圣水增加、卡牌变得合法时，λ可立刻变化；
- 场上单位、目标、塔血和手牌变化都会进入下一Tick Observation；
- 没有不可取消的WAIT承诺。

### 4.2 可表达攒费

当手里有关键高费组合时，策略可以把λ压到接近0；等圣水、手牌或场面达到目标，
再把λ迅速提高。

它学习的是“此状态下每秒多迫切地需要出牌”，而不是得到一个持有圣水Reward。

### 4.3 可表达主动过牌和骗牌

如果策略判断当前应主动过牌，可以在没有敌方动作的情况下提高λ，并由card head
选择低费牌；如果目标是骗出特定防守牌，同样是正常的主动PLAY决策。

## 5. Tick频率不变性

当状态近似不变、行动率为常数λ时，在时间 `T` 内一直不出牌的概率为：

```text
P(no play for T) = exp(-λT)
```

把同一时间切成更多Tick不会改变结果：

```text
[exp(-λT/m)]^m = exp(-λT)
```

这消除了“20 Hz比5 Hz拥有四倍随机出牌机会”的离散采样偏置。

策略仍然每Tick观察，所以响应能力不会因该修复下降。

## 6. 分层网络

共享Backbone保持：

```text
spatial encoder
public scalar encoder
LSTM
privileged critic encoder
value head
```

Actor改为：

```text
rate_head      → 1个rate_logit
card_head      → 4个当前手牌槽
position_head  → 每槽18×32原生合法格
```

旧的5类 `WAIT + 4手牌槽` card head不再存在。

## 7. 动作Mask

每个Tick先构建原有的最终训练Mask：

```text
card_mask[4]
position_mask[4, 576]
```

再计算：

```text
timing_choice_valid = any(card_mask)
```

规则：

- `timing_choice_valid=false`：强制不出牌，rate log-prob记0，rate entropy不进Loss；
- `timing_choice_valid=true`：采样Bernoulli行动时机；
- `play_now=true`：card head只保留合法手牌；
- position head继续由原生Validator和最终部署层约束；
- 原生命令门关闭时，所有card均不可选；
- 所有原生拒绝仍fail-closed。

这样不会用大量“圣水不足时被迫WAIT”的样本训练rate head。

## 8. Log Probability

有合法牌且本Tick不出牌：

```text
log π(a|s) = log(1 - p_play)
```

有合法牌且本Tick出牌：

```text
log π(a|s)
= log(p_play)
+ log π_card(slot|s)
+ log π_position(cell|slot,s)
```

没有合法牌：

```text
log π(a|s) = 0
timing_loss_mask = false
```

PPO必须保存 `play_now`、`timing_choice_valid`、card和position，并使用新策略重算
同一联合Log Probability。

## 9. Entropy

有合法牌时使用完整分层Entropy：

```text
H = H(Bernoulli(p_play))
  + p_play · [H(card) + E_card H(position | card)]
```

没有合法牌时，Actor entropy为0；Value仍正常训练。

位置Entropy对全部合法card按card概率求期望，不只统计本次采样到的手牌。

## 10. λ范围与数值稳定

第一版候选：

```text
λ_max = 20 PLAY events / second
```

因此最大单Tick出牌概率约为：

```text
1 - exp(-20 · 0.05) ≈ 0.6321
```

允许策略在紧急状态下平均约1–2 Tick内反应，同时避免概率精确等于1造成
Log Probability和KL数值问题。

实现必须使用稳定形式：

```text
log P(no play) = -λΔt
log P(play)    = log(-expm1(-λΔt))
```

禁止直接计算 `log(1-exp(...))` 导致小λ精度损失。

## 11. 初始行动率

不直接冻结一个拍脑袋先验。Stage 0对以下每秒行动率做无训练原生Rollout：

```text
λ0 ∈ {0.10, 0.20, 0.30, 0.50}
```

对应rate head bias：

```text
bias = log(λ0 / (λ_max - λ0))
```

选择标准不是胜率，而是：

- 每方实际PLAY/秒与v0.1同数量级；
- 低费牌不再刚合法就必然打出；
- 巨人、火枪手、野猪获得真实可下机会；
- 不出现整局不行动；
- 不产生明显10费溢出常态。

选定值写入Manifest，训练中不修改。

Stage 0的随机card/position只用于沙盒覆盖、Observation分布和机会统计，**不进入
正式Actor的PPO on-policy数据**。默认也不训练Critic；Critic warm-up如需验证，
必须作为独立ablation。

## 12. 初始化A/B实验

### 12.1 主线：全新初始化

第一条正式v0.2曲线使用全新初始化：

- CNN、公开scalar encoder、LSTM、Actor和Critic全部新建；
- rate head使用Stage 0选定的λ0先验；
- card/position head使用标准随机初始化；
- 新Optimizer、新RNG、新Run ID和新Checkpoint kind。

这条曲线最干净地判断行动率参数化本身能否修复低费偏置。

### 12.2 Ablation：只迁移共享表示

另开独立Run，在完全相同seed和预算下，只复制P010：

```text
spatial encoder
public scalar encoder
LSTM
privileged critic encoder
value head
```

不复制任何旧Actor head：

```text
不复制旧WAIT/card head
不复制position map/context
```

rate/card/position head全部新建。这样可以区分“共享表示迁移收益”和“继承v0.1
低费/塔后位置偏置”。两条Run都不迁移旧Optimizer。

两条Run的rate head均为：

```text
weights = 0
bias = Stage 0选定的λ0
```

不扩展Observation scalar：RNN仍逐Tick更新，当前Observation已经包含时间、手牌、
圣水、实体、塔血和上一Tick公开下牌事件。

## 13. Reward和GAE

Reward完全保持：

```text
r_t = z_t + 0.2(γΦ(s_(t+1)) - Φ(s_t))

z_terminal = +1 / 0 / -1
γ = 0.99995
Φ = 双方归一化皇冠塔剩余总HP之差
```

Trajectory仍然每个原生Tick一行，因此：

- 不需要宏Reward；
- 不需要变时长折扣；
- GAE公式保持v0.1；
- burn-in和train length时间含义保持v0.1；
- 原生transition仍是一Tick一次；
- Reward差分测试可直接复用。

这是相对承诺式WAIT方案的重要简化和风险降低。

## 14. Trajectory Schema

新增：

```text
play_now: bool
timing_choice_valid: bool
rate_logit: float（调试可选）
play_probability: float（调试可选）
card_slot: int 0..3（未出牌时填0并由mask忽略）
position: int 0..575
```

保留：

```text
grid/scalars/privileged
card_masks/position_masks
log_probability/value/reward/done
hidden_h/hidden_c
```

Checkpoint类型必须与v0.1不同。

## 15. 机会归一化行为指标

### 15.1 每张卡

```text
native ticks in hand
native ticks with legal deployment cells
native ticks affordable
native ticks playable
selected plays
legal / in_hand
affordable / in_hand
playable / in_hand
selected / playable
elixir at play
average hand hold duration
```

### 15.2 行动率

```text
λ mean / p50 / p95 / p99
p_play mean by elixir
p_play mean by playable card set
PLAY events per native second
no-play survival duration distribution
time from enemy play to response
time from card becoming legal to play
```

### 15.3 Position Head

每张当前合法的牌记录：

```text
H(position | legal mask)
H_normalized = H / log(number_of_legal_cells)
effective_cells = exp(H)
top1 / top5 position probability mass
```

只监控原生18×32空间，不加入桥头、塔后、中置等人工宏动作。

### 15.4 反坍缩

```text
整局无PLAY比例
10费停留比例
平均圣水分布
高费牌可下率
八卡零使用次数
Draw rate
平均局长
native rejection
```

全部只用于诊断和Early Stop，不进入Reward。

## 16. 为什么不把卡牌使用率设成目标

合理策略不保证八卡均匀使用：

- 某些对局巨人可以少用；
- 某些Matchup火枪手价值更高；
- 骷髅兵可能承担过牌与拉扯；
- 箭雨可能因对方阵容而长期保留。

验收只要求策略获得真实选择机会，不要求固定比例。

## 17. 代码隔离

新建独立包：

```text
selfplay_v2/
  action.py          # rate→per-tick概率与分层动作
  model.py           # rate/card/position recurrent policy
  migrate.py         # P010模型迁移
  rollout.py         # 20Hz collector
  ppo.py             # 分层LogProb/Entropy PPO
  metrics.py         # 机会归一化指标
  train.py
  evaluate.py
```

复用稳定组件：

- 原生Env和persistent RPC；
- compact one-tick transition；
- 原生部署Validator和最终Mask；
- v0.1 Observation空间语义；
- 塔血势函数Reward；
- RunStore、Resource Monitor和浏览器Dashboard。

数据根目录：

```text
D:\AI_data\cr-native-core\selfplay-v0.2
```

## 18. Checkpoint与Manifest

新类型：

```text
native_eight_card_continuous_rate_ppo_v2_checkpoint
```

Manifest冻结：

```text
native_tick_seconds = 0.05
rate_parameterization = bounded_poisson_hazard_v1
lambda_max
lambda_initial
forced_no_play_contract
hierarchical_logprob_contract
hierarchical_entropy_contract
initialization_mode = scratch / backbone_only
warm_start_parent_digest = null / P010 digest
reward_contract = tower_hp_only_v1
```

任何λ参数化、上限、先验、Mask或Log Probability变化都必须开新Run。

## 19. 数学测试

### 19.1 时间切分不变性

对随机λ、T和切分数m验证：

```text
exp(-λT) == [exp(-λT/m)]^m
```

### 19.2 概率与Log Probability

- `p_play`严格位于 `[0, 1)`；
- λ接近0时 `log P(play)`有限；
- no-play与play概率和为1；
- 采样Log Probability与PPO重算一致；
- 强制no-play的rate loss mask为false。

### 19.3 分层Entropy

小动作空间枚举全部联合动作，证明解析Entropy与枚举值一致。

### 19.4 v0.1退化边界

Reward、Value target、GAE、Native transition、state hash和RNN reset保持原证书。

## 20. 迁移测试

1. Scratch Run不存在任何从P010复制的参数组；
2. Backbone Ablation仅允许共享层逐Tensor bit-exact；
3. 两条Run的rate/card/position Actor head均为新初始化；
4. rate weights全0，bias对应Manifest λ0；
5. 旧Optimizer不能加载；
6. P010文件和digest只读；
7. 新Run不能写入v0.1目录。

## 21. 原生与Collector测试

1. 双方仍从同一个Tick状态决定；
2. 双方同TickPLAY仍按side0→side1执行；
3. 对手不下牌时，本方仍能任意Tick主动PLAY；
4. 对手下牌后，下一Tickrate可以变化；
5. card变合法后，下一Tickrate可以变化；
6. 20Hz tick、RNG、手牌、圣水和实体与v0.1环境一致；
7. 所有从Mask采样的动作原生接受；
8. Reset后双方Hidden独立归零；
9. CUDA Graph和eager采样语义一致；
10. 固定policy RNG可复现完整动作序列。

## 22. 分阶段验收

### Stage 0A：纯数学与迁移

```text
0 native training ticks
全部概率/Entropy/PPO/迁移测试
```

### Stage 0B：λ0无训练原生Sweep

对 `{0.10, 0.20, 0.30, 0.50}` 每档跑固定种子完整对局，不更新任何Actor或
Critic参数。

选择能够同时满足以下条件的先验：

- PLAY/sec处于可用范围；
- 三张高费牌可下率显著高于v0.1；
- 无整局静止；
- 10费溢出不过高；
- 0原生拒绝。

### Stage 1：10k Tick Smoke

```text
1 AVD / 1 Worker
新Optimizer
```

要求：

- 0 NaN/Inf；
- 0 Worker/RPC/动作失败；
- rate head参数确实变化；
- Checkpoint/RNG可恢复；
- 八卡机会指标完整；
- 主动PLAY在无敌方动作条件下出现。

### Stage 2：100k Tick初始化A/B行为门

```text
1 AVD / 4 Worker
```

Scratch与Backbone Ablation使用相同环境seed、训练预算和评估seed。分别要求：

- 巨人、火枪手、野猪不再全部长期不可下；
- 高费牌“在手时可下率”明显高于0.02%–0.03%基线；
- 不出现λ→0永久等待；
- 不出现λ→λ_max低费狂刷；
- 平局率和局长无异常；
- Reward、Value和Entropy健康。

失败则停止，不机械扩大样本。若两者都通过，用固定对战和行为稳定性选择一条进入
Stage 3，不能混合两条Run的数据。

### Stage 3：500k Tick学习门

```text
2 AVD / 8 Worker
```

固定种子交换side：

- v0.2 vs P010；
- v0.2 vs P000；
- v0.2 vs连续时间RandomLegal；
- v0.2 Checkpoint Cross-Play。

行为恢复不等于强度提升，必须用对战证明。

## 23. Continuous RandomLegal

旧RandomLegal每Tick均匀采样同样带有低费Hazard。新增公平基线：

```text
RandomRateLegal
```

它使用冻结λ、同一泊松行动过程，并在PLAY发生时从合法手牌和落点均匀采样。

固定独立policy RNG并交换side。

## 24. Opponent Pool版本边界

从Stage 0到Stage 3的第一条v0.2学习曲线仍使用Current-vs-Current，以便把变化
单独归因到行动率改造。训练过程中立即保存不可变历史候选，但暂不用于采样。

行动率版本通过500k学习门后，另开独立 `v0.2.1-opponent-pool` Run：

```text
50% current vs current
25% current vs latest historical
25% current vs uniform older historical
```

双方交换side；只更新current policy控制的一方，历史对手冻结且不产生梯度。
第一版不使用PFSP，先与纯Current-vs-Current做同预算A/B。Opponent Pool、Reward、
网络结构和PPO不能在同一个实验里同时改变。

## 25. 风险

| 风险 | 表现 | 处理 |
| --- | --- | --- |
| λ坍缩到0 | 永久等待、平局 | 行为Early Stop，不加Reward |
| λ长期饱和 | 低费狂刷 | λ分布告警与行为门 |
| forced WAIT污染 | 无牌可下也训练rate | timing loss mask关闭 |
| 评估不确定 | 随机出牌时机噪声 | 固定policy RNG、paired seeds、换side |
| 位置Entropy昂贵 | 4×576全条件计算 | Profile后优化，不先近似 |
| P010迁移伪兼容 | 旧Optimizer被加载 | 新Checkpoint kind与强校验 |
| 当前自博弈局限 | 双方形成相同局部策略 | 先过行为门，后评估历史对手池 |

## 26. 实施顺序

```text
1. 新建selfplay_v2独立包
2. 实现rate概率与稳定Log Probability
3. 实现分层动作和精确Entropy测试
4. 实现Scratch初始化和Backbone-only迁移器
5. 扩展Trajectory和机会归一化Metrics
6. 实现单Worker 20Hz collector
7. 实现Recurrent PPO update
8. 实现多Worker global batch
9. 接入Checkpoint、Evaluation和Dashboard
10. Stage 0A
11. Stage 0B λ0 Sweep
12. Stage 1 10k
13. Stage 2 100k
14. 根据行为证据决定是否Stage 3
15. Stage 3通过后再做Opponent Pool独立A/B
```

## 27. 最终边界

本设计只修复“动作概率随Tick采样次数放大”的结构偏置。

它不：

- 给圣水、伤害或卡牌使用率发Reward；
- 降低原生20Hz；
- 禁止主动过牌；
- 要求等待敌方先下牌；
- 强制高费牌比例；
- 保证单靠该机制就达到高水平；
- 在行为门失败后继续烧到2M/5M。

策略在每个Tick都保留完整行动权，真正的等待、过牌、试探和逼牌由胜负与塔血结果
自行学习。
