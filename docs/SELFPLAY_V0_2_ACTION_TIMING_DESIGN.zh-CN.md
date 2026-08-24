# Self-Play v0.2：可变等待动作设计

状态：**已否决，不得实现**

否决原因：承诺式 `WAIT(N)` 会让策略在等待期间放弃主动控制，不能完整表达主动
过牌、试探、逼牌、骗牌、突然进攻和随时改变计划的实时博弈。替代方案见
[`SELFPLAY_V0_2_CONTINUOUS_ACTION_RATE_DESIGN.zh-CN.md`](SELFPLAY_V0_2_CONTINUOUS_ACTION_RATE_DESIGN.zh-CN.md)。

基线：Self-Play v0.1 / P010 / `1,033,302 native ticks`  
目标：消除20 Hz逐Tick随机采样导致的低费循环，同时保持原生战斗逻辑20 Hz、
原生规则、终局目标和塔血势函数不变。

## 1. 结论

v0.2不继续在v0.1的5类动作头上打补丁，也不加入圣水、卡牌伤害、击杀、
过河或场面价值Reward。

策略动作改为分层、半马尔可夫动作：

```text
PLAY(card_slot, position)

WAIT(duration_ticks)
duration_ticks ∈ {1, 5, 10, 20, 40, 80}
```

原生 `libg` 仍逐个执行固定 `0.05 s` Tick。WAIT只减少策略重新采样次数，
不跳过、不近似、不重写任何战斗逻辑。

v0.2必须开新Run、新Manifest和新Checkpoint类型。P010只作为网络Warm Start，
不能把新动作语义写回v0.1 Run，也不能恢复v0.1 Optimizer。

## 2. v0.1问题的实测证据

### 2.1 不是最近一轮偶然波动

v0.1共28轮。每轮平均圣水都在 `0.99–1.02`；巨人、火枪手、野猪骑士的
使用率从第1轮起就长期低于约1%，没有随训练规模增加而恢复。

最后8局按“在手”和“真正可下”重新统计：

| 卡牌 | 在手Tick | 可下Tick | 打出 | 在手时可下率 | 可下时选择率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 骑士 | 19,560 | 433 | 180 | 2.21% | 41.57% |
| 弓箭手 | 19,449 | 393 | 182 | 2.02% | 46.31% |
| 巨人 | 78,155 | 14 | 2 | 0.02% | 14.29% |
| 骷髅兵 | 6,199 | 374 | 186 | 6.03% | 49.73% |
| 火枪手 | 77,732 | 13 | 3 | 0.02% | 23.08% |
| 野猪骑士 | 77,652 | 22 | 3 | 0.03% | 13.64% |
| 加农炮 | 19,420 | 443 | 181 | 2.28% | 40.86% |
| 箭雨 | 18,885 | 361 | 183 | 1.91% | 50.69% |

高费牌并非“可下但策略拒绝”，而是长期卡在手中，几乎从未积累到所需圣水。

### 2.2 20 Hz动作Hazard

v0.1每个原生Tick都从以下分布重新采样：

```text
WAIT + 当前4张手牌
```

当一张低费牌刚变为合法时，策略在之后每个Tick都有一次把它打出的机会。
如果单Tick打出概率为 `p`，连续等待 `N` Tick的概率是：

```text
P(wait N ticks) = (1 - p)^N
```

即使 `p` 不高，20 Hz下等待数秒的概率也会指数下降。实测低费牌一旦可下，
选择率达到约41%–51%，所以圣水几乎不可能继续增长到4–5费。

这是动作时间参数化造成的结构偏置，不是加一项Reward能可靠修复的问题。

## 3. 不采用的方案

### 3.1 不增加“持有圣水”Reward

可能诱发：

- 永远囤到10费；
- 为维持高圣水拒绝合理防守；
- 双方共同等待并进入平局；
- 把优化目标从胜负改成资源数字。

### 3.2 不增加逐卡伤害Reward

会错误评价拉扯、挡伤、吸引目标、逼迫走位和法术价值，并可能产生伤害刷分、
延长对局、双方喂单位等Reward Hacking。

### 3.3 不固定降为5 Hz或10 Hz

固定降频只能线性减少采样机会，仍不能让策略表达“我要等待4秒攒费”；同时会
永久降低紧急反应精度。

### 3.4 不给高费牌设置最低使用率Reward

牌组不存在正确的固定使用比例。强行追求均匀使用会让策略在错误时机下牌。

## 4. v0.2动作空间

### 4.1 分层Actor

共享Backbone和LSTM之后，Actor分成：

```text
mode_head      → WAIT / PLAY
wait_head      → 1 / 5 / 10 / 20 / 40 / 80 ticks
card_head      → 当前4个手牌槽
position_head  → 被选手牌的18×32原生合法格
```

合法性：

- WAIT始终合法；
- 没有任何可下手牌时，PLAY被Mask；
- card head只保留当前圣水和原生部署Mask都合法的槽；
- position head继续使用当前原生网格和适配层最终Mask；
- 原生命令门关闭时，PLAY被Mask；
- 不增加任何Python战斗规则。

### 4.2 Log Probability

WAIT动作：

```text
log π(a|s)
= log π_mode(WAIT|s)
+ log π_wait(duration|s)
```

PLAY动作：

```text
log π(a|s)
= log π_mode(PLAY|s)
+ log π_card(slot|s)
+ log π_position(cell|slot,s)
```

PPO必须保存并重算完整联合Log Probability，不能只训练mode head。

### 4.3 Entropy

使用分层分布的条件Entropy：

```text
H = H(mode)
  + P(WAIT) · H(wait_duration)
  + P(PLAY) · [H(card) + E_card H(position | card)]
```

这样不会因为当前分支未被采样就漏掉另一分支的探索量。

### 4.4 WAIT档位

第一版冻结候选：

| 档位 | Tick | 原生时间 |
| --- | ---: | ---: |
| W0 | 1 | 0.05 s |
| W1 | 5 | 0.25 s |
| W2 | 10 | 0.50 s |
| W3 | 20 | 1.00 s |
| W4 | 40 | 2.00 s |
| W5 | 80 | 4.00 s |

理由：

- `WAIT(1)`保留逐Tick精确反应能力；
- 20/40 Tick允许正常短期观察和攒费；
- 80 Tick允许明确的资源规划；
- 最大4秒，避免一次动作放弃过长控制权；
- 多次WAIT可以组合出任意更长等待。

档位必须在Stage 0/1由行为数据验证，不依据主观感觉继续扩展。

## 5. WAIT中断语义

双方WAIT倒计时完全独立。

初版只允许一种外部中断：

```text
敌方成功部署卡牌 → 等待方在下一原生Tick重新获得决策权
```

不会让等待方在同一Tick看到敌方动作后反向提交，仍保持双方同Tick信息对称。

初版不使用“实体变化”“目标变化”或“任意塔受伤”中断，因为这些事件频率高，
可能重新退化为逐Tick决策。策略面对已有场面威胁时应主动选择短WAIT。

终局和原生命令门事件始终可以提前结束等待。

## 6. 多Worker异步调度

### 6.1 单场状态

每场、每方独立保存：

```text
remaining_wait_ticks
pending_action
pending_start_state
pending_start_tick
pending_value
pending_log_probability
hidden_h / hidden_c
interrupted_by_enemy_play
```

### 6.2 一轮调度

1. 收集所有 `remaining_wait_ticks == 0` 的方；
2. 将不同Worker、不同方统一组成全局Inference Batch；
3. 对到期方采样PLAY或WAIT；
4. 同一场双方若都PLAY，仍生成原生 `side0 → side1` joint action；
5. 每场计算下一次到期的最小Tick数；
6. 各Worker并行调用一次原生宏步进；
7. 更新双方倒计时；
8. 若某方观察到敌方成功部署，将其倒计时设为0，在下一Tick决策；
9. 到期或终局时完成该方的Pending Transition。

不同Worker可以请求不同的step数量。全局Barrier仍等待最慢Worker，但RPC和推理
数量预计显著下降。

### 6.3 PLAY后的间隔

PLAY动作固定产生 `duration=1` 的Transition。该方最早在下一原生Tick再次决策。
不会在同一Tick重复提交两张牌。

## 7. 原生宏步进接口

新增训练专用、保持Debug API不变的：

```text
joint_training_transition_v2(actions, max_steps)
```

返回至少包含：

```text
joint_action
requested_steps
completed_core_updates
tick_before
tick_after
episode
next compact state（非终局）
timing_v2（可选）
```

JNI内部仍循环调用冻结的原生Battle Core，每次固定 `0.05f`。接口只合并RPC，
不能调用一次更大的 `dt`，也不能用Python外推中间战斗状态。

必须证明：

```text
WAIT(N)一次宏RPC的最终state hash
==
WAIT(1)连续N次RPC的最终state hash
```

## 8. 变时长Reward

Reward定义保持不变：

```text
r_t = z_t + α(γ Φ(s_{t+1}) - Φ(s_t))

z_terminal = win +1 / draw 0 / loss -1
α = 0.2
γ = 0.99995
Φ = 双方归一化皇冠塔剩余总HP之差
Φ(terminal absorbing state) = 0
```

对于跨越 `n` 个原生Tick的宏动作，保存折扣聚合Reward：

```text
R_t^(n) = Σ[k=0..n-1] γ^k r_(t+k)
```

由于中间Reward只有势函数差和最后可能出现的终局Reward，可直接精确化简：

```text
R_t^(n)
= γ^(n-1) · z_terminal_if_any
+ α(γ^n Φ(s_(t+n)) - Φ(s_t))
```

非终局时 `z=0`；终局时 `Φ(s_(t+n))=0`。

这不是新的Reward，只是把原有逐TickReward按相同γ精确折叠。

## 9. SMDP GAE与PPO

Trajectory新增：

```text
mode
wait_index
card_slot
position
duration_ticks
macro_reward
discount = γ^duration_ticks
trace_discount = (γ·λ)^duration_ticks
interrupted
```

变时长TD误差：

```text
δ_i = R_i
    + γ^(n_i) (1-done_i) V(s_(i+1))
    - V(s_i)
```

保持20 Hz时间语义的GAE：

```text
A_i = δ_i
    + (γ·λ)^(n_i) (1-done_i) A_(i+1)
```

不能错误地对每个宏动作只乘一次固定γ，否则WAIT(80)与WAIT(1)会拥有不同的
时间偏好并改变目标。

PPO clip、value coefficient、entropy coefficient和gradient clip第一轮保持不变；
序列长度需要按“决策步”重新标定，不能直接宣称v0.1的64/256仍等价。

Stage 1候选参数：

```text
burn_in_decisions = 16
train_length_decisions = 64
```

只有Profile证明不合适时再调整。

## 10. 网络迁移

### 10.1 从P010复制

逐值复制：

```text
spatial encoder
public scalar encoder
LSTM
privileged critic encoder
value head
position map
position context
```

旧 `card_head[1:5]` 复制到新4类card head。旧WAIT行丢弃。

### 10.2 新Head初始化

```text
mode_head weights = 0
mode prior = WAIT 75% / PLAY 25%

wait_head weights = 0
wait duration prior = 六档均匀
```

六档均匀WAIT的均值为26 Tick；计入25%的PLAY(duration=1)后，先验对应约
0.25次PLAY/秒/方，接近v0.1实际约0.23次/秒/方，但不会在每个低费合法Tick
反复抽签。

该先验属于动作时间参数化，不计入Reward。Manifest必须明确记录。

### 10.3 不迁移Optimizer

动作头形状和时间语义已经改变：

- 新建AdamW Optimizer；
- 不伪造v0.1 optimizer moments；
- 新建RNG seed和Run ID；
- P010记录为 `warm_start_parent`，不是resume checkpoint。

## 11. Observation与RNN

Actor仍只接收公开信息，Critic-only信息边界不变。

v0.2在公开scalar末尾追加：

```text
previous_duration_ticks / 80
previous_wait_interrupted（0/1）
```

扩展后的第一层权重从P010复制，新增列初始化为0。

LSTM只在该方真正决策时更新。WAIT期间不虚构中间Hidden，也不把未来状态倒灌。
下一次决策直接编码当前完整公开状态；敌方下牌中断时，公开动作事件和实体状态
都已存在。

这是v0.2的显式语义变化，必须通过消融评估RNN是否仍有价值。

## 12. 代码隔离

避免在v0.1文件上继续堆条件分支。新增独立包：

```text
selfplay_v2/
  action.py          # mode/wait/card/position动作契约
  schema.py          # v2 observation和trajectory schema
  model.py           # timed recurrent policy/value net
  migrate.py         # P010 → v0.2权重迁移
  rollout.py         # 异步双方倒计时与宏步进collector
  ppo.py             # variable-duration GAE/PPO
  train.py           # 新Run入口
  evaluate.py        # v2/v1混合时序评估
  metrics.py         # 机会归一化行为指标
```

允许复用：

- `native_core.env`稳定通信层；
- 原生部署Validator和Mask；
- `ObservationEncoder`的空间语义；
- RunStore的原子写入模式；
- Resource Monitor和浏览器Dashboard。

禁止复用v0.1 Trajectory/Checkpoint类型冒充兼容。

数据根目录：

```text
D:\AI_data\cr-native-core\selfplay-v0.2
```

## 13. Checkpoint与Manifest

新Checkpoint类型：

```text
native_eight_card_timed_recurrent_ppo_v2_checkpoint
schema_version = 1
```

额外冻结：

```text
wait_durations
interrupt_contract
mode_prior
wait_prior
macro_reward_contract
variable_discount_contract
observation_timing_features
warm_start_parent_digest
```

v0.2不能加载v0.1 Optimizer。任何WAIT档位、事件中断、折扣公式或Mask变化都必须
开新Run ID。

## 14. 必须增加的行为指标

### 14.1 卡牌机会归一化

每张卡记录：

```text
hand_decisions
playable_decisions
selected_plays
selected / playable
native ticks held in hand
native ticks playable
elixir at play
```

不再只显示“占全部出牌比例”。

### 14.2 时间动作

```text
WAIT各档使用率
平均/中位/最大决策间隔
每方每秒PLAY次数
敌方下牌中断次数
中断后反应Tick数
宏RPC平均/分位step数
native ticks / policy decision
```

### 14.3 反坍缩

```text
无动作整局比例
WAIT(80)比例
平均圣水和圣水分布
高费牌可下率
高费牌联合使用率
八卡零使用次数
平均局长/平局率
```

这些指标只触发诊断或Early Stop，不进入Reward。

## 15. 差分与单元测试

### 15.1 纯数学测试

1. 变时长Reward与逐Tick累积逐值一致；
2. 塔血势函数在宏步进下仍严格反对称；
3. variable-duration GAE与手算结果一致；
4. `duration=1`退化为v0.1公式；
5. 终局发生在宏动作末尾时，终局Reward折扣位置正确；
6. WAIT和PLAY联合Log Probability重算一致；
7. 分层Entropy与枚举分布一致；
8. 所有Mask组合至少保留WAIT。

### 15.2 权重迁移测试

1. 共享层逐Tensor bit-exact；
2. card head四行逐值等于P010旧行1..4；
3. 新scalar列权重为0；
4. mode/wait head只符合Manifest先验；
5. P010和迁移模型在强制PLAY同一手牌时，position logits一致。

### 15.3 调度测试

1. 双方WAIT倒计时独立；
2. 双方同Tick PLAY仍按side0→side1提交；
3. 一方PLAY后，另一方在下一Tick被中断；
4. 无敌方动作时WAIT准确到期；
5. Worker提前终局不会留下Pending Transition；
6. Reset后倒计时、Hidden和Pending全部清零；
7. 不同Worker宏步长不会串局。

### 15.4 原生差分测试

对多个seed和场面证明：

```text
macro WAIT(N) vs sequential WAIT(1)×N

tick
state hash
RNG
players/hand/elixir
entities
crown towers
episode
```

全部相等。

另需证明：

- PLAY + step(1)与v0.1 joint transition一致；
- WAIT期间原生时间、×2/×3圣水、加时和拼血不变；
- Macro API异常时fail-closed，不退回Python模拟。

## 16. 分阶段验收

### Stage 0：离线与短原生差分

```text
0训练
全部单元测试
至少100组宏步进/逐Tick state-hash差分
```

通过条件：0语义差异。

### Stage 1：10k native ticks Smoke

```text
1 AVD / 1 Worker
P010 Warm Start
新Optimizer
```

通过条件：

- 0 NaN/Inf；
- 0 RPC/Worker/原生拒绝；
- duration=1和长WAIT都实际出现；
- 所有Pending Transition闭合；
- Checkpoint可恢复；
- Reward宏折叠证书通过。

### Stage 2：100k native ticks 行为门

```text
1 AVD / 4 Worker
```

行为门是诊断门，不是Reward：

- 巨人/火枪手/野猪不再全部长期不可下；
- 三张高费牌至少都有实际合法机会和实际使用；
- 高费牌“在手时可下率”明显高于v0.1的0.02%–0.03%基线；
- 不允许WAIT(80)占比接近100%；
- 不允许整局无动作；
- 平均局长和Draw rate不能异常恶化。

若失败，不进入大规模训练。

### Stage 3：500k native ticks 学习门

```text
2 AVD / 8 Worker
```

固定种子、交换side评估：

- v0.2 vs P010；
- v0.2 vs P000；
- v0.2 vs RandomTimedLegal；
- v0.2不同Checkpoint Cross-Play。

只有行为恢复、工程稳定且强度无显著退化，才考虑累计2M。

## 17. Random Baseline

保留v0.1 RandomLegal用于横向证据，但新增：

```text
RandomTimedLegal
```

它必须：

- 严格遵守手牌、圣水、原生Mask；
- 随机选择WAIT/PLAY；
- WAIT duration从冻结分布采样；
- 使用固定独立RNG；
- 双方交换side。

否则逐TickRandomLegal本身也具有低费Hazard，不能作为v0.2公平基线。

## 18. 风险与对策

| 风险 | 表现 | 对策 |
| --- | --- | --- |
| 永久等待 | WAIT(80)接近100%、平局增加 | 行为Early Stop，不加Reward |
| 反应过慢 | 对敌方下牌无响应 | 下一Tick中断WAIT |
| 折扣错误 | 长WAIT被高估或低估 | SMDP公式与逐Tick差分 |
| 异步串局 | Hidden/Pending跨方或跨Worker | 独立状态与Reset测试 |
| 迁移伪兼容 | 加载旧Optimizer或旧动作头 | 新Checkpoint kind、新Run |
| 宏步进改规则 | 大dt或Python推演 | JNI逐次0.05f + state-hash证书 |
| 卡牌均匀化误导 | 为达到比例乱下牌 | 使用率只诊断，不进Reward |
| 吞吐虚高 | 只报native step | 同时报告ticks/s、decisions/s和PPO闭环 |

## 19. 实施顺序

```text
1. 新建 selfplay_v2 独立包
2. 定义Action/Trajectory/Checkpoint schema
3. 实现变时长Reward与GAE单元测试
4. 实现分层模型与P010迁移测试
5. 扩展compact macro transition元数据
6. 实现单Worker timed collector
7. 做macro vs sequential原生差分
8. 实现多Worker异步batch collector
9. 接入PPO、Checkpoint和Dashboard指标
10. Stage 1 10k
11. Stage 2 100k
12. 根据行为证据决定是否Stage 3 500k
```

## 20. 本设计不承诺的内容

- 不承诺WAIT机制单独就能产生高水平策略；
- 不承诺八卡使用率应当均匀；
- 不把100k行为恢复等同于强度提升；
- 不用PPO loss代替对战评估；
- 不在没有差分证书时宣称宏步进100%等价；
- 不在500k门失败后机械训练到2M或5M。

本设计解决的是已实测的“20 Hz低费动作Hazard”，并为后续Self-Play提供可表达、
可审计、可恢复的时间决策接口。
