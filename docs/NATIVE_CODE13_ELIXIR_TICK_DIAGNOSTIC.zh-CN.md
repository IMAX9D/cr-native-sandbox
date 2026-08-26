# Teacher-forced `result_code=13` 圣水与 Tick 诊断

## 1. 结论

`result_code=13` 已由诊断 Host 的原生命令证据确认是：

```text
D8D7C1: player.elixir_raw < packed_selection.card_cost * 10000
```

它不是落点、手牌或 Python 费用表的泛化错误。v6 的 8 例必须分成三类：

1. **6 例一 Tick 边界不确定**：T 点只差 `28..305 raw`，T+1 全部接受；
2. **Giant + Elixir Golem 资源事件**：T+9 出现一次 `+5000 raw` 非被动恢复，
   到 T+31 才接受；
3. **Spirit Empress 双形态缺口**：源事件没有逐次 3/6 费形态，当前原生
   canonical selection 在 `3.5581` 圣水时仍解析成 6 费 Mounted，T+69 才接受。

因此当前**没有证据支持把所有动作统一平移**。正式数据仍应 fail-closed；可以
保留严格前缀和显式不确定性，但不能把 A/B 中的最早接受 Tick 回写成源真值。

## 2. 证据与可复现物

- v3 pilot：
  `D:\AI_data\cr-native-core\expert-v1\native-teacher-forced-pilot-100-compact-v3`
- v6 pilot：
  `D:\AI_data\cr-native-core\expert-v1\native-teacher-forced-pilot-100-compact-v6`
- 完整 T..T+n A/B：
  `D:\AI_data\cr-native-core\expert-v1\native-code13-tick-ab-v1.json`
- A/B SHA-256：
  `0f33134f5988b29285ca5fc3861c4a96f85e3489c2437e3c4d6951ef03b97558`
- 诊断 Host：commit `5d90e56`
- 纯诊断脚本：`scripts/diagnose_expert_elixir_phase.py`
- 结果解析/费用 nibble：`expert_v1/elixir_tick_diagnostics.py`

v3 完成的 54 场里有 7 例 code 13。v6 的 100 场里有 8 例；v3 的 7 例在
v6 逐一复现，失败 Tick 和 `elixir_raw` 均一致，排除了瞬时 RPC/Worker 故障。

## 3. A/B 协议

每一个 offset 都执行：

1. 以 seed `424242` 全新 reset；
2. 校准该牌组的 native 4+4 layout；
3. 目标事件前的双方动作全部保持源 Tick、源卡、源坐标；
4. 只把首个 code-13 目标从 T 延迟到 T+n；
5. 记录原生 pre-action 手牌、next、refill、圣水、费用、resolved data id、
   command guards 和结果码；
6. n 从 0 逐一增加，首次接受后停止。

该实验没有修改 replay runner、源 JSON、Reward 或训练语义。它只回答“在同一
生成态 native trajectory 中，该动作最早何时能被当前原生命令接受”。

## 4. 8 例结果

| Battle | T | 卡 | 阶段 | T 圣水 raw | 原生命令费用 | 缺口 | 最早接受 |
|---|---:|---|---:|---:|---:|---:|---:|
| `080Y8LY0PQ9L` | 561 | Night Witch | ×1 | 39,858 | 40,000 | 142 | T+1 |
| `00CYPPG22CPJ` | 674 | Balloon Hero | ×1 | 49,972 | 50,000 | 28 | T+1 |
| `02YYPJRY0UGQ` | 941 | Giant Hero | ×1 | 39,600 | 50,000 | 10,400 | T+31 |
| `00YYPPGLR8YU` | 1685 | Hog Rider | ×1 | 39,930 | 40,000 | 70 | T+1 |
| `00VYPYPQV8QC` | 2078 | Goblin Drill | ×1 | 39,884 | 40,000 | 116 | T+1 |
| `02QY9L89CYGV` | 2435 | Royal Hogs | ×2 | 49,695 | 50,000 | 305 | T+1 |
| `09LP9JLR0U8Q` | 2771 | Valkyrie Evo | ×2 | 39,957 | 40,000 | 43 | T+1 |
| `00GYPP8QCJ9V` | 3361 | Spirit Empress | ×2 | 35,581 | 60,000 | 24,419 | T+69 |

费用不是 Python 根据名字猜的。旧 pilot 的 `packed_selection` 高 4 bit 和新
Host 的 `resource_before.card_cost` 一致；D8D520 实际消费的就是该费用。

## 5. 六个 T+1 样本

四个 ×1 样本从 T 到 T+1 都增加 `178 raw`；两个 ×2 样本都增加
`357 raw`。手牌、next deck index 和 refill timer 在这一步内不变，原生落点
validator 也一直为 valid。

这些样本说明：

- 默认赛程的 ×1/×2 恢复倍率正确；
- 不是卡费表错一整点圣水；
- 不是卡不在手或 refill 未完成；
- 不是 2400 Tick 倍率边界错误（两例分别在 2435、2771）；
- 失败恰好暴露了小于一次恢复量的边界。

但它们只证明 `[T,T+1]` 内存在命令相位不确定性。37 个 v6 成功对局中的数千
个动作在 T 已被接受；仅凭这 6 个资源临界点，不能证明所有 `data-t` 都应
统一加一。

## 6. Giant：不是固定 Tick 相位

`02YYPJRY0UGQ` 的对手牌组含 Elixir Golem。目标侧的原生前缀为：

| Tick | 卡 | pre-action raw | 费用 |
|---:|---|---:|---:|
| 241 | Giant Hero | 100,000 | 50,000 |
| 416 | Night Witch | 81,150 | 40,000 |
| 564 | Musketeer Evo | 67,494 | 40,000 |
| 711 | Bowler | 68,660 | 50,000 |
| 911 | Guards | 64,260 | 30,000 |
| 941 | Giant Hero | 39,600 | 50,000（拒绝） |

A/B 中：

```text
T+8   41024
T+9   46202   # +5178 = +178 passive + 5000 battle resource event
T+30  49940
T+31  50118   # 首次接受
```

这次 `+5000 raw` 与同 Tick 普通 ×1 恢复量叠加，是战斗实体死亡产生的资源
事件，而不是初始圣水或固定偏移。源对局能在 T 出 Giant，说明原局与当前
teacher-forced 生成态的 Elixir Golem 资源结算时序不相同。把动作改成 T+31
只能让当前生成态合法，不能恢复源状态。

## 7. Spirit Empress：逐次形态缺失

源 JSON 三次事件均只有：

```json
{"card":"spirit-empress","card_base":"spirit-empress",
 "card_form":"spirit-empress"}
```

牌组 token 也是 `spirit-empress`，没有 Evo/Hero form flag；materialized replay
写入基础 wrapper `28000025`，不写 `el`。冻结 DataTables 同时存在：

| Data id | 原生名 | 费用 |
|---:|---|---:|
| `28000025` | `MergeMaiden` wrapper | 6 |
| `26000104` | `MergeMaiden_Normal` | 3 |
| `26000105` | `MergeMaiden_Mounted` | 6 |

该卡的公开规则是：3–5.9 圣水自动使用地面 3 费形态，达到 6 圣水时自动使用
飞行 6 费形态。参见 RoyaleAPI 的
[Spirit Empress 机制说明](https://royaleapi.com/blog/spirit-empress-new-card-2025-july?lang=en)。

本次原生证据：

```text
T=3361
elixir_raw=35581
resolved_data_id=26000105 (Mounted)
packed cost=6
result_code=13

T+69
elixir_raw=60214
resolved_data_id=26000105 (Mounted)
packed cost=6
accepted=true
```

T..T+69 每 Tick 严格增加 `357 raw`，没有资源卡或额外圣水跳变。这里不能
归类为普通 Tick 相位：源事件没给逐次形态，而当前 headless canonical
selection 在低于 6 圣水时没有切到 `26000104`。源文件的聚合 elixir 表按通用
wrapper 统计，不能替代逐事件形态证据。

在补齐以下至少一项前，Spirit Empress 的这类事件必须隔离：

- 证明并修复 headless hand-entry/selection 的 3/6 费动态切换；
- 从原始 replay command 或实体出生记录恢复每次形态；
- 为 `elixir_raw` 在 3..5.9、>=6 两侧增加原生契约测试。

## 8. `time_raw` 到底是什么

本地采集器 `D:\皇室战争数据集\crawler\parsers.py` 的生成逻辑是：

1. 从 RoyaleAPI HTML 的 `blue/red marker` 读取 `data-t`；
2. 要求它是非负整数；
3. 原样保存为 `time_raw=t`；
4. 只派生 `time=round(t/20, 2)`。

采集器没有从视频帧、duration 或本地 libg 反推 Tick，也没有把值加一或减一。
外部同类采集实现也把 `data-t` 称为 raw time tick，并保留 `ticks/20` 的时间轴。

因此目前可以确认它是 **RoyaleAPI 上游 replay marker 的 20Hz 整数时间槽**；
不能确认它在 libg 的一个 Tick 内代表“被动恢复前”“命令入队时”还是“命令
结算后”。源字段也没有 sub-Tick/phase。将它直接解释成当前 Host 的
pre-action state Tick，是尚未被源契约证明的假设。

## 9. 不伪造数据的处理建议

### 9.1 当前正式策略

1. 原 Tick 上所有动作被 native 接受，才写入完整 exact trajectory；
2. code 13 保持 strict reject，不补圣水、不绕过原生命令；
3. 可另存 T 前的连续 exact prefix，并写明终止原因；
4. 六个临界样本记录 `source_tick_interval=[T,T+1]`，但不自动选 T+1；
5. Giant 标为 `generated_state_resource_event_divergence`，不得把 T+31 当源 Tick；
6. Spirit Empress 标为 `per_play_variant_unobserved`，在动态形态闭环前隔离。

### 9.2 何时才允许统一相位换算

只有额外观测能同时证明以下条件，才可新增一个**版本化**转换规则：

- 随机大样本的源 command phase 都稳定落在同一边；
- T/T+1 的手牌、圣水、实体出生和源可见状态同时对齐；
- ×1/×2/×3、同 Tick 双方动作、倍率边界均通过；
- Spirit Empress 和 Elixir Golem 等动态资源/形态样本被单独排除或闭环。

即使满足，也应保留原 `time_raw` 和转换 provenance，而不是覆盖源字段。

下一步应等待 seed-preserving materialize 完成，再在**同一批 100 场**上做两条
互斥执行分支：

```text
A: source label T，Host 在当前 T 边界执行（现状）
B: source label仍为 T，Host 在下一 native execution boundary 执行
```

比较两分支的 code 13、code 4、动作总接受率、完整终局/皇冠匹配和首个状态
分歧 Tick。B 分支只是诊断 provenance，不能先写成训练标签。只有整体统计一致
改善且动态资源/双形态例外已单独解释，才有资格把它升级为相位换算规则。
