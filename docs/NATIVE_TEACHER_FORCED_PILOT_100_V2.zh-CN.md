# 100 场原生 teacher-forced 逐 Tick Pilot

日期：2026-08-26

## 范围

本次从最终 100,000 场清单中确定性选择 100 场：

- source schema >= 3；
- 双方完整卡组、形态、等级和塔兵元数据齐全；
- 当前 Runtime 映射支持；
- 源数据明确报告 0 次主动技能，避免静默遗漏技能；
- 首个动作不早于原生 warm-up Tick 10。

这不是挑选“容易通过”的样本。任何手牌不符、原生动作拒绝、战斗提前
终止、牌组布局不收敛都会在首个差异处 fail-closed。

结果目录：

`D:\AI_data\cr-native-core\expert-v1\native-teacher-forced-pilot-100-compact-v6`

## 执行设计

- 1 AVD / 4 个隔离 libg Worker，Direct TCP `38031..38034`；
- SQLite WAL 中央 lease queue，四个 Worker 动态抢任务；
- 每个 Worker 生命周期内复用持久 TCP 和一次 seed shuffle bootstrap 校准；
- 对牌组相关的 4+4 布局最多使用三次实际 deck 再校准；
- 每次最多推进 64 Tick，并调用 `step_train_trace_v1` 返回每个 20 Hz
  原生状态；
- 成功场写入 Worker 私有 Anchor/Delta 分片，失败场不写入训练数据；
- terminal 皇冠只是独立诊断，不覆盖 deployment teacher-forced 结果。

仅在所有源动作之后的 source-duration fence，允许 Host 返回 Tick 不前进的
incomplete 终局/冻结后缀。动作之间仍严格要求每个完整帧恰好 `Tick+1`。

## 结果

| 指标 | 实测 |
|---|---:|
| 输入场次 | 100 |
| 完整 teacher-forced 成功 | 37 |
| 首差异失败 | 63 |
| 源部署动作 | 6,993 |
| 在首差异前接受的部署动作 | 4,494 |
| 在首差异前覆盖的源动作 | 64.264% |
| 实际送入 native execute 的动作 | 4,544 |
| 已尝试动作接受率 | 98.900% |
| 成功场逐 Tick 状态 | 153,707 |
| Pilot 墙钟 | 207.823 s |
| 端到端逐 Tick 吞吐 | 739.606 Tick/s |
| 成功场速度 | 640.93 场/小时 |
| Tick Store | 3,703,951 bytes |
| 压缩密度 | 24.097 B/Tick |

失败场分层：

`4,494 / 6,993 = 64.264%` 使用全部源动作作分母。由于 runner 在首个差异
处停止，后续动作根本没有尝试，它不是“已尝试动作接受率”。实际进入原生
execute 的 4,544 个动作中，4,494 个接受、50 个拒绝，接受率为 98.900%。

| 首差异 | 场次/事件 |
|---|---:|
| 原生 execute 拒绝 | 49 场 / 50 个拒绝事件 |
| `result_code=4` | 42 个事件 |
| `result_code=13` | 8 个事件 |
| 4+4 shuffle 三次仍不收敛 | 6 场 |
| 原生终局早于下一个源动作 | 8 场 |

成功 37 场的终局诊断：

- 皇冠一致 23；
- 皇冠不一致 1；
- source-duration fence 尚无终局 12；
- fence 处逻辑时钟冻结 1。

## 拒绝证据

### code 4

静态反汇编确认 code 4 来自 battle-logic predicate `D503D0`，不是落点
validator。42/42 个事件的运行态均记录 `command_gate_code=4`；样本中可见
本地皇冠和塔血已经进入与源回放不同的战斗结果，例如：

- `02RY9QJQ8QQR` 在 Tick 3765：源终局 `[1,0]`，本地已 `[1,3]`，
  side 0 King Tower HP 为 0；
- 当时 Electro Spirit 在手、圣水 `84505`、落点 `valid`，但
  `commands_allowed=false`。

将该场 avatar/hbd/support tower 设为 16，或把全部卡牌改为 11，均没有改变
首拒绝 Tick。模板 game mode 会把塔 HP 固定在 King `4824`、Princess
`3052`，因此不能把“账户卡牌等级”直接当作本场有效等级，也不能把等级
同步宣称为修复。

### code 13

静态反汇编确认 code 13 是原生执行 Tick 的 `current_elixir < card_cost`。
8 个事件全部保存当前 `elixir_raw`、卡费和差额；实测差额为：

`-28, -24419, -116, -70, -305, -10400, -142, -43`

其中小差额可能涉及源 marker 的 Tick 量化边界，较大差额还需检查卡牌多形态
费用或此前状态漂移。不得自动延迟动作、补圣水或吞掉样本。

每个首拒绝还包含：当前四手牌、next deck index、refill timer、距该方上一
动作的 Tick、六塔 HP、皇冠、battle/logic phase、command gate、实体数、
原生 execute 结果和部署 validator 结果。

## Seed 诊断

对 4 个成功样本改用 seed 1：

- 3 个样本能够完成 alternate replay；将 native deck slot 反映射为逻辑卡索引
  后，所有逐 Tick 训练状态 SHA-256 完全相同；
- 1 个样本在 alternate seed 的牌组布局校准阶段不收敛，未形成可比较轨迹；
- 没有观测到“两个可比较轨迹因 seed 而产生逻辑状态差异”的反例。

因此当前证据是 3/3 可比较样本一致，而不是把不可执行的第 4 个样本误报为
seed 差异。

## 数据完整性

四个 `.crts` 分片已全部重新解码：

- episode：37/37；
- Tick：153,707/153,707；
- 相邻 Tick：全部严格 `+1`；
- Store manifest 的 `every_native_tick_present=true`；
- 每个数据分片和索引均有独立 SHA-256。

关键文件 SHA-256：

- `summary.json`：`5BC978430025D059D0D1C09D6AC228497D7969CB23721C741AF31CDE32B33342`
- `results.jsonl`：`6BFDFA6967F08D79EF1C36F618970B6AD43C7AC476EF8A287D770F12ED8684EF`
- `selection.jsonl`：`5FA5239318CE2396934408CEAB624D507CCAF9A89143752ED80F458CA0127A3A`
- `shards/manifest.json`：`343E83ED2CC9D28292E6D3F9BFC895CEDB8A576FD6DC9F4FBFC72CD34826908C`

## 结论和下一步

逐 Tick 并发采集、压缩存储和首差异审计链已经可用，但当前 37% 的完整通过率
不足以直接扩到 100,000 场并宣称场景训练集已经就绪。

下一步按证据优先级处理：

1. 对 code 4 场做 seed / game-mode / replay-config 矩阵，定位本地战斗提前结束
   的首个状态分叉；
2. 对六个 layout 场记录每轮 actual 4+4、desired 4+4 和 logical→native mapping，
   修复布局校准而不是增加盲重试；
3. 对 code 13 检查 source marker 量化和多形态费用，保持原 Tick，不做容错平移；
4. 再跑包含 schema3 ability 的独立 pilot，唯一技能实体才执行，多候选显式分支；
5. 修复后使用同一固定 selection 重跑，只有逐项差分消失才扩大到 1,000/100,000。
