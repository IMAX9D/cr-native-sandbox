# 原生技能回放固定 100 场结果审计

## 结论

这轮固定 100 场 schema-3 技能回放证明了两件需要分开表述的事：

1. **外部技能接口已经真实打通。** 到达并唯一解析实体后，实际向 `libg.so`
   发送了 300 次技能命令，其中 299 次被接受，条件接受率为
   `299 / 300 = 99.667%`。这些命令覆盖 23 个 base card ID。
2. **100 场完整 teacher-forced 生成仍未通过严格验收。** 只有 58/100 场把全部
   部署和技能动作走完；42 场按 fail-closed 停止。剩余问题主要是来源事件缺少
   实体身份、无锚生成态逐渐漂移，以及一个尚未解释的上下文相关技能拒绝，不是
   “没有技能 RPC”。

不能把 summary 里的 `299 / 376` 称为“技能尝试接受率”。376 是来源文件中的
全部技能事件；首个失败后，本局后续动作不会继续执行。真正送入原生核心的技能
命令只有 300 次，真正的原生技能命令接受率才是 299/300。

本轮原始 `acceptance_pass` 为 `false`，因为严格条件是 100/100 场全部动作完成。
这不否定 58 个成功 episode 的动作完整性和 Tick Store 完整性。

## 固定输入与运行语义

执行目录：

```text
D:\AI_data\cr-native-core\expert-v1\native-ability-pilot-100-data-i-phase-plus1-v1
```

任务清单：

```text
D:\AI_data\cr-native-core\expert-v1\native-ability-pilot-100-plan\selected.jsonl
```

| 文件 | SHA-256 |
|---|---|
| 固定任务清单 | `2097f359fda18de4a08bf7e07ec43501d14569cebe36f3352ac1c5cf6666b250` |
| `results.jsonl` | `38e3c583d2867547f5c488184975b448e24458f5d7d964afda0665d81e160d53` |
| `summary.json` | `4fae69db4e508284fc4e5f80d8e7e13e78661564c8d45459da81657ee3286fdb` |

独立审计重新验证了：

- 任务清单 100 行、100 个唯一 Tag、`selection_index` 恰好为 0..99；
- 结果 100 行、100 个唯一 Tag，与任务 Tag 集合完全相同；
- 逐一重新读取 100 个来源 JSON，并计算实际文件 SHA-256；
- 100/100 来源文件都是 schema 3；
- 每场来源 `card_plays / ability_plays` 数量与 task 和 result 三方一致；
- 没有用 result 中自报的 source SHA 代替实际磁盘复核。

本轮固定 native execution 语义为：

```text
source label：RoyaleAPI time_raw T，原值不改
native execution boundary：T + 1
action_execution_tick_offset：1
```

100/100 结果以及 58/58 Tick Store episode metadata 都重新核对了上述 offset 和
provenance。318 个已经到达的技能 marker 也逐个满足
`execution_tick == source_tick + 1`。

坐标语义为：

```text
coordinate_provenance = royaleapi_raw_data_i_to_native_v1
data_i=0：rotate_18000_32000
data_i=1：identity
```

7,174 个部署事件全部来自原始 `data_i`：`data_i=0` 有 2,718 个，`data_i=1`
有 4,456 个，legacy XY fallback 为 0。

## 正确的动作分母

### 部署动作

| 阶段 | 数量 |
|---|---:|
| 来源部署事件 | 7,174 |
| 实际到达原生执行边界 | 5,981 |
| 原生接受 | 5,959 |
| 原生拒绝 | 22 |
| 首个失败后未继续尝试 | 1,193 |

真正的条件接受率为：

```text
5,959 / 5,981 = 99.632%
```

`5,959 / 7,174` 只是本轮在 fail-closed 规则下走到多深的覆盖率，不是部署接口
接受率。

### 技能动作

| 阶段 | 数量 |
|---|---:|
| 来源技能事件 | 376 |
| 已到达 exact-Tick 解析 marker | 318 |
| 唯一合法实体，可送入 libg | 300 |
| 多候选，要求显式分支 | 5 |
| 无合法匹配实体 | 13 |
| 真正送入原生核心 | 300 |
| 原生接受 | 299 |
| 原生拒绝 | 1 |
| 更早失败后未到达 | 58 |

真正的原生技能命令接受率为：

```text
299 / 300 = 99.667%
```

若观察完整“marker → 唯一实体 → 原生接受”管线，则是
`299 / 318 = 94.025%`。这个数字包含 5 个来源身份歧义和 13 个生成态无实体，
其语义也不同于原生接口接受率。

完整成功的 58 场中，4,168/4,168 个部署动作与 221/221 个技能动作全部接受。

## 42 场失败的精确分类

| 首失败类 | 场数 | 是否真的发送了该技能命令 | 解释 |
|---|---:|---|---|
| `ability_branch_required` | 5 | 否 | 同一 side 同时有多个合法能力实体，来源 marker 没有 entity ID；正确行为是 fail-closed，而不是猜一个 |
| `ability_entity_missing` | 13 | 否 | exact Tick 的生成态没有合法匹配实体；属于实体存活/能力合法性/生成态分叉 |
| `native_action_rejected` | 23 | 是（其中仅 1 个是技能） | 18 个部署 code4、4 个部署 code13、1 个技能 code1013 |
| `teacher_forced_failure` | 1 | 否 | 原生 Tick 冻结在 3681，无法前进到请求的 execution Tick 3744 |

23 个真实原生拒绝进一步拆分为：

| 原生命令 | result code | 数量 | 已观测 reason | 审计分类 |
|---|---:|---:|---|---|
| deployment | 4 | 18 | `battle_command_gate` | 生成态比来源动作序列更早关闭命令门，属于下游状态/终局漂移 |
| deployment | 13 | 4 | `insufficient_elixir` | exact-Tick 原生资源状态与来源动作时序分叉 |
| ability | 1013 | 1 | `native_rejected` | 上下文相关条件尚未解码，保持 `Unknown` |

因此，42 不能统称为“技能接口不支持”：

- 5 是来源标签不够精确；
- 13 是 exact-Tick 生成态中能力实体/合法性不再存在；
- 22 是部署动作先失败；
- 1 是原生逻辑不再推进；
- 只有 1 个是已经唯一解析并送入原生核心后被拒绝的技能命令。

## 唯一技能拒绝：Hero Mega Minion `code1013`

证据固定在 battle tag `099P9RVLP908`：

| 字段 | 值 |
|---|---|
| source Tick | 4056 |
| native execution Tick | 4057 |
| side / entity | `1 / 5000090` |
| base / native card ID | `26000039 / 203000039` |
| form | `MegaMinion / hero` |
| compact ability state | `ready`，state code 2 |
| available / charge / cooldown | `true / 1 / 0 ms` |
| native mana cost | 2 |
| player Elixir | `8.5175`（raw 85175） |
| episode command gate | open，`commands_allowed=true` |
| native result | rejected，code 1013 |

这些证据排除了“没有实体”“普通 ready 位未读出”“显式圣水不足”和“整个战斗
命令门关闭”四种简单解释，但**不能给 1013 强行命名**。

Supercell 对 [Hero Mega Minion / Wounding Warp 的官方说明](https://supercell.com/en/games/clashroyale/blog/news/new-season-midnight-mischief)
是：技能会寻找竞技场中最大生命值最低的敌人并传送过去。也就是说，该技能本身
依赖目标上下文。失败快照中虽然能看到存活的敌方非塔实体，但 compact state 并
没有暴露 libg 内部的目标资格判定、锁定状态和全部隐藏 guard；“看起来 ready”
不能证明 native target resolver 一定能在该 Tick 接受命令。

因此本轮只能记录：

```text
code1013 = Unknown / 可能是目标或其他上下文前置条件
```

不能把它写成“无目标”“冷却”“圣水不足”或其他未经验证的固定含义。下一步应对
Hero Mega Minion 单独做目标存在性、目标类型、出生后锁定延迟和目标切换的受控
矩阵，而不是在通用 runner 里绕过 1013。

## 终局诊断

终局只对 58 个 teacher-forced 完整成功 episode 有可比意义：

| terminal status | 58 个成功 episode |
|---|---:|
| crowns match | 46 |
| crowns mismatch | 3 |
| source-duration fence 时 native terminal missing | 9 |

在 49 个能明确比较皇冠的成功 episode 中，匹配率为
`46 / 49 = 93.878%`。

全 100 场 summary 中的 terminal 分布是 `46 match / 3 mismatch / 51 missing`；
其中 42 个 missing 来自动作尚未完整执行的失败场，不能放进成功终局匹配率分母。

这里的皇冠是**同版本 libg 在 teacher-forced 动作下生成的终局诊断**。它不是来源
服务器未公开的逐 Tick 隐藏状态真值，也不会反向否定已经完整接受的动作路径。

## Tick Store 完整性

58 个成功 episode 的 Tick Store 已逐 episode、逐 Tick 全量解码，不是只相信
summary 或 manifest：

- 4 个物理 shard 的 data/index SHA-256 全部与 manifest 一致；
- global/local shard manifest 逐对象一致；
- 每个 frame header、payload SHA-256 与 index 一致；
- 58 个 episode 全部解压，Tick 严格连续 `+1`；
- 首 Tick、末 Tick、Tick 数与 index/result 一致；
- 每个 episode 的 source SHA、坐标 provenance、T+1 provenance、20Hz 和
  `every_native_tick_present` 全部复核；
- store Tag 集合与 58 个 teacher-forced success Tag 集合完全相同。

| shard | episode | Tick | bytes | data SHA-256 | index SHA-256 |
|---|---:|---:|---:|---|---|
| `ability-worker-00-00000` | 18 | 79,747 | 2,069,930 | `dc47ed252e87df9a48064549dd008e461eaece6dfe7629528ea9e966d128d12d` | `e8f015ca216ae438fccc55335d629b5cd746430b83d4e1918965b5c18d60ef53` |
| `ability-worker-01-00000` | 12 | 56,227 | 1,372,959 | `be9d430c909d6290555780f1de691e55f99fecdc97dccfa3ebaf7fb9015740dd` | `e757005cacb93226427a750e1437a6f01c02afc8df71057d92f27de79aa20200` |
| `ability-worker-02-00000` | 14 | 65,147 | 1,784,001 | `0fabbe5035686dd70f5fe9471837e96f2a0a05493354f96821f1af91bb67e110` | `161c2ccd484a5f90a3d388105f89da8ee7b575f1972541092db194e8599033ea` |
| `ability-worker-03-00000` | 14 | 68,112 | 1,830,028 | `41db1a1db85c94e97a88e58f2de619986f30d7d1f13a08ca31af793a88f8dab9` | `73100f331f9415470de863f02ef30106a0982562f9eb91a8f3150f44ce0de034` |
| **合计** | **58** | **269,233** | **7,056,918** | — | — |

关键摘要：

```text
Tick Store manifest SHA-256
2077c30be7bdf5c78d5f41671b644daaa71d47d537a90f19b6b38fbcb61ddb70

Tick Store content SHA-256
8100fa69fe3aa611dcc633bccb2b0e6f90cde89daa3a04da40369668e7280de3

bytes / Tick
26.2112
```

“每 Tick 完整”在这里指每个已写入成功 episode 的原生 Tick 流连续、无缺帧，
并不表示失败的 42 场被截断前的片段也混入了正式成功分片。

## 对当前接口能力的判断

可以确认：

- entity-targeted ability command 已能从 Python 经 Host/JNI 到达 libg；
- 本轮覆盖到的 23 个 base card ID 中有 299 次真实接受证据；
- exact-Tick 解析、T+1 execution、坐标方向与 Tick Store metadata 能闭环审计；
- 多实体歧义时系统没有偷偷选第一个实体。

不能确认：

- 100/100 来源动作能在无中局状态锚点的生成态里始终复现；
- 每个来源技能 marker 都含有足够信息确定具体实体；
- `code1013` 的固定语义；
- 46/49 皇冠匹配等于原始隐藏服务器状态已经恢复。

所以剩余工作应按证据拆开：

1. 对 5 个 `branch_required` 做显式分支，不猜实体；
2. 对 13 个 `entity_missing`、18 个 code4、4 个 code13 和 1 个逻辑冻结研究
   分段 replay / 中局锚点，避免把生成态累计漂移误判成接口缺失；
3. 对 Hero Mega Minion 单独做上下文矩阵，保持 1013 为 Unknown；
4. 任何后续报表都同时记录 `source / reached / dispatched / accepted` 四个分母。

## 机器报告与复核

机器报告：

```text
D:\AI_data\cr-native-core\expert-v1\native-ability-pilot-100-data-i-phase-plus1-v1\ability-pilot-100-result-audit.json
```

当前报告 SHA-256：

```text
ac2efca164496030b5813ce7dfd66846495c177423630627dacca29b4c52bb14
```

报告包含 100 条逐 Tag 输入/结果摘要、42 条失败分类、23 条真实原生拒绝、唯一
Hero Mega Minion code1013 的现场摘要、4 个 shard 的全量解码/SHA 结果，以及 16 条
固定断言；当前全部为 `true`。

只读复核命令：

```powershell
D:\AI_data\runtime\venv\Scripts\python.exe `
  scripts\audit_native_ability_pilot_100.py
```

脚本只读取固定 pilot、任务清单、来源 JSON、diagnostics 与 Tick Store；唯一写入是
上述 audit JSON。它不连接 Worker、不修改 runner，也不重跑 libg。
