# Expert 100k 原生逐 Tick 资格审计

日期：2026-08-26

## 结论

最终 100,000 场中，当前可以进入**完整 native per-Tick teacher-forced 候选队列**
的不是 100,000 场，也不是当前编译器宽口径的 28,193 场，而是：

```text
权威静态候选：26,385 场
  deployment-only / 来源明确 0 次技能：2,359
  ability-positive / 来源有精确技能 Tick：24,026
```

这里的“权威”只表示以下输入事实完整：

- schema-v3；
- 每次部署都有原始 `x_raw / y_raw / data_i`；
- 双方完整八卡、等级、形态和塔兵；
- 卡牌、形态、塔兵、技能均能映射到当前 Runtime；
- 技能要么来源明确为 0 次，要么每次都有精确 Tick；
- 当前 `compile_battle()` 能构造无歧义的一 Tick 一命令计划。

它**不表示恢复了服务端隐藏 RNG 或原始逐 Tick 状态**。本项目采用已确认的
`native_teacher_forced` 语义：把同版本 JSON 专家动作按原 Tick/坐标送入同版本
`libg.so`，后续世界状态由原生核心生成。

当前 `compile_battle().native_replay_ready` 的宽口径是 28,193 场。其中额外
1,808 场来自 schema-v2、来源报告 0 次技能；它们没有 raw `data_i`，坐标只能走
历史 `x/y`，必须留在近似隔离层，不能混入上述 26,385 场。

## 审计输入与不变量

唯一基准清单：

```text
D:\AI_data\cr-native-core\expert-v1\training-dataset\version-window-20260804\accepted-cycle-clean.jsonl
```

- 行数：100,000；
- SHA-256：`3566ba54d3933d50521343ce758edcc1d11351837ead8aaea1bd6f11cbfd4a3a`；
- 审计逐一读取 `source_path`，对源文件计算 SHA-256；
- 不调用 libg，不连接 Worker；
- 不复制源 battle JSON；
- 每场都调用当前 `compile_battle()`，并复用当前 card/form、tower troop 和 ability
  capability 映射；
- 结果是“执行前资格审计”，不是 100k 原生实跑结果。

可复现入口：

```powershell
D:\AI_data\runtime\venv\Scripts\python.exe `
  scripts\audit_expert_100k_native_eligibility.py
```

## 全量分层

### Schema 与坐标

| Schema | 场次 | 坐标证据 | 处理结论 |
|---|---:|---|---|
| v1 | 50,229 | 仅历史 `x/y` | 缺等级/形态/塔兵，sequence-only |
| v2 | 23,205 | 仅历史 `x/y` | 元数据完整，但只能作为近似层 |
| v3 | 26,566 | 全部部署均有 raw `x/y/data_i` | 可进入权威资格判断 |

100k 共 7,410,172 个部署事件。其中 schema-v3 的 1,927,384 个事件全部保留
raw `data_i`：

- `data_i=0`：954,194；
- `data_i=1`：973,190；
- schema-v3 partial/missing：0 场；
- schema-v1/v2 legacy-only：73,434 场。

### 八卡、等级、形态与塔兵

| 字段 | 完整 | 缺失 |
|---|---:|---:|
| 双方八卡 | 49,771 | 50,229 |
| 卡牌等级 | 49,771 | 50,229 |
| 卡牌形态（按 schema-v2+ 契约） | 49,771 | 50,229 |
| 双方塔兵 | 49,771 | 50,229 |

49,771 正好等于 schema-v2 + schema-v3。当前可编译的 v2/v3 中没有未映射塔兵
或不支持的形态。三个 schema-v1 文件使用 `party-hut`，该 token 不在冻结 libg
卡表，按 fail-closed 拒绝。

### 技能日志

| 来源能力层 | schema-v1 | schema-v2 | schema-v3 | 合计 |
|---|---:|---:|---:|---:|
| 来源明确 0 次技能 | 4,217 | 1,820 | 2,378 | 8,415 |
| 只有次数、缺失 Tick | 46,012 | 21,385 | 0 | 67,397 |
| 精确技能 Tick | 0 | 0 | 24,188 | 24,188 |

schema-v3 共 90,547 个精确技能事件。旧 schema 的 67,397 场只有 Ability count，
不能补猜技能 Tick；它们最多用于部署序列或显式标注的近似实验，不能生成完整
native teacher-forced 轨迹。

### 当前编译结果

| 资格层 | 场次 | 是否进入完整原生队列 |
|---|---:|---|
| schema-v3、zero-ability、完整可编译 | 2,359 | 是，deployment-only |
| schema-v3、ability exact、完整可编译 | 24,026 | 是，含技能 |
| schema-v2 zero-ability、legacy coordinate | 1,808 | 否，近似隔离层 |
| schema-v2 ability count-only | 21,261 | 否，缺技能 Tick且坐标近似 |
| schema-v1 可编译 | 49,886 | 否，sequence-only |
| compile/mapping 拒绝 | 660 | 否 |
| **合计** | **100,000** |  |

因此：

```text
authoritative native full = 2,359 + 24,026 = 26,385
current compiler native_replay_ready = 26,385 + 1,808 = 28,193
```

660 个编译拒绝全部可解释：

| 首个拒绝 | 场次 |
|---|---:|
| 同一方同 Tick 有两个部署 | 641 |
| 同一方同 Tick 同时部署和技能 | 16 |
| 冻结卡表缺少 `party-hut` | 3 |

原生命令协议每方每 Tick 只能提交一个命令。前两类不能随意排序或平移 Tick；这样
会篡改专家动作语义，所以保持 fail-closed。按 schema 看，拒绝为 v1 343、v2
136、v3 181；v3 的 181 场正是 26,566 到 26,385 的差额。

## 100 场 Pilot 外推：只能作为估计

本节没有把 100 场结果伪装成 100k 实测。使用：

- deployment v10：89/100；
- ability pilot：58/100；
- 每层分别计算双侧 Wilson 95% 区间；
- 合并展示带把两层 Wilson 端点相加；它是规划带，不是联合 95% 置信区间；
- 默认 Pilot 对未来队列具有代表性、且成功率与对局时长近似独立。该假设未由
  100k 实跑证明。

| 层 | 静态候选 | Pilot | 成功率 95% Wilson | 预计完整成功场 |
|---|---:|---:|---:|---:|
| deployment-only | 2,359 | 89/100 | 81.37%–93.75% | 2,100；区间 1,919–2,212 |
| ability exact | 24,026 | 58/100 | 48.21%–67.20% | 13,935；区间 11,582–16,146 |
| **合并** | **26,385** | 分层 | — | **16,035；规划带 13,501–18,358** |

这意味着“26,385 场具备静态输入资格”，不等于“最终一定得到 26,385 条完整
轨迹”。按现有证据，完整可存轨迹的点估计约为 16k；主要损失仍在 ability
实体唯一解析、生成态长期漂移和 command gate，不是卡表或塔兵映射。

## Tick、存储与四 Worker 墙钟

### 权威 26,385 场

| 项 | 数值 |
|---|---:|
| deployment-only 来源时长 Tick | 9,981,260 |
| ability-exact 来源时长 Tick | 113,818,060 |
| 静态候选总 Tick | 123,799,320 |
| 预计成功轨迹 Tick | 74,897,796 |
| 成功轨迹 Tick 规划带 | 62,989,308–85,844,591 |

使用当前真实 Pilot：

- deployment v10：1,448.14 stored Tick/s，25.06 B/Tick；
- ability：1,035.12 stored Tick/s，26.21 B/Tick；
- 均为同一台机器、4 Worker、包含失败开销的端到端 Pilot 指标。

由此得到：

| 规划量 | 估计 |
|---|---:|
| 成功 Tick Store 点估计 | 1.95 GB（约 1.82 GiB） |
| 成功 Tick Store 规划带 | 1.64–2.24 GB（约 1.53–2.09 GiB） |
| 假设全部候选完整存储 | 3.23 GB（约 3.01 GiB） |
| 只按预计成功 Tick 计算的四 Worker工作量 | 16.28–22.32 小时 |
| 全部候选都推进到来源时长的四 Worker投影 | 32.46 小时 |

真实墙钟是 **Unknown**：失败对局会消耗非零前缀和 seed/layout 校准时间，而未来
队列的时长和失败深度分布也可能与两个 100 场 Pilot 不同。16.28–22.32 小时与
32.46 小时是规划区间，不是保证上下界。

### 仅作容量参照的全部 100k

全 corpus 来源时长合计 470,930,740 Tick。若不考虑资格、强行按当前两个 Pilot
的密度/吞吐线性换算：

- Tick Store 约 11.80–12.34 GB（10.99–11.50 GiB）；
- 四 Worker 约 90.3–126.4 小时。

这不是可执行建议，因为 73,615 场缺少完整原生资格。

## 输出与队列

审计根目录：

```text
D:\AI_data\cr-native-core\expert-v1\native-eligibility-v1
```

包含：

- `summary.json`：全量计数、Tick、事件数、Pilot 外推和置信区间；
- `manifest.json`：20 个审计 shard 与所有队列的行数、字节数、SHA-256；
- `shards/part-*.jsonl`：100,000 条逐场资格记录；
- `queues/authoritative-native-full.jsonl`：26,385 场正式候选；
- `queues/authoritative-deployment-only.jsonl`：2,359 场；
- `queues/authoritative-ability-exact.jsonl`：24,026 场；
- `queues/compiler-native-ready.jsonl`：28,193 场宽口径，仅用于核对；
- `queues/old-schema-approximate.jsonl`：23,069 场 schema-v2 近似层；
- `queues/sequence-only.jsonl`：49,886 场 schema-v1；
- `queues/compile-rejected.jsonl`：660 场 fail-closed 现场引用。

队列行只保存 Tag、源路径、源 SHA、schema、Tick/动作计数和资格字段，不包含源
battle JSON 内容。

## 下一步队列建议

### 先跑固定 1,000 场

从 `authoritative-native-full` 做确定性分层选择：

```text
deployment-only：100
ability-exact：900
```

总体比例接近 26,385 场母体，同时让较小的 deployment 层仍有 100 场独立分母。
固定 selection 与 source SHA，使用：

- 4 Worker；
- `action_execution_tick_offset=1`；
- raw `data_i` 坐标变换；
- 当前 bounded seed/layout calibration；
- 技能唯一实体才执行，多实体显式 branch，绝不猜；
- 成功和失败分别落目录，失败保留首差异；
- 每 100 场输出分层成功率、失败类别、Tick/s、B/Tick、墙钟和 Worker 错误。

1k 通过标准不是“所有场都成功”，而是：0 数据串局、0 静默吞动作、0 不可解释
拒绝、所有成功 Tick 连续且 shard 可全量解码；同时用实际 1k 结果替换本文的
100 场外推。

### 再跑全量 26,385

只有 1k 验收后，才消费 `authoritative-native-full`。正式产物按语义分开：

1. `complete-native`: 全动作接受、逐 Tick 连续，可进入原生场景训练；
2. `first-divergence`: 首差异诊断，不进入完整轨迹训练；
3. `ability-branch-required`: 等显式分支回放；
4. `terminal-diagnostic`: 仅诊断，不反向否定已完整接受的动作序列。

schema-v2 23,069 场继续留在 `old-schema-approximate`，schema-v1 49,886 场继续留在
sequence-only；除非重新抓取 raw `data_i`/技能 Tick/完整元数据，否则不要通过
“容错”把它们升级成权威原生轨迹。
