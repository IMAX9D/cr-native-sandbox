# Schema-v3 技能原生回放 100 场验收

这套验收与 deployment-only pilot 完全隔离，目标是验证：来源 JSON 中带精确
`time_raw` 的技能按钮事件，能在相同 Tick 上根据原生实体状态解析并通过 joint
command 送入 `libg.so`，同时把每个完整 20 Hz Tick 写入二进制 Tick Store。

## 语义边界

- 只选择 `schema_version == 3` 且 `ability_plays` 非空的对局。
- `ability_plays` 数量必须与 `elixir_stats.*.Ability.count` 一致。
- 计划必须通过完整卡牌形态、塔兵和技能静态映射检查。
- 技能触发 Tick 上只有一个合法原生实体时才执行。
- 同时存在多个合法实体时输出 `ability_branch_required`，不猜实体。
- 找不到实体或原生拒绝时停止该局，并保存来源、完整计划、当前紧凑原生状态、
  joint request/response、候选实体和最近 trace 边界。
- 终局匹配仅作诊断，不反向否定已经完整接受的 teacher-forced 动作序列。

## 两阶段入口

第一阶段只生成确定性任务清单，不连接 Worker：

```powershell
python scripts/pilot_expert_native_abilities.py prepare
```

选择算法为 `SHA256(selection_seed + "\\0" + battle_tag)` 升序，因此不受源清单
行序变化影响。当前正式 100k 清单生成结果为：

- 100 场 schema-v3 ability-positive 对局
- 7,174 个部署事件
- 376 个精确 Tick 技能事件
- 467,960 个来源时长 Tick
- 任务清单：
  `D:\AI_data\cr-native-core\expert-v1\native-ability-pilot-100-plan\selected.jsonl`
- SHA-256：`2097f359fda18de4a08bf7e07ec43501d14569cebe36f3352ac1c5cf6666b250`

第二阶段才会连接四个独占 Worker 并执行：

```powershell
python scripts/pilot_expert_native_abilities.py run
```

默认端口为 `38031..38034`，采用一个 Worker/连接/分片写入器和共享任务队列。
每个原生 gap 通过最多 64 Tick 的 compact trace RPC 获取，不使用每 Tick 一次 RPC。

## 输出与通过条件

默认执行目录：

`D:\AI_data\cr-native-core\expert-v1\native-ability-pilot-100-execution`

其中：

- `results.jsonl`：每局接受数、技能解析、终局诊断、Tick Store 索引。
- `diagnostics/*.json`：每个失败的完整现场。
- `shards/*.crts`：锚点 + delta 的逐 Tick 原生状态。
- `shards/manifest.json`：分片哈希、总局数、总 Tick 和总字节数。
- `summary.json`：吞吐、接受率、失败分类和最终验收结论。

只有以下条件同时成立，`acceptance_pass` 才为真：100/100 任务都有结果、全部部署
和技能动作被原生接受、每个成功轨迹的 Tick 范围连续、分片校验通过、无 Worker
初始化/连接错误。`branch_required` 是正确的 fail-closed 证据，但会使本轮非分支
验收不通过，后续应对该局做显式分支回放，不能人为指定任意实体。
