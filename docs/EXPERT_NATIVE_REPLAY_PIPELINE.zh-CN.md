# Expert-v1 原生复演与逐 Tick 数据方案

## 结论

当前 RoyaleAPI JSON 可以确定专家的普通部署 Tick、卡牌和坐标，并可从完整
八卡循环构造一个与动作序列兼容的初始手牌/队列。但它没有原始 `libg` 的 RNG
种子、游戏 Build、数值模式 ID、塔等级和逐 Tick 状态锚点。因此，将动作重新
输入 `libg` 得到的是：

```text
state_provenance = native_generated_unanchored
action_provenance = observed_deployments
```

它不能标成原对局 `authoritative state`。原生动作全部接受也只能证明生成场景
内部自洽，不能证明生成场景等于真人当时看到的场景。

生产源清单冻结为：

`D:\AI_data\cr-native-core\expert-v1\training-dataset\version-window-20260804\accepted.jsonl`

本次审计该清单有 73,556 场，SHA-256：

`9e3c4dec6e2baa7466f7ce81c4c5a1c5a7e641ce3c5be8923ed0c537f29e6855`

编译数据的 manifest 必须保存源清单绝对路径和 SHA-256；源清单改变后旧编译
结果必须过期，不能继续一键训练。

## 1. 当前源码路径

- JSON 长连接：`native_core/client.py`；
- Python 环境：`native_core/env.py`；
- 无窗口 Android Worker：`native_core/worker.py`；
- Java JSON Host：`android_probe/java/royale/nativehost/JniHost.java`；
- JNI/libg Bridge：`android_probe/native/jni_bridge.cpp`；
- 复演计划编译：`expert_v1/native_replay_plan.py`；
- gap-batched 原生执行：`expert_v1/native_replay_runner.py`；
- 数据资格审计：`scripts/audit_expert_native_replay.py`；
- 单场复演入口：`scripts/replay_expert_native.py`。

现有 Host 已经具备：

- Worker 生命周期内持久 TCP；
- 进程内 `reset`；
- `step(N)` 批量推进；
- `joint_act` 同 Tick 双方动作；
- `observe_train_v1` 紧凑训练状态；
- `joint_training_transition_v1` 单 Tick动作、推进、观察；
- `step_trace` 最多 64 Tick 的完整无损 Debug Trace。

离线专家复演不应逐 Tick 调 `joint_training_transition_v1`。动作之间应使用一次
`step(gap)`，只在需要落训练样本的 Tick 调紧凑观察。

## 2. 当前 73,556 场实测审计

审计输出：

`D:\AI_data\cr-native-core\expert-v1\native-replay-audit-current-window-v3-20260826`

16 线程纯编译实测：

| 指标 | 结果 |
| --- | ---: |
| 墙钟 | 268.398 s |
| 文件吞吐 | 274.06 场/s |
| 动作吞吐 | 20,270.73 动作/s |
| 源 Tick 扫描等价吞吐 | 1,281,165.55 tick/s |
| 编译成功 | 72,955 场 |
| 编译动作 | 5,440,619 |
| 源时长 | 343,862,040 tick |
| schema-v1 编译成功 | 49,886 |
| schema-v2 编译成功 | 23,069 |
| 循环/事件结构拒绝 | 601 |
| 当前可进入原生“候选复演” | 898 |
| 原对局状态可证实精确 | **0** |

这里的 898 场只表示：完整形态/等级可映射、双方公主塔、八卡循环有效、统计
表报告没有能力按钮事件。它们仍是 `native_generated_unanchored`，不是
`authoritative`。

单个 battle JSON 不含皇冠，但两轮源 `index.jsonl` 的 replay URL 保留了双方
皇冠，并且必须按源文件名而不是只按 battle tag 连接（同一局从另一玩家视角
抓取时双方会对调）。当前实现已为 73,556/73,556 场恢复
`terminal_provenance=source_index_crowns`。这使终局比较可测，但终局一致仍不能
替代逐 Tick 状态锚点。

主要缺口：

- 66,964 个已编译对局报告有能力按钮次数，但旧数据没有按钮 Tick/实体身份；
- 共缺 263,490 次能力按钮；
- 2,759 场含当前 Runtime 尚未映射的非公主塔塔兵；
- 10,214 场包含 `elite-barbarians-ev1`，当前冻结目录无法原生复演：当前
  `26000043` 没有 `evolution_form_id`，说明数据卡形态版本已经超过
  `libg 15.535.29` 的能力范围；这些场保留为 `action_sequence_only`，没有
  假装成基础形态；
- 其余拒绝包含八卡循环冲突、同方同 Tick 多动作和极少数未映射内容。

新爬虫已经开始保留 `_invalid` 能力 marker 的 Tick，并验证 marker 数量等于
Ability 统计。但 `ability_id` 目前仍为 unresolved；Tick 本身不能告诉原生接口
应该按哪个存活实体的技能按钮。只有将 marker 解析到真实技能/实体，或在该 Tick
由可靠状态锚点唯一解析，才可训练 Ability Head。

## 3. 兼容手牌的确定性解析

每方八张牌的所有 4-card 初始 hand 与其余四张 queue 只有 1,680 种。编译器对
整个专家出牌序列做状态过滤，选择一个兼容候选，并保留候选数。多个初始候选
常在前四次部署后收敛，因此：

- 后续循环/手牌可用于明确的序列标签；
- 不能声称选择出的初始 hand 就是真人原始 hand；
- 不能声称任意兼容 hand 对应的 RNG seed 是原始 seed。

计划逐方保存 `first_exact_action_index`。在该 index 之前，虽然专家打出的牌必然
属于某个兼容手牌，但另外三张牌不唯一，Card Head/Hand 特征的 source 标签必须
关闭；从唯一收敛点开始才写 `hand_provenance=inferred_exact`。不能因为生成的
libg replay 任意选了一副兼容初始手牌，就把开局歧义前缀升级成真人精确手牌。

实机证据已证明 shuffle 同时依赖卡牌身份与 slot，不能把 bootstrap deck 的排列
迁移到任意牌组。当前实现不再重排 deck slot：来源 `full_deck` 顺序和 Evo/Hero
form slot 原样保留，并在未知 seed 空间按升序做有界原生搜索。每个候选 seed 都
通过真实 `libg` reset 读取两方 4+4 布局，再用完整动作流验证八卡循环；两方同时
兼容才接受。默认上限 4096，耗尽即 fail-closed。chosen seed 只是兼容的合成
初始手牌变量，不是恢复出的真人原始 seed。详见
`NATIVE_TEACHER_FORCED_GUARD_AND_SEED_SEARCH.zh-CN.md`。

生产 generator 的 pipeline contract v4 只把升序扫描得到的**第一个**双边布局兼容
seed 送入无 trace semantic preflight。raw seed 仍由真实 libg reset 逐一验证，搜索
上限仍为 4096；变化仅是取消对第 2～8 个布局兼容 seed 的重复完整语义 replay。
固定 seed 的 full/audit-prefix trace 与 preflight parity、技能实体 fail-closed 均保持
不变。审计记录为 `single_semantic_seed_preflight_v2` 且 candidate limit 必须等于 1；
旧 cap=8 pipeline v3 的 run contract、结果或 Tick Store 不能续接到 v4 输出目录。

## 4. 高效并发执行

生产执行采用动态工作队列，不做同步 vector barrier：

```text
CPU 并行编译计划
        ↓
可复演计划队列
        ↓
N 个持久 Worker 各自 work-stealing
        ↓
进程内 reset
        ↓
step(距离下一采样点的 Tick 数)
        ↓
observe_train_v1
        ↓
验证 hand / mask / expert action
        ↓
joint_act（同 Tick 最多每方一次）
```

每场遇到下列任一情况立即 fail-closed，并保留首个证据：

- 当前手牌不含专家牌；
- Tick 漂移；
- 原生动作拒绝；
- 生成场景提前终局；
- 有界 seed 空间内找不到两方同时兼容的真实 libg 4+4 布局；
- 同方同 Tick 多动作；
- 能力事件缺失或无法唯一解析。

建议先在 1 AVD / 4 Worker 上跑 100、1,000 场验收，再对离线复演单独做
1/2/4/8 Worker sweep。每个 AVD 仍最多 4 Worker，Worker 之间用任务窃取避免长局
拖住所有短局。Self-Play 的 2 AVD/8 Worker甜点位不能直接当成离线复演实测值。

## 5. 为什么不能直接保存全部逐 Tick 稠密网格

当前编译成功部分已有 343,862,040 个 20Hz Tick。即便只保存训练脚手架最小
的 6-channel `uint8 [6,32,18]` grid，单 grid 就需要：

```text
343,862,040 × 6 × 32 × 18 = 1.188 TB（1.081 TiB）
```

还没有计算标量、卡牌 token、实体、Mask、索引、文件系统开销和下一轮补到
10 万场后的增量。完整 `step_trace` 还会在 JNI、Java、TCP 与 Python 之间构造
富 Debug JSON，不能用约 7,618 tick/s 的纯 `step(N)` 数字代表其吞吐。

生产方案保留 20Hz 原生演算，但稀疏存储训练点：

1. 所有真实部署/技能 Tick；
2. 每个等待区间的确定性均匀负样本；
3. 行动前 `-10/-5/-1 Tick` 的近决策负样本；
4. `delta_ticks` 与 `timing_exposure_ticks`；
5. 按采样概率写 `sample_weight`，避免 WAIT 下采样改变目标分布。

初始建议每秒一个均匀 WAIT 样本，再用小规模 A/B 测试 1 Hz、2 Hz 和自适应
采样。场内物理仍逐一执行全部 20Hz Tick，降低的只是重复训练记录数量，不是
游戏精度或策略可行动频率。

按 5,440,619 个部署动作只保存 6-channel grid，约 17.51 GiB；所选牌的
576-bit Mask 只约 0.365 GiB。与逐 Tick超过 1 TiB 相比，这才适合 mmap shard 和
多轮训练。

## 6. 与 Expert-v1 BC Contract 对接

最终目录仍为：

`D:\AI_data\cr-native-core\expert-v1\compiled\native-bc-v1`

manifest 至少增加：

```json
{
  "source_manifest": "...version-window-20260804\\accepted.jsonl",
  "source_manifest_sha256": "9e3c...6855",
  "state_provenance_counts": {
    "authoritative": 0,
    "native_generated_unanchored": 0,
    "sequence_only": 0
  },
  "terminal_validation": {
    "anchor_provenance": "source_index_crowns",
    "anchor_count": 73556,
    "status": "pending_native_replay",
    "mismatches": null
  }
}
```

不能在尚未运行 Native replay 时把终局写成 `terminal_mismatches=0`。Runner
会在源时长后读取原生皇冠逐场比较；未执行写 `null`，执行过才写计数。
即使皇冠一致也不能将 unanchored scene 升格为 authoritative。

Actor 记录必须在写盘前按单方投影；当前 runner 已删除敌方当前手牌和精确
圣水，只保留己方 hand/next/elixir 与公开场上实体。生成状态只能进入单独的
`native_generated_unanchored` ablation，不能借 `action accepted` 升格为
authoritative。旧 schema-v1 继续作为 `sequence_only`，各 Head 依证据设置独立
`*_label_mask`。

## 7. 当前 native benchmark 阻塞证据

本轮尝试启动 4 Worker 时 Android Emulator 立即退出；当前机器实测：

```text
emulator-check accel
accel: 6
Android Emulator hypervisor driver is not installed on this machine
```

因此本轮没有伪造新的 libg 复演吞吐。历史项目证书仍是：

- validated native短路径平均约 7,618 tick/s；
- 进程内 reset 平均约 11.475 ms；
- 4 Worker完整训练环境约 368.28 steps/s；
- 2 AVD/8 Worker完整训练约 526.99 env steps/s。

这些是不同路径的既有实测，不能当成本次 gap-batched Expert replay 的新
benchmark。恢复 WHPX/AEHD 后必须运行 100/1,000 场真实复演，记录
`source tick/s`、场/s、动作/s、reset/step/observe/probe/action 分项、拒绝率和
各失败原因，再确定正式 Worker 数。

## 8. 入口

纯资格审计：

```powershell
python scripts/audit_expert_native_replay.py `
  --manifest D:\AI_data\cr-native-core\expert-v1\training-dataset\version-window-20260804\accepted.jsonl `
  --output-root D:\AI_data\cr-native-core\expert-v1\native-replay-audit `
  --workers 16
```

单场原生复演（Runtime恢复后）：

```powershell
python scripts/replay_expert_native.py SOURCE.json `
  --port 38031 `
  --output D:\AI_data\cr-native-core\expert-v1\replay-probe.json
```

默认只接受 `native_replay_candidate`。非候选必须显式
`--allow-non-candidate`，且结果始终保留 synthetic provenance。
