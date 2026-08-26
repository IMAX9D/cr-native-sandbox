# 原生坐标修正后命令相位最终 100 场审计

## 结论

建议把 **native execution boundary** 从 source marker `T` 校准为 `T+1`：

```text
source label / RoyaleAPI time_raw：仍保存 T，不改写
native command execution boundary：T + 1
配置：action_execution_tick_offset = 1
```

这不是修补源数据，也不是把回放标签整体加一；它只定义源 marker 与 libg
命令消费边界之间的相位关系。本审计没有修改 runner 默认值，只给出下一步配置
变更建议。

固定坐标修正后的 99 场可比较样本给出了单调证据：

- offset 0 已成功的 `83/83` 场在 offset 1 下全部继续成功；
- offset 0 的 `6/6 code13` 全部转为完整成功；
- `10/10 code4` 的 battle tag 集合完全不变；
- 可比较成功率从 `83/99 = 83.84%` 提升到
  `89/99 = 89.90%`，增加 `6.06` 个百分点；
- 没有 success 退化，也没有 code13 迁移成 code4。

固定 selection 仍是 100 场。`089Y82CPYYY9` 单列为原生逻辑冻结诊断，不从
原始 100 场统计中删除，但不能把它误算成 offset 1 的动作拒绝，因此相位成功率
的分母是 99。

## 审计对象

| 分支 | 目录 | execution offset |
|---|---|---:|
| v9 | `D:\AI_data\cr-native-core\expert-v1\native-teacher-forced-pilot-100-data-i-v9` | 0 |
| v10 | `D:\AI_data\cr-native-core\expert-v1\native-teacher-forced-pilot-100-data-i-phase-plus1-v10` | 1 |

两次运行都已经包含正确的 RoyaleAPI → libg 坐标方向修复。除
`action_execution_tick_offset` 及其 provenance 文本外，两份 summary 的运行配置
完全相同：4 Worker、20Hz、trace batch 64、preferred seed 424242、最多搜索
4096 个 seed。

## 固定输入与 seed

| 检查 | 结果 |
|---|---:|
| selection battle tag | 100/100 相同 |
| selection 文件 | 逐字节相同 |
| selection SHA-256 | `5fa5239318ce2396934408ceab624d507ccaf9a89143752ed80f458ca0127a3a` |
| source SHA-256 | 100/100 相同 |
| chosen seed（结果中直接记录） | 99/99 相同 |

审计不仅比较结果行里的 source 字段，还重新读取并 SHA-256 计算了 100 个源 JSON；
磁盘内容与 selection 声明也全部相同。

冻结样本 `089Y82CPYYY9` 的 v9 chosen seed 是 `28`。v10 在 compact trace 抛错
时，通用 error row 没有序列化 `chosen_seed`，因此不能声称“100/100 都有两份
直接记录”。但 seed resolution 在执行 offset 被应用之前完成，且两次运行的
selection、source、preferred seed、最大搜索数及其他运行参数全部相同；所以该场
seed 相等是由执行路径结构确定的，机器报告明确标为 structural inference，而不是
伪装成直接观测。

## 为什么分母是 99，不是把 11 个 v10 failure 都当动作失败

`089Y82CPYYY9` 的证据为：

| 分支 | 证据 |
|---|---|
| v9 / offset 0 | 在 Tick 3681 执行动作，原生返回 code 3；`battle_command_hard_gate`，`hard_gate=true`，`logic_end_counter_198=4001` |
| v10 / offset 1 | 需要前进到 execution Tick 3682，但 compact trace 连续得到 `state_tick=3681`、`terminal=false`，libg 已不再推进 |

因此 v10 并没有在 T+1 边界执行并拒绝该动作；它根本到不了 T+1 边界。把它和
10 个真实 code4 一起写成“11 个 offset 1 动作失败”会误读结果。

本报告同时保留两套口径：

| 口径 | v9 | v10 |
|---|---:|---:|
| 原始固定 100 场 success / failure | 83 / 17 | 89 / 11 |
| 相位可比较 99 场 success / code13 / code4 | 83 / 6 / 10 | 89 / 0 / 10 |
| 单列逻辑冻结 | 1 | 1 |

冻结样本只从“相位接受率”分母排除，仍在 selection、source 与原始结果审计中保留。

## 逐类迁移

| v9 | v10 | 场数 |
|---|---|---:|
| success | success | 83 |
| code13 | success | 6 |
| code4 | code4 | 10 |
| logic-freeze excluded | logic-freeze excluded | 1 |

六个 `code13 → success`：

| Battle tag | v10 terminal diagnostic |
|---|---|
| `00CYPPG22CPJ` | match |
| `00VYPYPQV8QC` | match |
| `00YYPPGLR8YU` | missing at source-duration fence |
| `02QY9L89CYGV` | mismatch |
| `080Y8LY0PQ9L` | match |
| `09LP9JLR0U8Q` | match |

稳定的 10 个 code4 tag：

```text
008YLPVRJ09Y  00CYPY2LV28P  00LYPL9Y89JL  029YPJ0CC8PY
02GY9QRR0GY8  02PYPJGRV290  02UY8PLGYUUJ  08CPVRPY9PRJ
08PY829UP89G  090PPUJJJQ8G
```

这组结果比之前未修正坐标的相位 A/B 更有区分力：这次不是部分 code13 被
推迟成后续 code4，而是六场全部走完，且原有 code4 集合不增不减。

## 终局 match / mismatch

仅统计 teacher-forced success：

| terminal status | v9（83 场） | v10（89 场） | 变化 |
|---|---:|---:|---:|
| match | 65 | 68 | +3 |
| mismatch | 2 | 3 | +1 |
| missing at source-duration fence | 15 | 17 | +2 |
| logic frozen at fence | 1 | 1 | 0 |

六个新增成功贡献：`4 match + 1 mismatch + 1 missing`。

原来共同成功的 83 场中没有 `match → mismatch`：

- `match → match`：64；
- `mismatch → mismatch`：2；
- `missing → missing`：15；
- `match → logic frozen`：1；
- `logic frozen → missing`：1。

所以 mismatch 从 2 增到 3 来自一个以前无法走完、现在能够完整执行的新样本，
不是旧成功样本的终局退化。另一方面，精确可判定终局中的匹配率从
`65/67 = 97.01%` 变为 `68/71 = 95.77%`，不能隐藏这一点。

这里的 terminal crowns 是 teacher-forced 生成态的次级诊断。源 JSON 没有服务端
隐藏 RNG 和逐 Tick 原始 state truth，因此终局匹配不能单独裁决命令相位。它的
作用是检查新相位是否造成明显系统性退化；当前证据是 match 净增 3、旧成功集合
没有 match→mismatch，因而不推翻动作边界的单调证据。

## 按对局时长分层：剩余问题是长局缺锚累计漂移

| 源对局时长 | v9 success | v10 success | v10 成功率 |
|---|---:|---:|---:|
| `<= 180s` | 12 / 13 | 13 / 13 | 100.00% |
| `181–240s` | 49 / 55 | 52 / 55 | 94.55% |
| `> 240s` | 22 / 32 | 24 / 32 | 75.00% |

T+1 消除了三个时长层中的全部 code13，但 v10 的剩余 11 个原始 failure 高度集中
在长局：`181–240s` 只有 3 场失败，`>240s` 有 8 场失败；后者包含 7 个真实
code4 和单列的 logic-freeze。

这符合“缺少中局状态锚点导致累计漂移”的特征：当前 source 只提供专家动作，
teacher-forced libg 会从初始状态持续生成后续世界；对局越长，未观测 RNG、单位
交互、塔血和终局时序的微小差异越容易累积，最终可能让生成态 command gate 比
源动作序列更早关闭。短局已经 13/13，说明剩余失败不能继续笼统归因于坐标或统一
Tick 相位。

因此下一阶段应把两个问题分开：

1. 命令相位采用 T+1；
2. 针对长局研究可验证的周期状态锚点或分段 replay，不用另一个全局 offset 去掩盖
   累计漂移。

## Tick Store 全量解码与 SHA 审计

两边所有成功 episode 都逐 Tick 完整解码，不是只检查 manifest：

- 全局 `manifest.sha256` 与实际 `manifest.json` 一致；
- 每个 global/local shard manifest 完全一致；
- 每个 data/index SHA-256 与磁盘一致；
- 每个 episode frame header、payload SHA-256 与 index 一致；
- 所有 episode 逐块解压，Tick 严格 `+1` 连续；
- 首 Tick、末 Tick、Tick 数与 index/result 一致；
- store tag 集合与 teacher-forced success tag 集合相同；
- 全局 content SHA-256 重新计算一致。

| 分支 | 物理 shard | episode | Tick | bytes | manifest SHA-256 | content SHA-256 |
|---|---:|---:|---:|---:|---|---|
| v9 | 4 | 83 | 356,473 | 8,862,503 | `178afd3eb90acc426c3e7334e63fb9c12117757b3b59a5fcd35854a9432ef474` | `8bac863b91b3b72c7e766c25f87b3aec52e6cd6c3d9e779a67bbc023e78e3ab5` |
| v10 | 4 | 89 | 382,473 | 9,584,664 | `af9c24ffad1ad99d1fad4779a7373caffd85fbfdc6e178e2a0bbd2f7579e5a55` | `1c03615082ad71a240d8f1ce1ed9233b987342b3fe593c0e413348ec8942a59e` |

八个物理 shard 的 data/index SHA-256：

| 分支 / shard | episode | Tick | data SHA-256 | index SHA-256 |
|---|---:|---:|---|---|
| v9 / worker-00 | 17 | 80,062 | `404724c113143742f495f43f910169891a782d10bcc5fc9738e872d7f7d73c81` | `f5ff3aa5933689b675dcd30466b42839fcb435358ad8c3d9260f852a493e766a` |
| v9 / worker-01 | 23 | 92,626 | `1ac93ab56675cd98b6854d4f63048aa6d04ec8886d969c8e9aa796d5bc2dc266` | `ce9ae3dffca32584e791a2cee595f723fec6721c6a503c63069011c61bdb200f` |
| v9 / worker-02 | 24 | 102,286 | `d1d98dc53c303c28b119936671b972740a192ed3c21c4ee2476499f5925aade1` | `ba1af310aaf460ed6f6ef668c8170e1010001163bd4c8872905ba95e0d756cde` |
| v9 / worker-03 | 19 | 81,499 | `8d8a5a6d494fd942ebe9f8b027be12f80306312c0046f0c6ff7b88c2a219d9fb` | `152f7fc24b08e30e9a633fb569e9a06811a1e8b4a884b700a2615457a3f5a46e` |
| v10 / worker-00 | 21 | 95,175 | `500680fcb4d6f9f12720990302754faeb9b57c94d655ab5e4a9ca5cf19351c30` | `d36069c0fd4cf48199aa9c051e60aa76919069ef43445812ddc88d09be1d7fa6` |
| v10 / worker-01 | 22 | 90,884 | `efa2d68b16ea8f60f774468637342bc232bdeff7435ea9511af850834f5f96ca` | `2c32ba3778df028dfbf3fc03b63f04258dcb400bbcb91f4e467e56c6b89dc167` |
| v10 / worker-02 | 24 | 98,051 | `1374667583392aa4d328b0ef21caf3cd24b1fdbe2906000bfb907a157f601653` | `4ca8d55548a7de0adab5174ff2bca313b26bdfbc86e4451e98f0da2634604b29` |
| v10 / worker-03 | 22 | 98,363 | `89912bb855b97d4e4872963a60107826648ca7bdba359baf801474ed9d4995b3` | `8f9fdbb0870e5a2d744bd6f4d35b16963e2da034523e143ec6caaf5ea90fe917` |

## 最终配置建议

下一轮 native teacher-forced 数据生成应显式使用：

```text
action_execution_tick_offset = 1
action_tick_provenance = source label remains RoyaleAPI time_raw T;
                         native command executes at T+1
```

同时保留以下约束：

1. 不修改 source `time_raw`；
2. 每条结果与 Tick Store metadata 都记录 offset/provenance；
3. `089Y82CPYYY9` 这类“到不了目标执行边界”的逻辑冻结单列，不能归类成
   action reject；
4. code4 仍然是生成态 command gate 分叉，需要另行诊断，不能用相位偏移规避；
5. 正式切默认前，给 runner 默认值变更单独提交并运行现有 smoke/acceptance；本次
   审计提交不改默认。

## 机器报告与复核

机器报告：

```text
D:\AI_data\cr-native-core\expert-v1\native-teacher-forced-pilot-100-data-i-phase-plus1-v10\coordinate-phase-final-v9-v10.json
```

报告 SHA-256：

```text
4de9c74f17937114acaeecf28cfe268e0b4a939e17681c8da9a091b2ca58a0e6
```

报告包含 100 条逐 tag 迁移、原始与归一化分母、冻结证据、terminal 迁移、两边
全部 shard/episode/Tick/SHA 验证结果、时长分层及 12 条固定断言；当前全部为
`true`。

复核命令：

```powershell
D:\AI_data\runtime\venv\Scripts\python.exe `
  scripts\audit_native_coordinate_phase_final.py
```

脚本只读 v9/v10 pilot 目录；唯一写入是上述机器报告，不启动或修改 libg，不修改
runner 默认值。
