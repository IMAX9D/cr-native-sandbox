# 原生 Teacher-Forced Guard 与 Seed Search 诊断

## 结论

本轮解决的是原生回放适配层的两个问题，不是修改 `libg.so`：

1. `result_code=4/13` 已定位到冻结版 `libg` 的真实 guard；
2. 原先“重排八个 deck slot，再迭代校准 3 次”的布局算法已被替换。

新的布局路径：

- 始终保留来源 `full_deck` 的八卡顺序；
- 始终保留来源 Evo/Hero form slot；
- 不移动、不延迟、不跳过任何来源动作；
- 在缺失的 seed 空间内按 `1,2,3,...` 做确定性有界搜索；
- 只有当两方由真实 `libg` 给出的 `4手牌 + 4队列` 都能执行各自完整八卡动作流时才接受；
- 搜索耗尽则 fail-closed；
- `chosen_seed` 只是一个兼容的合成初始手牌变量，**不是恢复出的原始 seed**。

## `result_code=4`

冻结版 `DoSpellCommand::execute` 位于 `0xD8D520`。反汇编控制流为：

- 调用 `0xD503D0`；
- predicate 为真时返回 `4`；
- 此分支位于 placement 与卡费检查之前。

因此 code 4 的含义是 battle-level command gate 已关闭，不是落点错误，也不是
手牌映射错误。

JNI 现在在动作执行前只读导出：

- `D50CD0` hard gate；
- `D503D0` command gate；
- `battle phase / flag_1e9`；
- `logic state / substate`；
- `battleLogic+0x198` 终局计数；
- `battleLogic+0x60` player count；
- mode config `+0x110/+0x116/+0x191/+0x194`；
- 当前圣水、卡费、差额、refill、hand/cycle size。

提交：`5d90e56`。

### 两个真实矩阵

固定配置均测试 seeds `1 / 2 / 424242`：

| 样本 | 来源终局 | Friendly/Ranked 结果 | Ladder 16级结果 |
|---|---:|---|---|
| `02RY9QJQ8QQR` | `[1,0]` | Tick 3765，68动作，gate 4，`+0x198=101`，`crowns=[1,3]`，蓝王塔 0 HP | Tick 3526 已原生终局，`crowns=[3,1]` |
| `00CYPY2LLYJ2` | `[1,0]` | Tick 1185，10动作，gate 4，`+0x198=2901`，`crowns=[1,3]`，蓝王塔 0 HP | Tick 914，gate 4，`crowns=[3,0]` |

每个配置内三个 seed 的逻辑状态哈希完全相同。seed 经过布局归一后不是这两例
终局分叉的原因。

证据：

- `D:\AI_data\cr-native-core\expert-v1\native-teacher-forced-diagnostics\02RY9QJQ8QQR-command-gate-matrix.json`
- `D:\AI_data\cr-native-core\expert-v1\native-teacher-forced-diagnostics\00CYPY2LLYJ2-command-gate-matrix.json`

Friendly `72000007` 和 Ranked1v1 `72000323` 都把本矩阵的塔压到锦标赛级：
国王塔 `4824`，公主塔 `3052`。只把 `avatar.expLevel/kt/hbd.kt` 从 11 改为
16 不会改变实际 max HP。Ladder `72000006` 才得到 16 级国王塔 `7728`、
公主塔 `4858`。因此 RoyaleAPI 的账户卡级不能直接当作这批 Ranked 对局的
实际王塔等级；在缺少 league/cap 元数据时必须保留 provenance，不能猜。

这些矩阵说明 code 4 是正确拒绝：未锚定的生成态已经真实三冠终局，来源 JSON
却仍有后续动作。通过忽略 code 4 继续写 Tick 会伪造训练数据。

## `result_code=13`

`0xD8D7B2` 比较当前整数圣水与 packed selection 的高 4-bit 卡费。当前圣水小于
卡费时返回 `13`。详细 T/T+n A/B 见：

- `docs/NATIVE_CODE13_ELIXIR_TICK_DIAGNOSTIC.zh-CN.md`
- `D:\AI_data\cr-native-core\expert-v1\native-code13-tick-ab-v1.json`

本轮没有平移任何动作 Tick。

## 为什么旧布局算法不成立

旧路径先从一个 bootstrap deck 取得 shuffle permutation，再重排来源八卡的输入
slot，希望固定 seed 下得到某一个任意选中的兼容初始状态。如果实际不匹配，就把
本次结果作为下一轮 calibration，最多循环三次。

真实三轮证据表明：shuffle 还依赖卡牌身份，不能视为只依赖 slot 的固定排列。

以 `022YYLPR8C0R` 的 side 1 为例：所选逻辑队列要求最后两张为 `6 -> 7`。
当输入映射为 `6->slot0, 7->slot1` 时，libg 实际返回 `slot1 -> slot0`；交换输入
slot 后，libg 又返回 `slot0 -> slot1`。换回逻辑身份后，两次都仍是 `7 -> 6`，
所以旧固定点迭代形成稳定二周期，永远不会收敛。

三个失败样本均排除了重复卡：

- 八个 base data ID 唯一；
- Dark Prince Hero 真实解析为 `203000027`；
- Berserker Hero 真实解析为 `203000076`；
- Evo/Champion 入口也解析到预期原生数据。

所以根因是 calibration 算法假设错误，不是缺卡、重复 ID 或 Hero 形态未加载。

三轮原始证据保存在：

- `D:\AI_data\cr-native-core\expert-v1\native-teacher-forced-diagnostics\layout-three-rounds.json`
- 同目录 `layout-three-rounds-seed1.json`
- 同目录 `layout-three-rounds-seed2.json`

## 新 Seed Search

实现：`expert_v1/native_seed_search.py`。

### 搜索约束

对每一方：

1. 从 native observation 读取真实 `hand_deck_indices + cycle_deck_indices`；
2. 逐个应用完整来源卡牌序列；
3. 每次出牌必须在手；
4. 出牌后严格按原生八卡 refill 队列迁移；
5. 两方同时满足才接受 seed。

默认最多测试 `4096` 个 seed。上限可由 pilot CLI 的 `--maximum-seeds` 调整。
耗尽时抛出 `NativeSeedSearchError`，记录 battle tag、已测数量、上限和 legacy
preferred seed，不会退回重排 slot、跳动作或平移动作。

### 缓存

进程内缓存 key 包含：

- 两方有序的 `(card_id, form_flags, level)`；
- 两方 tower troop；
- 两方完整动作流对应的 compatible-origin set SHA-256。

缓存命中仍会在当前 Worker 上执行一次真实 reset 并复验完整约束；复验失败即驱逐
缓存并重新有界搜索。因此缓存不能绕过原生验证，也不会造成跨 Worker 串局。

### 历史三例回归

保留来源 deck/form slot 后，从 seed 1 递增：

| battle | chosen seed | seeds tested | 两方完整循环 |
|---|---:|---:|---|
| `008YLPVGR8GR` | 9 | 9 | 通过 |
| `022YYLPR8C0R` | 13 | 13 | 通过 |
| `02PYPJJRY9VG` | 53 | 53 | 通过 |

这些 seed 只是兼容 seed，不是原始对局 seed。

## 固定 v6 100 场 Seed Search 基准

输入为 v6 的原始固定 selection，4 个真实 Worker，fresh cache：

| 指标 | 结果 |
|---|---:|
| 成功找到兼容 seed | 100/100 |
| 搜索失败 | 0 |
| 总 native reset | 3,216 |
| 平均 seeds tested | 32.16 |
| median / p95 / p99 / max | 24 / 117 / 129 / 130 |
| 4 Worker wall | 7.502 s |
| battle/s | 13.329 |
| cache hit | 0（此固定 100 场 key 均唯一） |

保守假设 10 万场 key 全部唯一，seed resolution 约需：

- 3,216,000 次 native reset；
- 当前 4 Worker 实测速率约 `7,502 s = 2.08 h`。

这是 seed resolution 单独成本，不包含逐 Tick 回放。真实 10 万场若出现相同
deck/constraint key，缓存会降低 reset 数；在测到完整 10 万 key 分布前不能声称具体
命中收益。

完整基准：

- `D:\AI_data\cr-native-core\expert-v1\native-seed-search-benchmark-v1\summary.json`

## 固定 selection 小规模完整重放

使用 v6 selection 的前 10 场：

- 旧路径成功 `2/10`；
- 新路径成功 `2/10`；
- 两个旧 layout failure 均被消除，继续执行后分别在真实 code 13 处停止；
- 其余 code 4/code 13 与旧版一致；
- 没有通过延迟、跳过或忽略原生拒绝提高成功率。

输出：

- `D:\AI_data\cr-native-core\expert-v1\native-teacher-forced-seed-search-smoke-10-v1`

这说明布局适配已经修正，但 teacher-forced 全量数据仍受生成态与来源真实状态缺少
锚点、圣水相位和原生提前终局影响。Seed Search 只解决初始八卡循环，不把整个
JSON 回放升级成原始逐 Tick 状态真值。
