# 权威候选统一原生逐 Tick 生成器 v1

## 1. 目的与边界

本工具把已经通过静态资格审计的专家对局逐场送入原版 `libg.so`。旧
schema 3 队列继续可恢复运行；新抓取的 authoritative schema 5 队列使用同一生成器：

1. 按 RoyaleAPI 原始 `time_raw` 标签读取出牌和技能事件；
2. 按 profile v1 在 `T+1` 原生执行边界提交事件；
3. 原生核心以 20 Hz 连续演算；
4. 保存每一个完整的原生 Tick；
5. 第一处手牌、动作、技能实体、Tick 或逻辑阶段差异立即停止该场；
6. 只有整场 teacher-forced 动作全部接受后，才写入不可变 Tick Store。

工具只负责编排和审计，不实现、替换或修改任何游戏规则。

固定输入队列：

```text
D:\AI_data\cr-native-core\expert-v1\native-eligibility-v1\queues\authoritative-native-full.jsonl
```

旧冻结队列包含：

- 24,026 场带精确技能 Tick 的对局；
- 2,359 场来源明确报告没有技能事件的对局；
- 全部使用 schema 3 完整八卡、等级、形态、塔兵信息；
- 全部部署事件都有原始 `x/y/data_i`；
- 不包含旧 schema 的近似技能时间或 legacy 坐标。

新 schema 5 候选除了上述动作条件，还必须在候选验证和实际执行前两次通过：

- 冻结 native contract 三元组匹配；
- numeric game mode 与 battle index 完整；
- 双方塔兵等级完整；
- 来源最终六塔 HP 完整；
- `compile_battle()` 重新验证所有 schema 5 字段，不能只相信队列布尔值。

schema 3 任务格式和已有 SQLite 队列不变，因此无需迁移正在运行的旧任务。

## 2. 数据语义

### 2.1 时间

正式生成固定使用：

```text
royaleapi_native_teacher_forced profile v1
source marker = time_raw T（不可改写）
native command boundary = T+1
native Tick = 20 Hz / 50 ms
```

生成器没有开放 `T+0` 正式输出开关，避免把历史相位诊断误混入训练数据。

### 2.2 坐标

唯一允许的坐标来源为：

```text
royaleapi_raw_data_i_to_native_v1
data_i=0 -> rotate_18000_32000
data_i=1 -> identity
```

每场重新核对：

- `raw_data_i_events == deployment_actions`；
- `legacy_xy_fallback_events == 0`；
- 编译后的部署数、技能数与资格队列一致。

### 2.3 技能

技能按钮在原始事件 Tick 到来时，根据当时原生实体和卡组能力映射解析：

- 唯一合法实体：执行；
- 无合法实体：该场 fail-closed；
- 多个合法实体：记录 `branch_required`，该场 fail-closed；
- 永远不猜一个实体，也不凭空补技能 Tick。

### 2.4 原生状态声明

输出是“同版本 libg + 专家原始动作序列”的原生 teacher-forced 轨迹，不声明恢复 RoyaleAPI 未提供的原始 RNG/隐藏状态。每场 seed 只用于寻找与已观察八卡循环兼容的初始手牌布局。schema 5 会把来源 numeric game mode 与塔兵等级准确送入 replay；King Tower level 和 exact source build 仍是缺失字段，不能由此宣称原始状态完全一致。

正式 run contract v4 将 semantic candidate limit 固定为 1：升序 raw seed scan 保持
不变，找到首个双边布局兼容 seed 后只执行一次无 trace/no-mask preflight。该 seed
随后固定用于 full trace 或 censored prefix trace，并继续要求逐字段 semantic parity。
技能多候选仍直接失败，不能借性能优化选择或猜测技能实体。cap=8 的 v3 产物因
contract version、pipeline mode、audit kind 与显式 candidate-limit 字段不同而不能续跑。

schema 5 的终局六塔 HP 是诊断锚点，不影响 teacher-forced 动作接受的定义。由于
来源 Princess 槽位 0/1 尚无 native 左右映射证据，比较采用 King 精确值、两个
Princess HP 多重集合和 total；结果单独写入 tower-HP diagnostic 字段。

## 3. 输入不复制

源 JSON 始终在原位置只读：

1. 队列冻结 `source_path + source_sha256`；
2. 每次真正执行前重新计算 SHA-256；
3. SHA 不同立即停止该场；
4. 输出只保存源路径、SHA、编译计划和差异诊断；
5. 不把原始 JSON 复制到生成目录。

因此训练区不会再产生一份 100k 源回放副本。

## 4. 输出结构

默认目录：

```text
D:\AI_data\cr-native-core\expert-v1\native-authoritative-ticks-v1
```

结构：

```text
run-contract.json          不可变运行语义和组件 SHA
selection.jsonl            仅包含源指针/哈希/审计字段，不含源 JSON
selection.summary.json
work-queue.sqlite3         WAL lease 队列，可恢复、可抢占
results/<battle>.json      每场最终结果
diagnostics/<battle>.json  每场失败的完整结构化证据
workers/*.json             Worker 统计
shards/*.crts              成功场次的不可变 Tick 数据
shards/*.index.jsonl
shards/*.manifest.json
shards/manifest.json       全局 Tick Store 内容清单
results.jsonl              按 selection_index 原子聚合
summary.json               原子发布的运行统计
manifest.json              完整运行结束后才发布
manifest.sha256
```

每个失败诊断至少含：

- 首个差异阶段和错误分类；
- 首个原生拒绝的请求、result code、原生 Tick；
- 最近 reset/action/trace 边界状态；
- structured logic-freeze 诊断；
- 编译计划、seed search、坐标和技能解析 provenance；
- 异常 traceback（如有）。

## 5. 成功才写不可变分片

一场回放先进入内存暂存器。以下条件全部成立后才提交：

- `teacher_forced_success == true`；
- 已接受动作数等于源部署数加源技能数；
- 每个原生动作都有真实接受响应；
- Tick 连续；
- Tick 数等于完整 trace frame 数；
- 没有猜测技能分支。

提交使用 checksummed append-only frame。若进程在“frame 已 fsync、SQLite 尚未 complete”的狭小窗口崩溃，续跑会发现并逐字节校验孤儿 frame；内容完全一致才复用，否则停止，绝不静默覆盖。

## 6. Resume 与并发

- SQLite 使用 WAL 和租约抢占；
- `prepare` 与 `run` 共用同一个 OS 排他锁；
- 每个原生 Worker 独占自己的 append-only shard；
- 活跃任务定时续租；
- 续租一直覆盖到原子 result 写入和 SQLite terminal commit；
- 写 frame 前再次验证 lease owner/expiry；
- 进程硬退出后，OS run lock 自动释放；
- 下一次启动立即回收上次中断的租约，无需等待 15 分钟；
- 已完成/已失败任务不会重复执行；
- 运行契约或 selection 改变时拒绝在原目录续跑；
- summary、selection、results 聚合和最终 manifest 都以临时文件原子替换。

若进程在 `.partial -> .crts` 已完成、shard manifest 尚未发布的极小窗口退出，下一次启动会用 data/index 的逐 frame 校验重建缺失 manifest。已完整发布的 run 再次启动时，也会重新计算每个 `.crts` 和 index 的 SHA 并扫描 frame，而不是只相信顶层 manifest。

失败严格区分为三个 domain：

- `semantic`：明确白名单内的原生拒绝、技能实体缺失/分支、提前终局、structured logic freeze；永久止损；
- `source_integrity`：编译契约、手牌序列或来源计划不一致；永久止损并禁止发布；
- `infrastructure`：RPC/timeout、Host、协议计数/Tick、Tick Store 编码/磁盘、源 SHA 读取变化及任何未分类错误；单次运行最多尝试 3 次，仍失败则禁止发布。

新一次 `run` 默认只把上次的 `infrastructure` failed task 重新置为 pending，并给它新的 3 次尝试预算；`semantic` 和 `source_integrity` 永不自动重排。可用 `--no-retry-infrastructure-failures` 关闭跨运行重排。任何未知 runner failure 默认属于 infrastructure，绝不会被静默当成可发布的语义拒绝。

端口之间使用 SQLite work stealing，因此慢场不会永久拖住某一个固定分区。

## 7. 真正的动作接受率

不能用“已接受动作 / 全部计划动作”冒充原生接受率，因为一场在中途失败后，后续动作根本没有提交。

本工具分别记录：

```text
native_actions_attempted
native_actions_responded
native_actions_accepted
native_actions_rejected
native_actions_no_response
native_action_exceptions
true_attempted_acceptance_rate = accepted / attempted
```

部署和技能也分别提供真实尝试分母，并验证 `attempted = accepted + rejected + no_response`。`planned_actions` 仅表示数据覆盖，不参与原生接受率分母。

## 8. 使用方法

### 8.1 一键正式生成或续跑

双击：

```text
START_GENERATE_EXPERT_NATIVE_TICKS_V1.cmd
```

入口会先确保 4 个无界面原生 Worker 就绪，再开始或恢复默认输出目录。

### 8.2 命令行

```powershell
D:\AI_data\runtime\venv\Scripts\python.exe `
  scripts\generate_expert_native_ticks.py run `
  --workers 4 `
  --ports 38031 38032 38033 38034
```

查看状态：

```powershell
D:\AI_data\runtime\venv\Scripts\python.exe `
  scripts\generate_expert_native_ticks.py status
```

只冻结 selection 和创建恢复队列、不调用 libg：

```powershell
D:\AI_data\runtime\venv\Scripts\python.exe `
  scripts\generate_expert_native_ticks.py prepare
```

### 8.3 受控 smoke

```powershell
D:\AI_data\runtime\venv\Scripts\python.exe `
  scripts\generate_expert_native_ticks.py run `
  --limit 10 `
  --selection-seed mixed-smoke-v1 `
  --output-root D:\AI_data\cr-native-core\expert-v1\native-generator-mixed-smoke-v1
```

有限 selection 在候选池允许时强制至少包含一场技能正例和一场零技能对局，其余按 `sha256(selection_seed + battle_tag)` 稳定选择。

### 8.4 固定 1,000 场分层验收 selection

```powershell
D:\AI_data\runtime\venv\Scripts\python.exe `
  scripts\generate_expert_native_ticks.py run `
  --limit 1000 `
  --deployment-zero-quota 100 `
  --ability-exact-quota 900 `
  --selection-seed authoritative-1k-stratified-v1 `
  --output-root D:\AI_data\cr-native-core\expert-v1\native-authoritative-1k-v1
```

显式配额按两个 stratum 各自的 SHA 排名稳定抽样。必须同时提供两个配额，二者之和必须严格等于 `--limit`；任一层数量不足都会在创建队列前 fail-closed。默认全量入口不配置配额，仍消费全部 26,385 场。

## 9. 完成判据

`publication_ready=true` 表示：

- selection 中每个任务都有且只有一个最终结果；
- 无 pending/leased 任务；
- Worker 无基础设施错误；
- 源 SHA、profile v1、data_i 坐标 provenance 全部通过；
- 所有成功结果恰好对应 Tick Store 中的不可变 episode；
- 全局 shard/hash/episode/tick 数复核通过。

这不要求所有候选都成功。手牌差异、原生拒绝、`branch_required` 或结构化 logic freeze 是候选筛选的真实证据；它们保留完整诊断，但不会进入成功 Tick Store。
