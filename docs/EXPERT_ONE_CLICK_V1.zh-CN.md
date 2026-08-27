# Expert Schema5 / contract-v3 一键闭环 v1

双击仓库根目录的 `START_EXPERT_ONE_CLICK_V1.cmd`，入口会以前台常驻监督器的方式，从权威 Schema5、native ingest contract v3 数据采集一直推进到正式专家模型训练。这里的 `v3` 是 **ingest contract/产物代际**；战斗 JSON 文件格式仍固定为 `schema_version=5`。入口不会读取旧 Schema3 数据，也不会读取历史 `native-eligibility-v1` 队列。

## 固定流程

1. 校验 crawler 配置、原生 ingest contract v3 和独立输出目录 `authoritative-schema5-v3`。contract v2、`authoritative-schema5-v2` 或旧 one-click state 会立即 fail-closed。
2. 未满 100,000 场时启动或接管 `crawler.authoritative_production`，持续监督并等待；监督器意外退出会被恢复。
3. 正好达到 100,000 场后停止 crawler、checkpoint SQLite；此后才读取可用物理内存并把原生布局永久固化到 journal（可用内存至少 16 GiB 时为 2 AVD × 4 Worker，否则为 1 AVD × 4 Worker），随后冻结内容寻址 Schema5 manifest。
4. 从该冻结 manifest 重新生成 eligibility audit。native 候选队列会逐行验证：只能是 Schema5、必须有权威 contract-v3 标记、源文件必须位于 `authoritative-schema5-v3` 根目录且 SHA 相同。任何旧 contract 或 Schema3 行立即 fail-closed。
5. 获取跨所有 `--data-root` 共用的原生硬件 OS 锁，启动固化布局的 direct Worker，准备或恢复原生 Tick 生成。已有工作队列、结果和 CRTS shard 会原地续跑。
6. 候选、selected 和 processed 必须全部为 100,000；每个 JSON 都必须形成原生 attempt 终态，并且必须落入 Full Tick Store 或 Mask 完整、严格 right-censor 的 Prefix Store，二者互斥并集正好覆盖 100,000 场。teacher-forced full-success rate 与 `full_success_episodes` 继续记录为诊断，训练准入则使用经最终数组、Mask、censor 复验的 `admitted_training_episodes`；Prefix 不得伪装成 Full。训练有效性最终由逐-token 的卡牌、形态与 ability 标签门保证。完整分类写入 `native-generation-coverage.json`。
   Source/ability transcript 保持 v1；success、quota 与最终 token receipt 使用独立 v2 kind/schema，旧 v1 aggregate 一律拒绝。
7. 生成结束后先逐实例关闭全部 Worker 和 AVD；异常路径也会 best-effort 关闭，然后才释放全局硬件锁。
8. 对 Tick Store 做完整物理扫描：校验全部 CRTS/index SHA、帧计数，以及每场成功样本引用的内容寻址部署 Mask。
9. 并行编译 native BC 数据集。编译器必须重新认证上述 coverage receipt，并把 receipt SHA 与完全相同的能力覆盖对象写入最终 manifest；不满足能力门或编译后成功样本数不一致时拒绝发布。随后按确定性 shard 续跑，检查 split、Mask、标签、非有限值、原生拒绝和终局等 quality gates。
10. 使用正式编译数据而不是合成数据，运行 2 个 train batch 的真实小 smoke。
11. smoke 通过后调用 `expert_v1.training_v1.train --resume`，启动或恢复固定 Run `expert-v1-schema5-v3-100k`。

## 断点与安全边界

控制状态保存在：

```text
D:\AI_data\cr-native-core\expert-v1\one-click-schema5-v3\control\state.json
```

每个阶段在开始时写入输入文件 SHA-256，在完成时写入输出文件 SHA-256。重新双击时，已完成阶段只有在输入、输出逐字节未变化时才跳过；任何漂移都会停止并保留现场。单实例 OS 文件锁阻止两个入口同时写同一批数据，另一个固定的全局锁阻止不同数据根同时争用 AVD/ADB/direct ports。

采集阶段另外固化 `collection-runtime-fence-v1`：包含 one-click 自身、原生契约读取器、从 `authoritative_production/main/lane_watchdog/cf_recover` 静态递归得到的全部 crawler 项目模块、配置引用的种子/排除/升级清单、crawler Python/DLL、requirements，以及 curl_cffi、selectolax、patchright、PyYAML、ruyipage 的内容树 SHA。每次 30 秒轮询、命中 100,000 的同一轮、停止 crawler 前后、SQLite checkpoint 前后都会重新验签。活跃 crawler 的 OS 创建时间还必须不早于这些运行文件的最新 mtime；配置、契约、代码或依赖漂移会在旧 inputs 被标记 completed 前 fail-closed。

早期 state-schema-v2 曾在没有完整 runtime closure 的情况下启动采集。新版只对一种状态执行一次安全迁移：`collect_schema5_v3` 必须是唯一且仍为 `running` 的阶段，原六项 inputs 必须逐字节仍匹配，不得存在 native layout、后续阶段或 completed 输出，并且活跃 crawler 必须通过上述 OS 启动时间证据。迁移会先把原始 `state.json` 字节归档为 `state.pre-runtime-fence-v1.<sha16>.json`，再写入新 inputs 和迁移 receipt。任何条件不满足都会拒绝迁移；应保留现场并换新 data root。因为已经运行的 Python 不会热加载新代码，部署该加固后必须先正常停止旧 one-click 监督器，再重新双击；crawler 可保持运行，由新入口核验并接管。

v3 使用新的 state schema 和阶段名（`collect/freeze/audit_schema5_v3`）。把 v2 的 `state.json` 复制到 v3 根目录，或显式把 `--data-root` 指向 `one-click-schema5-v2`，都会拒绝启动；不能通过改目录名续跑旧 manifest、shard 或 checkpoint。

耗时阶段均具备恢复边界：crawler 使用原 SQLite 队列，native generator 使用自己的 SQLite work queue 和不可变 shard，编译器使用确定性 `compile-plan.json`/shard，训练使用完整 optimizer、normalizer 和 RNG checkpoint。

## 使用

生产运行（默认长期运行）：

```text
START_EXPERT_ONE_CLICK_V1.cmd
```

只查看状态，不启动或停止任何服务：

```powershell
scripts\start_expert_one_click_v1.ps1 -Status
```

正式编译数据已经完成且 AVD 停止凭据存在时，只重跑真实数据 smoke：

```powershell
scripts\start_expert_one_click_v1.ps1 -Smoke
```

日志保存在 `...\one-click-schema5-v3\logs`。失败时不要删除状态或产物；修复外部原因后重新双击，流程会从当前阶段恢复。`--smoke` 不会启动 crawler 或 AVD，也不会用合成数据填补缺失产物。

能力正样本门只能通过命令行显式豁免，并且必须留下原因：`--waive-ability-positive-coverage --ability-positive-waiver-reason "问题编号/原因"`。双击入口不会传入该参数。降低 `--minimum-ability-positive-success-count` 或 `--minimum-ability-positive-success-rate` 同样必须显式豁免；receipt 会永久记录 `waiver_applied` 和原因。

正式 v2→v3 迁移 apply 完成前，canonical contract 仍是 v2；此时入口会有意拒绝运行。只有 canonical contract、SQLite contract binding 和 crawler v3 输出目录三者一致后才能启动，禁止为了提前运行而关闭该检查。

## 并发配置

默认针对当前 20 逻辑核 / 32 GB 主机：

- crawler 使用 `config.authoritative.toml` 内自己的 22 lane、128 全局并发配置；
- eligibility audit 使用 20 个线程；
- 原生生成每 AVD 固定 4 Worker；crawler 完全停止后若可用内存至少 16 GiB，使用 2 AVD/8 Worker（38031–38038），否则使用 1 AVD/4 Worker（38031–38034）；
- BC 编译请求最多 32 个 I/O 校验线程和 10 个进程 Worker；实际进程数由容量预检
  按当前可用内存自动向下收敛并写入 receipt，无需换数据根重跑；
- 训练数据加载和 CUDA 设备选择沿用正式 `training_v1` 配置。

原生布局首次选择后单独写入 journal，恢复时直接读取，绝不会因当前 RAM 波动重新选择。非默认端口会 fail-closed；若要改变语义或并发，应使用新的 `--data-root`，避免拿新参数静默续跑旧产物。

## 逐 Token 训练覆盖门

冻结阶段会从 100,000 个已验签 Schema5 源文件重新计算并保存
`receipts/source-token-coverage-v1.json`。统计严格区分 180 个 deck/play token、
42 个 Evo 形态、16 个 Hero 形态和 25 个主动技能候选 token。原始
`ability_plays` 只形成候选与 Tick 注册表，绝不被当作具体技能身份。

完整 teacher-forced 成功局会在 generator 结果中留下两个 actor 的内容寻址
证据。部署标签绑定实际 Tick 的 Mask sidecar 与 `resolved_data_id`；技能标签
绑定冻结 source marker、libg candidate entity/card、selected entity，以及
Tick Store 中该实体的精确 native form。失败 Prefix 的证据数组必须为空，
它永远不能增加训练覆盖。

BC compiler 会再次读取 frozen source、generator result、Tick Store 和 Mask，
独立重建这些连接。每个观察到的 token 至少需要一条完整成功且最终编译的
监督样本，同时执行来源频率自适应门：Card 上限 16 局/64 标签，运行时形态
上限 8/16，主动技能上限 8/32。最终证据写入
`compiled/native-bc-v1/token-coverage-receipt.json`，其 file SHA、canonical
SHA 和无缺口 gate 一并绑定最终 dataset manifest。

本版不进行无限 repair loop。任一 token、形态或技能仍有缺口时，compiler
先持久化完整 deficit receipt，再以 `FAILED_COVERAGE` 停止；one-click journal
会保留该路径作为证据，不会发布 manifest，也不会进入 smoke 或正式训练。
