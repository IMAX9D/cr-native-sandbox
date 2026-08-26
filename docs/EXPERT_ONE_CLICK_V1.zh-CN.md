# Expert Schema5 / contract-v3 一键闭环 v1

双击仓库根目录的 `START_EXPERT_ONE_CLICK_V1.cmd`，入口会以前台常驻监督器的方式，从权威 Schema5、native ingest contract v3 数据采集一直推进到正式专家模型训练。这里的 `v3` 是 **ingest contract/产物代际**；战斗 JSON 文件格式仍固定为 `schema_version=5`。入口不会读取旧 Schema3 数据，也不会读取历史 `native-eligibility-v1` 队列。

## 固定流程

1. 校验 crawler 配置、原生 ingest contract v3 和独立输出目录 `authoritative-schema5-v3`。contract v2、`authoritative-schema5-v2` 或旧 one-click state 会立即 fail-closed。
2. 未满 100,000 场时启动或接管 `crawler.authoritative_production`，持续监督并等待；监督器意外退出会被恢复。
3. 正好达到 100,000 场后停止 crawler、checkpoint SQLite；此后才读取可用物理内存并把原生布局永久固化到 journal（可用内存至少 16 GiB 时为 2 AVD × 4 Worker，否则为 1 AVD × 4 Worker），随后冻结内容寻址 Schema5 manifest。
4. 从该冻结 manifest 重新生成 eligibility audit。native 候选队列会逐行验证：只能是 Schema5、必须有权威 contract-v3 标记、源文件必须位于 `authoritative-schema5-v3` 根目录且 SHA 相同。任何旧 contract 或 Schema3 行立即 fail-closed。
5. 获取跨所有 `--data-root` 共用的原生硬件 OS 锁，启动固化布局的 direct Worker，准备或恢复原生 Tick 生成。已有工作队列、结果和 CRTS shard 会原地续跑。
6. 候选、selected 和 processed 必须全部为 100,000；每个 JSON 都必须形成原生 attempt 终态。默认要求 teacher-forced 总成功率至少 50%。此外候选会按 `ability_events_observed > 0` 分为能力正样本和能力零样本，并分别记录尝试、成功、失败、成功率与失败类别；只要冻结候选中存在能力正样本，默认还要求至少 1 场成功且该组成功率至少 10%。因此普通对局的高成功率不能掩盖技能局全部失败。完整分类写入 `native-generation-coverage.json`。
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
