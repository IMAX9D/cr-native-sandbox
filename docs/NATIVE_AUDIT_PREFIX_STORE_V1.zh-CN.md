# 原生失败前缀 Actor-BC Store v2

正式原生生成使用两个物理隔离的 Tick Store：

- `shards/`：仅保存完整 teacher-forced 成功局，带完整部署 Mask，供 BC 编译器训练；
- `audit-prefix-shards/`：保存确定性 semantic preflight 失败局在首次失败前的连续原生 Tick，固定为 `training_admission=actor_bc_censored_prefix_v1`。

失败前缀使用升序扫描得到的首个 layout-compatible seed 固定重置；生产 semantic candidate limit=1，不猜测技能实体。Prefix trace 同时采集部署 Mask；只允许 partial slot metadata，但必须逐 Tick 证明每个可见手牌 deck index 均有内容寻址 native sidecar。合法的 `-1` 补牌空槽被无损保存为 PAD，且 card mask 恒为 false。重放的 failure、接受动作序列、最终 Tick、终局诊断等语义必须与 preflight 完全一致，否则按 instrumentation/infrastructure divergence 处理且不发布前缀。

每个前缀 episode 的 `native_replay_extent_v1` 固化：观察 Tick 范围、首次失败 source/execution Tick、首个无效事件、动作标签停止边界、timing censor 边界、已安全接受的历史动作摘要、可见手牌 Mask 覆盖审计，以及 `terminal_target=unknown_censored`。失败 Tick 可以作为 pre-action 审计状态物理保存，但只有 `tick < timing_censor_tick_exclusive` 的 Actor 行可进入训练；失败 Tick 及之后所有动作标签为 false，末端 timing 是 right-censor，不生成终局目标。

Prefix Store 保持独立 kind `cr_native_tick_prefix_audit_store_v1` 和独立 `deployment-masks-v1`。BC compiler 必须通过显式第二输入读取它，重新认证 extent、partial Mask、source action、native accepted transcript 和 prefix actor evidence；不得把 Prefix 文件混入 Full Store 或绕过 censor。

另有一个严格收窄的 Mask-censor 分支：仅当源回放继续向“模拟状态中仍被一座存活敌方公主塔锁定”的非桥、非全局法术 pocket 部署单位/建筑时，才允许 `actor_bc_mask_invalid_censored_prefix_v1`。它不能接收任意 Mask 差异。生产端必须额外执行一次同 seed、无 Mask 的参考重放，证明 preflight 语义一致、censor 前完整 TickState 逐字节一致、边界源动作确已被接受、Mask lane 未执行该动作，并固化 `native_mask_invalid_safe_censor_v3`。One-click 从已发布 Prefix frame 重算 pocket/塔血/sidecar；compiler 再从冻结 source、存储 Tick 与 sidecar 独立重算。英雄法术形态、桥边界、已破塔 pocket 或普通投影差异均不得走此分支。

One-click 的不可豁免覆盖门为：

```text
full_success_tags ∩ audit_prefix_tags = ∅
full_success_tags ∪ audit_prefix_tags = frozen_100k_tags
unframed_episodes = 0
audit_tick_coverage_rate = 1.0
compiled_full_episode_tags ∪ compiled_prefix_episode_tags = frozen_100k_tags
```

Full-success rate 和逐卡 `full_success_episodes` 继续记录为诊断指标，但不再要求至少 50%。训练 episode 准入使用独立的 `admitted_training_episodes`：只有经过最终数组、Mask 和 censor 复验的 Full 或 Prefix 才计数，Prefix 不会被伪装成 Full。训练覆盖由 Full + 安全 Prefix 的 100,000 场并集承担；卡牌、形态与 ability 的逐-token 门必须从最终编译数组的有效标签独立复算。RPC、source-integrity、seed-search、固定 seed 重放不一致、Mask 不完整或首帧前失败均不得伪造 Prefix；这些情况会令 one-click 停止并保留诊断现场。

版本边界保持分离：冻结 source coverage 与 ability resolution transcript 继续使用 v1；引入 `admitted_training_episodes` 的 success summary、adaptive quota 和最终 token coverage receipt 独立升级为 v2。消费者必须同时校验对应 v2 kind 与 `schema_version=2`，不得把旧 v1 aggregate 解释为新准入语义。
