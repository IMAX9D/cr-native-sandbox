# 原生失败前缀审计 Store v1

正式原生生成使用两个物理隔离的 Tick Store：

- `shards/`：仅保存完整 teacher-forced 成功局，带完整部署 Mask，供 BC 编译器训练；
- `audit-prefix-shards/`：仅保存确定性 semantic preflight 失败局在首次失败前的连续原生 Tick，固定为 `training_admission=audit_only`。

失败前缀使用 preflight 已选出的 seed 固定重置，不重新搜索 seed，不采集部署 Mask。重放的 failure、接受动作序列、最终 Tick、终局诊断等语义必须与 preflight 完全一致，否则按 instrumentation/infrastructure divergence 处理且不发布前缀。

每个前缀 episode 的 `native_replay_extent_v1` 固化：观察 Tick 范围、首次失败 source/execution Tick、首个无效事件、动作标签停止边界、timing censor 边界、已安全接受的历史动作摘要，以及 `terminal_target=unknown_censored`。失败 Tick 可以作为失败前的 pre-action 审计状态保存，但 `failure_tick_has_labels=false`，不得生成 PLAY、WAIT、卡牌、位置、技能或终局标签。

Audit Prefix Store 使用独立 kind `cr_native_tick_prefix_audit_store_v1`。BC compiler 遇到该 kind 必须明确拒绝，不能通过参数或 waiver 加入训练。

One-click 的不可豁免覆盖门为：

```text
full_success_tags ∩ audit_prefix_tags = ∅
full_success_tags ∪ audit_prefix_tags = frozen_100k_tags
unframed_episodes = 0
audit_tick_coverage_rate = 1.0
```

原有训练门保持独立：full-success rate 默认至少 50%，ability-positive coverage 仍须通过。RPC、source-integrity、seed-search、固定 seed 重放不一致或首帧前失败均不得伪造 Prefix；这些情况会令 one-click 停止并保留诊断现场。
