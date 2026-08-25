# Expert Sequence Pretraining v1

## 定位

这是完整原生状态模型之前的、可证明数据边界内的专家预训练。输入为当前版本窗口的 `accepted-cycle-clean.jsonl`，以及 schema-1/2/3 八卡循环求解结果。它训练：

- 基于不等间隔公开事件历史的部署时机；
- 当前四手牌中的基础卡选择；
- 18×32 部署格；
- 对手已公开卡牌和重复出牌历史的循环记忆。

它不训练英雄技能，不生成或猜测 libg 场面，不声称拥有原生落点合法 Mask。完整原生状态/技能模型仍需后续 native replay 数据。

## 可证明字段

每个玩家侧在第五次部署（`first_exact_action_index=4`）开始，写入：

- 自己基础八卡；
- 动作前精确四手牌和下一张牌；
- 当前动作 Tick、基础卡及部署格；
- 此前公开的敌方卡牌集合；
- 上一个公开部署/技能事件的阵营、卡牌和位置；
- 公开事件间隔。

循环只能证明四张牌的集合，不能证明回放客户端的 UI 槽位。因此四手牌按全局 card token 确定性排序，训练和在线推理必须使用同一排序；Card Head 的 0..3 是该规范顺序，而不是猜测出来的 UI 槽位。

同 Tick 事件按 joint tick 处理：当前 Tick 的敌方动作不会泄漏给同 Tick 的己方标签。敌方手牌、敌方精确圣水、未公开卡组、native RNG 和隐藏状态均不进入 Actor。

schema-3 如果尚无 `valid-sides.jsonl`，编译器会从 accepted manifest 的 `source_path` 并发运行同一循环求解器；以后 schema-N 也走该路径。schema-3 的 `ability_plays` 会作为公开历史事件纳入时序，但本阶段不把未解析技能 ID 当技能监督。

## 时机目标

本数据不是 20Hz 全 Tick 场面，不能伪装成 categorical `WAIT`。事件间隔采用分段指数点过程：

\[
\mathcal L_{timing}=\lambda(s)\Delta t-y\log\lambda(s)
\]

其中己方部署 `y=1`，只发生敌方公开事件的间隔为右删失 `y=0`。旧 schema 丢失技能事件时，该对局仍可训练卡牌和落点，但 `timing_label_mask=false`；不会拿不完整时间线训练 Timing Head。

## 磁盘契约

默认输出：

```text
D:\AI_data\cr-native-core\expert-v1\compiled\sequence-only-bc-v1
  manifest.json
  manifest.sha256
  split-assignments.jsonl
  shards/
    train-00000/*.npy
    validation-00000/*.npy
    test-00000/*.npy
```

`.npy` 是可 mmap 的未压缩数组。`observation_mode=sequence_only_v1`、`state_provenance.mode=sequence_only`、`native_grid_rows=0` 和 `native_replay_validated=false` 是 fail-closed 契约。Sequence shard 中根本没有 `grid.npy`，而不是用全零网格冒充原生状态。

Manifest 固化：

- accepted manifest 和所有 cycle 输入 SHA-256；
- 每个 shard 文件 SHA-256；
- 数据内容聚合 SHA-256；
- manifest 自身 SHA-256；
- 编译覆盖、拒绝原因和各 split 行数；
- card vocabulary；
- player-holdout 审计摘要。

## Split

同一 battle 的两个玩家侧只能进入同一 split，battle tag 和 source file 不跨 split。测试/验证玩家由稳定哈希选出；只要任一方是 holdout 玩家，整场进入对应 holdout，因此选中的 holdout 玩家不会泄漏到 Train。

## 一键入口

最终 accepted manifest 达到至少 100,000 场后双击：

```text
START_EXPERT_SEQUENCE_PRETRAIN_V1.cmd
```

入口依次执行：

1. 检查 accepted battle 数量门槛；
2. 自动发现 schema-1/2/3 cycle 文件，缺失的 schema-N 本地派生；
3. 原子编译 mmap shards；
4. 验证无网格伪造、无信息泄漏、精确手牌标签、player holdout 和所有哈希；
5. 启动 recurrent behaviour cloning；
6. 保存 run manifest、events、optimizer/RNG、latest/best checkpoint 和 Test 结果。

开发 smoke：

```text
SMOKE_EXPERT_SEQUENCE_PRETRAIN_V1.cmd
```

它使用真实 accepted 数据的前 300 场，而不是合成人造动作。生产入口默认拒绝少于 10 万场的数据；不会悄悄以小样本开始正式训练。

## 后续衔接

Sequence-only checkpoint 可作为 card embedding、事件历史 encoder、LSTM、Timing/Card/Position Head 的专家初始化。加入 libg 场景后应显式迁移兼容参数，并新增 native spatial encoder；不能把 sequence-only checkpoint 宣称为完整游戏状态专家。
