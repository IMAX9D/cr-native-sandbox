# Expert-v1 第一轮专家行为克隆技术方案

## 结论

第一轮训练采用独立的 recurrent behaviour cloning（BC），不使用 PPO、Reward 或固定八卡分类。模型每个原生 20Hz Tick 接收真人当时可见的信息，并依次预测：

1. `timing_hazard`：现在是否行动；
2. `action_kind`：部署卡牌或发动技能；
3. `hand_slot`：从当前四张手牌中选择；
4. `position`：条件于所选卡牌的 18×32 落点；
5. `ability`：从当前公开可用技能候选中选择；
6. `ability_position`：仅对需要目标的技能训练。

生产入口为 `START_EXPERT_TRAINING_V1.cmd`。入口只有在编译数据通过原生复演验收并标记为 `production_ready` 后才启动，任何字段、Mask、分割或标签错误都会 fail-closed。离线脚手架验收入口为 `SMOKE_EXPERT_TRAINING_V1.cmd`。

## 对现有训练代码的审计

旧 `training/` 是固定 8 卡、`WAIT + 4张牌` 单一 categorical head，并带 privileged critic、塔血 Reward 和 PPO；不能直接用于全卡专家模型。

`selfplay_v2/` 已把时机改成 20Hz continuous-rate hazard：

\[
p(\mathrm{act})=1-e^{-\lambda\Delta t}
\]

这个数学语义可复用，但其卡牌/位置仍为固定四槽实现，且包含 PPO value/critic。Expert-v1 只继承 hazard timing 的时间建模，不继承 PPO、Reward、optimizer 或 privileged critic。

`expert_base_cycle_side_v1` 可精确恢复旧 schema-v1 的基础八卡循环、第五次出牌后的四手牌、下一张牌、Tick 和落点；它没有原生场面、形态、等级及丢失的技能事件。因此它只能进入 `sequence_only` 辅助预训练，不能冒充权威原生场景样本。

## Actor 输入与信息边界

Actor 允许输入：

- 当前公开场面张量；
- 自己完整卡组；
- 当前四张手牌及下一张牌；
- 自己圣水、公开塔血、时间和倍率；
- 已公开的敌方卡牌；
- 公开历史动作；
- 当前合法卡牌、落点和技能 Mask；
- 距上一个输入的 Tick 间隔。

严禁输入：

- 敌方当前手牌；
- 敌方精确圣水；
- 尚未公开的敌方卡组；
- 原生 RNG、隐藏 AI/行为状态；
- 任何为 Critic 准备的 privileged state。

训练 shard 使用字段白名单；出现 `enemy_hand`、`opponent_elixir`、`privileged` 等数组会直接终止。Actor 模型 API 本身也没有 privileged 参数。

## 可变全卡模型

卡牌不是固定类别输出。每个基础/进化/英雄形态在数据 manifest 中有版本化 token，四手牌、己方卡组、下一张牌和已公开敌方卡牌共享 embedding。Card Head 对当前四个 hand-slot embedding 打分，所以任意合法八卡组合都能使用同一个模型。

Position Head 使用当前场面 cell feature 与 `recurrent context + 所选卡牌 embedding` 做条件打分，不需要为每张卡维护独立输出层。Ability Head 同样对当前公开可用的技能候选 embedding 打分。

网络只做 Actor：CNN 场面编码 + 公开标量/卡牌编码 + LSTM + 六个条件 Head。LSTM 用 burn-in window 做截断反向传播，随机窗口不会从完全空白 hidden 直接训练中局标签。

## BC Loss

总损失为：

\[
L=L_t+0.5L_k+L_c+L_p+L_a+L_{ap}
\]

- `L_t`：continuous-time hazard 的事件/删失负对数似然；
- `L_k`：行动后部署/技能分类；
- `L_c`：仅部署 Tick 的四手牌条件交叉熵；
- `L_p`：仅部署 Tick 的合法格条件交叉熵；
- `L_a`：仅技能 Tick 的技能候选交叉熵；
- `L_ap`：仅需要目标的技能 Tick 使用。

WAIT Tick 不产生 Card、Position 或 Ability 梯度。被原生规则判定无任何合法行动的 Tick 也不训练 timing。若编译器对 WAIT 做无偏抽样，必须写入 `sample_weight`；若把连续删失区间合并，必须写入 `timing_exposure_ticks`。不能简单删除 WAIT 后仍按普通二分类训练。

## 数据格式

生产数据根目录：

`D:\AI_data\cr-native-core\expert-v1\compiled\native-bc-v1`

当前有效源清单固定为：

`D:\AI_data\cr-native-core\expert-v1\training-dataset\version-window-20260804\accepted-cycle-clean.jsonl`

截至本方案落地时该窗口为 73,436 场“双方循环均有效”的唯一记录；后续下载、清洗会更新它。编译数据 manifest 必须固化该清单的绝对路径和 SHA-256。一键入口会重新计算当前清单哈希；源清单一旦变化，旧编译数据会被判为 stale，必须增量重编译，不能误用早先的 `combined-current-20260826.jsonl`。

根目录包含 `manifest.json`，每个 shard 是普通 `.npy` 目录，支持 mmap 和多进程 DataLoader。主要数组：

- `grid [N,C,32,18] uint8`
- `public_scalars [N,P]`
- `own_deck_tokens [N,8]`
- `hand_tokens [N,4]`
- `next_card_token [N]`
- `revealed_enemy_tokens [N,8]`
- `ability_tokens [N,A]`
- `card_mask [N,4]`
- `action_kind_mask [N,2]`
- `ability_mask [N,A]`
- `selected_position_mask_packed [N,72]`
- 六个条件标签及各自 `*_label_mask`
- `sequence_offsets`、`delta_ticks`、`timing_exposure_ticks`、`sample_weight`

落点 Mask 以 576 bit（72 bytes）保存；WAIT Tick 不存四份重复的落点 Mask，只保存专家所选卡对应的 Mask，Position Loss 只在实际部署时读取它。

采集到技能 Tick 但技能身份仍未解析时，可以训练 `timing` 和 `action_kind=ability`，但必须令 `ability_label_mask=false`。只有 libg 在该 Tick 唯一解析出真实可用技能后，才允许训练 Ability Head；不能根据卡组静态猜技能身份。

## 数据等级

- `authoritative`：完整卡组形态、等级、技能事件、原生动作全部接受且终局一致；参与全部 Head。
- `native_generated_unanchored`：libg 按观察动作生成，但源回放没有 seed/build/state anchor；只能证明“原生生成”，不能声称等于真人当时场面。默认一键入口拒绝把它当主场景数据。
- `sequence_only`：循环与手牌可证明，但缺原生场景或技能；只能训练明确可证明的辅助标签。
- `ambiguous`：循环、形态或技能不唯一；不训练。
- `rejected`：非法动作、模式错误、终局不一致或状态串局；隔离。

每个 Head 都有独立 label mask，弱数据不会污染它无法证明的目标。正式主模型的场景输入和完整动作 Head 以 `authoritative` 为准。

## 数据分割

最低要求：同一 battle tag、源文件和镜像视角绝不能跨 split。生产拆分：

- Train：约 90%；
- Validation：约 5%，用于 early stopping 和超参选择；
- Test：约 5%，全程只在训练结束评估；
- 另保留 player-holdout test：所选玩家的所有对局从 Train/Validation 移除，用于检查是否只记住玩家习惯。

manifest 必须声明并由编译器证明 `battle_tag_disjoint=true`、`source_file_disjoint=true`。建议按游戏版本、卡牌费用、卡种、形态、塔兵和对局时长分层，避免小众卡只落入一个 split。

## 第一轮训练日程

不把“8轮”当成固定真理。默认最大 20 epoch，每个 epoch 评估 Validation；连续 3 个 epoch 没有超过 `1e-4` 的改善就 early-stop，并保存 `best.pt`。训练先做小规模 overfit/合法性验收，再跑全量：

1. 100 场 overfit：确认 loss 可显著下降、所有 expert label 都在 Mask 内；
2. 1,000 场 controlled run：确认无 NaN、吞吐和显存稳定；
3. 全量约 10 万场：最大 20 epoch、Validation early-stop；
4. 冻结 Test 后离线评估；
5. libg 闭环与 GUI 人机验收；
6. 冻结 `Expert-v1`，后续才进入 DAgger/自博弈/强化微调。

## 验收指标

离线至少记录：

- timing NLL、Brier、按圣水/时间阶段校准；
- Card Top-1、Top-3 和 opportunity-normalized 每卡统计；
- Position NLL、平均格距、1格内命中率；
- Ability/Action-kind Top-1；
- 按卡牌、费用、形态、塔兵和 split 分层；
- Expert label 原生非法率必须为 0；
- NaN/Inf、数据跨 split、信息泄漏必须为 0。

生产 manifest 的 `quality_gates` 必须明确证明 split collision、禁用特征、非有限特征、标签/Mask 冲突、原生动作拒绝和终局不一致全部为 0；缺字段和非零值都会阻止一键入口启动。

特别注意：动作 Tick、卡牌和落点真实，并不自动证明 libg 生成的中间场面是真实回放场面。缺少原生 seed、精确 build 或状态 anchor 时，必须记录 `native_generated_unanchored_rows` 和 `terminal_validation_unknown`，不得写成 0 或冒充 `authoritative`。若确实要做近似场景实验，只能显式传入 `--allow-unanchored-native-states`；该选择会固化到 Run manifest，默认双击入口不会悄悄放宽。

在线必须完整走 libg：模型只从当前四手牌选择，原生动作拒绝为 0，技能可用性正确，能正常终局，并与随机、规则基线和冻结专家候选进行固定种子对战。第一轮 BC 的目标是得到可解释、可复现的专家初始化，不以自博弈胜率替代数据正确性。

## 入口

离线 smoke：

```powershell
.\SMOKE_EXPERT_TRAINING_V1.cmd
```

生产训练（编译数据未达到准入标准时会拒绝启动）：

```powershell
.\START_EXPERT_TRAINING_V1.cmd
```

直接命令：

```powershell
D:\AI_data\runtime\venv\Scripts\python.exe -m expert_v1.training_v1.train `
  --dataset-root D:\AI_data\cr-native-core\expert-v1\compiled\native-bc-v1 `
  --output-root D:\AI_data\cr-native-core\expert-v1\runs
```

每个 Run 原子写入 `manifest.json`、`events.jsonl`、`checkpoints/latest.pt`、`checkpoints/best.pt` 和 `result.json`；checkpoint 包含模型、optimizer、全部 RNG 状态及数据 manifest SHA-256。
