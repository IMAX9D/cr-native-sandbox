# fixed4 PPO：固定 IL 参考与 KL 限制

已接入真实 Worker 的固定对手训练、PPO 更新、价值网络、IL-KL 约束、胜率晋级和显式续训。首版只训练动作头；编码器、embedding、context 和 LSTM 冻结，价值网络不能向 Actor 传梯度。

## 启动

```bash
cd /home/lenovo/gh/cr-native-sandbox
conda activate r2dreamer
python start_hokoff_workers.py --workers 12
python train_hokoff_ppo.py --workers 12
```

默认只跑 **1 个 iteration**，并行环境数为12；未指定 `--episodes-per-iteration` 时，每轮采集 `max(workers,2)` 局完整对战，再做2个PPO epoch。GPU可用时默认CUDA，CPU计算线程总计4，可显式 `--device cpu`。

新训练默认**双方每局独立随机卡组**，对应 `--learner-deck random --opponent-deck random`。仅学习策略更新，对手模型仍然冻结，晋级才替换。`hokoff_model/fixtures/hog_2_6.json` 保留为场地、塔、等级模板；某一侧使用 `--learner-deck fixed` / `--opponent-deck fixed` 时，才沿用模板中的2.6卡组。模板卡组为冰雪精灵、滚木、戈仑冰人、骷髅兵、火枪手、野猪骑士、加农炮、火球，总费21、均费2.625。

可用 `--replay /path/to/fixture.json` 显式选择其它配置。`--resume` 始终使用实验目录已保存的 `fixture.json`，不会自动替换旧实验卡组；换卡组请启动新实验，固定对手和胜率统计从新实验开始。无需重启原生Worker，新局reset会加载指定卡组。

延长新实验：

```bash
python train_hokoff_ppo.py --iterations 10
```

显式续训（`--iterations` 是包含已完成轮次的累计目标）：

```bash
python train_hokoff_ppo.py \
  --device cuda \
  --resume /path/to/your-fixed-opponent-run/last.pt \
  --iterations 10
```

续训保留原始 `il_source.pt`、固定对手参数、胜率窗口、晋级编号、固定检查样本、Actor/critic 优化器、KL 系数和随机采样状态，不把最新 PPO 模型变成新的 IL 参考。训练配置必须与首次启动一致；如果当初改过配置，续训也要提供相同配置。设备也保持一致。`--win-rate-window` 和满费惩罚系数未指定时自动继承。保留整个实验目录。

旧版“双方都用当前策略”的 PPO checkpoint 没有固定对手与胜率历史，新入口拒绝直接续训这些旧实验。请从原始 IL 新开本实验，避免在旧结果中混入不同的训练协议。

Ctrl+C 退出，不自动续跑。`last.pt` 只在完整 iteration 后原子保存；中断/失败时保留上一个完整 iteration，部分采集日志留作诊断。显式 resume 后从该 checkpoint 重置新对局、重新采集未完成的轮次。

## 策略与优化范围

复用现有 `expert_selfplay_v1.ppo.recurrent_ppo_loss`、`variable_time_gae` 和 `DefensiveTowerReward`。适配当前模型的动作分布、采集数据及 KL 检查，不调用旧 Expert 模型的 Poisson hazard 动作概率。

每4 tick，每侧最多一次操作：

- `WAIT`；
- `DEPLOY(card_slot, position)`；
- `ABILITY(entity_slot)`，没有位置。

按原模型 sigmoid 时机概率及各合法条件分布**随机采样**。所有合法叶子动作共 `1 + 4*576 + 16 = 2321` 个，可完整求和计算概率、熵和 KL；不是仅在已采样动作上估计 KL。没有任何合法操作时，WAIT 概率强制为1。

训练采样不使用 BC 对战入口的0.5/0.7阈值。IL 参考也采用同一随机分布定义。原 BC 时机分支曾使用正类加权，其 sigmoid 不保证是校准后的专家动作频率；不能把采样后的出牌变化直接归因于 PPO 学习。

只更新：`timing`、`kind`、`card_score`、`ability_score`、`position_head`。初始 Actor lr=`1e-5`，梯度裁剪0.5。独立 critic 是读取冻结 context 的小 MLP，lr=`3e-4`，梯度裁剪1。PPO clip=0.1，value loss 系数0.5，熵系数0.001。

每批采集期间不更新 Actor，学习侧按对局交替。仅学习侧轨迹进入 PPO；另一侧使用独立冻结的对手参数，不参与优化。对手不会随每轮更新改变，只有胜率晋级时才替换。

当前只训练动作头，双方编码器、embedding、context和LSTM与最初IL逐项相同，因此复用每侧各自的冻结循环特征，对手侧使用冻结对手动作头。这不增加一次完整模型推理；加载/晋级时校验主干一致性，若未来解冻主干则需要改成独立循环计算。

## 固定对手与 beat_x 晋级

第1个对手是初始IL基模（日志 `opponent_number: 1`, `opponent_name: base`），仅学习策略做PPO更新。

默认规则：

- `--win-rate-window 100`：统计与**当前固定对手**最近100局完整对局，每侧各50局。
- `win_rate = wins / (wins + losses + draws)`；平局计为未胜。
- 必须打满窗口，且胜率**严格大于0.95**，默认至少96胜/100局；95胜不会晋级。
- 在本轮采集结束时检查，默认每2局检查一次。晋级前检查对手参数摘要未改变。
- 晋级时复制刚完成采集的学习策略参数，保存为 `beat_1.pt`，表示超过了第1个对手；它成为第2个固定对手。
- 此后依次保存 `beat_2.pt`、`beat_3.pt`。对手编号递增，胜率窗口和当前对手累计局数清零，历史晋级记录保留。
- 学习策略继续训练；IL-KL参考**始终是最初IL**，不会随着对手晋级移动。

胜率是训练期间的滚动统计，窗口内允许包含不同学习策略版本；它不是最新单个checkpoint的独立评测胜率。晋级记录保留窗口内各局结果和策略step。晋级快照取本批采集时的参数，先冻结，再更新学习策略，避免把尚未参与本批对战的更新后参数当作晋级模型。

`last.pt`包含当前学习策略、冻结对手、优化器、胜率窗口、晋级历史和随机状态。仅在完整轮次提交后导出 `beat_x.pt` 和 `opponents.json`；如在保存checkpoint后、导出前中断，resume会从已提交状态重建缺失的最新快照。未完成轮次的对局不重复计入持久化胜率。

`beat_x.pt`是用于推理/评测的策略快照，带完整原始IL验证集配置与晋级证据；恢复训练使用同目录 `last.pt`。不允许把beat快照作为新的IL锚点启动PPO。

例如：

```bash
python train_hokoff_ppo.py --iterations 1000 --win-rate-window 100
python run_hokoff_fixed.py --checkpoint /path/to/run/beat_1.pt
```

训练对战双方均按各自策略分布随机采样；最后这条单模型对战入口是贪心推理，不能将其结果直接当作训练晋级胜率。当前没有历史对手池，只维护一个固定对手。

## 双方随机卡组

新实验默认两侧都是random：每局独立为学习侧和对手抽一套卡组。两侧可以有相同卡牌，不强制互斥或镜像。学习侧仍交替站在side0/side1，卡组随机按学习/对手角色独立定义。各侧可用random/fixed单独控制。

随机池是原生目录与固定IL词表交集中的标准1v1卡牌；带技能的牌还要求技能词表支持。当前121张，抽8张不重复的牌、最多1张Champion，全部基础形态，不随机进化或英雄形态。包含普通部队、建筑、法术，允许冠军使用自身技能；不额外保证卡组强度或费用结构。

排除MergeMaiden（28000025）：虽然该牌本体在IL词表，但它生成的26000104/26000105形态不在。真实续训短测已复现该问题，随机池显式检查已知生成形态依赖；不增加随机embedding、不丢弃未知单位。卡池和排除原因都保存于checkpoint。旧122张随机池的试验恢复时会提前拒绝，避免重复发生同样错误。

卡组随机使用各自独立的确定性种子 `hokoff-learner-deck-v1:seed:episode_id` 和 `hokoff-opponent-deck-v1:seed:episode_id`，不会消耗动作采样或PPO优化器的随机状态，也不会因改变另一侧的卡组模式而重新抽样。卡池和抽样规则写入run contract/checkpoint，resume直接恢复，不因本地卡牌目录更新而重新选池。新局开始时保持原回放每个槽位等级、塔与场地；保存每局实际回放到 `episode-replays/episode-NNNNNN.json`。

每局摘要保存 `learner_deck_ids` 与 `opponent_deck_ids`，日志显示 `learner_deck_mode` 与 `opponent_deck_mode`。每个并行对局有独立的encoder卡组缓存，避免随机卡组使共享缓存持续增长。

恢复双方固定卡组：

```bash
python train_hokoff_ppo.py --learner-deck fixed --opponent-deck fixed --workers 12 --iterations 1000
```

双方随机（新默认）：

```bash
python train_hokoff_ppo.py --learner-deck random --opponent-deck random --workers 12 --iterations 1000
```

已启动并仍就绪的原生环境可以直接复用，不需要每次训练前再次启动；重启机器或关闭服务后再运行 `start_hokoff_workers.py`。

双方随机模式下，胜率同时受策略与卡组匹配影响，不能与原来的固定2.6对局胜率直接比较。当前是在兼容卡池中随机拼8张牌，**不是从专家卡组列表均匀抽整套卡组**，不保证复现BC训练卡组分布，也不保证每套牌搭配合理。95%晋级规则仍不变。resume不会自动改变旧实验的卡组模式；旧checkpoint缺少某一侧的模式字段时，该侧视为fixed；显式更换模式会拒绝，应从IL新开实验。若要保持己方2.6、仅对手随机，使用 `--learner-deck fixed --opponent-deck random`。

## 多环境并行与批量推理

`--workers N`连接 `--port`（默认39031）起的N个连续端口；每个原生Worker独立推进一局，一个线程负责该局的RPC、观测编码、合法动作掩码和奖励，GPU在主线程对就绪观测集中推理。

- 一份学习策略、一份冻结对手。每次批量推理最多处理 `2*N` 个角色观测，不为每个环境启动一份训练器。
- 每局有独立LSTM状态、公开出牌追踪、原生掩码缓存和随机数生成器；新局全部清空，实体数量不同的观测按掩码补齐。
- 环境异步完成后，可在空闲位置开始本轮下一局；每次推理等待当前活跃环境准备好，属于同步批量推理。
- 每局策略采样与原生回放种子均由 `seed + episode_id` 初始化。策略随机数彼此独立，不因线程完成顺序而交叉消耗；CUDA浮点批量计算仍可能有微小舍入差异。
- 本轮全部对局完整结束后，按episode_id顺序统计胜率，再晋级/更新。Actor和对手在整批采集过程中不变，不混入更新期间的旧策略数据。
- 持久化统计只接受整批成功完成的对局；一个Worker报错会使本轮失败，不自动重启或重复提交动作。
- Ctrl+C退出训练；正在执行的原生调用按socket超时收尾，保留上一完整轮checkpoint。Worker进程继续保持服务。

`last.pt`保存并行数、每轮局数与采样方案。resume时不填 `--workers`/`--episodes-per-iteration` 会继承；显式改变会拒绝，以免混用训练协议。旧版单环境固定对手checkpoint仍按原串行方案续训，新建实验使用新的按局独立采样方案。

用户态启动辅助脚本复用已就绪端口，只为缺失端口从已安装的 `/opt/cr-native-bionic/worker0` 建立独立目录，默认放在 `~/cr-data/native-workers/hokoff-ppo/`。它读取 `~/gh/cr-native-linux-bionic` 的运行时启动实现，不重新安装Android根目录，也不启动模拟器。每个服务有独立cache和日志。

```bash
python start_hokoff_workers.py --workers 12
python train_hokoff_ppo.py --workers 12 --iterations 1000
```

结束辅助脚本自己启动的进程：

```bash
python start_hokoff_workers.py --workers 12 --stop
```

它不会关闭复用的、由其它入口启动的原始Worker。

日志新增 `collection_seconds`、`environment_wait_seconds`（并行推进/观测准备的墙钟等待）、`inference_seconds`（含传输、批量模型、采样与输出）、`update_seconds`、`collection_ticks_per_second`、`iteration_ticks_per_second`、`mean_inference_batch_games`、`peak_cuda_mb`。不要把GPU推理时间理解为纯CUDA kernel时间，也不要把并行等待时间累加当作CPU占用。

复现完整采集+PPO性能比较：

```bash
python start_hokoff_workers.py --workers 12
python benchmark_hokoff_parallel.py --workers 4 8 12 --episodes 12
```

各档使用相同IL、种子、12局、奖励、PPO配置，单独建立实验目录。每轮对局数必须不少于并行数；尾部剩余对局会降低平均推理批量，整体吞吐包含这个实际开销。

## IL 约束

**KL 限制的是动作概率分布的偏离，不是梯度向量方向或参数距离。**

参考模型为启动时完整复制的 IL checkpoint，此后冻结，源文件和参数均校验摘要。使用方向：

`D_KL(π_IL || π_current)`。

优化目标包含原 PPO objective，以及 `β × (joint IL-KL + conditional-action IL-KL)`。后者单独约束“已经决定操作后”的选类型、选牌、位置、技能分布，避免大量 WAIT 把动作分支偏移稀释。β初始1，按实际偏移调节，范围0.1–100。

每次候选 Actor optimizer step 后，检查**整批采集状态**和**第一轮固定保留的512个检查状态**。参考和当前模型采用同一状态和同一合法掩码。

| 检查 | 默认上限 |
|---|---:|
| 相对 IL 的平均 joint KL | 0.02 |
| 有合法操作状态上的平均 conditional-action KL | 0.03 |
| 相对本轮采样策略的平均 joint KL | 0.01 |
| 相对本轮采样策略的平均 conditional-action KL | 0.01 |
| 单个检查状态的 IL joint/conditional KL 最大值 | 0.2 |

固定检查状态同样使用0.02/0.03均值和0.2最大值限制。第一次固定检查集来自第一批采集，不是假称覆盖全部专家集。

超过任一限制时：

1. 撤回本次 Actor 参数和 Adam 动量/步数。
2. Actor 学习率减半（最低1e-7），β加倍（最高100）。
3. 停止当前批剩余 Actor 更新，下一批重新采集。

critic 无共享可训练参数，可以保留其独立更新。候选检查期间中断或数值异常，也先恢复 Actor 和优化器，再退出。

这些硬限制保证**被检查状态上的指标**不超限，不是对所有未来未见状态的数学保证。遇到新一批状态已超出 IL 限制时，程序报错停止，不悄悄换参考模型或放宽限制。暂未解冻主干；以后若解冻，必须重新计算当前策略和 IL 策略各自的循环状态，不能继续复用本版冻结 context。

## 采集、奖励与时间

- 只接受完整原生终局；截断、非法动作、缺失完整塔状态、推进异常都会拒绝该轮。
- 记录原生回执，费用、手牌、位置和技能实例均由在线接口验证。
- 只有一次动作提交；有限次空命令解决延迟终局，避免重复执行。
- 保存真实 `delta_ticks`，末尾不足4 tick也按实际时长计算。
- 零tick的延迟终局不伪造新 PPO 决策，终局奖励合入前一条。
- GAE 使用 `gamma_per_tick=0.99995`、`lambda_per_tick=0.995`，按实际 tick 数取幂。塔奖励按观察区间前后净变化计算。

基础奖励保持现有默认值：敌塔掉血×0.001，己塔掉血×−0.0012，拆塔+5/掉塔−5，胜+10/负−10/平0终局项。不新增出牌次数奖励。

新实验默认启用满费空等奖励：`--full-elixir-penalty-per-tick 0.001`。

- 每个实际推进 tick，以该 tick **开始时**的 `elixir_raw >= 100000` 判定满10费。
- 满费且这一 tick 没有成功下牌：扣0.001；未满费：不扣。
- 成功下牌只豁免执行的那个 tick。之后费用低于10，自然不再扣分；免费牌不能豁免整个4 tick区间。
- 技能不算下牌：若 tick 开始时满费，释放技能当 tick仍扣0.001；技能耗费后，后续未满费 tick不扣。这是“满费不下牌”的字面规则。
- 对双方分别计算，仅学习侧回报用于对应 PPO 样本；开局100 tick预热和零tick终局不计罚。
- 启用时原生采集改为逐tick推进，精确记录费用，但模型/LSTM仍然每4 tick决策一次。原生RPC通常从每决策1次增至4次；不增加模型推理或训练样本数。BC推理入口不变。

例如满费空等1000 tick扣1分；6000 tick全部满费空等才扣6分。真实对局按实际时长累计，不硬编码6000上限。该项旨在减少满费浪费，也会让模型不愿在满费时主动等机会；0.001是初始实验值，不代表已找到最优强度。

关闭该项做对照：`python train_hokoff_ppo.py --full-elixir-penalty-per-tick 0`。
固定对手实验续训未指定该参数时继承 checkpoint 中的系数，不自动改变旧奖励。显式修改已存在实验的系数会被拒绝，应从 IL 启动新的独立实验。

## 输出与检查指标

默认目录：`/home/lenovo/cr-data/runs/hokoff-fixed-ppo-时间戳/`。

- `il_source.pt`：固定原始 IL checkpoint。
- `last.pt`：可显式恢复的 PPO checkpoint，包含 Actor、critic、冻结对手、胜率窗口、晋级历史、两个优化器、固定检查状态、采样 RNG。
- `beat_x.pt`：超过第x个对手时导出的策略快照，只在真实达到晋级条件后生成。
- `opponents.json`：当前对手、滚动胜率以及历次晋级的对手编号、模型摘要和对局结果。
- `rollout-NNNN.pt`：冻结特征、全部合法掩码、采样动作、旧log_prob、奖励、真实tick间隔、GAE目标。
- `iteration-NNNN-*-collection.jsonl`：原生动作/回执与奖励。
- `iteration-NNNN-update.json`：每次候选更新的 KL、是否接受、回滚原因及最终约束结果。
- `summary.json`：本次进程执行摘要，完整历史看各 iteration 文件。

奖励日志新增学习侧 `full_elixir_idle_ticks`、`full_elixir_penalty_sum`、`base_reward_sum`、`learner_card_plays`、`first_card_tick`。先看满费空等是否减少，再联合基础战斗回报和胜率评估，不能只看出牌变多。关闭该项时不逐tick测量，汇总 `full_elixir_ticks_measured=false`，逐样本计数为-1，命令行显示 `not_measured`，不能将其读作0浪费。

采集 JSONL 保存双方逐tick费用 `elixir_before_ticks`、原生回执、基础奖励与独立扣分，便于复算。`rollout` 中保存 `base_reward`、`full_elixir_penalty`、`full_elixir_idle_ticks`；GAE 使用两项相加的总奖励。

胜率重点看 `opponent_number`、`opponent_name`、`opponents_beaten`、`win_rate`、`win_rate_games`、`wins`、`losses`、`draws`、`side0_games`、`side1_games`。窗口未满时也显示实时胜率，但不会提前晋级。

重点看 `il_kl`、`il_action_kl`、`il_action_kl_max`、`update_kl`、`update_action_kl`、`rejected_updates`，以及 `reference_parameters_unchanged`、`encoder_lstm_unchanged`。小KL和学习率保留有效位数，不显示成误导性的0.0000。

模型可直接用现有 BC 对战入口做贪心评估：

```bash
python run_hokoff_fixed.py --checkpoint /path/to/ppo-run/last.pt
```

现有 `python -m hokoff_model.evaluate_fixed` 也支持 PPO checkpoint；它从同目录、经过摘要验证的 `il_source.pt` 读取原 IL 的固定验证集配置。

## 2026-09-08 短训验证

结果：`/home/lenovo/cr-data/runs/hokoff-fixed-ppo-20260908-kl-smoke/`。

两轮通过显式 resume 串接，共4局、4867个学习侧决策、40次 Actor 更新；432次双方原生命令全部接受。原始 IL、参考参数、编码器与 LSTM 均保持不变。

第二轮最终：

- IL joint KL：`7.86238e-06`。
- IL conditional-action KL：`0.000144179`。
- 单状态最大 conditional-action KL：`0.000546804`。
- 相对本轮采样策略 joint KL：`1.56941e-06`。
- 固定检查集 IL conditional-action KL：`0.000178205`。

自然短训没有触发回滚；另用故意过大学习率的单元测试验证参数和已有 Adam 状态逐项恢复，也验证候选检查期间中断时恢复。

同样100批、3200个 IL 验证窗口，AP `0.06209996 → 0.06208516`，选牌准确率 `53.1176% → 53.0700%`，位置准确率 `15.0881% → 15.0405%`，后两项各少正确1个样本。该结果支持“短训尚未明显偏离 IL”，不代表已经提升胜率或适合直接长训。

## 满费惩罚短测

结果目录：`/home/lenovo/cr-data/runs/hokoff-fixed-ppo-20260908-full-elixir-smoke/`，详情见其中 `ANALYSIS.zh-CN.md`。

110项测试通过；原生定向测试满费等待4 tick扣0.004，成功下3费牌后该区间扣0。两局自然采集共2378个决策、20次更新，学习侧满费空等均为0，新增奖励没有在这批轨迹中触发；不能据此宣称已经改善策略。IL conditional-action KL为0.000136102，固定参考和主干未变。

逐tick采集耗时52.72秒，先前同轨迹首轮19.16秒；原生RPC增多，模型推理和训练样本数不变。

## 固定对手短测

`/home/lenovo/cr-data/runs/hokoff-fixed-ppo-20260908-opponent-smoke/ANALYSIS.zh-CN.md` 保存本次结果。118项测试通过，包括95%不晋级、96%晋级、beat编号/快照恢复和冻结对手参数检查。

真实Worker两轮通过显式resume串接，共4局、40次更新。窗口从2局正确恢复到4局，最终2胜2负、双方位置各2局、win_rate=0.5，没有晋级。冻结对手全部参数仍与初始IL完全相同，学习策略动作头发生更新；最终IL conditional-action KL=0.000201125。真实实验没有生成beat快照；晋级导出流程由合成胜率单元测试验证。

## 并行实测与默认值

本机相同12局、相同IL/奖励/PPO配置：4路整轮116.12秒（500.39tick/s）、8路79.35秒（732.27tick/s）、12路67.54秒（860.34tick/s）。三个档位的实际动作、tick和奖励轨迹一致；显存峰值均2106.64MiB。

因此新实验默认12路。不是全局最优结论，尚未测16/24路。每轮默认局数也变为12，不再是旧版2局；显式 `--episodes-per-iteration` 可增加采集批量，但不能少于workers。

122项测试通过。12路又做了一轮显式resume，胜率窗口正确接续到24局（13胜11负），对手和原始IL参考未变。续训轮吞吐878.25tick/s，显存峰值2492.37MiB。

详细对照、资源占用和启动/续训命令见 `/home/lenovo/cr-data/runs/hokoff-fixed-ppo-parallel-bench-w12/PARALLEL_ANALYSIS.zh-CN.md`。本次仅短训与压测，Worker保持就绪，训练进程已结束。

## 默认2.6卡组验证

`/home/lenovo/cr-data/runs/hokoff-fixed-ppo-hog26-smoke/ANALYSIS.zh-CN.md` 保存替换卡组后的测试结果。两路两局、434次双方出牌全部接受，24次PPO更新，原有KL/固定参考检查通过。四张替换卡另通过原生定向出牌检查，包含自然短训中尚未采样到的火球。当前模型偏向低费牌，不代表已经掌握经典速猪打法。

## 随机对手短测

修正后的结果位于 `/home/lenovo/cr-data/runs/hokoff-fixed-ppo-random-compatible-smoke/ANALYSIS.zh-CN.md`。127项测试通过；4路两轮、显式resume，共8局8套不同对手卡组，己方保持2.6。721条原生命令全部接受，62次更新、0回滚，固定对手/原始IL/主干未变。最终conditional-action KL=0.000196396，胜率0胜8负，仅说明功能跑通，不代表已提高强度。

## 双方随机短测

结果目录：`/home/lenovo/cr-data/runs/hokoff-fixed-ppo-both-random-smoke/`。129项测试通过；4路两轮（第二轮显式resume）共8局，双方各8套不同卡组，逐局回放核对通过。395条命令全部接受，48次更新、0次回滚，固定对手和IL参考/主干未变，最终conditional-action KL=6.76938e-5。胜率窗口正确接续为4胜4负，仅为功能验证。
