# HoKoff 风格的皇室战争 LSTM BC 基线

这是 CR 专用实现，参考 [HoKoff 1v1 OneBaseModel](https://github.com/tencent-ailab/hokoff/blob/9b35f7e5891ad98df45a36e3a18f5192a31e72f4/hok1v1/offline_train/networkmodel/pytorch/module/OneBaseModel.py)
的实体 MLP、池化、单层 LSTM 思路。没有复制其源文件或权重，不依赖 HoK 游戏内核，也不是 OpenAI Five 复现。
上游仓库为 Tencent AI Lab 的 HoKoff（Apache-2.0），论文为 HoKoff: Real Game Dataset from Honor of Kings and its Offline Reinforcement Learning Benchmarks。

## 结构与范围

- 卡牌/单位位置/阵营嵌入＋共享实体 MLP，按敌我分别最大池化；动态实体数量，不截断为 15。
- 保留手牌槽位顺序、卡组、已揭示敌方牌、下一张牌、16 维公开标量和轻量 8 通道网格 CNN。
- 融合为 512 维，经过单层 512 维 LSTM，再映射为 256 维决策特征。
- 沿用 timing、kind、card、ability、576 格条件落点输出与原 BC 损失。标签掩码/专家落点合法掩码不作为观测。
- 不使用 Transformer，也不单独读取历史事件；历史来自连续战场观测。现有 loader 为兼容仍准备事件字段，但模型忽略它们。
- 数据词表为 181/42 时默认 **4,732,133 参数**，其中 LSTM 为 2,101,248。与原 HoKoff 的 290 万参数不同。
- `--width` 是编码/动作头宽度；`--hidden-size` 是 LSTM 宽度。没有 heads/layers 参数。

每个随机窗口从零状态开始，先处理最多 `frame_window-1` 个历史帧作为无梯度预热，再对 `targets` 个目标帧反传。
默认是最多 127 帧历史（6.35 秒）＋32 帧目标（1.6 秒）；对局开始不足历史时使用实际帧数。
补齐帧通过 packed sequences 排除，不推进隐藏状态；从不跨对局、玩家或随机批次携带状态。
对局推理接口是 `Policy.forward_stream(batch, state, reset)`，状态顺序为 `(hidden, cell)`；新对局传 `state=None` 或相应 reset=True。
推理请配合 `model.eval()`、`torch.no_grad()`。连续推理可携带较长历史，但有限窗口 BC 并不保证学会全局长期记忆。
训练 `forward` 仅保证目标区输出有意义，预热/补齐区输出不用于损失或评估。

数据适配有边界：全量 manifest 约 97.1% 行来自有效截断前缀，状态未与真实客户端逐帧锚定，技能覆盖不完整。
本版用于局部 BC 架构比较，未证明全局胜率，不把截断序列伪装为终局。等待/行动权重、采样与决策频率暂不改变。

## AutoDL 一条命令短训

在仓库根目录运行：

```bash
python train_hokoff.py
```

已预设当前服务器的数据/cache 路径、训练 split=validation、留出 split=train、FP16、
编码宽度 256、LSTM 512、batch 32、workers 8。累计成功更新 1000 步后验证并保存。
每次从零开始，检查点放入 `/root/autodl-tmp/runs/hokoff-lstm-check-时间戳/`，启动时打印实际路径。
无需重新安装。可用 `python train_hokoff.py --dry-run` 只看配置，
或 `python train_hokoff.py --workers 4` 覆盖单个参数。

## Linux 快速运行

在仓库根目录运行 `python -m ...`，无需安装 hokoff_model 或克隆参考仓库。
依赖与 policy_v1 相同：Python >=3.8、PyTorch >=2.0、NumPy >=1.22,<2。
已经能运行 cr-policy-train 的机器无需重新安装。全新环境先按 policy_v1 的独立安装说明准备依赖。

```bash
python -m hokoff_model.smoke
```

这是合成数据 CPU 短训，验证反传、验证集和检查点路径，不产出可用游戏策略。

已有完整数据和 policy-v1-cache 可直接复用；没有 cache 时按 policy_v1 README 用 cr-policy-prepare 生成。
下面保留历史数据的大训练 split=validation、小留出 split=train；test 不参与训练调参。
先做独立测速，不与长训同时运行：

```bash
python -m hokoff_model.benchmark \
  --data /root/autodl-tmp/expert-dataset/native-bc-v1 \
  --cache /root/autodl-tmp/policy-v1-cache \
  --split validation --device cuda --precision fp16 \
  --width 256 --hidden-size 512 --frame-window 128 --targets 32 \
  --batch-size 32 --workers 4 --warmup 10 --steps 100
```

该计时器同步 CUDA，结果用于瓶颈定位，不等于无扰动吞吐。它更新临时随机模型来测训练开销，不保存权重。
首次建议 workers=4；服务器此前的 SIGKILL 原因未确认，不能保证换模型会解决系统问题。

短训 1000 次成功更新，末尾评估 100 个留出批次，输出 last.pt/best.pt。确认测速和服务器稳定后运行：

```bash
python -m hokoff_model.train \
  --data /root/autodl-tmp/expert-dataset/native-bc-v1 \
  --cache /root/autodl-tmp/policy-v1-cache \
  --run-dir /root/autodl-tmp/runs/hokoff-lstm-check \
  --train-split validation --val-split train \
  --device cuda --precision fp16 \
  --width 256 --hidden-size 512 --frame-window 128 --targets 32 \
  --batch-size 32 --workers 4 --epochs 1 --max-steps 1000 \
  --log-every 100 --save-every 500 --eval-batches 100
```

只能恢复这个模型自己的检查点，不能加载原 Transformer 的权重。续训维持原数据/结构/训练契约参数，添加
`--resume /root/autodl-tmp/runs/hokoff-lstm-check/last.pt` 并提高 `--max-steps`；它是累计成功更新上限。
workers 可调整。共享训练引擎现在在索引层跳过旧批次，不会重新读取前几万批数据；仍需重建随机索引和核验缓存元数据。
不要重复使用已有 last.pt 的目录从零训练。

## 如何看效果

除原 loss/accuracy 外新增：

- `action_precision`：预测行动中有多少是实际行动。
- `action_recall`：实际行动中有多少被预测出来。
- `predicted_action_rate` / `actual_action_rate`：有效目标帧上的预测/真实行动比例。
- `action_tp` / `action_fp` / `action_fn`：对应原始计数，方便检查稀少样本。

这些是未按 sample_weight 加权的指标，仅在 timing_label_mask、loss_mask、frame_mask 都有效时计算。
阈值固定 logit>0（概率>0.5），不在此版本调阈值；分母为零时报告 0，应连同计数解读。
高 timing_accuracy 不能替代行动召回。小批验证不能保证覆盖技能和后期局面。

## 验证

```bash
python -m unittest discover -s hokoff_model/tests -v
python -m unittest discover -s policy_v1/tests -v
```

本地用既有 Python3.11/PyTorch2.8 CPU 验证；PyTorch2.0.0/2.0.1 CPU 加入 CI，尚无本次远端结果。
CUDA/FP16 吞吐与稳定性需在目标服务器实测，本机没有可用 CUDA，不据 CPU 数据估算 GPU 提速。

本次验证结果：新增 7 项测试＋原有 16 项测试通过；合成 CPU 短训、独立测速（workers=0/2）、双进程 CPU/Gloo 训练评估与保存均通过。
真实留出分片只做默认 473 万参数模型的只读前向检查，所有输出有限，没有在 test 数据上更新权重或调参。

## 行动召回为零时：独立时机诊断

先运行以下命令，不继续训练、不修改阈值或损失：

```bash
python eval_hokoff.py
```

自动选择 `/root/autodl-tmp/runs/hokoff-lstm-check*/last.pt` 中最近修改的检查点，启动时打印实际路径和 step。
从检查点的 val_split 随机抽取最多 200 批窗口（seed=123、batch=32、workers=4），保留自然等待比例。
它使用 FP32 前向，不创建优化器，不覆盖模型，结果保存为检查点目录下独立的 `timing-eval-时间戳.json`。

可指定模型：`python eval_hokoff.py --checkpoint /路径/last.pt`。
也可覆盖 `--data`、`--cache`、`--batches`、`--workers`；本地 CPU 使用 `--device cpu`。
不允许用训练 split 或 test split 做这项阈值诊断。

重点看：

- `action_probabilities` / `wait_probabilities`：两类帧的平均概率和分位数。
- `average_precision`：AP（阶梯 PR 曲线积分，正确合并并列分数）；越高表示越能将实际行动排在前面。
- `constant_score_ap_baseline`：全报同一分数时的 AP，等于真实行动比例。`ap_lift_over_prevalence` 是两者比值，不是胜率。
- `thresholds`：多个阈值对应的 precision、recall、误报率、预测行动比例以及 TP/FP/FN。
- `mean_predicted_probability` 与 `actual_action_rate`：帮助检查平均概率是否偏离实际频率。

指标不按 sample_weight 加权，与现有 accuracy 口径一致；没有预测行动时 precision=null，没有真实行动时 recall/AP=null。
仅概率高低有所区分并不代表可直接部署一个更低阈值；20Hz 连续执行还需验证重复出牌、动作合法性与对局效果。
阈值评估也不证明因果机制或全局策略能力。新脚本随机抽窗口，和此前验证首 100 批的样本不同，不能把指标变化全归因于模型。

## 下一轮诊断：先固定小样本，再比较行动权重

先执行：

```bash
python overfit_hokoff.py
```

默认从实际训练来源 `validation` 随机扫描窗口，固定选取 8 个含有效行动的窗口和 8 个纯等待窗口，
整批保存在内存中，反复更新 1000 次。每个窗口仍保留所有有效等待帧及原标签，动作标签不作为输入。
使用原 256/512 网络、FP32、行动正样本权重 32、原多头 BC，其余动作头损失不变。
前后记录同一训练批的 AP、行动召回与未加权 timing loss，最后输出 `phase: overfit_summary`。

这是带行动窗口富集的**记忆能力诊断**，其行动比例、AP 和准确率不能充当留出集表现。
权重 32 是固定对照值，并非已调好的最优值；小样本不收敛可能需要更多更新或排查输入/标签/优化过程，不能直接判定数据无用。
窗口来源和索引保存到 `selected-windows.json`，只有训练分片被使用；不读取 test。
`diagnostic.pt` 故意与正式训练检查点区分，不能用来替代通用 BC 模型或交给 eval_hokoff.py。

把最后一条 overfit_summary 发回分析，再决定是否进行第二项：

```bash
python compare_hokoff.py
```

第二项按顺序从零训练 baseline（权重 1）和 weighted（权重 32），各 2000 次成功更新。
两组均为 FP32、相同初始化种子/样本顺序、batch 32、workers 8、原始自然数据比例；只改变 timing 正例权重。
使用 FP32 是为了避免两组 FP16 溢出跳过不同样本，不能直接把本轮速度与旧 FP16 结果相比。
每组结束后自动用同样 seed=123 的随机留出窗口做时机评估，最后输出 `phase: comparison_summary`，无需另跑 eval_hokoff.py。
对比的是相同步数的 last.pt；不按不同加权目标的 best.pt 比较。

重点看留出集 AP/AUC，而非仅看 recall 或总 loss：提高正例权重本身就会抬高输出概率。
加权后的概率不能未经校准直接解释为每帧真实行动概率，也不自动改变部署阈值。
新日志 `timing_unweighted_loss` 保留相同口径的原 timing BCE；`timing_loss`/总 loss 在加权训练中是加权目标，跨权重不可直接比较。

两项都会新建 `/root/autodl-tmp/runs/hokoff-overfit-时间戳/` 或 `hokoff-weight-compare-时间戳/`，不覆盖旧模型。
支持覆盖 `--data`、`--cache`、`--run-dir`、`--device`、`--steps`、`--positive-weight` 等参数。
本机只用合成小样本验证实现；真实数据实验由服务器执行。

通用入口也支持 `python train_hokoff.py --timing-positive-weight 32`。
默认权重仍为 1，原 Transformer 训练目标不变；不同权重写入检查点契约，续训时不允许静默切换目标。

## 权重对照仍弱时：可行动分层与时间容差

```bash
python diagnose_hokoff.py
```

自动读取最近一个已完成权重对照目录的 baseline/last.pt 和 weighted/last.pt；也可传
`--comparison-dir /root/autodl-tmp/runs/hokoff-weight-compare-时间戳`。
两模型必须具有相同模型/训练契约和步数（除 timing 正例权重），不接受训练集或 test 作为诊断留出集。
可用 `--checkpoint /路径/last.pt` 只检查一个模型。

默认在相同留出集随机选 128 个玩家序列（seed=123），读取其全部窗口，沿用训练时的有限历史窗口前向，
拼接每个窗口的唯一目标帧。它不是改成无限历史 streaming 推理，不跨玩家、对局或无效 timing 区间匹配。
所有执行为 FP32/no_grad，无优化器，不更新检查点。默认 batch=32、workers=4，可覆盖。
结果保存在对照目录下 `timing-context-时间戳.json`；最后打印 `phase: timing_context_summary`。

### 可行动掩码的限制

编译器对某些动态卡牌只在卡牌监督时刻有精确合法掩码。因此全零 card/action 掩码不能证明“无费可下”。
本脚本仅报告 `mask_confirms_any_action`、`mask_confirms_card_play` 和 `no_action_confirmed_by_mask`，
不把最后一类强行标为 forced WAIT。掩码可受监督时刻覆盖影响，分层结果有选择偏差，不能据此直接过滤训练帧。
同时保留全量 AP/AUC，并按公开圣水比例分层；圣水分层也不等于已知每张牌的真实可负担性。

### 事件匹配口径

- `every_frame`：每个超过阈值的有效帧都是一个触发，直接暴露反复触发问题。
- `rising_edge`：仅从不超过阈值变为超过阈值时触发；连续高分平台只算一次。
  在有效片段开始时若已超过阈值会触发一次。这是固定的因果诊断规则，不是已调好的游戏执行策略。
- 每个触发最多匹配一个专家动作，每个专家动作也最多匹配一个触发；剩余触发为 FP，剩余动作为 FN。
- 报告精确时刻、±5 tick/±10 tick（±0.25/±0.5 秒）、仅提前 5/10 tick。
  偏差定义为预测 tick 减专家 tick；正数表示晚于专家。匹配按时间贪心最大化匹配数量，不保证最小时间偏差。
- 不虚构序列边界外或 label-mask 无效区间的数据；边界上下文不足的动作数量单列为 boundary_limited_events。

阈值网格由固定阈值与预测分数分位数组成。摘要的 best_event_f1_on_validation 是该留出样本内选出的最佳 F1，
并不是独立测试成绩，也不能直接用作部署阈值。
额外的 shift_control_best_f1 把每个连续有效片段的概率循环错位（seed=9187，长片段至少错开 11 tick），
保留概率分布和大部分局部形状，在相同阈值网格中也取最佳 F1。很短的片段无法保证移出最大容差。
这只是单次错位对照，不是统计显著性检验；宽时间容差本身会增加偶然匹配，不能只看容差后的召回变高。

摘要发送最后一条 timing_context_summary 即可；各阈值完整结果、圣水分层、序列抽样位置在 JSON 文件中。

## 核对动作前/后状态与标签

```bash
python audit_alignment.py
```

默认选择最近权重对照的 baseline 模型，抽取同一留出集 64 个完整玩家序列，
检查有效出牌标签前后各 10 tick（0.5 秒）的圣水、手牌和模型概率。
可用 `--arm weighted` 看另一组，或 `--checkpoint /路径/last.pt` 指定模型。
它不训练、不改数据、不移动标签。

只统计完整有效邻域、且邻域内没有其他己方行动的出牌，排除数单列。
比例差值定义为 offset=k 状态减去 offset=k-1 状态；圣水比例下降超过 0.02 记为一次明显下降，
这是观测变化阈值，不是根据卡牌费用表验证了真实扣费。
报告 label 当刻与下一 tick 的扣费/换牌比例，以及最大圣水下降、最近换牌、最大概率上升的相对 tick 分布。
事件对齐概率曲线含每个 offset 的均值、中位数和相对 t-1 的变化。
完整 JSON 还保存最多 8 个逐帧例子，含手牌 token、圣水、己方单位数和当前标签手牌 token 对应单位数。
这项 token 身份来自标签手牌槽，不能独立验证原始源事件到底出了哪张牌；法术、部署延迟和其他单位死亡也会影响实体计数。

代码意图：TickTraceAccumulator 保留边界的动作前状态，跳过后续 trace 同刻的动作后 initial_frame，
编译器按 source tick + episode execution offset 写标签。
因此若实际数组在 label+1 才出现换牌和圣水下降，与这一设计相容；不能为了让指标变好把标签移动到动作后帧。
脚本会核对本机编译器/生成器文件是否匹配 manifest 中的哈希，但这两项哈希不能单独证明 trace 实现或真实客户端对齐。
若数据和代码证据仍矛盾，需要原始 native tick/action transcript 进一步确认，不能从模型相关性自动修复标签。

结果写入检查点目录的 `alignment-audit-时间戳.json`。请发送最后一条 `phase: alignment_summary`。
