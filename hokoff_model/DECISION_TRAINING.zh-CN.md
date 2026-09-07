# HoKoff 决策序列训练

当前修正入口：`python train_hokoff_independent.py`。详见 [独立观测采样训练](INDEPENDENT_TRAINING.zh-CN.md)。下面保留旧事件采样实验的说明，用于历史复现；该采样存在已确认的时间间隔标签捷径。

这一入口只改训练端：保留实体编码器、网格 CNN 和单层 LSTM，增加时间输入和等待时长分类。现有 `train_hokoff.py`、原 `Policy`、导出与游戏推理路径保持原样。新检查点标记 `training_only=true`，不能交给旧推理代码部署。


## 一条命令启动

```bash
conda activate r2dreamer
cd /home/lenovo/gh/cr-native-sandbox
python train_hokoff_decisions.py
```

默认数据为 `~/cr-data/expert-dataset/native-bc-v1`，缓存为 `~/cr-data/hokoff-decision-cache-k8`，输出为 `~/cr-data/runs/hokoff-decisions-k8`。
缺少缓存时自动为 validation/train 准备全量决策索引，不解压数据。已有缓存直接复用。
CUDA 可用时默认 FP16、batch=32、workers=4；否则 CPU FP32、batch=8、workers=0。
已有 last.pt 时自动续训；每次默认最多新增 1000 次更新，仍受 epochs=10 的总轮数上限约束。

```bash
python train_hokoff_decisions.py --dry-run
python train_hokoff_decisions.py --steps 5000
python train_hokoff_decisions.py --max-steps 0 --epochs 10
```

`--steps` 指本次新增更新数；显式 `--max-steps` 是累计上限，0 表示只受 epochs 限制。
若已经完成 10 个 epoch，继续训练需要提高 `--epochs`。覆写训练参数的方式与模块入口相同；续训仍要求模型、batch、精度和损失合同一致。
开启新实验用新的 `--run-dir`。改变 K 时默认缓存和输出目录会相应改为 k4/k16 等后缀。

## 时间语义

一条训练记录表示：获取当前观测 → 当前操作一次或 NOOP → 等待 delay tick → 下次获取观测。默认最长等待 8 tick（400 ms），也允许 1～7 tick。出牌与技能事件全部保留；在空档中最多每 8 tick 增加一个决策点，不移动动作标签。

网络保留原 timing 头，预测“现在是否操作”。新 delay 头输出 WAIT、DEPLOY、ABILITY 三种当前动作类别下各 8 个时长概率，训练损失根据专家当前动作类别选择分支。网络计算所有分支，不将任何动作标签、delay 标签、标签 mask 或未来观测输入编码器。

`absolute_tick/6000` 和 `prev_elapsed_ticks/K` 经线性投影加到场景特征上。原 tick_fraction 在 6000 后饱和，准备缓存时先从连续原始序列恢复真实 tick。各有效片段首条 elapsed=0。

## 1. 准备决策缓存

在仓库根目录执行，使用已有解压的 native-bc-v1 目录。该命令不会自动解压 archives.zip，不改源数据，也不需要旧 policy-v1 事件缓存。

```bash
conda activate r2dreamer
python -m hokoff_model.decision_data \
  --data /root/autodl-tmp/expert-dataset/native-bc-v1 \
  --cache /root/autodl-tmp/hokoff-decision-cache-k8 \
  --max-delay 8 --splits validation train
```

`validation` 是这个历史数据集的大训练集合，`train` 是小留出集合，与既有启动器一致。test 默认不准备、不用于训练调参。这里沿用已有分组划分，不重新拆分玩家或对局。

准备过程以分片为单位读取标签、生成稀疏行索引，输出 source/valid/decision rows 和保留操作数量。缓存只存索引，不复制观测。默认检查元数据与时间结构；加 `--verify-hashes` 可完整核对所选分片全部使用的数组，额外增加顺序读取。

先检查少量分片可用：

```bash
python -m hokoff_model.decision_data \
  --data /root/autodl-tmp/expert-dataset/native-bc-v1 \
  --cache /root/autodl-tmp/hokoff-decision-pilot-k8 \
  --max-delay 8 --splits validation train \
  --max-shards-per-split 2 --verify-hashes
```

用新目录保存不同 K 或不同子集的缓存，不覆盖现有 index。缓存 manifest/hash 会进入检查点合同，避免以不同数据子集继续相同训练。

## 2. 训练

```bash
python -m hokoff_model.train_decisions \
  --data /root/autodl-tmp/expert-dataset/native-bc-v1 \
  --cache /root/autodl-tmp/hokoff-decision-cache-k8 \
  --run-dir /root/autodl-tmp/runs/hokoff-decisions-k8 \
  --train-split validation --val-split train \
  --device cuda --precision fp16 \
  --width 256 --hidden-size 512 \
  --max-delay 8 --frame-window 17 --targets 32 \
  --batch-size 32 --workers 4 \
  --epochs 10 --max-steps 1000 --eval-shuffle \
  --log-every 100 --save-every 500 --eval-batches 100
```

没有 GPU 时用 `--device cpu --precision fp32`，并适当减小 batch。示例路径需换成实际数据与缓存路径。本次 r2dreamer 环境中 CUDA 不可用，已测试的是 CPU FP32；未声称 CUDA 性能或完整训练收敛。

- `frame-window=17` 表示最多 16 个历史决策作预热；`targets=32` 表示 32 个决策步参与损失。它们不是下牌次数。
- 预热编码与 LSTM 都不记录梯度；补齐帧不进行场景编码或推进 LSTM。
- 每个窗口从当前有效片段内取预热历史，不跨玩家、对局或无效标签区间携带状态。
- 原 timing-positive-weight 和新增 delay-weight 默认均为 1。不要直接沿用旧的 32 倍行动权重，也不要把 delay 当作当前 timing BCE 的暴露时长。
- 新入口不接受 timing-horizon 参数，避免将“未来是否行动”的标签混入“现在是否行动”。
- 训练与留出集都使用专家构造的决策点。这是离线 BC 指标，不能验证模型自行决定观察时间后的闭环效果。

## 3. 续训和只评估

续训时使用上面的相同参数，增加：

```text
--resume /root/autodl-tmp/runs/hokoff-decisions-k8/last.pt --max-steps 2000
```

`max-steps` 是累计成功更新次数；`epochs` 必须足够覆盖目标更新数。只评估则同时加 `--resume ... --evaluate-only`。读取新检查点应使用新训练入口，原 `eval_hokoff.py` 面向旧架构。

改 K、缓存、模型宽度或损失合同需要新 run。旧 HoKoff 检查点不能直接作为新架构的 resume。

### 短等待加权实验

`--delay-short-weight` 默认 1，保留原目标并兼容已有检查点续训。设为 8 时，真实等待 1～7 tick 的有效标签权重乘以 8，8 tick 标签权重不变；按加权后的有效样本权重和归一化。它不改变数据顺序、模型结构、当前操作损失或原始准确率统计。截断/未知标签仍不参与等待损失。

改变权重应创建新实验；用 `--init-from` 只加载模型权重，优化器和步数从零开始。不能和 `--resume` 同用，输出目录必须为空或不存在。

```bash
python train_hokoff_decisions.py \
  --init-from ~/cr-data/runs/hokoff-decisions-k8/last.pt \
  --run ~/cr-data/runs/hokoff-decisions-k8-short8 \
  --delay-short-weight 8 --steps 1000
```

对照实验应从同一个固定检查点初始化，使用相同 seed/batch/lr 和验证窗口，只改变 `--delay-short-weight`，分别保存到新目录。不能直接比较不同权重下的总 `loss` 或 `delay_loss`；比较未加权指标和 `delay_unweighted_loss`。8 倍是实验起点，不保证足以消除多数类预测。

2026-09-07 从 35000 step 模型各训练 1000 步的短试验：权重 1/8/32/128 的验证短等待召回分别为 0%/0.23%/4.88%/56.74%。128 倍的短等待精确率仅 7.60%，精确 tick 命中率为 8.46%；能改变多数类输出，但不能据此认为时机已学准或它是最优权重。默认仍为 1，保留旧训练的目标。详细结果保存在 `~/cr-data/runs/delay-reweight-35000-20260907/REPORT.zh-CN.md`。

之后继续该实验时去掉 `--init-from`，保留权重和输出目录，启动器自动读取其 last.pt：

```bash
python train_hokoff_decisions.py \
  --run ~/cr-data/runs/hokoff-decisions-k8-short8 \
  --delay-short-weight 8 --steps 1000
```

## 看哪些指标

已有 timing/kind/card/position 损失与行动 precision、recall 均保留。新增：

- `delay_loss`：有效精确时长标签上的交叉熵，按 sample_weight 与短等待类别权重的乘积归一化。
- `delay_unweighted_loss`：不加短等待类别权重的交叉熵，仍保留原 sample_weight，可用于跨权重比较。
- `delay_short_loss` / `delay_max_loss`：短等待/最大等待样本各自的平均交叉熵。
- `delay_predicted_short_rate` / `delay_actual_short_rate`：预测/真实短等待比例，结合 precision 判断是否过度提前观察。
- `delay_short_probability_on_short` / `delay_short_probability_on_max`：两组样本上预测的短等待总概率（1～K-1 概率之和），用于区分概率变化和 argmax 是否改变。
- `delay_predicted_mean_ticks`：预测等待均值；`delay_predicted_1_rate`～`delay_predicted_8_rate` 与 `delay_target_*_rate` 展示预测和标签分布。
- `delay_accuracy`、`delay_mae_ticks`：精确时长命中率及平均误差。
- `delay_always_max_accuracy`：始终预测最大等待的基线。
- `delay_short_recall`：真实 delay<K 的样本中，预测也小于 K 的比例；这不是精确时长命中率。
- `delay_short_accuracy`：短等待样本中的精确命中率。
- `delay_short_late_rate`、`delay_short_mae_ticks`：关键短等待样本的过晚比例和误差。
- `delay_censored_count`、`delay_unknown_mode_count`：关闭精确时长监督的数量。

没有相应样本时，比例暂记为 0，必须一起检查 count。样本约 96% 的 delay=8，所以单独观察总准确率会严重高估效果。delay 指标选择专家当前动作分支，不代表模型选错动作后的实际运行效果。

## 尾部与数据读取

原 timing_label_mask=0 的区域不会变成 WAIT 标签。每一段连续有效区间独立生成索引，最后一条的 delay_label_mask=false；当前有效动作标签仍保留。完整回放也采用这个保守尾部策略，不用未知边界制造精确等待、终局奖励或输赢标签。

加载器在读取/恢复网格、实体和落点掩码之前选择 source_row。只对选中行解码，不先把中间全部 tick 展开成 dense 张量。网格/实体 CSR offsets 与稀疏落点 mask 都按原始行索引取值。

## 本次验证

在 conda r2dreamer（PyTorch 2.8.0+cu128、NumPy 1.26.0、CPU FP32）完成：

- hokoff_model 37 项测试（包含新增 7 项）和 policy_v1 17 项测试通过。
- 本地双进程 CPU/Gloo 训练、验证、保存通过。
- 新测试检查事件保留、无效区间切断、6000 tick 后时钟、稀疏观测与原 reader 对齐、无未来标签泄漏、预热梯度隔离、未知 delay 零梯度、小批损失下降及精确续训。
- 两个真实分片共 252,396 行，249,326 个有效 tick，转换为 31,828 个决策，保留 1,403 次操作；只使用历史 validation 作训练、train 作留出。
- 原 256/512 宽度模型训练 100 次更新，在 20 个留出批次（4,833 个有效决策）上：行动 precision=77.27%、recall=7.56%；条件卡牌准确率 43.30%、落点格准确率 8.48%。
- delay accuracy=95.95%，等于始终预测 8 的基线；195 个短等待样本的召回为 0。这次短训验证了训练管线与可学习梯度，没有证明等待策略或对局能力有效。条件卡牌/落点准确率也不等于完整策略的执行成功率。

复核：

```bash
python -m unittest discover -s hokoff_model/tests -v
python -m unittest discover -s policy_v1/tests -v
```

## 决策模型分阶段测速

```bash
python -m hokoff_model.benchmark_decisions \
  --device cuda --precision fp16 --workers 4 --warmup 10 --steps 100
```

默认使用与启动器相同的 ~/cr-data 数据与 k8 缓存，batch=32、width=256、hidden=512、frame-window=17。
分别输出 data_wait、host_to_device、forward、loss、backward、optimizer 的每批毫秒数。
使用临时随机模型，不读取或保存训练检查点。应独立于正在运行的训练进行测量；该命令不会自动暂停其他进程。
这是逐阶段 CUDA 同步计时，不等于无扰动训练吞吐。比较 workers 时需控制数据缓存冷热，不能把缓存收益当成 worker 收益。

## 数据加载优化（2026-09-07）

无需重建决策缓存，可直接从原检查点续训。改动保持模型、batch、样本顺序、标签、损失和检查点合同一致：

- 以 NumPy 批量恢复所选决策行的实体和网格，减少逐帧的小 Tensor 分配。
- 每个 batch 的补齐张量只分配一次，直接填充，避免先逐窗口 padding 再 stack。
- 缓存每个已访问分片的 dtype/shape/offset/order，重开 memmap 时复用文件布局；只缓存小元数据，打开分片仍受原 LRU 上限限制。
- 决策读取时不再打开不使用的源 tick 元数据数组，并复用初始化时已检查的路径。

在当前 RTX 5090 D v2、FP16、batch=32、width=256、hidden=512 上，使用相同随机种子和窗口序列，预热 30 批后计时 200 批：

| 配置 | 每批耗时 | 每秒 batch |
|---|---:|---:|
| 原版 4 worker，两次对照 | 121～142 ms | 7.1～8.3 |
| 优化版 4 worker，两次对照 | 58～66 ms | 15.1～17.2 |
| 优化版 8 worker，多次测试 | 43～80 ms | 12.4～23.2 |
| 优化版 16 worker | 107 ms | 9.3 |

4 worker 的代码优化约提高 1.8～2.4 倍吞吐。8 worker 波动较大，16 worker 反而更慢，因此启动器默认仍是 4 worker。数据缓存/系统负载会影响结果；同步分阶段计时不包含正常训练的验证、保存成本，不能据此保证整轮训练等比例提速，也不承诺 GPU 利用率达到 100%。

已完成 58 项测试、40 个真实窗口共 215 个 batch 张量逐元素精确比较，以及“原代码训练 2 步后切换优化版续训”的 CPU 检查，最终模型张量与原代码连续训练完全一致。测速期间短暂停止的原训练均已恢复，当前运行的旧进程不会自动加载代码更新。

下次启动/续训生效：

```bash
python train_hokoff_decisions.py --max-steps 0 --epochs 10
```


## 无 UI 的观察调度评估

见 [离线等待调度评估](DELAY_SCHEDULE_EVAL.zh-CN.md)：从原始逐 tick 状态测试固定间隔、模型自主跳帧和相同观测预算对照，无需启动游戏 UI。
