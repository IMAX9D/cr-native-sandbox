# Policy V1：独立离线 BC

小型空间 Transformer + 近期帧 Transformer + 公开出牌/技能事件 Transformer。
**只训练已有数据，不需要 APK、libg、Android、原生运行时或游戏资源。**
本目录是独立 Python 包；不要为此执行主仓库的 `pip install .[training]`。
模型是研究基线，离线 loss 和烟雾测试不代表实际游戏胜率。

## Linux 快速开始

推荐在单独的虚拟环境运行。支持 Python 3.8+、PyTorch 2.0+。
你提供的 `CUDA 11.8 / Python 3.8 / torch 2.0.0` 镜像可作为基础。

```bash
git clone https://github.com/IMAX9D/cr-native-sandbox.git
cd cr-native-sandbox
# 如果 V1 尚未合并 main：git switch feature/policy-v1-offline-bc
python -m pip install 'numpy>=1.22,<2'
# 镜像已有可用的 PyTorch 2.x，可跳过下一行。
python -m pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install --no-deps -e ./policy_v1
cr-policy-smoke
```

`cr-policy-smoke` 自动生成很小的合成数据、准备事件索引，在 CPU 上训练三步并评估、保存 checkpoint。
这只验证安装与训练链路，输出目录会打印到终端。可指定一个新目录：

```bash
cr-policy-smoke --output /tmp/cr-policy-smoke-1
```

依赖只有 NumPy、PyTorch；安装构建使用 setuptools。已有 GPU 驱动需要支持所安装的 CUDA 版 PyTorch。
数据集不随 Git 仓库发布，需要自行挂载/复制已解压的 `native-bc-v1`。

## 使用现有 10 万场数据

要求目录直接包含 `manifest.json` 和 `shards/`，是 `native_state_v1` 的已编译数据，
不是原始 battles JSON、压缩包或 `sequence_only_v1` 数据。
不更改源数据和原划分；缓存必须在源目录之外。

```bash
cr-policy-prepare \
  --data /data/native-bc-v1 \
  --cache /data/policy-v1-cache \
  --verify-hashes
```

准备时读取双方配对序列，提取已有有效标签对应的公开事件，不运行模拟器。
`--verify-hashes` 会一次性读取全部使用的源数组校验 SHA-256；大数据集需要时间。
省略该参数仍检查结构和事件约束，但不声称做过全量内容完整性校验。
缓存记录源 manifest、shard 元数据和事件文件的指纹，训练启动时核对；源数组应保持不可变。
准备中断可重新执行同一命令；完整缓存已有 `index.json` 时拒绝覆盖，请复用它或换新目录。

**这份历史 10 万场数据的大训练分区名字是 `validation`，真实留出分区名字是 `train`。**
应按数据来源及训练历史选定用途，不能因名字再次交换或把已训练分区当最终测试集。
针对本次讨论的历史语料，单卡命令为：

```bash
cr-policy-train \
  --data /data/native-bc-v1 --cache /data/policy-v1-cache \
  --run-dir /data/runs/policy-v1-small \
  --train-split validation --val-split train \
  --device cuda --precision fp16 \
  --batch-size 4 --workers 2 --epochs 10
```

新生成的常规命名数据集使用默认 `--train-split train --val-split validation`。
训练不使用 `test` 分区。两分区相同或包含相同对局时拒绝训练。

先加 `--max-steps 20 --log-every 1 --eval-batches 2` 测显存和耗时，再启动正式 run。
`--max-steps` 是总更新次数上限，包含恢复前的步骤；`--epochs` 也是总 epoch 上限。
默认验证最多每 rank 100 个 batch，只是固定前缀的快速检查；完整留出评估用 `--eval-batches 0`。
训练输出各头 loss、计数和准确率；WAIT 占比高，不能把 timing accuracy 当模型水平。

## 双 RTX 3090

两张 3090 是每卡 24GB；DDP 每卡保存一份模型，不自动合成一张 48GB 卡。
先单卡通过，再使用：

```bash
torchrun --standalone --nproc_per_node=2 -m policy_v1.train \
  --data /data/native-bc-v1 --cache /data/policy-v1-cache \
  --run-dir /data/runs/policy-v1-ddp \
  --train-split validation --val-split train \
  --device cuda --precision fp16 --batch-size 4 --workers 2 --epochs 10
```

`batch-size` 是每卡窗口数；上例全局每步约 8 个窗口，每窗口最多监督 32 个 Tick。
使用 DDP 全局有效标签权重归一化，稀有技能在不同 rank 的样本数不同时不等权混算。
训练 sampler 末尾可能补齐至 world size 整数倍；验证不重复尾部样本。
双卡也可各跑一个独立架构实验。实际 CUDA/双 3090 吞吐与峰值显存须在目标服务器测量。

## 恢复和离线评估

`last.pt` 定期及每次评估后原子保存；`best.pt` 按当前留出评估的六项 loss 之和保存。
包含模型、优化器、混合精度 scaler、每 rank RNG、epoch、batch 游标及数据合同。
恢复保留原模型、split、batch、精度、world size、学习率等参数，可提高总 epoch/step 上限。

```bash
cr-policy-train \
  --data /data/native-bc-v1 --cache /data/policy-v1-cache \
  --run-dir /data/runs/policy-v1-small \
  --train-split validation --val-split train \
  --device cuda --precision fp16 --batch-size 4 --workers 2 --epochs 10 \
  --resume /data/runs/policy-v1-small/last.pt
```

离线评估在同一命令加 `--evaluate-only --eval-batches 0`，也可将 resume 改为 `best.pt`。
这里只计算留出 BC 指标；真实对战评估尚未接入。checkpoint 使用 PyTorch pickle，只加载可信来源。

## V1 的确切结构与数据边界

- 默认三路编码器各 2 层，宽度 128，4 个注意力头、FFN 宽度 512；各帧共享空间编码器。
- 实体 token 拼接身份、格子位置、敌我 embedding 和 3 项数值，再经非线性 MLP。
  不使用把独立身份/位置简单相加后直接平均的旧结构。
- 八通道战场网格通过小 CNN 编码，保留塔和未纳入卡牌实体表的占位信息。
- 每帧汇总成一个 token。帧时序使用因果局部注意力，默认每层最多 128 帧。
  数据窗口含最多 127 帧前置上下文 + 32 帧目标；多层堆叠可能间接读取整个最多 159 帧窗口，
  因此不能把每个预测都称为严格只看 6.4 秒。前置上下文不计算 BC loss。
- 每个目标帧查询最近最多 128 个严格早于该 Tick 的公开事件；事件自身也是因果编码。
  读取窗口开始前最近 128 个事件及窗口内的事件，携带绝对时间编码。
  这是有限事件记忆，不承诺覆盖每局完整历史。
- 事件来自已编译数据的有效执行标签。它不是额外采集的全量事件日志，不补造被截断/未标注的事件。
  敌方事件只导出已执行的 token、时刻、坐标和类别，不导出私有手牌/圣水。
  与当前决策同 Tick 的双方动作都不进入该决策的历史。
- 输出 Tick 行动概率、普通牌/技能类型、条件槽位及 576 格落点。
  位置头依赖所选卡/技能 embedding 和融合后的场面表示。标签不进入编码器。
- Timing 保留逐 Tick WAIT 监督，当前实现要求 20Hz、每行一 Tick 暴露。
  不支持直接把采样频率改低而仍沿用同一动作概率；不使用固定出牌倍率。
- 各动作头按已有合法掩码与有效标签计算条件交叉熵，保留 sample_weight。
  技能位置缺标签时不训练该位置头；截断末尾按原标签掩码处理，不伪造终局/胜负。
- 未补齐细坐标、稳定实体 ID、攻击目标/相位、投射物生命周期；本版不训练世界模型或价值函数。
- 实体数按 batch 动态补齐，没有静默截掉第 33/64 个单位。

可调 `--width`、`--layers`、`--heads`、`--frame-window`、`--event-window`、`--targets`。
先测 128 维版本，再考虑 256 维；参数量在启动时打印，不能只按参数量推断激活显存。

## 开发检查

```bash
python -m unittest discover -s policy_v1/tests -v
```

覆盖因果性、同时刻事件隔离、敌我坐标旋转、单位排列不变性、身份位置敏感性、
空场/空历史、不同长度批次、全部动作头反向、合法标签及缓存/分区隔离。

## 本次本地验证范围

- Python 3.8 + PyTorch 2.0.1 CPU、Python 3.11 + PyTorch 2.8.0 CPU：10 项测试。
- 独立 wheel 安装后，从仓库之外运行 `cr-policy-smoke`，不依赖父项目导入路径。
- 现有语料的一份真实分片：源数组校验、公开事件准备、默认约 190 万参数模型的下牌/落点反向传播。
- 双进程 CPU/Gloo DDP：训练、全量合成留出评估、检查点保存；单进程续训与连续更新权重一致。
- 尚未验证 CUDA 混合精度、双 3090/NCCL 吞吐、完整 10 万场训练或真实游戏胜率。
