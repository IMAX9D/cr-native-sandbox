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
