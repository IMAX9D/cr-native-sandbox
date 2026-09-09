# timing 分支容量对照

```bash
conda activate r2dreamer
cd /home/lenovo/gh/cr-native-sandbox
python experiment_hokoff_timing_capacity.py
```

默认从正式固定周期模型的 `~/cr-data/runs/hokoff-fixed-p4/last.pt` 开始，在独立的 `~/cr-data/runs/hokoff-timing-capacity-时间戳/` 中运行两组各 5,000 步的短训及 AP 评估。

- `linear`：原有单层 timing 头，保留全部预训练参数。
- `residual`：原有 timing 分数加上 `Linear(256,256) → GELU → Linear(256,1)`。新增最后一层的权重和 bias 都为零，因而初始分数与原模型一致。新增 66,049 个参数（约 1.4%）。
- 两组都完整继承正式模型的 timing 权重。普通 `train_hokoff_fixed.py --init-from` 会重置 timing 头，本实验使用专门初始化入口，避免混淆容量和重新初始化的影响。
- 共享编码器、LSTM 和各动作头继续联合训练；不是冻结骨干的分类探针实验。delay 头保持冻结。
- 两组都重新创建 AdamW 优化器；不单独给对照组保留旧动量。学习率、权重衰减、梯度裁剪、batch、窗口和标签权重均继承源训练合同。
- 数据采样从实验 epoch 0 开始，使用同一 seed（默认 42），两组样本顺序一致。它不是从正式训练最后一个 batch 精确接着读。
- GPU 支持 BF16 时两组均使用 BF16，否则 FP32。禁用 FP16 对照，避免两组不同的溢出跳过批次改变成功更新所用样本。源正式训练使用 FP16，这次精度变动对两组一致。

开始前复制一份 source.pt，检查参数继承及实际观察窗口的初始输出逐元素一致。结束后检查成功更新数、样本游标、无 AMP 跳过、delay 权重未变，以及评估窗口编号和标签完全相同。

## 输出

- `experiment.json`：源 checkpoint 哈希、实验设置、参数增量、一致性检查、两组最终指标。
- `linear/last.pt`、`residual/last.pt`：各自训练到第 5,000 步的权重。
- `linear.console.log`、`residual.console.log`：结构化训练日志。
- 每组 `metrics.jsonl`：每 1,000 步的固定验证结果。
- 每组 `evaluation/`：最终 AP、阈值及同下牌预算的指标、可复查预测。

主要比较 AP、`action_budgets.actual_rate.precision`，同时检查选牌、位置和联合命中率是否退化。统一比较固定步数终点，不能分别挑选两组的 best.pt 后当作同预算结果。best.pt 仍按原训练引擎保留，供排查。

原生评估、梯度诊断和 `diagnose_hokoff_fixed.py` 已兼容容量组的新架构。游戏推理入口没有接入该架构。

## 调整规模与恢复单组

```bash
python experiment_hokoff_timing_capacity.py --steps 10000 --eval-every 2000
python experiment_hokoff_timing_capacity.py --seed 43
```

每次生成新目录；`--output` 可以指定不存在的目录。若需要恢复单组，用 `python -m hokoff_model.train_capacity`，提供该组 config.json 中同样的参数及原实验 source.pt，再加 `--resume 该组/last.pt`。不得用普通固定周期启动器恢复容量组。

这是单个配对 seed 的短训实验，尚不足以证明普遍提升；较小差异需要按对局进行配对评估，必要时重复不同训练 seed。离线专家周期命中也不等于对局胜率。

## 验证

新增 4 项容量测试覆盖初始输出和预训练权重精确一致、残差仅改变 timing 输出且接收梯度、精确断点恢复、两组训练和评估的一致性。HoKoff 72 项及共享 policy 17 项测试通过，共 89 项。两组各 2 步的真实 GPU BF16 功能检查通过。

## 2026-09-08 配对短训结果

从正式 534,691 步模型各训练 5,000 步，原结构 AP=6.152%，容量组 AP=6.257%；同专家下牌频率的 precision 分别为 10.747%、10.704%。位置准确率分别为 14.327%、14.279%，未见一致收益。

在相同 1,595 局、3,200 个验证窗口上做 500 次对局配对 bootstrap，容量组减原结构的 AP 差异为 +0.105 个百分点，95% 区间 [-0.162,+0.378]；同频率 precision 差异为 -0.042 个百分点，区间 [-0.818,+0.922]。区间只覆盖验证对局抽样，不覆盖训练 seed 波动。

本次结果不支持直接采用该容量组，保留原正式基线。不能据此排除所有容量方案；本次只测试一个隐藏维度和一个配对 seed 的短训。完整报告位于 `~/cr-data/runs/hokoff-timing-capacity-20260908-094502-018906/ANALYSIS.zh-CN.md`。
