# 固定周期模型：AP 与共享梯度诊断

在仓库根目录，用一条命令依次执行两项诊断：

```bash
conda activate r2dreamer
cd /home/lenovo/gh/cr-native-sandbox
python diagnose_hokoff_fixed.py
```

默认读取 `~/cr-data/runs/hokoff-fixed-p4/last.pt`，输出至同目录下带时间戳的 `diagnostics-*`。先复制一份 `model.pt`，两项诊断使用完全相同的 checkpoint，并记录 SHA256。原 checkpoint、优化器、源数据和训练缓存均不改写，不会启动或恢复训练。

默认配置：AP 使用与训练验证相同的固定 seed=123 验证窗口，100 batches；梯度诊断从实际训练集合均匀抽取 50 batches，seed=123，包含辅助动作窗口。batch size 继承 checkpoint（当前为 32），workers=2。CUDA 可用时自动使用 GPU。

其他用法：

```bash
# 评估按验证总 loss 保存的 best.pt；不同 checkpoint 使用相同的抽样规则。
python diagnose_hokoff_fixed.py --checkpoint ~/cr-data/runs/hokoff-fixed-p4/best.pt
# 单独执行一项。
python diagnose_hokoff_fixed.py --mode ap
python diagnose_hokoff_fixed.py --mode gradients
# 更大的梯度样本；也可以改为留出集诊断。
python diagnose_hokoff_fixed.py --mode gradients --gradient-batches 100 --gradient-split validation
# 小规模 CPU 功能检查，不用于判断最终效果。
python diagnose_hokoff_fixed.py --device cpu --workers 0 --cpu-threads 1 \
  --batch-size 2 --ap-batches 2 --gradient-batches 2
```

`--gradient-split` 是逻辑角色：training 对应本数据历史名 validation，validation 对应历史名 train。梯度默认从训练分布抽样，不能用只含主序列的验证集代表辅助样本带来的梯度影响。test split 不使用。

## AP 输出

- `ap/results.json`：`timing_ranking.average_precision`、正例比例参考值、AP/正例比例，以及多个阈值和下牌频率预算下的 precision/recall。
- `action_budgets.actual_rate`：把模型下牌数量限制在专家正例数量以内，再计算 precision/recall。用于控制“只是多预测下牌”的影响。
- 同分数样本作为一个整体，预算评估不根据标签或输入顺序拆分同分组。因此实际预测数量可能低于预算，报告会同时给出预算、实际数量和包含端点的阈值。
- `ap/pr_curve.npz`：所有不同分数阈值对应的 PR 曲线和预测频率。
- `ap/timing_predictions.npz`：完整分数、标签及窗口编号，方便离线复算。没有正例时 AP 为 null；缺少预测时 precision 记为 0。

计算 AP 与 PR 时不再给正例加权。模型输出是加权训练下的下牌分数，不应直接当作已校准概率。`metrics` 中的 loss 沿用原评估器的未加 timing 正例权重口径（weight=1），不能直接与训练日志的加权总 loss 比较。

这些仍是已知、合法对齐周期上的专家轨迹模仿指标。被屏蔽周期的统计保留在 `validation_label_audit`；没有执行模型出牌，不代表胜率。

## 梯度输出

`gradients/results.json` 按 `encoder`、`lstm`、`context` 和整体 `all_shared` 给出统计；参数名称列表也保存在报告中。关注 timing、card、position，但实际测量还包含 kind、ability、ability_position，避免忽略其他损失。

- `tasks.<任务>.norm`：各任务对该共享参数组的梯度 L2 大小。
- `tasks.<任务>.norm_fraction`：该任务梯度大小占各任务梯度大小之和的比例。这不是最终参数更新的贡献比例。
- `pairs.timing__position.mean` 等：同一 batch 内梯度余弦相似度的平均值。正值代表方向相近，负值代表有相反分量；同时报告中位数和 10%/90% 分位数。
- `negative_cosine_rate`：非零梯度有效比较中，余弦为负的比例。缺少标签或梯度为零时不参与余弦统计，返回 null，不能当作“没有冲突”。
- `cancellation_ratio`：梯度和的范数除以各梯度范数之和；越接近 1，抵消越少。
- `gradients/batches.jsonl`：每个 batch 的窗口编号、标签数、辅助动作数、任务 loss、梯度大小和余弦值，可定位异常样本。

梯度使用 checkpoint 记录的真实 timing 正例权重（当前为 32），复用训练损失的掩码和归一化方式。诊断使用 FP32，关闭 AMP、GradScaler 和 TF32；使用 `autograd.grad`，不写参数的 `.grad`，不执行 optimizer.step。

为支持 cuDNN LSTM 反向，模型处于 train 模式；当前架构无 dropout/BatchNorm，因此不会引入它们的随机性或状态更新。测量的是裁剪和 Adam 预处理前的原始任务梯度，不是实际 Adam 更新方向。不同 batch size 会改变梯度噪声；单进程结果也不等价于多 GPU 的梯度平均。

不要仅凭一次负余弦或较高的负余弦比例，就认定它导致了效果瓶颈。需结合梯度大小、多个 batch 的分布，并通过关闭某项 loss 等短训对照验证。

## 已完成验证

- 新增 7 项测试：已知排序的 AP、同分排序不作弊、预算不超额、无正例处理、已知梯度方向、分任务损失/梯度之和与原训练损失一致、辅助 timing 隔离、权重及 `.grad` 不变、统一入口输出和 checkpoint 快照一致性。
- HoKoff 68 项、共享 policy 17 项测试通过，共 85 项。
- 旧 2,000 步 checkpoint 的真实 GPU 回归：100 个验证 batch 的下牌分数、标签和窗口编号与原评估逐元素完全一致，AP 仍为 0.0298065438070978。
- GPU FP32 梯度诊断完成 3 个实际训练 batch。这里只验证代码可运行，不用这三个 batch 判断正式长训是否存在有害梯度冲突。
