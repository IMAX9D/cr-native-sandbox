# 固定 4 tick 动作基线

入口：仓库根目录 `train_hokoff_fixed.py`。只处理离线训练，游戏推理入口保持现状。

```bash
conda activate r2dreamer
cd /home/lenovo/gh/cr-native-sandbox
python train_hokoff_fixed.py --hours 15 --eval-every 2000 --save-every 1000
```

首次运行从零初始化；发现默认运行目录的 `last.pt` 时自动续训。`--hours 15` 表示本次训练约 15 小时后，在完成一个更新后验证、保存并退出，不会自行重启。最终验证和保存会多用少量时间。不指定 `--hours` 时，默认每次增加 1000 步。运行中的终端训练可用 Ctrl+C 中断，恢复点是最近一次已保存的 checkpoint。

- 数据：`~/cr-data/expert-dataset/native-bc-v1`，只读。
- 索引：`~/cr-data/hokoff-fixed-cache-p4`，缺失时自动生成，不再解压原始数据。
- 权重和日志：`~/cr-data/runs/hokoff-fixed-p4/{last.pt,best.pt,metrics.jsonl}`。
- 默认配置：周期 4 tick，LSTM hidden 512，32 个目标决策步，最多 16 步历史，batch 32，4 workers，FP16，学习率 3e-4。
- `--timing-positive-weight 32` 调整下牌正例的损失权重；没有把原始采样比例改成 50%。

在每个有效序列起点开始，固定每 4 tick 观察一次。完整周期 `[t,t+4)` 没有操作时监督 NOOP；恰好一次操作时，仅当该操作能在 t 合法执行，才把它对齐到 t（最多提前 3 tick）。手牌、当前可用性及王塔状态用于证明原生位置掩码可复用；数据编译器版本必须匹配已审核版本。提前释放技能、多个操作、尾部不完整或无法证明合法的周期不监督 timing，不能记成 NOOP。被排除的专家操作在训练 split 中保留为原时刻的条件动作辅助样本，其 timing 不参与训练。验证 split 只包含固定周期主序列，另报告排除数量。

当前观测始终取 t；未来操作只作监督标签和合法性审核，未来观测不进入模型。LSTM 的 16 步历史是 16 次周期观察，并非 16 次下牌。等待头停用，不计算 delay loss，不更新 delay 参数。

这里的源数据历史命名相反：`validation` 是实际训练集，`train` 是留出验证集。启动器已正确设置，test split 未使用。

主要关注 `action_precision`、`action_recall`、`predicted_action_rate`、`actual_action_rate`、`card_accuracy`、`position_accuracy`。后两个指标条件于专家动作标签；它们不能单独代表完整策略或胜率。`best.pt` 按总验证损失选择。

额外评估（对 checkpoint 所绑定的固定周期缓存）：

```bash
python -m hokoff_model.evaluate_fixed \
  --data ~/cr-data/expert-dataset/native-bc-v1 \
  --cache ~/cr-data/hokoff-fixed-cache-p4 \
  --checkpoint ~/cr-data/runs/hokoff-fixed-p4/last.pt \
  --output ~/cr-data/runs/hokoff-fixed-p4/evaluation-new
```

会输出 timing AP、阈值曲线、联合动作准确率、标签排除统计和可复查预测。输出目录须不存在。无需 UI；这是专家轨迹上的离线模仿评估，尚未验证模型实际出牌后的对局行为或胜率。

训练后的 AP 与共享梯度诊断可一起运行：`python diagnose_hokoff_fixed.py`。默认自动保存 checkpoint 快照并依次执行两项只读诊断，详见 [诊断说明](FIXED_DIAGNOSTICS.zh-CN.md)。

## 在线 BC 对战

已接通 fixed4 正式模型与 Linux Bionic Worker。仓库根目录运行 `python run_hokoff_fixed.py`。启动参数、验证结果及与 PPO 的边界见 [在线推理说明](LIVE_INFERENCE.zh-CN.md)。

## PPO 与 IL 保持

`python train_hokoff_ppo.py` 启动保守的 fixed4 PPO 短训，冻结 IL 参考与编码器/LSTM，并对 KL 超限更新回滚。默认1轮2局，详见 [PPO 训练说明](PPO_TRAINING.zh-CN.md)。
