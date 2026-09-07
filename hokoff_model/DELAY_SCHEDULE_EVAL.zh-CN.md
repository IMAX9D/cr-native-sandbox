# 无 UI 的等待调度评估

`evaluate_delay_schedule` 直接读取已解压数据中的原始逐 tick 公共状态，不启动游戏、UI 或视频播放器，不写源数据，不修改游戏推理接口。

```bash
conda activate r2dreamer
cd /home/lenovo/gh/cr-native-sandbox
python -m hokoff_model.evaluate_delay_schedule \
  --checkpoint ~/cr-data/runs/delay-reweight-35000-20260907/schedule-eval-step2000/source-2000.pt \
  --output /tmp/my-delay-schedule-eval \
  --window-batches 100 --battles 256 --shuffle-trials 100
```

输出目录必须不存在。默认 GPU FP16；CPU 可用 `--device cpu --workers 0`。本版评估合同针对 K=8，读取 checkpoint 对应的验证划分并检查缓存摘要。

评估分两部分：

1. 相同验证窗口上，比较固定 4/6/7/8 tick、模型预测、打乱等待预测。根据原始操作时间轴判断等待期间是否跨过下一次操作；并不把所有“小于 delay_target”都当成时机错误。专家当前动作类别只用于独立诊断，主结果由模型预测当前类别。
2. 从随机抽取的对局有效片段起点开始，模型决定下一次读取的原始行号。跳过的帧不编码、不更新 LSTM。统计观测次数、专家操作到下一次安排观测的延迟；与固定间隔、相同预算的均匀间隔和随机打乱间隔比较。

默认 `--replay-history continuous` 连续累积隐藏状态。`--replay-history rolling` 是诊断：只保留最近 `frame_window` 次实际选中的观测重算 LSTM，同样不会读取中间帧。

每个有效片段最后 8 tick 的操作统一不评分，保证所有策略的下一次观测仍在已知原始数据内；无效区间和玩家边界不串联。双方视角按同一对局做 bootstrap。随机与均匀预算对照保持模型每个片段的首末观测及观测次数；随机对照还保持内部间隔的多重集合。它们是事后构造的对照，不是已部署的在线策略。

输出 `results.json`、`point_predictions.npz`、`state_schedules.json`。`elapsed_shortcut_audit` 检查简单规则 `0 < prev_elapsed_ticks < 8 => play_now` 在专家采样点和自主观测点的表现。它检测采样捷径，不证明网络只使用这个规则。

注意：世界状态仍来自专家轨迹，模型下牌不执行。因此这些结果是离线观察调度和时间覆盖测试，不能验证实际游戏胜率。

2026-09-07 的 256 局测试中，连续状态模型比固定 8 tick 多用约 4.86% 观测，操作迟到率几乎相同。还发现该 elapsed 规则在专家决策点上的精确率为 100%、召回率为 87.71%，但在模型实际访问状态上的精确率仅 0.92%。旧压缩采样和自主观察之间存在明显的分布差异，需要先处理这一问题。

结果与图表：`~/cr-data/runs/delay-reweight-35000-20260907/schedule-eval-step2000/REPORT.zh-CN.md`。
