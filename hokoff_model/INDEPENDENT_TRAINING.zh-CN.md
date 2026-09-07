# 独立观测采样训练

旧的事件保留序列中，间隔小于 8 tick 几乎直接标记“当前发生专家操作”。本入口修复该采样捷径，继续复用现有模型结构、公共观测编码和 LSTM；不修改游戏推理端。

## 启动与续训

```bash
conda activate r2dreamer
cd /home/lenovo/gh/cr-native-sandbox
python train_hokoff_independent.py
```

默认读取 `~/cr-data/hokoff-independent-cache-k8`，保存到 `~/cr-data/runs/hokoff-independent-k8`。已有 last.pt 时默认增加最多 1000 个优化器更新；`--steps` 可调整增量。缺少缓存才会自动准备全量索引，不解压归档、不复制原始观测。

新入口默认实验参数为 `--sampling independent --delay-short-weight 128 --timing-positive-weight 8`。这是本轮验证使用的参数，不代表已找到最优权重。旧 `train_hokoff_decisions.py` 和旧缓存仍可用于历史复现；切换采样不能直接 resume 旧合同，需用 `--init-from` 加载模型权重并选择新的输出目录。

```bash
python train_hokoff_independent.py --dry-run
python train_hokoff_independent.py --steps 1000
# 从已有模型启动另一个实验；目录必须为空或不存在。
python train_hokoff_independent.py \
  --init-from /path/to/model.pt --run ~/cr-data/runs/independent-new-trial --steps 2000
```

## 采样与标签

- 主序列的观察间隔完全不读取动作标签：75% 的抽样选择最大间隔，其余 25% 在 1～K 中均匀抽取。K=8 时平均约 7.125 tick，短间隔也大量出现在 NOOP 帧。
- 主序列训练“当前是否操作”和等待时长；等待目标是 `min(距离下一次专家操作的 tick, K)`，排除当前操作，不使用随机抽到的下一次观察间隔作为答案。
- 当片段尾部的未来不可知时，屏蔽无法确定的等待标签；不跨无效区间、玩家或对局。
- 专家操作若未落在主序列上，则创建一个辅助样本，取该独立序列中最多 16 次之前的观察，加上原始操作帧。只有最后一帧的类别、卡牌、位置/技能标签参与损失；timing 和 delay 均不参与。辅助样本不是主序列下一次观察的锚点。
- 不平衡比例变化后，本轮将 timing 正类权重设为 8，以接近旧采样中正例的有效权重占比。指标仍按原始样本统计，不能把加权后的概率直接理解为已校准的真实事件概率。

原始输入和标签不改写。新的索引合同是 `independent_observations_with_action_aux_v2`。缓存额外保存 supervision 和 segment_roles，用于隔离辅助样本的损失；这些字段不输入观测编码器。

## 已完成的数据审计

完整训练集合包含 301774266 个有效原始 tick 和 1764267 次有效操作。新缓存保留 42439722 个主观测点，其中自然命中 247341 次操作；另有 1516926 个辅助操作样本。两者恰好覆盖全部有效操作。

主序列的动作率为 0.5828%；`0 < prev_elapsed_ticks < 8` 规则的精确率为 0.5875%，与基础动作率接近，不再是旧数据上的 100%。单元测试还直接验证了改变动作标签不会改变主序列的 rows/ticks/elapsed。

新缓存约 299 MiB，只存索引。辅助窗口带来额外训练工作，不能直接按旧 epoch 的步数估计：当前 batch=32 时约 91603 step/epoch。调参建议使用明确的短 `--steps`，先看真实调度指标。

## 验证边界

旧模型与新模型应在同一份独立验证采样上比较。对旧模型明确增加 `--allow-evaluation-cache-change`，评估器仍检查源 manifest 相同并使用 checkpoint 的验证划分。

```bash
python -m hokoff_model.evaluate_delay_schedule \
  --cache ~/cr-data/hokoff-independent-cache-k8 \
  --checkpoint ~/cr-data/runs/hokoff-independent-k8/last.pt \
  --output /tmp/my-independent-schedule-eval --window-batches 100 --battles 256
```

测试仍是无 UI 的原始状态序列调度，世界沿专家轨迹发展；不执行模型下牌，不代表胜率测试。

本轮 2000 步后，等预算随机对照的操作迟到率为 83.35%，模型为 81.16%；但均匀对照的平均迟到为 2.61 tick，优于模型的 2.89 tick。默认 0.5 阈值下动作召回仍为 0，动作排序 AP 从 0.689% 提升到 1.055%，仍然较弱。采样问题已修复，不能据此宣称整个游戏策略已经训练完成。

完整结果保存在 `~/cr-data/runs/hokoff-independent-k8/verification-2000/REPORT.zh-CN.md`。
