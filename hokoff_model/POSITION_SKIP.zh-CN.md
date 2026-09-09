# 空间直连落点头（BC 对照）

`--spatial-skip-channels 16` 增加一条不下采样的 32×18 空间路径：

```text
当前网格 + x/y 坐标平面 → 两层 3×3 CNN → 每格 16 维空间特征
LSTM 上下文 + 每张手牌 embedding → MLP → 每张牌的 16 维查询向量
两者逐格点积 / sqrt(16) → 加到原有的 576 格落点 logits
```

原来的实体编码、全局 CNN/LSTM、等待/选牌头及技能位置头不改。新路径使用相同观测：
原 8 通道网格，若显式开启 spatial-type-dim 则追加类别通道；不提供新数据、不恢复格内坐标。
特征图每个观测只计算一次，四张牌共用，不展开四份空间 CNN。
BC 只对 loss_mask 和 frame_mask 都有效的帧计算新分支，跳过预热/补齐。
新的 query 末层零初始化，初始空间修正为零；首次更新后 CNN 开始收到修正分支梯度。
等待/选牌头结构和初始输出保持一致，但后续位置损失仍可通过共享上下文改变它们的训练结果。

词表 181/42、width 256、hidden 512、history 4、spatial-type-dim 0 时：
原历史模型 4,786,829 参数，启用 16 通道后 4,799,085 参数，新增 12,256。
参数增量小不代表计算免费：额外全分辨率卷积增加激活显存和计算，需要在实际 GPU 比较吞吐。

## 开训

从零训练，复用已有数据、fixed4 缓存和历史 sidecar：

```bash
conda activate r2dreamer
cd /home/lenovo/gh/cr-native-sandbox
python train_hokoff_fixed.py \
  --history-length 4 --spatial-type-dim 0 --spatial-skip-channels 16 \
  --run-dir ~/cr-data/runs/hokoff-fixed-p4-history4-posskip16 \
  --width 256 --hidden-size 512 --batch-size 32 --workers 4 \
  --device cuda --precision fp16 --hours 2
```

重复相同命令从该目录 last.pt 续训。已有历史模型不能直接 resume 为新架构，必须新运行目录。
这条命令不迁移已训练历史权重；`--init-from` 也不是跨结构迁移入口。
关闭新分支的旧 checkpoint 仍可正常加载/续训。历史仍只支持 BC 和离线评估，尚未接入在线对战。
若已有历史对照使用不同宽度、batch 或空间通道设置，请沿用其设置，只改变 skip 和运行目录。

对照组改为 `--spatial-skip-channels 0` 并换目录。为比较相同更新数，可以将 `--hours 2`
替换成 `--steps 10000`；同时记录实际耗时。不要用旧五轮模型和新两小时模型直接判断结构胜负。

## 落点评估

```bash
python diagnose_hokoff_fixed.py \
  --checkpoint ~/cr-data/runs/hokoff-fixed-p4-history4-posskip16/last.pt \
  --mode ap --device cuda
```

新生成诊断目录的 ap/results.json 中：

- conditional_position：使用专家卡牌及其合法位置掩码，报告 accuracy、top5_accuracy、mean_grid_distance。
- conditional_position_by_card_token：按卡牌 token 分组；token 名称可从数据 manifest 的 card_vocabulary 查询。
- 原有 timing AP、行动预算指标、选牌/落点准确率与联合准确率仍保留。

这些位置指标只统计有效主周期的下牌样本，不包含不参与时机监督的辅助动作。
距离是格子坐标的欧氏距离，不能当作战术正确性；离线指标不代表实战胜率。

测试验证初始基线等价、非零学习梯度、全分辨率局部响应、卡牌条件、未来/标签隔离、
预热梯度隔离、历史+类别+skip 组合、训练续训、旧检查点兼容及离线评估。
