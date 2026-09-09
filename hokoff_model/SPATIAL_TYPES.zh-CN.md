# 空间类别输入对照

固定周期模型新增 `--spatial-type-dim`，默认 0 保持旧模型，8 表示敌我各 8 个类别通道。
从已有 entity_tokens、entity_positions、entity_relations、entity_mask 生成特征，复用原数据和缓存。
不读取动作标签，不改变落点精度、固定周期、损失或实体最大池化分支。

每格每个阵营：`sum(该格有效单位的可学习类型 embedding) / 16`。
与原 8 通道网格拼接为 24 通道，不裁剪，不按单位数量取平均。
聚合在模型内逐帧执行，可反向传播；不将稠密类别网格写入数据集。
类型表独立于实体分支 embedding。以词表 181、维度 8 计，新增 1448 个 embedding 参数和
2304 个卷积参数，共 3752 个；不是额外增加数百万参数。实际速度需在目标 GPU 测量。

首次训练（新目录，从零开始；先保留旧宽度进行单因素对照）：

```bash
conda activate r2dreamer
cd /home/lenovo/gh/cr-native-sandbox
python train_hokoff_fixed.py --spatial-type-dim 8 \
  --run-dir ~/cr-data/runs/hokoff-fixed-p4-spatial8 \
  --width 256 --hidden-size 512 --batch-size 32 --workers 4 \
  --device cuda --precision fp16 --hours 2
```

相同命令会从该目录 last.pt 续训；重新开始必须换目录。旧权重不能直接续训为新增分支模型。
旧检查点不含 spatial_type_dim 时，禁用分支仍可加载/续训。live/evaluate/PPO 根据配置构造分支，
无需新观测接口。不要把新模型跑到旧默认运行目录。运行 --dry-run 可检查解析参数。

对照组使用同一命令与不同目录，设置 `--spatial-type-dim 0`。比较相同更新步数的留出
时机 AP、行动预算 precision/recall、选牌与落点指标，以及相同耗时的训练进度。
需要精确控制更新预算时，用 `--steps 10000` 替代 `--hours 2`；epoch 上限仍生效。
默认数据 split 历史命名相反：validation 用于训练，train 用于留出。

确认收益后另起新目录测试 `--width 128 --hidden-size 256`。不要在同一个对照中同时改变
类别输入、模型宽度与监督定义。CPU 测试覆盖格子求和、阵营分离、padding/空场、梯度、
因果性、流式一致性、训练续训及旧配置兼容；不代表实战胜率或 GPU 性能验证。
