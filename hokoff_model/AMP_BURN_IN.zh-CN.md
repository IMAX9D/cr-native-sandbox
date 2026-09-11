# 无梯度预热与 AMP 权重缓存

2026-09-11 的静态属性消融发现：785157 步 checkpoint 的 entity.0.weight 新增20列全部为零，
对应 Adam 动量也为零。实体编码器 optimizer step=1，而 cards.weight 和 LSTM step=785157。
全量静态属性及 known 标记置零后，同一批3200窗口的所有落点预测和时机 AP 完全不变。
因此这次15小时训练不能归功于静态属性；其他网络部分确实更新了，权重不需要作废。

根因：DecisionPolicy 在同一个外层 autocast 作用域里，先在 no_grad 下编码 burn-in，
再在有梯度下编码 target。AMP 可复用在 no_grad 下生成的低精度权重缓存，导致 target
编码器不带梯度。CPU BF16（关闭不支持的 oneDNN RNN backward）复现了此机制；当前环境
没有可用 CUDA，不能声称已实测 GPU FP16 修复。

修复：no_grad_burn_in 上下文只在预热时关闭 autocast 权重缓存，结束后恢复原开关。
保持原低精度计算和 target 缓存，不必把所有训练改成 FP32，也不改变模型/数据/优化器格式。
同时覆盖 recurrent burn-in 和原始 Policy 的 recurrent burn-in。

真实样本修复验证：开启 AMP 缓存后，entity、grid、scene、history_summary 和 LSTM 都恢复
非空梯度，与关闭缓存的参考一致。回归测试检查属性新列从零变为非零、梯度有限、异常时
正确恢复开关；CUDA FP16 版本在有 GPU 时运行。

现有 checkpoint 可继续使用，但不要马上重复15小时。先做短续训，检查编码器 step 增长、
combat新增列变为非零，然后重新评估。由于以前未充分训练的编码器开始更新，训练曲线可能变化。
之前仅 FP32 的迁移/短训测试不足以发现这个问题，已补上混合精度回归测试。
