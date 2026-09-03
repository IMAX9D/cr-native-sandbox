# 123,740步低学习率恢复记录

## 状态

- 原运行：`expert-v1.1-cloud-177m-b32`，已按用户要求停止；最后进度记录147,437步，最后完整落盘139,207步。
- 恢复来源：123,740步（Epoch 2，batch 46,403）。
- 新运行：`expert-v1.1-r123740-lr1e-4-20260830`。
- 唯一训练超参数变化：学习率 `3e-4 → 1e-4`。
- 保留权重、AdamW动量/二阶矩、随机状态、数据排列位置和首轮最佳验证基线。
- 模型、Batch 32、BF16、序列128、burn-in 32、损失函数、梯度裁剪1.0等保持不变。
- 未创建或启用定时任务；用户要求在当前对话中监测。

## 永久备份

云机：`root@connect.weste.seetacloud.com:17461`，使用既有本地SSH密钥。

备份文件：

`/root/autodl-tmp/cr-expert-v1/frozen/expert-v1.1-step123740/checkpoint-123740.pt`

- 大小：2,125,199,333字节。
- SHA-256：`9a89986af3180b5b893b8aef294b3f0a70dade271b9e5b5c9251842a248f71a7`。
- 已校验并设为只读；不在滚动检查点目录，不会被三份轮转覆盖。
- 源运行清单与回执同目录保存。
- 异常日志和139,207步存档另存于 `frozen/expert-v1.1-instability-20260830`。
- 本地完整备份及回执：`D:\AI_data\cr-native-core\expert-v1\recovery\step123740-lr1e-4`。
  `checkpoint-123740.pt` 已下载完成，SHA-256与上述云端固定备份一致。

## 运行位置

- 新运行目录：`/root/autodl-tmp/cr-expert-v1/formal-runs/expert-v1.1-r123740-lr1e-4-20260830`
- 活动指针：`/root/autodl-tmp/cr-expert-v1/active-training-run.json`
- 进程ID以新目录的 `launch.json` 为准，不硬编码旧进程。
- 训练日志：新目录的 `train.log`；进度：`training-progress.json`；事件：`events.jsonl`。
- TensorBoard日志：`/root/tf-logs/expert-v1.1-r123740-lr1e-4-20260830`

## 新增观测（不改变训练计算）

- `train/loss`：真实批次窗口的算术均值，通常100批；断点恢复后的首个窗口可能不足100批，查看 `train/window_batches`。
- `train/loss_live`：每次发布时最后一个批次的Loss，保留旧口径。
- `train/loss_position_window_mean`、`train/loss_card_window_mean`：窗口分项均值。
- `train/gradient_norm_window_mean`、`train/gradient_norm_window_max`：裁剪前梯度范数。
- `train/loss_window_max`、`train/loss_window_gt10`、`train/loss_window_gt20`：窗口最大值和尖峰计数。
- `train/learning_rate`：优化器当前真实学习率。
- `train/position_logit_absmax`：发布时那个批次的原始落点logit绝对最大值，包含未监督位置；不能单独用它判定失败。

原`train/loss_live`不是100批均值，不能直接把新均值曲线更平滑解释为学习率优化成功。
应继续观察原约13万步之后的异常区间，并结合固定验证集判断。

## 迁移与验证

`expert_v1.training_v1.fork_run` 显式迁移学习率、optimizer identity和run signature，同时更新优化器的lr/initial_lr、scheduler的base_lrs/_last_lr。
不重置优化器moment或scheduler进度。

已执行：

- 原存档完整备份及哈希比对；
- 迁移后的逐张量权重、优化器状态及所有RNG核对；
- 新run严格续训契约检查；
- 13项迁移/原有续训测试；
- 新运行真实CUDA续训，进度已越过123,740，实际学习率为1e-4。

常驻缓存沿用68GiB请求上限与12GiB安全余量，实际驻留量由可用内存安全门确定。

## 初始观察

- 恢复后125,037步对应100批窗口，出现1个Loss>20的批次；窗口最大72.6688，均值6.1757，裁剪前梯度最大403.549。
- 此后观察窗口多数回落约5.5。不能据此宣称降学习率已经消除异常。
- 旧运行只记录每100批最后一个批次，新的窗口最大值覆盖全部批次，两者峰值不能直接同口径比较。
- 前台实时观察从126,837步开始，初始尖峰来自补查完整TensorBoard窗口计数，不应漏报。
