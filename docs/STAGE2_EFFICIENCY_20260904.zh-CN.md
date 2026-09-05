# Stage-2 训练效率修正与实测（2026-09-04）

本轮按完成 PPO 更新的有效对局数衡量效率。历史 `231,958 场/天` 是
192 场纯采集短测的折算值，不包含学习器、晋级评估或持续运行验证。

## 已修复的衔接问题

- 同一批次的多波分片可以进入 Stage-2：单分片不再被要求包含整个批次的场数。
- 在创建模型前按 collection 根目录去重 ledger，要求提交所有已登记分片，
  校验跨分片 episode 覆盖，整批只执行一次状态转换和 commit。
- 补充真实两波分片的 PPO 更新测试，覆盖缺片拒绝、连续两个策略版本更新。
- 旧更新的 `pre-update/checkpoint.pt` 纳入保留策略，避免硬链接使历史权重无限占盘。
- 主循环可直接以 Python 脚本启动，不再依赖临时 `PYTHONPATH`。
- 云端主入口改用 `19031 + slot`（可用 `--base-port` 覆盖），避开该机器的
  Linux 临时端口区间 `32768–60999`。旧的 390xx 段出现过冷启动 `EADDRINUSE`。

## 计算与数据路径

1. CPU 先拼装完整 minibatch，再传到 GPU；原来每个小 chunk 分别传输后再合并。
2. 对已验证、只读的 minibatch 使用有容量上限的 CPU/pinned-memory 缓存。
   before-audit、两次 PPO epoch、after-audit 共用一次拼装结果。
3. 在 CUDA 计算当前 minibatch 时，一个 CPU 线程准备下一个 minibatch。
4. 教师输出按完整序列窗口转到 CPU，避免逐 Tick、逐输出头的微小传输。
5. 卡牌与技能位置分布、熵和 KL 同时计算所有槽位；概率和梯度有逐槽参考对照测试。
6. 指标在设备端汇总后整体回传；保留 NaN、梯度和更新 Guard。
7. 首次尝试直接从当前权重更新，只有真正重试时恢复快照。
   恢复 optimizer 时复制状态，防止重试污染后续回滚所依赖的快照。
8. 保存流程复用同一份 CPU master 权重进行差异检查、哈希和 FP16 导出。

## 驻留学习器与准备流水线（含被替代的配置）

通过 `--persistent-learner --overlap-preparation` 启用：

- 未启用隔离时，Actor、冻结 BC 教师和优化器跨更新保留在进程中。
  此配置后续出现跨轮 OOM，已被下面的逐版本进程隔离配置替代；正式默认
  同时启用 `--isolate-updates`，仅在同一策略版本内驻留。
- 采集器原子发布一个完整分片后，学习器立即验证并准备该分片的 value、BC 输出和 GAE。
- 其他采集器仍可继续生成同一策略版本的数据。
- 只有整个批次 CLOSED 且所有分片完整时才进行 PPO 梯度更新。
- 同一策略版本的数据不会混入下一版本；未引入旧策略数据重放或 off-policy 修正。
- 每个更新记录真实场数、决策数、整个更新时间与完整更新场次/天折算值。

这项重叠针对教师/价值准备。PPO 参数更新仍保留严格版本屏障。

## 同数据 GPU A/B

设备：AutoDL F59 / RTX 5090 D 32GB。输入：相同 32 场原生完整对局，
8,940 个 learner 决策、153 个 recurrent chunks，两个 PPO epoch、chunk batch 8。
所有变体从相同 policy-19 checkpoint 开始，仅保存隔离试验产物。

| 变体 | PPO 与 Guard | 数据准备 | 初始化、准备、PPO、保存合计 | Guard |
| --- | ---: | ---: | ---: | --- |
| FP32，无输入缓存 | 28.98 秒 | 28.70 秒 | 75.89 秒 | 首次通过 |
| FP32，输入缓存 | 21.17 秒 | 同上 | 66.18 秒 | 首次通过 |
| BF16 + fused optimizer + 缓存 | 34.36 秒 | 同上 | 79.61 秒 | 首次通过，但更慢 |
| FP32，缓存 + 批量教师拷贝 + 槽位向量化 | 21.01 秒 | 19.76 秒 | 58.67 秒 | 首次通过 |

缓存样例：36 次构建、108 次命中，峰值约 1.06GB。
在 batch 8 下不采用 BF16/fused；后续 batch 32 验证成功后，BF16/fused 纳入高吞吐配置。
不同 CUDA 执行路径的权重 SHA 不完全相同，不能据此宣称位级一致；
CPU 缓存开关测试的更新参数逐值一致，GPU Guard 指标在相近范围内。

在相同 32 场上进一步使用 `chunk_batch_size=32`、`chunk_padding_multiple=80`：

| 精度 | PPO 与 Guard | 其中优化计算 | 完整学习器更新 | 峰值 allocated |
| --- | ---: | ---: | ---: | ---: |
| FP32 + cache | 12.70 秒 | 4.78 秒 | 49.48 秒 | 25.28 GiB |
| BF16 + fused + cache | 10.28 秒 | 2.87 秒 | 46.11 秒 | 18.91 GiB |
| FP16 + GradScaler + fused + cache | 10.18 秒 | 2.69 秒 | 46.27 秒 | 18.92 GiB |

三组均首次通过 Guard，没有删减有效样本。padding 行的 loss mask 为 false；
长短序列有效输出有对照测试。较大 minibatch 将两遍训练从 72 次 optimizer step
变为 10 次，改变了优化噪声和更新频率，不能把 minibatch 平均 loss 跨配置直接比较。
BF16 在 batch 8 时更慢、batch 32 时更快，因此精度选择必须连同 batch 实测。

上述仍是学习器 A/B，不能当成包含采集的完整训练吞吐。
96 Worker 的首个整链路更新（FP32、batch 8）完成 192 场、52,201 个决策，
耗时 300.35 秒，折算 55,231 场/天。该数字含采集、准备、PPO 和保存。
其第二次更新遇到显存不足：学习器仍保留约 22.3 GiB 的 allocator 缓存，
已增加跨更新清理无用梯度及 `empty_cache`。

12 个采集进程加驻留学习器的重叠方式也曾 OOM。当前候选改为 6 个采集进程
各带 16 个 Worker，准备窗口 128、准备 batch 2、CPU 输入缓存上限 8 GiB。
这使 Worker 并发保持 96，同时减少 GPU 模型副本。

### 最终连续运行验收

单纯调用 `empty_cache()` 后仍出现跨轮 OOM，因此最终采用 `--isolate-updates`：
每个策略版本在独立 CUDA 进程内完成准备与 PPO，进程退出后重建 MPS；
原生 Worker 继续复用。GPU 模型在同一版本内驻留，不再承诺跨版本驻留。

| 更新 | 完整对局 | 有效决策 | 完整耗时 | Guard |
| --- | ---: | ---: | ---: | --- |
| policy 19 → 20 | 192 | 51,484 | 252.43 秒 | 首次通过 |
| policy 20 → 21 | 192 | 50,842 | 251.77 秒 | 首次通过 |

共 384 场、102,326 个有效决策，流水线总用时 506.10 秒，折算约 6.56 万场/天。
这是有限批次测量，不是 24 小时耐久测试；20 万完整训练场/天尚未达成。
两轮均保存 FP32 master/optimizer/RNG checkpoint 与 FP16 Actor，验证结束自动关机。

最终 `throughput` 配置为：96 Worker、6 个 collector、2 波、12 Tick 决策窗口、
BF16 + fused AdamW、chunk batch 32、padding bucket 80、CPU 缓存上限 8 GiB、
准备窗口 128 × batch 2、重叠准备、按策略版本回收 GPU 进程。

12 Tick 是 600 ms 决策窗口，不是改变 libg 的 20 Hz 原生推进速度；
相较 4 Tick 配置它降低了决策分辨率。吞吐试验未证明防守反应和胜率不受影响。

一键续训入口：

```bash
python scripts/start_expert_selfplay_stage2.py --mode formal \
  --resume-run <已完整提交的Stage-2运行目录> --profile throughput --updates 100
```

该命令启动训练，不自动提交云端停机操作。需要结束即停机的有界任务，使用
`run_stage2_efficiency_trial.py --pipeline-updates <N> --isolate-updates --shutdown-on-finish`
并显式提供模型、Runtime、并发、精度与总时限；其日志会逐阶段输出，失败也保留现场。

## 运行与成本控制

`scripts/run_stage2_efficiency_trial.py` 提供有总时限的隔离试验；
`--shutdown-on-finish` 在保存诊断后调用当前 AutoDL 安装的关机脚本。
检测到非空 Trash 时不使用会清空它的厂商脚本，记录原因供人工处理。
该脚本没有新增循环定时任务，也不会自动发布试验模型作为正式策略。

本轮尚未把 league 40/40/20 和 Elo 晋级接入 Stage-2 主循环；当前对手仍是冻结 BASE。
这与计算效率优化分开验收。

## 实验工具与安全边界

- `benchmark_stage2_collection.py`：纯采集计时，不等于完整训练吞吐。
- `benchmark_stage2_preprocessing.py`：教师/value/GAE 准备计时。
- `benchmark_stage2_learner.py`：同数据、隔离产物的学习器 A/B。
- `benchmark_expert_actor_compile.py`：编译推理实验，非默认优化承诺。
- `serve_expert_selfplay_policy.py` / `remote_policy.py`：单机 Unix socket
  集中推理实验，未作为本次验收的默认配置。协议使用 pickle 与公开默认 authkey，
  只可用于同一可信用户的私有目录；它不是多用户安全隔离或公网推理服务。
- 滚动补位、异步写分片、dense sampling 和 compile 默认关闭。

公开仓库仅保留源码、测试和脱敏汇总，不包含实测原始 rollout、模型、
TensorBoard 事件、SSH 密钥或云账号凭据。所有 checkpoint/rollout 只从可信来源加载。
