# 专家v1.2：FP32落点＋softcap20接入记录

## 当前阶段

**第2轮已完成并保存于154,674步；AutoDL控制台已确认实例“已关机”。**

2026-08-31 00:01:52 UTC完成轮末验证和保存，00:02:09 UTC保护备份及SHA-256检查通过，00:02:14 UTC发出关机命令。
随后刷新控制台，实例`f8we5up1vu-e41e71d9`显示“已关机”、操作变为“开机”；不是仅根据SSH断开推断。
本轮全量验证Loss **5.5608998471**（前次5.5824724578）；选牌Top-1 **50.2822%**、1格内命中率 **19.4599%**。
平均落点误差7.4809格，比前次7.4449格略高，并非所有指标改善。
模型、Optimizer/RNG、FP16导出和TensorBoard日志均在云端受保护目录；本轮模型权重尚未下载到本地。关机后的补充SCP连接已关闭，本地回执由关机前实时输出和控制台证据整理。
本地确认回执：`D:\AI_data\cr-native-core\expert-v1\rollouts\softcap20-20260830\epoch002-shutdown-confirmed.json`。
关机终止按量算力计费，付费扩容数据盘仍按平台规则单独计费；没有释放或删除实例。

### 本次关机安排与执行历史

2026-08-30 23:20 UTC前后，已安全保存146,381步并接入`--stop-after-epoch 2`后续跑。
目标边界154,674步：完成第2轮全量验证、latest/best及epoch-002完整存档和FP16导出后退出，不进入第3轮。
`finish_expert_epoch_shutdown.py`在云端等待该训练进程退出，核对准确轮数/步数、完整Optimizer/RNG状态、有限数和文件SHA-256，保护性复制存档及TensorBoard日志、同步磁盘后，才执行AutoDL官方`/usr/bin/shutdown`。
关机保护状态：Run目录`control/epoch-shutdown-status.json`；失败时写`failed_no_shutdown`并保留现场，不自动重启或强制关机。
23:49 UTC监测时确认平台`/usr/bin/shutdown`是无shebang脚本；已改用固定参数`/bin/bash /usr/bin/shutdown`调用，避免直接执行出现ENOEXEC。只替换关机保护进程，训练PID44301未改变；新增调用方式测试通过。
这是云端训练任务的完成后动作，不是Codex定时任务；最终已通过控制台核对关机结果。
保护性输出目录：`checkpoints/manual/completed-epoch-002-shutdown-154674/`。数据保存在云端数据盘，不释放实例；本轮权重尚未下载到本地。

本次控制修改的轮末暂停/恢复及存档门槛测试通过；只改运行时停止条件，原20轮配置签名、学习率、Batch和模型设置不变。

## 续训历史

2026-08-30 21:57 UTC按用户要求从128,740步继续训练。

用户已授权将短程试验候选接入训练/推理并继续验证。
新运行：`expert-v1.2-softcap20-r123740-20260830`。
从上一低学习率分支已保护的123,740步恢复，先训练5,000步，在128,740步完整保存并暂停，再做同一全量验证集复测。该阶段已结束，本次从带最新验证元数据的`checkpoints/latest.pt`续训。
只移除了临时`--stop-at-step 128740`限制；学习率1e-4、Batch32、主干BF16、落点FP32＋softcap20、原始数据顺序和其余训练参数不变，仍按20 epochs上限及原有早停规则运行。
128,740步的完整手动快照保留。原启动记录存于Run的`launch-history/`，当前PID及命令见`launch.json`；TensorBoard复用同一个Run和已有镜像进程。
未创建或启用定时任务。

## 5,000步结果

- 实际更新：123,740 → 128,740，共5,000步。
- 训练Loss均值：5.4909883139；最大单批Loss：7.1633377075。
- 裁剪前梯度均值：6.0314948933；最大值：30.1289081573。
- Loss>10和>20的批次数均为0；无NaN/Inf/OOM。
- 稳态更新速度约3.62步/秒（TensorBoard事件时间估计，不含最后保存停顿）。
- 同一全量验证集Loss：5.6015052191 → **5.5824724578**。
- 选牌Top-1：49.4216% → **50.0573%**。
- 平均落点误差：7.4775 → **7.4449格**。
- 1格内命中率：19.2372% → **19.1226%**，略有下降，并非所有指标改善。

两次验证均覆盖64,358序列、46,249个有效落点。128,740步已更新为此新打分策略下的best；验证只更新评估元数据，模型、Optimizer和RNG与暂停时的存档逐项核对一致。

这5,000步没有覆盖全部旧13万～14万步异常区间，也没有进行新的实战强度验收，不能据此宣布长期问题彻底解决。

完整暂停快照与当时的best、运行清单、FP16导出位于Run目录：

`checkpoints/manual/stop-at-128740/`

- `checkpoint-128740.pt` SHA-256：`c7cb3b28de0d743d03207b5884a8ae93e201009cff687b692ac193b4bd14a41c`。
- `weights-128740-fp16.pt` SHA-256：`afc95c7eb4df6548d9385de247719d0b00ac6d9b3e4054571bdb77c3802131bc`。
- 全量复测后，Run的`checkpoints/latest.pt`和`best.pt`含最新验证元数据；上述手动快照保留保存当时的best引用及对应best副本。

原始汇总已下载到：`D:\AI_data\cr-native-core\expert-v1\rollouts\softcap20-20260830\rollout-report.json`。

云端项目：`/root/autodl-tmp/cr-expert-v1/src/CR-Native-Core`。
云端Run根目录：`/root/autodl-tmp/cr-expert-v1/formal-runs/expert-v1.2-softcap20-r123740-20260830`。
进程ID和实际启动参数以Run的`launch.json`为准。

## 实际变更

- 显式配置`position_head_fp32=true`、`position_logit_softcap=20.0`。
- 落点分数使用FP32查询/格子投影/点积，关闭TF32；分数变换为`20*tanh((score-mean(score))/20)`。
- 默认旧配置不新增序列化字段，保留旧checkpoint/model-config/run-signature行为。
- 训练和验证显式请求只计算有监督的落点行；默认推理接口不读取标签，输出全部四个手牌的落点分布。
- 半精度权重下，落点投影仍通过FP32函数计算；本地推理加载器按存档配置采用相同打分方式。
- 参数量177,092,661、Batch32、主干BF16、LR1e-4、梯度裁剪1.0、20 epochs及原数据划分等保持不变。
- 运行时`--stop-at-step 128740`只控制保存/暂停，不改变学习算法签名。

迁移来源不是原始高学习率123,740步，而是：

`/root/autodl-tmp/cr-expert-v1/frozen/expert-v1.1-r108272-lr1e-4-20260830-step123740/checkpoint-123740.pt`

SHA-256：`0287c9f65f0a6f32d3a16e9df25bc34e7cadb9c947e22c61b21c3b405687b95f`。
权重、Optimizer动量/二阶矩、更新次数、数据位置、所有RNG逐项核对保留；只迁移明确的新策略配置与相应Run/Optimizer身份。

## 验证基线

旧best的分数不能直接套到改变打分后的同一checkpoint，因此新Run重新建立全量验证基线，没有伪称旧best已在新策略下验证。

- 验证集：64,358个序列，2,012个Batch，46,249个有效落点。
- 123,740步新策略基线Loss：**5.601505219069305**。
- 平均落点误差：7.477488200118955格。
- 1格内命中率：19.237173004875932%。
- 选牌Top-1：49.421609896066115%。
- 原始数据在`initial-validation.json`和`validation-step-123740.json`。
- `val/loss`只写完整验证结果，部分验证进度写到`validation-progress.json`。

## 接入验收

- 本地38项相关测试通过；云端20项模型/迁移/续跑测试通过。
- 实际CUDA检查中，集成代码与隔离试验的Loss、落点输出、选牌输出一致；全参数梯度相对L2误差约`1.0837e-8`。
- CUDA检查只覆盖代表性固定批次，不等于所有输入逐位等价证明。
- 本地RTX3080以相同177M结构、已有epoch-001 FP16权重并显式启用新头部作兼容性测试：50次测量平均4.1345ms/tick，峰值已分配显存1310.125MiB。
- 上述3080测试是结构/资源兼容检查，不是新训练权重的实战验收；原始回执位于`D:\AI_data\cr-native-core\expert-v1\rollouts\softcap20-20260830\local-inference-compatibility.json`。

## 新增安全保存控制

无需等待每1%的自动保存点，可使用：

```text
python scripts/request_expert_checkpoint.py --run-root <Run目录>
python scripts/request_expert_checkpoint.py --run-root <Run目录> --stop-after-save
```

请求在下一个优化器更新完成后的安全边界处理，默认保留完整checkpoint、当时的best、运行清单和FP16导出。
路径位于`checkpoints/manual/<request_id>/`，回执位于`control/checkpoint-response.json`。
请求ID、Run身份、字段和时点会校验；已处理请求不重复执行，错误请求不会改训练参数。
在验证阶段或启动阶段，请求需等待进入训练安全边界，不承诺随时从任意算子中强制取出内存状态。

续跑测试发现并修复了RandomSampler在完整遍历末尾额外消耗空尾部randperm的边界差异。
新checkpoint记录`epoch_sampler_needs_tail`，保证手动中断后跨epoch的数据顺序与不中断一致。
旧checkpoint缺少该字段时维持历史显式suffix恢复行为；本次来源就是已恢复分支，不改变当前epoch的历史样本顺序。

## 运行注意

- 续训须带`CR_EXPERT_TRUST_EXISTING_INTEGRITY=1`以复用已认证数据回执，避免重复读取全量训练集；仍检查manifest和文件覆盖。
- 缓存请求上限68GiB、安全余量12GiB，实际锁页量由内存保护决定。锁页量降到0不等于没有系统文件页缓存。
- 原始数据、旧存档、旧试验均未删除。
- 后续恢复同一Run时，应明确移除或提高`--stop-at-step`，否则已到目标会保持暂停。
- 正式/阶段验证工具：`python -m expert_v1.training_v1.fork_position_run evaluate --run-root <Run目录>`，要求训练进程已退出以取得同一互斥锁。
