# BC 工作交接（2026-09-11）

## 后续补充：独立人机对局盒子

新增 [BC Release对局盒子](MATCH_BOX.zh-CN.md)，在受控原生对局中补齐已确认出牌历史，支持1037042步纯权重。旧`FixedPolicy.forward_stream`保护不变，旧PPO和MuMu入口仍未接在线历史；下文保留原交接时的状态。Windows入口为仓库根目录`HOKOFF_MATCH_BOX.cmd`。

## 仓库与修改记录

工作仓库为 https://github.com/IMAX9D/cr-native-sandbox ，分支 `feature/policy-v1-offline-bc`。不要把 `cr-native-linux-bionic` 的 main 当作 BC 工作分支。

| 提交 | 修改 | 说明 |
|---|---|---|
| a09622c | 保存已有固定周期诊断、在线推理及 PPO 工作 | 历史基线 |
| de2b73e | 可选空间卡牌类型通道 | 按敌我/格子累加类型嵌入，保留同格数量贡献 |
| 498b965 | 双方公开出牌历史摘要 | 每方最近4次普通出牌；只取当前 tick 之前的信息 |
| 4050313 | 历史索引缓存和向量化查询 | 修复随机窗口反复构建整 shard 历史导致的训练减速 |
| 2cb0843 | 全分辨率空间落点直连 | 保留32×18网格，按上下文和卡牌逐格评分，加到原落点 logits |
| dec6b77 | 静态战斗属性与续训迁移 | 10项属性+10项已知标记；保留原权重、优化器、随机状态，新输入列零初始化 |
| 2d8e86b | AMP burn-in 缓存修复 | 预热时关闭 autocast 权重缓存，恢复目标帧编码器梯度 |
| 6647832 | 续训评估记录 | 见 TRAINING_PROGRESS.zh-CN.md |

具体设计见 [空间类型](SPATIAL_TYPES.zh-CN.md)、[历史](HISTORY.zh-CN.md)、[落点直连](POSITION_SKIP.zh-CN.md)、[战斗属性](COMBAT.zh-CN.md)、[AMP修复](AMP_BURN_IN.zh-CN.md)。

## 必须保留的判断

- 785157步模型的实体等编码器 Adam step 仅为1，静态属性新增列全零。静态属性关闭/开启的离线输出完全相同，不能把该次15小时训练收益归因于战斗属性。
- 原因是同一 autocast 作用域内，no_grad 预热生成的低精度权重缓存被目标帧复用，导致梯度丢失。FP32测试不足以发现，已经增加混合精度回归测试。
- CPU BF16复现及回归通过；当时CUDA测试因环境无GPU跳过。用户随后GPU短续训确认编码器step同步增长、属性列开始更新。后续整体离线指标持续改善，但没有隔离静态属性的独立贡献。
- 早期空间直连开/关离线消融显示该分支被模型利用；这种消融不是“同预算分别从头训练”的架构因果证明。
- 旧训练中空间类型、历史效果有限的结论受到编码器未正常训练影响，不应直接沿用。
- 当前实体仍独立MLP编码后按敌我最大池化；静态属性进入实体分支，未直接作为显式空间通道输入。攻击目标、当前攻击冷却等动态属性不在现有BC数据中。
- 内核具备部分动态观测字段不等于已有专家数据已保存它们；需要重建数据才能补齐，且必须核对回放与专家动作的一致性。
- 历史分支目前仅支持BC/离线评估，forward_stream明确拒绝带历史模型；在线部署和自博弈尚需接入历史，不能直接声称已可实战。

## 当前状态和远程产物

最新本地 last.pt 为1037042步，best.pt为1026000步。详细指标与判断见 [训练记录](TRAINING_PROGRESS.zh-CN.md)。进步仍在，但最近3小时收益已缩小；暂未实施新的Transformer或关系特征。

旧版模型为 [785157步权重](https://github.com/IMAX9D/cr-native-sandbox/releases/tag/bc-weights-step785157-20260911)，为AMP修复前快照，包含模型和配置，不含优化器。误发的786000步冒烟Release已删除。原合作者的Release和Latest未改动。

已另行发布 [1037042步最新权重](https://github.com/IMAX9D/cr-native-sandbox/releases/tag/bc-weights-step1037042-20260911)，为AMP修复后模型，仅含模型参数、配置及步数，不含优化器。

**修改记录、代码及最新纯权重已上传；训练日志和数据未上传。1037042步完整checkpoint、1026000步best、专家数据和缓存仍是本地文件；仅clone仓库或下载纯权重不能恢复完整优化器状态继续训练。**

旧机本地路径（非远程下载地址）：

- 数据：`~/cr-data/expert-dataset/native-bc-v1`（约96GB）
- 数据压缩包：`~/gh/cr-native-sandbox/archives.zip`（约3.5GB，未入Git）
- 缓存：`~/cr-data/hokoff-fixed-cache-p4`（约328MB）
- 训练目录：`~/cr-data/runs/hokoff-fixed-p4-history4-posskip16-combat`

## 新电脑接续

```bash
git clone --branch feature/policy-v1-offline-bc https://github.com/IMAX9D/cr-native-sandbox.git
cd cr-native-sandbox
```

旧机训练环境为Python 3.11、PyTorch 2.8.0+cu128；新机需安装与GPU兼容的环境，依赖及数据准备参照现有训练文档。若另行取得完整checkpoint、相同数据及缓存，将checkpoint放进上述训练目录的last.pt，再执行：

```bash
python train_hokoff_combat.py --hours 3 --dry-run
python train_hokoff_combat.py --hours 3
```

路径不同时可显式传 `--run-dir`、`--data`、`--cache`。启动器校验manifest和cache index哈希，不应通过删除校验来绕过数据不一致。仅Release的纯权重不满足此完整续训流程。

详细评估（checkpoint路径需实际存在）：

```bash
python diagnose_hokoff_fixed.py \
  --checkpoint ~/cr-data/runs/hokoff-fixed-p4-history4-posskip16-combat/last.pt \
  --mode ap --device auto --ap-batches 100 --batch-size 32
```

偏好：务实优先，不要过度工程化；训练指令用前台命令，不使用nohup。
