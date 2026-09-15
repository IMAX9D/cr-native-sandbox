# R0 elite-v3

当前默认模型是关系注意力、空间CNN与LSTM结合的循环策略—价值网络。默认6层384维、12头注意力，768维LSTM；既有验证配置为36,853,407参数。方法参考FirstLight，属于本地适配实现，不兼容其原版权重，也不直接加载旧177M专家权重。

## 先运行离线检查

在仓库根目录、Python 3.11或以上环境中执行：

```bash
python -m pip install -e ".[training,data-processing,test]"
python -m training_r0.smoke --profile tiny --steps 8 --rich
python -m pytest tests/test_training_r0.py tests/test_training_r0_rich.py tests/test_training_r0_elite.py -q
```

这些命令使用合成输入，不连接游戏、不下载数据、不启动正式训练。`tiny`仅缩小测试容量；默认网络配置仍是elite。诊断允许不完整采集，正式IL入口继续拒绝此开关。

## 输入、决策与训练

- 输入保留72项实体字段、塔、卡牌、分组、事件、关系及命令历史；生命周期分支使用6项已发生、已确认伤害统计。
- 原生定义图和卡牌语义提供版本绑定的机制元数据；未知状态显式标记，不能填零冒充已知。
- 原生20 Hz，每5 Tick决策一次；输出等待或最多两条带候选、条件落点和时间偏移的动作。
- 已实现严格循环IL、PPO/GAE内核和循环状态管理。模仿标签与动作合法性必须匹配；正式数据需通过独立准入证据校验。

## 文件导航

| 位置 | 作用 |
| --- | --- |
| [config.py](config.py)、[model.py](model.py) | 默认尺寸、融合网络、价值头和联合动作解码 |
| [observation.py](observation.py)、[feature_contract.py](feature_contract.py) | 公共视角、字段顺序和观察契约 |
| [components.py](components.py)、[elite_encoders.py](elite_encoders.py) | 实体/卡牌、关系、空间与其他分支编码 |
| [semantics.py](semantics.py)、[native_graph.py](native_graph.py) | 机制语义、定义图及其指纹 |
| [actions.py](actions.py)、[session.py](session.py) | 合法动作、命令计划和按局/阵营管理的记忆 |
| [il_dataset.py](il_dataset.py)、[il_learning.py](il_learning.py) | 连续样本与模仿学习更新 |
| [rollout.py](rollout.py)、[learning.py](learning.py)、[rewards.py](rewards.py) | 自博弈轨迹、循环PPO及奖励 |
| [replay_capture.py](replay_capture.py)、[replay_adapter.py](replay_adapter.py) | 回放采集协议、对齐和特征组装 |
| [lifecycle.py](lifecycle.py)、[native_projection.py](native_projection.py) | 实体世代、伤害归属与原生状态投影 |
| [data/](data/) | 版本绑定的特征清单及机制元数据；不含模型权重或对局数据 |
| [../data_pipeline/](../data_pipeline/) | 原始回放整理、原生试处理、编译、证书与训练入口 |

## 当前验证范围

v17已完成主样本整局语义闭环，补齐延迟生成效果的原生归属，并有真实梯度更新与保存恢复记录。正式数据准入仍有待独立验证的编译器断言，跨机制规模验证尚未完成；本次发布不是已经训练完成的新专家模型。

完整状态见[发布说明](../docs/R0_ELITE_RELEASE.zh-CN.md)和[v17记录](../docs/R0_ATTRIBUTION_V17_STATUS.zh-CN.md)。架构简介见[模型框架](../docs/CURRENT_MODEL_FRAMEWORK.zh-CN.md)。
