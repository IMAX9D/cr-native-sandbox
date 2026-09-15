# R0 elite-v3 源码发布说明

发布范围：2026-09-14本地R0主线及v17相关修复。版本名为 **R0 elite-v3**，本地未定义另一个“R8 Elite”。

## 包含内容

1. `training_r0/` 完整输入、多分支编码、关系/空间/LSTM网络、动作解码、严格IL与PPO内核。
2. `data_pipeline/` 来源整理、原生处理契约、回放编译、证书准入和训练命令。
3. 原生采集桥接及相关telemetry头文件，包括延迟创建上下文、生命周期和伤害证据。
4. 对应离线回归测试、FirstLight的Apache-2.0许可与署名、字段及状态说明。

未包含模型权重、对局数据集、游戏APK/libg、浏览器会话、账号凭据、代理订阅或私有运行配置。其他MuMu策略切换及中间Expert V2实验不属于本次提交。

## 状态边界

| 项目 | 状态 |
| --- | --- |
| elite-v3模型和合成输入推理/反向 | 已实现，可离线检查 |
| v17主样本整局语义闭环 | 已有本地验证记录，12个域各1714/1714通过 |
| 真实窗口训练及保存恢复 | 已有局部验证记录；不是经过正式训练的专家权重 |
| 正式数据准入 | 尚未完成；需独立验证候选对齐、拒绝标签掩码、动作偏移、按局划分和分片回读等断言 |
| 跨机制规模验证与全量训练 | 尚未完成，不以单局或smoke成功代替 |

较早交接中“气球英雄归因未修复”的结论已被[v17记录](R0_ATTRIBUTION_V17_STATUS.zh-CN.md)更新。历史文档中的实验路径是作者本地证据位置，数据与权重未随仓库公开；不能视为下载后自动存在的文件。

## 配置自己的环境

基础安装与合成检查见[training_r0入口](../training_r0/README.md)。原生运行仍需按项目README配置匹配的runtime、桥接和资源。

| 配置 | 用途 |
| --- | --- |
| `CR_NATIVE_RESOURCE_ROOT` | 与绑定runtime匹配的`csv_logic`目录；默认`local/resources/csv_logic` |
| `CR_NATIVE_RUNTIME_EXPORT` | 原生导出目录；默认`local/runtime-export` |
| `CR_NATIVE_LEGACY_ARM_LIBRARY` | 可选旧ARM64库，仅用于历史覆盖审计 |
| `--downloader-root` | 处理命令显式指定下载器源码路径；不会自动读取作者机器目录 |
| `ROYALEAPI_DOWNLOADER_ROOT` | 外部下载器契约集成测试使用的路径 |

`training_r0/data` 中的`source`路径是生成时的来源记录；加载内置元数据不依赖该原始绝对路径。这些元数据与指定runtime绑定，不承诺适用于任意游戏更新版本。新资源需重新生成、核验指纹和准入证据。

下载器契约集成测试需要外部下载器及真实匹配资源；环境未准备时明确跳过。普通CPU回归不需要启动游戏或准备模型文件。

## 训练入口

```bash
# 查看帮助，不运行采集或训练
python -m data_pipeline.pilot100 capture --help
python -m data_pipeline.r0_compile --help
python -m data_pipeline.r0_training_smoke --help
python -m data_pipeline.r0_train --help
```

`r0_train` 接收已准入数据及对应ledger，持续检查数据/证书/分片绑定。原始回放JSON不能直接当作其训练输入，不能为绕过失败而开启不完整采集或修改证书。

## 本次本地发布检查

- CPU离线回归：110通过，4项需要外部资源的集成测试明确跳过。
- 指定本机匹配资源与下载器后：上述4项集成测试通过，共114项。
- tiny丰富输入smoke：推理、联合动作概率重算及PPO梯度更新通过。
- 本轮未启动正式训练或重新构建/认证原生游戏运行时；CI验证范围为CPU离线链路。
