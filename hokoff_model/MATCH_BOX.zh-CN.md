# BC Release 人机对局盒子

这是在原生人机GUI上新增的独立入口：人类蓝方、1037042步BC红方。旧专家和旧GUI入口保留，未修改模型参数、训练方式或R0。

## 已支持

- 读取两份已发布纯权重，默认1037042步；校验权重SHA、词表、静态属性和原生libg版本。
- 补齐双方最近4次**已确认普通出牌**历史；严格使用过去Tick，敌方坐标转到Actor视角，年龄以秒计。
- 普通/觉醒/英雄卡从实际己方卡组及手牌映射；合法位置、实时费用和技能状态由原生接口决定。
- 每4 Tick（200ms）推理一次，原生仍20Hz；阈值0.5、掩码内贪心，无1.5x/2x补偿。
- 从原生重置实际返回的Tick开始对齐、热身，Tick100才开放盒子控制；不伪造Tick0输入。
- 暂停不推进原生时钟；重开清空RNN、历史、公开已揭牌和掩码缓存。
- 重复启动保护、原生回执身份核对、数值和时钟异常停止，不盲目重复下牌。

`FixedPolicy.forward_stream`原来的BC-only保护仍保留。盒子使用单独的`HistoryOnlinePolicy`，只在在线历史齐全时进入流式路径；旧PPO、MuMu接管器和其他runner没有因此自动获得历史支持。

## Windows入口

1. 使用`feature/policy-v1-offline-bc`分支，Python>=3.11，PyTorch>=2.8、numpy、Tk。
2. 按主工程说明配置合法取得的`150535029/x86_64`原生运行时和AOSP Android31 worker。不要使用160402002/ARM64内核替代。
3. 复制`runtime.env.example.ps1`为不入Git的`runtime.env.ps1`并填本机路径；可设置`CR_MATCH_PYTHON`。
4. 双击仓库根目录`HOKOFF_MATCH_BOX.cmd`。

入口会下载并验证默认的19.3MB模型，使用仓库附带的**自有开源Java/C++宿主预编译件**，启动一个无窗口worker，然后打开盒子。预编译件不含libg、APK、游戏资源或模型；其二进制及对应源文件有SHA清单。已有不同宿主文件会先备份到`artifacts/hokoff-host-backups`。修改宿主源码后需显式`-RebuildHost`并配置JDK/NDK，不会静默运行过期二进制。

窗口默认暂停：点下方“开始对局 / 暂停”，选蓝方卡牌后点击战场；结束后点击“重置”再开始下一局。默认双方使用`examples/hog-2.6-evo-hero.json`，也可显式传其他相容卡组。

```powershell
. .\runtime.env.ps1
.\scripts\start_hokoff_match_box.ps1 -Replay .\examples\hog-2.6-evo-hero.json
# 已有正在运行的新版宿主，不重建、不重置它的启动阶段：
.\scripts\start_hokoff_match_box.ps1 -SkipBuild -SkipWorkerStart
```

## 冷启动资源说明

只有解码后的CSV/TOML目录不足以完成冷启动。如果已有worker不包含完整逻辑资源，可指定`CR_SANDBOX_BOOT_RESOURCES`目录，其中须有：

```text
asset-pack.apk
assets/assets.scdb
data/update/data_manifest.toml
```

本版本按这三个冻结文件的SHA检查（见`start_direct_service.ps1`）；它们来自已验证的完整worker模板，**不在本仓库中**。`data_manifest.toml`是原始编码资源，不能按名字当文本重写。缺少时应从已有合法运行时取得，不能用空文件或新版本资源凑数。

盒子的无头启动可设`CR_SANDBOX_BINDERLESS_BOOT=1`，以独立资源Context启动；`app_process`已注册Android框架JNI，不能再次执行裸ART的框架注册。脚本已区分这两种情况并补齐所需的mock类路径，失败会尽早报告，不再无条件等待5分钟。

## Linux/已有Worker

GUI可以在有Tk及图形桌面的Linux上运行；纯无桌面云机不具备显示窗口的能力。也可把原生服务通过SSH转发到自己的桌面机器，在桌面运行盒子。

先停止要更新的那个worker，将`tools/hokoff-match-host/lifecycle-probe.jar`及对应桥接库部署至它的独立目录并重新启动。**不要在正在进行的比赛中替换宿主文件，不要覆盖libg或关闭版本检查。** 裸ART启动保持原来的框架JNI注册；仅app_process使用skip-registration开关。

```bash
python scripts/download_hokoff_match_model.py
python -m hokoff_model.match_box --checkpoint models/hokoff-bc-step1037042.pt --host 127.0.0.1 --port 39031
```

纯权重缺少优化器和完整训练contract，不适合原样断点续训。这里补的是独立推理加载路径，不是伪造训练checkpoint。随附`match_encoder_contract.json`只包含编码维度、词表和哈希，不含原始比赛或数据集；卡牌词表与公开战斗属性表相符，技能词表取自原始native-bc-v1源manifest，来源哈希明确记录。发行权重本身未内嵌完整源manifest。

## 验收与限制

本次Windows/x86_64原生短测两个种子均到Tick260：脚本人类分别3/3次成功出牌，模型分别3/2次成功出牌；动态实体最大5/6，历史槽位最大6/5，RNN非零，重置后不串局。模型为1037042步、SHA `ff1fe42b76dbec2f68f840fd289aa5dda15357d5ab7ac47d3a1db1cbe6f02eb9`。

新增在线历史与原生回执测试通过，覆盖与离线`HistoryIndex`逐项一致、双方坐标、严格过去、重复、未知、清空、固定节拍、敌方私有信息隔离和原生版本拒绝；一次21项组合回归（含新增测试、本机CUDA FP16 AMP回归）也通过，后续新增的端口占用锁测试通过。既有Windows历史缓存并发写入测试曾出现WinError5，盒子不使用该离线磁盘缓存路径，未把该既有问题改写为通过。

这不是胜率验收或全部特殊技能认证；也没有运行新的IL/PPO训练。Linux显示端和远程网络延迟未在本轮实测。默认CPU单线程推理，提供`--device cuda`；不以使用率高低代替延迟验收。

每局日志与`box-status.json`在`artifacts/hokoff-match-box`，包括模型身份、真实Tick、出牌概率、历史数量和原生回执。该目录不入Git。关闭GUI会关闭控制连接；如需释放本地worker/AVD，使用原生worker管理命令，不会因此关闭MuMu。
