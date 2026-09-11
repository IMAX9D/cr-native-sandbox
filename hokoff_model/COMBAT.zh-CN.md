# 显式静态属性：15 小时续训实验

入口：`python train_hokoff_combat.py --hours 15`。
首次从 `~/cr-data/runs/hokoff-fixed-p4-history4-posskip16/last.pt` 迁移到独立目录
`~/cr-data/runs/hokoff-fixed-p4-history4-posskip16-combat`，之后从新目录续训。
不会覆盖源实验，不从零训练。新目录已有其他内容则拒绝覆盖。

迁移保留参数、AdamW 动量/step、GradScaler、RNG、epoch、采样游标和训练步数。
仅实体 MLP 第一层增加 20 列，初始权重和对应动量为零；在 FP32 中初始输出仅有
矩阵尺寸变化带来的浮点舍入差异。实际 240364 步权重与真实样本短测，最大输出差异
低于 1e-5，新属性列有非零梯度并能完成优化器更新；这不代表验证集收益。
256 宽度增加 5120 参数。观测缓存、空间直连头、历史、学习率和损失保持原配置。
每 2000 步保存和验证，epoch 上限提高至 100，15 小时预算后保存并停止；最后验证可能略超时。

## 数据含义与边界

版本 15.535.29，Worker libg SHA-256 与本仓库冻结版一致。
静态表由配套 Worker 的 csv_logic CSV/TOML 解码得到，表内保存配置文件哈希和映射审计。
当前词表 181 项，其中 110 项能映射基础单位配置；其余保留可确认的字段，未知字段同时输出 0 和 known=0。
对混合召唤卡不猜测实际子单位，技能状态、当前攻击目标、伤害等级缩放不接入。

10 个基础字段及对应 10 个 known 标记：圣水费用、是否建筑、攻击地面、攻击空中、
仅攻击建筑、基础 FlyingHeight、Speed、Range、HitSpeed、SightRange。
字段按表中 scales 归一化，是静态基础配置，不是当前速度/剩余冷却。
进化/英雄的特殊攻击序列、临时增益、动态射程不会被这一张表完整表达。
缺省布尔属性不武断当作已知 false；复杂表达式和冲突字段作为未知。

例如火枪手：AttacksGround=true、AttacksAir=true、Range=6000、HitSpeed=1000，
归一化后分别为 1、1、0.5、0.2。表的单位不是从其他版本游戏资料手工抄入。
完整表和 card_vocabulary 存入模型配置及 checkpoint；训练入口要求表与数据词表完全匹配。

## 后台开训并自动评估

```bash
conda activate r2dreamer
cd /home/lenovo/gh/cr-native-sandbox
nohup bash -c 'python train_hokoff_combat.py --hours 15 && python diagnose_hokoff_fixed.py --checkpoint ~/cr-data/runs/hokoff-fixed-p4-history4-posskip16-combat/last.pt --mode ap --device cuda' > ~/cr-data/combat-15h.log 2>&1 &
tail -f ~/cr-data/combat-15h.log
```

不要重复启动同一运行目录。`--dry-run` 可以查看继承的训练参数且不写入迁移结果。
历史模型目前仍只支持 BC 与离线评估，尚未接入真实对战。
此为属性增强实验，不保证优于继续训练原模型；同样不能保证架构已无瓶颈。

## 2026-09-11：AMP 预热修复

长训消融发现新增列未更新，已定位并修复 no_grad 预热污染 AMP 权重缓存的问题。
旧权重可以续训，但建议先短测再长训，详见[诊断说明](AMP_BURN_IN.zh-CN.md)。
