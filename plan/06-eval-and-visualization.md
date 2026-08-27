# 06 — 评估、可视化与最终产出

## 1. 独立评估程序

训练和评估完全分离。`eval.py`负责：

- 加载冻结checkpoint；
- 使用固定episode/physics seeds；
- 同时运行多个信息条件；
- 保存逐episode指标、视频和失败原因；
- 生成HTML报告。

不能从训练reward曲线直接下科学结论。

## 2. 每次评估必须输出

### 2.1 任务指标

- success；
- object effect error；
- completion time；
- dirt coverage（擦拭）；
- safety violation和峰值法向力。

### 2.2 Interaction诊断

- desired与actual region热图；
- 各表面点contact mode；
- task-relevant force/torque方向和幅值；
- phase与progress对齐；
- exact wrench与generalized force对比。

tracking指标与task success分开报告。成功但tracking差，说明executor可能绕过了interaction；tracking好但任务失败，说明interaction定义可能不充分。

### 2.3 Envelope 质量（新增）

按 `03` §8.1，每次评估必须同时报两个数，缺一不可：

- **coverage**：实际接触/受力完全落在 envelope 内的成功示教比例，目标 ≥90%；
- **width**：region 允许集合占物体表面积的比例；mechanics 上下界的相对宽度。

只报 coverage 会让过宽的 envelope 蒙混过关；只报 width 会让过窄的 envelope 看起来精确。

### 2.4 E-I 专属指标（新增）

- **留出任务零样本成功率**（主指标，见 `05` 实验二）；
- **可行性评价器 AUC**，以及"有评价器 vs 允许集合内随机取"的下游成功率差；
- **评价器高分区域与 empirical capability map 的重合度**——两者应互相印证（`04` §12），不重合处必须查明原因；
- 训练指令 vs 留出指令的跟踪质量差距。

### 2.5 扰动恢复指标（新增，对应 `05` 实验四）

- 恢复率；
- 恢复时间（从偏离到重新进入 envelope）；
- **纠正方向正确率**：扰动后第一个 200 ms 内动作变化方向是否与"回到 envelope"一致。

第三个指标最能说明机制，且只需 rollout 日志即可计算。必须按扰动类型分别报告，不要合并成一个平均数。

### 2.3 失败分类

- 未建立接触；
- 接触到错误区域；
- 滑脱；
- 力/力矩方向错误；
- 过力；
- 工具丢失或翻转；
- phase错位；
- 超时；
- 几何不可达。

## 3. 核心视频

### 3.1 Cross-embodiment四宫格（核心图）

同一Functional Envelope、相同物体参数和相同目标，**擦拭任务**：

| | |
|---|---|
| 双板 source（持工具） | Allegro（持工具） |
| Allegro（掌面直擦） | 垫头杆（垫面直擦） |

一个抓着工具、一个用手掌、一个是根完全不能抓东西的杆子——**同一份未改写的 envelope**。这是整个工作最有说服力的一张图，四格必须来自同一个 envelope 文件，且在视频里标出该文件的 hash。

视频按task phase或object progress同步，而不是强行按绝对时间同步。

### 3.2 留出任务零样本视频（新增）

垫头杆那一格单独出一条：标注"该执行器训练中从未见过擦拭任务、从未见过黑板擦、从未见过 dirt reward"。旁边并排放它的训练指令来源（预训练物体集上的方块/立柱/滑块）。

这是实验二的主结果图，重要性仅次于 3.1。

### 3.3 扰动恢复视频（新增）

同一 episode、同一扰动时刻、同一扰动幅度，并排 C0 与 C4。扰动注入时刻在视频里打标。若 C0 崩了而 C4 恢复了，这一条视频比任何表格都有效。

### 3.4 Source多策略视频

并排展示同一任务不同source策略，确认动作和接触明显不同，但最后目标一致。擦拭必须包含直擦策略（`01` §5.1(b)）。

### 3.5 Counterfactual视频

展示只改变 region、engage 方向、mode 或 mechanics 之一后产生的对应失败。必须确认失败来自物理变化，不是输入维度错误或程序崩溃。

首选镜头：旋钮上把 region 从销钉换到低摩擦轮缘 → 打滑失败。这是 `01` §3.1 设计的直接检验，也是最容易看懂的一条。

## 4. Region可视化

直接在物体表面点云上画热图：

- oracle per-demo region；
- oracle shared envelope（含 conformal 阈值后的允许集合边界）；
- model-predicted envelope；
- target actual contact；
- **可行性评价器在该 envelope 上的打分场**（`04` §5.3），用于看"允许集合里执行器实际选了哪一块"。

不要只画柱状图。分辨率按 `02` §2 的 S4.5 结论，可视化用高分辨率版本。

## 5. Shared Structure评估

- effect trajectory error；
- region heatmap的IoU/earth-mover distance；
- mode准确率；
- mechanics range coverage与宽度；
- source-strategy probe准确率；
- 使用预测envelope后的下游成功率。

最终以executor成功率验证sufficiency，以source identity leakage验证invariance，以字段删除验证minimality。

## 6. 动作差异量化

“视频看起来不同”之外，报告：

- joint/action trajectory distance；
- 接触点数量和拓扑；
- 实际使用的表面区域；
- 动作时长和regrasp次数；
- 相同功能结果下各执行器的energy/force profile。

## 7. 人工检查

每次正式评估前必须：

- 看至少5条成功视频；
- 看至少10条失败视频；
- 确认没有穿模、teleport或利用仿真漏洞；
- 确认matched counterfactual只改了指定字段；
- 确认不同信息条件使用同一批episode；
- 确认没有根据确认集结果继续调场景；
- **确认 E-I 的留出任务确实没有出现在其训练数据里**（查 dataloader 断言日志，不靠记忆）；
- **确认四宫格四格的 envelope 文件 hash 一致**。

最后两条是本版本新增，也是最容易在赶进度时被跳过、而一旦出错整篇作废的两条。

## 8. 最终报告结构

1. 定位与相对 KITE / ART-Glove / CHORD 的差异（`00-positioning.md`）；
2. 环境和几何可行性表；
3. Privileged Expert上限；
4. 分辨率与位姿噪声敏感度（实验零）；
5. Source策略多样性；
6. Shared Structure预测质量（含 coverage / width）；
7. 不同information条件的成功率和样本效率（实验一，含稀疏 reward 子实验）；
8. Mechanics最小化比较（C4 vs C5）；
9. **留出任务零样本（实验二，主结果）**；
10. Cross-embodiment结果（实验三）；
11. 扰动恢复（实验四）；
12. Matched counterfactual（实验五）；
13. 阴性结果和失败模式；
14. 本轮能证明与不能证明的结论。

## 9. 本轮通过后的科学结论上限

如果 Gate A–F 通过，可以声称：

> 在受控的刚性/工具操作任务中，从多种 source 策略学习出的 object-centric functional envelope，比仅描述 object effect、也比仅描述几何接触意图，更能支持接触主导任务；且一个**任务无关**的执行器可以在从未见过该任务的情况下，仅凭这份规格零样本实现它，多种形态差异明显的末端执行器均如此。

若 Gate D' 亦通过，可追加：

> 这一优势在物理扰动下放大，表明该表示提供的是纠正方向而非静态提示。

**不能声称**（与 `00-positioning.md` §4 一致）：

- 已解决任意任务的 universal representation；
- 已证明真人和真实传感器可用；
- 上游 interaction model 能泛化到新任务/新物体；
- 已实现**未见形态**零样本泛化（本轮每个形态各训一个执行器）；
- exact source force 就是最终答案。

