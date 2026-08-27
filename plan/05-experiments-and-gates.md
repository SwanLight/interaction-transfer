# 05 — 对照实验、因果干预与闸门

## 0. 实验总表

| # | 实验 | 载体 | 回答什么 | 闸门 | 需新训练 |
|---|---|---|---|---|:---:|
| 零 | 分辨率与位姿噪声敏感度 | E-T (C4) | 表示分辨率该定多少；物体位姿精度要求 | — | 部分 |
| 一 | 信息条件对照 C0–C5 | E-T | 哪类信息有增量价值；精确力是否冗余 | Gate D | 是 |
| **二** | **留出任务零样本** | **E-I** | **执行器是否任务无关** | **Gate E** | 是 |
| 三 | Cross-embodiment | E-I | 同一 envelope 能否被多形态实现 | Gate F | 否 |
| 四 | 扰动恢复 | E-T + E-I | 交互信息在偏离时是否提供纠正方向 | Gate D' | **否** |
| 五 | Matched counterfactual | E-T + E-I | executor 是否真的在用某个字段 | Gate C | 否 |
| 六 | Shared Structure | 两者 | 模型 envelope 能否替代 oracle envelope | Gate C | 否 |

实验三、四、五、六都是**纯评估**，在已有 checkpoint 上跑，代价接近零。这是把它们全部排进计划的理由。

---

## 1. 实验零：分辨率与位姿噪声敏感度

先做，因为它的结论冻结 `02` §2 的表示维度和 `07` 的硬件指标。

### 1.1 表面分辨率扫描

擦拭 × Allegro × C4，表面点数 {64, 256, 1024, 4096}，3 seed。取下游成功率饱和点。

同一个实验同时冻结三个数：表示维度、传感 pitch 目标、面位姿精度目标。初版把这三个数分别在 `02`、`idea` Phase II 的不同地方各自拍定，互不一致，这是设计缺陷。

预期：擦拭在 1024 左右饱和，抽屉/旋钮 256 就够。若擦拭在 4096 仍未饱和，说明接触斑块的精细结构确实重要，应据此上调硬件指标而非缩表示。

### 1.2 物体位姿噪声

**纯评估，代价近零，但回答部署时最脆弱的假设有多脆弱。**

整套表示是 object-centric 的，真机上需要 6D 物体跟踪 + mesh。在 executor 观测的 object pose 上加噪：

```text
位置：0 / 1 / 2 / 5 mm
姿态：0 / 1 / 2 / 5 deg
```

跑所有 Tier 1 checkpoint，找成功率拐点。输出直接进 `07` 作为视觉侧精度指标。

**同时看 C0 与 C4 的退化斜率**：若 C4 退化更慢，说明接触反馈在部分补偿物体位姿误差——这对硬件路线是重要利好，值得单独报告。

---

## 2. 实验一：信息条件对照（E-T）

### 2.1 问题

> 在相同当前反馈、相同 reward、相同训练预算下，增加哪一类 desired interaction information，能稳定提高任务成功率和泛化？

### 2.2 条件

| 条件 | 目的 |
|---|---|
| C0 结果轨迹 | effect-only baseline |
| C1 +区域 | 检验 interaction region 的增量价值 |
| C2 +engage 方向 | **≈ KITE 几何接触意图**，本工作的直接对照点 |
| C3 +接触模式 | 检验 stick/slide/separate |
| C4 完整功能交互 | 加入 task-relevant mechanics 范围 |
| C5 精确示教力 | 检验复制 source exact wrench 是否必要或过度 |

C1→C2→C4 的两段增量，就是本工作相对 KITE 的差值。这两段若都不显著，`00-positioning.md` §5 的第 3 条声称作废。

补充 leave-one-out：从 C4 中分别去掉区域、方向、模式或 mechanics，确认结论不是由固定添加顺序造成。

### 2.3 分层与 seed

按 `04` §11。Tier 1（擦拭）跑全部六条件 5 seed；Tier 2（旋钮）、Tier 3（抽屉）跑子集 3 seed。

### 2.4 稀疏 reward 子实验

按 `04` §8.2，Tier 1 组合额外跑 C0/C2/C4 的稀疏 reward 版本。

**这一组可能是实验一里最有信息量的。** 稠密 effect reward 本身在持续告诉执行器物体该怎么动，它替 C0 承担了大部分信息；去掉之后，interaction 字段作为"规格"的价值才暴露出来。

### 2.5 报告

task success、effect error、interaction tracking（仅诊断）、样本效率、failure mode、safety violation。

预期不是所有任务都要求完整交互：

- 抽屉：C0 可能已足够（阴性对照，见 Gate G）；
- 旋钮：主要检验 region 和 mechanics（`01` §3.1 的轮缘低摩擦是前提）；
- 擦拭：主要检验 mode 和法向/切向作用。

**如果某个任务上额外信息没有帮助，如实报告，不能调任务直到 baseline 失败。**

---

## 3. 实验二：留出任务零样本（E-I）—— 主实验

### 3.1 协议

按 `04` §5.4 的划分。对每个执行器：

1. 冻结其 E-I checkpoint 和可行性评价器；
2. 取**留出任务**确认集的 envelope（由 oracle 统计程序或 Shared Structure Model 产生）；
3. 直接执行。**不给 reward，不微调，不重训，不改 envelope。**

### 3.2 指标

| 指标 | 说明 |
|---|---|
| 零样本成功率 | 主指标 |
| 相对 E-T 同任务 C4 的差距 | E-T 在该任务上是训过的，这是上限参考而非公平对手 |
| 有无可行性评价器的差异 | 检验 §5.3 那一层是否真的有用 |
| 跟踪质量 vs 任务成功 | 分开报告，见 `06` §2.2 |

### 3.3 为什么这个实验没有循环论证

E-I 用交互跟踪 reward 训练，看起来"给了交互信息当然能跟踪交互"。但主指标不是跟踪质量，是**留出任务的成功率**——而留出任务的 success 判据（dirt 清除量）从未出现在 E-I 的任何 reward 中，其 envelope 也来自 E-I 从未训练过的物体。执行器要成功，必须真的把"规格"转成了有效动作。

**这也是为什么 `04` §8 的禁令只约束 E-T。** 两个实验的隔离靠训练数据和 reward 的划分保证，不靠同一个 reward 里的公平性。

### 3.4 失败时怎么办

若留出任务差而训练任务好：扩 `03` §2.4 的预训练物体集，**不得把留出任务加进训练**。后者会立刻让数字好看，也会立刻让主张归零。若扩了两轮仍不成立，如实报告 Gate E 未通过。

---

## 4. 实验三：Cross-Embodiment

同一条冻结 Functional Envelope 直接输入不同 executor，**禁止在主结果中根据 target 事后改写**。

### 4.1 主报告

- 未修改 envelope 的成功率；
- 每种 executor 的动作、接触区域和接触拓扑；
- 物体最终结果差异。

擦拭上的四宫格是核心图：双板 source（持工具）、Allegro（持工具）、Allegro（掌面直擦）、垫头杆（垫面直擦）。**同一份 envelope，一个抓着工具、一个用手掌、一个是根不能抓东西的杆子。**

### 4.2 可行性适配（可选、单独报告）

若允许 target 在 envelope 的"允许区域/允许力范围"内选择具体实现，这仍是同一 functional specification——这正是 `04` §5.3 可行性评价器做的事。

如果直接把 region、mode 或 wrench 替换成 envelope **之外**的值，则属于 embodiment-specific adaptation，不能再称为同一 interaction，必须单独成节报告。

---

## 5. 实验四：扰动恢复（新增）

### 5.1 为什么必须有

整个硬件 pitch 建立在触觉上，但初版计划把接触反馈统一发给所有条件之后，就再没有任何实验测它的价值。**审稿人会一眼看到这个洞。**

而且这大概率是 C0 真正失败的地方：物体已经偏离期望轨迹时，effect 只说"该到哪"，不说"为什么没到、该改什么"；而 mode（现在在 slide 而不是 stick）和 mechanics 范围（法向力不够）**直接指出了纠正方向**。

文献支持：[FoAR](https://arxiv.org/html/2411.15753) 显示视觉不足以完成需力反馈的任务；[CordViP](https://arxiv.org/pdf/2502.08449) 去掉接触预训练在 Flip-Cup 上显著掉点。但 [ForceFlow](https://arxiv.org/html/2605.11048) 报告单轴压力调节任务上力反馈单独就够——**因此本实验的结论必须按任务几何类别陈述，不得全局化**。

### 5.2 扰动类型

episode 中途（progress 40–60% 随机时刻）注入：

| 扰动 | 幅度 |
|---|---|
| 摩擦系数突变 | ×0.5 / ×2 |
| 物体质量/阻尼突变 | ×0.5 / ×2 |
| 外部推力冲量 | 沿随机方向，量级为任务典型接触力的 0.5–1.5 倍 |
| 诱导滑移 | 瞬时降低接触面摩擦 200 ms |

### 5.3 指标

- **恢复率**：扰动后仍完成任务的比例；
- **恢复时间**：从偏离到重新进入 envelope 的时长；
- **纠正方向正确率**：扰动后第一个 200 ms 内，动作变化方向是否与"回到 envelope"一致。

第三个指标最能说明机制，且只需要 rollout 日志就能算。

### 5.4 载体

E-T 的全部 Tier 1 checkpoint（C0–C5）+ E-I 的全部 checkpoint。纯评估。

**预期**：无扰动时 C0 与 C4 的差距小；有扰动时差距放大。若差距不放大，这是对 idea 的实质性反证，必须如实报告。

---

## 6. 实验五：Matched Counterfactual

不用随机打乱整条时间序列。乱序会产生训练分布外输入，无法证明某字段具有因果作用。

对冻结测试 episode 做受控替换：

- effect 保持不变，只换成另一条相同 phase 的 region；
- effect 和 region 保持不变，只换 engage 方向；
- effect、region、方向保持不变，只换 mode；
- 前四者保持不变，只旋转 task-relevant force/torque 方向；
- 保持字段数值落在训练分布内；
- **每次只改变一个因素。**

若改变某字段产生与任务物理一致的**特定**失败，才说明 executor 确实使用了它。

在旋钮上，"把 region 从销钉换到低摩擦轮缘"应产生打滑失败——这是 `01` §3.1 设计的直接检验，也是最容易看懂的一个反事实视频。

---

## 7. 实验六：Shared Structure

比较三种输入：

1. 某一条 source 的精确 interaction record；
2. 多条示教统计得到的 oracle Functional Envelope；
3. Shared Structure Model 预测的 Functional Envelope。

在四个测试集上比较：未见 episode、未见 source 策略、未见物理参数、小幅未见几何。**外加擦拭的跨实现测试**（`03` §7）。

Shared Structure 成立需要同时满足：

- 模型输出接近 oracle envelope，**coverage ≥ 90% 且 width ≤ oracle 的 1.5 倍**（`03` §8.1）；
- 未见 source 策略上 executor 仍成功；
- 输出不容易泄漏 source 策略身份；
- envelope 不比单条精确示教更依赖某一种 target morphology；
- 擦拭上两种实现的 envelope 可互换。

---

## 8. Mechanics 最小化比较

特别比较：exact source 6D wrench（C5）/ task-relevant generalized force / 只给方向 / 给允许范围（C4）。

**若 C4 与 C5 相当，说明 source 的其他 6D 分量是形态特异的冗余，应从 functional representation 中删除。** 这是本工作可发表的最小性结论，也是相对 CHORD 的差异点（见 `00-positioning.md` §2.3）。

---

## 9. 冻结与统计

- 开发集允许调环境和训练；确认集生成后冻结；
- 校准集独立于两者，只用于 conformal 阈值标定；
- 所有条件使用相同 episode seed 和物理参数，做 **paired comparison**；
- Tier 1 报 5 个训练 seed 的均值、标准差和 bootstrap 置信区间；Tier 2/3 报 3 seed；
- **不以"超过任意 25 个百分点"作为唯一判据**，而看置信区间、跨 seed 一致性和任务物理解释。（`log/progress.md` 早期版本写的 25pt 硬阈值以本条为准，已同步修正。）

---

## 10. 闸门

### Gate A：环境可学

Privileged Expert 在固定环境 ≥95%，随机环境 ≥85%。否则停止一切 representation 实验。

### Gate B：完整交互可执行

C4 输入下，可行 task×executor 组合在冻结测试集 ≥80%。否则先定位 executor，不要动表示。

### Gate C：executor 确实在用这些字段

实验五中，至少 region、mode、mechanics 各有一个字段的受控替换产生与物理一致的特定失败。若某字段替换后毫无影响，该字段在该任务上要么冗余、要么被忽略，两种情况都必须如实写出来。

### Gate D：信息增量可信

至少在擦拭或旋钮中，**C1→C2 或 C2→C4 的某一段**在相同 reward 下产生跨 seed 一致、统计可信、物理上可解释的提升。

注意闸门定在 C2 两侧，不是 C0 两侧——C0→C4 有提升只说明"接触信息有用"（已知结论），C2→C4 有提升才说明"本工作相对 KITE 有增量"。

### Gate D'：扰动下优势放大

实验四中，C4 相对 C0 的恢复率差距显著大于无扰动时的成功率差距。这一条不通过不阻断后续实验，但会削弱硬件必要性的论证，须在论文限制一节明确写出。

### Gate E：执行器任务无关（**主闸门**）

至少两个执行器在**留出任务**上零样本成功率 ≥60%，且其中至少一个是垫头杆或钩杆这类 0 自由度形态。

不通过 → 主张不成立，退回 `04` §5.2 检查指令多样性，或如实降级为"E-T 的表示消融"这一较弱的工作。

### Gate F：跨形态成立

至少**三种**形态明显不同的 executor 在未改写同一 envelope 的情况下成功，并且动作与接触实现明显不同（按 `06` §6 量化，不能只靠"视频看起来不同"）。

擦拭上的目标组合：Allegro 持工具、Allegro 掌面直擦、垫头杆直擦。

### Gate G：阴性结果可接受

抽屉上若 C0 与 C4 相当，视为**合理阴性对照**，不修改任务追求差距。同理，若 `02` §7 第 9 条的 region 可推导性探针显示某任务上 region 冗余，如实报告该任务不适合检验 region。

---

## 11. 如何解释失败

| 现象 | 优先解释 |
|---|---|
| Expert 失败 | 环境、几何、动作空间或 reward 有问题 |
| Expert 成功，C4 的 E-T 失败 | 网络、训练课程或观测定义有问题 |
| C4 成功，但 C0 也成功 | 该任务不需要额外信息，正常结果（Gate G） |
| C2 ≈ C4 | 几何接触意图已足够，mechanics/mode 在该任务冗余。**这是对本工作的实质性打击，必须正面写。** |
| C4 ≈ C5 | 精确力冗余，正面结论（§8） |
| E-I 训练任务好、留出任务差 | 预训练物体集覆盖不足，**不得加留出任务数据** |
| oracle envelope 成功，模型 envelope 失败 | Shared Structure Model 未学好 |
| 单条精确示教成功，shared envelope 失败 | envelope 构造过宽/过窄或时间对齐错误，查 coverage 与 width |
| 一个 target 成功、另一个失败 | 检查几何可行性和 executor 能力，不能立即否定表示 |
| counterfactual 不影响结果 | policy 忽略该字段，或该字段在该任务不必要 |

只有在 Expert、训练流程和几何可行性均通过后，多个任务和多个 executor 仍无法从任何 interaction 表示获益，才需要重新审视核心 idea。
