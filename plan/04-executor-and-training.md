# 04 — Executor 架构与训练

## 1. 总体原则

1. **两个执行器，两个实验，不要混**（§2）。E-I 是最终系统，E-T 只是对照实验的载体。
2. 每种执行器拥有自己的 executor；本轮不要求同一网络零样本控制新形态。
3. 不同 interaction 输入条件使用相同 reward、训练预算和当前反馈。
4. 先用 Privileged Expert 确认**环境可学**，避免把环境或控制问题误判成
   representation 失败。⚠️ **不要称它为"性能上限"**，见 §3 的修订。
5. 浮动底座由动力学 PD wrench 驱动，不能直接设置 root pose。

## 2. 两个执行器

### 2.1 为什么必须拆开

初版只有一个任务奖励执行器，并在 §8 明令禁止交互跟踪 reward。那条禁令**在它的语境里是对的**：若用交互跟踪 reward 跑 C0 vs C4 的对照，C0 连要跟踪的字段都没有，"C4 赢"是同义反复。

但代价是：**整个计划从头到尾没有训练过 idea 描述的那个东西。** idea 说的是"执行器先摸索自己形态的交互特性和能力，之后由预测的 interaction 驱动生成动作"——那需要一个任务无关的执行器，而任务奖励执行器不是。

同时，任务奖励执行器上的信息对照大概率会打平：执行器既能感知当前接触力（§4），又有稠密的物体结果 reward，PPO 完全可以自行重新发现该往哪接触、用多大力。desired region/mode/wrench 只是提前告知答案，而 RL 本来就擅长自己找答案。打平的原因与 idea 对错无关。

解法是拆成两个实验，各自回答一个不会被对方污染的问题。

### 2.2 对照表

| | **E-T** 任务奖励执行器 | **E-I** 交互跟踪执行器 |
|---|---|---|
| reward | 任务 reward（角度/开度/清除量） | 交互跟踪 reward，**无任务概念** |
| 训练数据 | 三个任务 | **预训练物体集 + 部分任务**，留出至少一个任务 |
| 回答的问题 | 交互信息能否帮助一个学任务的策略 | 该形态能否实现任意交互规格 |
| 主指标 | 成功率、样本效率 | interaction tracking；留出任务成功率是强泛化指标 |
| 对应实验 | 实验一（支撑） | **实验二（主）** |
| 进入最终系统 | 否 | 是 |

## 3. Privileged Expert（**按需训练**，见 D-30）

**强制的是脚本可行性验证，不是 Expert 训练。**

| 层 | 内容 | 范围 |
|---|---|---|
| **强制** | 脚本可行性验证（走 policy 动作接口）+ reward 量纲标定，`tools/s2_scripted.py` | 全部可行组合 |
| **按需** | Expert 训练 | Tier 1 的 2 个组合 + 任何需要甄别失败原因的组合 |

**硬规则**：

> **在没有为某个组合训练过 Expert 之前，绝不允许把该组合的失败归因于「表示信息不够」。**

这条不得违反。省的是什么时候花钱，不是严谨性。

### ⚠️ Expert 是"可学性检查"，不是"性能上限"（2026-08-30 修订）

原文一边称 Expert 为"性能上限"，一边按 D-32 用"能明确学会"作为 Gate A2 的
通过条件，而抽屉×钩杆实测只有 43.1% / 49.3%。**43% 不是任何东西的上限。**
两个说法必须二选一，这里选前者：

- Expert 的作用**只是**证明"这个组合物理上做得到、且 RL 学得动"；
- **一旦要用它来判断"是不是表示信息不够"**（D-30 的硬规则），
  就必须为那个具体组合训练出一个**明显更强**的 Expert 或 C4 参考，
  并如实报告它的成功率；拿 43% 去论证"表示不足"是没有依据的。

脚本验证的 100% 只证明动作接口与物理可行，不证明网络能稳定实现——
这两件事在 D-30 里已经分开，措辞上也要跟着分开。

**Expert 不进入最终系统，不用于证明任何科学结论。**

### Expert 观测

- executor 完整状态；
- 物体、工具和参照物完整状态；
- 当前接触集合和接触力；
- 物理参数；
- 任务目标。

### Expert 动作

- 浮动底座期望位移/姿态增量，经 PD 转换为外力和力矩；
- Allegro 关节 position targets；
- 夹爪开合 target。

### Expert reward

- 任务结果/进度；
- success bonus；
- 碰撞、关节限位、过大接触力、动作突变惩罚。

## 4. Executor 共同观测

E-T 和 E-I 的所有条件都获得**完全相同的当前反馈**：

- proprioception；
- 当前物体/工具/参照物状态；
- 当前实际接触 flag、接触力或物体响应（本轮仿真中作为统一反馈）；
- 物体表面点云及法向；
- 当前 phase 和 progress，**但必须由通用量导出，不得直接读任务的阶段机**（见下）。

### ⚠️ phase 与 progress 不能是任务 oracle（2026-08-30 修订）

采集脚本里的 `phase`（approach/establish/manipulate/release）和 `progress`
是**那个任务的状态机标签**。直接发给 E-I，等于递给它一个任务专用状态机——
"现在该进入操作阶段了""现在擦到 60% 了"——**"任务无关"当场作废**。

E-I 侧必须由这四个通用量导出，全程不含任务分支：

| 导出量 | 由什么算 |
|---|---|
| 是否已建立接触 | 指令 region 内的实际法向力是否超过阈值 |
| 交互进度 | 实际 effect 与指令 effect 的累计误差缩减比例（`effect/rigid` 与 `effect/surface_state`，D-53 的统一接口） |
| mode 是否达成 | 活跃表面点的实际 stick/slide 与指令 mode 的匹配率 |
| 阶段是否推进 | 上面三项满足后才前移指令窗口（`02` §5 的"按自己的进度选窗口"） |

**S4 记录里的 `phase` / `progress` 因此被移出模型输入**
（`records.SCHEMA_PREFIXES` 把它们归到丢弃前缀），只作为 S5 对齐示教的**标签**
与诊断量保留——标签可以用，观测不行。

条件之间唯一不同的是 **desired future interaction** 中包含哪些字段。这样避免把"目标指令不同"和"当前传感反馈不同"混为一谈。

## 5. E-I：交互跟踪执行器（主系统）

### 5.1 Reward：任务无关

```text
r = w_e · r_effect + w_r · r_region + w_m · r_mode + w_f · r_mech + r_safety
```

| 项 | 定义 |
|---|---|
| `r_effect` | 被操作物体实际状态变化与**指令 effect** 的负误差 |
| `r_region` | 落在指令 region 内的接触法向力占比；region 外的接触受罚 |
| `r_mode` | 各活跃表面点的实际 stick/slide 与指令 mode 的匹配率 |
| `r_mech` | 实际 object-frame surface traction 对当前 command 统计带的法向、切向方向与幅值误差；并检查其积分 6D wrench 一致性。全程不做任务专用广义力投影 |
| `r_safety` | 法向力上限、关节限位、动作平滑、非目标碰撞 |

**明令禁止出现在 E-I reward 中的项**：

- 任务 success bonus（"抽屉开到 100 mm 加分"）；
- dirt grid 清除量；
- 任何以任务关节角/位移为目标的项；
- 任何 task id、任务名或任务分支。

`r_effect` 看起来像任务项，但它跟踪的是**指令中的 effect**，不是任务定义的目标。执行器不知道自己在开抽屉，它只知道"这个物体应该这样动"。这是 E-I 任务无关性的关键，实现时必须确保 effect 目标来自指令通道而非环境配置。

### 5.2 指令从哪来

三个来源，混合采样：

1. **回放**：训练任务的 envelope（Allegro 用抽屉+旋钮，等等，见 §5.4）；
2. **扰动回放**：只对完整、真实出现过的 command sequence 做有限时间缩放和统一
   物理扰动，不把 region / traction / mode 独立拼接成不存在的交互；
3. **预训练物体集**：`03` §2.4 的方块/立柱/滑块/斜面上，由双板 source 脚本产生的物理有效指令。

第 3 项必须占相当比例。如果 E-I 只在三个任务的 envelope 上训练，"留出任务零样本"就退化成"三选一的插值"，说服力大打折扣。

**不要随机凭空生成指令**：随机采样的 (region, mode, wrench, effect) 四元组通常物理上互不相容，训出来的执行器会学会忽略指令。所有指令必须来自**真实跑通过的物理轨迹**。

### 5.3 本体能力如何进入系统

核心不是额外求一次集合交，而是**每种 embodiment 各自完成 interaction-conditioned
executor 训练**：其 policy、状态表示、动作空间与 replay 已经编码“这个身体怎样实现
给定 interaction”。部署时直接由该 executor 解码 command。

只有当 interaction artifact 明确包含多个替代实现，而且直接 decoder 无法稳定选择时，
才把 empirical executability evaluator 作为可选增强做独立消融。它不是 embodiment
物理能力真值，不是当前主流程的前置闸门，也不代表 idea 的算法就是“求交”。

### 5.4 训练/留出划分（泛化评估，不是协议定义）

每个执行器留出一个它**物理上做得到、但训练中从未见过**的任务。

| 执行器 | 训练指令来源 | **留出任务** |
|---|---|---|
| Allegro | 预训练集 + 抽屉 + 旋钮 | 擦拭（持工具 + 直擦两种实现） |
| 垫头杆 | **仅预训练集** | 擦拭 |
| 平行夹爪 | 预训练集 + 抽屉 | 擦拭 |
| 钩杆 | 仅预训练集 | 抽屉 + 旋钮 |

**垫头杆那一行是最强的一格**：一个 0 自由度、完全不能抓握的执行器，从未见过三个任务中的任何一个，直接执行擦拭的 envelope。如果这一格成立，"执行器学的是形态能力而非任务"就很难被反驳。

留出任务的 reward **绝不能**在 E-I 的任何训练阶段出现，包括 curriculum、包括调试。这一条要写进 dataloader 的断言里，不能只靠自觉。

## 6. E-T：任务奖励执行器（对照实验载体）

reward 与 §8 一致，所有信息条件相同。E-T 按 `02` §6 的 C0–C5 六个条件各训一遍，用于实验一。

E-T 不进入最终系统，其结果只支持"信息增量"这一条声称。

## 7. 网络结构

E-T 与 E-I 使用**相同结构**，只有指令通道的字段掩码不同。

### 几何与 Region 编码

输入表面点（分辨率由 S4.5 定，见 `02` §2）：坐标、法向、部件标签、desired region weight、desired engage direction。使用 PointNet：

```text
shared MLP 64 → 128 → 128
weighted/max pooling
输出 128 维
```

若 S4.5 判定需要高分辨率，按 `02` §2.3 采用"低分辨率全局 + 接触邻域局部高分辨率"。

### 时间指令编码

未来 10 步的 fixed effect、direction、mode 和 task-agnostic traction constraint 序列
进入两层 GRU，hidden 128。一次送入一段完整的 time-indexed interaction command，
不接收 task id 或 source strategy id。

**缺失字段用零值 + 独立 field mask 表示。不同条件保持完全相同的网络结构和参数量**——这是把性能差异归因到信息而非容量的前提。

E-I dataloader 使用 exact allowlist，只允许 `effect/rigid`、`effect/surface_state`、
surface geometry、region、engage、mode 与 task-agnostic traction constraints。
`mech/generalized`、任务原生 effect、任务 phase/progress、task id/name 任一出现都硬报错。

### 自身状态编码

proprioception、object state、current contact feedback 进入两层 MLP，输出 128 维。

### Actor/Critic

三个 128 维特征拼接，经过 256→256 的 MLP。Actor 输出动作分布；Critic 输出价值。

采用 asymmetric actor-critic：Actor 只看部署允许信息，Critic 可以看完整仿真状态和物理参数。

## 8. 公平 Reward（仅适用于 E-T）

所有信息条件使用相同 reward：

- task success / object effect tracking；
- 安全法向力上限；
- 非目标碰撞；
- action smoothness；
- joint limit。

**E-T 的主比较中禁止只给 C4 增加 region、mode 或 wrench tracking reward。**

这条禁令**只约束 E-T**。E-I 的 reward 本来就是交互跟踪，那不是作弊，因为 E-I 不用来比较信息条件——它用来测留出任务零样本。两者的隔离由 §2.2 保证。

### 8.1 E-T 的任务 Reward

**旋钮**：按目标方向和角度推进；达到目标角度 success；过大接触力或严重碰撞 terminate。

**抽屉**：按目标方向增加开度；达到目标开度 success；严重碰撞和超时失败。

**擦拭**：dirt grid 实际清除量；清除目标区域 success；过力或超时失败。持工具实现额外：工具失去抓持或翻转失败。

三种信息条件面对完全相同的物理成功定义。

### 8.2 稀疏 reward 变体（实验一的一个子条件）

除稠密 reward 外，对 Tier 1 组合额外跑一遍**只给 success/failure、不给稠密 effect tracking** 的版本。

理由：稠密 effect reward 本身就在持续告诉执行器"物体该怎么动"，它承担了 C0 条件的大部分信息。稀疏 reward 下 C0 几乎必然失败，而 interaction 字段提供的正是稠密的规格。**这一组最能说明 interaction 信息作为"规格"而非"提示"的价值。**

seed 数按 Tier 2 标准（3 seed），只在 Tier 1 组合上跑。

## 9. 主训练算法

使用 Isaac Lab 内置 rsl_rl PPO。

| 参数 | 初值 |
|---|---:|
| control frequency | 50 Hz |
| horizon | 32 |
| gamma / lambda | 0.99 / 0.95 |
| clip | 0.2 |
| learning rate | 3e-4 adaptive KL |
| epochs | 5 |
| parallel envs | 钩杆/垫头杆/夹爪 2048，Allegro 1024 |

## 10. 训练课程

### E-T

- **A 固定环境**：固定摩擦、阻尼和初始位置，用 C4。确认动作空间、网络和 reward 能学。
- **B 逐步随机化**：先随机初始姿态 → 再随机目标 → 最后加入摩擦、阻尼和小幅几何变化。
- **C 信息条件**：环境和训练流程冻结后，分别训练 C0–C5。

### E-I

- **A 单指令跟踪**：固定物体、固定一条指令，确认跟踪 reward 可学、各项量级合理。
- **B 预训练物体集**：在 `03` §2.4 四类物体上训练，指令随机化。这是"摸索自己形态能力"的主体阶段。
- **C 加入训练任务 envelope**：按 §5.4 加入该执行器的训练任务。
- **D 冻结并评估 executor 的 interaction coverage**：在未用于训练的物理指令、物体
  与任务上分别报告距离、跟踪率和失败模式。额外选择器若启用，另列实验，不并入主链。

**留出任务在 A–D 的任何阶段都不得出现。**

## 11. 计算预算与分层

初版要求"每个可行 task×executor 组合 × 6 条件 × 5 seed"，按可行性矩阵约 200+ 次 PPO 训练；而 `log/progress.md` 只列了 14 次。两者都不对。分层如下：

| 层 | 内容 | 条件 | seed | 运行数 |
|---|---|---|---:|---:|
| Expert | **按需**（Tier 1 的 2 个 + 失败甄别） | — | 1 | **2–4** |
| S4.5 | 分辨率扫描：擦拭 × Allegro × {64,256,1024,4096} | C4 | 3 | 12 |
| **E-I** | 4 个执行器 | — | 3 | **12** |
| **Tier 1** | 擦拭 × {Allegro, 垫头杆} | **C0–C5 全部** | **5** | **60** |
| Tier 1s | 同上，稀疏 reward | C0, C2, C4 | 3 | 18 |
| Tier 2 | 旋钮 × {Allegro, 钩杆} | C0, C2, C4 | 3 | 18 |
| Tier 3 | 抽屉 × {Allegro, 钩杆, 夹爪}（阴性对照） | C0, C4 | 3 | 18 |
| | | | **合计** | **147** |

**实验零（物体位姿噪声）和实验四（扰动恢复）是纯评估，不需要新训练**——直接在已有 checkpoint 上跑，代价接近零。这是把它们排进计划的主要理由。

按 Allegro contact-rich 单次 6–10 h、8×4090 并发 8 路估算，纯计算约 5–8 天，计入失败重跑和调试现实上是 3–5 周。Tier 2/3 是可裁的：若进度紧张，先砍 Tier 3 到 C0/C4 各 1 seed（阴性对照本来就不需要高统计功效），再砍 Tier 2 的 C2。

**不可裁的是 Tier 1 和 E-I。** 前者是唯一的完整条件序列，后者是主张本身。

## 12. 不可行目标与 Capability

不可行目标不混进正常成功训练。它们进入单独的 capability evaluation：

- 对每个候选目标从多个初始状态、多次控制优化；
- 分开记录物理不可达、policy 失败和安全限制失败；
- 输出称为 empirical capability map，不直接声称物理能力真值。

这张图用于理解训练后的覆盖边界，不自动成为部署模块。若未来启用 §5.3 的可选
evaluator，再单独比较两者并验证它是否真的提升下游成功率。

## 13. 诊断顺序

1. 几何脚本是否能完成；
2. Privileged Expert 是否达到上限；
3. E-I 在训练指令上跟踪是否收敛；
4. C4 单任务 E-T 是否学会；
5. reward 各项量级和 termination 分布是否正常；
6. **Actor 是否真正读取 desired 字段**（用 `05` 实验五的 matched counterfactual 查，不要靠猜）；
7. 最后才调 PPO 超参。

| 现象 | 优先解释 |
|---|---|
| Expert 失败 | 环境、几何、动作空间或 reward 有问题，**不是 idea 被否定** |
| Expert 成功，C4 的 E-T 失败 | 网络、训练课程或观测定义有问题 |
| E-T 全条件成功但 E-I 跟踪不收敛 | 交互跟踪 reward 的量级配比有问题，先查各项数量级 |
| E-I 训练任务好、留出任务差 | 预训练物体集覆盖不足，扩 `03` §2.4，**不要去加留出任务的数据** |

最后一行是最容易破防的地方。留出任务差的时候，把留出任务加进训练会立刻好看，也会立刻让主张归零。
