> # ⚠️ 本文件已被 `../plan/` 取代（SUPERSEDED，2026-08-27）
>
> **保留为背景与思路来源，不要照此执行。** 以下内容与现行计划存在直接冲突：
>
> | 本文件 | 现行 `../plan/` | 依据 |
> |---|---|---|
> | 仿真器用 Genesis（§3.2），双仿真器交叉验证 | Isaac Sim / Isaac Lab，单仿真器 | `log/decisions.md` D-01 |
> | Executor 用 BC → DAgger（§7.4） | goal-conditioned PPO；**且拆成 E-T / E-I 两个执行器** | D-08、**D-10** |
> | 任务为 drawer / wiping / **peg insertion**（§3.3） | 抽屉 / **擦拭（主）** / 旋钮 | `plan/README` §5 |
> | 表示 = 64–128 patch（§6.2） | 表面点云，分辨率由 S4.5 扫描确定 | D-15 |
> | 表示条件 R0–R3 四档（§8） | **C0–C5 六档**，新增 C2 几何接触意图 | **D-18** |
> | 主实验 = 表示消融（§8） | **主实验 = E-I 留出任务零样本** | **D-10** |
> | §25 参考文献含 Genesis tactile 接口两条 | 随 D-01 一并作废 | D-01 |
>
> 仍然有效并已被 `../plan/` 吸收的部分：**四阶段依赖顺序**（§2）、**Go/No-Go 闸门思想**（§22）、**先仿真定信息需求再反推硬件指标**（§24）。
>
> Phase II 的传感器研究（§12–§17）尚未被 `../plan/` 覆盖（本轮不做硬件），但其中的**面位姿精度**部分已被 `../plan/07` §3 实质性修改——规格是物体系的，因此世界系绝对位姿要求可以大幅降级。读 §15.4 前请先读 `../plan/07` §3。

---

# Functional Interaction Transfer：研究 Idea 与实验计划

## 0. 研究目标

本工作的核心不是把人的动作重定向到机器人，而是把**动作所实现的功能性交互（functional interaction）**从人的身体中抽离出来，再由不同机器人根据自己的形态和运动能力重新实现。

整条链路是：

```text
Human demonstration
        ↓
Human Interaction Acquisition
        ↓
Functional Interaction
        ↓
Embodiment-specific Executor
        ↓
Robot Action
```

其中最关键的中间层不是 human pose、joint angle，也不是简单的 object trajectory，而是：

> **完成任务所必需、但不规定具体身体怎么运动的交互约束。**

因此本研究要回答三个顺序明确的问题：

1. **任务真正需要保留什么交互信息？**
2. **这些信息在人类操作中最低需要怎样的传感系统才能获得？**
3. **不同机器人能否利用同一份交互信息完成任务？**

这三个问题不能同时验证。必须按依赖关系逐级推进。

---

# 1. Functional Interaction 的实验性定义

目前不先假定一个最终“万能表示”，而是先构造一个足够完整、可逐项删除的候选表示。

对于未来一小段时间内的任务过程，用一串 object-centric interaction states 表示。每个时刻只保留四类信息：

1. **Effect：物体应该怎样变化**  
   例如抽屉沿导轨移动、旋钮绕轴旋转、peg 相对孔继续插入。

2. **Interaction Region：哪里需要发生作用**  
   用物体表面的功能区域表示，例如 drawer handle、table wiping region，而不是人的某根手指或某个固定 fingertip contact point。

3. **Contact Mode：接触处于什么状态**  
   第一版只区分 `no-contact / stick / slide`，需要时再扩展。

4. **Mechanics：接触必须产生什么机械作用**  
   第一版用 object-frame wrench 或局部 3D traction 表示；后续再验证是否可以退化成“方向”“范围”或“可实现的 wrench capability”，而不需要精确复制人施加的力。

时间顺序本身由这一串 interaction states 表达，因此暂时不再额外引入一个复杂的 phase/temporal ontology。物体间的 referential constraint 也优先放进 object-relative state 中，而不是另外再加一套标签。

这不是最终结论，只是实验的最大候选集。真正目标是找到其中**最小而足够的子集**。

---

# 2. 整个研究分成四个阶段

```text
Phase I   Functional Requirement Discovery
          仿真器给上帝视角真实物理量
          → 找出任务真正需要哪些 interaction 信息

Phase II  Human-side Observability & Sensor Design
          已知 Phase I 的信息需求
          → 研究用什么传感器、多少精度才能把它们从人身上恢复出来

Phase III Real Human Interaction Acquisition
          做真实人类采集系统
          → 训练模型从真实传感数据预测 Functional Interaction

Phase IV  End-to-End Human → Robot
          人类数据 → Interaction → 多种机器人执行
```

其中 **Phase I 是第一优先级**。如果 Phase I 不能证明 interaction-level representation 比 effect/action-level representation 更有价值，就不应进入复杂硬件开发。

---

# 3. Phase I：Functional Requirement Discovery

## 3.1 目的

只回答一个问题：

> **如果仿真器把真实物理信息全部告诉我们，完成任务到底需要保留哪些 interaction 信息？**

这一阶段：

- 不使用真人；
- 不使用真实触觉传感器；
- 不模拟柔性皮肤；
- 不研究 surface-pose sensor；
- 不研究 raw RGB。

先把“信息本身是否必要”与“现实中怎么测”彻底分开。

---

## 3.2 仿真平台

### 主平台：Genesis

用于：

- 刚体/关节物理；
- 大规模并行环境；
- PPO 训练；
- 接触力与接触几何读取；
- 后续 tactile sensor ablation。

Phase I 只依赖刚体 contact oracle，因此不要求高保真柔性触觉仿真。

### 交叉验证：Newton

Phase I 得到主要结论后，只挑 1–2 个 contact-rich 任务，在 Newton 的 hydroelastic contact / contact sensor 下重做关键实验。

目的不是两套系统都开发，而是确认结论不完全依赖 Genesis 的单一 contact formulation。

---

## 3.3 第一批任务

不做几十个任务。第一批只做三个核心任务：

### Task 1：Open Drawer

验证：

- object effect 是否已经足够；
- functional contact region 是否必要；
- pulling direction / wrench 是否必要。

### Task 2：Wiping

验证：

- 持续 contact 是否必要；
- normal pressure 是否重要；
- shear / sliding information 是否提供额外价值。

### Task 3：Peg Insertion

验证：

- object-relative motion 是否足够；
- contact state 是否能帮助处理接触丰富的插入过程；
- spatial relation 与 contact mechanics 的作用。

等这三项跑通后，再加入 `Twist Knob/Cap` 作为 torque/stick-slip stress test。

`Pick-and-Lift` 只用于调试仿真、控制和训练流程，不作为核心 benchmark。

---

## 3.4 多执行器设置

机械臂先保持相同，只更换末端执行器，以避免把 arm reachability 与 hand morphology 混在一起。

第一阶段使用三类明显不同的末端执行器：

1. **Parallel-jaw gripper**：作为低自由度基线；
2. **Dexterous hand**：如 Allegro / LEAP 一类；
3. **Simple tool-like embodiment**：如 hook、pad 或其它结构极简的执行端。

不是要求每个任务都能被三种末端执行器完成，而是每个任务至少有两种明显不同 morphology 的可行解。

先建立一张 `Task × Embodiment Feasibility Matrix`，不可物理实现的组合不纳入 transfer 成败统计。

---

# 4. Phase I-A：先训练“会做任务”的 Privileged Experts

## 4.1 为什么先训练 Expert

后续 representation 实验如果失败，必须能确定是“表示信息不够”，而不是“机器人 controller 本来就不会做”。

因此每一个有效的 `task × embodiment` 组合，都先训练一个拥有仿真器上帝视角信息的 Expert。

Expert 可以看到：

- robot joint position / velocity；
- object pose / velocity；
- task target；
- contact point / normal / force；
- articulated joint state；
- 必要的物理参数。

---

## 4.2 控制方式

采用两段式控制，避免让 RL 同时学习无关的 free-space navigation。

### 接触前

用仿真器 IK / scripted target pose：

```text
current pose
→ pre-contact pose
→ contact-ready pose
```

### 接触后

用 PPO 学 manipulation policy。

PPO 可以理解为：机器人不断在并行仿真中尝试动作，依据 reward 更新策略。

第一版 action 不直接输出 motor torque，而是输出：

> **各关节下一时刻的 position target**

底层由 PD controller 跟踪。

这样可以减少低级电机控制对研究结论的干扰。

---

## 4.3 Expert 网络

第一版不需要 Transformer。

```text
Privileged state
      ↓
MLP: 256 → 256 → 128
      ↓
Joint-position target
```

- 激活函数：ELU / ReLU 均可；
- PPO：使用 RSL-RL 或等价成熟实现；
- control frequency：先从 50 Hz 开始；
- physics step 更高频，由仿真器内部积分。

---

## 4.4 Reward 设计原则

Reward 只服务于“稳定得到大量成功轨迹”。不把 reward engineering 当成论文创新点。

以 drawer 为例：

```text
靠近 handle
→ 正确建立接触
→ drawer displacement 增加
→ 接触维持
→ 到达目标开度
```

同时惩罚：

- 碰撞无关物体；
- 关节越界；
- 极端动作速度；
- 不合理的大力冲击。

Wiping 和 insertion 也按“任务进度 + 接触质量 + 成功条件”设计，不为每个细节增加复杂 reward。

---

## 4.5 Expert 通过标准

进入下一步前，Expert 至少满足：

- nominal environment 成功率稳定在约 90% 以上；
- 对初始位置、摩擦、物体质量等做适度随机化后，仍有稳定较高成功率；
- 失败不再主要来自 policy 本身不会做任务。

达不到则先修 Expert，不进入 representation ablation。

---

# 5. Phase I-B：生成 Oracle Interaction Dataset

Expert 成功执行任务时，以统一频率记录完整物理量。

每一帧保存：

```text
robot proprioception
object / reference object pose
object velocity
articulated joint state
contact position
contact normal
normal force
tangential force
relative tangential velocity
source embodiment action
success / failure
```

建议先按 50 Hz 记录。

第一轮每个有效 `task × embodiment` 组合先生成约 **500–1000 条成功轨迹**。如果 representation learning 仍明显欠拟合，再扩充数据，不一开始追求“海量”。

同时有意随机：

- 初始位置；
- 接触位置；
- 物体质量；
- 摩擦；
- drawer resistance / insertion clearance 等任务参数。

目的是让同一任务存在多种成功策略，而不是得到一堆几乎一样的轨迹。

---

# 6. Phase I-C：把 Oracle 数据转换成候选 Functional Interaction

## 6.1 Effect

全部在 object/reference-object 坐标系下表达。

例如：

- drawer：未来 joint displacement；
- wiping object：未来 object/effector 与 table 的相对运动；
- peg：peg 相对 hole 的 future pose。

第一版取未来 **1 秒**，均匀采样 **10 个时刻**。

---

## 6.2 Interaction Region

不要保存 source hand 的精确 fingertip ID。

在 manipulated object 表面建立固定 object-centric patch map。

第一版可把 mesh 分成约 64–128 个 patches；每帧保存：

- active patch ID / soft mask；
- patch centroid；
- patch local normal。

对于 drawer handle 这类有明确功能部件的物体，可直接使用 semantic part 作为 region；对于一般物体再使用 surface patch。

---

## 6.3 Contact Mode

第一版只分三类：

- `no-contact`
- `stick`
- `slide`

通过 simulator ground truth 判定：

- normal force 足够小 → no-contact；
- 有接触且切向相对速度很小 → stick；
- 有接触且切向相对速度明显 → slide。

阈值只用于生成一致标签，不把阈值本身作为研究结论。

---

## 6.4 Mechanics

把所有 contact force 转到 manipulated object frame。

第一版记录：

- 3D resultant force；
- 3D torque about object reference point。

也就是 6D object wrench。

但后续不直接假定“精确 wrench 数值必须迁移”，而是单独比较：

```text
exact wrench
vs.
wrench direction
vs.
minimum required capability / range
```

这样才能判断 human/robot 的精确力值是不是 embodiment-specific redundancy。

---

# 7. Phase I-D：训练 Representation-conditioned Executor

## 7.1 目的

现在开始真正检验：

> 如果只给机器人某种 interaction representation，它还会不会做任务？

每一种 embodiment 都先有自己的 executor。第一篇工作不要求同一个网络直接控制所有机器人。

---

## 7.2 输入输出

Executor 输入：

```text
robot current proprioception
+
current object state
+
未来 1 秒 Functional Interaction sequence
```

输出：

```text
next joint-position target
```

---

## 7.3 网络结构

第一版采用简单、可解释、容易训练的结构：

```text
interaction sequence (10 steps)
          ↓
      2-layer GRU
       hidden=128
          ↓
 interaction feature
          +
robot/object current state
          ↓
      MLP 256→256
          ↓
 joint-position target
```

GRU 只负责理解“这一串 interaction 随时间怎么变化”，不需要一开始上大型 Transformer。

---

## 7.4 训练方式：BC → DAgger

### 第一步：Behavior Cloning

直接使用 Expert rollout：

```text
(current state, interaction representation)
→ expert action
```

训练 Executor 模仿 Expert。

建议起始设置：

- optimizer：AdamW；
- learning rate：`3e-4`；
- batch size：64–128 sequences；
- 先训练到 validation imitation loss 基本收敛。

### 第二步：DAgger

纯 BC 的问题是 Student 稍微跑偏以后就进入训练数据没见过的状态。

因此再做约 5 轮 dataset aggregation：

```text
Student 自己执行
→ 到达新的状态
→ 用 Expert 对这些状态重新给动作标签
→ 加回数据集
→ 继续训练 Student
```

这样得到更稳的 closed-loop executor。

---

# 8. Phase I-E：Representation Ablation

为了避免变量太多，先只做一组**嵌套表示**：

```text
R0 = Effect
R1 = Effect + Interaction Region
R2 = Effect + Region + Contact Mode
R3 = Effect + Region + Mode + Mechanics
```

对于每个 R0–R3：

- 使用同样的网络；
- 使用同样的数据量；
- 使用同样的训练流程；
- 只改变输入信息。

这样才能把 performance 差异归因到 information，而不是 model capacity。

---

# 9. Phase I-F：Cross-Embodiment Test

这是整个 Phase I 的关键实验。

例如 source dexterous hand 做 drawer：

```text
source successful trajectory
→ 提取 R0 / R1 / R2 / R3
→ 不使用 source joint trajectory
```

然后分别输入：

- parallel gripper executor；
- dexterous-hand executor；
- hook executor。

每个 executor 用自己的关节动作去实现同一份 object-centric interaction。

成功标准不是动作相似，而是：

- 任务完成；
- desired object evolution 被实现；
- interaction constraints 被满足。

这样才能真正检验“transfer interaction, not action”。

---

# 10. Phase I-G：必须加入 Intervention，而不能只做普通 Ablation

Ablation 说明“加某类信息以后 performance 是否提高”；intervention 才能进一步说明这类信息是不是功能性约束。

只做几种最关键的干预：

### Region intervention

保持 desired object effect 不变，故意把 region 换到错误/不可施力区域。

### Mechanics intervention

保持 region 和 object goal 不变，把 desired wrench direction 旋转或削弱。

### Contact-mode intervention

在 wiping 等任务中，把应该维持 stick/controlled slide 的 mode 改掉。

### Physics intervention

改变 friction、mass、drawer resistance 等，再比较：

- exact force；
- direction；
- capability/range。

这一步决定最后保留的是“人的具体数值”，还是更抽象的功能性机械约束。

---

# 11. Phase I 的最终输出

Phase I 不输出硬件，而输出一份：

> **Minimal Functional Interaction Specification**

例如最后可能得到：

```text
未来 object-relative state
+
functional interaction region
+
contact mode
+
required wrench capability
```

也可能发现某些任务根本不需要 mechanics。

判定原则：

> 在多个任务、多个 embodiment 上，选择能达到接近 Full Representation performance 的最简表示；新增信息如果只带来很小收益，就不保留为核心字段。

这一步完成后，才进入人侧硬件问题。

---

# 12. Phase II：Human-side Observability & Sensor Design

## 12.1 目的

Phase I 已经告诉我们“任务需要什么物理信息”。

Phase II 不再重新问这些信息有没有用，而是问：

> **为了从人类 demonstration 恢复 Phase I 的表示，最低需要什么 sensing configuration？**

这一步的最终输出不是新 representation，而是一张 **Human Acquisition Hardware Specification**。

---

# 13. Phase II-A：建立虚拟传感器

先使用仿真中的 source hand / hand proxy 作为 demonstrator surface，在表面铺 virtual taxels。

不需要真实 FEM skin。直接把 oracle contact 聚合到 taxel footprint 中，再构造不同 sensor output。

第一轮只比较四种 sensing level：

```text
S0  Vision / object state only
S1  + Binary contact
S2  + Normal tactile
S3  + 3-axis tactile
```

Genesis 当前已经提供 ContactProbe、KinematicTaxel 等接口，可直接得到 binary contact、per-taxel force/torque，并加入 hysteresis、crosstalk、dead taxel 等硬件式扰动。

---

# 14. Phase II-B：训练 Interaction Estimator

## 输入

第一版不从 raw RGB 学视觉，以免把视觉算法问题混进来。

直接输入：

- object pose / relative pose；
- tactile taxel measurements；
- 每个 taxel 的 3D position 和 local normal。

## 网络

把每个 taxel 当成一个空间点：

```text
[x, y, z, nx, ny, nz, tactile channels]
```

采用：

```text
Taxel set
  ↓
PointNet-style encoder
  ↓
tactile feature
  +
object state
  ↓
2-layer GRU, hidden=128
  ↓
Future Functional Interaction
```

## 输出

输出严格对应 Phase I 最终保留下来的字段，不另外创造新 latent。

例如：

- future object state：regression；
- region：classification / surface mask；
- contact mode：classification；
- wrench：regression。

## Loss

使用简单的多任务监督：

- pose / wrench：Smooth L1；
- region / mode：cross-entropy；
- 各项 loss 做量级归一化后加权。

所有标签都来自 simulator oracle，因此这一阶段是 supervised learning，不需要再用 RL。

---

# 15. Phase II-C：把 sensing requirement 变成硬件指标

这一阶段不是无限扫参数，只扫真正影响硬件设计的四个维度。

## 1. Tactile modality

比较：

```text
binary
normal-only
3-axis force
```

如果 normal-only 已经能恢复足够好的 interaction，就不做全手 3-axis。

## 2. Coverage

先比较：

```text
fingertip-only
fingers + palm
whole interaction surface
```

Tactile Genesis 的近期结果显示，在其 dexterous tasks 中 coverage 对性能的影响明显大于继续堆高 taxel resolution；同时 per-taxel force/torque 是较稳健的 tactile representation。因此本项目把“coverage 是否更重要”当作实验先验，而不是直接当作结论。

## 3. Resolution / frequency

从较低到较高逐级提高，直到 downstream task success 基本饱和。

硬件规格以“继续提高已经不能显著改善任务成功率”为截止点，而不是追求传感器能做到的极限。

## 4. Surface-pose error

对 taxel world pose 人工加入位置和角度误差，例如：

```text
position: 0 / 1 / 2 / 5 mm
orientation: 0 / 1 / 2 / 5 deg
```

完整运行：

```text
noisy sensor
→ Interaction Estimator
→ Phase I Executor
→ task success
```

由任务成功率下降的拐点反推 surface-pose sensor 所需精度。

---

# 16. Phase II 的决策输出

最后必须明确形成这样的工程结论：

```text
需要 normal 还是 3-axis？
需要覆盖哪些手部区域？
需要多少 taxel？
最低采样频率？
允许多大 tactile noise？
surface pose 至少多准？
```

这时才真正冻结 Human Interaction Acquisition 的硬件指标。

---

# 17. 真实触觉硬件的推荐路线

最终方案由 Phase II 决定，但实验计划需要预留可落地的硬件路径。

## 如果 normal-only 已经足够

优先采用 **高覆盖柔性电容式 tactile skin**。

理由：

- 能持续测静态接触；
- 适合柔性、大面积覆盖；
- 易做多 taxel array；
- DexSkin 已经证明高覆盖 conformable capacitive skin 可以用于 contact-rich manipulation，并且可做跨传感器实例标定。

## 如果 3-axis information 确实必要

不直接跳到“全手每一点都是三轴”。

优先测试：

> **dense normal skin + sparse 3-axis nodes**

即全手/主要交互区域保持高覆盖 normal sensing，只在仿真证明 shear 最有价值的部位布置三轴节点。

如果最终必须发展柔性三轴单元，可参考 iontronic / capacitive multi-axis tactile 结构；已有工作证明软性三轴 iontronic sensor 可以解耦 normal 与 omnidirectional shear，但这种方案应在 Phase II 证明必要后再投入制造。

---

# 18. Phase III：真实 Human Interaction Acquisition

## 18.1 第一版真实系统目标

不是立刻做大规模 human foundation dataset，而是验证：

> **真实人类传感数据能否恢复 Phase I 定义的 interaction。**

第一轮 5–10 名被试、3 个核心任务即可。

每个任务要求故意改变操作策略：

- 不同抓法；
- 不同接触位置；
- 不同速度/力度。

目的是产生真正的 inter-human variation，而不是重复同一种动作。

---

## 18.2 真实传感数据

Human-side 只采 Phase II 已证明必要的模态：

```text
RGB-D / object tracking
+
tactile
+
tactile surface pose
```

开发阶段允许使用外部 ground truth 系统做标定和监督，但最终运行时不依赖它。

每个 tactile sample 同时存：

```text
raw tactile
calibrated tactile quantity
sensor local pose
sensor world/object pose
```

这样未来换传感器时，Functional Interaction 层不会和某一种 raw sensor channel 绑定。

---

# 19. Phase III-B：真实 Interaction Model

继续使用 Phase II 的网络结构，而不是另起一套系统：

```text
object state / visual feature
+
spatially registered tactile point set
          ↓
PointNet + GRU
          ↓
Future Functional Interaction
```

训练顺序：

1. simulation 数据预训练；
2. real human data fine-tune；
3. 测试未见过的人、不同动作策略、不同物体参数。

如果 sim-to-real gap 很大，再逐步增加 sensor-domain randomization，而不是一开始上复杂 domain adaptation。

---

# 20. Phase IV：完整 Human → Multiple Robots

最后才验证 PPT 中的完整链路：

```text
Human Demo
   ↓
Human Interaction Acquisition
   ↓
Interaction Estimator
   ↓
Functional Interaction
   ↓
Robot A Executor
Robot B Executor
Robot C Executor
```

机器人之间不共享 human joint trajectory。

第一版也不要求所有机器人使用同一组 executor 权重。

核心 claim 是：

> **同一份 Functional Interaction 可以由不同 embodiment-specific executors 通过不同动作实现。**

只有这一点成立后，再研究一个 morphology-conditioned generic executor 是否能对 held-out embodiment 泛化。

---

# 21. 最终 baseline 与核心对照

完整实验不需要堆大量 baseline，保留三类足够：

```text
Action-level transfer
Effect-only transfer
Functional Interaction transfer
```

其中 Functional Interaction 内部再使用 Phase I 的 R0–R3 ablation。

最终真正需要回答的是：

1. Effect-only 在哪些任务已经足够？
2. 哪些 contact-rich tasks 必须加入 interaction region / mode / mechanics？
3. 这些额外信息是否在不同 embodiment 上仍有一致价值？
4. 现实 human sensing 是否能以足够精度恢复它们？

---

# 22. Go / No-Go 节点

## Gate 1：Expert 是否可靠

不可靠 → 不进入 representation study。

## Gate 2：Interaction 是否真的提供额外价值

如果 R3 相比 Effect-only 在 contact-rich tasks 上没有稳定提升 → 暂停硬件开发，重新审视核心 hypothesis。

## Gate 3：Cross-embodiment 是否成立

如果 interaction 只能在 source morphology 上有效 → 说明表示仍携带过多 embodiment-specific 信息，需要重新设计 representation。

## Gate 4：传感器是否可观测

如果 Phase I 需要的信息在 realistic sensing 下无法稳定恢复 → 必须降低表示复杂度或重新设计 human acquisition hardware。

通过 Gate 1–4 后，才值得做大规模 human data collection。

---

# 23. 建议的实际研发顺序

## Sprint 1：把一个任务跑通

- Genesis 建 drawer；
- 一个 gripper；
- 读 object state/contact force；
- scripted / IK 接近；
- PPO manipulation expert；
- 保存成功轨迹。

目标：证明整个训练和数据记录链可以运行。

## Sprint 2：第一个 representation 实验

只做 drawer：

```text
Effect-only
vs.
Effect + Region + Mode + Mechanics
```

训练两个相同架构的 executor。

如果 Full Interaction 连 drawer 都没有价值，先不要扩展任务。

## Sprint 3：加入 Wiping + 第二种 embodiment

这里第一次重点观察 shear/contact mode 是否产生稳定收益，并开始 cross-embodiment test。

## Sprint 4：加入 Insertion，完成 R0–R3 系统 ablation

此时才形成 Phase I 的正式结论。

## Sprint 5：Sensor Study

根据 Phase I 最终表示，开始 binary / normal / 3-axis、coverage、surface-pose error 实验。

## Sprint 6：冻结硬件指标并进入真实人类 acquisition

不提前造过度复杂的 glove。

---

# 24. 这套方案最终验证的是什么

它不是一次性验证一个超大黑盒：

```text
Human big data
→ latent model
→ universal executor
```

而是按因果依赖拆开验证：

```text
第一步：任务到底需要什么 interaction information？

第二步：这种 information 能否被人侧传感系统可靠观测？

第三步：同一份 information 能否被不同机器人实现？

第四步：把真实 human acquisition 与 multi-embodiment execution 串起来。
```

因此每一步失败都能明确定位原因，也能在投入复杂硬件和大规模数据之前做 Go / No-Go 决策。

---

# 25. 关键参考与工具依据

1. **Tactile Genesis: Exploring Tactile Sensors at Scale for Learning Dexterous Tasks**, arXiv:2606.22332, 2026.  
   用于支持：统一 tactile abstraction、sensor placement / resolution / noise ablation、teacher–student tactile learning 的可行性。

2. **Genesis World tactile sensor documentation**, 2026.  
   当前已提供 `ContactProbe`、`KinematicTaxel`、`ElastomerTaxel`、`ProximityTaxel` 等接口；支持 per-taxel force/torque、hysteresis、crosstalk、dead taxel 等模拟。

3. **Newton Physics hydroelastic contact / SensorContact documentation**, 2026.  
   用于关键 contact-physics cross-check；支持 distributed contact area、total/friction force、force-weighted contact position。

4. **DexSkin: High-Coverage Conformable Robotic Skin for Learning Contact-Rich Manipulation**, CoRL 2025.  
   用于真实硬件 normal/high-coverage tactile skin 的参考路线。

5. **Hierarchically-Interlocked, Three-Axis Soft Iontronic Sensor for Omnidirectional Shear and Normal Forces**, Advanced Materials Technologies, DOI: 10.1002/admt.202401626.  
   用于三轴柔性 tactile 的可制造参考；只在 Phase II 证明 shear/3D force 必要后考虑。

---

## 最终一句话

> **先用仿真严格确定“任务真正需要什么 interaction 信息”，再把这个信息需求翻译成人侧传感器指标，最后才做真实人类采集与跨机器人执行。**

这条顺序是整个项目保持可验证、可收敛、不过早陷入硬件复杂度的核心。
