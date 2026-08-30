# Codex 接管审计 · 2026-08-30

> **后续更正**：本文 §4.1–4.3 曾把 joint components 与 executability evaluator 写成
> 主流程必需算法。用户澄清后，该强制结论已由 `log/decisions.md` D-63 撤销。仍有效的
> 是接口泄漏、成功数据不可证明必要充分性、KITE/CHORD 事实核对与传感边界等审计；
> S5 当前实现以直接的多示教统计传递为基线，不做显式“求交”。

## 结论先行

项目**没有被改成与原始 idea 无关的东西**。从原始 PPT/PDF/长文到当前 plan，主线仍是：

> 不迁移人的动作；从多种成功实现中提取物体中心的功能交互要求，再由不同形态根据
> 自身能力产生不同动作，实现相同功能结果。

现有 S0–S4 工作不是样子货：资产、接触数据、坐标不变性、source-action 泄漏、接触
模式、统一 effect 接口和探针原语数据都有代码、测试与 `out/` 证据。管理制度也总体
有效：plan 写意图、log 写事实、决策 append-only、每个数字落 artifact、留出失败后
不得补该任务数据，这些规则应保留。

但 **S5 之前仍有四个会使主张在形式上失效的架构漏洞**：task-specific mechanics
进入任务无关执行器、分字段边缘 envelope 可拼出物理非法组合、成功示教被过度解释成
必要充分约束、probe 的边缘覆盖被误写成张满交互空间。另有三处相关工作事实错误与
一处 sensor-sufficiency 越界。这些已按 D-57～D-62 修正；在新 contract 冻结前不应写
Shared Structure Model 或 E-I 网络。

## 1. 本次实际阅读与核验范围

完整阅读/检查：

- `README.md`、`HANDOFF.md`；
- `idea/` 的原始长文、11 页 PDF、10 页 PPT（逐页渲染检查）；
- `plan/README.md` 与 `plan/00`～`plan/07`；
- `log/decisions.md` D-01～D-56、`log/pitfalls.md` P-01～P-56、progress；
- `src/it/records.py`、`interaction.py`、surface/asset/environment 结构与全部 tools 入口；
- `out/s3_*`、`out/s4_records`、leak/verify/region probe、S4.6 sensing 报告；
- git 历史与当前分支状态。接管前工作树干净，`main` 比 `origin/main` ahead 37。

本地重新执行：

```text
tools/test_surfaces.py     12 tests  OK
tools/test_interaction.py  18 tests  OK
tools/test_records.py       8 tests  OK
python -m compileall src tools       OK
```

`test_surfaces.py` 用时约 105 s，其余约 4 s。上述结果证明本地数据契约与几何/提取单测
当前一致，不证明 S5/S6 算法可行。本次没有登录 8×4090 服务器重新跑 Isaac；远端训练
状态仍以本地同步的 report/artifact 为证，后续开始 S5/S6 前应做一次只读远端盘点。

## 2. 原始 idea 与当前方案的对应

### 保留下来的核心

- object-centric functional interaction，而非 human action/hand skeleton；
- effect、region、engage direction、contact mode、mechanics、temporal evolution；
- 多种 source 策略归纳共享约束；
- embodiment planner/executor 根据自身可行性选择具体实现；
- 同一规格跨显著不同形态实现。

### 有意且合理的工程替换

- Genesis → Isaac Sim / Isaac Lab；
- BC/DAgger 主线 → PPO；
- 原 peg 类任务 → knob，强化 mechanics/lever-arm 可辨识性；
- 固定 patch id → 有坐标/法向/面积的物体表面点云；
- 单一 task-reward executor → E-T（信息消融）与 E-I（任务无关主系统）分离；
- task-native effect → 固定 `effect/rigid + effect/surface_state`。

这些替换没有破坏 idea，反而修复了原方案中“只做任务策略消融、没有真正训练通用
interaction executor”的缺口。

## 3. 已完成工作的可信边界

### S3 数据

- drawer：800 episode / 739 success；
- wipe：2,700 / 2,606；
- knob：2,400 / 1,693；
- probe：10,740 / 9,250；
- 总计 16,640 episode。

十五个 probe primitive 均有至少两个物体提供成功样本。正确表述是“十五原语与五轴
边缘值均被覆盖”，不是“张满整个交互指令空间”。

### S4 Oracle Record v2

- source action 与 measurement pose 已分开审计；loader fail-closed；
- source body 数量被合并，输出维度不携带 plate/tool identity；
- object-frame contact、mode、force/wrench、effect 已落盘；
- `effect/rigid` 与 `effect/surface_state` 是固定 E-I effect contract；
- pose-difference mode 修复后 drawer/knob/wipe 的主要 mode 更符合任务物理；
- leak report 中 S5 才能完成的 strategy/envelope 检查仍明确标 DEFER，没有伪装 PASS。

### S4.6

它给的是**仿真观测模型下的字段可观测性**：位姿误差时间结构、taxel pitch、binary /
normal / shear 对 region/mode/mechanics 重建的影响。它还没有 executor checkpoint，不能
证明控制充分性；更不能证明真人装置可用。D-62 已收紧措辞。

## 4. S5 前必须冻结的算法契约

### 4.1 FunctionalEnvelope v1

E-I 允许的字段必须是固定、无任务分支的 exact allowlist：

```text
effect/rigid
effect/surface_state
surface/{xyz, normal, area, part}
region/{weight, allowed}
engage/direction_cone
mode/distribution
mechanics/{normal_traction_bounds, tangent_direction_cone,
           tangent_traction_bounds}
field masks + horizon validity
```

离散接触力到 traction field 必须用同部件、force-conserving 的归一化邻域核散射；核带宽
只随 surface/sensor pitch，积分合力严格守恒，积分力矩与 raw wrench 做硬阈值对拍。
直接 nearest-point force/area 会随分辨率产生伪尖峰，禁止使用。

禁止：task id/name、任务原生 effect、`mech/generalized`、采集 phase/progress、source
action/identity、source tool pose/geometry。task-relevant generalized force 只在 E-I 输出
以后做离线投影，不能参与 reward。

### 4.2 Envelope 是联合集合

第一版 oracle 不应是多个独立边缘头的笛卡尔积。应表示为少量 joint components 的并集；
每个 component 同时绑定 effect/region/direction/mode/traction。candidate sampling 一次取
完整元组，component id 不给策略。这样既保留多峰，又不把不同成功策略的字段拼成从未
存在的“混血”指令。

### 4.3 可证明的 coverage

- region：校准 90% episode 中至少 95% force-weighted contact mass 被允许区域覆盖；
- mechanics：quantile regression 后做 CQR；
- joint：episode-level nonconformity 检查同一 component 的字段同时满足；
- 明确 exchangeability 与 marginal coverage 边界，另外报告 task/strategy/physics/
  geometry subgroup coverage；
- success-only 输出称 empirical success-support，不称 necessary/sufficient feasible set。

### 4.4 经验可执行性评价器

它预测“当前 E-I 在当前数据分布下是否能跟踪”，不是形态物理能力。除 replay 外要对
边界/低密度 candidate 主动采样；用 ensemble conservative score 选择，并报告 calibration、
selection regret 与下游成功率。AUC 0.85 单独不够。

## 5. 相关工作核验结果

- [KITE](https://arxiv.org/html/2606.22113)：最接近的相邻架构，但不是完全相同系统；
  其 intent 是位置+engage direction，decoder 用无物理的 kinematics-only pairs 训练。
- [CHORD](https://arxiv.org/html/2607.00033)：比较 contact wrench-space support function，
  带 tolerance；不是 exact wrench replay。
- [Transferring Contact, Not Just Motion](https://arxiv.org/html/2606.15516)：共享 MANO
  motion latent + 标定 effort/load 接口，不是 contact-center-only grasp synthesis。
- [OmniContact](https://arxiv.org/html/2606.26201)：已有 contact-flow-conditioned unified
  low-level executor，必须正面对照；它仍含 body targets、binary contact 和手工 phase template。
- [GR00T modality augmentation](https://arxiv.org/html/2512.01358)：74/82/94 是并列
  G1 单任务配置，不是累积链；GR1 另为 51→63。
- [ForceFlow](https://arxiv.org/html/2605.11048)：interaction stage 去视觉后，平面单轴
  pressure 类 Stamp/Clean Whiteboard 仍为 80/90%，复杂几何任务为 0%，支持“模态充分性
  必须按任务几何陈述”，不能外推成全局结论。
- [Huang thesis](https://publications.ri.cmu.edu/robotic-manipulation-primitives)：支持用
  contact-mode lattice 枚举 primitive，但不支持本项目五轴组合的完备性。

## 6. 接管后的执行顺序

1. 先实现并测试 `FunctionalEnvelope` 联合 schema、traction 聚合与 E-I allowlist；
2. 用统计 oracle 在 drawer/wipe/knob 各生成 joint success-support envelope，先做可视化、
   coverage-width、跨策略/实现互换，不训练神经网络；
3. 冻结 oracle 文件 hash，用最小 E-I 单物体单 candidate 验证 task-agnostic reward；
4. 再做 probe 预训练、主动 executability 数据与留出任务；
5. oracle envelope 能驱动 E-I 后，才训练 Shared Structure Model；
6. S6 checkpoint 后补 S4.6 的下游 reconstructed-command 评估；
7. 任何留出失败不得加入留出任务数据；只能按已冻结诊断协议降级主张或记录新决策。

这条顺序把“表示错、统计聚合错、执行器学不会、模型预测错”四类失败拆开，避免一次
同时改多个层导致无法归因。
