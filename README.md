# Functional Interaction Transfer

不迁移人的动作，而是迁移动作所实现的**功能性交互**，再由不同末端执行器根据自身形态各自重新实现。

```text
Human demonstration → Functional Interaction → Embodiment Executor → Robot Action
```

本仓库是**仿真算法验证阶段**。真实触觉硬件、surface-pose 硬件和 sim-to-real 不在本轮范围内。

## 目录

| 目录 | 内容 |
|---|---|
| `HANDOFF.md` | **接手先读**：当前状态、服务器用法、已知的坑 |
| `log/takeover-audit-2026-08-30.md` | Codex 接管审计：idea 对齐、证据边界、S5 前阻塞项与文献复核 |
| `plan/` | 实验计划。**唯一权威**，按 `00` → `07` 顺序读 |
| `log/` | 工程日志：进度、踩坑、技术决策 |
| `src/it/` | 资产生成、控制器、接触工具、任务环境 |
| `tools/` | 环境工具、自检、录像、训练脚本 |
| `out/` | **产物**：视频与 HTML 报告，双击 `out/report.html` 就能看。S5 的全量结果在 `out/s5/README.md` |
| `idea/` | 原始 idea 与 PPT。**已被 `plan/` 取代**，仅作背景保留 |

## 从哪开始读

> **接手这个项目？先读 [`HANDOFF.md`](HANDOFF.md)** —— 当前进度、
> 服务器怎么用、哪些坑会浪费你半天，都在那里。

随后读 [`log/takeover-audit-2026-08-30.md`](log/takeover-audit-2026-08-30.md)：它记录
本轮接管时对原始 idea、现有证据和 S5 架构契约的完整复核。

1. [`plan/00-positioning.md`](plan/00-positioning.md) — 定位、相对 KITE/ART-Glove/CHORD 的差异、可声称的边界
2. [`plan/README.md`](plan/README.md) — 系统结构、两个执行器、六个信息条件、步骤与闸门
3. [`log/decisions.md`](log/decisions.md) — append-only 技术决策及其被否决的备选

## 核心系统与本轮验证

> 多位采集者为同一任务产生不同示教，上游把它们形成物体中心 interaction；每种
> embodiment 各自训练 decoder/executor，把同一 interaction 解码成自己的 action，
> 再由触觉或其他反馈闭环稳定执行。

本轮先验证 interaction 是否能跨示教者、跨执行器传递，以及 richer physical
interaction 是否优于 effect-only / KITE-style geometry-only。留出任务 Gate E 是对
executor 泛化能力的强测试，不是这套传递协议本身的定义。

明确**不**声称的内容见 `plan/00-positioning.md` §4——尤其：不把上游的新任务生成
能力混进传递协议，也不声称零样本控制未训练过的新形态。

## 环境

服务器 `root@10.0.6.98`，8 × RTX 4090。完整环境表和已知风险见 [`log/progress.md`](log/progress.md)。

| 项 | 版本 |
|---|---|
| Isaac Sim | 5.1.0-rc.19 |
| Isaac Lab | 2.3.1 = upstream `v2.3.1 (5c2ec81)` + 一处 `assets.py` 补丁；快照 commit `2ab57ade` |
| PyTorch | 2.7.0+cu128 |
| RL | rsl-rl-lib 3.0.1 |

版本锁定策略见 `log/decisions.md` **D-01 / D-20 / D-21**：**S2 开始训练后环境冻结**，此前允许修复。

## tools/

| 脚本 | 用途 |
|---|---|
| `fetch_assets.py` | 从 Omniverse S3 按前缀拉取 Isaac 资产到 `/mnt/isaacsim_assets` |
| `check_contact_sensor.py` | 接触传感器验证。**改动接触相关代码后必跑** |
| `contact_utils.py` | 逐接触点数据提取、stick/slide 判定、物体系转换、表面热力图 |

`src/it/contact_utils.py` 是每个环境都要用的基础件。它存在的原因见 `log/pitfalls.md` **P-16/P-17/P-18**——当前 Isaac Lab 版本的 `ContactSensor` 不暴露摩擦力，底层 buffer 又是跨 env 扁平打包的。

⚠️ **P-30 推翻了 P-18 的切片做法**：`counts`/`start_idx` 的下标不是 env 下标，
总数还会漏报。逐 env 归属改为**按接触点离哪个本体最近**判定，见
`extract_contact_points_padded`。静态自检发现不了这个问题。

⚠️ **P-36**：摩擦 buffer 有自己的 counts/start_idx，行数与接触 buffer 不同。
混用下标会让一半环境的摩擦翻倍、另一半全为零，`plan/02` §3.4 的
Contact Mode 因此整体错误。取摩擦一律走 `extract_contact_points_padded`。

## 最容易让结论作废的三件事

摘自 `log/README.md`，都会让数字变好看：

1. 留出任务不通过后动了训练数据（`pitfalls.md` P-13）
2. 为制造条件间差距而调任务参数或随机化范围（P-11）
3. 跨形态结果里对某个 target 单独改写了 envelope（P-15）

三者都必须记进 `decisions.md`，不得静默处理。
