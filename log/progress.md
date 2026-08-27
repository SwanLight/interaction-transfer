# 进度

> 本文件已按 2026-08-27 的修订同步（`decisions.md` D-10 ~ D-19）。旧版的 `R0/R3` 命名、32-patch 分辨率、25pt 硬阈值均已作废。

## 环境（2026-08-27 实测，中途不升级）

服务器 `root@10.0.6.98`（LXC 容器，PID1 = supervisord，主机名 `kaic-bb9c28fb-…-0`）。免密登录已配好（本地 `~/.ssh/id_rsa`，需 `ssh-add --apple-use-keychain`）。

| 项 | 值 |
|---|---|
| OS / 内核 | Ubuntu 24.04.2 LTS / 5.15.0-60-generic |
| **Isaac Sim** | **5.1.0-rc.19+release.26219.9c81211b.gl** ⚠️ 是 RC 不是正式版 |
| **Isaac Lab** | **2.3.1**（isaaclab 0.48.0 / tasks 0.11.6 / assets 0.2.3 / rl 0.4.4 / mimic 1.0.15）位于 `/workspace/isaaclab`，**editable 安装** |
| Isaac Lab commit | ❌ **无 .git，无法记录 commit**，见下方风险 R-2 |
| Python | 3.11.13（`/isaac-sim/kit/python/bin/python3`，经 `/isaac-sim/python.sh` 调用）；系统 python 是 3.12.3，**不要用它** |
| conda | 无 |
| PyTorch | 2.7.0+cu128 |
| RL 框架 | **rsl-rl-lib 3.0.1** ✅、rl_games 1.6.1、skrl 1.4.3、stable_baselines3 2.7.0 |
| GPU | 8 × RTX 4090（24564 MiB each），驱动 550.54.14 |
| CPU / 内存 | 128 核 Xeon Platinum 8358P @2.60GHz / 376 GB |
| 磁盘 | `/` overlay 14T，已用 907G，**可用 13T** |
| ffmpeg | 6.1.1-3ubuntu5 ✅ |
| nvcc | ❌ 未安装（torch 自带 runtime，仅在需要编译自定义 CUDA kernel 时才是问题） |
| 外网 | ✅ S3 资产源和 pypi 均可达 |
| **Isaac Lab 快照 commit** | **`2ab57ade`**（= upstream `v2.3.1 / 5c2ec81` + `assets.py` 补丁），见 D-22 |
| 项目 git commit（起点） | `2d454a2`（首次提交，2026-08-28） |

### 冒烟测试

`/isaac-sim/python.sh /tmp/isaac_smoke.py` → **EXIT=0**，16.5 s 完成启动→关闭。GPU/驱动/渲染栈正常。

headless 下 `GLFW initialization failed` / `failed to open the default display` / `carb.audio eDeviceLost` 是**正常噪音**，可忽略。

---

## ⚠️ 开工前必须处理的风险

### ✅ R-1 已解决（2026-08-27）

`/mnt/isaacsim_assets` 已填充：**308.1 MB / 297 文件 / 0 失败**。下载脚本留在服务器 `/tmp/fetch_assets.py`，进代码库时移到 `tools/fetch_assets.py`。

已拉取的前缀：

```
Isaac/Robots/WonikRobotics/AllegroHand/     15 文件    4.8 MB
Isaac/Robots/FrankaRobotics/FrankaPanda/    75 文件   37.3 MB
Isaac/IsaacLab/Robots/FrankaEmika/          57 文件   10.4 MB
Isaac/Props/Sektion_Cabinet/                18 文件    0.4 MB
Isaac/Props/Mounts/SeattleLabTable/         32 文件   73.3 MB
Isaac/Props/Blocks/                        100 文件  181.8 MB
```

只拉了用得到的（整树 >26.8 GB）。因为 D-07 规定新物件全用参数化 primitive，真正需要的内置资产就这几个。缺了再补拉。

### ✅ R-4 已解决（2026-08-27）

接触传感器在 Isaac Lab 2.3.1 上**完全可用**，但需要绕过包装层。三轮验证脚本：`/tmp/verify_contact.py`（v1）、`/tmp/verify2.py`（v2，定位根因）、`/tmp/verify3.py`（v3，最终确认）。

**最终结论**（v3，mass=0.5 kg，μ=0.4，μmg=1.962 N）：

| 施加力 | 摩擦力(单 env) | 速度 | 模式 |
|---:|---:|---:|---|
| 1.0 N | 1.000 N | ~0 | STICK |
| 1.5 N | 1.500 N | ~0 | STICK |
| 2.5 N | **1.962 N** | 0.54 m/s | SLIDE |
| 3.0 N | **1.962 N** | 1.04 m/s | SLIDE |

`counts=[4,4]`、`contact_pos_w` 无 NaN、`force_matrix_w=[0,0,4.905]`、points/normals 全部正常。**`plan/02` 的 Contact Mode 和 Interaction Region 两个字段均可实现。**

三个前提条件（缺一不可，已写进 `plan/01` §1 规则 7、8）：

1. filter 目标必须是 **rigid body**（不动的用 `kinematic_enabled=True`），静态碰撞体不行 → P-17
2. `filter_prim_paths_expr` 非空 + `max_contact_data_count_per_prim ≥ 1`
3. 逐点 buffer 按 env 打包，必须用 `counts`/`start_idx` 切片 → P-18

摩擦力走 `contact_physx_view.get_friction_data(dt)`，不升级 Isaac Lab，理由见 `decisions.md` **D-20**。

### ~~R-1 · 资产目录是空的~~（已解决，保留原始记录）

`/workspace/isaaclab/source/isaaclab/isaaclab/utils/assets.py:27` 把资产根**硬编码**为本地路径：

```python
# NUCLEUS_ASSET_ROOT_DIR = carb.settings.get_settings().get("/persistent/isaac/asset_root/cloud")
NUCLEUS_ASSET_ROOT_DIR = "/mnt/isaacsim_assets/Assets/Isaac/5.1"
```

而 `/mnt/isaacsim_assets` **是空的**。所有 `ISAAC_NUCLEUS_DIR` 引用都会解析失败。

云端 5.1 资产树存在且可达，本计划需要的三个资产已实测 200：

| 资产 | 路径 |
|---|---|
| Allegro | `Isaac/Robots/WonikRobotics/AllegroHand/allegro_hand_instanceable.usd` |
| Franka | `Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd` |
| Sektion 柜 | `Isaac/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd` |

整树 >26.8 GB / >40k 文件（S3 分页统计到 40 页仍未结束）。磁盘 13T 完全够。

**处理方案未定，见下方"待决"。**

### ✅ R-2 已解决（2026-08-27）

对 `/workspace/isaaclab` **原地** `git init` 做快照，不重新 clone、不改动任何文件。见 `decisions.md` **D-22**。

```
本地快照   2ab57ade5f9647e22503ce676553b6941187467d
upstream   v2.3.1 = 5c2ec81   (remote 已加，可随时 diff)
```

与 upstream 的实质差异**只有一处**——`source/isaaclab/isaaclab/utils/assets.py` (+3 -1)，即资产根路径补丁。其余 1155 个"改动"文件全是发行包裁掉的 `.github/` CI 配置、`.vscode` 模板和 `isaaclab.bat` 换行符差异。**确认没有其他隐藏的本地修改。**

复查命令：

```bash
ssh root@10.0.6.98 'cd /workspace/isaaclab && git diff --numstat upstream-v2.3.1 HEAD | awk "\$1+\$2>0"'
```

### ~~R-2 · Isaac Lab 不是 git 仓库~~（已解决，保留原始记录）

### R-3 · Isaac Sim 是 RC 版本（5.1.0-**rc.19**）

不是正式发布版。RC 可能有已知未修的 bug，且未来若被替换成正式版，行为可能变化。至少要把当前版本号锁死并在论文中如实写明。

### ~~R-4 · pitfalls P-01 ~ P-08 是针对旧版 Isaac Lab 写的~~（已解决，见上方）

复核结论已写进 `pitfalls.md` 顶部对照表 + 新增 P-16/P-17/P-18。P-06 ~ P-08（录像相关）仍未复核，留到 S0。

### R-5 · 待办：把验证脚本收进代码库

以下脚本目前只在服务器 `/tmp` 下，重启即丢：

| 服务器路径 | 应移到 |
|---|---|
| `/tmp/fetch_assets.py` | `tools/fetch_assets.py` |
| `/tmp/verify3.py` | `tools/check_contact_sensor.py` |

另需新写一个逐点接触数据的切片工具（约 30 行，按 `counts`/`start_idx` 拆 env），产出 `plan/02` §3.2 需要的格式。见 P-18。

---

## 步骤状态

| 步骤 | 内容 | 状态 | 完成日期 |
|---|---|---|---|
| S0 | 可视化链路 | ⬜ 未开始 | |
| S1 | 资产自检 | ⬜ 未开始 | |
| S2 | 场景组装 + 执行器接入 + Expert 上限 | ⬜ 未开始 | |
| S3 | Source 脚本 + 数据集 + 预训练物体集 | ⬜ 未开始 | |
| S4 | Interaction 提取 + 泄漏检查 | ⬜ 未开始 | |
| S4.5 | 分辨率敏感度扫描 | ⬜ 未开始 | |
| S5 | Shared Structure Model | ⬜ 未开始 | |
| S6 | **E-I 交互跟踪执行器** | ⬜ 未开始 | |
| S7 | **E-I 留出任务零样本（主闸门）** | ⬜ 未开始 | |
| S8 | E-T 信息条件对照 C0–C5 | ⬜ 未开始 | |
| S9 | 扰动恢复与鲁棒性 | ⬜ 未开始 | |
| S10 | Cross-embodiment + 反事实 + 报告 | ⬜ 未开始 | |

状态图例：⬜ 未开始 / 🟡 进行中 / ✅ 通过 / ❌ 受阻

**任务推进顺序：抽屉（打通管线）→ 擦拭（主任务）→ 旋钮（支撑）。** 不要在抽屉上做第一个表示实验——它是预期阴性对照。

---

## S0 — 可视化链路

判据见 `../plan/06-eval-and-visualization.md` §3。

- [ ] WebRTC 串流可用
- [ ] 最小 demo 能录 mp4
- [ ] report.html 骨架可生成
- [ ] `--enable_cameras` 不卡死（训练时不录像，见 `pitfalls.md` P-07）

记录：

---

## S1 — 资产自检

判据见 `../plan/01-assets-and-scenes.md` §7。

| 资产 | 自检脚本 | 状态 | 关键实测值 |
|---|---|---|---|
| 抽屉柜 | `check_cabinet.py` | ⬜ | |
| 平面 + dirt grid | `check_board.py` | ⬜ | |
| 旋钮 | `check_knob.py` | ⬜ | 轮缘/销钉摩擦标定：_待填_ |
| 预训练物体集（4 类） | `check_pretrain_objs.py` | ⬜ | |
| 采集板 | `check_plates.py` | ⬜ | PD 增益：_待填_ |
| 钩杆 | `check_hook.py` | ⬜ | |
| **垫头杆** | `check_padrod.py` | ⬜ | |
| Franka hand | — | ⬜ | 接触力可读？ |
| Allegro | — | ⬜ | 接触力可读？ |

关键自检项：

- [ ] **轮缘低摩擦标定通过**（安全力上限内纯轮缘无法达成目标转角，销钉可以）——D-14 的前提，不通过则旋钮无法检验 region
- [ ] **物体尺寸对所有执行器一致**（D-11，禁止单独 scale Allegro）
- [ ] 垫头杆能在平面上建立稳定面接触，3–8 N 下平稳滑移

记录：

---

## S2 — 场景组装 + Expert

- [ ] 四宫格视频能出来
- [ ] 物体不飞、不抖
- [ ] 接触力读数非零（`pitfalls.md` P-01/P-02）

Expert 上限（Gate A：固定 ≥95%，随机 ≥85%），9 个组合：

| 任务 | 执行器 | 固定 | 随机 | 通过 |
|---|---|---|---|---|
| 抽屉 | Allegro / 钩杆 / 夹爪 | | | |
| 旋钮 | Allegro / 钩杆 | | | |
| 擦拭 | Allegro(工具) / Allegro(直擦) / 夹爪 / 垫头杆 | | | |

记录：

---

## S3 — Source 数据集

判据见 `../plan/03-source-and-dataset.md` §4、§6、§7。

| 项 | 目标 | 实际 |
|---|---|---|
| 旋钮成功轨迹 | ≥1000（≥4 策略家族） | |
| 抽屉成功轨迹 | ≥450（≥3 策略家族） | |
| **擦拭成功轨迹** | **≥1250（≥5 家族，含直擦）** | |
| **预训练物体集** | **≥2500** | |
| 一致性校验标记率 | <15% | |
| 人工看过的轨迹视频 | ≥10 条 | |

- [ ] 策略分类器在**原始 source 动作**上准确率显著高于随机（多样性验收）
- [ ] 校准集已独立划出（D-17，不得与训练集/确认集重叠）
- [ ] 划分按 **episode** 而非帧（`pitfalls.md` P-10）

记录：

---

## S4 — Interaction 提取

`../plan/02-interaction-spec.md` §7 泄漏检查清单（9 条）：

- [ ] 1. 场景刚体旋转后物体系表示逐元素一致
- [ ] 2. 无 source root state
- [ ] 3. 改变 source 板数量后表示维度不变
- [ ] 4. 策略分类器难以从 envelope 识别策略
- [ ] 5. target 看不到 source 身份
- [ ] 6. 物理参数不直接写入表示
- [ ] 7. 无执行器专属 joint / contact 编号
- [ ] 8. **擦拭 envelope 与"是否用工具"无关**（两种实现的 envelope 可互换）
- [ ] 9. **region 可推导性探针**：只看 effect + 点云预测 region 的准确率显著低于上限

第 8、9 条为本版本新增。第 9 条若在某任务上不通过，如实报告该任务不适合检验 region，**不得调参掩盖**。

记录：

---

## S4.5 — 分辨率敏感度扫描

`../plan/05` 实验零 §1.1。擦拭 × Allegro × C4，3 seed。

| 表面点数 | 近似 pitch | 成功率 | 训练耗时 |
|---:|---:|---|---|
| 64 | ~15 mm | | |
| 256 | ~7.8 mm | | |
| 1024 | ~3.9 mm | | |
| 4096 | ~2.0 mm | | |

- [ ] 饱和点已确定：_待填_
- [ ] 该结论已同步写入 `../plan/02` §2 和 `../plan/07` §1
- [ ] 若需高分辨率，已实现"低分辨率全局 + 接触邻域局部"（`plan/02` §2.3）

记录：

---

## S5 — Shared Structure Model

判据见 `../plan/03` §9。

| 项 | 目标 | 实测 |
|---|---|---|
| coverage | ≥90% | |
| width | ≤ oracle 的 1.5× | |
| 冻结 executor 成功率下降 | ≤10 pt | |
| 策略身份 probe 准确率 | 显著低于原始动作 | |
| 擦拭跨实现互换后下降 | ≤10 pt | |

记录：

---

## S6 — E-I 交互跟踪执行器 ⚠️

**本轮最关键的一步。** `../plan/04` §5。

| 执行器 | 训练指令来源 | 留出任务 | 跟踪成功率 | 评价器 AUC | 状态 |
|---|---|---|---|---|---|
| Allegro | 预训练集 + 抽屉 + 旋钮 | 擦拭（两种实现） | | | ⬜ |
| **垫头杆** | **仅预训练集** | 擦拭 | | | ⬜ |
| 平行夹爪 | 预训练集 + 抽屉 | 擦拭 | | | ⬜ |
| 钩杆 | 仅预训练集 | 抽屉 + 旋钮 | | | ⬜ |

通过条件：训练指令上跟踪成功率 ≥80%，可行性评价器 AUC ≥0.85。

- [ ] **E-I reward 中不含任务 success bonus / dirt 清除量 / 任务关节目标**（`plan/04` §5.1 禁止清单）
- [ ] **dataloader 断言已生效**：留出任务在 A–D 任何阶段都不出现（不靠自觉，靠断言日志）

记录：

---

## S7 — 留出任务零样本（主闸门）⭐

`../plan/05` 实验二，Gate E。

| 执行器 | 留出任务 | 零样本成功率 | 有评价器 | 无评价器 | 通过 |
|---|---|---|---|---|---|
| Allegro | 擦拭(工具) | | | | |
| Allegro | 擦拭(直擦) | | | | |
| 垫头杆 | 擦拭 | | | | |
| 夹爪 | 擦拭 | | | | |
| 钩杆 | 抽屉 | | | | |
| 钩杆 | 旋钮 | | | | |

**Gate E**：至少两个执行器零样本 ≥60%，且其中至少一个是垫头杆/钩杆这类 0 自由度形态。

若训练任务好、留出任务差 → 扩预训练物体集（`plan/03` §2.4），**不得把留出任务加进训练**。扩两轮仍不成立则如实报告未通过。

记录：

---

## S8 — E-T 信息条件对照

`../plan/04` §11 分层，合计约 147 次训练（含 S2/S4.5/S6）。

### Tier 1（不可裁）：擦拭 × {Allegro, 垫头杆} × C0–C5 × 5 seed = 60

| 执行器 | C0 | C1 | C2 | C3 | C4 | C5 |
|---|---|---|---|---|---|---|
| Allegro | | | | | | |
| 垫头杆 | | | | | | |

### Tier 1s：稀疏 reward × {C0, C2, C4} × 3 seed = 18

| 执行器 | C0 | C2 | C4 |
|---|---|---|---|
| Allegro | | | |
| 垫头杆 | | | |

### Tier 2：旋钮 × {Allegro, 钩杆} × {C0, C2, C4} × 3 seed = 18

### Tier 3：抽屉 × {Allegro, 钩杆, 夹爪} × {C0, C4} × 3 seed = 18（阴性对照）

进度紧张时的裁剪顺序：Tier 3 降到 1 seed → Tier 2 砍 C2 → **Tier 1 与 E-I 不可裁**。

---

## S9 — 扰动恢复

`../plan/05` 实验四。**纯评估，用 S8 的 checkpoint，不需新训练。**

| 扰动 | C0 恢复率 | C2 | C4 | 纠正方向正确率 (C0/C4) |
|---|---|---|---|---|
| 摩擦 ×0.5 | | | | |
| 摩擦 ×2 | | | | |
| 质量/阻尼突变 | | | | |
| 外部冲量 | | | | |
| 诱导滑移 | | | | |

**Gate D'**：C4−C0 的恢复率差距显著大于无扰动时的成功率差距。

不通过不阻断后续，但削弱硬件必要性论证，须写进论文限制一节。

---

## S10 — 评估与报告

| 闸门 | 判据 | 实测 | 结论 |
|---|---|---|---|
| Gate A | Expert 固定≥95% / 随机≥85% | | |
| Gate B | C4 下可行组合 ≥80% | | |
| Gate C | region/mode/mechanics 各有一个反事实产生物理一致的特定失败 | | |
| Gate D | **C1→C2 或 C2→C4** 有跨 seed 一致的可信提升 | | |
| Gate D' | 扰动下 C4−C0 差距放大 | | |
| **Gate E** | **≥2 执行器留出任务零样本 ≥60%，含 1 个 0-DoF 形态** | | |
| Gate F | ≥3 种形态实现同一未改写 envelope | | |
| Gate G | 抽屉阴性结果可接受（不修改任务追求差距） | | |

**统计口径**（`plan/05` §9）：Tier 1 报 5 seed 均值 + 标准差 + bootstrap 置信区间；Tier 2/3 报 3 seed。**不以任意百分点硬阈值作为唯一判据**，而看置信区间、跨 seed 一致性和物理解释。

人工检查清单（`../plan/06` §7）每次 eval 都要走一遍：

- [ ] 看过 ≥5 条成功视频、≥10 条失败视频
- [ ] 看过四宫格，且四格 envelope 文件 hash 一致
- [ ] 确认 counterfactual 只改了指定字段，失败来自物理而非脚本崩
- [ ] 确认不同信息条件用同一批 episode
- [ ] **确认 E-I 留出任务未出现在训练数据里（查断言日志）**
- [ ] 确认没有根据确认集结果继续调场景

记录：
