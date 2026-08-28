# 踩坑记录

格式：**现象 → 原因 → 解法**。只记现象不记解法等于没记。

下面 P-01 ~ P-10 是**开工前就已知的坑**（来自文档和社区反馈），先写在这里避免重踩。之后按 P-11 往下追加。

---

> ## ⚠️ 2026-08-27 实测复核（Isaac Lab 2.3.1 / Isaac Sim 5.1.0-rc.19）
>
> P-01 ~ P-10 是针对**旧版**写的。已在服务器上跑 `/tmp/verify_contact.py` 逐条复核，结论如下。**以本表为准，正文中失效的条目已就地标注。**
>
> | 条目 | 复核结论 |
> |---|---|
> | P-01 activate_contact_sensors | ✅ **仍适用**。实测 `net_forces_w = [0, 0, 4.9050] N` |
> | P-02 track_friction_forces | ❌ **解法失效**，该配置项在 0.48.0 **不存在**（版本过旧）。现行解法见 **P-16** |
> | P-03 max_contact_data_count 默认 4 | ✅ **仍适用**。内省确认默认值 = 4 |
> | P-04 contact_pos_w 出现 NaN | ⚠️ **归因需修正**：NaN 不是 bug，是"该碰撞对未接触"的**哨兵值**。见下方修订 |
> | P-05 force_matrix_w 返回 None | ✅ **可规避**：设了 `filter_prim_paths_expr` 后正常返回 `shape=(2,1,1,3)` |
> | P-06 ~ P-08 | 未复核（S0 时验） |
> | P-09 力必须有物理意义 | ✅ **实测通过**：`\|F\|/mg = 1.0000` |
> | P-10 按 episode 划分 | 与仿真器无关，仍适用 |
>
> **新增坑：P-16（摩擦力接口）、P-17（filter 目标必须是刚体）、P-18（buffer 按 env 打包）。**
>
> 另外：`activate_contact_sensors=True` **只能加在有 rigid body 的 prim 上**。加在静态碰撞体（只有 `collision_props`、无 `rigid_props`）上会直接抛
> `ValueError: No contact sensors added to the prim`。
>
> 验证脚本留在服务器 `/tmp/verify_contact.py`（v1）、`/tmp/verify3.py`（v3，最终版）。**进代码库时移到 `tools/check_contact_sensor.py`。**

---

## P-01 · 接触传感器全是零

**现象**：`ContactSensor` 读出来的力全是 0，明明物体在接触。

**原因**：PhysX 的 ContactReporter 没激活。

**解法**：在**资产的 spawner**（不是 sensor cfg）上加 `activate_contact_sensors=True`。

---

## P-02 · 判不出 stick / slide ⚠️ 解法已失效，见 P-16

**现象**：切向力永远是 0，无法区分粘滞和滑动。

**原因**：`net_forces_w` **只是法向接触力之和**，不含切向分量。实测确认：静止时 `net_forces_w = [0, 0, 4.9050]`，切向分量恒为 0。

**旧解法（在 0.48.0 上无效）**：~~`ContactSensorCfg(track_friction_forces=True)`~~ —— **该配置项不存在**。

**现行解法**：见 **P-16**。

---

## P-03 · 接触点静默丢失

**现象**：接触点数量明显偏少，一致性校验（`plan/02` §7）残差异常大。

**原因**：`max_contact_data_count_per_prim` 默认只有 **4**，contact-rich 场景必然溢出，而且**不报错**。

**解法**：调到 64。这是首选排查项——一致性校验标记率高时先查这个。

---

## P-04 · contact_pos_w 出现 NaN ⚠️ 归因已修正（2026-08-27 实测）

**现象**：数据链里冒出 NaN，训练直接崩。

**~~旧归因~~**：~~reset 后被填成 `torch.nan`~~ —— 不准确。

**真实原因**：**NaN 是"该碰撞对当前未接触"的哨兵值，是设计如此，不是 bug。** 本地源码 docstring 原文：

```python
contact_pos_w: torch.Tensor | None = None
"""Average of the positions of contact points between sensor body and filter prim in world frame.
   Shape is (N, B, M, 3) ...
   Collision pairs not in contact will result in NaN.
"""
```

任何时刻只要该 (sensor, filter) 对没有接触，就是 NaN，与 reset 无关。

**解法**：

1. **把 NaN 当信息用**：`torch.isnan(contact_pos_w)` 直接就是 `no-contact` 判据，正好对应 `plan/02` §3.4 的第一档，不需要另外设力阈值；
2. 进入任何数值运算前 `torch.nan_to_num()` 或掩码，否则会顺着策略梯度传播；
3. **若观察到"明明在接触却全是 NaN"，先查 P-17**——多半是 filter 目标不是刚体，压根没配上对。

---

## P-05 · force_matrix_w 返回 None

**现象**：想拿"某个 body 对某个物体"的成对力，拿到 `None`。

**原因**：Isaac Lab 目前只支持 **many-to-one** 的接触力过滤；覆盖多个 body 的 sensor 上直接返回 `None`。

**解法**：要么拆成多个单 body sensor，要么改用 `ContactSensor.contact_physx_view.get_contact_data()` 取原始数据自己配对。

---

## P-06 · headless 下没有画面/录不出视频

**现象**：加了 `--video` 但没有 mp4，或者报渲染错误。

**原因**：headless 默认加载 `isaaclab.python.headless.kit`，没有渲染能力。

**解法**：必须同时加 `--enable_cameras`，它才会切到 `isaaclab.python.headless.rendering.kit`。另外确认 `ffmpeg` 已安装。

---

## P-07 · `--headless --video` 卡死

**现象**：训练脚本启动后无输出、无进程退出，尤其在容器里。

**原因**：社区有多起报告，与内部 `enable_cameras` 赋值 / 容器环境有关，未完全定位。

**解法**：**训练时不录像**，用独立的 eval 脚本录（`plan/06` §1.3）。不要在训练脚本里死磕。

---

## P-08 · 吸盘（SurfaceGripper）不能用于 RL 训练

**现象**：用了 Isaac Sim 的 `SurfaceGripper`，只能开很少的并行环境。

**原因**：截至 Isaac Sim 5.0，**Surface Gripper 只支持 CPU 后端**，官方 tutorial 要求 `--device=cpu`。

**解法**：本预演已改用钩杆，不涉及。若将来要做吸盘，自己实现"带断裂阈值的 6-DoF 软约束"（PD 外力 + latch/unlatch），全程 GPU 友好。见 `decisions.md` D-03。

---

## P-09 · 采集板的接触力没有物理意义

**现象**：板能施加任意大的力，Mechanics 字段数值离谱。

**原因**：把板设成了 kinematic body。kinematic body 不参与动力学求解，能施加无限大的力。

**解法**：板必须是**普通刚体 + PD 外力驱动**（`plan/01` §4.1）。自检判据：自由落体后稳态接触力 ≈ mg。

---

## P-10 · 测试成绩虚高

**现象**：test 成功率异常好，但换一批数据就崩。

**原因**：数据集按**帧**划分 train/test，同一条轨迹的相邻帧同时出现在两边。

**解法**：必须按 **episode** 划分（`plan/03` §7）。

---

下面 P-11 ~ P-15 是 2026-08-27 修订时识别出的**方法学坑**，不是工程坑。它们不会让程序崩，只会让结论作废，因此更危险。

---

## P-11 · Region 其实是从 effect 推出来的

**现象**：C1（+区域）相比 C0 毫无提升，或者提升不稳定。

**原因**：对刚性单自由度铰接物体，`effect + 物体几何` 在构造上几乎蕴含 region——销钉在点云里直接可见，转轴由 effect 推出。执行器不需要任何人告诉它去哪接触。这不是"捷径没堵死"，是这类物理系统本身如此。

**解法**：

1. 跑 `plan/02` §7 第 9 条的**可推导性探针**（只看 effect + 点云预测 region）先确认；
2. 若确认冗余，要么让"几何可达但功能后果不同"的区域存在（旋钮的轮缘低摩擦，D-14），要么如实报告该任务不适合检验 region；
3. **不要靠调随机化范围来制造差距**——那是调到 baseline 失败为止。

---

## P-12 · E-I 的 reward 里混进了任务

**现象**：E-I 在留出任务上零样本成功率意外地高，或者训练时收敛得意外地快。

**原因**：任务信息从后门进来了。常见途径：

- reward 里保留了 success bonus 或 dirt 清除量；
- effect 目标从环境配置读取而非从指令通道读取；
- 环境 reset 逻辑按任务分支，执行器能从初始状态分布反推任务；
- curriculum 里为了调试临时加了任务项，忘了删。

**解法**：`plan/04` §5.1 有明令禁止清单。在 dataloader 和 reward 函数里都加**断言**，并把断言日志存档——`plan/06` §7 要求每次 eval 前查这个日志。不靠自觉。

---

## P-13 · 留出任务差的时候，把它加进训练

**现象**：数字突然好看了。

**原因**：这是本项目最容易破防的地方。留出任务差 → 加数据 → 立刻好看 → 主张归零，且很难被自己发现。

**解法**：留出任务差时的**唯一**允许动作是扩预训练物体集（`plan/03` §2.4）。扩两轮仍不成立，如实报告 Gate E 未通过。写进 `decisions.md`，不要静默处理。

---

## P-14 · Envelope 靠"变宽"通过验收

**现象**：coverage 轻松达到 95%+，下游成功率也不错。

**原因**：envelope 宽到几乎不约束任何东西。只报 coverage 时这完全看不出来。

**解法**：**coverage 和 width 必须同时报**（`plan/03` §8.1、`plan/06` §2.3）。目标是 coverage ≥90% 约束下最小化 width。另外 mechanics 范围不要用"均值 ± k·标准差"——接触力分布通常单侧截断，对称假设不成立，用 pinball loss 回归分位数。

---

## P-15 · 四宫格四格用了不同的 envelope

**现象**：跨形态视频看起来很漂亮。

**原因**：为了让每一格都成功，各自用了各自 target 上表现最好的那条 envelope，或者对 envelope 做了 target-specific 的微调。这样整张图的意义就没了。

**解法**：四格必须来自**同一个 envelope 文件**，在视频里标出该文件的 hash（`plan/06` §3.1、§7）。允许 target 在 envelope 的允许集合内选择具体实现（那正是可行性评价器做的事），但**不允许把值改到允许集合之外**——后者属于 embodiment-specific adaptation，必须单独成节报告（`plan/05` §4.2）。

---

下面 P-16 ~ P-18 是 2026-08-27 在服务器上实测踩到的**真实工程坑**。

---

## P-16 · 拿不到摩擦力（stick/slide 判不出）

**现象**：`ContactSensorCfg` 里没有 `track_friction_forces`，`ContactSensorData` 里没有任何摩擦力字段。

**原因**：**版本过旧。** 该功能由 [PR #3563](https://github.com/isaac-sim/IsaacLab/pull/3563) 引入，**2025-12-10** 合并进 main。本地 isaaclab 版本 0.48.0，changelog 日期 **2025-11-10**，早一个月。不是被移除，是还没加进来。

**解法**：绕过 Isaac Lab 包装层，直接调底层 PhysX tensor view：

```python
# 前提：ContactSensorCfg 必须设 filter_prim_paths_expr（非空）
#       且 max_contact_data_count_per_prim >= 1
ff, fpts, counts, start = sensor.contact_physx_view.get_friction_data(dt=sim_physics_dt)
```

**实测验证**（`/tmp/verify3.py`，mass=0.5 kg，μ=0.4，μmg=1.962 N）：

| 施加力 | 摩擦力(单 env) | 速度 | 模式 |
|---:|---:|---:|---|
| 1.0 N | 1.000 N | ~0 | STICK（静摩擦跟随外力） |
| 1.5 N | 1.500 N | ~0 | STICK |
| 2.5 N | **1.962 N** | 0.54 m/s | SLIDE（饱和到 μN） |
| 3.0 N | **1.962 N** | 1.04 m/s | SLIDE |

教科书级库仑摩擦。**stick/slide 判据**：`|Ff| < μN` → stick；`|Ff| ≈ μN 且 v_tangential > 0` → slide。

**为什么不升级**：upstream 的 `friction_forces_w` 形状是 `(N,B,M,3)`，是**按 filter body 聚合求和**的，把空间分布抹掉了；而 `plan/02` §3.2 要的是逐接触点的力+位置+法向。升级之后仍然要调这两个底层接口。见 `decisions.md` D-20。

---

## P-17 · filter 相关的数据全是零 / 全是 NaN ⚠️ 最容易误判成 Isaac Lab bug

**现象**：物体明明压在地板上，`net_forces_w` 也正确（= mg），但是：

- `force_matrix_w = [0, 0, 0]`
- `contact_pos_w` 100% NaN
- `get_friction_data()` 全零
- `get_contact_data()` 的 **`counts = [0, 0]`**

**原因**：**filter 目标不是 rigid body。** PhysX 的成对接触过滤要求**两侧都是刚体**；只有 `collision_props` 而无 `rigid_props` 的静态碰撞体不会注册成可过滤的接触对。

`net_forces_w` 仍然正确，因为它**不走 filter 通道**——这个不一致极具迷惑性，会让人以为是传感器 bug。

**解法**：所有 filter 目标建成刚体。不动的东西（地板、桌面、柜体）用：

```python
rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)
```

**首选排查项**：拿不到 filter 数据时，第一个看 `get_contact_data(dt)[4]`（counts）。是 0 就是没配上对，不用再往下查。

**对本项目的影响**：`plan/01` §1 已加硬规则 7。擦拭平面、抽屉柜体如果建成静态碰撞体，**接触区域和摩擦力全部拿不到**，`plan/02` 的 Interaction Region 和 Contact Mode 两个字段直接作废。

---

## P-18 · 逐点接触数据是按 env 打包的，直接求和会串环境

**现象**：摩擦力/接触力数值恰好是预期的 N 倍（N = env 数）。

**原因**：`get_contact_data()` / `get_friction_data()` 返回的是**扁平打包 buffer**，形状 `(max_contact_data_count, 3)`，其中 `max_contact_data_count = max_contact_data_count_per_prim × num_envs × num_sensor_bodies`。所有 env 的接触点挤在同一个数组里。

实测：2 env × 每 env 4 个接触点 → 8 行非零，`counts = [4, 4]`，`start_idx = [0, 4]`。直接 `.sum(dim=0)` 会把两个 env 加在一起，正好差 2 倍。

**解法**：用 `counts` 和 `start_idx` 切片：

```python
forces, points, normals, seps, counts, start = view.get_contact_data(dt=dt)
for e in range(num_envs):
    s, n = start[e, 0].item(), counts[e, 0].item()
    env_points  = points[s:s+n]      # 该 env 的接触点
    env_forces  = forces[s:s+n]
    env_normals = normals[s:s+n]
```

**这层切片逻辑应该封装成项目自己的工具函数**（约 30 行），产出 `plan/02` §3.2 需要的"逐接触点 位置 / 法向 / 法向力 / 摩擦力"，不要在每个环境里重复写。

---

## P-19 · headless 下 `sim.step()` 卡死、`app.close()` 挂起

**日期**：2026-08-27（S1 开发时踩到）

**现象**：脚本跑到第一次物理步进就停住。进程还在，但 **CPU 0%、日志不增长、GPU 显存已分配但利用率 0%**。等 15 分钟也不动。

**原因**：两个独立问题，都出在渲染路径上。

1. `SimulationContext.step()` 默认 `render=True`。headless 且未加 `--enable_cameras` 时没有渲染能力，调用会阻塞。这和 P-06/P-07 是同一类问题，只是发生在 API 层而不是命令行层。
2. `SimulationApp.close()` 在本环境同样会挂起。最小复现脚本十步全部跑完、打印完最后一行，仍然 `timeout 124`——卡在关闭阶段。

**解法**：

```python
sim.step(render=False)      # 所有不需要出图的地方一律关渲染

...
# 报告落盘后直接退出，不走优雅关闭
sys.stdout.flush()
os._exit(exit_code)
```

`os._exit()` 跳过 atexit 和析构，对自检/评估脚本没有副作用（结果已经写文件了）。真正需要录像的评估脚本另说，那时必须加 `--enable_cameras`（P-06）。

**排查提示**：区分"卡死"和"慢"看 CPU——慢是 100%，卡死是 0%。本项目里 0% CPU + 显存已占，基本就是渲染阻塞。

---

## P-20 · `pkill -f <pattern>` 会杀掉自己所在的 shell

**日期**：2026-08-27（连踩两次）

**现象**：`ssh host 'pkill -f foo.py; cat > /tmp/foo.py'` 之后文件没被写入；或者 `pgrep -f foo.py` 明明进程已结束却一直报"运行中"。

**原因**：`pkill -f` / `pgrep -f` 匹配**完整命令行**，而执行它的那个 `bash -c` 的命令行里就含有这个模式字符串，于是匹配到自己。`pkill` 把自己所在的 shell 杀掉，后续命令全部不执行；`pgrep` 则永远至少匹配到 1 个。

**解法**：

```bash
pgrep -f "[f]oo.py"     # 字符类：模式本身不含字面 "foo.py"
```

或者把 kill 和后续操作拆成两条独立的 ssh 调用。

**代价**：这个坑最恶心的地方是**它静默失败**——第二次踩到时，我以为是资产生成有问题，实际上是生成脚本压根没被写进去。看到"命令明明跑了但文件没变"，先怀疑这个。

---

## P-21 · `set_joint_effort_target` 会一直保持，不会自动清零 ⚠️ 最难查的一个

**日期**：2026-08-28（S1，查了两小时）

**现象**：前一段测试里给关节施加过力矩，之后的测试中物体的**几何位置全错**——按配置算出的目标位姿完全对不上，执行器"穿过"物体却读不到任何接触。

**原因**：Isaac Lab 的 `set_joint_effort_target()` 设的是**持续指令**，不调用就一直保持上次的值。S1 里旋钮的"已知力矩响应"测试施加 1.5 N·m 把圆盘推到限位 3.491 rad（=200°），之后的摩擦标定阶段圆盘**仍在被 1.5 N·m 驱动**，销钉早就不在角度 0，而标定的目标位姿是按角度 0 算的。

**为什么极难定位**：

- 轮缘那一半**照常工作**（圆盘轴对称，转到哪都一样），只有销钉那一半失败——看起来像"销钉碰撞体有问题"；
- 单独测销钉碰撞体（扔球）**完全正常**；
- `write_joint_state_to_sim(0, 0)` 复位后下一步就又被推回去了，看不出来。

**解法**：换阶段时**显式清零**，并在新阶段的每一步持续清零：

```python
zero = torch.zeros(num_envs, num_joints, device=device)
knob.set_joint_effort_target(zero)
knob.write_joint_state_to_sim(zero, zero)
# 新阶段的每一步也要
def step_fn(i):
    knob.set_joint_effort_target(zero)
    ...
```

**排查提示**：几何对不上时，先打印物体**当前实际关节角**，不要相信"我 reset 过了"。

---

## P-22 · 接触力峰值包含撞击瞬态，不能当"稳态能传多少力"用

**日期**：2026-08-28

**现象**：明明指令法向力 25 N，实测峰值 66.5 N。

**原因**：物体接触瞬间的冲击尖峰。用峰值去论证"在安全力上限内能传多少力矩"，前提直接不成立——因为力早就超了上限。

**解法**：统计时过滤掉超出指令值的样本：

```python
if fnv > MAX_NORMAL_FORCE * 1.15:
    continue          # 撞击尖峰，不计入
```

过滤后旋钮的峰值法向力从 66.5 降到 28.15 N，落在上限内，结论才站得住。

**残留问题**：即便过滤后，τ_pin 实测 23.48 N·m vs 解析 1.30 N·m 仍差一个量级——峰值里还有求解器瞬态。**定性排序（rim < need < pin）可靠且裕量充足，但具体数值不可引用**。要写进论文需改用稳态窗口（接触建立后的中位数）重测。

---

## P-23 · 自由漂浮体在没有控制器接管的阶段会一直掉

**日期**：2026-08-28

**现象**：某个阶段结束后，浮动执行器的位置离谱（实测推子掉到 z = -30 m）。

**原因**：浮动刚体只在被 PD 驱动的阶段受控。前面的阶段如果没人管它，它就自由落体，掉多远取决于那个阶段多长。

**解法**：每个控制阶段开始前显式重置（`tools/s1_check.py::_reset_floating`），不要指望它还在原地。

---

## P-24 · headless 渲染的三个坑

**日期**：2026-08-28（S0）

**现象与解法**：

| 现象 | 原因 | 解法 |
|---|---|---|
| `RuntimeError: A camera was spawned without the --enable_cameras flag` | `SimulationApp({"enable_cameras": True})` **不起作用**。Camera 检查的是 carb 键 `/isaaclab/cameras_enabled`，只有 `AppLauncher` 会写它 | 用 `AppLauncher(headless=True, enable_cameras=True)`，它同时负责选 `isaaclab.python.headless.rendering.kit`（P-06） |
| 开了 cameras 后 `sim.step(render=True)` 仍卡死 | 与 P-19 同源 | 物理走 `step(render=False)`；出图时单独 `sim.render()` + `cam.update(dt)` |
| `cam.set_world_poses_from_view()` 卡死 | 未定位 | 相机位姿在 `CameraCfg.OffsetCfg(pos=..., rot=..., convention="opengl")` 里静态给定，四元数用 `tools/s0_record.py::look_at_quat()` 算 |

**顺带**：只设物理材质不设视觉材质，渲染出来全是白的，**功能区域完全看不出区别**。旋钮的轮缘和销钉一个颜色，而它们的区别就是 D-14 的全部内容。`src/it/build_assets.py::COLOR` 已给所有资产配色。

---

## P-25 · 可视化脚本的动作设计错了，会让正确的结论看起来是错的

**日期**：2026-08-28（S0）

**现象**：S1 数字全对，但录出来的视频要么看不出在验证什么，要么给出相反印象。

**三个实例**：

1. **推子飞出画面**：轮缘那段数字正确（圆盘没转），但画面里推子飞走了，观众合理地理解成"推子飞了所以没转"，而不是"摩擦不够所以没转"。
2. **两种情况用了同一条轨迹**：把销钉也写成"径向压紧 + 沿圆周走"——力穿过转轴，力矩恒为零，Δθ=0.0000。**视频看起来像销钉也推不动，与 D-14 的结论完全相反。**
3. **让推子追着销钉走**：推子的位置目标随圆盘一起动，两者互相拉扯，只转到 +0.13 rad（直线推法能到 +1.52）。

**解法**：可视化的动作必须**忠实反映被验证的物理**，不能图省事统一。旋钮的两种情况本来就不同：

```
销钉：切向直推      τ = F·r        与 μ 无关
轮缘：径向压+切向拖  τ = μ·F·R      只能靠摩擦
```

**教训**：视频不是"把数字配个画面"，它是独立的一道检查。`plan/06` §7 要求人工看视频，正是因为它能抓到数字抓不到的东西——但前提是**视频本身得是对的**。

---

<!-- 从 P-26 开始追加实际踩到的坑 -->
