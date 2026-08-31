# S6 — E-I 交互跟踪执行器

> **这一步的主张是**：每种形态各自训练一个 interaction-conditioned executor，
> 收到同一份 interaction 指令后生成自己的动作。它是最终系统（`plan/README` §8），
> 不是对照实验。

**当前状态（2026-08-31）**：错误的六路课程 A 已全部停止。修正后的 padrod ×
block/press 在 512 env、80 iteration 上通过 deterministic 评估，最佳 checkpoint 是
`runs/s6_corrected/canary_padrod_block_press_handoff_v1/model_79.pt`；继续训练到
250/300 会发生 policy drift，因此没有扩大到其余组合。

⚠️ **候选契约已升到 `interaction-transfer-v5`（D-91/D-92），代码读不了 v4 artifact。**
远端主目录仍是 v4；修正后的 v5 代码与 artifact 在**隔离目录**：
`/workspace/it_v5`（代码）、`/tmp/s5_v5`（3 份任务）、`/tmp/s6_v5/probe`（43 份探针）。
本地工作树是 v5 候选，禁止再用整树同步把两边无意混合。切换步骤见第 7 节。
接手前**必须先读第 3 节**——上一轮交接时"地基完成、验过"的那套里有 **9 处缺陷**，
其中 7 处不报错。

---

## 1. 指令来源（未变）

| | 份数 | 说明 |
|---|---:|---|
| 探针指令 | **43** | 10 个物体 × 15 条原语，按 `(物体, 原语)` 拆（D-76） |
| 任务指令 | 3 | 抽屉 / 擦拭 / 旋钮，合并五个策略家族 |

calibration 按 `(物体, 原语)` 分层重划到每组 30 条（`probe_split.txt`）。
43 份里 **6 份**的 region 用了兜底平滑（D-78）。
⚠️ `column/push` 只剩 `block` 一个承载物体，**S7 的 leave-one-primitive 分析必须单列**。

## 2. 跟踪 reward 的离线体检（`reward_probe.txt`）

三节全部通过：在线/离线 traction 中位比值 1.000/0.999/1.000（相关 0.93~0.997）；
reward 分得开自己的指令与错配指令，AUC **0.955 / 1.000 / 1.000**。

⚠️ **第一节喂进去的是已经构造好的 `mech/force_obj`**，所以它只对了**公式**，
没有走过"从原始 PhysX 量构造接触力"那条路径——而那条路径上有三处口径不同（P-80）。

---

## 3. 接手复查发现的 9 处缺陷（**动手前必读**）

上一轮的交接说"地基完成并通过 18 项单测 + 两份离线体检"。那些检查都真的跑了、
也真的绿了；问题是**它们查不到下面这些**。7 处不报错。

| | 缺陷 | 怎么发现的 | 记在哪 |
|---|---|---|---|
| 1 | **环境里没有台面**，探针物体开局自由落体 | 读代码 + 回头看 dry-run 日志：`region/mode/mech` **三项恒为 0** | P-75 / D-81 |
| 2 | **接触 filter 指到了错的 prim**（`slab`/`ridge` 的刚体在子 prim 上） | 同上 | P-75 / D-81 |
| 3 | **`own_radius` 比执行器还小**（默认 0.08 m，垫面离杆根 0.108 m），接触全被当 foreign 丢掉 | 同上 | P-75 |
| 4 | **提前终止可以自杀刷分**：五项 reward 全 ≤0，早退回报 0 = 最优 | 读代码（D-31 第 3 个洞） | P-77 / D-83 |
| 5 | **自由空间里也在做纯力控**，25 N 前馈把 0.3 kg 的杆推到 2.1 m/s | 读代码 | D-82 |
| 6 | **指令特征转到了错误的坐标系**（多了一次转置） | 写了一条会失败的测试，先在旧代码上确认它是红的 | P-81 |
| 7 | **标定出的 reward scale 根本没接进环境** | 读代码 | D-84 |
| 8 | **在线只在控制步末尾采一次接触**（P-31 原样重演） | 读代码 | 环境 docstring 第三条 |
| 9 | **在线接触力没按 S4 的口径构造**（符号/定向三处不同） | 读 S4 提取器 | P-80 |

### 3.1 修的过程里**新测出来**的 6 件事

这几件都是"改完之后跑一遍才知道"的，靠读代码读不出来：

| | 现象 | 结论 |
|---|---|---|
| **Isaac 下未捕获异常也以退出码 0 收场** | `KeyError` + 完整 traceback，而 `.done` 里是 0 | **P-74 ⭐⭐⭐**。最小复现五行。这是 P-55 的镜像：那次是假 FAIL，这次是**假 PASS** |
| 杆类执行器**自转 76 rad/s** | 旋转 PD 按质量归一，量纲错；杆绕自身轴的惯量小两个数量级，`ω_n·dt≈2.8` 直接发散 | **P-76 / D-85**。改按转动惯量归一后 76 → 0.67 rad/s，`r_mode` −31.6 → −0.46 |
| **悬停 −0.43/步，压对 −278/步** | 四项全 ≤0 时最优策略是"永远别碰那个物体" | **P-77 / D-84 ⭐⭐⭐**。`r_region` 改成带符号占比 |
| `r_mode` 报 **−27675** | `mode/slip_speed` 的允许盒半宽 6.6e-5 m/s，按半宽归一失去刻度 | **P-79 / D-88**。半宽给一个**物理**下限 |
| **400 轮 reward 恒等于 −99.13** | 逐轴 `pos_limit` 的可达立方体角（0.87 m）比失败门槛（0.75 m）还远，每条 episode 必然失败 | **P-78**。曲线只是"平"，看不出是收敛还是没梯度 |
| 策略跑到 **0.465 m 外坐着不动**，`touching` 恒为 0 | 势函数取负值时 `(γ−1)Φ` 奖励"离得越远越好" | **P-82 / D-89 ⭐⭐⭐**。Ng 等 1999 管的是最优解，不管 PPO 先爬上哪个坡 |

---

## 4. `tools/s6_smoke.py`：能失败的接触冒烟

**为什么它不可替代。** `--dry-run` 分不出"随机策略没碰到东西"和"环境根本测不到接触"
——两者在日志里都是三项恒为 0。冒烟**主动去建立接触**，所以任何一项还是零就是环境的问题。

八节 **18 项检查**，最要紧的三项：

| 检查 | 实测 | 它挡的是什么 |
|---|---|---|
| `r_region` 分得开压对/压错 | 内外差 **+0.92** | reward 对区域不敏感时训练曲线照样会涨 |
| **压对 > 悬停**（逐 env，条件在真的压进允许区域上） | **+0.97 vs −0.01** | "最优策略是别碰"（P-77） |
| 随机策略不能必然失败、且偶尔碰得到 | 失败率 **0.0004**、接触步占比 **2.9%** | "每条 episode 都一样 → 没有优势信号"（P-78）；"探索找不到接触"（D-86） |

⚠️ **`inside_region_share` 是报出来的数，不是判据。** `region/allowed` 只占表面积
1.35~4.4%（256 格里 4~10 格）。"能不能把力压进允许集合"正是 S6 训练要回答的问题，
在冒烟里断言它等于预设结论。

**三条指令上的实测**（开环脚本、垫头杆、按**当前命令格**的 region 质心瞄准）：

| 指令 | `inside_region_share` | 压对 vs 悬停 |
|---|---:|---|
| `ridge / press`（曲面线接触） | **0.640** | +1.38 vs −0.10 |
| `slab / rub`（平面扫掠） | **0.389** | +1.32 vs −0.13 |
| `block / press`（平面对平面） | **0.227** | +0.91 vs −0.14 |

⚠️ **这里有一条我先写错、又用实测改过来的结论。** 第一版冒烟瞄的是**整条指令的
接触锚点**（全局质心），`slab/rub` 只测到 **0.065** 并且判红；我据此写下了
"允许区域在相当程度上编码了采集端执行器的接触斑块几何"。**那个归因是错的**——
扫掠类原语的允许区域**逐格移动**，整条指令的质心落在扫掠路径中间，
而任何单独一格的允许区域都在别处。改成按当前格瞄准之后 0.065 → **0.389**。

**改正之后剩下的、仍然成立的那部分**：三条里 `block/press` 明显最低（0.227）。
它是唯一的**平面对平面**接触，允许区域是 256 格里的 4 格，位置正好是采集侧那块
35×25 mm 板的四个角；而垫头杆的垫面是 40×30 mm，四个角差约 2.5 mm，格距约 7 mm。
曲面线接触（ridge 0.64）与扫掠（slab 0.39）都没有这个问题。
**所以"允许区域编码了采集端斑块几何"这件事是真的，但只在平面对平面这一档上显著。**
它仍然关系到 `plan/00` 的"envelope 与形态无关"与 S4.5 的分辨率选择，
但影响范围比我最初写的窄得多。

---

## 5. 课程 A 正在跑什么

六张卡，六个 (执行器, 物体, 原语) 组合，各 800 轮 × 32 步 × 1024 env：

```
GPU0 padrod × block × press     GPU3 padrod × ridge × press
GPU1 padrod × block × push      GPU4 hook   × block × poke
GPU2 padrod × slab  × rub       GPU5 hook   × column × slide_push
```

**通过判据**（`plan/04` §13 第 3 条 + §10 A）：分项 reward 单调上升、
`diag/inside_region_share` 显著上升、`r_region` 转正、`diag/failed` < 5%。
任一不满足回到 §13 的诊断顺序，**不调 PPO 超参**。

**看训练必须看分项**（P-27）：`extras["log"]` 的每一项都进了 TensorBoard，
其中 `diag/touching`、`diag/inside_region_share`、`diag/anchor_distance`、
`diag/failed` 四个比 `Mean reward` 有用得多——上面那 6 条新发现里有 3 条
**只能从分项看出来**，总 reward 曲线全都是"平"或"在涨"。

## 6. 还欠什么

| 缺什么 | 卡在哪 |
|---|---|
| 课程 A 的结论 | ✅ 静态 block/press 在 80 iteration 收敛；`model_79` 100% 完成、0 timeout/failure、79.18 步。更长训练发生 policy drift，不能用最后 checkpoint |
| ~~`demand` 与"该格里实际发生的变化"~~ | ✅ **已测、已真正接入**。Claude 初版只写字段、tracker 未消费，且静压抖动制造 block/press 总 demand 43、column/press 583。D-92 修为 tracker 直读字段，离线/在线共享 0.5 mm 死区；静态 press/hold demand 全为 0，三任务对拍 0.931 / 1.000 / 1.002 |
| ~~原始 PhysX 量构造的接触力 vs `mech/force_obj`~~ | ✅ **已对拍**（`tools/s6_force_check.py`）。三个数据集中位余弦 **1.00000**、模长比 1.00000。并验过它能失败：法向定向反了 → −0.99843；摩擦符号反了 → p01 +0.914。**已知盲区**：离线记录的法向力已经是非负的，"忘了取绝对值"查不出来（P-83） |
| Allegro / 平行夹爪 | 资产里还没有夹爪；Allegro 要另配关节动作空间 |
| articulated 探针物体 | `probe_env_cfg` 显式拒绝：接它要一并搬 `probe_scene` 里的 joint 与 `damping_nominal`（采集时在运行时写进关节，会盖过资产值，P-38） |
| S4.5 / S4.6 下游段 | 都要 S6 的 checkpoint |

---

## 7. 切到 v5 契约怎么做

新代码（`WindowTracker.demand()` 读 `effect/bin_demand`）**读不了 v4 artifact**，
所以切换是一次性的、要一起做完：

```bash
# 1) 代码就位（先提交 v5 候选；确认没有训练在跑后再有意切换主目录）
ssh root@10.0.6.98 'ps aux | grep -c "[s]6_train"'      # 必须是 0
./tools/sync.sh

# 2) 重建 artifact（纯 numpy，分钟级）
IT_PY=/isaac-sim/python.sh IT_S6_OUT=/tmp/s6 bash tools/s6_commands.sh --only build
IT_PY=/isaac-sim/python.sh bash tools/s5_all.sh          # 三份任务 + 十一项闸门

# 3) 闸门重跑
PYTHONPATH=src /isaac-sim/python.sh tools/s6_reward_probe.py ...   # 四节
PYTHONPATH=src /isaac-sim/python.sh tools/s6_force_check.py ...
IT_GPU=0 ./tools/run_remote.sh "... tools/s6_smoke.py ..." s6smoke  # 18 项
```

**S5 的结论不会变**（已在 `/tmp/s5_v5` 上重跑核对过，coverage / width 与 v4
发布的数逐位一致），但重跑是为了留下证据，不是为了看有没有变。

## 8. 本轮纠正后的可复核结论

- v5 E-I 单元测试：29/29；新增测试会让“`bin_demand` 没接进 tracker”和“静压抖动
  产生非零 demand”直接失败。
- v5 block/press smoke：正确 +0.597、hover −0.0077、错误区域 −2.058；要求接触时
  touching 100%，失败 0。
- 同一 `model_79` 在 v4/v5 block/press deterministic rollout 的所有指标逐位相同：
  完成 100%、timeout/failure 0、79.18 步、要求接触时 touching 77.28%、区域内 69.01%。
- checkpoint 选择不能看最后一步：`model_250` 需 159.9 步且区域奖励降到 0.424；
  `model_300` 完成 0%、timeout 100%。当前 Stage A 产物固定为 `model_79`。
- v5 只在隔离目录验证完毕，尚未覆盖远端主目录；dynamic interaction 仍需在 v5 上
  单独设计 smoke/canary，不能用静态 press 证明 demand 的在线学习有效。
