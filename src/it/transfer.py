"""把同一任务的多条 S4 示教聚合成可传递的 interaction command。

这个模块实现项目 S5 的第一条、最保守的基线链路：

``多条成功 S4 record -> 相分段活动量对齐 -> 每条 episode 等权统计 -> InteractionTransfer``

它不推断任务语义，不声称恢复了任务的必要/充分条件，也不替执行器判断形态可行性。
输出只是示教中实际出现过的物体中心交互统计；不同 embodiment 各自训练的 E-I
executor 负责把这些物理字段解码成自己的 action。

采集脚本的 ``phase`` / ``progress`` 只用于**离线对齐**，不进入产物（D-55：标签
可以用，观测不行）。产物中的时间轴是一个没有任务语义的 command sequence index。
任务名仅留在 meta 供审计，``executor_arrays()`` 永远不返回 meta。

v3 -> v4 改了什么（D-72 / D-73 / D-74，实测依据见 ``out/s5/units_probe.txt``）
--------------------------------------------------------------------------

三个字段各自依赖了一个**不属于物理**的东西，而三条都不报错、所有单测与验收照样通过。
共同的形状是：**单元测试查得了"实现有没有 bug"，查不了"这个量的定义对不对"。**

**一、traction 不再随表面采样分辨率放大（P-68 / D-72）。** v3 把接触力整个塞给最近的
命令格再除以格面积，于是 ``N/m²`` 里的 m² 是我们随手选的分辨率。实测同一批示教从
64 格换到 1024 格，擦拭 178→2498、旋钮 12711→163033 N/m²，几乎正好按 1/面积 走。
v4 先用**固定物理带宽**（``SCATTER_SIGMA`` = 4 mm，跟拟传感 taxel pitch 走，不跟
分辨率走）的同部件守恒核把力散射到冻结表面点，再在格内按 ``|f|`` 加权池化。
四族池化做法的实测对照见 D-72——只有这一族三档一致（漂移 1.10~1.52×，抽屉那一格
1.52× 未过 1.50× 的门槛，残差在最粗档，如实记 FAIL 而不放宽门槛）。

**二、payload 补上连续滑移速度（P-69 / D-73）。** ``mode/prob`` 的四档是拿
``SLIP_SPEED_MIN`` 切出来的，实测把阈值在 1~10 mm/s 之间换，抽屉"黏住"占的力比例
从 0.592 摆到 0.964（37 个百分点）。S4 侧一直存着连续量，但 v3 **只把切完的标签导进
payload**，于是 ``r_mode`` 只能追那个任意约定。v4 增加 ``mode/slip_speed/{median,lo,hi}``
（与 mechanics 同一套标定），``mode/prob`` 降为诊断字段。

**三、effect 两路各带任务无关刻度（P-70 / D-74）。** ``effect/rigid`` 把米和弧度放进
同一个 6 维向量，而每个任务恰好只有一路非零：抽屉平移 p90 0.046 m / 旋转恒 0，旋钮
反过来 0.940 rad——直接取 L2 就是 D-31 第 2 个洞（量纲失衡）藏在"统一接口"里面。
v4 增加 ``effect/rigid/metric``（把 ξ 换算成表面平均位移的 6×6 度量）与两路各自的
刻度。换算之后抽屉 0.0458 m、旋钮 0.0504 m，从差 20.5× 变成差 1.10×。

v1 -> v2 改了什么（D-65 / D-66，实测依据见 ``out/s5_align/probe.txt``）
--------------------------------------------------------------------

**一、对齐轴换成"相分段 + 活动量"。** v1 按全局 ``progress`` 分格，而 ``progress``
是任务 effect 完成度：接近段全是 0、到达目标之后全是 1。实测 12 条真实抽屉记录
**84% 的帧落进 32 格里的 2 格**，中间 30 格每格中位数 2 帧；旋钮 73%。所有既有
校验照常通过，``support/episodes`` 还是满的——这是一条典型的静默失败（P-58）。

v2 先按 ``phase`` 把 episode 切成四段（这正是 `plan/03` §8 第 1 条写的"按 phase
**和** object progress 对齐"里被 v1 漏掉的那一半），格数按各段的**交互活动量占比**
分配（每段保底 ``PHASE_FLOOR`` 格），段内也按累计活动量分格。活动量 = 接触冲量率
+ effect 变化率，两路**各自在 episode 内归一化**后相加，因此不含任务分支，也不需要
在牛顿与米之间拍权重（D-31 的第 2 个洞）。

**二、每个 cell 的所有字段来自同一组 episode 和同一批帧。** v1 的 ``region`` /
``engage`` / ``mode`` 用 NaN 表示"这条 episode 没碰这里"（统计时被排除），而
``traction`` 用 0（统计时被算进去）。于是同一个 cell 上，方向是"碰过的人的平均"，
而力是"所有人的平均、被没碰的人稀释"。实测抽屉 10%、擦拭 15% 的 occupied cell 因此
拿到 ``[q10,q90] = [0,0]`` 的力带——命令同时说"这里该接触"和"这里的力必须为零"。
v2 把 ``traction`` / ``moment_density`` 也改成**接触条件化**，并新增
``region/support``（多少条 episode 碰过）与 ``region/duty``（碰过的人里，这个格内
有多大比例的帧在接触），让下游能分辨 1/12 与 12/12 的支持度。

**三、``engage`` 的集中度不再被丢掉。** 各 episode 的单位方向求平均后，合矢量长度
就是 ∈[0,1] 的集中度，正是方向多峰性的直接度量。v1 把它阈值化成一个 bool 就扔了。

**四、诊断措辞更正。** v1 报的"6D wrench 重建误差 6e-6 N"在代数上是恒等式
（cell 内先算局部力矩再求和 = 直接对接触点求力矩），那个数只反映浮点精度。真正有
信息量的是**表面投影完整性**：被 ``on_surface`` / ``weight>0`` 滤掉的接触占了多少
力，以及由此与 S4 独立记的 ``mech/wrench_obj`` 差多少。v2 按后者报，并额外报被丢掉
的力占比。跨 episode 取中位数/分位数之后守恒性**不再成立**，那是任何统计聚合的
固有性质，不得再声称"积分恢复 6D wrench"。
"""

from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from it.records import EpisodeRecord, META_KEY
from it.surfaces import SCATTER_K, SCATTER_SIGMA, Surface, surface_for

TRANSFER_SCHEMA_VERSION = "interaction-transfer-v4"
MODE_COUNT = 4
N_PHASES = 4
#: 每个**有帧**的 phase 至少分到的命令格数。松开段的活动量恰好是 0，若按活动量
#: 判"占用"，它会分到 0 格、那一段的帧被整段丢掉而不报错。
PHASE_FLOOR = 2

# E-I 输入契约是 exact allowlist，不靠前缀猜测。新增字段必须显式评审。
EXECUTOR_ARRAYS = frozenset({
    "command/fraction",
    "command/valid",
    "support/episodes",
    "surface/points_obj",
    "surface/normals_obj",
    "surface/area",
    "surface/part",
    "effect/rigid/median",
    "effect/rigid/q10",
    "effect/rigid/q90",
    "effect/rigid/valid_count",
    "effect/surface_state/median",
    "effect/surface_state/q10",
    "effect/surface_state/q90",
    "effect/surface_state/valid_count",
    "region/mass/mean",
    "region/mass/q10",
    "region/mass/q90",
    "region/allowed",
    "region/support",
    "region/duty",
    "engage/dir/mean",
    "engage/concentration",
    "engage/valid",
    "mode/prob",
    "mode/slip_speed/median",
    "mode/slip_speed/lo",
    "mode/slip_speed/hi",
    "effect/rigid/metric",
    "effect/rigid/scale_m",
    "effect/surface_state/scale",
    "mech/traction_obj/median",
    "mech/traction_obj/lo",
    "mech/traction_obj/hi",
    "mech/moment_density_obj/median",
    "mech/moment_density_obj/lo",
    "mech/moment_density_obj/hi",
})

#: 允许集合的"episode 被覆盖"判据：这么大比例的力加权接触点必须落进集合。
#: 与 `plan/03` §8.1 的 region 子指标同一口径，两个数才可比。
POINT_MASS_TARGET = 0.95
#: 标定目标：这么大比例的校准 episode 要被覆盖（`plan/03` §8.1 的 coverage ≥ 90%）。
TARGET_COVERAGE = 0.90
#: mechanics 尺度地板取本任务 traction 量级的这个比例。**它是标定唯一的自由参数**，
#: 必须随结果一起报（D-71：用绝对地板会让结论完全变样）。
MECH_FLOOR_RELATIVE = 0.05
#: region 密度的平滑带宽，单位是**命令格间距**。**默认 0 = 不平滑**：实测平滑会把
#: 允许面积在擦拭上从 4.42% 涨到 8.02%、旋钮 1.37%→2.71%（几乎翻倍），而 width 正是
#: `plan/03` §8.1 要最小化的那个数。平滑只作**兜底**用在原始估计量退化的那些
#: artifact 上（τ* 发散），由调用方决定并逐份记录，见 D-78。
REGION_SMOOTH = 0.0

#: region 允许集合的 τ 搜索网格。A(τ) 随 τ 嵌套，所以可直接对 τ 做 split conformal。
TAU_GRID = np.concatenate([np.linspace(0.50, 0.99, 50), [0.995, 0.999, 1.0]])
#: mechanics 盒子的 k 搜索网格。集合随 k 单调膨胀，同理。
K_GRID = np.concatenate([np.linspace(0.25, 8.0, 156), [10.0, 14.0, 20.0, 30.0]])

#: traction 的池化方式（P-68 / D-72）。**这是被实测选出来的，不是拍板的**，
#: 四族候选的对照见 ``tools/s5_units_probe.py`` 与 ``out/s5/units_probe.txt``。
#:
#: - ``nearest_area``：v2 的做法——接触力整个塞进最近的那个命令格再除格面积。
#:   格面积是**我们选的采样分辨率**，不是接触斑块的物理面积，于是 traction
#:   随分辨率线性放大（实测擦拭 64→1024 格：179→2500 N/m²）。D-58 明文禁止过
#:   这个实现，但代码里一直是它；
#: - ``kernel_area``：先按固定带宽核散射到冻结表面点，再在格内按面积平均。
#:   守恒，但格远大于核时被稀释，仍随分辨率变；
#: - ``kernel_point``：取该格代表点上的场值。分辨率无关，但粗档下欠采样；
#: - ``kernel_forcew``：散射到表面点后格内按 ``|f_j|`` 加权平均。分辨率无关，
#:   但比在线实现系统性低 1.48~1.75×，且隐含依赖精细采样密度（D-79 换掉了它）；
#: - ``contact_kernel``：**现行做法**。直接在接触点上求核和，与在线实现同一个式子。
TRACTION_POOLING = "contact_kernel"
TRACTION_POOLINGS = ("nearest_area", "kernel_area", "kernel_point", "kernel_forcew",
                     "contact_kernel")
#: 归一化的高斯面密度核常数：∫G dA = 1。与 `ei_reward._GAUSS_NORM` 必须相同。
_GAUSS_NORM = 1.0 / (2.0 * math.pi * SCATTER_SIGMA ** 2)

#: 每个 cell 上"只在接触过的 episode 之间统计"的字段。它们必须共用同一个
#: ``contact_frames > 0`` 掩码，否则同一个 cell 上不同字段来自不同的 episode 子集
#: （v1 就是这么错的）。
CONTACT_CONDITIONED = ("engage", "mode", "slip", "traction", "moment_density")

#: 不按命令时间格索引的 payload 字段：物体几何常量与两路 effect 的刻度。
#: 它们对整份 artifact 只有一个值，所以首维不是 bins。
TIME_INVARIANT_ARRAYS = frozenset({
    "effect/rigid/metric", "effect/rigid/scale_m", "effect/surface_state/scale",
})


class TransferError(ValueError):
    """S5 interaction transfer 违反数据契约。"""


# ---------------------------------------------------------------- 对齐

def activity_rate(record: EpisodeRecord, surface: Surface) -> np.ndarray:
    """逐帧交互活动量：接触冲量率 + effect 变化率，两路各自在 episode 内归一化。

    两路都先除以自己在本条 episode 内的峰值再相加，因此不需要跨"牛顿"与"米"的
    权重，也不含任何任务分支。整条 episode 某一路恒为零时那一路直接不贡献。

    effect 一路不直接取 ``norm(concat(dp, dr))``——那要在米和弧度之间拍一个权重。
    改为把 (dp, dr) 作用在冻结 surface 点上取**表面点平均位移**（米），
    ``surface_state`` 一路取面积加权变化量。
    """
    a = record.arrays
    contact = np.asarray(a["aux/frame_force"], dtype=np.float64)

    rigid = np.asarray(a["effect/rigid"], dtype=np.float64)[:, 0, :]
    state = np.asarray(a["effect/surface_state"], dtype=np.float64)[:, 0, :]
    n_level = state.shape[1]
    if n_level not in surface.parent:
        raise TransferError(f"surface 没有 effect field 所需的 {n_level} 点层级")
    points = np.asarray(surface.points[:n_level], dtype=np.float64)
    displacement = rigid[:, None, :3] + np.cross(rigid[:, None, 3:], points[None, :, :])
    effect = np.linalg.norm(displacement, axis=2).mean(axis=1)
    cell_area = np.bincount(surface.parent[n_level],
                            weights=np.asarray(surface.area, dtype=np.float64),
                            minlength=n_level)
    effect = effect + np.einsum("tc,c->t", np.abs(state), cell_area)

    total = np.zeros(len(contact), dtype=np.float64)
    for channel in (contact, effect):
        channel = np.abs(np.nan_to_num(channel, nan=0.0, posinf=0.0, neginf=0.0))
        peak = float(channel.max()) if len(channel) else 0.0
        if peak > 0:
            total = total + channel / peak
    return total


def surface_metric(surface: Surface) -> np.ndarray:
    """(6,6) 度量张量 ``M``：把刚体增量 ξ=(dp, dr) 换算成**表面点均方位移**（m²）。

    ``effect/rigid`` 把米和弧度放进同一个定长向量。维度统一了，**量纲没有统一**：
    实测每个任务恰好只有一路非零（抽屉平移 0.055 m / 旋转恒 0，旋钮反过来），
    两路量级差一个数量级。下游 ``r_effect`` 若直接对 6 维取 L2，同一组权重下旋钮的
    effect 项就比抽屉大一个数量级，而擦拭这一路完全没有信号——那正是 D-31 第 2 个洞
    （量纲失衡，最优解是"不动"），只是它藏在"统一接口"里面（P-70）。

    注意**不是** ``effect/rigid`` 设计错了：把关节量换算成刚体增量正是它消掉任务
    分支的办法。错在下游拿它算误差时缺一个任务无关的归一化。

    办法：把 ξ 作用在冻结 surface 点上，取表面点位移的均方根。表面点 ``p`` 处的位移是
    ``dp + dr × p = [I | -[p]_×] ξ``，于是

        mean_j a_j ‖dp + dr × p_j‖² / Σ_j a_j  =  ξᵀ M ξ

    这个 ``M`` 只由物体几何决定：任务无关（同一个公式）、有物理含义（"物体表面平均
    移动了多远"，单位米）、且自动处理了旋转与半径的关系——大物体转同样的角度表面走得
    更远，本来就该权重更大。``transfer.activity_rate`` 早就用同一个办法解决同一个
    问题，当时没想到 ``r_effect`` 也需要它。

    用**全分辨率**表面点做面积加权积分，因此 ``M`` 与命令分辨率无关。
    """
    points = np.asarray(surface.points, dtype=np.float64)
    area = np.asarray(surface.area, dtype=np.float64)
    total = float(area.sum())
    if total <= 0:
        raise TransferError("surface 总面积为零，无法定义 effect 度量")
    n = len(points)
    jacobian = np.zeros((n, 3, 6))
    jacobian[:, 0, 0] = jacobian[:, 1, 1] = jacobian[:, 2, 2] = 1.0
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    # -[p]_× ，使得 -[p]_× dr = dr × p
    jacobian[:, 0, 4], jacobian[:, 0, 5] = z, -y
    jacobian[:, 1, 3], jacobian[:, 1, 5] = -z, x
    jacobian[:, 2, 3], jacobian[:, 2, 4] = y, -x
    metric = np.einsum("n,nki,nkj->ij", area, jacobian, jacobian) / total
    return 0.5 * (metric + metric.T)          # 对称化，消掉浮点不对称


def rigid_surface_displacement(rigid: np.ndarray, metric: np.ndarray) -> np.ndarray:
    """把 (..., 6) 的刚体增量换算成表面平均位移（米）。"""
    values = np.asarray(rigid, dtype=np.float64)
    quadratic = np.einsum("...i,ij,...j->...", values, metric, values)
    return np.sqrt(np.maximum(quadratic, 0.0))


def phase_budget(records: Iterable[EpisodeRecord], surface: Surface, *, n_bins: int,
                 floor: int = PHASE_FLOOR) -> tuple[int, ...]:
    """按各 phase 的活动量占比分配命令格数，每个**有帧**的 phase 保底 ``floor`` 格。

    用跨 episode 的**中位数**占比，避免个别异常长的接近段决定整条命令轴。
    """
    shares: list[np.ndarray] = []
    occupancy = np.zeros(N_PHASES, dtype=bool)
    for record in records:
        activity = activity_rate(record, surface)
        phase = np.asarray(record.arrays["phase"], dtype=np.int64)
        valid = np.asarray(record.arrays["valid_s4"], dtype=bool)
        present = np.array([bool((valid & (phase == p)).any()) for p in range(N_PHASES)])
        occupancy |= present
        per_phase = np.array([activity[valid & (phase == p)].sum() for p in range(N_PHASES)])
        total = per_phase.sum()
        shares.append(per_phase / total if total > 0
                      else present / max(int(present.sum()), 1))
    if not shares:
        raise TransferError("phase_budget 至少需要一条 episode")
    share = np.where(occupancy, np.median(np.stack(shares), axis=0), 0.0)
    share = share / share.sum() if share.sum() > 0 else occupancy / max(int(occupancy.sum()), 1)

    budget = np.where(occupancy, floor, 0).astype(np.int64)
    spare = int(n_bins - budget.sum())
    if spare < 0:
        raise TransferError(f"n_bins={n_bins} 不足以给 {int(occupancy.sum())} 个 phase "
                            f"各保底 {floor} 格")
    extra = np.floor(share * spare).astype(np.int64)
    # 余数按小数部分从大到小补，保证总数正好 n_bins 且与 episode 顺序无关。
    remainder = share * spare - extra
    for index in np.argsort(-remainder, kind="stable")[:spare - int(extra.sum())]:
        extra[index] += 1
    return tuple(int(value) for value in budget + extra)


def bin_index(record: EpisodeRecord, surface: Surface, *, budget: tuple[int, ...]) -> np.ndarray:
    """把每个有效帧映射到命令格号；无效帧为 -1。"""
    a = record.arrays
    valid = np.asarray(a["valid_s4"], dtype=bool)
    phase = np.asarray(a["phase"], dtype=np.int64)
    activity = activity_rate(record, surface)
    offsets = np.concatenate([[0], np.cumsum(budget)]).astype(np.int64)
    out = np.full(len(valid), -1, dtype=np.int64)
    for p in range(N_PHASES):
        frames = np.flatnonzero(valid & (phase == p))
        width = int(budget[p])
        if not len(frames):
            continue
        if width <= 0:
            raise TransferError(
                f"phase {p} 有 {len(frames)} 个有效帧但分到 0 格；那些帧会被静默丢弃")
        if len(frames) == 1:
            local = np.array([0.5])
        else:
            cumulative = np.concatenate([[0.0], np.cumsum(activity[frames[1:]])])
            span = float(cumulative[-1])
            local = (cumulative / span * (1.0 - 1e-9) if span > 1e-12
                     else (np.arange(len(frames)) + 0.5) / len(frames))
        out[frames] = offsets[p] + np.minimum((local * width).astype(np.int64), width - 1)
    if np.any(valid & (out < 0)):
        raise TransferError("有有效帧没有落进任何命令格")
    return out


# ---------------------------------------------------------------- 产物

@dataclass
class InteractionTransfer:
    """一个任务实例的 embodiment-independent interaction command。"""

    meta: dict[str, Any]
    arrays: dict[str, np.ndarray]

    def validate(self) -> None:
        if self.meta.get("schema_version") != TRANSFER_SCHEMA_VERSION:
            raise TransferError(
                f"schema_version 必须是 {TRANSFER_SCHEMA_VERSION!r}，"
                f"实际是 {self.meta.get('schema_version')!r}。"
                "v1 的对齐轴与 cell 统计口径已作废（D-65/D-66）；v3 的 traction "
                "随分辨率放大、mode 只有阈值化标签、effect 缺任务无关刻度"
                "（D-72/D-73/D-74）。混用会让下游拿到一半新一半旧的量而看不出来")
        for key in ("transfer_id", "task", "object", "geometry_variant", "surface"):
            if key not in self.meta:
                raise TransferError(f"meta.{key} 是必填项")
        unknown = set(self.arrays) - EXECUTOR_ARRAYS
        missing = EXECUTOR_ARRAYS - set(self.arrays)
        if unknown or missing:
            raise TransferError(
                f"数组必须精确匹配 E-I allowlist；missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}")

        arrays = {name: np.asarray(value) for name, value in self.arrays.items()}
        if any(value.dtype == object for value in arrays.values()):
            raise TransferError("不允许 object dtype")
        fraction = arrays["command/fraction"]
        if fraction.ndim != 1 or len(fraction) < 2:
            raise TransferError("command/fraction 必须是一维且至少 2 个时间格")
        bins = len(fraction)
        if np.any(np.diff(fraction) <= 0) or fraction[0] < 0 or fraction[-1] > 1:
            raise TransferError("command/fraction 必须在 [0,1] 内严格递增")
        for name in EXECUTOR_ARRAYS:
            if name.startswith("surface/") or name in TIME_INVARIANT_ARRAYS:
                continue
            if arrays[name].shape[0] != bins:
                raise TransferError(f"{name} 的首维必须是时间格数 {bins}")
        if arrays["command/valid"].shape != (bins,) or arrays["command/valid"].dtype != np.bool_:
            raise TransferError("command/valid 必须是一维 bool")
        if not np.array_equal(arrays["command/valid"], arrays["support/episodes"] > 0):
            raise TransferError("command/valid 必须与 support/episodes > 0 一致")

        points = arrays["surface/points_obj"]
        normals = arrays["surface/normals_obj"]
        area = arrays["surface/area"]
        part = arrays["surface/part"]
        if points.ndim != 2 or points.shape[1] != 3:
            raise TransferError("surface/points_obj 必须是 (S,3)")
        n_surface = points.shape[0]
        if normals.shape != points.shape or area.shape != (n_surface,) or part.shape != (n_surface,):
            raise TransferError("surface arrays 的点数不一致")
        if np.any(area <= 0) or not np.isfinite(points).all() or not np.isfinite(normals).all():
            raise TransferError("surface arrays 含非法值")

        for name in ("region/mass/mean", "region/mass/q10", "region/mass/q90",
                     "region/allowed", "region/support", "region/duty",
                     "engage/concentration", "engage/valid"):
            if arrays[name].shape != (bins, n_surface):
                raise TransferError(f"{name} 必须是 (B,S)")
        if arrays["region/allowed"].dtype != np.bool_:
            raise TransferError("region/allowed 必须是 bool")
        for name in ("engage/dir/mean", "mech/traction_obj/median",
                     "mech/traction_obj/lo", "mech/traction_obj/hi",
                     "mech/moment_density_obj/median",
                     "mech/moment_density_obj/lo",
                     "mech/moment_density_obj/hi"):
            if arrays[name].shape != (bins, n_surface, 3):
                raise TransferError(f"{name} 必须是 (B,S,3)")
        if arrays["mode/prob"].shape != (bins, n_surface, MODE_COUNT):
            raise TransferError("mode/prob 必须是 (B,S,4)")
        for name in ("mode/slip_speed/median", "mode/slip_speed/lo", "mode/slip_speed/hi"):
            if arrays[name].shape != (bins, n_surface, 1):
                raise TransferError(f"{name} 必须是 (B,S,1)")
        if np.any(arrays["mode/slip_speed/lo"] < 0):
            raise TransferError("滑移速率是非负量，lo 不能为负")

        metric = arrays["effect/rigid/metric"]
        if metric.shape != (6, 6):
            raise TransferError("effect/rigid/metric 必须是 (6,6)")
        if not np.allclose(metric, metric.T, atol=1e-6):
            raise TransferError("effect/rigid/metric 必须对称")
        if np.linalg.eigvalsh(metric.astype(np.float64)).min() < -1e-9:
            raise TransferError("effect/rigid/metric 必须半正定——它是一个度量")
        for name in ("effect/rigid/scale_m", "effect/surface_state/scale"):
            value = arrays[name]
            if value.shape != () or not np.isfinite(value) or value <= 0:
                raise TransferError(f"{name} 必须是正的标量刻度")

        support = arrays["region/support"]
        if np.any(support < 0) or np.any(support > arrays["support/episodes"][:, None]):
            raise TransferError("region/support 必须在 [0, support/episodes] 内")
        duty = arrays["region/duty"]
        if np.any(duty < 0) or np.any(duty > 1.0 + 1e-6):
            raise TransferError("region/duty 必须在 [0,1] 内")
        if np.any((support == 0) & (duty > 0)):
            raise TransferError("没有 episode 支持的 cell 不能有非零 duty")
        # 接触条件化字段必须与 support 一致：没人碰过的 cell 上一律为 0，
        # 有人碰过的 cell 不得被静默当成"力为零"（v1 的 [q10,q90]=[0,0] 就是这么来的）。
        for name in ("engage/dir/mean", "mech/traction_obj/median",
                     "mech/moment_density_obj/median"):
            if np.any(np.linalg.norm(arrays[name], axis=2)[support == 0] != 0):
                raise TransferError(f"{name} 在 support==0 的 cell 上必须是 0")
        if np.any(arrays["mode/slip_speed/median"][support == 0] != 0):
            raise TransferError("mode/slip_speed 在 support==0 的 cell 上必须是 0")
        for name in ("mech/traction_obj", "mech/moment_density_obj", "mode/slip_speed"):
            lo, hi = arrays[f"{name}/lo"], arrays[f"{name}/hi"]
            if np.any(hi < lo):
                raise TransferError(f"{name} 的 hi 必须 ≥ lo")
            if np.any((lo > arrays[f"{name}/median"]) | (hi < arrays[f"{name}/median"])):
                raise TransferError(f"{name} 的中位数必须落在 [lo,hi] 内")
        if np.any(arrays["region/allowed"] & (arrays["region/mass/mean"] <= 0)):
            raise TransferError("region/allowed 不能包含 mass 为零的 cell")
        if np.any(arrays["mode/prob"][support == 0] != 0):
            raise TransferError("mode/prob 在 support==0 的 cell 上必须是 0")
        concentration = arrays["engage/concentration"]
        if np.any(concentration < -1e-6) or np.any(concentration > 1.0 + 1e-6):
            raise TransferError("engage/concentration 必须在 [0,1] 内")
        if not np.array_equal(arrays["engage/valid"], concentration > 1e-8):
            raise TransferError("engage/valid 必须与 engage/concentration > 0 一致")

        region = arrays["region/mass/mean"]
        nonempty = region.sum(axis=1) > 0
        if np.any(np.abs(region[nonempty].sum(axis=1) - 1.0) > 2e-5):
            raise TransferError("非空 region/mass/mean 每个时间格必须归一化")
        mode_sum = arrays["mode/prob"].sum(axis=2)
        occupied = support > 0
        if np.any(np.abs(mode_sum[occupied] - 1.0) > 2e-5):
            raise TransferError("有支持的 cell 上 mode/prob 必须归一化")
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise TransferError("产物中不允许 NaN/Inf；无支持位置应由 support/duty 表示")

    def executor_arrays(self) -> dict[str, np.ndarray]:
        """返回 E-I 可以读取的完整、且仅限 allowlist 的数组。"""
        self.validate()
        return {name: self.arrays[name] for name in sorted(EXECUTOR_ARRAYS)}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"无法 JSON 序列化：{type(value).__name__}")


def save_transfer(transfer: InteractionTransfer, path: str | os.PathLike[str]) -> str:
    """原子地保存一个 interaction transfer。"""
    transfer.validate()
    destination = Path(path)
    if destination.suffix != ".npz":
        raise TransferError("interaction transfer 路径必须以 .npz 结尾")
    destination.parent.mkdir(parents=True, exist_ok=True)
    meta = json.dumps(transfer.meta, ensure_ascii=False, sort_keys=True,
                      default=_json_default)
    payload: dict[str, Any] = {META_KEY: np.asarray(meta)}
    payload.update({name: np.asarray(value) for name, value in transfer.arrays.items()})
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, destination)
    return str(destination.resolve())


def load_transfer(path: str | os.PathLike[str]) -> InteractionTransfer:
    """读取并校验一个 interaction transfer。"""
    with np.load(path, allow_pickle=False) as data:
        if META_KEY not in data.files or data[META_KEY].ndim != 0:
            raise TransferError(f"{path} 缺少标量 {META_KEY}")
        meta = json.loads(str(data[META_KEY].item()))
        arrays = {name: np.array(data[name], copy=True)
                  for name in data.files if name != META_KEY}
    transfer = InteractionTransfer(meta=meta, arrays=arrays)
    transfer.validate()
    return transfer


# ---------------------------------------------------------------- 聚合

def _nan_stat(values: np.ndarray, stat: str) -> np.ndarray:
    """沿 episode 轴统计，完全无支持的位置填 0，由 companion mask/count 区分。"""
    # NumPy 对"所有 episode 在这里都无支持"的合法位置发 RuntimeWarning；
    # companion support/count 已明确表达它，统计数组按契约填 0。
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if stat == "mean":
            result = np.nanmean(values, axis=0)
        elif stat == "median":
            result = np.nanmedian(values, axis=0)
        elif stat == "q10":
            result = np.nanquantile(values, 0.10, axis=0)
        elif stat == "q90":
            result = np.nanquantile(values, 0.90, axis=0)
        else:  # pragma: no cover - internal programming error
            raise ValueError(stat)
    return np.nan_to_num(result, nan=0.0).astype(np.float32)


def _reference_scale(centre: np.ndarray, occupied: np.ndarray) -> float:
    """本任务 traction / moment 的量级，用来定尺度地板。

    ⚠️ 地板不能拍一个绝对值：训练集上离散度接近零的 cell 会被地板绑架，标定出来的
    标量完全由它们决定。D-71 实测用 1 N/m² 的绝对地板与用相对地板，四族的排序都变了。
    """
    magnitude = np.linalg.norm(centre, axis=-1)
    live = occupied & (magnitude > 0)
    return float(np.median(magnitude[live])) if live.any() else 1.0



def _positive_scale(values: np.ndarray) -> float:
    """一路 effect 通道的刻度：非零量的 p90。整路恒为零时返回 1.0。

    返回 1.0 而不是 0 是有意的：擦拭的 ``effect/rigid`` 与抽屉/旋钮的
    ``effect/surface_state`` 整条恒为零，除以 0 会让 ``r_effect`` 变成 NaN 而不是
    "这一路没有要求"。恒零的那一路分子也恒为零，所以除以 1 正好等价于不贡献。
    """
    finite = np.asarray(values, dtype=np.float64).ravel()
    finite = finite[np.isfinite(finite) & (finite > 0)]
    return float(np.percentile(finite, 90)) if len(finite) else 1.0


def _bincount2d(index: np.ndarray, weights: np.ndarray, size: int,
                shape: tuple[int, ...]) -> np.ndarray:
    """按最后一维逐通道 bincount，再 reshape 回 (B, S, C)。"""
    channels = weights.shape[1] if weights.ndim == 2 else 1
    flat = np.stack([np.bincount(index, weights=weights[:, c] if channels > 1 else weights,
                                 minlength=size)
                     for c in range(channels)], axis=1)
    return flat.reshape(*shape, channels) if channels > 1 else flat.reshape(shape)


def _fine_force_field(fine_idx: np.ndarray, force: np.ndarray, frame_of: np.ndarray,
                      surface: Surface, sigma: float, k: int
                      ) -> tuple[np.ndarray, np.ndarray]:
    """把每帧的接触力用**固定物理带宽**的同部件核散射到冻结表面点。

    返回 ``(key, f)``：``key = frame * S + j`` 的升序唯一键，``f`` 是该帧该表面点
    分到的力 (M, 3)。合力逐帧严格守恒（核每行权重和为 1）。

    为什么核心在这里：traction 要是"每平方米多少牛"，那个平方米必须是**物理尺度**。
    把接触力整个塞给最近的采样点再除以采样格面积，分母就是我们随手选的分辨率——
    实测换一档格子数，数字按 1/面积 线性变（P-68）。核带宽 ``sigma`` 只跟拟传感
    taxel pitch 走，与 ``LEVELS`` 无关，于是同一份示教在 64 / 256 / 1024 档上
    读出同一个 traction。

    核中心取**接触点被归到的那个采样点**而不是接触点的真实坐标：误差上界是半个
    采样 pitch（~1.5 mm）且远小于 sigma，换来的是整张核表可以按 surface 预计算
    并缓存。这个近似必须写下来，因为它是"traction 是在 sigma 尺度上平滑后的面密度"
    这句话的确切含义的一部分。
    """
    kern_idx, kern_w = surface.scatter_kernel(sigma, k)
    neighbour = kern_idx[fine_idx]                       # (K, k)
    weight = kern_w[fine_idx]                            # (K, k)
    keep = weight > 0
    spread = force[:, None, :] * weight[:, :, None]      # (K, k, 3)
    flat_key = (frame_of[:, None] * surface.n_points + neighbour)[keep]
    flat_force = spread[keep]
    key, inverse = np.unique(flat_key, return_inverse=True)
    field = np.stack([np.bincount(inverse, weights=flat_force[:, c], minlength=len(key))
                      for c in range(3)], axis=1)
    return key, field


def _cell_traction(*, fine_idx: np.ndarray, force: np.ndarray, frame_of: np.ndarray,
                   position_of: np.ndarray,
                   bin_of: np.ndarray, coarse: np.ndarray, surface: Surface,
                   n_bins: int, n_surface: int, contact_frames: np.ndarray,
                   occupied: np.ndarray, cell_area: np.ndarray,
                   pooling: str, sigma: float, kernel_k: int) -> np.ndarray:
    """每个命令格 × 表面 cell 的 object-frame traction (N/m²)。

    三层，每层的"对哪些样本求平均"都必须说得出来（P-59）：

    1. **帧内**：接触力散射成表面点上的力场 ``f_j``，``t_j = f_j / a_j``；
    2. **格内**：把 ``t_j`` 汇成该 cell 这一帧的一个值（``pooling`` 决定怎么汇）；
    3. **跨帧**：只在**最近点归属**判定该 cell 有接触的那些帧上取平均，
       除数正是 ``contact_frames``——与 ``engage`` / ``mode`` 用的是同一批帧，
       D-66 的"同一个 cell 的所有字段来自同一批帧"因此仍然成立。

    ``nearest_area`` 保留 v2 的实现，只为了让 ``tools/s5_units_probe.py`` 能把
    新旧放在同一张表里对照；它不是可选配置，见 D-72。
    """
    if pooling not in TRACTION_POOLINGS:
        raise TransferError(f"未知的 traction 池化方式 {pooling!r}")
    shape = (n_bins, n_surface)
    size = n_bins * n_surface
    if not len(frame_of):
        # 整条 episode 一个有效接触都没有。occupied 全 False，各字段照 NaN 填，
        # 与其余接触条件化字段一致（不是 0——"没碰过"不等于"力为零"，P-59）。
        return np.full(shape + (3,), np.nan)

    if pooling == "contact_kernel":
        # **现行做法（D-79）。** 直接在接触点自己的位置上求核和：
        #     t(x_k) = Σ_m F_m · G_σ(x_k - x_m)，  G_σ = exp(-d²/2σ²)/(2πσ²)
        # 再按 |F| 加权池化到格、按接触帧平均。
        #
        # 为什么不再经过冻结表面点：
        #
        # 1. **它与在线实现（`ei_reward.surface_traction`）是同一个式子**，不是
        #    "应该差不多"。先前的做法先散射到 16384 个表面点再在格内按 |f| 加权平均，
        #    实测比在线低 1.48~1.75×（相关 0.93~0.98，是干净的系统偏差）——
        #    于是一条**完美复现 source** 的轨迹在线上会读出 1.6 倍的 traction，
        #    落在指令盒外面，`r_mech` 反过来惩罚正确行为。这种错不会报错；
        # 2. 它连**精细采样密度**也不依赖了。散射版的值取决于表面采样了 4096 还是
        #    16384 个点（一个我从没测过的隐含依赖），而这一版只依赖 σ；
        # 3. 物理上更对：命令要说的是"你按住的地方该有多大面密度"，那就该在
        #    **力实际所在的位置**取值，而不是把整个 cell（含没有接触的大片）平均进去。
        frame_key = np.unique(frame_of)
        order = np.searchsorted(frame_key, frame_of)
        traction = np.zeros((len(force), 3))
        for f_index in range(len(frame_key)):
            rows = np.flatnonzero(order == f_index)
            if not len(rows):
                continue
            delta = position_of[rows][:, None, :] - position_of[rows][None, :, :]
            kernel = np.exp(-(delta ** 2).sum(-1) / (2.0 * sigma ** 2)) * _GAUSS_NORM
            traction[rows] = kernel @ force[rows]
        mass = np.linalg.norm(force, axis=1)
        flat = bin_of * n_surface + coarse
        denominator = np.bincount(flat, weights=mass, minlength=size)
        numerator = _bincount2d(flat, mass[:, None] * traction, size, shape)
        denominator = denominator.reshape(shape)[..., None]
        out = np.full(shape + (3,), np.nan)
        np.divide(numerator, denominator, out=out,
                  where=occupied[..., None] & (denominator > 0))
        return out

    if pooling == "nearest_area":
        force_sum = _bincount2d(bin_of * n_surface + coarse, force, size, shape)
        out = np.full(shape + (3,), np.nan)
        np.divide(force_sum, contact_frames[..., None], out=out, where=occupied[..., None])
        return out / cell_area[None, :, None]

    n_points = surface.n_points
    parent = np.asarray(surface.parent[n_surface])
    area_fine = np.asarray(surface.area, dtype=np.float64)
    key, field = _fine_force_field(fine_idx, force, frame_of, surface, sigma, kernel_k)
    frame = key // n_points
    point = key % n_points
    traction_fine = field / area_fine[point][:, None]
    cell = parent[point]

    # 帧内 → 每个 (frame, cell) 一个值。只保留最近点归属认可的那些 (frame, cell)：
    # 那是 contact_frames 数的同一批，除数与分子才对得上。
    assigned = np.unique(frame_of.astype(np.int64) * n_surface + coarse)
    pair = frame.astype(np.int64) * n_surface + cell
    live = np.isin(pair, assigned)
    if pooling == "kernel_point":
        # 代表点就是 fine 点 c 本身（LEVELS 是采样序列的前缀，D-68 之后仍成立）。
        live &= point == cell
    pair, traction_fine, field = pair[live], traction_fine[live], field[live]

    unique_pair, inverse = np.unique(pair, return_inverse=True)
    if pooling == "kernel_forcew":
        mass = np.linalg.norm(field, axis=1)
        denominator = np.bincount(inverse, weights=mass, minlength=len(unique_pair))
        numerator = np.stack(
            [np.bincount(inverse, weights=mass * traction_fine[:, c],
                         minlength=len(unique_pair)) for c in range(3)], axis=1)
        per_pair = np.divide(numerator, denominator[:, None],
                             out=np.zeros_like(numerator), where=denominator[:, None] > 0)
    elif pooling == "kernel_area":
        summed = np.stack([np.bincount(inverse, weights=field[:, c],
                                       minlength=len(unique_pair)) for c in range(3)], axis=1)
        per_pair = summed / cell_area[unique_pair % n_surface][:, None]
    else:                                                   # kernel_point
        per_pair = np.stack([np.bincount(inverse, weights=traction_fine[:, c],
                                         minlength=len(unique_pair))
                             for c in range(3)], axis=1)

    # 跨帧 → 每个 (bin, cell) 一个值。
    pair_frame = unique_pair // n_surface
    pair_cell = unique_pair % n_surface
    frame_to_bin = np.full(int(frame_of.max()) + 1, -1, dtype=np.int64)
    frame_to_bin[frame_of] = bin_of
    flat = frame_to_bin[pair_frame] * n_surface + pair_cell
    if np.any(frame_to_bin[pair_frame] < 0):
        raise TransferError("散射出的帧不在命令轴上——归属与分格用了不同的帧集合")
    total = _bincount2d(flat, per_pair, size, shape)
    out = np.full(shape + (3,), np.nan)
    np.divide(total, contact_frames[..., None], out=out, where=occupied[..., None])
    return out


def _episode_summary(record: EpisodeRecord, surface: Surface, *, budget: tuple[int, ...],
                     n_bins: int, n_surface: int,
                     pooling: str = TRACTION_POOLING,
                     sigma: float = SCATTER_SIGMA, kernel_k: int = SCATTER_K
                     ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """先汇总单条 episode；之后跨 episode 统计才能保证采集者等权。

    每个 cell 的 ``engage`` / ``mode`` / ``traction`` / ``moment_density`` 一律只用
    **该 cell 实际有接触的那些帧**，并配一份 ``contact_frames``。v1 让 traction 用
    全部帧而其余用接触帧，同一个 cell 上不同字段来自不同的帧集合。
    """
    a = record.arrays
    episode_id = record.meta["episode_id"]
    valid = np.asarray(a["valid_s4"], dtype=bool)
    index = bin_index(record, surface, budget=budget)

    idx = np.asarray(a["region/point_idx"], dtype=np.int64)
    slot_valid = np.asarray(a["region/valid"], dtype=bool)
    on_surface = np.asarray(a["region/on_surface"], dtype=bool)
    weight = np.asarray(a["region/weight"], dtype=np.float64)
    force = np.asarray(a["mech/force_obj"], dtype=np.float64)
    position = np.asarray(a["region/pos_obj"], dtype=np.float64)
    engage = np.asarray(a["engage/dir"], dtype=np.float64)
    mode = np.asarray(a["mode/label"], dtype=np.int64)
    if idx.shape != weight.shape or force.shape != idx.shape + (3,):
        raise TransferError(f"episode {episode_id} 的 contact slots 形状不一致")

    parent = surface.parent[n_surface]
    cell_area = np.bincount(parent, weights=np.asarray(surface.area, dtype=np.float64),
                            minlength=n_surface)
    if np.any(cell_area <= 0):
        raise TransferError("低分辨率 surface 出现零面积 cell")
    representative = np.asarray(surface.points[:n_surface], dtype=np.float64)

    live = slot_valid & on_surface & (weight > 0) & valid[:, None] & (index >= 0)[:, None]
    if np.any(idx[live] < 0) or np.any(idx[live] >= surface.n_points):
        raise TransferError(f"episode {episode_id} 的 surface index 越界")
    frame_of, slot_of = np.nonzero(live)
    coarse = parent[idx[frame_of, slot_of]]
    bin_of = index[frame_of]
    flat = bin_of * n_surface + coarse
    size = n_bins * n_surface
    shape = (n_bins, n_surface)

    live_mode = mode[frame_of, slot_of]
    if np.any((live_mode < 0) | (live_mode >= MODE_COUNT)):
        raise TransferError(f"episode {episode_id} 有未知 mode")
    live_weight = weight[frame_of, slot_of]
    live_force = force[frame_of, slot_of]
    live_moment = np.cross(position[frame_of, slot_of] - representative[coarse], live_force)

    region_sum = _bincount2d(flat, live_weight, size, shape)
    engage_sum = _bincount2d(flat, live_weight[:, None] * engage[frame_of, slot_of], size, shape)
    force_sum = _bincount2d(flat, live_force, size, shape)
    moment_sum = _bincount2d(flat, live_moment, size, shape)
    mode_sum = np.bincount(flat * MODE_COUNT + live_mode,
                           minlength=size * MODE_COUNT).reshape(n_bins, n_surface, MODE_COUNT)
    mode_sum = mode_sum.astype(np.float64)

    # 一帧里可能有多个接触点落进同一个 cell，按 slot 计数会把"接触了多少帧"数大。
    unique = np.unique(frame_of * n_surface + coarse)
    contact_frames = np.bincount(index[unique // n_surface] * n_surface + (unique % n_surface),
                                 minlength=size).reshape(shape).astype(np.float64)

    n_frames = np.bincount(index[valid], minlength=n_bins).astype(np.float64)
    present = n_frames > 0
    occupied = contact_frames > 0

    def conditioned(total: np.ndarray) -> np.ndarray:
        out = np.full(total.shape, np.nan)
        divisor = contact_frames[..., None] if total.ndim == 3 else contact_frames
        np.divide(total, divisor, out=out, where=(occupied[..., None] if total.ndim == 3
                                                  else occupied))
        return out

    traction = _cell_traction(
        fine_idx=idx[frame_of, slot_of], force=live_force, frame_of=frame_of,
        position_of=position[frame_of, slot_of],
        bin_of=bin_of, coarse=coarse, surface=surface, n_bins=n_bins,
        n_surface=n_surface, contact_frames=contact_frames, occupied=occupied,
        cell_area=cell_area, pooling=pooling, sigma=sigma, kernel_k=kernel_k)
    # ⚠️ moment_density 是**粗粒化残差**，不是分辨率无关的物理场：它是 cell 内的力
    # 相对代表点的力矩，力臂天然随 cell 边长走，格子越细它越趋近 0。它照旧用接触点的
    # **真实坐标**算（D-64 要的是 cell wrench 的记账项，核散射反而会把力臂抹掉）。
    # 因此它不能跨分辨率比较，S4.5 扫描时必须单列——见 D-72 与 out/s5/units_probe.txt。
    moment_density = conditioned(moment_sum) / cell_area[None, :, None]
    # 方向按接触力权重平均后再归一化；没有接触的 cell 留 NaN，跨 episode 统计时排除。
    engage_mean = np.full(shape + (3,), np.nan)
    np.divide(engage_sum, region_sum[..., None], out=engage_mean, where=occupied[..., None])
    norm = np.linalg.norm(np.nan_to_num(engage_mean), axis=2, keepdims=True)
    engage_mean = np.divide(engage_mean, norm, out=np.full(engage_mean.shape, np.nan),
                            where=(norm > 1e-12) & occupied[..., None])
    # 连续滑移速度：``mode/prob`` 的四档是拿 SLIP_SPEED_MIN 切出来的，实测把阈值在
    # 1~10 mm/s 之间换，抽屉/旋钮「黏住」占的力比例摆动 39 个百分点——那个标签基本是
    # 阈值的产物。S4 侧一直存着连续量，但 v3 之前**只有离散标签进 payload**，于是
    # `r_mode` 只能追那个任意约定（P-69）。加权口径与 engage 完全一致：同一批帧、
    # 同一个法向力权重，D-66 的不变量因此仍然成立。
    slip_mean = np.full(shape + (1,), np.nan)
    slip_sum = _bincount2d(flat, live_weight * np.asarray(
        a["mode/pose_slip"], dtype=np.float64)[frame_of, slot_of], size, shape)
    np.divide(slip_sum[..., None], region_sum[..., None], out=slip_mean,
              where=occupied[..., None])

    mode_prob = np.full(shape + (MODE_COUNT,), np.nan)
    mode_total = mode_sum.sum(axis=2, keepdims=True)
    np.divide(mode_sum, mode_total, out=mode_prob, where=occupied[..., None] & (mode_total > 0))

    region = np.full(shape, np.nan)
    row_total = region_sum.sum(axis=1, keepdims=True)
    np.divide(region_sum, row_total, out=region, where=row_total > 0)

    duty = np.zeros(shape)
    np.divide(contact_frames, n_frames[:, None], out=duty, where=present[:, None])

    rigid = np.asarray(a["effect/rigid"], dtype=np.float64)
    state = np.asarray(a["effect/surface_state"], dtype=np.float64)
    future_valid = np.asarray(a["effect/future_valid"], dtype=bool)
    if rigid.shape[:2] != future_valid.shape or state.shape[:2] != future_valid.shape:
        raise TransferError(f"episode {episode_id} 的 effect future mask 不匹配")
    horizon = rigid.shape[1]
    effect = {}
    for name, values in (("rigid", rigid), ("state", state)):
        summed = np.full((n_bins, horizon, values.shape[2]), np.nan)
        count = np.zeros((n_bins, horizon), dtype=np.int32)
        for hi in range(horizon):
            frames = np.flatnonzero(valid & future_valid[:, hi])
            if not len(frames):
                continue
            hits = np.bincount(index[frames], minlength=n_bins).astype(np.float64)
            totals = _bincount2d(index[frames], values[frames, hi], n_bins, (n_bins,))
            live_bins = hits > 0
            summed[live_bins, hi] = totals[live_bins] / hits[live_bins, None]
            count[live_bins, hi] = 1
        effect[name] = summed
        effect[f"{name}_count"] = count

    summary = {
        "present": present,
        "region": region,
        "contact_frames": contact_frames,
        "duty": duty,
        "engage": engage_mean,
        "mode": mode_prob,
        "slip": slip_mean,
        "traction": traction,
        "moment_density": moment_density,
        "rigid": effect["rigid"],
        "rigid_count": effect["rigid_count"],
        "state": effect["state"],
        "state_count": effect["state_count"],
    }
    return summary, _projection_diagnostics(record, valid, live, force_sum, moment_sum,
                                            representative, index, n_bins)


def _projection_diagnostics(record: EpisodeRecord, valid: np.ndarray, live: np.ndarray,
                            force_sum: np.ndarray, moment_sum: np.ndarray,
                            representative: np.ndarray, index: np.ndarray,
                            n_bins: int) -> dict[str, float]:
    """表面投影的**完整性**残差——不是"重建误差"。

    ⚠️ cell 内先算相对代表点的局部力矩、再与 ``cross(representative, F)`` 相加，
    在代数上恒等于直接对接触点求力矩。所以拿它跟自己对拍只会得到浮点精度级的数，
    那不构成任何验证（v1 的 README 把 6e-6 N 当成了 6D wrench 重建的证据）。

    有信息量的是：被 ``on_surface`` / ``weight>0`` 滤掉的接触占了多少力，以及由此
    与 S4 **独立**记下的 ``mech/wrench_obj`` 差多少。滤掉的力越多，这份表面表示
    就越不足以代表物体实际受到的作用。
    """
    a = record.arrays
    force = np.asarray(a["mech/force_obj"], dtype=np.float64)
    slot_valid = np.asarray(a["region/valid"], dtype=bool) & valid[:, None]
    magnitude = np.linalg.norm(force, axis=2)
    total_mass = float(magnitude[slot_valid].sum())
    dropped_mass = float(magnitude[slot_valid & ~live].sum())

    raw = np.asarray(a["mech/wrench_obj"], dtype=np.float64)[valid]
    kept_force = force_sum.sum(axis=(0, 1))
    kept_torque = (np.cross(representative[None, :, :], force_sum).sum(axis=(0, 1))
                   + moment_sum.sum(axis=(0, 1)))
    return {
        "projection_residual_force_N": float(np.linalg.norm(kept_force - raw[:, :3].sum(axis=0))),
        "projection_residual_torque_Nm": float(
            np.linalg.norm(kept_torque - raw[:, 3:].sum(axis=0))),
        "dropped_force_mass_fraction": float(dropped_mass / total_mass) if total_mass > 0 else 0.0,
        "valid_frame_fraction": float(valid.mean()),
        "empty_bin_fraction": float((np.bincount(index[valid], minlength=n_bins) == 0).mean()),
    }


def nested_region_sets(region: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """每个命令格上按 region mass 降序的 cell 次序及累计占比。

    ``A(τ)`` = 每格取累计占比刚好达到 τ 的前缀，是**面积最小**的超水平集，
    并且随 τ 嵌套——正是这一点让 split conformal 可以直接对 τ 做。
    """
    orders, cumulative = [], []
    for row in region:
        total = row.sum()
        if total <= 0:
            orders.append(np.zeros(0, dtype=np.int64))
            cumulative.append(np.zeros(0))
            continue
        order = np.argsort(-row, kind="stable")
        orders.append(order)
        cumulative.append(np.cumsum(row[order]) / total)
    return orders, cumulative


def region_allowed_at(orders, cumulative, tau: float, n_surface: int) -> np.ndarray:
    allowed = np.zeros((len(orders), n_surface), dtype=bool)
    for b, cum in enumerate(cumulative):
        if not len(cum):
            continue
        keep = int(np.searchsorted(cum, tau) + 1)
        allowed[b, orders[b][:keep]] = True
    return allowed


def _weighted_threshold(scores: np.ndarray, weights: np.ndarray, target: float) -> float:
    """让 ``target`` 比例的力加权接触点落进集合所需要的最小阈值。"""
    order = np.argsort(scores)
    cumulative = np.cumsum(weights[order]) / weights.sum()
    index = int(np.searchsorted(cumulative, target))
    return float(scores[order][min(index, len(order) - 1)])


def _conformal_quantile(required: np.ndarray, target: float) -> float:
    """split conformal 的 ``⌈(n+1)·target⌉/n`` 分位数。

    给的是 exchangeability 下的 **marginal** coverage，不是任意子群的条件保证
    （D-59）。子群 coverage 必须另报。
    """
    finite = np.sort(required[np.isfinite(required)])
    n = len(required)
    if n == 0:
        return float("nan")
    rank = int(np.ceil((n + 1) * target))
    return float(finite[rank - 1]) if rank <= len(finite) else float("inf")


def _calibrate_region(region: np.ndarray, summaries: list[dict[str, np.ndarray]],
                      target: float) -> float:
    orders, cumulative = nested_region_sets(region)
    lengths = np.array([[int(np.searchsorted(cum, tau) + 1) if len(cum) else 0
                         for cum in cumulative] for tau in TAU_GRID], dtype=np.int64)
    required = []
    for summary in summaries:
        episode = np.nan_to_num(summary["region"], nan=0.0)
        total = episode.sum()
        if total <= 0:
            continue
        inside = np.zeros(len(TAU_GRID))
        for b, order in enumerate(orders):
            if not len(order):
                continue
            cumulative_mass = np.concatenate([[0.0], np.cumsum(episode[b][order])])
            inside += cumulative_mass[np.minimum(lengths[:, b], len(order))]
        hit = np.flatnonzero(inside / total >= POINT_MASS_TARGET)
        required.append(float(TAU_GRID[hit[0]]) if len(hit) else float("inf"))
    return _conformal_quantile(np.asarray(required), target)


def _calibrate_box(centre: np.ndarray, scale: np.ndarray, key: str,
                   summaries: list[dict[str, np.ndarray]], support: np.ndarray,
                   target: float) -> float:
    """物体系三维轴对齐盒的联合标定标量（D-71）。

    形状选择是实测出来的：四族候选在同一 coverage 目标下比体积，方向锥形式体积大
    2.4~56 倍且无收益。见 ``tools/s5_mech_setform.py`` 与 ``out/s5/mech_setform.txt``。
    """
    required = []
    for summary in summaries:
        values = summary[key]
        mass = np.nan_to_num(summary["region"], nan=0.0)
        live = support & np.isfinite(values).all(axis=2) & (mass > 0)
        if not live.any():
            continue
        score = np.max(np.abs(np.nan_to_num(values) - centre) / scale, axis=-1)
        required.append(_weighted_threshold(score[live], mass[live], POINT_MASS_TARGET))
    return _conformal_quantile(np.asarray(required), target)


def episode_summary(record: EpisodeRecord, surface: Surface, *, budget: tuple[int, ...],
                    n_bins: int, n_surface: int
                    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """把一条 episode 投影到**已冻结的**命令轴上。

    S5 评估必须用 artifact 自己的 ``aggregation.phase_budget`` 来投影留出 episode，
    否则比的是两条不同的轴，coverage 数没有意义。
    """
    return _episode_summary(record, surface, budget=budget, n_bins=n_bins,
                            n_surface=n_surface)


def build_transfer(records: Iterable[EpisodeRecord], *, n_bins: int = 32,
                   n_surface: int = 256, transfer_id: str | None = None,
                   surface: Surface | None = None,
                   phase_floor: int = PHASE_FLOOR,
                   budget: tuple[int, ...] | None = None,
                   calibration: Iterable[EpisodeRecord] | None = None,
                   region_smooth: float = REGION_SMOOTH
                   ) -> InteractionTransfer:
    """从同一任务/物体/几何的多条成功 S4 records 构造统计 interaction command。

    每条 episode 在每个命令格内先汇总，跨 episode 再取均值/分位数。因此一个
    500 帧的采集者不会比一个 200 帧的采集者拥有 2.5 倍权重。

    ``budget`` 给定时用它而不重算命令轴。**比较两份 envelope 时必须这样做**：
    每个子集自己算 budget 会得到不同的轴，逐格比出来的距离就没有意义。

    每个 cell 上的 ``engage`` / ``mode`` / ``traction`` / ``moment_density`` 只在
    **实际接触过该 cell 的 episode** 之间统计，配套的 ``region/support`` 说明那是
    几条。这样同一个 cell 的各字段来自同一组 episode，不会出现"这里该接触"与
    "这里的力必须为零"并存的指令（v1 的实测占比：抽屉 10%、擦拭 15%）。
    """
    if n_bins < 2:
        raise TransferError("n_bins 至少为 2")
    materialized = list(records)
    if len(materialized) < 2:
        raise TransferError("至少需要 2 条成功示教；单条轨迹不能代表多人/多次采集统计")

    def validate_record(record: EpisodeRecord) -> None:
        record.validate()
        if record.meta.get("schema_version") != "s4-record-v2":
            raise TransferError("只接受 s4-record-v2")
        if not record.meta.get("success", False):
            raise TransferError("失败/near-success record 不能混进成功 interaction 聚合")

    for record in materialized:
        validate_record(record)
    first = materialized[0].meta
    expected = {key: str(first.get(key, "nominal"))
                for key in ("task", "object", "geometry_variant")}
    expected_surface_hash = str(first.get("surface", {}).get("sha256"))
    obj, geom = expected["object"], expected["geometry_variant"]
    surface = surface or surface_for(obj, geom)
    if surface.sha256 != expected_surface_hash:
        raise TransferError("传入 surface 与 record meta 的 sha256 不一致")
    if n_surface not in surface.parent:
        raise TransferError(
            f"surface 没有 {n_surface} 点层级；可用 {sorted(surface.parent)}")
    for record in materialized:
        for key, expected_value in expected.items():
            actual = str(record.meta.get(key, "nominal"))
            if actual != expected_value:
                raise TransferError(
                    f"所有 records 的 {key} 必须一致：期望 {expected_value!r}，"
                    f"实际 {actual!r}")
        if str(record.meta.get("surface", {}).get("sha256")) != expected_surface_hash:
            raise TransferError("不能聚合不同冻结 surface hash 的 records")

    if budget is None:
        budget = phase_budget(materialized, surface, n_bins=n_bins, floor=phase_floor)
    elif sum(budget) != n_bins or len(budget) != N_PHASES:
        raise TransferError(f"给定 budget {budget} 与 n_bins={n_bins} 不一致")
    summaries: list[dict[str, np.ndarray]] = []
    episode_diagnostics: list[dict[str, float]] = []
    episode_ids: list[str] = []
    strategy_counts: dict[str, int] = {}
    implementation_counts: dict[str, int] = {}
    for record in materialized:
        summary, diagnostics = _episode_summary(
            record, surface, budget=budget, n_bins=n_bins, n_surface=n_surface)
        summaries.append(summary)
        episode_diagnostics.append(diagnostics)
        episode_ids.append(str(record.meta["episode_id"]))
        for counts, key in ((strategy_counts, "strategy_family"),
                            (implementation_counts, "implementation")):
            name = str(record.meta.get(key, "unknown"))
            counts[name] = counts.get(name, 0) + 1

    def stack(name: str) -> np.ndarray:
        return np.stack([summary[name] for summary in summaries], axis=0)

    present = stack("present")
    contact_frames = stack("contact_frames")
    support = (contact_frames > 0).sum(axis=0).astype(np.int32)
    occupied = support > 0

    region = stack("region")
    region_mean = _nan_stat(region, "mean")
    if region_smooth > 0:
        # region 是"接触会落在哪"的**概率密度估计**，原始形式是 256 格上的直方图。
        # ~150 条示教、每格每帧几个接触点，没被碰到的格概率恰好是 0，于是它在任何
        # τ 下都进不了允许集合，split conformal 的 τ* 发散到 inf（D-78）。
        # 平滑一个格是最小的修法。⚠️ 这与 traction 的核**不是一回事**：那个的带宽是
        # 物理尺度且不能跟分辨率走（P-68），这个平滑的是直方图的格。
        _, kernel = surface.cell_kernel(n_surface, region_smooth)
        region_mean = region_mean @ kernel.T
    row_sum = region_mean.sum(axis=1, keepdims=True)
    region_mean = np.divide(region_mean, row_sum, out=np.zeros_like(region_mean),
                            where=row_sum > 0)

    duty = np.where(contact_frames > 0, stack("duty"), np.nan)
    engage = stack("engage")
    engage_mean = _nan_stat(engage, "mean")
    # 各 episode 的方向已是单位矢量，跨 episode 平均后的合矢量长度就是集中度 ∈[0,1]。
    # 它是方向多峰性的直接度量；v1 把它阈值化成 bool 就丢掉了。
    concentration = np.linalg.norm(engage_mean, axis=2)
    engage_mean = np.divide(engage_mean, concentration[..., None],
                            out=np.zeros_like(engage_mean),
                            where=concentration[..., None] > 1e-8)

    mode_prob = _nan_stat(stack("mode"), "mean")
    mode_sum = mode_prob.sum(axis=2, keepdims=True)
    mode_prob = np.divide(mode_prob, mode_sum, out=np.zeros_like(mode_prob),
                          where=mode_sum > 0)

    def zero_unsupported(values: np.ndarray) -> np.ndarray:
        return np.where(occupied[..., None] if values.ndim == 3 else occupied, values, 0)

    slip = stack("slip")
    slip_median = zero_unsupported(_nan_stat(slip, "median"))
    slip_scale = np.maximum(
        zero_unsupported(_nan_stat(slip, "q90") - _nan_stat(slip, "q10")) / 2.0,
        MECH_FLOOR_RELATIVE * _reference_scale(slip_median, occupied))

    traction = stack("traction")
    moment_density = stack("moment_density")

    traction_median = zero_unsupported(_nan_stat(traction, "median"))
    traction_scale = np.maximum(
        zero_unsupported(_nan_stat(traction, "q90") - _nan_stat(traction, "q10")) / 2.0,
        MECH_FLOOR_RELATIVE * _reference_scale(traction_median, occupied))
    moment_median = zero_unsupported(_nan_stat(moment_density, "median"))
    moment_scale = np.maximum(
        zero_unsupported(_nan_stat(moment_density, "q90")
                         - _nan_stat(moment_density, "q10")) / 2.0,
        MECH_FLOOR_RELATIVE * _reference_scale(moment_median, occupied))

    cell_area = np.bincount(surface.parent[n_surface],
                            weights=np.asarray(surface.area, dtype=np.float64),
                            minlength=n_surface).astype(np.float32)

    # effect 的任务无关刻度（P-70）。metric 只由物体几何决定，用**全分辨率**表面点
    # 积分，因此与命令分辨率无关；两个 scale 取本 artifact 命令轴上的 p90，是
    # "这份指令要求的变化有多大"，让 r_effect 的两项可以相加。
    effect_metric = surface_metric(surface)
    rigid_median = _nan_stat(stack("rigid"), "median")
    effect_rigid_scale = _positive_scale(
        rigid_surface_displacement(np.nan_to_num(rigid_median), effect_metric))
    # ⚠️ `effect/surface_state` 存的是**被擦掉的 dirt 格数**（`_dirt_field` 把格数
    # 累加到表面点上），不是面积。乘表面 cell 面积会得到一个既不是格数也不是面积的
    # 东西——本机实测那样算出来的"擦净面积"是 1.24 m²，而整块板只有 0.30 m²。
    # 刻度按这一路自己的 L1 量级定，量纲在 r_effect 里被约掉，因此不需要换算。
    state_median = _nan_stat(stack("state"), "median")
    effect_state_scale = _positive_scale(
        np.abs(np.nan_to_num(state_median)).sum(axis=-1))

    arrays = {
        "command/fraction": ((np.arange(n_bins, dtype=np.float32) + 0.5) / n_bins),
        "command/valid": present.any(axis=0),
        "support/episodes": present.sum(axis=0).astype(np.int32),
        "surface/points_obj": np.asarray(surface.points[:n_surface], dtype=np.float32),
        "surface/normals_obj": np.asarray(surface.normals[:n_surface], dtype=np.float32),
        "surface/area": cell_area,
        "surface/part": np.asarray(surface.part[:n_surface], dtype=np.int8),
        "effect/rigid/median": _nan_stat(stack("rigid"), "median"),
        "effect/rigid/q10": _nan_stat(stack("rigid"), "q10"),
        "effect/rigid/q90": _nan_stat(stack("rigid"), "q90"),
        "effect/rigid/valid_count": stack("rigid_count").sum(axis=0).astype(np.int32),
        "effect/surface_state/median": _nan_stat(stack("state"), "median"),
        "effect/surface_state/q10": _nan_stat(stack("state"), "q10"),
        "effect/surface_state/q90": _nan_stat(stack("state"), "q90"),
        "effect/surface_state/valid_count": stack("state_count").sum(axis=0).astype(np.int32),
        "region/mass/mean": region_mean.astype(np.float32),
        "region/mass/q10": _nan_stat(region, "q10"),
        "region/mass/q90": _nan_stat(region, "q90"),
        "region/allowed": np.zeros(region_mean.shape, dtype=bool),   # 标定后填
        "region/support": support,
        "region/duty": np.where(occupied, _nan_stat(duty, "mean"), 0.0).astype(np.float32),
        "engage/dir/mean": zero_unsupported(engage_mean).astype(np.float32),
        "engage/concentration": np.where(occupied, concentration, 0.0).astype(np.float32),
        "engage/valid": np.where(occupied, concentration, 0.0) > 1e-8,
        "mode/prob": zero_unsupported(mode_prob).astype(np.float32),
        # 连续量。`mode/prob` 降级为可解释性/诊断字段，`r_mode` 用这一路——阈值只在
        # 画图和讲故事时出现，不进研究结论（P-69）。滑移速度非负，lo 因此夹到 0。
        "mode/slip_speed/median": slip_median.astype(np.float32),
        "mode/slip_speed/lo": np.maximum(slip_median - slip_scale, 0.0).astype(np.float32),
        "mode/slip_speed/hi": (slip_median + slip_scale).astype(np.float32),
        # effect 两路的任务无关归一化（P-70）。metric 把 (dp, dr) 换算成表面平均位移，
        # 两个 scale 让 r_effect 的两项都变成无量纲相对误差——权重在成功轨迹上标定，
        # 不拍（D-31 第 2 个洞）。
        "effect/rigid/metric": effect_metric.astype(np.float32),
        "effect/rigid/scale_m": np.float32(effect_rigid_scale),
        "effect/surface_state/scale": np.float32(effect_state_scale),
        "mech/traction_obj/median": traction_median.astype(np.float32),
        "mech/traction_obj/lo": (traction_median - traction_scale).astype(np.float32),
        "mech/traction_obj/hi": (traction_median + traction_scale).astype(np.float32),
        "mech/moment_density_obj/median": moment_median.astype(np.float32),
        "mech/moment_density_obj/lo": (moment_median - moment_scale).astype(np.float32),
        "mech/moment_density_obj/hi": (moment_median + moment_scale).astype(np.float32),
    }

    calibration_meta: dict[str, Any] = {
        "calibrated": False,
        "point_mass_target": POINT_MASS_TARGET,
        "target_coverage": TARGET_COVERAGE,
        "mech_floor_relative": MECH_FLOOR_RELATIVE,
        "region_smooth_cells": region_smooth,
        "note": ("未标定：region/allowed 退化成 τ=0.95 的描述性超水平集，"
                 "mech 的 lo/hi 只是 10/90 分位数。二者都没有覆盖保证，"
                 "不得直接当作 E-I 的跟踪目标（D-67 / D-71）"),
    }
    orders, cumulative = nested_region_sets(region_mean.astype(np.float64))
    tau = POINT_MASS_TARGET
    if calibration is not None:
        calibration_summaries = []
        for record in calibration:
            validate_record(record)
            for key, expected_value in expected.items():
                if str(record.meta.get(key, "nominal")) != expected_value:
                    raise TransferError(f"校准 record 的 {key} 与训练集不一致")
            if str(record.meta.get("surface", {}).get("sha256")) != expected_surface_hash:
                raise TransferError("校准 record 的 frozen surface hash 与训练集不一致")
            calibration_summaries.append(_episode_summary(
                record, surface, budget=budget, n_bins=n_bins, n_surface=n_surface)[0])
        if len(calibration_summaries) < 20:
            raise TransferError(
                f"校准集只有 {len(calibration_summaries)} 条，split conformal 不可靠；"
                "宁可输出未标定 artifact 并显式标记，也不要给一个假的保证")
        tau = _calibrate_region(region_mean.astype(np.float64), calibration_summaries,
                                TARGET_COVERAGE)
        k_traction = _calibrate_box(traction_median.astype(np.float64),
                                    traction_scale.astype(np.float64), "traction",
                                    calibration_summaries, occupied, TARGET_COVERAGE)
        k_moment = _calibrate_box(moment_median.astype(np.float64),
                                  moment_scale.astype(np.float64), "moment_density",
                                  calibration_summaries, occupied, TARGET_COVERAGE)
        k_slip = _calibrate_box(slip_median.astype(np.float64),
                                slip_scale.astype(np.float64), "slip",
                                calibration_summaries, occupied, TARGET_COVERAGE)
        for name, centre, scale, k in (
                ("mech/traction_obj", traction_median, traction_scale, k_traction),
                ("mech/moment_density_obj", moment_median, moment_scale, k_moment),
                ("mode/slip_speed", slip_median, slip_scale, k_slip)):
            if not np.isfinite(k):
                raise TransferError(
                    f"{name} 的标定标量发散：校准集里达不到 {POINT_MASS_TARGET:.0%} "
                    "接触点覆盖的 episode 比例已超过 α，任何缩放都不够")
            low = centre - k * scale
            arrays[f"{name}/lo"] = (np.maximum(low, 0.0) if name == "mode/slip_speed"
                                    else low).astype(np.float32)
            arrays[f"{name}/hi"] = (centre + k * scale).astype(np.float32)
        calibration_meta = {
            "calibrated": True,
            "num_episodes": len(calibration_summaries),
            "point_mass_target": POINT_MASS_TARGET,
            "target_coverage": TARGET_COVERAGE,
            "mech_floor_relative": MECH_FLOOR_RELATIVE,
            "region_tau": float(tau),
            "region_smooth_cells": float(region_smooth),
            "mech_traction_k": float(k_traction),
            "mech_moment_k": float(k_moment),
            "mode_slip_speed_k": float(k_slip),
            "set_form": "object_frame_axis_aligned_box",
            "claim": ("exchangeability 下的 marginal coverage；对策略/物理/几何子群"
                      "没有条件保证，子群 coverage 必须另报（D-59）"),
        }
    if not np.isfinite(tau) or tau >= 1.0:
        raise TransferError(
            f"region 的 τ*={tau} 顶到上界：允许集合就是整个表面，envelope 没有约束"
            "任何东西，coverage 100% 是空的")
    arrays["region/allowed"] = region_allowed_at(orders, cumulative, tau, n_surface)

    task = str(first["task"])
    worst = lambda key: max(item[key] for item in episode_diagnostics)  # noqa: E731
    meta = {
        "schema_version": TRANSFER_SCHEMA_VERSION,
        "transfer_id": transfer_id or f"{task}-{obj}-{geom}",
        # 审计字段：不会由 executor_arrays() 返回。
        "task": task,
        "object": obj,
        "geometry_variant": geom,
        "num_episodes": len(summaries),
        "episode_ids": episode_ids,
        "strategy_family_counts": strategy_counts,
        "implementation_counts": implementation_counts,
        "aggregation": {
            "alignment": "phase_segmented_activity_bins_offline_only",
            "phase_budget": list(budget),
            "phase_floor": phase_floor,
            "episode_weighting": "equal_after_within_episode_bin_summary",
            "cell_statistics": "contact_conditioned_over_contacting_episodes",
            "n_bins": n_bins,
            "quantiles": [0.10, 0.90],
            "claim": "descriptive_demonstration_statistics",
            "mechanics": ("per-cell contact-conditioned traction + local moment density; "
                          "the 6D wrench identity holds per episode before aggregation "
                          "and is NOT preserved by cross-episode medians/quantiles"),
        },
        "surface": {
            "sha256": surface.sha256,
            "source_n_points": surface.n_points,
            "command_n_points": n_surface,
            "parts": list(surface.parts),
            "total_area_m2": float(cell_area.sum()),
        },
        "calibration": calibration_meta,
        "diagnostics": {
            "projection_residual_force_N": worst("projection_residual_force_N"),
            "projection_residual_torque_Nm": worst("projection_residual_torque_Nm"),
            "dropped_force_mass_fraction_max": worst("dropped_force_mass_fraction"),
            "empty_bin_fraction_max": worst("empty_bin_fraction"),
            "episode_valid_frame_fraction": [item["valid_frame_fraction"]
                                             for item in episode_diagnostics],
            "cell_support_median": float(np.median(support[occupied])) if occupied.any() else 0.0,
            "cell_support_under_half_fraction": (
                float((support[occupied] < len(summaries) / 2).mean()) if occupied.any() else 0.0),
        },
    }
    transfer = InteractionTransfer(meta=meta, arrays=arrays)
    transfer.validate()
    return transfer
