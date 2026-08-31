"""E-I 的任务无关交互跟踪 reward（`plan/04` §5.1）。

五项：``r = w_e·r_effect + w_r·r_region + w_m·r_mode + w_f·r_mech + r_safety``。
禁止出现的东西写在 `plan/04` §5.1：任务 success bonus、dirt 清除量、以任务关节角
为目标的项、任何 task id 或任务分支。本模块**只吃指令张量与实测接触量**，
没有任何一处知道自己在开抽屉还是在擦板。

三件必须一次说清的事
--------------------

**一、跟踪项是 hinge，不是到中位数的距离。** ``region/allowed`` 与
``mech/*/{lo,hi}``、``mode/slip_speed/{lo,hi}`` 都是**标定过的允许集合**
（D-67 / D-71 / D-73），不是点目标。落在集合内的一切实现都同样满足规格，
所以 reward 在集合内必须**恰好为零**，只惩罚越界量。

这不是风格问题。若改成"离中位数越近越好"，C4（允许集合）就被悄悄变成了
C5（精确复现 source 的 traction），而 `plan/02` §6 的整个 C4 vs C5 对照就是要
回答"复制 source 是不是多余的"——把 C4 实现成 C5，那个实验直接失去意义。

**二、effect 两路各自归一化，不对 6 维取 L2**（D-74 / P-70）。``effect/rigid`` 里
混着米和弧度，每个任务恰好只有一路非零，两路量级差 20.5 倍。用 payload 里的
``effect/rigid/metric`` 把 ξ 换算成"物体表面平均移动了多远"，再各除以各自的刻度。

**三、在线 traction 必须与 S5 离线的定义是同一个估计量**（D-72）。离线是
"固定 4 mm 带宽的同部件核散射到冻结表面点，再在格内按 |f| 加权池化"。在线不可能
每步都对 16384 个表面点做散射，所以用**连续形式**：直接在接触点上求和
``t(x) = Σ_k F_k·G_σ(x-x_k)``，再按 |F| 加权池化到格。两者在斑块远小于格子时
应当一致——**这一条不是推理出来的，是 `tools/s6_reward_probe.py` 在真实记录上
逐格对拍出来的**，误差随产物一起报。若哪天两者对不上，reward 追的就不是
artifact 里写的那个量，而这件事不会报错。

**四、`r_region` 必须能取正值，否则最优策略是"永远别碰那个物体"。** ⭐
第一版四项**全部 ≤ 0**：hinge 项在集合内恰好为零，而 `r_region` 写成
``占比 − 1`` 也是 ≤ 0，且没接触时恰好为 0。于是"什么都不做"的回报是 0，
而任何真实接触都是负的。`tools/s6_smoke.py` 实测：**悬停 −0.43/步，
而一段大致压对了的开环执行 −278/步**。PPO 会稳稳地学会躲开物体，
而训练曲线会一路上涨——因为它确实在最大化这个 reward。

这不是数值问题，是**规格里唯一说"该往哪碰"的那一项被写成了只能罚不能奖**。
现在 `r_region` 是**带符号的占比** ∈[−1, +1]：力全落在 ``region/allowed`` 内 = +1、
全落在外面 = −1、没接触 = 0。它是整套 reward 里唯一为正的项，也是建立接触的
全部动力。相应地 `w_region` 必须大于 `w_mech + w_mode`，否则"压对地方但力学不完美"
仍然不如不碰；这一条由 `s6_smoke.py` 的"压对 > 悬停"那一格实测把关，不靠推理。

⚠️ 这**不违反** hinge 那条底线：mech / mode 落在标定集合内仍然恰好为 0，
C4 没有被偷偷实现成 C5。region 本来就不是允许集合上的 hinge，
`plan/04` §5.1 的原话就是"落在指令 region 内的接触法向力**占比**"。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

#: 与 `surfaces.SCATTER_SIGMA` 必须一致：它是 traction 的定义的一部分，
#: 不是一个可以在这边单独调的超参。
from it.interaction import SLIP_SPEED_MIN
from it.surfaces import SCATTER_SIGMA
from it.transfer import EFFECT_RIGID_DEADBAND_M

#: 越界量的饱和点，单位是**允许集合的宽度**。越出去 3 个集合宽度就算"完全不满足"，
#: 再远不额外加罚。盒宽本身来自 split conformal 标定（D-67/D-71），所以这个刻度
#: 不是凭空拍的；截断点是拍的，因此 `RewardTerms` 把**截断发生的比例**当诊断量报出来。
VIOLATION_CAP = 3.0

#: 高斯面密度核的归一化常数：∫ G dA = 1 要求除以 2πσ²。
_GAUSS_NORM = 1.0 / (2.0 * math.pi * SCATTER_SIGMA ** 2)
_EPS = 1e-9
# `EFFECT_RIGID_DEADBAND_M` 定义在 transfer 契约模块；离线 demand 与在线 achieved
# 共享一个来源，防止两个碰巧相同的常数日后各自漂移。


def effect_magnitude(rigid: torch.Tensor, state: torch.Tensor, *,
                     metric: torch.Tensor, scale_rigid: torch.Tensor,
                     scale_state: torch.Tensor) -> torch.Tensor:
    """把一对 (刚体增量, 表面状态增量) 换算成**无量纲**的 effect 幅度。

    Args:
        rigid: (N, 6) 物体系刚体增量 (dp, dr)。
        state: (N, L) 逐表面点的状态变化。
        metric: (N, 6, 6) `effect/rigid/metric`。
    """
    quad = torch.einsum("ni,nij,nj->n", rigid, metric, rigid).clamp_min(0.0).sqrt()
    rigid_term = torch.where(
        quad >= EFFECT_RIGID_DEADBAND_M,
        quad / scale_rigid.clamp_min(EFFECT_RIGID_DEADBAND_M),
        torch.zeros_like(quad))
    return (rigid_term
            + state.abs().sum(-1) / scale_state.clamp_min(_EPS))


def effect_deficit(achieved: torch.Tensor, demand: torch.Tensor) -> torch.Tensor:
    """本命令格的 effect 完成缺口 ∈[0,1]。0 = 已经做到，1 = 一点没动。

    **为什么不是"实测 effect 与指令 effect 之差"。** 那是我第一版的写法，dry-run 上
    `r_effect` 报到 −26647 且逐步线性增长，比其余三项大六个数量级——两个错叠在一起：

    1. **比的不是同一段时间**。`effect/rigid[b, 0]` 是未来 0.1 s 的位移，而实测取的是
       一个控制步（0.02 s）的位移，差一个 5 倍；更要紧的是**命令格的时长本来就是
       弹性的**（`plan/02` §5：执行器按自己的进度选窗口，允许有限时间缩放），
       所以指令 effect 根本不是一个"速率"，而是"在这一格里总共要发生多少变化"。
       拿它跟每步位移比，量纲上就不对；
    2. **除以了一个可能接近零的刻度**。探针物体上"按住不动"那类原语的 effect 中位数
       几乎为零，`effect/rigid/scale_m` 跟着趋零，随机策略把物体碰飞之后
       `got/scale` 直接炸掉。

    改成缺口之后两个问题一起没了：**分子分母用同一个刻度，刻度约掉**；而且它是
    **势函数式**的——悬停时缺口一直是 1（一直被罚），推进时缺口单调下降，
    没法靠悬停刷分（D-31 第 1 个洞）。取值有界 [0,1]，与其余三项同量级。

    这一格不要求物体变化时（``demand`` ≈ 0）缺口恒为 0：接近段与松开段本来就不该
    因为"物体没动"挨罚。
    """
    wanted = demand > 1e-9
    ratio = torch.where(wanted, achieved / demand.clamp_min(1e-9),
                        torch.ones_like(demand))
    return (1.0 - ratio.clamp(0.0, 1.0)) * wanted.float()


def nearest_cell(contact_pos: torch.Tensor, cell_points: torch.Tensor) -> torch.Tensor:
    """纯最近邻归格 (N,K,3),(N,S,3) -> (N,K)。不做同侧判定，只用来**定向**。

    用 ``|p|² − 2p·c + |c|²`` 展开而不是 ``(p−c)`` 相减：后者要物化一个
    (N,K,S,3) 的张量，2048 env × 16 接触 × 256 格时是 100 MB，而这一步每个
    控制步都要跑。
    """
    cross = torch.einsum("nkd,nsd->nks", contact_pos, cell_points)
    return ((contact_pos ** 2).sum(-1)[:, :, None]
            - 2.0 * cross + (cell_points ** 2).sum(-1)[:, None, :]).argmin(-1)


def normal_orientation_sign(physx_normal: torch.Tensor, outward_normal: torch.Tensor,
                            normal_force: torch.Tensor, valid: torch.Tensor
                            ) -> torch.Tensor:
    """PhysX 报的这一束力作用在**谁**身上：+1 = 法向与物体外法向同向。(N,)

    取决于刚体对在 PhysX 内部的次序，换个场景就翻——这正是 P-37 那条
    "接触法向的正负约定"。判据用**力加权**的一致性，而不是逐点投票：
    掠射接触的法向估计不稳，但它们的力也小。

    返回 0 表示"这一帧看不出来"（没有接触，或者一致性正负相消）。调用方应当
    **latch** 住上一次看得出来的值，不要每帧重判——P-49 就是这个洞：
    离散选择在回路里每步重算时，会在两个等价解之间随机横跳。
    """
    weight = normal_force.abs() * valid.float()
    agree = (physx_normal * outward_normal).sum(-1)
    return torch.sign((agree * weight).sum(-1))


def contact_force_on_object(normal_force: torch.Tensor, friction: torch.Tensor,
                            outward_normal: torch.Tensor, valid: torch.Tensor,
                            sign: torch.Tensor) -> torch.Tensor:
    """作用在**物体**上的接触力 (N,K,3)，口径与 S4 离线逐字相同。⭐

    S4（`it.interaction` 第 4 步）是这么构造的：

    1. 法向大小取 ``|normal_force|``——PhysX 报的是带符号的；
    2. 方向取**表面采样点的外法向**取负（压进物体）。用表面的几何法向而不是
       PhysX 报的法向，是 P-37 的解法：几何量没有约定问题；
    3. 摩擦乘 ``on_object = −sign``，把"这束力作用在采集体上"翻成"作用在物体上"。

    在线侧原来直接写的是 ``normals * normal_forces + friction_forces``——
    用的是 PhysX 的带符号法向力、PhysX 的原始法向、原始摩擦，**三处都与离线不同**。
    后果是 traction 的方向可能整体翻转，而指令盒 ``mech/traction_obj/{lo,hi}``
    是在离线口径上标定的：一条完美复现 source 的轨迹会落在盒外，``r_mech``
    反过来惩罚正确行为。**这是 P-72 的第三例**——`tools/s6_reward_probe.py`
    第一节喂进去的是**已经构造好的** ``mech/force_obj``，所以它只对了公式，
    从没走过这条从原始量构造的路径。
    """
    live = valid.float()[..., None]
    f_normal = normal_force.abs()[..., None] * (-outward_normal)
    return (f_normal - sign[:, None, None] * friction) * live


def assign_cells(contact_pos: torch.Tensor, contact_normal: torch.Tensor,
                 contact_valid: torch.Tensor, cell_points: torch.Tensor,
                 cell_normals: torch.Tensor) -> torch.Tensor:
    """把接触点归到命令格。(N,K,3) -> (N,K) long，无效槽位给 0。

    **同侧约束**：先把法向相背（``n·n_c ≤ 0``）的格子排除，再取最近。薄物体上纯几何
    最近邻会把正面的接触归到背面去——黑板厚 20 mm 而 256 档的粗粒 pitch 约 50 mm，
    比厚度还大。D-68 在离线侧修的是同一个问题（那里靠 `parent[level]` 的同部件约束），
    在线侧没有 part 标签可用，法向同侧是等价且无参数的判据（P-37 的同一条规矩）。
    """
    delta = contact_pos[:, :, None, :] - cell_points[:, None, :, :]
    distance = (delta ** 2).sum(-1)
    agree = torch.einsum("nkd,nsd->nks", contact_normal, cell_normals) > 0.0
    # 一个同侧的都没有（掠射接触、法向估计不稳）就退回纯最近邻，而不是丢掉这个接触：
    # 丢掉会让 region 外的接触在 reward 里凭空消失，那正好是要惩罚的东西。
    has_side = agree.any(-1, keepdim=True)
    masked = torch.where(agree | ~has_side, distance, torch.full_like(distance, float("inf")))
    index = masked.argmin(-1)
    return torch.where(contact_valid, index, torch.zeros_like(index))


def match_functional_region(contact_pos: torch.Tensor,
                            contact_normal: torch.Tensor,
                            contact_valid: torch.Tensor,
                            cell_points: torch.Tensor,
                            cell_normals: torch.Tensor,
                            cell_area: torch.Tensor,
                            allowed: torch.Tensor,
                            sensor_sigma: float = SCATTER_SIGMA,
                            ) -> tuple[torch.Tensor, torch.Tensor]:
    """把 target embodiment 的连续接触匹配到指令的功能区域。

    返回 ``(matched_cell, compatibility)``，形状都是 ``(N,K)``。这里匹配的是
    **功能区域**，不是要求 target 复刻 source 的接触离散格或接触点数量：每个 target
    接触可以独立落到最近的允许表面格，接触拓扑与 source 不必相同。

    ``compatibility`` 是物理尺度上的连续相容度：允许格中心处为 1，离开后按高斯衰减。
    带宽取 ``max(sensor_sigma, sqrt(cell_area/pi))``，分别覆盖触觉/traction 的固定 4 mm
    空间分辨率和表面离散格本身的量化半径。旧实现先把连续接触硬归到 256 格，再做 bool
    membership；一个只差几毫米的 target 接触会跨过格边界，从“完全正确”瞬间变成
    “完全错误”。这会把采集端接触斑块的离散形状写进 executor reward，违背跨形态接口。

    法向相背的表面不参与匹配，避免薄物体正反面因欧氏距离接近而串格。若当前命令格
    不要求接触（``allowed`` 全空），相容度为 0，索引退回普通同侧最近邻。
    """
    if allowed.dtype != torch.bool:
        raise TypeError("match_functional_region 的 allowed 必须是 bool mask")
    delta = contact_pos[:, :, None, :] - cell_points[:, None, :, :]
    distance2 = (delta ** 2).sum(-1)
    normal_agree = torch.einsum("nkd,nsd->nks", contact_normal, cell_normals) > 0.0
    candidate = allowed[:, None, :] & normal_agree
    has_candidate = candidate.any(-1)
    masked = torch.where(candidate, distance2,
                         torch.full_like(distance2, float("inf")))
    matched = masked.argmin(-1)

    rows = torch.arange(contact_pos.shape[0], device=contact_pos.device)[:, None]
    matched_distance2 = masked[rows, torch.arange(contact_pos.shape[1],
                                                    device=contact_pos.device)[None, :],
                               matched]
    matched_area = cell_area.gather(1, matched)
    quant_sigma = (matched_area.clamp_min(_EPS) / math.pi).sqrt()
    sigma = torch.maximum(quant_sigma,
                          torch.full_like(quant_sigma, float(sensor_sigma)))
    compatibility = torch.exp(-matched_distance2 / (2.0 * sigma ** 2))
    compatibility = torch.where(has_candidate & contact_valid,
                                compatibility, torch.zeros_like(compatibility))

    fallback = assign_cells(contact_pos, contact_normal, contact_valid,
                            cell_points, cell_normals)
    matched = torch.where(has_candidate & contact_valid, matched, fallback)
    return matched, compatibility


def scatter_contact_compatibility(contact_force: torch.Tensor,
                                  contact_valid: torch.Tensor,
                                  cell_index: torch.Tensor,
                                  compatibility: torch.Tensor,
                                  n_surface: int) -> torch.Tensor:
    """把逐接触相容度按力质量池化到 matched 功能格，输出 ``(N,S)``。

    它与 :func:`surface_traction` 使用同一份 ``|F|`` 权重。这样 region、mode 与
    mechanics 三项看到的是同一组 target→command 匹配，而不是三套暗中不同的归格。
    """
    weight = contact_valid.float()
    mass = contact_force.norm(dim=-1) * weight
    n_env = mass.shape[0]
    flat = cell_index + torch.arange(n_env, device=mass.device)[:, None] * n_surface
    denominator = torch.zeros(n_env * n_surface, device=mass.device, dtype=mass.dtype)
    numerator = torch.zeros_like(denominator)
    denominator.scatter_add_(0, flat.reshape(-1), mass.reshape(-1))
    numerator.scatter_add_(0, flat.reshape(-1), (mass * compatibility).reshape(-1))
    return (numerator / denominator.clamp_min(_EPS)).view(n_env, n_surface)


def surface_traction(contact_pos: torch.Tensor, contact_force: torch.Tensor,
                     contact_valid: torch.Tensor, cell_index: torch.Tensor,
                     n_surface: int, sigma: float = SCATTER_SIGMA
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """每个命令格的实测 traction (N,S,3) 与该格的力质量 (N,S)。

    连续形式的核估计：``t(x) = Σ_k F_k · exp(-‖x-x_k‖²/2σ²) / (2πσ²)``，
    在**接触点自己的位置**上求值，再按 |F| 加权池化到格。σ 与离线完全相同。
    """
    weight = contact_valid.float()
    delta = contact_pos[:, :, None, :] - contact_pos[:, None, :, :]
    kernel = torch.exp(-(delta ** 2).sum(-1) / (2.0 * sigma ** 2)) * _GAUSS_NORM
    kernel = kernel * weight[:, None, :]
    # (N,K,3)：每个接触点处的 traction，含所有同帧接触的叠加（斑块重叠是真实物理）
    traction = torch.einsum("nkm,nmd->nkd", kernel, contact_force)

    mass = contact_force.norm(dim=-1) * weight                     # (N,K)
    n_env, n_contact = mass.shape
    flat = cell_index + torch.arange(n_env, device=mass.device)[:, None] * n_surface
    denominator = torch.zeros(n_env * n_surface, device=mass.device, dtype=mass.dtype)
    denominator.scatter_add_(0, flat.reshape(-1), mass.reshape(-1))
    numerator = torch.zeros(n_env * n_surface, 3, device=mass.device, dtype=mass.dtype)
    numerator.scatter_add_(0, flat.reshape(-1, 1).expand(-1, 3),
                           (mass[..., None] * traction).reshape(-1, 3))
    denominator = denominator.view(n_env, n_surface)
    numerator = numerator.view(n_env, n_surface, 3)
    return numerator / denominator[..., None].clamp_min(_EPS), denominator


def box_violation(value: torch.Tensor, low: torch.Tensor, high: torch.Tensor,
                  floor: float = 0.0) -> torch.Tensor:
    """逐格越界量，按盒子自身的半宽归一化。集合内**恰好为零**。

    归一化用半宽而不是绝对值：不同 cell 的允许盒宽度差几个量级（支持多的格窄、
    支持少的格宽），不归一化的话 reward 会被最宽的那几格主导。越界量因此读作
    "越出去了几个集合宽度"——这正是共形集合的自然刻度。

    ``floor``：半宽的**物理下限**。⚠️ 没有它的时候这一项会炸。实测
    （`tools/s6_smoke.py`，block/press）：``mode/slip_speed`` 的允许盒是
    ``[0, 1.3e-4] m/s``，半宽 6.6e-5——一个残余滑移只有 1 mm/s 的完美执行也会得到
    ``violation ≈ 14``，而实际压住时报到 **−27675**，比其余三项大六个数量级。
    半宽趋零时"越出去几个宽度"这个刻度本身失去意义，只能退回一个物理尺度。

    下限必须是**物理量**，不能是 1e-9 那种数值 epsilon（D-75：每个自定义量都要能
    列出它依赖了哪些不属于物理的东西）。滑移用 `interaction.SLIP_SPEED_MIN`
    （5 mm/s，"多大的滑移才算换了一种接触模式"）；traction 已经在 S5 标定时带了
    ``mech_floor_relative=0.05`` 的相对下限，这里默认不再加。

    ⚠️ 下限只改集合**外面**的罚的刻度，**不动零集合**——落在允许集合内仍然恰好为 0，
    所以 C4 不会因此被悄悄实现成 C5（D-71 那条底线没有松动）。
    """
    centre = 0.5 * (high + low)
    half = (0.5 * (high - low)).clamp_min(max(floor, _EPS))
    return ((value - centre).abs() / half - 1.0).clamp_min(0.0).amax(-1)


@dataclass
class RewardWeights:
    """五项的权重。**必须在成功示教上标定，不许拍**（D-31 第 2 个洞）。

    ``tools/s6_reward_calib.py`` 把 S4 的成功记录喂进本模块，让每一项在成功轨迹上
    的量级归一，再按这里的相对重要性配比。默认值是标定**之前**的占位，
    带 ``calibrated=False``，训练入口据此拒绝启动。
    """

    effect: float = 1.0
    region: float = 1.0
    mode: float = 1.0
    mech: float = 1.0
    safety: float = 1.0
    #: 各项在成功示教上的典型量级，由标定填。
    scale: dict[str, float] = field(default_factory=dict)
    calibrated: bool = False


@dataclass
class RewardTerms:
    """四项跟踪 + 一项安全。**四项跟踪都已经在 [−1, 1] 里**，权重就是相对重要性。

    不再除以标定出来的量级。除以"成功示教上的典型值"会让一条示教的每一项恰好
    得 −1（看着很整齐），但**分布外的值会被同一个小分母放大成天文数字**：实测
    `mech` 的标定值是 0.0138，而一次开环压的越界量是 3.5 个盒宽，相除得到 −254，
    而悬停不动只有 −0.43。这一版留下的问题见本模块 docstring 第四条。
    """

    effect: torch.Tensor
    region: torch.Tensor
    mode: torch.Tensor
    mech: torch.Tensor
    safety: torch.Tensor
    #: 越界量被 `VIOLATION_CAP` 截断的比例，只作诊断。截断长期居高不下 =
    #: 那一项已经饱和、没有梯度，要回头看是不是集合本身有问题。
    mech_saturated: torch.Tensor | None = None
    mode_saturated: torch.Tensor | None = None

    def total(self, w: RewardWeights) -> torch.Tensor:
        return (w.effect * self.effect + w.region * self.region
                + w.mode * self.mode + w.mech * self.mech
                + w.safety * self.safety)

    def as_log(self) -> dict[str, torch.Tensor]:
        """**每一项都必须进 extras["log"]**（P-27 的教训：没有分项记录，
        训练不收敛时只能猜，而我猜了三轮全错）。"""
        return {f"reward/{k}": v for k, v in self.__dict__.items() if v is not None}


def interaction_reward(*, effect_deficit: torch.Tensor,
                       traction: torch.Tensor, mass: torch.Tensor,
                       slip_speed: torch.Tensor,
                       allowed: torch.Tensor,
                       traction_lo: torch.Tensor, traction_hi: torch.Tensor,
                       slip_lo: torch.Tensor, slip_hi: torch.Tensor,
                       force_penalty: torch.Tensor,
                       region_outside_scale: float = 1.0,
                       violation_cap: float = VIOLATION_CAP) -> RewardTerms:
    """五项 reward。effect/mode/mech 是负误差，region 是带符号相容度，safety 另计。

    Args:
        effect_deficit: (N,) 本命令格的 effect **完成缺口** ∈[0,1]，
            由 `effect_deficit()` 算。0 = 这一格要求的变化已经做到。
        traction: (N,S,3) 实测逐格 traction。
        mass: (N,S) 逐格力质量，既是权重也是"这里有没有接触"。
        slip_speed: (N,S) 实测逐格滑移速率。
        allowed: (N,S) bool 或 [0,1]，指令允许区域或连续匹配相容度。
        force_penalty: (N,) 安全项（法向力超限、动作突变等），已是负值。
    """
    # ``allowed`` 既可为离线/单测使用的 bool mask，也可为在线 target→command
    # 连续匹配得到的相容度 [0,1]。后者是跨形态执行的关键：target 接触不必复刻
    # source 的离散 contact topology，只需在物理分辨率内匹配功能区域。
    allowed_weight = allowed.float().clamp(0.0, 1.0)
    total_mass = mass.sum(-1).clamp_min(_EPS)
    inside = (mass * allowed_weight).sum(-1)
    outside = (mass * (1.0 - allowed_weight)).sum(-1)
    # **带符号的软相容度**：完全相容 = +1，完全不相容 = −1，没接触 = 0。
    touching = mass.sum(-1) > _EPS
    region = torch.where(touching,
                         (inside - region_outside_scale * outside) / total_mass,
                         torch.zeros_like(inside))

    weight = mass * allowed_weight
    weight_sum = weight.sum(-1).clamp_min(_EPS)
    mech_raw = (weight * box_violation(traction, traction_lo, traction_hi)
                ).sum(-1) / weight_sum
    mode_raw = (weight * box_violation(slip_speed[..., None], slip_lo, slip_hi,
                                       floor=SLIP_SPEED_MIN)).sum(-1) / weight_sum
    cap = max(violation_cap, _EPS)
    return RewardTerms(effect=-effect_deficit, region=region,
                       mode=-(mode_raw / cap).clamp(max=1.0),
                       mech=-(mech_raw / cap).clamp(max=1.0),
                       safety=force_penalty,
                       mech_saturated=(mech_raw >= cap).float(),
                       mode_saturated=(mode_raw >= cap).float())
