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
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

#: 与 `surfaces.SCATTER_SIGMA` 必须一致：它是 traction 的定义的一部分，
#: 不是一个可以在这边单独调的超参。
from it.surfaces import SCATTER_SIGMA

#: 高斯面密度核的归一化常数：∫ G dA = 1 要求除以 2πσ²。
_GAUSS_NORM = 1.0 / (2.0 * math.pi * SCATTER_SIGMA ** 2)
_EPS = 1e-9


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
    return (quad / scale_rigid.clamp_min(_EPS)
            + state.abs().sum(-1) / scale_state.clamp_min(_EPS))


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


def box_violation(value: torch.Tensor, low: torch.Tensor, high: torch.Tensor
                  ) -> torch.Tensor:
    """逐格越界量，按盒子自身的半宽归一化。集合内**恰好为零**。

    归一化用半宽而不是绝对值：不同 cell 的允许盒宽度差几个量级（支持多的格窄、
    支持少的格宽），不归一化的话 reward 会被最宽的那几格主导。
    """
    centre = 0.5 * (high + low)
    half = 0.5 * (high - low).clamp_min(_EPS)
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
    effect: torch.Tensor
    region: torch.Tensor
    mode: torch.Tensor
    mech: torch.Tensor
    safety: torch.Tensor

    def total(self, w: RewardWeights) -> torch.Tensor:
        s = w.scale
        norm = lambda name, value: value / max(s.get(name, 1.0), _EPS)  # noqa: E731
        return (w.effect * norm("effect", self.effect)
                + w.region * norm("region", self.region)
                + w.mode * norm("mode", self.mode)
                + w.mech * norm("mech", self.mech)
                + w.safety * self.safety)

    def as_log(self) -> dict[str, torch.Tensor]:
        """**每一项都必须进 extras["log"]**（P-27 的教训：没有分项记录，
        训练不收敛时只能猜，而我猜了三轮全错）。"""
        return {f"reward/{k}": v for k, v in self.__dict__.items()}


def interaction_reward(*, effect_error: torch.Tensor,
                       traction: torch.Tensor, mass: torch.Tensor,
                       slip_speed: torch.Tensor,
                       allowed: torch.Tensor,
                       traction_lo: torch.Tensor, traction_hi: torch.Tensor,
                       slip_lo: torch.Tensor, slip_hi: torch.Tensor,
                       force_penalty: torch.Tensor,
                       region_outside_scale: float = 1.0) -> RewardTerms:
    """五项 reward。全部是**负的误差**（越大越好，上界 0），safety 另计。

    Args:
        effect_error: (N,) 无量纲 effect 跟踪误差，由 `effect_magnitude` 之差算。
        traction: (N,S,3) 实测逐格 traction。
        mass: (N,S) 逐格力质量，既是权重也是"这里有没有接触"。
        slip_speed: (N,S) 实测逐格滑移速率。
        allowed: (N,S) bool，指令允许区域。
        force_penalty: (N,) 安全项（法向力超限、动作突变等），已是负值。
    """
    total_mass = mass.sum(-1).clamp_min(_EPS)
    inside = (mass * allowed.float()).sum(-1)
    outside = (mass * (~allowed).float()).sum(-1)
    # 有接触才谈得上 region 分配；完全没接触时这一项为 0，由 effect 项去驱动建立接触。
    touching = mass.sum(-1) > _EPS
    region = torch.where(touching,
                         (inside - region_outside_scale * outside) / total_mass,
                         torch.zeros_like(inside)) - 1.0
    region = region * touching.float()      # 没接触时恰好 0，不是 -1

    weight = mass * allowed.float()
    weight_sum = weight.sum(-1).clamp_min(_EPS)
    mech = -(weight * box_violation(traction, traction_lo, traction_hi)).sum(-1) / weight_sum
    mode = -(weight * box_violation(slip_speed[..., None], slip_lo, slip_hi)
             ).sum(-1) / weight_sum
    return RewardTerms(effect=-effect_error, region=region, mode=mode, mech=mech,
                       safety=force_penalty)
