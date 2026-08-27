"""逐接触点数据提取工具。

Isaac Lab 2.3.1 (isaaclab 0.48.0) 的 ContactSensor 包装层不暴露摩擦力，
且底层 PhysX tensor view 返回的是**跨 env 扁平打包**的 buffer。
本模块负责把它拆成每个 env 的逐点数据，产出 `plan/02` §3.2 需要的格式。

前提条件（缺一不可，见 log/pitfalls.md P-16/P-17/P-18）：

1. filter 目标必须是 rigid body。不动的物体用
   ``RigidBodyPropertiesCfg(kinematic_enabled=True)``。静态碰撞体（只有
   collision_props）不会注册成可过滤接触对，此时 net_forces_w 仍正确但
   所有 filter 通道的数据静默失效。
2. ``ContactSensorCfg.filter_prim_paths_expr`` 非空。
3. ``ContactSensorCfg.max_contact_data_count_per_prim >= 1``（默认 4，
   contact-rich 场景不够，会静默丢接触点）。

实测依据：tools/check_contact_sensor.py，2026-08-27，Isaac Sim 5.1.0-rc.19。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ContactPoints:
    """单个 env 在单帧上的逐接触点数据，全部在世界系。

    所有张量第 0 维长度相同，等于该 env 当前的有效接触点数（可能为 0）。
    调用方需自行转换到物体系——本模块不做坐标变换，因为参照物体因任务而异。
    """

    positions: torch.Tensor
    """接触点位置，shape (K, 3)。"""

    normals: torch.Tensor
    """接触法向（由 filter 指向 sensor），shape (K, 3)。"""

    normal_forces: torch.Tensor
    """法向力标量，shape (K,)。"""

    friction_forces: torch.Tensor
    """摩擦力矢量，shape (K, 3)。切向，垂直于 normals。"""

    separations: torch.Tensor
    """分离距离，shape (K,)。负值表示穿透。"""

    @property
    def num_contacts(self) -> int:
        return int(self.positions.shape[0])

    def is_empty(self) -> bool:
        return self.num_contacts == 0


def extract_contact_points(
    sensor,
    dt: float,
    env_ids: torch.Tensor | list[int] | None = None,
    force_threshold: float = 1e-4,
) -> list[ContactPoints]:
    """从 ContactSensor 取出逐 env 的逐接触点数据。

    Args:
        sensor: 已初始化的 ``isaaclab.sensors.ContactSensor``。
        dt: 物理步长，即 ``SimulationContext.get_physics_dt()``。**不是控制步长。**
        env_ids: 要提取的 env 下标；None 表示全部。
        force_threshold: 法向力低于此值的接触点视为无效并剔除。PhysX 的 buffer
            里未使用的槽位是全零，必须过滤，否则会混入大量假接触点。

    Returns:
        长度等于 env 数的列表，每项是该 env 的 ContactPoints。

    Note:
        buffer 是跨 env 扁平打包的，总长为
        ``max_contact_data_count_per_prim * num_envs * num_sensor_bodies``。
        必须用 counts/start_idx 切片，直接对整个 buffer 求和会把所有 env 加在
        一起（见 P-18：2 env 时结果恰好是 2 倍，极易被误认为单位问题）。
    """
    view = sensor.contact_physx_view

    # get_contact_data -> (forces, points, normals, separations, counts, start_idx)
    c_forces, c_points, c_normals, c_seps, counts, start_idx = view.get_contact_data(dt=dt)
    # get_friction_data -> (friction_forces, friction_points, counts, start_idx)
    f_forces, _f_points, _f_counts, _f_start = view.get_friction_data(dt=dt)

    counts_flat = counts.flatten()
    start_flat = start_idx.flatten()
    num_envs = counts_flat.shape[0]

    if env_ids is None:
        env_ids = range(num_envs)
    elif isinstance(env_ids, torch.Tensor):
        env_ids = env_ids.tolist()

    out: list[ContactPoints] = []
    for e in env_ids:
        s = int(start_flat[e].item())
        n = int(counts_flat[e].item())
        if n == 0:
            empty = torch.empty((0, 3), device=c_points.device, dtype=c_points.dtype)
            empty1 = torch.empty((0,), device=c_points.device, dtype=c_points.dtype)
            out.append(ContactPoints(empty, empty, empty1, empty, empty1))
            continue

        sl = slice(s, s + n)
        nf = c_forces[sl].flatten()
        keep = nf.abs() > force_threshold

        out.append(
            ContactPoints(
                positions=c_points[sl][keep],
                normals=c_normals[sl][keep],
                normal_forces=nf[keep],
                friction_forces=f_forces[sl][keep],
                separations=c_seps[sl].flatten()[keep],
            )
        )
    return out


def classify_contact_mode(
    cp: ContactPoints,
    mu: float,
    slip_speed: torch.Tensor | None = None,
    stick_ratio: float = 0.95,
    slip_speed_threshold: float = 1e-3,
) -> torch.Tensor:
    """逐接触点判定 no-contact / stick / slide / separating。

    对应 `plan/02` §3.4 的四档。返回整型张量，取值：

    ==== ==============
    0    no contact
    1    sticking
    2    sliding
    3    separating
    ==== ==============

    判据来自库仑摩擦，已在 tools/check_contact_sensor.py 上实测验证
    （μ=0.4, m=0.5 kg → μN=1.962 N）：

    - 静摩擦跟随外力直到 μN：施力 1.0/1.5 N 时摩擦力恰为 1.0/1.5 N，v≈0
    - 之后饱和成动摩擦：施力 2.5/3.0 N 时摩擦力均为 1.962 N，物体滑动

    因此 ``|Ff| / (mu * Fn)`` 是比原始力值更稳健的判据——它对法向力大小
    归一化，不随压力变化而漂移。

    Args:
        cp: 单 env 的接触点数据。
        mu: 该接触对的摩擦系数。两侧材质不同时按仿真器的组合规则取值，
            不要想当然用其中一侧的。
        slip_speed: 每个接触点的切向相对速度，shape (K,)。为 None 时只用
            力的判据；提供后可消除"力已饱和但尚未起滑"的边界误判。
        stick_ratio: ``|Ff| < stick_ratio * mu * Fn`` 判为 stick。
        slip_speed_threshold: 切向速度阈值，单位 m/s。

    Note:
        阈值只用于生成一致的标签，**不把阈值本身作为研究结论**
        （`plan/02` §3.4 的原则）。
    """
    if cp.is_empty():
        return torch.empty((0,), dtype=torch.long, device=cp.positions.device)

    fn = cp.normal_forces.abs()
    ff = cp.friction_forces.norm(dim=-1)
    capacity = mu * fn

    mode = torch.full_like(fn, 2, dtype=torch.long)  # 默认 sliding
    mode[ff < stick_ratio * capacity] = 1            # sticking
    mode[fn <= 0] = 0                                # no contact
    mode[cp.separations > 0] = 3                     # separating

    if slip_speed is not None:
        # 力已饱和但实际没动 -> 仍是 stick
        stalled = (mode == 2) & (slip_speed.abs() < slip_speed_threshold)
        mode[stalled] = 1

    return mode


def to_object_frame(
    points_w: torch.Tensor,
    vectors_w: torch.Tensor,
    obj_pos_w: torch.Tensor,
    obj_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把世界系的接触点和矢量转到物体系。

    `plan/02` §1 要求所有表面点、接触区域和机械作用都表达在被操作物体坐标系中。
    §1.1 规定"被操作物体 = 任务成功判据所作用的那个物体"，中间传力的工具不是。

    Args:
        points_w: 世界系位置，shape (K, 3)。
        vectors_w: 世界系矢量（法向、力等），shape (K, 3)。位置做平移+旋转，
            矢量只做旋转。
        obj_pos_w: 物体世界系位置，shape (3,)。
        obj_quat_w: 物体世界系四元数 (w, x, y, z)，shape (4,)。

    Returns:
        (points_o, vectors_o)，均为 shape (K, 3)。
    """
    from isaaclab.utils.math import quat_apply_inverse

    n = points_w.shape[0]
    q = obj_quat_w.unsqueeze(0).expand(n, 4)
    points_o = quat_apply_inverse(q, points_w - obj_pos_w.unsqueeze(0))
    vectors_o = quat_apply_inverse(q, vectors_w)
    return points_o, vectors_o


def assign_to_surface_points(
    contact_pos_o: torch.Tensor,
    surface_points_o: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """把接触点按最近邻分配到物体表面采样点，形成加权热力图。

    实现 `plan/02` §3.2：以法向力加权形成表面热力图，**保留空间分布，
    不只保留质心**。

    这也是不升级 Isaac Lab 的原因之一：upstream 的 ``friction_forces_w``
    形状 (N, B, M, 3)，是按 filter body 聚合求和的，空间分布已被抹掉。
    见 log/decisions.md D-20。

    Args:
        contact_pos_o: 物体系接触点位置，shape (K, 3)。
        surface_points_o: 物体系表面采样点，shape (S, 3)。分辨率由 S4.5
            敏感度扫描确定（`plan/02` §2.1），不预先拍板。
        weights: 每个接触点的权重，通常是法向力，shape (K,)。

    Returns:
        每个表面采样点的累计权重，shape (S,)。
    """
    if contact_pos_o.shape[0] == 0:
        return torch.zeros(surface_points_o.shape[0], device=surface_points_o.device)

    d = torch.cdist(contact_pos_o.unsqueeze(0), surface_points_o.unsqueeze(0)).squeeze(0)
    nearest = d.argmin(dim=1)

    heat = torch.zeros(surface_points_o.shape[0], device=surface_points_o.device)
    heat.index_add_(0, nearest, weights)
    return heat
