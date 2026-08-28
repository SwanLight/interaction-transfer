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

        ⚠️ **这个函数返回的 ``friction_forces`` 是错的**，见 P-36：摩擦 buffer 有
        自己的 counts/start_idx，行数与接触 buffer 不同（实测接触 4 个点对应
        摩擦 2 个锚点），拿接触的切片去索引摩擦，env0 会多算一倍、env1 全是零。
        需要摩擦力时用 ``extract_contact_points_padded``。

        ⚠️ **这个函数只在"所有 env 同时有接触"时给出正确的 env 归属**。
        接触逐个 env 先后建立时，``start_idx`` 的下标与 env 下标会错开，
        本函数会把别的 env 的接触点交给你，且不报错——见 P-30 与
        ``extract_contact_points_padded``。需要逐 env 正确归属时用后者。
        本函数保留是因为它是 S1/S2 的既有依赖，且逐点数值本身是对的。
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


def contact_torque_about_axis(
    cp: ContactPoints,
    axis_point: torch.Tensor,
    axis_dir: torch.Tensor,
) -> torch.Tensor:
    """把逐接触点的力（法向 + 摩擦）合成为绕给定轴的广义力矩。

    实现 `plan/02` §8 要求的"用接触集合重建物体广义力"，也是 §3.5 中
    task-relevant generalized force 的计算方式（旋钮为绕轴力矩）。

    每个接触点的总力 = 法向力 × 法向 + 摩擦力矢量。力矩取 Σ (r_i × F_i) · axis。

    Args:
        cp: 单 env 的接触点数据，世界系。
        axis_point: 转轴上任一点，shape (3,)。
        axis_dir: 转轴单位方向，shape (3,)。

    Returns:
        标量张量，绕该轴的合力矩（N·m），符号按右手定则。
    """
    if cp.is_empty():
        return torch.zeros((), device=axis_point.device)

    f_total = cp.normal_forces.unsqueeze(-1) * cp.normals + cp.friction_forces
    r = cp.positions - axis_point.unsqueeze(0)
    tau = torch.cross(r, f_total, dim=-1).sum(dim=0)
    return torch.dot(tau, axis_dir / axis_dir.norm())


def extract_contact_points_padded(
    sensor,
    dt: float,
    body_pos_w: torch.Tensor,
    max_points: int = 16,
    force_threshold: float = 1e-4,
    own_radius: float = 0.08,
) -> dict[str, torch.Tensor]:
    """一次取出所有 env 的逐点接触数据，补齐到定长，**按位置归属到 env**。

    Args:
        sensor: 已初始化的 ``isaaclab.sensors.ContactSensor``。
        dt: **物理**步长。
        body_pos_w: (N, 3)，传感器所在刚体在各 env 的世界位置。**必填**——
            env 归属就是靠它判定的，理由见下面的 P-30。
        max_points: 每个 env 保留的最大接触点数，超出丢弃（同 P-03 的问题）。
        force_threshold: 法向力低于此值的槽位视为空槽。PhysX 未使用的槽位是全零。
        own_radius: 接触点到本体的最大距离。超出的点既不属于本 env、也找不到
            更近的本体，计进 ``foreign`` 并丢弃。取值要大于刚体自身尺寸的一半。

    Returns:
        dict，键为 ``positions`` (N, K, 3)、``normals`` (N, K, 3)、
        ``normal_forces`` (N, K)、``friction_forces`` (N, K, 3)、
        ``separations`` (N, K)、``valid`` (N, K) bool、``count`` (N,) long、
        ``foreign`` (标量 long，被丢弃的点数)、``dropped`` (标量 long，
        因超过 ``max_points`` 被截掉的点数)。

    Note:
        **P-30 · ``start_idx`` 的下标不是 env 下标。**

        `pitfalls.md` P-18 记的做法是"用 counts/start_idx 按 env 切片"。实测
        **那个下标不可信**：某一帧只有 env 2 有接触时，``counts`` 报的是
        ``[0, 0, 0, 2]``、``start_idx`` 是 ``[0, 0, 0, 0]``——数量和位置都对，
        但**记在了 env 3 名下**。于是 env 3 会拿到 env 2 的接触点，世界坐标
        整整差一个 ``env_spacing``，而且不报任何错。

        所有 env 同时有接触时下标恰好是恒等的，所以静态自检根本发现不了；
        S3 采集里接触是逐个 env 先后建立的，68% 的点归属错误。

        因此这里只用 ``counts``/``start_idx`` 圈定"buffer 里哪些槽位是这一帧
        新鲜的"（这一点它是对的），**env 归属改为按接触点离哪个本体最近来判**。

        ⚠️ 用过旧切片法的结论要重新审视：`decisions.md` D-34（钩杆接触部位
        分布）就是那样算的。它当时 16 个 env 几乎同时接触、下标大概率是恒等，
        但没有验证过。
    """
    view = sensor.contact_physx_view
    c_forces, c_points, c_normals, c_seps, counts, start_idx = view.get_contact_data(dt=dt)
    f_forces, f_points, _f_counts, _f_start = view.get_friction_data(dt=dt)

    device = c_points.device
    counts = counts.flatten().long()
    start = start_idx.flatten().long()
    total = int(c_points.shape[0])
    n_env = int(body_pos_w.shape[0])
    K = max_points

    def _empty(extra_foreign=0):
        z = torch.zeros(n_env, K, device=device)
        return {
            "positions": torch.zeros(n_env, K, 3, device=device),
            "normals": torch.zeros(n_env, K, 3, device=device),
            "normal_forces": z.clone(),
            "friction_forces": torch.zeros(n_env, K, 3, device=device),
            "separations": z.clone(),
            "valid": torch.zeros(n_env, K, dtype=torch.bool, device=device),
            "count": torch.zeros(n_env, dtype=torch.long, device=device),
            "foreign": torch.as_tensor(extra_foreign, device=device),
            "dropped": torch.zeros((), dtype=torch.long, device=device),
        }

    if total == 0:
        return _empty()

    # 1) 圈出这一帧有数据的槽位。
    #
    # **不用 counts/start_idx 圈范围**：实测它们连"总共有多少点"都会漏报——
    # 有 env 的板明明压进横杆 0.5 mm、抽屉被推开了 160 mm，counts 却从头到尾
    # 报 0，那些点其实躺在前缀和覆盖不到的槽位里。既然 env 归属已经改成按
    # 位置判（见下面的 P-30），范围也就不必再依赖它：**扫全 buffer**，
    # 用"法向力非零"+"离某块本体足够近"两道闸门筛。
    nf_all = c_forces.reshape(-1)
    fresh = nf_all.abs() > force_threshold
    sel = fresh.nonzero().flatten()
    if sel.numel() == 0:
        return _empty()

    # 2) 按位置归属：接触点必然贴在自己那块刚体上
    pts = c_points[sel]
    dist = (pts.unsqueeze(1) - body_pos_w.unsqueeze(0)).norm(dim=-1)   # (M, N)
    d_min, owner = dist.min(dim=1)
    near = d_min <= own_radius
    n_foreign = (~near).sum()
    sel, owner = sel[near], owner[near]
    if sel.numel() == 0:
        return _empty(n_foreign)

    # 3) 按 env 分组，组内排序号即定长数组的槽位
    order = torch.argsort(owner, stable=True)
    sel, owner = sel[order], owner[order]
    per_env = torch.bincount(owner, minlength=n_env)
    offsets = torch.cumsum(per_env, dim=0) - per_env
    rank = torch.arange(sel.numel(), device=device) - offsets[owner]
    keep = rank < K
    n_dropped = (~keep).sum()
    sel, owner, rank = sel[keep], owner[keep], rank[keep]

    out = _empty(n_foreign)
    out["dropped"] = n_dropped
    out["positions"][owner, rank] = c_points[sel]
    out["normals"][owner, rank] = c_normals[sel]
    out["normal_forces"][owner, rank] = nf_all[sel]
    out["separations"][owner, rank] = c_seps.reshape(-1)[sel]
    out["valid"][owner, rank] = True
    out["count"] = out["valid"].sum(dim=1)

    # 4) 摩擦：**摩擦 buffer 有自己的行数和自己的下标**，不能拿接触的下标去索引。
    #    实测同一帧接触 counts=[4,4] 而摩擦 counts=[2,2]（PhysX 的摩擦锚点是
    #    按**接触斑块**给的，一个斑块通常 2 个锚点，与接触点不是一一对应）。
    #    用接触下标去索引摩擦，env0 会读到"自己 2 个 + env1 的 2 个"= 恰好 2 倍，
    #    env1 读到全零 —— 见 P-36。
    _friction_into(out, f_forces, f_points, body_pos_w, own_radius)
    return out


def _friction_into(out, f_forces, f_points, body_pos_w, own_radius: float) -> None:
    """把摩擦锚点按位置归属到 env，再按法向力占比摊到各接触点上。

    摩擦是**斑块级**的量，接触点是点级的量，两者不同构。这里的摊派用的是
    库仑摩擦的标准假设——同一斑块内摩擦力正比于法向载荷——所以摊派后
    「每点的 |f_t| 与 μ·f_n 之比」仍然是判 stick/slide 的正确依据，
    且各点之和精确等于该斑块的实际摩擦力，不会凭空多出或少掉。
    """
    ff = f_forces.reshape(-1, 3)
    fp = f_points.reshape(-1, 3)
    live = ff.norm(dim=-1) > 1e-9
    idx = live.nonzero().flatten()
    if idx.numel() == 0:
        return
    d = (fp[idx].unsqueeze(1) - body_pos_w.unsqueeze(0)).norm(dim=-1)   # (A, N)
    d_min, a_owner = d.min(dim=1)
    near = d_min <= own_radius
    idx, a_owner = idx[near], a_owner[near]
    if idx.numel() == 0:
        return

    n_env, K = out["valid"].shape
    device = ff.device
    A = int(torch.bincount(a_owner, minlength=n_env).max().item())
    a_rank = torch.zeros_like(a_owner)
    order = torch.argsort(a_owner, stable=True)
    idx, a_owner = idx[order], a_owner[order]
    per = torch.bincount(a_owner, minlength=n_env)
    a_rank = (torch.arange(idx.numel(), device=device)
              - (torch.cumsum(per, 0) - per)[a_owner])

    anc_f = torch.zeros(n_env, A, 3, device=device)
    anc_p = torch.zeros(n_env, A, 3, device=device)
    anc_v = torch.zeros(n_env, A, dtype=torch.bool, device=device)
    anc_f[a_owner, a_rank] = ff[idx]
    anc_p[a_owner, a_rank] = fp[idx]
    anc_v[a_owner, a_rank] = True

    # 每个接触点认领离它最近的锚点
    dist = (out["positions"].unsqueeze(2) - anc_p.unsqueeze(1)).norm(dim=-1)  # (N,K,A)
    dist = dist.masked_fill(~anc_v.unsqueeze(1), float("inf"))
    claim = dist.argmin(dim=2)                                                # (N,K)
    ok = out["valid"] & anc_v.gather(1, claim)

    fn = out["normal_forces"].abs() * ok
    load = torch.zeros(n_env, A, device=device).scatter_add_(1, claim, fn)
    share = torch.where(load.gather(1, claim) > 1e-9,
                        fn / load.gather(1, claim).clamp_min(1e-9), torch.zeros_like(fn))
    out["friction_forces"] = (anc_f.gather(1, claim.unsqueeze(-1).expand(-1, -1, 3))
                              * share.unsqueeze(-1))


def classify_contact_mode_padded(
    normal_forces: torch.Tensor,
    friction_forces: torch.Tensor,
    separations: torch.Tensor,
    valid: torch.Tensor,
    mu: float,
    stick_ratio: float = 0.95,
) -> torch.Tensor:
    """``classify_contact_mode`` 的定长批量版本，判据完全相同。

    返回与 ``normal_forces`` 同形状的 int8 张量，取值 0=no contact /
    1=sticking / 2=sliding / 3=separating（`plan/02` §3.4）。

    与逐 env 版本的**唯一**差别：这里不接受 ``slip_speed``。逐点切向相对
    速度需要额外一次 PhysX 查询，采数据时不划算；「力已饱和但实际没动」的
    边界情况会被判成 sliding。S4 构造 Oracle Interaction Record 时若需要
    区分，用物体速度和接触点位置自行补判。
    """
    fn = normal_forces.abs()
    ff = friction_forces.norm(dim=-1)
    mode = torch.full(fn.shape, 2, dtype=torch.int8, device=fn.device)
    mode[ff < stick_ratio * mu * fn] = 1
    mode[separations > 0] = 3
    mode[fn <= 0] = 0
    mode[~valid] = 0
    return mode
