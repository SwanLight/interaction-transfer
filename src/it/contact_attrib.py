"""把接触点归到"物体的哪个部位"和"执行器的哪个面"。

**为什么必须有这个模块**：`log/decisions.md` D-34 的教训是，一个成功率看着
正常的策略，接触可能落在完全出乎意料的地方——钩杆 78% 的接触在主杆上，
真正的横钩只占 4.3%，而这件事**成功率、接触力、无穿模检查全都看不出来**，
是用户盯着录像发现的，之后才用探针脚本量化确认。

采集数据比训练策略更受这件事影响：`plan/02` §3.2 的 Interaction Region 就是
"接触落在物体表面哪里"的热图。如果 source 采集时板子实际是拿**边角**蹭在
**面板**上而不是拿**工作面**压在**把手**上，那么 region 字段从第一天起就是
错的，而下游一切都建立在它上面。

所以每条 episode 都带上部位归类的统计量，不是可选的诊断，是数据契约的一部分。

坐标全部先转到**被作用物体的局部系**再判定（`plan/02` §1 要求表示是物体系的），
不在世界系拍阈值——抽屉一拉出来世界系坐标就变了。
"""

from __future__ import annotations

import torch

MM = 0.001


def rotate_inverse(quat_wxyz: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """把世界系矢量转进局部系：``v_local = R(q)^T · v_world``。

    自己实现而不用 ``isaaclab.utils.math``，因为那里的函数名在版本间改过
    （``quat_rotate_inverse`` → ``quat_apply_inverse``），而本项目锁定
    Isaac Lab 2.3.1（D-20/D-21）却要能被后来的人在别的版本上读懂。
    公式是标准的 Rodrigues 展开，没有奇异点。

    Args:
        quat_wxyz: (N, 4)，(w, x, y, z)，必须已归一化。
        vec: (N, ..., 3)，世界系矢量。

    Returns:
        与 ``vec`` 同形状的局部系矢量。
    """
    shape = [quat_wxyz.shape[0]] + [1] * (vec.dim() - 2) + [1]
    w = quat_wxyz[:, 0].reshape(shape)
    u = quat_wxyz[:, 1:4].reshape(shape[:-1] + [3])
    dot = (u * vec).sum(dim=-1, keepdim=True)
    cross = torch.cross(u.expand_as(vec), vec, dim=-1)
    return (2.0 * w * w - 1.0) * vec + 2.0 * dot * u - 2.0 * w * cross


def to_local(points_w: torch.Tensor, origin_w: torch.Tensor,
             quat_w: torch.Tensor) -> torch.Tensor:
    """世界系点 → 刚体局部系点。

    Args:
        points_w: (N, K, 3)。
        origin_w: (N, 3) 刚体原点。
        quat_w: (N, 4) 刚体姿态。
    """
    return rotate_inverse(quat_w, points_w - origin_w.unsqueeze(1))


# ---------------------------------------------------------------- 执行器一侧

#: 板局部系的面归类。板的 +Z 是工作面法向，长边 35 mm 在 X、短边 25 mm 在 Y。
PLATE_PARTS = ("work_face", "back_face", "edge")


def classify_plate_face(normals_local: torch.Tensor, pos_local: torch.Tensor,
                        weights: torch.Tensor, valid: torch.Tensor,
                        face_align: float = 0.7) -> torch.Tensor:
    """判定接触落在板的哪个面，返回 (N, K) int8：0=工作面 / 1=背面 / 2=侧边。

    分两步，两步都**不依赖 PhysX 报法向的正负约定**：

    1. **面还是棱**，看法向与板面法向的夹角。板只有 3 mm 厚，接触点位置在
       两个表面之间浮动（实测正常的工作面接触，局部 z 在 0.3~1.7 mm 之间摆，
       而板半厚才 1.5 mm），用位置判会把一半正常接触判成侧边——那是 P-35。
       法向就干净：实测 100% 的面接触满足 ``|n_z| > 0.7``。
    2. **哪一面**，看接触点整体落在板的哪一侧，用**法向力加权的局部 z 均值**。

    ⚠️ 第 2 步**不能用法向的正负号**。PhysX 报的接触法向指向哪一侧取决于
    这一对刚体在内部的先后次序，**同一份代码换个场景就会翻**：抽屉场景里
    工作面接触的局部法向是 +Z，探针方块场景里是 −Z，于是同样的判据把
    100% 的正常接触判成了"用背面接触"。位置是几何量，没有这个问题；
    对整块板取加权均值也压掉了单点位置的抖动。

    Args:
        normals_local: (N, K, 3) 板局部系的接触法向。
        pos_local: (N, K, 3) 板局部系的接触点位置。
        weights: (N, K) 加权用的法向力幅值。
        valid: (N, K) 有效位掩码。
    """
    nz = normals_local[..., 2]
    w = weights.abs() * valid
    z_bar = (pos_local[..., 2] * w).sum(dim=1) / w.sum(dim=1).clamp_min(1e-9)
    work = (z_bar >= 0.0).unsqueeze(1).expand_as(nz)
    mode = torch.where(work, torch.zeros_like(nz, dtype=torch.int8),
                       torch.ones_like(nz, dtype=torch.int8))
    mode[nz.abs() <= face_align] = 2
    return mode


# ---------------------------------------------------------------- 抽屉一侧

#: 抽屉局部系的部位归类。顺序即返回的整数编码。
DRAWER_PARTS = (
    "bar_back",     # 把手横杆的**背面**（朝柜体一侧）——这是"手指伸进净空往外拉"的受力面
    "bar_front",    # 横杆的前面（朝外一侧）——拇指一侧
    "bar_top",      # 横杆上表面
    "bar_bottom",   # 横杆下表面
    "post",         # 支撑柱
    "panel",        # 抽屉面板
    "other",        # 托盘、面板侧面等，正常操作里不该有
)


def classify_drawer_local(
    pts_local: torch.Tensor,
    *,
    bar_x: float | torch.Tensor,
    bar_z: float,
    bar_radius: float,
    bar_half_len: float,
    post_half_spacing: float,
    post_radius: float,
    panel_t: float,
    tol: float = 6 * MM,
) -> torch.Tensor:
    """判定接触落在抽屉的哪个部位，返回 (N, K) int8，编码见 ``DRAWER_PARTS``。

    横杆按绕轴角度分成背/前/上/下四份：角度 ``atan2(z - bar_z, x - bar_x)``
    为 0 表示正前方（+X），±π 表示正后方。这个分法直接对应"手指勾在杆子
    背面往外拉"与"拇指按在杆子前面"，是操作常识层面能一眼看懂的判据。

    Args:
        pts_local: (N, K, 3)，已转进抽屉局部系。
        tol: 容差。接触点报在两碰撞面之间，且求解器允许少量穿透，
            所以判"在杆上"要放宽到半径 ± tol。
    """
    x, y, z = pts_local[..., 0], pts_local[..., 1], pts_local[..., 2]
    # bar_x 可以是标量，也可以是**逐 env** 的 (N,) 张量——几何变体
    # （`plan/03` §7）让把手净空按 env 变，横杆的 X 位置随之不同。
    if torch.is_tensor(bar_x) and bar_x.dim() == 1:
        bar_x = bar_x.view(-1, *([1] * (x.dim() - 1)))
    dx, dz = x - bar_x, z - bar_z
    r = torch.sqrt(dx * dx + dz * dz)

    on_bar = (r < bar_radius + tol) & (y.abs() < bar_half_len + tol)
    angle = torch.atan2(dz, dx)
    a = angle.abs()
    part = torch.full(x.shape, DRAWER_PARTS.index("other"), dtype=torch.int8,
                      device=x.device)

    part[on_bar & (a > 3.0 * torch.pi / 4)] = DRAWER_PARTS.index("bar_back")
    part[on_bar & (a < torch.pi / 4)] = DRAWER_PARTS.index("bar_front")
    part[on_bar & (angle >= torch.pi / 4) & (angle <= 3.0 * torch.pi / 4)] = \
        DRAWER_PARTS.index("bar_top")
    part[on_bar & (angle <= -torch.pi / 4) & (angle >= -3.0 * torch.pi / 4)] = \
        DRAWER_PARTS.index("bar_bottom")

    on_post = (
        ((y.abs() - post_half_spacing).abs() < post_radius + tol)
        & (x > panel_t - tol) & (x < bar_x + tol)
        & ((z - bar_z).abs() < post_radius + tol)
        & ~on_bar
    )
    part[on_post] = DRAWER_PARTS.index("post")

    on_panel = (x < panel_t + tol) & ~on_bar & ~on_post
    part[on_panel] = DRAWER_PARTS.index("panel")
    return part


def bar_span_fraction(pts_local: torch.Tensor, post_half_spacing: float,
                      post_radius: float) -> torch.Tensor:
    """接触点是否落在两根支撑柱**之间**的横杆自由段，返回 (N, K) bool。

    D-34 记录过钩杆的行为：37.6% 的接触落在柱外那 7 mm 的横杆末端，
    因为"侧向一靠就挂住"比"把杆子塞进 45 mm 净空"省事。采集数据时同样
    要盯住这一点——柱外末端不是正常的操作部位。
    """
    return pts_local[..., 1].abs() < (post_half_spacing - post_radius)


def roll_sign(z_axis: torch.Tensor, long_axis: torch.Tensor,
              refs: list[torch.Tensor]) -> torch.Tensor:
    """在 ±long_axis 两个**等价**解里挑一个，返回 (N, 1) 的 +1 / -1。

    ⚠️ **判据必须在退化时也是确定的，而且整条 episode 只能算一次。**

    第一版写成 `flip = (y·up < 0)`，而在圆柱侧面这类位点上 y 恰好水平，
    `y·up` 数学上正好是 0、实际是 ±1e-9 的浮点噪声。姿态每个控制步重算一次，
    于是目标四元数在相邻步之间**随机翻 180°**，转动 PD 每步都想把板拧半圈：
    实测立柱对捏的夹持力从 14.1 N 掉到 8.7 N、物体被甩出 1046 mm、
    成功率 51/52 -> 16/52。见 P-49。

    Args:
        refs: 参考方向按优先级排列。前一个与 y 近乎正交（|点积| < 1e-3）时
            退到下一个，因此结果与"哪个参考恰好退化"无关。
    """
    z = z_axis / z_axis.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    x = long_axis - (long_axis * z).sum(dim=-1, keepdim=True) * z
    y = torch.cross(z, x, dim=-1)
    score = torch.zeros(y.shape[0], 1, device=y.device, dtype=y.dtype)
    for r in refs:
        d = (y * r).sum(dim=-1, keepdim=True)
        score = torch.where(score.abs() < 1e-3, d, score)
    return torch.where(score < 0.0, -torch.ones_like(score), torch.ones_like(score))


def quat_face_and_up(z_axis: torch.Tensor, up_hint: torch.Tensor,
                     long_axis: torch.Tensor | None = None,
                     sign: torch.Tensor | None = None) -> torch.Tensor:
    """把板"正对某个表面"，并且**把朝向标记摆到一个确定的方向上**。

    `quat_from_frame` 只钉住工作面法向（局部 +Z）和局部 +X，**局部 +Y 是自己
    落下来的**——而深色鳍就长在 +Y 那条边上（`build_assets.PlateCfg`）。
    两块板一旦面对面（对捏、双面夹持），两个 +Z 相反，+Y 也就跟着相反：
    实测擦拭持工具时一块板的鳍朝下（-1.00）、另一块朝上（+1.00），
    探针上所有对捏原语的两块板夹角余弦都是 -1.00。

    **物理上这是恒等**——板是 35×25×3 的长方体，绕自身任一主轴转 180° 与自身
    重合，标记又只有视觉没有碰撞。但朝向标记的**唯一用途**就是让人在录像里
    读出板的姿态（`plan/06` §7 的人工检查），两块板的标记互相矛盾时它就废了。

    Args:
        z_axis: (N, 3) 工作面法向的世界方向（从板指向被接触的表面）。
        up_hint: (N, 3) 希望 **局部 +Y（鳍所在边）** 对齐的世界方向。
        long_axis: (N, 3)，可选。**长边（局部 +X，35 mm）被物理约束**时给它——
            例如推销钉时长边必须与销钉轴平行，横过来板会从柱面滑脱（P-46）。
            给了它就只在 ±long_axis 两个等价解里挑符号，长边的**轴**不动。
        sign: (N, 1)，可选。由 `roll_sign` **在 episode 开始时算一次**再传进来。
            姿态在控制回路里每步重算，符号却必须全程不变，否则目标会在两个
            等价解之间跳（P-49）。不给时就地按 up_hint 判，只适用于
            ``y·up_hint`` 明显不为零的场合。
    """
    z = z_axis / z_axis.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    if long_axis is not None:
        x = long_axis - (long_axis * z).sum(dim=-1, keepdim=True) * z
        if sign is None:
            y = torch.cross(z, x, dim=-1)
            sign = torch.where((y * up_hint).sum(dim=-1, keepdim=True) < 0.0,
                               -torch.ones_like(x[:, :1]), torch.ones_like(x[:, :1]))
        return quat_from_frame(z, x * sign)
    # 想要 Y ≈ up_hint，而 quat_from_frame 里 Y = Z × X，故取 X = up_hint × Z
    return quat_from_frame(z, torch.cross(up_hint, z, dim=-1))


def quat_from_frame(z_axis: torch.Tensor, x_hint: torch.Tensor) -> torch.Tensor:
    """由"局部 +Z 指向哪、局部 +X 大致指向哪"构造姿态四元数 (N, 4)。

    采集板的工作面法向是局部 +Z，所以"把板正对某个表面"这件事，
    自然的写法就是给定 z 轴；``x_hint`` 决定板绕自身法向的那一个自由度
    （板是长方形，长边朝哪不是无所谓的）。

    比逐个轴角相乘可靠：`plan/03` §2.4 的探针物体上有几十个接触位点，
    法向千奇百怪，用轴角拼会在某些朝向上碰到万向锁式的分支错误。

    Args:
        z_axis: (N, 3) 期望的局部 +Z 世界方向，不必归一化。
        x_hint: (N, 3) 期望的局部 +X 大致方向；会被正交化到垂直于 z_axis。
    """
    z = z_axis / z_axis.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    x = x_hint - (x_hint * z).sum(dim=-1, keepdim=True) * z
    bad = x.norm(dim=-1) < 1e-6
    if bad.any():
        # x_hint 与 z 共线时随便换一个不共线的参考方向
        alt = torch.zeros_like(x)
        alt[:, 0] = 1.0
        alt[z[:, 0].abs() > 0.9, 0] = 0.0
        alt[z[:, 0].abs() > 0.9, 1] = 1.0
        x = torch.where(bad.unsqueeze(-1),
                        alt - (alt * z).sum(dim=-1, keepdim=True) * z, x)
    x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    y = torch.cross(z, x, dim=-1)
    m = torch.stack([x, y, z], dim=-1)                       # 列向量即局部轴

    t = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    q = torch.zeros(z.shape[0], 4, device=z.device, dtype=z.dtype)
    big = t > 0
    s = torch.sqrt((t + 1.0).clamp_min(1e-12)) * 2.0
    q[big, 0] = 0.25 * s[big]
    q[big, 1] = (m[big, 2, 1] - m[big, 1, 2]) / s[big]
    q[big, 2] = (m[big, 0, 2] - m[big, 2, 0]) / s[big]
    q[big, 3] = (m[big, 1, 0] - m[big, 0, 1]) / s[big]
    # 迹为负时按最大对角元分支，避免除以接近零的数
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        sel = ~big & (m[:, i, i] >= m[:, j, j]) & (m[:, i, i] >= m[:, k, k])
        if not sel.any():
            continue
        si = torch.sqrt((1.0 + m[sel, i, i] - m[sel, j, j] - m[sel, k, k])
                        .clamp_min(1e-12)) * 2.0
        q[sel, 0] = (m[sel, k, j] - m[sel, j, k]) / si
        q[sel, 1 + i] = 0.25 * si
        q[sel, 1 + j] = (m[sel, j, i] + m[sel, i, j]) / si
        q[sel, 1 + k] = (m[sel, k, i] + m[sel, i, k]) / si
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)
