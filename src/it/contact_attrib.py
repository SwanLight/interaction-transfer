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


def classify_plate_face(normals_local: torch.Tensor,
                        face_align: float = 0.7) -> torch.Tensor:
    """判定接触落在板的哪个面，返回 (N, K) int8：0=工作面 / 1=背面 / 2=侧边。

    **按接触法向判，不按接触点的位置判。** 板只有 3 mm 厚，而 PhysX 报的接触点
    在两个表面之间浮动——实测一次正常的工作面接触，点的局部 z 在 0.3~1.7 mm
    之间摆（板半厚才 1.5 mm）。用"离哪个面近"去判，会把一半的正常接触判成侧边，
    于是验收表上凭空多出 36% 的"拿边角蹭"，而那是分类器的错，不是物理的错。

    法向就干净得多：实测 100% 的工作面接触满足 ``|n_z| > 0.7``，且法向在板局部系
    里恒指向**工作面朝外**的方向（+Z）。真正的边缘接触，法向躺在板平面内，
    ``|n_z|`` 会明显小于 1。

    Args:
        normals_local: (N, K, 3)，已转进**板局部系**的接触法向。
        face_align: 认定为"面接触"的 ``|n_z|`` 下限。
    """
    nz = normals_local[..., 2]
    mode = torch.full(nz.shape, 2, dtype=torch.int8, device=nz.device)
    mode[nz > face_align] = 0
    mode[nz < -face_align] = 1
    return mode


def plate_face_offset(pts_local: torch.Tensor) -> torch.Tensor:
    """接触点在板面内的位置，返回 (N, K, 2)，单位 m，原点在板心。

    用来看接触是压在面**中央**还是挂在**边角**上。板半宽 (17.5, 12.5) mm，
    所以 |x| 接近 17.5 就说明只有一条边在受力。
    """
    return pts_local[..., :2]


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
    bar_x: float,
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
