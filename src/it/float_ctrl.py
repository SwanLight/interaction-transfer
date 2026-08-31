"""浮动底座的 PD wrench 控制器。

`plan/01` §1 规则 3 规定所有浮动底座都是**动力学体，由 PD 外力和力矩驱动**，
禁止直接写 root pose——kinematic 体不参与动力学求解，能施加无限大的力，
接触力将失去物理意义（`log/pitfalls.md` P-09）。

本模块是那条规则的实现。S1 自检用它把执行器稳定在指定位姿，Phase I 的
Expert 和 Executor 的浮动底座动作也走同一条路径。
"""

from __future__ import annotations

import torch


def _quat_err(q_cur: torch.Tensor, q_des: torch.Tensor) -> torch.Tensor:
    """姿态误差的旋转矢量表示（世界系），shape (N, 3)。四元数为 (w, x, y, z)。

    小角度必须走 ``2*v`` 分支：误差趋近 0 时 sin(θ/2)→0，若直接用
    ``v / sin_half`` 归一化轴向量，即使 clamp 到 1e-12 也会让轴炸到 1e12，
    PD 输出巨大力矩把物体甩飞。实测钩杆因此飞到 -5 km 外。
    """
    from isaaclab.utils.math import quat_conjugate, quat_mul

    q_err = quat_mul(q_des, quat_conjugate(q_cur))
    q_err = torch.where(q_err[:, 0:1] < 0, -q_err, q_err)   # 最短路径
    v = q_err[:, 1:]
    vn = v.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(vn, q_err[:, 0:1].clamp(-1.0, 1.0))
    return torch.where(vn > 1e-6, v / vn.clamp_min(1e-6) * angle, 2.0 * v)


class FloatingPD:
    """把浮动刚体 PD 到目标位姿，可叠加前馈力/力矩。

    典型用法：让执行器悬停在接触位姿并压出指定法向力::

        pd = FloatingPD(rod, kp_pos=400.0, kd_pos=40.0, kp_rot=8.0, kd_rot=1.2)
        wrench = pd.compute(target_pos, target_quat, ff_force=press_force)
        rod.set_external_force_and_torque(*wrench)

    增益的量纲是"每千克"，内部乘质量，因此换不同质量的执行器不必重调。
    """

    def __init__(self, asset, kp_pos=400.0, kd_pos=40.0, kp_rot=20.0, kd_rot=4.0,
                 max_force=200.0, max_torque=20.0, compensate_gravity=True, g=9.81,
                 kd_force=30.0, rot_gain_basis="mass"):
        self.a = asset
        self.kp_pos, self.kd_pos = kp_pos, kd_pos
        self.kp_rot, self.kd_rot = kp_rot, kd_rot
        self.max_force, self.max_torque = max_force, max_torque
        self.compensate_gravity = compensate_gravity
        self.g = g
        self.kd_force = kd_force
        self.mass = asset.data.default_mass.sum(dim=-1).to(asset.device)
        self.rot_gain_basis = rot_gain_basis
        self.inertia = None
        if rot_gain_basis == "inertia":
            # (N, 9) 行主序的惯量张量，在**本体系**里、相对质心（Isaac Lab 的约定）。
            inertia = asset.data.default_inertia.to(asset.device).reshape(-1, 3, 3)
            self.inertia = inertia

    def compute(self, target_pos_w, target_quat_w=None, ff_force=None, ff_torque=None,
                force_mask=None, force_dir=None):
        """返回 (force, torque)，形状均为 (N, 1, 3)，可直接喂给
        ``set_external_force_and_torque``。

        Args:
            force_mask: (N, 3) bool。为 True 的轴走**纯力控**——输出直接取
                ``ff_force``（外加重力补偿），不做位置 PD。只能表达
                **与世界轴对齐**的力控方向。
            force_dir: (N, 3)。给定时改用**沿任意方向**的混合控制：
                沿 ``force_dir`` 力控，其正交补里仍然位置 PD。
                探针物体上的接触面法向千奇百怪，用 ``force_mask`` 只能整轴
                切换——实测把三个轴全设成力控之后，切向的位置指令完全失效，
                "捏住物体挪走"这类原语的物体位移恒等于 0。
                两个参数同时给时以 ``force_dir`` 为准。

        为什么需要混合控制：位置 PD 和前馈压力会互相抵消。PD 把物体牢牢固定
        在目标位姿，压不进接触面，接触力只有 ``kp·m·穿透量``——实测把推子
        放在距离表面 0.5 mm 处，法向力只有 0.06 N，而目标是 25 N。
        要产生受控的接触力，法向必须是力控，其余轴才用位置 PD。
        """
        d = self.a.data
        m = self.mass.unsqueeze(-1)

        e_p = target_pos_w - d.root_pos_w
        f = self.kp_pos * m * e_p - self.kd_pos * m * d.root_lin_vel_w
        if self.compensate_gravity:
            f = f + torch.tensor([0.0, 0.0, self.g], device=f.device) * m

        if target_quat_w is not None:
            e_r = _quat_err(d.root_quat_w, target_quat_w)
            alpha = self.kp_rot * e_r - self.kd_rot * d.root_ang_vel_w
        else:
            alpha = -self.kd_rot * d.root_ang_vel_w
        tq = self._torque(alpha, m)

        if ff_force is not None and force_dir is not None:
            n = force_dir / force_dir.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            grav = torch.zeros_like(f)
            if self.compensate_gravity:
                grav[:, 2] = self.g * m.squeeze(-1)
            # 位置 PD 只保留切向分量，法向换成力控 + 法向速度阻尼
            f_tan = f - (f * n).sum(-1, keepdim=True) * n
            f_nrm = ((ff_force * n).sum(-1, keepdim=True)
                     + (grav * n).sum(-1, keepdim=True)
                     - self.kd_force * m * (d.root_lin_vel_w * n).sum(-1, keepdim=True)) * n
            f = f_tan + f_nrm
        elif ff_force is not None:
            if force_mask is None:
                f = f + ff_force
            else:
                grav = torch.zeros_like(f)
                if self.compensate_gravity:
                    grav[:, 2] = self.g * m.squeeze(-1)
                # 力控轴必须带速度阻尼，否则是无阻尼自由加速：25 N / 0.2 kg
                # = 125 m/s²，8 mm 间隙 1.3 个物理步就撞上，撞击速度 1.37 m/s，
                # 单步穿透 11 mm —— 直接穿模弹飞，测不到任何接触力。
                # 稳态接触时 v→0，输出精确收敛到 ff_force。
                damp = self.kd_force * m * d.root_lin_vel_w
                f = torch.where(force_mask, ff_force + grav - damp, f)
        if ff_torque is not None:
            tq = tq + ff_torque

        f = f.clamp(-self.max_force, self.max_force).unsqueeze(1)
        tq = tq.clamp(-self.max_torque, self.max_torque).unsqueeze(1)
        return f, tq

    def _torque(self, alpha, m):
        """把期望角加速度换成力矩。

        ``rot_gain_basis="mass"`` 是历史行为（``τ = kp·m·e``）：**量纲上是错的**，
        力矩需要的是转动惯量而不是质量。双板执行器上它碰巧能用——板的三个主惯量
        量级相近。换成杆类执行器就不行了：垫头杆绕**自身轴**的惯量比横向小两个数量级，
        同一组增益给出的有效 ω_n ≈ 424 rad/s，而物理步长是 1/150 s
        （``ω_n·dt ≈ 2.8 ≫ 2``），显式积分直接发散。实测 E-I 的冒烟里
        **杆的角速度 76 rad/s**，接触点线速度随之虚高，`r_mode` 报到 −31.6。
        这与 P-52（板角速度顶在 PhysX 的 100 rad/s 上限）是同一族问题。

        ``rot_gain_basis="inertia"``：``τ = R·I_body·Rᵀ·α``，增益因此读作
        "每弧度多少 rad/s²"，与执行器的形状无关——**跨形态是这个项目的主线**，
        增益不该跟着形态重调。新代码一律用这一档；默认留 ``"mass"`` 是因为
        S1/S2/S3 的脚本与已冻结的结论都建立在旧行为上（D-21）。
        """
        if self.rot_gain_basis != "inertia":
            return m * alpha                       # 历史行为，与旧版逐位相同
        from isaaclab.utils.math import matrix_from_quat

        rot = matrix_from_quat(self.a.data.root_quat_w)
        body = torch.einsum("nji,nj->ni", rot, alpha)          # 世界 -> 本体
        body = torch.einsum("nij,nj->ni", self.inertia, body)
        return torch.einsum("nij,nj->ni", rot, body)           # 本体 -> 世界
