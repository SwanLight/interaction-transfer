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
    """姿态误差的轴角表示（世界系），shape (N, 3)。四元数为 (w, x, y, z)。"""
    from isaaclab.utils.math import quat_error_magnitude, quat_mul, quat_conjugate

    del quat_error_magnitude
    q_err = quat_mul(q_des, quat_conjugate(q_cur))
    # 取最短路径
    q_err = torch.where(q_err[:, 0:1] < 0, -q_err, q_err)
    angle = 2.0 * torch.acos(q_err[:, 0].clamp(-1.0, 1.0))
    sin_half = (1.0 - q_err[:, 0] ** 2).clamp_min(1e-12).sqrt()
    axis = q_err[:, 1:] / sin_half.unsqueeze(-1)
    return axis * angle.unsqueeze(-1)


class FloatingPD:
    """把浮动刚体 PD 到目标位姿，可叠加前馈力/力矩。

    典型用法：让执行器悬停在接触位姿并压出指定法向力::

        pd = FloatingPD(rod, kp_pos=400.0, kd_pos=40.0, kp_rot=8.0, kd_rot=1.2)
        wrench = pd.compute(target_pos, target_quat, ff_force=press_force)
        rod.set_external_force_and_torque(*wrench)

    增益的量纲是"每千克"，内部乘质量，因此换不同质量的执行器不必重调。
    """

    def __init__(self, asset, kp_pos=400.0, kd_pos=40.0, kp_rot=20.0, kd_rot=4.0,
                 max_force=200.0, max_torque=20.0, compensate_gravity=True, g=9.81):
        self.a = asset
        self.kp_pos, self.kd_pos = kp_pos, kd_pos
        self.kp_rot, self.kd_rot = kp_rot, kd_rot
        self.max_force, self.max_torque = max_force, max_torque
        self.compensate_gravity = compensate_gravity
        self.g = g
        self.mass = asset.data.default_mass.sum(dim=-1).to(asset.device)

    def compute(self, target_pos_w, target_quat_w=None, ff_force=None, ff_torque=None):
        """返回 (force, torque)，形状均为 (N, 1, 3)，可直接喂给
        ``set_external_force_and_torque``。"""
        d = self.a.data
        m = self.mass.unsqueeze(-1)

        e_p = target_pos_w - d.root_pos_w
        f = self.kp_pos * m * e_p - self.kd_pos * m * d.root_lin_vel_w
        if self.compensate_gravity:
            f = f + torch.tensor([0.0, 0.0, self.g], device=f.device) * m

        if target_quat_w is not None:
            e_r = _quat_err(d.root_quat_w, target_quat_w)
            tq = self.kp_rot * m * e_r - self.kd_rot * m * d.root_ang_vel_w
        else:
            tq = -self.kd_rot * m * d.root_ang_vel_w

        if ff_force is not None:
            f = f + ff_force
        if ff_torque is not None:
            tq = tq + ff_torque

        f = f.clamp(-self.max_force, self.max_force).unsqueeze(1)
        tq = tq.clamp(-self.max_torque, self.max_torque).unsqueeze(1)
        return f, tq
