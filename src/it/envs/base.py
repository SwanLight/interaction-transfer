"""浮动底座执行器的通用动作/观测处理。

`plan/01` §1 规则 3：所有浮动底座都是**动力学体，由 PD 外力和力矩驱动**，
禁止直接写 root pose——kinematic 体不参与动力学求解、能施加无限大的力，
接触力将失去物理意义（P-09）。

`plan/04` §2：Expert 动作是「浮动底座期望位移/姿态增量，经 PD 转换为外力和
力矩」。本模块实现这条：policy 输出增量 → 累加成目标位姿 → FloatingPD → wrench。
"""

from __future__ import annotations

import torch


class FloatingBaseAction:
    """把 policy 的 6 维增量动作转成浮动底座的目标位姿。

    动作是**增量**而非绝对位姿，因为绝对位姿会让策略必须先学会「世界坐标系
    在哪」，与本项目要研究的东西无关。增量还天然带一个动作幅度上限，
    对应 `plan/04` §8 的 action smoothness 惩罚。

    Args:
        num_envs: 环境数。
        device: torch 设备。
        pos_scale: 每步位移增量上限（m）。50 Hz 下 0.01 m → 0.5 m/s 上限。
        rot_scale: 每步姿态增量上限（rad）。
        pos_limit: 目标位置相对初始位置的活动范围（m），防止目标跑到场景外。
    """

    def __init__(self, num_envs: int, device, pos_scale: float = 0.01,
                 rot_scale: float = 0.06, pos_limit: float = 0.6):
        self.n, self.device = num_envs, device
        self.pos_scale, self.rot_scale, self.pos_limit = pos_scale, rot_scale, pos_limit
        self.target_pos = torch.zeros(num_envs, 3, device=device)
        self.target_quat = torch.zeros(num_envs, 4, device=device)
        self.target_quat[:, 0] = 1.0
        self.origin = torch.zeros(num_envs, 3, device=device)

    def reset(self, pos_w: torch.Tensor, quat_w: torch.Tensor, env_ids=None):
        if env_ids is None:
            self.target_pos[:] = pos_w
            self.target_quat[:] = quat_w
            self.origin[:] = pos_w
        else:
            self.target_pos[env_ids] = pos_w
            self.target_quat[env_ids] = quat_w
            self.origin[env_ids] = pos_w

    def step(self, action: torch.Tensor):
        """action: (N, 6)，已 clamp 到 [-1, 1]。返回 (target_pos, target_quat)。"""
        from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

        a = action.clamp(-1.0, 1.0)
        self.target_pos = self.target_pos + a[:, :3] * self.pos_scale
        # 限制在初始位置附近，避免目标漂到场景外导致 PD 输出饱和
        d = self.target_pos - self.origin
        self.target_pos = self.origin + d.clamp(-self.pos_limit, self.pos_limit)

        e = a[:, 3:6] * self.rot_scale
        dq = quat_from_euler_xyz(e[:, 0], e[:, 1], e[:, 2])
        self.target_quat = quat_mul(dq, self.target_quat)
        self.target_quat = self.target_quat / self.target_quat.norm(dim=-1, keepdim=True)
        return self.target_pos, self.target_quat


def contact_summary(sensor, dt: float, num_envs: int, device):
    """接触信息的定长摘要，供 Expert 的特权观测使用。

    每个 env 返回 8 维：
      [0]   接触点数（归一化）
      [1:4] 合法向力矢量（世界系）
      [4:7] 合摩擦力矢量（世界系）
      [7]   最大单点法向力

    定长是必须的——`plan/02` §7 第 3 条要求「改变接触体数量后表示维度不变」，
    观测同理，否则策略会依赖接触点的个数与顺序。
    """
    from it.contact_utils import extract_contact_points

    out = torch.zeros(num_envs, 8, device=device)
    cps = extract_contact_points(sensor, dt)
    for e in range(num_envs):
        cp = cps[e]
        if cp.is_empty():
            continue
        out[e, 0] = min(cp.num_contacts / 8.0, 1.0)
        out[e, 1:4] = (cp.normal_forces.unsqueeze(-1) * cp.normals).sum(dim=0)
        out[e, 4:7] = cp.friction_forces.sum(dim=0)
        out[e, 7] = cp.normal_forces.abs().max()
    return out
