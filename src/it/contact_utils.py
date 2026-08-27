

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
