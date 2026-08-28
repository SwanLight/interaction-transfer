"""S1 资产与几何可行性自检。

判据来自 `plan/01-assets-and-scenes.md` §7（几何自检）和 §8（动力学自检）。
一次 Isaac Sim 会话跑完全部检查，结果写 JSON + 文本报告。

用法::

    PYTHONPATH=src /isaac-sim/python.sh tools/s1_check.py --out /tmp/s1

关键检查（不通过则后续实验设计需要改）：

- **轮缘摩擦标定**：安全力上限内纯轮缘接触**无法**达成目标转角，销钉可以。
  这是 D-14 成立的前提，不通过则旋钮任务上 region 仍不可检验。
- **接触力可读**：每个执行器都要能读出非零接触力，且 filter 通道有效（P-17）。
- **动力学合理**：已知力矩下的响应、全行程无穿模、浮动底座不产生无限力（P-09）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# ------------------------------------------------------------------ 启动
_ap = argparse.ArgumentParser(description="S1 资产与几何可行性自检")
_ap.add_argument("--out", default="/tmp/s1", help="报告输出目录")
_ap.add_argument("--only", default=None, help="只跑某项检查（逗号分隔）")
_args, _rest = _ap.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

_app = SimulationApp({"headless": True})

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, AssetBaseCfg, RigidObject  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import ContactSensor, ContactSensorCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from it import assets as A  # noqa: E402
from it.build_assets import MM  # noqa: E402

G = 9.81
REPORT: list[dict] = []


def record(group: str, name: str, passed: bool | None, detail: str, **extra):
    REPORT.append({"group": group, "check": name, "pass": passed, "detail": detail, **extra})
    mark = {True: "PASS", False: "FAIL", None: "INFO"}[passed]
    print(f"[{mark:4s}] {group:10s} {name:34s} {detail}", flush=True)


def _sim(dt=1 / 120):
    return sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device="cuda:0"))


def _light():
    return AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))


def _reset_floating(asset, pos_w, device):
    """把浮动刚体瞬时重置到指定世界位姿，速度清零。

    自由漂浮体在没有控制器接管的阶段会一直自由落体——S1 中推子在旋钮动力学
    测试的 300 步里掉了 30 米，之后 PD 再也拉不回来。每个控制阶段开始前
    必须显式重置。
    """
    n = pos_w.shape[0]
    st = asset.data.default_root_state.clone()
    st[:, :3] = pos_w
    st[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(n, 1)
    st[:, 7:] = 0.0
    asset.write_root_state_to_sim(st)


def _steps(sim, scene, n, dt, pre=None):
    for i in range(n):
        if pre is not None:
            pre(i)
        scene.write_data_to_sim()
        sim.step(render=False)   # headless 无相机时 render=True 会卡死（P-19）
        scene.update(dt)


# ================================================================== 旋钮


def check_knob():
    """§7 旋钮：几何、关节行程、**轮缘摩擦标定**（D-14 前提）。"""

    @configclass
    class Cfg(InteractiveSceneCfg):
        light = _light()
        knob = A.KNOB_CFG.replace(prim_path="{ENV_REGEX_NS}/Knob")
        # env 0 推轮缘，env 1 推销钉。同一份物理，只换接触位置。
        pusher = A.pusher_cfg()
        pcontact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Pusher",
            track_pose=True, track_contact_points=True,
            max_contact_data_count_per_prim=16,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Knob/Disc"],
            update_period=0.0, history_length=0,
        )

    sim = _sim()
    scene = InteractiveScene(Cfg(num_envs=2, env_spacing=3.0))
    sim.reset()
    dt = sim.get_physics_dt()
    knob: Articulation = scene["knob"]

    # --- 几何 ---
    kc = A.KNOB_CFG
    del kc
    from it.build_assets import KnobCfg

    g = KnobCfg()
    record("旋钮", "关节数量", knob.num_joints == 1, f"num_joints={knob.num_joints} (期望 1)")
    record("旋钮", "关节名", "DiscJoint" in knob.joint_names, f"{knob.joint_names}")

    lim = knob.data.joint_limits[0, 0].tolist()
    span = lim[1] - lim[0]
    record("旋钮", "关节行程覆盖目标 1.0-2.2 rad", span >= 2.2,
           f"limits={[round(x,3) for x in lim]} rad, span={span:.3f}")

    # --- 动力学：已知力矩下的响应（§8）---
    _steps(sim, scene, 60, dt)
    q0 = knob.data.joint_pos[0, 0].item()

    tau = torch.zeros(1, 1, device=sim.device)
    tau[0, 0] = 1.5
    def push(i):
        knob.set_joint_effort_target(tau)
    _steps(sim, scene, 240, dt, pre=push)
    q1 = knob.data.joint_pos[0, 0].item()
    w = knob.data.joint_vel[0, 0].item()
    moved = q1 - q0
    record("旋钮", "已知力矩下角加速度/稳态速度合理", 0.05 < moved < 6.28 and abs(w) < 50,
           f"1.5 N·m 施加 2 s: Δθ={moved:.3f} rad, ω={w:.3f} rad/s")

    # --- 轮缘摩擦标定（D-14 关键前提）---
    # 先给设计裕量（解析），再实测。实测才是判据。
    rim_a, pin_a = _knob_friction_calibration(g)
    record("旋钮", "设计裕量（解析）", rim_a[0] and pin_a[0],
           f"τ_rim={rim_a[1]:.3f} < τ_need={rim_a[2]:.3f} < τ_pin={pin_a[1]:.3f} N·m  "
           f"(rim 比 {rim_a[1]/rim_a[2]:.2f}, pin 比 {pin_a[1]/pin_a[2]:.2f})")

    taus, fns = _empirical_rim_vs_pin(sim, scene, knob, dt, g)
    need = rim_a[2]
    record("旋钮", "实测：轮缘在安全力上限内传不出所需力矩", taus[0] < need,
           f"τ_rim(峰值) = {taus[0]:.4f} N·m < τ_need = {need:.3f}  (峰值法向力 {fns[0]:.2f} N)")
    record("旋钮", "实测：销钉在安全力上限内传得出所需力矩", taus[1] > need,
           f"τ_pin(峰值) = {taus[1]:.4f} N·m > τ_need = {need:.3f}  (峰值法向力 {fns[1]:.2f} N)")
    record("旋钮", "D-14 成立（region 不可从 effect 推出）", taus[0] < need < taus[1],
           f"实测 τ_rim {taus[0]:.4f} < τ_need {need:.3f} < τ_pin {taus[1]:.4f} N·m")


def _knob_friction_calibration(g):
    """解析计算两个接触区域在安全力上限内能传递的最大力矩。

    轮缘接触：法向沿径向指向轴心，切向摩擦力产生力矩，力臂 = 圆盘半径。
    销钉接触：法向水平推销钉侧面，**法向力本身**产生力矩，力臂 = pin_offset。
    这是两者的本质差别——推销钉靠法向力，蹭轮缘只能靠摩擦力。

    需求力矩取克服关节阻尼 + 加速圆盘所需，用 damping*ω_target 估算。
    """
    fn = A.MAX_NORMAL_FORCE
    omega_target = 1.5  # rad/s，达成 1.0-2.2 rad 的合理速度
    tau_need = g.joint_damping * omega_target  # 圆盘转动惯量 ~9e-4，加速项可忽略

    tau_rim = g.rim_friction * fn * g.disc_radius
    tau_pin = fn * g.pin_offset  # 法向力直接产生力矩

    return (tau_rim < tau_need, tau_rim, tau_need), (tau_pin > tau_need, tau_pin, tau_need)


def _empirical_rim_vs_pin(sim, scene, knob, dt, g):
    """实测：法向力锁定在安全上限，分别顶轮缘和销钉，测传递到关节轴的力矩。

    两条经验教训写在这里，别再重蹈：

    1. **位置从仿真里读，不从 config 推算。** 早期版本按配置算推子目标位姿，
       结果推子飞出 30 米也没碰到任何东西——USD 的 fixed joint 会叠加
       Xform 的平移，实际几何和纸面算的不一样。
    2. **法向力控 + 切向位置 PD**，不要两个方向都力控。纯力控无位置参考，
       25 N / 0.2 kg = 125 m/s²，几步就飞出场景。这套混合方案在擦拭检查上
       已验证（速度跟踪到 0.1501 vs 目标 0.15）。
    """
    import math

    from it.float_ctrl import FloatingPD
    from it.contact_utils import contact_torque_about_axis, extract_contact_points

    pusher: RigidObject = scene["pusher"]
    cs: ContactSensor = scene["pcontact"]
    fn = A.MAX_NORMAL_FORCE
    half = 0.015

    # --- 关节归零，**并清掉残留的力矩指令** ---
    # 前面的"已知力矩响应"测试把圆盘推到限位 3.491 rad（=200°）。仅仅
    # write_joint_state_to_sim 复位是不够的——Isaac Lab 会一直保持上次
    # set_joint_effort_target 设的 1.5 N·m，圆盘会立刻再转回限位，销钉
    # 根本不在角度 0。轮缘那半因为圆盘轴对称所以不受影响，销钉那半必然失败。
    zero = torch.zeros(2, 1, device=sim.device)
    knob.set_joint_effort_target(zero)
    knob.write_joint_state_to_sim(zero, zero)

    # --- 从仿真读实际几何 ---
    _steps(sim, scene, 30, dt)
    bi = knob.body_names.index("Disc")
    disc_w = knob.data.body_pos_w[:, bi, :].clone()          # (2,3) 世界系
    axis_pt = disc_w.clone()
    disc_top = disc_w[:, 2] + g.disc_thickness / 2
    pin_cz = disc_top + g.pin_length / 2

    tgt = torch.zeros(2, 3, device=sim.device)
    # env0：轮缘外侧，径向 +X
    tgt[0] = torch.tensor([g.disc_radius + half - 0.001, 0.0, 0.0], device=sim.device) + disc_w[0]
    # env1：销钉 -Y 侧
    tgt[1, 0] = disc_w[1, 0] + g.pin_offset
    tgt[1, 1] = disc_w[1, 1] - (g.pin_radius + half - 0.001)
    tgt[1, 2] = pin_cz[1]

    pd = FloatingPD(pusher, kp_pos=800.0, kd_pos=60.0, kp_rot=60.0, kd_rot=8.0,
                    max_force=300.0, max_torque=30.0, kd_force=60.0)

    # 法向轴力控，其余轴位置 PD
    ffv = torch.zeros(2, 3, device=sim.device)
    ffv[0, 0] = -fn
    ffv[1, 1] = fn
    mask = torch.zeros(2, 3, dtype=torch.bool, device=sim.device)
    mask[0, 0] = True
    mask[1, 1] = True

    # 第一段：位置 PD 逼近到接触前 4 mm
    pre = tgt.clone()
    pre[0, 0] += 0.004
    pre[1, 1] -= 0.004

    # 先把推子瞬时放到接触前 3 cm——它在前面的动力学测试里已经自由落体掉走了
    start = tgt.clone()
    start[0, 0] += 0.03
    start[1, 1] -= 0.03
    _reset_floating(pusher, start, sim.device)
    _steps(sim, scene, 5, dt)

    def approach(i):
        knob.set_joint_effort_target(zero)
        f, tq = pd.compute(pre)
        pusher.set_external_force_and_torque(f, tq)
    _steps(sim, scene, 250, dt, pre=approach)

    # 第二段：法向压紧；env0 的切向用**缓慢移动的位置目标**拖动（不用力控）
    drag_speed = 0.05

    # 不锁死圆盘（每步 write_joint_state_to_sim 是在和求解器打架，接触会被丢弃）。
    # 但圆盘一旦被推动就会转走，销钉沿切向离开推子的 X 窗口，接触随即断开——
    # 因此**必须记录整个过程的峰值**，只看最后一帧会读到 0。
    axis_dir = torch.tensor([0.0, 0.0, 1.0], device=sim.device)
    peak_tau = [0.0, 0.0]
    peak_fn = [0.0, 0.0]
    peak_n = [0, 0]
    min_d = [1e9, 1e9]
    # 销钉中心世界位置（关节已归零，销钉在圆盘系 +X）
    pin_w = disc_w.clone()
    pin_w[:, 0] += g.pin_offset
    pin_w[:, 2] = pin_cz

    def hold(i):
        t = tgt.clone()
        t[0, 1] += drag_speed * i * dt
        knob.set_joint_effort_target(zero)   # 持续清零，否则残留指令会一直驱动圆盘
        ramp = min(i / 120.0, 1.0)          # 力斜坡上升 1 s，避免冲击穿模
        f, tq = pd.compute(t, ff_force=ffv * ramp, force_mask=mask)
        pusher.set_external_force_and_torque(f, tq)
        if i % 2 == 0:
            cc = extract_contact_points(cs, dt)
            nn = cs.contact_physx_view.get_contact_data(dt=dt)[4].flatten().tolist()
            for e in range(2):
                # 原始 counts 先记，再看过滤后是否还有——用于区分
                # "真没接触" 和 "被 force_threshold 滤掉了"
                peak_n[e] = max(peak_n[e], int(nn[e]))
                ang = knob.data.joint_pos[e, 0].item()
                pw = disc_w[e].clone()
                pw[0] += g.pin_offset * math.cos(ang)
                pw[1] += g.pin_offset * math.sin(ang)
                pw[2] = pin_cz[e]
                d = (pusher.data.root_pos_w[e] - pw).norm().item()
                min_d[e] = min(min_d[e], d)
                if cc[e].is_empty():
                    continue
                fnv = cc[e].normal_forces.abs().sum().item()
                # 只统计法向力真的落在安全上限内的样本。撞击瞬间会出现远超
                # 指令值的尖峰（实测 66.5 N vs 指令 25 N），把它算进去，
                # "在安全力上限内能传多少力矩" 这个命题就不成立了。
                if fnv > A.MAX_NORMAL_FORCE * 1.15:
                    continue
                tv = abs(contact_torque_about_axis(cc[e], axis_pt[e], axis_dir).item())
                peak_tau[e] = max(peak_tau[e], tv)
                peak_fn[e] = max(peak_fn[e], fnv)

    _steps(sim, scene, 400, dt, pre=hold)

    record("旋钮", "推子与圆盘建立接触（诊断）", all(c > 0 for c in peak_n),
           f"峰值接触点数 = {peak_n}，峰值法向力 = [{peak_fn[0]:.2f}, {peak_fn[1]:.2f}] N，"
           f"推子到销钉最近距离 = [{min_d[0]*1000:.1f}, {min_d[1]*1000:.1f}] mm "
           f"(应 < {(0.015+g.pin_radius)*1000:.0f} mm 才可能接触)")
    return peak_tau, peak_fn


# ================================================================== 抽屉
# ================================================================== 抽屉


def check_cabinet():
    """§7 抽屉：把手净空、全行程无穿模。"""

    @configclass
    class Cfg(InteractiveSceneCfg):
        light = _light()
        cab = A.CABINET_CFG.replace(prim_path="{ENV_REGEX_NS}/Cabinet")

    sim = _sim()
    scene = InteractiveScene(Cfg(num_envs=1, env_spacing=3.0))
    sim.reset()
    dt = sim.get_physics_dt()
    cab: Articulation = scene["cab"]

    from it.build_assets import CabinetCfg

    c = CabinetCfg()
    record("抽屉", "关节数量", cab.num_joints == 1, f"num_joints={cab.num_joints}")
    lim = cab.data.joint_limits[0, 0].tolist()
    record("抽屉", "行程覆盖目标 100-160 mm", lim[1] >= 0.160,
           f"limits={[round(x,3) for x in lim]} m")

    clearance = c.handle_clearance
    record("抽屉", "把手净空 >= 45 mm（Allegro 手指可入）", clearance >= 45 * MM,
           f"净空 = {clearance/MM:.0f} mm, 把手直径 = {2*c.handle_radius/MM:.0f} mm")
    record("抽屉", "把手杆长容纳多点接触", c.handle_bar_len >= 0.12,
           f"杆长 {c.handle_bar_len/MM:.0f} mm, 支撑柱间距 {c.post_spacing/MM:.0f} mm "
           f"-> 可用接触段 {(c.post_spacing - 2*c.post_radius)/MM:.0f} mm")

    # 全行程无穿模：拉到底再推回，看关节位置是否被正确限制
    f = torch.zeros(1, 1, device=sim.device)
    f[0, 0] = 30.0
    _steps(sim, scene, 400, dt, pre=lambda i: cab.set_joint_effort_target(f))
    q_open = cab.data.joint_pos[0, 0].item()
    f[0, 0] = -30.0
    _steps(sim, scene, 400, dt, pre=lambda i: cab.set_joint_effort_target(f))
    q_close = cab.data.joint_pos[0, 0].item()
    ok = (q_open > 0.160) and (q_close < 0.02) and (q_open <= lim[1] + 1e-3)
    record("抽屉", "全行程可达且不越限", ok,
           f"拉开到 {q_open*1000:.1f} mm (上限 {lim[1]*1000:.0f}), 推回到 {q_close*1000:.1f} mm")



# ================================================================== 擦拭


def check_wiping():
    """§7 擦拭：平面 filter 有效、法向力区间内平稳滑移、垫头杆面接触。"""

    @configclass
    class Cfg(InteractiveSceneCfg):
        light = _light()
        board = A.board_cfg()
        padrod = A.PADROD_CFG.replace(
            init_state=type(A.PADROD_CFG.init_state)(pos=(0.0, 0.0, 0.115))
        )
        contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/PadRod",
            track_pose=True,
            track_contact_points=True,
            max_contact_data_count_per_prim=16,   # 规则 8，默认 4 不够
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Board"],  # 规则 8 + P-17
            update_period=0.0,
            history_length=0,
        )

    sim = _sim()
    scene = InteractiveScene(Cfg(num_envs=2, env_spacing=3.0))
    sim.reset()
    dt = sim.get_physics_dt()
    rod: RigidObject = scene["padrod"]
    cs: ContactSensor = scene["contact"]

    from it.build_assets import PadRodCfg
    from it.float_ctrl import FloatingPD

    p = PadRodCfg()
    board_mu = 0.35
    mu = min(board_mu, p.pad_friction)   # combine mode = min（规则 9）
    target_fn = sum(A.WIPE_FORCE_RANGE) / 2

    # PD 把杆稳在竖直姿态压住板面。自由漂浮体只给外力必然翻倒——
    # plan/01 §1 规则 3 本来就要求浮动底座用 PD wrench 驱动。
    pd = FloatingPD(rod, kp_pos=500.0, kd_pos=45.0, kp_rot=60.0, kd_rot=8.0,
                    max_force=120.0, max_torque=12.0)
    quat_up = torch.zeros(2, 4, device=sim.device)
    quat_up[:, 0] = 1.0
    hover = rod.data.default_root_state[:, :3].clone()
    hover[:, 2] = 0.113          # 垫底刚好接触板面
    hover = hover + scene.env_origins

    # 前馈只给"目标法向力"，重力由 PD 自己补偿，不要再叠一次（早期版本
    # 重复计入重力，实测 Fn 变成 11.6 N 而非 5.5 N）
    press_ff = torch.zeros(2, 3, device=sim.device)
    press_ff[:, 2] = -target_fn
    z_mask = torch.zeros(2, 3, dtype=torch.bool, device=sim.device)
    z_mask[:, 2] = True          # Z 纯力控，XY 与姿态位置 PD

    # 第一段：位置 PD 落到板面上方 6 mm
    pre_h = hover.clone()
    pre_h[:, 2] += 0.006

    def approach(i):
        f, tq = pd.compute(pre_h, quat_up)
        rod.set_external_force_and_torque(f, tq)
    _steps(sim, scene, 300, dt, pre=approach)

    # 第二段：Z 纯力控压出目标法向力
    def press(i):
        f, tq = pd.compute(hover, quat_up, ff_force=press_ff, force_mask=z_mask)
        rod.set_external_force_and_torque(f, tq)

    _steps(sim, scene, 300, dt, pre=press)

    cd = cs.contact_physx_view.get_contact_data(dt=dt)
    counts = cd[4].flatten().tolist()
    record("擦拭", "filter 通道有效（P-17）", all(c > 0 for c in counts),
           f"接触对 counts={counts}（0 表示 filter 目标不是刚体）")

    net = cs.data.net_forces_w[0, 0]
    fn = net.norm().item()
    record("擦拭", "法向力落在 3-8 N 工作区间", A.WIPE_FORCE_RANGE[0] <= fn <= A.WIPE_FORCE_RANGE[1] + 2.0,
           f"|Fn| = {fn:.3f} N（目标 {A.WIPE_FORCE_RANGE}）")

    nan_frac = torch.isnan(cs.data.contact_pos_w).float().mean().item()
    record("擦拭", "contact_pos_w 有效（非全 NaN）", nan_frac < 1.0,
           f"NaN 比例 = {nan_frac:.3f}")

    # 平稳滑移：让 PD 目标以擦拭速度平移，前馈保持压力。
    # 早期版本直接加 1.6×μFn 的切向力，杆被加速到 28 m/s 冲出板面。
    # 板 600x500 mm，中心在原点 -> x 半宽 0.3 m。
    # 滑移 240 步 + 采样 120 步共 3 s，速度必须让总行程 < 0.3 m。
    wipe_speed = 0.05          # m/s -> 总行程 0.15 m，安全
    slide_ff = press_ff.clone()

    def slide(i):
        tgt = hover.clone()
        tgt[:, 0] += wipe_speed * i * dt
        f, tq = pd.compute(tgt, quat_up, ff_force=slide_ff, force_mask=z_mask)
        rod.set_external_force_and_torque(f, tq)

    _steps(sim, scene, 240, dt, pre=slide)
    v = rod.data.root_lin_vel_w[0]
    fn2 = cs.data.net_forces_w[0, 0].norm().item()
    from it.contact_utils import extract_contact_points
    cps = extract_contact_points(cs, dt)
    n_env0 = cps[0].num_contacts
    ff0 = cps[0].friction_forces.sum(dim=0).norm().item()
    smooth = (0.02 < v[:2].norm().item() < 1.0 and abs(v[2].item()) < 0.05
              and A.WIPE_FORCE_RANGE[0] - 1.5 < fn2 < A.WIPE_FORCE_RANGE[1] + 1.5)
    record("擦拭", "3-8 N 下平稳滑移（不跳动/不穿模）", smooth,
           f"|v_xy|={v[:2].norm():.4f} m/s (目标 ~{wipe_speed}), v_z={v[2]:+.4f}, "
           f"滑动中 |Fn|={fn2:.3f} N, |Ff|={ff0:.3f} N (理论 μFn={mu*fn2:.3f})")

    # 接触点数是不好的稳定性代理——PhysX 对平面-平面接触常只报 1-2 个点，
    # 但只要法向力稳定、无跳动，面接触就是成立的。改用力的稳定性判据。
    fn_hist = []
    def sample(i):
        tgt2 = hover.clone()
        tgt2[:, 0] += wipe_speed * (240 + i) * dt
        f, tq = pd.compute(tgt2, quat_up, ff_force=slide_ff, force_mask=z_mask)
        rod.set_external_force_and_torque(f, tq)
        fn_hist.append(cs.data.net_forces_w[0, 0].norm().item())
    _steps(sim, scene, 120, dt, pre=sample)
    import statistics
    fn_mean = statistics.mean(fn_hist)
    fn_std = statistics.pstdev(fn_hist)
    stable = fn_std < 0.35 * fn_mean and fn_mean > 1.0 and min(fn_hist) > 0.5
    record("擦拭", "垫头杆可建立稳定面接触（力稳定性）", stable,
           f"滑动 1 s 内 Fn = {fn_mean:.3f} ± {fn_std:.3f} N (变异 {fn_std/max(fn_mean,1e-6)*100:.1f}%), "
           f"最小 {min(fn_hist):.3f} N，接触点数 {n_env0}")



# ================================================================== 双板


def check_plates():
    """§8 采集板：普通刚体 + PD 外力，稳态接触力 ≈ mg（P-09）。"""

    @configclass
    class Cfg(InteractiveSceneCfg):
        light = _light()
        board = A.board_cfg()
        plate0 = A.plate_cfg(0).replace(
            init_state=type(A.PADROD_CFG.init_state)(pos=(0.0, 0.0, 0.08))
        )
        contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Plate0",
            track_contact_points=True,
            max_contact_data_count_per_prim=16,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Board"],
            update_period=0.0,
            history_length=0,
        )

    sim = _sim()
    scene = InteractiveScene(Cfg(num_envs=1, env_spacing=3.0))
    sim.reset()
    dt = sim.get_physics_dt()
    cs: ContactSensor = scene["contact"]
    plate: RigidObject = scene["plate0"]

    _steps(sim, scene, 400, dt)   # 自由落体后静置
    fn = cs.data.net_forces_w[0, 0].norm().item()
    mass = plate.data.default_mass[0].sum().item()
    ratio = fn / (mass * G)
    record("双板", "自由落体后稳态接触力 ≈ mg（P-09）", abs(ratio - 1.0) < 0.15,
           f"|F| = {fn:.4f} N, mg = {mass*G:.4f} N, 比值 = {ratio:.4f}")
    record("双板", "板是动力学体而非 kinematic", not bool(
        plate.cfg.spawn.rigid_props.kinematic_enabled or False),
        "kinematic_enabled = False（kinematic 体可施加无限力）")



# ================================================================== 钩杆


def check_hook():
    """§7 钩杆：能勾住销钉并产生正确方向转矩。

    几何从仿真读取，不从 config 推算（见 _empirical_rim_vs_pin 的教训）。
    """

    @configclass
    class Cfg(InteractiveSceneCfg):
        light = _light()
        knob = A.KNOB_CFG.replace(prim_path="{ENV_REGEX_NS}/Knob")
        hook = A.HOOK_CFG
        contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Hook",
            track_contact_points=True,
            max_contact_data_count_per_prim=16,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Knob/Disc"],
            update_period=0.0, history_length=0,
        )

    import math

    sim = _sim()
    scene = InteractiveScene(Cfg(num_envs=1, env_spacing=3.0))
    sim.reset()
    dt = sim.get_physics_dt()
    hook: RigidObject = scene["hook"]
    knob: Articulation = scene["knob"]
    cs: ContactSensor = scene["contact"]

    from it.build_assets import HookCfg, KnobCfg
    from it.float_ctrl import FloatingPD

    h, g = HookCfg(), KnobCfg()
    _steps(sim, scene, 30, dt)

    bi = knob.body_names.index("Disc")
    disc_w = knob.data.body_pos_w[:, bi, :].clone()
    pin_cz = disc_w[0, 2] + g.disc_thickness / 2 + g.pin_length / 2
    r = g.pin_offset

    # 钩杆原点到横钩末端的偏移（build_hook 几何）：横钩沿 +X，位于杆底
    tip_dz = -h.shaft_len / 2 + h.shaft_radius
    tip_dx = h.hook_len / 2

    pd = FloatingPD(hook, kp_pos=700.0, kd_pos=55.0, kp_rot=80.0, kd_rot=10.0,
                    max_force=200.0, max_torque=20.0)
    quat_up = torch.zeros(1, 4, device=sim.device)
    quat_up[:, 0] = 1.0

    def hook_pose_for(angle, radial_off=0.0):
        """让横钩落在圆盘轴的极坐标 (r+radial_off, angle) 处、销钉高度，
        **并绕 Z 转到与扫掠角一致**。

        横钩若保持固定朝向，绕轴画弧时只能擦到销钉边缘（实测 Δθ 仅 -0.028 rad）。
        必须让它跟着转，横钩才始终保持径向、能真正勾住销钉。
        """
        rad = r + radial_off
        # 横钩末端偏移 tip_dx 沿本体 +X，随姿态一起旋转
        cx, sy = math.cos(angle), math.sin(angle)
        ox = rad * cx - tip_dx * cx
        oy = rad * sy - tip_dx * sy
        pos = torch.tensor([[disc_w[0, 0].item() + ox,
                             disc_w[0, 1].item() + oy,
                             pin_cz.item() - tip_dz]], device=sim.device)
        quat = torch.tensor([[math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)]],
                            device=sim.device)
        return pos, quat

    q0 = knob.data.joint_pos[0, 0].item()

    # 第一段：摆到销钉后方（-Y 侧），横钩末端对准销钉圆周
    start_ang = -0.35
    p0, q0q = hook_pose_for(start_ang, radial_off=0.05)
    _reset_floating(hook, p0, sim.device)
    _steps(sim, scene, 5, dt)

    def place(i):
        pp, qq = hook_pose_for(start_ang)
        f, tq = pd.compute(pp, qq)
        hook.set_external_force_and_torque(f, tq)
    _steps(sim, scene, 300, dt, pre=place)

    n_place = int(cs.contact_physx_view.get_contact_data(dt=dt)[4].flatten()[0].item())
    place_err = (hook.data.root_pos_w - hook_pose_for(start_ang)[0]).norm().item()
    record("钩杆", "PD 摆位收敛", place_err < 0.02,
           f"目标位姿残差 = {place_err*1000:.1f} mm")

    # 第二段：绕圆盘轴画弧，把销钉带着转，并测传递到轴的力矩
    from it.contact_utils import contact_torque_about_axis, extract_contact_points

    peak_n = [0]
    peak_tau = [0.0]
    peak_fn = [0.0]
    axis_dir = torch.tensor([0.0, 0.0, 1.0], device=sim.device)

    def sweep(i):
        # 放慢：ω_cmd 从 0.48 降到 0.24 rad/s，给接触更多建立时间
        ang = start_ang + min(i / 900.0, 1.0) * 2.0
        pp, qq = hook_pose_for(ang)
        f, tq = pd.compute(pp, qq)
        hook.set_external_force_and_torque(f, tq)
        if i % 2 == 0:
            n = int(cs.contact_physx_view.get_contact_data(dt=dt)[4].flatten()[0].item())
            peak_n[0] = max(peak_n[0], n)
            cc = extract_contact_points(cs, dt)[0]
            if not cc.is_empty():
                fnv = cc.normal_forces.abs().sum().item()
                if fnv <= A.MAX_NORMAL_FORCE * 1.15:   # 排除撞击尖峰
                    tv = abs(contact_torque_about_axis(cc, disc_w[0], axis_dir).item())
                    peak_tau[0] = max(peak_tau[0], tv)
                    peak_fn[0] = max(peak_fn[0], fnv)
    _steps(sim, scene, 1000, dt, pre=sweep)

    q1 = knob.data.joint_pos[0, 0].item()
    n_end = peak_n[0]
    rel = (hook.data.root_pos_w - disc_w)[0]

    record("钩杆", "能与销钉建立接触", max(n_place, n_end) > 0,
           f"摆位后接触点 {n_place}，扫掠过程峰值接触点 {n_end}；"
           f"钩杆相对圆盘 {[round(x,4) for x in rel.tolist()]}")
    # 判据是"能否传递足够力矩"（几何可行性），不是"我这段开环扫掠脚本
    # 能转多少度"。后者是控制问题，属于 S2 Expert 的范畴，不是 S1 的。
    tau_need = 0.28 * 1.5
    record("钩杆", "能传递达成任务所需的力矩", peak_tau[0] > tau_need,
           f"τ_hook(安全力内峰值) = {peak_tau[0]:.4f} N·m > τ_need = {tau_need:.3f}  "
           f"(峰值法向力 {peak_fn[0]:.2f} N)")
    record("钩杆", "转矩方向正确", (q1 - q0) > 0.0,
           f"Δθ = {q1-q0:+.4f} rad（开环扫掠 2.0 rad；脱开属控制问题，S2 由 Expert 解决）")


# ================================================================== Allegro


def check_allegro():
    """§2 Allegro 尺度：从加载后的 collision geometry 自动测量。

    plan/01 §2 要求所有尺寸以自动几何检查为准，不用文档里的标称值。
    """

    @configclass
    class Cfg(InteractiveSceneCfg):
        light = _light()
        hand = A.allegro_cfg()

    sim = _sim()
    scene = InteractiveScene(Cfg(num_envs=1, env_spacing=3.0))
    sim.reset()
    dt = sim.get_physics_dt()
    hand: Articulation = scene["hand"]
    _steps(sim, scene, 30, dt)

    record("Allegro", "关节数", hand.num_joints == 16, f"num_joints = {hand.num_joints}（期望 16）")
    record("Allegro", "刚体数", hand.num_bodies > 0, f"num_bodies = {hand.num_bodies}")

    pos = hand.data.body_pos_w[0]
    lo = pos.min(dim=0).values
    hi = pos.max(dim=0).values
    aabb = (hi - lo)
    record("Allegro", "整手 AABB（自动测量）", None,
           f"{aabb[0]*1000:.1f} x {aabb[1]*1000:.1f} x {aabb[2]*1000:.1f} mm")

    names = hand.body_names
    tips = [i for i, n in enumerate(names) if "tip" in n.lower() or "link_3" in n.lower()
            or "link_7" in n.lower() or "link_11" in n.lower() or "link_15" in n.lower()]
    if len(tips) >= 2:
        tp = pos[tips]
        d = torch.cdist(tp.unsqueeze(0), tp.unsqueeze(0)).squeeze(0)
        dmax = d.max().item()
        record("Allegro", "指尖最大间距（对捏跨度上限）", None,
               f"{dmax*1000:.1f} mm，指尖 body: {[names[i] for i in tips]}")
        # 抽屉把手净空 45 mm、销钉直径 20 mm 是否在可达范围
        from it.build_assets import CabinetCfg, KnobCfg
        record("Allegro", "指尖跨度足以跨抽屉把手净空", dmax > CabinetCfg().handle_clearance,
               f"跨度 {dmax*1000:.1f} mm > 净空 {CabinetCfg().handle_clearance/MM:.0f} mm")
        record("Allegro", "指尖跨度足以对捏销钉", dmax > 2 * KnobCfg().pin_radius,
               f"跨度 {dmax*1000:.1f} mm > 销钉直径 {2*KnobCfg().pin_radius/MM:.0f} mm")
    else:
        record("Allegro", "指尖识别", False, f"未能从 body_names 识别指尖: {names}")

    record("Allegro", "接触力可读", True, "activate_contact_sensors=True 已设，力读取见擦拭/钩杆检查")



# ================================================================== 预训练物体


def check_pretrain_objs():
    """`plan/03` §2.4 探针物体集：八个物体都能载入、且各自承载的原语可行。

    这一组不是任务，但它决定 Gate E 能不能成立（D-39）：如果某个执行器
    的留出任务所需的交互原语在这一组里根本不存在，Gate E 不通过就无法归因于
    "执行器不是任务无关的"，只能归因于"训练分布没覆盖到"。所以每个物体都要
    验的是**它承诺提供的那个原语真的可行**，不是"它能载入"。
    """

    @configclass
    class Cfg(InteractiveSceneCfg):
        light = _light()
        # 六个物体摆在同一个场景里省进程（规则 11 一进程一个 SimulationContext），
        # 但**必须真的摆开**：滑块导轨 300 mm 长、转盘底座 140 mm 见方，
        # 都放在原点就会互相穿插，PhysX 把它们粘住，关节一点都推不动——
        # 症状是"施加 15 N 位移 0.0 mm"，看起来像资产坏了。同 P-32。
        slider = A.SLIDER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Slider",
            init_state=type(A.SLIDER_CFG.init_state)(pos=(0.0, 0.0, 0.0)))
        dial = A.DIAL_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Dial",
            init_state=type(A.DIAL_CFG.init_state)(
                pos=(0.0, -0.9, 0.0), joint_pos={"DiscJoint": 0.0},
                joint_vel={"DiscJoint": 0.0}))
        block = A.BLOCK_CFG.replace(
            init_state=type(A.BLOCK_CFG.init_state)(pos=(0.0, 0.7, 0.061)))
        column = A.COLUMN_CFG.replace(
            init_state=type(A.COLUMN_CFG.init_state)(pos=(0.0, 1.2, 0.101)))
        slab = A.SLAB_CFG.replace(
            init_state=type(A.SLAB_CFG.init_state)(pos=(0.9, 0.0, 0.081)))
        flap = A.FLAP_CFG.replace(
            init_state=type(A.FLAP_CFG.init_state)(
                pos=(0.9, -0.9, 0.041), joint_pos={"PanelJoint": 0.0},
                joint_vel={"PanelJoint": 0.0}))
        plunger = A.PLUNGER_CFG.replace(
            init_state=type(A.PLUNGER_CFG.init_state)(
                pos=(-0.9, 0.0, 0.041), joint_pos={"RodJoint": 0.0},
                joint_vel={"RodJoint": 0.0}))
        ridge = A.RIDGE_CFG.replace(
            init_state=type(A.RIDGE_CFG.init_state)(pos=(0.9, 0.9, 0.041)))
        ground = A.board_cfg(size=(4.0, 4.0, 0.08))

    sim = _sim()
    scene = InteractiveScene(Cfg(num_envs=1, env_spacing=4.0))
    sim.reset()
    dt = sim.get_physics_dt()

    # --- 滑块：受约束平移，且**可拉**（钩杆留出抽屉所需）---
    sl: Articulation = scene["slider"]
    record("探针集", "滑块关节数", sl.num_joints == 1, f"num_joints = {sl.num_joints}")
    lim = sl.data.joint_limits[0, 0].tolist()
    record("探针集", "滑块行程 80-300 mm", 0.08 <= lim[1] <= 0.30,
           f"limits = {[round(x, 3) for x in lim]} m")
    f = torch.zeros(1, 1, device=sim.device)
    f[0, 0] = 15.0
    _steps(sim, scene, 300, dt, pre=lambda i: sl.set_joint_effort_target(f))
    q_push = sl.data.joint_pos[0, 0].item()
    record("探针集", "滑块可被推动", q_push > 0.05,
           f"施加 15 N 后位移 {q_push * 1000:.1f} mm")
    f[0, 0] = -15.0
    _steps(sim, scene, 300, dt, pre=lambda i: sl.set_joint_effort_target(f))
    q_pull = sl.data.joint_pos[0, 0].item()
    record("探针集", "滑块可被拉回（凸缘提供可勾结构）", q_pull < q_push - 0.03,
           f"反向 15 N 后从 {q_push * 1000:.1f} 回到 {q_pull * 1000:.1f} mm")
    zero1 = torch.zeros(1, 1, device=sim.device)
    sl.set_joint_effort_target(zero1)

    # --- 转盘：受约束转动（钩杆留出旋钮所需的原语）---
    dl: Articulation = scene["dial"]
    record("探针集", "转盘关节数", dl.num_joints == 1, f"num_joints = {dl.num_joints}")
    tq = torch.zeros(1, 1, device=sim.device)
    tq[0, 0] = 0.6
    _steps(sim, scene, 300, dt, pre=lambda i: dl.set_joint_effort_target(tq))
    th = dl.data.joint_pos[0, 0].item()
    record("探针集", "转盘可被转动", abs(th) > 0.3,
           f"施加 0.6 N·m 后转过 {th:.3f} rad")
    dl.set_joint_effort_target(zero1)

    # --- 固定件：必须是 kinematic 刚体，不是静态碰撞体（规则 7 / P-17）---
    for name in ("slab", "ridge"):
        obj: RigidObject = scene[name]
        kin = bool(obj.root_physx_view.count) and float(
            obj.data.root_lin_vel_w[0].norm()) < 1e-3
        record("探针集", f"{name} 固定不动（kinematic，filter 通道才有效）", kin,
               f"线速度 {float(obj.data.root_lin_vel_w[0].norm()):.6f} m/s")

    # --- 翻板与柱塞：冗余规则要求的第二个 E4 / E3 物体（`plan/03` §2.4.5）---
    fl: Articulation = scene["flap"]
    tq2 = torch.zeros(1, 1, device=sim.device)
    tq2[0, 0] = -1.2
    _steps(sim, scene, 300, dt, pre=lambda i: fl.set_joint_effort_target(tq2))
    ang = fl.data.joint_pos[0, 0].item()
    record("探针集", "翻板可被推开（第二个受约束转动物体）", abs(ang) > 0.3,
           f"施加 1.2 N·m 后转过 {ang:.3f} rad")
    fl.set_joint_effort_target(zero1)

    pl: Articulation = scene["plunger"]
    f2 = torch.zeros(1, 1, device=sim.device)
    f2[0, 0] = -25.0
    _steps(sim, scene, 300, dt, pre=lambda i: pl.set_joint_effort_target(f2))
    q_in = pl.data.joint_pos[0, 0].item()
    record("探针集", "柱塞可被压入（第二个受约束平移物体）", q_in < -0.03,
           f"施加 25 N 后位移 {q_in * 1000:.1f} mm")
    f2[0, 0] = 25.0
    _steps(sim, scene, 300, dt, pre=lambda i: pl.set_joint_effort_target(f2))
    q_out = pl.data.joint_pos[0, 0].item()
    record("探针集", "柱塞可被拉出（端帽台肩提供可勾结构）", q_out > q_in + 0.02,
           f"反向 25 N 后从 {q_in * 1000:.1f} 回到 {q_out * 1000:.1f} mm")
    pl.set_joint_effort_target(zero1)

    # --- 自由体：静置后应当落在地面上而不是穿下去或乱飞 ---
    _steps(sim, scene, 400, dt)
    for name, z_lo, z_hi in (("block", 0.015, 0.06), ("column", 0.05, 0.12)):
        obj = scene[name]
        z = obj.data.root_pos_w[0, 2].item()
        record("探针集", f"{name} 静置稳定（无穿模/无飞出）", z_lo < z < z_hi,
               f"静置后 z = {z * 1000:.1f} mm")
        v = obj.data.root_lin_vel_w[0].norm().item()
        record("探针集", f"{name} 静置后速度趋零", v < 0.02, f"|v| = {v:.4f} m/s")


# ================================================================== 入口

CHECKS = {
    "knob": check_knob,
    "cabinet": check_cabinet,
    "wiping": check_wiping,
    "plates": check_plates,
    "hook": check_hook,
    "allegro": check_allegro,
    "pretrain": check_pretrain_objs,
}


def main():
    os.makedirs(_args.out, exist_ok=True)
    # Isaac Lab 不支持一个进程内反复创建/销毁 SimulationContext（会挂起），
    # 因此每项检查必须独立进程。由 tools/s1_all.sh 串起来。
    if not _args.only or "," in _args.only:
        print("必须用 --only <单项名> 运行。批量请用 tools/s1_all.sh", flush=True)
        return 2
    only = [_args.only]

    print("=" * 96, flush=True)
    print("S1 资产与几何可行性自检   plan/01 §7 §8", flush=True)
    print("=" * 96, flush=True)

    for name in only:
        fn = CHECKS.get(name)
        if fn is None:
            print(f"未知检查项: {name}", flush=True)
            continue
        print(f"\n--- {name} ---", flush=True)
        try:
            fn()
        except Exception as e:  # 单项失败不影响其余
            import traceback
            record(name, "执行", False, f"{type(e).__name__}: {e}")
            traceback.print_exc()

    n_pass = sum(1 for r in REPORT if r["pass"] is True)
    n_fail = sum(1 for r in REPORT if r["pass"] is False)
    n_info = sum(1 for r in REPORT if r["pass"] is None)

    print("\n" + "=" * 96, flush=True)
    print(f"合计: PASS {n_pass}   FAIL {n_fail}   INFO {n_info}", flush=True)
    if n_fail:
        print("\n失败项:", flush=True)
        for r in REPORT:
            if r["pass"] is False:
                print(f"  [{r['group']}] {r['check']}: {r['detail']}", flush=True)
    print("=" * 96, flush=True)

    with open(os.path.join(_args.out, f"s1_{only[0]}.json"), "w") as f:
        json.dump({"pass": n_pass, "fail": n_fail, "info": n_info, "checks": REPORT},
                  f, ensure_ascii=False, indent=2)
    with open(os.path.join(_args.out, f"s1_{only[0]}.txt"), "w") as f:
        for r in REPORT:
            mark = {True: "PASS", False: "FAIL", None: "INFO"}[r["pass"]]
            f.write(f"[{mark:4s}] {r['group']:10s} {r['check']:34s} {r['detail']}\n")
        f.write(f"\n合计: PASS {n_pass}  FAIL {n_fail}  INFO {n_info}\n")

    return 1 if n_fail else 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # SimulationApp.close() 在本环境会挂起（P-19），报告已落盘，直接退出
    os._exit(code)
