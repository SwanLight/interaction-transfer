"""S0：把 S1 的各个场景录成 mp4，用于人工核对。

`plan/06` §7 要求每次正式评估前人工看视频。S1 的经历证明了为什么——
第一版旋钮标定拿到"轮缘推不动"的 PASS，实际是推子被底座卡住，
**看一眼画面就能发现，靠数字发现不了**。

三条本环境特有的约束（见 log/pitfalls.md P-06/P-07/P-19/P-24）：

1. 必须用 ``AppLauncher(headless=True, enable_cameras=True)``，不能用裸
   ``SimulationApp``——它负责选 headless.rendering.kit 并写 carb 设置
   ``/isaaclab/cameras_enabled``。
2. ``sim.step(render=True)`` 仍会卡死。物理走 ``step(render=False)``，
   要出图时单独调 ``sim.render()`` + ``cam.update()``。
3. ``cam.set_world_poses_from_view()`` 会卡死。相机位姿必须在
   ``CameraCfg.OffsetCfg`` 里静态给定，用本模块的 ``look_at_quat()`` 算。

用法::

    PYTHONPATH=src /isaac-sim/python.sh tools/s0_record.py --scene knob --out /tmp/s0
"""

from __future__ import annotations

import argparse
import math
import os
import sys

_ap = argparse.ArgumentParser(description="S0 场景录像")
_ap.add_argument("--scene", required=True,
                 choices=["knob_rim", "knob_pin", "cabinet", "wiping", "hook", "slider"])
_ap.add_argument("--out", default="/tmp/s0")
_ap.add_argument("--width", type=int, default=960)
_ap.add_argument("--height", type=int, default=540)
_ap.add_argument("--fps", type=int, default=30)
_args, _ = _ap.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

_launcher = AppLauncher(headless=True, enable_cameras=True)
_app = _launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, AssetBaseCfg, RigidObject  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from it import assets as A  # noqa: E402
from it.build_assets import CabinetCfg, HookCfg, KnobCfg, PadRodCfg  # noqa: E402
from it.float_ctrl import FloatingPD  # noqa: E402

G = 9.81


def look_at_quat(eye, target, up=(0.0, 0.0, 1.0)):
    """USD/OpenGL 相机约定（-Z 朝前，+Y 朝上）的四元数 (w, x, y, z)。"""
    eye = np.asarray(eye, float); target = np.asarray(target, float); up = np.asarray(up, float)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    z = -fwd
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, qx, qy, qz = 0.25 * s, (R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        w, qx, qy, qz = (R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s
    elif R[1,1] > R[2,2]:
        s = math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        w, qx, qy, qz = (R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s
    else:
        s = math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        w, qx, qy, qz = (R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s
    return (float(w), float(qx), float(qy), float(qz))


def cam_cfg(eye, target, w, h):
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/Cam", update_period=0.0, height=h, width=w,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=20.0, clipping_range=(0.03, 30.0)),
        offset=CameraCfg.OffsetCfg(pos=tuple(eye), rot=look_at_quat(eye, target),
                                   convention="opengl"))


def _lights():
    return (AssetBaseCfg(prim_path="/World/dome",
                         spawn=sim_utils.DomeLightCfg(intensity=280.0, color=(0.85, 0.88, 1.0))),
            AssetBaseCfg(prim_path="/World/sun",
                         spawn=sim_utils.DistantLightCfg(intensity=900.0, angle=6.0)))


class Recorder:
    """物理按 dt 走，按 fps 抽帧。"""

    def __init__(self, sim, scene, cam, out_dir, tag, fps):
        self.sim, self.scene, self.cam = sim, scene, cam
        self.dt = sim.get_physics_dt()
        self.every = max(1, int(round(1.0 / (fps * self.dt))))
        self.frames = []
        self.out_dir, self.tag, self.fps = out_dir, tag, fps
        self.i = 0

    def steps(self, n, pre=None):
        for k in range(n):
            if pre is not None:
                pre(k)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)      # render=True 会卡死
            self.scene.update(self.dt)
            if self.i % self.every == 0:
                self.sim.render()            # 单独渲染
                self.cam.update(self.dt)
                rgb = self.cam.data.output["rgb"][0, ..., :3]
                self.frames.append(rgb.detach().cpu().numpy().astype(np.uint8))
            self.i += 1

    def save(self):
        import imageio.v2 as iio
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, f"{self.tag}.mp4")
        with iio.get_writer(path, fps=self.fps, codec="libx264",
                            quality=8, macro_block_size=1) as wr:
            for f in self.frames:
                wr.append_data(f)
        print(f"WROTE {path} ({len(self.frames)} frames, {os.path.getsize(path)} bytes)", flush=True)
        return path


def _reset_floating(asset, pos_w, device):
    n = pos_w.shape[0]
    st = asset.data.default_root_state.clone()
    st[:, :3] = pos_w
    st[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(n, 1)
    st[:, 7:] = 0.0
    asset.write_root_state_to_sim(st)


# ============================================================ 场景

def _rec_knob_case(out, W, H, fps, case):
    """D-14 标定的可视化。case="rim"（蓝色轮缘）或 "pin"（橙色销钉）。

    **两种情况的动作本来就不同**，这正是 D-14 依赖的物理区别：

    - 销钉：**切向直推**，法向力直接产生力矩 τ = F·r，与 μ 无关
    - 轮缘：**径向压 + 切向拖**，力矩只能靠摩擦传递 τ = μ·F·R

    两条弯路记在这里，别再走：

    1. 把两者统一成"径向压紧 + 沿圆周走"——销钉那条的力穿过转轴，
       力矩恒为零（实测 Δθ=0.0000），视频看起来像销钉也推不动。
    2. 让推子"追着销钉走"——推子的位置目标随圆盘一起动，两者互相
       拉扯，只转到 +0.13 rad。

    直线推法最简单也最有效（销钉 +1.30 rad）。圆盘转到目标后停止驱动，
    否则推子会被持续加速飞出画面。
    """
    g = KnobCfg()

    @configclass
    class Cfg(InteractiveSceneCfg):
        dome, sun = _lights()
        knob = A.KNOB_CFG.replace(prim_path="{ENV_REGEX_NS}/Knob")
        pusher = A.pusher_cfg()
        cam = cam_cfg((0.38, -0.36, 0.30), (0.0, 0.0, 0.09), W, H)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/120, device="cuda:0"))
    scene = InteractiveScene(Cfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    knob: Articulation = scene["knob"]
    pusher: RigidObject = scene["pusher"]
    rec = Recorder(sim, scene, scene["cam"], out, f"knob_{case}", fps)
    dt = rec.dt
    FN = A.MAX_NORMAL_FORCE
    half = 0.015

    zero = torch.zeros(1, 1, device=sim.device)
    knob.set_joint_effort_target(zero)
    knob.write_joint_state_to_sim(zero, zero)
    rec.steps(15)

    disc = knob.data.body_pos_w[:, knob.body_names.index("Disc"), :].clone()
    pin_cz = (disc[0, 2] + g.disc_thickness / 2 + g.pin_length / 2).item()

    tgt = torch.zeros(1, 3, device=sim.device)
    ffv = torch.zeros(1, 3, device=sim.device)
    mask = torch.zeros(1, 3, dtype=torch.bool, device=sim.device)
    RIM_R = g.disc_radius + half - 0.001
    if case == "rim":
        tgt[0] = torch.tensor([RIM_R, 0.0, 0.0], device=sim.device) + disc[0]
        mask[0, 0] = mask[0, 1] = True   # 径向压紧，两个水平轴都参与
        start = tgt.clone(); start[0, 0] += 0.03
        pre = tgt.clone();   pre[0, 0] += 0.004
    else:
        tgt[0, 0] = disc[0, 0] + g.pin_offset
        tgt[0, 1] = disc[0, 1] - (g.pin_radius + half - 0.001)
        tgt[0, 2] = pin_cz
        ffv[0, 1] = FN           # 切向直推（+Y），法向力本身产生力矩
        mask[0, 1] = True
        start = tgt.clone(); start[0, 1] -= 0.03
        pre = tgt.clone();   pre[0, 1] -= 0.004

    pd = FloatingPD(pusher, kp_pos=800.0, kd_pos=60.0, kp_rot=60.0, kd_rot=8.0,
                    max_force=300.0, max_torque=30.0, kd_force=60.0)
    _reset_floating(pusher, start, sim.device)
    rec.steps(5)
    rec.steps(220, lambda i: pusher.set_external_force_and_torque(*pd.compute(pre)))

    STOP_AT = 1.8      # rad，转到这里就撤力，免得推子被持续加速飞出画面
    def drive(i):
        knob.set_joint_effort_target(zero)
        done = knob.data.joint_pos[0, 0].item() >= STOP_AT
        ramp = 0.0 if done else min(i / 120.0, 1.0)
        if case == "rim":
            # 切向拖必须**沿圆周**。早期用直线拖，推子几步就偏离圆周、
            # 脱离接触，然后径向力把它推飞出画面——证据还在（圆盘没转），
            # 但画面看起来像"推子飞了所以没转"，说服力全无。
            ang = min(i / 600.0, 1.0) * 2.0
            t = disc.clone()
            t[0, 0] += RIM_R * math.cos(ang)
            t[0, 1] += RIM_R * math.sin(ang)
            ff = torch.zeros(1, 3, device=sim.device)
            ff[0, 0] = -FN * math.cos(ang) * ramp
            ff[0, 1] = -FN * math.sin(ang) * ramp
        else:
            t = tgt.clone()
            ff = ffv * ramp
        f, tq = pd.compute(t, ff_force=ff, force_mask=mask if not done else None)
        pusher.set_external_force_and_torque(f, tq)
    rec.steps(700, drive)

    dq = knob.data.joint_pos[0, 0].item()
    verdict = "转进目标区间 1.0–2.2" if dq >= 1.0 else "推不动（低于目标下限 1.0）"
    print(f"RESULT {case}: Δθ={dq:+.4f} rad -> {verdict}", flush=True)
    return rec.save()


def rec_knob_rim(out, W, H, fps):
    return _rec_knob_case(out, W, H, fps, "rim")


def rec_knob_pin(out, W, H, fps):
    return _rec_knob_case(out, W, H, fps, "pin")


def rec_cabinet(out, W, H, fps):
    @configclass
    class Cfg(InteractiveSceneCfg):
        dome, sun = _lights()
        cab = A.CABINET_CFG.replace(prim_path="{ENV_REGEX_NS}/Cabinet")
        cam = cam_cfg((0.62, -0.52, 0.36), (-0.05, 0.0, 0.09), W, H)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/120, device="cuda:0"))
    scene = InteractiveScene(Cfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    cab: Articulation = scene["cab"]
    rec = Recorder(sim, scene, scene["cam"], out, "cabinet_travel", fps)
    f = torch.zeros(1, 1, device=sim.device)
    rec.steps(30)
    f[0, 0] = 30.0
    rec.steps(400, lambda i: cab.set_joint_effort_target(f))
    q_open = cab.data.joint_pos[0, 0].item()
    f[0, 0] = -30.0
    rec.steps(400, lambda i: cab.set_joint_effort_target(f))
    print(f"RESULT 拉开 {q_open*1000:.1f} mm，推回 {cab.data.joint_pos[0,0].item()*1000:.1f} mm", flush=True)
    return rec.save()


def rec_wiping(out, W, H, fps):
    p = PadRodCfg()

    @configclass
    class Cfg(InteractiveSceneCfg):
        dome, sun = _lights()
        board = A.board_cfg()
        padrod = A.PADROD_CFG.replace(
            init_state=type(A.PADROD_CFG.init_state)(pos=(-0.10, 0.0, 0.14)))
        cam = cam_cfg((0.26, -0.34, 0.22), (0.06, 0.0, 0.04), W, H)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/120, device="cuda:0"))
    scene = InteractiveScene(Cfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    rod: RigidObject = scene["padrod"]
    rec = Recorder(sim, scene, scene["cam"], out, "wiping_padrod", fps)
    dt = rec.dt

    target_fn = sum(A.WIPE_FORCE_RANGE) / 2
    pd = FloatingPD(rod, kp_pos=500.0, kd_pos=45.0, kp_rot=60.0, kd_rot=8.0,
                    max_force=120.0, max_torque=12.0)
    quat_up = torch.zeros(1, 4, device=sim.device); quat_up[:, 0] = 1.0
    hover = rod.data.default_root_state[:, :3].clone()
    hover[:, 2] = 0.113
    hover = hover + scene.env_origins
    ff = torch.zeros(1, 3, device=sim.device); ff[:, 2] = -target_fn
    zmask = torch.zeros(1, 3, dtype=torch.bool, device=sim.device); zmask[:, 2] = True

    pre_h = hover.clone(); pre_h[:, 2] += 0.006
    rec.steps(240, lambda i: rod.set_external_force_and_torque(*pd.compute(pre_h, quat_up)))
    rec.steps(180, lambda i: rod.set_external_force_and_torque(
        *pd.compute(hover, quat_up, ff_force=ff, force_mask=zmask)))

    def slide(i):
        t = hover.clone(); t[:, 0] += 0.05 * i * dt
        rod.set_external_force_and_torque(*pd.compute(t, quat_up, ff_force=ff, force_mask=zmask))
    rec.steps(500, slide)
    v = rod.data.root_lin_vel_w[0]
    print(f"RESULT 擦拭速度 |v_xy|={v[:2].norm():.4f} m/s, v_z={v[2]:+.4f}", flush=True)
    return rec.save()


def rec_hook(out, W, H, fps):
    h, g = HookCfg(), KnobCfg()

    @configclass
    class Cfg(InteractiveSceneCfg):
        dome, sun = _lights()
        knob = A.KNOB_CFG.replace(prim_path="{ENV_REGEX_NS}/Knob")
        hook = A.HOOK_CFG
        cam = cam_cfg((0.34, -0.32, 0.32), (0.0, 0.0, 0.11), W, H)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/120, device="cuda:0"))
    scene = InteractiveScene(Cfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    hook: RigidObject = scene["hook"]
    knob: Articulation = scene["knob"]
    rec = Recorder(sim, scene, scene["cam"], out, "hook_sweep", fps)
    rec.steps(20)

    disc = knob.data.body_pos_w[:, knob.body_names.index("Disc"), :].clone()
    pin_cz = disc[0, 2] + g.disc_thickness / 2 + g.pin_length / 2
    tip_dz = -h.shaft_len / 2 + h.shaft_radius
    tip_dx = h.hook_len / 2
    r = g.pin_offset
    pd = FloatingPD(hook, kp_pos=700.0, kd_pos=55.0, kp_rot=80.0, kd_rot=10.0,
                    max_force=200.0, max_torque=20.0)

    def pose(angle, off=0.0):
        rad = r + off
        c, s2 = math.cos(angle), math.sin(angle)
        pos = torch.tensor([[disc[0,0].item() + rad*c - tip_dx*c,
                             disc[0,1].item() + rad*s2 - tip_dx*s2,
                             pin_cz.item() - tip_dz]], device=sim.device)
        q = torch.tensor([[math.cos(angle/2), 0.0, 0.0, math.sin(angle/2)]], device=sim.device)
        return pos, q

    q0 = knob.data.joint_pos[0, 0].item()
    a0 = -0.35
    _reset_floating(hook, pose(a0, 0.05)[0], sim.device)
    rec.steps(5)
    rec.steps(280, lambda i: hook.set_external_force_and_torque(*pd.compute(*pose(a0))))
    def sweep(i):
        hook.set_external_force_and_torque(*pd.compute(*pose(a0 + min(i/900.0, 1.0)*2.0)))
    rec.steps(1000, sweep)
    print(f"RESULT 钩杆 Δθ={knob.data.joint_pos[0,0].item()-q0:+.4f} rad", flush=True)
    return rec.save()


def rec_slider(out, W, H, fps):
    @configclass
    class Cfg(InteractiveSceneCfg):
        dome, sun = _lights()
        slider = A.SLIDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Slider")
        cam = cam_cfg((0.26, -0.32, 0.22), (0.0, 0.0, 0.04), W, H)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/120, device="cuda:0"))
    scene = InteractiveScene(Cfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    sl: Articulation = scene["slider"]
    rec = Recorder(sim, scene, scene["cam"], out, "slider_pretrain", fps)
    f = torch.zeros(1, 1, device=sim.device)
    rec.steps(30)
    f[0, 0] = 15.0
    rec.steps(400, lambda i: sl.set_joint_effort_target(f))
    print(f"RESULT 滑块位移 {sl.data.joint_pos[0,0].item()*1000:.1f} mm", flush=True)
    return rec.save()


SCENES = {"knob_rim": rec_knob_rim, "knob_pin": rec_knob_pin, "cabinet": rec_cabinet,
          "wiping": rec_wiping, "hook": rec_hook, "slider": rec_slider}

if __name__ == "__main__":
    SCENES[_args.scene](_args.out, _args.width, _args.height, _args.fps)
    sys.stdout.flush()
    os._exit(0)     # app.close() 会挂起（P-19）
