"""S3 · 旋钮任务的双板 source 采集。

`plan/01` §3：圆盘绕 Z 转，偏心销钉伸出盘面，**轮缘 μ=0.10 / 销钉 μ=0.80**（D-14）。
那个摩擦差是旋钮任务存在的理由——它让 region 在信息论上**不可**从 effect 推出：
销钉靠法向力直接产生力矩（与 μ 无关），蹭轮缘只能靠摩擦（正比于 μ）。

## `rim_only` 这一档是**故意会失败**的

`plan/03` §2.1 原本把"双板接触轮缘、不使用销钉"列为一个正常策略家族。
那是 D-14（轮缘低摩擦）之前写的。加上低摩擦之后，S1 实测：

    τ_rim = 0.170 N·m  <  τ_need = 0.420 N·m  <  τ_pin = 23.5 N·m

也就是说轮缘家族在安全力上限内**转不到目标角**。所以本脚本仍然采它，
但它的 episode 会被判为失败、进 ``failed`` 桶（`plan/03` §6 要求失败样本
单独保存）。它的价值不在于当示教用，而在于：

* 它是 D-14 成立的**在真实操作轨迹上**的证据，不只是 S1 那种静态标定；
* `plan/05` 实验五要的反事实镜头「把 region 从销钉换到轮缘 -> 打滑失败」
  需要真实的轮缘接触数据才能做。

⚠️ **转速必须落在设计点附近。** τ_need = 阻尼 × ω，D-14 的判据是按
ω = 1.5 rad/s 算的；转得很慢时 τ_need 会掉到 0.14 N·m，轮缘反而推得动。
本脚本把指令角速度控制在 1.2~1.8 rad/s，与 `plan/01` §3.1 的设计点一致。

## 用法

    ./tools/run_remote.sh "PYTHONPATH=src /isaac-sim/python.sh \\
        tools/s3_source_knob.py --envs 60 --batches 25 --out /tmp/s3_knob" knob
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

_ap = argparse.ArgumentParser(description="S3 旋钮双板 source 采集")
_ap.add_argument("--envs", type=int, default=60)
_ap.add_argument("--batches", type=int, default=25)
_ap.add_argument("--out", default="/tmp/s3_knob")
_ap.add_argument("--seed", type=int, default=20260829)
_ap.add_argument("--family", default="all")
_ap.add_argument("--holdout-family", default="pin_regrasp")
_ap.add_argument("--physics-variant-frac", type=float, default=0.15)
_ap.add_argument("--video", action="store_true")
_ap.add_argument("--width", type=int, default=1280)
_ap.add_argument("--height", type=int, default=720)
_ap.add_argument("--fps", type=int, default=25)
_a, _ = _ap.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

_app = AppLauncher(headless=True, enable_cameras=_a.video).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg, ContactSensorCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from it import assets as A  # noqa: E402
from it import build_assets as B  # noqa: E402
from it.contact_attrib import (  # noqa: E402
    quat_from_frame,
    rotate_inverse,
    to_local,
)
from it.contact_utils import (  # noqa: E402
    classify_contact_mode_padded,
    extract_contact_points_padded,
)
from it.float_ctrl import FloatingPD  # noqa: E402
from it.records import (  # noqa: E402
    SCHEMA_VERSION,
    EpisodeRecord,
    save_episode,
    split_episode_entries,
    write_manifest,
)
from it.viz import look_at_quat  # noqa: E402

MM = 0.001
DT = 1.0 / 300.0
DECIMATION = 6
CONTROL_DT = DT * DECIMATION
MAX_CONTACTS = 24
PHASE_NAMES = ("approach", "establish", "manipulate", "release")
PHASE_STEPS = (50, 55, 160, 45)
N_FRAMES = sum(PHASE_STEPS)

_K = B.KnobCfg()
_P = B.PlateCfg()
#: 圆盘上表面高度、销钉中段高度（旋钮局部系，销钉是圆盘的子几何）
DISC_TOP = _K.disc_thickness / 2
PIN_Z = DISC_TOP + _K.pin_length * 0.55
GOAL_RANGE = (1.0, 2.2)
#: 指令角速度。**必须落在设计点附近**（`plan/01` §3.1 按 1.5 rad/s 标定），
#: 转得慢 τ_need 会掉下来，轮缘反而推得动，rim_only 就失去了对照意义。
OMEGA_RANGE = (1.2, 1.8)
#: 1.2 mm。3 mm 时接触力被顶到 25~96 N，远超 25 N 安全上限——
#: 板面 35×25 压进刚体的深度必须小，力才留得住。
GRIP_INTERF = 1.2 * MM
PRE_GAP = 1.5 * MM
STANDOFF = 0.12
SAFE_R = 0.055
MAX_NORMAL = A.MAX_NORMAL_FORCE      # 25 N 安全上限，对所有条件相同
MAX_VALID_FORCE = 200.0
MAX_PLATE_DIST = 1.4

FAMILIES = ("pin_pinch", "pin_push_single", "pin_push_dual", "pin_regrasp", "rim_only")


def _camera_cfg():
    eye = (0.30, -0.30, 0.26)
    at = (0.0, 0.0, 0.10)
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/Cam", update_period=0.0,
        height=_a.height, width=_a.width, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.02, 30.0)),
        offset=CameraCfg.OffsetCfg(pos=eye, rot=look_at_quat(eye, at),
                                   convention="opengl"))


def _contact_cfg(idx: int):
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Plate{idx}", track_pose=True,
        track_contact_points=True, max_contact_data_count_per_prim=MAX_CONTACTS,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Knob/Disc"],
        update_period=0.0, history_length=0)


@configclass
class SceneCfg(InteractiveSceneCfg):
    dome = AssetBaseCfg(prim_path="/World/dome",
                        spawn=sim_utils.DomeLightCfg(intensity=300.0,
                                                     color=(0.86, 0.89, 1.0)))
    sun = AssetBaseCfg(prim_path="/World/sun",
                       spawn=sim_utils.DistantLightCfg(intensity=1100.0, angle=5.0))
    knob = A.KNOB_CFG.replace(prim_path="{ENV_REGEX_NS}/Knob")
    plate0 = A.plate_cfg(0)
    plate1 = A.plate_cfg(1)
    contact0 = _contact_cfg(0)
    contact1 = _contact_cfg(1)
    if _a.video:
        cam = _camera_cfg()


def _u(rng, lo, hi, n):
    return torch.from_numpy(rng.uniform(lo, hi, size=n).astype(np.float32))


def sample_family(fam: str, rng, n: int, device) -> dict:
    """逐 env 参数。

    Returns:
        ``on`` (n,2) 哪块板参与；``side`` (n,2) 板在销钉切向的哪一侧（±1）；
        ``radius`` (n,) 接触半径（销钉 or 轮缘）；``on_pin`` (n,) 是否接销钉；
        ``regrasp`` (n,) 是否中途松开重抓。
    """
    on = torch.ones(n, 2)
    side = torch.zeros(n, 2)
    on_pin = torch.ones(n)
    regrasp = torch.zeros(n)
    if fam == "pin_pinch":
        side[:, 0], side[:, 1] = -1.0, +1.0        # 切向两侧对捏销钉
    elif fam == "pin_push_single":
        side[:, 0] = -1.0                          # 只用一块板推
        on[:, 1] = 0.0
    elif fam == "pin_push_dual":
        side[:, :] = -1.0                          # 两块板同侧推，分担载荷
    elif fam == "pin_regrasp":
        side[:, 0], side[:, 1] = -1.0, +1.0
        regrasp = torch.ones(n)
    elif fam == "rim_only":
        side[:, 0], side[:, 1] = -1.0, +1.0
        on_pin = torch.zeros(n)
    else:
        raise ValueError(f"未知策略家族：{fam}")
    return {k: v.to(device) for k, v in
            dict(on=on, side=side, on_pin=on_pin, regrasp=regrasp).items()}


def _git_sha() -> str:
    here = Path(__file__).resolve().parent.parent / ".git_sha"
    if here.exists() and here.read_text(encoding="utf-8").strip():
        return here.read_text(encoding="utf-8").strip()
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


class Buffers:
    def __init__(self, n, device):
        z = lambda *s: torch.zeros(N_FRAMES, *s, device=device)  # noqa: E731
        self.phase = torch.zeros(N_FRAMES, n, dtype=torch.int8, device=device)
        self.progress = z(n)
        self.valid = torch.ones(N_FRAMES, n, dtype=torch.bool, device=device)
        self.angle = z(n)
        self.omega = z(n)
        self.obj_pos = z(n, 3)
        self.obj_quat = z(n, 4)
        self.pos_obj = z(2, n, MAX_CONTACTS, 3)
        self.nrm_obj = z(2, n, MAX_CONTACTS, 3)
        self.fri_obj = z(2, n, MAX_CONTACTS, 3)
        self.fn = z(2, n, MAX_CONTACTS)
        self.sep = z(2, n, MAX_CONTACTS)
        self.cvalid = torch.zeros(N_FRAMES, 2, n, MAX_CONTACTS, dtype=torch.bool,
                                  device=device)
        self.mode = torch.zeros(N_FRAMES, 2, n, MAX_CONTACTS, dtype=torch.int8,
                                device=device)
        self.on_pin_c = torch.zeros(N_FRAMES, 2, n, MAX_CONTACTS, dtype=torch.bool,
                                    device=device)
        self.tau = z(n)
        self.src_pose = z(2, n, 7)
        self.src_vel = z(2, n, 6)
        self.src_tgt = z(2, n, 7)
        self.src_cmd = z(2, n, 6)
        self.foreign = z(2)
        self.dropped = z(2)
        self.rot_err = z(2, n)


def run_batch(scene, sim, camera, fam_of, rng, device, batch):
    n = scene.cfg.num_envs
    knob: Articulation = scene["knob"]
    plates = [scene["plate0"], scene["plate1"]]
    sensors = [scene["contact0"], scene["contact1"]]
    jid = knob.find_joints("DiscJoint")[0][0]
    bid = knob.body_names.index("Disc")

    par = {k: torch.zeros(n, 2, device=device) for k in ("on", "side")}
    par.update({k: torch.zeros(n, device=device) for k in ("on_pin", "regrasp")})
    for fam in sorted(set(fam_of)):
        idx = [i for i, f in enumerate(fam_of) if f == fam]
        sub = sample_family(fam, rng, len(idx), device)
        sel = torch.tensor(idx, device=device, dtype=torch.long)
        for k in par:
            par[k][sel] = sub[k]
    on, side, on_pin, regrasp = par["on"], par["side"], par["on_pin"], par["regrasp"]

    goal = _u(rng, *GOAL_RANGE, n).to(device)
    omega = _u(rng, *OMEGA_RANGE, n).to(device)
    turn_dir = torch.from_numpy(
        rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=n)).to(device)
    # 接触半径：销钉家族接销钉侧面，轮缘家族接圆盘外缘
    radius = torch.where(on_pin > 0,
                         torch.full((n,), _K.pin_offset, device=device),
                         torch.full((n,), _K.disc_radius, device=device))
    # 接触高度：销钉在盘面之上，轮缘在盘厚中段
    height = torch.where(on_pin > 0,
                         torch.full((n,), PIN_Z, device=device),
                         torch.zeros(n, device=device))
    # 贴合面到接触中心的距离
    surf = torch.where(on_pin > 0,
                       torch.full((n,), _K.pin_radius, device=device),
                       torch.zeros(n, device=device))

    # 物理变体（`plan/03` §7）：留出关节阻尼区间
    is_var = torch.from_numpy((rng.random(n) < _a.physics_variant_frac)
                              .astype(np.bool_)).to(device)
    damp = _u(rng, 0.22, 0.34, n).to(device)
    lo, hi = ((0.10, 0.16) if batch % 2 == 0 else (0.42, 0.56))
    damp = torch.where(is_var, _u(rng, lo, hi, n).to(device), damp)
    knob.write_joint_damping_to_sim(damp.unsqueeze(-1), joint_ids=[jid])
    if (knob.data.joint_damping[:, jid] - damp).abs().max() > 1e-4:
        raise RuntimeError("关节阻尼没写进去，物理变体划分会变成假的")

    z1 = torch.zeros(n, 1, device=device)
    th0 = _u(rng, -0.10, 0.10, n).to(device)
    knob.write_joint_state_to_sim(th0.unsqueeze(-1), z1, joint_ids=[jid])
    knob.set_joint_effort_target(z1, joint_ids=[jid])
    scene.update(DT)
    axis = knob.data.body_pos_w[:, bid, :]

    # 推的那块板压进去，跟的那块板留缝。
    #
    # 两块板都用位置控制压住销钉时，**跟随的那块变成刹车**：它被销钉推着走，
    # 而位置指令不让它走，于是销钉被夹在"推"和"刹"之间——实测内力 40 N、
    # 净力矩只有 0.06 N·m。留缝之后跟随板只在需要时挡一下，不吃力矩。
    push_side = -turn_dir
    interf = torch.where(side * push_side.unsqueeze(-1) > 0,
                         torch.full_like(side, GRIP_INTERF),
                         torch.full_like(side, -6.0 * MM))
    # 跟随板留 6 mm 缝：3 mm 时它仍然被销钉顶上，pin_pinch 的内力
    # 一直卡在 27 N（净力矩只有 0.10 N·m）。
    #: **同侧两块板必须沿销钉轴向错开**，否则被指到同一个点上互相穿插——
    #: pin_push_dual 实测 66% 的操作步脱手、力矩上不去，与 P-32 同类。
    same_side = (side[:, 0] * side[:, 1] > 0) & (on[:, 0] > 0) & (on[:, 1] > 0)
    z_stag = torch.stack([-0.012 * same_side.float(), 0.012 * same_side.float()], dim=-1)

    def arc(theta, k):
        """给定指令角，返回第 k 块板的世界目标位置与姿态。

        板贴在接触体的**切向**一侧：销钉靠法向力直接给力矩（与 μ 无关），
        轮缘只能靠摩擦——这正是 D-14 要检验的差别。

        ⚠️ 轮缘家族两块板必须**径向对置**（θ 与 θ+π）。第一版把 side 乘了 0，
        两块板被指到同一个点上，直接互相穿插，接触力 34 N 而力矩为零——
        与 P-32 是同一类错误。
        """
        th_k = theta + (0.0 if k == 0 else math.pi)
        th_use = torch.where(on_pin > 0, theta, th_k)
        c, s_ = torch.cos(th_use), torch.sin(th_use)
        rad = torch.stack([c, s_, torch.zeros_like(c)], dim=-1)          # 径向
        tan = torch.stack([-s_, c, torch.zeros_like(c)], dim=-1)         # 切向
        h_k = height + z_stag[:, k] * (on_pin > 0).float()
        center = axis + rad * radius.unsqueeze(-1) \
            + torch.stack([torch.zeros_like(c), torch.zeros_like(c), h_k], dim=-1)
        off_pin = (surf + _P.size[2] / 2 - interf[:, k]).unsqueeze(-1)
        pin_pos = center + tan * off_pin * side[:, k].unsqueeze(-1)
        rim_pos = center + rad * (_P.size[2] / 2 - GRIP_INTERF)
        pos = torch.where(on_pin.unsqueeze(-1) > 0, pin_pos, rim_pos)
        z_ax = torch.where(on_pin.unsqueeze(-1) > 0,
                           -tan * side[:, k].unsqueeze(-1), -rad)
        q = quat_from_frame(z_ax, torch.tensor([[0.0, 0.0, 1.0]], device=device)
                            .expand(n, 3))
        return pos, q

    pds = [FloatingPD(pl, kp_pos=3000.0, kd_pos=110.0, kp_rot=6.0, kd_rot=0.025,
                      max_force=45.0, max_torque=4.0, kd_force=15.0) for pl in plates]

    targets, quats = [], []
    for k, pl in enumerate(plates):
        p0, q0 = arc(th0, k)
        st = pl.data.default_root_state.clone()
        st[:, :3] = p0 + torch.tensor([0.0, 0.0, STANDOFF], device=device)
        st[:, 3:7] = q0
        st[:, 7:] = 0.0
        pl.write_root_state_to_sim(st)
        targets.append(st[:, :3].clone())
        quats.append(q0)
    for _ in range(6):
        knob.set_joint_effort_target(z1, joint_ids=[jid])
        for pl in plates:
            zw = torch.zeros(n, 1, 3, device=device)
            pl.set_external_force_and_torque(zw, zw, is_global=True)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(DT)

    buf = Buffers(n, device)
    prev_tgt = [t.clone() for t in targets]
    preview: list[np.ndarray] = []
    every = max(1, round(1.0 / (CONTROL_DT * _a.fps)))
    mu_pin = min(_P.friction, _K.pin_friction)
    mu_rim = min(_P.friction, _K.rim_friction)
    # 领先角由目标角速度反推：ω = τ/阻尼 = (k·lead·r)·r/阻尼
    #   -> lead = ω · 阻尼 / (k · r²)，k = kp_pos·m = 1500 N/m，r = 销钉偏心距
    r_eff = torch.where(on_pin > 0, torch.full((n,), _K.pin_offset, device=device),
                        torch.full((n,), _K.disc_radius, device=device))
    # ×2.2 是补板自己的跟踪滞后。板要以 ω·r ≈ 0.078 m/s 追一个动目标，
    # 位置 PD 的稳态滞后 kd·v/kp ≈ 2.9 mm，而领先角本身只折合 5.4 mm——
    # 一多半被吃掉了，实测推力只有 2.3 N 而不是模型说的 8 N，圆盘转不到目标角。
    lead_ang = (2.2 * omega * damp / (1500.0 * r_eff * r_eff)).clamp(0.02, 0.30)

    frame = 0
    for phase_id, plen in enumerate(PHASE_STEPS):
        for kk in range(plen):
            if phase_id == 2:
                # **跟着圆盘走、领先一个固定角度**，而不是开环走一条预定的弧。
                #
                # 开环推进时圆盘一旦跟不上，板与销钉的位置差就一直累积，
                # PD 力按 1500 N/m × 滞后 涨上去——实测滞后到 36 mm、接触力
                # 54 N，远超 25 N 安全上限，而其中大部分是两块板隔着销钉
                # 互相顶的**内力**，对绕轴力矩毫无贡献。
                #
                # 领先角直接决定推力：F = k·lead·r，τ = F·r，ω = τ/阻尼。
                # lead 由目标 ω 反推（见上面 lead_ang），力自然落在安全区内。
                th_now = knob.data.joint_pos[:, jid]
                th_end = th0 + turn_dir * goal
                th_cmd = th_now + turn_dir * lead_ang
                th_cmd = torch.where(turn_dir > 0,
                                     torch.minimum(th_cmd, th_end),
                                     torch.maximum(th_cmd, th_end))
            else:
                th_cmd = th0.clone()
            back = torch.zeros(n, device=device)
            if phase_id == 2 and bool((regrasp > 0).any()):
                r_ph = (kk / plen)
                if 0.45 < r_ph < 0.60:
                    back = regrasp * SAFE_R * math.sin(
                        math.pi * (r_ph - 0.45) / 0.15)
                    # 松开期间不再驱动，让圆盘自己停下
                    th_cmd = torch.where(regrasp > 0, knob.data.joint_pos[:, jid], th_cmd)

            for k in range(2):
                p, q = arc(th_cmd, k)
                quats[k] = q
                tgt = targets[k]
                lift = torch.zeros(n, 3, device=device)
                if phase_id == 0:
                    a = min((kk + 1) / max(plen - 8, 1), 1.0)
                    lift[:, 2] = STANDOFF * (1.0 - a) + PRE_GAP * 4 * a
                elif phase_id == 1:
                    a = min((kk + 1) / max(plen - 8, 1), 1.0)
                    lift[:, 2] = PRE_GAP * 4 * (1.0 - a)
                elif phase_id == 3:
                    lift[:, 2] = STANDOFF * min((kk + 1) / max(plen, 1), 1.0)
                # regrasp 的松开：沿径向退出去再贴回来
                c, s = torch.cos(th_cmd), torch.sin(th_cmd)
                rad = torch.stack([c, s, torch.zeros_like(c)], dim=-1)
                tgt[:] = p + lift + rad * back.unsqueeze(-1)
                # 不参与的板停在远处
                tgt[:] = torch.where(on[:, k].unsqueeze(-1) > 0, tgt,
                                     p + torch.tensor([0.0, 0.0, STANDOFF],
                                                      device=device))
                buf.src_cmd[frame, k, :, :3] = tgt - prev_tgt[k]
                prev_tgt[k] = tgt.clone()

            best = [None, None]
            best_fn = torch.zeros(2, n, device=device)
            for _sub in range(DECIMATION):
                # P-21：力矩指令会一直保持，每步都要显式清零
                knob.set_joint_effort_target(z1, joint_ids=[jid])
                for k, pl in enumerate(plates):
                    f, tq = pds[k].compute(targets[k], quats[k])
                    pl.set_external_force_and_torque(f, tq, is_global=True)
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(DT)
                for k, pl in enumerate(plates):
                    cp = extract_contact_points_padded(
                        sensors[k], DT, body_pos_w=pl.data.root_pos_w,
                        max_points=MAX_CONTACTS, own_radius=0.07)
                    tot = cp["normal_forces"].abs().sum(dim=1)
                    buf.foreign[frame, k] += cp["foreign"]
                    buf.dropped[frame, k] += cp["dropped"]
                    if best[k] is None:
                        best[k], best_fn[k] = cp, tot
                    else:
                        take = (tot > best_fn[k]).unsqueeze(-1)
                        for key in ("normal_forces", "separations"):
                            best[k][key] = torch.where(take, cp[key], best[k][key])
                        for key in ("positions", "normals", "friction_forces"):
                            best[k][key] = torch.where(take.unsqueeze(-1), cp[key],
                                                       best[k][key])
                        best[k]["valid"] = torch.where(take, cp["valid"], best[k]["valid"])
                        best_fn[k] = torch.maximum(best_fn[k], tot)

            th = knob.data.joint_pos[:, jid]
            o_pos = knob.data.body_pos_w[:, bid, :]
            o_quat = knob.data.body_quat_w[:, bid, :]
            buf.phase[frame] = phase_id
            buf.angle[frame] = th - th0
            buf.omega[frame] = knob.data.joint_vel[:, jid]
            buf.progress[frame] = ((th - th0) * turn_dir / goal).clamp(0.0, 1.0)
            buf.obj_pos[frame], buf.obj_quat[frame] = o_pos, o_quat
            tau = torch.zeros(n, device=device)
            ok = torch.ones(n, dtype=torch.bool, device=device)
            for k, pl in enumerate(plates):
                cp = best[k]
                pl_loc = to_local(cp["positions"], o_pos, o_quat)
                buf.pos_obj[frame, k] = pl_loc * cp["valid"].unsqueeze(-1)
                nrm_l = rotate_inverse(o_quat, cp["normals"])
                fri_l = rotate_inverse(o_quat, cp["friction_forces"])
                buf.nrm_obj[frame, k] = nrm_l
                buf.fri_obj[frame, k] = fri_l
                buf.fn[frame, k] = cp["normal_forces"]
                buf.sep[frame, k] = cp["separations"]
                buf.cvalid[frame, k] = cp["valid"]
                mu = torch.where(on_pin > 0, torch.full_like(on_pin, mu_pin),
                                 torch.full_like(on_pin, mu_rim))
                buf.mode[frame, k] = classify_contact_mode_padded(
                    cp["normal_forces"], cp["friction_forces"], cp["separations"],
                    cp["valid"], mu=float(mu_pin))
                # 接触在销钉上还是轮缘上（物体系半径判定）
                r_c = pl_loc[..., :2].norm(dim=-1)
                buf.on_pin_c[frame, k] = cp["valid"] & (
                    (pl_loc[..., :2] - torch.tensor([_K.pin_offset, 0.0], device=device)
                     ).norm(dim=-1) < _K.pin_radius + 0.012)
                # 绕轴力矩：接触力对圆盘轴的 Z 分量（作用在圆盘上 = 取负）
                f_w = -(cp["normal_forces"].unsqueeze(-1) * cp["normals"]
                        + cp["friction_forces"])
                r_w = cp["positions"] - o_pos.unsqueeze(1)
                tau = tau + torch.cross(r_w, f_w, dim=-1)[..., 2].sum(dim=1)
                buf.src_pose[frame, k] = torch.cat(
                    [pl.data.root_pos_w, pl.data.root_quat_w], dim=-1)
                buf.src_vel[frame, k] = torch.cat(
                    [pl.data.root_lin_vel_w, pl.data.root_ang_vel_w], dim=-1)
                buf.src_tgt[frame, k, :, :3] = targets[k]
                buf.src_tgt[frame, k, :, 3:] = quats[k]
                dq = (pl.data.root_quat_w * quats[k]).sum(dim=-1).abs().clamp(max=1.0)
                buf.rot_err[frame, k] = 2.0 * torch.acos(dq)
                ok &= torch.isfinite(pl.data.root_pos_w).all(dim=-1)
                ok &= (pl.data.root_pos_w - scene.env_origins).norm(dim=-1) < MAX_PLATE_DIST
                del r_c
            buf.tau[frame] = tau
            buf.valid[frame] = ok & torch.isfinite(th) & (
                buf.fn[frame].abs().sum(dim=(0, 2)) <= MAX_VALID_FORCE)

            if camera is not None and frame % every == 0:
                sim.render()
                camera.update(CONTROL_DT)
                preview.append(camera.data.output["rgb"][0, ..., :3]
                               .detach().cpu().numpy().astype(np.uint8))
            frame += 1

    n_cut, n_keep = float(buf.dropped.sum()), float(buf.cvalid.sum())
    print(f"  接触点：保留 {n_keep:.0f}，离所有板都远而丢弃 "
          f"{float(buf.foreign.sum()):.0f}，超上限截掉 {n_cut:.0f}", flush=True)
    if n_cut > 0.01 * max(n_keep, 1.0):
        raise RuntimeError(f"{n_cut:.0f} 个接触点被静默截掉（P-03），调大 MAX_CONTACTS")
    return buf, dict(goal=goal, omega=omega, turn_dir=turn_dir, damp=damp,
                     is_var=is_var, on_pin=on_pin, th0=th0), preview


def diagnostics(buf: Buffers, m: dict, e: int) -> dict[str, Any]:
    fn = buf.fn[:, :, e, :].abs()
    total = float(fn.sum().item())
    manip = buf.phase[:, e] == 2
    in_contact = fn.sum(dim=(1, 2)) > 0.05
    ang = buf.angle[:, e] * m["turn_dir"][e]

    def share(codes, table):
        return {nm: (float((fn * (codes == i)).sum().item() / total) if total > 0 else 0.0)
                for i, nm in enumerate(table)}

    return {
        "max_angle_rad": float(ang.max().item()),
        "goal_rad": float(m["goal"][e].item()),
        "reached_goal": bool(ang.max().item() >= m["goal"][e].item() - 0.05),
        "on_pin_force_share": float(
            (fn * buf.on_pin_c[:, :, e, :]).sum().item() / total) if total > 0 else 0.0,
        "manip_no_contact_fraction": float(
            ((manip & ~in_contact).sum() / max(int(manip.sum().item()), 1)).item()),
        "mean_torque_Nm": float((buf.tau[:, e] * m["turn_dir"][e])[manip].mean().item()),
        "peak_torque_Nm": float((buf.tau[:, e] * m["turn_dir"][e]).max().item()),
        "mean_contact_force_N": float(fn.sum(dim=(1, 2))[manip].mean().item()),
        "peak_point_force_N": float(fn.max().item()),
        # 只在 **manipulate 且有接触** 的帧上算超限比例。
        # 贴合瞬间的求解器瞬态一定会尖峰（P-27、以及 progress.md 里
        # "τ_pin 实测 23.48 vs 解析 1.30"那一条已经写明峰值不能当稳态用），
        # 拿全程的最大值去判安全上限，等于把瞬态当成持续受力。
        "over_safe_force_fraction": float(
            ((fn.sum(dim=(1, 2)) > MAX_NORMAL) & manip & in_contact).float().sum().item()
            / max(int((manip & in_contact).sum().item()), 1)),
        "p95_force_N": float(torch.quantile(
            fn.sum(dim=(1, 2))[manip & in_contact], 0.95).item())
        if int((manip & in_contact).sum().item()) > 0 else 0.0,
        "mode_share": share(buf.mode[:, :, e, :],
                            ("no_contact", "sticking", "sliding", "separating")),
        "mean_contacts_per_frame": float(
            buf.cvalid[:, :, e, :].float().sum(dim=(1, 2)).mean().item()),
        "invalid_frame_fraction": float(1.0 - buf.valid[:, e].float().mean().item()),
        "mean_orientation_error_deg": float(
            torch.rad2deg(buf.rot_err[:, :, e]).mean().item()),
        "uses_pin": bool(m["on_pin"][e] > 0),
    }


def judge(d: dict) -> tuple[bool, list[str]]:
    fails = []
    if not d["reached_goal"]:
        fails.append("没转到目标角")
    if d["manip_no_contact_fraction"] > 0.30:
        fails.append("操作阶段脱手过多")
    if d["invalid_frame_fraction"] > 0.10:
        fails.append("脏帧过多")
    if d["over_safe_force_fraction"] > 0.05:
        fails.append("接触力超过安全上限")
    if d["uses_pin"] and d["on_pin_force_share"] < 0.60:
        fails.append("接触没打在销钉上")
    return (not fails), fails


def to_arrays(buf: Buffers, e: int) -> dict[str, np.ndarray]:
    cpu = lambda t: t.detach().cpu().numpy()  # noqa: E731
    out = {
        "phase": cpu(buf.phase[:, e]),
        "progress": cpu(buf.progress[:, e]).astype(np.float32),
        "valid_frame": cpu(buf.valid[:, e]),
        "object/disc_angle": cpu(buf.angle[:, e]).astype(np.float32)[:, None],
        "object/disc_omega": cpu(buf.omega[:, e]).astype(np.float32)[:, None],
        "object/axis_torque": cpu(buf.tau[:, e]).astype(np.float32)[:, None],
        "source/disc_pos_w": cpu(buf.obj_pos[:, e]).astype(np.float32),
        "source/disc_quat_w": cpu(buf.obj_quat[:, e]).astype(np.float32),
    }
    for k in range(2):
        c, s = f"contact/plate{k}", f"source/plate{k}"
        out[f"{c}/pos_obj"] = cpu(buf.pos_obj[:, k, e]).astype(np.float32)
        out[f"{c}/normal_obj"] = cpu(buf.nrm_obj[:, k, e]).astype(np.float32)
        out[f"{c}/friction_obj"] = cpu(buf.fri_obj[:, k, e]).astype(np.float32)
        out[f"{c}/normal_force"] = cpu(buf.fn[:, k, e]).astype(np.float32)
        out[f"{c}/separation"] = cpu(buf.sep[:, k, e]).astype(np.float32)
        out[f"{c}/valid"] = cpu(buf.cvalid[:, k, e])
        out[f"{c}/mode"] = cpu(buf.mode[:, k, e])
        out[f"{c}/on_pin"] = cpu(buf.on_pin_c[:, k, e])
        out[f"{s}/root_pose"] = cpu(buf.src_pose[:, k, e]).astype(np.float32)
        out[f"{s}/root_velocity"] = cpu(buf.src_vel[:, k, e]).astype(np.float32)
        out[f"{s}/target_pose"] = cpu(buf.src_tgt[:, k, e]).astype(np.float32)
        out[f"{s}/cmd_delta"] = cpu(buf.src_cmd[:, k, e]).astype(np.float32)
    return out


def write_preview(path: Path, frames) -> None:
    if not frames:
        return
    import imageio.v2 as iio
    path.parent.mkdir(parents=True, exist_ok=True)
    with iio.get_writer(path, fps=_a.fps, codec="libx264", quality=8,
                        macro_block_size=1) as w:
        for fr in frames:
            w.append_data(fr)


def report(out_dir: Path, records, splits) -> None:
    lines: list[str] = []
    add = lines.append
    metas = [r.meta for _, r in records]
    ok = [m for m in metas if m["success"]]
    add(f"S3 旋钮双板 source · {len(metas)} episode，成功 {len(ok)} "
        f"({100.0 * len(ok) / max(len(metas), 1):.1f}%)")
    add("")
    add(f"{'家族':<18}{'条数':>5}{'成功':>6}{'转角rad':>9}{'目标rad':>9}{'脱手%':>8}"
        f"{'销钉上%':>9}{'力矩Nm':>9}{'力N':>7}{'峰值N':>8}")
    for fam in sorted({m["strategy_family"] for m in metas}):
        sub = [m for m in metas if m["strategy_family"] == fam]
        g = lambda k: float(np.mean([m["diagnostics"][k] for m in sub]))  # noqa: E731
        add(f"{fam:<18}{len(sub):>5}{sum(m['success'] for m in sub):>6}"
            f"{g('max_angle_rad'):>9.3f}{g('goal_rad'):>9.3f}"
            f"{100 * g('manip_no_contact_fraction'):>8.1f}"
            f"{100 * g('on_pin_force_share'):>9.1f}{g('mean_torque_Nm'):>9.3f}"
            f"{g('mean_contact_force_N'):>7.2f}{g('peak_point_force_N'):>8.2f}")
    add("")
    add("⚠️ `rim_only` **预期失败**：D-14 的低摩擦轮缘在安全力上限内传不出所需力矩。")
    add("   它的 episode 进 failed 桶，用作 D-14 的操作级证据与 `plan/05` 实验五的反事实。")
    add("")
    add("接触模式分布（按法向力加权，`plan/02` §3.4）")
    for nm in ("no_contact", "sticking", "sliding", "separating"):
        v = float(np.mean([m["diagnostics"]["mode_share"][nm] for m in metas]))
        add(f"  {nm:<14}{100 * v:>7.2f}%")
    add("")
    for k in ("peak_point_force_N", "p95_force_N", "over_safe_force_fraction",
              "invalid_frame_fraction", "mean_orientation_error_deg"):
        add(f"  {k:<30}{float(np.mean([m['diagnostics'][k] for m in metas])):>9.4f}")
    add("")
    add("划分（按 episode，P-10）")
    for nm, ids in splits.items():
        add(f"  {nm:<24}{len(ids):>5}")
    bad = [m for m in metas if not m["success"]]
    if bad:
        add("")
        add("失败原因分布")
        rs: dict[str, int] = {}
        for m in bad:
            for r in m["failure_reasons"]:
                rs[r] = rs.get(r, 0) + 1
        for r, c in sorted(rs.items(), key=lambda kv: -kv[1]):
            add(f"  {r:<26}{c:>5}")
    text = "\n".join(lines) + "\n"
    (out_dir / "report.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)


def main() -> int:
    out_dir = Path(_a.out)
    (out_dir / "episodes").mkdir(parents=True, exist_ok=True)
    fams = FAMILIES if _a.family == "all" else tuple(_a.family.split(","))
    for f in fams:
        if f not in FAMILIES:
            raise SystemExit(f"未知家族 {f}，可选：{FAMILIES}")

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=DT, device="cuda:0",
        physx=sim_utils.PhysxCfg(gpu_max_rigid_contact_count=2 ** 22,
                                 gpu_max_rigid_patch_count=2 ** 20)))
    scene = InteractiveScene(SceneCfg(num_envs=_a.envs, env_spacing=1.4,
                                      replicate_physics=True))
    sim.reset()
    camera: Camera | None = scene["cam"] if _a.video else None
    device, rng, sha = sim.device, np.random.default_rng(_a.seed), _git_sha()

    records: list[tuple[str, EpisodeRecord]] = []
    for b in range(_a.batches):
        fam_of = [fams[(b * _a.envs + i) % len(fams)] for i in range(_a.envs)]
        print(f"\n=== knob batch {b + 1}/{_a.batches} · {_a.envs} env ===", flush=True)
        buf, m, preview = run_batch(scene, sim, camera, fam_of, rng, device, b)
        if camera is not None:
            write_preview(out_dir / "videos" / f"knob_{fam_of[0]}_b{b}.mp4", preview)
        for e in range(_a.envs):
            d = diagnostics(buf, m, e)
            good, fails = judge(d)
            fam = fam_of[e]
            eid = f"knob-{fam}-b{b:02d}e{e:03d}"
            meta = {
                "schema_version": SCHEMA_VERSION, "episode_id": eid, "task": "knob",
                "source_embodiment": "two_dynamic_plates",
                "strategy_family": fam, "strategy_variant": f"b{b:02d}e{e:03d}",
                "physics_variant": "heldout_damping" if bool(m["is_var"][e]) else "nominal",
                "success": good, "failure_reasons": fails,
                "expected_to_fail": fam == "rim_only",
                "seed": int(_a.seed + b * _a.envs + e),
                "control_hz": 1.0 / CONTROL_DT, "physics_hz": 1.0 / DT,
                "phase_names": list(PHASE_NAMES), "phase_steps": list(PHASE_STEPS),
                "generator_git_sha": sha,
                "goal_rad": float(m["goal"][e].item()),
                "physics": {"joint_damping": float(m["damp"][e].item()),
                            "rim_friction": _K.rim_friction,
                            "pin_friction": _K.pin_friction,
                            "plate_friction": _P.friction},
                "source_params": {"omega_cmd": float(m["omega"][e].item()),
                                  "turn_dir": float(m["turn_dir"][e].item()),
                                  "start_angle": float(m["th0"][e].item()),
                                  "uses_pin": bool(m["on_pin"][e] > 0)},
                "diagnostics": d,
            }
            rec = EpisodeRecord(meta=meta, arrays=to_arrays(buf, e))
            records.append((save_episode(rec, out_dir / "episodes" / f"{eid}.npz"), rec))
        del buf

    entries = [r.to_manifest_entry("x") for _, r in records]
    splits = split_episode_entries(entries, seed=_a.seed,
                                   holdout_strategy_family=_a.holdout_family)
    write_manifest(records, out_dir / "manifest.json", dataset_name="s3_knob_source",
                   generator_git_sha=sha, splits=splits,
                   extra={"task": "knob", "families": list(fams),
                          "holdout_strategy_family": _a.holdout_family})
    report(out_dir, records, splits)
    return 0


try:
    code = main()
except Exception:
    import traceback
    traceback.print_exc()
    code = 1
sys.stdout.flush()
os._exit(code)   # P-19
