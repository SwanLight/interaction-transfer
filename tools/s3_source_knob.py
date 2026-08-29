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
    quat_face_and_up,
    rotate_inverse,
    to_local,
)
from it.contact_utils import (  # noqa: E402
    classify_contact_mode_padded,
    contact_rel_vel,
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
PHASE_STEPS = (50, 55, 155, 45)
N_FRAMES = sum(PHASE_STEPS)

_K = B.KnobCfg()
_P = B.PlateCfg()
#: 圆盘上表面高度、销钉中段高度（旋钮局部系，销钉是圆盘的子几何）
DISC_TOP = _K.disc_thickness / 2
PIN_Z = DISC_TOP + _K.pin_length * 0.55
#: 指令角速度。**必须落在设计点附近**（`plan/01` §3.1 按 1.5 rad/s 标定），
#: 转得慢 τ_need 会掉下来，轮缘反而推得动，rim_only 就失去了对照意义。
OMEGA_RANGE = (1.2, 1.8)
#: 目标角**由角速度导出**：goal = ω × T_PUSH。
#: 初版独立采 goal 和 ω，于是驱动时长在 0.5~1.8 s 之间变，而 manipulate
#: 相位固定 3.2 s——转到位之后板就撤力，剩下的两秒全是空转，
#: "操作阶段脱手" 因此高达 63%，判据把本来干净的 episode 全判失败。
T_PUSH = 1.55
GOAL_RANGE = (OMEGA_RANGE[0] * T_PUSH, OMEGA_RANGE[1] * T_PUSH)
#: 跟随板留的缝。它只在需要时挡一下，不吃力矩（见 `arc` 的说明）。
FOLLOW_GAP = 6.0 * MM
#: 推力上限。25 N 是 `plan/01` §3.2 的硬约束。
#: 18 N 时 `pin_push_dual` 实测合力均值 16.6 N、有 6.6% 的操作帧越过 25 N——
#: 它两块板上下夹住销钉，指令力几乎全部落到物体上，而单板家族只落到四成多。
#: 上限对所有家族一律取 13 N：压得比允许的轻永远是合规的，
#: 按家族分别放宽才是把上限当成可调参数。
MAX_PUSH_FORCE = 13.0
#: 位置 PD 的等效接触刚度 kp_pos × 质量 = 3000 × 0.5 = 1500 N/m。
#: 干涉深度 δ 与推力的换算全靠它。
K_CONTACT = 1500.0
#: 轮缘家族的切向领先角。轮缘靠摩擦拖，必须有相对滑移才有拖力。
RIM_LEAD = 0.08
#: 轮缘家族的径向位置干涉，见 `press_full` 处的实测说明。
RIM_PRESS = 6.5 * MM
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

    **板站哪一侧不写死，由转向决定。** 圆盘往 ``turn_dir`` 转时，能推的那一侧
    是 ``-turn_dir``；写死成 -1 的话，转向为 -1 的那一半 episode 里推的板
    会被判成"跟随板"而留缝，根本碰不到销钉——实测 `pin_push_single`
    正好约一半的操作步零接触。

    Returns:
        ``role`` (n,2)：+1 = 推、-1 = 跟随（留缝）、0 = 不参与；
        ``on_pin`` (n,) 接销钉还是接轮缘；``regrasp`` (n,) 是否中途松开重抓。
    """
    role = torch.zeros(n, 2)
    on_pin = torch.ones(n)
    regrasp = torch.zeros(n)
    if fam == "pin_pinch":
        role[:, 0], role[:, 1] = 1.0, -1.0         # 一推一跟，形成对捏
    elif fam == "pin_push_single":
        role[:, 0], role[:, 1] = 1.0, 0.0          # 只用一块板推
    elif fam == "pin_push_dual":
        role[:, :] = 1.0                           # 两块板同侧推，沿销钉轴向错开
    elif fam == "pin_regrasp":
        role[:, 0], role[:, 1] = 1.0, -1.0
        regrasp = torch.ones(n)
    elif fam == "rim_only":
        role[:, 0], role[:, 1] = 1.0, 1.0          # 径向对置压轮缘，靠摩擦拖
        on_pin = torch.zeros(n)
    else:
        raise ValueError(f"未知策略家族：{fam}")
    return {k: v.to(device) for k, v in
            dict(role=role, on_pin=on_pin, regrasp=regrasp).items()}


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
        self.rvel_obj = z(2, n, MAX_CONTACTS, 3)
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

    par = {"role": torch.zeros(n, 2, device=device)}
    par.update({k: torch.zeros(n, device=device) for k in ("on_pin", "regrasp")})
    for fam in sorted(set(fam_of)):
        idx = [i for i, f in enumerate(fam_of) if f == fam]
        sub = sample_family(fam, rng, len(idx), device)
        sel = torch.tensor(idx, device=device, dtype=torch.long)
        for k in par:
            par[k][sel] = sub[k]
    role, on_pin, regrasp = par["role"], par["on_pin"], par["regrasp"]

    omega = _u(rng, *OMEGA_RANGE, n).to(device)
    goal = omega * T_PUSH
    # **只能往正方向转。** 旋钮的 revolute 关节限位是 -10°~+200°
    # （`build_assets.KnobCfg.joint_limit_deg`），它是个带死挡的单向旋钮。
    # 初版把转向做成随机 ±1，于是**一半 episode 一上来就顶在 -10° 死挡上**，
    # 目标角 1.0~2.2 rad 在那个方向根本不存在——两次控制方案完全不同的实验里
    # 转角均值却几乎一模一样（0.780 vs 0.782、0.814 vs 0.813），
    # 正是因为这一半 episode 与控制无关地钉死在限位上。见 P-43。
    turn_dir = torch.ones(n, device=device)
    # **几何变体**（`plan/03` §7 的"小幅几何变化"）：销钉偏心距按 env 轮转，
    # 46 / 52 / 58 mm。`MultiUsdFileCfg(random_choice=False)` 的分配是
    # env 下标 % 槽位数，所以这里能确定地知道每个 env 拿到的是哪一个。
    geom_tag = [A.geom_tag_of(i) for i in range(n)]
    pin_off = torch.tensor(
        [float(B.variant_cfg("knob", t).pin_offset) for t in geom_tag], device=device)
    # 接触半径：销钉家族接销钉侧面，轮缘家族接圆盘外缘
    radius = torch.where(on_pin > 0, pin_off,
                         torch.full((n,), _K.disc_radius, device=device))
    # 接触高度：销钉在盘面之上，轮缘在盘厚中段
    height = torch.where(on_pin > 0,
                         torch.full((n,), PIN_Z, device=device),
                         torch.zeros(n, device=device))
    if os.environ.get("IT_PIN_SWEEP"):
        # 诊断用：把接触高度沿销钉从下到上扫一遍，量销钉碰撞体的实际竖直范围
        height = torch.linspace(0.004, 0.058, n, device=device)
        print("SWEEP 高度(mm):", [round(float(h) * 1000, 1) for h in height], flush=True)
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

    #: **同侧两块板必须沿销钉轴向错开**，否则被指到同一个点上互相穿插——
    #: pin_push_dual 实测 66% 的操作步脱手、力矩上不去，与 P-32 同类。
    #:
    #: 错开量由几何定死，不是调出来的：板横放时竖直方向占 25 mm，销钉可用段
    #: 是 z ∈ [7.5, 55.5] mm（盘面之上到销钉顶）。两块 25 mm 高的板放进 48 mm
    #: 里，中心只能取 22.0 与 48.0 mm——下缘 9.5 mm 高于盘面 2 mm，两板之间
    #: 留 1 mm。初版按 PIN_Z ± 12 mm 放，下缘落到 4.4 mm，**低于盘面**，
    #: 一直在蹭圆盘顶面：`pin_push_dual` 的"接触在销钉上"因此只有 67%。
    both_push = (role[:, 0] > 0) & (role[:, 1] > 0) & (on_pin > 0)
    _zc = (DISC_TOP + _P.size[1] / 2 + 2 * MM,
           DISC_TOP + _P.size[1] / 2 + 2 * MM + _P.size[1] + 1 * MM)
    z_stag = torch.stack([(_zc[k] - PIN_Z) * both_push.float() for k in (0, 1)], dim=-1)

    def arc(theta, k, dl):
        """给定圆盘的**实际**角度，返回第 k 块板的目标位姿与**推力方向**。

        参考系挂在**销钉当前所在的位置**上，板贴在该处的切向一侧，
        推力方向恒为"从板指向销钉"——也就是纯切向。力全部转成力矩。

        ⚠️ 第一版把参考系挂在"领先角"上：板既沿弧提前了 r·lead、又沿切向
        偏了 off，两个偏移叠加，板跑到销钉的**斜前方**。实测领先角 0.20 rad
        时接触法向偏离切向 42°，三分之一的力顶着销钉往圆心推、不产生任何
        力矩；为了补偿又要更大的力，正反馈直到 PD 饱和（力恒定在 49.9 N，
        圆盘停在 1.709 rad 转不动）。见 P-42。

        侧向由**转向**决定，不能写死：圆盘往 ``turn_dir`` 转时能推的是
        ``-turn_dir`` 那一侧。写死成 -1 的话，转向为 -1 的那一半 episode 里
        推板会被当成跟随板留缝，全程零接触——`pin_push_single` 正是如此。

        轮缘家族两块板**径向对置**（θ 与 θ+π），推力沿径向向内，并沿切向
        领先 ``RIM_LEAD``：摩擦拖动必须有相对滑移才有拖力。
        """
        s_k = torch.where(role[:, k] > 0, -turn_dir, turn_dir)
        gap = torch.where(role[:, k] > 0, -dl,
                          torch.full((n,), FOLLOW_GAP, device=device))
        # 轮缘板与销钉在角度上**永远差 90°**。初版只领先圆盘 RIM_LEAD=0.08 rad
        # (4.6°)，在 r≈62 mm 处离销钉才 5 mm，而板宽 25 mm——两块板全程压在
        # 销钉上，"接触在销钉上"高达 92%，这一档就不再是"不用销钉"的对照了。
        th_rim = (theta + turn_dir * RIM_LEAD + math.pi / 2
                  + (0.0 if k == 0 else math.pi))
        th_use = torch.where(on_pin > 0, theta, th_rim)
        c, s_ = torch.cos(th_use), torch.sin(th_use)
        rad = torch.stack([c, s_, torch.zeros_like(c)], dim=-1)          # 径向
        tan = torch.stack([-s_, c, torch.zeros_like(c)], dim=-1)         # 切向
        h_k = height + z_stag[:, k] * (on_pin > 0).float()
        center = axis + rad * radius.unsqueeze(-1) \
            + torch.stack([torch.zeros_like(c), torch.zeros_like(c), h_k], dim=-1)
        off_pin = (surf + _P.size[2] / 2 + gap).unsqueeze(-1)
        pin_pos = center + tan * off_pin * s_k.unsqueeze(-1)
        rim_pos = center + rad * (_P.size[2] / 2 + gap).unsqueeze(-1)
        pos = torch.where(on_pin.unsqueeze(-1) > 0, pin_pos, rim_pos)
        # 从板指向接触体的方向：销钉家族是切向，轮缘家族是径向朝内。
        # 它同时是板的局部 +Z（工作面法向，见 `contact_attrib.quat_face_and_up`）
        # 和力控方向。
        n_push = torch.where(on_pin.unsqueeze(-1) > 0,
                             -tan * s_k.unsqueeze(-1), -rad)
        # **板的长边必须与销钉轴平行**（竖放，35 mm 沿销钉）。
        #
        # 实测：把同一块板横过来（25 mm 沿销钉、35 mm 沿径向），切向力控下
        # 板会**径直从 Ø20 的销钉里穿过去**，接触点数全程为零。沿销钉高度
        # 扫一遍，横放时纯销钉段的接触力是 0.0~0.4 N，竖放时是 3.7~13.3 N，
        # 而**静态**位置控制下两种朝向都正常（2 mm 干涉都量到 3 N）——
        # 所以这是动态接触问题，不是几何摆错。见 P-46。
        #
        # 唯一的例外是 `pin_push_dual`：两块板要沿销钉上下错开，而销钉可用段
        # 只有 48 mm，放不下两块 35 mm 的竖板（需要 71 mm）。那一档只能横放，
        # 而横放对它是可行的——两块板上下夹着销钉互相约束，实测能稳定推转。
        _up = torch.tensor([[0.0, 0.0, 1.0]], device=device).expand(n, 3)
        # 轮缘家族同样横放：竖放时 35 mm 的板跨在只有 15 mm 厚的盘缘上，
        # 实测操作阶段全程脱手；横放实测稳定压住、法向力 19.5 N。
        #
        # 深色鳍（局部 +Y）的朝向必须**两块板一致**，否则录像里看着像其中一块
        # 翻了 180°（实测两块板的鳍夹角余弦 -0.99）。见 `quat_face_and_up`：
        #   竖放家族 —— 长边被约束成竖直，鳍只能水平，统一取"径向朝外"；
        #   横放家族 —— 长边不受约束，鳍朝上。
        vert = ((on_pin > 0) & ~both_push).unsqueeze(-1)
        q_vert = quat_face_and_up(n_push, rad, long_axis=_up)
        q_flat = quat_face_and_up(n_push, _up)
        return pos, torch.where(vert, q_vert, q_flat), n_push

    pds = [FloatingPD(pl, kp_pos=3000.0, kd_pos=110.0, kp_rot=6.0, kd_rot=0.025,
                      max_force=45.0, max_torque=4.0, kd_force=40.0) for pl in plates]

    targets, quats = [], []
    for k, pl in enumerate(plates):
        p0, q0, _ = arc(th0, k, torch.zeros(n, device=device))
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
    # **切向走力控，压力不再靠位置干涉换。**
    #
    # 位置干涉的等效刚度只有 kp_pos×质量 = 1500 N/m：要推出 10 N 得压进
    # 6.7 mm，而销钉半径才 10 mm——板等于插进销钉里去了。实测单板推力
    # 只有 2.45 N（设计值 11 N），圆盘转到一半就停。
    #
    # 力控方向是**销钉当前位置的切向**，每个控制步重算，随圆盘一起转
    # （`FloatingPD` 的 ``force_dir``，P-39）；正交补里仍走位置 PD，
    # 板的半径与高度因此被稳稳按住。
    #
    #   需要的力矩 τ = 阻尼 × ω，力臂是接触半径 -> F = 阻尼·ω / r（×1.15 余量）
    r_eff = torch.where(on_pin > 0, pin_off,
                        torch.full((n,), _K.disc_radius, device=device))
    n_push_plates = (role > 0).float().sum(dim=1).clamp_min(1.0)
    f_total = ((damp * omega / r_eff) * 1.35).clamp(max=MAX_PUSH_FORCE)
    # 轮缘家族直接给到上限：μ 只有 0.10，即便如此也拖不动——D-14 要验的正是
    # 这个。τ_rim = μ·F·R = 0.10 × 18 × 0.070 = 0.126 N·m ≪ τ_need ≈ 0.42。
    f_total = torch.where(on_pin > 0, f_total,
                          torch.full_like(f_total, MAX_PUSH_FORCE))
    # **两块板一起推时按块平分**。25 N 安全上限（`plan/01` §3.2）是对被操作
    # 物体上的合力判的，不能因为多加一块板就翻倍——那是把上限放宽来让数据
    # 过关。平分之后力矩不变、合力不变。
    f_share = f_total / n_push_plates
    if os.environ.get("IT_PIN_STATIC"):
        f_share = torch.zeros_like(f_share)          # 只留位置控制，测静态几何
    # 板要跟着销钉一起以 v = ω·r 走，而力控轴的阻尼是**对地**的（P-33）：
    # 不补的话稳态时白白扣掉 kd·m·v。
    v_follow = omega * r_eff
    #: 销钉家族在 establish 相位先用**位置干涉**压出 0.8 mm 的实接触，
    #: 再把切向交给力控。直接从"贴着但零穿透"起步时接触力是零，前馈力让板
    #: 自由加速——实测一个控制步就冲进销钉 5~7 mm，随后整块板从 Ø20 的销钉里
    #: 穿过去撞在另一块板背上（接触点数全程 0，因为板与板的接触不在传感器
    #: 过滤名单里）。见 P-45。
    PIN_ENGAGE = 2.0 * MM if os.environ.get("IT_PIN_STATIC") else 0.8 * MM
    PIN_HOLD = 2.0 * MM
    # **轮缘家族不走力控，走位置干涉。**
    # 径向力控在光滑柱面上没有任何位置反馈拦着板：实测板压进轮缘 8 mm 之后
    # 顺着 15 mm 厚的边带滑下去，钻到圆盘底下卡在立柱上，接触力 1229 N
    # 而力矩为零（见 P-44）。位置干涉是自限的——r 和 z 都被位置 PD 按住。
    # 干涉量取安全上限对应的深度：这就是轮缘能被公平地压到的最大程度。
    # 干涉量按**实测**力定：12 mm 时量到 32 N，超过 25 N 安全上限，
    # 6.5 mm 折合约 17 N。这是轮缘能被公平地压到的最大程度，不是为了让它
    # 转不动而调小的——D-14 的对照必须给足力才算数。
    press_full = torch.where(on_pin > 0,
                             torch.full((n,), PIN_ENGAGE, device=device),
                             torch.full((n,), RIM_PRESS, device=device))
    # 操作阶段的干涉量。切向走力控时它不起作用（那个轴被投影掉了），
    # 只在**转到位、撤掉推力之后**决定"把住旋钮"的接触深度：
    # 0.8 mm 只有 1.2 N，量到的接触时有时无，2 mm 稳定在 3 N。
    press_manip = torch.where(on_pin > 0,
                              torch.full((n,), PIN_HOLD, device=device),
                              torch.full((n,), RIM_PRESS, device=device))

    frame = 0
    for phase_id, plen in enumerate(PHASE_STEPS):
        for kk in range(plen):
            # 参考角**永远取圆盘的实际角度**，板始终骑在销钉当前位置上。
            th_cmd = knob.data.joint_pos[:, jid]
            reached = (th_cmd - th0) * turn_dir >= goal
            # 推力前 15 步线性加载，避免板一落位就是满力；转到位就撤力，
            # 让圆盘靠关节阻尼自己停住。
            if phase_id == 2:
                ramp = min((kk + 1) / 25.0, 1.0)
                push_on = torch.where(reached, torch.zeros_like(f_share),
                                      f_share * ramp)
                # **转到位之后停止推进，但仍轻轻把住旋钮**（保持 press_full 的
                # 微干涉），而不是撒手。撒手时剩下的窗口全是空转，
                # "操作阶段脱手" 会因此报到 30% 以上——那是"转完了"，
                # 不是"脱手"。把住旋钮既符合操作常识，也不用去动判据。
                press = press_manip
            elif phase_id == 1:
                push_on = torch.zeros(n, device=device)
                press = press_manip * min((kk + 1) / max(plen - 8, 1), 1.0)
            else:
                push_on = torch.zeros(n, device=device)
                press = torch.zeros(n, device=device)
            back = torch.zeros(n, device=device)
            if phase_id == 2 and bool((regrasp > 0).any()):
                r_ph = (kk / plen)
                if 0.45 < r_ph < 0.60:
                    back = regrasp * SAFE_R * math.sin(
                        math.pi * (r_ph - 0.45) / 0.15)
                    # 松开期间不再推，让圆盘靠阻尼自己停下
                    push_on = torch.where(regrasp > 0, torch.zeros_like(push_on),
                                          push_on)
                    press = torch.where(regrasp > 0, torch.zeros_like(press), press)  # 重抓时确实松开

            ff_f = [None, None]
            dir_f = [None, None]
            for k in range(2):
                p, q, n_push = arc(th_cmd, k, press)
                quats[k] = q
                # 只有"推"的板走力控；不推的板 force_dir 取零向量，
                # `FloatingPD` 会退回纯位置 PD（法向分量为零）。
                act = ((role[:, k] > 0) & (push_on > 0)
                       & (on_pin > 0)).float().unsqueeze(-1)
                dir_f[k] = n_push * act
                ff_f[k] = n_push * ((push_on + pds[k].kd_force * pds[k].mass
                                     * v_follow) * act.squeeze(-1)).unsqueeze(-1)
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
                tgt[:] = torch.where(role[:, k].unsqueeze(-1) != 0, tgt,
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
                    f, tq = pds[k].compute(targets[k], quats[k],
                                           ff_force=ff_f[k], force_dir=dir_f[k])
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
                # `plan/03` §5 要求逐帧记录接触点的**相对速度**。PhysX 的接触
                # 缓冲不给，但两个接触体都是刚体，接触点处的速度是解析的。
                rv = contact_rel_vel(
                    cp["positions"], pl.data.root_pos_w,
                    pl.data.root_lin_vel_w, pl.data.root_ang_vel_w,
                    o_pos, knob.data.body_lin_vel_w[:, bid, :],
                    knob.data.body_ang_vel_w[:, bid, :])
                buf.rvel_obj[frame, k] = rotate_inverse(o_quat, rv) * cp["valid"].unsqueeze(-1)
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
                    (pl_loc[..., :2] - torch.stack(
                        [pin_off, torch.zeros_like(pin_off)], dim=-1).unsqueeze(1)
                     ).norm(dim=-1) < _K.pin_radius + 0.005)
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

            if os.environ.get("IT_KNOB_DEBUG") and frame % 10 == 0:
                for e_ in range(min(2, n)):
                    row = [f"f{frame:3d} ph{phase_id} e{e_} th{float(th[e_]):+.3f}"
                           f" tau{float(tau[e_]):+.3f}"]
                    row.append(f"push{float(push_on[e_]):5.2f}")
                    for k, pl in enumerate(plates):
                        d_ = pl.data.root_pos_w[e_] - o_pos[e_]
                        err = targets[k][e_] - pl.data.root_pos_w[e_]
                        lc = to_local(pl.data.root_pos_w[e_].view(1, 1, 3),
                                      o_pos[e_].view(1, 3), o_quat[e_].view(1, 4))[0, 0]
                        row.append(f"| P{k} loc({float(lc[0])*1000:6.1f},"
                                   f"{float(lc[1])*1000:6.1f},{float(lc[2])*1000:6.1f})"
                                   f" r{float(d_[:2].norm())*1000:6.1f}"
                                   f" z{float(d_[2])*1000:+6.1f}"
                                   f" err{float(err.norm())*1000:6.1f}"
                                   f" dirN{float(dir_f[k][e_].norm()):4.1f}"
                                   f" ff{float(ff_f[k][e_].norm()):5.1f}"
                                   f" np{int(buf.cvalid[frame, k, e_].sum())}"
                                   f" F{float(buf.fn[frame, k, e_].abs().sum()):6.1f}")
                    print(" ".join(row), flush=True)

            if camera is not None and frame % every == 0:
                sim.render()
                camera.update(CONTROL_DT)
                preview.append(camera.data.output["rgb"][0, ..., :3]
                               .detach().cpu().numpy().astype(np.uint8))
            frame += 1

    if os.environ.get("IT_PIN_SWEEP"):
        mf = buf.fn[:, :, :, :].abs().sum(dim=(1, 3))          # (F, N)
        mm_ = buf.phase[:, 0] == 2
        print("SWEEP 力(N) :", [round(float(v), 2) for v in mf[mm_].mean(dim=0)],
              flush=True)
    n_cut, n_keep = float(buf.dropped.sum()), float(buf.cvalid.sum())
    print(f"  接触点：保留 {n_keep:.0f}，离所有板都远而丢弃 "
          f"{float(buf.foreign.sum()):.0f}，超上限截掉 {n_cut:.0f}", flush=True)
    if n_cut > 0.01 * max(n_keep, 1.0):
        raise RuntimeError(f"{n_cut:.0f} 个接触点被静默截掉（P-03），调大 MAX_CONTACTS")
    return buf, dict(goal=goal, omega=omega, turn_dir=turn_dir, damp=damp,
                     is_var=is_var, on_pin=on_pin, th0=th0,
                     geom_tag=geom_tag, pin_off=pin_off), preview


def diagnostics(buf: Buffers, m: dict, e: int) -> dict[str, Any]:
    fn = buf.fn[:, :, e, :].abs()
    total = float(fn.sum().item())
    manip = buf.phase[:, e] == 2
    in_contact = fn.sum(dim=(1, 2)) > 0.05
    # **"脱手"只在驱动段判。** 转到目标角之后板就不再推、只轻轻把住，
    # 那段窗口里接触时有时无是"转完了"，不是"脱手"。把它算进去时同一批
    # 干净轨迹会因为窗口留得长而被判失败（实测窗口 130→155 步，脱手率
    # 27%→38%，而驱动段本身没有任何变化）。保持段的接触率单独报，不藏。
    drive = manip & (buf.progress[:, e] < 1.0)
    if int(drive.sum().item()) < 30:
        drive = manip                       # 没真正驱动过，退回整段判
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
            ((drive & ~in_contact).sum() / max(int(drive.sum().item()), 1)).item()),
        "drive_frames": int(drive.sum().item()),
        "hold_contact_fraction": float(
            ((manip & ~drive & in_contact).sum()
             / max(int((manip & ~drive).sum().item()), 1)).item()),
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
        out[f"{c}/rel_vel_obj"] = cpu(buf.rvel_obj[:, k, e]).astype(np.float32)
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
    add("「脱手%」只统计**驱动段**（转到目标角之前）；转到位之后板不再推、"
        "只轻轻把住，那段的接触率单独列在「保持接触%」。")
    add("")
    add(f"{'家族':<18}{'条数':>5}{'成功':>6}{'转角rad':>9}{'目标rad':>9}{'脱手%':>8}"
        f"{'销钉上%':>9}{'力矩Nm':>9}{'力N':>7}{'峰值N':>8}{'驱动帧':>7}{'保持接触%':>10}")
    for fam in sorted({m["strategy_family"] for m in metas}):
        sub = [m for m in metas if m["strategy_family"] == fam]
        g = lambda k: float(np.mean([m["diagnostics"][k] for m in sub]))  # noqa: E731
        add(f"{fam:<18}{len(sub):>5}{sum(m['success'] for m in sub):>6}"
            f"{g('max_angle_rad'):>9.3f}{g('goal_rad'):>9.3f}"
            f"{100 * g('manip_no_contact_fraction'):>8.1f}"
            f"{100 * g('on_pin_force_share'):>9.1f}{g('mean_torque_Nm'):>9.3f}"
            f"{g('mean_contact_force_N'):>7.2f}{g('peak_point_force_N'):>8.2f}"
            f"{g('drive_frames'):>7.0f}{100 * g('hold_contact_fraction'):>10.1f}")
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
                                      # **必须 False**：`replicate_physics=True` 时
                                      # Isaac Lab 把 env_0 的物理整体复制给所有 env，
                                      # `MultiUsdFileCfg` 的多资产会被抹平——实测 24 个
                                      # env 全部拿到名义几何，而代码以为其中 4 个是变体。
                                      # 几何变体（`plan/03` §7）靠它才成立。
                                      replicate_physics=False))
    sim.reset()
    camera: Camera | None = scene["cam"] if _a.video else None
    device, rng, sha = sim.device, np.random.default_rng(_a.seed), _git_sha()

    records: list[tuple[str, EpisodeRecord]] = []
    for b in range(_a.batches):
        # 用 (b + i) 而不是 (b*envs + i)：录像只录 env 0，而 envs 是 5 的倍数时
        # env 0 永远落在同一个家族上，25 个批次只能拍到一个家族。
        fam_of = [fams[(b + i) % len(fams)] for i in range(_a.envs)]
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
                "geometry_variant": m["geom_tag"][e],
                "implementation": "two_plate_pin",
                "success": good, "failure_reasons": fails,
                "expected_to_fail": fam == "rim_only",
                "seed": int(_a.seed + b * _a.envs + e),
                "control_hz": 1.0 / CONTROL_DT, "physics_hz": 1.0 / DT,
                "phase_names": list(PHASE_NAMES), "phase_steps": list(PHASE_STEPS),
                "generator_git_sha": sha,
                "goal_rad": float(m["goal"][e].item()),
                "geometry": {"pin_offset_mm": float(m["pin_off"][e].item() * 1000)},
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
