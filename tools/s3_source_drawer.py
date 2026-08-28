"""S3 · 抽屉任务的双板 source 采集器（脚本化阶段机，代替人）。

`plan/03` §1：双板 source 只负责产生**多样、成功、物理有效**的示教，它不是
最终算法的一部分。`plan/03` §3 规定第一轮用脚本化阶段机保证策略覆盖：

    approach → establish → manipulate → release

## 动作设计的依据

两块 35×25×3 mm 的板要把抽屉拉开，几何上只有这么几种做法。人开带横杆把手
的抽屉时，是**手指伸进把手后面的净空、拇指按在杆子前面，然后往外拉**——
拉力主要来自手指压在杆子**背面**的法向力，拇指提供对捏和稳定。本脚本照此
实现，并由此派生出五个策略家族（`plan/03` §2.2 要求抽屉 ≥3 个）。

净空几何（抽屉局部系，mm）：

    面板  x ∈ [0, 18]          横杆  x ∈ [63, 85]，z ∈ [79, 101]，轴沿 Y
    净空  x ∈ [18, 63]         支撑柱 |y| = 62.5 ± 8，x ∈ [18, 74]
    -> |y| < 54.5 这一段净空是通的，35 mm 宽的板可以竖着插进去

所以板从**杆子下方**平飞进来、升到杆心高度、再用 X 轴力控贴上去。这条路径
上不会撞到任何东西，且看起来就是"把手指伸到把手底下再勾上去"。

## 为什么每条 episode 都带接触部位统计

D-34 的教训：钩杆 78% 的接触在主杆上、真正的横钩只占 4.3%，而成功率、
接触力、无穿模检查**全都看不出来**。采集数据比训练策略更受这件事影响——
`plan/02` §3.2 的 Interaction Region 就是"接触落在物体表面哪里"。如果板子
实际是拿边角蹭面板而不是拿工作面压把手，region 字段从第一天起就是错的。
`it.contact_attrib` 把归类做成数据契约的一部分，不是可选诊断。

## 用法

    ./tools/run_remote.sh "PYTHONPATH=src /isaac-sim/python.sh \\
        tools/s3_source_drawer.py --batches 8 --envs 64 --out /tmp/s3_drawer" s3

    # 验收录像（每个家族一条，只录 env 0）
    ./tools/run_remote.sh "PYTHONPATH=src /isaac-sim/python.sh \\
        tools/s3_source_drawer.py --video --envs 4 --batches 1 \\
        --out /tmp/s3_vid" s3vid

**必须 pin 单卡**（P-29），``run_remote.sh`` 已内置 ``IT_GPU``。
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

_ap = argparse.ArgumentParser(description="S3 抽屉双板 source 采集")
_ap.add_argument("--envs", type=int, default=64, help="并行环境数 = 每批 episode 数")
_ap.add_argument("--batches", type=int, default=8)
_ap.add_argument("--out", default="/tmp/s3_drawer")
_ap.add_argument("--seed", type=int, default=20260829)
_ap.add_argument("--family", default="all", help="逗号分隔的家族名，或 all")
_ap.add_argument("--holdout-family", default="asym_primary",
                 help="划进 unseen_strategy_test 的家族（`plan/03` §7）")
_ap.add_argument("--physics-variant-frac", type=float, default=0.15,
                 help="采样到留出物理区间的 episode 比例，用于 unseen_physics_test")
_ap.add_argument("--video", action="store_true", help="录 env 0 的预览视频")
_ap.add_argument("--width", type=int, default=1280)
_ap.add_argument("--height", type=int, default=720)
_ap.add_argument("--fps", type=int, default=25)
_a, _ = _ap.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

_app = AppLauncher(headless=True, enable_cameras=_a.video).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, AssetBaseCfg, RigidObject  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg, ContactSensorCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import quat_from_angle_axis, quat_mul  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from it import assets as A  # noqa: E402
from it.build_assets import CabinetCfg, PlateCfg  # noqa: E402
from it.contact_attrib import (  # noqa: E402
    DRAWER_PARTS,
    PLATE_PARTS,
    bar_span_fraction,
    classify_drawer_local,
    classify_plate_face,
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

C, P = CabinetCfg(), PlateCfg()
MM = 0.001
#: 物理 300 Hz、decimation 6 -> 记录仍是 50 Hz（`plan/03` §5 要求的控制/记录频率）。
#: 比 S2 的 150 Hz 细一倍，是为了**姿态 PD 的数值稳定性**：板只有 35×25 mm，
#: 最小转动惯量 I ≈ 2.6e-5，显式积分的稳定条件 ω·dt < 2 会把姿态刚度压到
#: 守不住姿态（150 Hz 时实测平均偏 20°、峰值 74°，且无接触时也偏——是极限环，
#: 不是接触扰动）。dt 减半 -> 允许的刚度变成 4 倍。采集阶段多一倍算力很便宜。
DT = 1.0 / 300.0
DECIMATION = 6
CONTROL_DT = DT * DECIMATION
#: 每个 prim 的接触数据槽位。**不是每 env 的上限，是共享池的每 env 份额**——
#: 池子总长 = 本值 × env 数，某个 env 暴涨会挤占别人的槽位，超出池子就串环境
#: 且不报错（P-30）。S3 实测单板峰值 ~12 点，取 64 留 5 倍余量。
MAX_CONTACTS = 24

#: 把手横杆中心相对抽屉局部原点
BAR_X = C.panel_t + C.handle_clearance + C.handle_radius     # 74 mm
BAR_Z = C.panel_h / 2                                        # 90 mm
#: 板心贴到杆面时的 x 偏移（杆半径 + 板半厚），再留 1 mm 预接触间隙
ENGAGE_DX = C.handle_radius + P.size[2] / 2
PRE_GAP = 1.0 * MM
#: 接近阶段在杆下方多少米飞行。低于杆底 79 mm 且高于托盘，路径全程无遮挡。
APPROACH_DZ = -0.060
HOME_DX = 0.150
#: 两块板的起始位在 X 上错开多少。**必须错开**：pinch 家族两块板的 y 只差
#: 几毫米，而板是 35×25 mm——起始位若相同，两块板生成时就是互相穿插的，
#: PhysX 把它们粘在一起，approach 走不到位，最后一起卡在横杆底下，
#: 接触力 74 N、接触部位 100% 判成 bar_bottom。
#: 这一条解释了 pinch 两个家族全部的失败，而 hook/asym（y 相距 ≥38 mm）
#: 从一开始就是 8/8。
HOME_STAGGER = 0.080
#: 升降时离杆面留多少余量。**这一条是必须的**：第一版让板贴着 1 mm 间隙
#: 直接升上去，结果升的过程中就蹭到了杆的下半圈——采出来的 region 会说
#: "开抽屉要压把手底面"。倾斜手面那一档更糟，直接楔进杆底，接触力冲到 75 N。
SAFE_DX = 0.024

PHASE_NAMES = ("approach", "establish", "manipulate", "release")
#: manipulate 从 250 降到 160：第一次 pilot 里抽屉 0.7 s 就拉完了，
#: 剩下 82% 的 manipulate 是空转，把所有统计量都稀释掉了。
PHASE_STEPS = (60, 70, 170, 60)
N_FRAMES = sum(PHASE_STEPS)
#: establish 内部三段：在安全距离上升到杆心 -> 横向贴到杆面 -> 轻轻建立接触。
#: **establish 只负责建立接触，不许把物体推动**——第一版让力在 establish 里
#: 就斜坡到满，实测有 episode 在 establish 结束时抽屉已经开了 138 mm，
#: manipulate 阶段反而无事可做，phase 标签与实际发生的事对不上
#: （`plan/02` §3.6 的 phase 是要进 Oracle Record 的，错了下游全错）。
RISE_STEPS = 40
CLOSE_STEPS = 55
#: establish 末尾的接触力只到满量的这个比例，够贴住、不足以推动抽屉
TOUCH_FRAC = 0.25
#: manipulate 开头把力从 TOUCH_FRAC 拉到满量用多少步
RAMP_STEPS = 20

GOAL_RANGE = (0.100, 0.160)
GOAL_TOL = 0.008
#: 名义物理区间（训练分布）与**留出**物理区间（`plan/03` §7 的 unseen_physics_test）。
#: 留出区间与名义区间不重叠，否则"没见过的物理"里装的其实是同分布数据——
#: records.py 里已经因为同一类错误修过一次 bug。
DAMPING_NOMINAL = (24.0, 36.0)
DAMPING_HELDOUT = ((16.0, 21.0), (39.0, 46.0))
#: 单帧法向力合计超过这个值就判为脏帧（P-27 的求解器尖峰）
MAX_VALID_FORCE = 200.0
MAX_PLATE_DIST = 1.2

FAMILIES = ("pinch_center", "pinch_offset", "single_finger", "hook_both", "asym_primary")


# ---------------------------------------------------------------- 场景


def _camera_cfg():
    """斜前上方看把手。位姿必须静态给定，运行时 set_world_poses_from_view 会挂死（P-24）。"""
    # 抽屉全程从 x=0.074 走到 0.234，取中段 0.15 作注视点，机位压低到接近
    # 把手高度——验收要看的是"板贴在杆的哪一侧"，俯视角度大了就看不出来。
    eye = (0.40, -0.33, 0.20)
    target = (0.15, 0.0, 0.095)
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/Cam", update_period=0.0,
        height=_a.height, width=_a.width, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=26.0, clipping_range=(0.02, 30.0)),
        offset=CameraCfg.OffsetCfg(pos=eye, rot=look_at_quat(eye, target),
                                   convention="opengl"),
    )


def _contact_cfg(idx: int):
    """规则 8：``filter_prim_paths_expr`` 非空 + ``max_contact_data_count_per_prim ≥ 1``，
    否则逐点数据全是空的（P-03）。filter 目标必须是刚体，抽屉是（规则 7 / P-17）。"""
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Plate{idx}",
        track_pose=True, track_contact_points=True,
        max_contact_data_count_per_prim=MAX_CONTACTS,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cabinet/Drawer"],
        update_period=0.0, history_length=0,
    )


@configclass
class SceneCfg(InteractiveSceneCfg):
    dome = AssetBaseCfg(prim_path="/World/dome",
                        spawn=sim_utils.DomeLightCfg(intensity=300.0,
                                                     color=(0.86, 0.89, 1.0)))
    sun = AssetBaseCfg(prim_path="/World/sun",
                       spawn=sim_utils.DistantLightCfg(intensity=1100.0, angle=5.0))
    cabinet = A.CABINET_CFG.replace(prim_path="{ENV_REGEX_NS}/Cabinet")
    plate0 = A.plate_cfg(0)
    plate1 = A.plate_cfg(1)
    contact0 = _contact_cfg(0)
    contact1 = _contact_cfg(1)
    if _a.video:
        cam = _camera_cfg()


# ---------------------------------------------------------------- 家族参数


def _u(rng, lo, hi, n):
    return torch.from_numpy(rng.uniform(lo, hi, size=n).astype(np.float32))


def sample_family(family: str, rng: np.random.Generator, n: int, device) -> dict:
    """给一批 env 采样该家族的**逐 env 参数**。

    五个家族共用同一个阶段机，只有这里返回的参数不同——这样一批 64 个 env
    可以混着跑不同家族，也保证家族之间除了参数没有别的隐藏差异。

    y 的两条硬约束（越界就穿模，两个都留了余量）：

    * 板宽 35 mm -> 同侧两块板的中心间距必须 > 35 mm，否则两块板自己撞在一起；
    * 支撑柱内缘 |y| = 54.5 mm -> 板心 |y| ≤ 54.5 − 17.5 = 37 mm。

    ⚠️ 没有"手面倾斜"这一档。试过，**几何上不成立**：平板贴圆柱只有在
    板面与柱轴平行时才是面接触，绕任何轴一转，最近的就变成板的一条棱，
    棱扎进杆里 -> 接触力冲到 75 N、把板压低 23 mm，采出来的 region 变成
    "压把手底面"。想要"不同手面姿态"，靠的是 ``z_off``（贴在杆背的高低不同）
    和 ``single_finger``（只用一块板），不是靠转手腕。

    Returns:
        ``side`` (N,2)：-1 = 板在杆**背面**（手指），+1 = 在杆**前面**（拇指）；
        ``y_off`` / ``z_off`` (N,2) 板心相对把手中心的偏移；
        ``force`` (N,2) 沿 X 推向杆子的力（正值幅度）；
        ``use`` (N,2) 这块板参不参与，0 表示全程停在起始位不动。
    """
    side = torch.zeros(n, 2)
    y_off = torch.zeros(n, 2)
    z_off = torch.zeros(n, 2)
    force = torch.zeros(n, 2)
    use = torch.ones(n, 2)
    sgn = torch.from_numpy(rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=n))
    # 贴在杆背的高低。±4 mm 对半径 11 mm 的杆是 ±21° 的接触角变化——
    # region 和 engage 方向都真的不同，又不会离开"杆背面"这个正确部位。
    z_off[:, 0] = _u(rng, -0.004, 0.004, n)
    z_off[:, 1] = _u(rng, -0.004, 0.004, n)

    if family == "pinch_center":
        # 中央对称夹持：手指在后、拇指在前，都在把手中段
        side[:, 0], side[:, 1] = -1.0, +1.0
        y_off[:, 0] = _u(rng, -0.006, 0.006, n)
        y_off[:, 1] = y_off[:, 0] + _u(rng, -0.004, 0.004, n)
        force[:, 0] = _u(rng, 5.5, 7.5, n)
        force[:, 1] = _u(rng, 1.2, 2.2, n)
    elif family == "pinch_offset":
        # 偏左/偏右夹持：同样的对捏，整体沿杆平移
        side[:, 0], side[:, 1] = -1.0, +1.0
        y_off[:, 0] = sgn * _u(rng, 0.014, 0.026, n)
        y_off[:, 1] = y_off[:, 0] + _u(rng, -0.004, 0.004, n)
        force[:, 0] = _u(rng, 5.5, 7.5, n)
        force[:, 1] = _u(rng, 1.2, 2.2, n)
    elif family == "single_finger":
        # **只用一块板**。另一块全程停在起始位，从不接触。
        # 这一档不只是多一种打开方式：`plan/02` §7 第 3 条要求"改变 source
        # 板数量后表示维度不变"，而那条泄漏检查需要真的存在只有一个接触体的
        # 示教才能验。没有这一档，那条检查就只能靠嘴说。
        side[:, 0], side[:, 1] = -1.0, -1.0
        y_off[:, 0] = sgn * _u(rng, 0.0, 0.012, n)
        y_off[:, 1] = 0.0
        force[:, 0] = _u(rng, 4.5, 6.5, n)
        use[:, 1] = 0.0
    elif family == "hook_both":
        # 双板都在杆背面，无拇指——接触拓扑与 pinch 明显不同
        side[:, :] = -1.0
        y_off[:, 0] = -_u(rng, 0.019, 0.034, n)
        y_off[:, 1] = +_u(rng, 0.019, 0.034, n)
        force[:, 0] = _u(rng, 2.0, 3.2, n)
        force[:, 1] = _u(rng, 2.0, 3.2, n)
    elif family == "asym_primary":
        # 一块承担主要拉力，另一块只是轻扶（`plan/03` §2.2 第 3 条）。
        # 站位与 hook_both 相同，差别在**力的分配**——同样的接触拓扑、
        # 明显不同的力分布，正是"同一份 envelope 该抹掉哪些差异"要检验的。
        side[:, :] = -1.0
        flip = (sgn > 0).float()
        main_y = -_u(rng, 0.019, 0.026, n)
        aux_y = +_u(rng, 0.028, 0.036, n)
        y_off[:, 0] = torch.where(flip > 0, main_y, aux_y)
        y_off[:, 1] = torch.where(flip > 0, aux_y, main_y)
        main_f = _u(rng, 3.4, 4.6, n)
        aux_f = _u(rng, 0.5, 1.2, n)
        force[:, 0] = torch.where(flip > 0, main_f, aux_f)
        force[:, 1] = torch.where(flip > 0, aux_f, main_f)
    else:
        raise ValueError(f"未知策略家族：{family}")

    return {k: v.to(device) for k, v in
            dict(side=side, y_off=y_off, z_off=z_off, force=force, use=use).items()}


def plate_quat(side: torch.Tensor) -> torch.Tensor:
    """板的目标姿态 (N, 4)。

    板局部 +Z 是工作面法向。基准姿态把 +Z 转到世界 +X（手指，面朝把手背面），
    拇指再绕世界 Z 转 180°。``yaw`` 是绕世界 Z 的手面偏角。
    """
    n, dev = side.shape[0], side.device
    ax_x = torch.tensor([1.0, 0.0, 0.0], device=dev).expand(n, 3)
    ax_y = torch.tensor([0.0, 1.0, 0.0], device=dev).expand(n, 3)
    ax_z = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
    ang = torch.full((n,), math.pi / 2, device=dev)
    # 先绕 X 转 90°（局部 +Y -> 世界 +Z），再绕 Z 转 90°（局部 +Z -> 世界 +X）
    base = quat_mul(quat_from_angle_axis(ang, ax_z), quat_from_angle_axis(ang, ax_x))
    flip = quat_from_angle_axis(torch.where(side > 0, math.pi, 0.0) * torch.ones(n, device=dev),
                                ax_z)
    return quat_mul(flip, base)


# ---------------------------------------------------------------- 采集


def _git_sha() -> str:
    """代码版本。服务器上没有 .git，由 ``sync.sh`` 写一个 ``.git_sha`` 带过来。"""
    here = Path(__file__).resolve().parent.parent / ".git_sha"
    if here.exists():
        sha = here.read_text(encoding="utf-8").strip()
        if sha:
            return sha
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


class Buffers:
    """一批 episode 的逐帧缓冲，全程留在 GPU，最后一次性拷回 CPU。"""

    def __init__(self, n: int, device):
        z = lambda *s: torch.zeros(n, *s, device=device)  # noqa: E731
        self.phase = torch.zeros(N_FRAMES, n, dtype=torch.int8, device=device)
        self.progress = torch.zeros(N_FRAMES, n, device=device)
        self.valid = torch.ones(N_FRAMES, n, dtype=torch.bool, device=device)
        self.open = torch.zeros(N_FRAMES, n, device=device)
        self.vel = torch.zeros(N_FRAMES, n, device=device)
        self.obj_pos = torch.zeros(N_FRAMES, n, 3, device=device)
        self.obj_quat = torch.zeros(N_FRAMES, n, 4, device=device)
        self.pos_obj = torch.zeros(N_FRAMES, 2, n, MAX_CONTACTS, 3, device=device)
        self.nrm_obj = torch.zeros(N_FRAMES, 2, n, MAX_CONTACTS, 3, device=device)
        self.fri_obj = torch.zeros(N_FRAMES, 2, n, MAX_CONTACTS, 3, device=device)
        self.fn = torch.zeros(N_FRAMES, 2, n, MAX_CONTACTS, device=device)
        self.sep = torch.zeros(N_FRAMES, 2, n, MAX_CONTACTS, device=device)
        self.cvalid = torch.zeros(N_FRAMES, 2, n, MAX_CONTACTS, dtype=torch.bool, device=device)
        self.mode = torch.zeros(N_FRAMES, 2, n, MAX_CONTACTS, dtype=torch.int8, device=device)
        self.part = torch.zeros(N_FRAMES, 2, n, MAX_CONTACTS, dtype=torch.int8, device=device)
        self.face = torch.zeros(N_FRAMES, 2, n, MAX_CONTACTS, dtype=torch.int8, device=device)
        self.inspan = torch.zeros(N_FRAMES, 2, n, MAX_CONTACTS, dtype=torch.bool, device=device)
        self.src_pose = torch.zeros(N_FRAMES, 2, n, 7, device=device)
        self.src_vel = torch.zeros(N_FRAMES, 2, n, 6, device=device)
        self.src_tgt = torch.zeros(N_FRAMES, 2, n, 7, device=device)
        self.src_cmd = torch.zeros(N_FRAMES, 2, n, 6, device=device)
        self.src_pos_w = torch.zeros(N_FRAMES, 2, n, MAX_CONTACTS, 3, device=device)
        self.net_force = torch.zeros(N_FRAMES, 2, n, device=device)
        self.fm_force = torch.zeros(N_FRAMES, 2, n, device=device)
        self.pd_sat = torch.zeros(N_FRAMES, 2, n, device=device)
        self.rot_err = torch.zeros(N_FRAMES, 2, n, device=device)
        self.foreign = torch.zeros(N_FRAMES, 2, device=device)
        self.dropped = torch.zeros(N_FRAMES, 2, device=device)
        self.raw_count = torch.zeros(N_FRAMES, 2, device=device)
        self.sub_hit = torch.zeros(N_FRAMES, 2, n, device=device)
        del z


def run_batch(scene, sim, camera, family_of_env, rng, device, batch_idx):
    """跑一批 episode（每个 env 一条），返回 (Buffers, 逐 env 元数据)。"""
    n = scene.cfg.num_envs
    cabinet: Articulation = scene["cabinet"]
    plates: list[RigidObject] = [scene["plate0"], scene["plate1"]]
    sensors = [scene["contact0"], scene["contact1"]]
    jid = cabinet.find_joints("DrawerJoint")[0][0]
    drawer_body = cabinet.body_names.index("Drawer")

    # --- 逐 env 参数：家族参数 + 目标开度 + 物理变体 ---
    params = {k: torch.zeros(n, 2, device=device)
              for k in ("side", "y_off", "z_off", "force", "use")}
    for fam in sorted(set(family_of_env)):
        idx = [i for i, f in enumerate(family_of_env) if f == fam]
        sub = sample_family(fam, rng, len(idx), device)
        sel = torch.tensor(idx, device=device, dtype=torch.long)
        for k in params:
            params[k][sel] = sub[k]
    side, y_off, z_off, force_mag, use = (
        params[k] for k in ("side", "y_off", "z_off", "force", "use"))
    goal = _u(rng, *GOAL_RANGE, n).to(device)

    is_variant = torch.from_numpy(
        (rng.random(n) < _a.physics_variant_frac).astype(np.bool_)).to(device)
    damping = _u(rng, *DAMPING_NOMINAL, n).to(device)
    lo, hi = DAMPING_HELDOUT[0] if batch_idx % 2 == 0 else DAMPING_HELDOUT[1]
    damping = torch.where(is_variant, _u(rng, lo, hi, n).to(device), damping)
    cabinet.write_joint_damping_to_sim(damping.unsqueeze(-1), joint_ids=[jid])
    applied = cabinet.data.joint_damping[:, jid]
    if (applied - damping).abs().max() > 1e-3:
        raise RuntimeError(
            "关节阻尼没写进去：写 {:.2f} 读回 {:.2f}。物理变体划分会变成假的，"
            "宁可在这里停住。".format(damping[0].item(), applied[0].item()))

    # --- 复位 ---
    zero1 = torch.zeros(n, 1, device=device)
    cabinet.write_joint_state_to_sim(zero1, zero1, joint_ids=[jid])
    cabinet.set_joint_effort_target(zero1, joint_ids=[jid])
    scene.update(DT)

    handle = cabinet.data.body_pos_w[:, drawer_body, :] + torch.tensor(
        [BAR_X, 0.0, BAR_Z], device=device)
    quats = [plate_quat(side[:, p]) for p in range(2)]
    # **两块板都从前方（+X）下方飞进来**：手指那块要伸进净空，只能从杆子
    # 底下穿过去再升上来；从 -X 侧进入意味着穿过柜体，物理上根本没有那条路。
    home_x_of = [handle[:, 0] + HOME_DX + i * HOME_STAGGER for i in range(2)]

    # 增益按板的**转动惯量**定，不能照搬钩杆的。板 35×25×3 mm、0.5 kg，
    # 绕长边的 I = m(b²+c²)/12 ≈ 2.6e-5 kg·m²，比钩杆小两个量级。
    # 显式积分的稳定条件 ω·dt < 2，ω = sqrt(kp_rot·m/I)：
    #   kp_rot=50（本仓库其他地方的默认值），150 Hz -> ω·dt = 5.4，直接发散
    #   kp_rot=6，300 Hz                        -> ω·dt = 1.2，稳
    # 姿态刚度 kp_rot·m = 3.0 N·m/rad，0.15 N·m 的接触扰动偏 2.9°。
    # 线性方向没有这个问题：f = kp·m·e 意味着 ω = sqrt(kp_pos)，与质量无关。
    #
    # kd_force 从 35 降到 15 是另一个实测教训：力控轴的速度阻尼 kd_force·m
    # 会**从接触力里扣掉**。35 时阻尼是 17.5 N·s/m，而板必须跟着抽屉以
    # 0.13 m/s 走 —— 白扣 2.3 N，比某些家族的指令力还大，板直接跟不上、
    # 脱开接触。降到 15（7.5 N·s/m）同样的速度只扣 1 N，同时保留足够阻尼
    # 抑制"贴上去-弹开-再贴"的高频断续（降到 8 时子步接触率反而更差）。
    pds = [FloatingPD(pl, kp_pos=3000.0, kd_pos=110.0, kp_rot=6.0, kd_rot=0.025,
                      max_force=120.0, max_torque=4.0, kd_force=15.0) for pl in plates]
    targets = []
    for p, plate in enumerate(plates):
        st = plate.data.default_root_state.clone()
        st[:, 0] = home_x_of[p]
        st[:, 1] = handle[:, 1] + y_off[:, p]
        st[:, 2] = handle[:, 2] + APPROACH_DZ
        st[:, 3:7] = quats[p]
        st[:, 7:] = 0.0
        plate.write_root_state_to_sim(st)
        targets.append(st[:, :3].clone())
    for _ in range(6):
        cabinet.set_joint_effort_target(zero1, joint_ids=[jid])
        for plate in plates:
            zw = torch.zeros(n, 1, 3, device=device)
            plate.set_external_force_and_torque(zw, zw, is_global=True)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(DT)

    buf = Buffers(n, device)
    engage = torch.zeros(n, 2, device=device)     # 力的斜坡系数 0~1
    prev_tgt = [t.clone() for t in targets]
    preview: list[np.ndarray] = []
    every = max(1, round(1.0 / (CONTROL_DT * _a.fps)))
    mu_eff = min(P.friction, C.friction)

    frame = 0
    for phase_id, phase_len in enumerate(PHASE_STEPS):
        for k in range(phase_len):
            handle = cabinet.data.body_pos_w[:, drawer_body, :] + torch.tensor(
                [BAR_X, 0.0, BAR_Z], device=device)
            opening = cabinet.data.joint_pos[:, jid]
            drawer_v = cabinet.data.joint_vel[:, jid]
            active = opening < (goal - GOAL_TOL)
            # 目标一律相对**当前**把手位置算。抽屉一旦拉开 130 mm，用复位时
            # 那个把手位置算出来的撤出点就落在柜体里了。
            engage_x = [handle[:, 0] + side[:, p] * (ENGAGE_DX + PRE_GAP) for p in range(2)]
            home_x_of = [handle[:, 0] + HOME_DX + i * HOME_STAGGER for i in range(2)]

            for p in range(2):
                tgt = targets[p]
                tgt[:, 1] = handle[:, 1] + y_off[:, p]
                safe_x = handle[:, 0] + side[:, p] * (ENGAGE_DX + SAFE_DX)
                home_x = home_x_of[p]
                on = use[:, p] > 0
                if phase_id == 0:
                    # approach：从前下方平飞到**安全距离**（不是贴着杆）。
                    # 两块板错开时间窗——手指那块要穿过拇指的最终位置，
                    # 同时进场必然对撞。
                    t0, t1 = (0.0, 0.55) if bool(side[0, p] < 0) else (0.55, 0.92)
                    if bool((side[:, p] < 0).all()) and p == 1:
                        t0, t1 = 0.30, 0.85
                    u = (k + 1) / phase_len
                    a = min(max((u - t0) / (t1 - t0), 0.0), 1.0)
                    tgt[:, 0] = home_x + (safe_x - home_x) * a
                    tgt[:, 2] = handle[:, 2] + APPROACH_DZ
                elif phase_id == 1:
                    # establish：安全距离上升到杆心 -> 横向贴上去 -> 轻接触
                    a = min((k + 1) / RISE_STEPS, 1.0)
                    b = min(max((k + 1 - RISE_STEPS)
                                / max(CLOSE_STEPS - RISE_STEPS, 1), 0.0), 1.0)
                    tgt[:, 0] = safe_x + (engage_x[p] - safe_x) * b
                    tgt[:, 2] = handle[:, 2] + z_off[:, p] + APPROACH_DZ * (1.0 - a)
                    engage[:, p] = TOUCH_FRAC * min(
                        max((k + 1 - CLOSE_STEPS) / max(phase_len - CLOSE_STEPS, 1), 0.0), 1.0)
                elif phase_id == 2:
                    # manipulate：把力拉到满量并保持；到量后卸力，不继续推
                    tgt[:, 0] = plates[p].data.root_pos_w[:, 0]
                    tgt[:, 2] = handle[:, 2] + z_off[:, p]
                    full = TOUCH_FRAC + (1.0 - TOUCH_FRAC) * min((k + 1) / RAMP_STEPS, 1.0)
                    engage[:, p] = torch.where(
                        active, torch.full_like(engage[:, p], full),
                        (engage[:, p] - 0.08).clamp(min=0.0))
                else:
                    # release：先横向退到安全距离，再下沉，最后撤出
                    engage[:, p] = 0.0
                    a = min((k + 1) / 12.0, 1.0)
                    b = min(max((k + 1 - 12) / 22.0, 0.0), 1.0)
                    c_ = min(max((k + 1 - 34) / max(phase_len - 34, 1), 0.0), 1.0)
                    x_out = safe_x + (home_x - safe_x) * c_
                    tgt[:, 0] = engage_x[p] + (x_out - engage_x[p]) * a
                    tgt[:, 2] = handle[:, 2] + APPROACH_DZ * b

                # 不参与的板全程停在起始位：single_finger 需要一块**真的**
                # 从头到尾不接触的板，而不是"力设成 0 但还是杵在把手边上"。
                tgt[:, 0] = torch.where(on, tgt[:, 0], home_x)
                tgt[:, 2] = torch.where(on, tgt[:, 2], handle[:, 2] + APPROACH_DZ)
                engage[:, p] = torch.where(on, engage[:, p],
                                           torch.zeros_like(engage[:, p]))
                buf.src_cmd[frame, p, :, :3] = tgt - prev_tgt[p]
                prev_tgt[p] = tgt.clone()

            # 每个**物理子步**都取一次接触，保留法向力最大的那一子步。
            #
            # 只在控制步末尾取一次是不够的：pilot 里有 episode 抽屉明明被拉开
            # 了 97 mm，却一个接触点都没记到——接触是真实存在的，只是恰好不在
            # 被采样的那个子步上。50 Hz 的记录频率是 `plan/03` §5 定的，不能改；
            # 能改的是"这一帧拿哪个子步代表"。取力最大的子步，而不是最后一个。
            best_fn = torch.zeros(2, n, device=device)
            best: list[dict | None] = [None, None]
            for _sub in range(DECIMATION):
                # P-21：力矩指令会一直保持，每步都要显式清零，否则抽屉被持续驱动
                cabinet.set_joint_effort_target(zero1, joint_ids=[jid])
                for p, plate in enumerate(plates):
                    ff = torch.zeros(n, 3, device=device)
                    # side=-1（手指，在杆背面）要往 +X 推；side=+1（拇指）往 -X 推
                    # 力控轴的速度阻尼是相对**世界**的，而板必须跟着抽屉走。
                    # 把 kd_force·m·v_drawer 前馈回去，等价于"阻尼相对被操作
                    # 物体"，于是接触力就等于指令力，与抽屉开多快无关。
                    #
                    # 不补的后果按板的角色不同而不同，两个都致命：
                    #   手指（推 +X）：白扣 kd·m·v，指令力小的家族直接跟不上；
                    #   拇指（推 −X）：指令方向与运动方向相反，一接触就分离，
                    #     之后只剩阻尼、以 0.27 m/s 飘进柜体撞面板，
                    #     实测把 pinch 家族的平均接触力顶到 30 N。
                    ff[:, 0] = (-side[:, p] * force_mag[:, p] * engage[:, p]
                                + pds[p].kd_force * pds[p].mass * drawer_v)
                    mask = torch.zeros(n, 3, dtype=torch.bool, device=device)
                    mask[:, 0] = engage[:, p] > 0
                    f, tq = pds[p].compute(targets[p], quats[p], ff_force=ff, force_mask=mask)
                    buf.pd_sat[frame, p] += (
                        f.squeeze(1).abs() >= pds[p].max_force - 1e-3).any(dim=-1).float()
                    plate.set_external_force_and_torque(f, tq, is_global=True)
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(DT)

                for p, plate in enumerate(plates):
                    cp = extract_contact_points_padded(
                        sensors[p], DT, body_pos_w=plate.data.root_pos_w,
                        max_points=MAX_CONTACTS, own_radius=0.06)
                    tot = cp["normal_forces"].abs().sum(dim=1)
                    buf.sub_hit[frame, p] += (tot > 0.05).float()
                    buf.foreign[frame, p] += cp["foreign"]
                    buf.dropped[frame, p] += cp["dropped"]
                    if best[p] is None:
                        best[p] = cp
                        best_fn[p] = tot
                    else:
                        take = (tot > best_fn[p]).unsqueeze(-1)
                        for key in ("normal_forces", "separations"):
                            best[p][key] = torch.where(take, cp[key], best[p][key])
                        for key in ("positions", "normals", "friction_forces"):
                            best[p][key] = torch.where(
                                take.unsqueeze(-1), cp[key], best[p][key])
                        best[p]["valid"] = torch.where(take, cp["valid"], best[p]["valid"])
                        best_fn[p] = torch.maximum(best_fn[p], tot)

            # ---------------- 记录 ----------------
            opening = cabinet.data.joint_pos[:, jid]
            obj_pos = cabinet.data.body_pos_w[:, drawer_body, :]
            obj_quat = cabinet.data.body_quat_w[:, drawer_body, :]
            buf.phase[frame] = phase_id
            buf.progress[frame] = (opening / goal.clamp_min(1e-6)).clamp(0.0, 1.0)
            buf.open[frame] = opening
            buf.vel[frame] = cabinet.data.joint_vel[:, jid]
            buf.obj_pos[frame] = obj_pos
            buf.obj_quat[frame] = obj_quat

            ok = torch.ones(n, dtype=torch.bool, device=device)
            for p, plate in enumerate(plates):
                cp = best[p]
                buf.raw_count[frame, p] = cp["valid"].sum().float()
                pos_l = to_local(cp["positions"], obj_pos, obj_quat)
                buf.pos_obj[frame, p] = pos_l * cp["valid"].unsqueeze(-1)
                # 法向和摩擦力是**矢量**，只旋转不平移
                buf.nrm_obj[frame, p] = rotate_inverse(obj_quat, cp["normals"])
                buf.fri_obj[frame, p] = rotate_inverse(obj_quat, cp["friction_forces"])
                buf.fn[frame, p] = cp["normal_forces"]
                buf.sep[frame, p] = cp["separations"]
                buf.cvalid[frame, p] = cp["valid"]
                buf.src_pos_w[frame, p] = cp["positions"]
                buf.mode[frame, p] = classify_contact_mode_padded(
                    cp["normal_forces"], cp["friction_forces"], cp["separations"],
                    cp["valid"], mu=mu_eff)
                buf.part[frame, p] = torch.where(
                    cp["valid"],
                    classify_drawer_local(
                        pos_l, bar_x=BAR_X, bar_z=BAR_Z, bar_radius=C.handle_radius,
                        bar_half_len=C.handle_bar_len / 2,
                        post_half_spacing=C.post_spacing / 2, post_radius=C.post_radius,
                        panel_t=C.panel_t),
                    torch.full_like(buf.part[frame, p], -1))
                nrm_plate = rotate_inverse(plate.data.root_quat_w, cp["normals"])
                buf.face[frame, p] = torch.where(
                    cp["valid"], classify_plate_face(nrm_plate),
                    torch.full_like(buf.face[frame, p], -1))
                buf.inspan[frame, p] = cp["valid"] & bar_span_fraction(
                    pos_l, C.post_spacing / 2, C.post_radius)

                buf.src_pose[frame, p] = torch.cat(
                    [plate.data.root_pos_w, plate.data.root_quat_w], dim=-1)
                buf.src_vel[frame, p] = torch.cat(
                    [plate.data.root_lin_vel_w, plate.data.root_ang_vel_w], dim=-1)
                buf.src_tgt[frame, p, :, :3] = targets[p]
                buf.src_tgt[frame, p, :, 3:] = quats[p]
                buf.net_force[frame, p] = sensors[p].data.net_forces_w[:, 0].norm(dim=-1)
                # Isaac Lab 自己按 env 组织的**过滤后**接触力。逐点 buffer 漏报时
                # 它仍然是对的，所以拿它当逐点提取的对照：两者应当同时为零、
                # 同时非零。不一致就说明逐点通道又在静默丢数据。
                buf.fm_force[frame, p] = sensors[p].data.force_matrix_w[:, 0, 0].norm(dim=-1)
                # 姿态跟踪误差。板的转动惯量只有 3.9e-5，PD 增益被数值稳定性
                # 卡得很低（见上面的增益注释），所以必须实测它到底稳不稳，
                # 不能假设"设了 PD 姿态就守得住"。
                dq = (plate.data.root_quat_w * quats[p]).sum(dim=-1).abs().clamp(max=1.0)
                buf.rot_err[frame, p] = 2.0 * torch.acos(dq)
                ok &= torch.isfinite(plate.data.root_pos_w).all(dim=-1)
                ok &= (plate.data.root_pos_w - scene.env_origins).norm(dim=-1) < MAX_PLATE_DIST

            total_fn = buf.fn[frame].abs().sum(dim=(0, 2))
            buf.valid[frame] = ok & torch.isfinite(opening) & (total_fn <= MAX_VALID_FORCE)

            if camera is not None and frame % every == 0:
                sim.render()
                camera.update(CONTROL_DT)
                preview.append(camera.data.output["rgb"][0, ..., :3]
                               .detach().cpu().numpy().astype(np.uint8))
            frame += 1

    n_far, n_cut = float(buf.foreign.sum()), float(buf.dropped.sum())
    n_kept = float(buf.raw_count.sum())
    print(f"  接触点归属：保留 {n_kept:.0f}，离所有板都远而丢弃 {n_far:.0f}，"
          f"超过每 env 上限截掉 {n_cut:.0f}", flush=True)
    if n_cut > 0.01 * max(n_kept, 1.0):
        raise RuntimeError(
            f"{n_cut:.0f} 个接触点因超过 MAX_CONTACTS={MAX_CONTACTS} 被截掉"
            f"（保留 {n_kept:.0f}）。截断是静默的（P-03），调大上限再跑。")
    meta = dict(goal=goal, damping=damping, is_variant=is_variant,
                side=side, y_off=y_off, z_off=z_off, force=force_mag, use=use)
    return buf, meta, preview


# ---------------------------------------------------------------- 统计与落盘


def episode_diagnostics(buf: Buffers, e: int) -> dict[str, Any]:
    """一条 episode 的验收数字。**接触部位分布是这里的重点**（D-34）。"""
    fn = buf.fn[:, :, e, :].abs()                      # (T, 2, K)
    valid = buf.cvalid[:, :, e, :]
    part = buf.part[:, :, e, :]
    face = buf.face[:, :, e, :]
    total = fn.sum().item()

    def share(codes, table):
        out = {}
        for i, name in enumerate(table):
            out[name] = float((fn * (codes == i)).sum().item() / total) if total > 0 else 0.0
        return out

    vel = buf.vel[:, e]
    moving = vel.abs() > 0.005
    in_contact = fn.sum(dim=(1, 2)) > 0.05
    n_moving = int(moving.sum().item())
    # 只统计 manipulate 阶段：approach/release 本来就没有接触
    manip = buf.phase[:, e] == 2
    m_moving = moving & manip

    # 接触力对抽屉的合力沿 +X（传感器报的是作用在板上的力，取负号得作用在抽屉上的）
    f_on_drawer_x = -(buf.fn[:, :, e, :].unsqueeze(-1) * buf.nrm_obj[:, :, e, :, :]
                      + buf.fri_obj[:, :, e, :, :]).sum(dim=(1, 2))[:, 0]
    m_pushing = m_moving & in_contact

    return {
        "max_open_mm": float(buf.open[:, e].max().item() * 1000.0),
        "final_open_mm": float(buf.open[-1, e].item() * 1000.0),
        "contact_frame_fraction": float(in_contact.float().mean().item()),
        "manip_no_contact_fraction": float(
            ((m_moving & ~in_contact).sum() / max(int(m_moving.sum().item()), 1)).item()),
        "moving_no_contact_fraction": float(
            ((moving & ~in_contact).sum() / max(n_moving, 1)).item()),
        # 只在**有接触**的帧上算方向对不对。把"没接触"也算成"方向不对"，
        # 等于把同一件事罚两次，而且掩盖了真正想问的问题：接触上了的时候，
        # 力是不是真的在把抽屉往外拉。脱手多少由 manip_no_contact_fraction 单独报。
        "pull_direction_ok_fraction": float(
            ((f_on_drawer_x > 0.05) & m_pushing).sum().item()
            / max(int(m_pushing.sum().item()), 1)),
        "manip_contact_frames": int(m_pushing.sum().item()),
        "mean_contact_force_N": float(fn.sum(dim=(1, 2)).mean().item()),
        "max_contact_force_N": float(fn.sum(dim=(1, 2)).max().item()),
        "peak_point_force_N": float(fn.max().item()),
        "mean_contacts_per_frame": float(valid.float().sum(dim=(1, 2)).mean().item()),
        "substep_contact_fraction": float(
            (buf.sub_hit[:, :, e][manip].mean() / DECIMATION).item()),
        "invalid_frame_fraction": float(1.0 - buf.valid[:, e].float().mean().item()),
        "pd_saturation_fraction": float((buf.pd_sat[:, :, e] > 0).float().mean().item()),
        "mean_orientation_error_deg": float(
            torch.rad2deg(buf.rot_err[:, :, e]).mean().item()),
        "max_orientation_error_deg": float(
            torch.rad2deg(buf.rot_err[:, :, e]).max().item()),
        "mean_pull_force_x_N": float(f_on_drawer_x[m_moving].mean().item())
        if int(m_moving.sum().item()) > 0 else 0.0,
        # 逐点通道与 Isaac Lab 的 force_matrix 对照：两者判"有没有接触"应当一致
        "matrix_contact_fraction": float(
            ((buf.fm_force[:, :, e].sum(dim=1) > 0.05) & manip).float().sum().item()
            / max(int(manip.sum().item()), 1)),
        "points_contact_fraction": float(
            (in_contact & manip).float().sum().item() / max(int(manip.sum().item()), 1)),
        "unfiltered_force_ratio": float(
            (buf.net_force[:, :, e].sum() / max(fn.sum().item(), 1e-6)).item()),
        "drawer_part_force_share": share(part, DRAWER_PARTS),
        "plate_face_force_share": share(face, PLATE_PARTS),
        "bar_inside_posts_share": float(
            (fn * buf.inspan[:, :, e, :]).sum().item() / total) if total > 0 else 0.0,
        # 四种 mode 都报（`plan/02` §3.4）。只报 stick 比例会掩盖"接触其实是
        # 断续的"——separating 占比高就是那个信号。
        "mode_share": {
            name: (float((fn * (buf.mode[:, :, e, :] == i)).sum().item() / total)
                   if total > 0 else 0.0)
            for i, name in enumerate(("no_contact", "sticking", "sliding", "separating"))
        },
    }


def to_arrays(buf: Buffers, e: int) -> dict[str, np.ndarray]:
    """抽出一条 episode 的逐帧数组。

    前缀分层见 ``it.records``：``source/*`` 只用于事后追查，永远不进模型输入。
    **世界系的东西一律进 source/**——`plan/02` §1 要求表示是物体系的，
    把世界坐标放在 ``contact/`` 下等于给下游留了一条抄近路的口子。

    ``contact/plateK/*`` 是**逐采集体**的原始记录，不是最终表示：
    `plan/02` §7 第 3 条要求"改变 source 板数量后表示维度不变"，
    那是 S4 构造 Oracle Interaction Record 时**合并**两块板的接触集合来满足的，
    不是这一层的事。S4 不合并就会把"两块板"这个 source 特有的结构漏下去。
    """
    cpu = lambda t: t.detach().cpu().numpy()  # noqa: E731
    out: dict[str, np.ndarray] = {
        "phase": cpu(buf.phase[:, e]),
        "progress": cpu(buf.progress[:, e]).astype(np.float32),
        "valid_frame": cpu(buf.valid[:, e]),
        "object/drawer_joint_pos": cpu(buf.open[:, e]).astype(np.float32)[:, None],
        "object/drawer_joint_vel": cpu(buf.vel[:, e]).astype(np.float32)[:, None],
        "source/drawer_pos_w": cpu(buf.obj_pos[:, e]).astype(np.float32),
        "source/drawer_quat_w": cpu(buf.obj_quat[:, e]).astype(np.float32),
    }
    for p in range(2):
        k = f"contact/plate{p}"
        out[f"{k}/pos_obj"] = cpu(buf.pos_obj[:, p, e]).astype(np.float32)
        out[f"{k}/normal_obj"] = cpu(buf.nrm_obj[:, p, e]).astype(np.float32)
        out[f"{k}/friction_obj"] = cpu(buf.fri_obj[:, p, e]).astype(np.float32)
        out[f"{k}/normal_force"] = cpu(buf.fn[:, p, e]).astype(np.float32)
        out[f"{k}/separation"] = cpu(buf.sep[:, p, e]).astype(np.float32)
        out[f"{k}/valid"] = cpu(buf.cvalid[:, p, e])
        out[f"{k}/mode"] = cpu(buf.mode[:, p, e])
        out[f"{k}/drawer_part"] = cpu(buf.part[:, p, e])
        s = f"source/plate{p}"
        out[f"{s}/root_pose"] = cpu(buf.src_pose[:, p, e]).astype(np.float32)
        out[f"{s}/root_velocity"] = cpu(buf.src_vel[:, p, e]).astype(np.float32)
        out[f"{s}/target_pose"] = cpu(buf.src_tgt[:, p, e]).astype(np.float32)
        out[f"{s}/cmd_delta"] = cpu(buf.src_cmd[:, p, e]).astype(np.float32)
        out[f"{s}/contact_pos_w"] = cpu(buf.src_pos_w[:, p, e]).astype(np.float32)
        out[f"{s}/contact_plate_face"] = cpu(buf.face[:, p, e])
    return out


#: episode 判成功的条件。前两条是任务本身，后两条是**做法**——
#: P-28 那种"捅一下让抽屉自己滑"在数字上完全合格，只有第 3 条能挡住。
SUCCESS_RULES = {
    "max_open_mm": ("≥ 目标开度", None),
    "manip_no_contact_fraction": ("≤ 0.35", 0.35),
    "pull_direction_ok_fraction": ("≥ 0.85（仅在有接触的帧上算）", 0.85),
    "invalid_frame_fraction": ("≤ 0.10", 0.10),
}


def judge(diag: dict, goal_m: float) -> tuple[bool, list[str]]:
    fails = []
    if diag["max_open_mm"] < goal_m * 1000.0 - GOAL_TOL * 1000.0:
        fails.append("开度不足")
    if diag["manip_no_contact_fraction"] > 0.35:
        fails.append("操作阶段脱手过多")
    if diag["manip_contact_frames"] < 15:
        fails.append("操作阶段几乎没接触")
    elif diag["pull_direction_ok_fraction"] < 0.85:
        fails.append("受力方向不对")
    if diag["invalid_frame_fraction"] > 0.10:
        fails.append("脏帧过多")
    return (not fails), fails


def write_preview(path: Path, frames: list[np.ndarray]) -> None:
    if not frames:
        return
    import imageio.v2 as iio
    path.parent.mkdir(parents=True, exist_ok=True)
    with iio.get_writer(path, fps=_a.fps, codec="libx264", quality=8,
                        macro_block_size=1) as w:
        for fr in frames:
            w.append_data(fr)


def main() -> int:
    out_dir = Path(_a.out)
    (out_dir / "episodes").mkdir(parents=True, exist_ok=True)
    families = FAMILIES if _a.family == "all" else tuple(_a.family.split(","))
    for f in families:
        if f not in FAMILIES:
            raise SystemExit(f"未知家族 {f}，可选：{FAMILIES}")

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=DT, device="cuda:0",
        physx=sim_utils.PhysxCfg(gpu_max_rigid_contact_count=2 ** 22,
                                 gpu_max_rigid_patch_count=2 ** 20)))
    scene = InteractiveScene(SceneCfg(num_envs=_a.envs, env_spacing=2.2,
                                      replicate_physics=True))
    sim.reset()
    camera: Camera | None = scene["cam"] if _a.video else None
    device = sim.device
    rng = np.random.default_rng(_a.seed)
    git_sha = _git_sha()

    records: list[tuple[str, EpisodeRecord]] = []
    for b in range(_a.batches):
        fam_of_env = [families[(b * _a.envs + i) % len(families)] for i in range(_a.envs)]
        print(f"\n=== batch {b + 1}/{_a.batches} · {_a.envs} env ===", flush=True)
        buf, m, preview = run_batch(scene, sim, camera, fam_of_env, rng, device, b)
        if camera is not None:
            write_preview(out_dir / "videos" / f"drawer_{fam_of_env[0]}_b{b}.mp4", preview)
        for e in range(_a.envs):
            diag = episode_diagnostics(buf, e)
            ok, fails = judge(diag, m["goal"][e].item())
            fam = fam_of_env[e]
            variant = bool(m["is_variant"][e].item())
            eid = f"drawer-{fam}-b{b:02d}e{e:03d}"
            meta = {
                "schema_version": SCHEMA_VERSION,
                "episode_id": eid,
                "task": "drawer",
                "source_embodiment": "two_dynamic_plates",
                "strategy_family": fam,
                "strategy_variant": f"b{b:02d}e{e:03d}",
                "physics_variant": "heldout_damping" if variant else "nominal",
                "success": ok,
                "failure_reasons": fails,
                "seed": int(_a.seed + b * _a.envs + e),
                "control_hz": 1.0 / CONTROL_DT,
                "physics_hz": 1.0 / DT,
                "phase_names": list(PHASE_NAMES),
                "phase_steps": list(PHASE_STEPS),
                "goal_open_m": float(m["goal"][e].item()),
                "generator_git_sha": git_sha,
                "physics": {
                    "drawer_joint_damping": float(m["damping"][e].item()),
                    "drawer_mass_kg": C.drawer_mass,
                    "drawer_friction": C.friction,
                    "plate_mass_kg": P.mass,
                    "plate_friction": P.friction,
                },
                "source_params": {
                    "side": [float(x) for x in m["side"][e].tolist()],
                    "y_offset_mm": [float(x * 1000) for x in m["y_off"][e].tolist()],
                    "force_N": [float(x) for x in m["force"][e].tolist()],
                    "z_offset_mm": [float(x * 1000) for x in m["z_off"][e].tolist()],
                    "plate_used": [bool(x) for x in m["use"][e].tolist()],
                },
                "diagnostics": diag,
            }
            rec = EpisodeRecord(meta=meta, arrays=to_arrays(buf, e))
            path = save_episode(rec, out_dir / "episodes" / f"{eid}.npz")
            records.append((path, rec))
        del buf

    entries = [r.to_manifest_entry("x") for _, r in records]
    splits = split_episode_entries(entries, seed=_a.seed,
                                   holdout_strategy_family=_a.holdout_family)
    write_manifest(records, out_dir / "manifest.json", dataset_name="s3_drawer_source",
                   generator_git_sha=git_sha, splits=splits,
                   extra={"task": "drawer", "families": list(families)})
    report(out_dir, records, splits)
    return 0


def report(out_dir: Path, records, splits) -> None:
    """人可读的验收表。接触部位一栏就是"操作方式对不对"的证据。"""
    lines: list[str] = []
    add = lines.append
    metas = [r.meta for _, r in records]
    ok = [m for m in metas if m["success"]]
    add(f"S3 抽屉双板 source · {len(metas)} episode，成功 {len(ok)} "
        f"({100.0 * len(ok) / max(len(metas), 1):.1f}%)")
    add("")
    add(f"{'家族':<16}{'条数':>5}{'成功':>6}{'开度mm':>9}{'脱手%':>8}"
        f"{'受力向%':>9}{'杆背面%':>9}{'工作面%':>9}{'柱间%':>8}{'力N':>7}")
    for fam in sorted({m["strategy_family"] for m in metas}):
        sub = [m for m in metas if m["strategy_family"] == fam]
        s = [m for m in sub if m["success"]]
        g = lambda key: np.mean([m["diagnostics"][key] for m in sub])  # noqa: E731
        add(f"{fam:<16}{len(sub):>5}{len(s):>6}"
            f"{g('max_open_mm'):>9.1f}{100 * g('manip_no_contact_fraction'):>8.1f}"
            f"{100 * g('pull_direction_ok_fraction'):>9.1f}"
            f"{100 * np.mean([m['diagnostics']['drawer_part_force_share']['bar_back'] for m in sub]):>9.1f}"
            f"{100 * np.mean([m['diagnostics']['plate_face_force_share']['work_face'] for m in sub]):>9.1f}"
            f"{100 * g('bar_inside_posts_share'):>8.1f}{g('mean_contact_force_N'):>7.2f}")
    add("")
    add("接触落在抽屉的哪个部位（按法向力加权，全体 episode 平均）")
    for name in DRAWER_PARTS:
        v = np.mean([m["diagnostics"]["drawer_part_force_share"][name] for m in metas])
        add(f"  {name:<14}{100 * v:>7.2f}%")
    add("")
    add("接触落在板的哪个面")
    for name in PLATE_PARTS:
        v = np.mean([m["diagnostics"]["plate_face_force_share"][name] for m in metas])
        add(f"  {name:<14}{100 * v:>7.2f}%")
    add("")
    add("其他")
    for key in ("peak_point_force_N", "invalid_frame_fraction", "pd_saturation_fraction",
                "unfiltered_force_ratio", "mean_contacts_per_frame",
                "mean_orientation_error_deg", "max_orientation_error_deg",
                "mean_pull_force_x_N", "substep_contact_fraction",
                "matrix_contact_fraction", "points_contact_fraction"):
        add(f"  {key:<28}{np.mean([m['diagnostics'][key] for m in metas]):>9.4f}")
    add("")
    add("接触模式分布（按法向力加权，`plan/02` §3.4）")
    for name in ("no_contact", "sticking", "sliding", "separating"):
        v = np.mean([m["diagnostics"]["mode_share"][name] for m in metas])
        add(f"  {name:<14}{100 * v:>7.2f}%")
    add("")
    add("划分（按 episode，P-10）")
    for name, ids in splits.items():
        add(f"  {name:<24}{len(ids):>5}")
    bad = [m for m in metas if not m["success"]]
    if bad:
        add("")
        add("失败原因分布")
        reasons: dict[str, int] = {}
        for m in bad:
            for r in m["failure_reasons"]:
                reasons[r] = reasons.get(r, 0) + 1
        for r, c in sorted(reasons.items(), key=lambda kv: -kv[1]):
            add(f"  {r:<24}{c:>5}")
    text = "\n".join(lines) + "\n"
    (out_dir / "report.txt").write_text(text, encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps({"num_episodes": len(metas), "num_success": len(ok),
                    "splits": {k: len(v) for k, v in splits.items()}},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(text, flush=True)


try:
    code = main()
except Exception:
    import traceback
    traceback.print_exc()
    code = 1
sys.stdout.flush()
# P-19：SimulationApp.close() 在本环境会挂起，进程变僵尸占显存
os._exit(code)
