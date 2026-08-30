"""S3 · 擦拭任务的双板 source 采集（主任务）。

`plan/01` §5.1：擦拭是**清除平面目标区域的污渍**，允许两种实现——

* **(a) 持工具擦**：双板夹住黑板擦，用它的底垫擦；
* **(b) 自身接触面直擦**：不用工具，双板直接用板面擦。

两者产生**同一份 Functional Envelope**，因为 `plan/02` §1.1 把擦拭的被操作
物体定在**平面**上。这是 D-12 的核心，也是 `plan/02` §7 第 8 条那条泄漏检查
（envelope 不得包含工具的存在与否）的数据来源——没有 (b) 就没法验。

## 记什么、不记什么

`plan/02` §3.1 经 **D-42** 修订后，擦拭的 Object Effect **只有平面上的
dirt 状态变化**。初版还写了"擦拭接触体相对平面的未来 6D 位姿增量"，
那一条已删除：接触体是黑板擦或执行器自己的面，**都是 source 侧的东西**，
写进 effect 等于让 envelope 携带"用什么姿态、沿什么路径移动你的手"，
那是动作层面的迁移，而垫头杆那一格（一个不能抓握的杆子执行同一份 envelope）
根本无从执行。工具位姿只进 ``source/*`` 追查字段。

## 接触体是"任一接触体"

`plan/01` §5.3 的判据已从"黑板擦底面"泛化为**任一接触体**。所以三个物体
（黑板擦、板 0、板 1）各挂一个接触传感器、**都过滤到平面**，
板侧的接触集合就是它们的并集——实现 (a) 时是黑板擦在接触，
实现 (b) 时是两块板在接触，记录格式完全一样。

## 用法

    ./tools/run_remote.sh "PYTHONPATH=src /isaac-sim/python.sh \\
        tools/s3_source_wipe.py --envs 60 --batches 24 --out /tmp/s3_wipe" wipe
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

_ap = argparse.ArgumentParser(description="S3 擦拭双板 source 采集")
_ap.add_argument("--envs", type=int, default=60)
_ap.add_argument("--batches", type=int, default=24)
_ap.add_argument("--out", default="/tmp/s3_wipe")
_ap.add_argument("--seed", type=int, default=20260829)
_ap.add_argument("--family", default="all")
_ap.add_argument("--holdout-implementation", default="direct",
                 help="留出哪一种实现（with_tool / direct / 空字符串表示不留）")
_ap.add_argument("--holdout-family", default="tool_tilt",
                 help="划进 unseen_strategy_test 的家族（`plan/03` §7）")
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
from isaaclab.assets import AssetBaseCfg, RigidObject  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg, ContactSensorCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from it import assets as A  # noqa: E402
from it import build_assets as B  # noqa: E402
from it.contact_attrib import (  # noqa: E402
    PLATE_PARTS,
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
PHASE_STEPS = (40, 50, 260, 40)
N_FRAMES = sum(PHASE_STEPS)

_E = B.EraserCfg()
_P = B.PlateCfg()
BOARD = (600 * MM, 500 * MM, 20 * MM)
BOARD_MU = 0.35

#: 目标区域（平面局部系，XY，单位 m）与网格。10 mm 一格。
#: X 宽 190 = 接触体宽度（垫 80 / 双板并排 95）+ 道心漂移 116，留一点余量。
#: 区域比一次扫掠能覆盖的还宽，就成了做不到的任务，不是难任务。
REGION = (0.190, 0.290)
CELL = 0.010
GRID = (int(round(REGION[0] / CELL)), int(round(REGION[1] / CELL)))   # (21, 30)
N_PASS = 3
#: 清除判据（`plan/01` §5.3）：法向力在区间内、有切向相对运动、扫过该格
CLEAR_RADIUS = 6 * MM
WIPE_LO, WIPE_HI = A.WIPE_FORCE_RANGE          # (3.0, 8.0) N
SLIDE_V_MIN = 0.008                            # m/s

#: **三个轴全部走位置控制**，靠位置干涉产生夹持力和下压力。
#:
#: 一开始 Z 走力控，结果**同一个指令力在不同家族上量出来差一个数量级**
#: （2.4 N 的指令，tool_center 量到 7.5 N、tool_light_fast 量到 0.67 N）。
#: 原因是力从板经"摩擦抓持"传到黑板擦再传到平面，这条链是软的，
#: 而力控轴没有位置参考，板会在 260 帧里慢慢漂走。
#: 位置干涉则把高度钉死：下压力 = 2 × kp_pos × m × δ = 3000 δ，可预测。
#: 3.0 mm × 1500 N/m ≈ 4.5 N 夹持力，摩擦 μ=0.9 -> 单板 4 N、双板 8 N，
#: 相对 0.35×5 ≈ 1.8 N 的擦拭阻力有 4 倍余量。
#: 一度以为滑脱是夹持不够，加到 4.5 mm 反而更糟（下压力被顶到 8 N）——
#: 真正的原因是 release 阶段的目标阶跃，见下面 u_ph 的注释。
GRIP_INTERF = 3.0 * MM
#: 下落时夹持轴上额外张开的量。**必须先张开再夹拢**，理由见 `grip_targets`。
GRIP_OPEN = 6.0 * MM
#: 下压干涉与力的换算：单块板的位置刚度 kp_pos × m = 1500 N/m。
#:
#: **持工具与直擦的换算不同**：持工具时两块板共同把**一个**黑板擦压向平面，
#: 接触体只有一个，press 是它的总压力 -> δ = press / (2 × 1500)；
#: 直擦时**每块板各自是一个接触体**，press 是每块板的压力 -> δ = press / 1500。
#: `plan/01` §5.3 的清除判据是**逐接触体**判法向力在 3~8 N 内，
#: 按总压力去分摊会让每块板只有 1.6 N，一格都擦不掉——实测清除率卡在 30%。
PLATE_STIFF = 1500.0
STANDOFF = 0.12
MAX_VALID_FORCE = 200.0
MAX_PLATE_DIST = 1.4

FAMILIES = ("tool_center", "tool_offset", "tool_tilt", "tool_heavy_slow",
            "tool_light_fast", "direct_wipe")


def _camera_cfg():
    eye = (0.34, -0.42, 0.30)
    at = (0.0, 0.0, 0.02)
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/Cam", update_period=0.0,
        height=_a.height, width=_a.width, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=22.0, clipping_range=(0.02, 30.0)),
        offset=CameraCfg.OffsetCfg(pos=eye, rot=look_at_quat(eye, at),
                                   convention="opengl"))


def _contact_cfg(prim: str):
    """三个接触体各一个传感器，**都过滤到平面**（`plan/01` §5.3 的"任一接触体"）。

    规则 7：平面必须是 kinematic 刚体而不是静态碰撞体，否则 filter 通道
    静默失效，region 与 mode 两个字段直接作废（P-17）。
    """
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim}", track_pose=True, track_contact_points=True,
        max_contact_data_count_per_prim=MAX_CONTACTS,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Board"],
        update_period=0.0, history_length=0)


@configclass
class SceneCfg(InteractiveSceneCfg):
    dome = AssetBaseCfg(prim_path="/World/dome",
                        spawn=sim_utils.DomeLightCfg(intensity=300.0,
                                                     color=(0.86, 0.89, 1.0)))
    sun = AssetBaseCfg(prim_path="/World/sun",
                       spawn=sim_utils.DistantLightCfg(intensity=1100.0, angle=5.0))
    board = A.board_cfg(size=BOARD, friction=BOARD_MU)
    eraser = A.ERASER_CFG
    plate0 = A.plate_cfg(0)
    plate1 = A.plate_cfg(1)
    contact_e = _contact_cfg("Eraser")
    contact_p0 = _contact_cfg("Plate0")
    contact_p1 = _contact_cfg("Plate1")
    if _a.video:
        cam = _camera_cfg()


# ---------------------------------------------------------------- 家族参数


def _u(rng, lo, hi, n):
    return torch.from_numpy(rng.uniform(lo, hi, size=n).astype(np.float32))


def sample_family(fam: str, rng, n: int, device) -> dict:
    """逐 env 参数。六个家族共用一个阶段机，只有这里返回的参数不同。

    Returns:
        ``tool``：1 = 持工具擦，0 = 直擦；
        ``grip_off`` 抓持点沿黑板擦**长轴 X** 的偏移（偏心抓）；
        ``tilt`` 绕扫掠轴的侧倾角；
        ``press`` 下压力（N）；``n_pass`` 蛇形趟数（趟数越多扫得越快）；
        ``y_dir`` 起始扫掠方向（±1）。
    """
    z = lambda: torch.zeros(n)  # noqa: E731
    tool, grip_off, tilt = torch.ones(n), z(), z()
    press = _u(rng, 4.0, 5.5, n)
    n_pass = torch.full((n,), 3.0)
    y_dir = torch.from_numpy(rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=n))

    if fam == "tool_center":
        pass
    elif fam == "tool_offset":
        # 偏心抓：板贴在黑板擦两个 ±Y **长侧面**上，但沿长轴 X 偏离中心。
        # 长侧面 90 mm、板长边 35 mm，两边各有 27.5 mm 余量，
        # ±6~10 mm 的偏心完全落在面内。（贴短端面时余量只有 5 mm，
        # 同样的偏心会让板有一小半悬空。）
        grip_off = torch.from_numpy(
            rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=n)) \
            * _u(rng, 0.006, 0.010, n)
    elif fam == "tool_tilt":
        # 轻微倾斜抓持：绕**扫掠轴 Y** 侧倾，底垫单边受力更重。
        # 绕 Y 转**不改变 ±Y 面的法向**，所以抓持面始终与板平贴；
        # 板也跟着一起倾（见 `q_grip`），就是人捏着长边把工具侧一侧。
        # 不绕竖直轴偏摆——那样夹持面就不再是黑板擦的 ±Y 面了。
        tilt = torch.from_numpy(
            rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=n)) \
            * _u(rng, math.radians(3), math.radians(7), n)
    elif fam == "tool_heavy_slow":
        # 慢：趟数少 -> 每趟走得慢。**不能用 speed<1 去减速**，
        # 那样路径根本走不完，实测清除率卡在 67%。
        press, n_pass = _u(rng, 6.0, 7.5, n), torch.full((n,), 2.0)
    elif fam == "tool_light_fast":
        press, n_pass = _u(rng, 3.2, 4.0, n), torch.full((n,), 4.0)
    elif fam == "direct_wipe":
        # 不用工具，两块板直接以板面擦，沿 X 并排拉开覆盖更宽的带。
        # press 在这里是**每块板**的压力（见 PLATE_STIFF 的说明）。
        tool = torch.zeros(n)
        press = _u(rng, 3.8, 5.5, n)
    else:
        raise ValueError(f"未知策略家族：{fam}")
    return {k: v.to(device) for k, v in
            dict(tool=tool, grip_off=grip_off, tilt=tilt, press=press,
                 n_pass=n_pass, y_dir=y_dir).items()}


def raster(u: torch.Tensor, y_dir: torch.Tensor, n_pass: torch.Tensor):
    """蛇形栅格路径。返回平面局部系的 (x, y)，单位 m。

    沿 Y 来回扫，同时沿 X **连续**漂移。趟数由家族决定：
    趟数多 = 同样时间里走得快，趟数少 = 走得慢，而**路径总能走完**。

    ⚠️ X 必须连续。第一版把 X 写成按趟数跳变的阶梯（每趟换道跳 65 mm），
    位置 PD 收到 65 mm 的阶跃指令会瞬间输出 1500 N/m × 0.065 = 97 N，
    板横向猛甩、黑板擦被打飞——实测工具漂移 1.9~2.7 **米**。
    换成整段线性漂移之后路径仍然覆盖同一片区域（垫子 80 mm 宽、
    道心从 −65 走到 +65，覆盖 X ∈ [−105, 105]），而且处处连续。
    """
    t = u.clamp(0.0, 1.0)                              # 路径**总是走完**
    p = t * n_pass
    idx = torch.minimum(p.floor(), n_pass - 1)
    local = p - idx
    half = REGION[1] / 2 * 0.94
    fwd = ((idx % 2) == 0).float() * 2.0 - 1.0          # 奇数趟反向，形成蛇形
    y = y_dir * fwd * (-half + 2 * half * local)
    x = (t - 0.5) * 2.0 * 0.058
    return x, y


# ---------------------------------------------------------------- 采集


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
        self.dirt = torch.zeros(N_FRAMES, n, *GRID, dtype=torch.bool, device=device)
        self.cleared = z(n)
        self.pos_obj = z(3, n, MAX_CONTACTS, 3)
        self.nrm_obj = z(3, n, MAX_CONTACTS, 3)
        self.fri_obj = z(3, n, MAX_CONTACTS, 3)
        self.rvel_obj = z(3, n, MAX_CONTACTS, 3)
        self.fn = z(3, n, MAX_CONTACTS)
        self.sep = z(3, n, MAX_CONTACTS)
        self.cvalid = torch.zeros(N_FRAMES, 3, n, MAX_CONTACTS, dtype=torch.bool,
                                  device=device)
        self.mode = torch.zeros(N_FRAMES, 3, n, MAX_CONTACTS, dtype=torch.int8,
                                device=device)
        self.in_region = torch.zeros(N_FRAMES, 3, n, MAX_CONTACTS, dtype=torch.bool,
                                     device=device)
        self.body_fn = z(3, n)
        self.body_v = z(3, n)
        self.src_pose = z(2, n, 7)
        self.src_vel = z(2, n, 6)
        self.src_tgt = z(2, n, 7)
        self.src_cmd = z(2, n, 6)
        self.tool_pose = z(n, 7)
        #: 被操作物体（平面）的**世界位姿**。平面是 kinematic、全程不动，
        #: 所以这是一列常量——但它必须被记下来：S4 要用"接触体位姿 − 物体位姿"
        #: 算接触点的相对速度，两者必须在**同一个世界系**里。缺了它只能假定
        #: 平面在原点，而板的位姿含 env 原点偏移（env 间距 2.2 m），
        #: 混用会算出 8 m 的力臂、几十毫米的假滑移，**且零征兆**（P-54）。
        self.board_pose = z(n, 7)
        self.foreign = z(3)
        self.dropped = z(3)
        self.rot_err = z(2, n)



class DirtVis:
    """把 dirt 网格画成平面上一片片**纯视觉、无碰撞**的小方片（只在录像时开）。

    为什么要有它：擦拭的 effect **只有** dirt 状态变化（`plan/02` §3.1 经
    D-42 修订），而 dirt 一直是个内部张量、没有任何可视几何——录像里看得到
    板和黑板擦怎么动，**看不到污渍**。"动作看着对、区域没擦干净"这两件事
    在画面上分不出来，所以一直只能另出一张 `dirt_tc.png` 覆盖图。
    现在录像里也能直接看到污渍一格格消失，覆盖图仍然保留（它给的是全体
    episode 的统计，录像只有一条）。

    只给 **env 0** 建（相机只拍它），只在 ``--video`` 时建，格子不带碰撞、
    不带物理 API，**对物理和数据没有任何影响**。擦除是单调的，所以每次渲染
    只需要把"这一帧新变干净的"藏掉，累计不超过 630 次写。
    """

    def __init__(self, stage, origin, top_z):
        from pxr import Gf, UsdGeom
        self._UsdGeom = UsdGeom
        self.prims = {}
        root = UsdGeom.Xform.Define(stage, "/World/DirtVis")
        UsdGeom.Xformable(root).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
        for i in range(GRID[0]):
            for j in range(GRID[1]):
                x = float(origin[0]) + (i + 0.5) * CELL - REGION[0] / 2
                y = float(origin[1]) + (j + 0.5) * CELL - REGION[1] / 2
                c = UsdGeom.Cube.Define(stage, f"/World/DirtVis/c_{i}_{j}")
                c.CreateSizeAttr(1.0)
                xf = UsdGeom.Xformable(c)
                xf.AddTranslateOp().Set(Gf.Vec3d(x, y, float(top_z) + 0.0006))
                # 铺满单元格（0.96 而不是 0.42）：占比小的时候看着像一片散点，
                # 铺满之后脏区连成一片，"哪一块被擦干净了"一眼就能看出来。
                xf.AddScaleOp().Set(Gf.Vec3f(CELL * 0.96, CELL * 0.96, 0.0004))
                c.CreateDisplayColorAttr([Gf.Vec3f(0.16, 0.12, 0.09)])
                self.prims[(i, j)] = c.GetPrim()
        self.shown = {k: True for k in self.prims}

    def update(self, dirt_env0):
        """``dirt_env0``：(GRID) 的 bool，True = 还脏。把新变干净的藏掉。"""
        clean = (~dirt_env0).nonzero(as_tuple=False).tolist()
        for i, j in clean:
            if self.shown.get((i, j)):
                self._UsdGeom.Imageable(self.prims[(i, j)]).MakeInvisible()
                self.shown[(i, j)] = False


def run_batch(scene, sim, camera, fam_of_env, rng, device, batch):
    n = scene.cfg.num_envs
    board: RigidObject = scene["board"]
    eraser: RigidObject = scene["eraser"]
    plates = [scene["plate0"], scene["plate1"]]
    sensors = [scene["contact_e"], scene["contact_p0"], scene["contact_p1"]]
    bodies = [eraser] + plates

    par = {k: torch.zeros(n, device=device)
           for k in ("tool", "grip_off", "tilt", "press", "n_pass", "y_dir")}
    for fam in sorted(set(fam_of_env)):
        idx = [i for i, f in enumerate(fam_of_env) if f == fam]
        sub = sample_family(fam, rng, len(idx), device)
        sel = torch.tensor(idx, device=device, dtype=torch.long)
        for k in par:
            par[k][sel] = sub[k]
    tool, grip_off = par["tool"], par["grip_off"]
    tilt, press, n_pass, y_dir = (
        par["tilt"], par["press"], par["n_pass"], par["y_dir"])
    is_tool = tool > 0.5

    # 物理变体（`plan/03` §7）：改平面摩擦做不到（材质在 USD 里），
    # 改**下压力区间**同样是"没见过的物理条件"，且对 mechanics 字段是真变化。
    is_var = torch.from_numpy((rng.random(n) < _a.physics_variant_frac)
                              .astype(np.bool_)).to(device)
    lo, hi = ((8.5, 9.5) if batch % 2 == 0 else (2.0, 2.6))
    press = torch.where(is_var, _u(rng, lo, hi, n).to(device), press)

    # --- 复位 ---
    b_pos = board.data.root_pos_w
    b_quat = board.data.root_quat_w
    top = b_pos + torch.tensor([0.0, 0.0, BOARD[2] / 2], device=device)
    # **几何变体**（`plan/03` §7 的"小幅几何变化"）：黑板擦长度按 env 轮转，
    # 80 / 90 / 100 mm。夹持面在 ±Y、位置由 body[1] 决定且三档不变，
    # 所以采集器不必按 env 取夹持几何，只需标签——变的是擦拭覆盖的足迹。
    geom_tag = [A.geom_tag_of(i) for i in range(n)]

    x0, y0 = raster(torch.zeros(n, device=device), y_dir, n_pass)

    e_st = eraser.data.default_root_state.clone()
    e_st[:, 0] = top[:, 0] + x0
    e_st[:, 1] = top[:, 1] + y0
    e_st[:, 2] = top[:, 2] + 0.0005
    ax_y = torch.tensor([0.0, 1.0, 0.0], device=device).expand(n, 3)
    from isaaclab.utils.math import quat_from_angle_axis, quat_mul
    e_st[:, 3:7] = quat_from_angle_axis(tilt, ax_y)
    e_st[:, 7:] = 0.0
    # 直擦的那一档把黑板擦挪到一边，别让它挡路
    e_st[:, 0] = torch.where(is_tool, e_st[:, 0], top[:, 0] + 0.26)
    e_st[:, 1] = torch.where(is_tool, e_st[:, 1], top[:, 1] + 0.20)
    eraser.write_root_state_to_sim(e_st)

    # 板的姿态：工作面法向指向黑板擦（持工具）或指向平面（直擦）
    # 两块板从两侧夹黑板擦，工作面法向相反。**深色鳍（局部 +Y）必须都朝上**，
    # 否则录像里一块朝上一块朝下、看着像翻了 180°（实测正好 -1.00）。
    _up = torch.tensor([[0.0, 0.0, 1.0]], device=device).expand(n, 3)
    # **夹的是两个 90×25 的长侧面（±Y），不是 45×25 的短端面。**
    # 生活中捏黑板擦就是捏长边；而且扫掠恰好沿 Y，捏长侧面时推力由**法向**
    # 直接传，捏短端面时只能靠摩擦传。
    q_grip = [quat_mul(
        quat_from_angle_axis(tilt, ax_y),
        quat_face_and_up(
            torch.tensor([[0.0, -1.0, 0.0]], device=device).expand(n, 3)
            * (1 if k == 0 else -1), _up)) for k in range(2)]
    # 直擦时板的**长边沿 X**（换道方向），两块板合起来 95 mm 宽，
    # 与黑板擦垫子的 80 mm 同量级；长边沿 Y 只有 69 mm，覆盖不住。
    # 直擦时板是水平的、法向朝下，"鳍朝上"退化（up 与法向共线）；
    # 两块板取同一个水平参考即可，保证互相一致。
    q_flat = quat_face_and_up(
        torch.tensor([[0.0, 0.0, -1.0]], device=device).expand(n, 3),
        torch.tensor([[0.0, -1.0, 0.0]], device=device).expand(n, 3),
        long_axis=torch.tensor([[1.0, 0.0, 0.0]], device=device).expand(n, 3))
    quats = [torch.where(is_tool.unsqueeze(-1), q_grip[k], q_flat) for k in range(2)]

    pds = [FloatingPD(pl, kp_pos=3000.0, kd_pos=110.0, kp_rot=6.0, kd_rot=0.025,
                      max_force=120.0, max_torque=4.0, kd_force=15.0) for pl in plates]

    def grip_targets(cx, cy, press_frac: float, open_extra: float = 0.0):
        """给定接触体中心的平面局部 (x, y)，返回两块板的世界目标位置。

        ``press_frac`` 是下压量的比例（0~1）。**必须等夹拢之后才加压。**
        张开着落下去时板不受阻，会一路落到"已经下压"的那个高度；等夹拢时
        竖直方向的位置误差已经归零，压力也就归零了——实测板的 z 误差从
        +1.5 mm 掉到 +0.15 mm，平面上的法向力从 4.8 N 掉到 0.45 N，
        清除率 0%。顺序必须是：张开落下 -> 夹拢 -> 再下压。

        ``open_extra`` 是夹持轴上额外张开的量（m）。**下落时必须先张开。**
        夹持位比黑板擦本体窄 2×GRIP_INTERF，直上直下落下去等于拿两块板的
        下边缘楔在工具顶面上：捏短端面时工具要沿长轴挪 45 mm 才能脱出，
        勉强夹得住；捏长侧面时只要挪 22 mm，实测工具被顶高 23 mm、
        随即甩出 200 mm，六个持工具家族全线失败（力 1.0 N、清除率 29%）。
        先张开 6 mm 落到位、再夹拢，就是人拿黑板擦的顺序。
        """
        out = []
        for k in range(2):
            sgn = 1.0 - 2.0 * k                     # +1 / -1
            p = torch.zeros(n, 3, device=device)
            if True:
                # 持工具：贴在黑板擦 ±Y **长侧面**上，压进 GRIP_INTERF；
                # 偏心抓沿黑板擦长轴 X 偏移。
                gy = _E.body[1] / 2 + _P.size[2] / 2 - GRIP_INTERF + open_extra
                tx = top[:, 0] + cx + grip_off
                ty = top[:, 1] + cy + sgn * gy
                on_ = press_frac
                dz_tool = press / (2.0 * PLATE_STIFF) * on_
                dz_direct = press / PLATE_STIFF * on_
                tz = top[:, 2] + _E.pad[2] + _E.body[2] / 2 - dz_tool
                # 直擦：两块板并排压在平面上，沿 X 拉开 ±30 mm。
                # ±22 时两块板合起来只有 69 mm 宽，加上漂移覆盖不到 190 mm 的
                # 目标区域，清除率卡在 26%——那是任务做不到，不是做法不好。
                dx = top[:, 0] + cx + sgn * 0.030
                dy = top[:, 1] + cy
                dz = top[:, 2] + _P.size[2] / 2 - dz_direct
                p[:, 0] = torch.where(is_tool, tx, dx)
                p[:, 1] = torch.where(is_tool, ty, dy)
                p[:, 2] = torch.where(is_tool, tz, dz)
            out.append(p)
        return out

    targets = []
    for k, pl in enumerate(plates):
        st = pl.data.default_root_state.clone()
        g = grip_targets(x0, y0, 0.0, GRIP_OPEN)[k]
        st[:, :3] = g + torch.tensor([0.0, 0.0, STANDOFF], device=device)
        st[:, 3:7] = quats[k]
        st[:, 7:] = 0.0
        pl.write_root_state_to_sim(st)
        targets.append(st[:, :3].clone())
    for _ in range(8):
        for pl in plates:
            zw = torch.zeros(n, 1, 3, device=device)
            pl.set_external_force_and_torque(zw, zw, is_global=True)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(DT)

    buf = Buffers(n, device)
    dirt = torch.ones(n, *GRID, dtype=torch.bool, device=device)
    dirt_vis = None
    if camera is not None:
        import omni.usd
        stg = omni.usd.get_context().get_stage()
        if stg.GetPrimAtPath("/World/DirtVis"):
            stg.RemovePrim("/World/DirtVis")
        dirt_vis = DirtVis(stg, top[0].tolist(), float(top[0, 2]))
    # 网格中心（平面局部系）
    gx = (torch.arange(GRID[0], device=device) + 0.5) * CELL - REGION[0] / 2
    gy = (torch.arange(GRID[1], device=device) + 0.5) * CELL - REGION[1] / 2
    prev_tgt = [t.clone() for t in targets]
    preview: list[np.ndarray] = []
    every = max(1, round(1.0 / (CONTROL_DT * _a.fps)))
    n_dirt0 = float(GRID[0] * GRID[1])

    frame = 0
    for phase_id, plen in enumerate(PHASE_STEPS):
        for kk in range(plen):
            # release **必须从路径终点继续**，不能回到起点。
            # 原来 release 用 u=0 取路径起点，于是抬手那一刻目标从终点跳回起点——
            # 蛇形趟数是奇数时终点与起点相隔 282 mm，位置 PD 收到这个阶跃就把
            # 黑板擦甩出去。趟数为偶数的家族终点恰好等于起点，所以毫发无损，
            # 失败按家族劈成两半——**那个分布就是线索**（同 P-32）。
            u_ph = {0: 0.0, 1: 0.0, 2: (kk + 1) / plen, 3: 1.0}[phase_id]
            u = torch.full((n,), u_ph, device=device)
            cx, cy = raster(u, y_dir, n_pass)
            # establish 相位分三段，与人拿黑板擦的顺序一致：
            #   u < 0.40   张开着落到位
            #   0.40~0.70  夹拢（张开量线性归零）
            #   0.70~1.00  加压（下压量线性升到 1）
            if phase_id == 0:
                open_extra, press_frac = GRIP_OPEN, 0.0
            elif phase_id == 1:
                u = (kk + 1) / max(plen, 1)
                open_extra = GRIP_OPEN * min(max((0.70 - u) / 0.30, 0.0), 1.0)
                press_frac = min(max((u - 0.70) / 0.30, 0.0), 1.0)
            else:
                open_extra, press_frac = 0.0, 1.0
            g = grip_targets(cx, cy, press_frac, open_extra)
            for k in range(2):
                tgt = targets[k]
                if phase_id == 0:                        # approach：张开着降到接触上方
                    a = min((kk + 1) / max(plen - 6, 1), 1.0)
                    tgt[:] = g[k] + torch.tensor([0.0, 0.0, STANDOFF], device=device) \
                        * (1.0 - a) + torch.tensor([0.0, 0.0, 0.012], device=device) * a
                elif phase_id == 1:                      # establish：落到位、夹拢、加压
                    tgt[:] = g[k] + torch.tensor([0.0, 0.0, 0.012], device=device) \
                        * max(1.0 - (kk + 1) / max(plen - 10, 1), 0.0)
                elif phase_id == 2:                      # manipulate：栅格擦拭
                    tgt[:] = g[k]
                else:                                    # release：先松夹持再抬起
                    # 不先松开就抬，等于把黑板擦一起提走——实测工具被举高
                    # 120 mm，"工具被举离平面"成了第一失败原因。
                    sgn = 1.0 - 2.0 * k
                    a = min((kk + 1) / 12.0, 1.0)
                    b = min(max((kk + 1 - 12) / max(plen - 12, 1), 0.0), 1.0)
                    # 沿**夹持轴 Y** 张开（夹的是长侧面）。原来写成沿 X 张开，
                    # 那是贴短端面时的夹持轴，改了抓持面就不再对应。
                    open_g = torch.zeros(n, 3, device=device)
                    open_g[:, 1] = sgn * 0.010 * a
                    tgt[:] = g[k] + open_g \
                        + torch.tensor([0.0, 0.0, STANDOFF], device=device) * b
                buf.src_cmd[frame, k, :, :3] = tgt - prev_tgt[k]
                prev_tgt[k] = tgt.clone()

            best = [None, None, None]
            best_fn = torch.zeros(3, n, device=device)
            for _sub in range(DECIMATION):
                for k, pl in enumerate(plates):
                    # 三轴全位置控制：X 夹持干涉、Y 扫掠、Z 下压干涉（见 PRESS_STIFF）
                    f, tq = pds[k].compute(targets[k], quats[k])
                    pl.set_external_force_and_torque(f, tq, is_global=True)
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(DT)
                for bi, body in enumerate(bodies):
                    cp = extract_contact_points_padded(
                        sensors[bi], DT, body_pos_w=body.data.root_pos_w,
                        max_points=MAX_CONTACTS, own_radius=0.09)
                    tot = cp["normal_forces"].abs().sum(dim=1)
                    buf.foreign[frame, bi] += cp["foreign"]
                    buf.dropped[frame, bi] += cp["dropped"]
                    if best[bi] is None:
                        best[bi], best_fn[bi] = cp, tot
                    else:
                        take = (tot > best_fn[bi]).unsqueeze(-1)
                        for key in ("normal_forces", "separations"):
                            best[bi][key] = torch.where(take, cp[key], best[bi][key])
                        for key in ("positions", "normals", "friction_forces"):
                            best[bi][key] = torch.where(take.unsqueeze(-1), cp[key],
                                                        best[bi][key])
                        best[bi]["valid"] = torch.where(take, cp["valid"],
                                                        best[bi]["valid"])
                        best_fn[bi] = torch.maximum(best_fn[bi], tot)

            # ---------------- 清除污渍（`plan/01` §5.3）----------------
            for bi, body in enumerate(bodies):
                cp = best[bi]
                pl_loc = to_local(cp["positions"], b_pos, b_quat)
                buf.pos_obj[frame, bi] = pl_loc * cp["valid"].unsqueeze(-1)
                buf.nrm_obj[frame, bi] = rotate_inverse(b_quat, cp["normals"])
                buf.fri_obj[frame, bi] = rotate_inverse(b_quat, cp["friction_forces"])
                # `plan/03` §5 的相对速度。这里的另一侧是**平面**——它是
                # kinematic 刚体、全程不动，所以省略 B 即可（见 `contact_rel_vel`）。
                rv = contact_rel_vel(cp["positions"], body.data.root_pos_w,
                                     body.data.root_lin_vel_w, body.data.root_ang_vel_w)
                buf.rvel_obj[frame, bi] = rotate_inverse(b_quat, rv) * cp["valid"].unsqueeze(-1)
                buf.fn[frame, bi] = cp["normal_forces"]
                buf.sep[frame, bi] = cp["separations"]
                buf.cvalid[frame, bi] = cp["valid"]
                buf.mode[frame, bi] = classify_contact_mode_padded(
                    cp["normal_forces"], cp["friction_forces"], cp["separations"],
                    cp["valid"], mu=min(_P.friction, BOARD_MU))
                fn_b = cp["normal_forces"].abs().sum(dim=1)
                v_b = body.data.root_lin_vel_w[:, :2].norm(dim=-1)
                buf.body_fn[frame, bi] = fn_b
                buf.body_v[frame, bi] = v_b
                # 容差取清除半径：贴着边界的接触确实清除了边界那一格，
                # 用严格不等式会把正常的边缘扫掠算成"接触落在区域外"。
                inside = (pl_loc[..., 0].abs() < REGION[0] / 2 + CLEAR_RADIUS) & \
                         (pl_loc[..., 1].abs() < REGION[1] / 2 + CLEAR_RADIUS) & \
                         cp["valid"]
                buf.in_region[frame, bi] = inside
                # 三个条件同时满足才擦掉：法向力在区间内、有切向相对运动、扫过该格
                ok_b = (fn_b >= WIPE_LO) & (fn_b <= WIPE_HI) & (v_b > SLIDE_V_MIN)
                if not bool(ok_b.any()):
                    continue
                m = cp["valid"].unsqueeze(-1)
                big = torch.where(m, pl_loc, torch.full_like(pl_loc, -1e3))
                small = torch.where(m, pl_loc, torch.full_like(pl_loc, 1e3))
                hi_xy = big[..., :2].max(dim=1).values + CLEAR_RADIUS
                lo_xy = small[..., :2].min(dim=1).values - CLEAR_RADIUS
                has = cp["valid"].any(dim=1) & ok_b
                cover = ((gx.view(1, -1, 1) >= lo_xy[:, 0].view(-1, 1, 1))
                         & (gx.view(1, -1, 1) <= hi_xy[:, 0].view(-1, 1, 1))
                         & (gy.view(1, 1, -1) >= lo_xy[:, 1].view(-1, 1, 1))
                         & (gy.view(1, 1, -1) <= hi_xy[:, 1].view(-1, 1, 1)))
                dirt = dirt & ~(cover & has.view(-1, 1, 1))

            buf.dirt[frame] = dirt
            buf.cleared[frame] = 1.0 - dirt.float().sum(dim=(1, 2)) / n_dirt0
            buf.phase[frame] = phase_id
            buf.progress[frame] = buf.cleared[frame]
            buf.tool_pose[frame] = torch.cat(
                [eraser.data.root_pos_w, eraser.data.root_quat_w], dim=-1)
            buf.board_pose[frame] = torch.cat(
                [board.data.root_pos_w, board.data.root_quat_w], dim=-1)
            ok = torch.ones(n, dtype=torch.bool, device=device)
            for k, pl in enumerate(plates):
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
            buf.valid[frame] = ok & (buf.fn[frame].abs().sum(dim=(0, 2)) <= MAX_VALID_FORCE)

            if camera is not None and frame % every == 0:
                if dirt_vis is not None:
                    dirt_vis.update(dirt[0])       # 先更新污渍，再渲染
                sim.render()
                camera.update(CONTROL_DT)
                preview.append(camera.data.output["rgb"][0, ..., :3]
                               .detach().cpu().numpy().astype(np.uint8))
            frame += 1

    n_cut, n_keep = float(buf.dropped.sum()), float(buf.cvalid.sum())
    print(f"  接触点：保留 {n_keep:.0f}，离所有接触体都远而丢弃 "
          f"{float(buf.foreign.sum()):.0f}，超上限截掉 {n_cut:.0f}", flush=True)
    if n_cut > 0.01 * max(n_keep, 1.0):
        raise RuntimeError(f"{n_cut:.0f} 个接触点被静默截掉（P-03），调大 MAX_CONTACTS")
    return buf, dict(is_tool=is_tool, press=press, n_pass=n_pass, tilt=tilt,
                     grip_off=grip_off, is_var=is_var, y_dir=y_dir,
                     geom_tag=geom_tag), preview


# ---------------------------------------------------------------- 判定与落盘


def diagnostics(buf: Buffers, m: dict, e: int) -> dict[str, Any]:
    fn = buf.fn[:, :, e, :].abs()
    total = float(fn.sum().item())
    manip = buf.phase[:, e] == 2
    body_fn = buf.body_fn[:, :, e]
    wiper = 0 if bool(m["is_tool"][e]) else 1          # 工具 or 板 0
    fw = body_fn[:, wiper]
    touching = fw > 0.05
    in_range = (fw >= WIPE_LO) & (fw <= WIPE_HI)

    def share(codes, table):
        return {nm: (float((fn * (codes == i)).sum().item() / total) if total > 0 else 0.0)
                for i, nm in enumerate(table)}

    reg = float((fn * buf.in_region[:, :, e, :]).sum().item() / total) if total > 0 else 0.0
    mid = 0.5 * (buf.src_pose[:, 0, e, :2] + buf.src_pose[:, 1, e, :2])
    rel = buf.tool_pose[:, e, :2] - mid
    slip = (rel - rel[0]).norm(dim=-1)
    return {
        "cleared_fraction": float(buf.cleared[-1, e].item()),
        "manip_no_contact_fraction": float(
            ((manip & ~touching).sum() / max(int(manip.sum().item()), 1)).item()),
        "force_in_range_fraction": float(
            ((in_range & manip).sum() / max(int((touching & manip).sum().item()), 1)).item()),
        "in_region_force_share": reg,
        "mean_wipe_force_N": float(fw[manip & touching].mean().item())
        if int((manip & touching).sum().item()) > 0 else 0.0,
        "peak_point_force_N": float(fn.max().item()),
        "mode_share": share(buf.mode[:, :, e, :],
                            ("no_contact", "sticking", "sliding", "separating")),
        "mean_contacts_per_frame": float(
            buf.cvalid[:, :, e, :].float().sum(dim=(1, 2)).mean().item()),
        "invalid_frame_fraction": float(1.0 - buf.valid[:, e].float().mean().item()),
        "mean_orientation_error_deg": float(
            torch.rad2deg(buf.rot_err[:, :, e]).mean().item()),
        "tool_used": bool(m["is_tool"][e]),
        # 量**抓持有没有打滑**，不是量工具走了多远——蛇形路径本来就要走
        # 三百毫米，拿绝对位移当漂移会把正常擦拭判成失败。
        # 判据是工具相对两块板中点的偏移有没有变。
        "tool_slip_mm": float(slip.max().item() * 1000.0)
        if bool(m["is_tool"][e]) else 0.0,
        "tool_lift_mm": float(buf.tool_pose[manip, e, 2].max().item() * 1000.0)
        if bool(m["is_tool"][e]) else 0.0,
    }


def judge(d: dict) -> tuple[bool, list[str]]:
    fails = []
    if d["cleared_fraction"] < 0.70:
        fails.append("目标区域没擦干净")
    if d["manip_no_contact_fraction"] > 0.25:
        fails.append("擦拭阶段脱手过多")
    if d["force_in_range_fraction"] < 0.60:
        fails.append("法向力常在允许区间之外")
    if d["in_region_force_share"] < 0.60:
        fails.append("接触多数落在目标区域之外")
    if d["invalid_frame_fraction"] > 0.10:
        fails.append("脏帧过多")
    # 持工具那一档额外查工具有没有掉（`plan/01` §7）
    if d["tool_used"] and d["tool_lift_mm"] > 40.0:
        fails.append("工具被举离平面")
    if d["tool_used"] and d["tool_slip_mm"] > 25.0:
        fails.append("工具从夹持中滑脱")
    return (not fails), fails


def to_arrays(buf: Buffers, e: int) -> dict[str, np.ndarray]:
    """前缀分层同 ``it.records``。

    ``object/dirt_grid`` 是**擦拭的 effect 本身**（`plan/02` §3.1 经 D-42 修订后
    只保留它）。工具位姿在 ``source/tool_pose``——它是 source 侧的东西，
    进 envelope 就等于迁移动作而不是迁移交互。
    """
    cpu = lambda t: t.detach().cpu().numpy()  # noqa: E731
    out = {
        "phase": cpu(buf.phase[:, e]),
        "progress": cpu(buf.progress[:, e]).astype(np.float32),
        "valid_frame": cpu(buf.valid[:, e]),
        "object/dirt_grid": cpu(buf.dirt[:, e]),
        "object/dirt_cleared": cpu(buf.cleared[:, e]).astype(np.float32)[:, None],
        "source/tool_pose": cpu(buf.tool_pose[:, e]).astype(np.float32),
        # 平面的世界位姿。它进 `source/*` 是因为**世界系的量一律进 source**
        # （不进表示），但 S4 必须能读到它才能把接触体位姿转到物体系（P-54）。
        "source/board_pos_w": cpu(buf.board_pose[:, e, :3]).astype(np.float32),
        "source/board_quat_w": cpu(buf.board_pose[:, e, 3:7]).astype(np.float32),
    }
    for bi, nm in enumerate(("tool", "plate0", "plate1")):
        c = f"contact/{nm}"
        out[f"{c}/pos_obj"] = cpu(buf.pos_obj[:, bi, e]).astype(np.float32)
        out[f"{c}/normal_obj"] = cpu(buf.nrm_obj[:, bi, e]).astype(np.float32)
        out[f"{c}/friction_obj"] = cpu(buf.fri_obj[:, bi, e]).astype(np.float32)
        out[f"{c}/rel_vel_obj"] = cpu(buf.rvel_obj[:, bi, e]).astype(np.float32)
        out[f"{c}/normal_force"] = cpu(buf.fn[:, bi, e]).astype(np.float32)
        out[f"{c}/separation"] = cpu(buf.sep[:, bi, e]).astype(np.float32)
        out[f"{c}/valid"] = cpu(buf.cvalid[:, bi, e])
        out[f"{c}/mode"] = cpu(buf.mode[:, bi, e])
        out[f"{c}/in_region"] = cpu(buf.in_region[:, bi, e])
    for k in range(2):
        s = f"source/plate{k}"
        out[f"{s}/root_pose"] = cpu(buf.src_pose[:, k, e]).astype(np.float32)
        out[f"{s}/root_velocity"] = cpu(buf.src_vel[:, k, e]).astype(np.float32)
        out[f"{s}/target_pose"] = cpu(buf.src_tgt[:, k, e]).astype(np.float32)
        out[f"{s}/cmd_delta"] = cpu(buf.src_cmd[:, k, e]).astype(np.float32)
    return out


def write_preview(path: Path, frames: list[np.ndarray]) -> None:
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
    add(f"S3 擦拭双板 source · {len(metas)} episode，成功 {len(ok)} "
        f"({100.0 * len(ok) / max(len(metas), 1):.1f}%)")
    add("")
    add(f"{'家族':<18}{'条数':>5}{'成功':>6}{'清除%':>8}{'脱手%':>8}{'力在区间%':>11}"
        f"{'区域内%':>9}{'力N':>7}{'工具滑脱mm':>11}")
    for fam in sorted({m["strategy_family"] for m in metas}):
        sub = [m for m in metas if m["strategy_family"] == fam]
        g = lambda k: float(np.mean([m["diagnostics"][k] for m in sub]))  # noqa: E731
        add(f"{fam:<18}{len(sub):>5}{sum(m['success'] for m in sub):>6}"
            f"{100 * g('cleared_fraction'):>8.1f}{100 * g('manip_no_contact_fraction'):>8.1f}"
            f"{100 * g('force_in_range_fraction'):>11.1f}"
            f"{100 * g('in_region_force_share'):>9.1f}{g('mean_wipe_force_N'):>7.2f}"
            f"{g('tool_slip_mm'):>11.1f}")
    add("")
    add("接触模式分布（按法向力加权，`plan/02` §3.4）")
    for nm in ("no_contact", "sticking", "sliding", "separating"):
        v = float(np.mean([m["diagnostics"]["mode_share"][nm] for m in metas]))
        add(f"  {nm:<14}{100 * v:>7.2f}%")
    add("")
    for k in ("peak_point_force_N", "invalid_frame_fraction",
              "mean_orientation_error_deg", "mean_contacts_per_frame"):
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
    scene = InteractiveScene(SceneCfg(num_envs=_a.envs, env_spacing=1.6,
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
        # 用 (b + i) 而不是 (b*envs + i)：录像只录 env 0，而 envs 是家族数的
        # 倍数时 env 0 永远落在同一个家族上——45 个批次全都只录到 tool_center。
        fam_of = [fams[(b + i) % len(fams)] for i in range(_a.envs)]
        print(f"\n=== wipe batch {b + 1}/{_a.batches} · {_a.envs} env ===", flush=True)
        buf, m, preview = run_batch(scene, sim, camera, fam_of, rng, device, b)
        if camera is not None:
            write_preview(out_dir / "videos" / f"wipe_{fam_of[0]}_b{b}.mp4", preview)
        for e in range(_a.envs):
            d = diagnostics(buf, m, e)
            good, fails = judge(d)
            fam = fam_of[e]
            eid = f"wipe-{fam}-b{b:02d}e{e:03d}"
            meta = {
                "schema_version": SCHEMA_VERSION, "episode_id": eid, "task": "wipe",
                "source_embodiment": "two_dynamic_plates",
                "implementation": "tool" if d["tool_used"] else "direct",
                "strategy_family": fam, "strategy_variant": f"b{b:02d}e{e:03d}",
                "physics_variant": "heldout_press" if bool(m["is_var"][e]) else "nominal",
                "geometry_variant": m["geom_tag"][e],
                # `plan/01` §5.1 的两种**实现**。`plan/03` §7 要求至少留出一种
                # 实现的全部 episode 作跨实现测试（对应 `02` §7 第 8 条：
                # envelope 与是否使用工具无关）。
                "implementation": "with_tool" if bool(m["is_tool"][e]) else "direct",
                "success": good, "failure_reasons": fails,
                "seed": int(_a.seed + b * _a.envs + e),
                "control_hz": 1.0 / CONTROL_DT, "physics_hz": 1.0 / DT,
                "phase_names": list(PHASE_NAMES), "phase_steps": list(PHASE_STEPS),
                "generator_git_sha": sha,
                "region_m": list(REGION), "grid": list(GRID), "cell_m": CELL,
                "physics": {"press_N": float(m["press"][e].item()),
                            "board_friction": BOARD_MU,
                            "eraser_pad_friction": _E.pad_friction,
                            "plate_friction": _P.friction},
                "source_params": {"tilt_deg": math.degrees(float(m["tilt"][e].item())),
                                  "grip_offset_mm": float(m["grip_off"][e].item() * 1000),
                                  "n_pass": float(m["n_pass"][e].item()),
                                  "sweep_dir": float(m["y_dir"][e].item())},
                "diagnostics": d,
            }
            rec = EpisodeRecord(meta=meta, arrays=to_arrays(buf, e))
            records.append((save_episode(rec, out_dir / "episodes" / f"{eid}.npz"), rec))
        del buf

    entries = [r.to_manifest_entry("x") for _, r in records]
    # `plan/03` §7 末尾：两种实现中至少留出**一种实现的全部 episode**
    # 作跨实现测试（对应 `02` §7 第 8 条：envelope 与是否使用工具无关）。
    splits = split_episode_entries(entries, seed=_a.seed,
                                   holdout_strategy_family=_a.holdout_family,
                                   holdout_implementation=_a.holdout_implementation)
    write_manifest(records, out_dir / "manifest.json", dataset_name="s3_wipe_source",
                   generator_git_sha=sha, splits=splits,
                   extra={"task": "wipe", "families": list(fams),
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
