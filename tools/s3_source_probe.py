"""S3 · 探针物体集的双板 source 采集（交互原语库）。

`plan/03` §2.4：这一组物体**只用于产生指令多样性，永不作为任务评估**。
它的选取依据是一套**独立于本项目三个任务**的交互分类学（Huang 的接触模式枚举 +
Bullock 的手中心描述子 + Lynch & Mason 的非抓握原语命名），判据是**张满分类学**，
不是"覆盖留出任务"。理由与被否掉的旧做法见 `log/decisions.md` D-41。

## 这个脚本的组织方式：原语 × 位点，不是"每个物体一段脚本"

物体只声明**它表面有哪些接触位点**（物体局部系的位置 + 外法向 + 切向），
原语只声明**贴上去之后怎么动**（法向力多大、切向是扫掠还是加力还是脉冲、
用一块板还是两块对置）。两者叉乘即可，加一个物体不需要写新的阶段机。

这不只是省代码。`plan/02` 的交互规格本身就是「region（位点）+ engage 方向（法向）
+ mode + mechanics」，脚本按同样的结构组织，采出来的数据与规格字段天然对齐。

## 阶段机

    approach → establish → manipulate → release

与抽屉采集器同构（`tools/s3_source_drawer.py`），并沿用它踩出来的全部教训：
P-30 接触归属按位置判、P-31 逐子步取力最大者、P-32 两块板起始位错开、
P-33 力控阻尼相对被操作物体、P-34 物理 300 Hz + 板 0.5 kg、P-36 摩擦按自己的下标取。

## 用法

    # 单个物体
    ./tools/run_remote.sh "PYTHONPATH=src /isaac-sim/python.sh \\
        tools/s3_source_probe.py --object block --envs 60 --batches 4 \\
        --out /tmp/s3_probe/block" probe_block

    # 八个物体并行（8 张卡）
    for i in 0 1 2 3 4 5 6 7; do
      obj=$(echo "block column slider plunger dial flap ridge slab" | cut -d' ' -f$((i+1)))
      IT_GPU=$i ./tools/run_remote.sh "... --object $obj ..." "probe_$obj" &
    done; wait
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_ap = argparse.ArgumentParser(description="S3 探针物体集 source 采集")
_ap.add_argument("--object", required=True)
_ap.add_argument("--envs", type=int, default=60)
_ap.add_argument("--batches", type=int, default=4)
_ap.add_argument("--out", default="/tmp/s3_probe")
_ap.add_argument("--seed", type=int, default=20260829)
_ap.add_argument("--primitive", default="all", help="逗号分隔的原语名，或 all")
_ap.add_argument("--holdout-primitive", default="",
                 help="划进 unseen_strategy_test 的原语；留空表示不留出")
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
from isaaclab.assets import Articulation, AssetBaseCfg, RigidObject  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg, ContactSensorCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from it import assets as A  # noqa: E402
from it import build_assets as B  # noqa: E402
from it.contact_attrib import (  # noqa: E402
    PLATE_PARTS,
    classify_plate_face,
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
PHASE_STEPS = (50, 60, 150, 50)
N_FRAMES = sum(PHASE_STEPS)
RISE_STEPS, CLOSE_STEPS = 32, 46
TOUCH_FRAC, RAMP_STEPS = 0.25, 18
STANDOFF = 0.130
SAFE_GAP = 0.026
PRE_GAP = 1.0 * MM
PLATE_T = B.PlateCfg().size[2]
MAX_VALID_FORCE = 200.0
MAX_PLATE_DIST = 1.3
#: 物体速度上限，超过就收力（见 run_batch 里的速度闸）。
#: 0.05 m/s × 3 s 的 manipulate ≈ 150 mm 行程，是一次示教该有的量级；
#: 取 0.10 时方块被推出 550 mm，那更像"把它推走"而不是"推移它"。
V_MAX = 0.05
#: 切向扫掠的往复周期（控制步）。60 步（1.2 s）时位置 PD 的跟踪滞后
#: 会产生约 20 N 的切向力，把板从接触面上甩开；120 步降到约 7 N。
SWEEP_PERIOD = 120.0
#: 接触点离目标位点多远仍算"打在这个位点上"。板半对角 21.5 mm，放宽到 45 mm。
SITE_TOL = 0.045


# ---------------------------------------------------------------- 位点与原语


@dataclass(frozen=True)
class Site:
    """物体表面的一个接触位点，全部在**物体局部系**。

    Attributes:
        pos: 位点中心。
        normal: 表面外法向。板贴上来之后沿 **−normal** 施力。
        sweep: 切向运动/切向力的方向，必须垂直于 normal。
        long: 板长边（局部 +X）对齐的方向；默认与 sweep 相同。
            圆柱面上通常要让长边**沿柱轴**才是线接触，此时两者不同。
        approach: 板从哪个方向飞进来，默认沿 +normal 直进。
            凹进去的位点（如柱塞端帽的内侧台肩）必须侧向进入，
            否则路径会穿过物体本身。
    """

    pos: tuple[float, float, float]
    normal: tuple[float, float, float]
    sweep: tuple[float, float, float]
    long: tuple[float, float, float] | None = None
    approach: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class Primitive:
    """一条交互原语：贴上位点之后怎么动。

    Attributes:
        sites: 用哪些位点。一个 = 单面接触，两个 = 双面。
        f_normal: 法向力幅值范围（N）。
        tan_mode: ``none`` / ``sweep``（切向位置扫掠）/ ``push``（切向加力）/
            ``pulse``（法向力通断，做通断接触）。
        tan_amp: sweep 时是位移幅值（m）；push 时是**相对摩擦锥**的比例
            （<1 保持 sticking，>1 推出摩擦锥变成 sliding）。
        sweep_frame: ``object`` 扫掠目标跟着物体走（贴着表面滑，如 rub）；
            ``world`` 扫掠目标锚在世界系（带着物体走，如 pinch_move）。
            两者的区别是实打实的：跟着物体扫等于原地不动，
            pinch_move 第一版就是这样，物体位移恒等于 0.0 mm。
        expect: 期望的物体响应，用于逐原语判成功。
            ``static`` 物体基本不动 / ``move`` 明显位移 / ``turn`` 明显转动 /
            ``joint+`` ``joint-`` 关节朝某个方向明显变化 / ``cycle`` 通断次数。
    """

    sites: tuple[str, ...]
    f_normal: tuple[float, float]
    tan_mode: str = "none"
    tan_amp: tuple[float, float] = (0.0, 0.0)
    sweep_frame: str = "object"
    #: 第二块板的切向方向取反。同向扫是"捏着挪走"，反向扫才是"捏着转"——
    #: 同一组位点靠这个开关区分两条原语，不必再定义一套反向的位点。
    opposed_sweep: bool = False
    #: 法向走**速度控制**而不是力控。纯力控推不动"几乎没有阻力"的物体——
    #: 卧柱的滚动阻力接近零，板给多大力它就跑多快，而板自己也要被同一个力
    #: 加速，质量比一定的情况下板永远追不上（实测 95% 的操作步脱手）。
    #: 这一档对应 Lynch & Mason 的 stable pushing：推的人按**速度**走，
    #: 接触力由物体的阻力自己决定。
    normal_velocity: float = 0.0
    expect: str = "static"


#: 交互原语库（`plan/03` §2.4.4）。名字沿用文献既有词汇。
PRIMITIVES: dict[str, Primitive] = {
    "press":       Primitive((), (4.0, 10.0), expect="static"),
    "push":        Primitive((), (4.0, 9.0), expect="move"),
    "slide_push":  Primitive((), (4.0, 8.0), "push", (1.02, 1.25), expect="move"),
    "rub":         Primitive((), (4.0, 9.0), "sweep", (0.020, 0.050), expect="static"),
    "shear":       Primitive((), (6.0, 12.0), "push", (0.25, 0.55), expect="static"),
    "poke":        Primitive((), (4.0, 9.0), "pulse", expect="cycle"),
    "pivot":       Primitive((), (5.0, 12.0), expect="turn"),
    # roll 不用切向扫掠：卧柱受侧向推力就滚，扫掠反而把它拖走。
    # 法向走速度控制，理由见 Primitive.normal_velocity。
    "roll":        Primitive((), (2.0, 5.0), normal_velocity=0.035, expect="turn"),
    "crank":       Primitive((), (4.0, 10.0), expect="joint"),
    "slide_along": Primitive((), (6.0, 14.0), expect="joint"),
    "hook_pull":   Primitive((), (6.0, 14.0), expect="joint"),
    "twist":       Primitive((), (4.0, 9.0), "push", (0.30, 0.70),
                         opposed_sweep=True, expect="turn"),
    "pinch_hold":  Primitive((), (4.0, 9.0), expect="static"),
    "pinch_move":  Primitive((), (5.0, 10.0), "sweep", (0.030, 0.060),
                         sweep_frame="world", expect="move"),
    "pinch_turn":  Primitive((), (5.0, 10.0), "sweep", (0.020, 0.050),
                         sweep_frame="world", opposed_sweep=True, expect="turn"),
}


@dataclass
class ObjectSpec:
    """一个探针物体：资产、可动体、位点表、承载的原语。"""

    cfg: Any
    articulated: bool
    body: str | None                  # articulation 里被操作的 link 名
    body_path: str                    # 接触 filter 的 prim 路径（相对 env）
    sites: dict[str, Site]
    prims: dict[str, tuple[str, ...]]  # 原语 -> 位点名
    init_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    joint: str | None = None
    #: 关节初值。滑块必须从**行程中点**起步，否则它停在 0 端，
    #: "往回推"这一档一开始就顶在限位上，实测 slide_along 关节位移恒为 0。
    joint_init: float = 0.0
    damping_nominal: tuple[float, float] = (0.0, 0.0)
    damping_heldout: tuple[tuple[float, float], ...] = ()
    cam_eye: tuple[float, float, float] = (0.42, -0.36, 0.28)
    cam_at: tuple[float, float, float] = (0.0, 0.0, 0.06)


def _s(pos, normal, sweep, long=None, approach=None):
    return Site(pos, normal, sweep, long, approach)


_BL = B.BlockCfg()
_CO = B.ColumnCfg()
_SL = B.SliderCfg()
_PL = B.PlungerCfg()
_DI = B.DialCfg()
_FL = B.FlapCfg()
_RI = B.RidgeCfg()
_RO = B.RollerCfg()
_BA = B.BallCfg()
_SB = B.SlabCfg()

_bx, _by, _bz = (v / 2 for v in _BL.size)

OBJECTS: dict[str, ObjectSpec] = {
    # --- 自由体：E1 平移 / E2 转动 ---
    "block": ObjectSpec(
        cfg=A.BLOCK_CFG, articulated=False, body=None, body_path="Block",
        init_pos=(0.0, 0.0, _bz),
        sites={
            "face_x":  _s((+_bx, 0, 0), (1, 0, 0), (0, 1, 0)),
            "face_nx": _s((-_bx, 0, 0), (-1, 0, 0), (0, 1, 0)),   # 与 face_x 同向，
            # 由 Primitive.opposed_sweep 决定要不要把第二块板反过来
            "top":     _s((0, 0, +_bz), (0, 0, 1), (1, 0, 0)),
            # 撬翻推**窄面**。先翻不先滑的条件是 μ·h > 半宽：
            # 推 60 mm 宽的面要 μ·h > 30 mm，推 45 mm 宽的面只要 > 22.5 mm。
            # μ=0.9（板与方块都设 0.9，规则 9 的 min 组合才取得到）、h=35 mm
            # -> 0.0315 > 0.0225，裕量 40%。推宽面时实测是先滑走。
            "high_y":  _s((0, +_by, +_bz * 0.75), (0, 1, 0), (1, 0, 0)),
        },
        # 不在自由方块上做 rub：板-方块与方块-地面同为 μ=0.9，
        # "板滑而方块不动"的力窗口只有 2 N 宽，实测方块被拖走 35 m。
        # P4 rub 由 slab / ridge / flap 三个固定物体承载，冗余规则仍满足。
        prims={"press": ("top",), "push": ("face_x",), "slide_push": ("face_x",),
               "shear": ("top",), "poke": ("face_x",),
               "pivot": ("high_y",), "pinch_hold": ("face_x", "face_nx"),
               "pinch_move": ("face_x", "face_nx"),
               "pinch_turn": ("face_x", "face_nx"), "twist": ("face_x", "face_nx")},
    ),
    "column": ObjectSpec(
        cfg=A.COLUMN_CFG, articulated=False, body=None, body_path="Column",
        init_pos=(0.0, 0.0, _CO.height / 2),
        sites={
            # 柱面位点：法向沿径向，切向沿圆周，板长边沿柱轴（线接触）
            # 推移位点放低（离地 25 mm）：μ·h = 0.7×0.025 < 半径 0.028 -> 先滑不先倒
            "side_low": _s((_CO.radius, 0, -_CO.height / 2 + 0.025), (1, 0, 0),
                           (0, 1, 0), (0, 0, 1)),
            # 撬翻位点放高（离地 70 mm）：μ·h = 0.7×0.07 > 0.028 -> 先倒不先滑
            "side_high": _s((_CO.radius, 0, _CO.height / 2 - 0.010), (1, 0, 0),
                            (0, 1, 0), (0, 0, 1)),
            "side_a": _s((_CO.radius, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
            "side_b": _s((-_CO.radius, 0, 0), (-1, 0, 0), (0, 1, 0), (0, 0, 1)),
            "top":    _s((0, 0, _CO.height / 2), (0, 0, 1), (1, 0, 0)),
        },
        # slide_push 与 push 用同一个低位点：把它推到打滑（M3 stick->slip）
        # 与稳定推移是同一处接触的两种力学状态，正是这两条原语的区别所在。
        # 加它是为了让 P3 不再只挂在方块一个物体上（冗余硬规则）。
        prims={"press": ("top",), "push": ("side_low",),
               "slide_push": ("side_low",), "poke": ("side_a",),
               "pivot": ("side_high",), "twist": ("side_a", "side_b"),
               "pinch_turn": ("side_a", "side_b")},
        cam_eye=(0.40, -0.34, 0.26), cam_at=(0.0, 0.0, 0.08),
    ),
    "roller": ObjectSpec(
        cfg=A.ROLLER_CFG, articulated=False, body=None, body_path="Roller",
        init_pos=(0.0, 0.0, _RO.radius),
        sites={
            # 推侧面中心高度 -> 滚；压顶 -> 压住；戳侧面 -> 通断
            "side":  _s((_RO.radius, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 0)),
            "top":   _s((0, 0, _RO.radius), (0, 0, 1), (1, 0, 0), (0, 1, 0)),
            # **两个平端面**（Ø60 的圆盘面，轴线沿 Y）。捏住两端沿轴向力封闭，
            # 是几何上与方块完全不同的一副对捏面：方块是两个平行的矩形侧面，
            # 这里是一根曲面柱体的两个圆端面。
            # 提起来 -> pinch_move；原地夹住 -> pinch_hold。
            "end_a": _s((0, _RO.length / 2, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0)),
            "end_b": _s((0, -_RO.length / 2, 0), (0, -1, 0), (0, 0, 1), (1, 0, 0)),
            # 同样是两端面，但切向取水平、两块板反向 -> 绕**竖直轴**偏转
            # （力臂是 140 mm 的端面间距，靠端面摩擦传力，F2/E2）。
            # 绕自身轴转在这个物体上就是 roll，不能拿来当 twist。
            "yaw_a": _s((0, _RO.length / 2, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1)),
            "yaw_b": _s((0, -_RO.length / 2, 0), (0, -1, 0), (1, 0, 0), (0, 0, 1)),
        },
        # 侧面只做 roll 和 poke：压顶会让圆柱侧向滑走、推侧面就是滚，
        # 与 roll 是同一件事——分成两个名字只会让标签自相矛盾。
        # 端面对捏（`end_*` / `yaw_*` 位点）**实测不成立**，已撤回：
        # 自由卧柱在两块板夹上来之前就先滚跑了，1020 条里 pinch_hold 只成功 2 条、
        # pinch_move 与 twist 各 0 条，物体位移 215~1399 mm。位点留在表里
        # 是为了让下一个人不必重试一遍。
        prims={"roll": ("side",), "poke": ("side",)},
        cam_eye=(0.34, -0.32, 0.20), cam_at=(0.0, 0.0, 0.03),
    ),
    "ball": ObjectSpec(
        cfg=A.BALL_CFG, articulated=False, body=None, body_path="Ball",
        init_pos=(0.0, 0.0, _BA.radius),
        sites={"side": _s((_BA.radius, 0, 0), (1, 0, 0), (0, 1, 0), (0, 1, 0)),
               "top":  _s((0, 0, _BA.radius), (0, 0, 1), (1, 0, 0), (0, 1, 0)),
               # 两侧对置、切向反向 -> 绕**竖直轴**搓转（F2 摩擦驱动）。
               # 立柱之外的第二个 twist 承载物体：立柱是竖直柱面，
               # 球是球面，几何上确实不同（冗余硬规则要的就是这个）。
               "side_a": _s((_BA.radius, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
               "side_b": _s((-_BA.radius, 0, 0), (-1, 0, 0), (0, 1, 0), (0, 0, 1))},
        prims={"roll": ("side",), "press": ("top",), "poke": ("side",),
               "twist": ("side_a", "side_b")},
        cam_eye=(0.32, -0.30, 0.20), cam_at=(0.0, 0.0, 0.035),
    ),
    # --- 受约束平移 E3 ---
    "slider": ObjectSpec(
        cfg=A.SLIDER_CFG, articulated=True, body="Block", body_path="Slider/Block",
        joint="BlockJoint", damping_nominal=(1.4, 2.8), joint_init=0.075,
        damping_heldout=((0.7, 1.1), (3.4, 4.6)),
        sites={
            # 挡片高出块顶，两个面都够得到：压 −X 面推走，压 +X 面拉回
            "tab_back":  _s((_SL.block[0] / 2 - 6 * MM - _SL.tab[0], 0,
                             _SL.block[2] / 2 + _SL.tab[2] / 2),
                            (-1, 0, 0), (0, 1, 0), None, (0, 0, 1)),
            "tab_front": _s((_SL.block[0] / 2 - 6 * MM, 0,
                             _SL.block[2] / 2 + _SL.tab[2] / 2),
                            (1, 0, 0), (0, 1, 0), None, (0, 0, 1)),
            "top":       _s((-0.012, 0, _SL.block[2] / 2), (0, 0, 1), (0, 1, 0)),
            # 挡片的**两个窄侧面**（10 mm 厚 × 34 mm 高，比块本身宽 10 mm，
            # 从两侧都够得到）。捏住它再沿导轨推 -> 力封闭的对捏 + 受约束平移，
            # 正是 P14 pinch-move 的 E3 那一支。
            # 挡片的 ±X 面不能拿来做 pinch_move：那两个面的法向就是导轨方向，
            # 捏住之后要移动的方向与法向重合，几何上不成立。
            "tab_side_a": _s((_SL.block[0] / 2 - 6 * MM - _SL.tab[0] / 2,
                              _SL.tab[1] / 2,
                              _SL.block[2] / 2 + _SL.tab[2] / 2),
                             (0, 1, 0), (1, 0, 0), (0, 0, 1)),
            "tab_side_b": _s((_SL.block[0] / 2 - 6 * MM - _SL.tab[0] / 2,
                              -_SL.tab[1] / 2,
                              _SL.block[2] / 2 + _SL.tab[2] / 2),
                             (0, -1, 0), (1, 0, 0), (0, 0, 1)),
        },
        prims={"slide_along": ("tab_front",), "hook_pull": ("tab_back",),
               "press": ("top",), "poke": ("tab_front",),
               "pinch_hold": ("tab_side_a", "tab_side_b"),
               "pinch_move": ("tab_side_a", "tab_side_b")},
        cam_eye=(0.34, -0.40, 0.26), cam_at=(-0.02, 0.0, 0.05),
    ),
    "plunger": ObjectSpec(
        cfg=A.PLUNGER_CFG, articulated=True, body="Rod", body_path="Plunger/Rod",
        joint="RodJoint", damping_nominal=(1.1, 2.2),
        damping_heldout=((0.5, 0.9), (2.7, 3.8)),
        sites={
            "cap_front": _s((_PL.rod_len / 2, 0, 0), (1, 0, 0), (0, 1, 0)),
            # 端帽内侧台肩：法向朝 −X，压它就是把杆往外拉。必须**侧向**进入。
            "cap_back":  _s((_PL.rod_len / 2 - _PL.cap_len - 0.004,
                             0.5 * (_PL.rod_radius + _PL.cap_radius), 0),
                            (-1, 0, 0), (0, 0, 1), None, (0, 1, 0)),
            "rod_side":  _s((0.02, _PL.rod_radius, 0), (0, 1, 0), (1, 0, 0), (1, 0, 0)),
        },
        prims={"slide_along": ("cap_front",), "hook_pull": ("cap_back",),
               "press": ("cap_front",), "poke": ("cap_front",)},
        cam_eye=(0.40, -0.36, 0.24), cam_at=(0.04, 0.0, 0.04),
    ),
    # --- 受约束转动 E4 ---
    "dial": ObjectSpec(
        cfg=A.DIAL_CFG, articulated=True, body="Disc", body_path="Dial/Disc",
        joint="DiscJoint", damping_nominal=(0.14, 0.28),
        damping_heldout=((0.06, 0.11), (0.34, 0.48)),
        sites={
            # 凸耳的圆周侧面：压它产生绕轴力矩
            "lug_side": _s((_DI.lug_offset, -_DI.lug_radius,
                            _DI.disc_thickness / 2 + _DI.lug_height / 2),
                           (0, -1, 0), (0, 0, 1), (0, 0, 1)),
            "rim_a":    _s((_DI.disc_radius, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
            "rim_b":    _s((-_DI.disc_radius, 0, 0), (-1, 0, 0), (0, -1, 0), (0, 0, 1)),
            "top":      _s((0, 0, _DI.disc_thickness / 2), (0, 0, 1), (1, 0, 0)),
        },
        # poke 放盘顶而不是凸耳侧面：Ø18 的凸耳被平板戳到的是棱，
        # 实测面接触度只有 0.64、32% 判成侧边。
        prims={"crank": ("lug_side",), "press": ("top",), "poke": ("top",)},
        cam_eye=(0.36, -0.32, 0.30), cam_at=(0.0, 0.0, 0.09),
    ),
    "flap": ObjectSpec(
        cfg=A.FLAP_CFG, articulated=True, body="Panel", body_path="Flap/Panel",
        # ⚠️ 这里的值在运行时写进关节，**盖过资产里的 joint_damping**。
        # 改了 build_assets 而没改这里 = 什么都没改（实测两次结果逐位相同）。
        joint="PanelJoint", damping_nominal=(1.4, 2.8),
        damping_heldout=((0.7, 1.1), (3.4, 4.6)),
        sites={
            # 板面上离铰链远近不同的两个位点：力臂不同，所需力不同
            "face_far":  _s((_FL.panel[0] / 2, +_FL.panel[1] / 3, 0), (1, 0, 0), (0, 0, 1)),
            "face_near": _s((_FL.panel[0] / 2, -_FL.panel[1] / 3, 0), (1, 0, 0), (0, 0, 1)),
            "face_back": _s((-_FL.panel[0] / 2, +_FL.panel[1] / 3, 0), (-1, 0, 0),
                            (0, 0, -1)),
        },
        # 不做 press/rub：自由摆动的门不是静止表面，一压就转，
        # 而那两条原语的 expect 是 static。P1/P4 由固定物体承载，冗余仍满足。
        # twist / pinch_turn 从转盘挪到这里：门板 10 mm 厚，两面对捏很自然，
        # 而转盘的薄轮缘被两块板径向夹住时会卡死（实测接触力冲到 279 N）。
        # 不做 twist / pinch_turn：门的开合方向恰好**就是**对捏轴，
        # 而对捏轴是力控的，位置指令表达不出"捏住往那边推"。
        # 那两条原语由 block 与 column 承载，冗余仍满足。
        prims={"crank": ("face_far",), "poke": ("face_back",)},
        cam_eye=(0.40, -0.38, 0.26), cam_at=(0.02, 0.04, 0.09),
    ),
    # --- 固定体 E5 ---
    "ridge": ObjectSpec(
        cfg=A.RIDGE_CFG, articulated=False, body=None, body_path="Ridge/Body",
        sites={
            "ridge_top":  _s((0, 0, _RI.base[2] / 2 + _RI.ridge_radius), (0, 0, 1),
                             (1, 0, 0), (0, 1, 0)),
            "ridge_side": _s((_RI.ridge_radius, 0, _RI.base[2] / 2 + _RI.ridge_radius),
                             (1, 0, 0), (0, 1, 0), (0, 1, 0)),
            "base_top":   _s((0.10, 0, _RI.base[2] / 2), (0, 0, 1), (1, 0, 0)),
        },
        prims={"press": ("ridge_top",), "rub": ("base_top",), "shear": ("base_top",),
               "poke": ("ridge_top",)},
        cam_eye=(0.36, -0.34, 0.22), cam_at=(0.0, 0.0, 0.03),
    ),
    "slab": ObjectSpec(
        cfg=A.SLAB_CFG, articulated=False, body=None, body_path="Slab/Board",
        init_pos=(0.0, 0.0, 0.08),
        sites={"top": _s((0, 0, _SB.size[2] / 2), (0, 0, 1), (1, 0, 0)),
               "edge": _s((_SB.size[0] / 2, 0, 0), (1, 0, 0), (0, 1, 0))},
        prims={"press": ("top",), "rub": ("top",), "shear": ("top",), "poke": ("top",)},
        cam_eye=(0.40, -0.34, 0.30), cam_at=(0.0, 0.0, 0.09),
    ),
}


# ---------------------------------------------------------------- 场景

SPEC = OBJECTS.get(_a.object)
if SPEC is None:
    raise SystemExit(f"未知物体 {_a.object}，可选：{sorted(OBJECTS)}")
PRIM_NAMES = tuple(sorted(SPEC.prims)) if _a.primitive == "all" \
    else tuple(x for x in _a.primitive.split(","))
for _p in PRIM_NAMES:
    if _p not in SPEC.prims:
        raise SystemExit(f"{_a.object} 不承载原语 {_p}，它有：{sorted(SPEC.prims)}")


def _contact_cfg(idx: int):
    """规则 8：filter 非空 + max_contact_data_count ≥1；filter 目标必须是刚体（规则 7）。"""
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Plate{idx}",
        track_pose=True, track_contact_points=True,
        max_contact_data_count_per_prim=MAX_CONTACTS,
        filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/{SPEC.body_path}"],
        update_period=0.0, history_length=0,
    )


@configclass
class SceneCfg(InteractiveSceneCfg):
    dome = AssetBaseCfg(prim_path="/World/dome",
                        spawn=sim_utils.DomeLightCfg(intensity=300.0,
                                                     color=(0.86, 0.89, 1.0)))
    sun = AssetBaseCfg(prim_path="/World/sun",
                       spawn=sim_utils.DistantLightCfg(intensity=1100.0, angle=5.0))
    # 台面摩擦 0.9：撬翻原语要求"先翻不先滑"，那是 μ 决定的（见 high_x 位点）
    ground = A.board_cfg(size=(2.4, 2.4, 0.08), friction=0.9)
    target = SPEC.cfg.replace(
        init_state=type(SPEC.cfg.init_state)(
            pos=SPEC.init_pos,
            **({"joint_pos": {SPEC.joint: SPEC.joint_init},
                "joint_vel": {SPEC.joint: 0.0}}
               if SPEC.articulated else {})),
    )
    plate0 = A.plate_cfg(0)
    plate1 = A.plate_cfg(1)
    contact0 = _contact_cfg(0)
    contact1 = _contact_cfg(1)
    if _a.video:
        cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Cam", update_period=0.0,
            height=_a.height, width=_a.width, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=26.0,
                                             clipping_range=(0.02, 30.0)),
            offset=CameraCfg.OffsetCfg(pos=SPEC.cam_eye,
                                       rot=look_at_quat(SPEC.cam_eye, SPEC.cam_at),
                                       convention="opengl"))


# ---------------------------------------------------------------- 工具


def _u(rng, lo, hi, n):
    return torch.from_numpy(rng.uniform(lo, hi, size=n).astype(np.float32))


def _v(t, device, n):
    return torch.tensor(t, dtype=torch.float32, device=device).expand(n, 3).contiguous()


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
    def __init__(self, n: int, device, n_state: int):
        z = lambda *s: torch.zeros(N_FRAMES, *s, device=device)  # noqa: E731
        self.phase = torch.zeros(N_FRAMES, n, dtype=torch.int8, device=device)
        self.progress = z(n)
        self.valid = torch.ones(N_FRAMES, n, dtype=torch.bool, device=device)
        self.state = z(n, n_state)
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
        self.face = torch.zeros(N_FRAMES, 2, n, MAX_CONTACTS, dtype=torch.int8,
                                device=device)
        self.nz = z(2, n, MAX_CONTACTS)      # 板局部系法向的 z 分量
        self.pz = z(2, n, MAX_CONTACTS)      # 板局部系接触点的 z 分量
        self.z_side = torch.zeros(2, n, device=device)
        self.nz_abs = torch.zeros(2, n, device=device)
        self.on_site = torch.zeros(N_FRAMES, 2, n, MAX_CONTACTS, dtype=torch.bool,
                                   device=device)
        self.src_pose = z(2, n, 7)
        self.src_vel = z(2, n, 6)
        self.src_tgt = z(2, n, 7)
        self.src_cmd = z(2, n, 6)
        self.src_pos_w = z(2, n, MAX_CONTACTS, 3)
        self.foreign = z(2)
        self.dropped = z(2)
        self.rot_err = z(2, n)
        self.pd_sat = z(2, n)


# ---------------------------------------------------------------- 采集


def run_batch(scene, sim, camera, prim_of_env, rng, device, batch):
    n = scene.cfg.num_envs
    tgt_obj = scene["target"]
    plates: list[RigidObject] = [scene["plate0"], scene["plate1"]]
    sensors = [scene["contact0"], scene["contact1"]]
    art: Articulation | None = tgt_obj if SPEC.articulated else None
    jid = art.find_joints(SPEC.joint)[0][0] if art is not None else 0
    bid = art.body_names.index(SPEC.body) if art is not None else 0
    mu_eff = min(B.PlateCfg().friction, 0.6)

    def obj_frame():
        if art is not None:
            return art.data.body_pos_w[:, bid, :], art.data.body_quat_w[:, bid, :]
        return tgt_obj.data.root_pos_w, tgt_obj.data.root_quat_w

    def obj_state():
        if art is not None:
            return art.data.joint_pos[:, jid: jid + 1]
        p, q = obj_frame()
        return torch.cat([p - scene.env_origins, q], dim=-1)

    # --- 逐 env 参数 ---
    n_site = torch.zeros(n, dtype=torch.long)
    site_idx = torch.zeros(n, 2, dtype=torch.long)
    f_norm = torch.zeros(n, 2, device=device)
    tan_amp = torch.zeros(n, device=device)
    site_names = sorted(SPEC.sites)
    for i, pname in enumerate(prim_of_env):
        pr = PRIMITIVES[pname]
        used = SPEC.prims[pname]
        n_site[i] = len(used)
        for k, sname in enumerate(used):
            site_idx[i, k] = site_names.index(sname)
        f_norm[i, :] = float(rng.uniform(*pr.f_normal))
        tan_amp[i] = float(rng.uniform(*pr.tan_amp)) if pr.tan_mode != "none" else 0.0
    site_idx = site_idx.to(device)
    n_site = n_site.to(device)
    tan_is = {m: torch.tensor([PRIMITIVES[p].tan_mode == m for p in prim_of_env],
                              device=device) for m in ("sweep", "push", "pulse")}
    sweep_world = torch.tensor([PRIMITIVES[p].sweep_frame == "world"
                                for p in prim_of_env], device=device)
    sweep_flip = torch.tensor([PRIMITIVES[p].opposed_sweep for p in prim_of_env],
                              device=device)
    v_cmd = torch.tensor([PRIMITIVES[p].normal_velocity for p in prim_of_env],
                         device=device)
    vel_mode = v_cmd > 0

    # 位点几何（物体局部系）按 env 取出
    def site_field(attr, fallback=None):
        out = torch.zeros(n, 2, 3, device=device)
        for k in range(2):
            for i in range(n):
                st = SPEC.sites[site_names[int(site_idx[i, k])]]
                v = getattr(st, attr) or (getattr(st, fallback) if fallback else None)
                out[i, k] = torch.tensor(v, device=device)
        return out

    s_pos = site_field("pos")
    s_nrm = site_field("normal")
    s_swp = site_field("sweep")
    s_lng = site_field("long", "sweep")
    s_app = site_field("approach", "normal")
    s_nrm = s_nrm / s_nrm.norm(dim=-1, keepdim=True)
    s_app = s_app / s_app.norm(dim=-1, keepdim=True)
    s_swp = s_swp / s_swp.norm(dim=-1, keepdim=True)
    s_swp[:, 1] = torch.where(sweep_flip.unsqueeze(-1), -s_swp[:, 1], s_swp[:, 1])

    # 物理变体（`plan/03` §7 的 unseen_physics_test）
    is_var = torch.from_numpy((rng.random(n) < _a.physics_variant_frac)
                              .astype(np.bool_)).to(device)
    damping = torch.zeros(n, device=device)
    if art is not None and SPEC.damping_nominal[1] > 0:
        damping = _u(rng, *SPEC.damping_nominal, n).to(device)
        lo, hi = SPEC.damping_heldout[batch % len(SPEC.damping_heldout)]
        damping = torch.where(is_var, _u(rng, lo, hi, n).to(device), damping)
        art.write_joint_damping_to_sim(damping.unsqueeze(-1), joint_ids=[jid])
        got = art.data.joint_damping[:, jid]
        if (got - damping).abs().max() > 1e-3:
            raise RuntimeError("关节阻尼没写进去，物理变体划分会变成假的")
    else:
        is_var = torch.zeros(n, dtype=torch.bool, device=device)

    # --- 复位 ---
    if art is not None:
        z1 = torch.zeros(n, 1, device=device)
        art.write_joint_state_to_sim(z1 + SPEC.joint_init, z1, joint_ids=[jid])
        art.set_joint_effort_target(z1, joint_ids=[jid])
    else:
        st0 = tgt_obj.data.default_root_state.clone()
        st0[:, :3] = scene.env_origins + torch.tensor(SPEC.init_pos, device=device)
        st0[:, 7:] = 0.0
        tgt_obj.write_root_state_to_sim(st0)
    scene.update(DT)

    def world_of(local):
        p, q = obj_frame()
        from isaaclab.utils.math import quat_apply
        return p.unsqueeze(1) + quat_apply(q.unsqueeze(1).expand(-1, 2, -1), local)

    def dir_world(local):
        from isaaclab.utils.math import quat_apply
        _, q = obj_frame()
        return quat_apply(q.unsqueeze(1).expand(-1, 2, -1), local)

    pds = [FloatingPD(pl, kp_pos=3000.0, kd_pos=110.0, kp_rot=6.0, kd_rot=0.025,
                      max_force=120.0, max_torque=4.0, kd_force=15.0) for pl in plates]

    quats = []
    for k in range(2):
        quats.append(quat_from_frame(-dir_world(s_nrm)[:, k], dir_world(s_lng)[:, k]))

    targets = []
    for k, plate in enumerate(plates):
        stp = plate.data.default_root_state.clone()
        # 起始位沿各自的进入方向拉远，并额外错开（P-32：起始位重叠会互相穿插）
        home = (world_of(s_pos)[:, k] + dir_world(s_app)[:, k]
                * (STANDOFF + k * 0.075))
        stp[:, :3] = home
        stp[:, 3:7] = quats[k]
        stp[:, 7:] = 0.0
        plate.write_root_state_to_sim(stp)
        targets.append(stp[:, :3].clone())
    for _ in range(6):
        if art is not None:
            art.set_joint_effort_target(torch.zeros(n, 1, device=device), joint_ids=[jid])
        for plate in plates:
            zw = torch.zeros(n, 1, 3, device=device)
            plate.set_external_force_and_torque(zw, zw, is_global=True)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(DT)

    n_state = obj_state().shape[-1]
    buf = Buffers(n, device, n_state)
    state0 = obj_state().clone()
    engage = torch.zeros(n, 2, device=device)
    prev_tgt = [t.clone() for t in targets]
    preview: list[np.ndarray] = []
    every = max(1, round(1.0 / (CONTROL_DT * _a.fps)))
    make_break = torch.zeros(n, device=device)
    prev_touch = torch.zeros(n, dtype=torch.bool, device=device)
    # manipulate 开始那一刻的贴合点与扫掠方向，供 world 系扫掠锚定
    anchor = torch.zeros(n, 2, 3, device=device)
    anchor_dir = torch.zeros(n, 2, 3, device=device)
    anchor_nrm = torch.zeros(n, 2, 3, device=device)

    frame = 0
    for phase_id, phase_len in enumerate(PHASE_STEPS):
        for kk in range(phase_len):
            p_site = world_of(s_pos)
            d_nrm, d_app, d_swp = dir_world(s_nrm), dir_world(s_app), dir_world(s_swp)
            for k in range(2):
                quats[k] = quat_from_frame(-d_nrm[:, k], dir_world(s_lng)[:, k])
            p_eng = p_site + d_nrm * (PLATE_T / 2 + PRE_GAP)
            p_safe = p_eng + d_app * SAFE_GAP
            p_home = p_site + d_app * (STANDOFF + torch.tensor(
                [0.0, 0.075], device=device).view(1, 2, 1))
            on = (torch.arange(2, device=device).view(1, 2) < n_site.view(n, 1))

            for k in range(2):
                tgt = targets[k]
                if phase_id == 0:                       # approach：飞到安全距离
                    t0, t1 = (0.0, 0.55) if k == 0 else (0.45, 0.95)
                    u = (kk + 1) / phase_len
                    a = min(max((u - t0) / (t1 - t0), 0.0), 1.0)
                    tgt[:] = p_home[:, k] + (p_safe[:, k] - p_home[:, k]) * a
                elif phase_id == 1:                     # establish：贴上去，轻接触
                    b = min(max((kk + 1 - RISE_STEPS)
                                / max(CLOSE_STEPS - RISE_STEPS, 1), 0.0), 1.0)
                    tgt[:] = p_safe[:, k] + (p_eng[:, k] - p_safe[:, k]) * b
                    touch = TOUCH_FRAC * min(
                        max((kk + 1 - CLOSE_STEPS) / max(phase_len - CLOSE_STEPS, 1),
                            0.0), 1.0)
                    # 速度控制档全程不加力：贴合那一下的冲量正是把滚柱打飞的
                    # 原因（0.5 m/s 的初速在 1.05 N·s/m 阻尼下能滑 165 mm，
                    # 而推板整个 manipulate 才前进 105 mm）。
                    engage[:, k] = torch.where(vel_mode,
                                               torch.zeros_like(engage[:, k]), touch)
                elif phase_id == 2:                     # manipulate：执行原语
                    if kk == 0:
                        # 速度档从**板实际所在**位置起算推进，不从名义贴合点，
                        # 否则残余的位置误差会在第一步变成一个阶跃指令。
                        anchor[:, k] = torch.where(
                            vel_mode.unsqueeze(-1),
                            plates[k].data.root_pos_w, p_eng[:, k])
                        anchor_dir[:, k] = d_swp[:, k]
                        anchor_nrm[:, k] = d_nrm[:, k]
                    full = TOUCH_FRAC + (1 - TOUCH_FRAC) * min((kk + 1) / RAMP_STEPS, 1.0)
                    duty = torch.where(
                        tan_is["pulse"],
                        ((kk % 16) < 6) * torch.ones(n, device=device),
                        torch.ones(n, device=device))
                    engage[:, k] = full * duty
                    base = plates[k].data.root_pos_w.clone()
                    # 法向由力控接管，切向按原语走位置扫掠
                    ph = 2.0 * math.pi * (kk + 1) / SWEEP_PERIOD
                    # object 系：基点跟着物体走 -> 板贴着表面滑（rub / roll）
                    # world  系：基点锚在开始接触的位置 -> 板带着物体走（pinch_*）
                    # 速度控制档：目标沿法向匀速推进，接触力由物体阻力决定
                    base = torch.where(
                        vel_mode.unsqueeze(-1),
                        # 推进方向在**贴合那一刻**锁定，不跟着物体转。
                        # 物体系法向会随物体一起转，滚柱转过 178° 时推进
                        # 方向也跟着转了 178°——人推东西时手不会跟着转。
                        anchor[:, k] - anchor_nrm[:, k]
                        * (v_cmd * CONTROL_DT * (kk + 1)).unsqueeze(-1),
                        base)
                    b0 = torch.where(sweep_world.unsqueeze(-1),
                                     anchor[:, k], p_eng[:, k])
                    dsw = torch.where(sweep_world.unsqueeze(-1),
                                      anchor_dir[:, k], d_swp[:, k])
                    off = (tan_amp * math.sin(ph)).unsqueeze(-1) * dsw
                    tgt[:] = torch.where(tan_is["sweep"].unsqueeze(-1), b0 + off, base)
                    # pulse：不加力的半程把板退开，让接触真的断掉
                    tgt[:] = torch.where((tan_is["pulse"] & (duty < 0.5)).unsqueeze(-1),
                                         p_eng[:, k] + d_app[:, k] * 0.012, tgt)
                else:                                   # release：退出
                    engage[:, k] = 0.0
                    a = min((kk + 1) / 14.0, 1.0)
                    b = min(max((kk + 1 - 14) / max(phase_len - 14, 1), 0.0), 1.0)
                    tgt[:] = (p_safe[:, k] + (p_home[:, k] - p_safe[:, k]) * b) * a \
                        + p_eng[:, k] * (1 - a)

                tgt[:] = torch.where(on[:, k].unsqueeze(-1), tgt, p_home[:, k])
                engage[:, k] = torch.where(on[:, k], engage[:, k],
                                           torch.zeros_like(engage[:, k]))
                buf.src_cmd[frame, k, :, :3] = tgt - prev_tgt[k]
                prev_tgt[k] = tgt.clone()

            best_fn = torch.zeros(2, n, device=device)
            best: list[dict | None] = [None, None]
            # **速度闸**。恒定的推力对自由体是发散的：0.25 kg 的方块被 6 N 推
            # 3 秒，扣掉地面摩擦仍有 18 m/s²，跑出去 80 m。实测第一版就是这样，
            # 物体位移 33 米、转了 179°。
            #
            # 真实的推移是准静态的：推的人看着物体速度收力。这里照做——
            # 物体速度超过 v_max 就按比例收指令力。对本来就不动的物体
            # （press/rub/shear）速度恒为零，这一项不起作用。
            v_obj = (art.data.joint_vel[:, jid].abs() if art is not None
                     else tgt_obj.data.root_lin_vel_w.norm(dim=-1))
            # 软伺服，不是硬切。第一版用 clamp(1-(v-V)/V, 0, 1)，速度一超标
            # 力就归零，物体减速、力又回来——平均接触力只剩 0.8 N，接触若有若无，
            # 连"接触落在板哪一面"都判不稳。按比例收力则接触始终压得住。
            brake = (V_MAX / v_obj.clamp_min(V_MAX)).clamp(0.25, 1.0)
            for _sub in range(DECIMATION):
                if art is not None:
                    # P-21：力矩指令会一直保持，每步都要显式清零
                    art.set_joint_effort_target(torch.zeros(n, 1, device=device),
                                                joint_ids=[jid])
                for k, plate in enumerate(plates):
                    amp = (f_norm[:, k] * engage[:, k] * brake).unsqueeze(-1)
                    # 速度控制档不给前馈力：位置推进已经在推了，再叠一份力
                    # 等于推两次，实测滚柱照样跑出 400 mm、95% 的步脱手。
                    ff = torch.where(vel_mode.unsqueeze(-1),
                                     torch.zeros_like(d_nrm[:, k]), -d_nrm[:, k] * amp)
                    # 切向加力：按摩擦锥的比例给，<1 保持 sticking，>1 推出锥外
                    ftan = (tan_amp * mu_eff * f_norm[:, k] * engage[:, k]
                            * brake).unsqueeze(-1)
                    ff = ff + torch.where(tan_is["push"].unsqueeze(-1),
                                          d_swp[:, k] * ftan, torch.zeros_like(ff))
                    # 沿接触**法向**力控、切向仍然位置 PD。用 force_mask 只能
                    # 整轴切换，而探针物体的法向不与世界轴对齐——把三轴全设成
                    # 力控会让切向指令彻底失效（pinch_move 物体位移恒为 0）。
                    live = ((engage[:, k] > 0) & ~vel_mode).unsqueeze(-1)
                    f, tq = pds[k].compute(
                        targets[k], quats[k], ff_force=ff,
                        force_dir=torch.where(live, -d_nrm[:, k],
                                              torch.zeros_like(d_nrm[:, k]))
                        if bool(live.any()) else None)
                    buf.pd_sat[frame, k] += (
                        f.squeeze(1).abs() >= pds[k].max_force - 1e-3).any(dim=-1).float()
                    plate.set_external_force_and_torque(f, tq, is_global=True)
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(DT)
                # P-31：接触报告逐子步断续，取法向力最大的那一子步作代表
                for k, plate in enumerate(plates):
                    cp = extract_contact_points_padded(
                        sensors[k], DT, body_pos_w=plate.data.root_pos_w,
                        max_points=MAX_CONTACTS, own_radius=0.06)
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
                        best[k]["valid"] = torch.where(take, cp["valid"],
                                                       best[k]["valid"])
                        best_fn[k] = torch.maximum(best_fn[k], tot)

            # ---------------- 记录 ----------------
            o_pos, o_quat = obj_frame()
            buf.phase[frame] = phase_id
            buf.state[frame] = obj_state()
            buf.obj_pos[frame], buf.obj_quat[frame] = o_pos, o_quat
            buf.progress[frame] = (frame + 1) / N_FRAMES

            ok = torch.ones(n, dtype=torch.bool, device=device)
            touch = torch.zeros(n, dtype=torch.bool, device=device)
            for k, plate in enumerate(plates):
                cp = best[k]
                pl = to_local(cp["positions"], o_pos, o_quat)
                buf.pos_obj[frame, k] = pl * cp["valid"].unsqueeze(-1)
                buf.nrm_obj[frame, k] = rotate_inverse(o_quat, cp["normals"])
                buf.fri_obj[frame, k] = rotate_inverse(o_quat, cp["friction_forces"])
                buf.fn[frame, k] = cp["normal_forces"]
                buf.sep[frame, k] = cp["separations"]
                buf.cvalid[frame, k] = cp["valid"]
                buf.src_pos_w[frame, k] = cp["positions"]
                buf.mode[frame, k] = classify_contact_mode_padded(
                    cp["normal_forces"], cp["friction_forces"], cp["separations"],
                    cp["valid"], mu=mu_eff)
                buf.nz[frame, k] = rotate_inverse(
                    plate.data.root_quat_w, cp["normals"])[..., 2]
                buf.pz[frame, k] = to_local(cp["positions"], plate.data.root_pos_w,
                                            plate.data.root_quat_w)[..., 2]
                # 接触有没有打在**指定的位点**上——这是"做法对不对"的判据
                # 扫掠类原语的接触点**本来就要在表面上移动**，容差要把扫幅算进去
                tol = SITE_TOL + tan_amp.unsqueeze(-1)
                buf.on_site[frame, k] = cp["valid"] & (
                    (pl - s_pos[:, k].unsqueeze(1)).norm(dim=-1) < tol)
                buf.src_pose[frame, k] = torch.cat(
                    [plate.data.root_pos_w, plate.data.root_quat_w], dim=-1)
                buf.src_vel[frame, k] = torch.cat(
                    [plate.data.root_lin_vel_w, plate.data.root_ang_vel_w], dim=-1)
                buf.src_tgt[frame, k, :, :3] = targets[k]
                buf.src_tgt[frame, k, :, 3:] = quats[k]
                dq = (plate.data.root_quat_w * quats[k]).sum(dim=-1).abs().clamp(max=1.0)
                buf.rot_err[frame, k] = 2.0 * torch.acos(dq)
                touch |= cp["normal_forces"].abs().sum(dim=1) > 0.05
                ok &= torch.isfinite(plate.data.root_pos_w).all(dim=-1)
                ok &= (plate.data.root_pos_w - scene.env_origins).norm(dim=-1) \
                    < MAX_PLATE_DIST
            make_break += (touch & ~prev_touch).float()
            prev_touch = touch
            buf.valid[frame] = ok & torch.isfinite(buf.state[frame]).all(dim=-1) \
                & (buf.fn[frame].abs().sum(dim=(0, 2)) <= MAX_VALID_FORCE)

            if camera is not None and frame % every == 0:
                sim.render()
                camera.update(CONTROL_DT)
                preview.append(camera.data.output["rgb"][0, ..., :3]
                               .detach().cpu().numpy().astype(np.uint8))
            frame += 1

    # 面归属在**整条 episode 上**判一次，不逐帧判。
    #
    # 面还是棱由 |n_z| 决定（P-35：位置在两表面之间浮动，判不了）；
    # 哪一面由接触点的加权 z 均值决定（法向的正负号不可信，见
    # `contact_attrib.classify_plate_face` 的说明）。但**逐帧**的 z 均值在
    # 接触很轻时会在零附近抖，于是同一次贴合被判得一半正面一半背面——
    # 实测 push 这类物体在动的原语，工作面占比因此只有 37%。
    # 脚本化的板在一条 episode 里只会贴住同一个面，所以整条判一次才是对的。
    w_all = buf.fn.abs() * buf.cvalid
    z_side = ((buf.pz * w_all).sum(dim=(0, 3))
              / w_all.sum(dim=(0, 3)).clamp_min(1e-9))          # (2, n)
    buf.z_side = z_side
    buf.nz_abs = ((buf.nz.abs() * w_all).sum(dim=(0, 3))
                  / w_all.sum(dim=(0, 3)).clamp_min(1e-9))
    work = (z_side >= 0).unsqueeze(0).unsqueeze(-1)
    buf.face[:] = torch.where(work, torch.zeros_like(buf.face),
                              torch.ones_like(buf.face))
    buf.face[buf.nz.abs() <= 0.7] = 2
    buf.face[~buf.cvalid] = -1

    n_cut = float(buf.dropped.sum())
    n_keep = float(buf.cvalid.sum())
    print(f"  接触点：保留 {n_keep:.0f}，离所有板都远而丢弃 {float(buf.foreign.sum()):.0f}，"
          f"超上限截掉 {n_cut:.0f}", flush=True)
    if n_cut > 0.01 * max(n_keep, 1.0):
        raise RuntimeError(f"{n_cut:.0f} 个接触点被静默截掉（P-03），调大 MAX_CONTACTS")

    meta = dict(state0=state0, f_norm=f_norm, tan_amp=tan_amp, damping=damping,
                is_var=is_var, make_break=make_break, site_idx=site_idx,
                site_names=site_names, n_site=n_site)
    return buf, meta, preview


# ---------------------------------------------------------------- 判定与落盘


def diagnostics(buf: Buffers, m: dict, e: int, prim: str) -> dict[str, Any]:
    fn = buf.fn[:, :, e, :].abs()
    total = float(fn.sum().item())
    manip = buf.phase[:, e] == 2
    # 「有没有接到该接的地方」只在**最初建立接触的那几帧**上量。
    #
    # 拿整段去量是错的，而且错法与原语有关：撬翻会把物体翻 150°、
    # 滚动更是**按定义**让接触点在物体表面上连续迁移——那是 P7 这一格
    # 的全部内容，不是做法不对。用"最初接触的 10 帧"既能抓住"接错地方"，
    # 又不会把物理本身当成错误。
    first = (fn.sum(dim=(1, 2)) > 0.05).nonzero().flatten()
    eng = torch.zeros_like(manip)
    if first.numel():
        eng[first[:10]] = True
    else:
        eng = buf.phase[:, e] == 1
    in_contact = fn.sum(dim=(1, 2)) > 0.05

    def share(codes, table):
        return {nm: (float((fn * (codes == i)).sum().item() / total) if total > 0 else 0.0)
                for i, nm in enumerate(table)}

    st = buf.state[:, e]
    if SPEC.articulated:
        d_state = float((st[:, 0] - m["state0"][e, 0]).abs().max().item())
        d_signed = float((st[:, 0] - m["state0"][e, 0])[manip].abs().max().item())
        d_move = d_rot = 0.0
    else:
        d_move = float((st[:, :3] - m["state0"][e, :3]).norm(dim=-1).max().item())
        dq = (st[:, 3:] * m["state0"][e, 3:]).sum(-1).abs().clamp(max=1.0)
        d_rot = float((2 * torch.acos(dq)).max().item())
        d_state = d_signed = 0.0

    return {
        "contact_frame_fraction": float(in_contact.float().mean().item()),
        "manip_no_contact_fraction": float(
            ((manip & ~in_contact).sum() / max(int(manip.sum().item()), 1)).item()),
        "on_site_force_share": float(
            (fn * buf.on_site[:, :, e, :])[eng].sum().item()
            / max(float(fn[eng].sum().item()), 1e-9)),
        "plate_face_force_share": share(buf.face[:, :, e, :], PLATE_PARTS),
        "mode_share": share(buf.mode[:, :, e, :],
                            ("no_contact", "sticking", "sliding", "separating")),
        "mean_contact_force_N": float(fn.sum(dim=(1, 2)).mean().item()),
        "peak_point_force_N": float(fn.max().item()),
        "mean_contacts_per_frame": float(
            buf.cvalid[:, :, e, :].float().sum(dim=(1, 2)).mean().item()),
        "invalid_frame_fraction": float(1.0 - buf.valid[:, e].float().mean().item()),
        "pd_saturation_fraction": float((buf.pd_sat[:, :, e] > 0).float().mean().item()),
        "mean_orientation_error_deg": float(
            torch.rad2deg(buf.rot_err[:, :, e]).mean().item()),
        "object_translation_mm": d_move * 1000.0,
        "object_rotation_deg": math.degrees(d_rot),
        "joint_change": d_state,
        "joint_change_manip": d_signed,
        "make_break_count": float(m["make_break"][e].item()),
        # 判"哪一面"用的那个量本身。正 = 接触落在工作面一侧。
        "plate_z_side_mm": float(buf.z_side[:, e].abs().max().item() * 1000.0
                                 * (1.0 if buf.z_side[0, e] >= 0 else -1.0)),
        "face_normal_align": float(buf.nz_abs[:, e].max().item()),
        "primitive": prim,
    }


#: 逐原语的期望物体响应。**不同原语必须用不同判据**——用统一的"任务完成"判据
#: 会把 press（物体本来就不该动）判成失败，把 push（物体该动）判成成功一次了事。
def judge(d: dict, prim: str) -> tuple[bool, list[str]]:
    fails = []
    # pulse 类原语（poke）**本来就该断开接触**，用同一条脱手判据罚它是错的
    if PRIMITIVES[prim].tan_mode != "pulse" and d["manip_no_contact_fraction"] > 0.35:
        fails.append("操作阶段脱手过多")
    if d["on_site_force_share"] < 0.70:
        fails.append("接触没打在指定位点上")
    # 判据是**面接触还是棱接触**，不是"正面还是背面"。
    #
    # 正面/背面这个区分在轻接触下测不出来：它靠接触点在板局部系的 z 的正负，
    # 而板半厚只有 1.5 mm，接触力 1 N 左右时那个量在零附近抖（实测同一批
    # episode 的加权均值只有 +0.4 mm）。而且脚本化的板从不掉头，
    # "用背面接触"在几何上根本到不了——测一个到不了的东西没有意义。
    #
    # 真正要防的是"拿边角在蹭"（D-34 那类问题），那由法向与板面法向的
    # 夹角判，稳得多：实测正常面接触 |n_z| = 0.9985。
    if d["face_normal_align"] < 0.70:
        fails.append("拿边角在蹭而不是用平面接触")
    if d["mean_orientation_error_deg"] > 20.0:
        fails.append("板没有保持指令姿态")
    if d["invalid_frame_fraction"] > 0.10:
        fails.append("脏帧过多")

    exp = PRIMITIVES[prim].expect
    if exp == "static":
        # 关节物体的位姿是常量，"没动"必须查关节，否则一扇被推开 100° 的门
        # 也会因为 translation/rotation 都是 0 而判成"静止"。
        if (d["object_translation_mm"] > 12.0 or d["object_rotation_deg"] > 8.0
                or d["joint_change"] > 0.05):
            fails.append("物体本不该明显移动")
    elif exp == "move":
        # 与 "turn" 分支同理：受约束平移的物体（滑轨块、柱塞）沿导轨走了
        # 46 mm，而它的**世界位姿**是常量——只查 translation 会把一条完全
        # 正确的"捏住沿导轨推"判成"物体没有被推动"。`02` §3.1 把 E3
        # （受约束平移）和 E1（自由平移）都算作"物体被移动"。
        if d["object_translation_mm"] < 8.0 and d["joint_change"] < 0.02:
            fails.append("物体没有被推动")
        # 推移应当是**推着走**，不是把它推翻。翻掉了接触拓扑完全变样，
        # 那条轨迹不该被当成 push/slide_push 的示教。
        if d["object_rotation_deg"] > 60.0:
            fails.append("物体被推翻而不是推移")
    elif exp == "turn":
        if d["object_rotation_deg"] < 8.0 and d["joint_change"] < 0.12:
            fails.append("物体没有被转动")
    elif exp == "joint":
        if d["joint_change_manip"] < 0.015:
            fails.append("关节没有明显变化")
    elif exp == "cycle":
        if d["make_break_count"] < 3:
            fails.append("通断次数不足")
    return (not fails), fails


def to_arrays(buf: Buffers, e: int) -> dict[str, np.ndarray]:
    """前缀分层同 ``it.records``：``source/*`` 只作追查，世界系一律进 source/。"""
    cpu = lambda t: t.detach().cpu().numpy()  # noqa: E731
    out = {
        "phase": cpu(buf.phase[:, e]),
        "progress": cpu(buf.progress[:, e]).astype(np.float32),
        "valid_frame": cpu(buf.valid[:, e]),
        "object/state": cpu(buf.state[:, e]).astype(np.float32),
        "source/object_pos_w": cpu(buf.obj_pos[:, e]).astype(np.float32),
        "source/object_quat_w": cpu(buf.obj_quat[:, e]).astype(np.float32),
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
        out[f"{c}/on_site"] = cpu(buf.on_site[:, k, e])
        out[f"{s}/root_pose"] = cpu(buf.src_pose[:, k, e]).astype(np.float32)
        out[f"{s}/root_velocity"] = cpu(buf.src_vel[:, k, e]).astype(np.float32)
        out[f"{s}/target_pose"] = cpu(buf.src_tgt[:, k, e]).astype(np.float32)
        out[f"{s}/cmd_delta"] = cpu(buf.src_cmd[:, k, e]).astype(np.float32)
        out[f"{s}/contact_pos_w"] = cpu(buf.src_pos_w[:, k, e]).astype(np.float32)
        out[f"{s}/contact_plate_face"] = cpu(buf.face[:, k, e])
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


def _fs(sub, name):
    return float(np.mean([m["diagnostics"]["plate_face_force_share"][name] for m in sub]))


def report(out_dir: Path, records, splits) -> None:
    lines: list[str] = []
    add = lines.append
    metas = [r.meta for _, r in records]
    ok = [m for m in metas if m["success"]]
    add(f"S3 探针物体 `{_a.object}` · {len(metas)} episode，"
        f"成功 {len(ok)} ({100.0 * len(ok) / max(len(metas), 1):.1f}%)")
    add("")
    add(f"{'原语':<14}{'条数':>5}{'成功':>6}{'脱手%':>8}{'打中位点%':>10}"
        f"{'面接触度':>9}{'工作面%':>9}{'侧边%':>8}{'力N':>7}{'峰值N':>8}"
        f"{'位移mm':>9}{'转角°':>8}{'关节Δ':>8}")
    for pn in sorted({m["strategy_family"] for m in metas}):
        sub = [m for m in metas if m["strategy_family"] == pn]
        g = lambda k: float(np.mean([m["diagnostics"][k] for m in sub]))  # noqa: E731
        add(f"{pn:<14}{len(sub):>5}{sum(m['success'] for m in sub):>6}"
            f"{100 * g('manip_no_contact_fraction'):>8.1f}"
            f"{100 * g('on_site_force_share'):>10.1f}"
            f"{float(np.mean([m['diagnostics']['face_normal_align'] for m in sub])):>9.3f}"
            f"{100 * _fs(sub, 'work_face'):>9.1f}{100 * _fs(sub, 'edge'):>8.1f}"
            f"{g('mean_contact_force_N'):>7.2f}{g('peak_point_force_N'):>8.2f}"
            f"{g('object_translation_mm'):>9.1f}{g('object_rotation_deg'):>8.1f}"
            f"{g('joint_change'):>8.3f}")
    add("")
    add("接触模式分布（按法向力加权，`plan/02` §3.4）")
    for nm in ("no_contact", "sticking", "sliding", "separating"):
        v = float(np.mean([m["diagnostics"]["mode_share"][nm] for m in metas]))
        add(f"  {nm:<14}{100 * v:>7.2f}%")
    add("")
    for key in ("invalid_frame_fraction", "pd_saturation_fraction",
                "mean_orientation_error_deg", "mean_contacts_per_frame",
                "plate_z_side_mm", "face_normal_align"):
        add(f"  {key:<30}{float(np.mean([m['diagnostics'][key] for m in metas])):>9.4f}")
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
            add(f"  {r:<24}{c:>5}")
    text = "\n".join(lines) + "\n"
    (out_dir / "report.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)


def main() -> int:
    out_dir = Path(_a.out)
    (out_dir / "episodes").mkdir(parents=True, exist_ok=True)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=DT, device="cuda:0",
        physx=sim_utils.PhysxCfg(gpu_max_rigid_contact_count=2 ** 22,
                                 gpu_max_rigid_patch_count=2 ** 20)))
    scene = InteractiveScene(SceneCfg(num_envs=_a.envs, env_spacing=1.8,
                                      replicate_physics=True))
    sim.reset()
    camera: Camera | None = scene["cam"] if _a.video else None
    device, rng, sha = sim.device, np.random.default_rng(_a.seed), _git_sha()

    records: list[tuple[str, EpisodeRecord]] = []
    for b in range(_a.batches):
        prim_of = [PRIM_NAMES[(b * _a.envs + i) % len(PRIM_NAMES)] for i in range(_a.envs)]
        print(f"\n=== {_a.object} batch {b + 1}/{_a.batches} · {_a.envs} env ===",
              flush=True)
        buf, m, preview = run_batch(scene, sim, camera, prim_of, rng, device, b)
        if camera is not None:
            write_preview(out_dir / "videos" / f"{_a.object}_{prim_of[0]}_b{b}.mp4",
                          preview)
        for e in range(_a.envs):
            pn = prim_of[e]
            d = diagnostics(buf, m, e, pn)
            good, fails = judge(d, pn)
            eid = f"{_a.object}-{pn}-b{b:02d}e{e:03d}"
            meta = {
                "schema_version": SCHEMA_VERSION, "episode_id": eid,
                "task": f"probe_{_a.object}", "source_embodiment": "two_dynamic_plates",
                "strategy_family": pn, "strategy_variant": f"b{b:02d}e{e:03d}",
                "physics_variant": "heldout_damping" if bool(m["is_var"][e]) else "nominal",
                "success": good, "failure_reasons": fails,
                "seed": int(_a.seed + b * _a.envs + e),
                "control_hz": 1.0 / CONTROL_DT, "physics_hz": 1.0 / DT,
                "phase_names": list(PHASE_NAMES), "phase_steps": list(PHASE_STEPS),
                "generator_git_sha": sha,
                "probe_object": _a.object,
                "primitive": pn,
                "sites": [m["site_names"][int(m["site_idx"][e, k])]
                          for k in range(int(m["n_site"][e]))],
                "physics": {"joint_damping": float(m["damping"][e].item()),
                            "plate_mass_kg": B.PlateCfg().mass,
                            "plate_friction": B.PlateCfg().friction},
                "source_params": {"f_normal_N": [float(x) for x in m["f_norm"][e].tolist()],
                                  "tangential_amp": float(m["tan_amp"][e].item())},
                "diagnostics": d,
            }
            rec = EpisodeRecord(meta=meta, arrays=to_arrays(buf, e))
            records.append((save_episode(rec, out_dir / "episodes" / f"{eid}.npz"), rec))
        del buf

    entries = [r.to_manifest_entry("x") for _, r in records]
    splits = split_episode_entries(
        entries, seed=_a.seed,
        holdout_strategy_family=_a.holdout_primitive or None)
    write_manifest(records, out_dir / "manifest.json",
                   dataset_name=f"s3_probe_{_a.object}", generator_git_sha=sha,
                   splits=splits,
                   extra={"probe_object": _a.object, "primitives": list(PRIM_NAMES),
                          "holdout_strategy_family": _a.holdout_primitive or None})
    report(out_dir, records, splits)
    return 0


try:
    code = main()
except Exception:
    import traceback
    traceback.print_exc()
    code = 1
sys.stdout.flush()
os._exit(code)   # P-19：SimulationApp.close() 会挂起
