"""探针物体集的**场景事实**：物体怎么摆、刚体在哪个 prim、台面什么参数。

这个模块存在的理由是 P-72 的通用形状——**同一组事实有两份实现，早晚会分家**。

这些事实原来只写在 `tools/s3_source_probe.py::OBJECTS` 里，而 S6 的 E-I 环境
（`it.envs.interaction`）需要**同一组**事实才能复现"指令是在什么物理条件下产生的"。
第一版 E-I 环境自己另写了一套，两处都不报错，合起来有两个后果：

1. **没有台面**。采集侧的场景里有一张 μ=0.9 的台面，物体按 ``init_pos`` 放在上面；
   E-I 环境只建了灯、物体、执行器，而 ``block``/``column``/``ball``/``roller``
   都是 ``disable_gravity=False`` 的自由刚体——开局自由落体。指令说的是
   "在桌面上推方块"，环境里根本没有桌面。
2. **接触 filter 指到了错的 prim**。``ridge`` 的刚体在 ``Ridge/Body``、
   ``slab`` 在 ``Slab/Board``，而 E-I 环境硬写成物体根 prim。那正是 P-17：
   ``net_forces_w`` 照常工作，**接触位置与摩擦力全部静默失效**。

所以两边现在都从这里取。**本模块不依赖 pxr 也不依赖 Isaac**（与 `geom_cfg` 同样的
理由），本机可读、可测。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from it import geom_cfg as B

MM = 0.001

#: 台面摩擦。撬翻原语要求"先翻不先滑"，那是 μ 决定的（`s3_source_probe` 的 high_x
#: 位点）。**规则 9**：所有材质 ``friction_combine_mode="min"``，所以两侧都要 0.9
#: 才取得到 0.9。
TABLE_FRICTION = 0.9
#: 台面厚度。顶面恰好在 z=0，所有 ``init_pos`` 的 z 都是相对这个面给的。
TABLE_THICKNESS = 0.08

#: 采集侧的台面是 2.4 m 见方（`s3_source_probe` 的 `SceneCfg.ground`）。E-I 训练要
#: 2048 个 env 同时存在，这里缩到 1.2 m 见方。**只有外廓不同**：材质、厚度、顶面高度
#: 与 kinematic 属性逐字相同，而物体在探针原语里的最大行程远小于 0.6 m
#: （`FloatingBaseAction.pos_limit` 就是 0.6）。环境另有一条诊断统计"物体跑出台面"
#: 的 env 数——**缩小台面若真的造成了影响，那条诊断会非零**，不靠这段注释保证。
TABLE_EXTENT_TRAIN = 1.2
TABLE_EXTENT_SOURCE = 2.4


@dataclass(frozen=True)
class ProbeObject:
    """一个探针物体的场景事实。**只放事实，不放策略**。

    ``body_path`` 是**相对物体根 prim** 的刚体路径。自由刚体的 ``RigidBodyAPI``
    挂在根上（``build_assets`` 里 ``_rigid(root.GetPrim())``），所以是空串；
    固定体与 articulation 的刚体在子 prim 上。ContactSensor 的
    ``filter_prim_paths_expr`` 必须指到**那个**刚体，否则整条过滤通道静默失效（P-17）。
    """

    #: 物体根 prim 在资产里的名字，例如 ``Block``。
    prim_name: str
    #: 相对根 prim 的刚体子路径；根自身即刚体时为空串。
    body_subpath: str
    #: 初始位置（m），z 相对台面顶面（z=0）。
    init_pos: tuple[float, float, float]
    articulated: bool = False
    joint: str | None = None
    joint_init: float = 0.0
    #: 采集时**在运行时写进关节**的阻尼范围，会盖过资产里的值（P-38）。
    #: E-I 接 articulated 物体时必须照抄，否则物理与产生指令时不是同一套。
    damping_nominal: tuple[float, float] = (0.0, 0.0)
    cam_eye: tuple[float, float, float] = (0.42, -0.36, 0.28)
    cam_at: tuple[float, float, float] = (0.0, 0.0, 0.06)
    #: 自由刚体：静止时质心离台面的高度应当等于它的"半高"。
    #: `tools/test_probe_scene.py` 拿它做一致性检查，摆错一个数就红。
    rest_half_height: float | None = None

    @property
    def body_path(self) -> str:
        """采集侧用的相对路径（``Ridge/Body`` 这种）。"""
        return f"{self.prim_name}/{self.body_subpath}" if self.body_subpath else self.prim_name

    def filter_expr(self, object_prim: str) -> str:
        """ContactSensor 的 filter 表达式。``object_prim`` 是物体根 prim 的完整路径。"""
        return f"{object_prim}/{self.body_subpath}" if self.body_subpath else object_prim


_BL = B.BlockCfg()
_CO = B.ColumnCfg()
_RO = B.RollerCfg()
_BA = B.BallCfg()

PROBE_OBJECTS: dict[str, ProbeObject] = {
    # --- 自由体 E1/E2：刚体就是根 prim ---
    "block": ProbeObject("Block", "", (0.0, 0.0, _BL.size[2] / 2),
                         rest_half_height=_BL.size[2] / 2),
    "column": ProbeObject("Column", "", (0.0, 0.0, _CO.height / 2),
                          cam_eye=(0.40, -0.34, 0.26), cam_at=(0.0, 0.0, 0.08),
                          rest_half_height=_CO.height / 2),
    "roller": ProbeObject("Roller", "", (0.0, 0.0, _RO.radius),
                          cam_eye=(0.34, -0.32, 0.20), cam_at=(0.0, 0.0, 0.03),
                          rest_half_height=_RO.radius),
    "ball": ProbeObject("Ball", "", (0.0, 0.0, _BA.radius),
                        cam_eye=(0.32, -0.30, 0.20), cam_at=(0.0, 0.0, 0.035),
                        rest_half_height=_BA.radius),
    # --- 受约束平移 E3 ---
    "slider": ProbeObject("Slider", "Block", (0.0, 0.0, 0.0), articulated=True,
                          joint="BlockJoint", joint_init=0.075,
                          damping_nominal=(1.4, 2.8),
                          cam_eye=(0.34, -0.40, 0.26), cam_at=(-0.02, 0.0, 0.05)),
    "plunger": ProbeObject("Plunger", "Rod", (0.0, 0.0, 0.0), articulated=True,
                           joint="RodJoint", damping_nominal=(1.1, 2.2),
                           cam_eye=(0.40, -0.36, 0.24), cam_at=(0.04, 0.0, 0.04)),
    # --- 受约束转动 E4 ---
    "dial": ProbeObject("Dial", "Disc", (0.0, 0.0, 0.0), articulated=True,
                        joint="DiscJoint", damping_nominal=(0.14, 0.28),
                        cam_eye=(0.36, -0.32, 0.30), cam_at=(0.0, 0.0, 0.09)),
    "flap": ProbeObject("Flap", "Panel", (0.0, 0.0, 0.0), articulated=True,
                        joint="PanelJoint", damping_nominal=(1.4, 2.8),
                        cam_eye=(0.40, -0.38, 0.26), cam_at=(0.02, 0.04, 0.09)),
    # --- 固定体 E5：kinematic，刚体在子 prim 上 ---
    "ridge": ProbeObject("Ridge", "Body", (0.0, 0.0, 0.0),
                         cam_eye=(0.36, -0.34, 0.22), cam_at=(0.0, 0.0, 0.03)),
    "slab": ProbeObject("Slab", "Board", (0.0, 0.0, 0.08),
                        cam_eye=(0.40, -0.34, 0.30), cam_at=(0.0, 0.0, 0.09)),
}
"""十个探针物体的场景事实。键与 `tools/s3_source_probe.py::OBJECTS` 一一对应。"""


#: 各执行器的"末端"相对其根 prim 的偏移（m），用于把执行器摆到接触位点上方。
#: 采集侧的双板有 `plate_cfg` 的工作面约定；E-I 侧的杆类执行器只有一个末端面，
#: 这里显式写出来，免得每处各估一遍。
EXECUTOR_TIP: dict[str, tuple[float, float, float]] = {
    # 垫头杆：杆长 200 mm 居中，垫贴在 −Z 端，垫厚 8 mm -> 垫底面在 −0.108 m
    "padrod": (0.0, 0.0, -(B.PadRodCfg().shaft_len / 2 + B.PadRodCfg().pad[2])),
    # 钩杆：横钩在主杆 −Z 端，半径 8 mm -> 钩心在 −0.117 m
    "hook": (0.0, 0.0, -(B.HookCfg().shaft_len / 2 - B.HookCfg().shaft_radius)),
}

#: 各 target embodiment 对功能区域的接触空间分辨率（Gaussian sigma, m）。
#: 这属于 embodiment-specific decoder，不进入 interaction artifact。取接触面最短半径
#: ``r`` 对应的半高斯宽度 ``sigma=r/sqrt(2 ln 2)``：接触斑块边缘仍有 0.5 相容度。
EXECUTOR_REGION_SIGMA: dict[str, float] = {
    "padrod": (min(B.PadRodCfg().pad[:2]) / 2) / math.sqrt(2.0 * math.log(2.0)),
    "hook": B.HookCfg().shaft_radius / math.sqrt(2.0 * math.log(2.0)),
}
