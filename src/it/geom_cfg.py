"""资产的几何参数（**不依赖 pxr / Isaac Sim**）。

原本这些 dataclass 长在 `it.build_assets` 里，而那个模块在 import 时就要
`from pxr import ...`——拿不到 pxr 时它会自己拉起一个 SimulationApp。于是
"读一下把手多高"这种纯数值的事，也必须先启动 Isaac Sim，本机根本做不到。

S4 的表面采样（`it.surfaces`）要按 episode 的几何变体取尺寸，本机要能跑单元
测试，所以把参数与建模分开：**本模块只有数值，`build_assets` 负责把它们写成
USD**。`build_assets` 原样再导出这些名字，`from it import build_assets as B;
B.KnobCfg()` 这类既有写法不受影响。

尺寸全部来自 `plan/01-assets-and-scenes.md`，单位 mm，内部转米。
改这里的值等于改资产——D-07 规定资产只能从代码参数生成，不许手工编辑 USD。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

MM = 0.001

@dataclass
class KnobCfg:
    """旋钮：圆盘 + 偏心销钉 + revolute joint。plan/01 §3。"""

    disc_radius: float = 70 * MM
    disc_thickness: float = 15 * MM
    pin_offset: float = 52 * MM
    pin_radius: float = 10 * MM
    pin_length: float = 48 * MM
    base_size: tuple[float, float, float] = (170 * MM, 170 * MM, 40 * MM)
    # 底座半宽 85 mm > 圆盘半径 70 mm，径向接近轮缘会先撞底座。
    # 用立柱把圆盘抬起，让轮缘在侧向可达（S1 实测发现）。
    riser_radius: float = 30 * MM
    riser_height: float = 30 * MM
    disc_mass: float = 0.35
    # D-14：轮缘低摩擦使 region 在信息论上不可从 effect 推出
    rim_friction: float = 0.10
    pin_friction: float = 0.80
    joint_damping: float = 0.28
    joint_limit_deg: tuple[float, float] = (-10.0, 200.0)


@dataclass
class CabinetCfg:
    """抽屉：面板 + 把手 + prismatic joint。plan/01 §4。"""

    panel_w: float = 300 * MM
    panel_h: float = 180 * MM
    panel_t: float = 18 * MM
    handle_bar_len: float = 140 * MM
    handle_radius: float = 11 * MM
    handle_clearance: float = 45 * MM
    post_spacing: float = 125 * MM
    post_radius: float = 8 * MM
    travel: float = 180 * MM
    tray_depth: float = 250 * MM
    tray_t: float = 10 * MM
    wall_t: float = 15 * MM
    drawer_mass: float = 1.2
    friction: float = 0.6
    # 阻尼必须大到「松手就停」，否则策略会学会**捅一下让抽屉自己滑**。
    # 滑行距离 = v0 · m/c。原来 c=3.0 时，0.2 m/s 能滑 80 mm——
    # 抽屉总行程才 180 mm，一次冲量滑完大半，根本不需要持续拉。
    # S2 实测：打开过程中 60% 的控制步接触力为零，速度衰减反解出的
    # 阻尼恰好是 3.06，确认是自由滑行（P-28）。
    # c=30 时滑行距离 0.2×1.2/30 = 8 mm（原来 80 mm），既压掉了「捅一下滑行」
    # 这个退化解，又不至于把拉力需求推得太高。
    joint_damping: float = 30.0


@dataclass
class EraserCfg:
    """黑板擦：主体 + 底部接触垫。plan/01 §5.2。"""

    body: tuple[float, float, float] = (90 * MM, 45 * MM, 25 * MM)
    pad: tuple[float, float, float] = (80 * MM, 35 * MM, 5 * MM)
    mass: float = 0.08
    body_friction: float = 0.9
    pad_friction: float = 0.5


@dataclass
class HookCfg:
    """钩杆：L 形，0 自由度。plan/01 §6。"""

    shaft_radius: float = 8 * MM
    shaft_len: float = 250 * MM
    hook_len: float = 50 * MM
    mass: float = 0.25
    friction: float = 0.7


@dataclass
class PadRodCfg:
    """垫头杆：主杆 + 末端平垫，0 自由度。plan/01 §6，D-12。"""

    shaft_radius: float = 8 * MM
    shaft_len: float = 200 * MM
    pad: tuple[float, float, float] = (40 * MM, 30 * MM, 8 * MM)
    mass: float = 0.30
    pad_friction: float = 0.5


@dataclass
class PlateCfg:
    """双板 source 的单块薄板。plan/01 §6，D-02。

    局部坐标：**+Z 是工作面法向**，长边 35 mm 在 X，短边 25 mm 在 Y。

    ``face_*`` / ``fin_*`` 是**只有视觉、没有碰撞**的朝向标记。加它们的理由：
    薄板前后对称、四边接近对称，S3 录像验收时肉眼无法判断哪一面在接触、
    哪条边朝上，也就无法确认"接触发生在该发生的面上"——而那正是
    `plan/06` §7 要人工看视频的目的。

    **为什么不给标记加碰撞**：碰撞几何一改，S1 已验过的接触判据
    （稳态接触力/mg = 1.0000）和 D-02 的板尺寸就都要重验。标记的唯一
    用途是让画面可读，不该改变物理。代价是标记会与物体视觉穿插而无提示，
    因此它们只放在**背面和顶边**——工作面一侧保持干净。
    """

    size: tuple[float, float, float] = (35 * MM, 25 * MM, 3 * MM)
    #: 0.05 -> 0.5 kg。理由是数值的，不是物理的：板只有 35×25 mm，绕长边的
    #: 转动惯量 I = m(b²+c²)/12，50 g 时只有 2.6e-6 kg·m²。显式积分下姿态 PD 的
    #: 稳定条件 ω·dt < 2 会把刚度压到守不住姿态（实测极限环，平均偏 20°）。
    #: 惯量正比于质量，加重是唯一不改尺寸的解。物理上也说得通——采集面背后
    #: 本来就该有一只手或一段器械，不是一片孤立的塑料片。
    mass: float = 0.5
    friction: float = 0.9
    #: 工作面贴片（浅色）：告诉观众哪一面是接触面。每边内缩 1 mm 便于看出边界。
    face_inset: float = 1 * MM
    face_t: float = 0.4 * MM
    #: 顶边鳍（深色）：告诉观众哪条边朝上。突出 6 mm，占板高 24%，
    #: 在 960×540 的录像里能看清——2 mm 级的标记在这个尺度上只有 1~2 像素。
    fin: tuple[float, float, float] = (30 * MM, 6 * MM, 3 * MM)


@dataclass
class RollerCfg:
    """探针物体集：卧倒的滚柱。plan/03 §2.4，原语 P7 roll / P2 push / P1 press / P12 poke。

    轴线水平躺在台面上的圆柱，推它就滚。

    立着的 `column` 提供不了这一格：竖直圆柱受侧推只会**倒**，不会滚
    （实测 roll 那一档 97.7% 的操作步脱手、物体跑出 2 米）。滚动是接触点
    在物体表面上移动而不打滑的那一类接触（Huang 把它归在 sticking 里），
    与滑移是不同的格，必须有能真滚的物体。
    """

    radius: float = 30 * MM
    length: float = 140 * MM
    mass: float = 0.35
    friction: float = 0.8


@dataclass
class BallCfg:
    """探针物体集：自由球。plan/03 §2.4，原语 P7 roll / P1 press / P12 poke。

    P7 rolling contact 的第二个承载物体（冗余规则）。圆柱只能绕一根轴滚，
    球在任意方向都滚——两者的接触拓扑也不同（线接触 vs 点接触）。
    """

    radius: float = 35 * MM
    mass: float = 0.30
    friction: float = 0.8


@dataclass
class SliderCfg:
    """预训练物体集：滑块导轨。plan/03 §2.4。"""

    rail_len: float = 300 * MM
    rail_w: float = 40 * MM
    rail_h: float = 20 * MM
    block: tuple[float, float, float] = (60 * MM, 50 * MM, 40 * MM)
    #: 块顶上竖起来的挡片。**必须有**：没有它滑块只能被"推"。挡片高出块顶，
    #: 于是它的**两个面都够得到**——从一侧压是推，绕到另一侧压就是拉。
    #: P11 hook-pull 这一格靠的就是"压一个背向自己的面"，与抽屉手指压把手
    #: 背面是同一件事（`plan/03` §2.4.4）。
    #: 早先做成侧向凸缘，与块的侧面齐平，根本没有可够到的台肩。
    tab: tuple[float, float, float] = (10 * MM, 70 * MM, 34 * MM)
    travel: float = 150 * MM
    mass: float = 0.4
    friction: float = 0.5
    joint_damping: float = 2.0


@dataclass
class BlockCfg:
    """预训练物体集：桌面方块。plan/03 §2.4。

    自由刚体，产生"推移 / 侧推翻倒 / 按住不动"三类交互。它覆盖的是
    **无约束物体**这一档——effect 是自由的平移和翻转，与滑块/转盘的
    受约束运动形成对照。
    """

    size: tuple[float, float, float] = (60 * MM, 45 * MM, 40 * MM)
    mass: float = 0.25
    #: 0.6 -> 0.9。P6 pivot（撬翻）要求"先翻不先滑"，条件是 μ·h > 半宽，
    #: 而规则 9 规定摩擦按 **min** 组合——地面设成 0.9 而方块留 0.6 的话，
    #: 实际接触摩擦仍是 0.6，方块只会被推着滑走。两边都得是 0.9。
    friction: float = 0.9


@dataclass
class ColumnCfg:
    """预训练物体集：立柱。plan/03 §2.4。

    自由刚体，产生"侧推 / 双面搓转 / 推倒"。搓转那一档是**切向摩擦主导**的
    交互，与"法向力主导"（推方块、勾抽屉）形成对照——`plan/02` §3.5 的
    mechanics 字段要覆盖这两种，否则 range 的意义无从检验。
    """

    radius: float = 28 * MM
    #: 120 -> 80 mm。先倒不先滑的条件是 μ·h > 半径：h=120 时推中部就已经
    #: μ·h = 0.7×0.06 = 0.042 > 0.028，怎么推都是推倒，push/roll 全部失效。
    #: h=80 之后，推低处（h=25 mm）是推移，推高处（h=70 mm）才是撬翻，
    #: 同一个物体上两种原语都成立。
    height: float = 80 * MM
    mass: float = 0.30
    friction: float = 0.7


@dataclass
class DialCfg:
    """预训练物体集：转盘。plan/03 §2.4 的**新增项**，理由见 D-39。

    圆盘 + 3 个均布凸耳 + revolute joint，阻尼可随机。它覆盖
    **受约束转动**这一档——钩杆的留出任务里有旋钮，若预训练集完全没有
    绕固定轴转动这件事，钩杆就不是"没学过这个任务"，而是"没学过这类动作"，
    Gate E 不通过将无法归因。

    **与旋钮资产刻意做得不一样**：无偏心销钉、凸耳有三个而非一个、
    半径 90 vs 70 mm、无低摩擦轮缘设计（D-14 是旋钮特有的）。
    """

    disc_radius: float = 45 * MM
    disc_thickness: float = 14 * MM
    lug_radius: float = 9 * MM
    lug_height: float = 30 * MM
    lug_offset: float = 32 * MM
    base_size: tuple[float, float, float] = (140 * MM, 140 * MM, 30 * MM)
    riser_radius: float = 22 * MM
    riser_height: float = 26 * MM
    mass: float = 0.30
    friction: float = 0.7
    joint_damping: float = 0.20


@dataclass
class SlabCfg:
    """探针物体集：斜板。plan/03 §2.4，原语 P1 press / P4 rub / P5 shear / P12 poke。

    固定（kinematic）平板，可带倾角。它是 **E5（物体不动）× M2（持续 sliding）**
    这一格最简单的实现之一。

    ⚠️ **这一格不由它独占**：`block` 被压住时它的顶面同样是静止表面，
    `ridge` 也提供固定曲面。冗余规则（`plan/03` §2.4.5）要求每条原语
    至少两个几何不同的物体承载——删掉本物体，P1/P4/P5 仍然被覆盖。
    这一点是"物体集不是照着擦拭任务反推的"的可检验依据。

    200×150 是**探针**尺度，不是工作台尺度；无 dirt grid、无黑板擦、
    可倾斜。它不产生 envelope，只产生 interaction record。

    规则 7：必须是 kinematic 刚体而不是静态碰撞体，否则 filter 通道失效（P-17）。
    """

    size: tuple[float, float, float] = (200 * MM, 150 * MM, 20 * MM)
    tilt_deg: float = 15.0
    friction: float = 0.45


@dataclass
class FlapCfg:
    """探针物体集：立板门。plan/03 §2.4，原语 P9 crank / P4 rub / P1 press / P12 poke。

    竖直立着、沿一条**竖直边**铰接的板，推它的板面使其绕立柱转开。

    ⚠️ 最初设计成"平放在底座上、沿水平边铰接的翻板"，**几何上做不到**：
    平板执行器没法伸到板底下把它掀起来，只能压，而压是被底座挡住的。
    改成立着的门之后，"推板面"就是最自然的动作，而且**接触点离转轴多远**
    直接改变所需力矩——P9 crank 的力臂含义因此是真的，不是名义上的。

    与 `dial` 同属 E4（受约束转动）但几何完全不同：转轴在物体边缘而非中心、
    接触面是平板而非圆柱凸耳、力臂随接触位置连续变化。
    """

    base: tuple[float, float, float] = (120 * MM, 120 * MM, 16 * MM)
    post_radius: float = 12 * MM
    panel: tuple[float, float, float] = (10 * MM, 150 * MM, 120 * MM)
    mass: float = 0.20
    friction: float = 0.6
    #: 0.05 -> 2.0。推力矩约 0.5~1.4 N·m，阻尼 0.05 时角速度 10 rad/s，
    #: 门 0.08 s 就甩到限位，采集板追不上（44% 的步脱手）。
    #: 2.0 对应 0.27~0.7 rad/s，开满 45° 要 1~3 s，正好是一个 manipulate 阶段。
    joint_damping: float = 2.0
    #: ±45°：开到 ±100° 时板绕铰链扫过的弧长太大，采集板的位置 PD 跟不上，
    #: 实测 crank 有 45% 的操作步是脱手的。
    joint_limit_deg: tuple[float, float] = (-45.0, 45.0)


@dataclass
class PlungerCfg:
    """探针物体集：柱塞。plan/03 §2.4，原语 P10 slide-along / P11 hook-pull / P1 / P12。

    圆柱在固定套筒里滑动，外端带一个比杆粗的帽——帽的背面提供可勾的台肩。
    与 `slider` 同属 E3（受约束平移）但几何完全不同：圆柱 vs 方块、
    套筒约束 vs 导轨约束、端帽台肩 vs 侧向凸缘。
    冗余规则要求 E3 与 P11 这两格不能只有一个物体。
    """

    sleeve_radius: float = 26 * MM
    sleeve_len: float = 90 * MM
    rod_radius: float = 16 * MM
    rod_len: float = 200 * MM
    cap_radius: float = 30 * MM
    cap_len: float = 14 * MM
    travel: float = 110 * MM
    mass: float = 0.25
    friction: float = 0.5
    joint_damping: float = 1.5


@dataclass
class RidgeCfg:
    """预训练物体集：凸棱台。plan/03 §2.4 的**新增项**，理由见 D-39。

    固定台面上横着一条圆柱棱。覆盖**曲面上的接触**——抽屉把手是圆柱，
    而钩杆的留出任务正是抽屉；预训练集若只有平面，执行器从没在曲面上
    建立过接触，engage 方向随曲率变化这件事也无从学起。

    产生的交互：贴着棱侧推（线接触）、跨棱滑过（接触点沿棱移动）、
    沿棱滑动（接触点不动而物体表面在动）。
    """

    base: tuple[float, float, float] = (300 * MM, 200 * MM, 20 * MM)
    ridge_radius: float = 10 * MM
    ridge_len: float = 200 * MM
    friction: float = 0.55


@dataclass
class BuildCfg:
    knob: KnobCfg = field(default_factory=KnobCfg)
    cabinet: CabinetCfg = field(default_factory=CabinetCfg)
    eraser: EraserCfg = field(default_factory=EraserCfg)
    hook: HookCfg = field(default_factory=HookCfg)
    padrod: PadRodCfg = field(default_factory=PadRodCfg)
    plate: PlateCfg = field(default_factory=PlateCfg)
    block: BlockCfg = field(default_factory=BlockCfg)
    column: ColumnCfg = field(default_factory=ColumnCfg)
    roller: RollerCfg = field(default_factory=RollerCfg)
    ball: BallCfg = field(default_factory=BallCfg)
    dial: DialCfg = field(default_factory=DialCfg)
    slab: SlabCfg = field(default_factory=SlabCfg)
    flap: FlapCfg = field(default_factory=FlapCfg)
    plunger: PlungerCfg = field(default_factory=PlungerCfg)
    ridge: RidgeCfg = field(default_factory=RidgeCfg)
    slider: SliderCfg = field(default_factory=SliderCfg)

#: 物体名 -> `BuildCfg` 上的字段名。`build_assets.BUILDERS` 与 `variant_cfg`
#: 共用它，避免两处各写一份对应关系。
CFG_ATTR: dict[str, str] = {
    "knob": "knob",
    "cabinet": "cabinet",
    "eraser": "eraser",
    "hook": "hook",
    "padrod": "padrod",
    "plate0": "plate",
    "plate1": "plate",
    "slider": "slider",
    "block": "block",
    "column": "column",
    "roller": "roller",
    "ball": "ball",
    "dial": "dial",
    "slab": "slab",
    "flap": "flap",
    "plunger": "plunger",
    "ridge": "ridge",
}

#: **小幅几何变化**（`plan/03` §7 表第 6 行）。每个任务物体额外生成两个变体，
#: 采集时按 env 轮转混进去，它们的 episode 单独进 ``unseen_geometry_test``。
#:
#: 变化量刻意取小（±12% 以内）：judge 与 envelope 都不该因为它失效，
#: 这一格要测的是"同一份交互规格在略微不同的几何上还成不成立"，
#: 不是"换了个物体"。改的都是**采集器必须知道的那一个尺寸**——
#: 销钉偏心距、把手离面板的距离、黑板擦长度——所以采集器也要按 env 取值。
GEOM_VARIANTS: dict[str, list[tuple[str, dict]]] = {
    "knob": [("nominal", {}),
             ("g1", {"pin_offset": 46 * MM}),
             ("g2", {"pin_offset": 58 * MM})],
    "cabinet": [("nominal", {}),
                ("g1", {"handle_clearance": 38 * MM}),
                ("g2", {"handle_clearance": 52 * MM})],
    "eraser": [("nominal", {}),
               ("g1", {"body": (80 * MM, 45 * MM, 25 * MM),
                       "pad": (70 * MM, 35 * MM, 5 * MM)}),
               ("g2", {"body": (100 * MM, 45 * MM, 25 * MM),
                       "pad": (90 * MM, 35 * MM, 5 * MM)})],
}


def variant_cfg(name: str, tag: str, cfg: "BuildCfg | None" = None):
    """取某个物体某个几何变体的配置对象（``tag`` = nominal / g1 / g2）。"""
    cfg = cfg or BuildCfg()
    base = getattr(cfg, CFG_ATTR[name])
    over = dict(next(o for t, o in GEOM_VARIANTS[name] if t == tag))
    return replace(base, **over) if over else base
