"""从代码生成参数化 USD 资产。

D-07 禁止手工编辑 USD 和 URDF 导入。本模块用 pxr API 从参数生成 USD：
参数错了改 config 重跑即可，git revert 就能回退，不会陷进网格编辑。

尺寸全部来自 `plan/01-assets-and-scenes.md`，单位 mm，内部转米。

用法::

    /isaac-sim/python.sh -m it.build_assets --out /workspace/interaction_transfer/assets_gen
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

try:
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
except ModuleNotFoundError:  # pxr 需要先启动 Isaac Sim 才可用
    from isaacsim import SimulationApp

    _APP = SimulationApp({"headless": True})
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

MM = 0.001


# ---------------------------------------------------------------- 参数


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
class SliderCfg:
    """预训练物体集：滑块导轨。plan/03 §2.4。"""

    rail_len: float = 300 * MM
    rail_w: float = 40 * MM
    rail_h: float = 20 * MM
    block: tuple[float, float, float] = (60 * MM, 50 * MM, 40 * MM)
    travel: float = 150 * MM
    mass: float = 0.4
    friction: float = 0.5
    joint_damping: float = 2.0


@dataclass
class BuildCfg:
    knob: KnobCfg = field(default_factory=KnobCfg)
    cabinet: CabinetCfg = field(default_factory=CabinetCfg)
    eraser: EraserCfg = field(default_factory=EraserCfg)
    hook: HookCfg = field(default_factory=HookCfg)
    padrod: PadRodCfg = field(default_factory=PadRodCfg)
    plate: PlateCfg = field(default_factory=PlateCfg)
    slider: SliderCfg = field(default_factory=SliderCfg)


# ---------------------------------------------------------------- USD 辅助


def _new_stage(path: str) -> Usd.Stage:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    return stage


def _phys_material(stage, path: str, static_f: float, dynamic_f: float, restitution: float = 0.0,
                   combine: str = "min"):
    """物理材质。

    ``combine="min"`` 至关重要：PhysX 默认的摩擦组合模式是 **average**，
    于是 D-14 设计的低摩擦轮缘（μ=0.10）碰上高摩擦执行器（μ=0.7）时，
    实际接触摩擦是 (0.10+0.7)/2 = 0.4——低摩擦设计完全失效，region 又变成
    可从 effect 推出的了。用 min 才能保证"轮缘对任何东西都是低摩擦"。
    """
    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr().Set(static_f)
    api.CreateDynamicFrictionAttr().Set(dynamic_f)
    api.CreateRestitutionAttr().Set(restitution)
    px = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
    px.CreateFrictionCombineModeAttr().Set(combine)
    px.CreateRestitutionCombineModeAttr().Set("min")
    return mat


#: 视觉配色。让 S0 录像里一眼能分辨功能区域——尤其轮缘(低摩擦)与销钉(高摩擦)，
#: 那是 D-14 的关键，全白渲染根本看不出区别。
COLOR = {
    "base":    (0.45, 0.45, 0.48),
    "rim":     (0.20, 0.45, 0.85),   # 蓝 = 低摩擦，推不动
    "pin":     (0.95, 0.55, 0.10),   # 橙 = 高摩擦，可推转
    "cabinet": (0.55, 0.40, 0.28),
    "handle":  (0.85, 0.85, 0.88),
    "eraser":  (0.85, 0.25, 0.25),
    "pad":     (0.20, 0.75, 0.55),
    "hook":    (0.95, 0.85, 0.20),
    "rod":     (0.30, 0.75, 0.90),
    "block":   (0.70, 0.35, 0.75),
    "plate0":  (0.90, 0.55, 0.15),   # 橙板
    "plate1":  (0.15, 0.55, 0.90),   # 蓝板
    "face":    (0.96, 0.96, 0.92),   # 浅色 = 工作面（两块板相同，看一次就记住）
    "marker":  (0.10, 0.10, 0.12),   # 深色顶边鳍 = 这条边朝上
}


def _vis_material(stage, path: str, rgb, rough: float = 0.55):
    """UsdPreviewSurface 视觉材质。"""
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def _bind_visual(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants)


def _bind_material(prim, mat):
    binding = UsdShade.MaterialBindingAPI.Apply(prim)
    binding.Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def _xform(stage, path: str, pos=(0.0, 0.0, 0.0), rot_wxyz=None):
    xf = UsdGeom.Xform.Define(stage, path)
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    if rot_wxyz is not None:
        xf.AddOrientOp().Set(Gf.Quatf(rot_wxyz[0], Gf.Vec3f(*rot_wxyz[1:])))
    return xf


def _box(stage, path: str, size, pos=(0.0, 0.0, 0.0), mat=None, rot_wxyz=None, vis=None,
         collision: bool = True):
    """size 为 (sx, sy, sz) 实际边长。Cube 基准边长 1，用 scale 缩放。

    ``collision=False`` 生成**纯视觉**几何（不 Apply CollisionAPI，也不参与
    惯性张量计算）。用于朝向标记一类"只为看得清、不该改变物理"的东西。
    """
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*pos))
    if rot_wxyz is not None:
        cube.AddOrientOp().Set(Gf.Quatf(rot_wxyz[0], Gf.Vec3f(*rot_wxyz[1:])))
    cube.AddScaleOp().Set(Gf.Vec3f(*size))
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    if mat is not None:
        _bind_material(cube.GetPrim(), mat)
    if vis is not None:
        _bind_visual(cube.GetPrim(), vis)
    return cube


def _cyl(stage, path: str, radius, height, pos=(0.0, 0.0, 0.0), axis="Z", mat=None, vis=None):
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(height)
    cyl.CreateAxisAttr(axis)
    half = radius if axis != "Z" else height / 2.0
    ext = {
        "Z": (Gf.Vec3f(-radius, -radius, -height / 2), Gf.Vec3f(radius, radius, height / 2)),
        "X": (Gf.Vec3f(-height / 2, -radius, -radius), Gf.Vec3f(height / 2, radius, radius)),
        "Y": (Gf.Vec3f(-radius, -height / 2, -radius), Gf.Vec3f(radius, height / 2, radius)),
    }[axis]
    cyl.CreateExtentAttr([ext[0], ext[1]])
    cyl.AddTranslateOp().Set(Gf.Vec3d(*pos))
    UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
    if mat is not None:
        _bind_material(cyl.GetPrim(), mat)
    if vis is not None:
        _bind_visual(cyl.GetPrim(), vis)
    del half
    return cyl


def _rigid(prim, mass: float | None = None, kinematic: bool = False):
    rb = UsdPhysics.RigidBodyAPI.Apply(prim)
    rb.CreateKinematicEnabledAttr(kinematic)
    if mass is not None:
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
    PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    return rb


def _joint(stage, path, kind, body0, body1, axis, pos0, pos1, limits=None, damping=0.0):
    """kind: 'revolute' | 'prismatic' | 'fixed'。revolute 的 limit 单位是度。

    ``body0=None`` 表示锚定到 world——用于给 articulation 的根链接做固定约束。
    Isaac Lab 的 ``fix_root_link=True`` 要求根 prim 自带 RigidBodyAPI，而我们的
    根是纯 Xform，所以在 USD 里显式建这个 fixed joint 更可靠。
    """
    cls = {
        "revolute": UsdPhysics.RevoluteJoint,
        "prismatic": UsdPhysics.PrismaticJoint,
        "fixed": UsdPhysics.FixedJoint,
    }[kind]
    j = cls.Define(stage, path)
    if body0 is not None:
        j.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    else:
        j.CreateBody0Rel().SetTargets([])
    j.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    j.CreateLocalPos0Attr().Set(Gf.Vec3f(*pos0))
    j.CreateLocalPos1Attr().Set(Gf.Vec3f(*pos1))
    j.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
    j.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    if kind != "fixed":
        j.CreateAxisAttr(axis)
        if limits is not None:
            j.CreateLowerLimitAttr().Set(limits[0])
            j.CreateUpperLimitAttr().Set(limits[1])
        drive_kind = "angular" if kind == "revolute" else "linear"
        drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), drive_kind)
        drive.CreateTypeAttr().Set("force")
        drive.CreateStiffnessAttr().Set(0.0)
        drive.CreateDampingAttr().Set(damping)
        drive.CreateTargetVelocityAttr().Set(0.0)
        drive.CreateMaxForceAttr().Set(1e6)
    return j


# ---------------------------------------------------------------- 各资产


def build_knob(path: str, cfg: KnobCfg) -> str:
    """底座固定，圆盘绕 Z 转，销钉偏心竖直伸出。轮缘与销钉摩擦不同（D-14）。"""
    stage = _new_stage(path)
    root = _xform(stage, "/Knob")

    m_rim = _phys_material(stage, "/Knob/PhysMat_Rim", cfg.rim_friction, cfg.rim_friction)
    m_pin = _phys_material(stage, "/Knob/PhysMat_Pin", cfg.pin_friction, cfg.pin_friction)
    v_base = _vis_material(stage, "/Knob/VisBase", COLOR["base"])
    v_rim = _vis_material(stage, "/Knob/VisRim", COLOR["rim"])
    v_pin = _vis_material(stage, "/Knob/VisPin", COLOR["pin"])

    bx, by, bz = cfg.base_size
    base = _xform(stage, "/Knob/Base", pos=(0.0, 0.0, bz / 2))
    _rigid(base.GetPrim(), mass=20.0)
    # ArticulationRootAPI 必须挂在**有 RigidBodyAPI 的 link** 上，Isaac Lab 的
    # fix_root_link=True 才能给它建世界固定关节。挂在纯 Xform 根上会报
    # NotImplementedError；而自己在 USD 里建 body0 为空的 FixedJoint **不生效**——
    # S1 实测整个旋钮以 20 m/s 自由落体到 z=-46 m。
    UsdPhysics.ArticulationRootAPI.Apply(base.GetPrim())
    _box(stage, "/Knob/Base/geom", (bx, by, bz), mat=m_rim, vis=v_base)

    _cyl(stage, "/Knob/Base/riser", cfg.riser_radius, cfg.riser_height,
         pos=(0.0, 0.0, bz / 2 + cfg.riser_height / 2), mat=m_rim, vis=v_base)

    disc_z = bz + cfg.riser_height + cfg.disc_thickness / 2
    disc = _xform(stage, "/Knob/Disc", pos=(0.0, 0.0, disc_z))
    _rigid(disc.GetPrim(), mass=cfg.disc_mass)
    _cyl(stage, "/Knob/Disc/rim", cfg.disc_radius, cfg.disc_thickness, mat=m_rim, vis=v_rim)
    _cyl(
        stage,
        "/Knob/Disc/pin",
        cfg.pin_radius,
        cfg.pin_length,
        pos=(cfg.pin_offset, 0.0, cfg.disc_thickness / 2 + cfg.pin_length / 2),
        mat=m_pin, vis=v_pin,
    )

    _joint(
        stage, "/Knob/DiscJoint", "revolute", "/Knob/Base", "/Knob/Disc",
        "Z", (0.0, 0.0, bz / 2 + cfg.riser_height), (0.0, 0.0, -cfg.disc_thickness / 2),
        limits=cfg.joint_limit_deg, damping=cfg.joint_damping,
    )
    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().Save()
    return path


def build_cabinet(path: str, cfg: CabinetCfg) -> str:
    """柜体固定，抽屉沿 +X 拉出。把手前方留净空供手指/钩杆伸入。"""
    stage = _new_stage(path)
    root = _xform(stage, "/Cabinet")
    mat = _phys_material(stage, "/Cabinet/PhysMat", cfg.friction, cfg.friction)
    v_cab = _vis_material(stage, "/Cabinet/VisBody", COLOR["cabinet"])
    v_hdl = _vis_material(stage, "/Cabinet/VisHandle", COLOR["handle"], rough=0.25)

    inner_w = cfg.panel_w - 2 * cfg.wall_t
    inner_h = cfg.panel_h
    depth = cfg.tray_depth + cfg.travel
    t = cfg.wall_t

    body = _xform(stage, "/Cabinet/Body", pos=(0.0, 0.0, 0.0))
    _rigid(body.GetPrim(), mass=50.0)
    UsdPhysics.ArticulationRootAPI.Apply(body.GetPrim())
    _box(stage, "/Cabinet/Body/bottom", (depth, cfg.panel_w, t), pos=(-depth / 2, 0.0, -t / 2), mat=mat, vis=v_cab)
    _box(stage, "/Cabinet/Body/top", (depth, cfg.panel_w, t), pos=(-depth / 2, 0.0, inner_h + t / 2), mat=mat, vis=v_cab)
    _box(stage, "/Cabinet/Body/left", (depth, t, inner_h), pos=(-depth / 2, (inner_w + t) / 2, inner_h / 2), mat=mat, vis=v_cab)
    _box(stage, "/Cabinet/Body/right", (depth, t, inner_h), pos=(-depth / 2, -(inner_w + t) / 2, inner_h / 2), mat=mat, vis=v_cab)
    _box(stage, "/Cabinet/Body/back", (t, cfg.panel_w, inner_h), pos=(-depth - t / 2, 0.0, inner_h / 2), mat=mat, vis=v_cab)

    drawer = _xform(stage, "/Cabinet/Drawer", pos=(0.0, 0.0, 0.0))
    _rigid(drawer.GetPrim(), mass=cfg.drawer_mass)
    _box(stage, "/Cabinet/Drawer/panel", (cfg.panel_t, cfg.panel_w, cfg.panel_h),
         pos=(cfg.panel_t / 2, 0.0, cfg.panel_h / 2), mat=mat, vis=v_cab)
    _box(stage, "/Cabinet/Drawer/tray", (cfg.tray_depth, inner_w - 2 * MM, cfg.tray_t),
         pos=(-cfg.tray_depth / 2, 0.0, cfg.tray_t / 2), mat=mat, vis=v_cab)

    hx = cfg.panel_t + cfg.handle_clearance + cfg.handle_radius
    hz = cfg.panel_h / 2
    _cyl(stage, "/Cabinet/Drawer/handle_bar", cfg.handle_radius, cfg.handle_bar_len,
         pos=(hx, 0.0, hz), axis="Y", mat=mat, vis=v_hdl)
    post_len = cfg.handle_clearance + cfg.handle_radius
    for sgn, nm in ((1, "l"), (-1, "r")):
        _cyl(stage, f"/Cabinet/Drawer/handle_post_{nm}", cfg.post_radius, post_len,
             pos=(cfg.panel_t + post_len / 2, sgn * cfg.post_spacing / 2, hz), axis="X", mat=mat, vis=v_hdl)

    _joint(
        stage, "/Cabinet/DrawerJoint", "prismatic", "/Cabinet/Body", "/Cabinet/Drawer",
        "X", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
        limits=(0.0, cfg.travel), damping=cfg.joint_damping,
    )
    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().Save()
    return path


def build_eraser(path: str, cfg: EraserCfg) -> str:
    """主体 + 更小的底部接触垫，两者摩擦不同。"""
    stage = _new_stage(path)
    root = _xform(stage, "/Eraser")
    _rigid(root.GetPrim(), mass=cfg.mass)
    m_body = _phys_material(stage, "/Eraser/PhysMat_Body", cfg.body_friction, cfg.body_friction)
    m_pad = _phys_material(stage, "/Eraser/PhysMat_Pad", cfg.pad_friction, cfg.pad_friction)
    v_bod = _vis_material(stage, "/Eraser/VisBody", COLOR["eraser"])
    v_pad = _vis_material(stage, "/Eraser/VisPad", COLOR["pad"])
    bw, bd, bh = cfg.body
    pw, pd, ph = cfg.pad
    _box(stage, "/Eraser/body", (bw, bd, bh), pos=(0.0, 0.0, ph + bh / 2), mat=m_body, vis=v_bod)
    _box(stage, "/Eraser/pad", (pw, pd, ph), pos=(0.0, 0.0, ph / 2), mat=m_pad, vis=v_pad)
    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().Save()
    return path


def build_hook(path: str, cfg: HookCfg) -> str:
    """L 形钩杆：竖直主杆 + 水平横钩。"""
    stage = _new_stage(path)
    root = _xform(stage, "/Hook")
    _rigid(root.GetPrim(), mass=cfg.mass)
    mat = _phys_material(stage, "/Hook/PhysMat", cfg.friction, cfg.friction)
    vis = _vis_material(stage, "/Hook/Vis", COLOR["hook"])
    _cyl(stage, "/Hook/shaft", cfg.shaft_radius, cfg.shaft_len, pos=(0.0, 0.0, 0.0), axis="Z", mat=mat, vis=vis)
    _cyl(stage, "/Hook/tip", cfg.shaft_radius, cfg.hook_len,
         pos=(cfg.hook_len / 2, 0.0, -cfg.shaft_len / 2 + cfg.shaft_radius), axis="X", mat=mat, vis=vis)
    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().Save()
    return path


def build_padrod(path: str, cfg: PadRodCfg) -> str:
    """垫头杆：主杆 + 末端平垫。用于擦拭的自身接触面直擦（D-12）。"""
    stage = _new_stage(path)
    root = _xform(stage, "/PadRod")
    _rigid(root.GetPrim(), mass=cfg.mass)
    mat = _phys_material(stage, "/PadRod/PhysMat", cfg.pad_friction, cfg.pad_friction)
    v_rod = _vis_material(stage, "/PadRod/VisRod", COLOR["rod"])
    v_pad = _vis_material(stage, "/PadRod/VisPad", COLOR["pad"])
    pw, pd, ph = cfg.pad
    _cyl(stage, "/PadRod/shaft", cfg.shaft_radius, cfg.shaft_len, pos=(0.0, 0.0, 0.0), axis="Z", mat=mat, vis=v_rod)
    _box(stage, "/PadRod/pad", (pw, pd, ph), pos=(0.0, 0.0, -cfg.shaft_len / 2 - ph / 2), mat=mat, vis=v_pad)
    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().Save()
    return path


def _build_plate(path: str, cfg: PlateCfg, body_rgb) -> str:
    """一块采集板：一个碰撞盒 + 两个纯视觉朝向标记。

    碰撞几何**只有** ``pad`` 那一个盒子，与 S1 验过的
    ``sim_utils.CuboidCfg(size=(35,25,3) mm)`` 完全等价——标记不参与碰撞，
    也不参与惯性（质量由 MassAPI 显式给定）。
    """
    stage = _new_stage(path)
    root = _xform(stage, "/Plate")
    _rigid(root.GetPrim(), mass=cfg.mass)
    mat = _phys_material(stage, "/Plate/PhysMat", cfg.friction, cfg.friction)
    v_body = _vis_material(stage, "/Plate/VisBody", body_rgb)
    v_face = _vis_material(stage, "/Plate/VisFace", COLOR["face"], rough=0.30)
    v_fin = _vis_material(stage, "/Plate/VisFin", COLOR["marker"], rough=0.70)

    sx, sy, sz = cfg.size
    _box(stage, "/Plate/pad", (sx, sy, sz), mat=mat, vis=v_body)
    _box(stage, "/Plate/face_mark",
         (sx - 2 * cfg.face_inset, sy - 2 * cfg.face_inset, cfg.face_t),
         pos=(0.0, 0.0, sz / 2 + cfg.face_t / 2), vis=v_face, collision=False)
    fx, fy, fz = cfg.fin
    _box(stage, "/Plate/fin", (fx, fy, fz),
         pos=(0.0, sy / 2 + fy / 2, 0.0), vis=v_fin, collision=False)

    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().Save()
    return path


def build_plate0(path: str, cfg: PlateCfg) -> str:
    """橙板。两块板几何完全相同，只有机身颜色不同（用于在录像里区分彼此）。"""
    return _build_plate(path, cfg, COLOR["plate0"])


def build_plate1(path: str, cfg: PlateCfg) -> str:
    """蓝板。"""
    return _build_plate(path, cfg, COLOR["plate1"])


def build_slider(path: str, cfg: SliderCfg) -> str:
    """预训练物体集：滑块沿导轨平移，阻尼可随机化。"""
    stage = _new_stage(path)
    root = _xform(stage, "/Slider")
    mat = _phys_material(stage, "/Slider/PhysMat", cfg.friction, cfg.friction)
    v_rail = _vis_material(stage, "/Slider/VisRail", COLOR["base"])
    v_blk = _vis_material(stage, "/Slider/VisBlock", COLOR["block"])

    rail = _xform(stage, "/Slider/Rail", pos=(0.0, 0.0, cfg.rail_h / 2))
    _rigid(rail.GetPrim(), mass=20.0)
    UsdPhysics.ArticulationRootAPI.Apply(rail.GetPrim())
    _box(stage, "/Slider/Rail/geom", (cfg.rail_len, cfg.rail_w, cfg.rail_h), mat=mat, vis=v_rail)

    bw, bd, bh = cfg.block
    block = _xform(stage, "/Slider/Block", pos=(-cfg.travel / 2, 0.0, cfg.rail_h + bh / 2))
    _rigid(block.GetPrim(), mass=cfg.mass)
    _box(stage, "/Slider/Block/geom", (bw, bd, bh), mat=mat, vis=v_blk)

    _joint(
        stage, "/Slider/BlockJoint", "prismatic", "/Slider/Rail", "/Slider/Block",
        "X", (-cfg.travel / 2, 0.0, cfg.rail_h / 2), (0.0, 0.0, -bh / 2),
        limits=(0.0, cfg.travel), damping=cfg.joint_damping,
    )
    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().Save()
    return path


# ---------------------------------------------------------------- 入口

BUILDERS = {
    "knob": (build_knob, "knob"),
    "cabinet": (build_cabinet, "cabinet"),
    "eraser": (build_eraser, "eraser"),
    "hook": (build_hook, "hook"),
    "padrod": (build_padrod, "padrod"),
    "plate0": (build_plate0, "plate"),
    "plate1": (build_plate1, "plate"),
    "slider": (build_slider, "slider"),
}

DEFAULT_OUT = "/workspace/interaction_transfer/assets_gen"


def build_all(out_dir: str = DEFAULT_OUT, cfg: BuildCfg | None = None) -> dict[str, str]:
    cfg = cfg or BuildCfg()
    os.makedirs(out_dir, exist_ok=True)
    made = {}
    for name, (fn, attr) in BUILDERS.items():
        p = os.path.join(out_dir, f"{name}.usd")
        fn(p, getattr(cfg, attr))
        made[name] = p
    return made


def main():
    ap = argparse.ArgumentParser(description="生成参数化 USD 资产")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    made = build_all(args.out)
    for name, p in made.items():
        print(f"  {name:9s} -> {p}  ({os.path.getsize(p)} bytes)")
    print(f"\n共 {len(made)} 个资产写入 {args.out}")


if __name__ == "__main__":
    main()
    # SimulationApp 的优雅关闭在本环境会挂起（P-19）。文件已落盘，直接退出。
    import sys
    sys.stdout.flush()
    os._exit(0)
