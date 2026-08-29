"""从代码生成参数化 USD 资产。

D-07 禁止手工编辑 USD 和 URDF 导入。本模块用 pxr API 从参数生成 USD：
参数错了改 config 重跑即可，git revert 就能回退，不会陷进网格编辑。

尺寸参数在 `it.geom_cfg`（不依赖 pxr，本机可读）；本模块只负责把它们写成 USD。

用法::

    /isaac-sim/python.sh -m it.build_assets --out /workspace/interaction_transfer/assets_gen
"""

from __future__ import annotations

import argparse
import math
import os

try:
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
except ModuleNotFoundError:  # pxr 需要先启动 Isaac Sim 才可用
    from isaacsim import SimulationApp

    _APP = SimulationApp({"headless": True})
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

# 几何参数在 `it.geom_cfg` 里，那个模块**不依赖 pxr**——本机（无 Isaac Sim）
# 也要能读尺寸做表面采样与单元测试。这里原样再导出，既有的
# `from it import build_assets as B; B.KnobCfg()` 写法不变。
from it.geom_cfg import (  # noqa: F401
    MM,
    BallCfg,
    BlockCfg,
    BuildCfg,
    CabinetCfg,
    CFG_ATTR,
    ColumnCfg,
    DialCfg,
    EraserCfg,
    FlapCfg,
    GEOM_VARIANTS,
    HookCfg,
    KnobCfg,
    PadRodCfg,
    PlateCfg,
    PlungerCfg,
    RidgeCfg,
    RollerCfg,
    SlabCfg,
    SliderCfg,
    variant_cfg,
)


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
    "column":  (0.35, 0.70, 0.45),
    "dial":    (0.80, 0.70, 0.30),
    "lug":     (0.95, 0.45, 0.20),
    "slab":    (0.30, 0.45, 0.60),
    "flap":    (0.55, 0.60, 0.35),
    "plunger": (0.60, 0.40, 0.55),
    "ridge":   (0.75, 0.55, 0.35),
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
    tw, td, th = cfg.tab
    _box(stage, "/Slider/Block/tab", (tw, td, th),
         pos=(bw / 2 - tw / 2 - 6 * MM, 0.0, bh / 2 + th / 2 - 6 * MM), mat=mat, vis=v_blk)

    _joint(
        stage, "/Slider/BlockJoint", "prismatic", "/Slider/Rail", "/Slider/Block",
        "X", (-cfg.travel / 2, 0.0, cfg.rail_h / 2), (0.0, 0.0, -bh / 2),
        limits=(0.0, cfg.travel), damping=cfg.joint_damping,
    )
    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().Save()
    return path


def build_block(path: str, cfg: BlockCfg) -> str:
    """预训练物体集：自由方块。"""
    stage = _new_stage(path)
    root = _xform(stage, "/Block")
    _rigid(root.GetPrim(), mass=cfg.mass)
    mat = _phys_material(stage, "/Block/PhysMat", cfg.friction, cfg.friction)
    vis = _vis_material(stage, "/Block/Vis", COLOR["block"])
    _box(stage, "/Block/geom", cfg.size, mat=mat, vis=vis)
    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().Save()
    return path


def build_column(path: str, cfg: ColumnCfg) -> str:
    """预训练物体集：自由立柱。竖直放置，可被侧推、搓转或推倒。"""
    stage = _new_stage(path)
    root = _xform(stage, "/Column")
    _rigid(root.GetPrim(), mass=cfg.mass)
    mat = _phys_material(stage, "/Column/PhysMat", cfg.friction, cfg.friction)
    vis = _vis_material(stage, "/Column/Vis", COLOR["column"])
    _cyl(stage, "/Column/geom", cfg.radius, cfg.height, axis="Z", mat=mat, vis=vis)
    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().Save()
    return path


def build_roller(path: str, cfg: RollerCfg) -> str:
    """探针物体集：卧倒的自由滚柱，轴线沿 Y。"""
    stage = _new_stage(path)
    root = _xform(stage, "/Roller")
    _rigid(root.GetPrim(), mass=cfg.mass)
    mat = _phys_material(stage, "/Roller/PhysMat", cfg.friction, cfg.friction)
    vis = _vis_material(stage, "/Roller/Vis", COLOR["column"])
    _cyl(stage, "/Roller/geom", cfg.radius, cfg.length, axis="Y", mat=mat, vis=vis)
    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().Save()
    return path


def build_ball(path: str, cfg: BallCfg) -> str:
    """探针物体集：自由球。P7 的第二个承载物体。"""
    stage = _new_stage(path)
    root = _xform(stage, "/Ball")
    _rigid(root.GetPrim(), mass=cfg.mass)
    mat = _phys_material(stage, "/Ball/PhysMat", cfg.friction, cfg.friction)
    vis = _vis_material(stage, "/Ball/Vis", COLOR["lug"])
    sph = UsdGeom.Sphere.Define(stage, "/Ball/geom")
    sph.CreateRadiusAttr(cfg.radius)
    sph.CreateExtentAttr([Gf.Vec3f(-cfg.radius, -cfg.radius, -cfg.radius),
                          Gf.Vec3f(cfg.radius, cfg.radius, cfg.radius)])
    UsdPhysics.CollisionAPI.Apply(sph.GetPrim())
    _bind_material(sph.GetPrim(), mat)
    _bind_visual(sph.GetPrim(), vis)
    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().Save()
    return path


def build_dial(path: str, cfg: DialCfg) -> str:
    """预训练物体集：转盘。底座固定，圆盘绕 Z 转，三个均布凸耳供推动。

    与旋钮（``build_knob``）刻意不同：三耳而非单销、无低摩擦轮缘（D-14 是
    旋钮特有的设计）、半径更小。它提供的是"绕固定轴转动"这个**能力**，
    不是旋钮那个**任务**。
    """
    stage = _new_stage(path)
    root = _xform(stage, "/Dial")
    mat = _phys_material(stage, "/Dial/PhysMat", cfg.friction, cfg.friction)
    v_base = _vis_material(stage, "/Dial/VisBase", COLOR["base"])
    v_disc = _vis_material(stage, "/Dial/VisDisc", COLOR["dial"])
    v_lug = _vis_material(stage, "/Dial/VisLug", COLOR["lug"])

    bw, bd, bh = cfg.base_size
    base = _xform(stage, "/Dial/Base", pos=(0.0, 0.0, bh / 2))
    _rigid(base.GetPrim(), mass=20.0)
    UsdPhysics.ArticulationRootAPI.Apply(base.GetPrim())
    _box(stage, "/Dial/Base/geom", (bw, bd, bh), mat=mat, vis=v_base)
    # 立柱把圆盘抬到底座之上，否则径向接近轮缘会先撞底座（同 D-26）
    _cyl(stage, "/Dial/Base/riser", cfg.riser_radius, cfg.riser_height,
         pos=(0.0, 0.0, bh / 2 + cfg.riser_height / 2), axis="Z", mat=mat, vis=v_base)

    disc_z = bh + cfg.riser_height + cfg.disc_thickness / 2
    disc = _xform(stage, "/Dial/Disc", pos=(0.0, 0.0, disc_z))
    _rigid(disc.GetPrim(), mass=cfg.mass)
    _cyl(stage, "/Dial/Disc/geom", cfg.disc_radius, cfg.disc_thickness,
         axis="Z", mat=mat, vis=v_disc)
    for i in range(3):
        ang = 2.0 * math.pi * i / 3.0
        _cyl(stage, f"/Dial/Disc/lug{i}", cfg.lug_radius, cfg.lug_height,
             pos=(cfg.lug_offset * math.cos(ang), cfg.lug_offset * math.sin(ang),
                  cfg.disc_thickness / 2 + cfg.lug_height / 2),
             axis="Z", mat=mat, vis=v_lug)

    _joint(stage, "/Dial/DiscJoint", "revolute", "/Dial/Base", "/Dial/Disc", "Z",
           (0.0, 0.0, cfg.riser_height + bh / 2), (0.0, 0.0, -cfg.disc_thickness / 2),
           limits=(-720.0, 720.0), damping=cfg.joint_damping)
    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().Save()
    return path


def build_slab(path: str, cfg: SlabCfg) -> str:
    """探针物体集：固定斜板。

    规则 7：kinematic 刚体，不是静态碰撞体——静态碰撞体会让 filter 通道
    静默失效，region 和 mode 两个字段直接作废（P-17）。
    """
    stage = _new_stage(path)
    root = _xform(stage, "/Slab")
    mat = _phys_material(stage, "/Slab/PhysMat", cfg.friction, cfg.friction)
    vis = _vis_material(stage, "/Slab/Vis", COLOR["slab"])
    half = math.radians(cfg.tilt_deg) / 2.0
    board = _xform(stage, "/Slab/Board", pos=(0.0, 0.0, 0.0),
                   rot_wxyz=(math.cos(half), 0.0, math.sin(half), 0.0))
    _rigid(board.GetPrim(), mass=100.0, kinematic=True)
    _box(stage, "/Slab/Board/geom", cfg.size, mat=mat, vis=vis)
    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().Save()
    return path


def build_flap(path: str, cfg: FlapCfg) -> str:
    """探针物体集：竖直铰接的立板门。转轴在板的一条竖边上。"""
    stage = _new_stage(path)
    root = _xform(stage, "/Flap")
    mat = _phys_material(stage, "/Flap/PhysMat", cfg.friction, cfg.friction)
    v_base = _vis_material(stage, "/Flap/VisBase", COLOR["base"])
    v_pan = _vis_material(stage, "/Flap/VisPanel", COLOR["flap"])

    bw, bd, bh = cfg.base
    pw, pd, ph = cfg.panel
    base = _xform(stage, "/Flap/Base", pos=(0.0, 0.0, bh / 2))
    _rigid(base.GetPrim(), mass=20.0)
    UsdPhysics.ArticulationRootAPI.Apply(base.GetPrim())
    _box(stage, "/Flap/Base/geom", (bw, bd, bh), mat=mat, vis=v_base)
    _cyl(stage, "/Flap/Base/post", cfg.post_radius, ph + bh,
         pos=(0.0, 0.0, (ph + bh) / 2), axis="Z", mat=mat, vis=v_base)

    # 板竖着立在立柱旁边，铰链轴过立柱中心、沿 Z
    panel = _xform(stage, "/Flap/Panel", pos=(0.0, pd / 2 + cfg.post_radius, bh + ph / 2))
    _rigid(panel.GetPrim(), mass=cfg.mass)
    _box(stage, "/Flap/Panel/geom", (pw, pd, ph), mat=mat, vis=v_pan)

    _joint(stage, "/Flap/PanelJoint", "revolute", "/Flap/Base", "/Flap/Panel", "Z",
           (0.0, 0.0, bh / 2 + ph / 2), (0.0, -pd / 2 - cfg.post_radius, 0.0),
           limits=cfg.joint_limit_deg, damping=cfg.joint_damping)
    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().Save()
    return path


def build_plunger(path: str, cfg: PlungerCfg) -> str:
    """探针物体集：圆柱在固定套筒内滑动，外端带帽，帽背面是可勾的台肩。"""
    stage = _new_stage(path)
    root = _xform(stage, "/Plunger")
    mat = _phys_material(stage, "/Plunger/PhysMat", cfg.friction, cfg.friction)
    v_slv = _vis_material(stage, "/Plunger/VisSleeve", COLOR["base"])
    v_rod = _vis_material(stage, "/Plunger/VisRod", COLOR["plunger"])

    # 轴线高度必须让**端帽**离地：帽半径 30 mm 比套筒半径还大，
    # 按套筒半径 26 mm 放轴线的话帽子有 4 mm 埋进台面，杆被地面卡住，
    # 实测 slide_along / hook_pull 的关节位移恒为 0。
    z = cfg.cap_radius + 6 * MM
    sleeve = _xform(stage, "/Plunger/Sleeve", pos=(0.0, 0.0, z))
    _rigid(sleeve.GetPrim(), mass=20.0)
    UsdPhysics.ArticulationRootAPI.Apply(sleeve.GetPrim())
    # 套筒做成 U 形槽，**不能是实心圆柱**——实心的话杆推进去就是两个实体
    # 互相穿插，PhysX 直接把关节卡死，实测 slide_along 的关节位移恒为 0。
    # 约束本来就由 prismatic joint 提供，套筒只负责"看起来像个导向"。
    gap = cfg.rod_radius + 4 * MM
    wall = 8 * MM
    _box(stage, "/Plunger/Sleeve/bottom", (cfg.sleeve_len, 2 * (gap + wall), wall),
         pos=(0.0, 0.0, -gap - wall / 2), mat=mat, vis=v_slv)
    for sgn, nm in ((1, "l"), (-1, "r")):
        _box(stage, f"/Plunger/Sleeve/side_{nm}", (cfg.sleeve_len, wall, 2 * gap),
             pos=(0.0, sgn * (gap + wall / 2), 0.0), mat=mat, vis=v_slv)

    rod_x = cfg.sleeve_len / 2 + cfg.rod_len / 2 - 30 * MM
    rod = _xform(stage, "/Plunger/Rod", pos=(rod_x, 0.0, z))
    _rigid(rod.GetPrim(), mass=cfg.mass)
    _cyl(stage, "/Plunger/Rod/geom", cfg.rod_radius, cfg.rod_len,
         axis="X", mat=mat, vis=v_rod)
    _cyl(stage, "/Plunger/Rod/cap", cfg.cap_radius, cfg.cap_len,
         pos=(cfg.rod_len / 2 - cfg.cap_len / 2, 0.0, 0.0), axis="X", mat=mat, vis=v_rod)

    # 行程对称：杆停在中点，推得进也拉得出（同 slider 的 joint_init）
    _joint(stage, "/Plunger/RodJoint", "prismatic", "/Plunger/Sleeve", "/Plunger/Rod",
           "X", (rod_x, 0.0, 0.0), (0.0, 0.0, 0.0),
           limits=(-cfg.travel / 2, cfg.travel / 2), damping=cfg.joint_damping)
    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().Save()
    return path


def build_ridge(path: str, cfg: RidgeCfg) -> str:
    """预训练物体集：固定台面 + 一条圆柱凸棱，提供曲面接触。"""
    stage = _new_stage(path)
    root = _xform(stage, "/Ridge")
    mat = _phys_material(stage, "/Ridge/PhysMat", cfg.friction, cfg.friction)
    v_base = _vis_material(stage, "/Ridge/VisBase", COLOR["base"])
    v_rid = _vis_material(stage, "/Ridge/VisRidge", COLOR["ridge"])
    bw, bd, bh = cfg.base
    body = _xform(stage, "/Ridge/Body", pos=(0.0, 0.0, bh / 2))
    _rigid(body.GetPrim(), mass=100.0, kinematic=True)
    _box(stage, "/Ridge/Body/geom", (bw, bd, bh), mat=mat, vis=v_base)
    _cyl(stage, "/Ridge/Body/ridge", cfg.ridge_radius, cfg.ridge_len,
         pos=(0.0, 0.0, bh / 2 + cfg.ridge_radius), axis="Y", mat=mat, vis=v_rid)
    stage.SetDefaultPrim(root.GetPrim())
    stage.GetRootLayer().Save()
    return path


# ---------------------------------------------------------------- 入口

#: 物体名 -> 建模函数。配置字段名不在这里重复写，统一取 `CFG_ATTR`
#: （`it.geom_cfg`），否则加物体时要在两处保持一致。
_BUILD_FN = {
    "knob": build_knob,
    "cabinet": build_cabinet,
    "eraser": build_eraser,
    "hook": build_hook,
    "padrod": build_padrod,
    "plate0": build_plate0,
    "plate1": build_plate1,
    "slider": build_slider,
    "block": build_block,
    "column": build_column,
    "roller": build_roller,
    "ball": build_ball,
    "dial": build_dial,
    "slab": build_slab,
    "flap": build_flap,
    "plunger": build_plunger,
    "ridge": build_ridge,
}

BUILDERS = {name: (fn, CFG_ATTR[name]) for name, fn in _BUILD_FN.items()}

DEFAULT_OUT = "/workspace/interaction_transfer/assets_gen"


def build_all(out_dir: str = DEFAULT_OUT, cfg: BuildCfg | None = None) -> dict[str, str]:
    cfg = cfg or BuildCfg()
    os.makedirs(out_dir, exist_ok=True)
    made = {}
    for name, (fn, attr) in BUILDERS.items():
        p = os.path.join(out_dir, f"{name}.usd")
        fn(p, getattr(cfg, attr))
        made[name] = p
        for tag, _ in GEOM_VARIANTS.get(name, [])[1:]:
            pv = os.path.join(out_dir, f"{name}_{tag}.usd")
            fn(pv, variant_cfg(name, tag, cfg))
            made[f"{name}_{tag}"] = pv
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
