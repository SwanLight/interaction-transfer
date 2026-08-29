"""Isaac Lab 资产配置。

尺寸来自 `plan/01-assets-and-scenes.md`，几何由 `it.build_assets` 从代码生成。

**硬规则**（`plan/01` §1）：

- 规则 1：所有执行器操作完全相同尺寸的物体，不为单一执行器缩放（D-11）
- 规则 7：所有需要读接触数据的物体必须是 rigid body。不动的用
  ``kinematic_enabled=True``，静态碰撞体会让 filter 通道静默失效（P-17）
- 规则 8：ContactSensor 必须设非空 ``filter_prim_paths_expr`` 且
  ``max_contact_data_count_per_prim >= 1``
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg

ASSETS_GEN = os.environ.get("IT_ASSETS_GEN", "/workspace/interaction_transfer/assets_gen")

MM = 0.001

#: 安全法向力上限，对所有实验条件相同（`plan/01` §3.2）
MAX_NORMAL_FORCE = 25.0

#: 擦拭任务的法向力工作区间（`plan/01` §7）
WIPE_FORCE_RANGE = (3.0, 8.0)


def _usd(name: str) -> str:
    return os.path.join(ASSETS_GEN, f"{name}.usd")


def _rigid_props(max_lin_vel: float = 20.0, max_depenetration_velocity: float = 1.0,
                 angular_damping: float = 0.05, linear_damping: float = 0.05, **kw):
    """刚体属性。

    ``max_linear_velocity`` 防穿模：力控轴没有位置参考，物体会一直加速；
    单步位移超过障碍物厚度就直接穿过去，测不到任何接触。S1 实测中推子以
    2.08 m/s 穿过 20 mm 的销钉（单步位移 17 mm）。

    ``max_depenetration_velocity`` 防接触力爆炸：默认 5.0 m/s 意味着求解器
    可以用极高的速度把互相穿透的物体推开，产生巨大的冲量。S2 训练中实测
    **峰值接触力 4384 N**（均值只有 13 N），这些尖峰把奖励曲线甩到 -256。
    降到 1.0 m/s 后穿透恢复得慢一些，但力不会爆。
    """
    return sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        max_depenetration_velocity=max_depenetration_velocity,
        max_linear_velocity=max_lin_vel,
        linear_damping=linear_damping,
        angular_damping=angular_damping,
        **kw,
    )


# ---------------------------------------------------------------- 任务物体

KNOB_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Knob",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_usd("knob"),
        activate_contact_sensors=True,
        rigid_props=_rigid_props(),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, fix_root_link=True
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0), joint_pos={"DiscJoint": 0.0}, joint_vel={"DiscJoint": 0.0}
    ),
    actuators={
        "disc": ImplicitActuatorCfg(
            joint_names_expr=["DiscJoint"], effort_limit=200.0, velocity_limit=20.0,
            stiffness=0.0, damping=0.28,   # 必须与 build_assets 的 joint_damping 一致
        )
    },
)
"""旋钮。轮缘 μ=0.15 / 销钉 μ=0.8（D-14），使 region 不可从 effect 推出。"""


CABINET_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Cabinet",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_usd("cabinet"),
        activate_contact_sensors=True,
        rigid_props=_rigid_props(),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, fix_root_link=True
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0), joint_pos={"DrawerJoint": 0.0}, joint_vel={"DrawerJoint": 0.0}
    ),
    actuators={
        "drawer": ImplicitActuatorCfg(
            joint_names_expr=["DrawerJoint"], effort_limit=200.0, velocity_limit=5.0,
            # 必须与 build_assets 的 joint_damping 一致；理由见那里的注释
            stiffness=0.0, damping=30.0,
        )
    },
)
"""抽屉。阴性对照任务（Gate G），物理参数不刻意制造隐藏困难。"""


ERASER_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Eraser",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_usd("eraser"), activate_contact_sensors=True, rigid_props=_rigid_props()
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.05)),
)
"""黑板擦。擦拭实现 (a) 持工具擦用它；实现 (b) 直擦不用（D-12）。"""


def board_cfg(size=(600 * MM, 500 * MM, 20 * MM), friction: float = 0.35) -> RigidObjectCfg:
    """擦拭平面。

    **必须是 kinematic 刚体，不能是静态碰撞体**——它是 ContactSensor 的 filter
    目标，静态碰撞体会让 ``force_matrix_w`` / ``contact_pos_w`` /
    ``get_friction_data`` 全部静默失效（规则 7，P-17）。

    按 `plan/02` §1.1，擦拭任务的**被操作物体就是这个平面**（不是黑板擦），
    因为任务成功判据是平面上的污渍被清除。这也是两种实现能共享同一份
    envelope 的前提。
    """
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Board",
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=friction, dynamic_friction=friction, restitution=0.0,
                friction_combine_mode="min", restitution_combine_mode="min",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.35, 0.25)),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -size[2] / 2)),
    )


# ---------------------------------------------------------------- 执行器

def plate_cfg(idx: int) -> RigidObjectCfg:
    """双板 source 的单块薄板（0 或 1），由 ``build_assets`` 生成的 USD 载入。

    **普通刚体 + PD 外力驱动，绝不能设 kinematic**——kinematic body 不参与
    动力学求解，能施加无限大的力，接触力将失去物理意义（P-09）。
    自检判据：自由落体后稳态接触力 ≈ mg。

    局部坐标 **+Z 是工作面法向**，长边 35 mm 在 X、短边 25 mm 在 Y。
    USD 里带两个**纯视觉**朝向标记（工作面浅色贴片 + 顶边深色鳍），
    碰撞几何仍然只有那一个 35×25×3 mm 的盒子，与改版前逐字等价。
    改用 USD 的原因见 ``build_assets.PlateCfg``：录像里必须能看出板子朝向，
    否则无法人工确认"接触发生在工作面上"。

    ``max_lin_vel`` 收到 6 m/s：PD 的力上限能给出很大的加速度，一旦目标位姿
    跳变就会在一个物理步内穿过把手。

    ``max_depenetration_velocity`` 从全局的 1.0 再降到 0.25：板贴着圆柱做
    力控时，求解器每次把微小穿透以 1 m/s 弹开，板被弹离再被推回，接触变成
    高频断续——S3 实测只有 12% 的物理子步能报到接触，而抽屉是被一连串冲击
    推开的。`plan/02` §3.4 要记 stick/slide，断续接触既不是 stick 也不是 slide，
    这种数据没法用。
    """
    if idx not in (0, 1):
        raise ValueError(f"只有 plate0 / plate1 两块板，收到 idx={idx}")
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Plate{idx}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_usd(f"plate{idx}"), activate_contact_sensors=True,
            rigid_props=_rigid_props(max_lin_vel=6.0,
                                     max_depenetration_velocity=0.25),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.3)),
    )


# ---------------------------------------------------------------- 预训练物体集
#
# `plan/03` §2.4：这一组**只用于产生指令多样性，永不作为任务评估**。
# 它不进 E-T、不进 Shared Structure Model、不进任何评估。
# 覆盖设计与"够不够"的论证见 `plan/03` §2.4 与 `log/decisions.md` D-39。

BLOCK_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Block",
    spawn=sim_utils.UsdFileCfg(usd_path=_usd("block"), activate_contact_sensors=True,
                               rigid_props=_rigid_props()),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.02)),
)
"""自由方块：推移 / 侧推翻倒 / 按住不动。覆盖**无约束物体**这一档。"""


COLUMN_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Column",
    spawn=sim_utils.UsdFileCfg(usd_path=_usd("column"), activate_contact_sensors=True,
                               rigid_props=_rigid_props()),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.06)),
)
"""自由立柱：侧推 / 双面搓转 / 推倒。覆盖**切向摩擦主导**的力学。"""


ROLLER_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Roller",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_usd("roller"), activate_contact_sensors=True,
        # 滚动阻力：PhysX 不建模滚阻，自由圆柱一推就无限加速——实测被推着
        # 跑出 2.9 m、94% 的操作步脱手。角阻尼是滚阻的标准替代，
        # 8.0 对应约 0.12 s 的衰减时间常数——低于这个值滚柱会带着动量滑出去，
        # 实测 3.0 时它冲过推板 3 倍的行程。滚柱只在被推时滚。
        rigid_props=_rigid_props(angular_damping=8.0, linear_damping=8.0)),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.03)),
)
"""卧倒的自由滚柱：原语 P7 roll / P2 push / P1 press / P12 poke。

冗余规则要求 P7 不能只有一个物体，而立着的 `column` 受侧推只会倒不会滚。
"""


BALL_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Ball",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_usd("ball"), activate_contact_sensors=True,
        # 滚阻替代，同 roller：PhysX 不建模滚阻，自由球一推就无限加速
        rigid_props=_rigid_props(angular_damping=8.0, linear_damping=8.0)),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.035)),
)
"""自由球：P7 rolling contact 的第二个承载物体（冗余规则）。

圆柱只能绕一根轴滚，球在任意方向都滚；接触拓扑也不同（线接触 vs 点接触）。
"""


DIAL_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Dial",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_usd("dial"), activate_contact_sensors=True, rigid_props=_rigid_props(),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, fix_root_link=True),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0), joint_pos={"DiscJoint": 0.0}, joint_vel={"DiscJoint": 0.0}),
    actuators={
        "disc": ImplicitActuatorCfg(
            joint_names_expr=["DiscJoint"], effort_limit=100.0, velocity_limit=8.0,
            # 必须与 build_assets 的 joint_damping 一致
            stiffness=0.0, damping=0.20,
        )
    },
)
"""转盘：覆盖**受约束转动**。与旋钮刻意不同（三耳、无低摩擦轮缘），见 D-39。"""


SLAB_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Slab",
    spawn=sim_utils.UsdFileCfg(usd_path=_usd("slab"), activate_contact_sensors=True),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
)
"""固定斜板：原语 P1 press / P4 rub / P5 shear / P12 poke。

USD 里已设 ``kinematic_enabled=True``（规则 7 / P-17），所以这里不再传
``rigid_props``——传了会用默认值把 kinematic 覆盖掉。
"""


FLAP_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Flap",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_usd("flap"), activate_contact_sensors=True, rigid_props=_rigid_props(),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, fix_root_link=True),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0), joint_pos={"PanelJoint": 0.0}, joint_vel={"PanelJoint": 0.0}),
    actuators={
        "panel": ImplicitActuatorCfg(
            joint_names_expr=["PanelJoint"], effort_limit=60.0, velocity_limit=8.0,
            # 必须与 build_assets 的 joint_damping 一致
            stiffness=0.0, damping=2.0,
        )
    },
)
"""翻板：原语 P9 crank / P4 rub / P1 press / P12 poke。

与 `dial` 同属受约束转动（E4）但转轴在物体**边缘**、力臂随开合角变化，
是 `plan/03` §2.4.5 冗余规则要求的第二个 E4 物体。
"""


PLUNGER_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Plunger",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_usd("plunger"), activate_contact_sensors=True,
        rigid_props=_rigid_props(),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, fix_root_link=True),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0), joint_pos={"RodJoint": 0.0}, joint_vel={"RodJoint": 0.0}),
    actuators={
        "rod": ImplicitActuatorCfg(
            joint_names_expr=["RodJoint"], effort_limit=120.0, velocity_limit=5.0,
            stiffness=0.0, damping=1.5,
        )
    },
)
"""柱塞：原语 P10 slide-along / P11 hook-pull / P1 press / P12 poke。

与 `slider` 同属受约束平移（E3）但几何完全不同（圆柱+套筒 vs 方块+导轨，
端帽台肩 vs 侧向凸缘），是冗余规则要求的第二个 E3 与 P11 物体。
"""


RIDGE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Ridge",
    spawn=sim_utils.UsdFileCfg(usd_path=_usd("ridge"), activate_contact_sensors=True),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
)
"""固定凸棱台：覆盖**曲面上的接触**。抽屉把手是圆柱，而钩杆要零样本抽屉。"""


HOOK_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Hook",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_usd("hook"), activate_contact_sensors=True,
        rigid_props=_rigid_props(max_lin_vel=1.0),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.4)),
)
"""L 形钩杆，0 自由度。用于旋钮和抽屉，不参加擦拭（无法稳定施加法向压力）。"""


PADROD_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/PadRod",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_usd("padrod"), activate_contact_sensors=True, rigid_props=_rigid_props()
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.4)),
)
"""垫头杆，0 自由度，只做擦拭直擦。

`plan/04` §5.4 里它是最强的一格：一个完全不能抓握的执行器，训练中只见过
预训练物体集，直接零样本执行擦拭 envelope。
"""


SLIDER_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Slider",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_usd("slider"),
        activate_contact_sensors=True,
        rigid_props=_rigid_props(),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, fix_root_link=True
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(joint_pos={"BlockJoint": 0.0}),
    actuators={
        "block": ImplicitActuatorCfg(
            joint_names_expr=["BlockJoint"], effort_limit=100.0, velocity_limit=5.0,
            stiffness=0.0, damping=2.0,
        )
    },
)
"""预训练物体集之一。只用于给 E-I 产生交互指令多样性，永不作为任务评估。"""


# ---------------------------------------------------------------- 内置资产

ISAAC_ASSETS = {
    "allegro": "Isaac/Robots/WonikRobotics/AllegroHand/allegro_hand_instanceable.usd",
    "franka": "Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
    "panda_instanceable": "Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd",
}
"""已由 tools/fetch_assets.py 拉到 /mnt/isaacsim_assets 的内置资产相对路径。"""


def allegro_cfg(prim_path: str = "{ENV_REGEX_NS}/Allegro") -> ArticulationCfg:
    """Allegro 手，浮动底座。

    按规则 1（D-11），物体尺寸对所有执行器一致，**不为 Allegro 单独缩放**。
    若几何不可行，整场景统一缩放。
    """
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/WonikRobotics/AllegroHand/allegro_hand_instanceable.usd",
            activate_contact_sensors=True,
            rigid_props=_rigid_props(),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, fix_root_link=False
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            # thumb_joint_0 的限位是 [0.279, 1.571]，USD 默认 0.0 越界会抛 ValueError。
            # 只覆盖它——Isaac Lab 不允许 ".*" 与具体名字的模式重叠。
            joint_pos={"thumb_joint_0": 0.4},
        ),
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[".*"], effort_limit=0.5, velocity_limit=100.0,
                stiffness=3.0, damping=0.1,
            )
        },
    )


def pusher_cfg(size=(30 * MM, 30 * MM, 30 * MM), mass: float = 0.2,
               friction: float = 0.9) -> RigidObjectCfg:
    """标定用推子。仅用于 S1 摩擦标定，不进入任何任务。

    自身摩擦故意设高（0.9），用来验证 ``friction_combine_mode="min"`` 确实生效：
    高摩擦推子碰低摩擦轮缘时，接触摩擦必须取 min(0.9, 0.10) = 0.10。
    若组合模式是默认的 average，这里会得到 0.5，D-14 直接失效。
    """
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Pusher",
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=_rigid_props(max_lin_vel=0.6),   # 单步位移 5 mm < 销钉直径 20 mm
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=friction, dynamic_friction=friction, restitution=0.0,
                friction_combine_mode="min", restitution_combine_mode="min",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.2)),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, 0.0, 0.1)),
    )
