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


def _rigid_props(max_lin_vel: float = 20.0, **kw):
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
        max_depenetration_velocity=1.0,
        max_linear_velocity=max_lin_vel,
        linear_damping=0.05,
        angular_damping=0.05,
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
            stiffness=0.0, damping=3.0,
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

def plate_cfg(idx: int, size=(35 * MM, 25 * MM, 3 * MM), mass: float = 0.05,
              friction: float = 0.9) -> RigidObjectCfg:
    """双板 source 的单块薄板。

    **普通刚体 + PD 外力驱动，绝不能设 kinematic**——kinematic body 不参与
    动力学求解，能施加无限大的力，接触力将失去物理意义（P-09）。
    自检判据：自由落体后稳态接触力 ≈ mg。
    """
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Plate{idx}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=friction, dynamic_friction=friction, restitution=0.0,
                friction_combine_mode="min", restitution_combine_mode="min",
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.9, 0.6, 0.2) if idx == 0 else (0.2, 0.6, 0.9)
            ),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.3)),
    )


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
