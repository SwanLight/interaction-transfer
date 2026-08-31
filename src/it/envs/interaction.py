"""E-I：交互跟踪执行器的环境（`plan/04` §5）。主系统就是它。

与 `envs/drawer.py`（E-T / Expert 的载体）的根本区别：**这个环境不知道任务是什么**。
它的 reward 只比"实测的交互"与"指令要求的交互"，没有任何一项读任务的成功判据、
关节角或 dirt 网格。物体是什么、要拉开多少，全部只经由 artifact 的指令通道进来。

六个设计点，都是被实测或文献钉住的
----------------------------------

**一、场景必须与"产生这份指令时"的场景一致。** 台面、物体初始位姿、接触 filter 指到
哪个 prim，全部从 `it.probe_scene` 取——那是采集侧与这里**共用的唯一真值**。
第一版这里自己另写了一套，两个后果都不报错：探针物体全是带重力的自由刚体而场景里
**没有台面**（开局自由落体），`ridge`/`slab` 的刚体在子 prim 上而 filter 指到了根
（P-17：接触位置与摩擦力静默失效）。2026-08-30 那次 dry-run 的
``region/mode/mech`` 三项**恒等于 0**，当时被读成"量级合理"，实际含义是全程一次
接触都没有发生。

**二、动作空间是混合力/位控，不是纯位姿增量。** 指令要跟踪的是 *traction*
（面密度），而接触力是穿透量的刚性函数：`float_ctrl` 的文档里记着实测，把推子放在
距表面 0.5 mm 处法向力只有 0.06 N，而目标是 25 N。纯位置动作等于要求 PPO 学会
亚毫米定位。接触密集操作的 RL 文献一致的做法是把**柔顺/力设定值放进动作空间**
（variable impedance / admittance，Buchli；Martín-Martín；Beltran-Hernandez
[arXiv 2003.00628]）。

**但力控只在有接触时才打开。** 混合力/位控里"可以力控的子空间"**就是由接触约束
定义的**（Raibert & Craig 的原始表述），自由空间里没有那个子空间。第一版只按动作
幅度判是否力控：0.3 kg 的杆 + 25 N 前馈 + ``kd_force=40`` 的终端速度是 2.1 m/s，
随机策略一定把自己甩飞。采集侧的 ``live = engage>0`` 是同一条规矩，P-45
（"先用位置干涉压出实接触，再交给力控"）也是。

**方向由策略自己出，不直接用指令里的 engage 方向。** 否则 C0/C1（没有方向字段）
的动作空间就和 C4 不一样了，而 `plan/04` §7 要求所有条件的结构与参数量逐位相同。

**三、接触逐物理子步采，取法向力最大的那一子步**（P-31）。采集侧就是这么记的
（`s3_source_probe.py` 的 `best` 循环）。只在控制步末尾采一次会漏掉大半接触——
那会变成 P-72 的第二例：同一个量的离线/在线两份实现，`s6_reward_probe.py`
只对过**公式**，没有对过**采样**。

**四、滑移速率用位姿差分算，与 S4 同一个判据（D-49）。** PhysX 报的瞬时相对速度在
角速度饱和时不可信（P-52），且离线的 `mode/pose_slip` 用的就是两刚体位姿差分——
在线换一个算法，reward 追的就不是 artifact 里写的那个量。它按**控制步**算，
与 S4 逐帧记录的粒度一致（所以这一处不跟着子步走）。

**五、失败终止必须有显式惩罚**（D-31 第 3 个洞）。五项跟踪 reward 全部 ≤ 0
（hinge：落在允许集合内恰好为零），于是"立刻结束 episode"的回报是 0，与完美执行
并列最优。第一版把"执行器跑出 1.5 m"当无惩罚的 terminated，配上第二条那个自由
空间力控，就是一条**学得会的作弊路径**：开局把自己甩出去。惩罚按**剩余步数**给，
所以早退不会比留下来干活便宜。

**六、reward 各项按成功示教上标定的量级归一**（D-31 第 2 个洞）。四项实测量级差
210×（`effect 0.147` vs `mech 0.0007`）。标定值由 `tools/s6_reward_probe.py` 产生，
**没有标定就不许构造这个环境**——第一版是训练入口读了、打印了、然后丢掉。

**七、接触取 `extract_contact_points_padded`。** `contact_summary` /
`extract_contact_points` 那条路径的摩擦力是错的（P-36），而且它逐 env 循环，
2048 个 env 下根本跑不动。
"""

from __future__ import annotations

import glob

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObject
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

from it import assets as A
from it import probe_scene as PS
from it.contact_utils import extract_contact_points_padded
from it.ei_command import CommandBank, WindowTracker
from it.ei_reward import (RewardWeights, assign_cells, contact_force_on_object,
                          effect_magnitude, interaction_reward,
                          match_functional_region, nearest_cell,
                          normal_orientation_sign, scatter_contact_compatibility,
                          surface_traction)
from it.envs.base import FloatingBaseAction
from it.float_ctrl import FloatingPD

#: 每个 env 每步保留的接触点数。定长是必须的（`plan/02` §7 第 3 条：
#: 改变接触体数量后表示维度不变）。
MAX_CONTACTS = 16
#: 判"有接触"的法向力合力阈值（N）。与采集侧 `s3_source_probe` 的 ``touch`` 判据
#: （0.05 N）取同一个数——力控门控与"接触已建立"用同一条线，免得两处各拍一个。
CONTACT_FORCE_MIN = 0.05
_TINY = 1e-9
#: 本体/反馈观测的维度，见 `_proprio`：
#: 执行器位姿 7 + 速度 6 + 物体位姿 7 + 相对位置 3 + 物体速度 6 + 上一动作 9
#: + 接触反馈 6 + **通用进度量 6**。
#:
#: ⚠️ 那 6 个进度量是**策略**观测不是特权观测。`plan/04` §4 要 E-I 有"当前 phase 与
#: progress"，但必须由通用量导出——它们正是那个导出结果（是否已建立接触、effect 完成
#: 比例、窗口位置、停留占比、是否超时、是否跑完）。放进特权里策略就看不到，
#: 而没有它策略无从知道自己在指令的哪一段。
PROPRIO_DIM = 50
#: critic 独享的特权观测维度：物体质量 1 + 力设定值 3 + 停留步数 1。
PRIVILEGED_DIM = 5


@configclass
class InteractionEnvCfg(DirectRLEnvCfg):
    decimation = 3                      # 1/150 × 3 = 50 Hz（`plan/04` §9）
    #: 32 个命令格 × `MIN_DWELL`=2 步 = 64 步是下限；300 步（6 s）留了四倍余量，
    #: 而更短的 episode 意味着更频繁地重置回锚点附近——那是探索的主要来源。
    episode_length_s = 6.0
    action_space = 9                    # 6 位姿增量 + 3 力矢量
    #: 观测里**不放**指令张量本身：逐格空间场是 256×34，2048 个 env × horizon 32
    #: 的 PPO buffer 会到几个 GB。放的是 (command_index, bin_index)，
    #: 由 `ei_policy` 那侧按下标去 `CommandBank` 取——指令在 episode 内是常量，
    #: 变的只有格号。
    observation_space = PROPRIO_DIM + 2 + 4
    state_space = PROPRIO_DIM + 2 + 4 + PRIVILEGED_DIM

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1 / 150, render_interval=3,
        physx=sim_utils.PhysxCfg(gpu_max_rigid_contact_count=2 ** 23,
                                 gpu_max_rigid_patch_count=2 ** 21))
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=2.0,
                                                     replicate_physics=True)
    light: object = AssetBaseCfg(prim_path="/World/light",
                                 spawn=sim_utils.DomeLightCfg(intensity=800.0))

    #: 被操作物体与执行器由 curriculum 换（`plan/04` §10 的 A/B/C 三段）。
    #: **不要手写这三项**，用 `probe_env_cfg()`——那里会一并把台面、初始位姿和
    #: 接触 filter 配成与采集侧一致的一套。
    object_cfg: object = A.BLOCK_CFG.replace(prim_path="/World/envs/env_.*/Object")
    executor_cfg: object = A.PADROD_CFG.replace(prim_path="/World/envs/env_.*/Executor")
    #: 物体刚体相对其根 prim 的子路径（`probe_scene.ProbeObject.body_subpath`）。
    #: 空串表示根 prim 自身就是刚体。**指错这里 = P-17，接触位置与摩擦力静默失效。**
    object_body_subpath: str = ""
    #: 执行器名字（`probe_scene.EXECUTOR_TIP` 的键）。用来把末端摆到接触锚点上。
    executor_name: str = "padrod"
    #: 台面外廓（m）。物体就放在它上面，顶面在 z=0。
    table_extent: float = PS.TABLE_EXTENT_TRAIN
    #: 接触点归属到本 env 的最大半径（m），传给 `extract_contact_points_padded`。
    #:
    #: ⚠️ **默认值 0.08 对杆类执行器是错的**：垫头杆的垫底面离根 prim 0.108 m、
    #: 钩杆的钩心 0.117 m，接触全部会被当成 "foreign" 丢掉——一个不报错、只让
    #: 接触整体消失的坑。采集侧用 0.06 是因为双板很小。这里按执行器的最大外廓取，
    #: 上限由 env_spacing（2.0 m）给，取 0.25 有一个数量级的裕量。
    contact_own_radius: float = 0.25
    #: target embodiment 接触斑块的空间分辨率（Gaussian sigma, m）。它属于 decoder，
    #: 不写回 interaction：大垫面与细钩可以用不同接触拓扑实现同一功能区域。
    region_match_sigma: float = 4.0e-3

    #: 指令库：一批冻结 artifact 的路径。**留出任务由 `forbid` 在入口拦**
    #: （`plan/04` §5.4：不得只靠自觉）。
    command_glob: str = "/tmp/s6/probe/*.npz"
    # 训练课程应把解析后的**确切路径**传进来；glob 只保留给 smoke/单场景装配。
    # 否则课程 C 虽解析了 task artifact，环境却可能又只展开 probe glob。
    command_paths: tuple[str, ...] = ()
    forbid: tuple[str, ...] = ()

    #: 各项 reward 在成功示教上的量级，由 `tools/s6_reward_probe.py` 标定。
    #: **必须给**，缺了就在构造时报错（D-31 第 2 个洞）。
    reward_scale: dict = {}
    max_contact_force: float = A.MAX_NORMAL_FORCE
    #: 动作里力矢量的量程（N）。上限取安全法向力，越界由 r_safety 罚。
    force_scale: float = A.MAX_NORMAL_FORCE
    #: 接触控制用毫米级增量。旧值 10 mm 配合 std=0.8 时，300 步随机游走 RMS
    #: 约 14 cm，远大于整个接触邻域；PPO 学到任何接触反馈前就已经漂走。
    action_pos_scale: float = 0.002
    action_rot_scale: float = 0.02
    action_pos_limit: float = 0.12
    #: 接触前用 interaction 自己的 region + normal 做通用伺服。它不读 task，也不读
    #: source 动作；只把任意 embodiment 的已知末端带到 contact-ready pose。原始方案
    #: 本来就把 free-space approach 与 contact RL 分开，E-I 学的是接触后的物理解码。
    use_precontact_servo: bool = True
    precontact_clearance: float = 0.004
    precontact_handoff_tolerance: float = 0.002
    #: 四项跟踪 reward 的权重。四项都已经在 [−1, 1] 里（`ei_reward.RewardTerms`），
    #: 所以权重就是**相对重要性**，可以直接读。
    #:
    #: ⚠️ `w_region` 必须大于 `w_mech + w_mode`。`r_region` 是唯一能取正值的项，
    #: 也是"建立接触"的全部动力；若压对地方的正收益抵不过力学不完美的负收益，
    #: 最优策略就是**永远别碰那个物体**（`ei_reward` docstring 第四条，实测
    #: 悬停 −0.43/步 vs 压住 −278/步）。这一条由 `tools/s6_smoke.py` 的
    #: "压对 > 悬停" 那一格实测把关，**不靠这段注释保证**。
    w_effect: float = 1.0
    w_region: float = 2.0
    w_mode: float = 0.25
    w_mech: float = 0.75
    w_force: float = 0.05
    w_action: float = 0.002
    #: 失败终止的惩罚，按**剩余控制步数的折现和**给（见模块 docstring 第五条）。
    #: 单位是"每剩余一步的归一化 reward"，量级应当 ≥ 悬停不动时的单步代价，
    #: 否则早退仍然划算。默认值由 `tools/s6_smoke.py` 实测的悬停代价定。
    fail_penalty_per_step: float = 4.0
    #: 折现率，**必须与 PPO 的 gamma 一致**（`plan/04` §9 是 0.99）。用折现和而不是
    #: 裸的剩余步数：裸和在第 1 步失败时是 −1600，比正常回报大一个数量级，会把
    #: value 的目标撑坏；折现和的上界是 `1/(1-γ)` 倍单步代价，与正常回报同量级。
    fail_penalty_gamma: float = 0.99
    #: 执行器初始站位：沿指令接触锚点的外法向退开这么远（m），区间内均匀采样。
    #: 见 `ei_command.CommandBank._contact_anchor` 的说明。
    init_standoff: tuple[float, float] = (0.02, 0.10)
    #: 站位的横向抖动与姿态抖动。0 会让 2048 个 env 完全相同，探索没有多样性。
    init_jitter_m: float = 0.02
    init_jitter_rad: float = 0.15
    #: 接近段的稠密代价的量程（无量纲）：离锚点 `approach_ref` 以外每步罚 `w_approach`，
    #: 贴到锚点恰好 0。取 0.5 = "压对地方"（`w_region`=2）的四分之一。0 = 关掉。
    #:
    #: 为什么需要它：跟踪 reward 里唯一为正的 `r_region` 只在**有接触时**才非零，
    #: 而 `r_effect` 的缺口在接近段与距离无关。于是接近段完全没有梯度。实测随机
    #: 策略只有 **0.6%** 的步碰得到物体（`tools/s6_smoke.py` 第八节），探索几乎
    #: 发现不了那点正收益。
    #:
    #: 为什么可以加：这一项是 `F = γΦ(s') − Φ(s)` 的**势函数式**塑形
    #: （Ng, Harada & Russell 1999），**可证明不改变最优策略**——它只重新分配
    #: 回报的时间分布，不能把"贴着物体不干活"变成更优。而且 Φ 只读指令里的
    #: 接触锚点（`CommandBank.anchor_pos`），**不含任何任务量**。
    #: `plan/04` §8 那条"禁止只给 C4 加 tracking reward"只约束 E-T。
    #:
    #: ⚠️ **Φ 必须非负、且在远处为零。** 第一版取 `Φ = −5·d`（离得越远越负），
    #: 在 γ<1 下 `F = γΦ' − Φ` 原地不动时等于 `(γ−1)Φ = +0.01·5·d`——
    #: **离得越远，白拿的越多**。实测 24 轮之内策略就跑到 `anchor_distance=0.465 m`
    #: 上坐着不动，`shaping=+0.019/步`、总回报 +7.0 并从此持平，
    #: 而 `diag/touching` 全程 **0.0000**。Ng 等 1999 的策略不变性对**无穷时域最优解**
    #: 成立，但它管不住"有限时域里 PPO 会先爬上哪个坡"——而那条坡的方向由 Φ 的符号定。
    #: 现在 `Φ = w·(1 − min(d, d_ref)/d_ref) ≥ 0`：远处恰好为 0（不白拿），
    #: 贴着物体不干活反而每步漏掉 `0.01·w`（很小，但方向是对的）。
    w_approach: float = 0.5
    #: 接近塑形的作用距离（m）。超出这个距离 Φ=0：没有梯度，也没有漂移。
    #:
    #: ⚠️ **必须覆盖整个可达集合**（`√3·pos_limit` ≈ 0.43 m），否则执行器一旦
    #: 随机游走出这个半径就进入一片**完全平坦**的区域，再也没有任何信号把它带回来。
    #: 实测 `approach_ref=0.15` 时，三个 padrod 的组合在第 5 轮就已经停在
    #: `anchor_distance ≈ 0.46 m`（正好是立方体的角）且 `touching` 恒为 0。
    #: `_init_buffers` 里有一条硬检查钉住这个关系。
    approach_ref: float = 0.26


def probe_env_cfg(object_name: str, executor_name: str, *, num_envs: int,
                  command_glob: str, forbid: tuple[str, ...],
                  reward_scale: dict) -> InteractionEnvCfg:
    """按 `it.probe_scene` 装配一个与采集侧物理一致的 E-I 环境配置。

    **训练入口与冒烟脚本都必须走这里**，不要各自拼 cfg——"同一组事实有两份实现"
    正是 P-72 的形状，也正是第一版没有台面、filter 指错 prim 的来源。
    """
    executors = {"padrod": A.PADROD_CFG, "hook": A.HOOK_CFG}
    objects = {"block": A.BLOCK_CFG, "column": A.COLUMN_CFG, "ball": A.BALL_CFG,
               "roller": A.ROLLER_CFG, "slab": A.SLAB_CFG, "ridge": A.RIDGE_CFG}
    if executor_name not in executors:
        raise SystemExit(f"{executor_name} 的资产还没接进来；现有 {sorted(executors)}")
    if object_name not in objects:
        raise SystemExit(f"{object_name} 还没接进来；现有 {sorted(objects)}")
    spec = PS.PROBE_OBJECTS[object_name]
    if spec.articulated:
        raise SystemExit(
            f"{object_name} 是 articulation。接它要一并搬 probe_scene 里的 joint 与"
            "damping_nominal（采集时是在运行时写进关节的，会盖过资产里的值，P-38），"
            "本轮还没做")

    cfg = InteractionEnvCfg()
    cfg.scene.num_envs = num_envs
    object_cfg = objects[object_name]
    cfg.object_cfg = object_cfg.replace(
        prim_path="/World/envs/env_.*/Object",
        init_state=type(object_cfg.init_state)(pos=spec.init_pos))
    cfg.object_body_subpath = spec.body_subpath
    cfg.executor_cfg = executors[executor_name].replace(
        prim_path="/World/envs/env_.*/Executor")
    cfg.executor_name = executor_name
    cfg.region_match_sigma = PS.EXECUTOR_REGION_SIGMA[executor_name]
    # P-52 的建议："下轮采集显式设 max_angular_velocity"。PhysX 默认 100 rad/s，
    # 杆类执行器一旦被接触冲量拍到那个上限，`root_ang_vel_w` 就不再可信，
    # 而它既进观测又进滑移诊断。20 rad/s 远高于任何有意义的操作转速。
    cfg.executor_cfg.spawn.rigid_props.max_angular_velocity = 20.0
    cfg.command_glob = command_glob
    cfg.forbid = forbid
    cfg.reward_scale = dict(reward_scale)
    return cfg


class InteractionEnv(DirectRLEnv):
    """交互跟踪执行器。**reward 里没有任何一项知道任务是什么。**"""

    cfg: InteractionEnvCfg

    def _setup_scene(self):
        if not self.cfg.reward_scale:
            raise ValueError(
                "reward_scale 是空的。各项 reward 的量级必须在成功示教上标定，不许拍"
                "（D-31 第 2 个洞：量纲失衡时最优解是不动）——先跑 "
                "tools/s6_reward_probe.py，把它的 scale 传进 cfg")
        self.obj = RigidObject(self.cfg.object_cfg)
        self.executor = RigidObject(self.cfg.executor_cfg)
        # 台面：与采集侧同一份材质、同一个厚度、顶面同样在 z=0（`it.probe_scene`）。
        # 没有它，block/column/ball/roller 这些自由刚体开局就自由落体，
        # 而指令说的是"在桌面上推方块"。
        e = self.cfg.table_extent
        self.table = RigidObject(A.board_cfg(
            size=(e, e, PS.TABLE_THICKNESS), friction=PS.TABLE_FRICTION
        ).replace(prim_path="/World/envs/env_.*/Table"))
        root = "/World/envs/env_.*/Object"
        sub = self.cfg.object_body_subpath
        filter_expr = f"{root}/{sub}" if sub else root
        self.contact = ContactSensor(ContactSensorCfg(
            prim_path="/World/envs/env_.*/Executor", track_contact_points=True,
            max_contact_data_count_per_prim=MAX_CONTACTS,
            filter_prim_paths_expr=[filter_expr],
            update_period=0.0, history_length=0))
        self.scene.rigid_objects["object"] = self.obj
        self.scene.rigid_objects["executor"] = self.executor
        self.scene.rigid_objects["table"] = self.table
        self.scene.sensors["contact"] = self.contact
        self.cfg.light.spawn.func("/World/light", self.cfg.light.spawn)
        self.scene.clone_environments(copy_from_source=False)

    def _configure_gym_env_spaces(self):
        super()._configure_gym_env_spaces()
        self._init_buffers()

    def _init_buffers(self):
        n, dev = self.num_envs, self.device
        paths = (list(self.cfg.command_paths) if self.cfg.command_paths
                 else sorted(glob.glob(self.cfg.command_glob)))
        self.bank = CommandBank(paths, device=dev,
                                forbid=self.cfg.forbid)
        self.tracker = WindowTracker(self.bank, n, device=dev)
        # ⚠️ `pos_limit` 是**逐轴**的，可达集合是个立方体，对角线是 √3·pos_limit。
        # 第一版 pos_limit=0.5 而 `far` 门槛 0.75 m：立方体的角在 0.87 m，
        # 随机策略把目标走到角上就必然触发失败终止。实测后果是
        # **400 轮训练里 mean reward 恒等于 −99.13**——每条 episode 都以失败收场，
        # 优势信号几乎为零，PPO 完全学不动，而曲线看起来只是"平"。
        # 交互本身发生在 10 cm 尺度上，0.25 m 的活动范围已经宽裕。
        # ⚠️ 探索体积必须与**交互本身的尺度**相称，不是与场景的尺度相称。
        # 交互发生在锚点附近 ~5 cm 内，初始站距 2~10 cm。`pos_limit=0.25`（逐轴）
        # 的可达立方体角在 0.43 m，实测策略在 15 轮内就扩散到 `anchor_distance≈0.46`
        # 并再也回不来（`diag/touching` 恒为 0）——不是学坏了，是随机游走在一个
        # 比目标大一个数量级的体积里根本撞不到目标。0.12 m 仍然比初始站距宽裕一倍。
        self.act = FloatingBaseAction(
            n, dev, pos_scale=self.cfg.action_pos_scale,
            rot_scale=self.cfg.action_rot_scale, pos_limit=self.cfg.action_pos_limit)
        # 旋转增益按**转动惯量**归一（`float_ctrl` 的 `rot_gain_basis`）。用质量归一
        # 是量纲错的，杆类执行器绕自身轴的惯量小两个数量级，同一组增益在 1/150 s 的
        # 步长下直接发散——实测冒烟里杆的角速度 76 rad/s（P-52 的同一族）。
        # 增益因此读作"每弧度多少 rad/s²"：ω_n=√60≈7.8 rad/s、ζ=8/(2√60)≈0.52。
        self.pd = FloatingPD(self.executor, kp_pos=600.0, kd_pos=50.0,
                             kp_rot=60.0, kd_rot=8.0, max_force=180.0,
                             max_torque=6.0, kd_force=40.0,
                             rot_gain_basis="inertia")
        self.weights = RewardWeights(
            effect=self.cfg.w_effect, region=self.cfg.w_region, mode=self.cfg.w_mode,
            mech=self.cfg.w_mech, safety=1.0,
            scale=dict(self.cfg.reward_scale), calibrated=True)
        self.actions = torch.zeros(n, self.cfg.action_space, device=dev)
        self.prev_actions = torch.zeros_like(self.actions)
        self.tgt_pos = torch.zeros(n, 3, device=dev)
        self.tgt_quat = torch.zeros(n, 4, device=dev)
        self.tgt_quat[:, 0] = 1.0
        self.ff_force = torch.zeros(n, 3, device=dev)
        self.progress_obs = torch.zeros(n, 6, device=dev)
        self.contact_obs = torch.zeros(n, 6, device=dev)
        self.spawn_pos = torch.zeros(n, 3, device=dev)
        reach = 3.0 ** 0.5 * self.act.pos_limit
        if self.cfg.w_approach > 0 and self.cfg.approach_ref < reach:
            raise ValueError(
                f"approach_ref={self.cfg.approach_ref} 小于可达半径 {reach:.3f} m"
                f"（√3·pos_limit）。执行器游走出这个半径之后 Φ 恒为 0，"
                "接近段就再没有梯度——实测策略会停在立方体的角上、全程不接触")
        self.tip_local = torch.tensor(PS.EXECUTOR_TIP[self.cfg.executor_name],
                                      device=dev, dtype=torch.float32).expand(n, 3)
        # 位姿差分算滑移要上一控制步的两个刚体位姿（D-49，与 S4 同一个判据）。
        self.prev_obj_pos = torch.zeros(n, 3, device=dev)
        self.prev_obj_quat = torch.zeros(n, 4, device=dev)
        self.prev_obj_quat[:, 0] = 1.0
        self.prev_exec_pos = torch.zeros(n, 3, device=dev)
        self.prev_exec_quat = torch.zeros(n, 4, device=dev)
        self.prev_exec_quat[:, 0] = 1.0
        self.prev_effect_pose = torch.zeros(n, 7, device=dev)
        self._sub = 0
        self._best = self._empty_contact()
        self._best_force = torch.zeros(n, device=dev)
        self._live = torch.zeros(n, dtype=torch.bool, device=dev)
        self._approach_handoff = torch.zeros(n, dtype=torch.bool, device=dev)
        # PhysX 报的接触束作用在谁身上（±1）。**latch** 住，不每帧重判（P-49）。
        self._normal_sign = torch.ones(n, device=dev)
        self.shaping = torch.zeros(n, device=dev)
        self._terms = None
        self._failed = torch.zeros(n, dtype=torch.bool, device=dev)
        self._diag: dict[str, torch.Tensor] = {}

    def _empty_contact(self) -> dict[str, torch.Tensor]:
        """定长的空接触。**存的是原始量**——作用在物体上的力要按 S4 的口径构造，
        那一步放在 `_advance` 里做一次，见 `ei_reward.contact_force_on_object`。"""
        n, k, dev = self.num_envs, MAX_CONTACTS, self.device
        return {"pos": torch.zeros(n, k, 3, device=dev),
                "normal": torch.zeros(n, k, 3, device=dev),
                "friction": torch.zeros(n, k, 3, device=dev),
                "normal_force": torch.zeros(n, k, device=dev),
                "valid": torch.zeros(n, k, dtype=torch.bool, device=dev)}

    # ------------------------------------------------------------ 动作

    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions = self.actions
        self.actions = actions.clone().clamp(-1.0, 1.0)
        self.tgt_pos, self.tgt_quat = self.act.step(self.actions[:, :6])
        self.ff_force = self.actions[:, 6:9] * self.cfg.force_scale
        if self.cfg.use_precontact_servo:
            ready_pos, ready_quat = self._contact_ready_pose()
            d = self.executor.data
            tip = d.root_pos_w + quat_apply(d.root_quat_w, self.tip_local)
            ready_tip = ready_pos + quat_apply(ready_quat, self.tip_local)
            reached = (tip - ready_tip).norm(dim=-1) <= self.cfg.precontact_handoff_tolerance
            self._approach_handoff |= reached | self._live
            servo = ~self._approach_handoff
            # 直接改的是 PD **目标**，不是刚体位姿；动力学与接触求解仍完整保留。
            self.act.target_pos = torch.where(servo[:, None], ready_pos,
                                              self.act.target_pos)
            self.act.target_quat = torch.where(servo[:, None], ready_quat,
                                               self.act.target_quat)
            self.tgt_pos, self.tgt_quat = self.act.target_pos, self.act.target_quat
            # 伺服阶段不接受 policy 力动作；交接后仍由 `_apply_action` 的真实 contact
            # gate 保证自由空间不做力控。
            self.ff_force = torch.where(servo[:, None], torch.zeros_like(self.ff_force),
                                        self.ff_force)
        self._sub = 0

    def _contact_ready_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """只由当前 interaction 格导出的 contact-ready 根位姿（世界系）。

        当前格有 region 质量时取其力质量质心/平均法向；接近或释放格为空时退回整条
        command 的 anchor。没有 task phase、目标关节或 source tool pose。
        """
        cmd, b = self.tracker.command_index, self.tracker.bin_index
        points = self.bank.gather("surface/points_obj", cmd)
        normals = self.bank.gather("surface/normals_obj", cmd)
        mass = self.bank.gather_bin("region/mass/mean", cmd, b)
        total = mass.sum(-1, keepdim=True)
        weight = mass / total.clamp_min(1e-9)
        site = torch.einsum("ns,nsd->nd", weight, points)
        normal = torch.einsum("ns,nsd->nd", weight, normals)
        site = torch.where(total > 1e-9, site, self.bank.anchor_pos[cmd])
        normal = torch.where(total > 1e-9, normal, self.bank.anchor_normal[cmd])
        normal = torch.nn.functional.normalize(normal, dim=-1)

        obj_pos, obj_quat = self._object_frame()
        world_normal = quat_apply(obj_quat, normal)
        world_site = (obj_pos + quat_apply(obj_quat, site)
                      + self.scene.env_origins)
        # 只送到表面**外**的 contact-ready pose，不替 policy 建立接触或施压。
        tip_goal = world_site + world_normal * self.cfg.precontact_clearance
        quat = self._quat_from_z(world_normal)
        root = tip_goal - quat_apply(quat, self.tip_local)
        return root, quat

    def _apply_action(self):
        # 逐子步采接触（P-31）。传感器里躺着的是**上一次 sim.step 之后**的结果，
        # 所以第一个子步（_sub == 0）读到的属于上一个控制步——那一次只用来做力控
        # 门控（差一个子步无所谓），不进 best，best 在这里清空。
        if self._sub == 0:
            self._best = self._empty_contact()
            self._best_force = torch.zeros(self.num_envs, device=self.device)
        else:
            self._accumulate_contact()
        self._sub += 1

        # **力控只在有接触时打开**（模块 docstring 第二条）。自由空间里给
        # FloatingPD 一个 force_dir 会把整个位置 PD 沿该方向投影掉，25 N 前馈
        # 在 0.3 kg 的杆上是 83 m/s²，随机策略必然把自己甩飞。
        magnitude = self.ff_force.norm(dim=-1, keepdim=True)
        use_force = (magnitude > 1e-3) & self._live[:, None]
        direction = torch.where(use_force, self.ff_force / magnitude.clamp_min(1e-9),
                                torch.zeros_like(self.ff_force))
        zero = torch.zeros_like(self.ff_force)
        f, tq = self.pd.compute(self.tgt_pos, self.tgt_quat,
                                ff_force=torch.where(use_force, self.ff_force, zero),
                                force_dir=torch.where(use_force, direction, zero))
        self.executor.set_external_force_and_torque(f, tq)

    # ------------------------------------------------------------ 交互量

    def _object_frame(self):
        d = self.obj.data
        pos = d.root_pos_w - self.scene.env_origins
        return pos, d.root_quat_w

    def _sample_contact(self) -> dict[str, torch.Tensor]:
        """当前子步的接触，全部换算到**物体系**——`plan/02` §1 的坐标约定。"""
        raw = extract_contact_points_padded(
            self.contact, self.physics_dt,
            self.executor.data.root_pos_w, max_points=MAX_CONTACTS,
            own_radius=self.cfg.contact_own_radius)
        obj_pos, obj_quat = self._object_frame()
        inv = quat_conjugate(obj_quat)
        world = raw["positions"] - self.scene.env_origins[:, None, :]
        k = world.shape[1]
        to_obj = lambda v: quat_apply(inv[:, None, :].expand(-1, k, -1), v)  # noqa: E731
        return {"pos": to_obj(world - obj_pos[:, None, :]),
                "normal": to_obj(raw["normals"]),
                "friction": to_obj(raw["friction_forces"]),
                "valid": raw["valid"], "normal_force": raw["normal_forces"]}

    def _accumulate_contact(self):
        """取法向力最大的那一子步作代表（P-31，与采集侧 `s3_source_probe` 同一条）。"""
        cp = self._sample_contact()
        total = cp["normal_force"].abs().sum(dim=1)
        self._live = total > CONTACT_FORCE_MIN
        take = total > self._best_force
        for key, value in cp.items():
            mask = take.view(-1, *([1] * (value.dim() - 1)))
            self._best[key] = torch.where(mask, value, self._best[key])
        self._best_force = torch.maximum(self._best_force, total)

    def _force_on_object(self, c: dict, cells: torch.Tensor,
                         cell_normals: torch.Tensor):
        """把这一步的原始接触量拼成**作用在物体上**的力，口径与 S4 逐字相同。

        三步，顺序不能换（`it.interaction` 的第 3、4 步就是这个顺序）：

        1. **纯最近邻**归一次格，只为拿到每个接触点所在处的**表面外法向**——
           几何量，没有正负约定问题（P-37）；
        2. 用它判 PhysX 报的这束力作用在谁身上，并**latch** 住这个符号。
           每帧重判会在两个等价解之间横跳（P-49）；掠射帧的 ``sign`` 是 0，
           这时沿用上一次看得出来的值；
        3. 用**定向后**的法向再做一次同侧归格（D-68 的在线版）。第一版拿
           PhysX 的原始法向做同侧判定，方向翻转时那个判据整体失效而不报错。
        """
        outward = cell_normals.gather(
            1, nearest_cell(c["pos"], cells)[..., None].expand(-1, -1, 3))
        sign = normal_orientation_sign(c["normal"], outward,
                                       c["normal_force"], c["valid"])
        self._normal_sign = torch.where(sign != 0, sign, self._normal_sign)
        oriented = self._normal_sign[:, None, None] * c["normal"]
        cell_index = assign_cells(c["pos"], oriented, c["valid"], cells, cell_normals)
        outward = cell_normals.gather(1, cell_index[..., None].expand(-1, -1, 3))
        force = contact_force_on_object(c["normal_force"], c["friction"], outward,
                                        c["valid"], self._normal_sign)
        return force, cell_index, outward

    def _slip_speed(self, pos_obj: torch.Tensor, normal_obj: torch.Tensor,
                    valid: torch.Tensor) -> torch.Tensor:
        """接触斑块的切向滑移速率，由**两刚体的位姿差分**给出（D-49 / P-52）。

        把同一个接触点分别当作固连在物体上和固连在执行器上，看这一步之后两者
        分开了多远；差的切向分量除以 dt 就是滑移速率。**不用 PhysX 报的瞬时相对
        速度**——角速度饱和时它整体虚高，实测积出 59 mm 滑移而接触点在物体系里
        一动没动。

        粒度是**控制步**（不跟着子步走）：S4 的 `mode/pose_slip` 就是逐帧位姿差分，
        在线换一个粒度，reward 追的就不是 artifact 里写的那个量。
        """
        obj_pos, obj_quat = self._object_frame()
        exec_pos = self.executor.data.root_pos_w - self.scene.env_origins
        exec_quat = self.executor.data.root_quat_w
        k = pos_obj.shape[1]
        expand = lambda q: q[:, None, :].expand(-1, k, -1)  # noqa: E731
        world_now = obj_pos[:, None, :] + quat_apply(expand(obj_quat), pos_obj)
        # 同一世界点在上一步的执行器系里的位置
        local_exec = quat_apply(expand(quat_conjugate(exec_quat)),
                                world_now - exec_pos[:, None, :])
        moved_by_obj = (self.prev_obj_pos[:, None, :]
                        + quat_apply(expand(self.prev_obj_quat), pos_obj))
        moved_by_exec = (self.prev_exec_pos[:, None, :]
                         + quat_apply(expand(self.prev_exec_quat), local_exec))
        delta = moved_by_exec - moved_by_obj
        normal_world = quat_apply(expand(obj_quat), normal_obj)
        tangential = delta - (delta * normal_world).sum(-1, keepdim=True) * normal_world
        dt = self.physics_dt * self.cfg.decimation
        return (tangential.norm(dim=-1) / dt) * valid.float()

    # ------------------------------------------------------------ 每步只算一次

    def _advance(self):
        """取接触 → 归格 → 推窗口 → 算五项 reward。**一个控制步只算一次。**

        Isaac Lab 的 `DirectRLEnv.step` 先调 `_get_dones` 再调 `_get_rewards`
        （`direct_rl_env.py` 里 `reset_terminated[:] = self._get_dones()` 在
        `reward_buf = self._get_rewards()` 上面）。所以推窗口必须在 `_get_dones`
        里就完成，否则"指令跑完"要晚一整步才被看见。
        """
        self._accumulate_contact()          # 最后一个子步
        c = self._best
        cmd, b = self.tracker.command_index, self.tracker.bin_index
        cells = self.bank.gather("surface/points_obj", cmd)
        cell_normals = self.bank.gather("surface/normals_obj", cmd)
        force, _physical_cell, outward = self._force_on_object(c, cells, cell_normals)
        allowed = self.bank.gather_bin("region/allowed", cmd, b)
        # target embodiment 的接触拓扑不需要复刻 source。连续接触在固定物理分辨率内
        # 匹配到最近的功能区域格；这是 executor-side decode，不改 interaction artifact。
        cell_index, contact_compatibility = match_functional_region(
            c["pos"], outward, c["valid"], cells, cell_normals,
            self.bank.gather("surface/area", cmd), allowed,
            sensor_sigma=self.cfg.region_match_sigma)
        traction, mass = surface_traction(c["pos"], force, c["valid"],
                                          cell_index, self.bank.n_surface)
        region_membership = scatter_contact_compatibility(
            force, c["valid"], cell_index, contact_compatibility,
            self.bank.n_surface)
        slip = self._slip_speed(c["pos"], outward, c["valid"])
        slip_cell = torch.zeros(self.num_envs, self.bank.n_surface, device=self.device)
        slip_cell.scatter_reduce_(1, cell_index, slip, reduce="amax", include_self=True)

        # effect：本控制步物体实际发生了多少变化（无量纲，两路各除各自刻度）。
        # 它先喂给窗口跟踪器累计，reward 用的是**本格的完成缺口**而不是逐步差值——
        # 命令格的时长是弹性的，指令 effect 是"这一格里总共要发生多少"，
        # 不是一个速率。逐步差值那一版在 dry-run 上炸到 −26647（见 ei_reward.effect_deficit）。
        got_rigid, got_state = self._effect_rate()
        metric = self.bank.gather("effect/rigid/metric", cmd)
        scale_r = self.bank.gather("effect/rigid/scale_m", cmd)
        scale_s = self.bank.gather("effect/surface_state/scale", cmd)
        got = effect_magnitude(got_rigid, got_state, metric=metric,
                               scale_rigid=scale_r, scale_state=scale_s)

        over = (c["normal_force"].abs()
                - self.cfg.max_contact_force).clamp_min(0.0).sum(-1)
        smooth = (self.actions - self.prev_actions).pow(2).sum(-1)
        safety = -(self.cfg.w_force * over + self.cfg.w_action * smooth)

        # 先推窗口：缺口要用**累计到本步为止**的完成量算，所以顺序不能反。
        region_force = (mass * region_membership).sum(-1)
        progress = self.tracker.step(effect_increment=got,
                                     region_normal_force=region_force)
        self.progress_obs = torch.stack(list(progress.values()), dim=-1)

        self._terms = interaction_reward(
            effect_deficit=self.tracker.last_deficit,
            traction=traction, mass=mass, slip_speed=slip_cell,
            allowed=region_membership,
            traction_lo=self.bank.gather_bin("mech/traction_obj/lo", cmd, b),
            traction_hi=self.bank.gather_bin("mech/traction_obj/hi", cmd, b),
            slip_lo=self.bank.gather_bin("mode/slip_speed/lo", cmd, b),
            slip_hi=self.bank.gather_bin("mode/slip_speed/hi", cmd, b),
            force_penalty=safety)

        self.contact_obs = torch.stack([
            c["valid"].float().sum(-1) / MAX_CONTACTS,
            c["normal_force"].abs().sum(-1) / self.cfg.max_contact_force,
            region_force / self.cfg.max_contact_force,
            mass.sum(-1) / self.cfg.max_contact_force,
            slip_cell.amax(-1), traction.norm(dim=-1).amax(-1) * 1e-5], dim=-1)

        # 接近段的稠密信号（`w_approach`）。见 `_approach_cost` 的说明。
        self.shaping = self._approach_cost()

        touching = mass.sum(-1) > 1e-9
        contact_required = self.bank.gather_bin(
            "region/mass/mean", cmd, b).sum(-1) > 0
        inside = region_force / mass.sum(-1).clamp_min(1e-9)
        self._diag = {
            "diag/touching": touching.float(),
            # 这是推进前当前命令格的要求，与上面的 contact 数据属于同一步。
            # smoke 必须用它排除 interaction 明确指定的 approach/release 空接触格；
            # 但 reward 比较仍看全时段无条件均值，不能借条件化隐藏坏奖励。
            "diag/contact_required": contact_required.float(),
            "diag/inside_region_share": torch.where(touching, inside,
                                                    torch.zeros_like(inside)),
            "diag/contact_points": c["valid"].float().sum(-1),
            "diag/peak_normal_force": (c["normal_force"].abs()
                                       * c["valid"].float()).amax(-1),
            "diag/object_off_table": self._off_table().float(),
            "diag/slip_in_region": (slip_cell * region_membership).amax(-1),
            "diag/slip_max": slip_cell.amax(-1),
            # 越界量被截断的比例。长期居高不下 = 那一项已经饱和、没有梯度了。
            "diag/mech_saturated": self._terms.mech_saturated,
            "diag/mode_saturated": self._terms.mode_saturated,
            # 分得开"垫子真的在物体表面上滑"和"杆在绕自己转"：末端离根 108 mm，
            # 2 rad/s 的自转就是 0.22 m/s 的接触点线速度（P-52 的同一族问题）。
            "diag/exec_ang_speed": self.executor.data.root_ang_vel_w.norm(dim=-1),
            "diag/exec_lin_speed": self.executor.data.root_lin_vel_w.norm(dim=-1),
            # 垫面与接触面的贴合角（度）。平贴时接触流形是四个角点，歪着就只剩一个角——
            # `inside_region_share` 会因此从"四选四"退化成"四选一"。
            "diag/pad_tilt_deg": self._pad_tilt(c),
            "diag/anchor_distance": self._anchor_distance(),
            "diag/shaping": self.shaping,
        }
        self._remember_poses()

    def _approach_cost(self) -> torch.Tensor:
        """接近段的稠密信号：`−w·min(d, d_ref)/d_ref` ∈ [−w, 0]，贴到锚点时恰好 0。

        **为什么不是势函数式的**（这一条推翻了 D-86/D-89 的做法，见 D-90）。
        势函数式塑形 `F = γΦ' − Φ` 原地不动时等于 `(γ−1)Φ`，而这个漂移的量级是
        `(1−γ)·|Φ|·T`：episode 有 300 步而 `1/(1−γ)` 只有 100，**漂移是 telescope
        出来的那点收益的三倍**。于是不论 Φ 取什么符号，"停在 Φ=0 的地方"都是最省的，
        实测两个符号都试过：

        - `Φ = −5d`（远处最负）：24 轮内策略跑到 0.465 m 外坐着不动（P-82）；
        - `Φ = w(1 − d/d_ref)`（远处为 0）：80 轮后停在 d≈0.17~0.27，
          正好是 `Φ≈0` 的那一圈，`diag/touching` 仍然近乎为 0。

        Ng 等 1999 的策略不变性在这里是**反作用**的：它保证塑形改变不了最优解，
        而我们恰恰需要"离物体近"这件事在**有限时域**里真的更划算。

        **为什么这一项不会被 D-31 第 1 个洞套利**：它是一个**有上界 0 的代价**，
        不是一个可以反复领取的奖励。贴着锚点不动只是**不再付钱**，领不到任何东西；
        而真正压对地方还有 `+w_region` 可拿。梯子因此是
        「远处 −w」<「贴着不动 0」<「压对地方 +2」，每一级都严格更好。
        """
        d = self._anchor_distance() / max(self.cfg.approach_ref, 1e-6)
        return -self.cfg.w_approach * d.clamp(max=1.0)

    def _anchor_distance(self) -> torch.Tensor:
        """执行器末端到**指令接触锚点**的距离（m）。只读指令，不读任务。"""
        obj_pos, obj_quat = self._object_frame()
        anchor = obj_pos + quat_apply(obj_quat,
                                      self.bank.anchor_pos[self.tracker.command_index])
        d = self.executor.data
        tip = (d.root_pos_w - self.scene.env_origins
               + quat_apply(d.root_quat_w, self.tip_local))
        return (tip - anchor).norm(dim=-1)

    def _pad_tilt(self, c: dict) -> torch.Tensor:
        """执行器末端轴（本体 −Z）与接触法向的夹角（度）。力加权，无接触时为 0。"""
        axis = quat_apply(self.executor.data.root_quat_w,
                          torch.tensor([0.0, 0.0, -1.0], device=self.device
                                       ).expand(self.num_envs, 3))
        _, obj_quat = self._object_frame()
        k = c["normal"].shape[1]
        normal_w = quat_apply(obj_quat[:, None, :].expand(-1, k, -1), c["normal"])
        cos = (normal_w * axis[:, None, :]).sum(-1).abs().clamp(0.0, 1.0)
        w = c["normal_force"].abs() * c["valid"].float()
        total = w.sum(-1)
        angle = torch.rad2deg(torch.acos(cos))
        return torch.where(total > _TINY, (w * angle).sum(-1) / total.clamp_min(_TINY),
                           torch.zeros_like(total))

    def _off_table(self) -> torch.Tensor:
        """物体跑出台面（含掉下去）。缩小台面若真造成了影响，这一条会非零。"""
        pos, _ = self._object_frame()
        half = self.cfg.table_extent / 2
        return (pos[:, :2].abs().amax(-1) > half) | (pos[:, 2] < -0.05)

    # ------------------------------------------------------------ 终止与 reward

    def _get_dones(self):
        self._advance()
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        pos = self.executor.data.root_pos_w - self.scene.env_origins
        # `far` 只该是**数值发散**的兜底，不能是策略正常动作就够得到的东西。
        # 目标被逐轴限在 spawn 附近 `pos_limit` 内，所以可达集合的最远点是
        # √3·pos_limit（立方体的角），门槛必须比它大——否则"把目标走到角上"
        # 就是一条必然触发失败终止的路径（第一版就是这样，见 `_init_buffers`）。
        far_limit = 3.0 ** 0.5 * self.act.pos_limit + 0.15
        far = (pos - self.spawn_pos).norm(dim=-1) > far_limit
        broken = ~torch.isfinite(pos).all(dim=-1)
        self._failed = far | broken | self._off_table()
        return self.tracker.finished | self._failed, timeout

    def _get_rewards(self) -> torch.Tensor:
        terms = self._terms
        total = terms.total(self.weights)
        # **失败终止必须有显式惩罚**（D-31 第 3 个洞）。五项跟踪 reward 全部 ≤ 0，
        # 早退的回报是 0——与完美执行并列最优。按**剩余步数**给，早退因此不会比
        # 留下来干活便宜；而正常跑完（tracker.finished）不罚。
        remaining = (self.max_episode_length - 1 - self.episode_length_buf).clamp_min(0)
        g = self.cfg.fail_penalty_gamma
        horizon = (1.0 - g ** remaining.float()) / max(1.0 - g, 1e-9)
        total = total - self._failed.float() * horizon * self.cfg.fail_penalty_per_step
        # 接近段的代价。终止步上不再计（失败已经单独罚过，成功不该再挨这一刀）。
        total = total + self.shaping * (~self._failed & ~self.tracker.finished).float()

        # **每一项都进 extras["log"]**：没有分项记录，训练不收敛时只能猜（P-27）。
        log = {k: v.mean() for k, v in terms.as_log().items()}
        log.update({k: v.mean() for k, v in self._diag.items()})
        log["diag/failed"] = self._failed.float().mean()
        log["diag/finished"] = self.tracker.finished.float().mean()
        self.extras.setdefault("log", {}).update(log)
        return total

    def _effect_rate(self):
        """本控制步物体实际发生的变化，与指令 effect 同一个表示（`plan/02` §3.1）。"""
        pos, quat = self._object_frame()
        prev_pos, prev_quat = self.prev_effect_pose[:, :3], self.prev_effect_pose[:, 3:7]
        inv = quat_conjugate(prev_quat)
        dp = quat_apply(inv, pos - prev_pos)
        dq = quat_mul(inv, quat)
        dq = torch.where(dq[:, 0:1] < 0, -dq, dq)
        angle = 2.0 * torch.atan2(dq[:, 1:].norm(dim=-1, keepdim=True),
                                  dq[:, 0:1].clamp(-1.0, 1.0))
        axis = dq[:, 1:] / dq[:, 1:].norm(dim=-1, keepdim=True).clamp_min(1e-9)
        rigid = torch.cat([dp, axis * angle], dim=-1)
        level = self.bank.gather("effect/surface_state/median",
                                 self.tracker.command_index).shape[-1]
        # 本轮仿真里没有"表面状态"这种可变量的物体（dirt 只在擦拭上，而擦拭是
        # 留出任务）。恒零是**事实**，不是占位——留出任务上它才会非零。
        state = torch.zeros(self.num_envs, level, device=self.device)
        return rigid, state

    def _remember_poses(self, env_ids=None):
        """记下本步的两个刚体位姿，供下一步做位姿差分。

        ``env_ids`` 必须能限定范围：`_reset_idx` 只重置一部分 env，若在那里无差别地
        刷新全部 env 的"上一步位姿"，**没被重置的那些 env 会有一帧滑移速率是错的**
        ——而且不报错。这与 P-30 是同一类"下标范围没对齐"的错。
        """
        pos, quat = self._object_frame()
        exec_pos = self.executor.data.root_pos_w - self.scene.env_origins
        exec_quat = self.executor.data.root_quat_w
        if env_ids is None:
            env_ids = slice(None)
        self.prev_obj_pos[env_ids] = pos[env_ids].clone()
        self.prev_obj_quat[env_ids] = quat[env_ids].clone()
        self.prev_exec_pos[env_ids] = exec_pos[env_ids].clone()
        self.prev_exec_quat[env_ids] = exec_quat[env_ids].clone()
        self.prev_effect_pose[env_ids] = torch.cat([pos, quat], dim=-1)[env_ids].clone()

    # ------------------------------------------------------------ 观测

    def _proprio(self) -> torch.Tensor:
        d = self.executor.data
        pos, quat = self._object_frame()
        rel = quat_apply(quat_conjugate(quat), d.root_pos_w - self.scene.env_origins - pos)
        return torch.cat([
            d.root_pos_w - self.scene.env_origins, d.root_quat_w,      # 7
            d.root_lin_vel_w, d.root_ang_vel_w,                        # 6
            pos, quat,                                                 # 7
            rel,                                                       # 3
            self.obj.data.root_lin_vel_w, self.obj.data.root_ang_vel_w,  # 6
            self.actions,                                              # 9
            self.contact_obs,                                          # 6
            self.progress_obs,                                         # 6
        ], dim=-1)

    def _get_observations(self) -> dict:
        obs = torch.cat([
            self._proprio(),
            self.tracker.command_index.float()[:, None],
            self.tracker.bin_index.float()[:, None],
            self._object_frame()[1],
        ], dim=-1)
        return {"policy": obs, "critic": torch.cat([obs, self._privileged()], dim=-1)}

    def _privileged(self) -> torch.Tensor:
        """critic 独享：仿真里才有的量。只影响价值估计，不进策略。"""
        return torch.cat([
            self.obj.data.default_mass.to(self.device).sum(-1, keepdim=True),
            self.ff_force,
            self.tracker.dwell.float()[:, None] / max(self.tracker.max_dwell, 1),
        ], dim=-1)

    # ------------------------------------------------------------ 重置

    def _reset_idx(self, env_ids):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.obj._ALL_INDICES
        super()._reset_idx(env_ids)
        if not hasattr(self, "act"):
            self._init_buffers()
        n, dev = len(env_ids), self.device
        command = torch.randint(0, len(self.bank), (n,), device=dev)
        self.tracker.reset(env_ids, command)

        st = self.obj.data.default_root_state[env_ids].clone()
        st[:, :3] += self.scene.env_origins[env_ids]
        self.obj.write_root_state_to_sim(st, env_ids=env_ids)

        ex = self.executor.data.default_root_state[env_ids].clone()
        ex[:, :7] = self._spawn_pose(command, st[:, :3], st[:, 3:7])
        ex[:, 7:] = 0.0
        self.executor.write_root_state_to_sim(ex, env_ids=env_ids)
        self.spawn_pos[env_ids] = ex[:, :3] - self.scene.env_origins[env_ids]
        self.act.reset(ex[:, :3], ex[:, 3:7], env_ids=env_ids)
        self.tgt_pos[env_ids], self.tgt_quat[env_ids] = ex[:, :3], ex[:, 3:7]
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.contact_obs[env_ids] = 0.0
        self.progress_obs[env_ids] = 0.0
        self._live[env_ids] = False
        self._approach_handoff[env_ids] = False
        self.shaping[env_ids] = 0.0
        self._best_force[env_ids] = 0.0
        self._normal_sign[env_ids] = 1.0
        self._remember_poses(env_ids)

    def _spawn_pose(self, command: torch.Tensor, obj_pos_w: torch.Tensor,
                    obj_quat: torch.Tensor) -> torch.Tensor:
        """执行器的初始位姿：沿指令接触锚点的外法向退开一个随机站距。

        **只读指令**（`CommandBank.anchor_pos/anchor_normal`），不读任务、不读物体
        名字，所以 S7 零样本评估照用同一条规则。理由见
        `ei_command.CommandBank._contact_anchor`：接近段没有任何稠密梯度，
        把执行器丢在 0.25 m 外等于要求随机游走先撞上物体。
        """
        n, dev = command.shape[0], self.device
        anchor = self.bank.anchor_pos[command]                        # 物体系
        normal = self.bank.anchor_normal[command]
        anchor_w = obj_pos_w + quat_apply(obj_quat, anchor)
        normal_w = quat_apply(obj_quat, normal)
        lo, hi = self.cfg.init_standoff
        standoff = torch.rand(n, 1, device=dev) * (hi - lo) + lo
        jitter = (torch.rand(n, 3, device=dev) - 0.5) * 2.0 * self.cfg.init_jitter_m
        tip_w = anchor_w + normal_w * standoff + jitter

        # 让执行器的末端指向物体：局部 +Z 对齐到外法向。
        quat = self._quat_from_z(normal_w)
        quat = quat_mul(self._small_random_rotation(n), quat)
        tip_local = torch.tensor(PS.EXECUTOR_TIP[self.cfg.executor_name],
                                 device=dev, dtype=torch.float32).expand(n, 3)
        root_w = tip_w - quat_apply(quat, tip_local)
        return torch.cat([root_w, quat], dim=-1)

    @staticmethod
    def _quat_from_z(direction: torch.Tensor) -> torch.Tensor:
        """把局部 +Z 转到 ``direction`` 的最短弧四元数 (N,4)，(w,x,y,z)。

        ``direction`` 恰好是 −Z 时最短弧没有唯一解，必须显式挑一个分支——
        P-49 就是这个洞：闭环里每步重算的目标含离散选择时，四元数会在两个等价解
        之间随机横跳。这里的选择在 reset 里算一次，**不在回路内**。
        """
        n = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        w = 1.0 + n[:, 2:3]
        axis = torch.stack([-n[:, 1], n[:, 0], torch.zeros_like(n[:, 0])], dim=-1)
        quat = torch.cat([w, axis], dim=-1)
        flipped = torch.zeros_like(quat)
        flipped[:, 1] = 1.0                       # 绕 +X 转 180°
        quat = torch.where(w > 1e-6, quat, flipped)
        return quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-9)

    def _small_random_rotation(self, n: int) -> torch.Tensor:
        """小幅随机姿态抖动。2048 个 env 完全相同的话探索没有多样性。"""
        dev = self.device
        axis = torch.nn.functional.normalize(torch.randn(n, 3, device=dev), dim=-1)
        angle = (torch.rand(n, 1, device=dev) - 0.5) * 2.0 * self.cfg.init_jitter_rad
        return torch.cat([torch.cos(angle / 2), axis * torch.sin(angle / 2)], dim=-1)
