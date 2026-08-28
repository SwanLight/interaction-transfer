"""抽屉任务的 Privileged Expert 环境（S2）。

`plan/README` §5：抽屉是**阴性对照**任务——预期「只给物体结果」就足够，
不需要额外的交互信息（Gate G）。它在 S2 的作用是打通训练管线，
不是产生科学结论。

`plan/04` §3：Expert 观测给上帝视角（执行器完整状态、物体完整状态、
当前接触集合与接触力、物理参数、任务目标），reward 只服务于「稳定得到
大量成功轨迹」，不把 reward engineering 当创新点。

Gate A（`plan/05` §10）：固定环境成功率 ≥95%，随机环境 ≥85%。
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg, RigidObject
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass

from it import assets as A
from it.build_assets import CabinetCfg
from it.envs.base import FloatingBaseAction, contact_summary
from it.float_ctrl import FloatingPD

_C = CabinetCfg()


@configclass
class DrawerEnvCfg(DirectRLEnvCfg):
    # 50 Hz 控制（plan/04 §9）。dt=1/150 × decimation 3 = 50 Hz，
    # 物理步长比 1/120 更细，contact-rich 场景数值更稳。
    decimation = 3
    episode_length_s = 8.0

    action_space = 6           # 浮动底座 6 维位姿增量
    observation_space = 39
    state_space = 0

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1 / 150,
        render_interval=3,
        physx=sim_utils.PhysxCfg(
            # contact-rich 场景默认缓存会静默溢出（P-03）
            gpu_max_rigid_contact_count=2 ** 22,
            gpu_max_rigid_patch_count=2 ** 20,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=2.5,
                                                     replicate_physics=True)

    cabinet: object = A.CABINET_CFG.replace(prim_path="/World/envs/env_.*/Cabinet")
    executor: object = A.HOOK_CFG.replace(prim_path="/World/envs/env_.*/Hook")
    contact: object = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Hook",
        track_contact_points=True,
        max_contact_data_count_per_prim=16,        # 规则 8
        filter_prim_paths_expr=["/World/envs/env_.*/Cabinet/Drawer"],   # 规则 8 + P-17
        update_period=0.0, history_length=0,
    )
    light: object = AssetBaseCfg(prim_path="/World/light",
                                 spawn=sim_utils.DomeLightCfg(intensity=800.0))

    # --- 任务参数 ---
    goal_range: tuple[float, float] = (0.100, 0.160)    # plan/01 §4 目标开度
    goal_tol: float = 0.010
    hold_steps: int = 10                                # 达标需保持的控制步数
    # 脚本化验证专用：关掉提前终止，让一条完整轨迹跑完再统计。
    # 否则 env 会在中途自动重置，reward 标定会跨多个 episode，数字没意义。
    disable_termination: bool = False

    # --- 随机化（Curriculum B，`plan/04` §10）---
    randomize: bool = False
    exec_pos_noise: float = 0.03
    damping_range: tuple[float, float] = (2.0, 4.5)

    # --- reward 权重（`plan/04` §8.1）---
    # 权重由 tools/s2_scripted.py 在**一条成功的脚本轨迹**上标定，不是拍的。
    # 标定前（w_progress=12 / w_success=8 / w_action=0.008）的加权贡献：
    #     success +122 / progress +2.2 / reach +1.0 / action -32
    # 成功奖励占 97%，稠密引导几乎没有梯度，而 action 惩罚是第二大项——
    # 第一次训练 40 轮 reward 从 -44.8 只爬到 -19.7、episode 长度 399/400
    # （从没成功过），最优解就是「不动」。
    # 现按标定值重配，成功轨迹总分约 +79，其中稠密项占 ~30%：
    #     progress +18 / reach +5 / success +60 / action -4
    w_progress: float = 150.0     # × 初始差距 0.13 m ≈ +20
    w_success: float = 4.0        # × 保持 15 步 ≈ +60
    w_reach: float = 22.0         # × 初始距离 0.23 m ≈ +5
    w_force: float = 0.05
    w_action: float = 0.001       # × ~10/步 × 400 步 ≈ -4
    w_fail: float = 30.0          # 飞出边界的一次性惩罚，必须大于放弃任务的收益
    max_contact_force: float = A.MAX_NORMAL_FORCE


class DrawerEnv(DirectRLEnv):
    cfg: DrawerEnvCfg

    def _setup_scene(self):
        self.cabinet = Articulation(self.cfg.cabinet)
        self.executor = RigidObject(self.cfg.executor)
        self.contact = ContactSensor(self.cfg.contact)
        self.scene.articulations["cabinet"] = self.cabinet
        self.scene.rigid_objects["executor"] = self.executor
        self.scene.sensors["contact"] = self.contact
        self.cfg.light.spawn.func("/World/light", self.cfg.light.spawn)
        self.scene.clone_environments(copy_from_source=False)

    def __post_init__(self):
        pass

    def _configure_gym_env_spaces(self):
        super()._configure_gym_env_spaces()
        self._init_buffers()

    def _init_buffers(self):
        n, dev = self.num_envs, self.device
        self.act = FloatingBaseAction(n, dev, pos_scale=0.012, rot_scale=0.05, pos_limit=0.7)
        self.pd = FloatingPD(self.executor, kp_pos=600.0, kd_pos=50.0,
                             kp_rot=60.0, kd_rot=8.0,
                             max_force=180.0, max_torque=18.0, kd_force=40.0)
        self.goal = torch.zeros(n, device=dev)
        self.hold = torch.zeros(n, device=dev, dtype=torch.long)
        self.prev_open = torch.zeros(n, device=dev)
        self.prev_act = torch.zeros(n, 6, device=dev)
        self.prev_dist = torch.zeros(n, device=dev)
        self.prev_gap = torch.zeros(n, device=dev)
        self._far_buf = torch.zeros(n, dtype=torch.bool, device=dev)
        self.term_opening = torch.zeros(n, device=dev)
        self.term_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.actions = torch.zeros(n, 6, device=dev)
        # reset 会先调 _get_observations，那时 _pre_physics_step 还没跑过，
        # 目标位姿必须在这里就有值，否则 AttributeError。
        self.tgt_pos = torch.zeros(n, 3, device=dev)
        self.tgt_quat = torch.zeros(n, 4, device=dev)
        self.tgt_quat[:, 0] = 1.0
        self.success_buf = torch.zeros(n, dtype=torch.bool, device=dev)
        self._dj = self.cabinet.find_joints("DrawerJoint")[0]

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        self.tgt_pos, self.tgt_quat = self.act.step(self.actions)

    def _apply_action(self):
        f, tq = self.pd.compute(self.tgt_pos, self.tgt_quat)
        self.executor.set_external_force_and_torque(f, tq)

    # ------------------------------------------------------------ 观测

    def _get_observations(self) -> dict:
        d = self.executor.data
        cab = self.cabinet.data
        opening = cab.joint_pos[:, self._dj[0]]
        handle = self._handle_pos_w()
        cs = contact_summary(self.contact, self.physics_dt, self.num_envs, self.device)

        obs = torch.cat([
            d.root_pos_w - self.scene.env_origins,      # 3  执行器位置
            d.root_quat_w,                              # 4  姿态
            d.root_lin_vel_w,                           # 3
            d.root_ang_vel_w,                           # 3
            self.tgt_pos - self.scene.env_origins,      # 3  当前目标位姿
            self.tgt_quat,                              # 4
            handle - self.scene.env_origins,            # 3  把手位置（物体状态）
            opening.unsqueeze(-1),                      # 1  开度
            cab.joint_vel[:, self._dj].reshape(-1, 1),  # 1
            self.goal.unsqueeze(-1),                    # 1  任务目标
            (self.goal - opening).unsqueeze(-1),        # 1  剩余
            (handle - d.root_pos_w),                    # 3  相对向量
            cs,                                         # 8  接触摘要（特权）
            self.joint_damping.unsqueeze(-1),           # 1  物理参数（特权）
        ], dim=-1)
        return {"policy": obs}

    def _handle_pos_w(self):
        """把手杆中心的世界位置。抽屉沿 +X 拉出，把手随之平移。"""
        base = self.cabinet.data.body_pos_w[:, self._drawer_body, :]
        off = torch.tensor([_C.panel_t + _C.handle_clearance + _C.handle_radius,
                            0.0, _C.panel_h / 2], device=self.device)
        return base + off

    # ------------------------------------------------------------ reward

    def _get_rewards(self) -> torch.Tensor:
        """奖励各项**必须分别记录**。

        第一次训练 239 轮曲线全程不收敛、在 +55 和 -256 之间来回甩，
        而我只有总 reward，只能猜是哪一项在爆。猜了三轮都没猜对。
        分项记录之后一眼就能看出来是谁。
        """
        cab = self.cabinet.data
        opening = cab.joint_pos[:, self._dj[0]]
        d = self.executor.data

        # 1) 朝目标开度收敛（主项），**势函数式**：只奖励「与目标的差距」的减少量。
        # 早期写成 progress × sign(goal - opening)，开度一超过目标符号就翻转，
        # 奖励出现悬崖，训练曲线剧烈震荡（52 → -90 → +17 → -51）。
        # 用差距的差分则处处连续，且求和 telescope 到「初始差距 - 最终差距」。
        gap = (self.goal - opening).abs()
        r_prog = self.cfg.w_progress * (self.prev_gap - gap).clamp(-0.02, 0.02)
        self.prev_gap = gap.clone()
        r = r_prog

        # 2) 靠近把手：**势函数式塑形**（只奖励距离的减少量）。
        # 早期用 exp(-4d) 直接给分，随机策略每步就能拿 +0.29、累计 +31.65，
        # 而抽屉几乎没开——策略可以靠「在把手附近悬停但不开抽屉」无限刷分。
        # 差分形式求和会 telescope，总量等于起止距离之差，刷不了。
        dist = (self._handle_pos_w() - d.root_pos_w).norm(dim=-1)
        r_reach = self.cfg.w_reach * (self.prev_dist - dist).clamp(-0.05, 0.05)
        self.prev_dist = dist.clone()
        r = r + r_reach

        # 3) 达标
        reached = (opening - self.goal).abs() < self.cfg.goal_tol
        r_succ = self.cfg.w_success * reached.float()
        r = r + r_succ

        # 4) 惩罚：过大接触力、动作突变
        cs = contact_summary(self.contact, self.physics_dt, self.num_envs, self.device)
        # 力惩罚**必须封顶**。它原来是无界的：接触力尖峰到几百牛时，
        # 单步就能扣掉几十分，400 步累计上千——这就是曲线甩到 -256 的来源。
        over = (cs[:, 7] - self.cfg.max_contact_force).clamp(min=0.0, max=20.0)
        r_force = self.cfg.w_force * over
        r = r - r_force
        r_act = self.cfg.w_action * (self.actions - self.prev_act).pow(2).sum(dim=-1)
        r = r - r_act

        # 5) 失败终止的显式惩罚。
        # 没有它就有一个致命漏洞：每步 reward 可能为负，而策略只要让执行器
        # 飞出边界就能立刻结束 episode——**主动自杀比继续做任务划算**。
        # 实测表现为 episode 长度掉到 250 左右而 reward 一路走低。
        r_fail = self.cfg.w_fail * self._far_buf.float()
        r = r - r_fail

        self.prev_open = opening.clone()
        self.prev_act = self.actions.clone()

        # 分项写进 extras，rsl_rl 会打进 tensorboard 并在终端汇总
        self.extras.setdefault("log", {})
        self.extras["log"].update({
            "rew/1_progress": r_prog.mean().item(),
            "rew/2_reach": r_reach.mean().item(),
            "rew/3_success": r_succ.mean().item(),
            "rew/4_force": (-r_force).mean().item(),
            "rew/5_action": (-r_act).mean().item(),
            "rew/6_fail": (-r_fail).mean().item(),
            "rew/total": r.mean().item(),
            "diag/opening_mm": (opening * 1000).mean().item(),
            "diag/contact_force_N": cs[:, 7].mean().item(),
            "diag/contact_force_max_N": cs[:, 7].max().item(),
            "diag/over_force_N": over.mean().item(),
        })
        return r

    # ------------------------------------------------------------ 终止

    def _get_dones(self):
        opening = self.cabinet.data.joint_pos[:, self._dj[0]]
        reached = (opening - self.goal).abs() < self.cfg.goal_tol
        self.hold = torch.where(reached, self.hold + 1, torch.zeros_like(self.hold))
        success = self.hold >= self.cfg.hold_steps

        # 执行器跑太远 = 失败（reward 里有显式惩罚，见 _get_rewards 第 5 项）
        far = (self.executor.data.root_pos_w - self.scene.env_origins).norm(dim=-1) > 1.2
        self._far_buf = far
        # 终止时的诊断量必须在这里存下来。DirectRLEnv 在 step() 内部会自动
        # 重置终止的 env，评估脚本在 step() 返回后再读 joint_pos / episode_length_buf
        # 读到的是**重置后的 0**，不是终止时的值。
        self.term_opening = opening.clone()
        self.term_steps = self.episode_length_buf.clone()
        if self.cfg.disable_termination:
            z = torch.zeros_like(far)
            self.success_buf = success
            return z, self.episode_length_buf >= self.max_episode_length - 1
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        self.success_buf = success
        return success | far, timeout

    # ------------------------------------------------------------ reset

    def _reset_idx(self, env_ids):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.cabinet._ALL_INDICES
        super()._reset_idx(env_ids)
        if not hasattr(self, "act"):
            self._init_buffers()
        n = len(env_ids)
        dev = self.device

        # 抽屉归零
        z = torch.zeros(n, 1, device=dev)
        self.cabinet.write_joint_state_to_sim(z, z, joint_ids=self._dj, env_ids=env_ids)
        self.cabinet.set_joint_effort_target(z, joint_ids=self._dj, env_ids=env_ids)

        # 目标开度
        lo, hi = self.cfg.goal_range
        self.goal[env_ids] = torch.rand(n, device=dev) * (hi - lo) + lo

        # 执行器初始位姿：把手前方
        st = self.executor.data.default_root_state[env_ids].clone()
        base = self.scene.env_origins[env_ids].clone()
        st[:, 0] = base[:, 0] + _C.panel_t + _C.handle_clearance + 0.12
        st[:, 1] = base[:, 1]
        st[:, 2] = base[:, 2] + _C.panel_h / 2 + 0.125     # 钩杆重心在杆中部
        if self.cfg.randomize:
            st[:, :3] += (torch.rand(n, 3, device=dev) - 0.5) * 2 * self.cfg.exec_pos_noise
        st[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev).repeat(n, 1)
        st[:, 7:] = 0.0
        self.executor.write_root_state_to_sim(st, env_ids=env_ids)

        self.act.reset(st[:, :3], st[:, 3:7], env_ids=env_ids)
        self.tgt_pos[env_ids] = st[:, :3]
        self.tgt_quat[env_ids] = st[:, 3:7]
        self.hold[env_ids] = 0
        self.prev_open[env_ids] = 0.0
        self.prev_act[env_ids] = 0.0
        self.prev_dist[env_ids] = (self._handle_pos_w()[env_ids] - st[:, :3]).norm(dim=-1)
        self.prev_gap[env_ids] = self.goal[env_ids].abs()
        self._far_buf[env_ids] = False

    # ------------------------------------------------------------ 杂项

    @property
    def _drawer_body(self):
        if not hasattr(self, "_db"):
            self._db = self.cabinet.body_names.index("Drawer")
        return self._db

    @property
    def joint_damping(self):
        return self.cabinet.data.joint_damping[:, self._dj[0]]
