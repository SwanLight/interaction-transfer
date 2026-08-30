"""E-I：交互跟踪执行器的环境（`plan/04` §5）。主系统就是它。

与 `envs/drawer.py`（E-T / Expert 的载体）的根本区别：**这个环境不知道任务是什么**。
它的 reward 只比"实测的交互"与"指令要求的交互"，没有任何一项读任务的成功判据、
关节角或 dirt 网格。物体是什么、要拉开多少，全部只经由 artifact 的指令通道进来。

三个设计点，都是被实测或文献钉住的
----------------------------------

**一、动作空间是混合力/位控，不是纯位姿增量。** 指令要跟踪的是 *traction*
（面密度），而接触力是穿透量的刚性函数——`float_ctrl` 的文档里记着实测：把推子放在
距表面 0.5 mm 处，法向力只有 0.06 N，而目标是 25 N。纯位置动作等于要求 PPO 学会
亚毫米定位。接触密集操作的 RL 文献一致的做法是把**柔顺/力设定值放进动作空间**
（variable impedance / admittance，Buchli；Martín-Martín；Beltran-Hernandez
[arXiv 2003.00628]），近期的 sim-to-real 消融也显示去掉法向力调节会让成功率大幅下降。
所以动作 = 6 维位姿增量 + **3 维力矢量**，力矢量交给 `FloatingPD(force_dir=...)`
做沿任意方向的混合控制（P-39：力控掩码只能表达世界轴对齐的方向，这里不能用）。

**方向由策略自己出，不直接用指令里的 engage 方向。** 否则 C0/C1（没有方向字段）
的动作空间就和 C4 不一样了，而 `plan/04` §7 要求所有条件的结构与参数量逐位相同。

**二、滑移速率用位姿差分算，与 S4 同一个判据（D-49）。** PhysX 报的瞬时相对速度在
角速度饱和时不可信（P-52），且离线的 `mode/pose_slip` 用的就是两刚体位姿差分——
在线换一个算法，reward 追的就不是 artifact 里写的那个量。

**三、接触取 `extract_contact_points_padded`。** `contact_summary` /
`extract_contact_points` 那条路径的摩擦力是错的（P-36），而且它逐 env 循环，
2048 个 env 下根本跑不动。
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObject
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

from it import assets as A
from it.contact_utils import extract_contact_points_padded
from it.ei_command import CommandBank, WindowTracker
from it.ei_reward import (assign_cells, effect_deficit, effect_magnitude,
                          interaction_reward, surface_traction)
from it.envs.base import FloatingBaseAction
from it.float_ctrl import FloatingPD

#: 每个 env 每步保留的接触点数。定长是必须的（`plan/02` §7 第 3 条：
#: 改变接触体数量后表示维度不变）。
MAX_CONTACTS = 16
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
    episode_length_s = 8.0
    action_space = 9                    # 6 位姿增量 + 3 力矢量
    #: 观测里**不放**指令张量本身：逐格空间场是 256×34，2048 个 env × horizon 32
    #: 的 PPO buffer 会到几个 GB。放的是 (command_index, bin_index)，
    #: 由 `ei_policy` 那侧按下标去 `CommandBank` 取——指令在 episode 内是常量，
    #: 变的只有格号。
    observation_space = PROPRIO_DIM + 2 + 4
    state_space = PROPRIO_DIM + 2 + 4 + PRIVILEGED_DIM

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1 / 150, render_interval=3,
        physx=sim_utils.PhysxCfg(gpu_max_rigid_contact_count=2 ** 22,
                                 gpu_max_rigid_patch_count=2 ** 20))
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=2.0,
                                                     replicate_physics=True)
    light: object = AssetBaseCfg(prim_path="/World/light",
                                 spawn=sim_utils.DomeLightCfg(intensity=800.0))

    #: 被操作物体与执行器由 curriculum 换（`plan/04` §10 的 A/B/C 三段）。
    object_cfg: object = A.BLOCK_CFG.replace(prim_path="/World/envs/env_.*/Object")
    executor_cfg: object = A.PADROD_CFG.replace(prim_path="/World/envs/env_.*/Executor")

    #: 指令库：一批冻结 artifact 的路径。**留出任务由 `forbid` 在入口拦**
    #: （`plan/04` §5.4：不得只靠自觉）。
    command_glob: str = "/tmp/s6/probe/*.npz"
    forbid: tuple[str, ...] = ()

    max_contact_force: float = A.MAX_NORMAL_FORCE
    #: 动作里力矢量的量程（N）。上限取安全法向力，越界由 r_safety 罚。
    force_scale: float = A.MAX_NORMAL_FORCE
    w_effect: float = 1.0
    w_region: float = 1.0
    w_mode: float = 0.5
    w_mech: float = 1.0
    w_force: float = 0.05
    w_action: float = 0.002


class InteractionEnv(DirectRLEnv):
    """交互跟踪执行器。**reward 里没有任何一项知道任务是什么。**"""

    cfg: InteractionEnvCfg

    def _setup_scene(self):
        self.obj = RigidObject(self.cfg.object_cfg)
        self.executor = RigidObject(self.cfg.executor_cfg)
        self.contact = ContactSensor(ContactSensorCfg(
            prim_path="/World/envs/env_.*/Executor", track_contact_points=True,
            max_contact_data_count_per_prim=MAX_CONTACTS,
            filter_prim_paths_expr=["/World/envs/env_.*/Object"],
            update_period=0.0, history_length=0))
        self.scene.rigid_objects["object"] = self.obj
        self.scene.rigid_objects["executor"] = self.executor
        self.scene.sensors["contact"] = self.contact
        self.cfg.light.spawn.func("/World/light", self.cfg.light.spawn)
        self.scene.clone_environments(copy_from_source=False)

    def _configure_gym_env_spaces(self):
        super()._configure_gym_env_spaces()
        self._init_buffers()

    def _init_buffers(self):
        import glob
        n, dev = self.num_envs, self.device
        self.bank = CommandBank(sorted(glob.glob(self.cfg.command_glob)), device=dev,
                                forbid=self.cfg.forbid)
        self.tracker = WindowTracker(self.bank, n, device=dev)
        self.act = FloatingBaseAction(n, dev, pos_scale=0.010, rot_scale=0.04,
                                      pos_limit=0.6)
        self.pd = FloatingPD(self.executor, kp_pos=600.0, kd_pos=50.0,
                             kp_rot=60.0, kd_rot=8.0, max_force=180.0,
                             max_torque=18.0, kd_force=40.0)
        self.actions = torch.zeros(n, self.cfg.action_space, device=dev)
        self.prev_actions = torch.zeros_like(self.actions)
        self.tgt_pos = torch.zeros(n, 3, device=dev)
        self.tgt_quat = torch.zeros(n, 4, device=dev)
        self.tgt_quat[:, 0] = 1.0
        self.ff_force = torch.zeros(n, 3, device=dev)
        self.progress_obs = torch.zeros(n, 6, device=dev)
        self.contact_obs = torch.zeros(n, 6, device=dev)
        # 位姿差分算滑移要上一控制步的两个刚体位姿（D-49，与 S4 同一个判据）。
        self.prev_obj_pos = torch.zeros(n, 3, device=dev)
        self.prev_obj_quat = torch.zeros(n, 4, device=dev)
        self.prev_obj_quat[:, 0] = 1.0
        self.prev_exec_pos = torch.zeros(n, 3, device=dev)
        self.prev_exec_quat = torch.zeros(n, 4, device=dev)
        self.prev_exec_quat[:, 0] = 1.0
        self.prev_effect_pose = torch.zeros(n, 7, device=dev)
        self.reward_terms: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------ 动作

    def _pre_physics_step(self, actions: torch.Tensor):
        self.prev_actions = self.actions
        self.actions = actions.clone().clamp(-1.0, 1.0)
        self.tgt_pos, self.tgt_quat = self.act.step(self.actions[:, :6])
        self.ff_force = self.actions[:, 6:9] * self.cfg.force_scale

    def _apply_action(self):
        # 力矢量长度接近零时退回纯位控：给 FloatingPD 一个零方向会让它把整个
        # 位置 PD 投影掉（P-39 的同一处），执行器直接失控漂走。
        magnitude = self.ff_force.norm(dim=-1, keepdim=True)
        use_force = magnitude > 1e-3
        direction = torch.where(use_force, self.ff_force / magnitude.clamp_min(1e-9),
                                torch.zeros_like(self.ff_force))
        f, tq = self.pd.compute(self.tgt_pos, self.tgt_quat,
                                ff_force=torch.where(use_force, self.ff_force,
                                                     torch.zeros_like(self.ff_force)),
                                force_dir=torch.where(use_force, direction,
                                                      torch.zeros_like(direction)))
        self.executor.set_external_force_and_torque(f, tq)

    # ------------------------------------------------------------ 交互量

    def _object_frame(self):
        d = self.obj.data
        pos = d.root_pos_w - self.scene.env_origins
        return pos, d.root_quat_w

    def _contacts(self):
        """本控制步的接触，全部换算到**物体系**——`plan/02` §1 的坐标约定。"""
        raw = extract_contact_points_padded(
            self.contact, self.physics_dt,
            self.executor.data.root_pos_w, max_points=MAX_CONTACTS)
        obj_pos, obj_quat = self._object_frame()
        inv = quat_conjugate(obj_quat)
        world = raw["positions"] - self.scene.env_origins[:, None, :]
        k = world.shape[1]
        to_obj = lambda v: quat_apply(inv[:, None, :].expand(-1, k, -1), v)  # noqa: E731
        pos_obj = to_obj(world - obj_pos[:, None, :])
        normal_obj = to_obj(raw["normals"])
        force_obj = to_obj(raw["normals"] * raw["normal_forces"][..., None]
                           + raw["friction_forces"])
        return {"pos": pos_obj, "normal": normal_obj, "force": force_obj,
                "valid": raw["valid"], "normal_force": raw["normal_forces"]}

    def _slip_speed(self, pos_obj: torch.Tensor, normal_obj: torch.Tensor,
                    valid: torch.Tensor) -> torch.Tensor:
        """接触斑块的切向滑移速率，由**两刚体的位姿差分**给出（D-49 / P-52）。

        把同一个接触点分别当作固连在物体上和固连在执行器上，看这一步之后两者
        分开了多远；差的切向分量除以 dt 就是滑移速率。**不用 PhysX 报的瞬时相对
        速度**——角速度饱和时它整体虚高，实测积出 59 mm 滑移而接触点在物体系里
        一动没动。
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

    # ------------------------------------------------------------ reward

    def _get_rewards(self) -> torch.Tensor:
        c = self._contacts()
        cmd, b = self.tracker.command_index, self.tracker.bin_index
        cells = self.bank.gather("surface/points_obj", cmd)
        cell_normals = self.bank.gather("surface/normals_obj", cmd)
        cell_index = assign_cells(c["pos"], c["normal"], c["valid"], cells, cell_normals)
        traction, mass = surface_traction(c["pos"], c["force"], c["valid"],
                                          cell_index, self.bank.n_surface)
        slip = self._slip_speed(c["pos"], c["normal"], c["valid"])
        slip_cell = torch.zeros(self.num_envs, self.bank.n_surface, device=self.device)
        slip_cell.scatter_reduce_(1, cell_index, slip, reduce="amax", include_self=True)

        allowed = self.bank.gather_bin("region/allowed", cmd, b)
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

        over = (c["normal_force"] - self.cfg.max_contact_force).clamp_min(0.0).sum(-1)
        smooth = (self.actions - self.prev_actions).pow(2).sum(-1)
        safety = -(self.cfg.w_force * over + self.cfg.w_action * smooth)

        # 先推窗口：缺口要用**累计到本步为止**的完成量算，所以顺序不能反。
        region_force = (mass * allowed.float()).sum(-1)
        progress = self.tracker.step(effect_increment=got,
                                     region_normal_force=region_force)
        self.progress_obs = torch.stack(list(progress.values()), dim=-1)

        terms = interaction_reward(
            effect_deficit=effect_deficit(self.tracker.achieved, self.tracker.demand()),
            traction=traction, mass=mass, slip_speed=slip_cell, allowed=allowed,
            traction_lo=self.bank.gather_bin("mech/traction_obj/lo", cmd, b),
            traction_hi=self.bank.gather_bin("mech/traction_obj/hi", cmd, b),
            slip_lo=self.bank.gather_bin("mode/slip_speed/lo", cmd, b),
            slip_hi=self.bank.gather_bin("mode/slip_speed/hi", cmd, b),
            force_penalty=safety)

        self.contact_obs = torch.stack([
            c["valid"].float().sum(-1) / MAX_CONTACTS,
            c["normal_force"].sum(-1) / self.cfg.max_contact_force,
            region_force / self.cfg.max_contact_force,
            mass.sum(-1) / self.cfg.max_contact_force,
            slip_cell.amax(-1), traction.norm(dim=-1).amax(-1) * 1e-5], dim=-1)
        self._remember_poses()

        # **每一项都进 extras["log"]**：没有分项记录，训练不收敛时只能猜（P-27）。
        self.extras.setdefault("log", {}).update(
            {k: v.mean() for k, v in terms.as_log().items()})
        return (self.cfg.w_effect * terms.effect + self.cfg.w_region * terms.region
                + self.cfg.w_mode * terms.mode + self.cfg.w_mech * terms.mech
                + terms.safety)

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

    # ------------------------------------------------------------ 终止与重置

    def _get_dones(self):
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        far = (self.executor.data.root_pos_w
               - self.scene.env_origins).norm(dim=-1) > 1.5
        return self.tracker.finished | far, timeout

    def _reset_idx(self, env_ids):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.obj._ALL_INDICES
        super()._reset_idx(env_ids)
        if not hasattr(self, "act"):
            self._init_buffers()
        n, dev = len(env_ids), self.device
        self.tracker.reset(env_ids, torch.randint(0, len(self.bank), (n,), device=dev))

        st = self.obj.data.default_root_state[env_ids].clone()
        st[:, :3] += self.scene.env_origins[env_ids]
        self.obj.write_root_state_to_sim(st, env_ids=env_ids)
        ex = self.executor.data.default_root_state[env_ids].clone()
        ex[:, :3] += self.scene.env_origins[env_ids]
        self.executor.write_root_state_to_sim(ex, env_ids=env_ids)
        self.act.reset(ex[:, :3], ex[:, 3:7], env_ids=env_ids)
        self.tgt_pos[env_ids], self.tgt_quat[env_ids] = ex[:, :3], ex[:, 3:7]
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.contact_obs[env_ids] = 0.0
        self.progress_obs[env_ids] = 0.0
        self._remember_poses(env_ids)
