"""S2 前置：脚本化验证任务在几何上可完成，并标定 reward 量纲。

`plan/04` §13 的诊断顺序第 1 条就是「几何脚本是否能完成」——**在训练之前**。
S1 只验了抽屉的行程和把手净空尺寸，没验过钩杆真能勾住把手拉开。

顺带解决 reward 量纲问题：跑一条**成功**的脚本轨迹，把各项 reward 累计值
打出来，就知道权重配得对不对。凭空猜权重是 S2 第一次训练失败的原因
（progress 总量 1.6，action 惩罚 -32，最优解是不动）。
"""
import argparse, os, sys, math

_ap = argparse.ArgumentParser()
_ap.add_argument("--envs", type=int, default=4)
_ap.add_argument("--video", action="store_true")
_a, _ = _ap.parse_known_args()

from isaaclab.app import AppLauncher
_app = AppLauncher(headless=True, enable_cameras=_a.video).app

import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from it.envs.drawer import DrawerEnv, DrawerEnvCfg
from it.build_assets import CabinetCfg, HookCfg

log = open("/tmp/s2_scripted.txt", "w", buffering=1)
def P(*x): print(*x, file=log)

C, H = CabinetCfg(), HookCfg()
cfg = DrawerEnvCfg()
cfg.scene.num_envs = _a.envs
cfg.disable_termination = True   # 一条完整轨迹跑完再统计
cfg.episode_length_s = 20.0
env = DrawerEnv(cfg)
obs, _ = env.reset()
dev = env.device
dt = cfg.sim.dt * cfg.decimation

P("脚本化可行性验证：钩杆能否勾住把手拉开抽屉")
P(f"把手：杆心距面板 {(C.panel_t + C.handle_clearance + C.handle_radius)*1000:.0f} mm，"
  f"净空 {C.handle_clearance*1000:.0f} mm，杆半径 {C.handle_radius*1000:.0f} mm")
P(f"钩杆：主杆半径 {H.shaft_radius*1000:.0f} mm，横钩长 {H.hook_len*1000:.0f} mm")
P(f"目标开度：{env.goal[0].item()*1000:.1f} mm\n")

# --- 几何分析与「真正勾住」的动作设计 ---
#
# 把手区可通行空间：
#   面板前面 x = 18 mm，把手杆心 x = 74 mm，杆背面 x = 63 mm
#   -> 净空 x ∈ [18, 63]，中心 40.5 mm；主杆直径 16 mm，两侧各余 14.5 mm
#   支撑柱在 y = ±62.5 mm，y = 0 这一列是空的，竖直主杆可自由下探
#
# **为什么必须真的勾住**（S2 实测教训）：
# 早期设计让横钩指 +Y（纯避让），拉力全靠**主杆圆柱压把手圆柱**——
# 两根轴垂直的圆柱接触是**点接触**。抽屉阻尼小时勉强够用；阻尼提高后
# 需要的力变大，点接触立刻滑脱（实测钩杆飞到 319 mm 外，抽屉只开了 3.6 mm）。
#
# 正确的动作分三段：
#   ① 下探：横钩转到 +Y 避让，竖直主杆落进净空（横钩指 +X 会扫过把手撞上）
#   ② 转正：到位后绕 Z 转 90°，横钩转到 +X，伸到**把手杆下方**
#   ③ 拉：主杆前面压把手背面提供拉力，横钩挡住向上脱出 -> **形封闭**
#
# 高度约束：横钩顶面必须低于把手杆底面（79 mm），否则转正时会撞上。
# 横钩相对原点 z 偏移 = -shaft_len/2 + r = -117 mm，横钩半径 8 mm
# -> 原点高度 ≤ 79 + 117 - 8 = 188 mm。取 180 mm，余 8 mm 间隙。
tip_dz = -H.shaft_len / 2 + H.shaft_radius
GAP_X = C.panel_t + C.handle_clearance / 2
BAR_X = C.panel_t + C.handle_clearance + C.handle_radius
handle = env._handle_pos_w()
base_xy = handle.clone()
base_xy[:, 0] = handle[:, 0] - BAR_X + GAP_X
base_xy[:, 1] = handle[:, 1]

P(f"净空 x ∈ [{C.panel_t*1000:.0f}, {(C.panel_t+C.handle_clearance)*1000:.0f}] mm，"
  f"中心 {GAP_X*1000:.1f} mm；主杆直径 {2*H.shaft_radius*1000:.0f} mm -> 两侧各余 "
  f"{(C.handle_clearance/2 - H.shaft_radius)*1000:.1f} mm")
P(f"横钩顶面需低于把手底面 {(handle[0,2].item() - C.handle_radius)*1000:.0f} mm "
  f"-> 原点高度上限 {(handle[0,2].item() - C.handle_radius - tip_dz - H.shaft_radius)*1000:.0f} mm")

Z_ENGAGE = 0.180
Q_AVOID = math.pi / 2      # 下探时横钩指 +Y
Q_HOOK = 0.0               # 勾住时横钩指 +X（伸到把手下方）


def phase_pose(i):
    """① 悬停对位 ② 下探（横钩避让）③ 原地转正让横钩伸到把手下 ④ 拉开"""
    t = base_xy.clone()
    ang = Q_AVOID
    if i < 50:
        t[:, 2] = 0.34
    elif i < 140:
        t[:, 2] = 0.34 + (Z_ENGAGE - 0.34) * (i - 50) / 90.0
    elif i < 190:
        t[:, 2] = Z_ENGAGE
        ang = Q_AVOID * (1.0 - (i - 140) / 50.0)      # 转到 Q_HOOK
    else:
        t[:, 2] = Z_ENGAGE
        ang = Q_HOOK
        t[:, 0] += min((i - 190) / 170.0, 1.0) * 0.34
    q = torch.zeros(env.num_envs, 4, device=dev)
    q[:, 0] = math.cos(ang / 2)
    q[:, 3] = math.sin(ang / 2)
    return t, q


def phase_target(i):
    return phase_pose(i)[0]


# **通过 policy 的动作接口驱动**，不直接调底层控制器。
#
# 这一点很重要：直接调 PD 只能验证「几何上做得到」，验证不了「策略能不能
# 用它的动作表达出这个动作」。动作是位姿增量、每步有幅度上限，如果脚本
# 要求的运动超过上限，策略再聪明也做不到——那就是动作空间设计的问题，
# 必须在这里暴露，而不是等训练不收敛了再回头猜。
#
# 转换：action = (期望目标位姿 - 当前目标位姿) / 每步上限，再 clamp 到 [-1,1]。
# clamp 生效的比例就是「动作空间够不够快」的直接度量。
clamp_hits = torch.zeros(env.num_envs, device=dev)
n_steps_done = 0

st = env.executor.data.default_root_state.clone()
st[:, :3], _q0 = phase_pose(0)[0], phase_pose(0)[1]
st[:, 3:7] = _q0
st[:, 7:] = 0.0
env.executor.write_root_state_to_sim(st)
env.act.reset(st[:, :3], st[:, 3:7])
for _ in range(10):
    env.scene.write_data_to_sim(); env.sim.step(render=False); env.scene.update(cfg.sim.dt)

terms = {k: torch.zeros(env.num_envs, device=dev) for k in
         ["progress", "reach", "success", "force", "action"]}
prev_open = torch.zeros(env.num_envs, device=dev)
prev_dist = (handle - env.executor.data.root_pos_w).norm(dim=-1)
max_open = torch.zeros(env.num_envs, device=dev)
max_fn = torch.zeros(env.num_envs, device=dev)

for i in range(400):
    want, want_q = phase_pose(i)
    # 位姿增量 -> 归一化动作。姿态用 yaw 增量（本任务只需绕 Z 转）。
    dpos = (want - env.act.target_pos) / env.act.pos_scale
    cur_yaw = 2.0 * torch.atan2(env.act.target_quat[:, 3], env.act.target_quat[:, 0])
    want_yaw = 2.0 * torch.atan2(want_q[:, 3], want_q[:, 0])
    dyaw = torch.atan2(torch.sin(want_yaw - cur_yaw), torch.cos(want_yaw - cur_yaw))
    a = torch.zeros(env.num_envs, cfg.action_space, device=dev)
    a[:, :3] = dpos
    a[:, 5] = dyaw / env.act.rot_scale
    clamp_hits += (a[:, :3].abs() > 1.0).any(dim=-1).float()
    n_steps_done += 1
    a = a.clamp(-1.0, 1.0)
    env.step(a)          # 走完整的环境接口：动作 -> 目标位姿 -> PD -> wrench

    op = env.cabinet.data.joint_pos[:, env._dj[0]]
    max_open = torch.maximum(max_open, op)
    dist = (env._handle_pos_w() - env.executor.data.root_pos_w).norm(dim=-1)
    fn = env.contact.data.net_forces_w[:, 0].norm(dim=-1)
    max_fn = torch.maximum(max_fn, fn)

    terms["progress"] += (op - prev_open).clamp(-0.02, 0.02)
    terms["reach"] += (prev_dist - dist).clamp(-0.05, 0.05)
    terms["success"] += ((op - env.goal).abs() < cfg.goal_tol).float()
    terms["force"] += (fn - cfg.max_contact_force).clamp(min=0.0)
    prev_open, prev_dist = op.clone(), dist.clone()

    if i % 80 == 0:
        P(f"  step {i:3d}  开度 {op.mean()*1000:6.1f} mm  接触力 {fn.mean():5.2f} N  "
          f"钩-把手距 {dist.mean()*1000:5.1f} mm")

P(f"\n结果：")
P(f"  最大开度 = {max_open.mean()*1000:.1f} mm（目标 {env.goal.mean()*1000:.1f}，"
  f"任务区间 {cfg.goal_range[0]*1000:.0f}–{cfg.goal_range[1]*1000:.0f}）")
P(f"  峰值接触力 = {max_fn.mean():.2f} N（安全上限 {cfg.max_contact_force}）")
ok = (max_open >= cfg.goal_range[0]).float().mean().item()
P(f"  达到任务下限的比例 = {ok*100:.0f}%")
P(f"  判定：{'钩杆几何上能完成抽屉任务' if ok > 0.5 else '❌ 钩杆够不到/拉不开，需改初始位姿或几何'}")
rate = (clamp_hits / max(n_steps_done, 1)).mean().item()
P(f"\n  动作空间是否够用：{rate*100:.1f}% 的步需要超过单步幅度上限")
P(f"  判定：{'够用' if rate < 0.15 else '❌ 动作空间太慢，策略表达不出这个动作，需放大 pos_scale'}")

P(f"\nreward 量纲标定（一条成功轨迹上各项的累计值，未乘权重）：")
for k, v in terms.items():
    P(f"  {k:9s} {v.mean().item():+10.4f}")
P(f"\n当前权重下的加权值：")
w = dict(progress=cfg.w_progress, reach=cfg.w_reach, success=cfg.w_success,
         force=-cfg.w_force, action=-cfg.w_action)
tot = 0.0
for k, v in terms.items():
    x = w[k] * v.mean().item()
    tot += x
    P(f"  {k:9s} 权重 {w[k]:+8.4f} -> {x:+10.3f}")
P(f"  {'合计':9s} {' ':16s}{tot:+10.3f}")
P(f"\n注：action 惩罚未在脚本轨迹里计（脚本不产生 policy 动作）。"
  f"训练时它约为 -w_action × 10 × 400 = {-cfg.w_action*4000:.1f}，"
  f"必须远小于 progress 的贡献，否则最优解是「不动」。")
log.close(); os._exit(0)
