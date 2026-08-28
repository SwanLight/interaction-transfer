"""S2 冒烟：环境能否建起来、步进、观测维度对不对、reward 有没有信号。

在花 GPU 小时训练之前先跑这个。`plan/04` §13 的诊断顺序第 1、3 条。
"""
import argparse, os, sys

_ap = argparse.ArgumentParser()
_ap.add_argument("--task", default="drawer")
_ap.add_argument("--envs", type=int, default=16)
_ap.add_argument("--steps", type=int, default=120)
_a, _ = _ap.parse_known_args()

from isaaclab.app import AppLauncher
_app = AppLauncher(headless=True).app

import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from it.envs.drawer import DrawerEnv, DrawerEnvCfg

log = open(f"/tmp/s2_smoke_{_a.task}.txt", "w", buffering=1)
def P(*x): print(*x, file=log)

cfg = DrawerEnvCfg()
cfg.scene.num_envs = _a.envs
env = DrawerEnv(cfg)
P(f"环境已建：num_envs={env.num_envs}  device={env.device}")
P(f"控制频率 = {1.0/(cfg.sim.dt*cfg.decimation):.1f} Hz  (物理 {1/cfg.sim.dt:.0f} Hz, decimation {cfg.decimation})")
P(f"episode 最大步数 = {env.max_episode_length}")

obs, _ = env.reset()
o = obs["policy"]
P(f"\n观测 shape = {tuple(o.shape)}  (cfg 声明 {cfg.observation_space})")
P(f"  是否含 NaN/Inf: {bool(torch.isnan(o).any() or torch.isinf(o).any())}")
P(f"  数值范围 [{o.min():.3f}, {o.max():.3f}]")
assert o.shape[1] == cfg.observation_space, f"观测维度不符！实际 {o.shape[1]}"

P("\n--- 随机动作 rollout ---")
tot_r = torch.zeros(env.num_envs, device=env.device)
n_succ = n_term = 0
max_open = torch.zeros(env.num_envs, device=env.device)
for i in range(_a.steps):
    a = torch.rand(env.num_envs, cfg.action_space, device=env.device) * 2 - 1
    obs, rew, term, trunc, _ = env.step(a)
    tot_r += rew
    op = env.cabinet.data.joint_pos[:, env._dj[0]]
    max_open = torch.maximum(max_open, op)
    n_succ += int(env.success_buf.sum())
    n_term += int(term.sum())
    if i % 40 == 0:
        P(f"  step {i:3d}  rew {rew.mean():+.4f}  开度 max {op.max()*1000:6.1f} mm  "
          f"接触力 {env.contact.data.net_forces_w[:,0].norm(dim=-1).max():.2f} N")

P(f"\n随机策略 {_a.steps} 步：")
P(f"  累计 reward 均值 = {tot_r.mean():+.3f}  范围 [{tot_r.min():+.2f}, {tot_r.max():+.2f}]")
P(f"  最大开度 = {max_open.max()*1000:.1f} mm （目标 {cfg.goal_range[0]*1000:.0f}–{cfg.goal_range[1]*1000:.0f}）")
P(f"  成功次数 = {n_succ}（随机策略应接近 0）")
P(f"  终止次数 = {n_term}")
P(f"\n判定：")
P(f"  观测无 NaN            {'OK' if not bool(torch.isnan(o).any()) else 'FAIL'}")
P(f"  reward 有梯度信号     {'OK' if tot_r.std() > 1e-3 else 'FAIL（所有 env 一样，学不动）'}")
P(f"  接触力可读            {'OK' if float(env.contact.data.net_forces_w.norm(dim=-1).max()) > 0 else '未接触（随机策略正常）'}")
log.close()
os._exit(0)
