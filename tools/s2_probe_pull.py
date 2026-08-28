"""诊断：抽屉是被「持续拉开」的，还是被「弹一下然后自己滑」的？

用户看录像时的质疑：「感觉像抽屉自己有动力在动一样」。

这个怀疑有具体的物理依据：抽屉质量 1.2 kg、关节阻尼 3.0。如果策略学会了
给一个冲量然后让它滑行，奖励里的 progress 项照样给分，画面上就像
「抽屉自己往外走」。**那是退化解，不是在做任务。**

判据（逐控制步记录，看开度上升期间）：

- **持续拉**：接触力全程 > 0，且抽屉速度与接触力同步
- **弹一下**：接触力只有开头几步非零，之后归零而抽屉继续滑

顺带记录钩杆姿态，确认横钩到底朝哪、有没有参与接触。
"""
import argparse, os, sys

_ap = argparse.ArgumentParser()
_ap.add_argument("--run", required=True)
_ap.add_argument("--ckpt", default="model_100.pt")
_ap.add_argument("--envs", type=int, default=8)
_ap.add_argument("--steps", type=int, default=150)
_ap.add_argument("--runs_dir", default="/workspace/interaction_transfer/runs")
_a, _ = _ap.parse_known_args()

from isaaclab.app import AppLauncher
_app = AppLauncher(headless=True).app

import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from it.envs.drawer import DrawerEnv, DrawerEnvCfg
from it.contact_utils import extract_contact_points

TRAIN_CFG = {
    "algorithm": {"class_name": "PPO", "clip_param": 0.2, "desired_kl": 0.01,
                  "entropy_coef": 0.006, "gamma": 0.99, "lam": 0.95,
                  "learning_rate": 3.0e-4, "max_grad_norm": 1.0,
                  "num_learning_epochs": 5, "num_mini_batches": 4,
                  "schedule": "adaptive", "use_clipped_value_loss": True,
                  "value_loss_coef": 1.0},
    "policy": {"class_name": "ActorCritic", "activation": "elu",
               "actor_hidden_dims": [256, 256, 128],
               "critic_hidden_dims": [256, 256, 128], "init_noise_std": 1.0},
    "obs_groups": {"policy": ["policy"], "critic": ["policy"]},
    "num_steps_per_env": 32, "save_interval": 100,
    "empirical_normalization": True, "logger": "tensorboard", "max_iterations": 1,
}

log = open("/tmp/s2_probe_pull.txt", "w", buffering=1)
def P(*x): print(*x, file=log)

cfg = DrawerEnvCfg()
cfg.scene.num_envs = _a.envs
cfg.disable_termination = True      # 跑完整过程，不被自动重置打断
cfg.episode_length_s = 20.0
env_raw = DrawerEnv(cfg)
env = RslRlVecEnvWrapper(env_raw)
runner = OnPolicyRunner(env, TRAIN_CFG, log_dir=os.path.join(_a.runs_dir, _a.run),
                        device=str(env_raw.device))
runner.load(os.path.join(_a.runs_dir, _a.run, _a.ckpt))
policy = runner.get_inference_policy(device=env_raw.device)

obs = env.get_observations()
if isinstance(obs, tuple):
    obs = obs[0]
dt = cfg.sim.dt * cfg.decimation

P("逐步记录（env 0）")
P("")
P("**决定性判据**：抽屉沿 +X 打开。作用在抽屉上的接触力若是 +X，是正常拉开；")
P("若接触力是 -X 而抽屉仍在 +X 移动，说明它不是被推开的——那是求解器异常或别的驱动源。")
P("")
P(f"{'步':>4} {'开度mm':>8} {'速度mm/s':>10} {'|F|N':>7} {'Fx':>8} {'Fz':>8} "
  f"{'点数':>5} {'钩杆vx':>9} {'钩杆vz':>9}")

from isaaclab.utils.math import quat_apply
hist = []
with torch.inference_mode():
    for i in range(_a.steps):
        act = policy(obs)
        out = env.step(act)
        obs = out[0]
        if isinstance(obs, tuple):
            obs = obs[0]

        op = env_raw.cabinet.data.joint_pos[:, env_raw._dj[0]]
        vel = env_raw.cabinet.data.joint_vel[:, env_raw._dj[0]]
        cps = extract_contact_points(env_raw.contact, cfg.sim.dt)
        # 作用在**抽屉**上的接触力 = -(作用在钩杆上的力)。
        # 传感器挂在钩杆上，法向由 filter(抽屉) 指向 sensor(钩杆)。
        fvec = torch.zeros(env_raw.num_envs, 3, device=env_raw.device)
        fmag = torch.zeros(env_raw.num_envs, device=env_raw.device)
        nc = []
        for e, c in enumerate(cps):
            nc.append(c.num_contacts)
            if c.is_empty():
                continue
            f_on_hook = (c.normal_forces.unsqueeze(-1) * c.normals).sum(dim=0) \
                        + c.friction_forces.sum(dim=0)
            fvec[e] = -f_on_hook          # 反作用力 = 作用在抽屉上的力
            fmag[e] = c.normal_forces.abs().sum()
        hv = env_raw.executor.data.root_lin_vel_w
        hist.append((op[0].item(), vel[0].item(), fmag[0].item(), nc[0], fvec[0, 0].item()))
        if i % 6 == 0:
            P(f"{i:4d} {op[0]*1000:8.2f} {vel[0]*1000:10.2f} {fmag[0]:7.2f} "
              f"{fvec[0,0]:+8.2f} {fvec[0,2]:+8.2f} {nc[0]:5d} "
              f"{hv[0,0]*1000:+9.1f} {hv[0,2]*1000:+9.1f}")

# ---- 判定 ----
moving = [x for x in hist if abs(x[1]) > 0.005]     # 抽屉在动的步
if moving:
    no_contact = [x for x in moving if x[2] < 0.05]
    P(f"\n抽屉在动的控制步数 = {len(moving)}")
    P(f"  其中**接触力接近零**的步数 = {len(no_contact)}  "
      f"({len(no_contact)/len(moving)*100:.1f}%)")
    P(f"  运动期间平均接触力 = {sum(x[2] for x in moving)/len(moving):.3f} N")
    P(f"  运动期间平均接触点数 = {sum(x[3] for x in moving)/len(moving):.2f}")
    frac = len(no_contact) / len(moving)
    if frac > 0.5:
        verdict = "弹一下然后滑行（退化解）"
    elif frac > 0.2:
        verdict = "部分滑行"
    else:
        verdict = "持续接触拉开（正常）"
    P(f"\n判定 A（接触是否持续）：{verdict} —— 无接触占比 {frac*100:.1f}%")

    # 判定 B：抽屉在 +X 打开时，作用在它身上的力是不是 +X
    opening_steps = [x for x in hist if x[1] > 0.005]      # 速度 > 5 mm/s 且为正
    if opening_steps:
        push_out = [x for x in opening_steps if x[4] > 0.05]
        push_in = [x for x in opening_steps if x[4] < -0.05]
        no_f = [x for x in opening_steps if abs(x[4]) <= 0.05]
        P(f"\n判定 B（力的方向是否解释运动）：抽屉正在打开的步数 = {len(opening_steps)}")
        P(f"  受力 +X（推开，正常）  {len(push_out):4d}  ({len(push_out)/len(opening_steps)*100:5.1f}%)")
        P(f"  受力 -X（推回，反常）  {len(push_in):4d}  ({len(push_in)/len(opening_steps)*100:5.1f}%)")
        P(f"  几乎无力（滑行）      {len(no_f):4d}  ({len(no_f)/len(opening_steps)*100:5.1f}%)")
        if len(push_out) / len(opening_steps) > 0.6:
            P("  -> 抽屉是被正常推开的")
        elif len(no_f) / len(opening_steps) > 0.5:
            P("  -> **抽屉主要在无外力滑行**：策略给冲量后让它自己走（退化解）")
        else:
            P("  -> **异常**：抽屉在受反向力的情况下仍在打开，物理不自洽")
else:
    P("\n抽屉全程没动")

# 横钩朝向统计
P(f"\n横钩世界系朝向（最后一步，各 env）：")
q = env_raw.executor.data.root_quat_w
tipdir = quat_apply(q, torch.tensor([[1.0, 0.0, 0.0]], device=env_raw.device)
                    .repeat(env_raw.num_envs, 1))
for e in range(min(_a.envs, 8)):
    d = tipdir[e]
    P(f"  env{e}: [{d[0]:+.2f}, {d[1]:+.2f}, {d[2]:+.2f}]  "
      f"{'指向柜体(-X，能勾)' if d[0] < -0.5 else ('指向侧面(±Y，勾不上)' if abs(d[1]) > 0.5 else '指向外(+X，勾不上)')}")
log.close()
os._exit(0)
