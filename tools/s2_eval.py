"""S2：评估 Expert 的成功率（Gate A 判据）。

`plan/06` §1：**训练和评估完全分离**，不能从训练 reward 曲线直接下科学结论。
训练曲线里 episode 变短有两种可能——成功终止、或飞出边界失败终止，分不清。

Gate A（`plan/05` §10）A2：固定环境 ≥95%，随机环境 ≥85%。

用法::

    PYTHONPATH=src /isaac-sim/python.sh tools/s2_eval.py \\
        --run expert_drawer_hook_s0 --ckpt model_final.pt --episodes 200
"""
import argparse, os, sys

_ap = argparse.ArgumentParser()
_ap.add_argument("--task", default="drawer_hook")
_ap.add_argument("--run", required=True)
_ap.add_argument("--ckpt", default="model_final.pt")
_ap.add_argument("--envs", type=int, default=256)
_ap.add_argument("--episodes", type=int, default=256, help="至少跑满这么多条 episode")
_ap.add_argument("--randomize", action="store_true")
_ap.add_argument("--out", default="/workspace/interaction_transfer/runs")
_a, _ = _ap.parse_known_args()

from isaaclab.app import AppLauncher
_app = AppLauncher(headless=True).app

import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from it.envs.drawer import DrawerEnv, DrawerEnvCfg

TASKS = {"drawer_hook": (DrawerEnv, DrawerEnvCfg)}
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

tag = f"{_a.run}_{'rand' if _a.randomize else 'fixed'}"
log = open(f"/tmp/s2_eval_{tag}.txt", "w", buffering=1)
def P(*x): print(*x, file=log)

EnvCls, CfgCls = TASKS[_a.task]
cfg = CfgCls()
cfg.scene.num_envs = _a.envs
cfg.randomize = _a.randomize
env_raw = EnvCls(cfg)
env = RslRlVecEnvWrapper(env_raw)

log_dir = os.path.join(_a.out, _a.run)
runner = OnPolicyRunner(env, TRAIN_CFG, log_dir=log_dir, device=str(env_raw.device))
ckpt = os.path.join(log_dir, _a.ckpt)
runner.load(ckpt)
policy = runner.get_inference_policy(device=env_raw.device)

P(f"评估 {_a.run} / {_a.ckpt}")
P(f"环境：{'随机化' if _a.randomize else '固定'}，{_a.envs} 并行，目标 ≥{_a.episodes} 条 episode")
P(f"Gate A 判据：固定 ≥95%，随机 ≥85%\n")

n_done = n_succ = n_far = n_timeout = 0
open_at_end, steps_at_end = [], []
obs = env.get_observations()
if isinstance(obs, tuple):
    obs = obs[0]
max_steps = int(env_raw.max_episode_length) + 5

with torch.inference_mode():
    while n_done < _a.episodes:
        act = policy(obs)
        out = env.step(act)
        obs, dones = out[0], out[2]
        if isinstance(obs, tuple):
            obs = obs[0]
        if dones.any():
            idx = dones.nonzero(as_tuple=False).flatten()
            # 这些量必须用环境在终止时刻存下的快照，不能现读——
            # DirectRLEnv 已经在 step() 内部把终止的 env 重置了。
            succ = env_raw.success_buf[idx]
            far = env_raw._far_buf[idx]
            n_done += len(idx)
            n_succ += int(succ.sum())
            n_far += int((far & ~succ).sum())
            n_timeout += int((~succ & ~far).sum())
            open_at_end += env_raw.term_opening[idx].tolist()
            steps_at_end += env_raw.term_steps[idx].tolist()

rate = n_succ / max(n_done, 1)
thr = 0.85 if _a.randomize else 0.95
import statistics
P(f"完成 episode 数 = {n_done}")
P(f"  成功        {n_succ:5d}  ({rate*100:5.1f}%)")
P(f"  飞出边界    {n_far:5d}  ({n_far/max(n_done,1)*100:5.1f}%)")
P(f"  超时        {n_timeout:5d}  ({n_timeout/max(n_done,1)*100:5.1f}%)")
P(f"\n终止时开度  均值 {statistics.mean(open_at_end)*1000:.1f} mm  "
  f"中位数 {statistics.median(open_at_end)*1000:.1f} mm  "
  f"(任务区间 {cfg.goal_range[0]*1000:.0f}-{cfg.goal_range[1]*1000:.0f})")
P(f"终止时步数  均值 {statistics.mean(steps_at_end):.1f}  "
  f"中位数 {statistics.median(steps_at_end):.0f}  (上限 {env_raw.max_episode_length})")
P(f"\nGate A ({'随机' if _a.randomize else '固定'}环境，门槛 {thr*100:.0f}%)："
  f"{'通过' if rate >= thr else '未通过'}  实测 {rate*100:.1f}%")
log.close()
os._exit(0 if rate >= thr else 1)
