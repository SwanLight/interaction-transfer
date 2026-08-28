"""S2：Privileged Expert 训练（rsl_rl PPO）。

`plan/04` §9 的超参。Gate A（`plan/05` §10）：固定环境 ≥95%，随机环境 ≥85%。

Expert 只用于**性能上限与调试**，不用于证明任何科学结论（`plan/04` §3）。
它存在的意义是：后面 representation 实验失败时，能确定是「表示信息不够」
而不是「机器人本来就不会做」。

用法::

    PYTHONPATH=src /isaac-sim/python.sh tools/s2_train.py --task drawer_hook \\
        --envs 2048 --iters 1500 --run expert_drawer_hook
"""
import argparse, os, sys

_ap = argparse.ArgumentParser()
_ap.add_argument("--task", default="drawer_hook")
_ap.add_argument("--envs", type=int, default=2048)
_ap.add_argument("--iters", type=int, default=1500)
_ap.add_argument("--run", default=None)
_ap.add_argument("--seed", type=int, default=0)
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

# plan/04 §9 起始超参
TRAIN_CFG = {
    "algorithm": {
        "class_name": "PPO",
        "clip_param": 0.2, "desired_kl": 0.01, "entropy_coef": 0.006,
        "gamma": 0.99, "lam": 0.95, "learning_rate": 3.0e-4,
        "max_grad_norm": 1.0, "num_learning_epochs": 5,
        "num_mini_batches": 4, "schedule": "adaptive",
        "use_clipped_value_loss": True, "value_loss_coef": 1.0,
    },
    "policy": {
        "class_name": "ActorCritic",
        "activation": "elu",
        "actor_hidden_dims": [256, 256, 128],
        "critic_hidden_dims": [256, 256, 128],
        "init_noise_std": 1.0,
    },
    # rsl-rl 3.x 需要显式声明观测分组。S2 的 Expert 是特权策略，
    # actor 和 critic 看同一组上帝视角观测（plan/04 §3）。
    # 到了 S8 的 E-T / S6 的 E-I 才启用非对称 actor-critic（plan/04 §7）：
    # actor 只看部署允许的信息，critic 额外看物理参数。
    "obs_groups": {"policy": ["policy"], "critic": ["policy"]},
    "num_steps_per_env": 32,        # plan/04 §9 horizon
    "save_interval": 100,
    "empirical_normalization": True,
    "logger": "tensorboard",
    "max_iterations": _a.iters,
}


def main():
    EnvCls, CfgCls = TASKS[_a.task]
    cfg = CfgCls()
    cfg.scene.num_envs = _a.envs
    cfg.randomize = _a.randomize
    cfg.seed = _a.seed

    env = EnvCls(cfg)
    env = RslRlVecEnvWrapper(env)

    run = _a.run or f"{_a.task}_s{_a.seed}{'_rand' if _a.randomize else ''}"
    log_dir = os.path.join(_a.out, run)
    os.makedirs(log_dir, exist_ok=True)

    tc = dict(TRAIN_CFG)
    tc["max_iterations"] = _a.iters
    runner = OnPolicyRunner(env, tc, log_dir=log_dir, device=str(env.unwrapped.device))
    print(f"START run={run} envs={_a.envs} iters={_a.iters} randomize={_a.randomize}", flush=True)
    runner.learn(num_learning_iterations=_a.iters)
    runner.save(os.path.join(log_dir, "model_final.pt"))
    print(f"DONE saved {log_dir}/model_final.pt", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
