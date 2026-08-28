"""诊断：接触**具体发生在钩杆的哪个部位、把手的哪个部位**。

用户观察：「勾的是把手侧面凸出来的那一小段，而不是正中心那片区域，
而且用的是长杆子那一端，不是更短的钩子」。

不靠看画面猜。把接触点分别转到钩杆本体系和抽屉系，按几何分区归类。

钩杆本体系（原点=主杆中心）：
    主杆   半径 8，沿 Z，z ∈ [-125, +125] mm
    横钩   半径 8，沿 X，x ∈ [0, 50]，位于 z = -117 mm  ← 真正的「钩子」
    主杆下端 z = -125，**比横钩还低 8 mm**

把手（抽屉系）：
    横杆   半径 11，沿 Y，y ∈ [-70, +70]，x = 74，z = 90 mm
    支撑柱 半径 8，沿 X，位于 y = ±62.5，x ∈ [18, 74] mm
    中央可用段 |y| < 54 mm；柱外伸出段 62.5 < |y| < 70 mm
"""
import argparse, os, sys

_ap = argparse.ArgumentParser()
_ap.add_argument("--run", required=True)
_ap.add_argument("--ckpt", default="model_100.pt")
_ap.add_argument("--envs", type=int, default=16)
_ap.add_argument("--steps", type=int, default=200)
_ap.add_argument("--runs_dir", default="/workspace/interaction_transfer/runs")
_a, _ = _ap.parse_known_args()

from isaaclab.app import AppLauncher
_app = AppLauncher(headless=True).app

import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from it.envs.drawer import DrawerEnv, DrawerEnvCfg
from it.contact_utils import extract_contact_points, to_object_frame
from it.build_assets import CabinetCfg, HookCfg

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

C, H = CabinetCfg(), HookCfg()
log = open("/tmp/s2_probe_where.txt", "w", buffering=1)
def P(*x): print(*x, file=log)

cfg = DrawerEnvCfg()
cfg.scene.num_envs = _a.envs
cfg.disable_termination = True
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

MM = 1000.0
hook_bins = {"横钩(钩子本身)": 0, "主杆下端(<-117)": 0, "主杆中下段": 0,
             "主杆中上段": 0, "其他": 0}
handle_bins = {"横杆中央 |y|<54": 0, "横杆柱间过渡 54-62.5": 0,
               "横杆柱外伸出 62.5-70": 0, "支撑柱": 0, "非把手(面板/托盘)": 0}
force_by_hook = {k: 0.0 for k in hook_bins}
force_by_handle = {k: 0.0 for k in handle_bins}
n_pts = 0

drawer_bi = env_raw.cabinet.body_names.index("Drawer")

with torch.inference_mode():
    for i in range(_a.steps):
        act = policy(obs)
        out = env.step(act)
        obs = out[0]
        if isinstance(obs, tuple):
            obs = obs[0]
        cps = extract_contact_points(env_raw.contact, cfg.sim.dt)
        for e, c in enumerate(cps):
            if c.is_empty():
                continue
            # -> 钩杆本体系
            hp, _ = to_object_frame(c.positions, c.normals,
                                    env_raw.executor.data.root_pos_w[e],
                                    env_raw.executor.data.root_quat_w[e])
            # -> 抽屉系
            dp, _ = to_object_frame(c.positions, c.normals,
                                    env_raw.cabinet.data.body_pos_w[e, drawer_bi],
                                    env_raw.cabinet.data.body_quat_w[e, drawer_bi])
            for k in range(hp.shape[0]):
                f = c.normal_forces[k].abs().item()
                n_pts += 1
                x, y, z = (hp[k] * MM).tolist()
                if z < -109 and x > 8:
                    kb = "横钩(钩子本身)"
                elif z < -117:
                    kb = "主杆下端(<-117)"
                elif z < 0:
                    kb = "主杆中下段"
                elif z < 125:
                    kb = "主杆中上段"
                else:
                    kb = "其他"
                hook_bins[kb] += 1
                force_by_hook[kb] += f

                dx, dy, dz = (dp[k] * MM).tolist()
                ay = abs(dy)
                if abs(dz - C.panel_h / 2 * MM) < 20 and dx > 55:
                    if ay < 54:
                        hb = "横杆中央 |y|<54"
                    elif ay < 62.5:
                        hb = "横杆柱间过渡 54-62.5"
                    elif ay < 72:
                        hb = "横杆柱外伸出 62.5-70"
                    else:
                        hb = "非把手(面板/托盘)"
                elif 54 < ay < 72 and 15 < dx < 76:
                    hb = "支撑柱"
                else:
                    hb = "非把手(面板/托盘)"
                handle_bins[hb] += 1
                force_by_handle[hb] += f

P(f"统计样本：{_a.envs} env × {_a.steps} 控制步，共 {n_pts} 个接触点\n")
P("=" * 62)
P("接触落在**钩杆**的哪个部位")
P("=" * 62)
P(f"{'部位':<22}{'接触点数':>10}{'占比':>9}{'累计法向力(N)':>16}")
for k, v in hook_bins.items():
    pct = v / max(n_pts, 1) * 100
    P(f"{k:<22}{v:>10}{pct:>8.1f}%{force_by_hook[k]:>16.1f}")
P("")
P("=" * 62)
P("接触落在**把手**的哪个部位")
P("=" * 62)
P(f"{'部位':<24}{'接触点数':>10}{'占比':>9}{'累计法向力(N)':>16}")
for k, v in handle_bins.items():
    pct = v / max(n_pts, 1) * 100
    P(f"{k:<24}{v:>10}{pct:>8.1f}%{force_by_handle[k]:>16.1f}")
log.close()
os._exit(0)
