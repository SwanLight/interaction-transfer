"""S2：录制训练好的策略在跑什么（`plan/06` §7 的人工检查）。

**成功率 100% 不等于做对了。** S1 出现过多次假 PASS——数字对、画面一看就
不对（推子被底座卡住却报「推不动」）。策略更有可能找到物理漏洞：
穿模、抖动、利用求解器瑕疵。这些只能看出来。

用法::

    PYTHONPATH=src /isaac-sim/python.sh tools/s2_record.py \\
        --run expert_drawer_hook_v2 --ckpt model_100.pt --out /tmp/s2vid
"""
import argparse, math, os, sys

_ap = argparse.ArgumentParser()
_ap.add_argument("--task", default="drawer_hook")
_ap.add_argument("--run", required=True)
_ap.add_argument("--ckpt", default="model_100.pt")
_ap.add_argument("--out", default="/tmp/s2vid")
_ap.add_argument("--envs", type=int, default=4)
_ap.add_argument("--seconds", type=float, default=6.0)
_ap.add_argument("--randomize", action="store_true")
_ap.add_argument("--width", type=int, default=960)
_ap.add_argument("--height", type=int, default=540)
_ap.add_argument("--fps", type=int, default=30)
_ap.add_argument("--runs_dir", default="/workspace/interaction_transfer/runs")
_a, _ = _ap.parse_known_args()

from isaaclab.app import AppLauncher
_app = AppLauncher(headless=True, enable_cameras=True).app

import numpy as np, torch
import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils import configclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from it.envs.drawer import DrawerEnv, DrawerEnvCfg
from it.build_assets import CabinetCfg

from it.viz import look_at_quat            # 相机位姿必须静态给定（P-24）

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

C = CabinetCfg()
EYE = (0.62, -0.52, 0.36)
TGT = (-0.02, 0.0, 0.09)

EnvCls, CfgCls = TASKS[_a.task]
cfg = CfgCls()
cfg.scene.num_envs = _a.envs
cfg.scene.env_spacing = 1.6
cfg.randomize = _a.randomize
cfg.cam = CameraCfg(
    prim_path="/World/envs/env_.*/Cam", update_period=0.0,
    height=_a.height, width=_a.width, data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(focal_length=20.0, clipping_range=(0.03, 30.0)),
    offset=CameraCfg.OffsetCfg(pos=EYE, rot=look_at_quat(EYE, TGT), convention="opengl"))

env_raw = EnvCls(cfg)
# 相机不在 env 的 _setup_scene 里，单独挂上去
cam = Camera(cfg.cam)
env_raw.scene.sensors["cam"] = cam
env_raw.sim.reset()

env = RslRlVecEnvWrapper(env_raw)
log_dir = os.path.join(_a.runs_dir, _a.run)
runner = OnPolicyRunner(env, TRAIN_CFG, log_dir=log_dir, device=str(env_raw.device))
runner.load(os.path.join(log_dir, _a.ckpt))
policy = runner.get_inference_policy(device=env_raw.device)

dt = cfg.sim.dt * cfg.decimation
n_ctrl = int(_a.seconds / dt)
every = max(1, int(round(1.0 / (_a.fps * dt))))
frames = []
succ_at = [None] * _a.envs

obs = env.get_observations()
if isinstance(obs, tuple):
    obs = obs[0]

with torch.inference_mode():
    for i in range(n_ctrl):
        act = policy(obs)
        out = env.step(act)
        obs = out[0]
        if isinstance(obs, tuple):
            obs = obs[0]
        for e in range(_a.envs):
            if succ_at[e] is None and bool(env_raw.success_buf[e]):
                succ_at[e] = i
        if i % every == 0:
            env_raw.sim.render()
            cam.update(dt)
            frames.append(cam.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8))

os.makedirs(_a.out, exist_ok=True)
tag = f"expert_{_a.task}{'_rand' if _a.randomize else ''}"
path = os.path.join(_a.out, f"{tag}.mp4")
import imageio.v2 as iio
with iio.get_writer(path, fps=_a.fps, codec="libx264", quality=8, macro_block_size=1) as w:
    for f in frames:
        w.append_data(f)

op = env_raw.term_opening
print(f"WROTE {path} ({len(frames)} frames)", flush=True)
print(f"RESULT 成功 env: {sum(1 for x in succ_at if x is not None)}/{_a.envs}；"
      f"首次成功控制步 {[x for x in succ_at]}", flush=True)
print(f"RESULT 终止开度 {op.mean().item()*1000:.1f} mm；"
      f"峰值接触力观察请看 tensorboard diag/contact_force_max_N", flush=True)
sys.stdout.flush()
os._exit(0)
