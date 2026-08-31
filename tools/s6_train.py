#!/usr/bin/env python3
"""E-I 交互跟踪执行器的训练入口（`plan/04` §5 / §9 / §10）。

**课程分三段，留出任务在任何一段都不得出现**（`plan/04` §5.4，写进 dataloader 的
断言里而不是靠自觉——`CommandBank(forbid=...)` 在入口就报错）：

======  ==================================================================
``a``   单指令跟踪。固定一份 artifact、固定一个物体，确认跟踪 reward 可学、
        各项量级合理。**这一段过不了就不要往下走**（`plan/04` §13 第 3 条）
``b``   预训练物体集。43 份 (物体, 原语) 指令随机采样，这是"摸索自己形态能力"
        的主体阶段
``c``   加入该执行器的训练任务 envelope（`plan/04` §5.4 的划分表）
======  ==================================================================

开训前的两道闸门，缺一不可：

1. ``tools/s6_reward_probe.py`` —— 各项 reward 的量级标定（D-31 第 2 个洞）
   与"reward 分不分得开指令"（AUC）；
2. ``tools/s6_smoke.py`` —— **接触真的建立得起来吗**。`--dry-run` 分不出
   "随机策略没碰到"和"环境根本测不到接触"：2026-08-30 那次退出码 0、
   `region/mode/mech` 三项恒为 0，被读成了"量级合理"。

用法（服务器上，一张卡一条，P-29 必须 pin 单卡）::

    IT_GPU=0 IT_WAIT=0 ./tools/run_remote.sh \\
        "PYTHONPATH=src /isaac-sim/python.sh tools/s6_train.py --stage a \\
         --executor padrod --object block --primitive press \\
         --iterations 600 --envs 1024" s6a_block_press
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

#: 每个执行器的**留出任务**（`plan/04` §5.4）。这些 token 会被塞进
#: `CommandBank(forbid=...)`，命中即抛异常——留出任务的指令与 reward 在
#: **任何**训练阶段都不得出现，包括 curriculum 与调试。
HELD_OUT = {
    "allegro": ("wipe", "board"),
    "padrod": ("wipe", "board", "drawer", "knob"),   # 仅预训练集
    "gripper": ("wipe", "board"),
    "hook": ("wipe", "board", "drawer", "knob"),     # 仅预训练集
}
#: 每段课程用哪些 artifact。``{obj}`` / ``{prim}`` 由命令行填。
STAGES = {
    "a": ["{probe_root}/probe_{obj}-{obj}-nominal-{prim}-train.npz"],
    "b": ["{probe_root}/*.npz"],
    "c": ["{probe_root}/*.npz", "{task_root}/*/*-train.npz"],
}

#: PPO 超参，`plan/04` §9。两项 §9 没规定的写在下面，理由见 `log/decisions.md`。
PPO_CFG = {
    "num_steps_per_env": 32,            # §9 的 horizon
    "save_interval": 50,
    "obs_groups": {"policy": ["policy"], "critic": ["critic"]},
    "policy": {
        "class_name": "InteractionActorCritic",
        # 观测里有 command_index / bin_index 两个整数下标，归一化会把它们抹掉。
        "actor_obs_normalization": False,
        "critic_obs_normalization": False,
        # §9 没规定。动作已在环境入口 clamp 到 [-1,1]，std=1.0 等于接近均匀采样，
        # 9 维增量动作上 std=0.8 会形成 Brownian random walk：位置每步约 8 mm，
        # 300 步 RMS 约 14 cm，已经大于整个接触工作区。课程 A 用 0.25，从局部接触
        # 邻域探索；log_std 仍由 PPO 学习，不把它固定死。
        "init_noise_std": 0.20,
    },
    "algorithm": {
        "class_name": "PPO",
        "num_learning_epochs": 5,       # §9
        "num_mini_batches": 4,
        "clip_param": 0.2,              # §9
        "gamma": 0.99,                  # §9
        "lam": 0.95,                    # §9
        "value_loss_coef": 1.0,
        # §9 没规定。跟踪 reward 是 hinge，集合内恰好为零，梯度在集合内消失，
        # 需要一点熵维持探索；0.01 会让 9 维动作的噪声压不下去（实测另记）。
        "entropy_coef": 0.001,
        "learning_rate": 3.0e-4,        # §9
        "schedule": "adaptive",         # §9 的 adaptive KL
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
    },
}


def resolve(stage: str, obj: str, prim: str, *,
            probe_root: str | Path = "/tmp/s6/probe",
            task_root: str | Path = "/tmp/s5") -> list[str]:
    paths: list[str] = []
    for pattern in STAGES[stage]:
        paths.extend(sorted(glob.glob(pattern.format(
            obj=obj, prim=prim, probe_root=probe_root, task_root=task_root))))
    if not paths:
        raise SystemExit(
            f"课程 {stage} 没有匹配到任何 artifact："
            f"{[p.format(obj=obj, prim=prim, probe_root=probe_root, task_root=task_root) for p in STAGES[stage]]}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--executor", choices=sorted(HELD_OUT), default="padrod")
    parser.add_argument("--object", default="block")
    parser.add_argument("--primitive", default="press",
                        help="课程 a 用哪一条原语的指令")
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run", default=None)
    parser.add_argument("--runs-root", type=Path, default=Path("runs/s6"))
    parser.add_argument("--probe-root", type=Path, default=Path("/tmp/s6/probe"))
    parser.add_argument("--task-root", type=Path, default=Path("/tmp/s5"))
    parser.add_argument("--resume-checkpoint", type=Path, default=None,
                        help="从显式 checkpoint 续训。路径必须存在；会同时恢复 PPO "
                             "optimizer/iteration，日志仍写到本次 --run，避免覆盖来源 run")
    parser.add_argument("--scale", type=Path,
                        default=Path("/tmp/s6/reward_probe.json"),
                        help="reward 各项的量级，由 tools/s6_reward_probe.py 标定。"
                             "**不给标定就不许开训**（D-31 第 2 个洞）")
    parser.add_argument("--smoke", type=Path, default=None,
                        help="与 executor/object/primitive 绑定的 tools/s6_smoke.py JSON。"
                             "默认自动取 /tmp/s6/smoke_<executor>_<object>_<primitive>.json")
    parser.add_argument("--allow-without-smoke", action="store_true",
                        help="跳过冒烟闸门。只用于调试，会在日志里显式记一行")
    parser.add_argument("--dry-run", action="store_true",
                        help="只装配、跑几步、打印各项量级，不训练")
    args = parser.parse_args()

    from it.s6_gate import implementation_fingerprint, smoke_path
    if args.smoke is None:
        args.smoke = smoke_path(args.executor, args.object, args.primitive)
    if args.allow_without_smoke and not args.dry_run:
        raise SystemExit("--allow-without-smoke 只允许 --dry-run；真实训练没有绕过闸门的入口")

    if not args.scale.exists():
        raise SystemExit(
            f"找不到 {args.scale}。reward 的权重必须在成功示教上标定，不许拍——"
            "先跑 tools/s6_reward_probe.py（D-31 第 2 个洞：量纲失衡时最优解是不动）")
    scale = json.loads(args.scale.read_text(encoding="utf-8"))["scale"]

    fail_penalty = None
    if args.smoke.exists():
        smoke = json.loads(args.smoke.read_text(encoding="utf-8"))
        if smoke.get("failed") and not args.allow_without_smoke:
            raise SystemExit(
                f"{args.smoke} 记着冒烟未通过：{smoke['failed']}。"
                "接触建立不起来时训练曲线照样会涨，学到的只是'别乱动'——"
                "先修环境，不要先开训（--allow-without-smoke 可强行跳过）")
        expected = {"object": args.object, "executor": args.executor,
                    "primitive": args.primitive,
                    "implementation_fingerprint": implementation_fingerprint()}
        actual = smoke.get("meta", {})
        mismatch = {key: (actual.get(key), value) for key, value in expected.items()
                    if actual.get(key) != value}
        if mismatch and not args.allow_without_smoke:
            raise SystemExit(
                f"{args.smoke} 不是当前组合/当前实现的闸门：{mismatch}。"
                "每个组合必须在改完代码后单独重跑 smoke，旧报告不能复用")
        fail_penalty = smoke.get("suggested_fail_penalty_per_step")
    elif not args.allow_without_smoke:
        raise SystemExit(
            f"找不到 {args.smoke}。先跑 tools/s6_smoke.py——`--dry-run` 分不出"
            "'随机策略没碰到'和'环境根本测不到接触'（2026-08-30 那次就是后者）")

    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=True)
    simulation_app = launcher.app

    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    import rsl_rl.runners.on_policy_runner as opr

    from it.ei_policy import InteractionActorCritic
    from it.envs.interaction import InteractionEnv, probe_env_cfg

    # ⚠️ P-19 要求脚本末尾用 os._exit()（SimulationApp.close() 会挂起、进程变僵尸
    # 占显存），**但 os._exit 不刷 stdout 缓冲**——非 tty 下 print 全进缓冲，
    # 日志里只剩 Isaac 自己的输出而退出码还是 0，看起来像"跑了但什么都没做"（P-73）。
    say = lambda *a: print(*a, flush=True)  # noqa: E731

    paths = resolve(args.stage, args.object, args.primitive,
                    probe_root=args.probe_root, task_root=args.task_root)
    glob_pattern = paths[0]
    cfg = probe_env_cfg(args.object, args.executor, num_envs=args.envs,
                        command_glob=glob_pattern,
                        forbid=HELD_OUT[args.executor], reward_scale=scale)
    cfg.command_paths = tuple(paths)
    if args.stage == "a":
        # 课程 A 是接触解码 canary，不拿 25 N 的全量程噪声轰一个约 3 N 的静态指令。
        # 后续课程按指令集实测的力范围逐级放宽；安全上限仍由 max_contact_force 独立约束。
        cfg.force_scale = min(cfg.force_scale, 6.0)
    if fail_penalty is not None:
        cfg.fail_penalty_per_step = float(fail_penalty)
    cfg.fail_penalty_gamma = PPO_CFG["algorithm"]["gamma"]

    env = InteractionEnv(cfg=cfg)
    torch.manual_seed(args.seed)
    say(f"[S6] 课程 {args.stage} / 执行器 {args.executor} / 物体 {args.object}")
    say(f"[S6] 指令 {len(env.bank)} 份，留出禁令 {cfg.forbid}")
    say(f"[S6] reward scale {scale}")
    say(f"[S6] 失败终止惩罚 {cfg.fail_penalty_per_step}/剩余步（γ={cfg.fail_penalty_gamma}）")

    if args.dry_run:
        obs, _ = env.reset()
        assert obs["policy"].shape[-1] == cfg.observation_space, (
            f"观测维度 {obs['policy'].shape[-1]} 与 cfg 的 "
            f"{cfg.observation_space} 对不上")
        action = torch.zeros(env.num_envs, cfg.action_space, device=env.device)
        for step in range(8):
            obs, reward, done, timeout, extras = env.step(action.uniform_(-1.0, 1.0))
            log = extras.get("log", {})
            say(f"  step {step}: reward {float(reward.mean()):+.4f}  "
                + "  ".join(f"{k.split('/')[-1]} {float(v):+.4f}"
                            for k, v in sorted(log.items())))
        say("[S6] dry-run 结束。⚠️ dry-run **不能**当接触验证：随机策略碰不到东西，"
            "与环境测不到接触长得一模一样。那一条看 tools/s6_smoke.py。")
        env.close(); simulation_app.close()
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(0)

    # rsl_rl 的 `_construct_algorithm` 用 `eval(policy_cfg["class_name"])` 在**它自己的
    # 模块命名空间**里解析策略类，所以自定义类必须先注进去。这是 rsl_rl 3.x 唯一的
    # 扩展点（Isaac Lab 的 `RslRlPpoActorCriticCfg.class_name` 也是这么用的）。
    opr.InteractionActorCritic = InteractionActorCritic

    run = args.run or (f"{args.stage}_{args.executor}_{args.object}_"
                       f"{args.primitive}_s{args.seed}")
    log_dir = str(args.runs_root / run)
    train_cfg = json.loads(json.dumps(PPO_CFG))       # 深拷贝：runner 会 pop 掉键
    train_cfg["policy"]["bank"] = env.bank
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    runner = OnPolicyRunner(wrapped, train_cfg, log_dir=log_dir, device=str(env.device))
    if args.resume_checkpoint is not None:
        if not args.resume_checkpoint.exists():
            raise FileNotFoundError(f"续训 checkpoint 不存在：{args.resume_checkpoint}")
        runner.load(str(args.resume_checkpoint))
        say(f"[S6] 续训来源 {args.resume_checkpoint}")
    say(f"[S6] 策略参数 {sum(p.numel() for p in runner.alg.policy.parameters()):,}")
    say(f"[S6] 日志 {log_dir}")
    say(f"[S6] 迭代 {args.iterations} × {PPO_CFG['num_steps_per_env']} 步 × "
        f"{args.envs} env = {args.iterations * PPO_CFG['num_steps_per_env'] * args.envs:,} 帧")
    # `extras["log"]` 的每一项都会被 runner 写进 TensorBoard（含 "/" 的 key 直接
    # add_scalar），这正是 P-27 要求的"每一项 reward 都要有分项记录"。
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
    say("[S6] 训练结束")

    env.close()
    simulation_app.close()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    # ⚠️ P-74：创建过 SimulationApp 之后，**未捕获异常也会以退出码 0 收场**。
    # 训练脚本尤其要自己兜住：一次崩掉的训练报 PASS，比崩掉本身更贵。
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(1)
