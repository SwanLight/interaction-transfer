#!/usr/bin/env python3
"""E-I 交互跟踪执行器的训练入口（`plan/04` §5 / §10）。

**课程分三段，留出任务在任何一段都不得出现**（`plan/04` §5.4，写进 dataloader 的
断言里而不是靠自觉——`CommandBank(forbid=...)` 在入口就报错）：

======  ==================================================================
``a``   单指令跟踪。固定一份 artifact、固定一个物体，确认跟踪 reward 可学、
        各项量级合理。**这一段过不了就不要往下走**（`plan/04` §13 第 3 条）
``b``   预训练物体集。43 份 (物体, 原语) 指令随机采样，这是"摸索自己形态能力"
        的主体阶段
``c``   加入该执行器的训练任务 envelope（`plan/04` §5.4 的划分表）
======  ==================================================================

用法（服务器上）::

    IT_PY=/isaac-sim/python.sh ./tools/run_remote.sh \\
        "/isaac-sim/python.sh tools/s6_train.py --stage a --executor padrod \\
         --iterations 300 --envs 1024" s6a
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

#: 每个执行器的**留出任务**（`plan/04` §5.4）。这些 token 会被塞进
#: `CommandBank(forbid=...)`，命中即抛异常——留出任务的指令与 reward 在
#: **任何**训练阶段都不得出现，包括 curriculum 与调试。
HELD_OUT = {
    "allegro": ("wipe", "board"),
    "padrod": ("wipe", "board", "drawer", "knob"),   # 仅预训练集
    "gripper": ("wipe", "board"),
    "hook": ("wipe", "board", "drawer", "knob"),     # 仅预训练集
}
#: 每段课程用哪些 artifact。
STAGES = {
    "a": ["/tmp/s6/probe/probe_block-block-nominal-press-train.npz"],
    "b": ["/tmp/s6/probe/*.npz"],
    "c": ["/tmp/s6/probe/*.npz", "/tmp/s5/*/*-train.npz"],
}


def resolve(stage: str, executor: str) -> list[str]:
    paths: list[str] = []
    for pattern in STAGES[stage]:
        paths.extend(sorted(glob.glob(pattern)))
    if not paths:
        raise SystemExit(f"课程 {stage} 没有匹配到任何 artifact：{STAGES[stage]}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--executor", choices=sorted(HELD_OUT), default="padrod")
    parser.add_argument("--object", default="block")
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run", default=None)
    parser.add_argument("--scale", type=Path,
                        default=Path("/tmp/s6/reward_probe.json"),
                        help="reward 各项的量级，由 tools/s6_reward_probe.py 标定。"
                             "**不给标定就不许开训**（D-31 第 2 个洞）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只装配、跑几步、打印各项量级，不训练")
    args = parser.parse_args()

    if not args.scale.exists():
        raise SystemExit(
            f"找不到 {args.scale}。reward 的权重必须在成功示教上标定，不许拍——"
            "先跑 tools/s6_reward_probe.py（D-31 第 2 个洞：量纲失衡时最优解是不动）")
    scale = json.loads(args.scale.read_text(encoding="utf-8"))["scale"]

    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=True)
    simulation_app = launcher.app

    import torch
    from it import assets as A
    from it.envs.interaction import (InteractionEnv, InteractionEnvCfg,
                                     PRIVILEGED_DIM, PROPRIO_DIM)
    from it.ei_policy import InteractionActorCritic

    executors = {"padrod": A.PADROD_CFG, "hook": A.HOOK_CFG}
    objects = {"block": A.BLOCK_CFG, "column": A.COLUMN_CFG, "ball": A.BALL_CFG,
               "slab": A.SLAB_CFG, "ridge": A.RIDGE_CFG, "roller": A.ROLLER_CFG}
    if args.executor not in executors:
        raise SystemExit(f"{args.executor} 的资产还没接进来；现有 {sorted(executors)}")

    paths = resolve(args.stage, args.executor)
    cfg = InteractionEnvCfg()
    cfg.scene.num_envs = args.envs
    cfg.object_cfg = objects[args.object].replace(prim_path="/World/envs/env_.*/Object")
    cfg.executor_cfg = executors[args.executor].replace(
        prim_path="/World/envs/env_.*/Executor")
    cfg.command_glob = paths[0] if len(paths) == 1 else "/tmp/s6/probe/*.npz"
    cfg.forbid = HELD_OUT[args.executor]

    env = InteractionEnv(cfg=cfg)
    torch.manual_seed(args.seed)
    policy = InteractionActorCritic(env.bank, action_dim=cfg.action_space,
                                    proprio_dim=PROPRIO_DIM,
                                    privileged_dim=PRIVILEGED_DIM).to(env.device)
    print(f"[S6] 课程 {args.stage} / 执行器 {args.executor} / 物体 {args.object}")
    print(f"[S6] 指令 {len(env.bank)} 份，留出禁令 {cfg.forbid}")
    print(f"[S6] 策略参数 {sum(p.numel() for p in policy.parameters()):,}")
    print(f"[S6] reward scale {scale}")

    obs, _ = env.reset()
    assert obs["policy"].shape[-1] == PROPRIO_DIM + 6, (
        f"观测维度 {obs['policy'].shape[-1]} 与 PROPRIO_DIM+6={PROPRIO_DIM + 6} 对不上")
    for step in range(8 if args.dry_run else 0):
        with torch.no_grad():
            action = policy.act(obs["policy"])
        obs, reward, done, timeout, extras = env.step(action)
        log = extras.get("log", {})
        print(f"  step {step}: reward {float(reward.mean()):+.4f}  "
              + "  ".join(f"{k.split('/')[-1]} {float(v):+.4f}" for k, v in log.items()))
    if args.dry_run:
        print("[S6] dry-run 结束：装配通过，各项量级见上。")
        env.close()
        simulation_app.close()
        os._exit(0)

    raise SystemExit(
        "PPO 训练循环还没接上 rsl_rl。先用 --dry-run 确认环境与各项量级，"
        "再接 OnPolicyRunner——那一步要按 `plan/04` §9 的超参表配，"
        "且必须把 extras['log'] 的每一项都送进 TensorBoard（P-27）。")


if __name__ == "__main__":
    main()
