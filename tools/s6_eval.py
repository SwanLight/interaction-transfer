#!/usr/bin/env python3
"""Deterministic first-episode evaluation for an S6 checkpoint.

Training logs contain exploration noise and mix asynchronous auto-resets.  This tool loads one
checkpoint, uses the actor mean, and reports only each environment's first episode so model_0 and
the trained checkpoint can be compared on exactly the same distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executor", default="padrod")
    parser.add_argument("--object", default="block")
    parser.add_argument("--primitive", default="press")
    parser.add_argument("--run", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--runs-root", type=Path, default=Path("runs/s6_corrected"))
    parser.add_argument("--probe-root", type=Path, default=Path("/tmp/s6/probe"))
    parser.add_argument("--task-root", type=Path, default=Path("/tmp/s5"))
    parser.add_argument("--smoke", type=Path, default=None)
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--scale", type=Path, default=Path("/tmp/s6/reward_probe.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from isaaclab.app import AppLauncher
    simulation_app = AppLauncher(headless=True).app

    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    import rsl_rl.runners.on_policy_runner as opr

    from it.ei_policy import InteractionActorCritic
    from it.envs.interaction import InteractionEnv, probe_env_cfg
    from it.s6_gate import implementation_fingerprint, smoke_path
    from s6_train import HELD_OUT, PPO_CFG, resolve

    torch.manual_seed(args.seed)
    scale = json.loads(args.scale.read_text(encoding="utf-8"))["scale"]
    command = resolve("a", args.object, args.primitive,
                      probe_root=args.probe_root, task_root=args.task_root)[0]
    cfg = probe_env_cfg(args.object, args.executor, num_envs=args.envs,
                        command_glob=command, forbid=HELD_OUT[args.executor],
                        reward_scale=scale)
    cfg.force_scale = min(cfg.force_scale, 6.0)
    smoke_file = (args.smoke if args.smoke is not None
                  else smoke_path(args.executor, args.object, args.primitive))
    if smoke_file.exists():
        smoke = json.loads(smoke_file.read_text(encoding="utf-8"))
        if smoke.get("suggested_fail_penalty_per_step") is not None:
            cfg.fail_penalty_per_step = float(smoke["suggested_fail_penalty_per_step"])
    cfg.fail_penalty_gamma = PPO_CFG["algorithm"]["gamma"]

    env_raw = InteractionEnv(cfg=cfg)
    wrapped = RslRlVecEnvWrapper(env_raw, clip_actions=1.0)
    opr.InteractionActorCritic = InteractionActorCritic
    train_cfg = json.loads(json.dumps(PPO_CFG))
    train_cfg["policy"]["bank"] = env_raw.bank
    run_dir = args.runs_root / args.run
    checkpoint = run_dir / args.checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    # 评估不能往训练 run 目录追加事件文件；checkpoint 只通过显式路径读取。
    runner = OnPolicyRunner(wrapped, train_cfg,
                            log_dir=f"/tmp/s6_eval_{args.checkpoint.replace('.', '_')}",
                            device=str(env_raw.device))
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=env_raw.device)

    # 用 raw DirectRLEnv rollout，保留 terminated 与 truncated 的区别；rsl_rl wrapper
    # 会把二者合成 dones，曾导致 300 步 timeout 被误报成任务完成。
    obs, _ = env_raw.reset()
    alive = torch.ones(args.envs, dtype=torch.bool, device=env_raw.device)
    sums = {name: 0.0 for name in (
        "reward", "touching", "contact_required", "inside_region_share",
        "anchor_distance", "region", "mode", "mech", "effect", "safety",
        "action_abs")}
    counts = {name: 0 for name in sums}
    required_touch = 0.0
    required_count = 0
    finished = torch.zeros_like(alive)
    failed = torch.zeros_like(alive)
    timed_out = torch.zeros_like(alive)

    def add(name: str, value: torch.Tensor, mask: torch.Tensor) -> None:
        nonlocal sums, counts
        sums[name] += float(value[mask].sum())
        counts[name] += int(mask.sum())

    with torch.inference_mode():
        for _step in range(args.steps):
            if not bool(alive.any()):
                break
            action = policy(obs)
            obs, reward, terminated, truncated, _extras = env_raw.step(action)
            terminated, truncated = terminated.bool(), truncated.bool()
            done = terminated | truncated
            mask = alive.clone()
            diag = env_raw._diag
            terms = env_raw._terms
            add("reward", reward, mask)
            add("touching", diag["diag/touching"], mask)
            add("contact_required", diag["diag/contact_required"], mask)
            add("inside_region_share", diag["diag/inside_region_share"], mask)
            add("anchor_distance", diag["diag/anchor_distance"], mask)
            add("region", terms.region, mask)
            add("mode", terms.mode, mask)
            add("mech", terms.mech, mask)
            add("effect", terms.effect, mask)
            add("safety", terms.safety, mask)
            add("action_abs", action.abs().mean(-1), mask)
            required = mask & diag["diag/contact_required"].bool()
            required_touch += float(diag["diag/touching"][required].sum())
            required_count += int(required.sum())
            failed_now = done & env_raw._failed.bool()
            failed |= failed_now
            finished |= terminated & ~failed_now
            timed_out |= truncated
            alive &= ~done

    mean = {name: sums[name] / max(counts[name], 1) for name in sums}
    result = {
        "meta": {
            "executor": args.executor, "object": args.object,
            "primitive": args.primitive, "run": args.run,
            "checkpoint": args.checkpoint, "seed": args.seed,
            "envs": args.envs, "max_steps": args.steps,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "implementation_fingerprint": implementation_fingerprint(),
        },
        "mean": mean,
        "touching_when_contact_required": required_touch / max(required_count, 1),
        "finished_share": float(finished.float().mean()),
        "failed_share": float(failed.float().mean()),
        "timed_out_share": float(timed_out.float().mean()),
        "unfinished_share": float(alive.float().mean()),
        "evaluated_env_steps": counts["reward"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    env_raw.close(); simulation_app.close()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(1)
