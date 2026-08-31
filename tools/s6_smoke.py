#!/usr/bin/env python3
"""E-I 环境的**接触冒烟**：不带策略，用脚本把执行器压到指令 region 上。

**为什么必须有这一条。** 2026-08-30 的 `--dry-run` 退出码是 0、四项 reward 都"在
0~1 之间"，被记成"装配通过，各项量级合理"。回头看那份日志（`/tmp/s6dry3.log`），
八个步里 ``region / mode / mech`` **三项恒等于 0.0000**——真实含义是**全程一次接触
都没有发生**（场景里没有台面、物体自由落体；`own_radius` 默认 0.08 又比垫头杆的
垫面到根 prim 的 0.108 m 还小，就算碰上了也会被当成 foreign 丢掉）。

一个随机策略"没碰到东西"和一个环境"根本测不到接触"，在 dry-run 里长得一模一样。
这个脚本把两者分开：**它主动去建立接触**，所以三项里任何一项还是零，就是环境的问题。

五节，每节都能失败
------------------

======  ==========================================================================
一      压在**允许区域内**：必须真的接触上，且 ≥90% 的接触力落在 ``region/allowed``
二      ``r_region`` 必须趋近 0（hinge：集合内恰好为零）
三      压到**允许区域外**：``r_region`` 必须显著变负。⭐ 这一节才是判别力所在——
        只做第一、二节的话，一个"永远返回 0"的实现也能通过
四      ``r_mech`` / ``r_mode`` 必须有限且非零，量级与 `reward_probe.json` 同数量级
五      悬停不动时的单步代价：用来标定 `InteractionEnvCfg.fail_penalty_per_step`
        （D-31 第 3 个洞：失败终止的惩罚要 ≥ 留下来干活的代价，否则早退划算）
======  ==========================================================================

⚠️ 判据是**退出码**，不是报告文件的开头（P-55）。

用法（服务器上）::

    IT_PY=/isaac-sim/python.sh ./tools/run_remote.sh \\
        "PYTHONPATH=src /isaac-sim/python.sh tools/s6_smoke.py \\
         --out /tmp/s6/smoke.txt" s6smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

_p = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
_p.add_argument("--object", default="block")
_p.add_argument("--executor", default="padrod")
_p.add_argument("--command", default="/tmp/s6/probe/probe_block-block-nominal-press-train.npz")
_p.add_argument("--envs", type=int, default=64)
_p.add_argument("--steps", type=int, default=120)
_p.add_argument("--press", type=float, default=3.0, help="法向压力设定值（N）")
_p.add_argument("--scale", type=Path, default=Path("/tmp/s6/reward_probe.json"))
_p.add_argument("--out", type=Path, default=None)
_a = _p.parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

_launcher = AppLauncher(headless=True)
simulation_app = _launcher.app

import torch  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402

from it import probe_scene as PS  # noqa: E402
from it.envs.interaction import InteractionEnv, probe_env_cfg  # noqa: E402
from it.float_ctrl import _quat_err  # noqa: E402
from it.s6_gate import implementation_fingerprint, smoke_path  # noqa: E402

if _a.out is None:
    primitive = Path(_a.command).stem.split("-")[-2]
    _a.out = smoke_path(_a.executor, _a.object, primitive).with_suffix(".txt")

say = lambda *a: print(*a, flush=True)  # noqa: E731  P-73：os._exit 不刷缓冲
LINES: list[str] = []
FAILED: list[str] = []


def report(text: str = ""):
    LINES.append(text)
    say(text)


def check(name: str, ok: bool, detail: str):
    mark = "PASS" if ok else "FAIL"
    report(f"  [{mark}] {name} —— {detail}")
    if not ok:
        FAILED.append(name)


def tip_world(env: InteractionEnv) -> torch.Tensor:
    """执行器末端在世界系的位置。压的是末端，不是根 prim。"""
    d = env.executor.data
    local = torch.tensor(PS.EXECUTOR_TIP[env.cfg.executor_name],
                         device=env.device, dtype=torch.float32).expand(env.num_envs, 3)
    return d.root_pos_w + quat_apply(d.root_quat_w, local)


def target_site(env: InteractionEnv, outside: bool):
    """要压的表面点（世界系）与它的外法向。

    ``outside=True`` 时挑一个 ``region/allowed`` 为假、且离允许区域最远的表面点——
    "压错地方" 必须让 ``r_region`` 显著变负，否则这个 reward 对区域不敏感。
    """
    cmd, b = env.tracker.command_index, env.tracker.bin_index
    points = env.bank.gather("surface/points_obj", cmd)            # (N,S,3)
    normals = env.bank.gather("surface/normals_obj", cmd)
    obj_pos, obj_quat = env._object_frame()
    if not outside:
        # **按当前命令格的 region 质心**，不是整条指令的锚点。
        # 扫掠类原语（rub / push / slide_push）的允许区域是随格移动的：
        # 整条指令的质心落在扫掠路径的中间，而任何**单独一格**的允许区域都在别处。
        # 实测 `slab/rub` 按整条锚点静压只有 0.065 落进允许区域，
        # 而 `ridge/press`（静态原语）有 0.571——差别不在环境，在这个脚本瞄哪。
        mass = env.bank.gather_bin("region/mass/mean", cmd, b)      # (N,S)
        total = mass.sum(-1, keepdim=True).clamp_min(1e-9)
        w = mass / total
        site = torch.einsum("ns,nsd->nd", w, points)
        nrm = torch.nn.functional.normalize(
            torch.einsum("ns,nsd->nd", w, normals), dim=-1)
        # 质心可能落在物体外部（曲面上的两片区域取平均）。投到最近的表面点上。
        near = (points - site[:, None, :]).norm(dim=-1).argmin(-1)
        rows = torch.arange(env.num_envs, device=env.device)
        site, nrm = points[rows, near], normals[rows, near]
    else:
        # 挑"同一个面上、离允许区域最远、但**够得到**"的点。
        # 只按距离挑会挑到底面——那一面被台面挡着，压不上去，测出来的是
        # "没碰到"而不是"压错了地方"（实测第一版 touching 只有 0.17）。
        allowed_any = env.bank._stack["region/allowed"][cmd].any(dim=1)   # (N,S)
        same_face = (normals * env.bank.anchor_normal[cmd][:, None, :]).sum(-1) > 0.9
        anchor = env.bank.anchor_pos[cmd][:, None, :]
        distance = (points - anchor).norm(dim=-1)
        distance = torch.where(allowed_any | ~same_face,
                               torch.full_like(distance, -1.0), distance)
        pick = distance.argmax(dim=-1)
        rows = torch.arange(env.num_envs, device=env.device)
        site, nrm = points[rows, pick], normals[rows, pick]
    world = obj_pos + quat_apply(obj_quat, site) + env.scene.env_origins
    return world, quat_apply(obj_quat, nrm)


def run_case(env: InteractionEnv, mode: str, steps: int, press: float) -> dict:
    """跑一段脚本控制，返回后半段的统计。

    ``mode``：``inside`` 压允许区域、``outside`` 压区域外、``hover`` 原地不动。
    """
    env.reset()
    n, dev = env.num_envs, env.device
    outside = mode == "outside"
    site_w, normal_w = target_site(env, outside)
    level_quat = InteractionEnv._quat_from_z(normal_w)
    retarget = mode == "inside"     # 跟着命令格走
    action = torch.zeros(n, env.cfg.action_space, device=dev)
    stats: list[dict] = []
    per_env: list[tuple] = []
    # DirectRLEnv 会在成功/失败后自动 reset。不同 case 的完成率不同，若继续把
    # reset 后的下一轮接近段混进均值，成功越快的 controller 反而越像“接触不稳”。
    # 所有判据只比较每个 env 的首个 episode；终止步本身仍计入。
    first_episode = torch.ones(n, dtype=torch.bool, device=dev)
    for step in range(steps):
        if mode == "hover":
            action.zero_()
        elif mode == "random":
            action.uniform_(-1.0, 1.0)
        else:
            if retarget and step % 5 == 0:
                site_w, normal_w = target_site(env, outside)
                level_quat = InteractionEnv._quat_from_z(normal_w)
            # 目标点稍微沉进表面 2 mm：位置干涉先压出实接触，力控才接得上（P-45）
            goal = site_w - normal_w * 0.002
            delta = goal - tip_world(env)
            # **接触之后法向不再走位置**：法向已经交给力控，位置误差却还在积分，
            # 目标会一路沉进台面里（`pos_limit` 允许 0.5 m）。一旦某帧接触断开，
            # 位置 PD 就按几十厘米的误差输出上百牛把杆砸下去 -> 弹跳 -> 实测
            # 滑移速率 0.22 m/s、`r_mode` −41.8。采集侧的做法就是法向力控、
            # 切向位置（`s3_source_probe` 的 `force_dir`），这里照做。
            live = env._live[:, None].float()
            tangential = delta - (delta * normal_w).sum(-1, keepdim=True) * normal_w
            delta = live * tangential + (1.0 - live) * delta
            action[:, :3] = (delta / env.act.pos_scale).clamp(-1.0, 1.0)
            # **姿态也要开环控住**：垫面必须与表面平行。初始站位带 ±0.15 rad 的
            # 抖动，不摆平的话垫子是用一条棱压上去的——接触点会散在整片顶面上，
            # 而 `region/allowed` 只有 4 个格。第一版漏了这一路，实测
            # inside_region_share = 0.000，看起来像归格错了，其实是压歪了。
            err = _quat_err(env.act.target_quat, level_quat)
            action[:, 3:6] = (err / env.act.rot_scale).clamp(-1.0, 1.0)
            action[:, 6:9] = -normal_w * (press / env.cfg.force_scale)
        _, reward, terminated, truncated, extras = env.step(action)
        if step >= steps // 2:                       # 只统计后半段（已经压稳）
            log = {k: float(v) for k, v in extras.get("log", {}).items()}
            log["reward"] = float(reward.mean())
            log["terminated"] = float(terminated.float().mean())
            stats.append(log)
            # **逐 env 也要留一份**：整体平均把"这一格压对了吗"和"压对了划不划算"
            # 混成一个数。前者是本脚本的开环控制水平，后者才是 reward 的性质。
            per_env.append((reward.detach().clone(),
                            env._diag["diag/inside_region_share"].clone(),
                            env._diag["diag/touching"].clone(),
                            env._diag["diag/contact_required"].clone(),
                            first_episode.clone()))
        first_episode &= ~(terminated | truncated)
    keys = sorted(stats[0])
    out = {k: sum(s[k] for s in stats) / len(stats) for k in keys}
    rewards = torch.stack([row[0] for row in per_env])
    shares = torch.stack([row[1] for row in per_env])
    touching = torch.stack([row[2] for row in per_env])
    required = torch.stack([row[3] for row in per_env])
    first_episode = torch.stack([row[4] for row in per_env])
    hit = shares > 0.8
    # 覆盖 extras 里的全 env 标量均值：后者不可剔除自动 reset 后的样本。
    out["reward"] = float(rewards[first_episode].mean())
    out["reward_when_region_satisfied"] = (
        float(rewards[hit & first_episode].mean())
        if bool((hit & first_episode).any()) else float("nan"))
    out["share_of_steps_region_satisfied"] = float(
        hit[first_episode].float().mean())
    out["share_of_steps_contact_required"] = float(
        required[first_episode].float().mean())
    required_first = required.bool() & first_episode
    out["touching_when_contact_required"] = float(
        touching[required_first].float().mean()) if bool(required_first.any()) else float("nan")
    return out


def debug_dump(env: InteractionEnv, tag: str):
    """env 0 的逐格实况。**猜不如看**：接触落在哪个格、那个格允不允许、
    允许的格又在哪，三件事摆在一起才知道是归格错了、还是压错了地方、还是格空。"""
    cmd, b = env.tracker.command_index, env.tracker.bin_index
    c = env._best
    cells = env.bank.gather("surface/points_obj", cmd)
    normals = env.bank.gather("surface/normals_obj", cmd)
    _force, index, _outward = env._force_on_object(c, cells, normals)
    allowed = env.bank.gather_bin("region/allowed", cmd, b)
    mass_mean = env.bank.gather_bin("region/mass/mean", cmd, b)
    report(f"  --- debug [{tag}] env0 ---")
    report(f"      bin {int(b[0])}/{env.bank.n_bins}  dwell {int(env.tracker.dwell[0])}  "
           f"allowed cells {int(allowed[0].sum())}  "
           f"bin region mass {float(mass_mean[0].sum()):.4f}")
    report(f"      整条指令的 allowed 格数（按 bin 求和 >0）"
           f" {int(env.bank._stack['region/allowed'][cmd[0]].any(0).sum())}")
    report(f"      anchor_obj {env.bank.anchor_pos[cmd[0]].tolist()}  "
           f"normal {env.bank.anchor_normal[cmd[0]].tolist()}")
    valid = c["valid"][0]
    report(f"      valid {int(valid.sum())}  fn {c['normal_force'][0][valid].tolist()}")
    report(f"      contact_obj {[[round(v, 4) for v in p] for p in c['pos'][0][valid].tolist()]}")
    report(f"      assigned cells {index[0][valid].tolist()}  "
           f"allowed? {allowed[0][index[0][valid]].tolist()}")
    live = allowed[0].nonzero().flatten()
    report(f"      allowed cell 坐标 "
           f"{[[round(v, 4) for v in p] for p in cells[0][live][:6].tolist()]}")
    report(f"      本格 mass 最大的 4 个格 "
           f"{mass_mean[0].topk(min(4, mass_mean.shape[-1])).indices.tolist()}")
    report("")


def main():
    scale = json.loads(_a.scale.read_text(encoding="utf-8"))["scale"] \
        if _a.scale.exists() else None
    if scale is None:
        raise SystemExit(f"找不到 {_a.scale}——先跑 tools/s6_reward_probe.py")

    cfg = probe_env_cfg(_a.object, _a.executor, num_envs=_a.envs,
                        command_glob=_a.command, forbid=(), reward_scale=scale)
    env = InteractionEnv(cfg=cfg)
    report("E-I 环境接触冒烟")
    report("=" * 88)
    report(f"物体 {_a.object} / 执行器 {_a.executor} / 指令 {Path(_a.command).name}")
    report(f"env {_a.envs}，每档 {_a.steps} 步，压力设定值 {_a.press} N")
    report(f"reward scale {scale}")
    report("")

    inside = run_case(env, "inside", _a.steps, _a.press)
    debug_dump(env, "inside")
    outside = run_case(env, "outside", _a.steps, _a.press)
    debug_dump(env, "outside")
    hover = run_case(env, "hover", _a.steps, _a.press)
    torch.manual_seed(0)
    rand = run_case(env, "random", _a.steps, _a.press)

    report("一、压在允许区域内：接触真的建立起来了吗")
    report("-" * 88)
    report("  怎么读：`touching` 是有接触的 env 占比，`contact_points` 是逐 env 的接触点数。")
    report("        这两项若为零，后面四节全部没有意义——而 dry-run 分不出这一点。")
    for key in ("diag/touching", "diag/contact_points", "diag/peak_normal_force",
                "diag/inside_region_share"):
        report(f"      {key:32s} {inside[key]:+10.4f}")
    report(f"      指令要求接触的步占比              "
           f"{inside['share_of_steps_contact_required']:+10.4f}")
    report(f"      要求接触时的 touching              "
           f"{inside['touching_when_contact_required']:+10.4f}")
    check("接触建立", inside["touching_when_contact_required"] > 0.80,
          f"指令要求接触时，有接触的 env 占比 "
          f"{inside['touching_when_contact_required']:.3f}（门槛 0.80；"
          f"全时段无条件值 {inside['diag/touching']:.3f}）")
    check("接触点非空", inside["diag/contact_points"] > 0.5,
          f"逐 env 平均 {inside['diag/contact_points']:.2f} 个接触点")
    chance = 4.0 / 256.0
    check("落在允许区域显著高于偶然", inside["diag/inside_region_share"] > 10 * chance,
          f"允许区域内的力占比 {inside['diag/inside_region_share']:.3f}，"
          f"偶然水平 {chance:.4f}（门槛 10×）")
    report("")
    report("  ⚠️ **`inside_region_share` 只是报出来的数，不是执行器能力的判据。**")
    report("     `region/allowed` 是**标定过的**允许集合，S5 实测只占表面积 1.35~4.4%")
    report("     （256 格里 4~10 格）。本节用的是一段开环脚本、且垫头杆的垫面")
    report("     40×30 mm 比采集侧的板 35×25 mm 大——四个角差约 2.5 mm，而 256 档的")
    report("     格距约 7 mm。'能不能把力压进允许集合'正是 S6 训练要回答的问题，")
    report("     在冒烟里断言它等于预设结论。这里只要求它显著高于偶然水平。")
    report("")

    report("二、连续功能区域匹配必须给正确接触正收益")
    report("-" * 88)
    report(f"      r_region（区域内）           {inside['reward/region']:+10.4f}")
    report(f"      r_region（区域外）           {outside['reward/region']:+10.4f}")
    check("正确接触的 r_region 为正", inside["reward/region"] > 0.05,
          f"区域内 {inside['reward/region']:+.4f}（门槛 > +0.05）")
    report("")

    report("三、r_region 分得开'压对地方'和'压错地方' ⭐")
    report("-" * 88)
    report("  怎么读：这一节是判别力所在。只看'接触建立'的话，一个'永远返回 0'的")
    report("        实现照样全绿；跟踪 reward 对区域不敏感时训练曲线也照样会涨。")
    report(f"      区域外的允许区域内力占比      {outside['diag/inside_region_share']:+10.4f}")
    gap = inside["reward/region"] - outside["reward/region"]
    check("区域外接触被识别", outside["diag/touching"] > 0.50,
          f"有接触的 env 占比 {outside['diag/touching']:.3f}——没碰到就谈不上'压错了'")
    check("r_region 分得开区域内外", gap > 0.5,
          f"内外差 {gap:+.4f}（门槛 > 0.5）")
    check("压对时的区域内占比更高",
          inside["diag/inside_region_share"] > outside["diag/inside_region_share"] + 0.05,
          f"{inside['diag/inside_region_share']:.3f} vs "
          f"{outside['diag/inside_region_share']:.3f}")
    report("")

    report("四、静压时的滑移与自转必须是小量")
    report("-" * 88)
    report("  怎么读：这一节挡的是 P-52 那一族——执行器自转会让接触点的线速度虚高，")
    report("        而滑移速率既进 `r_mode` 又进观测。垫面离杆根 108 mm，")
    report("        2 rad/s 的自转就是 0.22 m/s 的接触点线速度。")
    report("        实测把旋转 PD 的增益从按质量归一改成按**转动惯量**归一之后，")
    report("        角速度 76 -> 0.67 rad/s、滑移 0.165 -> 0.003 m/s、r_mode -31.6 -> -0.46。")
    for key in ("diag/slip_in_region", "diag/slip_max", "diag/exec_ang_speed",
                "diag/exec_lin_speed"):
        report(f"      {key:32s} {inside[key]:+10.4f}")
    check("静压时滑移是小量", inside["diag/slip_in_region"] < 0.05,
          f"允许区域内的滑移 {inside['diag/slip_in_region']:.4f} m/s（门槛 0.05）")
    check("执行器不自转", inside["diag/exec_ang_speed"] < 5.0,
          f"角速度 {inside['diag/exec_ang_speed']:.3f} rad/s（门槛 5.0）")
    report("")

    report("五、r_mech / r_mode 的量级")
    report("-" * 88)
    report("  怎么读：与 reward_probe 第三节标定的 scale 比。差两个数量级的项，")
    report("        在合成 reward 里等于不存在或等于淹没其余。")
    for key, name in (("reward/mech", "mech"), ("reward/mode", "mode"),
                      ("reward/effect", "effect")):
        report(f"      {key:24s} {inside[key]:+10.4f}   标定 scale {scale[name]:.4f}")
    finite = all(abs(inside[k]) < 1e6 for k in ("reward/mech", "reward/mode",
                                                "reward/effect"))
    check("各项有限", finite, "没有出现 inf/nan 量级")
    check("mech 非零", abs(inside["reward/mech"]) > 1e-6,
          f"{inside['reward/mech']:+.6f}——恒为零说明力学通道没接上")
    check("mode 没有炸", abs(inside["reward/mode"]) < 10.0,
          f"{inside['reward/mode']:+.4f}（门槛 |·|<10）——盒子半宽趋零时这一项"
          "会失去刻度，实测炸到 −27675")
    report("")

    report("六、压对地方必须比什么都不做更划算 ⭐⭐")
    report("-" * 88)
    report("  怎么读：跟踪 reward 的四项里只有 `r_region` 能取正值。若一段大致压对了的")
    report("        执行还不如悬停不动，PPO 就会稳稳地学会躲开物体——**而训练曲线**")
    report("        **会一路上涨**，因为它确实在最大化这个 reward。第一版四项全部 ≤0，")
    report("        实测悬停 −0.43/步、压住 −278/步，这一格当场变红。")
    report(f"      压住时单步 reward           {inside['reward']:+10.4f}")
    report(f"      悬停单步 reward             {hover['reward']:+10.4f}")
    report(f"      压错地方单步 reward         {outside['reward']:+10.4f}")
    report(f"      其中真正压进允许区域的那些 (share>0.8) "
           f"{inside['reward_when_region_satisfied']:+10.4f}"
           f"   占比 {inside['share_of_steps_region_satisfied']:.3f}")
    report("")
    report("  判据必须看**整段脚本的无条件均值**。按 inside_share 再筛样本，会把")
    report("  '大多数 target 接触因形态不同而被惩罚'这一失败条件化掉；旧闸门正是因此")
    report("  把 −0.745/步的正确脚本误判为 PASS，随后 PPO 学会完全避开接触。")
    check("压对 > 悬停（整段无条件均值）",
          inside["reward"] > hover["reward"] + 0.1,
          f"{inside['reward']:+.4f} vs {hover['reward']:+.4f}"
          f"（差 {inside['reward'] - hover['reward']:+.4f}，门槛 +0.1）")
    check("压对 > 压错", inside["reward"] > outside["reward"] + 0.1,
          f"{inside['reward']:+.4f} vs {outside['reward']:+.4f}")
    report("")

    report("七、失败终止的惩罚该给多大")
    report("-" * 88)
    report("  怎么读：五项 reward 全部 ≤0，所以'立刻结束 episode'的回报是 0。")
    report("        失败终止的惩罚必须 ≥ 悬停不动的单步代价，否则开局把自己甩出去")
    report("        就是最优策略（D-31 第 3 个洞）。")
    report(f"      悬停单步 reward             {hover['reward']:+10.4f}")
    report(f"      压住时单步 reward           {inside['reward']:+10.4f}")
    suggested = max(1.0, round(abs(hover["reward"]) * 1.5, 2))
    report(f"      建议 fail_penalty_per_step  {suggested:.2f}   "
           f"（悬停代价的 1.5 倍，留一档裕量）")
    report("")

    report("八、随机策略不能必然失败 ⭐")
    report("-" * 88)
    report("  怎么读：失败终止带一个按剩余步数折现的大惩罚。若随机策略**必然**触发它，")
    report("        每条 episode 的回报就都一样，优势信号归零，PPO 什么都学不到——")
    report("        而训练曲线只是'平'，看不出是学不动还是已经收敛。实测第一版")
    report("        `pos_limit`（逐轴 0.5 m）的立方体角在 0.87 m，而 `far` 门槛是")
    report("        0.75 m：400 轮训练 mean reward **恒等于 −99.13**。")
    for key in ("diag/failed", "diag/touching", "diag/exec_lin_speed"):
        report(f"      {key:32s} {rand[key]:+10.4f}")
    report(f"      {'随机策略单步 reward':32s} {rand['reward']:+10.4f}")
    check("随机策略不是必然失败", rand["diag/failed"] < 0.01,
          f"逐步失败率 {rand['diag/failed']:.4f}（门槛 < 0.01，即平均 100 步以上才失败一次）")
    check("随机策略偶尔能碰到物体", rand["diag/touching"] > 0.02,
          f"有接触的步占比 {rand['diag/touching']:.4f}——一次都碰不到的话，"
          "探索找不到接触，`r_region` 那点正收益就无从被发现")
    report("")

    report("九、没有 env 在压的过程中失败")
    report("-" * 88)
    for key in ("diag/failed", "diag/object_off_table", "diag/finished"):
        report(f"      {key:32s} {inside[key]:+10.4f}")
    check("压的过程中不失败", inside["diag/failed"] < 0.05,
          f"失败率 {inside['diag/failed']:.3f}（门槛 < 0.05）")

    report("")
    report("=" * 88)
    if FAILED:
        report(f"[FAIL] 未通过：{', '.join(FAILED)}")
    else:
        report("[PASS] 全部通过")

    _a.out.parent.mkdir(parents=True, exist_ok=True)
    _a.out.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    _a.out.with_suffix(".json").write_text(json.dumps(
        {"meta": {"object": _a.object, "executor": _a.executor,
                  "primitive": Path(_a.command).stem.split("-")[-2],
                  "command": str(Path(_a.command).resolve()),
                  "implementation_fingerprint": implementation_fingerprint()},
         "inside": inside, "outside": outside, "hover": hover, "random": rand,
         "suggested_fail_penalty_per_step": suggested,
         "failed": FAILED}, indent=2, sort_keys=True), encoding="utf-8")
    env.close()
    simulation_app.close()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(1 if FAILED else 0)


if __name__ == "__main__":
    # ⚠️ P-74：创建过 SimulationApp 之后，**未捕获异常也会以退出码 0 收场**
    # （Kit 的关停流程接管了进程收尾）。判据是退出码的上层脚本会因此报假 PASS，
    # 所以这里自己兜住，显式 os._exit(1)。
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(1)
