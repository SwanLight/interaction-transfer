#!/usr/bin/env python3
"""E-I 交互跟踪 reward 的离线体检：**三个都能失败的问题**。

在写环境、写网络、烧 GPU 之前，reward 本身能不能立住是可以在纯 numpy/torch 里
查清楚的。三节：

**一、在线 traction 估计量与 S5 离线的定义是不是同一个量？**
    离线（D-72）是"固定 4 mm 带宽的同部件核散射到冻结表面点，再在格内按 |f| 加权
    池化"；在线不可能每步散射 16384 个点，用的是连续形式
    ``t(x)=Σ F_k·G_σ(x-x_k)`` 在接触点上求值再池化。**两者必须是同一个量**，
    否则 reward 追的不是 artifact 里写的那个东西——而这件事不会报错。
    判据：逐格比值的中位数落在 [1/tol, tol] 内，且相关系数 ≥ 下限。

**二、reward 分得开"这条示教在执行它自己的指令"和"在执行别的指令"吗？**
    这是本工具最要紧的一节。交互跟踪 reward 若对指令不敏感，E-I 学到的就是
    "做点什么"而不是"实现这份规格"，而训练曲线照样会涨。
    做法：把每条成功示教分别按**它自己的** artifact 和**别的任务/原语的** artifact
    打分，比两者的差。判据：自己的指令必须显著更高，且区分度 AUC ≥ 下限。
    **这一条不通过，后面的一切都没有意义。**

**三、各项的量级配比。**
    D-31 第 2 个洞：量纲失衡时最优解是"不动"。权重必须在**成功轨迹**上标定，
    不许拍。本节报每一项在成功示教上的分布，并输出可直接喂给
    `ei_reward.RewardWeights` 的 ``scale``。

用法::

    tools/s6_reward_probe.py \\
        --task drawer=/tmp/s4_drawer=/tmp/s5/drawer/drawer-drawer-nominal-train.npz \\
        --task wipe=/tmp/s4_wipe=/tmp/s5/wipe/wipe-board-nominal-train.npz \\
        --task knob=/tmp/s4_knob=/tmp/s5/knob/knob-knob-nominal-train.npz \\
        --out out/s6/reward_probe.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from it.ei_reward import (  # noqa: E402
    assign_cells, box_violation, effect_deficit, effect_magnitude, interaction_reward,
    surface_traction)
from it.records import load_episode, read_manifest  # noqa: E402
from it.surfaces import SCATTER_SIGMA, load_surface  # noqa: E402
from it.transfer import bin_index, episode_summary, load_transfer, phase_budget  # noqa: E402

#: 在线/离线的比值容差。D-79 之后两边是**同一个式子**，剩下的差别只有两处离散化：
#: 离线按记录的 `region/point_idx → parent` 归格，在线按法向同侧的最近命令点归格；
#: 离线的有效接触还多一层 `on_surface & weight>0` 过滤。所以容差收到 1.25。
#:
#: ⚠️ 第一版这里拍的是 2.0，**于是 1.75 倍的系统偏差"通过"了**（P-72）。
#: 容差是拍出来的，拍出来的容差挡不住系统偏差——真正发现问题的是相关系数：
#: 比值稳定在 1.6 而不是散在 1 附近，那不是噪声。所以下面同时看比值和相关。
TRACTION_RATIO_TOL = 1.25
TRACTION_CORR_MIN = 0.70
MATCH_AUC_MIN = 0.75


def _records(dataset: Path, split: str, limit: int):
    manifest = read_manifest(dataset / "manifest.json")
    taken = 0
    for entry in manifest["episodes"]:
        if str(entry.get("split")) != split or not entry.get("success"):
            continue
        if not (dataset / entry["path"]).exists():
            continue
        record = load_episode(dataset / entry["path"])
        if not record.meta.get("success", False):
            continue
        yield record
        taken += 1
        if taken >= limit:
            return


def frame_tensors(record, surface, n_surface: int):
    """把一条 record 摊成逐帧的接触张量（在线 reward 的输入形状）。"""
    a = record.arrays
    valid = np.asarray(a["valid_s4"], dtype=bool)
    live = (np.asarray(a["region/valid"], bool) & np.asarray(a["region/on_surface"], bool)
            & (np.asarray(a["region/weight"]) > 0) & valid[:, None])
    idx = np.asarray(a["region/point_idx"], dtype=np.int64).clip(0, surface.n_points - 1)
    normals = np.asarray(surface.normals, dtype=np.float32)[idx]
    return {
        "valid": torch.as_tensor(live),
        "pos": torch.as_tensor(np.asarray(a["region/pos_obj"], dtype=np.float32)),
        "force": torch.as_tensor(np.asarray(a["mech/force_obj"], dtype=np.float32)),
        "normal": torch.as_tensor(normals),
        "slip": torch.as_tensor(np.asarray(a["mode/pose_slip"], dtype=np.float32)),
        "rigid": torch.as_tensor(np.asarray(a["effect/rigid"], dtype=np.float32)[:, 0, :]),
        "state": torch.as_tensor(
            np.asarray(a["effect/surface_state"], dtype=np.float32)[:, 0, :]),
        "frame_valid": torch.as_tensor(valid),
    }


def online_traction_per_bin(frames, surface, index, n_bins, n_surface):
    """在线估计量按命令格取平均，口径与离线的"只在接触帧上平均"一致。"""
    cells = torch.as_tensor(np.asarray(surface.points[:n_surface], dtype=np.float32))
    cell_normals = torch.as_tensor(np.asarray(surface.normals[:n_surface], dtype=np.float32))
    n_frames = frames["pos"].shape[0]
    cell_index = assign_cells(frames["pos"], frames["normal"], frames["valid"],
                              cells.expand(n_frames, -1, -1),
                              cell_normals.expand(n_frames, -1, -1))
    traction, mass = surface_traction(frames["pos"], frames["force"], frames["valid"],
                                      cell_index, n_surface)
    out = torch.full((n_bins, n_surface, 3), float("nan"))
    counts = torch.zeros(n_bins, n_surface)
    accum = torch.zeros(n_bins, n_surface, 3)
    live_bin = torch.as_tensor(index)
    for b in range(n_bins):
        rows = torch.nonzero(live_bin == b, as_tuple=True)[0]
        if not len(rows):
            continue
        touched = mass[rows] > 0
        counts[b] = touched.float().sum(0)
        accum[b] = (traction[rows] * touched[..., None].float()).sum(0)
    occupied = counts > 0
    out[occupied] = accum[occupied] / counts[occupied][..., None]
    return out, occupied


def section_traction(records, surface, budget, n_bins, n_surface):
    ratios, pairs = [], []
    for record in records:
        summary, _ = episode_summary(record, surface, budget=budget, n_bins=n_bins,
                                     n_surface=n_surface)
        offline = torch.as_tensor(np.asarray(summary["traction"], dtype=np.float32))
        frames = frame_tensors(record, surface, n_surface)
        index = bin_index(record, surface, budget=budget)
        online, occupied = online_traction_per_bin(frames, surface, index, n_bins, n_surface)
        both = occupied & torch.isfinite(offline).all(-1) & torch.isfinite(online).all(-1)
        if not both.any():
            continue
        a = offline[both].norm(dim=-1)
        b = online[both].norm(dim=-1)
        keep = (a > 0) & (b > 0)
        if not keep.any():
            continue
        ratios.append((b[keep] / a[keep]).numpy())
        pairs.append(np.stack([a[keep].numpy(), b[keep].numpy()]))
    if not ratios:
        return {"median_ratio": float("nan"), "corr": float("nan"), "n": 0}
    ratio = np.concatenate(ratios)
    joint = np.concatenate(pairs, axis=1)
    corr = float(np.corrcoef(np.log(joint[0]), np.log(joint[1]))[0, 1])
    return {"median_ratio": float(np.median(ratio)),
            "p10_ratio": float(np.percentile(ratio, 10)),
            "p90_ratio": float(np.percentile(ratio, 90)),
            "corr": corr, "n": int(ratio.size)}


def score_episode(record, surface, transfer, budget, n_bins, n_surface):
    """一条示教在给定 artifact 下的逐项 reward 均值。"""
    arrays = transfer.arrays
    frames = frame_tensors(record, surface, n_surface)
    index = torch.as_tensor(bin_index(record, surface, budget=budget))
    keep = torch.nonzero((index >= 0) & frames["frame_valid"], as_tuple=True)[0]
    if not len(keep):
        return None
    b = index[keep]
    n = len(keep)
    cells = torch.as_tensor(np.asarray(surface.points[:n_surface], dtype=np.float32))
    cell_normals = torch.as_tensor(np.asarray(surface.normals[:n_surface], dtype=np.float32))
    cell_index = assign_cells(frames["pos"][keep], frames["normal"][keep],
                              frames["valid"][keep],
                              cells.expand(n, -1, -1), cell_normals.expand(n, -1, -1))
    traction, mass = surface_traction(frames["pos"][keep], frames["force"][keep],
                                      frames["valid"][keep], cell_index, n_surface)
    take = lambda name: torch.as_tensor(np.asarray(arrays[name], dtype=np.float32))[b]  # noqa: E731
    slip_cell = torch.zeros(n, n_surface)
    slip_cell.scatter_reduce_(1, cell_index, frames["slip"][keep] * frames["valid"][keep],
                              reduce="amax", include_self=True)
    metric = torch.as_tensor(np.asarray(arrays["effect/rigid/metric"], dtype=np.float32))
    scale_r = torch.as_tensor(float(arrays["effect/rigid/scale_m"]))
    scale_s = torch.as_tensor(float(arrays["effect/surface_state/scale"]))
    want_rigid = take("effect/rigid/median")[:, 0, :]
    want_state = take("effect/surface_state/median")[:, 0, :]
    got = effect_magnitude(frames["rigid"][keep], frames["state"][keep],
                           metric=metric.expand(n, -1, -1),
                           scale_rigid=scale_r.expand(n), scale_state=scale_s.expand(n))
    want = effect_magnitude(want_rigid, want_state, metric=metric.expand(n, -1, -1),
                            scale_rigid=scale_r.expand(n), scale_state=scale_s.expand(n))
    terms = interaction_reward(
        # 与环境同一个口径：本格的完成缺口，不是逐步差值（见 ei_reward.effect_deficit）。
        effect_deficit=effect_deficit(got.cumsum(0), want.clamp_min(1e-9)),
        traction=traction, mass=mass, slip_speed=slip_cell,
        allowed=take("region/allowed").bool(),
        traction_lo=take("mech/traction_obj/lo"), traction_hi=take("mech/traction_obj/hi"),
        slip_lo=take("mode/slip_speed/lo"), slip_hi=take("mode/slip_speed/hi"),
        force_penalty=torch.zeros(n))
    return {k: float(v.mean()) for k, v in terms.__dict__.items()}


def _auc(positive: np.ndarray, negative: np.ndarray) -> float:
    if not len(positive) or not len(negative):
        return float("nan")
    wins = (positive[:, None] > negative[None, :]).mean()
    ties = (positive[:, None] == negative[None, :]).mean()
    return float(wins + 0.5 * ties)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", dest="tasks", action="append", required=True,
                        metavar="名字=数据集=artifact")
    parser.add_argument("--surface", action="append", default=[], metavar="[名字=]路径")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--n-bins", type=int, default=32)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    surfaces: dict[str | None, Path] = {}
    for item in args.surface:
        key, _, value = item.rpartition("=")
        surfaces[key or None] = Path(value)

    loaded = []
    for spec in args.tasks:
        name, dataset, artifact = spec.split("=", 2)
        dataset = Path(dataset)
        transfer = load_transfer(Path(artifact))
        manifest = read_manifest(dataset / "manifest.json")
        obj = str(transfer.meta["object"]); geom = str(transfer.meta["geometry_variant"])
        path = surfaces.get(name, surfaces.get(None))
        if path is None:
            info = manifest.get("surfaces", {}).get(f"{obj}/{geom}", {})
            path = dataset / info["path"]
        surface = load_surface(path)
        records = list(_records(dataset, "train", args.limit))
        if len(records) < 2:
            raise SystemExit(f"{name}: 训练划分里成功 record 不足")
        budget = tuple(transfer.meta["aggregation"]["phase_budget"])
        loaded.append({"name": name, "surface": surface, "transfer": transfer,
                       "records": records, "budget": budget,
                       "n_surface": int(transfer.arrays["surface/points_obj"].shape[0])})

    lines = ["E-I 交互跟踪 reward 的离线体检", "=" * 88, ""]
    failures = 0

    lines += ["一、在线 traction 估计量 == S5 离线的定义吗？",
              "-" * 88,
              "  怎么读：比值是「在线 / 离线」，应当在 1 附近；相关系数在 log 域上算。",
              f"  判据：中位比值 ∈ [{1/TRACTION_RATIO_TOL:.2f}, {TRACTION_RATIO_TOL:.2f}] "
              f"且相关 ≥ {TRACTION_CORR_MIN}。",
              "  ⚠️ 比值稳定（相关高）而偏离 1，是**系统偏差**不是噪声——那比比值本身",
              "     更值得看。第一版容差拍成 2.0，1.75 倍的偏差就那样通过了（P-72）。",
              f"  （两者的 σ 都是 {SCATTER_SIGMA*1e3:.0f} mm——σ 是 traction 定义的一部分，"
              "不是可以两边各调的超参）", ""]
    lines.append("      " + f"{'任务':<8s}{'中位比值':>10s}{'p10':>9s}{'p90':>9s}"
                 f"{'log 相关':>10s}{'样本':>9s}   判定")
    per_task_traction = {}
    for item in loaded:
        stat = section_traction(item["records"][:8], item["surface"], item["budget"],
                                args.n_bins, item["n_surface"])
        per_task_traction[item["name"]] = stat
        ok = (1 / TRACTION_RATIO_TOL <= stat["median_ratio"] <= TRACTION_RATIO_TOL
              and stat["corr"] >= TRACTION_CORR_MIN)
        failures += 0 if ok else 1
        lines.append("      " + f"{item['name']:<8s}{stat['median_ratio']:10.3f}"
                     f"{stat['p10_ratio']:9.3f}{stat['p90_ratio']:9.3f}"
                     f"{stat['corr']:10.3f}{stat['n']:9d}   {'PASS' if ok else 'FAIL'}")
    lines.append("")

    lines += ["二、reward 分得开「执行自己的指令」和「执行别人的指令」吗？",
              "-" * 88,
              "  做法：每条成功示教分别按它自己的 artifact 与其余任务的 artifact 打分。",
              "  怎么读：AUC = 随机取一条自评分与一条错配分，自评分更高的概率。",
              f"  判据：AUC ≥ {MATCH_AUC_MIN}。**这一条不过，后面一切都没有意义**——",
              "        跟踪 reward 对指令不敏感时，训练曲线照样会涨，而学到的是"
              "「做点什么」而不是「实现这份规格」。", ""]
    matched: dict[str, list[float]] = {}
    mismatched: dict[str, list[float]] = {}
    for item in loaded:
        for record in item["records"][:args.limit]:
            own = score_episode(record, item["surface"], item["transfer"],
                                item["budget"], args.n_bins, item["n_surface"])
            if own is None:
                continue
            matched.setdefault(item["name"], []).append(
                own["effect"] + own["region"] + own["mode"] + own["mech"])
            for other in loaded:
                if other["name"] == item["name"] or other["n_surface"] != item["n_surface"]:
                    continue
                # 换指令但**不换物体**：错配的是「要求」，不是「在哪个物体上」。
                alien = score_episode(record, item["surface"], other["transfer"],
                                      other["budget"], args.n_bins, item["n_surface"])
                if alien is not None:
                    mismatched.setdefault(item["name"], []).append(
                        alien["effect"] + alien["region"] + alien["mode"] + alien["mech"])
    lines.append("      " + f"{'任务':<8s}{'自己的指令':>12s}{'错配指令':>12s}"
                 f"{'差':>10s}{'AUC':>8s}   判定")
    auc_all = {}
    for item in loaded:
        name = item["name"]
        pos = np.array(matched.get(name, []))
        neg = np.array(mismatched.get(name, []))
        auc = _auc(pos, neg)
        auc_all[name] = auc
        ok = np.isfinite(auc) and auc >= MATCH_AUC_MIN
        failures += 0 if ok else 1
        lines.append("      " + f"{name:<8s}{pos.mean() if len(pos) else float('nan'):12.3f}"
                     f"{neg.mean() if len(neg) else float('nan'):12.3f}"
                     f"{(pos.mean()-neg.mean()) if len(pos) and len(neg) else float('nan'):10.3f}"
                     f"{auc:8.3f}   {'PASS' if ok else 'FAIL'}")
    lines.append("")

    lines += ["三、各项在成功示教上的量级（用来标定权重，D-31 第 2 个洞）",
              "-" * 88,
              "  怎么读：四项都是负的误差，0 最好。某一项若比别人小两个数量级，",
              "        它在合成 reward 里就等于不存在；若大两个数量级，别的项就等于不存在。",
              "  ⚠️ effect 这一列在本工具里是**整条 episode 累计**算的，而环境里是**逐格**",
              "     累计（每推进一格清零）。所以这一列偏乐观，只用来看量级配比，",
              "     不能当作训练时 r_effect 的预期值。第二节的 AUC 不受影响——",
              "     自评分与错配分用的是同一个口径。",
              ""]
    lines.append("      " + f"{'任务':<8s}{'effect':>12s}{'region':>12s}"
                 f"{'mode':>12s}{'mech':>12s}")
    scales: dict[str, float] = {}
    per_term: dict[str, list[float]] = {"effect": [], "region": [], "mode": [], "mech": []}
    for item in loaded:
        rows = [score_episode(r, item["surface"], item["transfer"], item["budget"],
                              args.n_bins, item["n_surface"])
                for r in item["records"][:args.limit]]
        rows = [r for r in rows if r]
        if not rows:
            continue
        means = {k: float(np.mean([r[k] for r in rows])) for k in per_term}
        for k, v in means.items():
            per_term[k].append(v)
        lines.append("      " + f"{item['name']:<8s}" + "".join(
            f"{means[k]:12.4f}" for k in ("effect", "region", "mode", "mech")))
    for k, values in per_term.items():
        scales[k] = float(np.median(np.abs(values))) if values else 1.0
    lines += ["",
              "  标定出的 scale（喂给 ei_reward.RewardWeights.scale）：",
              "      " + json.dumps({k: round(v, 6) for k, v in scales.items()},
                                    ensure_ascii=False), ""]

    lines += ["=" * 88, f"FAIL {failures} 项" if failures else "全部通过"]
    text = "\n".join(lines)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps(
        {"traction": per_task_traction, "auc": auc_all, "scale": scales},
        ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(text)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
