#!/usr/bin/env python3
"""比较 S5 把多条示教对齐到同一条 command 轴的几种候选做法。

`plan/03` §8 第 1 条要求"按 **phase 和** object progress 对齐"。v1 实现只用了
`progress`，而 `progress` 是**任务 effect 完成度**：接近全部接近段的帧都是 0、
到达目标之后的帧都是 1。实测（抽屉 12 条真实 S4 记录）**84% 的帧落进 32 格里的
2 格**，中间 30 格每格中位数 2 帧——命令序列的绝大部分容量被浪费在两个退化格上，
而真正的操作段被 1~2 帧描述。所有既有校验都通过，`support/episodes` 还是满的。

这个脚本不猜哪种对齐更好，它测四种候选：

=================  =====================================================
键                  定义
=================  =====================================================
``progress``       全局 progress（v1 的做法）
``phase_time``     phase 均分 8/8/8/8，段内按归一化帧序
``phase_progress`` phase 均分，段内按归一化 running-max progress
``phase_effect``   phase 均分，段内按归一化累计 effect 变化量
``activity``       **按活动量分配格数**，段内按归一化累计活动量
=================  =====================================================

前四个把 32 格**平均分给 4 个 phase**。实测这样做会让抽屉只有 18% 的命令格里
有接触——因为接近段和松开段本来就没有接触，而操作段里到达目标之后还有很长一截
无接触的保持。命令通道的容量被花在"这一步不需要接触"上。

``activity`` 因此改成：格数**按每个 phase 的活动量占比**分配（每段保底
``--phase-floor`` 格），段内也按累计活动量分格。活动量是两路**各自在 episode
内归一化**后相加的无量纲量：

- 接触通道：逐帧法向力之和（接触冲量率）；
- effect 通道：把固定 effect 契约（D-53）的 (dp, dr) 作用在冻结 surface 点上取
  **表面点平均位移**（米）。这样不需要在米和弧度之间拍权重（D-31 的第 2 个洞）；
  ``surface_state`` 通道取面积加权变化量。

两路各自归一化后相加，因此不含任务分支，也不需要跨异质量纲的权重。整段活动量
为零时退化成均匀帧序。

判据四项：

1. **占用均匀度**——单格最大帧占比、空格率。退化对齐在这里就露馅；
2. **接触格占比**——命令通道有多少格真的在描述接触。越高越好；
3. **家族内 region 弥散度**——同一策略家族的两条示教，在同一命令格上的接触分布
   本应接近，差异主要来自对齐误差。**这一项越低越好**；
4. **家族内 / 家族间弥散比**——好的对齐把家族内压低而不抹掉家族间差异。
   两个一起降说明它在抹平真实的策略差异，不是对齐变好。

用法::

    PYTHONPATH=src python3 tools/s5_align_probe.py \
        --inputs drawer=out/s5_smoke_inputs/drawer:out/s5_smoke_inputs/drawer-nominal.npz \
        --out out/s5_align/probe.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from it.records import EpisodeRecord, load_episode, read_manifest  # noqa: E402
from it.surfaces import Surface, load_surface  # noqa: E402
# activity 一档直接调用**实际会发货的那份实现**，避免探针与产物各写一遍而悄悄分叉。
from it.transfer import activity_rate, bin_index as transfer_bin_index, phase_budget  # noqa: E402

KEYS = ("progress", "phase_time", "phase_progress", "phase_effect", "activity")
N_PHASES = 4


def _phase_offsets(budget: tuple[int, ...]) -> np.ndarray:
    """每个 phase 的起始格号。"""
    return np.concatenate([[0], np.cumsum(budget)]).astype(np.int64)


def _bin_index(record: EpisodeRecord, key: str, surface: Surface, *,
               budget: tuple[int, ...], n_bins: int) -> np.ndarray:
    """把每个有效帧映射到命令格号；无效帧为 -1。"""
    a = record.arrays
    valid = np.asarray(a["valid_s4"], dtype=bool)
    progress = np.asarray(a["progress"], dtype=np.float64)
    phase = np.asarray(a["phase"], dtype=np.int64)
    out = np.full(len(valid), -1, dtype=np.int64)

    if key == "progress":
        local = np.clip(progress, 0.0, 1.0 - 1e-9)
        out[valid] = np.minimum((local[valid] * n_bins).astype(np.int64), n_bins - 1)
        return out

    if key == "activity":
        return transfer_bin_index(record, surface, budget=budget)
    activity = activity_rate(record, surface) if key == "phase_effect" else None
    offsets = _phase_offsets(budget)
    for p in range(N_PHASES):
        frames = np.flatnonzero(valid & (phase == p))
        width = int(budget[p])
        if not len(frames) or width <= 0:
            continue
        if key == "phase_time" or len(frames) == 1:
            local = (np.arange(len(frames)) + 0.5) / len(frames)
        elif key == "phase_progress":
            local = _normalize_monotone(np.maximum.accumulate(progress[frames]), len(frames))
        else:
            cumulative = np.concatenate([[0.0], np.cumsum(activity[frames[1:]])])
            local = _normalize_monotone(cumulative, len(frames))
        out[frames] = offsets[p] + np.minimum((local * width).astype(np.int64), width - 1)
    return out


def _normalize_monotone(values: np.ndarray, n: int) -> np.ndarray:
    """把一条非降序列压到 [0,1)；完全没有变化时退化成均匀帧序。"""
    span = float(values[-1] - values[0])
    if not np.isfinite(span) or span <= 1e-12:
        return (np.arange(n) + 0.5) / n
    return (values - values[0]) / span * (1.0 - 1e-9)


def _region_per_bin(record: EpisodeRecord, surface: Surface, bin_index: np.ndarray, *,
                    n_bins: int, n_surface: int) -> tuple[np.ndarray, np.ndarray]:
    """每个命令格上的法向力加权接触分布 + 该格的帧数。"""
    a = record.arrays
    parent = surface.parent[n_surface]
    idx = np.asarray(a["region/point_idx"], dtype=np.int64)
    live = (np.asarray(a["region/valid"], dtype=bool)
            & np.asarray(a["region/on_surface"], dtype=bool)
            & (np.asarray(a["region/weight"], dtype=np.float64) > 0))
    weight = np.asarray(a["region/weight"], dtype=np.float64)
    region = np.zeros((n_bins, n_surface))
    counts = np.zeros(n_bins, dtype=np.int64)
    for t in np.flatnonzero(bin_index >= 0):
        counts[bin_index[t]] += 1
        m = live[t]
        if np.any(m):
            np.add.at(region[bin_index[t]], parent[idx[t, m]], weight[t, m])
    total = region.sum(axis=1, keepdims=True)
    region = np.divide(region, total, out=np.zeros_like(region), where=total > 0)
    return region, counts


def _js_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon 距离，∈[0,1]。两个分布完全不重叠时为 1。"""
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = [np.where(x > 0, x * np.log(np.where(m > 0, x / np.where(m > 0, m, 1), 1)), 0.0)
                 for x in (p, q)]
    divergence = 0.5 * float(terms[0].sum() + terms[1].sum())
    return float(np.sqrt(max(divergence, 0.0) / np.log(2.0)))


def _bin_homogeneity(record: EpisodeRecord, surface: Surface, bin_index: np.ndarray, *,
                     n_bins: int, n_surface: int) -> float:
    """同一条 episode、同一个命令格内，各接触帧的接触分布彼此有多不一致。

    这是**不受占用量混淆**的对齐判据：``within_family_js`` 会被"格子越大、平均得
    越平、看起来越像"带偏，而这一项直接问"这个格子有没有把物理上不同的接触状态
    混进同一条命令"。越低越好。只统计有 ≥2 个接触帧的格。
    """
    a = record.arrays
    parent = surface.parent[n_surface]
    idx = np.asarray(a["region/point_idx"], dtype=np.int64)
    live = (np.asarray(a["region/valid"], dtype=bool)
            & np.asarray(a["region/on_surface"], dtype=bool)
            & (np.asarray(a["region/weight"], dtype=np.float64) > 0))
    weight = np.asarray(a["region/weight"], dtype=np.float64)

    per_frame: dict[int, list[np.ndarray]] = {}
    for t in np.flatnonzero(bin_index >= 0):
        m = live[t]
        if not np.any(m):
            continue
        row = np.zeros(n_surface)
        np.add.at(row, parent[idx[t, m]], weight[t, m])
        total = row.sum()
        if total > 0:
            per_frame.setdefault(int(bin_index[t]), []).append(row / total)

    scores: list[float] = []
    for rows in per_frame.values():
        if len(rows) < 2:
            continue
        stacked = np.stack(rows)
        centre = stacked.mean(axis=0)
        scores.extend(_js_distance(row, centre) for row in stacked)
    return float(np.mean(scores)) if scores else float("nan")


def _dispersion(regions: np.ndarray, families: list[str]) -> tuple[float, float, int, int]:
    """同格内的家族内 / 家族间平均 JS 距离，以及各自的样本对数。"""
    within: list[float] = []
    across: list[float] = []
    n_eps, n_bins, _ = regions.shape
    for b in range(n_bins):
        present = [i for i in range(n_eps) if regions[i, b].sum() > 0]
        for ai in range(len(present)):
            for bi in range(ai + 1, len(present)):
                i, j = present[ai], present[bi]
                distance = _js_distance(regions[i, b], regions[j, b])
                (within if families[i] == families[j] else across).append(distance)
    return (float(np.mean(within)) if within else float("nan"),
            float(np.mean(across)) if across else float("nan"),
            len(within), len(across))


def probe_task(name: str, root: Path, surface: Surface, *, n_bins: int, n_surface: int,
               limit: int | None, floor: int) -> dict:
    manifest = read_manifest(root / "manifest.json")
    entries = [e for e in manifest["episodes"]
               if e.get("success") and (root / e["path"]).exists()]
    entries.sort(key=lambda e: str(e["episode_id"]))
    if limit:
        entries = entries[:limit]
    if len(entries) < 2:
        raise SystemExit(f"{name}: 至少需要 2 条成功 episode，实际 {len(entries)}")
    records = [load_episode(root / e["path"]) for e in entries]
    families = [str(r.meta.get("strategy_family", "unknown")) for r in records]

    even = n_bins // N_PHASES
    even_budget = tuple([even] * (N_PHASES - 1) + [n_bins - even * (N_PHASES - 1)])
    activity_budget = phase_budget(records, surface, n_bins=n_bins, floor=floor)

    result = {"task": name, "episodes": len(records), "n_bins": n_bins,
              "families": sorted(set(families)), "even_budget": list(even_budget),
              "activity_budget": list(activity_budget), "keys": {}}
    for key in KEYS:
        budget = activity_budget if key == "activity" else even_budget
        regions, counts, homogeneity = [], [], []
        for record in records:
            index = _bin_index(record, key, surface, budget=budget, n_bins=n_bins)
            region, count = _region_per_bin(record, surface, index, n_bins=n_bins,
                                            n_surface=n_surface)
            regions.append(region)
            counts.append(count)
            homogeneity.append(_bin_homogeneity(record, surface, index, n_bins=n_bins,
                                                n_surface=n_surface))
            if np.any((index < 0) & np.asarray(record.arrays["valid_s4"], dtype=bool)):
                raise SystemExit(f"{key}: 有有效帧没有落进任何命令格（budget={budget}）")
        regions = np.stack(regions)
        counts = np.stack(counts).astype(np.float64)
        share = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
        within, across, n_within, n_across = _dispersion(regions, families)
        result["keys"][key] = {
            "budget": list(budget),
            "max_bin_frame_share": float(share.max(axis=1).mean()),
            "top2_bin_frame_share": float(np.sort(share, axis=1)[:, -2:].sum(axis=1).mean()),
            "empty_bin_fraction": float((counts == 0).mean()),
            "median_frames_per_bin": float(np.median(counts)),
            "contact_bin_fraction": float((regions.sum(axis=2) > 0).mean()),
            "within_bin_frame_js": float(np.nanmean(homogeneity)),
            "within_family_js": within,
            "across_family_js": across,
            "within_over_across": (float(within / across)
                                   if across == across and across > 0 else float("nan")),
            "n_within_pairs": n_within,
            "n_across_pairs": n_across,
        }
    return result


def _format(results: list[dict]) -> str:
    lines = ["S5 command-axis alignment probe",
             "=" * 78,
             "判据：max/top2 帧占比越低越好（退化格）；within_family_js 越低越好（对齐误差）；",
             "      binJS = 格内逐帧接触分布的不一致度，**不受格子大小混淆**，越低越好；",
             "      within_over_across 越低越好，但 across 本身不应被一起压低（那是抹平策略差异）。",
             ""]
    for result in results:
        lines.append(f"--- {result['task']}  ({result['episodes']} eps, "
                     f"{len(result['families'])} families)  "
                     f"activity budget {result['activity_budget']}")
        lines.append(f"{'key':16s} {'max%':>6s} {'top2%':>6s} {'empty%':>7s} "
                     f"{'contact%':>8s} {'binJS':>6s} {'within':>7s} {'across':>7s} {'w/a':>6s}")
        for key, value in result["keys"].items():
            lines.append(
                f"{key:16s} {value['max_bin_frame_share']*100:6.1f} "
                f"{value['top2_bin_frame_share']*100:6.1f} "
                f"{value['empty_bin_fraction']*100:7.1f} "
                f"{value['contact_bin_fraction']*100:8.1f} "
                f"{value['within_bin_frame_js']:6.3f} "
                f"{value['within_family_js']:7.3f} {value['across_family_js']:7.3f} "
                f"{value['within_over_across']:6.3f}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="name=records_dir:frozen_surface.npz")
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--surface-points", type=int, default=256)
    parser.add_argument("--limit", type=int, help="每个任务最多用多少条 episode")
    parser.add_argument("--phase-floor", type=int, default=2,
                        help="activity 键给每个非空 phase 的保底格数")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    results = []
    for spec in args.inputs:
        name, _, rest = spec.partition("=")
        root, _, surface_path = rest.partition(":")
        surface = load_surface(Path(surface_path))
        results.append(probe_task(name, Path(root), surface, n_bins=args.bins,
                                  n_surface=args.surface_points, limit=args.limit,
                                  floor=args.phase_floor))
    text = _format(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    args.out.with_suffix(".json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
