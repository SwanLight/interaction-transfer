#!/usr/bin/env python3
"""S5 闸门：在**冻结的**留出划分上评估一份 interaction transfer artifact。

`plan/README` §7 给 S5 的通过条件是"未见 source 策略上仍能输出正确 functional
envelope"，`plan/03` §8.1 把它落成两个可测量的数——**coverage** 与 **width**，
目标是"coverage ≥ 90% 的约束下最小化 width"。`plan/02` §7 的第 4、8 两条泄漏检查
在 S4 阶段如实标了 DEFER，因为它们要 envelope；这里补上。

七项检查
--------

===  =====================================================================
1    **coverage**：留出 episode 的接触质量有多大比例落进允许区域；mechanics /
     effect 的逐点带内率与 episode 级同时带内率
2    **width**：允许区域面积占物体表面积的比例；mechanics 带的相对宽度
3    **策略子群 coverage**：按 strategy family 分开报。envelope 若被某一族主导，
     整体 coverage 可以很好看而其余族被系统性排除——这才是第 4 条泄漏的实质
4    **family envelope 距离**：pooled envelope 与各族自建 envelope 的距离谱。
     贴着某一族＝聚合塌到了那一族上
5    **多峰性**：pooled envelope 到各 episode 的距离 vs episode 到最近邻 episode
     的距离。前者显著更大＝聚合出了一个谁都没做过的混血指令（D-63 的触发条件）
6    **跨实现可互换**（仅擦拭，`plan/02` §7 第 8 条）：只用持工具的 episode 与只用
     直擦的 episode 各建一份 envelope，比较二者
7    **接口不变量**：payload 恰好是 allowlist；改变 source 接触体数量不改变维度
===  =====================================================================

⚠️ 判据是**退出码**，不是文件开头（P-55）。任一项 FAIL 则整体非零。10/90 分位数
是描述性统计，不是 conformal guarantee，因此第 1 项里 mechanics 的带内率**只报数、
不设通过门槛**（D-59/D-63）：要给保证得先做 CQR，那要等冻结校准集另立一步。

用法::

    PYTHONPATH=src python3 tools/s5_eval_envelope.py \\
        --artifact out/s5/drawer/drawer-drawer-nominal-train.npz \\
        --manifest /tmp/s4_drawer/manifest.json \\
        --surface /tmp/s4_drawer/surfaces/drawer-nominal.npz \\
        --out out/s5/drawer/eval.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from it.records import EpisodeRecord, load_episode, read_manifest  # noqa: E402
from it.surfaces import Surface, load_surface, surface_for  # noqa: E402
from it.transfer import (  # noqa: E402
    InteractionTransfer,
    build_transfer,
    episode_summary,
    load_transfer,
)

#: 允许区域取"按 region mass 降序累计到这个比例"的最小 cell 集合（超水平集）。
#: `plan/03` §8.1 的 region 子指标就是"至少 95% 法向力加权接触质量落入允许集合"。
REGION_MASS_TARGET = 0.95
#: coverage 的通过门槛（`plan/03` §8.1）。只对 region 生效；mechanics 只报数。
COVERAGE_GATE = 0.90
#: 第 4、6 项要用子集重建 envelope，每组最多保留这么多条 record，避免整份数据集驻留内存。
#: 按 episode_id 排序后取前 N，与运行顺序无关。
SUBSET_CAP = 200
#: 允许集合的大小只由一个参数控制：累计到 pooled region mass 的哪一档。
#: 这一族集合是**嵌套**的（τ 越大集合越大），所以 split conformal 可以直接在 τ 上做。
TAU_GRID = np.concatenate([np.linspace(0.50, 0.99, 50), [0.995, 0.999, 1.0]])


class Check:
    """一项检查的结论。``status`` 只能是 PASS / FAIL / DEFER。"""

    def __init__(self, name: str, status: str, detail: str, data: dict | None = None):
        if status not in ("PASS", "FAIL", "DEFER"):
            raise ValueError(status)
        self.name, self.status, self.detail, self.data = name, status, detail, data or {}

    def line(self) -> str:
        return f"[{self.status}] {self.name}: {self.detail}"


# ------------------------------------------------------------ 基础量

def _js_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon 距离 ∈ [0,1]；两个分布完全不重叠时为 1。"""
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)
    total = 0.0
    for x in (p, q):
        mask = x > 0
        total += 0.5 * float(np.sum(x[mask] * np.log(x[mask] / m[mask])))
    return float(np.sqrt(max(total, 0.0) / np.log(2.0)))


def _nested_allowed(region: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """每个命令格上按 pooled region mass 降序的 cell 次序，及其累计质量占比。

    ``A(τ)`` = 每格取累计占比刚好达到 τ 的前缀。这一族集合随 τ **嵌套**，
    因此可以直接把 τ 当成 split conformal 的一维阈值来标定。
    """
    orders, cumulative = [], []
    for row in region:
        total = row.sum()
        if total <= 0:
            orders.append(np.zeros(0, dtype=np.int64))
            cumulative.append(np.zeros(0))
            continue
        order = np.argsort(-row, kind="stable")
        orders.append(order)
        cumulative.append(np.cumsum(row[order]) / total)
    return orders, cumulative


def _prefix_lengths(cumulative: list[np.ndarray], tau: float) -> list[int]:
    return [int(np.searchsorted(cum, tau) + 1) if len(cum) else 0 for cum in cumulative]


def _lengths_by_tau(cumulative: list[np.ndarray]) -> np.ndarray:
    """(len(TAU_GRID), n_bins) 的前缀长度表。它只跟 pooled envelope 有关，算一次即可。"""
    return np.array([_prefix_lengths(cumulative, tau) for tau in TAU_GRID], dtype=np.int64)


def _mass_inside_curve(episode: np.ndarray, orders: list[np.ndarray],
                       lengths: np.ndarray) -> np.ndarray:
    """这条 episode 在 ``A(τ)`` 里的接触质量占比，对整个 τ 网格一次算完。"""
    total = episode.sum()
    if total <= 0:
        return np.full(len(TAU_GRID), np.nan)
    inside = np.zeros(len(TAU_GRID))
    for b, order in enumerate(orders):
        if not len(order):
            continue
        cumulative_mass = np.concatenate([[0.0], np.cumsum(episode[b][order])])
        inside += cumulative_mass[np.minimum(lengths[:, b], len(order))]
    return inside / total


def _required_tau(curve: np.ndarray) -> float:
    """让这条 episode 的接触质量达到 95% 所需要的最小 τ；达不到就是 +inf。"""
    hit = np.flatnonzero(curve >= REGION_MASS_TARGET)
    return float(TAU_GRID[hit[0]]) if len(hit) else float("inf")


def _conformal_tau(required: np.ndarray, alpha: float = 1.0 - COVERAGE_GATE) -> float:
    """split conformal：校准集上 ``⌈(n+1)(1-α)⌉/n`` 分位数。

    要求 exchangeability，给的是 **marginal** coverage，不是对任意子群的条件保证
    （D-59）。所以子群 coverage 必须另报，那正是第 3 项。
    """
    finite = np.sort(required[np.isfinite(required)])
    n = len(required)
    if n == 0:
        return float("nan")
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    if rank > len(finite):
        return float("inf")     # 校准集里达不到的比例已经超过 α，任何 τ 都不够
    return float(finite[rank - 1])


def _allowed_at(orders: list[np.ndarray], cumulative: list[np.ndarray], tau: float,
                n_surface: int) -> np.ndarray:
    allowed = np.zeros((len(orders), n_surface), dtype=bool)
    for b, k in enumerate(_prefix_lengths(cumulative, tau)):
        if k > 0 and len(orders[b]):
            allowed[b, orders[b][:k]] = True
    return allowed


def _allowed_region(region: np.ndarray, target: float = REGION_MASS_TARGET) -> np.ndarray:
    """每个命令格上、累计质量达到 ``target`` 的**最小** cell 集合。

    取超水平集而不是"mass > 阈值"：给定要覆盖的质量，超水平集的面积最小，
    因此 width 这个数才有"在 coverage 约束下最小化"的含义。
    """
    allowed = np.zeros(region.shape, dtype=bool)
    for b, row in enumerate(region):
        if row.sum() <= 0:
            continue
        order = np.argsort(-row, kind="stable")
        cumulative = np.cumsum(row[order])
        keep = int(np.searchsorted(cumulative, target * row.sum()) + 1)
        allowed[b, order[:keep]] = True
    return allowed


def _episode_region(summary: dict[str, np.ndarray]) -> np.ndarray:
    """一条 episode 投影到命令轴后的逐格接触分布（无接触的格为全零）。"""
    return np.nan_to_num(summary["region"], nan=0.0)


def _pooled_region(transfer: InteractionTransfer) -> np.ndarray:
    return np.asarray(transfer.arrays["region/mass/mean"], dtype=np.float64)


def _frozen_axis(transfer: InteractionTransfer) -> dict:
    """artifact 自己的命令轴参数。比较任何两份 envelope 都必须用同一条轴。"""
    aggregation = transfer.meta["aggregation"]
    return {"n_bins": int(aggregation["n_bins"]),
            "n_surface": int(transfer.meta["surface"]["command_n_points"]),
            "budget": tuple(int(v) for v in aggregation["phase_budget"])}


def _summaries(records: list[EpisodeRecord], surface: Surface, transfer: InteractionTransfer
               ) -> list[dict[str, np.ndarray]]:
    """用 artifact 自己的命令轴投影一批 episode——不能各投各的轴。"""
    axis = _frozen_axis(transfer)
    return [episode_summary(record, surface, **axis)[0] for record in records]


# ------------------------------------------------------------ 检查项

def check_coverage_and_width(transfer: InteractionTransfer, groups: dict[str, list],
                             summaries: dict[str, list[dict[str, np.ndarray]]]
                             ) -> tuple[list[Check], float]:
    """第 1、2 项：coverage 与 width，允许集合在**冻结的校准集**上做 split conformal。

    `plan/03` §8.1 的目标函数是"在 coverage ≥ 90% 的约束下最小化 width"。未标定的
    描述性超水平集（τ = 0.95）做不到——实测抽屉只有 0.68~0.79。允许集合的大小只由
    一个参数 τ 控制，而这一族集合随 τ 嵌套，所以直接对 τ 做 split conformal：
    在校准集上取每条 episode"达到 95% 接触质量所需的最小 τ"的
    ``⌈(n+1)(1-α)⌉/n`` 分位数，得到 τ*。之后 **width 才是要报的数**，coverage 是
    被约束住的那一个。

    ⚠️ 给的是 exchangeability 下的 **marginal** coverage，不是任意子群的条件保证
    （D-59）。子群 coverage 另报，见第 3 项。
    """
    arrays = transfer.arrays
    region = _pooled_region(transfer)
    baked = transfer.meta.get("calibration", {})
    orders, cumulative = _nested_allowed(region)
    area = np.asarray(arrays["surface/area"], dtype=np.float64)
    total_area = float(area.sum())
    n_surface = region.shape[1]
    active = region.sum(axis=1) > 0

    lengths = _lengths_by_tau(cumulative)
    curves = {split: [_mass_inside_curve(_episode_region(item), orders, lengths)
                      for item in items] for split, items in summaries.items()}
    required = {split: np.array([_required_tau(c) for c in values if np.isfinite(c).any()])
                for split, values in curves.items()}

    def width_at(tau: float) -> float:
        if not np.isfinite(tau) or not active.any():
            return float("nan")
        allowed = _allowed_at(orders, cumulative, tau, n_surface)
        return float(np.mean((allowed[active] * area).sum(axis=1)) / total_area)

    def coverage_at(split: str, tau: float) -> float:
        values = required.get(split)
        if values is None or not len(values):
            return float("nan")
        return float(np.mean(values <= tau))

    checks: list[Check] = []
    tau_star = float("nan")
    calibration = required.get("calibration", np.array([]))
    held_out = [s for s in summaries if s not in ("train", "calibration")]

    # (a) 未标定的描述性集合，留作对照——它是 v1 报告里那一档。
    plain = {s: coverage_at(s, REGION_MASS_TARGET) for s in held_out}
    checks.append(Check(
        "1a region coverage（未标定 τ=0.95，仅作对照）", "DEFER",
        "；".join(f"{k} {v:.3f} (n={len(required[k])})" for k, v in sorted(plain.items()))
        + f"；width {width_at(REGION_MASS_TARGET)*100:.2f}%。"
        "描述性超水平集没有覆盖保证，达不到门槛属预期，判据看 1a'",
        {"per_split": plain, "width": width_at(REGION_MASS_TARGET)}))

    if baked.get("calibrated"):
        # artifact 自带标定（D-67 / D-71）。评估**必须**用它，不能自己重标一个——
        # 否则报的 coverage 不是 S6 实际会拿到的那个集合的 coverage。
        tau_star = float(baked["region_tau"])
    elif len(calibration) < 20:
        checks.append(Check("1a' region coverage（conformal）", "DEFER",
                            f"artifact 未标定，且校准集只有 {len(calibration)} 条"))
        checks.append(Check("2a region width", "DEFER", "τ* 未标定，width 无从谈起"))
        tau_star = float("nan")
    else:
        tau_star = _conformal_tau(calibration)
    if np.isfinite(tau_star):
        covered = {s: coverage_at(s, tau_star) for s in held_out}
        finite = [v for v in covered.values() if np.isfinite(v)]
        status = "PASS" if finite and min(finite) >= COVERAGE_GATE else "FAIL"
        # τ* 顶到 1.0 意味着允许集合就是整个表面：coverage 必然 100%，而 envelope
        # 没有约束任何东西。这种"通过"是空的，必须当失败报。
        if not np.isfinite(tau_star) or tau_star >= 1.0:
            status = "FAIL"
        checks.append(Check(
            "1a' region coverage（conformal）", status,
            ("artifact 自带标定；" if baked.get("calibrated") else "评估侧现场标定；")
            + f"τ*={tau_star:.3f}（校准 n={baked.get('num_episodes', len(calibration))}）；留出 "
            + "；".join(f"{k} {v:.3f} (n={len(required[k])})" for k, v in sorted(covered.items()))
            + f"；门槛 {COVERAGE_GATE}",
            {"tau_star": tau_star, "per_split": covered,
             "calibration_n": int(len(calibration))}))
        allowed = _allowed_at(orders, cumulative, tau_star, n_surface)
        checks.append(Check(
            "2a region width", "PASS",
            f"τ*={tau_star:.3f} 下允许区域占物体表面积 {width_at(tau_star)*100:.2f}%"
            f"（未标定时 {width_at(REGION_MASS_TARGET)*100:.2f}%）；"
            f"平均 {float(allowed[active].sum(axis=1).mean()) if active.any() else 0:.1f}/"
            f"{n_surface} 个 cell。**这才是要最小化的数**",
            {"allowed_area_fraction": width_at(tau_star),
             "allowed_area_fraction_uncalibrated": width_at(REGION_MASS_TARGET),
             "total_area_m2": total_area}))

    band = _mechanics_band_report(transfer, summaries)
    checks.append(Check(
        "1b mechanics band（只报数，不设门槛）", "DEFER",
        "10/90 分位数是描述性统计，不是 conformal guarantee（D-59/D-63）；"
        + "；".join(f"{k} 逐点带内 全部 {v['all']:.3f} / 支持≥5 {v['support>=5']:.3f}"
                    f" / 支持≥10 {v['support>=10']:.3f}、episode 全中 {v['simultaneous']:.3f}"
                    for k, v in sorted(band["per_split"].items()) if k != "train")
        + f"；cell 支持分布 {band['cell_support_histogram']}"
        + "。⚠️ 逐点数要求三分量**同时**命中，独立时就该是单轴的三次方（P-65）；"
          "有覆盖保证的是 episode 级那一列",
        band))
    checks.append(Check(
        "2b mechanics width", "PASS",
        f"traction 带相对宽度中位数 全部 {band['relative_width_median']:.3f} / "
        f"支持≥5 {band['relative_width_median_support5']:.3f}（与 1b 并读："
        "带很宽而带内率仍低，说明问题不是带窄了）",
        {"relative_width_median": band["relative_width_median"],
         "relative_width_median_support5": band["relative_width_median_support5"]}))
    return checks, tau_star


def _mechanics_band_report(transfer: InteractionTransfer,
                           summaries: dict[str, list[dict[str, np.ndarray]]]) -> dict:
    """traction 带的带内率与宽度，**按 cell 支持度分层报**。

    只有 1 条 episode 支持的 cell 上 q10 = q90，带宽恒为 0，留出 episode 几乎必然落在
    带外。不分层的话整体带内率会被这些 cell 拖低，看不出"支持足够的地方带够不够用"。
    """
    arrays = transfer.arrays
    low = np.asarray(arrays["mech/traction_obj/lo"], dtype=np.float64)
    high = np.asarray(arrays["mech/traction_obj/hi"], dtype=np.float64)
    median = np.asarray(arrays["mech/traction_obj/median"], dtype=np.float64)
    support = np.asarray(arrays["region/support"], dtype=np.int64)
    strata = {"all": support > 0, "support>=5": support >= 5, "support>=10": support >= 10}

    scale = np.linalg.norm(median, axis=2)
    relative = np.divide(np.linalg.norm(high - low, axis=2), scale,
                         out=np.full(scale.shape, np.nan), where=scale > 0)

    per_split: dict[str, dict] = {}
    for split, items in summaries.items():
        pointwise = {name: [] for name in strata}
        simultaneous = []
        for summary in items:
            traction = summary["traction"]
            finite = np.isfinite(traction).all(axis=2)
            inside = ((traction >= low - 1e-9) & (traction <= high + 1e-9)).all(axis=2)
            for name, mask in strata.items():
                live = mask & finite
                if live.any():
                    pointwise[name].append(float(inside[live].mean()))
            live_all = strata["all"] & finite
            if live_all.any():
                simultaneous.append(bool(inside[live_all].all()))
        if pointwise["all"]:
            per_split[split] = {
                **{name: (float(np.mean(values)) if values else float("nan"))
                   for name, values in pointwise.items()},
                "simultaneous": float(np.mean(simultaneous)) if simultaneous else float("nan"),
                "episodes": len(pointwise["all"]),
            }
    return {"per_split": per_split,
            "relative_width_median": (float(np.nanmedian(relative[strata["all"]]))
                                      if strata["all"].any() else float("nan")),
            "relative_width_median_support5": (float(np.nanmedian(relative[strata["support>=5"]]))
                                               if strata["support>=5"].any() else float("nan")),
            "cell_support_histogram": {
                "support==1": int((support == 1).sum()),
                "support>=5": int((support >= 5).sum()),
                "support>=10": int((support >= 10).sum()),
                "occupied": int((support > 0).sum())}}


def check_strategy_subgroups(transfer: InteractionTransfer,
                             summaries: dict[str, list[dict[str, np.ndarray]]],
                             families: dict[str, list[str]], tau_star: float) -> Check:
    """第 3 项：按 strategy family 分开报 coverage。

    `plan/02` §7 第 4 条要的是"envelope 不轻易泄漏策略身份"。单份聚合 artifact 没有
    可分类的标签，真正的风险是**聚合塌到某一族上**：整体 coverage 好看，而其余族被
    系统性排除。所以这一项报的是各族 coverage 的**落差**。
    """
    # ⚠️ 必须用第 1a' 项标定出来的**同一个** τ*。早先这里用未标定的 τ=0.95，
    # 于是 1a' 在标定集合上 PASS 而本项在另一个集合上 FAIL，两个数不可比。
    region = _pooled_region(transfer)
    if not np.isfinite(tau_star):
        return Check("3 策略子群 coverage", "DEFER", "τ* 未标定，子群 coverage 无从比较")
    orders, cumulative = _nested_allowed(region)
    allowed = _allowed_at(orders, cumulative, tau_star, region.shape[1])
    per_family: dict[str, list[float]] = defaultdict(list)
    for split, items in summaries.items():
        if split == "train":
            continue
        for family, summary in zip(families[split], items):
            episode = _episode_region(summary)
            weight = episode.sum(axis=1)
            rows = weight > 0
            if rows.any():
                per_family[family].append(
                    float((episode * allowed).sum(axis=1)[rows].sum() / weight[rows].sum()))
    if len(per_family) < 2:
        return Check("3 策略子群 coverage", "DEFER",
                     f"留出划分里只有 {len(per_family)} 个策略族，无法比较落差")
    # ⚠️ 必须与第 1a 项**同一个判据**（episode 级：接触质量 ≥95% 才算被覆盖）。
    # 早先这里报的是 mass-inside 的均值，于是 1a FAIL 而本项 PASS，两个数看起来矛盾
    # 其实测的根本不是一回事。均值作为辅助量并排报。
    coverage = {k: float(np.mean(np.asarray(v) >= REGION_MASS_TARGET))
                for k, v in sorted(per_family.items())}
    mean_mass = {k: float(np.mean(v)) for k, v in sorted(per_family.items())}
    spread = max(coverage.values()) - min(coverage.values())
    status = "PASS" if min(coverage.values()) >= COVERAGE_GATE else "FAIL"
    return Check("3 策略子群 coverage", status,
                 f"τ*={tau_star:.3f} 下 "
                 + "；".join(f"{k} {coverage[k]:.3f}（均值 {mean_mass[k]:.3f}, n={len(per_family[k])}）"
                             for k in coverage)
                 + f"；落差 {spread:.3f}。conformal 只保证 marginal coverage，"
                   "子群落差必须另报（D-59）",
                 {"per_family_coverage": coverage, "per_family_mean_mass": mean_mass,
                  "spread": spread, "counts": {k: len(v) for k, v in per_family.items()}})


def check_family_envelopes(records_by_family: dict[str, list[EpisodeRecord]],
                           surface: Surface, transfer: InteractionTransfer,
                           family_splits: dict[str, set[str]]) -> Check:
    """第 4 项：pooled envelope 与各族自建 envelope 的距离谱。

    **留出族也要一起比**。pooled envelope 只由 train 构造，所以"它离某个从未参与
    构造的族有多远"正是最该看的一列——标 ``*`` 的就是留出族。
    """
    axis = _frozen_axis(transfer)
    usable = {k: v for k, v in records_by_family.items() if len(v) >= 2}
    if len(usable) < 2:
        return Check("4 family envelope 距离", "DEFER",
                     f"只有 {len(usable)} 个族有 ≥2 条训练 episode")
    pooled = _pooled_region(transfer)
    active = pooled.sum(axis=1) > 0
    distances: dict[str, float] = {}
    for family, items in sorted(usable.items()):
        built = build_transfer(items, surface=surface, transfer_id=f"family-{family}", **axis)
        own = _pooled_region(built)
        rows = active & (own.sum(axis=1) > 0)
        distances[family] = (float(np.mean([_js_distance(pooled[b], own[b])
                                            for b in np.flatnonzero(rows)]))
                             if rows.any() else float("nan"))
    finite = [v for v in distances.values() if np.isfinite(v)]
    spread = float(max(finite) - min(finite)) if finite else float("nan")
    # 只报数：多大的落差算"塌到一族上"要靠下游成功率定，现在没有 E-I 可比。
    held_out = {k for k, splits in family_splits.items() if "train" not in splits}
    return Check("4 family envelope 距离", "DEFER",
                 "；".join(f"{k}{'*' if k in held_out else ''} {v:.3f}"
                           for k, v in distances.items())
                 + f"；落差 {spread:.3f}（* = 未参与构造的留出族；"
                   "判据要 S6 的下游成功率，现在只报数）",
                 {"per_family": distances, "spread": spread,
                  "held_out_families": sorted(held_out)})


def check_multimodality(transfer: InteractionTransfer,
                        summaries: dict[str, list[dict[str, np.ndarray]]]) -> Check:
    """第 5 项：聚合出来的指令是不是"谁都没做过的混血"。

    比较 pooled envelope 到各 episode 的 JS 距离与 **episode 两两之间**的 JS 距离。
    若示教分布是单峰的，质心到成员的距离应当**小于**成员之间的平均距离；若是多峰的，
    质心落在两峰之间的低密度谷里，两者会接近甚至反超。那正是 D-63 说的"真实数据显示
    严重多峰"时该换聚类/条件模型的信号。

    ⚠️ **参照量用两两平均，不用最近邻。** 最近邻距离随 episode 数增加必然趋于 0
    （任何连续分布都如此），拿它当分母会让比值随样本量单调放大，与多峰性无关。
    最近邻仍然报，但只作辅助，且必须与 episode 数一起读。
    """
    train = summaries.get("train", [])
    if len(train) < 3:
        return Check("5 多峰性", "DEFER", f"训练 episode 只有 {len(train)} 条，最近邻无意义")
    pooled = _pooled_region(transfer)
    regions = [_episode_region(s) for s in train]
    rng = np.random.default_rng(0)
    to_pooled, pairwise, to_neighbour = [], [], []
    for b in np.flatnonzero(pooled.sum(axis=1) > 0):
        rows = [r[b] for r in regions if r[b].sum() > 0]
        if len(rows) < 3:
            continue
        to_pooled.extend(_js_distance(pooled[b], row) for row in rows)
        # episode 多时两两全算是 O(N²)；固定种子抽样，保证可复现。
        picked = (rows if len(rows) <= 24
                  else [rows[i] for i in rng.choice(len(rows), 24, replace=False)])
        for i in range(len(picked)):
            others = [_js_distance(picked[i], picked[j])
                      for j in range(len(picked)) if j != i]
            pairwise.extend(others)
            to_neighbour.append(min(others))
    if not to_pooled:
        return Check("5 多峰性", "DEFER", "没有足够的共同占用格")
    pooled_mean = float(np.mean(to_pooled))
    pairwise_mean = float(np.mean(pairwise))
    neighbour_mean = float(np.mean(to_neighbour))
    ratio = pooled_mean / pairwise_mean if pairwise_mean > 0 else float("inf")

    support = np.asarray(transfer.arrays["region/support"])
    strong = support >= max(2, int(np.median(support[support > 0])) if (support > 0).any() else 2)
    concentration = np.asarray(transfer.arrays["engage/concentration"])[strong]
    mode = np.asarray(transfer.arrays["mode/prob"])[strong]
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.nansum(np.where(mode > 0, mode * np.log(mode), 0.0), axis=1)
    return Check("5 多峰性", "DEFER",
                 f"pooled→episode {pooled_mean:.3f} vs episode 两两 {pairwise_mean:.3f}"
                 f"（比值 {ratio:.2f}，>1 即质心落在示教之外）；最近邻 {neighbour_mean:.3f}"
                 f"（n={len(train)}，随样本量缩，只作辅助）；engage 集中度中位数 "
                 f"{float(np.median(concentration)) if concentration.size else float('nan'):.3f}"
                 f"；mode 熵中位数 {float(np.median(entropy)) if entropy.size else float('nan'):.3f}"
                 "。判据要 S6 的下游成功率，现在只报数",
                 {"pooled_to_episode": pooled_mean, "episode_pairwise": pairwise_mean,
                  "episode_to_neighbour": neighbour_mean, "ratio_vs_pairwise": ratio,
                  "train_episodes": len(train),
                  "engage_concentration_median": (float(np.median(concentration))
                                                  if concentration.size else None),
                  "mode_entropy_median": float(np.median(entropy)) if entropy.size else None})


def check_cross_implementation(records_by_implementation: dict[str, list[EpisodeRecord]],
                               surface: Surface, transfer: InteractionTransfer) -> list[Check]:
    """第 6 项 = `plan/02` §7 第 8 条：擦拭两种实现必须产生可互换的 envelope。"""
    usable = {k: v for k, v in records_by_implementation.items() if len(v) >= 2}
    if len(usable) < 2:
        return [Check("6 跨实现可互换", "DEFER",
                      f"数据里只有 {len(usable)} 种 implementation（非擦拭任务正常）")]
    axis = _frozen_axis(transfer)
    built = {name: build_transfer(items, surface=surface, transfer_id=f"impl-{name}", **axis)
             for name, items in sorted(usable.items())}
    names = sorted(built)
    regions = {n: _pooled_region(built[n]) for n in names}
    pairs = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            rows = np.flatnonzero((regions[a].sum(axis=1) > 0) & (regions[b].sum(axis=1) > 0))
            pairs[f"{a}|{b}"] = (float(np.mean([_js_distance(regions[a][r], regions[b][r])
                                                for r in rows])) if len(rows) else float("nan"))
    # 硬约束：产物里不得出现工具的存在与否/位姿/几何。字段名与 meta 一并查。
    payload_names = set(transfer.executor_arrays())
    forbidden = [n for n in payload_names
                 if any(token in n for token in ("tool", "impl", "plate", "gripper"))]
    checks = [Check("6a payload 无实现字段", "FAIL" if forbidden else "PASS",
                    ("payload 含实现相关字段 " + str(forbidden)) if forbidden
                    else "payload 里没有工具/实现/接触体字段",
                    {"implementations": {k: len(v) for k, v in usable.items()}})]
    # ⚠️ 距离只报数，**不判 PASS/FAIL**。D-69：`direct_wipe` 的记录里有一个每帧都在、
    # 力方差 2%、固定在板角的寄生接触（没被使用的黑板擦），稳定占 10% 的力。
    # 在它被重采修掉之前，这个距离既不能证明可互换，也不能证明不可互换——
    # 把采集缺陷写成"两种实现的功能交互不同"是本项目明令禁止的那一类事。
    checks.append(Check(
        "6b 两实现 envelope 距离（只报数）", "DEFER",
        "region JS 距离 " + "；".join(f"{k} {v:.3f}" for k, v in pairs.items())
        + "。⚠️ D-69：direct_wipe 记录含寄生接触（未使用的工具全程压在板角，占 ~10% 的力），"
          "该数在重采之前不可解读；最终判据是把一份 envelope 交给只会另一种实现的执行器，要 S6",
        {"pairwise_region_js": pairs}))
    return checks


def check_interface_invariance(records_by_family: dict[str, list[EpisodeRecord]],
                               surface: Surface, transfer: InteractionTransfer) -> list[Check]:
    """第 7 项 = `plan/02` §7 第 3、7 条：产物维度不得携带 source 侧的任何数量。

    分两条报，因为强弱形式的可得性不同：

    - **7a 结构不变量**（总能跑）：用两组互不相交的 episode 各建一份 envelope，
      维度必须逐个数组相同。它保证维度只由 (n_bins, n_surface) 决定；
    - **7b 接触体数量不变量**（要数据里真的有不同的 ``fields.n_bodies``）：
      S4 已经把多个 source body 合并掉了，同一个任务里 ``n_bodies`` 往往是常数，
      那时如实 DEFER，不得拿 7a 冒充它。
    """
    axis = _frozen_axis(transfer)
    everything = [record for items in records_by_family.values() for record in items]
    everything.sort(key=lambda r: str(r.meta["episode_id"]))

    def shapes(items: list[EpisodeRecord], tag: str) -> dict[str, tuple[int, ...]]:
        built = build_transfer(items, surface=surface, transfer_id=tag, **axis)
        return {name: tuple(np.asarray(value).shape)
                for name, value in built.executor_arrays().items()}

    checks: list[Check] = []
    if len(everything) >= 4:
        left, right = shapes(everything[::2], "half-a"), shapes(everything[1::2], "half-b")
        bad = sorted(name for name in left if left[name] != right[name])
        checks.append(Check("7a 结构不变量", "FAIL" if bad else "PASS",
                            f"两组互不相交示教产出的维度不同：{bad}" if bad else
                            f"{len(everything)} 条对半分，{len(left)} 个数组维度全部一致",
                            {"episodes": len(everything)}))
    else:
        checks.append(Check("7a 结构不变量", "DEFER",
                            f"训练 episode 只有 {len(everything)} 条，分不出两组"))

    by_bodies: dict[int, list[EpisodeRecord]] = defaultdict(list)
    for record in everything:
        by_bodies[int(record.meta.get("fields", {}).get("n_bodies", -1))].append(record)
    usable = {k: v for k, v in by_bodies.items() if k > 0 and len(v) >= 2}
    if len(usable) < 2:
        checks.append(Check("7b 接触体数量不变量", "DEFER",
                            f"训练集里 fields.n_bodies 只有 {sorted(by_bodies)}；"
                            "S4 已合并 source body，同任务内通常是常数，需跨任务/跨实现数据"))
        return checks
    grouped = {key: shapes(items, f"bodies-{key}") for key, items in sorted(usable.items())}
    reference = grouped[min(grouped)]
    bad = sorted({name for group in grouped.values() for name, shape in group.items()
                  if reference[name] != shape})
    checks.append(Check("7b 接触体数量不变量", "FAIL" if bad else "PASS",
                        f"维度随接触体数量变化：{bad}" if bad else
                        "；".join(f"n_bodies={k} {len(v)} 条" for k, v in sorted(usable.items()))
                        + f"，{len(reference)} 个数组维度全部一致",
                        {"groups": {str(k): len(v) for k, v in usable.items()}}))
    return checks


# ------------------------------------------------------------ 驱动

def _load_group(manifest_path: Path, manifest: dict, transfer: InteractionTransfer,
                splits: list[str], limit: int | None, existing_only: bool
                ) -> tuple[dict[str, list], dict[str, list[str]], dict[str, int]]:
    meta = transfer.meta
    want = (str(meta["task"]), str(meta["object"]), str(meta["geometry_variant"]),
            str(meta["surface"]["sha256"]))
    groups: dict[str, list] = {split: [] for split in splits}
    #  artifact 绑定在一份冻结 surface 上，几何变体各有各的点序，无法在同一条命令轴上
    #  打分。把被这条规则排除掉的成功 episode 数如实记下来，不能让它们悄悄消失。
    excluded: dict[str, int] = defaultdict(int)
    for entry in manifest["episodes"]:
        entry_meta = entry.get("meta", {})
        key = (str(entry.get("task")), str(entry_meta.get("object")),
               str(entry.get("geometry_variant", entry_meta.get("geometry_variant", "nominal"))),
               str(entry_meta.get("surface", {}).get("sha256")))
        split = str(entry.get("split"))
        if not entry.get("success") or split not in groups:
            continue
        if key != want:
            excluded[split] += 1
            continue
        if existing_only and not (manifest_path.parent / entry["path"]).exists():
            continue
        groups[split].append(entry)
    for split in groups:
        groups[split].sort(key=lambda e: str(e["episode_id"]))
        if limit:
            groups[split] = groups[split][:limit]
    families = {split: [str(e.get("meta", {}).get("strategy_family", "unknown"))
                        for e in entries] for split, entries in groups.items()}
    return groups, families, dict(excluded)


def _resolve_surface(explicit: Path | None, manifest_path: Path, manifest: dict,
                     transfer: InteractionTransfer) -> Surface:
    """优先用显式给的冻结 surface，其次用 manifest 里登记的，最后才现场重算。

    现场重算是 P-57 的风险来源：FPS 的 tie-breaking 会随 NumPy/BLAS 版本变，
    在另一台机器上得到的点序与数据里的 ``region/point_idx`` 对不上。hash 不符时
    下面的调用方会直接失败，但能不走这条路就不走。
    """
    if explicit:
        return load_surface(explicit)
    key = f"{transfer.meta['object']}/{transfer.meta['geometry_variant']}"
    info = manifest.get("surfaces", {}).get(key, {})
    candidate = manifest_path.parent / str(info.get("path", ""))
    if info.get("path") and candidate.exists():
        return load_surface(candidate)
    print(f"⚠️ manifest 没有登记 {key} 的冻结 surface，改为现场重算（P-57 风险）；"
          "先跑 tools/s5_freeze_surfaces.py", file=sys.stderr)
    return surface_for(str(transfer.meta["object"]), str(transfer.meta["geometry_variant"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--surface", type=Path, help="冻结 surface；跨环境运行时必须给")
    parser.add_argument("--splits", nargs="+",
                        default=["train", "calibration", "in_distribution_test",
                                 "unseen_strategy_test", "unseen_physics_test",
                                 "unseen_geometry_test", "unseen_implementation_test"])
    parser.add_argument("--limit", type=int, help="每个划分最多用多少条；只用于 smoke")
    parser.add_argument("--existing-only", action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    transfer = load_transfer(args.artifact)
    manifest_path = args.manifest.resolve()
    manifest = read_manifest(manifest_path)
    surface = _resolve_surface(args.surface, manifest_path, manifest, transfer)
    if surface.sha256 != str(transfer.meta["surface"]["sha256"]):
        raise SystemExit("surface hash 与 artifact 不一致；跨环境重新 FPS 会改点序（P-57）")

    groups, families, excluded = _load_group(manifest_path, manifest, transfer, args.splits,
                                             args.limit, args.existing_only)
    # 逐条投影完就丢，只按 (family / implementation) 各留至多 SUBSET_CAP 条 record，
    # 供第 4、6、7 项重建子集 envelope。否则整份数据集会同时驻留内存。
    #
    # ⚠️ 分组**跨全部划分**，不能只用 train：擦拭的 `direct_wipe` 373 条全部在
    # `unseen_implementation_test` 里（这是有意的留出设计），只看 train 的话
    # `plan/02` §7 第 8 条那项检查永远不会触发，而且会 DEFER 得像"数据不支持"。
    axis = _frozen_axis(transfer)
    summaries: dict[str, list[dict[str, np.ndarray]]] = {}
    by_family: dict[str, list[EpisodeRecord]] = defaultdict(list)
    by_implementation: dict[str, list[EpisodeRecord]] = defaultdict(list)
    family_splits: dict[str, set[str]] = defaultdict(set)
    for split, entries in groups.items():
        if not entries:
            continue
        collected = []
        for entry in entries:
            record = load_episode(manifest_path.parent / entry["path"])
            collected.append(episode_summary(record, surface, **axis)[0])
            family = str(record.meta.get("strategy_family", "unknown"))
            implementation = str(record.meta.get("implementation", "unknown"))
            family_splits[family].add(split)
            if len(by_family[family]) < SUBSET_CAP:
                by_family[family].append(record)
            if len(by_implementation[implementation]) < SUBSET_CAP:
                by_implementation[implementation].append(record)
        summaries[split] = collected

    checks, tau_star = check_coverage_and_width(transfer, groups, summaries)
    checks.append(check_strategy_subgroups(transfer, summaries, families, tau_star))
    checks.append(check_family_envelopes(by_family, surface, transfer, family_splits))
    checks.append(check_multimodality(transfer, summaries))
    checks.extend(check_cross_implementation(by_implementation, surface, transfer))
    checks.extend(check_interface_invariance(by_family, surface, transfer))
    if excluded:
        checks.append(Check(
            "8 跨几何 coverage", "DEFER",
            "以下成功 episode 因为几何变体不同（各有各的冻结 surface 与点序）无法在这条"
            "命令轴上打分：" + "；".join(f"{k} {v} 条" for k, v in sorted(excluded.items()))
            + "。跨几何要么各建各的 artifact（那就不再是「未见几何」），要么需要一张"
              "表面对应表；本步不做，如实记为未完成",
            {"excluded_by_geometry": excluded}))

    tally = {status: sum(1 for c in checks if c.status == status)
             for status in ("PASS", "FAIL", "DEFER")}
    header = [f"S5 envelope 评估 — {transfer.meta['transfer_id']}",
              f"artifact  {args.artifact}",
              f"manifest  {manifest_path}  ({manifest.get('dataset_name')})",
              f"episodes  " + "，".join(f"{k} {len(v)}" for k, v in sorted(groups.items()) if v),
              f"命令轴    phase budget {transfer.meta['aggregation']['phase_budget']}，"
              f"{transfer.meta['aggregation']['n_bins']} 格",
              "=" * 78]
    body = [check.line() for check in checks]
    footer = ["=" * 78,
              f"PASS {tally['PASS']}  FAIL {tally['FAIL']}  DEFER {tally['DEFER']}",
              "⚠️ 判据是退出码，不是这份文件的开头（P-55）。DEFER 的项目不得引用成通过。"]
    text = "\n".join(header + body + footer)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps(
        {"transfer_id": transfer.meta["transfer_id"],
         "artifact": str(args.artifact), "tally": tally,
         "checks": [{"name": c.name, "status": c.status, "detail": c.detail, "data": c.data}
                    for c in checks]},
        ensure_ascii=False, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")
    print(text)
    sys.exit(1 if tally["FAIL"] else 0)


if __name__ == "__main__":
    main()
