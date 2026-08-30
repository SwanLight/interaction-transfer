#!/usr/bin/env python3
"""mechanics 的**允许集合该长什么形状**——用 coverage vs volume 实测决定，不靠规格拍板。

背景：v2 把每个 (命令格, cell) 的 traction 存成物体系下三个分量各自的 10/90 分位数。
评估时报的"带内率 0.24~0.68"曾被读成"这种表示不行"，但那是**指标的问题**：它要求
三轴同时落在各自的边缘区间里，而逐轴实测是 0.75~0.89（名义 0.80），三轴独立时联合
就该是 0.8³≈0.51。边缘刻度是准的，**从来没有人构造过一个联合区域**。

所以真正待定的不是"用不用三维力"（坐标只是换基，执行器拿 `surface/normals_obj`
自己就能转），而是**允许集合的形状**。这个脚本把它当成一个可测的问题：

对每一族候选集合，用**同一套** split conformal 在冻结校准集上标定一个标量 k，
使 episode 级 coverage 达到同一个目标（≥95% 的力加权接触点落在集合内的 episode
占 90%），然后在冻结测试集上比 **coverage 与体积**。同一 coverage 下体积最小的赢。
这正是 `plan/03` §8.1 那句"在 coverage ≥ 90% 的约束下最小化 width"用在集合形状上。

候选：

===============  =========================================================
``box_obj``      物体系轴对齐盒（v2 现状）
``box_surface``  以 cell 外法向为一轴的局部系轴对齐盒
``ellipsoid``    以逐 cell 协方差定的马氏椭球（能表达分量间相关）
``cone``         法向区间 × 切向幅值区间 × 切向方向锥（D-58 写的那种）
===============  =========================================================

**四族都只是假设。** 谁赢由数据说了算；输的那几族的数字一并落盘，免得下一个人
重新猜一遍。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from it.records import load_episode, read_manifest  # noqa: E402
from it.surfaces import load_surface  # noqa: E402
from it.transfer import episode_summary, load_transfer  # noqa: E402

FORMS = ("box_obj", "box_surface", "ellipsoid", "cone")
#: episode 被判为"落在集合内"所需的力加权接触点比例。与 region 子指标同一个口径
#: （`plan/03` §8.1），这样两个数才可比。
POINT_MASS_TARGET = 0.95
#: 校准目标：这么大比例的校准 episode 要被覆盖。
TARGET_COVERAGE = 0.90
#: k 的搜索网格。集合随 k 单调变大，所以可以直接对 k 做 split conformal。
K_GRID = np.concatenate([np.linspace(0.25, 6.0, 116), [8.0, 12.0, 20.0]])
_EPS = 1e-12


def tangent_basis(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """给每个 cell 造一组切向基。

    ⚠️ 用固定参考轴叉乘会在法向与参考轴平行时退化（P-49 的同一个形状：离散选择
    落进回路里）。这里按法向的最小分量选参考轴，保证处处良态，且只算一次。
    """
    n = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), _EPS)
    reference = np.zeros_like(n)
    reference[np.arange(len(n)), np.argmin(np.abs(n), axis=1)] = 1.0
    u = np.cross(n, reference)
    u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), _EPS)
    return u, np.cross(n, u)


class SetFamily:
    """一族随 k 单调膨胀的集合。``score`` 返回"要多大的 k 才装得下这个点"。"""

    def __init__(self, name: str, centre: np.ndarray, scale: np.ndarray,
                 basis: np.ndarray | None = None, extra: dict | None = None):
        self.name, self.centre, self.scale, self.basis = name, centre, scale, basis
        self.extra = extra or {}

    def score(self, values: np.ndarray) -> np.ndarray:
        """(B,S,3) -> (B,S)：装下每个点所需的最小 k。"""
        delta = values - self.centre
        if self.basis is not None:
            delta = np.einsum("skj,bsj->bsk", self.basis, delta)
        if self.name == "ellipsoid":
            return np.sqrt(np.maximum(np.einsum("bsi,bsij,bsj->bs", delta,
                                                self.extra["inverse"], delta), 0.0))
        if self.name == "cone":
            normal, tangent = delta[..., 0], delta[..., 1:]
            magnitude = np.linalg.norm(values_tangent(values, self.basis), axis=-1)
            reference = self.extra["tangent_magnitude"]
            angle = angular_gap(values_tangent(values, self.basis), self.extra["direction"])
            return np.maximum.reduce([
                np.abs(normal) / np.maximum(self.scale[..., 0], _EPS),
                np.abs(magnitude - reference) / np.maximum(self.scale[..., 1], _EPS),
                angle / np.maximum(self.extra["angle_scale"], _EPS),
            ])
        return np.max(np.abs(delta) / np.maximum(self.scale, _EPS), axis=-1)

    def log_volume(self, k: float) -> np.ndarray:
        """每个 cell 的集合体积取对数（体积跨几个数量级，线性平均没有意义）。"""
        if self.name == "ellipsoid":
            return (np.log(4.0 / 3.0 * np.pi) + 3.0 * np.log(k)
                    + 0.5 * self.extra["log_det"])
        if self.name == "cone":
            half = np.clip(k * self.extra["angle_scale"], _EPS, np.pi)
            inner = np.maximum(self.extra["tangent_magnitude"] - k * self.scale[..., 1], 0.0)
            outer = self.extra["tangent_magnitude"] + k * self.scale[..., 1]
            area = np.maximum(half * (outer ** 2 - inner ** 2), _EPS)
            return np.log(2.0 * k * np.maximum(self.scale[..., 0], _EPS)) + np.log(area)
        return np.sum(np.log(np.maximum(2.0 * k * self.scale, _EPS)), axis=-1)


def values_tangent(values: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return np.einsum("skj,bsj->bsk", basis[:, 1:], values)


def angular_gap(tangent: np.ndarray, direction: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(tangent, axis=-1)
    unit = tangent / np.maximum(norm, _EPS)[..., None]
    cosine = np.clip(np.einsum("bsk,bsk->bs", unit, direction), -1.0, 1.0)
    # 幅值为零时方向无意义，不该因此被判出界。
    return np.where(norm > _EPS, np.arccos(cosine), 0.0)


def _quiet_nan(function, *args, **kwargs):
    """整列都是 NaN 的 cell 是合法情形（没人碰过），由 support 掩码表达。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return function(*args, **kwargs)


def resolve_floor(samples: np.ndarray, absolute: float, relative: float) -> float:
    """尺度地板。

    ⚠️ 用一个绝对值当地板会把结论带偏：训练集上离散度接近 0 的 cell（低支持、
    或那一格的力本来就很稳）拿到 floor 之后，任何偏差除以它都是巨大的分数，
    标定出来的 k 就被这些 cell 绑架。实测用 1 N/m² 的绝对地板时，擦拭的 k 是 38、
    椭球 327、锥 956——那不是形状的差别，是地板的差别。

    改为**相对于本任务 traction 量级**取地板：与各 cell 中位数幅值的中位数挂钩。
    """
    centre = np.nan_to_num(_quiet_nan(np.nanmedian, samples, axis=0))
    magnitude = np.linalg.norm(centre, axis=-1)
    live = magnitude > 0
    reference = float(np.median(magnitude[live])) if live.any() else 0.0
    return float(max(absolute, relative * reference))


def build_families(samples: np.ndarray, weights: np.ndarray, normals: np.ndarray,
                   floor: float) -> dict[str, SetFamily]:
    """从训练集的逐 cell 样本构造四族集合的中心与形状（尚未标定 k）。

    ``samples`` (E,B,S,3) 含 NaN（该 episode 没碰过这个 cell），一律按"只用碰过的
    episode"统计——与 D-66 同一条口径。
    """
    centre = np.nan_to_num(_quiet_nan(np.nanmedian, samples, axis=0))
    spread = np.nan_to_num(_quiet_nan(np.nanquantile, samples, 0.90, axis=0)
                           - _quiet_nan(np.nanquantile, samples, 0.10, axis=0))
    # 尺度地板：只有一条 episode 支持的 cell 上分位数会收成 0，除以它会得到 inf。
    scale = np.maximum(spread / 2.0, floor)

    u, v = tangent_basis(normals)
    basis = np.stack([normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True),
                                           _EPS), u, v], axis=1)          # (S,3,3)

    families: dict[str, SetFamily] = {
        "box_obj": SetFamily("box_obj", centre, scale),
        "box_surface": SetFamily("box_surface", centre,
                                 _surface_scale(samples, centre, basis, floor), basis),
    }

    b, s = centre.shape[:2]
    covariance = np.zeros((b, s, 3, 3))
    inverse = np.zeros_like(covariance)
    log_det = np.zeros((b, s))
    flat = samples.reshape(len(samples), b * s, 3)
    for i in range(b * s):
        column = flat[:, i, :]
        column = column[np.isfinite(column).all(axis=1)]
        bi, si = divmod(i, s)
        if len(column) < 4:
            cov = np.eye(3) * (floor ** 2)
        else:
            cov = np.cov(column.T) + np.eye(3) * (floor ** 2)
        covariance[bi, si] = cov
        inverse[bi, si] = np.linalg.inv(cov)
        log_det[bi, si] = float(np.linalg.slogdet(cov)[1])
    families["ellipsoid"] = SetFamily("ellipsoid", centre, scale,
                                      extra={"inverse": inverse, "log_det": log_det})

    families["cone"] = _cone_family(samples, centre, basis, floor)
    del weights
    return families


def _surface_scale(samples: np.ndarray, centre: np.ndarray, basis: np.ndarray,
                   floor: float) -> np.ndarray:
    delta = samples - centre
    local = np.einsum("skj,ebsj->ebsk", basis, delta)
    spread = np.nan_to_num(_quiet_nan(np.nanquantile, local, 0.90, axis=0)
                           - _quiet_nan(np.nanquantile, local, 0.10, axis=0))
    return np.maximum(spread / 2.0, floor)


def _cone_family(samples: np.ndarray, centre: np.ndarray, basis: np.ndarray,
                 floor: float) -> SetFamily:
    local = np.einsum("skj,ebsj->ebsk", basis, samples)
    normal_spread = np.nan_to_num(_quiet_nan(np.nanquantile, local[..., 0], 0.90, axis=0)
                                  - _quiet_nan(np.nanquantile, local[..., 0], 0.10, axis=0))
    tangent = local[..., 1:]
    magnitude = np.linalg.norm(tangent, axis=-1)
    magnitude_centre = np.nan_to_num(_quiet_nan(np.nanmedian, magnitude, axis=0))
    magnitude_spread = np.nan_to_num(_quiet_nan(np.nanquantile, magnitude, 0.90, axis=0)
                                     - _quiet_nan(np.nanquantile, magnitude, 0.10, axis=0))
    unit = tangent / np.maximum(magnitude, _EPS)[..., None]
    resultant = np.nan_to_num(_quiet_nan(np.nanmean, unit, axis=0))
    length = np.linalg.norm(resultant, axis=-1)
    direction = resultant / np.maximum(length, _EPS)[..., None]
    # 圆形标准差：集中度越低，方向锥的天然尺度越大。
    angle_scale = np.maximum(np.sqrt(np.maximum(-2.0 * np.log(np.clip(length, _EPS, 1.0)),
                                                0.0)), 1e-3)
    scale = np.stack([np.maximum(normal_spread / 2.0, floor),
                      np.maximum(magnitude_spread / 2.0, floor)], axis=-1)
    return SetFamily("cone", centre, scale, basis,
                     extra={"tangent_magnitude": magnitude_centre,
                            "direction": direction, "angle_scale": angle_scale})


def episode_required_k(family: SetFamily, traction: np.ndarray, mass: np.ndarray,
                       support: np.ndarray) -> float:
    """这条 episode 要多大的 k 才能让 95% 的力加权接触点落进集合。"""
    live = support & np.isfinite(traction).all(axis=2) & (mass > 0)
    if not live.any():
        return float("nan")
    scores = family.score(np.nan_to_num(traction))[live]
    weights = mass[live]
    order = np.argsort(scores)
    cumulative = np.cumsum(weights[order]) / weights.sum()
    index = int(np.searchsorted(cumulative, POINT_MASS_TARGET))
    return float(scores[order][min(index, len(order) - 1)])


def conformal_k(required: np.ndarray, target: float = TARGET_COVERAGE) -> float:
    finite = np.sort(required[np.isfinite(required)])
    n = len(required)
    if n == 0:
        return float("nan")
    rank = int(np.ceil((n + 1) * target))
    return float(finite[rank - 1]) if rank <= len(finite) else float("inf")


def collect(entries, root: Path, surface, axis: dict) -> tuple[np.ndarray, np.ndarray]:
    traction, mass = [], []
    for entry in entries:
        summary, _ = episode_summary(load_episode(root / entry["path"]), surface, **axis)
        traction.append(summary["traction"])
        mass.append(np.nan_to_num(summary["region"], nan=0.0))
    return np.stack(traction), np.stack(mass)


def run_task(name: str, dataset: Path, artifact: Path, *, limit: int, floor: float,
             floor_relative: float) -> dict:
    manifest_path = (dataset / "manifest.json").resolve()
    manifest = read_manifest(manifest_path)
    transfer = load_transfer(artifact)
    key = f"{transfer.meta['object']}/{transfer.meta['geometry_variant']}"
    surface = load_surface(manifest_path.parent / manifest["surfaces"][key]["path"])
    axis = {"n_bins": int(transfer.meta["aggregation"]["n_bins"]),
            "n_surface": int(transfer.meta["surface"]["command_n_points"]),
            "budget": tuple(transfer.meta["aggregation"]["phase_budget"])}
    want = str(transfer.meta["surface"]["sha256"])

    def pick(split: str) -> list:
        rows = [e for e in manifest["episodes"]
                if e.get("success") and e.get("split") == split
                and str(e.get("meta", {}).get("surface", {}).get("sha256")) == want]
        rows.sort(key=lambda e: str(e["episode_id"]))
        return rows[:limit]

    train, calibration = pick("train"), pick("calibration")
    tests = {s: pick(s) for s in ("in_distribution_test", "unseen_physics_test",
                                  "unseen_strategy_test")}
    if len(calibration) < 20:
        raise SystemExit(f"{name}: 校准集只有 {len(calibration)} 条，标定不可靠")

    train_traction, train_mass = collect(train, manifest_path.parent, surface, axis)
    support = np.asarray(transfer.arrays["region/support"]) > 0
    normals = np.asarray(transfer.arrays["surface/normals_obj"], dtype=np.float64)
    resolved = resolve_floor(train_traction, floor, floor_relative)
    families = build_families(train_traction, train_mass, normals, resolved)
    del train_traction, train_mass

    calibration_traction, calibration_mass = collect(calibration, manifest_path.parent,
                                                     surface, axis)
    test_data = {s: collect(rows, manifest_path.parent, surface, axis)
                 for s, rows in tests.items() if rows}

    result = {"task": name, "train": len(train), "calibration": len(calibration),
              "scale_floor_N_per_m2": resolved,
              "point_mass_target": POINT_MASS_TARGET, "target_coverage": TARGET_COVERAGE,
              "forms": {}}
    for form, family in families.items():
        required = np.array([episode_required_k(family, calibration_traction[i],
                                                calibration_mass[i], support)
                             for i in range(len(calibration))])
        k = conformal_k(required)
        coverage = {}
        for split, (traction, mass) in test_data.items():
            values = np.array([episode_required_k(family, traction[i], mass[i], support)
                               for i in range(len(traction))])
            coverage[split] = float(np.mean(values <= k)) if len(values) else float("nan")
        log_volume = family.log_volume(k)[support]
        result["forms"][form] = {
            "k": k,
            "coverage": coverage,
            "median_log_volume": float(np.median(log_volume)),
            "median_linear_extent_N_per_m2": float(np.exp(np.median(log_volume) / 3.0)),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", required=True,
                        help="name=dataset_dir=artifact.npz")
    parser.add_argument("--limit", type=int, default=400, help="每个划分最多用多少条")
    parser.add_argument("--floor", type=float, default=1.0,
                        help="尺度地板的绝对下限（N/m²）")
    parser.add_argument("--floor-relative", type=float, default=0.05,
                        help="尺度地板取本任务 traction 量级的这个比例（见 resolve_floor）")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    results = []
    for spec in args.tasks:
        name, dataset, artifact = spec.split("=", 2)
        results.append(run_task(name, Path(dataset), Path(artifact),
                                limit=args.limit, floor=args.floor,
                                floor_relative=args.floor_relative))

    lines = ["mechanics 允许集合的形状比较（同一 coverage 目标下比体积）",
             "=" * 88,
             "标定：冻结校准集上 split conformal 定 k，使 ≥90% 的 episode 有 ≥95% 的",
             "      力加权接触点落进集合。之后在冻结测试集上报 coverage，并比体积。",
             "体积用**立方根**（N/m²）报，因为它是三维集合的线性尺度，可直接与 traction 幅值比。",
             ""]
    for item in results:
        lines.append(f"--- {item['task']}  train {item['train']} / calib {item['calibration']}"
                     f"  尺度地板 {item['scale_floor_N_per_m2']:.1f} N/m²")
        lines.append(f"{'form':14s} {'k':>6s} {'尺度(N/m²)':>12s}  " + "  ".join(
            f"{s.replace('_test',''):>16s}" for s in sorted(
                next(iter(item['forms'].values()))['coverage'])))
        for form, value in item["forms"].items():
            lines.append(f"{form:14s} {value['k']:6.2f} "
                         f"{value['median_linear_extent_N_per_m2']:12.1f}  " + "  ".join(
                             f"{value['coverage'][s]:16.3f}"
                             for s in sorted(value["coverage"])))
        lines.append("")
    text = "\n".join(lines)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    args.out.with_suffix(".json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
