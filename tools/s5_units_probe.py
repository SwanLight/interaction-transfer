#!/usr/bin/env python3
"""三个"自定义量"的单位与刻度体检：它们依赖了哪些**不属于物理**的东西？

背景。S5 的 payload 里每一个字段都是我们自己定义出来的。单元测试查得了"实现有没有
bug"，查不了"这个量的定义对不对"——P-68 / P-69 / P-70 三个洞全部是在所有测试、
独立验收和泄漏检查**都通过**的情况下存活下来的。这个脚本把那个查不了的问题变成
一张可以被反驳的表：**给每个量列出它的非物理依赖，然后实测那个依赖有多大。**

三节，各自有明确判据：

**一、traction 依赖表面采样分辨率吗（P-68）**
    ``N/m²`` 读起来像压强，于是没人再问那个 m² 是从哪来的。把命令表面在
    {64, 256, 1024} 三档之间换，同一批示教上 traction 的中位数应当**不变**——
    分辨率是我们选的，物体受到的压强不是。四种池化做法放在同一张表里比：
    漂移比 ``max/min`` ≤ ``--drift-tol``（默认 1.5×）为 PASS。
    这一节同时是 D-72 选型的依据，**输的三族的数字一并落盘**。

**二、mode 标签依赖阈值吗（P-69）**
    ``mode/prob`` 的四档是用滑移速度阈值 ``SLIP_SPEED_MIN`` 切出来的。把阈值在
    {1, 2, 3, 5, 10} mm/s 之间换，"黏住"占的**力**比例摆动多少个百分点？
    摆动大 = 这个标签基本是阈值的产物，下游若只拿得到切完的标签，那个任意约定
    就成了研究结论的一部分。判据不是"摆动必须小"（物理上本来就可能落在模糊带里），
    而是**payload 必须同时带连续量**——脚本据此检查 artifact 而不是数据。

**三、effect 的两路能放进同一个范数吗（P-70）**
    ``effect/rigid`` 是 6 维 (平移 m, 旋转 rad)。每个任务只有一路非零，两路量级
    差一个数量级——下游 ``r_effect`` 若直接取 6 维 L2，就是 D-31 第 2 个洞（量纲
    失衡）藏在"统一接口"里面。报三个数：平移 p90、旋转 p90、以及把 (dp, dr) 作用
    在冻结 surface 点上得到的**表面平均位移**（米，任务无关）。判据是后者对三个
    任务落在同一个数量级。

**四、核本身的不变量**（能失败的对拍，不是恒等式——P-60）
    合力守恒、同部件、以及"核带宽不随分辨率变"。

用法::

    tools/s5_units_probe.py \
        --task drawer=/tmp/s4_drawer=/tmp/s5/drawer/drawer-drawer-nominal-train.npz \
        --out out/s5/units_probe.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from it.interaction import SLIP_SPEED_MIN  # noqa: E402
from it.records import load_episode, read_manifest  # noqa: E402
from it.surfaces import LEVELS, SCATTER_K, SCATTER_SIGMA, load_surface  # noqa: E402
from it.transfer import (  # noqa: E402
    TRACTION_POOLING, TRACTION_POOLINGS, _episode_summary, load_transfer,
    phase_budget, rigid_surface_displacement, surface_metric)

#: 分辨率扫描用哪几档。4096 太慢且与 256 的结论一致，默认不扫。
SWEEP_LEVELS = (64, 256, 1024)
#: mode 阈值敏感度扫的候选（m/s）。5 mm/s 是现用值。
SLIP_GRID = (1.0e-3, 2.0e-3, 3.0e-3, 5.0e-3, 1.0e-2)


def _records(dataset: Path, split: str, limit: int):
    manifest = read_manifest(dataset / "manifest.json")
    # ⚠️ 划分取**每条 episode 自己的 `split` 字段**，不取顶层的 `manifest["splits"]`。
    # 后者在四份数据集之间格式并不一致：抽屉/旋钮是名字列表，而擦拭被
    # `s5_exclude_episodes.py`（D-70 移出 direct_wipe）改写成了**计数**。
    # 照名字列表读会在擦拭上抛 `'int' object is not iterable`——服务器上实际炸过一次。
    # `s5_eval_envelope.py::_load_group` 一直用的就是逐 episode 字段，这里对齐它。
    taken = 0
    for entry in manifest["episodes"]:
        if str(entry.get("split")) != split or not entry.get("success"):
            continue
        # 本机只放了 smoke 子集，所以**先筛存在再截断**——反过来会在全量划分的
        # 前 limit 条都不在本机时得到空集合，而那看起来像"数据集是空的"。
        if not (dataset / entry["path"]).exists():
            continue
        record = load_episode(dataset / entry["path"])
        if not record.meta.get("success", False):
            continue
        yield record
        taken += 1
        if taken >= limit:
            return


def traction_resolution(records, surface, budget, n_bins):
    """一、每种池化做法在三档分辨率上的 traction 中位数（N/m²）。"""
    out = {}
    for pooling in TRACTION_POOLINGS:
        row = {}
        for level in SWEEP_LEVELS:
            if level not in surface.parent:
                continue
            values = []
            for record in records:
                summary, _ = _episode_summary(record, surface, budget=budget,
                                              n_bins=n_bins, n_surface=level,
                                              pooling=pooling)
                traction = summary["traction"]
                finite = np.isfinite(traction).all(axis=-1)
                values.append(np.linalg.norm(traction[finite], axis=-1))
            pooled = np.concatenate(values) if values else np.zeros(1)
            row[level] = float(np.median(pooled))
        span = list(row.values())
        # 也报**去掉最粗那一档**的漂移。理由不是为了让数字好看（判据仍然用全档），
        # 而是这两个数的差本身有信息：全档漂移大而细档漂移小，说明残差来自
        # "一个粗格里同时装下了好几个互不相干的接触斑块"（按 |f| 平均时会被高force
        # 的那个拉高），那是**粗粒化本身的损失**，与 P-68 的 1/面积 标度是两回事。
        fine = [row[n] for n in sorted(row)[1:]]
        out[pooling] = {
            "median_N_per_m2": row,
            "drift": float(max(span) / max(min(span), 1e-9)) if span else float("nan"),
            "drift_without_coarsest": (float(max(fine) / max(min(fine), 1e-9))
                                       if len(fine) > 1 else float("nan")),
        }
    return out


def moment_resolution(records, surface, budget, n_bins):
    """粗粒化残差 moment_density 的分辨率标度——它**应该**随格子变小而变小。

    单列它是为了不让下一个人把"moment 也漂移"当成第二个 P-68：力臂随 cell 边长走
    是这个量的定义决定的，不是实现错误。报出来，S4.5 扫描时才知道要归一化它。
    """
    row = {}
    for level in SWEEP_LEVELS:
        if level not in surface.parent:
            continue
        values = []
        for record in records:
            summary, _ = _episode_summary(record, surface, budget=budget,
                                          n_bins=n_bins, n_surface=level)
            moment = summary["moment_density"]
            finite = np.isfinite(moment).all(axis=-1)
            values.append(np.linalg.norm(moment[finite], axis=-1))
        pooled = np.concatenate(values) if values else np.zeros(1)
        area = np.bincount(surface.parent[level],
                           weights=np.asarray(surface.area, dtype=np.float64),
                           minlength=level)
        row[level] = {"median_N_per_m": float(np.median(pooled)),
                      "cell_pitch_mm": float(np.sqrt(np.median(area)) * 1e3)}
    return row


def mode_threshold(records):
    """二、"黏住"占的力比例随滑移速度阈值怎么变。"""
    slip, force = [], []
    for record in records:
        arrays = record.arrays
        valid = np.asarray(arrays["valid_s4"], dtype=bool)
        active = (np.asarray(arrays["mode/label"]) > 0) & valid[:, None]
        slip.append(np.asarray(arrays["mode/pose_slip"], dtype=np.float64)[active])
        force.append(np.linalg.norm(
            np.asarray(arrays["mech/force_obj"], dtype=np.float64), axis=-1)[active])
    slip = np.concatenate(slip) if slip else np.zeros(0)
    force = np.concatenate(force) if force else np.zeros(0)
    total = force.sum()
    share = {t: float(force[slip <= t].sum() / total) if total > 0 else float("nan")
             for t in SLIP_GRID}
    values = list(share.values())
    return {
        "stick_force_share": {f"{t * 1e3:g}mm_s": v for t, v in share.items()},
        "swing_pp": float((max(values) - min(values)) * 100.0),
        "current_threshold_mm_s": SLIP_SPEED_MIN * 1e3,
        "slip_p50_mm_s": float(np.median(slip) * 1e3) if len(slip) else float("nan"),
    }


def effect_scale(records, surface):
    """三、effect 两路的量级，以及任务无关的表面平均位移（米）。"""
    metric = surface_metric(surface)
    translation, rotation, displacement, cleaned = [], [], [], []
    for record in records:
        arrays = record.arrays
        valid = (np.asarray(arrays["valid_s4"], dtype=bool)[:, None]
                 & np.asarray(arrays["effect/future_valid"], dtype=bool))
        rigid = np.asarray(arrays["effect/rigid"], dtype=np.float64)[valid]
        if not len(rigid):
            continue
        translation.append(np.linalg.norm(rigid[:, :3], axis=-1))
        rotation.append(np.linalg.norm(rigid[:, 3:], axis=-1))
        displacement.append(rigid_surface_displacement(rigid, metric))
        # surface_state 是被擦掉的 **dirt 格数**。换算成面积要用 dirt 网格的格边长
        # （episode meta 里的 cell_m），不是表面采样 cell 的面积——后者算出来会比
        # 整块板还大。这一步只在诊断里做，payload 用无量纲刻度。
        state = np.asarray(arrays["effect/surface_state"], dtype=np.float64)[valid]
        dirt_cell = float(record.meta.get("cell_m", 0.01)) ** 2
        cleaned.append(np.abs(state).sum(axis=-1) * dirt_cell)
    percentile = lambda parts: (float(np.percentile(np.concatenate(parts), 90))  # noqa: E731
                                if parts else float("nan"))
    return {
        "translation_p90_m": percentile(translation),
        "rotation_p90_rad": percentile(rotation),
        "surface_displacement_p90_m": percentile(displacement),
        "cleaned_area_p90_m2": percentile(cleaned),
    }


def kernel_invariants(surface):
    """四、核的三条不变量。每一条都能失败——这不是恒等式对拍（P-60）。"""
    index, weight = surface.scatter_kernel()
    part = np.asarray(surface.part)
    probe = np.zeros(surface.n_points)
    probe[::37] = 1.0                    # 稀疏探针力，覆盖全表面
    scattered = np.bincount(index.ravel(),
                            weights=(weight * probe[:, None]).ravel(),
                            minlength=surface.n_points)
    return {
        "row_sum_max_error": float(np.abs(weight.sum(axis=1) - 1.0).max()),
        "force_conservation_error": float(abs(scattered.sum() - probe.sum())),
        "cross_part_weight": float(weight[part[index] != part[:, None]].sum()),
        "sigma_m": SCATTER_SIGMA,
        "neighbours": SCATTER_K,
    }


def run_task(name: str, dataset: Path, artifact: Path | None, *, surface_path: Path | None,
             limit: int, n_bins: int) -> dict:
    manifest = read_manifest(dataset / "manifest.json")
    records = list(_records(dataset, "train", limit))
    if len(records) < 2:
        raise SystemExit(f"{name}: train split 里成功 record 不足 2 条")
    meta = records[0].meta
    obj, geom = str(meta["object"]), str(meta.get("geometry_variant", "nominal"))
    if surface_path is None:
        info = manifest.get("surfaces", {}).get(f"{obj}/{geom}", {})
        surface_path = dataset / info["path"] if info.get("path") else None
    if surface_path is None:
        raise SystemExit(f"{name}: 找不到冻结 surface（P-57：不得现场重算）")
    surface = load_surface(surface_path)
    if surface.sha256 != str(meta.get("surface", {}).get("sha256")):
        raise SystemExit(f"{name}: 冻结 surface hash 与 record 不一致")
    budget = phase_budget(records, surface, n_bins=n_bins)

    payload_has_continuous_mode = None
    payload_has_effect_scale = None
    if artifact is not None and artifact.exists():
        arrays = load_transfer(artifact).executor_arrays()
        payload_has_continuous_mode = any(key.startswith("mode/slip_speed")
                                          for key in arrays)
        payload_has_effect_scale = ("effect/rigid/metric" in arrays
                                    and "effect/rigid/scale_m" in arrays
                                    and "effect/surface_state/scale" in arrays)
    return {
        "task": name,
        "object": obj,
        "num_episodes": len(records),
        "n_bins": n_bins,
        "traction": traction_resolution(records, surface, budget, n_bins),
        "moment": moment_resolution(records, surface, budget, n_bins),
        "mode": mode_threshold(records),
        "effect": effect_scale(records, surface),
        "kernel": kernel_invariants(surface),
        "payload_has_continuous_mode": payload_has_continuous_mode,
        "payload_has_effect_scale": payload_has_effect_scale,
    }


def render(results: list[dict], drift_tol: float) -> tuple[str, int]:
    lines, failures = [], [0]
    lines += ["自定义量的单位与刻度体检（P-68 / P-69 / P-70）",
              "=" * 92,
              "每一节问同一句话：这个量依赖了哪些不属于物理的东西，那个依赖有多大？",
              ""]

    lines += ["一、traction 依赖表面采样分辨率吗？",
              "-" * 92,
              "  问句：把命令表面从 64 格换到 1024 格，同一批示教的 traction 中位数该不该变？",
              "  怎么读：分辨率是我们选的，物体受到的压强不是——所以**不该变**。",
              f"  判据：**现行做法**（{TRACTION_POOLING}）漂移比 max/min ≤ {drift_tol:g}×。",
              "        另外三族是**对照臂**：它们漂移是这个实验的结论、不是本次运行的",
              "        失败，所以标「对照」而不计退出码——一条永远红的闸门等于没有闸门。",
              "        反过来，对照臂**不**漂移才要红：那说明这批数据触不到 P-68，",
              "        整张表失去区分能力。",
              ""]
    for item in results:
        levels = sorted(next(iter(item["traction"].values()))["median_N_per_m2"])
        lines.append(f"  --- {item['task']}（{item['num_episodes']} 条示教）")
        lines.append("      " + f"{'池化做法':<16s}" + "".join(f"{f'{n} 格':>13s}" for n in levels)
                     + f"{'漂移':>10s}{'去掉最粗档':>12s}   判定")
        for pooling, value in item["traction"].items():
            drift = value["drift"]
            if pooling == TRACTION_POOLING:
                ok = drift <= drift_tol
                mark = "PASS" if ok else "FAIL"
                failures[0] += 0 if ok else 1
            else:
                degenerate = drift <= drift_tol
                mark = "对照（无区分力→FAIL）" if degenerate else "对照"
                failures[0] += 1 if degenerate else 0
            fine_drift = value["drift_without_coarsest"]
            fmt = lambda d: (f"{d:8.2f}×" if d < 1e4 else f"{'>10000':>8s}×")  # noqa: E731
            lines.append("      " + f"{pooling:<16s}"
                         + "".join(f"{value['median_N_per_m2'][n]:13.1f}" for n in levels)
                         + fmt(drift) + f"{fmt(fine_drift):>12s}" + f"   {mark}")
        lines.append("")
    lines += ["  注：`nearest_area` 是出问题的那一版，列在这里是为了让这张表**能失败**。",
              "      四族里只有 `kernel_forcew` 通得过——这就是选它的全部理由（D-72）。",
              "      `kernel_area` 值得单看：它已经在用核了，照样漂 3~11×。",
              "      **核散射是必要条件不是充分条件，池化那一步也得治。**",
              ""]

    lines += ["  moment_density 的分辨率标度（**预期就会变**，单列以免被误读成第二个 P-68）",
              "  -" * 46]
    for item in results:
        cells = ", ".join(f"{n} 格 pitch {v['cell_pitch_mm']:.1f} mm → "
                          f"{v['median_N_per_m']:.1f} N/m"
                          for n, v in sorted(item["moment"].items()))
        lines.append(f"      {item['task']:8s} {cells}")
    lines += ["      力臂随 cell 边长走是这个量的定义决定的：它是 cell 内相对代表点的",
              "      粗粒化残差，不是物理场。实测还**不单调**（格子变小同时压低力臂、",
              "      又把一个斑块分到更多格里）。所以它跨分辨率不可比，S4.5 扫描时必须",
              "      单列，也不能照着 traction 那条判据去要求它。",
              ""]

    lines += ["二、mode 标签依赖阈值吗？",
              "-" * 92,
              "  问句：把 stick/slide 的滑移速度阈值在 1~10 mm/s 之间换，"
              "「黏住」占的力比例摆动多少？",
              "  怎么读：摆动大 = 这个标签基本是阈值的产物。",
              "  判据：**不是**「摆动必须小」（接触本来就可能落在模糊带里），",
              "        而是 payload 必须同时带连续量，让下游能自己重判（P-69）。",
              ""]
    grid = [f"{t * 1e3:g}mm_s" for t in SLIP_GRID]
    lines.append("      " + f"{'任务':<10s}" + "".join(f"{g:>12s}" for g in grid)
                 + f"{'摆动':>10s}  {'连续量进 payload':>18s}")
    for item in results:
        mode = item["mode"]
        has = item["payload_has_continuous_mode"]
        flag = {True: "是  PASS", False: "否  FAIL", None: "（无 artifact）"}[has]
        failures[0] += 1 if has is False else 0
        lines.append("      " + f"{item['task']:<10s}"
                     + "".join(f"{mode['stick_force_share'][g]:12.3f}" for g in grid)
                     + f"{mode['swing_pp']:8.1f}pp  {flag:>18s}")
    lines.append("")

    lines += ["三、effect 的两路能放进同一个范数吗？",
              "-" * 92,
              "  问句：`effect/rigid` 是 6 维 (平移 m, 旋转 rad)。下游若直接取 L2，",
              "        米和弧度就被当成同一种东西加起来了。",
              "  怎么读：前两列每个任务**只有一路非零**、量级差一个数量级 = 不能直接取范数。",
              "        第三列把 (dp, dr) 作用在冻结 surface 点上取表面平均位移——同一个",
              "        公式、单位是米，抽屉与旋钮因此可比。第四列是另一路",
              "        `effect/surface_state` 的物理量级（被擦掉的面积）。",
              "  判据：**刚体一路**有信号的任务（抽屉/旋钮）在第三列同数量级。",
              "        擦拭第三列为零不是缺陷：它的物体是运动学固定的平面，effect 全在",
              "        第四列——这正是 r_effect 要两路各自刻度、而不是一个 6 维范数的原因。",
              ""]
    lines.append("      " + f"{'任务':<10s}{'平移 p90 (m)':>15s}{'旋转 p90 (rad)':>16s}"
                 f"{'表面位移 p90 (m)':>19s}{'擦净面积 p90 (m²)':>20s}")
    for item in results:
        effect = item["effect"]
        flag = {True: "   两路刻度进 payload：是  PASS",
                False: "   两路刻度进 payload：否  FAIL"}.get(
                    item["payload_has_effect_scale"], "")
        failures[0] += 1 if item["payload_has_effect_scale"] is False else 0
        lines.append("      " + f"{item['task']:<10s}"
                     f"{effect['translation_p90_m']:15.4f}"
                     f"{effect['rotation_p90_rad']:16.4f}"
                     f"{effect['surface_displacement_p90_m']:19.4f}"
                     f"{effect['cleaned_area_p90_m2']:20.5f}" + flag)
    lines.append("")

    lines += ["四、散射核自己的不变量（每条都能失败）",
              "-" * 92]
    lines.append("      " + f"{'任务':<10s}{'行和误差':>14s}{'合力守恒误差':>16s}"
                 f"{'跨部件权重':>14s}{'带宽 σ (mm)':>14s}")
    for item in results:
        kernel = item["kernel"]
        bad = (kernel["row_sum_max_error"] > 1e-9
               or kernel["force_conservation_error"] > 1e-9
               or kernel["cross_part_weight"] > 0.0)
        failures[0] += 1 if bad else 0
        lines.append("      " + f"{item['task']:<10s}"
                     f"{kernel['row_sum_max_error']:14.2e}"
                     f"{kernel['force_conservation_error']:16.2e}"
                     f"{kernel['cross_part_weight']:14.2e}"
                     f"{kernel['sigma_m'] * 1e3:14.1f}")
    lines += ["      带宽是**固定物理尺度**，与表面采样分辨率无关——这正是第一节能通过的原因。",
              ""]

    lines += ["=" * 92,
              f"FAIL {failures[0]} 项" if failures[0] else "全部通过"]
    return "\n".join(lines), failures[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", dest="tasks", action="append", required=True,
                        metavar="名字=数据集[=artifact]",
                        help="可给多次；artifact 用于检查 payload 是否带连续 mode")
    parser.add_argument("--surface", action="append", default=[], metavar="[名字=]路径",
                        help="冻结 surface；不给则从数据集 manifest 的 surfaces.path 找。"
                             "本机跑时 manifest 里没有 path，用 名字=路径 逐任务指定")
    parser.add_argument("--limit", type=int, default=40,
                        help="每个任务最多用多少条 train episode（分辨率扫描很慢）")
    parser.add_argument("--n-bins", type=int, default=32)
    parser.add_argument("--drift-tol", type=float, default=1.5)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    surfaces: dict[str | None, Path] = {}
    for item in args.surface:
        key, _, value = item.rpartition("=")
        surfaces[key or None] = Path(value)

    results = []
    for spec in args.tasks:
        parts = spec.split("=")
        name, dataset = parts[0], Path(parts[1])
        artifact = Path(parts[2]) if len(parts) > 2 else None
        results.append(run_task(name, dataset, artifact,
                                surface_path=surfaces.get(name, surfaces.get(None)),
                                limit=args.limit, n_bins=args.n_bins))

    text, failures = render(results, args.drift_tol)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    args.out.with_suffix(".json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(text)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
