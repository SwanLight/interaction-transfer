#!/usr/bin/env python3
"""找出"搁着没动却被记成交互"的寄生接触。

这是 P-64 / D-69 的**结构性补救**。那次的情形是：擦拭 `direct_wipe` 不再抓黑板擦，
但工具本体仍在场景里、仍在接触过滤名单里，就那么压在板角上，被**逐帧**记成
"物体受到的交互"，稳定占 9~12% 的接触力。它不报错、不影响 effect（那个角上没有
污渍）、录像里也看不出来（一块静止的黑板擦看着本来就该是静止的），
只污染 region 与 mechanics——而那正是 envelope 的主体。

它足以让结论翻转：`direct_wipe` 在 tool-only envelope 下 coverage 变成 0.000，
照字面读就是"擦拭的两种实现产生的功能交互不可互换"，而 `plan/README` §5 的四宫格
正建立在可互换上。**把采集缺陷写成科学结论是本项目明令禁止的事**，所以这条检查
必须能自动跑，不能靠人想起来查。

判据（三条同时满足才算寄生）：

1. **接触率高**——该处在 ≥ ``--min-frame-rate`` 的有效帧里都有接触。真正的操作接触
   会随阶段建立和断开；
2. **力几乎没有方差**——法向力的变异系数 < ``--max-force-cv``。压着不动的东西给的是
   自身重量，恒定；
3. **位置几乎不动**——接触点在物体系里的位移 std < ``--max-pos-std``。

三条缺一不可：光看接触率会把"全程按住"的合法接触也报出来；光看力方差会把匀速滑移
误报。用**物体系**坐标是因为寄生体相对物体不动，而这正是 P-53/P-54 反复强调的
"这个量活在哪个坐标系"。

用法::

    PYTHONPATH=src python3 tools/s5_parasite_check.py /tmp/s4_wipe --sample 40
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from it.records import load_episode, read_manifest  # noqa: E402


def audit_episode(record, *, cell: float, min_frame_rate: float, max_force_cv: float,
                  max_pos_std: float) -> list[dict]:
    """把接触点按物体系位置粗分格，逐格判"是不是搁着没动的"。"""
    a = record.arrays
    valid = np.asarray(a["valid_s4"], dtype=bool)
    live = (np.asarray(a["region/valid"], dtype=bool)
            & np.asarray(a["region/on_surface"], dtype=bool)
            & (np.asarray(a["region/weight"], dtype=np.float64) > 0)
            & valid[:, None])
    if not live.any():
        return []
    frames, slots = np.nonzero(live)
    position = np.asarray(a["region/pos_obj"], dtype=np.float64)[frames, slots]
    force = np.linalg.norm(np.asarray(a["mech/force_obj"], dtype=np.float64)[frames, slots],
                           axis=1)
    n_valid = int(valid.sum())

    key = np.round(position / cell).astype(np.int64)
    buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for i, k in enumerate(map(tuple, key)):
        buckets[k].append(i)

    findings = []
    total_force = float(force.sum())
    for k, rows in buckets.items():
        rows = np.asarray(rows)
        occupied = len(np.unique(frames[rows]))
        rate = occupied / max(n_valid, 1)
        if rate < min_frame_rate:
            continue
        f = force[rows]
        mean = float(f.mean())
        cv = float(f.std() / mean) if mean > 0 else float("inf")
        spread = float(np.linalg.norm(position[rows].std(axis=0)))
        if cv > max_force_cv or spread > max_pos_std:
            continue
        findings.append({
            "centre_obj": [round(float(v), 4) for v in position[rows].mean(axis=0)],
            "frame_rate": round(rate, 4),
            "force_mean_N": round(mean, 4),
            "force_cv": round(cv, 4),
            "pos_std_m": round(spread, 5),
            "force_share": round(float(f.sum() / total_force), 4) if total_force > 0 else 0.0,
        })
    return findings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--sample", type=int, default=40, help="每个分组抽多少条")
    parser.add_argument("--cell", type=float, default=0.03, help="位置分格边长（m）")
    parser.add_argument("--min-frame-rate", type=float, default=0.95)
    parser.add_argument("--max-force-cv", type=float, default=0.05)
    parser.add_argument("--max-pos-std", type=float, default=0.05)
    parser.add_argument("--max-force-share", type=float, default=0.01,
                        help="寄生接触占总接触力的比例超过它就判 FAIL")
    parser.add_argument("--existing-only", action="store_true",
                        help="只读本地实际存在的 episode；用于下载样本 smoke")
    args = parser.parse_args()

    failed = False
    for dataset in args.datasets:
        manifest_path = (dataset / "manifest.json").resolve()
        manifest = read_manifest(manifest_path)
        groups: dict[str, list] = defaultdict(list)
        for entry in manifest["episodes"]:
            if args.existing_only and not (manifest_path.parent / entry["path"]).exists():
                continue
            if entry.get("success"):
                meta = entry.get("meta", {})
                groups[f"{meta.get('implementation', '?')}/{entry.get('strategy_family', '?')}"
                       ].append(entry)
        print(f"=== {dataset}  ({manifest.get('dataset_name')})")
        for name, entries in sorted(groups.items()):
            entries = sorted(entries, key=lambda e: str(e["episode_id"]))[:args.sample]
            hits: list[dict] = []
            for entry in entries:
                record = load_episode(manifest_path.parent / entry["path"])
                hits.extend(audit_episode(
                    record, cell=args.cell, min_frame_rate=args.min_frame_rate,
                    max_force_cv=args.max_force_cv, max_pos_std=args.max_pos_std))
            if not entries:
                continue
            if not hits:
                print(f"  [PASS] {name:34s} n={len(entries):3d}  没有恒定静止接触")
                continue
            share = float(np.mean([h["force_share"] for h in hits]))
            episodes = len({tuple(h["centre_obj"]) for h in hits})
            status = "FAIL" if share > args.max_force_share else "WARN"
            failed |= status == "FAIL"
            print(f"  [{status}] {name:34s} n={len(entries):3d}  "
                  f"{len(hits)} 处恒定静止接触，占总接触力 {share:.1%}，"
                  f"{episodes} 个不同位置")
            for h in sorted(hits, key=lambda x: -x["force_share"])[:3]:
                print(f"           位置 {h['centre_obj']}  接触率 {h['frame_rate']:.3f}  "
                      f"力 {h['force_mean_N']:.3f} N (cv {h['force_cv']:.3f})  "
                      f"位移 std {h['pos_std_m']*1000:.1f} mm  力占比 {h['force_share']:.1%}")
    print()
    if failed:
        print("[FAIL] 有分组含显著的寄生接触——那些力不属于功能交互，"
              "envelope 的 region/mechanics 会被污染（P-64 / D-69）")
        sys.exit(1)
    print("[PASS] 没有分组的寄生接触超过阈值")


if __name__ == "__main__":
    main()
