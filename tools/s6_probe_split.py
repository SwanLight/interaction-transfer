#!/usr/bin/env python3
"""把探针数据集的 calibration 划分**按 (物体, 原语) 分层**重划，供 S6 造指令用。

为什么要动它。探针数据集的 train/calibration 是在**物体**这一层按比例切的
（每个物体约 10% 进校准）。而 S6 的指令要按 **(物体, 原语)** 构造——"戳球"和
"滚球"是两种不同的交互，混进一份 envelope 里得到的是一个没有物理意义的平均。
按原语拆开之后，每组分到的校准条数就掉到 3~27，44 组里只有 24 组够 split conformal
的下限（`transfer.build_transfer` 要求 ≥20，低于此宁可输出未标定 artifact 也不给假保证）。

**这不是"调数据让结果好看"**，三条边界：

1. **只在 train 与 calibration 之间移动**，任何 `*_test` 划分一个都不碰——
   评估集保持冻结（P-13 的规矩针对的正是评估集）；
2. **在任何 E-I 训练之前做**，此刻不存在任何可以被倒推的结果；
3. **确定性**：按 `sha256(episode_id)` 排序取前 N 条，与运行顺序、文件系统顺序无关，
   谁重跑都得到同一份划分。

不够条数的组**如实报告并排除**，不降低门槛（那会让 conformal 的保证名存实亡）。

排除之后核对 D-41 的硬规则：**每条原语至少两个几何不同的物体承载**。注意这里有
**两个不同的声称**，不能混为一谈：

- **数据集的原语覆盖**（`03` §2.4 声称的那个）——由 S3 采到的成功轨迹决定，
  本脚本一条都不改，永远成立；
- **可标定指令集的原语覆盖**——只数那些校准条数够做 split conformal 的组。
  它可能比前者窄，而 E-I 就是在它上面预训练的。D-41 那条"至少两个几何不同的物体"
  的**理由**（不让执行器学到物体专用的捷径）对后者同样适用，所以窄下去必须被看见。

某条原语在后者上只剩一个承载物体时脚本非零退出，除非用 ``--accept-thin`` 显式列出来
——这样"知情地接受"和"悄悄丢掉"就分得开，而且**新出现的**变窄仍然会让闸门变红。

用法::

    tools/s6_probe_split.py /tmp/s4_probe --target 30 --min-train 12 --apply
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from it.records import apply_splits, read_manifest  # noqa: E402

#: 只有这两个划分参与重划。其余（各种 *_test、failed）逐字保留。
MOVABLE = ("train", "calibration")


def _order_key(episode_id: str) -> str:
    """确定性排序键。用 hash 而不是 episode_id 本身，避免按采集批次顺序取，
    那会让校准集整块来自最后几个 batch。"""
    return hashlib.sha256(episode_id.encode("utf-8")).hexdigest()


def restratify(manifest: dict, *, target: int, min_train: int) -> tuple[dict, list[dict]]:
    """返回 (新 manifest, 每组的报告)。"""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    frozen: dict[str, list[str]] = defaultdict(list)
    for entry in manifest["episodes"]:
        split = str(entry.get("split"))
        episode_id = str(entry["episode_id"])
        if split not in MOVABLE or not entry.get("success"):
            frozen[split].append(episode_id)
            continue
        obj = str(entry.get("meta", {}).get("object", "?"))
        groups[(obj, str(entry.get("strategy_family", "?")))].append(entry)

    new_splits: dict[str, list[str]] = {k: list(v) for k, v in frozen.items()}
    new_splits.setdefault("train", [])
    new_splits.setdefault("calibration", [])
    report = []
    for (obj, family), entries in sorted(groups.items()):
        entries.sort(key=lambda e: _order_key(str(e["episode_id"])))
        available = len(entries)
        take = min(target, max(0, available - min_train))
        usable = take >= 20
        for index, entry in enumerate(entries):
            bucket = "calibration" if index < take else "train"
            new_splits[bucket].append(str(entry["episode_id"]))
        report.append({
            "object": obj, "strategy_family": family, "available": available,
            "calibration": take, "train": available - take, "usable": usable,
        })
    return apply_splits(manifest, new_splits), report


def check_primitive_coverage(report: list[dict]) -> dict[str, list[str]]:
    """D-41 的硬规则：每条原语至少两个几何不同的物体承载。"""
    carriers: dict[str, list[str]] = defaultdict(list)
    for row in report:
        if row["usable"]:
            carriers[row["strategy_family"]].append(row["object"])
    return {k: sorted(v) for k, v in carriers.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="/tmp/s4_probe（下面每个物体一个子目录）")
    parser.add_argument("--target", type=int, default=30,
                        help="每 (物体, 原语) 组的校准条数目标")
    parser.add_argument("--min-train", type=int, default=12,
                        help="重划后每组至少留多少条训练；不够就少取校准")
    parser.add_argument("--apply", action="store_true",
                        help="不给则只报告，不写盘")
    parser.add_argument("--accept-thin", nargs="*", default=[], metavar="原语",
                        help="已知并接受只剩 1 个承载物体的原语。**必须在 decisions.md "
                             "里有对应记录**；列在这里只是让闸门不因为已知情况长红，"
                             "新出现的变窄照样报错")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    manifests = sorted(args.root.glob("*/manifest.json"))
    if not manifests:
        raise SystemExit(f"{args.root} 下没有 */manifest.json")

    lines = ["探针数据集 calibration 划分按 (物体, 原语) 分层重划",
             "=" * 78,
             f"目标每组 {args.target} 条校准，训练至少留 {args.min_train} 条。",
             "**只在 train 与 calibration 之间移动，任何 *_test 划分一条都不动。**",
             "确定性：按 sha256(episode_id) 排序取前 N 条。",
             ""]
    all_report: list[dict] = []
    for path in manifests:
        manifest = read_manifest(path)
        updated, report = restratify(manifest, target=args.target, min_train=args.min_train)
        all_report += report
        if args.apply:
            path.write_text(json.dumps(updated, ensure_ascii=False, indent=2,
                                       sort_keys=True) + "\n", encoding="utf-8")
        for row in report:
            lines.append("  %-8s %-12s 可用 %4d → 校准 %3d / 训练 %4d  %s"
                         % (row["object"], row["strategy_family"], row["available"],
                            row["calibration"], row["train"],
                            "" if row["usable"] else "← 条数不足，排除出指令集"))

    usable = [r for r in all_report if r["usable"]]
    dropped = [r for r in all_report if not r["usable"]]
    carriers = check_primitive_coverage(all_report)
    thin = {k: v for k, v in carriers.items() if len(v) < 2}

    lines += ["",
              "-" * 78,
              f"可用组 {len(usable)} / {len(all_report)}，"
              f"合计训练 {sum(r['train'] for r in usable)} 条、"
              f"校准 {sum(r['calibration'] for r in usable)} 条",
              ""]
    if dropped:
        lines.append("被排除的组（条数不足，**不降低 conformal 门槛**）：")
        for row in dropped:
            lines.append("  %-8s %-12s 只有 %d 条可移动"
                         % (row["object"], row["strategy_family"], row["available"]))
        lines.append("")
    lines.append(f"原语承载核对（D-41：每条原语 ≥2 个几何不同的物体），共 {len(carriers)} 条原语：")
    for family, objects in sorted(carriers.items()):
        lines.append("  %-12s %d 个承载物体：%s"
                     % (family, len(objects), "、".join(objects)))
    accepted = {k: v for k, v in thin.items() if k in set(args.accept_thin)}
    unexpected = {k: v for k, v in thin.items() if k not in set(args.accept_thin)}
    lines.append("")
    if accepted:
        lines += ["⚠️ 以下原语在**可标定指令集**上只剩 1 个承载物体，已知情接受"
                  "（数据集本身的覆盖不变）：",
                  *[f"  {k}：{v}" for k, v in sorted(accepted.items())],
                  "  E-I 在这条原语上只见过一种几何，S7 的 leave-one-primitive 分析"
                  "必须把它单列（D-60）。"]
    if unexpected:
        lines += ["❌ 以下原语只剩 <2 个承载物体，且**没有**被 --accept-thin 列出：",
                  *[f"  {k}：{v}" for k, v in sorted(unexpected.items())],
                  "  要么调 --target / --min-train，要么在 decisions.md 记一条再列进来。"]
    if not thin:
        lines.append("✅ 每条原语在可标定指令集上都仍有 ≥2 个几何不同的物体承载。")
    lines.append("")
    lines.append("（未加 --apply，本次只报告，没有写盘）" if not args.apply
                 else "已写回各 manifest。")

    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        args.out.with_suffix(".json").write_text(
            json.dumps({"groups": all_report, "carriers": carriers},
                       ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    sys.exit(1 if unexpected else 0)


if __name__ == "__main__":
    main()
