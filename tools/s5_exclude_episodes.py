#!/usr/bin/env python3
"""把一批 episode 移出数据集范围，并把理由写进 manifest。

用途是**范围决定**，不是清洗数据。两者的区别必须留下痕迹：范围决定是"这一档任务
变体本身不该做"，清洗数据是"这一档结果不好看所以拿掉"。后者正是 P-11 / P-13 禁止的。
所以这个工具强制要求 ``--reason``，并把它连同时间、匹配条件一起写进 manifest 的
``exclusions`` 列表，谁都能查到某一批数据是什么时候、因为什么被移出去的。

**不删文件。** 被标记的 episode 仍在磁盘上，只是 ``split`` 被改成
``excluded_<tag>``；下游一律跳过 ``excluded`` 前缀的划分。这样决定可复核、可撤销，
而对所有消费者的效果与删掉完全一样。真要腾空间再单独删，那是另一件事。

用法::

    PYTHONPATH=src python3 tools/s5_exclude_episodes.py \\
        --manifest /tmp/s4_wipe/manifest.json \\
        --implementation direct --tag out_of_scope \\
        --reason "用户决定：擦拭只保留持工具实现"
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from it.records import read_manifest  # noqa: E402

EXCLUDED_PREFIX = "excluded_"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--implementation", help="按 meta.implementation 匹配")
    parser.add_argument("--strategy-family", help="按 strategy_family 匹配")
    parser.add_argument("--tag", default="out_of_scope",
                        help="新划分名是 excluded_<tag>")
    parser.add_argument("--reason", required=True,
                        help="为什么移出范围。**必填**：范围决定与"
                             "'结果不好看就拿掉'必须能被后人分辨")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not (args.implementation or args.strategy_family):
        raise SystemExit("至少要给 --implementation 或 --strategy-family 之一")

    manifest_path = args.manifest.resolve()
    manifest = read_manifest(manifest_path)
    target = f"{EXCLUDED_PREFIX}{args.tag}"

    matched = []
    for entry in manifest["episodes"]:
        meta = entry.get("meta", {})
        if args.implementation and str(meta.get("implementation")) != args.implementation:
            continue
        if args.strategy_family and str(entry.get("strategy_family")) != args.strategy_family:
            continue
        matched.append(entry)
    if not matched:
        raise SystemExit("没有 episode 匹配上；先用 --dry-run 核对条件")

    before = Counter(str(e.get("split")) for e in matched)
    print(f"匹配 {len(matched)} 条，原划分分布：{dict(before)}")
    already = sum(1 for e in matched if str(e.get("split", "")).startswith(EXCLUDED_PREFIX))
    if already:
        print(f"其中 {already} 条已经被排除过，将保持不变")
    if args.dry_run:
        print("--dry-run：没有写回")
        return

    for entry in matched:
        if not str(entry.get("split", "")).startswith(EXCLUDED_PREFIX):
            entry["previous_split"] = entry.get("split")
            entry["split"] = target
    manifest.setdefault("exclusions", []).append({
        "split": target,
        "reason": args.reason,
        "match": {"implementation": args.implementation,
                  "strategy_family": args.strategy_family},
        "episodes": len(matched),
        "previous_splits": {k: v for k, v in sorted(before.items())},
    })
    # manifest 顶层的 splits 统计要跟着改，否则它与 episodes 表自相矛盾。
    manifest["splits"] = dict(Counter(str(e.get("split")) for e in manifest["episodes"]))

    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(manifest_path)
    print(f"已写回 {manifest_path}")
    print(f"新的划分分布：{manifest['splits']}")


if __name__ == "__main__":
    main()
