"""交互原语库的**张满与冗余**核对（`plan/03` §2.4）。

这是探针物体集唯一的验收判据。两条，缺一不可：

1. **数量**：每条原语 ≥200 条成功轨迹（`plan/03` §6）；
2. **冗余**：每条原语由 **≥2 个几何结构不同的物体**承载。

第 2 条是 D-41 的核心。物体集不是照着留出任务反推的这件事，靠的就是它——
满足冗余之后，"你是照着任务 X 设计物体 Y 的"这句指控失去力量，
因为删掉 Y 那一格仍然被别的物体覆盖着。

**张不满就报张不满。** 把某一格从分类学里删掉以求"全绿"，等于把分类学改成
"我们做得到的那些"，而分类学独立于我们的实现正是它全部的意义。

用法::

    python3 tools/s3_coverage.py /tmp/s3_probe
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

#: 原语顺序与 `plan/03` §2.4.4 的表一致
PRIMITIVE_ORDER = (
    "press", "push", "slide_push", "rub", "shear", "poke", "pivot", "roll",
    "crank", "slide_along", "hook_pull", "twist", "pinch_hold", "pinch_move",
    "pinch_turn",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="探针数据集根目录，其下每个物体一个子目录")
    ap.add_argument("--min-success", type=int, default=200)
    ap.add_argument("--min-objects", type=int, default=2)
    a = ap.parse_args()

    total = collections.Counter()
    good = collections.Counter()
    carriers: dict[str, set[str]] = collections.defaultdict(set)
    n_all = n_ok = 0
    manifests = sorted(Path(a.root).glob("*/manifest.json"))
    if not manifests:
        print(f"{a.root} 下没有找到任何 manifest.json")
        return 2

    for m in manifests:
        d = json.loads(m.read_text(encoding="utf-8"))
        obj = d.get("probe_object", m.parent.name)
        for e in d["episodes"]:
            p = e["strategy_family"]
            total[p] += 1
            n_all += 1
            if e["success"]:
                good[p] += 1
                n_ok += 1
                carriers[p].add(obj)

    out = [
        "S3 探针物体集 · 交互原语库覆盖核对",
        f"{len(manifests)} 个物体，{n_all} 条 episode，成功 {n_ok} "
        f"({100.0 * n_ok / max(n_all, 1):.1f}%)",
        "",
        f"判据：每条原语成功 ≥{a.min_success} 条，且由 ≥{a.min_objects} "
        f"个几何不同的物体承载",
        "",
        f"{'':<4}{'原语':<14}{'总数':>7}{'成功':>7}{'物体':>6}  承载物体",
    ]
    n_full = 0
    unmet: list[str] = []
    for p in PRIMITIVE_ORDER:
        n_obj = len(carriers[p])
        enough, redundant = good[p] >= a.min_success, n_obj >= a.min_objects
        if enough and redundant:
            mark, n_full = "OK  ", n_full + 1
        elif enough or redundant:
            mark = "!!  "
            unmet.append(f"{p}（{'数量不足' if not enough else '只有一个承载物体'}）")
        else:
            mark = "XX  "
            unmet.append(f"{p}（数量与冗余都不满足）")
        out.append(f"{mark}{p:<14}{total[p]:>7}{good[p]:>7}{n_obj:>6}  "
                   f"{sorted(carriers[p]) if carriers[p] else '—'}")

    out += ["", f"两条判据都满足：{n_full}/{len(PRIMITIVE_ORDER)}",
            "!! = 只满足一条    XX = 两条都不满足"]
    if unmet:
        out += ["", "未满足的格（不得因为它们不好看就从分类学里删掉，见 D-41）："]
        out += [f"  - {u}" for u in unmet]
    text = "\n".join(out) + "\n"
    print(text)
    (Path(a.root) / "coverage.txt").write_text(text, encoding="utf-8")
    return 0 if not unmet else 1


if __name__ == "__main__":
    sys.exit(main())
