"""采集板朝向标记的一致性核对（`plan/06` §7 的人工检查配套）。

板上两个标记都**只有视觉、没有碰撞**（`build_assets.PlateCfg`）：
工作面上的浅色贴片说"这一面是接触面"，深色鳍长在**局部 +Y** 那条边上、
说"这条边朝上"。它们唯一的用途是让人在录像里读出板的姿态。

**两块板的标记必须指向同一个世界方向。** 不然录像里看着像其中一块翻了
180°，标记就废了——用户正是这么发现 P-48 的。而绕工作面法向的滚转是个
自由度：`quat_from_frame` 只钉住法向和局部 X，+Y 是叉乘掉出来的，
两块板一旦面对面，+Y 必然相反。

这个量**不影响任何物理结论**（长方体绕主轴 180° 自映射，板位姿又是
`source/*` 审计字段、被隔离在模型输入之外），所以它不是数据质量判据，
而是**产物可读性**判据。但它不能只靠肉眼看几条录像抽查——本文件把它
固化成可重跑的核对。

用法::

    PYTHONPATH=src /isaac-sim/python.sh tools/s3_plate_orient.py \\
        /tmp/s3_knob /tmp/s3_wipe /tmp/s3_drawer /tmp/s3_probe/block
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from it.records import load_episode, read_manifest  # noqa: E402


def local_y(q: np.ndarray) -> np.ndarray:
    """四元数 (…,4) -> 局部 +Y 的世界方向（深色鳍所在的边）。"""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([2 * (x * y - w * z), 1 - 2 * (x * x + z * z),
                     2 * (y * z + w * x)], axis=-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", help="数据集根目录，可给多个")
    ap.add_argument("--min-cos", type=float, default=0.5,
                    help="两块板 +Y 方向余弦的下限；低于它判为标记互相矛盾")
    ap.add_argument("--per-family", type=int, default=3)
    a = ap.parse_args()

    print("两块板的 +Y（深色鳍所在边）方向余弦。+1 = 完全一致，-1 = 反了 180°")
    print(f"{'数据集':<22}{'家族/原语':<18}{'余弦':>7}   鳍朝（世界系）")
    bad: list[str] = []
    total = 0
    for root_s in a.roots:
        root = Path(root_s)
        man = read_manifest(root / "manifest.json")
        name = man.get("dataset_name", root.name)
        seen: dict[str, list] = {}
        for e in man["episodes"]:
            f = e["strategy_family"]
            if len(seen.get(f, [])) >= a.per_family:
                continue
            rec = load_episode(root / e["path"])
            keep = np.asarray(rec.arrays["phase"]) == 2
            v = []
            for p in (0, 1):
                key = f"source/plate{p}/root_pose"
                if key not in rec.arrays:
                    break
                q = np.asarray(rec.arrays[key])[keep, 3:7]
                m = local_y(q).mean(0)
                v.append(m / (np.linalg.norm(m) + 1e-9))
            if len(v) == 2:
                seen.setdefault(f, []).append((float(np.dot(v[0], v[1])), v[0]))
        for f, rows in sorted(seen.items()):
            cos = float(np.mean([r[0] for r in rows]))
            m = rows[0][1]
            total += 1
            flag = " " if cos >= a.min_cos else "!"
            print(f"{flag}{name:<21}{f:<18}{cos:>7.2f}   "
                  f"({m[0]:+.2f},{m[1]:+.2f},{m[2]:+.2f})")
            if cos < a.min_cos:
                bad.append(f"{name}/{f} 余弦 {cos:+.2f}")

    print(f"\n共 {total} 个家族/原语，标记互相矛盾的 {len(bad)} 个")
    for b in bad:
        print(f"  [FAIL] {b}")
    print(f"[{'PASS' if not bad else 'FAIL'}] 两块板的朝向标记一致（P-48 / P-49）")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
