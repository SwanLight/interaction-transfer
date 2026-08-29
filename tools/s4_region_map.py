"""把 Oracle Record 的 region 热图画成图——录像替代不了它。

`plan/06` §7 要求每次评估人工看一遍。录像看得到板在动，**看不到接触落在
物体的哪一片、力怎么分布**，而那正是 `plan/02` §3.2 的 region 字段本身。
S3 的 `s3_dirt_map.py`（擦拭的污渍）和 `s3_knob_contact.py`（销钉的受力方向）
是同样的道理：数字对了不等于对，但图看一眼就能发现。

每个策略家族一行，三个正交视图（俯视 XY / 正视 XZ / 侧视 YZ）：
浅灰是物体表面的轮廓，颜色越暖表示该处累计法向力越大。

与 `s3_dirt_map.py` 一样只用 numpy 拼画布 + imageio 落盘，
不引入 matplotlib（本项目没有这个依赖，字体也没有）。

用法::

    PYTHONPATH=src /isaac-sim/python.sh tools/s4_region_map.py /tmp/s4_drawer \\
        --out region_drawer.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from it.interaction import region_heatmap  # noqa: E402
from it.records import load_episode, read_manifest  # noqa: E402
from it.surfaces import surface_for  # noqa: E402

PANEL = 190          # 单个视图的像素边长
PAD = 8
BG = 250             # 背景
SIL = 214            # 物体轮廓的灰度
VIEWS = ((0, 1, "俯视 XY"), (0, 2, "正视 XZ"), (1, 2, "侧视 YZ"))


def _ramp(t: np.ndarray) -> np.ndarray:
    """0~1 -> RGB。浅黄 -> 橙 -> 深红，低值那端刻意与灰色轮廓拉开。"""
    t = np.clip(t, 0.0, 1.0)[..., None]
    lo = np.array([255.0, 232.0, 150.0])
    mid = np.array([245.0, 140.0, 30.0])
    hi = np.array([150.0, 20.0, 25.0])
    first = lo + (mid - lo) * np.clip(t * 2.0, 0, 1)
    return np.where(t < 0.5, first, mid + (hi - mid) * np.clip(t * 2.0 - 1.0, 0, 1))


def _view(points: np.ndarray, heat: np.ndarray, ax0: int, ax1: int) -> np.ndarray:
    """一个正交视图。同一像素上取最大热度（不是求和——求和会让重叠面偏亮）。"""
    img = np.full((PANEL, PANEL, 3), BG, dtype=np.uint8)
    p = points[:, [ax0, ax1]]
    lo, hi = p.min(axis=0), p.max(axis=0)
    span = np.maximum(hi - lo, 1e-6).max()
    q = (p - (lo + hi) / 2) / span * (PANEL - 2 * PAD) + PANEL / 2
    col = np.clip(q[:, 0].astype(int), 0, PANEL - 1)
    row = np.clip((PANEL - 1 - q[:, 1]).astype(int), 0, PANEL - 1)

    img[row, col] = SIL                                  # 先铺轮廓
    if heat.max() > 0:
        norm = heat / heat.max()
        strong = norm > 1e-3
        order = np.argsort(norm[strong])                 # 热的后画，压在冷的上面
        rr, cc = row[strong][order], col[strong][order]
        img[rr, cc] = _ramp(norm[strong][order]).astype(np.uint8)
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="region.png")
    ap.add_argument("--per-family", type=int, default=40, help="每个家族累计几条")
    ap.add_argument("--group-by", default="strategy_family",
                    choices=("strategy_family", "implementation"))
    a = ap.parse_args()

    root = Path(a.root)
    man = read_manifest(root / "manifest.json")
    ok = [e for e in man["episodes"] if e["success"]]
    if not ok:
        print(f"{root} 里没有成功的 episode")
        return 2

    groups: dict[str, list] = {}
    for e in ok:
        groups.setdefault(str(e[a.group_by]), []).append(e)

    rows, labels, skipped = [], [], 0
    for name in sorted(groups):
        surf = None
        heat = None
        tag = None
        for e in groups[name][: a.per_family]:
            rec = load_episode(root / e["path"])
            info = rec.meta["surface"]
            # ⚠️ **只能叠加同一个几何变体的热图。** 变体之间点数一样，但那是
            # 两套**不同的采样**（把手位置差 ±7 mm），同一个下标指的不是同一处。
            # 混着叠加实测会凭空造出 6~7% 的"面板/托盘接触"，而逐 episode
            # 归一化的统计里那两处只有 0.16%——图会骗人，数不会。
            if tag is None:
                tag = info["geom_tag"]
                surf = surface_for(info["object"], tag)
                heat = np.zeros(surf.n_points)
            if info["geom_tag"] != tag:
                skipped += 1
                continue
            heat += region_heatmap(rec, surf.n_points, phase=2)
        if surf is None or heat.max() <= 0:
            continue
        panels = [_view(surf.points.astype(np.float64), heat, i, j)
                  for i, j, _ in VIEWS]
        row = np.concatenate([np.pad(p, ((PAD, PAD), (PAD, PAD), (0, 0)),
                                     constant_values=BG) for p in panels], axis=1)
        rows.append(row)
        labels.append((name, int((heat > 0).any()) and len(groups[name][: a.per_family]),
                       float(heat.sum()), surf.parts, surf.part, heat))

    if not rows:
        print("没有可画的家族")
        return 2
    canvas = np.concatenate(rows, axis=0)
    import imageio.v2 as iio                              # 惰性导入，同 s3_dirt_map

    iio.imwrite(a.out, canvas)

    print(f"写出 {a.out}，{canvas.shape[1]}×{canvas.shape[0]} 像素")
    if skipped:
        print(f"跳过 {skipped} 条几何变体不同的 episode——不同变体是两套不同的采样，"
              "下标不能混用")
    print(f"每行一个{'策略家族' if a.group_by == 'strategy_family' else '实现'}，"
          f"三列依次是 " + " / ".join(v[2] for v in VIEWS))
    print(f"{'分组':<18}{'条数':>6}{'累计法向力 N':>14}   力最大的三个部件")
    for name, n, tot, parts, part_of, heat in labels:
        share = sorted(((parts[i], float(heat[part_of == i].sum() / max(heat.sum(), 1e-9)))
                        for i in range(len(parts))), key=lambda kv: -kv[1])[:3]
        txt = "  ".join(f"{k} {100 * v:.1f}%" for k, v in share if v > 1e-4)
        print(f"{name:<18}{n:>6}{tot:>14.0f}   {txt}")
    print("\n说明：浅灰是物体表面轮廓，颜色越暖 = 该处累计法向力越大。")
    print("      图上看的是**接触落在哪**；力打在哪一侧、哪片在滑要看 region/mode 的数。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
