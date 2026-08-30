#!/usr/bin/env python3
"""把 S5 的说明书画出来：允许接触区域长什么样、命令轴是怎么分帧的。

S5 没有录像，全部结论都是数。而这一步最该被看见的两件事，**光看数看不出来**：

1. **允许区域到底圈在物体的哪一片。** "占表面积 1.35%" 是个抽象的数；
   画出来才知道它圈的是不是把手横杆背面、是不是销钉、是不是黑板中间那条带。
   圈错地方而数字好看，是完全可能的；
2. **命令轴把帧分到哪去了。** P-58 那个坑（84% 的帧挤进 2 格）当初能活下来，
   正是因为没有任何地方把逐格帧数画出来过。

沿用 `s4_region_map.py` 的画法：只用 numpy 拼画布 + imageio 落盘，不引 matplotlib。

用法::

    PYTHONPATH=src python3 tools/s5_visualize.py \\
        --artifact out/s5/drawer/xxx.npz --dataset /tmp/s4_drawer --out out/s5/region_drawer.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from it.records import load_episode, read_manifest  # noqa: E402
from it.surfaces import load_surface  # noqa: E402
from it.transfer import bin_index, load_transfer  # noqa: E402

PANEL, PAD, GAP = 210, 10, 6
BG = np.array([250, 250, 252], dtype=np.uint8)
SIL = np.array([226, 226, 234], dtype=np.uint8)
#: 轮廓抽稀。16384 个点铺满 210px 会糊成一团实心块，反而看不出物体形状。
SIL_STRIDE = 5
#: 命令点只有几个到十几个，1 像素在 210px 的画布上根本看不见，一律加粗。
DOT = 1
ALLOW = np.array([26, 122, 210], dtype=np.uint8)      # 允许集合：蓝
BARBG = np.array([236, 236, 242], dtype=np.uint8)
BAR = np.array([70, 70, 82], dtype=np.uint8)


def _ramp(t: np.ndarray) -> np.ndarray:
    """0~1 -> RGB。浅黄 -> 橙 -> 深红。与 s4_region_map 保持同一套配色。"""
    t = np.clip(t, 0.0, 1.0)[..., None]
    lo, mid, hi = (np.array([255.0, 232.0, 150.0]), np.array([245.0, 140.0, 30.0]),
                   np.array([150.0, 20.0, 25.0]))
    first = lo + (mid - lo) * np.clip(t * 2.0, 0, 1)
    return np.where(t < 0.5, first, mid + (hi - mid) * np.clip(t * 2.0 - 1.0, 0, 1))


def _project(points: np.ndarray, ax0: int, ax1: int) -> tuple[np.ndarray, np.ndarray]:
    p = points[:, [ax0, ax1]]
    lo, hi = p.min(axis=0), p.max(axis=0)
    span = np.maximum(hi - lo, 1e-6).max()
    q = (p - (lo + hi) / 2) / span * (PANEL - 2 * PAD) + PANEL / 2
    return (np.clip((PANEL - 1 - q[:, 1]).astype(int), 0, PANEL - 1),
            np.clip(q[:, 0].astype(int), 0, PANEL - 1))


def _view(full: np.ndarray, points: np.ndarray, heat: np.ndarray | None,
          allowed: np.ndarray | None, ax0: int, ax1: int) -> np.ndarray:
    """一个正交视图。先铺整个物体的轮廓，再画命令点。"""
    img = np.full((PANEL, PANEL, 3), BG, dtype=np.uint8)
    row, col = _project(full[::SIL_STRIDE], ax0, ax1)
    img[row, col] = SIL
    row, col = _project(points, ax0, ax1)

    def blot(mask: np.ndarray, colour: np.ndarray) -> None:
        for dr in range(-DOT, DOT + 1):
            for dc in range(-DOT, DOT + 1):
                img[np.clip(row[mask] + dr, 0, PANEL - 1),
                    np.clip(col[mask] + dc, 0, PANEL - 1)] = colour

    if heat is not None and heat.max() > 0:
        norm = heat / heat.max()
        live = norm > 1e-3
        order = np.argsort(norm[live])            # 热的后画，压在冷的上面
        colours = _ramp(norm[live][order]).astype(np.uint8)
        index = np.flatnonzero(live)[order]
        for i, colour in zip(index, colours):
            single = np.zeros(len(row), dtype=bool)
            single[i] = True
            blot(single, colour)
    if allowed is not None and allowed.any():
        blot(allowed, ALLOW)
    return img


def _bars(values: np.ndarray, width: int, height: int, highlight: np.ndarray | None = None):
    """一排竖直条形图。用来画"每个命令格里有多少帧"。"""
    img = np.full((height, width, 3), BG, dtype=np.uint8)
    n = len(values)
    step = max(width // n, 1)
    peak = max(float(values.max()), 1e-9)
    for i, value in enumerate(values):
        x0 = i * step
        x1 = min(x0 + step - 1, width)
        bar = int(round(value / peak * (height - 2)))
        img[:, x0:x1] = BARBG
        if bar > 0:
            colour = ALLOW if (highlight is not None and highlight[i]) else BAR
            img[height - bar:, x0:x1] = colour
    return img


def _stack(rows: list[np.ndarray]) -> np.ndarray:
    width = max(r.shape[1] for r in rows)
    out = []
    for r in rows:
        if r.shape[1] < width:
            pad = np.full((r.shape[0], width - r.shape[1], 3), BG, dtype=np.uint8)
            r = np.concatenate([r, pad], axis=1)
        out.append(r)
        out.append(np.full((GAP, width, 3), BG, dtype=np.uint8))
    return np.concatenate(out[:-1], axis=0)


def region_figure(transfer, surface) -> np.ndarray:
    """三视图 × 两行：上行是力加权接触分布，下行是标定出来的允许集合。"""
    n = int(transfer.meta["surface"]["command_n_points"])
    points = np.asarray(surface.points[:n], dtype=np.float64)
    full = np.asarray(surface.points, dtype=np.float64)
    mass = np.asarray(transfer.arrays["region/mass/mean"], dtype=np.float64).sum(axis=0)
    allowed = np.asarray(transfer.arrays["region/allowed"]).any(axis=0)
    views = ((0, 1), (0, 2), (1, 2))
    heat_row = [_view(full, points, mass, None, a, b) for a, b in views]
    allow_row = [_view(full, points, None, allowed, a, b) for a, b in views]
    join = lambda imgs: np.concatenate(
        [x for pair in zip(imgs, [np.full((PANEL, GAP, 3), BG, dtype=np.uint8)] * len(imgs))
         for x in pair][:-1], axis=1)
    return _stack([join(heat_row), join(allow_row)])


def axis_figure(transfer, surface, records) -> np.ndarray:
    """两行条形图：上行按任务完成度分帧（v1，P-58），下行按相分段活动量分帧（v2）。"""
    n_bins = int(transfer.meta["aggregation"]["n_bins"])
    budget = tuple(int(v) for v in transfer.meta["aggregation"]["phase_budget"])
    old = np.zeros(n_bins)
    new = np.zeros(n_bins)
    for record in records:
        valid = np.asarray(record.arrays["valid_s4"], dtype=bool)
        progress = np.clip(np.asarray(record.arrays["progress"], dtype=np.float64), 0, 1 - 1e-9)
        old += np.bincount(np.minimum((progress[valid] * n_bins).astype(int), n_bins - 1),
                           minlength=n_bins)
        index = bin_index(record, surface, budget=budget)
        new += np.bincount(index[valid], minlength=n_bins)
    width = n_bins * 26
    return _stack([_bars(old, width, 90), _bars(new, width, 90)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    import imageio.v3 as iio

    transfer = load_transfer(args.artifact)
    manifest_path = (args.dataset / "manifest.json").resolve()
    manifest = read_manifest(manifest_path)
    key = f"{transfer.meta['object']}/{transfer.meta['geometry_variant']}"
    surface = load_surface(manifest_path.parent / manifest["surfaces"][key]["path"])

    entries = [e for e in manifest["episodes"] if e.get("success")
               and e.get("split") == "train"
               and str(e.get("meta", {}).get("surface", {}).get("sha256"))
               == transfer.meta["surface"]["sha256"]]
    entries.sort(key=lambda e: str(e["episode_id"]))
    records = [load_episode(manifest_path.parent / e["path"])
               for e in entries[:args.episodes]]

    figure = _stack([region_figure(transfer, surface),
                     axis_figure(transfer, surface, records)])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(args.out, figure)
    print(f"WROTE {args.out}  {figure.shape[1]}x{figure.shape[0]}")


if __name__ == "__main__":
    main()
