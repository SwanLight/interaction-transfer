"""把擦拭 episode 的 dirt 网格画成图（`plan/06` §7 的人工检查）。

录像里看得到板和黑板擦怎么动，**看不到污渍**——dirt 是一个内部网格，
没有可视几何。而擦拭任务的 effect 恰恰**只有** dirt 状态变化（`plan/02` §3.1
经 D-42 修订）。所以光看录像不足以验收擦拭：动作看着对、区域没擦干净，
两者在画面上分不出来。

这个脚本把每条 episode 的最终 dirt 网格摊成一张图：白 = 已清除，黑 = 仍有污渍。
一眼能看出是"整片擦干净"还是"只擦了中间一条"。

用法::

    PYTHONPATH=src /isaac-sim/python.sh tools/s3_dirt_map.py /tmp/s3_wipe \\
        --out /tmp/dirt.png --per-family 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from it.records import load_episode, read_manifest  # noqa: E402

SCALE = 8          # 每个格子放大成 8×8 像素
PAD = 6
LABEL_H = 0        # 不画文字，避免引入字体依赖；家族顺序在 stdout 里给


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="dirt.png")
    ap.add_argument("--per-family", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    root = Path(a.root)
    man = read_manifest(root / "manifest.json")
    fams = sorted({e["strategy_family"] for e in man["episodes"]})
    rng = np.random.default_rng(a.seed)

    rows: list[np.ndarray] = []
    print("每行一个策略家族，从左到右是随机抽的几条 episode：")
    for fam in fams:
        pool = [e for e in man["episodes"]
                if e["strategy_family"] == fam and e["success"]]
        if not pool:
            pool = [e for e in man["episodes"] if e["strategy_family"] == fam]
        pick = [pool[i] for i in rng.choice(len(pool), min(a.per_family, len(pool)),
                                            replace=False)]
        tiles = []
        for e in pick:
            rec = load_episode(root / e["path"])
            grid = np.asarray(rec.arrays["object/dirt_grid"])[-1]      # (H, W) bool
            cleared = 1.0 - float(grid.mean())
            img = np.where(grid, 30, 245).astype(np.uint8)             # 黑=脏 白=净
            img = np.kron(img, np.ones((SCALE, SCALE), dtype=np.uint8))
            tiles.append(np.pad(img, PAD, constant_values=140))
            print(f"  {fam:<18}{e['episode_id']:<26}清除 {100 * cleared:5.1f}%"
                  f"  {'成功' if e['success'] else '失败'}")
        rows.append(np.concatenate(tiles, axis=1))

    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, w - r.shape[1])), constant_values=140) for r in rows]
    canvas = np.concatenate(rows, axis=0)
    import imageio.v2 as iio
    iio.imwrite(a.out, canvas)
    print(f"\n家族顺序（从上到下）：{fams}")
    print(f"写出 {a.out}，{canvas.shape[1]}×{canvas.shape[0]} 像素")
    return 0


if __name__ == "__main__":
    sys.exit(main())
