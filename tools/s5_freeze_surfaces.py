#!/usr/bin/env python3
"""把 S4 数据集用到的 surface 冻结成文件，补上 P-57 / D-64 要求的那一半。

现有四份 S4 记录只在 manifest 里存了 surface 的 **identity hash**，没有存点序本身。
FPS 在几何对称点上的 tie-breaking 会随 BLAS / NumPy 版本变化，所以在别的环境里
重新生成不保证得到同一套点序——而 `region/point_idx` 是按那套点序写的。

这个脚本在**产生这些数据的那台机器上**跑一次：逐个重新生成、与 manifest 记的 hash
逐个核对，核对通过才落盘成 `frozen-surface-v1`，并把路径写回 manifest 的 `surfaces`
表。之后 S5 一律用 `--surface`，不再依赖"在哪台机器上跑"。

**hash 对不上就直接失败，不写文件。** 那说明本机重算出来的点序与数据不是一套，
继续跑只会把接触点归到错误的表面点上（而且完全不报错）。

用法::

    PYTHONPATH=src /isaac-sim/python.sh tools/s5_freeze_surfaces.py \\
        /tmp/s4_drawer /tmp/s4_wipe /tmp/s4_knob
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from it.records import read_manifest  # noqa: E402
from it.surfaces import load_surface, save_surface, surface_for  # noqa: E402


def _stale_reason(path: Path) -> str | None:
    """已冻结的文件是不是旧代码产的。

    ⚠️ identity ``sha256`` 只覆盖 points / normals / part，**不含 ``parent``**。
    所以池化规则改了之后，旧文件照样能通过 hash 核对而被 SKIP 掉——修好的代码
    等于没生效（P-38 的同一个形状：结果逐位相同就该怀疑改动没生效）。
    这里直接查当前要求的不变量：**粗粒代表点必须与被代表的点同部件**。
    """
    try:
        surface = load_surface(path)
    except Exception as error:                     # 读不动就当它旧，重生成
        return f"读取失败（{error}）"
    part = np.asarray(surface.part)
    for level, mapping in sorted(surface.parent.items()):
        crossed = int((part != part[mapping]).sum())
        if crossed:
            return (f"{level} 档有 {crossed} 个点（{crossed / len(part):.1%}）"
                    "的代表点跨了部件")
    return None


def freeze(dataset: Path, *, force: bool) -> list[str]:
    manifest_path = dataset / "manifest.json"
    manifest = read_manifest(manifest_path)
    surfaces = manifest.get("surfaces", {})
    if not surfaces:
        raise SystemExit(f"{manifest_path} 没有 surfaces 表")
    out_dir = dataset / "surfaces"
    out_dir.mkdir(parents=True, exist_ok=True)

    messages = []
    changed = False
    for key, info in sorted(surfaces.items()):
        obj, geom = key.split("/", 1)
        destination = out_dir / f"{obj}-{geom}.npz"
        expected = str(info.get("sha256"))
        stale = _stale_reason(destination) if destination.exists() else None
        if destination.exists() and not force and stale is None:
            existing = load_surface(destination)
            if existing.sha256 != expected:
                raise SystemExit(f"{destination} 已存在但 hash 与 manifest 不符；"
                                 "先查清楚它是哪来的，不要覆盖")
            messages.append(f"SKIP  {key} 已冻结 {destination.name}")
        else:
            if stale:
                messages.append(f"STALE {key}：{stale}，重新生成")
            surface = surface_for(obj, geom)
            if surface.sha256 != expected:
                raise SystemExit(
                    f"{key}: 本机重算的 surface hash {surface.sha256[:16]}… 与 manifest 记的 "
                    f"{expected[:16]}… 不符。数据里的 region/point_idx 是按 manifest 那套点序写的，"
                    "在这台机器上重算得到的是另一套（P-57）。必须在产生数据的环境里跑本脚本。")
            save_surface(surface, destination)
            messages.append(f"WROTE {key} -> {destination.name} "
                            f"({surface.n_points} 点, {surface.total_area:.5f} m²)")
        relative = str(destination.relative_to(dataset))
        if info.get("path") != relative:
            info["path"] = relative
            changed = True

    if changed:
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=1,
                                        sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(manifest_path)
        messages.append(f"UPDATED {manifest_path}（surfaces[*].path 已写回）")
    return messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--force", action="store_true", help="重写已存在的冻结 surface")
    args = parser.parse_args()
    for dataset in args.datasets:
        print(f"=== {dataset}")
        for message in freeze(dataset.resolve(), force=args.force):
            print(" ", message)


if __name__ == "__main__":
    main()
