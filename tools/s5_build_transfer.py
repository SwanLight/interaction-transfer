#!/usr/bin/env python3
"""从一个 S4 manifest 构造同任务的 statistical interaction transfer。

示例（训练集构造，确认集只用于之后评估）：

    PYTHONPATH=src python3 tools/s5_build_transfer.py \
      --manifest /tmp/s4_drawer/manifest.json --output out/s5_drawer

程序按 ``task / object / geometry_variant / frozen surface hash`` 自动分组；每组产出
一个 `.npz` 与同名 `.report.json`。默认只读成功的 train split。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from it.records import load_episode, read_manifest, sha256_file  # noqa: E402
from it.surfaces import load_surface  # noqa: E402
from it.transfer import TransferError, build_transfer, save_transfer  # noqa: E402


def _allowed_cells_mean(transfer) -> float:
    active = transfer.arrays["region/mass/mean"].sum(axis=1) > 0
    allowed = transfer.arrays["region/allowed"]
    return float(allowed[active].sum(axis=1).mean()) if active.any() else 0.0


def _slug(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(value)).strip("-") or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", default="train",
                        help="manifest split；用 'all' 读取全部成功记录")
    parser.add_argument("--calibration-split", default="calibration",
                        help="用来标定允许集合的**冻结**划分；设成 'none' 则输出未标定 artifact")
    parser.add_argument("--task", help="可选，只构造指定 task")
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--surface-points", type=int, default=256)
    parser.add_argument("--region-smooth-fallback", type=float, default=1.0,
                        help="原始 region 估计量退化时退到几格平滑（0 = 不兜底，直接失败）。"
                             "**只是兜底**：默认路径不平滑，因为平滑会让 width 翻倍")
    parser.add_argument("--split-by-family", action="store_true",
                        help="再按 strategy_family 拆一层。**探针集要开，三个主任务不能开**："
                             "主任务的 envelope 就是要归纳多种策略的共同要求，"
                             "而探针集一个物体上的不同原语是不同的交互")
    parser.add_argument("--surface", type=Path,
                        help="冻结 surface artifact；跨环境运行时必须优先使用")
    parser.add_argument("--max-episodes", type=int,
                        help="只用于 smoke；正式 artifact 不应设置")
    parser.add_argument("--existing-only", action="store_true",
                        help="只读本地实际存在的 episode；用于下载样本 smoke")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = read_manifest(manifest_path)
    if manifest.get("schema_version") != "s4-record-v2":
        raise SystemExit("manifest 必须是 s4-record-v2")
    # 分组键默认是 (task, object, geometry, surface_hash)。探针集要再按 strategy_family
    # 拆一层：`03` §2.4 的一个物体上跑了多条**不同的**原语（"戳球"和"滚球"是两种交互），
    # 混进一份 envelope 得到的是一个没有物理意义的平均——那正是 D-59 说的
    # "把不同成功策略的字段拼成从未存在的混血指令"，只不过发生在构造这一侧。
    # 三个主任务反过来**必须**合并策略家族：同一个"拉开 20 mm"由五种策略实现，
    # 归纳它们共同的功能要求正是本项目的主张（`02` §4）。所以这是个 flag，不是默认。
    groups: dict[tuple, list[dict]] = {}
    calibration_groups: dict[tuple, list[dict]] = {}
    for entry in manifest["episodes"]:
        meta = entry.get("meta", {})
        if args.existing_only and not (manifest_path.parent / entry["path"]).exists():
            continue
        if not entry.get("success", False):
            continue
        # 被移出范围的划分一律跳过，"all" 也不例外（见 tools/s5_exclude_episodes.py）。
        if str(entry.get("split", "")).startswith("excluded_"):
            continue
        task = str(entry.get("task"))
        if args.task and task != args.task:
            continue
        obj = str(meta.get("object"))
        geom = str(entry.get("geometry_variant", meta.get("geometry_variant", "nominal")))
        surface_hash = str(meta.get("surface", {}).get("sha256"))
        family = str(entry.get("strategy_family", "all"))
        key = ((task, obj, geom, surface_hash, family) if args.split_by_family
               else (task, obj, geom, surface_hash))
        split = str(entry.get("split"))
        if args.calibration_split != "none" and split == args.calibration_split:
            calibration_groups.setdefault(key, []).append(entry)
        if args.split != "all" and split != args.split:
            continue
        groups.setdefault(key, []).append(entry)
    if not groups:
        raise SystemExit("筛选后没有成功 episode")

    args.output.mkdir(parents=True, exist_ok=True)
    explicit_surface = load_surface(args.surface) if args.surface else None
    index = []
    failures: list[dict] = []
    for key, entries in sorted(groups.items()):
        task, obj, geom, surface_hash = key[:4]
        family = key[4] if len(key) > 4 else None
        entries.sort(key=lambda item: str(item["episode_id"]))
        if args.max_episodes is not None:
            entries = entries[:args.max_episodes]
        if len(entries) < 2:
            print(f"SKIP {task}/{obj}/{geom}/{family}: 只有 {len(entries)} 条",
                  file=sys.stderr)
            continue

        def records():
            for entry in entries:
                path = (manifest_path.parent / entry["path"]).resolve()
                if entry.get("sha256") and sha256_file(path) != entry["sha256"]:
                    raise RuntimeError(f"SHA-256 不匹配：{path}")
                yield load_episode(path)

        parts = (task, obj, geom, args.split) if family is None else \
            (task, obj, geom, family, args.split)
        transfer_id = "-".join(map(_slug, parts))
        group_surface = explicit_surface
        if group_surface is None:
            info = manifest.get("surfaces", {}).get(f"{obj}/{geom}", {})
            candidate = manifest_path.parent / str(info.get("path", ""))
            if info.get("path") and candidate.exists():
                group_surface = load_surface(candidate)
        calibration_entries = sorted(
            calibration_groups.get(key, []),
            key=lambda item: str(item["episode_id"]))

        def calibration_records():
            for entry in calibration_entries:
                yield load_episode((manifest_path.parent / entry["path"]).resolve())

        if not calibration_entries:
            print(f"⚠️ {transfer_id}: 没有 {args.calibration_split!r} 划分的 episode，"
                  "输出**未标定** artifact；它没有覆盖保证，不得直接喂给 E-I",
                  file=sys.stderr)
        # region 密度默认**不平滑**：实测平滑会让允许面积在擦拭上翻倍，而 width 正是
        # 要最小化的数。只有当原始直方图估计量退化（τ* 发散：校准 episode 的接触落在
        # 训练从没碰过的格上，那些格概率恰好是 0，任何 τ 都进不了允许集合）才退到
        # 平滑估计量，并把用了哪一档逐份记进 report（D-78）。
        def build(smooth: float):
            return build_transfer(records(), n_bins=args.bins,
                                  n_surface=args.surface_points,
                                  transfer_id=transfer_id, surface=group_surface,
                                  calibration=calibration_records()
                                  if calibration_entries else None,
                                  region_smooth=smooth)

        smooth_used = 0.0
        try:
            try:
                transfer = build(0.0)
            except TransferError as first_error:
                if "τ*" not in str(first_error) or args.region_smooth_fallback <= 0:
                    raise
                smooth_used = float(args.region_smooth_fallback)
                transfer = build(smooth_used)
                print(f"SMOOTHED {transfer_id}: 原始直方图估计量退化，"
                      f"退到 {smooth_used} 格平滑（D-78）", file=sys.stderr)
        except TransferError as error:
            # 一个 group 造不出来不该带走同一个物体的其余 group——探针集一个物体上
            # 有 2~9 条原语，早先的实现在第一条失败时整个物体都没了，而日志里只看得到
            # 一条 traceback。失败逐条记下来，末尾统一非零退出，**不静默跳过**。
            print(f"FAILED {transfer_id}: {error}", file=sys.stderr)
            failures.append({"transfer_id": transfer_id, "error": str(error),
                             "episodes": len(entries),
                             "calibration_episodes": len(calibration_entries)})
            continue
        artifact = args.output / f"{transfer_id}.npz"
        save_transfer(transfer, artifact)
        report = {
            "artifact": artifact.name,
            "artifact_sha256": sha256_file(artifact),
            "manifest": str(manifest_path),
            "manifest_dataset": manifest.get("dataset_name"),
            "source_split": args.split,
            "calibration_split": args.calibration_split,
            "calibration_episodes": len(calibration_entries),
            "source_surface_sha256": surface_hash,
            "meta": transfer.meta,
            "support": {
                "min_episodes_per_bin": int(transfer.arrays["support/episodes"].min()),
                "max_episodes_per_bin": int(transfer.arrays["support/episodes"].max()),
                "empty_bins": int((transfer.arrays["support/episodes"] == 0).sum()),
                "phase_budget": list(transfer.meta["aggregation"]["phase_budget"]),
                "occupied_cells": int((transfer.arrays["region/support"] > 0).sum()),
                "cell_support_median": transfer.meta["diagnostics"]["cell_support_median"],
                "cell_support_under_half_fraction":
                    transfer.meta["diagnostics"]["cell_support_under_half_fraction"],
                # 只在**有接触质量**的命令格上平均，与 s5_eval_envelope.py 的 2a 项
                # 同一个分母；否则接近段那些空格会把这个数稀释掉。
                "allowed_cells_mean": _allowed_cells_mean(transfer),
            },
            "calibration": transfer.meta["calibration"],
            "region_smooth_cells": smooth_used,
        }
        report_path = artifact.with_suffix(".report.json")
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                          sort_keys=True) + "\n", encoding="utf-8")
        index.append(report)
        print(f"WROTE {artifact} ({len(entries)} episodes)")

    if not index:
        raise SystemExit("没有任何 group 造出 artifact")
    index_path = args.output / "index.json"
    previous = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else []
    # 同一个输出目录会被多个物体依次写入，合并而不是覆盖（按 artifact 名去重）。
    merged = {item["artifact"]: item for item in previous}
    merged.update({item["artifact"]: item for item in index})
    index_path.write_text(
        json.dumps(sorted(merged.values(), key=lambda item: item["artifact"]),
                   ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    if failures:
        print(f"\n{len(failures)} 个 group 造不出来：", file=sys.stderr)
        for item in failures:
            print(f"  {item['transfer_id']}（train {item['episodes']} / "
                  f"calib {item['calibration_episodes']}）：{item['error'][:110]}",
                  file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
