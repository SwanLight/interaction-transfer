"""S4：把一份 S3 source 数据集批量转成 Oracle Interaction Record。

`plan/02` §3 的表示提取。输入是 S3 的 episode（`s3-episode-v1`），输出是
`s4-record-v1`：物体中心的 effect / region / engage 方向 / mode / mechanics / phase。
划分**原样继承** S3 manifest——重新划会让校准集混进训练集，而 `plan/02` §4.1 的
conformal 覆盖率保证正是建立在"校准集独立"上的。

不需要 Isaac Sim（纯 numpy），但服务器的系统 python 没装 numpy，所以还是走
Isaac 的解释器::

    PYTHONPATH=src /isaac-sim/python.sh tools/s4_extract.py /tmp/s3_drawer_v3 \\
        --out /tmp/s4_drawer --jobs 32

探针集那种"一个物体一个子目录"的数据集，直接把根目录传进来即可，会逐个处理。

**这个脚本自己报的数不算验收**。落盘之后必须跑 `tools/s4_verify_records.py`
从文件重读一遍，并与 S3 已验收的接触部位分布对拍——D-34 的教训是接触落在
完全出乎意料的地方时，成功率和接触力全都正常。
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from it.interaction import (  # noqa: E402
    extract,
    part_force_share,
    spec_for,
)
from it.records import (  # noqa: E402
    load_episode,
    read_manifest,
    save_episode,
    write_manifest,
)
from it.surfaces import surface_for  # noqa: E402

MODE_NAMES = ("no_contact", "sticking", "sliding", "separating")

#: 归不到表面的接触点占比超过这个数就判失败。它不是精度指标，是**归属错了**的
#: 信号：几何变体没对上、或者接触落在没建模的部件上，两种都会让 region 字段作废。
MAX_OFF_SURFACE = 0.05

#: 法向符号在一条 episode 内的加权一致度下限。明显小于 1 说明同一条 episode 里
#: PhysX 的约定翻过（P-37），那条数据的力方向不可信。
MIN_NORMAL_AGREEMENT = 0.90

_SURFACES: dict[tuple[str, str], object] = {}


def _episode_stats(rec, out_rec) -> dict:
    """一条 episode 的统计量。落盘的是记录本身，这些数只进汇总报告。"""
    valid = np.asarray(out_rec.arrays["valid_s4"])
    active = np.asarray(out_rec.arrays["region/valid"]) & valid[:, None]
    w = np.asarray(out_rec.arrays["region/weight"], dtype=np.float64) * active
    total = float(w.sum())
    raw = np.asarray(out_rec.arrays["mode/raw"])
    lab = np.asarray(out_rec.arrays["mode/label"])

    def share(mode: np.ndarray) -> list[float]:
        if total <= 0:
            return [0.0] * 4
        return [float(w[mode == i].sum() / total) for i in range(4)]

    # 滑移阈值的敏感度（工作方式第 4 条：引用任何数字前先问它对参数敏不敏感）。
    slip = np.asarray(out_rec.arrays["mode/slip_speed"], dtype=np.float64)
    stick_at = [float((w[active] > 0).sum() and
                      (w[active & (slip <= thr)]).sum() / max(float(w[active].sum()), 1e-9))
                for thr in (1e-3, 3e-3, 5e-3, 10e-3)]
    off = int(np.asarray(out_rec.arrays["aux/off_surface"]).sum())
    n_contact = int(active.sum())
    ex = out_rec.meta["extraction"]
    return {
        "episode_id": out_rec.meta["episode_id"],
        "task": out_rec.meta["task"],
        "family": out_rec.meta["strategy_family"],
        "success": bool(out_rec.meta["success"]),
        "split": out_rec.meta["split"],
        "geom": out_rec.meta["geometry_variant"],
        "frames": int(len(valid)),
        "dirty_frames": int((~valid).sum()),
        "contacts": n_contact,
        "off_surface": off,
        "peak_force_N": float(np.asarray(out_rec.arrays["aux/frame_force"]).max()),
        "mode_share_raw": share(raw),
        "mode_share_new": share(lab),
        "stick_share_by_threshold": stick_at,
        "mode_source": ex["mode_source"],
        "normal_agreement": float(ex["normal_sign_agreement"]),
        "force_sign": float(ex["force_on_object_sign"]),
        "has_rel_vel": bool(ex["has_rel_vel"]),
        "gen_abs_mean": [float(v) for v in
                         np.abs(np.asarray(out_rec.arrays["mech/generalized"],
                                           dtype=np.float64)).mean(axis=0)],
        "part_share": part_force_share(out_rec, _SURFACES[(
            out_rec.meta["surface"]["object"], out_rec.meta["surface"]["geom_tag"])]),
    }


def _one(job: tuple[str, str, str]) -> tuple[str, str, dict | None, str]:
    """处理一条 episode：读 -> 提取 -> 落盘 -> 回报统计。"""
    src_path, out_path, episode_id = job
    try:
        rec = load_episode(src_path)
        spec = spec_for(rec.meta)
        tag = str(rec.meta.get("geometry_variant", "nominal"))
        surface = _SURFACES[(spec.obj, tag)]
        out_rec = extract(rec, surface)
        save_episode(out_rec, out_path)
        return (episode_id, out_path, _episode_stats(rec, out_rec), "")
    except Exception as exc:                      # noqa: BLE001
        return (episode_id, out_path, None, f"{type(exc).__name__}: {exc}")


def _agg(rows: list[dict]) -> dict:
    def wmean(key: str) -> float:
        return float(np.mean([r[key] for r in rows])) if rows else 0.0

    contacts = sum(r["contacts"] for r in rows)
    off = sum(r["off_surface"] for r in rows)
    frames = sum(r["frames"] for r in rows)
    dirty = sum(r["dirty_frames"] for r in rows)
    raw = np.array([r["mode_share_raw"] for r in rows]) if rows else np.zeros((1, 4))
    new = np.array([r["mode_share_new"] for r in rows]) if rows else np.zeros((1, 4))
    weight = np.array([max(r["contacts"], 1) for r in rows], dtype=np.float64)
    weight = weight / weight.sum()
    parts: dict[str, float] = {}
    for r in rows:
        for name, val in r["part_share"].items():
            parts[name] = parts.get(name, 0.0) + val
    n = max(len(rows), 1)
    return {
        "episodes": len(rows),
        "frames": frames,
        "dirty_frame_fraction": dirty / max(frames, 1),
        "contacts": contacts,
        "off_surface_fraction": off / max(contacts + off, 1),
        "peak_force_N": max((r["peak_force_N"] for r in rows), default=0.0),
        "normal_agreement_min": min((r["normal_agreement"] for r in rows), default=1.0),
        "normal_agreement_mean": wmean("normal_agreement"),
        "mode_share_raw": (raw * weight[:, None]).sum(axis=0).tolist(),
        "mode_share_new": (new * weight[:, None]).sum(axis=0).tolist(),
        "part_share": {k: v / n for k, v in sorted(parts.items(), key=lambda kv: -kv[1])},
    }


def _report(name: str, rows: list[dict], surfaces: dict, elapsed: float) -> str:
    ok = [r for r in rows if r["success"]]
    lines = [f"===== S4 提取：{name} =====",
             f"episode {len(rows)} 条（成功 {len(ok)}），耗时 {elapsed:.0f} s", ""]

    lines.append("表面采样（`plan/02` §2，冻结并记进每条记录的 meta）")
    lines.append(f"{'物体/变体':<18}{'点数':>8}{'面积cm²':>10}{'pitch mm':>10}  sha256")
    for (obj, tag), surf in sorted(surfaces.items()):
        pitch = (surf.total_area / surf.n_points) ** 0.5 * 1000.0
        lines.append(f"{obj + '/' + tag:<18}{surf.n_points:>8}"
                     f"{surf.total_area * 1e4:>10.1f}{pitch:>10.2f}  {surf.sha256[:16]}")
    lines.append("")

    fams = sorted({r["family"] for r in rows})
    lines.append("逐家族（只统计成功 episode）")
    lines.append(f"{'家族':<18}{'条数':>6}{'脏帧%':>8}{'接触点':>10}"
                 f"{'归不到面%':>11}{'峰值力N':>10}{'法向一致':>10}")
    for fam in fams:
        sub = [r for r in ok if r["family"] == fam]
        if not sub:
            continue
        a = _agg(sub)
        lines.append(f"{fam:<18}{a['episodes']:>6}{100 * a['dirty_frame_fraction']:>8.2f}"
                     f"{a['contacts']:>10}{100 * a['off_surface_fraction']:>11.2f}"
                     f"{a['peak_force_N']:>10.1f}{a['normal_agreement_min']:>10.3f}")
    lines.append("")

    agg = _agg(ok)
    lines.append("contact mode（按法向力加权；D-49 的双报）")
    lines.append(f"{'判据':<22}" + "".join(f"{m:>13}" for m in MODE_NAMES))
    lines.append(f"{'S3 原始（摩擦锥）':<22}" +
                 "".join(f"{100 * v:>12.1f}%" for v in agg["mode_share_raw"]))
    lines.append(f"{'S4 重判（滑移速度）':<22}" +
                 "".join(f"{100 * v:>12.1f}%" for v in agg["mode_share_new"]))
    lines.append("")

    thr_rows = np.array([r["stick_share_by_threshold"] for r in rows]) if rows \
        else np.zeros((1, 4))
    lines.append("滑移阈值敏感度（sticking 占比随阈值怎么变）")
    lines.append(f"{'阈值 mm/s':<22}" + "".join(f"{t:>13}" for t in (1, 3, 5, 10)))
    lines.append(f"{'sticking 占比':<22}" +
                 "".join(f"{100 * v:>12.1f}%" for v in thr_rows.mean(axis=0)))
    src = {}
    for r in rows:
        src[r["mode_source"]] = src.get(r["mode_source"], 0) + 1
    lines.append(f"滑移判据来源：{src}")
    lines.append("  pose_diff  = 两刚体位姿差分（主判据，D-49）；也是真实装置能测的量")
    lines.append("  patch_drift= 位姿缺失时的退路：接触斑块在物体表面上的位移")
    lines.append("  ⚠️ PhysX 报的瞬时相对速度只作诊断（`mode/inst_slip`），"
                 "它被采集板的姿态极限环污染，见 P-52")
    lines.append("")

    lines.append("接触力按物体部件（与 S3 的接触部位统计对拍）")
    for part, val in agg["part_share"].items():
        if val > 1e-4:
            lines.append(f"  {part:<18}{100 * val:>8.2f}%")
    lines.append("")
    lines.append("说明：")
    lines.append("  归不到面% —— 接触点离最近表面采样点超过容差的比例。它不是精度指标，")
    lines.append("               是**归属错了**的信号（几何变体没对上 / 接触落在没建模的部件上）。")
    lines.append("  法向一致 —— 一条 episode 内 PhysX 报的法向与几何外法向的加权一致度。")
    lines.append("               明显小于 1 说明约定在 episode 内翻过（P-37），力的方向不可信。")
    lines.append("  这里的数是**生成侧自报**的。验收要跑 tools/s4_verify_records.py 从文件重读。")
    return "\n".join(lines)


def _datasets(root: Path) -> list[Path]:
    if (root / "manifest.json").exists():
        return [root]
    subs = sorted(p for p in root.iterdir() if (p / "manifest.json").exists())
    if not subs:
        raise SystemExit(f"{root} 下既没有 manifest.json，也没有带 manifest 的子目录")
    return subs


def process(src: Path, out: Path, jobs: int, limit: int | None) -> tuple[list[dict], dict]:
    man = read_manifest(src / "manifest.json")
    entries = man["episodes"][:limit] if limit else man["episodes"]

    # 表面采样在**建进程池之前**造好：Linux 下 fork 出来的子进程直接共享，
    # 否则每个 worker 都要自己重算一遍（大物体单次 ~17 s）。
    need = set()
    for e in entries:
        spec = spec_for(e["meta"])
        need.add((spec.obj, str(e.get("geometry_variant", "nominal"))))
    for obj, tag in sorted(need):
        if (obj, tag) not in _SURFACES:
            t0 = time.time()
            _SURFACES[(obj, tag)] = surface_for(obj, tag)
            print(f"  表面采样 {obj}/{tag}: {_SURFACES[(obj, tag)].n_points} 点"
                  f"（{time.time() - t0:.0f} s）", flush=True)

    (out / "episodes").mkdir(parents=True, exist_ok=True)
    jobs_list = [(str(src / e["path"]), str(out / "episodes" / f"{e['episode_id']}.npz"),
                  e["episode_id"]) for e in entries]

    rows, failures, paths = [], [], {}
    t0 = time.time()
    if jobs > 1:
        with mp.Pool(jobs) as pool:
            for i, (eid, path, stat, err) in enumerate(
                    pool.imap_unordered(_one, jobs_list, chunksize=4), 1):
                if err:
                    failures.append(f"{eid}: {err}")
                else:
                    rows.append(stat)
                    paths[eid] = path
                if i % 500 == 0:
                    print(f"  {i}/{len(jobs_list)}（{time.time() - t0:.0f} s）", flush=True)
    else:
        for job in jobs_list:
            eid, path, stat, err = _one(job)
            if err:
                failures.append(f"{eid}: {err}")
            else:
                rows.append(stat)
                paths[eid] = path

    if failures:
        for line in failures[:10]:
            print(f"  [提取失败] {line}")
        raise SystemExit(f"{len(failures)} 条 episode 提取失败，不写 manifest")

    # 划分原样继承 S3：重新划会让校准集混进训练集（`plan/02` §4.1）
    records = [(paths[e["episode_id"]], load_episode(paths[e["episode_id"]]))
               for e in entries]
    splits = {k: [i for i in v if i in paths]
              for k, v in (man.get("splits") or {}).items()}
    write_manifest(
        records, out / "manifest.json",
        dataset_name=f"s4_{man['dataset_name']}",
        generator_git_sha=_git_sha(),
        extra={
            "source_dataset": str(src),
            "source_manifest_sha": man.get("generator_git_sha", "unknown"),
            "surfaces": {f"{o}/{t}": {"n_points": s.n_points, "sha256": s.sha256,
                                      "total_area_m2": s.total_area}
                         for (o, t), s in _SURFACES.items()},
        },
        splits=splits or None,
    )
    return rows, {f"{o}/{t}": s for (o, t), s in _SURFACES.items()}


def _git_sha() -> str:
    p = Path(__file__).resolve().parent.parent / ".git_sha"
    return p.read_text(encoding="utf-8").strip() if p.exists() else "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="S3 数据集根目录（或含多个子数据集的目录）")
    ap.add_argument("--out", required=True, help="S4 记录写到哪")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 8) // 2))
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条，调试用")
    # 两个阈值都按"错了以后结论作废"的量级给，不是精度指标
    ap.add_argument("--max-off-surface", type=float, default=MAX_OFF_SURFACE)
    ap.add_argument("--min-normal-agreement", type=float, default=MIN_NORMAL_AGREEMENT)
    a = ap.parse_args()

    root, out_root = Path(a.root), Path(a.out)
    bad = []
    for src in _datasets(root):
        out = out_root if src == root else out_root / src.name
        print(f"\n===== {src} -> {out} =====", flush=True)
        t0 = time.time()
        rows, surfaces = process(src, out, a.jobs, a.limit or None)
        text = _report(src.name, rows, {tuple(k.split("/")): v
                                        for k, v in surfaces.items()},
                       time.time() - t0)
        (out / "report.txt").write_text(text + "\n", encoding="utf-8")
        (out / "stats.json").write_text(json.dumps(rows, ensure_ascii=False), "utf-8")
        print(text)

        ok = [r for r in rows if r["success"]]
        agg = _agg(ok)
        if agg["off_surface_fraction"] > a.max_off_surface:
            bad.append(f"{src.name}: 归不到表面的接触点 "
                       f"{100 * agg['off_surface_fraction']:.2f}% > "
                       f"{100 * a.max_off_surface:.2f}%")
        if agg["normal_agreement_min"] < a.min_normal_agreement:
            bad.append(f"{src.name}: 法向符号一致度最低 "
                       f"{agg['normal_agreement_min']:.3f} < {a.min_normal_agreement}")

    if bad:
        print("\n===== 未通过 =====")
        for line in bad:
            print("[FAIL] " + line)
        return 1
    print("\n[PASS] 提取完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
