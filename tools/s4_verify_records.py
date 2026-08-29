"""S4 记录的独立验收：从落盘文件重读一遍，不看提取脚本自己报的数。

`plan/README` §7 给 S4 的通过条件是"坐标、动力学一致性和泄漏检查全部通过"。
泄漏检查在 `tools/s4_leak_checks.py`，这里管另外两件：

1. **结构与契约**——schema、字段归类、划分是否原样继承、表面采样是否冻结、
   SHA-256 抽查、有没有 NaN；
2. **动力学一致性**（`plan/02` §8）——用接触集合重建的广义力，与物体自己的
   动力学核对。**旋钮这一格最硬**：S3 采集时另记了一路 `object/axis_torque`，
   那是仿真器直接给的绕轴力矩，与我们从接触点重建出来的量可以逐帧对拍。
   抽屉用 `m·a + c·v` 对拍。擦拭平面是 kinematic 的，动力学恒等式不成立，
   **如实标 N/A，不拿别的数凑**。

用法::

    PYTHONPATH=src /isaac-sim/python.sh tools/s4_verify_records.py \\
        /tmp/s4_drawer /tmp/s3_drawer_v3 --sample 60

第二个参数是对应的 S3 数据集（对拍动力学与划分要用）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from it.interaction import RECORD_SCHEMA, part_force_share  # noqa: E402
from it.records import (  # noqa: E402
    IR_SCHEMA_VERSION,
    load_episode,
    read_manifest,
    sha256_file,
)
from it.surfaces import surface_for  # noqa: E402

FAIL: list[str] = []

#: S3 的安全法向力上限（`plan/01` §3.2）。超过它的帧不是错误——S3 已经验收过——
#: 但 mechanics 字段（C4/C5）直接建立在这些力上，占比必须看得见。
SAFETY_FORCE_N = 25.0


def check(ok: bool, name: str, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    line = f"[{tag}] {name}" + (f"  —— {detail}" if detail else "")
    print(line)
    if not ok:
        FAIL.append(line)


def info(name: str, detail: str = "") -> None:
    print(f"[INFO] {name}" + (f"  —— {detail}" if detail else ""))


def _dyn_knob(rec, src) -> tuple[float, float]:
    """重建的绕轴力矩 vs 仿真器直接记的 `object/axis_torque`。

    这一路是 S3 采集时另外记下来的，与接触点数据是**两条独立路径**：
    一条是关节的广义力，一条是逐接触点的力乘力臂。它们对得上，说明
    法向定向、力的作用对象、力臂全都没错。
    """
    tau = np.asarray(rec.arrays["mech/generalized"], dtype=np.float64)[:, 0]
    ref = np.asarray(src.arrays["object/axis_torque"], dtype=np.float64)[:, 0]
    m = (np.asarray(rec.arrays["phase"]) == 2) & np.asarray(rec.arrays["valid_s4"])
    m &= np.abs(ref) > 0.02
    if m.sum() < 20:
        return float("nan"), float("nan")
    return (float(np.corrcoef(tau[m], ref[m])[0, 1]),
            float(np.median(np.abs(tau[m])) / max(np.median(np.abs(ref[m])), 1e-9)))


def _dyn_drawer(rec, src) -> tuple[float, float]:
    """重建的沿轨力 vs `m·a + c·v`（质量与阻尼在 S3 的 meta 里）。"""
    f = np.asarray(rec.arrays["mech/generalized"], dtype=np.float64)[:, 0]
    v = np.asarray(src.arrays["object/drawer_joint_vel"], dtype=np.float64)[:, 0]
    phys = src.meta.get("physics", {})
    mass = float(phys.get("drawer_mass_kg", 1.2))
    damp = float(phys.get("drawer_joint_damping", 30.0))
    dt = 1.0 / float(src.meta.get("control_hz", 50.0))
    acc = np.gradient(v, dt)
    need = mass * acc + damp * v
    m = (np.asarray(rec.arrays["phase"]) == 2) & np.asarray(rec.arrays["valid_s4"])
    m &= np.abs(need) > 0.2
    if m.sum() < 20:
        return float("nan"), float("nan")
    return (float(np.corrcoef(f[m], need[m])[0, 1]),
            float(np.median(np.abs(f[m])) / max(np.median(np.abs(need[m])), 1e-9)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="S4 记录目录")
    ap.add_argument("src", help="对应的 S3 数据集目录")
    ap.add_argument("--sample", type=int, default=60, help="抽查几条读回并校验")
    ap.add_argument("--seed", type=int, default=0)
    # 动力学一致性的门槛按"方向对不对"给，不按"数值差多少"——接触力里含
    # 内力与瞬态，幅值本来就不该逐帧相等（P-22）。
    ap.add_argument("--min-dyn-corr", type=float, default=0.5)
    a = ap.parse_args()

    root, src_root = Path(a.root), Path(a.src)
    man = read_manifest(root / "manifest.json")
    src_man = read_manifest(src_root / "manifest.json")
    rng = np.random.default_rng(a.seed)

    check(man["schema_version"] == IR_SCHEMA_VERSION, "manifest schema",
          man["schema_version"])
    check(man["num_episodes"] == src_man["num_episodes"],
          "episode 数与 S3 一致",
          f"{man['num_episodes']} vs {src_man['num_episodes']}")

    # --- 划分必须原样继承（重划会污染校准集，`plan/02` §4.1）---
    s4_split = {e["episode_id"]: e["split"] for e in man["episodes"]}
    s3_split = {e["episode_id"]: e["split"] for e in src_man["episodes"]}
    moved = [k for k in s3_split if s4_split.get(k) != s3_split[k]]
    check(not moved, "划分原样继承 S3",
          f"{len(moved)} 条被移动" if moved else
          f"{len(s3_split)} 条一致，校准集 "
          f"{sum(1 for v in s3_split.values() if v == 'calibration')} 条")

    # --- 表面采样冻结：同一 (物体, 几何变体) 只能有一个 hash ---
    surf_hash: dict[tuple[str, str], set] = {}
    for e in man["episodes"]:
        s = e["meta"]["surface"]
        surf_hash.setdefault((s["object"], s["geom_tag"]), set()).add(s["sha256"])
    multi = {k: v for k, v in surf_hash.items() if len(v) > 1}
    check(not multi, "表面采样已冻结（每个几何变体一个 hash）",
          f"{len(surf_hash)} 个变体" if not multi else str(multi))
    for (obj, tag), hs in sorted(surf_hash.items()):
        live = surface_for(obj, tag)
        got = next(iter(hs))
        check(live.sha256 == got, f"表面 hash 可复现：{obj}/{tag}",
              f"{got[:16]}" if live.sha256 == got else f"记录 {got[:16]} vs 重算 {live.sha256[:16]}")

    # --- 抽查读回：结构、字段归类、NaN、SHA ---
    entries = list(man["episodes"])
    pick = rng.choice(len(entries), size=min(a.sample, len(entries)), replace=False)
    bad_sha, bad_nan, bad_field, fieldsets = [], [], [], set()
    src_by_id = {e["episode_id"]: e for e in src_man["episodes"]}
    dyn_corr, dyn_ratio, over_safety, frames_total = [], [], 0, 0
    part_acc: dict[str, float] = {}
    mode_pairs = np.zeros((4, 4))

    for i in pick:
        e = entries[i]
        path = root / e["path"]
        if sha256_file(path) != e["sha256"]:
            bad_sha.append(e["episode_id"])
        rec = load_episode(path)
        if rec.meta["schema_version"] != RECORD_SCHEMA:
            bad_field.append(f"{e['episode_id']} schema")
        try:
            model = rec.model_arrays()
        except Exception as exc:                     # noqa: BLE001
            bad_field.append(f"{e['episode_id']}: {exc}")
            continue
        if any(k.startswith(("source/", "aux/")) for k in model):
            bad_field.append(f"{e['episode_id']} 模型输入里混进了审计字段")
        fieldsets.add(tuple(sorted(rec.arrays)))
        for name, arr in rec.arrays.items():
            arr = np.asarray(arr)
            if arr.dtype.kind == "f" and not np.isfinite(arr).all():
                bad_nan.append(f"{e['episode_id']}:{name}")

        f_total = np.asarray(rec.arrays["aux/frame_force"], dtype=np.float64)
        over_safety += int((f_total > SAFETY_FORCE_N).sum())
        frames_total += len(f_total)

        raw = np.asarray(rec.arrays["mode/raw"]).ravel()
        new = np.asarray(rec.arrays["mode/label"]).ravel()
        w = np.asarray(rec.arrays["region/weight"], dtype=np.float64).ravel()
        for r in range(4):
            for n in range(4):
                mode_pairs[r, n] += float(w[(raw == r) & (new == n)].sum())

        surf = surface_for(rec.meta["surface"]["object"], rec.meta["surface"]["geom_tag"])
        if rec.meta["success"]:
            for k, v in part_force_share(rec, surf, phase=2).items():
                part_acc[k] = part_acc.get(k, 0.0) + v

        # --- 动力学一致性 ---
        src_entry = src_by_id.get(e["episode_id"])
        if src_entry is not None and rec.meta["success"]:
            src = load_episode(src_root / src_entry["path"])
            task = rec.meta["task"]
            if task == "knob":
                c, r = _dyn_knob(rec, src)
            elif task == "drawer":
                c, r = _dyn_drawer(rec, src)
            else:
                c, r = float("nan"), float("nan")
            if np.isfinite(c):
                dyn_corr.append(c)
                dyn_ratio.append(r)

    check(not bad_sha, "抽查 SHA-256 一致", f"{len(pick)} 条")
    check(not bad_nan, "无 NaN / Inf", "; ".join(bad_nan[:3]))
    check(not bad_field, "字段归类 fail-closed 通过", "; ".join(bad_field[:3]))
    check(len(fieldsets) == 1, "抽查的 episode 字段集合一致",
          f"{len(fieldsets)} 种字段集合")

    n_ok = sum(1 for i in pick if entries[i]["meta"]["success"])
    if part_acc and n_ok:
        share = {k: v / n_ok for k, v in sorted(part_acc.items(), key=lambda kv: -kv[1])}
        info("接触力按部件（操作阶段，抽查集）",
             "  ".join(f"{k} {100 * v:.2f}%" for k, v in share.items() if v > 1e-4))

    tot = mode_pairs.sum()
    if tot > 0:
        names = ("no_contact", "sticking", "sliding", "separating")
        print("[INFO] mode 迁移矩阵（行 = S3 原始，列 = S4 重判，法向力加权 %）")
        print("       " + "".join(f"{n:>13}" for n in names))
        for r in range(4):
            print(f"       {names[r]:<12}" +
                  "".join(f"{100 * mode_pairs[r, n] / tot:>12.1f}%" for n in range(4)))

    info("超安全上限的帧",
         f"{100 * over_safety / max(frames_total, 1):.2f}%（总接触力 > {SAFETY_FORCE_N} N）")

    if dyn_corr:
        med_c = float(np.median(dyn_corr))
        med_r = float(np.median(dyn_ratio))
        check(med_c >= a.min_dyn_corr, "动力学一致性（`plan/02` §8）",
              f"重建广义力与物体动力学的相关中位 {med_c:.3f}，"
              f"幅值比中位 {med_r:.2f}（{len(dyn_corr)} 条）")
    else:
        info("动力学一致性", "该任务不适用（擦拭平面是 kinematic 刚体，"
                             "接触力不进入它的运动方程）—— 如实标 N/A")

    if FAIL:
        print(f"\n===== {len(FAIL)} 项未通过 =====")
        for line in FAIL:
            print(line)
        return 1
    print("\n===== 全部通过 =====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
