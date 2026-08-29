"""S4 泄漏检查：`plan/02` §7 的九条，逐条给状态。

**有两条本步做不完，如实标"留到 S5"，不写成通过。** 第 4 条要的是"从
*Functional Envelope* 预测策略身份"，第 8 条要的是"两种实现的 envelope
可互换"——envelope 是 S5 的产物，S4 只有逐条示教的 Oracle Record。
这里能做的是它们的**记录级前身**，也照样报数，但标签必须写清楚是哪一级。

第 9 条（region 可推导性探针）单独一个脚本 `tools/s4_region_probe.py`，
因为它要训一个网络，跑起来比这里慢一个量级。

用法::

    PYTHONPATH=src /isaac-sim/python.sh tools/s4_leak_checks.py \\
        /tmp/s4_drawer /tmp/s3_drawer_v3

第 1 条（刚体旋转不变性）做的是**代数等变性测试**：对已采 episode 施加任意
旋转 R——把物体系里的接触位置/法向/摩擦力/相对速度、物体位姿和表面采样一起
转过去——再看提取器的输出。**限制必须写明**：它检验的是提取器里有没有混进
硬编码的世界轴，**不检验仿真器的坐标处理**；后者由 S3 已有的接触部位统计
间接支撑（抽屉 90.2% 落在把手背面这类数字，坐标错了不可能对）。

``mech/generalized`` 也参加这个测试。它按物体的**标准轴**定义（抽屉沿 +X、
旋钮绕 +Z、平面压 −Z），而那些轴长在物体自己身上——场景一转它们跟着转，
所以广义力在场景旋转下就该**逐元素不变**，不是等变。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from it.interaction import SLIP_SPEED_MIN, extract, spec_for  # noqa: E402
from it.records import EpisodeRecord, load_episode, read_manifest  # noqa: E402
from it.surfaces import surface_for  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s3_diversity import features as source_features  # noqa: E402
from s3_diversity import softmax_fit  # noqa: E402

FAIL: list[str] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    line = f"[{tag}] {name}" + (f"  —— {detail}" if detail else "")
    print(line)
    if not ok:
        FAIL.append(line)


def defer(name: str, detail: str) -> None:
    print(f"[DEFER] {name}  —— {detail}")


def info(name: str, detail: str = "") -> None:
    print(f"[INFO] {name}" + (f"  —— {detail}" if detail else ""))


def _rand_rot(rng: np.random.Generator) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    th = rng.uniform(0.3, 2.8)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(th) * k + (1 - np.cos(th)) * (k @ k)


def _quat_left(rot: np.ndarray, quat: np.ndarray) -> np.ndarray:
    from it.interaction import _quat_to_rot  # noqa: PLC0415

    m = np.einsum("ij,tjk->tik", rot, _quat_to_rot(quat))
    w = np.sqrt(np.clip(1.0 + np.trace(m, axis1=1, axis2=2), 1e-12, None)) / 2.0
    return np.stack([w, (m[:, 2, 1] - m[:, 1, 2]) / (4 * w),
                     (m[:, 0, 2] - m[:, 2, 0]) / (4 * w),
                     (m[:, 1, 0] - m[:, 0, 1]) / (4 * w)], axis=1)


def rotate_scene(rec: EpisodeRecord, rot: np.ndarray) -> EpisodeRecord:
    """把**整个场景**绕世界原点转过去：所有世界系字段转，物体系字段一个不动。

    这才是 `plan/02` §7 第 1 条的原话——"整个场景刚体旋转后，物体系表示
    **逐元素一致**"。场景一转，物体的世界位姿、接触体的世界位姿全都跟着转，
    而"接触点在物体系里的坐标"按定义不变。于是提取器的输出必须**逐位相同**；
    一旦某处硬编码了世界轴（"+X 是拉出方向"、"up = +Z"），这个测试立刻炸。

    ⚠️ 第一版写错过：那一版把物体系的 `contact/*` 也转了，同时又把物体的世界
    姿态往同方向转——等于把同一个旋转记了两次，于是滑移速度差到 0.52 m/s。
    错的是测试，不是提取器。擦拭那份恰好没暴露（它的物体是静止且姿态恒等的）。
    """
    arrays = {}
    for k, v in rec.arrays.items():
        a = np.asarray(v)
        world_pose = a.ndim == 2 and a.shape[-1] == 7 and (
            k.endswith("root_pose") or k.endswith("_pose") or k == "object/state")
        world_pos = a.ndim == 2 and a.shape[-1] == 3 and k.endswith("_pos_w")
        world_quat = a.ndim == 2 and a.shape[-1] == 4 and k.endswith("_quat_w")
        world_pts = a.ndim == 3 and a.shape[-1] == 3 and k.endswith("contact_pos_w")
        if world_pose:
            p = np.einsum("ij,tj->ti", rot, a[:, :3].astype(np.float64))
            q = _quat_left(rot, a[:, 3:7].astype(np.float64))
            arrays[k] = np.concatenate([p, q], axis=1).astype(a.dtype)
        elif world_pos:
            arrays[k] = np.einsum("ij,tj->ti", rot, a.astype(np.float64)).astype(a.dtype)
        elif world_quat:
            arrays[k] = _quat_left(rot, a.astype(np.float64)).astype(a.dtype)
        elif world_pts:
            arrays[k] = np.einsum("ij,tkj->tki", rot, a.astype(np.float64)).astype(a.dtype)
        else:
            arrays[k] = a          # contact/* 是物体系的，按定义不动
    return EpisodeRecord(meta=dict(rec.meta), arrays=arrays)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="S4 记录目录")
    ap.add_argument("src", help="对应的 S3 数据集目录")
    ap.add_argument("--sample", type=int, default=20, help="等变性测试抽几条")
    ap.add_argument("--clf-sample", type=int, default=400, help="分类器用几条")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    root, src_root = Path(a.root), Path(a.src)
    man = read_manifest(root / "manifest.json")
    src_man = read_manifest(src_root / "manifest.json")
    rng = np.random.default_rng(a.seed)
    src_by_id = {e["episode_id"]: e for e in src_man["episodes"]}
    ok_entries = [e for e in man["episodes"] if e["success"]]

    # ---------------- 1. 场景刚体旋转后物体系表示逐元素一致 ----------------
    pick = rng.choice(len(ok_entries), size=min(a.sample, len(ok_entries)), replace=False)
    worst = {"idx": 0, "mode": 0, "effect": 0.0, "wrench": 0.0, "slip": 0.0}
    for i in pick:
        e = ok_entries[int(i)]
        src = load_episode(src_root / src_by_id[e["episode_id"]]["path"])
        spec = spec_for(src.meta)
        tag = str(src.meta.get("geometry_variant", "nominal"))
        surf = surface_for(spec.obj, tag)
        base = extract(src, surf)
        rot = _rand_rot(rng)
        # 表面采样不跟着转：场景转了，但物体在**自己**的坐标系里没变
        turned = extract(rotate_scene(src, rot), surf)

        worst["idx"] = max(worst["idx"], int(
            (np.asarray(base.arrays["region/point_idx"])
             != np.asarray(turned.arrays["region/point_idx"])).sum()))
        # mode 比**连续量**，不比标签：标签是按 5 mm/s 切出来的，卡在阈值上的点
        # 会因为旋转多绕几步浮点运算而翻面。那是数值噪声，不是坐标依赖。
        # 判据因此是：滑移速度的差要小，且**离阈值不近**的点标签必须一致。
        s0 = np.asarray(base.arrays["mode/slip_speed"], dtype=np.float64)
        s1 = np.asarray(turned.arrays["mode/slip_speed"], dtype=np.float64)
        worst["slip"] = max(worst["slip"], float(np.abs(s0 - s1).max()))
        far = np.abs(s0 - SLIP_SPEED_MIN) > 0.05 * SLIP_SPEED_MIN
        worst["mode"] = max(worst["mode"], int(
            ((np.asarray(base.arrays["mode/label"])
              != np.asarray(turned.arrays["mode/label"])) & far).sum()))
        worst["effect"] = max(worst["effect"], float(np.abs(
            np.asarray(base.arrays["effect/future"], dtype=np.float64)
            - np.asarray(turned.arrays["effect/future"], dtype=np.float64)).max()))
        # 物体系的力与力矩在场景旋转下**不变**（不是等变）——它们本来就表达在
        # 物体自己的坐标系里，场景怎么摆与它无关。
        for key in ("mech/wrench_obj", "mech/generalized"):
            worst["wrench"] = max(worst["wrench"], float(np.abs(
                np.asarray(base.arrays[key], dtype=np.float64)
                - np.asarray(turned.arrays[key], dtype=np.float64)).max()))

    measured = bool(load_episode(root / ok_entries[0]["path"])
                    .meta["extraction"].get("object_pose_measured", True))
    if not measured:
        # 这份数据没记被操作物体的位姿（擦拭平面是 kinematic 的），场景一转
        # 平面也该跟着转，而记录里没有那个自由度。**不许在假定之上报通过。**
        defer("1. 场景刚体旋转",
              "该数据集没有记录被操作物体的位姿（擦拭平面是 kinematic），"
              "无法如实施加场景旋转。下一轮采集把平面的 root pose 记上（一列常量）"
              "即可补测；region / effect / 力与力矩三项在下面单独报")
        check(worst["idx"] == 0 and worst["effect"] < 1e-5,
              "1a. 场景旋转下 region 与 effect 仍逐元素一致",
              f"region 索引差 {worst['idx']} 个、effect 最大差 {worst['effect']:.2e}"
              "（mode 那一路因上述原因不参与）")
    else:
        check(worst["idx"] == 0 and worst["mode"] == 0 and worst["effect"] < 1e-5
              and worst["slip"] < 1e-4,
              "1. 场景刚体旋转：不变量逐元素一致",
              f"{len(pick)} 条 × 随机旋转；region 索引差 {worst['idx']} 个、"
              f"滑移速度最大差 {worst['slip']:.2e} m/s、"
              f"离阈值 5% 以外的 mode 差 {worst['mode']} 个、"
              f"effect 最大差 {worst['effect']:.2e}")

    check(worst["wrench"] < 1e-3, "1b. 物体系的力、力矩与广义力逐元素不变",
          f"最大残差 {worst['wrench']:.2e} N / N·m")
    info("1c. 这条测试的限制",
         "只检验提取器有没有混进硬编码世界轴；不检验仿真器的坐标处理")

    # ---------------- 2 / 5 / 7. 结构性隔离 ----------------
    sample = load_episode(root / ok_entries[0]["path"])
    keys = sorted(sample.arrays)
    check(not [k for k in keys if k.startswith("source/")],
          "2. 记录里没有 source/* 字段", f"{len(keys)} 个数组")

    # 更强的一条：把输入里的 source/* 全删掉，看输出有哪些字段会变。
    # **允许变的只有滑移速度那两路**——mode 的主判据是接触体与物体的位姿差分
    # （见 `interaction._pose_slip`），那是 oracle 用特权数据算出来的**物理量**，
    # 不是把 source 字段写进表示。`plan/02` §1 禁的是后者。
    # 其余任何字段一旦随 source 变化，就说明采集侧的东西漏进表示了。
    src0 = load_episode(src_root / src_by_id[ok_entries[0]["episode_id"]]["path"])
    spec0 = spec_for(src0.meta)
    surf0 = surface_for(spec0.obj, str(src0.meta.get("geometry_variant", "nominal")))
    stripped = EpisodeRecord(meta=dict(src0.meta),
                             arrays={k: v for k, v in src0.arrays.items()
                                     if not k.startswith("source/")})
    a0, a1 = extract(src0, surf0).arrays, extract(stripped, surf0).arrays
    differ = sorted(k for k in a0
                    if not np.array_equal(np.asarray(a0[k]), np.asarray(a1[k])))
    # 允许变的就是滑移那一路：`mode/slip_speed` 是主判据的值本身，
    # `mode/pose_slip` 是它的诊断副本，`mode/label` 是按它切出来的标签。
    allowed = {"mode/pose_slip", "mode/slip_speed", "mode/label"}
    check(set(a0) == set(a1) and set(differ) <= allowed,
          "2b. 去掉 source/* 后，只有滑移速度那一路会变",
          f"实际变化的字段：{differ or '无'}（允许：{sorted(allowed)}）")

    model_keys = sorted(sample.model_arrays())
    banned = [k for k in model_keys
              if any(t in k for t in ("plate", "tool", "body", "joint_id", "family",
                                      "strategy", "friction", "damping", "mass"))]
    check(not banned, "7. 表示里没有执行器专属编号 / 部件名", str(banned))
    check(not any(k.startswith("aux/") for k in model_keys),
          "5. 模型输入里没有 source 身份类字段",
          "strategy_family 只在 meta 的审计字段里")

    # ---------------- 3. 改变接触体数量，维度不变 ----------------
    by_family: dict[str, list] = {}
    for e in ok_entries:
        by_family.setdefault(e["strategy_family"], []).append(e)
    shapes = {}
    for fam, lst in sorted(by_family.items()):
        rec = load_episode(root / lst[0]["path"])
        shapes[fam] = (tuple(sorted(rec.arrays)),
                       rec.arrays["mech/wrench_obj"].shape[1],
                       rec.arrays["effect/future"].shape[1:],
                       rec.meta["fields"]["n_bodies"])
    fieldsets = {v[0] for v in shapes.values()}
    dims = {(v[1], v[2]) for v in shapes.values()}
    bodies = {v[3] for v in shapes.values()}
    check(len(fieldsets) == 1 and len(dims) == 1,
          "3. 改变 source 接触体数量后表示维度不变",
          f"{len(shapes)} 个家族（含 {sorted(bodies)} 个接触体的实现）"
          f"共用同一套字段与维度")

    # ---------------- 4. 从表示识别策略（记录级） ----------------
    fams = sorted(by_family)
    if len(fams) >= 2:
        n_each = max(2, a.clf_sample // len(fams))
        xs_rec, xs_src, ys = [], [], []
        for ci, fam in enumerate(fams):
            for e in by_family[fam][:n_each]:
                rec = load_episode(root / e["path"])
                xs_rec.append(record_features(rec))
                xs_src.append(source_features(
                    load_episode(src_root / src_by_id[e["episode_id"]]["path"])))
                ys.append(ci)
        acc_rec = _cv_acc(np.array(xs_rec), np.array(ys), len(fams), a.seed)
        acc_src = _cv_acc(np.array(xs_src), np.array(ys), len(fams), a.seed)
        chance = 1.0 / len(fams)
        info("4. 策略身份可识别性（**记录级**，不是 envelope 级）",
             f"从 Oracle Record {acc_rec:.3f} vs 从原始 source 动作 {acc_src:.3f}"
             f"（随机 {chance:.3f}，{len(fams)} 类）")
        defer("4. envelope 级的那一半",
              "`plan/02` §7 第 4 条要的是从 Functional Envelope 预测策略身份，"
              "envelope 是 S5 的产物；两个数将来并排报")

    # ---------------- 6. 物理参数不进表示 ----------------
    phys = {}
    for e in ok_entries:
        phys.setdefault(e["physics_variant"], []).append(e)
    if len(phys) >= 2:
        names = sorted(phys)
        n_each = max(2, a.clf_sample // len(names))
        xs, ys = [], []
        for ci, name in enumerate(names):
            for e in phys[name][:n_each]:
                xs.append(record_features(load_episode(root / e["path"])))
                ys.append(ci)
        acc = _cv_acc(np.array(xs), np.array(ys), len(names), a.seed)
        info("6. 物理参数可推断性（诊断，不设门槛）",
             f"从记录预测 physics_variant {acc:.3f}（随机 {1 / len(names):.3f}，"
             f"{names}）—— 物理参数没有被直接写进表示，但它必然通过"
             "实际交互体现出来，这个数不该是 0")

    # ---------------- 8. 擦拭：与"是否用工具"无关 ----------------
    impls = sorted({e["implementation"] for e in ok_entries})
    if len(impls) >= 2:
        tool_fields = [k for k in model_keys if "tool" in k or "eraser" in k]
        check(not tool_fields, "8. 表示里没有工具的存在性/位姿/几何字段",
              f"两种实现：{impls}")
        stats = {}
        for impl in impls:
            sub = [e for e in ok_entries if e["implementation"] == impl][:200]
            stats[impl] = np.mean([record_features(load_episode(root / e["path"]))
                                   for e in sub], axis=0)
        d = np.abs(stats[impls[0]] - stats[impls[1]])
        scale = np.abs(stats[impls[0]]) + np.abs(stats[impls[1]]) + 1e-9
        info("8b. 两种实现的记录级统计差异",
             f"逐特征相对差中位 {float(np.median(d / scale)):.3f}"
             f"（1.0 = 完全不同，0 = 完全一样）")
        defer("8. envelope 互换实验",
              "要把实现 (a) 的 envelope 喂给只会实现 (b) 的执行器，属 S5/S7")
    else:
        info("8. 擦拭跨实现检查", f"该数据集只有一种实现（{impls}），不适用")

    defer("9. region 可推导性探针", "单独一个脚本：tools/s4_region_probe.py")

    if FAIL:
        print(f"\n===== {len(FAIL)} 项未通过 =====")
        for line in FAIL:
            print(line)
        return 1
    print("\n===== 可在 S4 完成的项全部通过（DEFER 的两条留到 S5）=====")
    return 0


def record_features(rec: EpisodeRecord) -> np.ndarray:
    """一条 Oracle Record 的低阶统计特征。**只取模型可见字段**。

    与 `s3_diversity.features` 对称：那边只取 `source/*`（动作层面），
    这边只取表示层面。两个准确率并排报才说明"表示抹掉了多少策略特异性"。
    """
    arrays = rec.model_arrays()
    act = np.asarray(arrays["region/valid"])
    w = np.asarray(arrays["region/weight"], dtype=np.float64) * act
    tot = max(float(w.sum()), 1e-9)
    parts: list[float] = []

    idx = np.asarray(arrays["region/point_idx"])
    pos = np.asarray(arrays["region/pos_obj"], dtype=np.float64)
    cen = (pos * w[..., None]).sum(axis=(0, 1)) / tot
    spread = np.sqrt(((pos - cen) ** 2 * w[..., None]).sum(axis=(0, 1)) / tot)
    parts += list(cen) + list(spread)
    parts.append(float((idx >= 0).sum()) / max(act.sum(), 1))

    eng = np.asarray(arrays["engage/dir"], dtype=np.float64)
    parts += list((eng * w[..., None]).sum(axis=(0, 1)) / tot)

    mode = np.asarray(arrays["mode/label"])
    parts += [float(w[mode == i].sum() / tot) for i in range(4)]
    for key in ("mode/slip_speed", "mode/cone_ratio"):
        v = np.asarray(arrays[key], dtype=np.float64)[act]
        parts += [float(v.mean()) if v.size else 0.0,
                  float(np.quantile(v, 0.9)) if v.size else 0.0]

    for key in ("mech/wrench_obj", "mech/generalized"):
        v = np.asarray(arrays[key], dtype=np.float64)
        parts += list(v.mean(axis=0)) + list(v.std(axis=0)) + list(np.abs(v).max(axis=0))

    eff = np.asarray(arrays["effect/current"], dtype=np.float64)
    parts += list(eff[-1] - eff[0]) + list(eff.std(axis=0))
    return np.asarray(parts, dtype=np.float64)


def _cv_acc(x: np.ndarray, y: np.ndarray, n_cls: int, seed: int) -> float:
    """按 episode 划分的留出准确率（P-10：绝不按帧划）。"""
    x = np.nan_to_num(x)
    mu, sd = x.mean(0), x.std(0) + 1e-9
    x = (x - mu) / sd
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    cut = int(len(x) * 0.7)
    tr, te = order[:cut], order[cut:]
    w, b = softmax_fit(x[tr], y[tr], n_cls, seed=seed)
    pred = np.argmax(x[te] @ w + b, axis=1)
    return float((pred == y[te]).mean())


if __name__ == "__main__":
    sys.exit(main())
