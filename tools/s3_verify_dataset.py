"""对采完的 source 数据集做**独立**验收，不依赖生成脚本的自我报告。

生成脚本报的数字来自它自己的内存缓冲；这里从落盘的文件重新读一遍，
检查的是数据集作为**数据集**是否成立：文件没坏、划分不重不漏、
留出集名副其实、审计字段没漏进模型输入。

`plan/03` §7 的四个测试集加一个校准集，只要有一个装错东西，
下游的泛化数字就是假的而且不报错——records.py 里已经因为
"unseen_physics_test 用随机 shuffle 填充"修过一次同类 bug。

只用 numpy，不需要 Isaac Sim，系统 python3 就能跑::

    python3 tools/s3_verify_dataset.py /tmp/s3_drawer_v3 --sample 40
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from it.records import (  # noqa: E402
    MODEL_INPUT_PREFIXES,
    SOURCE_PREFIX,
    load_episode,
    read_manifest,
    sha256_file,
)

FAIL = []


def check(ok: bool, name: str, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    line = f"[{tag}] {name}" + (f"  —— {detail}" if detail else "")
    print(line)
    if not ok:
        FAIL.append(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--sample", type=int, default=40, help="抽查几条 episode 读回")
    ap.add_argument("--seed", type=int, default=0)
    # 数量门槛按数据集类型给，不写死。`plan/03` §6：
    #   抽屉  --min-total 450 --min-per-family 150 --min-families 3
    #   探针  --min-total 0   --min-per-family 200 --min-families 1（逐原语看）
    ap.add_argument("--min-total", type=int, default=450)
    ap.add_argument("--min-per-family", type=int, default=150)
    ap.add_argument("--min-families", type=int, default=3)
    a = ap.parse_args()

    root = Path(a.root)
    man = read_manifest(root / "manifest.json")
    eps = man["episodes"]
    splits = man["splits"]
    print(f"数据集 {man['dataset_name']}  schema {man['schema_version']}  "
          f"{man['num_episodes']} episode  生成于 {man['generator_git_sha'][:8]}\n")

    # ---- 划分：不重不漏 ----
    all_ids = {e["episode_id"] for e in eps}
    assigned = [i for v in splits.values() for i in v]
    check(len(assigned) == len(set(assigned)), "划分之间不重叠",
          f"{len(assigned)} 个分配、{len(set(assigned))} 个唯一")
    check(set(assigned) == all_ids, "划分覆盖全部 episode",
          f"漏 {len(all_ids - set(assigned))}，多 {len(set(assigned) - all_ids)}")

    by_id = {e["episode_id"]: e for e in eps}
    ids_all = list(by_id)

    # ---- 留出集名副其实（这两条就是 records.py 修过的那个 bug 的守门人）----
    #
    # 判据按**数据集自己声明的意图**来，不假定每个数据集都有留出集：
    # 探针物体集就没有策略留出（它整体都是 E-I 的训练指令来源，
    # 留出是在**任务**层面做的，见 `plan/04` §5.4），自由刚体也没有关节阻尼可变。
    # 但"声明了却没兑现"和"装错了东西"都必须抓住。
    want_fam = man.get("holdout_strategy_family")
    hold_fam = splits.get("unseen_strategy_test", [])
    if want_fam:
        fam_hold = {by_id[i]["strategy_family"] for i in hold_fam}
        fam_train = {by_id[i]["strategy_family"] for i in splits.get("train", [])}
        check(fam_hold == {want_fam} and not (fam_hold & fam_train),
              "unseen_strategy_test 只含声明的留出家族，且该家族不在训练集里",
              f"声明留出 {want_fam}，实际留出 {sorted(fam_hold)}，训练 {sorted(fam_train)}")
    else:
        check(not hold_fam, "未声明策略留出，unseen_strategy_test 应为空",
              f"却有 {len(hold_fam)} 条")
        print("[INFO] 本数据集未声明留出策略家族")

    # ---- 几何留出（`plan/03` §7 表第 6 行）----
    has_geom = [i for i in ids_all
                if by_id[i].get("geometry_variant", "nominal") != "nominal"]
    hold_geom = splits.get("unseen_geometry_test", [])
    if has_geom:
        tags = {by_id[i].get("geometry_variant") for i in hold_geom}
        check("nominal" not in tags and tags,
              "unseen_geometry_test 只含非名义几何", f"{sorted(tags)}")
        # 与物理留出同理：划分有优先级，前面的桶要扣掉。
        expect = (set(has_geom) - set(splits.get("failed", []))
                  - set(splits.get("unseen_strategy_test", []))
                  - set(splits.get("unseen_implementation_test", [])))
        check(set(hold_geom) == expect,
              "所有成功、且不属于前序留出的几何变体都进了 unseen_geometry_test",
              f"变体 {len(has_geom)} 条，应留出 {len(expect)} 条，实际 {len(hold_geom)} 条")
        check(all(by_id[i].get("geometry_variant", "nominal") == "nominal"
                  for i in splits.get("train", [])),
              "训练集只含名义几何")
    else:
        check(not hold_geom, "没有几何变体样本，unseen_geometry_test 应为空",
              f"却有 {len(hold_geom)} 条")

    # ---- 跨实现留出（`plan/03` §7 末尾，对应 `02` §7 第 8 条）----
    impls = {by_id[i].get("implementation", "default") for i in ids_all}
    hold_impl = splits.get("unseen_implementation_test", [])
    if len(impls) > 1:
        check(bool(hold_impl), "多实现数据集必须留出一种实现作跨实现测试",
              f"实现有 {sorted(impls)}，留出 {len(hold_impl)} 条")
        held = {by_id[i].get("implementation") for i in hold_impl}
        check(len(held) == 1, "unseen_implementation_test 只含**一种**实现",
              f"{sorted(held)}")
        check(all(by_id[i].get("implementation") not in held
                  for i in splits.get("train", [])),
              "训练集不含被留出的那种实现")
        n_success_held = len([i for i in ids_all
                              if by_id[i].get("implementation") in held
                              and by_id[i]["success"]])
        check(len(hold_impl) == n_success_held,
              "被留出实现的**全部**成功 episode 都在该桶里",
              f"该实现成功 {n_success_held} 条，桶里 {len(hold_impl)} 条")

    has_var = [i for i in ids_all
               if by_id[i]["meta"].get("physics_variant", "nominal") != "nominal"]
    hold_phys = splits.get("unseen_physics_test", [])
    if has_var:
        phys_hold = {by_id[i]["meta"].get("physics_variant") for i in hold_phys}
        check("nominal" not in phys_hold and phys_hold,
              "unseen_physics_test 只含非名义物理参数", f"{sorted(phys_hold)}")
        # 划分是有优先级的（失败 -> 跨实现 -> 策略 -> 几何 -> 物理），
        # 一条既带物理变体、又落在前面任何一个桶里的 episode 会被前面拿走。
        # 判据必须把**前面所有桶**都扣掉，否则会得到一个假的 FAIL——
        # 加几何桶那次就当场撞上了（5 条变体里 2 条被几何桶先拿走）。
        expect = (set(has_var) - set(splits.get("failed", []))
                  - set(splits.get("unseen_implementation_test", []))
                  - set(splits.get("unseen_strategy_test", []))
                  - set(splits.get("unseen_geometry_test", [])))
        check(set(hold_phys) == expect,
              "所有成功、且不属于留出家族的物理变体都进了 unseen_physics_test",
              f"变体 {len(has_var)} 条，应留出 {len(expect)} 条，实际 {len(hold_phys)} 条")
        damp_h = [by_id[i]["meta"]["physics"].get("joint_damping",
                  by_id[i]["meta"]["physics"].get("drawer_joint_damping", 0.0))
                  for i in hold_phys]
        damp_t = [by_id[i]["meta"]["physics"].get("joint_damping",
                  by_id[i]["meta"]["physics"].get("drawer_joint_damping", 0.0))
                  for i in splits.get("train", [])]
        if damp_h and damp_t and max(damp_t) > 0:
            # 逐个值判，不比区间端点：留出区间是**训练区间两侧各一段**，
            # 两段的凸包当然把训练区间包在里面，比端点会得到假的 FAIL。
            lo, hi = min(damp_t), max(damp_t)
            inside = [v for v in damp_h if lo <= v <= hi]
            check(not inside, "没有留出样本落在训练的物理参数区间内",
                  f"训练 [{lo:.2f}, {hi:.2f}]，留出 {len(inside)} 个落在区间内；"
                  f"留出取值 [{min(damp_h):.2f}, {max(damp_h):.2f}]")
    else:
        check(not hold_phys, "没有物理变体样本，unseen_physics_test 应为空",
              f"却有 {len(hold_phys)} 条")
        print("[INFO] 本数据集没有可变的物理参数（自由刚体无关节阻尼）")
    check(all(by_id[i]["meta"].get("physics_variant", "nominal") == "nominal"
              for i in splits.get("train", [])),
          "训练集只含名义物理参数")

    check(len(splits.get("calibration", [])) > 0,
          "校准集非空（conformal 阈值要用，D-17）",
          f"{len(splits.get('calibration', []))} 条")

    # ---- 文件完整性 + 契约 ----
    rng = random.Random(a.seed)
    picked = rng.sample(eps, min(a.sample, len(eps)))
    bad_hash, bad_load, leaked, n_frames = [], [], [], set()
    for e in picked:
        path = root / e["path"]
        try:
            if e.get("sha256") and sha256_file(path) != e["sha256"]:
                bad_hash.append(e["episode_id"])
            rec = load_episode(path)
            n_frames.add(rec.num_frames)
            model = rec.model_arrays()
            if any(k.startswith(SOURCE_PREFIX) for k in model):
                leaked.append(e["episode_id"])
        except Exception as exc:  # noqa: BLE001
            bad_load.append(f"{e['episode_id']}: {type(exc).__name__}: {exc}")
    check(not bad_hash, f"抽查 {len(picked)} 条的 SHA-256 与 manifest 一致",
          ", ".join(bad_hash[:3]))
    check(not bad_load, "抽查的 episode 都能读回并通过契约校验",
          "; ".join(bad_load[:2]))
    check(not leaked, "model_arrays() 里没有 source/* 审计字段", ", ".join(leaked[:3]))
    check(len(n_frames) == 1, "所有 episode 帧数一致", f"{sorted(n_frames)}")

    # ---- 数值健康 ----
    rec = load_episode(root / picked[0]["path"])
    fields = sorted(rec.arrays)
    n_src = sum(k.startswith(SOURCE_PREFIX) for k in fields)
    n_mod = sum(k.startswith(MODEL_INPUT_PREFIXES) for k in fields)
    print(f"\n字段 {len(fields)} 个：模型输入 {n_mod}，审计 {n_src}")
    nan = [k for k, v in rec.arrays.items()
           if np.issubdtype(np.asarray(v).dtype, np.floating)
           and not np.isfinite(np.asarray(v)).all()]
    check(not nan, "抽查 episode 无 NaN/Inf", ", ".join(nan[:3]))

    ok_rate = sum(e["success"] for e in eps) / max(len(eps), 1)
    fam = {}
    for e in eps:
        if e["success"]:
            fam[e["strategy_family"]] = fam.get(e["strategy_family"], 0) + 1
    print(f"\n成功率 {100 * ok_rate:.1f}%，各家族成功轨迹数：")
    for k in sorted(fam):
        print(f"  {k:<16}{fam[k]:>5}")
    if a.min_total:
        check(sum(fam.values()) >= a.min_total,
              f"成功轨迹总数 ≥{a.min_total}（`plan/03` §6）", f"{sum(fam.values())}")
    n_ok = sum(1 for v in fam.values() if v >= a.min_per_family)
    check(n_ok >= a.min_families,
          f"至少 {a.min_families} 个家族各有 ≥{a.min_per_family} 条成功轨迹"
          f"（`plan/03` §6）",
          f"达标 {n_ok} 个，未达标 "
          f"{sorted(k for k, v in fam.items() if v < a.min_per_family)}")

    print()
    if FAIL:
        print(f"===== {len(FAIL)} 项未通过 =====")
        for line in FAIL:
            print(line)
        return 1
    print("===== 全部通过 =====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
