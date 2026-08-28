"""对采完的 source 数据集做**独立**验收，不依赖生成脚本的自我报告。

生成脚本报的数字来自它自己的内存缓冲；这里从落盘的文件重新读一遍，
检查的是数据集作为**数据集**是否成立：文件没坏、划分不重不漏、
留出集名副其实、审计字段没漏进模型输入。

`plan/03` §7 的四个测试集加一个校准集，只要有一个装错东西，
下游的泛化数字就是假的而且不报错——records.py 里已经因为
"unseen_physics_test 用随机 shuffle 填充"修过一次同类 bug。

只用 numpy，不需要 Isaac Sim，系统 python3 就能跑::

    python3 tools/s3_verify_dataset.py /tmp/s3_drawer_full --sample 40
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

    # ---- 留出集名副其实（这两条就是 records.py 修过的那个 bug 的守门人）----
    fam_hold = {by_id[i]["strategy_family"] for i in splits.get("unseen_strategy_test", [])}
    fam_train = {by_id[i]["strategy_family"] for i in splits.get("train", [])}
    check(len(fam_hold) == 1 and not (fam_hold & fam_train),
          "unseen_strategy_test 只含留出家族，且该家族不在训练集里",
          f"留出 {sorted(fam_hold)}，训练 {sorted(fam_train)}")

    phys_hold = {by_id[i]["meta"].get("physics_variant") for i in
                 splits.get("unseen_physics_test", [])}
    phys_train = {by_id[i]["meta"].get("physics_variant") for i in splits.get("train", [])}
    check("nominal" not in phys_hold and phys_hold,
          "unseen_physics_test 只含非名义物理参数", f"{sorted(phys_hold)}")
    check(phys_train == {"nominal"}, "训练集只含名义物理参数", f"{sorted(phys_train)}")

    damp_h = [by_id[i]["meta"]["physics"]["drawer_joint_damping"]
              for i in splits.get("unseen_physics_test", [])]
    damp_t = [by_id[i]["meta"]["physics"]["drawer_joint_damping"]
              for i in splits.get("train", [])]
    if damp_h and damp_t:
        # 逐个值判，不比区间端点：留出区间是**训练区间两侧各一段**，
        # 两段的凸包当然把训练区间包在里面，比端点会得到假的 FAIL。
        lo, hi = min(damp_t), max(damp_t)
        inside = [v for v in damp_h if lo <= v <= hi]
        check(not inside, "没有留出样本落在训练的物理参数区间内",
              f"训练 [{lo:.1f}, {hi:.1f}]，留出 {len(inside)} 个落在区间内；"
              f"留出取值 [{min(damp_h):.1f}, {max(damp_h):.1f}]")

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
    check(sum(fam.values()) >= 450, "抽屉成功轨迹总数 ≥450（`plan/03` §6）",
          f"{sum(fam.values())}")
    check(sum(1 for v in fam.values() if v >= 150) >= 3,
          "至少 3 个策略家族各有 ≥150 条成功轨迹（`plan/03` §6）",
          f"达标家族 {sum(1 for v in fam.values() if v >= 150)} 个")

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
