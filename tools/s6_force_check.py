#!/usr/bin/env python3
"""在线构造的接触力 vs S4 离线的 `mech/force_obj`——**从同一批原始量开始**对拍。

**为什么需要这一条。** P-72 定下的规矩是"同一个量的两份实现，必须有一条把它们放在
同一批输入上对拍的测试"。`tools/s6_reward_probe.py` 第一节确实对过 traction，
**但它喂进去的是已经构造好的 `mech/force_obj`**——只覆盖了那个式子下游的一段，
而错误在上游：E-I 环境里原来写的是 ``normals * normal_forces + friction_forces``，
与 S4 的口径**三处都不同**（P-80）：

1. PhysX 报的法向力是**带符号**的，S4 取 ``|normal_force|``；
2. PhysX 报的法向有正负约定问题（P-37），S4 用的是**表面的几何外法向**取负；
3. 摩擦作用在采集体还是物体上，取决于刚体对在 PhysX 内部的次序，
   S4 用 ``on_object = −sign`` 翻过来，**整条 episode 判一次**（P-49）。

后果是 traction 的方向可能整体翻转，而指令盒是在离线口径上标定的——
一条完美复现 source 的轨迹会落在盒外，`r_mech` 反过来惩罚正确行为。**不会报错。**

**这条对拍从原始量开始**：读 S3 记录里 PhysX 原样存下的
``contact/{body}/{normal_force, normal_obj, friction_obj, valid}``，
复现 S4 的排序，喂给**在线**的 `ei_reward.contact_force_on_object`，
再与 S4 落盘的 `mech/force_obj` 逐点比余弦与模长比。

判据：力加权的中位余弦 ≥ 0.999 且中位模长比 ∈ [0.98, 1.02]。

**这条检查能不能失败（P-60，实测过，不是推理）**——拿三种真实可能写错的实现喂进去：

======================  ===========  ===========  ==============================
实现                     中位余弦      p01 余弦      结论
======================  ===========  ===========  ==============================
修好的                   **+1.00000**  +1.00000     PASS
法向定向反了              **−0.99843**  −0.99996     一眼就红（后果最坏的那一种）
摩擦作用在谁身上判反了      +0.99843     **+0.91425**  过不了 0.999 的门槛
忘了对法向力取绝对值        +1.00000     +1.00000     **查不出来** ← 见下
======================  ===========  ===========  ==============================

⚠️ **最后一行是这条检查的已知盲区，必须写下来。** S3 记录里落盘的
``contact/{body}/normal_force`` **已经是非负的**（实测 min +0.000 / max +2.927），
所以 ``|fn|`` 与 ``fn`` 在离线数据上逐位相同，取不取绝对值测不出来。
**但在线的 PhysX 传感器报的是带符号的**（E-I 冒烟里实测到 −4.38 N）。
也就是说：**离线记录与在线传感器的符号约定不一样**，而这条对拍只走得到离线那一半。
在线那一半只能靠 `ei_reward.contact_force_on_object` 里的 ``.abs()`` 与
`tools/s6_smoke.py` 的 `diag/peak_normal_force`（它就是因为没取绝对值而报过 0）兜住。

用法（服务器上，纯 numpy，不需要 Isaac）::

    PYTHONPATH=src /isaac-sim/python.sh tools/s6_force_check.py \\
        --pair block=/tmp/s3_probe/block=/tmp/s4_probe/block \\
        --pair drawer=/tmp/s3_drawer_v3=/tmp/s4_drawer \\
        --out /tmp/s6/force_check.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from it.ei_reward import contact_force_on_object, normal_orientation_sign  # noqa: E402
from it.interaction import CONTACT_FORCE_MIN  # noqa: E402
from it.records import load_episode, read_manifest  # noqa: E402

COS_MIN = 0.999
RATIO_LO, RATIO_HI = 0.98, 1.02


def bodies(record) -> list[str]:
    """S4 的 `cat()` 按**排序后的**体名拼接，这里必须一致，否则槽位对不上。"""
    names = {k.split("/")[1] for k in record.arrays if k.startswith("contact/")}
    return sorted(names)


def compare(raw, derived) -> tuple[np.ndarray, np.ndarray]:
    """返回 (逐点余弦, 逐点模长比)，只在两边都有效的点上。"""
    names = bodies(raw)
    cat = lambda f: np.concatenate(  # noqa: E731
        [np.asarray(raw.arrays[f"contact/{b}/{f}"]) for b in names], axis=1)
    fn_raw = cat("normal_force").astype(np.float64)
    nrm = cat("normal_obj").astype(np.float64)
    fri = cat("friction_obj").astype(np.float64)
    valid = cat("valid").astype(bool)

    fn = np.abs(fn_raw)
    active = valid & (fn > CONTACT_FORCE_MIN)
    # 复现 S4 的排序（`it.interaction` 第 8 步：力大的排前面）
    order = np.argsort(-(fn * active), axis=1)
    rows = np.arange(fn.shape[0])[:, None]
    take = lambda a: a[rows, order] if a.ndim == 2 else a[rows, order, :]  # noqa: E731

    fn_s, fri_s, nrm_s = take(fn), take(fri), take(nrm)
    active_s = take(active)
    # 外法向：用 S4 自己定过向的那一份（`engage/dir` 就是 normal_in = −外法向）。
    # 这里要检的是**合成方式与符号**，不是"最近表面点查得对不对"。
    outward = -np.asarray(derived.arrays["engage/dir"], dtype=np.float64)
    live = active_s & (np.linalg.norm(outward, axis=-1) > 0.5)
    if not live.any():
        return np.array([]), np.array([])

    t = lambda a: torch.as_tensor(a, dtype=torch.float64)  # noqa: E731
    sign = normal_orientation_sign(t(nrm_s), t(outward), t(fn_s), t(live))
    # 整条 episode 判一次（P-49：离散选择不能逐帧算）
    weight = (fn_s * live).sum()
    episode_sign = torch.sign(
        (sign * torch.as_tensor((fn_s * live).sum(axis=1))).sum()) if weight > 0 else 1.0
    sign_full = torch.full((fn_s.shape[0],), float(episode_sign), dtype=torch.float64)

    online = contact_force_on_object(t(fn_s), t(fri_s), t(outward), t(live),
                                     sign_full).numpy()
    offline = np.asarray(derived.arrays["mech/force_obj"], dtype=np.float64) * live[..., None]

    a, b = online[live], offline[live]
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    ok = (na > 1e-9) & (nb > 1e-9)
    cos = (a[ok] * b[ok]).sum(-1) / (na[ok] * nb[ok])
    return cos, na[ok] / nb[ok]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pair", dest="pairs", action="append", required=True,
                   metavar="名字=S3数据集=S4数据集")
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    lines = ["在线构造的接触力 vs S4 的 mech/force_obj（从同一批原始量开始）",
             "=" * 88, "",
             "  怎么读：余弦比模长比更要紧——**方向翻转会让余弦变成 −1**，",
             "        而那正是 P-80 那个错误的后果（指令盒在离线口径上标定，",
             "        在线量出反向的 traction，`r_mech` 反过来惩罚正确行为）。",
             f"  判据：力加权中位余弦 ≥ {COS_MIN}，中位模长比 ∈ [{RATIO_LO}, {RATIO_HI}]。",
             "",
             f"      {'数据集':<12}{'点数':>9}{'中位余弦':>11}{'p01 余弦':>11}"
             f"{'中位模长比':>12}{'判定':>7}"]
    failed = 0
    for spec in a.pairs:
        name, s3, s4 = spec.split("=", 2)
        m4 = read_manifest(Path(s4) / "manifest.json")
        cos_all, ratio_all = [], []
        taken = 0
        for entry in m4["episodes"]:
            if not entry.get("success") or taken >= a.limit:
                continue
            p3, p4 = Path(s3) / entry["path"], Path(s4) / entry["path"]
            if not (p3.exists() and p4.exists()):
                continue
            cos, ratio = compare(load_episode(p3), load_episode(p4))
            if len(cos):
                cos_all.append(cos); ratio_all.append(ratio); taken += 1
        if not cos_all:
            lines.append(f"      {name:<12}{'—':>9}   没有可比的点（检查数据集配对）")
            failed += 1
            continue
        cos = np.concatenate(cos_all); ratio = np.concatenate(ratio_all)
        med_c, p01, med_r = (float(np.median(cos)), float(np.percentile(cos, 1)),
                             float(np.median(ratio)))
        ok = med_c >= COS_MIN and RATIO_LO <= med_r <= RATIO_HI
        failed += 0 if ok else 1
        lines.append(f"      {name:<12}{len(cos):>9}{med_c:>11.5f}{p01:>11.5f}"
                     f"{med_r:>12.5f}{'PASS' if ok else 'FAIL':>7}")

    lines += ["", "=" * 88,
              "[PASS] 全部通过" if not failed else f"[FAIL] {failed} 个数据集未通过"]
    text = "\n".join(lines)
    print(text, flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(text + "\n", encoding="utf-8")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
