"""旋钮 source 的接触部位与受力方向核对（`plan/06` §7 的人工检查）。

录像能看出板在推销钉，**看不出力打在销钉的哪一侧、方向偏了多少**。而旋钮
任务的全部力学就在这两件事上：销钉靠**法向力**直接给力矩（与 μ 无关），
方向一偏，力就变成顶向圆心的无用功（P-42 实测偏 42° 时三分之一的力白费）。

本脚本在**物体局部系**里统计，因此圆盘转到哪个角度都不影响结论：

* 力打在哪：销钉柱面 / 圆盘顶面 / 轮缘 —— 按法向力加权；
* 打在销钉的哪一侧：接触点相对销钉轴心的方位角，转换成"推进方向的前/后"；
* 方向对不对：接触法向与该处**圆盘切向**的夹角，0° = 力全部转成力矩；
* 打在销钉的什么高度：盘面之上 0~48 mm。

用法::

    PYTHONPATH=src /isaac-sim/python.sh tools/s3_knob_contact.py /tmp/s3_knob
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from it import build_assets as B  # noqa: E402
from it.records import load_episode, read_manifest  # noqa: E402

_K = B.KnobCfg()
PIN_XY = np.array([_K.pin_offset, 0.0], dtype=np.float64)
DISC_TOP = _K.disc_thickness / 2


def _hist(vals, w, edges):
    h, _ = np.histogram(vals, bins=edges, weights=w)
    return h / max(h.sum(), 1e-9)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--max-episodes", type=int, default=400)
    a = ap.parse_args()

    root = Path(a.root)
    man = read_manifest(root / "manifest.json")
    eps = [e for e in man["episodes"] if e["success"]]
    fams = sorted({e["strategy_family"] for e in man["episodes"]})
    print(f"数据集 {man['dataset_name']}：{len(eps)} 条成功 episode")

    rows = {}
    for fam in fams:
        pool = [e for e in man["episodes"] if e["strategy_family"] == fam
                and (e["success"] or fam == "rim_only")][: a.max_episodes // len(fams)]
        acc = dict(w=0.0, pin=0.0, face=0.0, rim=0.0, behind=0.0,
                   ang=[], wang=[], hgt=[], whgt=[])
        for meta in pool:
            rec = load_episode(root / meta["path"])
            phase = np.asarray(rec.arrays["phase"])
            keep = phase == 2
            for k in (0, 1):
                p = np.asarray(rec.arrays[f"contact/plate{k}/pos_obj"])[keep]
                nrm = np.asarray(rec.arrays[f"contact/plate{k}/normal_obj"])[keep]
                f = np.abs(np.asarray(rec.arrays[f"contact/plate{k}/normal_force"])[keep])
                v = np.asarray(rec.arrays[f"contact/plate{k}/valid"])[keep]
                w = (f * v).ravel()
                if w.sum() <= 0:
                    continue
                pp = p.reshape(-1, 3)
                nn = nrm.reshape(-1, 3)
                d_pin = np.linalg.norm(pp[:, :2] - PIN_XY, axis=-1)
                r_ax = np.linalg.norm(pp[:, :2], axis=-1)
                on_pin = (d_pin < _K.pin_radius + 0.005) & (pp[:, 2] > DISC_TOP - 0.002)
                on_rim = r_ax > _K.disc_radius - 0.010
                on_face = (~on_pin) & (~on_rim) & (pp[:, 2] > DISC_TOP - 0.004)
                acc["w"] += w.sum()
                acc["pin"] += w[on_pin].sum()
                acc["rim"] += w[on_rim].sum()
                acc["face"] += w[on_face].sum()
                if on_pin.sum() == 0:
                    continue
                q = pp[on_pin]
                nq = nn[on_pin]
                wq = w[on_pin]
                # 该处圆盘切向（物体系里销钉在 +X 轴上，切向即 +Y）
                tang = np.zeros_like(nq)
                tang[:, 1] = 1.0
                cos = np.abs((nq * tang).sum(-1)) / (
                    np.linalg.norm(nq, axis=-1) + 1e-9)
                acc["ang"].append(np.degrees(np.arccos(np.clip(cos, 0, 1))))
                acc["wang"].append(wq)
                # 接触在销钉的"后方"（−Y 侧）= 推着它往 +θ 走
                side = q[:, 1] - PIN_XY[1]
                acc["behind"] += wq[side < 0].sum()
                acc["hgt"].append((q[:, 2] - DISC_TOP) * 1000.0)
                acc["whgt"].append(wq)
        rows[fam] = acc

    print("\n按法向力加权的接触部位（操作阶段）")
    print(f"{'家族':<18}{'销钉%':>8}{'盘顶面%':>9}{'轮缘%':>8}"
          f"{'销钉后侧%':>11}{'法向偏切向°':>13}{'销钉高度mm':>12}")
    bad = []
    for fam, acc in rows.items():
        w = max(acc["w"], 1e-9)
        pin, face, rim = 100 * acc["pin"] / w, 100 * acc["face"] / w, 100 * acc["rim"] / w
        if acc["ang"]:
            ang = np.concatenate(acc["ang"])
            wang = np.concatenate(acc["wang"])
            a_mean = float((ang * wang).sum() / wang.sum())
            hgt = np.concatenate(acc["hgt"])
            whg = np.concatenate(acc["whgt"])
            h_mean = float((hgt * whg).sum() / whg.sum())
            behind = 100 * acc["behind"] / max(acc["pin"], 1e-9)
        else:
            a_mean = h_mean = behind = float("nan")
        print(f"{fam:<18}{pin:>8.1f}{face:>9.1f}{rim:>8.1f}"
              f"{behind:>11.1f}{a_mean:>13.1f}{h_mean:>12.1f}")
        if fam != "rim_only":
            if pin < 95.0:
                bad.append(f"{fam}：只有 {pin:.1f}% 的力打在销钉上")
            if a_mean > 25.0:
                bad.append(f"{fam}：接触法向偏离切向 {a_mean:.1f}°，力矩有效分量不足")
            if behind < 90.0:
                bad.append(f"{fam}：只有 {behind:.1f}% 的力打在销钉的推进后侧")
        elif rim < 90.0:
            bad.append(f"rim_only：只有 {rim:.1f}% 的力打在轮缘上，对照不成立")

    print("\n说明：")
    print("  · 销钉%   —— 力打在销钉柱面上的比例（其余是盘顶面或轮缘）")
    print("  · 销钉后侧% —— 打在销钉**推进方向后方**的比例；推着走就该全在后方")
    print("  · 法向偏切向° —— 0° 表示力全部转成绕轴力矩，越大越多力顶向圆心（P-42）")
    print("  · 销钉高度mm —— 盘面之上的高度，销钉长 48 mm")
    for b in bad:
        print(f"  [FAIL] {b}")
    print(f"\n[{'PASS' if not bad else 'FAIL'}] 接触部位与受力方向核对")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
