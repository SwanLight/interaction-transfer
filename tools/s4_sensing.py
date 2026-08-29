"""S4.6 传感可观测性：**装置能测到的那点东西，够不够重建交互规格**。

这是本项目里唯一直接检验 idea 的**硬件前提**的实验。核心主张是"不采人手动作，
只采各接触面的位姿 + 面上的触觉"；此前整轮实验没有一处碰这句话——`plan/00` §4
把它整个推给了 Phase II，于是本轮验证的只剩"物体中心表示 + 任务无关执行器"，
而那部分与 CHORD 的 object-centric contact wrench、contact-centric 抓取迁移
高度重叠。**属于本方案自己的那一条，必须自己给证据。**

做法：拿已经验收过的 S4 Oracle Record 当真值，**模拟装置的观测模型**再重建一遍：

```text
装置测得                              重建出
─────────────────────────────────    ─────────────────────
面相对物体的 6D 位姿（带噪）      ┐
面上的触觉阵列（pitch、模态）     ├─>  region 热图 / contact mode / mechanics
物体位姿与 mesh（带噪）           ┘
```

然后与 oracle 逐项比。输出的是三条降级曲线，直接对应 `plan/07` §1 里
被标成"Phase II 待定"的三个硬件指标：**taxel pitch、触觉模态、面位姿精度**。

三件必须说清楚的事：

1. **这不是"给表示加噪声"**。加噪只回答"表示对误差敏不敏感"；这里换的是
   **信息来源**——只允许用装置能测的量重新算一遍 region / mode / mechanics。
   比如切向力：法向-only 的触觉根本给不出它，那一档的 mechanics 就只能是残缺的，
   而不是"带噪的"。
2. **相对位姿，不是绝对位姿**（`plan/07` §3）。整套表示是物体系的，所以装置
   需要的是"面相对物体"的位姿，不是世界系绝对位姿——这是把一个高精度无链路
   6D 跟踪问题降级成"粗跟踪 + 触觉配准"的关键，也是本方案与 ART-Glove 那类
   简化外骨骼的真正差异。
3. **mode 用位姿差分算**，与 oracle 里那一路是同一个式子（`interaction._pose_slip`）——
   oracle 用的是仿真给的真值位姿，这里用的是带噪的观测位姿。两者之差就是
   "面位姿精度要到多少才判得准 stick/slide"，那正是要冻结的指标。

用法::

    PYTHONPATH=src /isaac-sim/python.sh tools/s4_sensing.py /tmp/s4_drawer \\
        /tmp/s3_drawer_v3 --episodes 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from it.interaction import (  # noqa: E402
    CONTACT_FORCE_MIN,
    SLIP_SPEED_MIN,
    _object_pose,
    _quat_to_rot,
    contact_bodies,
    region_heatmap,
    spec_for,
)
from it.records import load_episode, read_manifest  # noqa: E402
from it.surfaces import assign_to_surface, surface_for  # noqa: E402

#: 触觉模态。装置能给的东西按这个顺序递增。
MODALITIES = ("binary", "normal", "shear")

#: 面位姿噪声档位（mm / deg）。与 `plan/05` 实验零 §1.2 的物体位姿噪声同量级，
#: 但这里加的是**面相对物体**的位姿——`plan/07` §3 说那才是真瓶颈。
POSE_NOISE = ((0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (2.0, 2.0), (5.0, 5.0))

#: 噪声的**时间结构**，比幅值更要命。
#: - ``white``：每帧独立。mode 由位姿**差分**得到，白噪声会被差分放大成
#:   ``√2·σ/dt``——0.5 mm 的白噪声就等于 17 mm/s 的假滑移，直接压过 5 mm/s 的阈值。
#: - ``bias``：整条 episode 一个固定偏置 + 一成的白噪声抖动。基于配准的估计器
#:   （`plan/07` §3 的"粗跟踪 + 触觉配准"）给的是这一类：**偏置大而帧间抖动小**。
#: 两种都要报。只报 white 会把结论写成"面位姿必须到 0.05 mm"，那是噪声模型的
#: 结论，不是物理的结论。
NOISE_KINDS = ("bias", "white")

#: taxel pitch（mm）。1.25 mm 是真实指尖 16×16 覆盖 20×20 mm 的水平；
#: 8 mm 是"稀疏节点"路线。
PITCHES = (1.0, 2.0, 4.0, 8.0)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=-1)


def _noise_quat(rng: np.random.Generator, n: int, deg: float) -> np.ndarray:
    """随机小角度扰动四元数。"""
    if deg <= 0:
        q = np.zeros((n, 4))
        q[:, 0] = 1.0
        return q
    axis = rng.normal(size=(n, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    half = np.deg2rad(rng.normal(0.0, deg, size=n)) / 2.0
    return np.concatenate([np.cos(half)[:, None], axis * np.sin(half)[:, None]], axis=1)


def _body_pose(arrays: dict, body: str) -> np.ndarray | None:
    for key in (f"source/{body}/root_pose", f"source/{body}_pose"):
        if key in arrays:
            return np.asarray(arrays[key], dtype=np.float64)
    return None


class Device:
    """装置的观测模型：面相对物体的位姿 + 面上的触觉阵列。

    **只暴露装置测得到的东西。** 真值里的接触点世界坐标、逐点摩擦矢量、
    PhysX 报的法向，一概不经过这里。
    """

    def __init__(self, src_arrays: dict, meta: dict, rng: np.random.Generator,
                 pos_mm: float, rot_deg: float, pitch_mm: float, modality: str,
                 noise_kind: str = "bias"):
        self.bodies = contact_bodies(src_arrays)
        self.a = src_arrays
        self.pitch = pitch_mm * 1e-3
        self.modality = modality
        op = _object_pose(src_arrays, meta)
        if op is None:
            raise ValueError("拿不到被操作物体的位姿")
        self.obj_pos, self.obj_rot = op[0], op[1]
        n = len(self.obj_pos)

        # 面相对物体的位姿 = 真值 ∘ 噪声。噪声加在**相对位姿**上（`plan/07` §3）
        self.rel_pos, self.rel_rot = {}, {}
        for b in self.bodies:
            pose = _body_pose(src_arrays, b)
            if pose is None:
                continue
            rb = _quat_to_rot(pose[:, 3:7])
            # 世界 -> 物体系的相对位姿
            rel_p = np.einsum("tij,tj->ti", self.obj_rot.transpose(0, 2, 1),
                              pose[:, :3] - self.obj_pos)
            rel_r = np.einsum("tij,tjk->tik", self.obj_rot.transpose(0, 2, 1), rb)
            if noise_kind == "bias":
                # 固定偏置 + 一成抖动：配准类估计器的形状
                dp = (rng.normal(0.0, pos_mm * 1e-3, size=(1, 3))
                      + rng.normal(0.0, 0.1 * pos_mm * 1e-3, size=(n, 3)))
                q = _quat_mul(_noise_quat(rng, 1, rot_deg),
                              _noise_quat(rng, n, 0.1 * rot_deg))
            else:
                dp = rng.normal(0.0, pos_mm * 1e-3, size=(n, 3))
                q = _noise_quat(rng, n, rot_deg)
            dp = dp if pos_mm > 0 else 0.0
            dq = _quat_to_rot(q)
            self.rel_pos[b] = rel_p + dp
            self.rel_rot[b] = np.einsum("tij,tjk->tik", dq, rel_r)

    def taxels(self, body: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """该面上的触觉读数。

        返回 ``(物体系接触点, 法向压力, 切向读数)``。接触点由**面上的 taxel 位置**
        经过面位姿映射到物体系——这正是装置的信息通路：它不知道接触在物体上的
        绝对位置，只知道"我这块面的哪几个 taxel 被压了"和"这块面此刻贴在物体的哪"。
        """
        fn = np.abs(np.asarray(self.a[f"contact/{body}/normal_force"], dtype=np.float64))
        valid = np.asarray(self.a[f"contact/{body}/valid"])
        act = valid & (fn > CONTACT_FORCE_MIN)
        pos_obj = np.asarray(self.a[f"contact/{body}/pos_obj"], dtype=np.float64)

        # 真值接触点 -> 面局部系（真值位姿），再按 pitch 量化 = taxel 命中
        rel_p, rel_r = self.rel_pos[body], self.rel_rot[body]
        pose = _body_pose(self.a, body)
        true_r = _quat_to_rot(pose[:, 3:7])
        true_rel_p = np.einsum("tij,tj->ti", self.obj_rot.transpose(0, 2, 1),
                               pose[:, :3] - self.obj_pos)
        true_rel_r = np.einsum("tij,tjk->tik", self.obj_rot.transpose(0, 2, 1), true_r)
        local = np.einsum("tij,tkj->tki", true_rel_r.transpose(0, 2, 1),
                          pos_obj - true_rel_p[:, None, :])
        local = np.round(local / self.pitch) * self.pitch          # taxel 量化

        # 面局部系 -> 物体系（**带噪的**观测位姿）
        obs = rel_p[:, None, :] + np.einsum("tij,tkj->tki", rel_r, local)

        if self.modality == "binary":
            press = act.astype(np.float64)          # 只知道"这个 taxel 被压了"
        else:
            press = fn * act
        if self.modality == "shear":
            shear = np.asarray(self.a[f"contact/{body}/friction_obj"], dtype=np.float64)
        else:
            shear = np.zeros_like(pos_obj)          # 法向-only 的皮肤给不出切向
        return obs, press * act, shear * act[..., None]

    def slip(self, body: str, dt: float) -> np.ndarray:
        """由**观测到的**相对位姿差分算滑移速度——与 oracle 同一个式子。"""
        rp, rr = self.rel_pos[body], self.rel_rot[body]
        pos_obj = np.asarray(self.a[f"contact/{body}/pos_obj"], dtype=np.float64)
        loc = np.einsum("tij,tkj->tki", rr.transpose(0, 2, 1), pos_obj - rp[:, None, :])
        nxt = rp[2:, None, :] + np.einsum("tij,tkj->tki", rr[2:], loc[1:-1])
        prv = rp[:-2, None, :] + np.einsum("tij,tkj->tki", rr[:-2], loc[1:-1])
        out = np.zeros(pos_obj.shape[:2])
        out[1:-1] = np.linalg.norm(nxt - prv, axis=-1) / (2.0 * dt)
        out[0], out[-1] = out[1], out[-2]
        return out


def reconstruct(dev: Device, surface, level: int, dt: float,
                truth_pos: np.ndarray, truth_w: np.ndarray) -> dict:
    """从装置读数重建 region 热图、逐帧 stick 比例、法向/切向合力。

    ⚠️ **mode 要逐帧比，不能把两边的接触点各自摊平了按下标对。** 记录里的槽位
    是按力排过序的，装置这边是按接触体拼的，下标根本不对应——第一版就是这么
    比的，零噪声下都只有 0.806 的"一致率"，那 0.194 全是错位。
    """
    heat = np.zeros(surface.n_points)
    n_frames = truth_w.shape[0]
    stick_num = np.zeros(n_frames)
    stick_den = np.zeros(n_frames)
    disp = []
    f_normal = np.zeros(n_frames)
    f_shear = np.zeros((n_frames, 3))
    obs_all, w_all = [], []
    for b in dev.bodies:
        if b not in dev.rel_pos:
            continue
        obs, press, shear = dev.taxels(b)
        slip = dev.slip(b, dt)
        live = press > 0
        if not live.any():
            continue
        idx, ok, _ = assign_to_surface(obs[live], surface, max_dist=8e-3)
        w = press[live]
        np.add.at(heat, idx[ok], w[ok])
        stick_num += (press * (slip <= SLIP_SPEED_MIN)).sum(axis=1)
        stick_den += press.sum(axis=1)
        f_normal += press.sum(axis=1)
        f_shear += shear.sum(axis=1)
        obs_all.append((obs * press[..., None]).sum(axis=1))
        w_all.append(press.sum(axis=1))
    if not w_all:
        return {}
    par = surface.parent[level]
    coarse = np.zeros(level)
    np.add.at(coarse, par, heat)
    wsum = np.sum(w_all, axis=0)
    cen_obs = np.sum(obs_all, axis=0) / np.maximum(wsum, 1e-9)[:, None]
    live_f = (wsum > 0) & (truth_w.sum(axis=1) > 0)
    if live_f.any():
        cen_true = ((truth_pos * truth_w[..., None]).sum(axis=1)
                    / np.maximum(truth_w.sum(axis=1), 1e-9)[:, None])
        disp = np.linalg.norm(cen_obs[live_f] - cen_true[live_f], axis=1)
    return {"heat": coarse,
            "stick": np.divide(stick_num, np.maximum(stick_den, 1e-9)),
            "live": stick_den > 0,
            "disp_mm": float(np.mean(disp) * 1000.0) if len(disp) else float("nan"),
            "f_normal": f_normal, "f_shear": f_shear}


def oracle_side(rec, surface, level: int) -> dict:
    par = surface.parent[level]
    heat = np.zeros(level)
    np.add.at(heat, par, region_heatmap(rec, surface.n_points))
    act = np.asarray(rec.arrays["region/valid"])
    w = np.asarray(rec.arrays["region/weight"], dtype=np.float64) * act
    mode = np.asarray(rec.arrays["mode/label"])
    den = w.sum(axis=1)
    return {"heat": heat,
            "stick": np.divide((w * (mode == 1)).sum(axis=1), np.maximum(den, 1e-9)),
            "live": den > 0,
            "w": w,
            "pos": np.asarray(rec.arrays["region/pos_obj"], dtype=np.float64),
            "f_normal": den}


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / max(na * nb, 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="S4 记录目录")
    ap.add_argument("src", help="对应的 S3 数据集目录")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--level", type=int, default=1024,
                    help="热图池化档位。256 太粗（格子比噪声还大），默认 1024")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    root, src_root = Path(a.root), Path(a.src)
    man = read_manifest(root / "manifest.json")
    src_man = read_manifest(src_root / "manifest.json")
    src_by_id = {e["episode_id"]: e for e in src_man["episodes"]}
    rng = np.random.default_rng(a.seed)
    ok = [e for e in man["episodes"] if e["success"]]
    ok = [ok[i] for i in rng.permutation(len(ok))[: a.episodes]]

    print(f"===== S4.6 传感可观测性：{root.name} =====")
    print(f"{len(ok)} 条成功 episode，热图池化到 {a.level} 点\n")

    def sweep(label: str, configs: list, make) -> None:
        print(f"{label:<24}{'region 余弦':>12}{'接触点偏移mm':>14}"
              f"{'stick比例一致':>14}{'法向力误差':>12}{'切向可得':>10}")
        for cfg in configs:
            rows_r, rows_m, rows_f, rows_d, has_shear = [], [], [], [], []
            for e in ok:
                rec = load_episode(root / e["path"])
                src = load_episode(src_root / src_by_id[e["episode_id"]]["path"])
                surf = surface_for(rec.meta["surface"]["object"],
                                   rec.meta["surface"]["geom_tag"])
                dt = 1.0 / float(rec.meta.get("control_hz", 50.0))
                try:
                    dev = make(src, rec.meta, cfg)
                except (ValueError, KeyError):
                    continue
                truth = oracle_side(rec, surf, a.level)
                got = reconstruct(dev, surf, a.level, dt, truth["pos"], truth["w"])
                if not got or truth["heat"].sum() <= 0:
                    continue
                rows_r.append(cos(got["heat"], truth["heat"]))
                live = got["live"] & truth["live"]
                if live.any():
                    # 逐帧的 stick 比例之差——两边都是"这一帧有多少力在贴着"
                    rows_m.append(1.0 - float(np.abs(
                        got["stick"][live] - truth["stick"][live]).mean()))
                    fn_t = truth["f_normal"][live]
                    rows_f.append(float(np.abs(got["f_normal"][live] - fn_t).sum()
                                        / max(fn_t.sum(), 1e-9)))
                rows_d.append(got["disp_mm"])
                has_shear.append(float(np.linalg.norm(got["f_shear"]) > 0))
            if not rows_r:
                continue
            print(f"{str(cfg):<24}{np.mean(rows_r):>12.3f}"
                  f"{np.nanmean(rows_d):>14.2f}{np.mean(rows_m):>14.3f}"
                  f"{np.mean(rows_f):>12.3f}{np.mean(has_shear):>10.2f}")
        print()

    for kind in NOISE_KINDS:
        sweep(f"面位姿噪声 {kind} (mm,deg)", list(POSE_NOISE),
              lambda src, meta, c, k=kind: Device(
                  src.arrays, meta, np.random.default_rng(a.seed),
                  c[0], c[1], 2.0, "shear", k))
    sweep("taxel pitch (mm)", list(PITCHES),
          lambda src, meta, c: Device(src.arrays, meta, np.random.default_rng(a.seed),
                                      0.0, 0.0, c, "shear", "bias"))
    sweep("触觉模态", list(MODALITIES),
          lambda src, meta, c: Device(src.arrays, meta, np.random.default_rng(a.seed),
                                      1.0, 1.0, 2.0, c, "bias"))

    print("怎么读：")
    print("  region 余弦 —— 重建热图与 oracle 的相似度。它对面位姿噪声最敏感，")
    print("               因为接触点在物体上的位置**完全由面位姿决定**。")
    print("  mode 一致   —— stick/slide 标签与 oracle 的加权一致率。它对面位姿的")
    print("               **时间差分**敏感，所以比 region 掉得更快——这条决定")
    print("               `plan/07` §1 的面位姿精度指标。")
    print("  切向可得    —— 法向-only 的皮肤给不出切向力，那一档 mechanics 是")
    print("               **残缺**而不是带噪的；C4 的力方向与范围直接依赖它。")
    print("\n⚠️ 这里只换信息来源，不改任何判据；oracle 那一侧是已验收的 S4 记录。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
