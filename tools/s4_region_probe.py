"""泄漏检查第 9 条：region 到底能不能从 effect + 物体点云推出来。

`plan/02` §7 第 9 条（D-14 的配套）：

> 训练一个只看 ``effect + 物体点云`` 的探针网络去预测 oracle region。
> 若探针准确率接近直接给 region 的上限，说明该任务上 region 在信息论上是
> **冗余的**，该任务不能用于检验 region 的必要性，**应如实报告而非调参掩盖**。

**为什么这条不设通过门槛**：它不是"做得好不好"，是"这个任务适不适合检验
region"。Gate G 明确写了阴性结果可接受。调参让它不可推导，就是 P-11。

### 三个数并排比，才读得出结论

同一个任务里物体点云对所有 episode 是**同一份**（几何变体除外），所以
"effect + 点云 → region" 实际上就是"effect → region"。三个预测器：

| 预测器 | 看得到什么 | 意义 |
|---|---|---|
| `mean` | 什么都不看，永远输出全数据集的平均热图 | 下界：不用 effect 能做到多好 |
| `probe` | **只看 effect**（当前值 + 未来窗口 + 物体点云） | 待检验的那一个 |
| `family` | 作弊上界：知道这条示教属于哪个策略家族，输出该家族的平均热图 | 上界：热图本身有多可重复 |

- `probe ≈ mean` → effect 推不出 region，**region 有独立信息**，该任务可以用来
  检验 region 的必要性；
- `probe ≈ family` → region 基本可由 effect 推出，该任务**不适合**检验 region，
  如实报告。

指标用**热图残差的可解释方差**，不用余弦相似度。原因是实测发现的：同一个任务里
所有 episode 的热图都压在同一片区域上（抽屉永远是把手背面、旋钮永远是销钉），
余弦相似度因此全在 0.83~0.98 之间**饱和**，三个预测器差不到 0.01，读不出结论。
把全体均值减掉之后再看解释了多少方差，`mean` 预测器按定义是 0，
`family` 与 `probe` 的数就直接可比了。

用法::

    PYTHONPATH=src /isaac-sim/python.sh tools/s4_region_probe.py /tmp/s4_drawer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from it.interaction import region_heatmap  # noqa: E402
from it.records import load_episode, read_manifest  # noqa: E402
from it.surfaces import surface_for  # noqa: E402


def coarse_heatmap(rec, level: int) -> np.ndarray:
    """把 region 热图池化到 `LEVELS` 里的某一档，再归一化成分布。

    池化用的是表面采样的 `parent` 映射，低分辨率是高分辨率的**前缀**（D-50），
    所以这一步是无损的合并，不是重新最近邻。
    """
    s = surface_for(rec.meta["surface"]["object"], rec.meta["surface"]["geom_tag"])
    heat = region_heatmap(rec, s.n_points, phase=2)
    par = s.parent[level]
    out = np.zeros(level)
    np.add.at(out, par, heat)
    n = out.sum()
    return out / n if n > 0 else out


def effect_features(rec) -> np.ndarray:
    """只用 effect 通道。**不碰 region / mode / mechanics**，否则这条检查没有意义。"""
    arrays = rec.model_arrays()
    cur = np.asarray(arrays["effect/current"], dtype=np.float64)
    fut = np.asarray(arrays["effect/future"], dtype=np.float64)
    ok = np.asarray(arrays["effect/future_valid"])
    manip = np.asarray(arrays["phase"]) == 2
    if not manip.any():
        manip = np.ones(len(cur), dtype=bool)
    f = [cur[manip].mean(0), cur[manip].std(0), cur[-1] - cur[0],
         cur[manip].min(0), cur[manip].max(0)]
    fw = fut[manip]
    ow = ok[manip]
    f += [(fw * ow[..., None]).mean(axis=0).ravel(),
          np.abs(fw * ow[..., None]).max(axis=0).ravel()]
    return np.nan_to_num(np.concatenate([np.atleast_1d(v) for v in f]))


def ridge(x: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    x1 = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    a = x1.T @ x1 + lam * np.eye(x1.shape[1])
    return np.linalg.solve(a, x1.T @ y)


def cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    return (a * b).sum(axis=1) / np.maximum(na * nb, 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--level", type=int, default=256, help="热图池化到哪一档")
    ap.add_argument("--limit", type=int, default=600, help="最多用几条成功 episode")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    root = Path(a.root)
    man = read_manifest(root / "manifest.json")
    ok = [e for e in man["episodes"] if e["success"]]
    # **先打乱再截断。** manifest 按 episode_id 排序，直接取前 N 条会只覆盖到
    # 一两个策略家族——实测旋钮那份 300 条全是同一个家族，family 这条上界
    # 因此恒等于 mean，读不出任何东西。
    ok = [ok[i] for i in np.random.default_rng(a.seed).permutation(len(ok))[: a.limit]]
    if len(ok) < 40:
        print(f"成功 episode 只有 {len(ok)} 条，样本太少，不做这条检查")
        return 2

    xs, ys, fams = [], [], []
    for e in ok:
        rec = load_episode(root / e["path"])
        h = coarse_heatmap(rec, a.level)
        if h.sum() <= 0:
            continue
        xs.append(effect_features(rec))
        ys.append(h)
        fams.append(e["strategy_family"])
    x = np.array(xs)
    y = np.array(ys)
    fam = np.array(fams)
    mu, sd = x.mean(0), x.std(0) + 1e-9
    x = (x - mu) / sd

    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(x))
    cut = int(len(x) * 0.7)
    tr, te = order[:cut], order[cut:]

    base = y[tr].mean(0)
    resid = y[te] - base                     # 要解释的就是这一部分
    denom = float((resid ** 2).sum())

    def ev(pred: np.ndarray) -> float:
        """可解释方差：1 - 残差平方和 / 总平方和。mean 预测器按定义是 0。"""
        return float(1.0 - ((y[te] - pred) ** 2).sum() / max(denom, 1e-12))

    w = ridge(x[tr], y[tr])
    probe_pred = np.maximum(
        np.concatenate([x[te], np.ones((len(te), 1))], axis=1) @ w, 0.0)
    fam_mean = {f: y[tr][fam[tr] == f].mean(0) for f in np.unique(fam[tr])}
    fam_pred = np.array([fam_mean.get(f, base) for f in fam[te]])
    mean_pred = np.repeat(base[None, :], len(te), axis=0)

    ev_probe, ev_fam = ev(probe_pred), ev(fam_pred)
    cos_mean = float(cos(mean_pred, y[te]).mean())
    cos_probe = float(cos(probe_pred, y[te]).mean())
    cos_fam = float(cos(fam_pred, y[te]).mean())

    print(f"===== region 可推导性探针：{root.name} =====")
    print(f"{len(x)} 条成功 episode，热图池化到 {a.level} 点，"
          f"训练 {len(tr)} / 留出 {len(te)}，{len(np.unique(fam))} 个策略家族")
    print()
    print(f"{'预测器':<32}{'可解释方差':>12}{'余弦相似度':>12}")
    print(f"{'mean（什么都不看）':<32}{0.0:>12.3f}{cos_mean:>12.3f}")
    print(f"{'probe（只看 effect + 点云）':<32}{ev_probe:>12.3f}{cos_probe:>12.3f}")
    print(f"{'family（作弊：知道策略家族）':<32}{ev_fam:>12.3f}{cos_fam:>12.3f}")
    print()
    print("余弦相似度那一列会**饱和**（同一任务的热图都压在同一片区域上），"
          "读结论看可解释方差那一列。")
    print()

    span = ev_fam - 0.0
    if span > 1e-6:
        print(f"probe 解释掉的方差占 family 那个作弊上界的 "
              f"{100 * ev_probe / span:.0f}%——上界本身多大也要一起看，"
              f"上界小就说明这个任务的热图本来就没什么可解释的差异")
        print()
    if ev_probe < 0.10:
        verdict = ("effect **推不出** region（可解释方差 < 0.10）—— region 在这个"
                   "任务上有独立信息，可以用来检验 region 的必要性")
    elif ev_probe > 0.50:
        verdict = ("region 相当程度上可由 effect 推出 —— **该任务检验 region 必要性的"
                   "能力被削弱**，按 `plan/02` §7 第 9 条与 Gate G 如实报告，不得调参掩盖")
    else:
        verdict = ("effect 只解释了一部分（0.10~0.50），需要连同 `plan/05` 实验一的"
                   "结果一起解读，不能单独下结论")
    print(f"结论：{verdict}")
    if ev_fam < 0.05:
        print("附注：family 这个作弊上界也接近 0，说明热图在**家族之间**本来就差别很小，")
        print("      这个任务上 region 的形态几乎只有一种——这件事本身要写进报告。")
    print()
    print("说明：这条检查**不设通过门槛**。它回答的是「这个任务适不适合检验 region」，")
    print("      不是「我们做得好不好」。调参让它变成不可推导，就是 P-11。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
