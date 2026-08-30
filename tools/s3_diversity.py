"""S3 多样性验收：策略分类器（`plan/03` §4）。

判据原文：

> 训练一个简单 source-strategy 分类器时，**原始 source 动作上的准确率
> 应明显高于随机水平**。不能只用"视频看起来有点歪"作为多样性证据。

也就是说：不同策略家族必须在**动作层面**真的不一样，否则"多条不同示教归纳出
同一份 envelope"这件事就没有内容——归纳一堆本来就相同的东西不算归纳。

⚠️ **这个脚本证明的是"动作确实不同"，不是"envelope 相同"。**
后者是 `plan/02` §7 第 4 条的泄漏检查（从 envelope 预测策略身份的准确率
应当**显著低于**从原始动作预测），要等 S4/S5 有了 envelope 才能做。
两个数将来要并排报，一高一低才说明"表示抹掉了策略特异性而保留了功能"。

特征只取 ``source/*``——那正是被隔离在模型输入之外的审计字段。
用它们做分类是合规的：这里要的就是"从 source 动作能不能认出策略"。

只用 numpy，不引入 sklearn。多分类逻辑回归用梯度下降训，够用且没有依赖。

用法::

    PYTHONPATH=src /isaac-sim/python.sh tools/s3_diversity.py /tmp/s3_drawer_v3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from it.records import load_episode, read_manifest  # noqa: E402


def features(rec) -> np.ndarray:
    """一条 episode 的动作特征。只取 source 侧的指令增量与本体速度。

    刻意只用**低阶统计量**（均值/标准差/分位数），不喂时间序列——
    判据要的是"动作分布明显不同"，用一个能记住轨迹的强模型反而说明不了问题。
    """
    parts: list[np.ndarray] = []
    for key in sorted(rec.arrays):
        if not (key.endswith("/cmd_delta") or key.endswith("/root_velocity")):
            continue
        a = np.asarray(rec.arrays[key], dtype=np.float64)
        parts += [a.mean(0), a.std(0), np.abs(a).mean(0),
                  np.quantile(a, 0.9, axis=0), np.quantile(a, 0.1, axis=0)]
    return np.concatenate(parts) if parts else np.zeros(1)


def softmax_fit(x, y, n_cls, epochs=400, lr=0.5, l2=1e-3, seed=0):
    """多分类逻辑回归，全批量梯度下降。"""
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.01, size=(x.shape[1], n_cls))
    b = np.zeros(n_cls)
    onehot = np.eye(n_cls)[y]
    for _ in range(epochs):
        z = x @ w + b
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        g = (p - onehot) / len(x)
        w -= lr * (x.T @ g + l2 * w)
        b -= lr * g.sum(0)
    return w, b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--success-only", action="store_true", default=True)
    a = ap.parse_args()

    root = Path(a.root)
    man = read_manifest(root / "manifest.json")
    eps = [e for e in man["episodes"] if e["success"] or not a.success_only]
    fams = sorted({e["strategy_family"] for e in eps})
    if len(fams) < 2:
        print(f"只有 {len(fams)} 个策略家族，多样性判据不适用")
        return 0

    print(f"数据集 {man['dataset_name']}：{len(eps)} 条成功 episode，"
          f"{len(fams)} 个策略家族")
    xs, ys = [], []
    for e in eps:
        rec = load_episode(root / e["path"])
        xs.append(features(rec))
        ys.append(fams.index(e["strategy_family"]))
    x = np.stack(xs)
    y = np.array(ys)

    # **按 episode 划分**（P-10），且标准化只用训练集的统计量
    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(len(x))
    n_te = max(1, int(len(x) * a.test_frac))
    te, tr = idx[:n_te], idx[n_te:]
    mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-9
    xtr, xte = (x[tr] - mu) / sd, (x[te] - mu) / sd

    w, b = softmax_fit(xtr, y[tr], len(fams), seed=a.seed)
    pred = (xte @ w + b).argmax(1)
    acc = float((pred == y[te]).mean())
    chance = 1.0 / len(fams)
    # 按类频次的多数类基线——类别不均衡时它比 1/K 更难打败
    major = float(max((y[tr] == c).mean() for c in range(len(fams))))

    print(f"\n特征维度 {x.shape[1]}，训练 {len(tr)} / 测试 {len(te)}")
    print(f"准确率 {acc:.3f}   随机 {chance:.3f}   多数类 {major:.3f}")
    print(f"相对随机提升 {acc / chance:.1f}×")
    print("\n混淆矩阵（行=真值，列=预测）")
    cm = np.zeros((len(fams), len(fams)), dtype=int)
    for t, p in zip(y[te], pred):
        cm[t, p] += 1
    head = "".join(f"{f[:9]:>10}" for f in fams)
    print(f"{'':<18}{head}")
    for i, f in enumerate(fams):
        print(f"{f:<18}" + "".join(f"{v:>10}" for v in cm[i]))

    ok = acc > max(chance * 2.0, major + 0.10)
    print(f"\n[{'PASS' if ok else 'FAIL'}] 原始 source 动作上的准确率显著高于随机"
          f"（`plan/03` §4）")
    print("\n注：本判据只说明**动作层面确实不同**。`plan/02` §7 第 4 条要的是"
          "\n    从 envelope 预测策略身份**显著更难**，要等 S4/S5 有 envelope 才能做，"
          "\n    届时两个数并排报，一高一低才说明表示抹掉了策略特异性。")
    (root / "diversity.txt").write_text(
        f"accuracy={acc:.4f} chance={chance:.4f} majority={major:.4f} "
        f"families={len(fams)} n={len(x)}\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
