"""`it.surfaces` 的本地单元测试。

**故意不依赖 Isaac Sim / torch**，系统 python 就能跑，改完立刻验::

    PYTHONPATH=src python3 tools/test_surfaces.py

这里测的都是"错了以后 region 字段会静默地不对"的性质：

- 法向**朝外**——法向反了，`plan/02` §3.3 的 engage 方向整体反向，
  而热图看起来完全正常（P-37 已经在接触法向上栽过一次）；
- 采样**确定性**——同一个物体两次采出的点必须逐位相同，否则不同批次
  提取出的 region 索引不可比，而这件事从数字上看不出来；
- 分辨率**嵌套**——S4.5 扫 {64,256,1024,4096} 时，四档必须是同一套点的前缀，
  否则扫出来的差异里混着"采样点换了位置"；
- 面积守恒——`plan/03` §8.1 的 width 用它做分母。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from it import geom_cfg as G  # noqa: E402
from it.surfaces import (  # noqa: E402
    LEVELS,
    assign_to_surface,
    object_geometry,
    object_names,
    surface_for,
)


class TestSurfaces(unittest.TestCase):
    def test_every_object_builds(self):
        for obj in object_names():
            s = surface_for(obj)
            self.assertGreaterEqual(s.n_points, LEVELS[-1], obj)
            self.assertEqual(s.points.shape, (s.n_points, 3), obj)
            self.assertEqual(s.normals.shape, (s.n_points, 3), obj)
            self.assertEqual(s.part.shape, (s.n_points,), obj)
            self.assertTrue(np.isfinite(s.points).all(), obj)
            self.assertTrue(np.isfinite(s.normals).all(), obj)
            # 法向必须是单位向量
            n = np.linalg.norm(s.normals.astype(np.float64), axis=1)
            self.assertLess(float(np.abs(n - 1.0).max()), 1e-5, obj)

    def test_every_part_is_present(self):
        """每个声明的部件都要有点。

        少一个部件不会报错，只会让那片表面上的接触**全部归到别处**，
        热图看起来仍然正常。
        """
        for obj in object_names():
            s = surface_for(obj)
            counts = [int((s.part == i).sum()) for i in range(len(s.parts))]
            self.assertTrue(all(c > 0 for c in counts), f"{obj}: {dict(zip(s.parts, counts))}")

    def test_normals_point_outward(self):
        """沿 +n 走一小步必须在所有实体之外，沿 −n 走一小步必须落进某个实体。

        判内外用的是造面片时用的**同一套实体**，所以这是精确判据、不是近似。
        它抓的是"某个面片的 (u, v, n) 手性写反了"——那种错误在点云图上完全
        看不出来，却会让 `plan/02` §3.3 的 engage 方向整体反向。
        """
        eps = 0.2e-3
        for obj in object_names():
            s = surface_for(obj)
            _, solids, _ = object_geometry(obj)
            p = s.points.astype(np.float64)
            n = s.normals.astype(np.float64)
            outside = np.zeros(len(p), dtype=bool)
            inside = np.zeros(len(p), dtype=bool)
            for solid in solids:
                outside |= solid(p + eps * n, 0.0)      # 朝外走却仍在实体里 = 法向反了
                inside |= solid(p - eps * n, 0.0)
            self.assertEqual(int(outside.sum()), 0,
                             f"{obj}: {int(outside.sum())} 个点沿法向走仍在实体内")
            self.assertGreater(float(inside.mean()), 0.99,
                               f"{obj}: 只有 {inside.mean():.1%} 的点反向走进入实体")

    def test_deterministic(self):
        a = surface_for("knob")
        # 绕开进程内缓存，直接重建一次
        from it.surfaces import _build  # noqa: PLC0415

        b = _build("knob", "nominal", a.n_points)
        self.assertEqual(a.sha256, b.sha256)
        np.testing.assert_array_equal(a.points, b.points)
        np.testing.assert_array_equal(a.part, b.part)

    def test_levels_are_prefixes(self):
        """低分辨率必须是高分辨率的前缀（S4.5 的扫描依赖这一条）。"""
        s = surface_for("knob")
        for level in LEVELS:
            self.assertIn(level, s.parent)
            par = s.parent[level]
            self.assertEqual(par.shape, (s.n_points,))
            self.assertTrue((par < level).all())
            # 前 level 个点必须以自己为代表
            np.testing.assert_array_equal(par[:level], np.arange(level))

    def test_all_parts_reachable_at_lowest_level(self):
        """最粗那一档（64 点）也不该整个部件是空的。"""
        for obj in ("drawer", "knob", "slider"):
            s = surface_for(obj)
            present = {int(v) for v in s.part[: LEVELS[0]]}
            self.assertEqual(len(present), len(s.parts),
                             f"{obj} 在 {LEVELS[0]} 点那档丢了部件："
                             f"{set(range(len(s.parts))) - present}")

    def test_area_matches_analytic(self):
        """采样面积之和应当等于解析面积（球最好验，只有一个封闭表达式）。"""
        s = surface_for("ball")
        r = G.BallCfg().radius
        self.assertAlmostEqual(s.total_area, 4 * np.pi * r * r, delta=1e-4)

    def test_geometry_variants_differ(self):
        """几何变体必须真的换一份采样，且只有变的那个尺寸变。"""
        nom = surface_for("drawer", "nominal")
        g1 = surface_for("drawer", "g1")
        self.assertNotEqual(nom.sha256, g1.sha256)
        bar = nom.parts.index("bar_back")
        x_nom = float(nom.points[nom.part == bar][:, 0].mean())
        x_g1 = float(g1.points[g1.part == bar][:, 0].mean())
        # g1 把 handle_clearance 从 45 收到 38 mm，横杆整体前移 7 mm
        self.assertAlmostEqual(x_nom - x_g1, 7e-3, delta=1.5e-3)

    def test_variant_on_object_without_variants_raises(self):
        with self.assertRaises(ValueError):
            surface_for("block", "g1")

    def test_rotation_equivariance(self):
        """`plan/02` §7 第 1 条：接触数据与表面采样一起转，region 索引逐元素不变。"""
        rng = np.random.default_rng(0)
        s = surface_for("block")
        pts = s.points[rng.choice(s.n_points, 200, replace=False)].astype(np.float64)
        pts = pts * 1.001                       # 稍微离开表面，模拟真实接触点
        idx0, ok0, d0 = assign_to_surface(pts, s, max_dist=5e-3)

        # 任意旋转（罗德里格斯）
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        th = 0.7
        k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        rot = np.eye(3) + np.sin(th) * k + (1 - np.cos(th)) * (k @ k)

        idx1, ok1, d1 = assign_to_surface(pts @ rot.T, s.rotated(rot), max_dist=5e-3)
        np.testing.assert_array_equal(idx0, idx1)
        np.testing.assert_array_equal(ok0, ok1)
        np.testing.assert_allclose(d0, d1, atol=1e-9)

    def test_assign_rejects_far_points(self):
        """离表面太远的点必须被判为不在容差内，而不是硬塞给最近的采样点。"""
        s = surface_for("block")
        far = np.array([[0.5, 0.5, 0.5]])
        _, ok, _ = assign_to_surface(far, s, max_dist=5e-3)
        self.assertFalse(bool(ok[0]))

    def test_penetrating_points_still_belong(self):
        """穿进物体内部的接触点仍然算落在这个物体上（撞击瞬态下常见）。"""
        s = surface_for("ball")
        deep = np.array([[0.029, 0.0, 0.0]])          # 球半径 35 mm，穿进 6 mm
        _, ok, d = assign_to_surface(deep, s, max_dist=1e-3)
        self.assertTrue(bool(ok[0]))
        self.assertGreater(float(d[0]), 5e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
