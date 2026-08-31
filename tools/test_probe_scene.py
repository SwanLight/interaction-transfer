#!/usr/bin/env python3
"""`it.probe_scene` 的一致性检查。本机跑，纯 python，不要 numpy/torch/Isaac。

**这几条都是能失败的检查**，不是"跑一遍看不报错"：

1. 自由刚体的 ``init_pos[2]`` 必须等于它的半高——也就是"物体恰好放在 z=0 的台面上"。
   摆错一个数就红。这一条挡的是 F1 那类错误：E-I 环境第一版没有台面，物体开局
   自由落体，而 dry-run 的 ``region/mode/mech`` 三项恒为 0 被读成了"量级合理"。
2. ``body_subpath`` 必须与 `tools/s3_source_probe.py::OBJECTS` 里的 ``body_path``
   逐字相同。那张表是**产生数据**时用的，接触 filter 指错 prim 就是 P-17
   （``net_forces_w`` 照常工作，接触位置与摩擦力静默失效）。
   ⚠️ 这里不 import 采集脚本（它 import Isaac，本机跑不了），改为**读源文件里的
   字面量**——比"两处各写一遍"强，比 import 便宜。
3. 台面外廓必须大于执行器目标位姿的活动范围，否则物体可能被推下桌而看不出来。

用法::

    PYTHONPATH=src python3 tools/test_probe_scene.py
"""

from __future__ import annotations

import ast
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from it import geom_cfg as B          # noqa: E402
from it.probe_scene import (          # noqa: E402
    EXECUTOR_TIP, PROBE_OBJECTS, TABLE_EXTENT_TRAIN, TABLE_FRICTION, TABLE_THICKNESS)

SOURCE = Path(__file__).resolve().parent / "s3_source_probe.py"


def _collector_body_paths() -> dict[str, str]:
    """从采集脚本的源码里抠出 ``OBJECTS`` 每一项的 ``body_path`` 字面量。

    用 AST 而不是正则：正则会被注释里的同名字符串骗到，而这张表的注释很长。
    """
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.AnnAssign):
            continue
        if not (isinstance(node.target, ast.Name) and node.target.id == "OBJECTS"):
            continue
        assert isinstance(node.value, ast.Dict), "OBJECTS 不再是字面 dict，这个检查要改"
        out: dict[str, str] = {}
        for key, value in zip(node.value.keys, node.value.values):
            assert isinstance(key, ast.Constant) and isinstance(value, ast.Call)
            for kw in value.keywords:
                if kw.arg == "body_path":
                    assert isinstance(kw.value, ast.Constant)
                    out[key.value] = kw.value.value
        return out
    raise AssertionError("在 s3_source_probe.py 里没找到 OBJECTS")


class TestRestingOnTheTable(unittest.TestCase):
    """自由刚体必须恰好放在台面上。"""

    def test_free_bodies_sit_exactly_on_the_table_top(self):
        for name, spec in PROBE_OBJECTS.items():
            if spec.rest_half_height is None:
                continue
            with self.subTest(object=name):
                self.assertAlmostEqual(
                    spec.init_pos[2], spec.rest_half_height, places=9,
                    msg=f"{name} 的初始高度 {spec.init_pos[2]} 与半高 "
                        f"{spec.rest_half_height} 对不上——它不是放在台面上，"
                        "而是悬空或埋进台面里")

    def test_every_free_rigid_object_declares_its_half_height(self):
        """自由刚体（body_subpath 为空且非 kinematic）必须给出半高。

        没有半高，上一条检查就会被静默跳过——那正是"检查存在但从不失败"。
        """
        free = {"block", "column", "roller", "ball"}
        for name in free:
            self.assertIsNotNone(PROBE_OBJECTS[name].rest_half_height,
                                 f"{name} 是自由刚体，必须声明 rest_half_height")

    def test_half_heights_come_from_geom_cfg_not_from_a_literal(self):
        """半高必须与 `geom_cfg` 一致：改了几何而没改这里 = 什么都没改（P-38）。"""
        expected = {"block": B.BlockCfg().size[2] / 2,
                    "column": B.ColumnCfg().height / 2,
                    "roller": B.RollerCfg().radius,
                    "ball": B.BallCfg().radius}
        for name, value in expected.items():
            self.assertAlmostEqual(PROBE_OBJECTS[name].rest_half_height, value, places=12)


class TestFilterPrimPaths(unittest.TestCase):
    """接触 filter 必须指到真正带 RigidBodyAPI 的那个 prim（P-17 / 规则 7）。"""

    def test_body_paths_match_the_collector_verbatim(self):
        collector = _collector_body_paths()
        self.assertEqual(set(collector), set(PROBE_OBJECTS),
                         "两张表的物体集合不一致")
        for name, path in collector.items():
            with self.subTest(object=name):
                self.assertEqual(PROBE_OBJECTS[name].body_path, path,
                                 f"{name} 的刚体路径与采集侧不一致：指错 prim 之后"
                                 "接触位置与摩擦力会静默失效（P-17）")

    def test_filter_expr_composes_the_full_path(self):
        obj = "/World/envs/env_.*/Object"
        self.assertEqual(PROBE_OBJECTS["block"].filter_expr(obj), obj)
        self.assertEqual(PROBE_OBJECTS["slab"].filter_expr(obj), obj + "/Board")
        self.assertEqual(PROBE_OBJECTS["ridge"].filter_expr(obj), obj + "/Body")

    def test_sub_prim_objects_are_exactly_the_non_free_ones(self):
        """自由刚体的刚体在根上，其余都在子 prim 上。反了就说明表被改坏了。"""
        at_root = {n for n, s in PROBE_OBJECTS.items() if not s.body_subpath}
        self.assertEqual(at_root, {"block", "column", "roller", "ball"})


class TestTableAndTips(unittest.TestCase):
    def test_table_is_larger_than_the_reachable_target_range(self):
        """台面半宽必须大于执行器目标位姿的活动半径，否则物体可能被推下桌。

        ``pos_limit`` 从 E-I 环境的源码里读，不在这里另写一个数——两处各写一遍
        正是本模块要消灭的东西。（不 import：`envs/interaction` 要 torch 与 Isaac。）
        """
        env_src = Path(__file__).resolve().parent.parent / "src/it/envs/interaction.py"
        found = re.findall(r"pos_limit=([0-9.]+)", env_src.read_text(encoding="utf-8"))
        self.assertTrue(found, "在 envs/interaction.py 里没找到 pos_limit")
        reach = max(float(v) for v in found)
        self.assertGreater(TABLE_EXTENT_TRAIN / 2, reach * 0.9,
                           f"台面半宽 {TABLE_EXTENT_TRAIN / 2} 比执行器够得到的范围 "
                           f"{reach} 还小，物体会被推下桌而没人发现")

    def test_table_material_matches_the_collection_scene(self):
        """μ 与厚度是**产生指令时**的物理条件，不能在 E-I 侧另取一个值。"""
        text = SOURCE.read_text(encoding="utf-8")
        self.assertIn(f"size=(2.4, 2.4, {TABLE_THICKNESS})", text)
        self.assertIn(f"friction={TABLE_FRICTION}", text)

    def test_executor_tips_point_below_the_root(self):
        for name, tip in EXECUTOR_TIP.items():
            with self.subTest(executor=name):
                self.assertLess(tip[2], 0.0, f"{name} 的末端应当在根 prim 下方")
                self.assertGreater(abs(tip[2]), 0.05, f"{name} 的末端偏移小得不合理")


if __name__ == "__main__":
    unittest.main(verbosity=2)
