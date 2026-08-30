"""`it.interaction`（Oracle Interaction Record 提取器）的本地单元测试。

**不依赖 Isaac Sim / torch**，系统 python 就能跑::

    PYTHONPATH=src python3 tools/test_interaction.py

用一条**手工构造**的 episode 来验，而不是真数据：接触点、法向、摩擦力、
相对速度全部人写死，于是"作用在物体上的合力和力矩应该是多少"可以笔算，
提取器算错了立刻能看出来。真数据上的验收是另一回事，靠的是与 S3 已验收的
接触部位分布对拍（`tools/s4_verify_records.py`）。

这里覆盖的都是"错了以后没有任何报错、只是结论作废"的性质：

- **法向定向**（P-37）：PhysX 报的法向正负两种约定，提取出的"作用在物体上的力"
  必须一样；
- **旋转等变**（`plan/02` §7 第 1 条）：整个场景转一下，region 索引与 mode
  逐元素不变，力与力矩跟着转，effect 不变；
- **接触体合并**（§7 第 3 条）：一块板与两块板产生**同样字段、同样维度**的记录；
- **mode 重判**（D-49）：带着力却被 PhysX 标成 separating 的接触，重判之后
  必须是 sticking 或 sliding。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from it.interaction import RECORD_SCHEMA, extract  # noqa: E402
from it.records import SCHEMA_VERSION, EpisodeRecord  # noqa: E402
from it.surfaces import surface_for  # noqa: E402

#: 帧数取 64：未来窗口是 1 s = 50 帧，要让第 0 帧的整窗都存在。
T = 64
SLOTS = 4
#: 方块 +X 面上的一个接触点：法向力 5 N 压进去，摩擦 0.5 N，接触体沿 +Y 滑。
CONTACT_X = 0.030
FN = 5.0
FT = 0.5
SLIP = 0.010
#: env 原点偏移。真实数据里 env 之间隔 2.2 m，物体与接触体的世界坐标都含这一项，
#: 而 `object/state` 不含——P-54 就是这两套混用出来的。夹具必须照搬这个结构，
#: 否则测不出那一类错。
ENV_ORIGIN = np.array([7.2, -5.4, 0.0])


def make_episode(*, bodies=("plate0", "plate1"), reported_normal_sign=+1.0,
                 raw_mode=3, rot=None, slide=True) -> EpisodeRecord:
    """造一条 probe_block 的 S3 episode。

    Args:
        bodies: 有几个接触体。只有第一个真的有接触，其余全空——
            对应 `single_finger` 那种只用一块板的示教。
        reported_normal_sign: PhysX 报的法向是 +外法向（这一束力作用在采集体上）
            还是 −外法向（作用在物体上）。两种都要能提取出同一个结果（P-37）。
        raw_mode: S3 写进记录的 mode。默认 3 = separating，正是 D-49 要重判的那一档。
        rot: 若给了 (3,3)，把所有物体系矢量与物体位姿一起转过去（等变性测试用）。
        slide: 接触点是否沿 +X 面向 +Y 移动。True 时斑块以 SLIP 的速度在物体表面上
            走（真滑移）；False 时接触点钉死不动，但相对速度**照给**——那正是
            抽屉/旋钮数据里的情形（板抖动使瞬时速度虚高，P-52），用来验
            "路程不相容就不采信瞬时速度"这条。
    """
    rot = np.eye(3) if rot is None else np.asarray(rot, dtype=np.float64)
    arrays: dict[str, np.ndarray] = {
        "phase": np.full(T, 2, dtype=np.int8),
        "progress": (np.arange(T, dtype=np.float32) + 1) / T,
        "valid_frame": np.ones(T, dtype=bool),
    }

    # 物体位姿：沿 +Y 匀速平移 1 mm/帧，姿态不变。
    # ⚠️ **两套坐标要与真实数据一致**（P-54）：`object/state` 是**减去 env 原点**的
    # 相对坐标，`source/object_*_w` 才是含 env 偏移的世界坐标；接触体位姿也是世界系。
    # 夹具里如果只写 `object/state`，提取器会拿不到世界位姿而直接报错——那是有意的。
    pos = np.zeros((T, 3))
    pos[:, 1] = np.arange(T) * 1e-3
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (T, 1))
    rel = np.concatenate([pos @ rot.T, _quat_mul_left(rot, quat)], axis=1)
    arrays["object/state"] = rel.astype(np.float32)              # env 相对
    world = pos + ENV_ORIGIN                                     # 含 env 偏移
    arrays["source/object_pos_w"] = (world @ rot.T).astype(np.float32)
    arrays["source/object_quat_w"] = _quat_mul_left(rot, quat).astype(np.float32)

    outward = np.array([1.0, 0.0, 0.0])          # 方块 +X 面的外法向
    # 滑移时接触点沿 +Y 在 +X 面上移动（面在 Y 上有 ±22.5 mm，走得下）
    y = (np.arange(T) * SLIP / 50.0) if slide else np.zeros(T)
    point = np.stack([np.full(T, CONTACT_X), y, np.zeros(T)], axis=1)
    # 报的这一束若是"作用在采集体上"，摩擦就与接触体的相对运动反向
    friction = np.array([0.0, -FT, 0.0]) * reported_normal_sign
    rel = np.array([0.0, SLIP, 0.0])             # 接触体相对物体沿 +Y 滑

    for k, body in enumerate(bodies):
        live = (k == 0)
        z3 = np.zeros((T, SLOTS, 3), dtype=np.float32)
        z1 = np.zeros((T, SLOTS), dtype=np.float32)
        c = f"contact/{body}"
        arrays[f"{c}/pos_obj"] = z3.copy()
        arrays[f"{c}/normal_obj"] = z3.copy()
        arrays[f"{c}/friction_obj"] = z3.copy()
        arrays[f"{c}/rel_vel_obj"] = z3.copy()
        arrays[f"{c}/normal_force"] = z1.copy()
        arrays[f"{c}/separation"] = z1.copy()
        arrays[f"{c}/valid"] = np.zeros((T, SLOTS), dtype=bool)
        arrays[f"{c}/mode"] = np.zeros((T, SLOTS), dtype=np.int8)
        if live:
            arrays[f"{c}/pos_obj"][:, 0] = (point @ rot.T).astype(np.float32)
            arrays[f"{c}/normal_obj"][:, 0] = (rot @ (reported_normal_sign * outward)
                                               ).astype(np.float32)
            arrays[f"{c}/friction_obj"][:, 0] = (rot @ friction).astype(np.float32)
            arrays[f"{c}/rel_vel_obj"][:, 0] = (rot @ rel).astype(np.float32)
            arrays[f"{c}/normal_force"][:, 0] = FN
            # PhysX 常常在带着力的同时报出正的 separation（P-31 的采样伪影），
            # S3 的判据会因此把它标成 separating。
            arrays[f"{c}/separation"][:, 0] = 1e-4
            arrays[f"{c}/valid"][:, 0] = True
            arrays[f"{c}/mode"][:, 0] = raw_mode
        # 板的位姿要与场景自洽：mode 的主判据是**位姿差分**（见 `_pose_slip`），
        # 写一堆零等于告诉提取器"板一动没动"，滑移就无从谈起。
        # 物体沿 +Y 走 1 mm/帧；滑移时板再多走 SLIP，不滑时与物体同速。
        plate_y = np.arange(T) * (1e-3 + (SLIP / 50.0 if slide else 0.0))
        plate_p = (np.stack([np.full(T, CONTACT_X + 0.02), plate_y, np.zeros(T)], axis=1)
                   + ENV_ORIGIN)          # 板也在世界系里，同样含 env 偏移
        arrays[f"source/{body}/root_pose"] = np.concatenate(
            [plate_p @ rot.T, _quat_mul_left(rot, quat)], axis=1).astype(np.float32)

    meta = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": "block-push-t000",
        "task": "probe_block",
        "probe_object": "block",
        "strategy_family": "push",
        "success": True,
        "control_hz": 50.0,
        "physics": {"plate_friction": 0.9},
    }
    return EpisodeRecord(meta=meta, arrays=arrays)


def _quat_mul_left(rot: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """把旋转 R 作用到一串四元数左边（R∘q），用矩阵绕一圈算，避免再写一遍四元数乘法。"""
    from it.interaction import _quat_to_rot  # noqa: PLC0415

    m = np.einsum("ij,tjk->tik", rot, _quat_to_rot(quat))
    w = np.sqrt(np.clip(1.0 + np.trace(m, axis1=1, axis2=2), 1e-12, None)) / 2.0
    return np.stack([w,
                     (m[:, 2, 1] - m[:, 1, 2]) / (4 * w),
                     (m[:, 0, 2] - m[:, 2, 0]) / (4 * w),
                     (m[:, 1, 0] - m[:, 0, 1]) / (4 * w)], axis=1)


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class TestExtract(unittest.TestCase):
    def test_schema_and_fields(self):
        out = extract(make_episode())
        out.validate()
        self.assertEqual(out.meta["schema_version"], RECORD_SCHEMA)
        for key in ("region/pos_obj", "region/point_idx", "region/weight",
                    "engage/dir", "mode/label", "mode/raw", "mech/wrench_obj",
                    "mech/generalized", "effect/future", "valid_s4"):
            self.assertIn(key, out.arrays)
        # `aux/*` 是诊断量，不该进模型输入；`source/*` 更不该出现
        model = out.model_arrays()
        self.assertFalse([k for k in model if k.startswith(("aux/", "source/"))])
        self.assertFalse([k for k in out.arrays if k.startswith("source/")])

    def test_force_on_object_is_convention_free(self):
        """P-37：PhysX 报的法向两种正负约定，提取出的力必须一样。

        手算：法向力 5 N 压进 +X 面 -> 物体受 −X 方向 5 N；
        接触体沿 +Y 滑 -> 摩擦拖着物体沿 +Y，0.5 N。
        力矩 r×f，r = (0.03, 0, 0) -> (0, 0, 0.03×0.5) = (0, 0, 0.015)。
        """
        expect_f = np.array([-FN, FT, 0.0])
        expect_t = np.array([0.0, 0.0, CONTACT_X * FT])
        for sign in (+1.0, -1.0):
            out = extract(make_episode(reported_normal_sign=sign))
            w = np.asarray(out.arrays["mech/wrench_obj"])[0]
            np.testing.assert_allclose(w[:3], expect_f, atol=1e-4,
                                       err_msg=f"报的法向符号 {sign} 时合力错了")
            np.testing.assert_allclose(w[3:], expect_t, atol=1e-6,
                                       err_msg=f"报的法向符号 {sign} 时力矩错了")

    def test_engage_direction_points_into_object(self):
        out = extract(make_episode())
        d = np.asarray(out.arrays["engage/dir"])[0, 0]
        np.testing.assert_allclose(d, [-1.0, 0.0, 0.0], atol=1e-5)

    def test_mode_rederived_from_slip(self):
        """带着力却被标成 separating 的接触，重判后必须是 sliding（D-49）。"""
        out = extract(make_episode(raw_mode=3, slide=True))
        self.assertEqual(int(np.asarray(out.arrays["mode/raw"])[0, 0]), 3)
        self.assertEqual(int(np.asarray(out.arrays["mode/label"])[0, 0]), 2)
        self.assertEqual(out.meta["extraction"]["mode_source"], "pose_diff")

    def test_pose_diff_beats_corrupted_instant_velocity(self):
        """板与物体同速（没在滑）却报着 10 mm/s 的瞬时相对速度 —— 判 sticking。

        这正是抽屉/旋钮数据里的情形（P-52）：PhysX 报的瞬时速度被采集板的姿态
        极限环污染，积出来是几十毫米的滑移，而位姿差分说两者根本没有相对运动。
        mode 必须听位姿差分的，另外两路只作诊断留在记录里。
        """
        out = extract(make_episode(slide=False))
        self.assertEqual(out.meta["extraction"]["mode_source"], "pose_diff")
        self.assertFalse(out.meta["extraction"]["rel_vel_trusted"][0])
        self.assertEqual(int(np.asarray(out.arrays["mode/label"])[10, 0]), 1)
        # 三路证据都留在记录里，谁也不掩盖谁
        self.assertGreater(float(np.asarray(out.arrays["mode/inst_slip"])[10, 0]), 5e-3)
        self.assertLess(float(np.asarray(out.arrays["mode/pose_slip"])[10, 0]), 1e-3)

    def test_pose_diff_detects_real_sliding(self):
        """板真的相对物体滑 10 mm/s 时，位姿差分要判成 sliding。"""
        out = extract(make_episode(slide=True))
        self.assertGreater(float(np.asarray(out.arrays["mode/pose_slip"])[10, 0]), 5e-3)
        self.assertEqual(int(np.asarray(out.arrays["mode/label"])[10, 0]), 2)

    def test_separating_only_without_force(self):
        """力没了、几何上正在分开，才是 separating。"""
        rec = make_episode(slide=False)
        rec.arrays["contact/plate0/normal_force"][:, 0] = 0.0
        out = extract(rec)
        self.assertEqual(int(np.asarray(out.arrays["mode/label"])[0, 0]), 3)

    def test_body_count_invariance(self):
        """`plan/02` §7 第 3 条：改变接触体数量，记录的字段与维度不变。"""
        one = extract(make_episode(bodies=("plate0",)))
        two = extract(make_episode(bodies=("plate0", "plate1")))
        self.assertEqual(set(one.arrays), set(two.arrays))
        for key in ("mech/wrench_obj", "mech/generalized", "effect/future"):
            self.assertEqual(one.arrays[key].shape, two.arrays[key].shape, key)
            np.testing.assert_allclose(one.arrays[key], two.arrays[key], atol=1e-6)
        # 逐接触槽的字段行数会随接触体数量变（那是**容量**不是维度），
        # 有效接触的内容必须一致
        for key in ("region/pos_obj", "engage/dir", "mode/label"):
            a = np.asarray(one.arrays[key])[0, 0]
            b = np.asarray(two.arrays[key])[0, 0]
            np.testing.assert_allclose(np.asarray(a, float), np.asarray(b, float),
                                       atol=1e-6, err_msg=key)
        self.assertEqual(one.meta["fields"]["n_slots"], SLOTS)
        self.assertEqual(two.meta["fields"]["n_slots"], 2 * SLOTS)

    def test_rotation_equivariance(self):
        """`plan/02` §7 第 1 条：场景整体旋转后，物体系表示的不变量逐元素不变。

        这是本轮选定的做法（代数等变性测试）：把接触数据、物体位姿和表面采样
        一起转，检查
        **region 索引 / mode / effect 增量**不变，**力与力矩**跟着转。
        它抓的是提取器里混进硬编码世界轴（"+X 是拉出方向""up = +Z"）这一类错误。
        """
        rot = _rot_z(0.7) @ _rot_z(0.0)
        base = extract(make_episode())
        surf = surface_for("block").rotated(rot)
        turned = extract(make_episode(rot=rot), surface=surf)

        np.testing.assert_array_equal(base.arrays["region/point_idx"],
                                      turned.arrays["region/point_idx"])
        np.testing.assert_array_equal(base.arrays["mode/label"],
                                      turned.arrays["mode/label"])
        np.testing.assert_allclose(base.arrays["effect/future"],
                                   turned.arrays["effect/future"], atol=1e-6)
        w0 = np.asarray(base.arrays["mech/wrench_obj"], dtype=np.float64)
        w1 = np.asarray(turned.arrays["mech/wrench_obj"], dtype=np.float64)
        np.testing.assert_allclose(w1[:, :3], w0[:, :3] @ rot.T, atol=1e-5)
        np.testing.assert_allclose(w1[:, 3:], w0[:, 3:] @ rot.T, atol=1e-6)

    def test_future_window(self):
        """未来窗口是 1 s / 10 点的**增量**，末尾不足处标 invalid、不外插。"""
        out = extract(make_episode())
        fut = np.asarray(out.arrays["effect/future"])
        ok = np.asarray(out.arrays["effect/future_valid"])
        self.assertEqual(fut.shape, (T, 10, 6))
        # 物体沿 +Y 每帧 1 mm，步长 5 帧 -> 第 j 个采样点是 5(j+1) mm
        np.testing.assert_allclose(fut[0, :, 1], 5e-3 * np.arange(1, 11), atol=1e-6)
        self.assertTrue(ok[0].all())            # 第 0 帧的整窗都在
        self.assertFalse(ok[-1].any())          # 最后一帧没有未来
        np.testing.assert_allclose(fut[-1], 0.0)

    def test_dirty_frame_filter(self):
        """力尖峰（P-27）必须被 `valid_s4` 挡掉，且不覆盖 S3 的 `valid_frame`。"""
        rec = make_episode()
        rec.arrays["contact/plate0/normal_force"][5, 0] = 4000.0
        out = extract(rec)
        self.assertFalse(bool(np.asarray(out.arrays["valid_s4"])[5]))
        self.assertTrue(bool(np.asarray(out.arrays["valid_frame"])[5]))
        self.assertTrue(bool(np.asarray(out.arrays["valid_s4"])[4]))

    def test_mixed_frames_must_raise(self):
        """物体位姿与接触体位姿不在同一个系时，必须**直接报错**（P-54）。

        这是那一类 bug 的唯一硬防线：它零征兆——逐帧数值、接触部位分布、
        动力学一致性、其余单元测试**全部正常**，只有力臂大得离谱。
        """
        rec = make_episode()
        # 把物体的世界位姿换成 env 相对的（正是 P-54 里那半个错）
        rec.arrays["source/object_pos_w"] = np.asarray(
            rec.arrays["object/state"])[:, :3].copy()
        with self.assertRaises(ValueError) as cm:
            extract(rec)
        self.assertIn("坐标系", str(cm.exception))

    def test_missing_object_pose_must_raise(self):
        """拿不到物体的世界位姿时宁可炸，也不许假定它在原点（P-54 的另一半）。"""
        rec = make_episode()
        for k in ("source/object_pos_w", "source/object_quat_w"):
            rec.arrays.pop(k)
        with self.assertRaises((ValueError, KeyError, TypeError)):
            extract(rec)

    def test_region_lands_on_the_right_part(self):
        """接触点必须归到 +X 面上，不是别的面。"""
        out = extract(make_episode(slide=False))
        surf = surface_for("block")
        idx = int(np.asarray(out.arrays["region/point_idx"])[0, 0])
        self.assertGreaterEqual(idx, 0)
        self.assertEqual(surf.parts[int(surf.part[idx])], "face_px")
        self.assertTrue(bool(np.asarray(out.arrays["region/on_surface"])[0, 0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
