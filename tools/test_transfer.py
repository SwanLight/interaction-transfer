"""S5 statistical interaction-transfer contract 的本地单元测试。

运行：``PYTHONPATH=src python3 tools/test_transfer.py``。不依赖 Isaac Sim。
"""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from it.interaction import extract  # noqa: E402
from it.transfer import (  # noqa: E402
    EXECUTOR_ARRAYS,
    TRANSFER_SCHEMA_VERSION,
    TransferError,
    bin_index,
    build_transfer,
    load_transfer,
    phase_budget,
    save_transfer,
)
from it.surfaces import surface_for  # noqa: E402
from test_interaction import make_episode  # noqa: E402


def records(count: int = 3):
    out = []
    for index in range(count):
        record = extract(make_episode(slide=(index != 1)))
        record.meta["episode_id"] = f"demo-{index:02d}"
        record.meta["strategy_family"] = "sliding" if index != 1 else "sticking"
        out.append(record)
    return out


class TestInteractionTransfer(unittest.TestCase):
    def test_builds_task_agnostic_executor_payload(self):
        transfer = build_transfer(records(), n_bins=8, n_surface=64)
        transfer.validate()
        payload = transfer.executor_arrays()
        self.assertEqual(set(payload), set(EXECUTOR_ARRAYS))
        self.assertFalse(any("task" in name or "joint" in name or "source" in name
                             or "generalized" in name or "phase" in name
                             or "progress" in name for name in payload))
        self.assertEqual(payload["region/mass/mean"].shape, (8, 64))
        np.testing.assert_array_equal(payload["command/valid"],
                                      payload["support/episodes"] > 0)
        self.assertEqual(payload["mode/prob"].shape, (8, 64, 4))
        self.assertEqual(payload["mech/traction_obj/median"].shape, (8, 64, 3))
        self.assertEqual(payload["mech/moment_density_obj/median"].shape, (8, 64, 3))
        occupied = payload["region/mass/mean"].sum(axis=1) > 0
        np.testing.assert_allclose(payload["region/mass/mean"][occupied].sum(axis=1), 1.0,
                                   atol=2e-5)

    def test_surface_projection_is_complete(self):
        """表面投影残差 + 被滤掉的力占比。

        ⚠️ 这**不是** "6D wrench 重建误差"。cell 内先算局部力矩再求和，在代数上恒等于
        直接对接触点求力矩，拿它跟自己对拍只会得到浮点精度级的数（v1 的 README 把
        6e-6 N 当成了重建的证据）。有信息量的是被 ``on_surface``/``weight>0`` 滤掉的
        接触占了多少力，以及由此与 S4 独立记的 ``mech/wrench_obj`` 差多少。
        """
        diagnostics = build_transfer(records(), n_bins=8, n_surface=64).meta["diagnostics"]
        self.assertLessEqual(diagnostics["projection_residual_force_N"], 1e-6)
        self.assertLessEqual(diagnostics["projection_residual_torque_Nm"], 1e-6)
        self.assertLessEqual(diagnostics["dropped_force_mass_fraction_max"], 1e-6)

    def test_every_cell_field_comes_from_the_same_episodes(self):
        """同一个 cell 上"该接触"与"力必须为零"不得并存。

        v1 的 region/engage/mode 用 NaN 排除没碰过的 episode，而 traction 用 0 把它们
        算进去。真实 12 条示教上，**抽屉 58% / 擦拭 81% / 旋钮 49%** 的 occupied cell
        因此拿到"接触质量 > 0 而力中位数恰为 0"的自相矛盾指令（P-59）。
        """
        payload = build_transfer(records(), n_bins=8, n_surface=64).executor_arrays()
        support = payload["region/support"]
        occupied = support > 0
        self.assertTrue(occupied.any())
        traction = np.linalg.norm(payload["mech/traction_obj/median"], axis=2)
        self.assertEqual(int((traction[occupied] == 0).sum()), 0)
        # 反过来，没有任何 episode 支持的 cell 上所有接触条件化字段必须干净为零。
        self.assertEqual(float(np.abs(traction[~occupied]).max(initial=0.0)), 0.0)
        self.assertEqual(float(np.abs(payload["mode/prob"][~occupied]).max(initial=0.0)), 0.0)
        self.assertEqual(float(payload["region/duty"][~occupied].max(initial=0.0)), 0.0)
        # duty 与 support 是下游分辨 1/N 与 N/N 的唯一依据。
        self.assertTrue(np.all(support <= payload["support/episodes"][:, None]))
        self.assertTrue(np.all((payload["region/duty"] >= 0)
                               & (payload["region/duty"] <= 1.0 + 1e-6)))

    def test_engage_concentration_is_kept_not_thresholded_away(self):
        """各 episode 单位方向的合矢量长度就是方向多峰性的直接度量，不能只留 bool。"""
        payload = build_transfer(records(), n_bins=8, n_surface=64).executor_arrays()
        concentration = payload["engage/concentration"]
        occupied = payload["region/support"] > 0
        self.assertTrue(np.all((concentration >= -1e-6) & (concentration <= 1.0 + 1e-6)))
        np.testing.assert_array_equal(payload["engage/valid"], concentration > 1e-8)
        # 造出来的示教方向一致，集中度应接近 1；若被实现成"先归一化再平均"就恒为 1，
        # 因此同时检查它确实是从未归一化的平均量算出来的（有支持处 > 0）。
        self.assertGreater(float(concentration[occupied].min()), 0.0)

    def test_alignment_never_silently_drops_frames(self):
        """任何**有帧**的 phase 都必须分到格子，否则那一段会被整段丢掉而不报错。"""
        demos = records(2)
        surface = surface_for("block", "nominal")
        for record in demos:
            phase = np.asarray(record.arrays["phase"]).copy()
            phase[: len(phase) // 4] = 0        # 造一个活动量为零的接近段
            record.arrays["phase"] = phase
        budget = phase_budget(demos, surface, n_bins=8)
        self.assertEqual(sum(budget), 8)
        self.assertGreaterEqual(budget[0], 2, "零活动量的接近段仍要拿到保底格数")
        self.assertEqual(budget[1], 0, "没有帧的 phase 不应占格子")
        for record in demos:
            index = bin_index(record, surface, budget=budget)
            valid = np.asarray(record.arrays["valid_s4"], dtype=bool)
            self.assertTrue(np.all(index[valid] >= 0))
            self.assertEqual(int(index[valid].max()) < 8, True)

    def test_v1_artifacts_are_rejected(self):
        transfer = build_transfer(records(2), n_bins=8, n_surface=64)
        self.assertEqual(transfer.meta["schema_version"], TRANSFER_SCHEMA_VERSION)
        transfer.meta["schema_version"] = "interaction-transfer-v1"
        with self.assertRaises(TransferError):
            transfer.validate()

    def test_episode_not_frame_is_statistical_unit(self):
        demos = records(2)
        second = demos[1]
        # 改变第二位采集者的接触力；region 仍按每条 episode 归一化，不能因为力更大
        # 或帧更多而获得更多采集者票数。
        second.arrays["region/weight"] *= 10.0
        second.arrays["mech/force_obj"] *= 10.0
        second.arrays["mech/wrench_obj"] *= 10.0
        transfer = build_transfer(demos, n_bins=8, n_surface=64)
        occupied = transfer.arrays["region/mass/mean"].sum(axis=1) > 0
        np.testing.assert_allclose(transfer.arrays["region/mass/mean"][occupied].sum(axis=1),
                                   1.0, atol=2e-5)
        self.assertEqual(transfer.meta["aggregation"]["episode_weighting"],
                         "equal_after_within_episode_bin_summary")

    def test_rejects_mixed_task_failure_and_unknown_executor_field(self):
        demos = records(2)
        bad_task = copy.deepcopy(demos[1])
        bad_task.meta["task"] = "another_task"
        with self.assertRaises(TransferError):
            build_transfer([demos[0], bad_task], n_bins=8, n_surface=64)

        failed = copy.deepcopy(demos[1])
        failed.meta["success"] = False
        with self.assertRaises(TransferError):
            build_transfer([demos[0], failed], n_bins=8, n_surface=64)

        transfer = build_transfer(demos, n_bins=8, n_surface=64)
        transfer.arrays["phase"] = np.zeros(8, dtype=np.int8)
        with self.assertRaises(TransferError):
            transfer.executor_arrays()

    def test_round_trip(self):
        transfer = build_transfer(records(), n_bins=8, n_surface=64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transfer.npz"
            save_transfer(transfer, path)
            loaded = load_transfer(path)
        self.assertEqual(loaded.meta, transfer.meta)
        for name in EXECUTOR_ARRAYS:
            np.testing.assert_array_equal(loaded.arrays[name], transfer.arrays[name])


if __name__ == "__main__":
    unittest.main()
