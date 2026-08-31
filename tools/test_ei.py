"""E-I 指令通道 / reward / 网络的单元测试。

⚠️ **这一套要在服务器上跑**，因为它依赖 torch：

    ssh root@10.0.6.98 'cd /workspace/interaction_transfer && \\
        PYTHONPATH=src /isaac-sim/python.sh tools/test_ei.py'

本机那四套（surfaces / interaction / records / transfer）是纯 numpy，继续留在本机。
分界线是 **torch**，不是 Isaac——这一套不需要 Isaac，但必须用服务器那个
torch 2.7.0+cu128，因为训练跑的是它（D-21 冻结环境；拿另一个版本验过的代码
在服务器上跑的是另一份实现）。
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from it.ei_command import (  # noqa: E402
    FIELD_GROUPS, SPATIAL_DIM, TEMPORAL_DIM, CommandBank, CommandError,
    WindowTracker, spatial_features, temporal_features)
from it.ei_policy import InteractionActorCritic, InteractionPolicy  # noqa: E402
from it.ei_reward import (  # noqa: E402
    EFFECT_RIGID_DEADBAND_M, _GAUSS_NORM, assign_cells, box_violation,
    effect_deficit, effect_magnitude,
    interaction_reward, match_functional_region,
    scatter_contact_compatibility, surface_traction)
from it.surfaces import SCATTER_SIGMA  # noqa: E402


def _artifacts(root: str | None = None, limit: int = 4) -> list[Path]:
    # 契约升 v5 之后，v4 与 v5 的 artifact 在服务器上并存（前者还在被训练用着）。
    # 用环境变量指哪一套，免得测试悄悄跑在过期的那一套上。
    root = root or os.environ.get("IT_PROBE_DIR", "/tmp/s6/probe")
    paths = sorted(Path(root).glob("*.npz"))[:limit]
    if len(paths) < 2:
        raise unittest.SkipTest(f"{root} 下不足 2 份 artifact")
    return paths


class TestReward(unittest.TestCase):
    def test_box_violation_is_zero_inside_the_set(self):
        """允许集合是**集合**不是点目标：集合内必须恰好 0（D-71 / D-67）。

        若改成"离中位数越近越好"，C4 就被悄悄实现成了 C5（精确复现 source 的力），
        而 `plan/02` §6 的 C4 vs C5 对照正是要回答"复制 source 是不是多余的"。
        """
        lo = torch.tensor([[[-1.0, -2.0, -1.0]]])
        hi = torch.tensor([[[3.0, 2.0, 1.0]]])
        for inside in ([0.0, 0.0, 0.0], [-1.0, -2.0, -1.0], [3.0, 2.0, 1.0]):
            value = torch.tensor([[inside]])
            self.assertEqual(float(box_violation(value, lo, hi)), 0.0)
        near = float(box_violation(torch.tensor([[[3.5, 0.0, 0.0]]]), lo, hi))
        far = float(box_violation(torch.tensor([[[5.0, 0.0, 0.0]]]), lo, hi))
        self.assertGreater(near, 0.0)
        self.assertGreater(far, near, "越界越多必须罚得越重")

    def test_single_contact_traction_matches_the_analytic_gaussian(self):
        """一个孤立接触点上的 traction 必须等于 F/(2πσ²)——核是归一化面密度。

        这条能失败：把归一化常数写错、或把 σ 换成采样 pitch，数值立刻不对。
        """
        force = torch.tensor([[[0.0, 0.0, 5.0]]])
        pos = torch.zeros(1, 1, 3)
        valid = torch.ones(1, 1, dtype=torch.bool)
        cell = torch.zeros(1, 1, dtype=torch.long)
        traction, mass = surface_traction(pos, force, valid, cell, n_surface=4)
        # 量级是 5×10⁴，float32 给不了绝对精度到小数点后三位，比**相对**误差。
        self.assertAlmostEqual(float(traction[0, 0, 2]) / (5.0 * _GAUSS_NORM),
                               1.0, places=5)
        self.assertAlmostEqual(float(mass[0, 0]), 5.0, places=6)
        self.assertAlmostEqual(float(traction[0, 1].abs().sum()), 0.0, places=9)

    def test_two_contacts_within_a_bandwidth_add_up(self):
        """斑块重叠是真实物理，必须叠加而不是各算各的。"""
        pos = torch.tensor([[[0.0, 0.0, 0.0], [SCATTER_SIGMA * 0.1, 0.0, 0.0]]])
        force = torch.tensor([[[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]]])
        valid = torch.ones(1, 2, dtype=torch.bool)
        cell = torch.zeros(1, 2, dtype=torch.long)
        traction, _ = surface_traction(pos, force, valid, cell, n_surface=2)
        single = 2.0 * _GAUSS_NORM
        self.assertGreater(float(traction[0, 0, 2]), 1.8 * single)

    def test_assignment_respects_the_contact_side(self):
        """薄物体上纯几何最近邻会把正面的接触归到背面（D-68 的在线版）。"""
        cells = torch.tensor([[[0.0, 0.0, 0.01], [0.0, 0.0, -0.01]]])
        normals = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]])
        # 接触点更靠近背面的那个格，但法向朝上 -> 必须归到正面
        pos = torch.tensor([[[0.0, 0.0, -0.006]]])
        contact_normal = torch.tensor([[[0.0, 0.0, 1.0]]])
        valid = torch.ones(1, 1, dtype=torch.bool)
        index = assign_cells(pos, contact_normal, valid, cells, normals)
        self.assertEqual(int(index[0, 0]), 0)

    def test_region_term_bounds_and_no_contact_case(self):
        allowed = torch.tensor([[True, False]])
        common = dict(traction=torch.zeros(1, 2, 3), slip_speed=torch.zeros(1, 2),
                      traction_lo=-torch.ones(1, 2, 3), traction_hi=torch.ones(1, 2, 3),
                      slip_lo=torch.zeros(1, 2, 1), slip_hi=torch.ones(1, 2, 1),
                      force_penalty=torch.zeros(1), effect_deficit=torch.zeros(1))
        full_in = interaction_reward(mass=torch.tensor([[3.0, 0.0]]), allowed=allowed,
                                     **common)
        full_out = interaction_reward(mass=torch.tensor([[0.0, 3.0]]), allowed=allowed,
                                      **common)
        none = interaction_reward(mass=torch.zeros(1, 2), allowed=allowed, **common)
        # **带符号的占比**：全落在允许区域 +1、全落在外面 −1、没接触 0。
        # 这三个数的**次序**是整套 reward 能不能学的关键：`r_region` 是唯一能取正值
        # 的项，若"压对地方"不比"什么都不做"高，最优策略就是永远躲开物体
        # （实测第一版悬停 −0.43/步、压住 −278/步，PPO 会稳稳学会躲开）。
        self.assertAlmostEqual(float(full_in.region), 1.0, places=6)
        self.assertAlmostEqual(float(full_out.region), -1.0, places=6)
        self.assertAlmostEqual(float(none.region), 0.0, places=6,
                               msg="完全没接触时 region 项应为 0")
        self.assertGreater(float(full_in.region), float(none.region),
                           "压对地方必须严格优于不碰，否则最优策略是躲开物体")
        self.assertGreater(float(none.region), float(full_out.region),
                           "压错地方必须严格劣于不碰")

    def test_functional_region_matching_is_continuous_and_topology_free(self):
        """target 接触匹配功能区域，不复刻 source 的格号与接触点数量。

        旧实现把连续点硬归到 256 格后做 bool membership：跨一条格边界就从 +1 跳到
        −1。这里验证 2 mm 的形态/量化偏差仍是正匹配、30 mm 的错位是负匹配；两个
        target 接触还可以同时匹配同一个 source 功能格。
        """
        cells = torch.tensor([[[0.0, 0.0, 0.0], [0.03, 0.0, 0.0]]])
        normals = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]])
        area = torch.full((1, 2), np.pi * SCATTER_SIGMA ** 2)
        allowed = torch.tensor([[True, False]])
        pos = torch.tensor([[[0.002, 0.0, 0.0], [0.003, 0.0, 0.0],
                             [0.030, 0.0, 0.0]]])
        normal = torch.tensor([[[0.0, 0.0, 1.0]]]).expand(1, 3, 3)
        valid = torch.ones(1, 3, dtype=torch.bool)
        matched, compatibility = match_functional_region(
            pos, normal, valid, cells, normals, area, allowed)
        self.assertEqual(matched[0, :2].tolist(), [0, 0],
                         "contact topology 不同也应匹配同一功能格")
        self.assertGreater(float(compatibility[0, 0]), 0.5)
        self.assertGreater(float(compatibility[0, 1]), 0.5)
        self.assertLess(float(compatibility[0, 2]), 1e-4)

        force = torch.tensor([[[0.0, 0.0, 2.0], [0.0, 0.0, 1.0],
                               [0.0, 0.0, 3.0]]])
        pooled = scatter_contact_compatibility(force, valid, matched,
                                                compatibility, n_surface=2)
        self.assertTrue(bool(((pooled >= 0) & (pooled <= 1)).all()))

    def test_soft_region_compatibility_preserves_reward_order(self):
        common = dict(traction=torch.zeros(1, 1, 3), slip_speed=torch.zeros(1, 1),
                      traction_lo=-torch.ones(1, 1, 3),
                      traction_hi=torch.ones(1, 1, 3),
                      slip_lo=torch.zeros(1, 1, 1), slip_hi=torch.ones(1, 1, 1),
                      force_penalty=torch.zeros(1), effect_deficit=torch.zeros(1),
                      mass=torch.ones(1, 1))
        near = interaction_reward(allowed=torch.tensor([[0.9]]), **common)
        far = interaction_reward(allowed=torch.tensor([[0.1]]), **common)
        self.assertAlmostEqual(float(near.region), 0.8, places=6)
        self.assertAlmostEqual(float(far.region), -0.8, places=6)

    def test_violation_is_capped_and_reports_saturation(self):
        """越界量必须有饱和点，且饱和本身要被报出来。

        不封顶时 `mech` 会被"除以标定值"放大成 −254，而悬停只有 −0.43——
        整套 reward 于是奖励"别碰"。封顶把它压回 [−1, 0]，而**截断比例**是诊断量：
        长期居高不下说明那一项已经没有梯度，要回头查集合本身。
        """
        far = interaction_reward(
            mass=torch.tensor([[3.0, 0.0]]), allowed=torch.tensor([[True, False]]),
            traction=torch.tensor([[[1e6, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
            slip_speed=torch.zeros(1, 2),
            traction_lo=-torch.ones(1, 2, 3), traction_hi=torch.ones(1, 2, 3),
            slip_lo=torch.zeros(1, 2, 1), slip_hi=torch.ones(1, 2, 1),
            force_penalty=torch.zeros(1), effect_deficit=torch.zeros(1))
        self.assertAlmostEqual(float(far.mech), -1.0, places=6, msg="必须封顶到 −1")
        self.assertAlmostEqual(float(far.mech_saturated), 1.0, places=6,
                               msg="封顶发生了就必须报出来")

    def test_effect_deficit_is_bounded_and_scale_free(self):
        """第一版 `r_effect` 取"实测与指令之差"，dry-run 上炸到 −26647、逐步线性增长，
        比其余三项大六个数量级。两个错叠在一起：比的不是同一段时间（命令格的时长是
        弹性的，指令 effect 是"这一格总共要发生多少"而不是速率），以及除以了一个
        可能接近零的刻度。缺口形式把两个问题一起消掉——刻度在分子分母里约掉。
        """
        demand = torch.tensor([1.0, 1.0, 1.0, 0.0])
        achieved = torch.tensor([0.0, 0.5, 2.0, 5.0])
        d = effect_deficit(achieved, demand)
        self.assertAlmostEqual(float(d[0]), 1.0, places=6, msg="一点没动 = 满缺口")
        self.assertAlmostEqual(float(d[1]), 0.5, places=6)
        self.assertAlmostEqual(float(d[2]), 0.0, places=6, msg="超额完成不该有负缺口")
        self.assertAlmostEqual(float(d[3]), 0.0, places=6,
                               msg="这一格不要求物体变化时，不该因为物体没动挨罚")
        self.assertTrue(bool(((d >= 0) & (d <= 1)).all()), "缺口必须有界 [0,1]")
        # 刻度约掉：把两边同乘一个因子，结果不变
        for factor in (1e-6, 1e6):
            torch.testing.assert_close(effect_deficit(achieved * factor,
                                                      demand * factor), d)

    def test_effect_magnitude_separates_metres_and_radians(self):
        """P-70：两路各自归一化，不对 6 维取 L2。"""
        metric = torch.eye(6)[None]
        metric[0, 3:, 3:] *= 0.05 ** 2               # 半径 50 mm 的物体
        rigid = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])   # 转 1 rad
        got = effect_magnitude(rigid, torch.zeros(1, 4), metric=metric,
                               scale_rigid=torch.tensor([1.0]),
                               scale_state=torch.tensor([1.0]))
        self.assertAlmostEqual(float(got), 0.05, places=6)

    def test_static_effect_uses_a_physical_deadband(self):
        metric = torch.eye(6)[None]
        tiny = torch.tensor([[EFFECT_RIGID_DEADBAND_M * 0.1, 0, 0, 0, 0, 0]])
        got = effect_magnitude(tiny, torch.zeros(1, 1), metric=metric,
                               scale_rigid=torch.tensor([1e-9]),
                               scale_state=torch.tensor([1.0]))
        self.assertEqual(float(got), 0.0,
                         "微米级静态抖动不能被 tiny scale 放大成任务 effect")


class TestOnlineOfflineAgree(unittest.TestCase):
    """在线（torch）与离线（numpy）的 traction 必须是**同一个式子**，不是"差不多"。

    这条测试的由来值得记：第一版离线是"散射到 16384 个冻结表面点、再在格内按 |f|
    加权平均"，在线是"直接在接触点上求核和"。两者都自洽、都通过各自的单测，
    但 `tools/s6_reward_probe.py` 实测在线**系统性高 1.48~1.75 倍**
    （log 相关 0.93~0.98，是干净的偏差不是噪声）。后果：一条完美复现 source 的
    轨迹在线上读出 1.6 倍的 traction，落在指令盒外，`r_mech` 反过来惩罚正确行为。
    **而这件事不会报错。**

    D-79 因此把离线换成同一个式子。有了这条测试，两边再分家就会立刻变红。
    """

    def test_numpy_and_torch_traction_match_to_float_precision(self):
        rng = np.random.default_rng(0)
        n_contact, n_surface = 9, 6
        position = rng.normal(scale=0.01, size=(n_contact, 3))
        force = rng.normal(scale=2.0, size=(n_contact, 3))
        cell = rng.integers(0, n_surface, size=n_contact)

        online, _ = surface_traction(
            torch.as_tensor(position, dtype=torch.float64)[None],
            torch.as_tensor(force, dtype=torch.float64)[None],
            torch.ones(1, n_contact, dtype=torch.bool),
            torch.as_tensor(cell)[None], n_surface)

        from it.transfer import _GAUSS_NORM as OFFLINE_NORM
        self.assertAlmostEqual(OFFLINE_NORM, _GAUSS_NORM, places=12,
                               msg="两边的核归一化常数必须相同")
        delta = position[:, None, :] - position[None, :, :]
        kernel = np.exp(-(delta ** 2).sum(-1) / (2.0 * SCATTER_SIGMA ** 2)) * OFFLINE_NORM
        traction = kernel @ force
        mass = np.linalg.norm(force, axis=1)
        offline = np.zeros((n_surface, 3))
        for c in range(n_surface):
            rows = np.flatnonzero(cell == c)
            if len(rows):
                offline[c] = (mass[rows, None] * traction[rows]).sum(0) / mass[rows].sum()
        np.testing.assert_allclose(online[0].numpy(), offline, rtol=1e-9, atol=1e-9)


class TestCommandChannel(unittest.TestCase):
    def test_bank_loads_and_shapes_agree(self):
        bank = CommandBank(_artifacts())
        self.assertGreaterEqual(len(bank), 2)
        self.assertEqual(bank.n_surface, 256)
        index = torch.zeros(3, dtype=torch.long)
        bins = torch.tensor([0, 1, 2])
        cells = spatial_features(bank, index, bins)
        self.assertEqual(cells.shape, (3, bank.n_surface, SPATIAL_DIM))
        window = temporal_features(bank, index, bins)
        self.assertEqual(window.shape[0], 3)
        self.assertEqual(window.shape[2], TEMPORAL_DIM)
        self.assertTrue(torch.isfinite(cells).all() and torch.isfinite(window).all())

    def test_held_out_task_is_refused_at_the_dataloader(self):
        """`plan/04` §5.4：留出任务在**任何**训练阶段都不得出现，包括 curriculum
        与调试。这一条要写进 dataloader 的断言里，不能只靠自觉。"""
        paths = _artifacts()
        token = CommandBank([paths[0]]).ids[0].split("-")[0]
        with self.assertRaises(CommandError):
            CommandBank(paths, forbid=(token,))

    def test_field_masks_zero_the_block_and_flip_the_flag(self):
        """C0–C5 的差别只能是掩码，网络结构不许变（`plan/04` §7）。"""
        bank = CommandBank(_artifacts())
        index = torch.zeros(2, dtype=torch.long)
        bins = torch.zeros(2, dtype=torch.long)
        full = spatial_features(bank, index, bins)
        off = spatial_features(bank, index, bins,
                               enabled={name: False for name in FIELD_GROUPS})
        self.assertEqual(full.shape, off.shape, "掩码不能改变输入维度")
        self.assertTrue(torch.allclose(off[..., -len(FIELD_GROUPS):],
                                       torch.zeros_like(off[..., -len(FIELD_GROUPS):])))
        # 几何那 7 维（点/法向/面积）在任何条件下都保留
        self.assertTrue(torch.allclose(full[..., :7], off[..., :7]))
        self.assertFalse(torch.allclose(full[..., 7:-len(FIELD_GROUPS)],
                                        off[..., 7:-len(FIELD_GROUPS)]))

    def test_window_stalls_on_a_bin_that_wants_contact(self):
        """`plan/02` §5：接触没建立就不许前移。

        ⚠️ 只对**要求接触的格**成立。接近段那些格的 region 质量为零，本来就不该
        等接触——第一版测试从第 0 格测起，而第 0 格正是接近段，于是"推进了"被
        误读成 bug。这里先找出第一个真正要求接触的格再测。
        """
        bank = CommandBank(_artifacts())
        wants = bank.gather("region/mass/mean", torch.zeros(1, dtype=torch.long))[0].sum(-1)
        contact_bins = torch.nonzero(wants > 0, as_tuple=True)[0]
        if not len(contact_bins):
            self.skipTest("这份 artifact 没有要求接触的命令格")
        start = int(contact_bins[0])
        tracker = WindowTracker(bank, num_envs=2)
        tracker.reset(torch.arange(2), torch.zeros(2, dtype=torch.long))
        tracker.bin_index[:] = start
        for _ in range(tracker.max_dwell - 1):
            tracker.step(effect_increment=torch.full((2,), 10.0),
                         region_normal_force=torch.zeros(2))
        self.assertTrue(bool((tracker.bin_index == start).all()),
                        "没有建立接触就不该推进（plan/02 §5）")
        tracker.reset(torch.arange(2), torch.zeros(2, dtype=torch.long))
        tracker.bin_index[:] = start
        for _ in range(4):
            tracker.step(effect_increment=torch.full((2,), 10.0),
                         region_normal_force=torch.full((2,), 5.0))
        self.assertTrue(bool((tracker.bin_index > start).all()), "达成后必须推进")

    def test_window_reward_is_settled_before_advancing(self):
        """完成当前格的步不能因为先跳到下一格而重新得到满缺口。"""
        bank = CommandBank(_artifacts())
        tracker = WindowTracker(bank, num_envs=1, min_dwell=1)
        tracker.reset(torch.arange(1), torch.zeros(1, dtype=torch.long))
        demand = torch.zeros(1)
        for b in range(bank.n_bins):
            tracker.bin_index[:] = b
            demand = tracker.demand()
            if float(demand) > 1e-6:
                break
        else:
            self.skipTest("抽到的 artifact 没有非零 effect 命令格")
        old_bin = tracker.bin_index.clone()
        tracker.step(effect_increment=demand + 1.0,
                     region_normal_force=torch.full((1,), 10.0))
        self.assertEqual(float(tracker.last_deficit), 0.0)
        self.assertGreater(int(tracker.bin_index[0]), int(old_bin[0]))

    def test_window_demand_reads_the_artifact_field(self):
        """D-91 不能只把字段写进 schema；真实 tracker 必须消费它。"""
        bank = CommandBank(_artifacts())
        sentinel = torch.linspace(0.125, 3.875, bank.n_bins)
        bank._stack["effect/bin_demand"][0] = sentinel
        tracker = WindowTracker(bank, num_envs=1)
        tracker.reset(torch.arange(1), torch.zeros(1, dtype=torch.long))
        for b in (0, bank.n_bins // 2, bank.n_bins - 1):
            tracker.bin_index[:] = b
            self.assertAlmostEqual(float(tracker.demand()), float(sentinel[b]), places=6)

    def test_static_press_has_zero_rigid_demand_after_deadband(self):
        """静压的 1e-7 m 级抖动不能变成几十倍任务 effect。"""
        root = Path(os.environ.get("IT_PROBE_DIR", "/tmp/s6/probe"))
        paths = sorted(root.glob("probe_block-block-nominal-press-train.npz"))
        if not paths:
            self.skipTest(f"{root} 下没有 block/press artifact")
        bank = CommandBank(paths)
        demand = bank._stack["effect/bin_demand"][0]
        self.assertEqual(float(demand.abs().max()), 0.0)

    def test_approach_bins_do_not_wait_for_contact(self):
        """反过来的一半：不要求接触的格若也等接触，接近段就只能靠超时爬过去。"""
        bank = CommandBank(_artifacts())
        wants = bank.gather("region/mass/mean", torch.zeros(1, dtype=torch.long))[0].sum(-1)
        free_bins = torch.nonzero(wants <= 0, as_tuple=True)[0]
        if not len(free_bins):
            self.skipTest("这份 artifact 每一格都要求接触")
        tracker = WindowTracker(bank, num_envs=1)
        tracker.reset(torch.arange(1), torch.zeros(1, dtype=torch.long))
        tracker.bin_index[:] = int(free_bins[0])
        for _ in range(tracker.min_dwell + 1):
            tracker.step(effect_increment=torch.zeros(1),
                         region_normal_force=torch.zeros(1))
        self.assertGreater(int(tracker.bin_index[0]), int(free_bins[0]))
        self.assertLess(tracker.dwell.max().item(), tracker.max_dwell,
                        "不该是靠超时爬过去的")

    def test_window_times_out_and_never_runs_off_the_end(self):
        bank = CommandBank(_artifacts())
        tracker = WindowTracker(bank, num_envs=2)
        tracker.reset(torch.arange(2), torch.zeros(2, dtype=torch.long))
        for _ in range(tracker.max_dwell * (bank.n_bins + 3)):
            tracker.step(effect_increment=torch.zeros(2),
                         region_normal_force=torch.zeros(2))
        self.assertTrue(bool((tracker.bin_index <= bank.n_bins - 1).all()))
        self.assertTrue(bool(tracker.finished.all()), "跑完应当标记 finished 而不是越界")


class TestPolicy(unittest.TestCase):
    def _policy(self, privileged: int = 0) -> InteractionPolicy:
        torch.manual_seed(0)
        return InteractionPolicy(proprio_dim=32, action_dim=7, privileged_dim=privileged)

    def test_information_conditions_share_the_exact_parameter_count(self):
        """把性能差异归因到**信息**而非容量的前提（`plan/04` §7）。

        条件之间只有输入里的掩码不同，网络本身逐位相同——所以这个测试查的是
        "掩码没有偷偷改变结构"。
        """
        a, b = self._policy(), self._policy()
        self.assertEqual(sum(p.numel() for p in a.parameters()),
                         sum(p.numel() for p in b.parameters()))

    def test_forward_shapes_and_privileged_critic(self):
        policy = self._policy(privileged=11)
        n, s, h = 5, 256, 8
        cells = torch.randn(n, s, SPATIAL_DIM)
        weight = torch.rand(n, s)
        window = torch.randn(n, h, TEMPORAL_DIM)
        proprio = torch.randn(n, 32)
        dist = policy.act(cells, weight, window, proprio)
        self.assertEqual(dist.mean.shape, (n, 7))
        value = policy.value(cells, weight, window, proprio, torch.randn(n, 11))
        # (n, 1)：rsl_rl 的 RolloutStorage 把 values 存成 (T, N, 1)，
        # 返回 (n,) 会在 time-out bootstrap 那一步被广播成 (N, N)。
        self.assertEqual(value.shape, (n, 1))
        with self.assertRaises(ValueError):
            policy.value(cells, weight, window, proprio)

    def test_actor_actually_reads_the_command(self):
        """`plan/04` §13 第 6 条：**Actor 是否真正读取 desired 字段，要查不要猜。**

        这里只查最弱的版本——换一份指令，动作分布必须变。真正的判据是实验五的
        matched counterfactual，那要等训练完。
        """
        policy = self._policy()
        n, s, h = 4, 256, 8
        weight, proprio = torch.rand(n, s), torch.randn(n, 32)
        window = torch.randn(n, h, TEMPORAL_DIM)
        a = policy.act(torch.randn(n, s, SPATIAL_DIM), weight, window, proprio).mean
        b = policy.act(torch.randn(n, s, SPATIAL_DIM), weight, window, proprio).mean
        self.assertGreater(float((a - b).abs().max()), 1e-6)

    def test_continuous_input_normalization_handles_physical_unit_ranges(self):
        """Pa 级 command 与米级 proprio 同时输入时不能让网络数值失控。"""
        policy = self._policy()
        n, s, h = 4, 256, 8
        cells = torch.randn(n, s, SPATIAL_DIM)
        cells[..., 20:29] *= 1e5       # traction 的真实数量级
        window = torch.randn(n, h, TEMPORAL_DIM)
        window[..., 18:22] *= 1e5
        proprio = torch.randn(n, 32)
        dist = policy.act(cells, torch.rand(n, s), window, proprio)
        loss = dist.mean.square().mean()
        loss.backward()
        self.assertTrue(bool(torch.isfinite(dist.mean).all()))
        self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all())
                            for p in policy.parameters()))


class TestActorCriticWrapper(unittest.TestCase):
    """rsl_rl 外壳：观测里只放下标，特征现取（省 rollout buffer 的几个 GB）。"""

    def _bank(self):
        return CommandBank(_artifacts())

    @staticmethod
    def _make(bank, *, proprio: int, privileged: int = 0, **kw):
        """按 rsl_rl 3.0.1 的构造签名装配：``(obs, obs_groups, num_actions, **cfg)``。

        维度由 obs 推，所以这里给一份**形状正确**的观测模板就够了——`OnPolicyRunner`
        也正是拿 `env.get_observations()` 的第一帧来做这件事。
        """
        policy = proprio + InteractionActorCritic.INDEX_FIELDS
        obs = {"policy": torch.zeros(1, policy),
               "critic": torch.zeros(1, policy + privileged)}
        groups = {"policy": ["policy"], "critic": ["critic"]}
        return InteractionActorCritic(obs, groups, 9, bank=bank, **kw)

    def test_rejects_observation_normalization(self):
        """观测里有整数下标，归一化会把指令通道抹掉——必须在构造时就拦。"""
        with self.assertRaises(ValueError):
            self._make(self._bank(), proprio=45, actor_obs_normalization=True)

    def test_roundtrip_shapes_and_rotation(self):
        bank = self._bank()
        ac = self._make(bank, proprio=45, privileged=13)
        n = 6
        proprio = torch.randn(n, 45)
        command = torch.randint(0, len(bank), (n, 1)).float()
        bins = torch.randint(0, bank.n_bins, (n, 1)).float()
        quat = torch.nn.functional.normalize(torch.randn(n, 4), dim=-1)
        obs = torch.cat([proprio, command, bins, quat], dim=-1)
        self.assertEqual(ac.act(obs).shape, (n, 9))
        critic_obs = torch.cat([obs, torch.randn(n, 13)], dim=-1)
        # (n, 1) 不是 (n,)：rsl_rl 的 storage 与 time-out bootstrap 都按 (T, N, 1) 走
        self.assertEqual(ac.evaluate(critic_obs).shape, (n, 1))
        self.assertEqual(ac.get_actions_log_prob(ac.act(obs)).shape, (n,))

    def test_command_fields_land_in_the_world_frame(self):
        """指令的表面点与向量场必须被转到**世界系**，不是别的什么系。⭐

        artifact 里的一切都在物体系；执行器活在世界系。观测里带物体四元数就是为了
        让网络把两边对上。`_quat_to_matrix(q_obj)` 已经是 ``R_obj→world``——
        再转置一次得到的是 ``R_world→obj``，拿它去乘物体系的数组，结果落在一个
        **不存在的参考系**里。

        这一条在旧实现上是红的，而三条已有的 wrapper 测试全都测不出来：
        它们用的随机四元数只检查**形状**，而真正会暴露的场景（物体转了个角度）
        在方块平放时 ``q≈(1,0,0,0)``、``R=Rᵀ=I``，两种写法逐位相同。
        P-53 / P-54 那三次也都是"逐帧数值、接触部位、单元测试全部正常，只有结论错"。
        """
        from it.ei_policy import _quat_to_matrix

        bank = self._bank()
        ac = self._make(bank, proprio=8)
        half = float(np.pi / 4)                       # 绕 z 转 90°
        quat = torch.tensor([[np.cos(half), 0.0, 0.0, np.sin(half)]], dtype=torch.float32)
        obs = torch.cat([torch.zeros(1, 8), torch.zeros(1, 1), torch.zeros(1, 1), quat],
                        dim=-1)
        cells = ac._unpack(obs)[0]
        points_obj = bank.gather("surface/points_obj", torch.zeros(1, dtype=torch.long))
        expect = torch.einsum("nij,nsj->nsi", _quat_to_matrix(quat), points_obj)
        # 绕 z 转 90°：物体系的 +x 必须变成世界系的 +y。转置版会给出 −y。
        self.assertLess(float((cells[..., 0:3] - expect).abs().max()), 1e-5,
                        "表面点没有被转到世界系——检查 _unpack 里的 rotation")
        # 法向与 engage 方向必须**跟着一起转**（P-53/P-54：整组一起转，不能只转一部分）
        normals_obj = bank.gather("surface/normals_obj", torch.zeros(1, dtype=torch.long))
        expect_n = torch.einsum("nij,nsj->nsi", _quat_to_matrix(quat), normals_obj)
        self.assertLess(float((cells[..., 3:6] - expect_n).abs().max()), 1e-5,
                        "法向没有和表面点用同一个旋转")

    def test_conditions_differ_only_by_mask_not_by_parameter_count(self):
        """C0（只有 effect）与 C4（全字段）必须是同一个网络（`plan/04` §7）。"""
        bank = self._bank()
        torch.manual_seed(0)
        c4 = self._make(bank, proprio=45)
        torch.manual_seed(0)
        c0 = self._make(bank, proprio=45,
                        enabled={g: False for g in FIELD_GROUPS})
        self.assertEqual(sum(p.numel() for p in c4.parameters()),
                         sum(p.numel() for p in c0.parameters()))
        n = 4
        obs = torch.cat([torch.randn(n, 45),
                         torch.zeros(n, 1), torch.zeros(n, 1),
                         torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1)], dim=-1)
        # 同样的权重、同样的观测，只是掩码不同 -> 输出必须不同，否则掩码没接上
        self.assertGreater(float((c4.act_inference(obs) - c0.act_inference(obs)
                                  ).abs().max()), 1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
