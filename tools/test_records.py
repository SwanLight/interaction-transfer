"""`it.records` 数据契约的本地单元测试。

**故意不依赖 Isaac Sim**，用系统 python 就能跑，改完 records.py 立刻验::

    PYTHONPATH=src python3 tools/test_records.py
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from it.records import (
    SCHEMA_VERSION,
    EpisodeRecord,
    RecordError,
    apply_splits,
    load_episode,
    read_manifest,
    save_episode,
    split_episode_entries,
    write_manifest,
)


class RecordContractTests(unittest.TestCase):
    def _record(self, episode_id: str = "ep-000") -> EpisodeRecord:
        return EpisodeRecord(
            meta={
                "schema_version": SCHEMA_VERSION,
                "episode_id": episode_id,
                "task": "drawer",
                "strategy_family": "centered_symmetric",
                "strategy_variant": "left",
                "success": True,
                "physics": {"drawer_damping": 30.0},
            },
            arrays={
                "phase": np.array([0, 1, 2], dtype=np.int8),
                "progress": np.array([0.0, 0.5, 1.0], dtype=np.float32),
                "valid_frame": np.array([True, True, False], dtype=np.bool_),
                "object/drawer_joint_pos": np.zeros((3, 1), dtype=np.float32),
                "contact/plate0/positions": np.zeros((3, 4, 3), dtype=np.float32),
                "contact/plate0/valid": np.zeros((3, 4), dtype=np.bool_),
                "source/plate0/root_pose": np.zeros((3, 7), dtype=np.float32),
                "source/action": np.zeros((3, 12), dtype=np.float32),
            },
        )

    def test_落盘读回不变且审计字段被隔开(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episodes" / "ep-000.npz"
            record = self._record()
            loaded = load_episode(save_episode(record, path))
            self.assertEqual(loaded.meta, record.meta)
            self.assertEqual(set(loaded.arrays), set(record.arrays))
            for name in record.arrays:
                np.testing.assert_array_equal(loaded.arrays[name], record.arrays[name])
            model = loaded.model_arrays()
            self.assertNotIn("source/action", model)          # 不许泄漏
            self.assertNotIn("source/plate0/root_pose", model)
            self.assertIn("object/drawer_joint_pos", model)
            self.assertIn("contact/plate0/positions", model)

    def test_未归类字段必须报错而不是放行(self) -> None:
        record = self._record()
        record.arrays["robot/joint_pos"] = np.zeros((3, 1), dtype=np.float32)
        with self.assertRaises(RecordError):
            record.model_arrays()

    def test_帧数对不上要拒绝(self) -> None:
        record = self._record()
        record.arrays["object/bad"] = np.zeros((2, 1), dtype=np.float32)
        with self.assertRaisesRegex(RecordError, "2 帧"):
            record.validate()

    def test_manifest_哈希_相对路径_划分(self) -> None:
        families = ["centered", "offset", "asymmetric"] * 3 + ["centered"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pairs = []
            for i, family in enumerate(families):
                record = self._record(f"ep-{i:03d}")
                record.meta["strategy_family"] = family
                episode_path = root / "episodes" / f"ep-{i:03d}.npz"
                save_episode(record, episode_path)
                pairs.append((str(episode_path), record))
            entries = [r.to_manifest_entry(str(Path(p).relative_to(root)), sha256="unused")
                       for p, r in pairs]
            splits = split_episode_entries(entries, seed=17,
                                           holdout_strategy_family="asymmetric")
            manifest_path = root / "manifest.json"
            write_manifest(pairs, manifest_path, dataset_name="unit",
                           generator_git_sha="test", splits=splits)
            loaded = read_manifest(manifest_path)
            self.assertEqual(loaded["num_episodes"], 10)
            self.assertEqual(loaded["generator_git_sha"], "test")
            self.assertTrue(all(e["sha256"] for e in loaded["episodes"]))
            self.assertTrue(all(e["split"] == "unseen_strategy_test"
                                for e in loaded["episodes"]
                                if e["strategy_family"] == "asymmetric"))

    def test_划分确定且按_episode(self) -> None:
        entries = [{"episode_id": f"ep-{i}", "strategy_family": "a" if i < 2 else "b"}
                   for i in range(10)]
        first = split_episode_entries(entries, seed=3, holdout_strategy_family="b")
        second = split_episode_entries(entries, seed=3, holdout_strategy_family="b")
        self.assertEqual(first, second)
        self.assertEqual(set(first["unseen_strategy_test"]), {f"ep-{i}" for i in range(2, 10)})
        self.assertEqual(set().union(*map(set, first.values())),
                         {e["episode_id"] for e in entries})

    def test_物理留出集只收_physics_variant_非_nominal(self) -> None:
        """2026-08-28 修的真 bug：旧实现用随机 shuffle 填 unseen_physics_test，
        于是「没见过的物理」里装的是同分布 episode，泛化数字是假的。"""
        entries = [{"episode_id": f"ep-{i}", "strategy_family": "a",
                    "physics_variant": "high_friction" if i < 3 else "nominal"}
                   for i in range(12)]
        s = split_episode_entries(entries, seed=1)
        self.assertEqual(set(s["unseen_physics_test"]), {"ep-0", "ep-1", "ep-2"})
        # 没有变体时该集合必须为空，不许拿随机 episode 顶替
        plain = [{"episode_id": f"ep-{i}", "strategy_family": "a"} for i in range(12)]
        self.assertEqual(split_episode_entries(plain, seed=1)["unseen_physics_test"], [])

    def test_manifest_条目数对不上要拒绝(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps({
                "manifest_version": "s3-manifest-v1",
                "schema_version": SCHEMA_VERSION,
                "num_episodes": 1, "episodes": [],
            }))
            with self.assertRaises(RecordError):
                read_manifest(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
