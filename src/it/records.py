"""S3 source 数据集的 episode 记录格式（带版本号）。

一个 episode = 一个压缩 `.npz`；元数据以 UTF-8 JSON 存在压缩包内部的
`__meta__` 里，所以单个文件可以随便搬，不依赖旁边的 manifest。

**文件里有意混着两层数据，用前缀区分**：

- ``source/*``：采集侧的东西（板的位姿、PD 目标、脚本动作）。
  **只用于事后追查，永远不许进模型输入。** `plan/02` §7 第 2 条要求
  表示里不含 source root state；这里靠前缀在数据层就把它隔开。
- ``object/*`` / ``contact/*``：物体侧的物理观测，S4 从这些构造
  Oracle Interaction Record。

`model_arrays()` 是 fail-closed 的：碰到没有归类的前缀直接抛异常，
而不是默默放行——新加一个审计字段时，忘记归类会立刻炸，不会悄悄泄漏。

划分只在 manifest 里做，且**只按 episode，不按帧**（P-10）。
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

SCHEMA_VERSION = "s3-episode-v1"
#: S4 的 Oracle Interaction Record（`it.interaction`）。同一套读写与划分逻辑，
#: 只是允许的字段前缀不同——它已经是物体中心的表示，不再有 `contact/<板>/` 这层。
IR_SCHEMA_VERSION = "s4-record-v1"
MANIFEST_VERSION = "s3-manifest-v1"
META_KEY = "__meta__"

#: 前缀白名单故意写死。宁可 loader 报错，也不要把审计字段当特征喂进去。
SOURCE_PREFIX = "source/"
MODEL_INPUT_PREFIXES = ("object/", "contact/", "phase", "progress", "valid_")

#: 每个 schema：(允许进模型输入的前缀, 只作追查/诊断而被丢弃的前缀)。
#: 两个集合都写出来，是为了让"新加字段忘了归类"落到 `model_arrays` 的异常上，
#: 而不是被默默放行——S3 那条 fail-closed 规则对 S4 同样适用。
SCHEMA_PREFIXES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    SCHEMA_VERSION: (MODEL_INPUT_PREFIXES, (SOURCE_PREFIX,)),
    IR_SCHEMA_VERSION: (
        ("effect/", "region/", "engage/", "mode/", "mech/",
         "phase", "progress", "valid_"),
        (SOURCE_PREFIX, "aux/"),
    ),
}


class RecordError(ValueError):
    """episode 违反数据契约时抛出。"""


@dataclass
class EpisodeRecord:
    """一条定频 episode + 它的 JSON 可序列化元数据。"""

    meta: dict[str, Any]
    arrays: dict[str, np.ndarray]

    def validate(self) -> None:
        if not isinstance(self.meta, dict):
            raise RecordError("meta 必须是 dict")
        if self.meta.get("schema_version") not in SCHEMA_PREFIXES:
            raise RecordError(
                f"schema_version 必须是 {sorted(SCHEMA_PREFIXES)} 之一，"
                f"实际是 {self.meta.get('schema_version')!r}"
            )
        for key in ("episode_id", "task", "strategy_family"):
            if not self.meta.get(key):
                raise RecordError(f"meta.{key} 是必填项")
        if not self.arrays:
            raise RecordError("episode 至少要有一个逐帧数组")

        n_frames: int | None = None
        for name, value in self.arrays.items():
            if name == META_KEY:
                raise RecordError(f"{META_KEY!r} 是保留名")
            if not isinstance(name, str) or not name:
                raise RecordError("数组名必须是非空字符串")
            arr = np.asarray(value)
            if arr.ndim == 0:
                raise RecordError(f"数组 {name!r} 必须有帧维度")
            if n_frames is None:
                n_frames = int(arr.shape[0])
            elif int(arr.shape[0]) != n_frames:
                raise RecordError(
                    f"数组 {name!r} 有 {arr.shape[0]} 帧，期望 {n_frames}"
                )
            if arr.dtype == object:
                raise RecordError(f"数组 {name!r} 不能是 object dtype")
        if n_frames is None or n_frames < 1:
            raise RecordError("episode 至少要有 1 帧")
        if "valid_frame" in self.arrays:
            valid = np.asarray(self.arrays["valid_frame"])
            if valid.ndim != 1 or valid.dtype != np.bool_:
                raise RecordError("valid_frame 必须是一维 bool 数组")
        for key in ("phase", "progress"):
            if key in self.arrays and np.asarray(self.arrays[key]).ndim != 1:
                raise RecordError(f"{key} 必须是一维")

    @property
    def num_frames(self) -> int:
        self.validate()
        first = next(iter(self.arrays.values()))
        return int(np.asarray(first).shape[0])

    def model_arrays(self) -> dict[str, np.ndarray]:
        """返回允许进入下游交互管线的数组。

        这是数据层的护栏，**不替代** dataloader 里的断言（P-12 要求两边都有）。
        ``source/*`` 按构造排除；未知前缀直接报错，防止新字段静默泄漏。
        """
        self.validate()
        allowed, dropped = SCHEMA_PREFIXES[self.meta["schema_version"]]
        result: dict[str, np.ndarray] = {}
        for name, value in self.arrays.items():
            if name.startswith(dropped):
                continue
            if name.startswith(allowed):
                result[name] = value
            else:
                raise RecordError(
                    f"数组 {name!r} 没有归类：要么改名到被丢弃的前缀 {dropped}，"
                    f"要么加进 SCHEMA_PREFIXES[{self.meta['schema_version']!r}]，不许含糊"
                )
        return result

    def to_manifest_entry(self, path: str, sha256: str | None = None) -> dict[str, Any]:
        self.validate()
        entry = {
            "episode_id": self.meta["episode_id"],
            "path": path,
            "num_frames": self.num_frames,
            "task": self.meta["task"],
            "strategy_family": self.meta["strategy_family"],
            "strategy_variant": self.meta.get("strategy_variant", "default"),
            "physics_variant": self.meta.get("physics_variant", "nominal"),
            # `plan/03` §7 的另外两条划分依据。它们必须**提到条目层**，
            # 因为 `split_episode_entries` 只看条目、不下钻 meta。
            "geometry_variant": self.meta.get("geometry_variant", "nominal"),
            "implementation": self.meta.get("implementation", "default"),
            "success": bool(self.meta.get("success", False)),
            "split": self.meta.get("split", "unassigned"),
            "meta": self.meta,
        }
        if sha256 is not None:
            entry["sha256"] = sha256
        return entry


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"无法 JSON 序列化：{type(value).__name__}")


def _json_text(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default)


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def save_episode(record: EpisodeRecord, path: str | os.PathLike[str]) -> str:
    """校验并落盘一条 episode，返回绝对路径。"""
    record.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix != ".npz":
        raise RecordError(f"episode 路径必须以 .npz 结尾，实际是 {destination}")
    payload: dict[str, Any] = {META_KEY: np.asarray(_json_text(record.meta))}
    payload.update({name: np.asarray(value) for name, value in record.arrays.items()})
    # 先写临时名再原子替换：Isaac 进程被 kill 时（P-19 经常要 kill）
    # 不能留下一个「manifest 指着半个文件」的数据集。
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, destination)
    return str(destination.resolve())


def load_episode(path: str | os.PathLike[str], *, validate: bool = True) -> EpisodeRecord:
    """读一条 episode。``allow_pickle=False``，不接受 object 数组。"""
    with np.load(path, allow_pickle=False) as data:
        if META_KEY not in data.files:
            raise RecordError(f"{path} 没有 {META_KEY} 元数据")
        raw = data[META_KEY]
        if raw.ndim != 0:
            raise RecordError(f"{path} 的元数据不是标量 JSON")
        meta = json.loads(str(raw.item()))
        arrays = {n: np.array(data[n], copy=True) for n in data.files if n != META_KEY}
    record = EpisodeRecord(meta=meta, arrays=arrays)
    if validate:
        record.validate()
    return record


def write_manifest(
    records: Iterable[tuple[str, EpisodeRecord]],
    path: str | os.PathLike[str],
    *,
    dataset_name: str,
    generator_git_sha: str = "unknown",
    extra: Mapping[str, Any] | None = None,
    splits: Mapping[str, Iterable[str]] | None = None,
) -> str:
    """写 manifest。``records`` 是 ``(路径, record)`` 对的可迭代对象。

    路径按 manifest 所在目录存相对路径，整个数据集目录可以整体搬走。
    每条都记 SHA-256，事后能验证文件没被改过。
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    schema = SCHEMA_VERSION
    for episode_path, record in records:
        record.validate()
        schema = record.meta["schema_version"]
        p = Path(episode_path).resolve()
        try:
            rel = os.path.relpath(p, destination.parent.resolve())
        except ValueError as exc:
            raise RecordError(f"episode 和 manifest 不在同一个盘上：{p}") from exc
        entries.append(record.to_manifest_entry(rel, sha256=sha256_file(p)))
    entries.sort(key=lambda item: str(item["episode_id"]))
    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "schema_version": schema,
        "dataset_name": dataset_name,
        "generator_git_sha": generator_git_sha,
        "num_episodes": len(entries),
        "episodes": entries,
    }
    if extra:
        manifest.update(dict(extra))
    if splits is not None:
        manifest = apply_splits(manifest, splits)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return str(destination.resolve())


def read_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    """读 manifest 并做结构校验。"""
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise RecordError("manifest_version 缺失或不支持")
    if manifest.get("schema_version") not in SCHEMA_PREFIXES:
        raise RecordError(
            f"manifest 的 schema_version {manifest.get('schema_version')!r} 不认识")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list):
        raise RecordError("manifest.episodes 必须是 list")
    if manifest.get("num_episodes") != len(episodes):
        raise RecordError("num_episodes 与实际条目数对不上")
    ids = [entry.get("episode_id") for entry in episodes]
    if any(not item for item in ids) or len(set(ids)) != len(ids):
        raise RecordError("episode_id 必须非空且唯一")
    return manifest


def split_episode_entries(
    entries: list[dict[str, Any]],
    *,
    seed: int = 0,
    holdout_strategy_family: str | None = None,
    holdout_implementation: str | None = None,
) -> dict[str, list[str]]:
    """确定性地按 **episode** 分配划分（P-10：绝不按帧）。

    `plan/03` §7 要一个校准集 + **五个**冻结测试集，另加一个 ``failed`` 桶。
    五条不靠随机抽、由元数据决定的规则，**按以下顺序**生效：

    1. ``failed``：``success == False`` 的全部。
       `plan/03` §6 要求"失败、近成功和物理不可行样本单独保存，
       **不与成功 shared-structure 标签混合**"。它们仍然留在数据集里可供检查，
       但不进训练集、不进任何测试集——**一条失败的示教不是示教**，
       把它放进 `unseen_strategy_test` 会让那个集合的泛化数字失去意义。
    2. ``unseen_implementation_test``：``implementation == holdout_implementation``
       的成功样本。`plan/03` §7 末尾要求擦拭"两种实现中至少留出一种实现的
       **全部** episode 作跨实现测试"，对应 `02` §7 第 8 条那条泄漏检查
       （envelope 与是否使用工具无关）。它排在策略留出**之前**——
       "实现"是比"策略家族"更粗的一层，一种实现里可能有多个家族。
    3. ``unseen_strategy_test``：``strategy_family == holdout_strategy_family`` 的成功样本。
    4. ``unseen_geometry_test``：``geometry_variant != "nominal"`` 的其余成功样本
       （`plan/03` §7 表第 6 行"小幅几何变化"：销钉偏心距 / 把手高度 / 黑板擦尺寸）。
    5. ``unseen_physics_test``：``physics_variant != "nominal"`` 的其余成功样本。

    ⚠️ 第 3 条是 2026-08-28 修的一个真 bug：原实现用随机 shuffle 填它，
    于是这个名字叫「没有见过的物理」的集合里装的其实是同分布 episode，
    **跑出来的泛化数字会是假的，而且从结果上看不出来**。
    没有对应元数据时该集合就是**空的**，不拿随机 episode 顶替。
    """
    if not entries:
        raise RecordError("不能划分空的 episode 列表")
    ids = [str(entry["episode_id"]) for entry in entries]
    if len(set(ids)) != len(ids):
        raise RecordError("划分前 episode_id 必须唯一")

    by_id = {str(e["episode_id"]): e for e in entries}
    failed = [i for i in ids if not by_id[i].get("success", False)]
    taken = set(failed)

    holdout_impl = [
        i for i in ids
        if i not in taken and holdout_implementation is not None
        and by_id[i].get("implementation") == holdout_implementation
    ]
    taken |= set(holdout_impl)
    holdout_strategy = [
        i for i in ids
        if i not in taken and holdout_strategy_family is not None
        and by_id[i].get("strategy_family") == holdout_strategy_family
    ]
    taken |= set(holdout_strategy)
    holdout_geom = [
        i for i in ids
        if i not in taken and by_id[i].get("geometry_variant", "nominal") != "nominal"
    ]
    taken |= set(holdout_geom)
    holdout_physics = [
        i for i in ids
        if i not in taken and by_id[i].get("physics_variant", "nominal") != "nominal"
    ]
    taken |= set(holdout_physics)

    remaining = [i for i in ids if i not in taken]
    rng = np.random.default_rng(seed)
    rng.shuffle(remaining)
    n = len(remaining)
    n_cal = max(1, round(n * 0.10)) if n >= 4 else 0
    n_test = max(1, round(n * 0.15)) if n >= 4 else 0
    # 保证各子集不重叠，且训练集至少留 1 条
    while n_cal + n_test >= max(n, 1):
        if n_test:
            n_test -= 1
        else:
            n_cal = max(0, n_cal - 1)

    n_train = max(n - n_cal - n_test, 0)
    result = {
        "train": remaining[:n_train],
        "calibration": remaining[n_train: n_train + n_cal],
        "in_distribution_test": remaining[n_train + n_cal: n_train + n_cal + n_test],
        "unseen_physics_test": holdout_physics,
        "unseen_geometry_test": holdout_geom,
        "unseen_strategy_test": holdout_strategy,
        "unseen_implementation_test": holdout_impl,
        "failed": failed,
    }
    assigned = [item for values in result.values() for item in values]
    if sorted(assigned) != sorted(ids):
        raise RecordError("划分过程丢了或重复了 episode")
    return result


def apply_splits(manifest: dict[str, Any], splits: Mapping[str, Iterable[str]]) -> dict[str, Any]:
    """返回打好 split 标签的 manifest 副本。"""
    result = json.loads(json.dumps(manifest))
    by_id = {
        str(episode_id): split
        for split, ids_for_split in splits.items()
        for episode_id in ids_for_split
    }
    if set(by_id) != {str(item["episode_id"]) for item in result["episodes"]}:
        raise RecordError("splits 必须不重不漏地覆盖 manifest 里每一条 episode")
    for entry in result["episodes"]:
        split = by_id[str(entry["episode_id"])]
        entry["split"] = split
        entry.setdefault("meta", {})["split"] = split
    result["splits"] = {name: list(ids) for name, ids in splits.items()}
    return result
