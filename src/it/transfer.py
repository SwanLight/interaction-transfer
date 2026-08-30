"""把同一任务的多条 S4 示教聚合成可传递的 interaction command。

这个模块实现项目 S5 的第一条、最保守的基线链路：

``多条成功 S4 record -> 相分段活动量对齐 -> 每条 episode 等权统计 -> InteractionTransfer``

它不推断任务语义，不声称恢复了任务的必要/充分条件，也不替执行器判断形态可行性。
输出只是示教中实际出现过的物体中心交互统计；不同 embodiment 各自训练的 E-I
executor 负责把这些物理字段解码成自己的 action。

采集脚本的 ``phase`` / ``progress`` 只用于**离线对齐**，不进入产物（D-55：标签
可以用，观测不行）。产物中的时间轴是一个没有任务语义的 command sequence index。
任务名仅留在 meta 供审计，``executor_arrays()`` 永远不返回 meta。

v1 -> v2 改了什么（D-65 / D-66，实测依据见 ``out/s5_align/probe.txt``）
--------------------------------------------------------------------

**一、对齐轴换成"相分段 + 活动量"。** v1 按全局 ``progress`` 分格，而 ``progress``
是任务 effect 完成度：接近段全是 0、到达目标之后全是 1。实测 12 条真实抽屉记录
**84% 的帧落进 32 格里的 2 格**，中间 30 格每格中位数 2 帧；旋钮 73%。所有既有
校验照常通过，``support/episodes`` 还是满的——这是一条典型的静默失败（P-58）。

v2 先按 ``phase`` 把 episode 切成四段（这正是 `plan/03` §8 第 1 条写的"按 phase
**和** object progress 对齐"里被 v1 漏掉的那一半），格数按各段的**交互活动量占比**
分配（每段保底 ``PHASE_FLOOR`` 格），段内也按累计活动量分格。活动量 = 接触冲量率
+ effect 变化率，两路**各自在 episode 内归一化**后相加，因此不含任务分支，也不需要
在牛顿与米之间拍权重（D-31 的第 2 个洞）。

**二、每个 cell 的所有字段来自同一组 episode 和同一批帧。** v1 的 ``region`` /
``engage`` / ``mode`` 用 NaN 表示"这条 episode 没碰这里"（统计时被排除），而
``traction`` 用 0（统计时被算进去）。于是同一个 cell 上，方向是"碰过的人的平均"，
而力是"所有人的平均、被没碰的人稀释"。实测抽屉 10%、擦拭 15% 的 occupied cell 因此
拿到 ``[q10,q90] = [0,0]`` 的力带——命令同时说"这里该接触"和"这里的力必须为零"。
v2 把 ``traction`` / ``moment_density`` 也改成**接触条件化**，并新增
``region/support``（多少条 episode 碰过）与 ``region/duty``（碰过的人里，这个格内
有多大比例的帧在接触），让下游能分辨 1/12 与 12/12 的支持度。

**三、``engage`` 的集中度不再被丢掉。** 各 episode 的单位方向求平均后，合矢量长度
就是 ∈[0,1] 的集中度，正是方向多峰性的直接度量。v1 把它阈值化成一个 bool 就扔了。

**四、诊断措辞更正。** v1 报的"6D wrench 重建误差 6e-6 N"在代数上是恒等式
（cell 内先算局部力矩再求和 = 直接对接触点求力矩），那个数只反映浮点精度。真正有
信息量的是**表面投影完整性**：被 ``on_surface`` / ``weight>0`` 滤掉的接触占了多少
力，以及由此与 S4 独立记的 ``mech/wrench_obj`` 差多少。v2 按后者报，并额外报被丢掉
的力占比。跨 episode 取中位数/分位数之后守恒性**不再成立**，那是任何统计聚合的
固有性质，不得再声称"积分恢复 6D wrench"。
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from it.records import EpisodeRecord, META_KEY
from it.surfaces import Surface, surface_for

TRANSFER_SCHEMA_VERSION = "interaction-transfer-v2"
MODE_COUNT = 4
N_PHASES = 4
#: 每个**有帧**的 phase 至少分到的命令格数。松开段的活动量恰好是 0，若按活动量
#: 判"占用"，它会分到 0 格、那一段的帧被整段丢掉而不报错。
PHASE_FLOOR = 2

# E-I 输入契约是 exact allowlist，不靠前缀猜测。新增字段必须显式评审。
EXECUTOR_ARRAYS = frozenset({
    "command/fraction",
    "command/valid",
    "support/episodes",
    "surface/points_obj",
    "surface/normals_obj",
    "surface/area",
    "surface/part",
    "effect/rigid/median",
    "effect/rigid/q10",
    "effect/rigid/q90",
    "effect/rigid/valid_count",
    "effect/surface_state/median",
    "effect/surface_state/q10",
    "effect/surface_state/q90",
    "effect/surface_state/valid_count",
    "region/mass/mean",
    "region/mass/q10",
    "region/mass/q90",
    "region/support",
    "region/duty",
    "engage/dir/mean",
    "engage/concentration",
    "engage/valid",
    "mode/prob",
    "mech/traction_obj/median",
    "mech/traction_obj/q10",
    "mech/traction_obj/q90",
    "mech/moment_density_obj/median",
    "mech/moment_density_obj/q10",
    "mech/moment_density_obj/q90",
})

#: 每个 cell 上"只在接触过的 episode 之间统计"的字段。它们必须共用同一个
#: ``contact_frames > 0`` 掩码，否则同一个 cell 上不同字段来自不同的 episode 子集
#: （v1 就是这么错的）。
CONTACT_CONDITIONED = ("engage", "mode", "traction", "moment_density")


class TransferError(ValueError):
    """S5 interaction transfer 违反数据契约。"""


# ---------------------------------------------------------------- 对齐

def activity_rate(record: EpisodeRecord, surface: Surface) -> np.ndarray:
    """逐帧交互活动量：接触冲量率 + effect 变化率，两路各自在 episode 内归一化。

    两路都先除以自己在本条 episode 内的峰值再相加，因此不需要跨"牛顿"与"米"的
    权重，也不含任何任务分支。整条 episode 某一路恒为零时那一路直接不贡献。

    effect 一路不直接取 ``norm(concat(dp, dr))``——那要在米和弧度之间拍一个权重。
    改为把 (dp, dr) 作用在冻结 surface 点上取**表面点平均位移**（米），
    ``surface_state`` 一路取面积加权变化量。
    """
    a = record.arrays
    contact = np.asarray(a["aux/frame_force"], dtype=np.float64)

    rigid = np.asarray(a["effect/rigid"], dtype=np.float64)[:, 0, :]
    state = np.asarray(a["effect/surface_state"], dtype=np.float64)[:, 0, :]
    n_level = state.shape[1]
    if n_level not in surface.parent:
        raise TransferError(f"surface 没有 effect field 所需的 {n_level} 点层级")
    points = np.asarray(surface.points[:n_level], dtype=np.float64)
    displacement = rigid[:, None, :3] + np.cross(rigid[:, None, 3:], points[None, :, :])
    effect = np.linalg.norm(displacement, axis=2).mean(axis=1)
    cell_area = np.bincount(surface.parent[n_level],
                            weights=np.asarray(surface.area, dtype=np.float64),
                            minlength=n_level)
    effect = effect + np.einsum("tc,c->t", np.abs(state), cell_area)

    total = np.zeros(len(contact), dtype=np.float64)
    for channel in (contact, effect):
        channel = np.abs(np.nan_to_num(channel, nan=0.0, posinf=0.0, neginf=0.0))
        peak = float(channel.max()) if len(channel) else 0.0
        if peak > 0:
            total = total + channel / peak
    return total


def phase_budget(records: Iterable[EpisodeRecord], surface: Surface, *, n_bins: int,
                 floor: int = PHASE_FLOOR) -> tuple[int, ...]:
    """按各 phase 的活动量占比分配命令格数，每个**有帧**的 phase 保底 ``floor`` 格。

    用跨 episode 的**中位数**占比，避免个别异常长的接近段决定整条命令轴。
    """
    shares: list[np.ndarray] = []
    occupancy = np.zeros(N_PHASES, dtype=bool)
    for record in records:
        activity = activity_rate(record, surface)
        phase = np.asarray(record.arrays["phase"], dtype=np.int64)
        valid = np.asarray(record.arrays["valid_s4"], dtype=bool)
        present = np.array([bool((valid & (phase == p)).any()) for p in range(N_PHASES)])
        occupancy |= present
        per_phase = np.array([activity[valid & (phase == p)].sum() for p in range(N_PHASES)])
        total = per_phase.sum()
        shares.append(per_phase / total if total > 0
                      else present / max(int(present.sum()), 1))
    if not shares:
        raise TransferError("phase_budget 至少需要一条 episode")
    share = np.where(occupancy, np.median(np.stack(shares), axis=0), 0.0)
    share = share / share.sum() if share.sum() > 0 else occupancy / max(int(occupancy.sum()), 1)

    budget = np.where(occupancy, floor, 0).astype(np.int64)
    spare = int(n_bins - budget.sum())
    if spare < 0:
        raise TransferError(f"n_bins={n_bins} 不足以给 {int(occupancy.sum())} 个 phase "
                            f"各保底 {floor} 格")
    extra = np.floor(share * spare).astype(np.int64)
    # 余数按小数部分从大到小补，保证总数正好 n_bins 且与 episode 顺序无关。
    remainder = share * spare - extra
    for index in np.argsort(-remainder, kind="stable")[:spare - int(extra.sum())]:
        extra[index] += 1
    return tuple(int(value) for value in budget + extra)


def bin_index(record: EpisodeRecord, surface: Surface, *, budget: tuple[int, ...]) -> np.ndarray:
    """把每个有效帧映射到命令格号；无效帧为 -1。"""
    a = record.arrays
    valid = np.asarray(a["valid_s4"], dtype=bool)
    phase = np.asarray(a["phase"], dtype=np.int64)
    activity = activity_rate(record, surface)
    offsets = np.concatenate([[0], np.cumsum(budget)]).astype(np.int64)
    out = np.full(len(valid), -1, dtype=np.int64)
    for p in range(N_PHASES):
        frames = np.flatnonzero(valid & (phase == p))
        width = int(budget[p])
        if not len(frames):
            continue
        if width <= 0:
            raise TransferError(
                f"phase {p} 有 {len(frames)} 个有效帧但分到 0 格；那些帧会被静默丢弃")
        if len(frames) == 1:
            local = np.array([0.5])
        else:
            cumulative = np.concatenate([[0.0], np.cumsum(activity[frames[1:]])])
            span = float(cumulative[-1])
            local = (cumulative / span * (1.0 - 1e-9) if span > 1e-12
                     else (np.arange(len(frames)) + 0.5) / len(frames))
        out[frames] = offsets[p] + np.minimum((local * width).astype(np.int64), width - 1)
    if np.any(valid & (out < 0)):
        raise TransferError("有有效帧没有落进任何命令格")
    return out


# ---------------------------------------------------------------- 产物

@dataclass
class InteractionTransfer:
    """一个任务实例的 embodiment-independent interaction command。"""

    meta: dict[str, Any]
    arrays: dict[str, np.ndarray]

    def validate(self) -> None:
        if self.meta.get("schema_version") != TRANSFER_SCHEMA_VERSION:
            raise TransferError(
                f"schema_version 必须是 {TRANSFER_SCHEMA_VERSION!r}，"
                f"实际是 {self.meta.get('schema_version')!r}。"
                "v1 的对齐轴与 cell 统计口径都已作废（D-65/D-66），不得混用")
        for key in ("transfer_id", "task", "object", "geometry_variant", "surface"):
            if key not in self.meta:
                raise TransferError(f"meta.{key} 是必填项")
        unknown = set(self.arrays) - EXECUTOR_ARRAYS
        missing = EXECUTOR_ARRAYS - set(self.arrays)
        if unknown or missing:
            raise TransferError(
                f"数组必须精确匹配 E-I allowlist；missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}")

        arrays = {name: np.asarray(value) for name, value in self.arrays.items()}
        if any(value.dtype == object for value in arrays.values()):
            raise TransferError("不允许 object dtype")
        fraction = arrays["command/fraction"]
        if fraction.ndim != 1 or len(fraction) < 2:
            raise TransferError("command/fraction 必须是一维且至少 2 个时间格")
        bins = len(fraction)
        if np.any(np.diff(fraction) <= 0) or fraction[0] < 0 or fraction[-1] > 1:
            raise TransferError("command/fraction 必须在 [0,1] 内严格递增")
        for name in EXECUTOR_ARRAYS:
            if name.startswith("surface/"):
                continue
            if arrays[name].shape[0] != bins:
                raise TransferError(f"{name} 的首维必须是时间格数 {bins}")
        if arrays["command/valid"].shape != (bins,) or arrays["command/valid"].dtype != np.bool_:
            raise TransferError("command/valid 必须是一维 bool")
        if not np.array_equal(arrays["command/valid"], arrays["support/episodes"] > 0):
            raise TransferError("command/valid 必须与 support/episodes > 0 一致")

        points = arrays["surface/points_obj"]
        normals = arrays["surface/normals_obj"]
        area = arrays["surface/area"]
        part = arrays["surface/part"]
        if points.ndim != 2 or points.shape[1] != 3:
            raise TransferError("surface/points_obj 必须是 (S,3)")
        n_surface = points.shape[0]
        if normals.shape != points.shape or area.shape != (n_surface,) or part.shape != (n_surface,):
            raise TransferError("surface arrays 的点数不一致")
        if np.any(area <= 0) or not np.isfinite(points).all() or not np.isfinite(normals).all():
            raise TransferError("surface arrays 含非法值")

        for name in ("region/mass/mean", "region/mass/q10", "region/mass/q90",
                     "region/support", "region/duty", "engage/concentration",
                     "engage/valid"):
            if arrays[name].shape != (bins, n_surface):
                raise TransferError(f"{name} 必须是 (B,S)")
        for name in ("engage/dir/mean", "mech/traction_obj/median",
                     "mech/traction_obj/q10", "mech/traction_obj/q90",
                     "mech/moment_density_obj/median",
                     "mech/moment_density_obj/q10",
                     "mech/moment_density_obj/q90"):
            if arrays[name].shape != (bins, n_surface, 3):
                raise TransferError(f"{name} 必须是 (B,S,3)")
        if arrays["mode/prob"].shape != (bins, n_surface, MODE_COUNT):
            raise TransferError("mode/prob 必须是 (B,S,4)")

        support = arrays["region/support"]
        if np.any(support < 0) or np.any(support > arrays["support/episodes"][:, None]):
            raise TransferError("region/support 必须在 [0, support/episodes] 内")
        duty = arrays["region/duty"]
        if np.any(duty < 0) or np.any(duty > 1.0 + 1e-6):
            raise TransferError("region/duty 必须在 [0,1] 内")
        if np.any((support == 0) & (duty > 0)):
            raise TransferError("没有 episode 支持的 cell 不能有非零 duty")
        # 接触条件化字段必须与 support 一致：没人碰过的 cell 上一律为 0，
        # 有人碰过的 cell 不得被静默当成"力为零"（v1 的 [q10,q90]=[0,0] 就是这么来的）。
        for name in ("engage/dir/mean", "mech/traction_obj/median",
                     "mech/moment_density_obj/median"):
            if np.any(np.linalg.norm(arrays[name], axis=2)[support == 0] != 0):
                raise TransferError(f"{name} 在 support==0 的 cell 上必须是 0")
        if np.any(arrays["mode/prob"][support == 0] != 0):
            raise TransferError("mode/prob 在 support==0 的 cell 上必须是 0")
        concentration = arrays["engage/concentration"]
        if np.any(concentration < -1e-6) or np.any(concentration > 1.0 + 1e-6):
            raise TransferError("engage/concentration 必须在 [0,1] 内")
        if not np.array_equal(arrays["engage/valid"], concentration > 1e-8):
            raise TransferError("engage/valid 必须与 engage/concentration > 0 一致")

        region = arrays["region/mass/mean"]
        nonempty = region.sum(axis=1) > 0
        if np.any(np.abs(region[nonempty].sum(axis=1) - 1.0) > 2e-5):
            raise TransferError("非空 region/mass/mean 每个时间格必须归一化")
        mode_sum = arrays["mode/prob"].sum(axis=2)
        occupied = support > 0
        if np.any(np.abs(mode_sum[occupied] - 1.0) > 2e-5):
            raise TransferError("有支持的 cell 上 mode/prob 必须归一化")
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise TransferError("产物中不允许 NaN/Inf；无支持位置应由 support/duty 表示")

    def executor_arrays(self) -> dict[str, np.ndarray]:
        """返回 E-I 可以读取的完整、且仅限 allowlist 的数组。"""
        self.validate()
        return {name: self.arrays[name] for name in sorted(EXECUTOR_ARRAYS)}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"无法 JSON 序列化：{type(value).__name__}")


def save_transfer(transfer: InteractionTransfer, path: str | os.PathLike[str]) -> str:
    """原子地保存一个 interaction transfer。"""
    transfer.validate()
    destination = Path(path)
    if destination.suffix != ".npz":
        raise TransferError("interaction transfer 路径必须以 .npz 结尾")
    destination.parent.mkdir(parents=True, exist_ok=True)
    meta = json.dumps(transfer.meta, ensure_ascii=False, sort_keys=True,
                      default=_json_default)
    payload: dict[str, Any] = {META_KEY: np.asarray(meta)}
    payload.update({name: np.asarray(value) for name, value in transfer.arrays.items()})
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, destination)
    return str(destination.resolve())


def load_transfer(path: str | os.PathLike[str]) -> InteractionTransfer:
    """读取并校验一个 interaction transfer。"""
    with np.load(path, allow_pickle=False) as data:
        if META_KEY not in data.files or data[META_KEY].ndim != 0:
            raise TransferError(f"{path} 缺少标量 {META_KEY}")
        meta = json.loads(str(data[META_KEY].item()))
        arrays = {name: np.array(data[name], copy=True)
                  for name in data.files if name != META_KEY}
    transfer = InteractionTransfer(meta=meta, arrays=arrays)
    transfer.validate()
    return transfer


# ---------------------------------------------------------------- 聚合

def _nan_stat(values: np.ndarray, stat: str) -> np.ndarray:
    """沿 episode 轴统计，完全无支持的位置填 0，由 companion mask/count 区分。"""
    # NumPy 对"所有 episode 在这里都无支持"的合法位置发 RuntimeWarning；
    # companion support/count 已明确表达它，统计数组按契约填 0。
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if stat == "mean":
            result = np.nanmean(values, axis=0)
        elif stat == "median":
            result = np.nanmedian(values, axis=0)
        elif stat == "q10":
            result = np.nanquantile(values, 0.10, axis=0)
        elif stat == "q90":
            result = np.nanquantile(values, 0.90, axis=0)
        else:  # pragma: no cover - internal programming error
            raise ValueError(stat)
    return np.nan_to_num(result, nan=0.0).astype(np.float32)


def _bincount2d(index: np.ndarray, weights: np.ndarray, size: int,
                shape: tuple[int, ...]) -> np.ndarray:
    """按最后一维逐通道 bincount，再 reshape 回 (B, S, C)。"""
    channels = weights.shape[1] if weights.ndim == 2 else 1
    flat = np.stack([np.bincount(index, weights=weights[:, c] if channels > 1 else weights,
                                 minlength=size)
                     for c in range(channels)], axis=1)
    return flat.reshape(*shape, channels) if channels > 1 else flat.reshape(shape)


def _episode_summary(record: EpisodeRecord, surface: Surface, *, budget: tuple[int, ...],
                     n_bins: int, n_surface: int
                     ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """先汇总单条 episode；之后跨 episode 统计才能保证采集者等权。

    每个 cell 的 ``engage`` / ``mode`` / ``traction`` / ``moment_density`` 一律只用
    **该 cell 实际有接触的那些帧**，并配一份 ``contact_frames``。v1 让 traction 用
    全部帧而其余用接触帧，同一个 cell 上不同字段来自不同的帧集合。
    """
    a = record.arrays
    episode_id = record.meta["episode_id"]
    valid = np.asarray(a["valid_s4"], dtype=bool)
    index = bin_index(record, surface, budget=budget)

    idx = np.asarray(a["region/point_idx"], dtype=np.int64)
    slot_valid = np.asarray(a["region/valid"], dtype=bool)
    on_surface = np.asarray(a["region/on_surface"], dtype=bool)
    weight = np.asarray(a["region/weight"], dtype=np.float64)
    force = np.asarray(a["mech/force_obj"], dtype=np.float64)
    position = np.asarray(a["region/pos_obj"], dtype=np.float64)
    engage = np.asarray(a["engage/dir"], dtype=np.float64)
    mode = np.asarray(a["mode/label"], dtype=np.int64)
    if idx.shape != weight.shape or force.shape != idx.shape + (3,):
        raise TransferError(f"episode {episode_id} 的 contact slots 形状不一致")

    parent = surface.parent[n_surface]
    cell_area = np.bincount(parent, weights=np.asarray(surface.area, dtype=np.float64),
                            minlength=n_surface)
    if np.any(cell_area <= 0):
        raise TransferError("低分辨率 surface 出现零面积 cell")
    representative = np.asarray(surface.points[:n_surface], dtype=np.float64)

    live = slot_valid & on_surface & (weight > 0) & valid[:, None] & (index >= 0)[:, None]
    if np.any(idx[live] < 0) or np.any(idx[live] >= surface.n_points):
        raise TransferError(f"episode {episode_id} 的 surface index 越界")
    frame_of, slot_of = np.nonzero(live)
    coarse = parent[idx[frame_of, slot_of]]
    bin_of = index[frame_of]
    flat = bin_of * n_surface + coarse
    size = n_bins * n_surface
    shape = (n_bins, n_surface)

    live_mode = mode[frame_of, slot_of]
    if np.any((live_mode < 0) | (live_mode >= MODE_COUNT)):
        raise TransferError(f"episode {episode_id} 有未知 mode")
    live_weight = weight[frame_of, slot_of]
    live_force = force[frame_of, slot_of]
    live_moment = np.cross(position[frame_of, slot_of] - representative[coarse], live_force)

    region_sum = _bincount2d(flat, live_weight, size, shape)
    engage_sum = _bincount2d(flat, live_weight[:, None] * engage[frame_of, slot_of], size, shape)
    force_sum = _bincount2d(flat, live_force, size, shape)
    moment_sum = _bincount2d(flat, live_moment, size, shape)
    mode_sum = np.bincount(flat * MODE_COUNT + live_mode,
                           minlength=size * MODE_COUNT).reshape(n_bins, n_surface, MODE_COUNT)
    mode_sum = mode_sum.astype(np.float64)

    # 一帧里可能有多个接触点落进同一个 cell，按 slot 计数会把"接触了多少帧"数大。
    unique = np.unique(frame_of * n_surface + coarse)
    contact_frames = np.bincount(index[unique // n_surface] * n_surface + (unique % n_surface),
                                 minlength=size).reshape(shape).astype(np.float64)

    n_frames = np.bincount(index[valid], minlength=n_bins).astype(np.float64)
    present = n_frames > 0
    occupied = contact_frames > 0

    def conditioned(total: np.ndarray) -> np.ndarray:
        out = np.full(total.shape, np.nan)
        divisor = contact_frames[..., None] if total.ndim == 3 else contact_frames
        np.divide(total, divisor, out=out, where=(occupied[..., None] if total.ndim == 3
                                                  else occupied))
        return out

    traction = conditioned(force_sum) / cell_area[None, :, None]
    moment_density = conditioned(moment_sum) / cell_area[None, :, None]
    # 方向按接触力权重平均后再归一化；没有接触的 cell 留 NaN，跨 episode 统计时排除。
    engage_mean = np.full(shape + (3,), np.nan)
    np.divide(engage_sum, region_sum[..., None], out=engage_mean, where=occupied[..., None])
    norm = np.linalg.norm(np.nan_to_num(engage_mean), axis=2, keepdims=True)
    engage_mean = np.divide(engage_mean, norm, out=np.full(engage_mean.shape, np.nan),
                            where=(norm > 1e-12) & occupied[..., None])
    mode_prob = np.full(shape + (MODE_COUNT,), np.nan)
    mode_total = mode_sum.sum(axis=2, keepdims=True)
    np.divide(mode_sum, mode_total, out=mode_prob, where=occupied[..., None] & (mode_total > 0))

    region = np.full(shape, np.nan)
    row_total = region_sum.sum(axis=1, keepdims=True)
    np.divide(region_sum, row_total, out=region, where=row_total > 0)

    duty = np.zeros(shape)
    np.divide(contact_frames, n_frames[:, None], out=duty, where=present[:, None])

    rigid = np.asarray(a["effect/rigid"], dtype=np.float64)
    state = np.asarray(a["effect/surface_state"], dtype=np.float64)
    future_valid = np.asarray(a["effect/future_valid"], dtype=bool)
    if rigid.shape[:2] != future_valid.shape or state.shape[:2] != future_valid.shape:
        raise TransferError(f"episode {episode_id} 的 effect future mask 不匹配")
    horizon = rigid.shape[1]
    effect = {}
    for name, values in (("rigid", rigid), ("state", state)):
        summed = np.full((n_bins, horizon, values.shape[2]), np.nan)
        count = np.zeros((n_bins, horizon), dtype=np.int32)
        for hi in range(horizon):
            frames = np.flatnonzero(valid & future_valid[:, hi])
            if not len(frames):
                continue
            hits = np.bincount(index[frames], minlength=n_bins).astype(np.float64)
            totals = _bincount2d(index[frames], values[frames, hi], n_bins, (n_bins,))
            live_bins = hits > 0
            summed[live_bins, hi] = totals[live_bins] / hits[live_bins, None]
            count[live_bins, hi] = 1
        effect[name] = summed
        effect[f"{name}_count"] = count

    summary = {
        "present": present,
        "region": region,
        "contact_frames": contact_frames,
        "duty": duty,
        "engage": engage_mean,
        "mode": mode_prob,
        "traction": traction,
        "moment_density": moment_density,
        "rigid": effect["rigid"],
        "rigid_count": effect["rigid_count"],
        "state": effect["state"],
        "state_count": effect["state_count"],
    }
    return summary, _projection_diagnostics(record, valid, live, force_sum, moment_sum,
                                            representative, index, n_bins)


def _projection_diagnostics(record: EpisodeRecord, valid: np.ndarray, live: np.ndarray,
                            force_sum: np.ndarray, moment_sum: np.ndarray,
                            representative: np.ndarray, index: np.ndarray,
                            n_bins: int) -> dict[str, float]:
    """表面投影的**完整性**残差——不是"重建误差"。

    ⚠️ cell 内先算相对代表点的局部力矩、再与 ``cross(representative, F)`` 相加，
    在代数上恒等于直接对接触点求力矩。所以拿它跟自己对拍只会得到浮点精度级的数，
    那不构成任何验证（v1 的 README 把 6e-6 N 当成了 6D wrench 重建的证据）。

    有信息量的是：被 ``on_surface`` / ``weight>0`` 滤掉的接触占了多少力，以及由此
    与 S4 **独立**记下的 ``mech/wrench_obj`` 差多少。滤掉的力越多，这份表面表示
    就越不足以代表物体实际受到的作用。
    """
    a = record.arrays
    force = np.asarray(a["mech/force_obj"], dtype=np.float64)
    slot_valid = np.asarray(a["region/valid"], dtype=bool) & valid[:, None]
    magnitude = np.linalg.norm(force, axis=2)
    total_mass = float(magnitude[slot_valid].sum())
    dropped_mass = float(magnitude[slot_valid & ~live].sum())

    raw = np.asarray(a["mech/wrench_obj"], dtype=np.float64)[valid]
    kept_force = force_sum.sum(axis=(0, 1))
    kept_torque = (np.cross(representative[None, :, :], force_sum).sum(axis=(0, 1))
                   + moment_sum.sum(axis=(0, 1)))
    return {
        "projection_residual_force_N": float(np.linalg.norm(kept_force - raw[:, :3].sum(axis=0))),
        "projection_residual_torque_Nm": float(
            np.linalg.norm(kept_torque - raw[:, 3:].sum(axis=0))),
        "dropped_force_mass_fraction": float(dropped_mass / total_mass) if total_mass > 0 else 0.0,
        "valid_frame_fraction": float(valid.mean()),
        "empty_bin_fraction": float((np.bincount(index[valid], minlength=n_bins) == 0).mean()),
    }


def episode_summary(record: EpisodeRecord, surface: Surface, *, budget: tuple[int, ...],
                    n_bins: int, n_surface: int
                    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """把一条 episode 投影到**已冻结的**命令轴上。

    S5 评估必须用 artifact 自己的 ``aggregation.phase_budget`` 来投影留出 episode，
    否则比的是两条不同的轴，coverage 数没有意义。
    """
    return _episode_summary(record, surface, budget=budget, n_bins=n_bins,
                            n_surface=n_surface)


def build_transfer(records: Iterable[EpisodeRecord], *, n_bins: int = 32,
                   n_surface: int = 256, transfer_id: str | None = None,
                   surface: Surface | None = None,
                   phase_floor: int = PHASE_FLOOR,
                   budget: tuple[int, ...] | None = None) -> InteractionTransfer:
    """从同一任务/物体/几何的多条成功 S4 records 构造统计 interaction command。

    每条 episode 在每个命令格内先汇总，跨 episode 再取均值/分位数。因此一个
    500 帧的采集者不会比一个 200 帧的采集者拥有 2.5 倍权重。

    ``budget`` 给定时用它而不重算命令轴。**比较两份 envelope 时必须这样做**：
    每个子集自己算 budget 会得到不同的轴，逐格比出来的距离就没有意义。

    每个 cell 上的 ``engage`` / ``mode`` / ``traction`` / ``moment_density`` 只在
    **实际接触过该 cell 的 episode** 之间统计，配套的 ``region/support`` 说明那是
    几条。这样同一个 cell 的各字段来自同一组 episode，不会出现"这里该接触"与
    "这里的力必须为零"并存的指令（v1 的实测占比：抽屉 10%、擦拭 15%）。
    """
    if n_bins < 2:
        raise TransferError("n_bins 至少为 2")
    materialized = list(records)
    if len(materialized) < 2:
        raise TransferError("至少需要 2 条成功示教；单条轨迹不能代表多人/多次采集统计")

    def validate_record(record: EpisodeRecord) -> None:
        record.validate()
        if record.meta.get("schema_version") != "s4-record-v2":
            raise TransferError("只接受 s4-record-v2")
        if not record.meta.get("success", False):
            raise TransferError("失败/near-success record 不能混进成功 interaction 聚合")

    for record in materialized:
        validate_record(record)
    first = materialized[0].meta
    expected = {key: str(first.get(key, "nominal"))
                for key in ("task", "object", "geometry_variant")}
    expected_surface_hash = str(first.get("surface", {}).get("sha256"))
    obj, geom = expected["object"], expected["geometry_variant"]
    surface = surface or surface_for(obj, geom)
    if surface.sha256 != expected_surface_hash:
        raise TransferError("传入 surface 与 record meta 的 sha256 不一致")
    if n_surface not in surface.parent:
        raise TransferError(
            f"surface 没有 {n_surface} 点层级；可用 {sorted(surface.parent)}")
    for record in materialized:
        for key, expected_value in expected.items():
            actual = str(record.meta.get(key, "nominal"))
            if actual != expected_value:
                raise TransferError(
                    f"所有 records 的 {key} 必须一致：期望 {expected_value!r}，"
                    f"实际 {actual!r}")
        if str(record.meta.get("surface", {}).get("sha256")) != expected_surface_hash:
            raise TransferError("不能聚合不同冻结 surface hash 的 records")

    if budget is None:
        budget = phase_budget(materialized, surface, n_bins=n_bins, floor=phase_floor)
    elif sum(budget) != n_bins or len(budget) != N_PHASES:
        raise TransferError(f"给定 budget {budget} 与 n_bins={n_bins} 不一致")
    summaries: list[dict[str, np.ndarray]] = []
    episode_diagnostics: list[dict[str, float]] = []
    episode_ids: list[str] = []
    strategy_counts: dict[str, int] = {}
    implementation_counts: dict[str, int] = {}
    for record in materialized:
        summary, diagnostics = _episode_summary(
            record, surface, budget=budget, n_bins=n_bins, n_surface=n_surface)
        summaries.append(summary)
        episode_diagnostics.append(diagnostics)
        episode_ids.append(str(record.meta["episode_id"]))
        for counts, key in ((strategy_counts, "strategy_family"),
                            (implementation_counts, "implementation")):
            name = str(record.meta.get(key, "unknown"))
            counts[name] = counts.get(name, 0) + 1

    def stack(name: str) -> np.ndarray:
        return np.stack([summary[name] for summary in summaries], axis=0)

    present = stack("present")
    contact_frames = stack("contact_frames")
    support = (contact_frames > 0).sum(axis=0).astype(np.int32)
    occupied = support > 0

    region = stack("region")
    region_mean = _nan_stat(region, "mean")
    row_sum = region_mean.sum(axis=1, keepdims=True)
    region_mean = np.divide(region_mean, row_sum, out=np.zeros_like(region_mean),
                            where=row_sum > 0)

    duty = np.where(contact_frames > 0, stack("duty"), np.nan)
    engage = stack("engage")
    engage_mean = _nan_stat(engage, "mean")
    # 各 episode 的方向已是单位矢量，跨 episode 平均后的合矢量长度就是集中度 ∈[0,1]。
    # 它是方向多峰性的直接度量；v1 把它阈值化成 bool 就丢掉了。
    concentration = np.linalg.norm(engage_mean, axis=2)
    engage_mean = np.divide(engage_mean, concentration[..., None],
                            out=np.zeros_like(engage_mean),
                            where=concentration[..., None] > 1e-8)

    mode_prob = _nan_stat(stack("mode"), "mean")
    mode_sum = mode_prob.sum(axis=2, keepdims=True)
    mode_prob = np.divide(mode_prob, mode_sum, out=np.zeros_like(mode_prob),
                          where=mode_sum > 0)

    def zero_unsupported(values: np.ndarray) -> np.ndarray:
        return np.where(occupied[..., None] if values.ndim == 3 else occupied, values, 0)

    traction = stack("traction")
    moment_density = stack("moment_density")
    cell_area = np.bincount(surface.parent[n_surface],
                            weights=np.asarray(surface.area, dtype=np.float64),
                            minlength=n_surface).astype(np.float32)
    arrays = {
        "command/fraction": ((np.arange(n_bins, dtype=np.float32) + 0.5) / n_bins),
        "command/valid": present.any(axis=0),
        "support/episodes": present.sum(axis=0).astype(np.int32),
        "surface/points_obj": np.asarray(surface.points[:n_surface], dtype=np.float32),
        "surface/normals_obj": np.asarray(surface.normals[:n_surface], dtype=np.float32),
        "surface/area": cell_area,
        "surface/part": np.asarray(surface.part[:n_surface], dtype=np.int8),
        "effect/rigid/median": _nan_stat(stack("rigid"), "median"),
        "effect/rigid/q10": _nan_stat(stack("rigid"), "q10"),
        "effect/rigid/q90": _nan_stat(stack("rigid"), "q90"),
        "effect/rigid/valid_count": stack("rigid_count").sum(axis=0).astype(np.int32),
        "effect/surface_state/median": _nan_stat(stack("state"), "median"),
        "effect/surface_state/q10": _nan_stat(stack("state"), "q10"),
        "effect/surface_state/q90": _nan_stat(stack("state"), "q90"),
        "effect/surface_state/valid_count": stack("state_count").sum(axis=0).astype(np.int32),
        "region/mass/mean": region_mean.astype(np.float32),
        "region/mass/q10": _nan_stat(region, "q10"),
        "region/mass/q90": _nan_stat(region, "q90"),
        "region/support": support,
        "region/duty": np.where(occupied, _nan_stat(duty, "mean"), 0.0).astype(np.float32),
        "engage/dir/mean": zero_unsupported(engage_mean).astype(np.float32),
        "engage/concentration": np.where(occupied, concentration, 0.0).astype(np.float32),
        "engage/valid": np.where(occupied, concentration, 0.0) > 1e-8,
        "mode/prob": zero_unsupported(mode_prob).astype(np.float32),
        "mech/traction_obj/median": zero_unsupported(_nan_stat(traction, "median")),
        "mech/traction_obj/q10": zero_unsupported(_nan_stat(traction, "q10")),
        "mech/traction_obj/q90": zero_unsupported(_nan_stat(traction, "q90")),
        "mech/moment_density_obj/median": zero_unsupported(_nan_stat(moment_density, "median")),
        "mech/moment_density_obj/q10": zero_unsupported(_nan_stat(moment_density, "q10")),
        "mech/moment_density_obj/q90": zero_unsupported(_nan_stat(moment_density, "q90")),
    }

    task = str(first["task"])
    worst = lambda key: max(item[key] for item in episode_diagnostics)  # noqa: E731
    meta = {
        "schema_version": TRANSFER_SCHEMA_VERSION,
        "transfer_id": transfer_id or f"{task}-{obj}-{geom}",
        # 审计字段：不会由 executor_arrays() 返回。
        "task": task,
        "object": obj,
        "geometry_variant": geom,
        "num_episodes": len(summaries),
        "episode_ids": episode_ids,
        "strategy_family_counts": strategy_counts,
        "implementation_counts": implementation_counts,
        "aggregation": {
            "alignment": "phase_segmented_activity_bins_offline_only",
            "phase_budget": list(budget),
            "phase_floor": phase_floor,
            "episode_weighting": "equal_after_within_episode_bin_summary",
            "cell_statistics": "contact_conditioned_over_contacting_episodes",
            "n_bins": n_bins,
            "quantiles": [0.10, 0.90],
            "claim": "descriptive_demonstration_statistics",
            "mechanics": ("per-cell contact-conditioned traction + local moment density; "
                          "the 6D wrench identity holds per episode before aggregation "
                          "and is NOT preserved by cross-episode medians/quantiles"),
        },
        "surface": {
            "sha256": surface.sha256,
            "source_n_points": surface.n_points,
            "command_n_points": n_surface,
            "parts": list(surface.parts),
            "total_area_m2": float(cell_area.sum()),
        },
        "diagnostics": {
            "projection_residual_force_N": worst("projection_residual_force_N"),
            "projection_residual_torque_Nm": worst("projection_residual_torque_Nm"),
            "dropped_force_mass_fraction_max": worst("dropped_force_mass_fraction"),
            "empty_bin_fraction_max": worst("empty_bin_fraction"),
            "episode_valid_frame_fraction": [item["valid_frame_fraction"]
                                             for item in episode_diagnostics],
            "cell_support_median": float(np.median(support[occupied])) if occupied.any() else 0.0,
            "cell_support_under_half_fraction": (
                float((support[occupied] < len(summaries) / 2).mean()) if occupied.any() else 0.0),
        },
    }
    transfer = InteractionTransfer(meta=meta, arrays=arrays)
    transfer.validate()
    return transfer
