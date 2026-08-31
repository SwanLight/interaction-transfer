"""E-I 的指令通道：把冻结的 interaction artifact 变成执行器每一步看到的张量。

这个模块只做三件事，每件都有一条不能违反的规矩：

1. **加载与白名单**。E-I 只准看 `transfer.EXECUTOR_ARRAYS` 里的数组。task id、
   任务名、`phase`、`progress`、`mech/generalized`、任务原生 effect 一律**硬报错**，
   不靠调用者自觉（`plan/04` §7）。artifact 的 `meta` 永远不出现在给策略的张量里。

2. **指令窗口**。给 PointNet 的是**当前格**的逐 cell 空间场，给 GRU 的是
   **未来 H 格**的低维摘要。为什么这样切见 D-77。

3. **窗口推进**。`plan/02` §5 与 `plan/04` §4 要求执行器**按自己的实际进度**选窗口，
   而且推进判据必须由**通用量**导出、不得读任务的状态机——采集脚本里的
   approach/establish/manipulate/release 是那个任务的状态机标签，直接发给 E-I
   等于递给它一个任务专用状态机，"任务无关"当场作废。

   本模块的推进判据全部来自指令与实测的比对：

   ===================  ==================================================
   已建立接触            指令 `region/allowed` 内的实测法向力 ≥ 阈值
   交互进度              实测 effect 相对本格指令 effect 的完成比例
                        （用 D-74 的表面度量算，任务无关）
   推进                  两者都满足，或本格停留超过 `max_dwell`
   ===================  ==================================================

   `min_dwell` / `max_dwell` 就是 `plan/02` §5 说的"允许对 effect 轨迹做有限时间
   缩放，但不得改变功能目标"——把时间缩放的范围显式写成两个数，而不是让它
   隐含在某个超参里。

**一条容易踩的**：artifact 的向量场（engage 方向、traction、moment）都在**物体系**。
执行器活在世界系。本模块提供 `rotate_fields`，把它们和表面点一起转到调用者指定的
参考系里；**必须整组一起转**，否则会重演 P-53/P-54 那一类"这个量活在哪个坐标系
没标清楚"的错误——那三次都是逐帧数值、接触部位、单元测试全部正常。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

from it.transfer import EXECUTOR_ARRAYS, InteractionTransfer, load_transfer

#: 未来窗口给 GRU 看几格。与 S4 的 `FUTURE_SAMPLES` 无关——那是 effect 的未来采样点数，
#: 这是**命令轴**上的格数。
HORIZON = 8

#: 判"已建立接触"的法向力阈值（N）。与 `interaction.CONTACT_FORCE_MIN` 同量级，
#: 但这里是**整个允许区域上的合力**，所以取大一档。
CONTACT_ESTABLISHED_N = 0.5

#: 判"本格 effect 已完成"的比例。不取 1.0：指令是多条示教的中位数，
#: 要求逐格精确达成等于要求复制某一条示教。
EFFECT_PROGRESS_RATIO = 0.8

#: 每格至少 / 至多停留多少个控制步。这两个数就是允许的时间缩放范围。
MIN_DWELL, MAX_DWELL = 2, 60


class CommandError(ValueError):
    """E-I 指令通道的契约被违反。"""


@dataclass(frozen=True)
class CommandSpec:
    """一份 artifact 里 E-I 能看的全部东西（torch 张量，已上设备）。

    ``arrays`` 的键**精确等于** `transfer.EXECUTOR_ARRAYS`。这里不存 meta：
    审计信息留在 npz 里，进不了模型。
    """

    command_id: str
    n_bins: int
    n_surface: int
    arrays: dict[str, torch.Tensor]

    @property
    def horizon(self) -> int:
        return int(self.arrays["effect/rigid/median"].shape[1])


def load_command(path: str | Path, device: torch.device | str = "cpu") -> CommandSpec:
    """读一份冻结 artifact，并在入口处**硬校验** E-I 白名单。"""
    transfer = load_transfer(Path(path))
    arrays = transfer.executor_arrays()          # 内部会 validate()
    leaked = set(arrays) - EXECUTOR_ARRAYS
    if leaked:
        raise CommandError(f"artifact 里混进了非白名单数组：{sorted(leaked)}")
    for name in arrays:
        lowered = name.lower()
        for banned in ("task", "phase", "progress", "generalized", "strategy",
                       "implementation", "source"):
            if banned in lowered:
                raise CommandError(
                    f"字段 {name!r} 含被禁前缀 {banned!r}；E-I 的指令通道是 exact "
                    "allowlist，不能靠调用者自觉不取（plan/04 §7）")
    tensors = {name: torch.as_tensor(np.asarray(value), device=device)
               for name, value in arrays.items()}
    return CommandSpec(
        command_id=str(transfer.meta["transfer_id"]),
        n_bins=int(tensors["command/fraction"].shape[0]),
        n_surface=int(tensors["surface/points_obj"].shape[0]),
        arrays=tensors)


class CommandBank:
    """一批 artifact 的只读集合。所有 artifact 必须同 ``n_bins`` / ``n_surface``。

    形状一致不是实现方便，是**表示层的要求**：不同任务、不同物体的指令必须是
    同一个定长接口，否则 E-I 的输入维度会随任务变，"任务无关"在接口层就失败了。
    """

    def __init__(self, paths: Iterable[str | Path], device="cpu",
                 forbid: Sequence[str] = ()):
        self.device = torch.device(device)
        self.specs: list[CommandSpec] = []
        forbidden = tuple(forbid)
        for path in sorted(map(Path, paths)):
            spec = load_command(path, self.device)
            # 留出任务的断言（`plan/04` §5.4）：不能只靠自觉，dataloader 入口就拦。
            for token in forbidden:
                if token and token in spec.command_id:
                    raise CommandError(
                        f"{spec.command_id!r} 命中留出禁令 {token!r}。留出任务的指令与"
                        "reward 在 E-I 的**任何**训练阶段都不得出现，包括 curriculum "
                        "与调试（plan/04 §5.4）")
            self.specs.append(spec)
        if not self.specs:
            raise CommandError("指令库是空的")
        shapes = {(s.n_bins, s.n_surface, s.horizon) for s in self.specs}
        if len(shapes) != 1:
            raise CommandError(
                f"所有 artifact 必须同形状，实际有 {sorted(shapes)}；"
                "定长接口是「任务无关」的前提")
        self.n_bins, self.n_surface, self.horizon = shapes.pop()
        self._stack = {name: torch.stack([s.arrays[name] for s in self.specs])
                       for name in EXECUTOR_ARRAYS}
        self.anchor_pos, self.anchor_normal = self._contact_anchor()

    def _contact_anchor(self) -> tuple[torch.Tensor, torch.Tensor]:
        """每份指令的**接触锚点**：(C,3) 物体系位置 + (C,3) 物体系外法向。

        取整条指令上 ``region/mass/mean`` 加权的表面点质心与平均法向——也就是
        "这份指令主要要求在物体的哪一片、从哪个方向作用"。

        它**只读指令**，不读任务、不读环境配置，所以拿它初始化执行器的站位不会
        把任务信息漏进来；S7 零样本评估照用同一条规则即可（`plan/04` §5.4）。

        为什么需要它：跟踪 reward 在没有接触时 ``r_region`` 按设计恒为 0
        （`ei_reward.interaction_reward` 的 ``touching`` 分支），``r_effect`` 的缺口
        在接近段与距离无关地恒为 1。**接近段没有任何稠密梯度**，把执行器丢在
        0.25 m 外等于要求随机游走先撞上物体。采集侧的双板也是直接生成在位点附近的。

        全零质量（整条指令都不要求接触）时退回表面质心 + ``+Z``，不让它变成 NaN。
        """
        mass = self._stack["region/mass/mean"].sum(dim=1)            # (C, S)
        points = self._stack["surface/points_obj"]                   # (C, S, 3)
        normals = self._stack["surface/normals_obj"]
        total = mass.sum(dim=-1, keepdim=True)                       # (C, 1)
        has_mass = total.squeeze(-1) > 0
        weight = torch.where(total > 0, mass / total.clamp_min(1e-12),
                             torch.full_like(mass, 1.0 / max(self.n_surface, 1)))
        pos = torch.einsum("cs,csd->cd", weight, points)
        nrm = torch.einsum("cs,csd->cd", weight, normals)
        norm = nrm.norm(dim=-1, keepdim=True)
        fallback = torch.zeros_like(nrm)
        fallback[:, 2] = 1.0
        nrm = torch.where(norm > 1e-6, nrm / norm.clamp_min(1e-12), fallback)
        if not bool(has_mass.all()):
            missing = [self.ids[i] for i in (~has_mass).nonzero().flatten().tolist()]
            raise CommandError(
                f"这些 artifact 的 region/mass/mean 全为零，无法定出接触锚点：{missing}。"
                "整条指令都不要求接触的 artifact 不该进 E-I 的训练集")
        return pos, nrm

    def __len__(self) -> int:
        return len(self.specs)

    @property
    def ids(self) -> list[str]:
        return [s.command_id for s in self.specs]

    def gather(self, name: str, command_index: torch.Tensor) -> torch.Tensor:
        """取 (N, ...)：每个 env 按它当前用的那份 artifact 取一个字段。"""
        return self._stack[name][command_index]

    def gather_bin(self, name: str, command_index: torch.Tensor,
                   bin_index: torch.Tensor) -> torch.Tensor:
        """取 (N, ...)：再按每个 env 当前的命令格取一格。"""
        return self._stack[name][command_index, bin_index]

    def gather_window(self, name: str, command_index: torch.Tensor,
                      bin_index: torch.Tensor, horizon: int = HORIZON) -> torch.Tensor:
        """取 (N, horizon, ...)：从当前格开始的未来窗口，越界处夹到最后一格。

        夹到最后一格而不是补零：补零会让"指令结束"和"这一格要求零作用"变成同一个
        输入，而这两件事对执行器的意义完全相反（一个是可以松手，一个是要保持接触
        但不施力）。夹住的语义是"保持最后的要求"，与 `plan/02` §5
        "object progress 落后时 desired 窗口不继续盲目前移"一致。
        """
        offsets = torch.arange(horizon, device=bin_index.device)
        index = (bin_index[:, None] + offsets[None, :]).clamp_max(self.n_bins - 1)
        flat = self._stack[name][command_index[:, None].expand_as(index), index]
        return flat


class WindowTracker:
    """按执行器**自己的实际进度**推进命令窗口（`plan/02` §5 / `plan/04` §4）。

    维护每个 env 的 (command_index, bin_index, dwell, 本格累计 effect)。
    ``step`` 吃实测量，吐出推进后的格号与一组**通用**的进度标量——那组标量既是
    推进判据，也正好是 `plan/04` §4 要求 E-I 观测里带的"由通用量导出的 phase/progress"。
    """

    def __init__(self, bank: CommandBank, num_envs: int, device="cpu",
                 min_dwell: int = MIN_DWELL, max_dwell: int = MAX_DWELL,
                 contact_threshold: float = CONTACT_ESTABLISHED_N,
                 progress_ratio: float = EFFECT_PROGRESS_RATIO):
        self.bank, self.device = bank, torch.device(device)
        self.min_dwell, self.max_dwell = int(min_dwell), int(max_dwell)
        self.contact_threshold = float(contact_threshold)
        self.progress_ratio = float(progress_ratio)
        z = lambda dtype: torch.zeros(num_envs, dtype=dtype, device=self.device)  # noqa: E731
        self.command_index = z(torch.long)
        self.bin_index = z(torch.long)
        self.dwell = z(torch.long)
        self.achieved = z(torch.float32)      # 本格累计的 effect 完成量（米）
        self.last_deficit = z(torch.float32)  # 本步切格**之前**结算的完成缺口
        self.finished = z(torch.bool)

    def reset(self, env_ids: torch.Tensor, command_index: torch.Tensor) -> None:
        self.command_index[env_ids] = command_index.to(self.device)
        self.bin_index[env_ids] = 0
        self.dwell[env_ids] = 0
        self.achieved[env_ids] = 0.0
        self.last_deficit[env_ids] = 0.0
        self.finished[env_ids] = False

    def demand(self) -> torch.Tensor:
        """本格要求的 effect 幅度，**无量纲**。0 表示这一格不要求物体发生变化。

        直接消费 artifact 的 ``effect/bin_demand``：它是在同一命令格内累计逐步 effect、
        再跨 episode 聚合的量。不能从 ``effect/rigid/median[b,0]`` 重算；后者是未来
        0.1 s 的局部变化，与这里的整格累计量时间基准不同（P-84 / D-91）。
        """
        return self.bank.gather_bin("effect/bin_demand", self.command_index,
                                    self.bin_index)

    def step(self, *, effect_increment: torch.Tensor,
             region_normal_force: torch.Tensor) -> dict[str, torch.Tensor]:
        """推进一步。

        Args:
            effect_increment: (N,) 本控制步实测的 effect 幅度（**无量纲**，两路各自
                除以自己的刻度后相加），由 `ei_reward.effect_magnitude` 算——
                必须与 `demand()` 用同一个口径，否则比值没有意义。
            region_normal_force: (N,) 落在**指令允许区域内**的实测法向力合力（N）。

        Returns:
            一组通用进度标量，同时用作观测（`plan/04` §4）。
        """
        self.dwell += 1
        self.achieved += effect_increment.to(self.device)
        demand = self.demand()
        needs_effect = demand > 1e-6
        # 不要求物体动的格子（接近段、松开段）只看接触条件；否则 ratio 恒为 0，
        # 那些格永远只能靠 max_dwell 超时才前进，等于把接近段变成一段固定的空转。
        ratio = torch.where(needs_effect, self.achieved / demand.clamp_min(1e-9),
                            torch.ones_like(demand))
        # 必须在修改 bin_index / 清零 achieved **之前**结算。旧实现由环境在 step()
        # 之后读取 tracker.achieved/demand；恰好推进的 env 已跳到下一格且 achieved=0，
        # 因而“完成得越快”越频繁收到下一格的满缺口。
        self.last_deficit = (1.0 - ratio.clamp(0.0, 1.0)) * needs_effect.float()
        established = region_normal_force.to(self.device) >= self.contact_threshold
        # 这一格若整格不要求接触（允许区域上没有力的要求），接触条件自动满足。
        wants_contact = self.bank.gather_bin(
            "region/mass/mean", self.command_index, self.bin_index).sum(-1) > 0
        contact_ok = established | ~wants_contact
        ready = contact_ok & (ratio >= self.progress_ratio) & (self.dwell >= self.min_dwell)
        advance = (ready | (self.dwell >= self.max_dwell)) & ~self.finished

        self.bin_index = torch.where(advance, self.bin_index + 1, self.bin_index)
        self.finished = self.finished | (self.bin_index >= self.bank.n_bins)
        self.bin_index = self.bin_index.clamp_max(self.bank.n_bins - 1)
        self.dwell = torch.where(advance, torch.zeros_like(self.dwell), self.dwell)
        self.achieved = torch.where(advance, torch.zeros_like(self.achieved), self.achieved)
        return {
            "contact_established": contact_ok.float(),
            "effect_progress": ratio.clamp(0.0, 2.0),
            "bin_fraction": self.bin_index.float() / max(self.bank.n_bins - 1, 1),
            "dwell_fraction": self.dwell.float() / self.max_dwell,
            "timed_out": (advance & ~ready).float(),
            "finished": self.finished.float(),
        }


# ---------------------------------------------------------------- 给网络的特征

#: 指令按字段分组，C0–C5 的信息条件就是**开关这几组**（`plan/02` §6）。
#: 每组配一个 mask 位，缺失时**填零 + mask 置 0**，网络结构与参数量在所有条件下
#: 完全相同——这是把性能差异归因到信息而非容量的前提（`plan/04` §7）。
FIELD_GROUPS = ("region", "engage", "mode", "mech")

#: 逐格空间特征的维度：几何 7 + region 4 + engage 4 + mode 7 + mech 12 + 4 个 mask 位。
SPATIAL_DIM = 7 + 4 + 4 + 7 + 12 + len(FIELD_GROUPS)
#: 未来窗口每格的摘要维度 + 4 个 mask 位 + 1 个有效位。摘要 23 =
#: effect 刚体 6 + 表面状态 1 + region 质心 3 + 展布 1 + 允许占比 1
#: + engage 方向 3 + 集中度 1 + 滑移三分位 3 + traction 均值 3 + 盒宽 1。
TEMPORAL_DIM = 23 + len(FIELD_GROUPS) + 1


def _masked(value: torch.Tensor, enabled: bool) -> torch.Tensor:
    return value if enabled else torch.zeros_like(value)


def spatial_features(bank: "CommandBank", command_index: torch.Tensor,
                     bin_index: torch.Tensor, *, rotation: torch.Tensor | None = None,
                     translation: torch.Tensor | None = None,
                     enabled: dict[str, bool] | None = None) -> torch.Tensor:
    """当前命令格的逐格空间特征 (N, S, SPATIAL_DIM)，喂给 PointNet。

    ``rotation`` / ``translation`` 把**表面点与所有向量场一起**转到执行器的参考系。
    必须整组一起转：只转点不转 engage/traction，或者只转一部分，就会重演
    P-53/P-54 那一类"这个量活在哪个坐标系没标清楚"——那三次都是逐帧数值、
    接触部位、单元测试全部正常，错的只有结论。
    """
    on = {name: True for name in FIELD_GROUPS} | dict(enabled or {})
    take = lambda name: bank.gather_bin(name, command_index, bin_index)  # noqa: E731
    points = bank.gather("surface/points_obj", command_index)
    normals = bank.gather("surface/normals_obj", command_index)
    area = bank.gather("surface/area", command_index)[..., None]
    engage = take("engage/dir/mean")
    traction = torch.cat([take("mech/traction_obj/median"), take("mech/traction_obj/lo"),
                          take("mech/traction_obj/hi")], dim=-1)
    moment = take("mech/moment_density_obj/median")
    if rotation is not None:
        rot = lambda v: torch.einsum("nij,nsj->nsi", rotation, v)  # noqa: E731
        points, normals, engage, moment = rot(points), rot(normals), rot(engage), rot(moment)
        traction = torch.cat([rot(traction[..., 0:3]), rot(traction[..., 3:6]),
                              rot(traction[..., 6:9])], dim=-1)
        if translation is not None:
            points = points + translation[:, None, :]
    n_env, n_cell = points.shape[:2]
    blocks = [
        torch.cat([points, normals, area], dim=-1),
        _masked(torch.cat([take("region/mass/mean")[..., None],
                           take("region/allowed").float()[..., None],
                           (take("region/support").float()
                            / bank.gather_bin("support/episodes", command_index,
                                              bin_index).float().clamp_min(1.0)[..., None]
                            )[..., None],
                           take("region/duty")[..., None]], dim=-1), on["region"]),
        _masked(torch.cat([engage, take("engage/concentration")[..., None]],
                          dim=-1), on["engage"]),
        _masked(torch.cat([take("mode/slip_speed/median"), take("mode/slip_speed/lo"),
                           take("mode/slip_speed/hi"), take("mode/prob")],
                          dim=-1), on["mode"]),
        _masked(torch.cat([traction, moment], dim=-1), on["mech"]),
        torch.stack([torch.full((n_env, n_cell), float(on[name]),
                                device=points.device) for name in FIELD_GROUPS], dim=-1),
    ]
    return torch.cat(blocks, dim=-1)


def temporal_features(bank: "CommandBank", command_index: torch.Tensor,
                      bin_index: torch.Tensor, *, horizon: int = HORIZON,
                      rotation: torch.Tensor | None = None,
                      enabled: dict[str, bool] | None = None) -> torch.Tensor:
    """未来 ``horizon`` 格的**低维摘要** (N, horizon, TEMPORAL_DIM)，喂给 GRU。

    为什么未来窗口给摘要而不给逐格全图（D-77）：逐格空间场是 (H, S, 34)，
    H=8、S=256、2048 个 env 时每个前向要过 8 次 PointNet，而 PPO 每轮还要重放几遍。
    摘要保留了"下一步该往哪推、推多大、允许多滑"这些决定动作的量，
    丢掉的是"未来某一格里第 137 个 cell 的细节"——那个细节等窗口推进到那一格时，
    空间通道会原样给出来。这与 OmniContact 的 contact flow 用**稀疏未来目标**
    而不是稠密全身轨迹是同一个取舍。
    """
    on = {name: True for name in FIELD_GROUPS} | dict(enabled or {})
    win = lambda name: bank.gather_window(name, command_index, bin_index, horizon)  # noqa: E731
    points = bank.gather("surface/points_obj", command_index)               # (N,S,3)
    mass = win("region/mass/mean")                                          # (N,H,S)
    total = mass.sum(-1, keepdim=True).clamp_min(1e-9)
    centroid = torch.einsum("nhs,nsd->nhd", mass / total, points)
    # 质量加权的空间展布（米）：接触要求是"集中在一小片"还是"铺开在一大片"。
    offset = (points[:, None, :, :] - centroid[:, :, None, :]).pow(2).sum(-1)   # (N,H,S)
    spread = ((mass / total) * offset).sum(-1, keepdim=True).clamp_min(0).sqrt()
    allowed_fraction = win("region/allowed").float().mean(-1, keepdim=True)
    engage = torch.einsum("nhs,nhsd->nhd", mass / total, win("engage/dir/mean"))
    concentration = (mass / total * win("engage/concentration")).sum(-1, keepdim=True)
    slip = torch.stack([(mass / total * win(f"mode/slip_speed/{k}")[..., 0]).sum(-1)
                        for k in ("median", "lo", "hi")], dim=-1)
    traction_mean = torch.einsum("nhs,nhsd->nhd", mass / total,
                                 win("mech/traction_obj/median"))
    traction_width = (mass / total * (win("mech/traction_obj/hi")
                                      - win("mech/traction_obj/lo")).norm(dim=-1)
                      ).sum(-1, keepdim=True)
    rigid = win("effect/rigid/median")[:, :, 0, :]                          # (N,H,6)
    state = win("effect/surface_state/median")[:, :, 0, :].abs().sum(-1, keepdim=True)
    if rotation is not None:
        rot = lambda v: torch.einsum("nij,nhj->nhi", rotation, v)  # noqa: E731
        centroid, engage, traction_mean = rot(centroid), rot(engage), rot(traction_mean)
        rigid = torch.cat([rot(rigid[..., :3]), rot(rigid[..., 3:])], dim=-1)
    valid = win("command/valid").float()[..., None]
    n_env, h = valid.shape[:2]
    blocks = [
        torch.cat([rigid, state], dim=-1),                                  # 7
        _masked(torch.cat([centroid, spread, allowed_fraction], dim=-1), on["region"]),  # 5
        _masked(torch.cat([engage, concentration], dim=-1), on["engage"]),  # 4
        _masked(slip, on["mode"]),                                          # 3
        _masked(torch.cat([traction_mean, traction_width], dim=-1), on["mech"]),  # 4
        valid,                                                              # 1
        torch.stack([torch.full((n_env, h), float(on[name]), device=valid.device)
                     for name in FIELD_GROUPS], dim=-1),
    ]
    return torch.cat(blocks, dim=-1)
