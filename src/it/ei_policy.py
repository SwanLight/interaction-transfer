"""E-I / E-T 的网络（`plan/04` §7）。两者**结构完全相同**，只有指令通道的字段掩码不同。

结构一致不是省事，是实验一能成立的前提：C0–C5 之间若容量也变了，"C4 更好"就分不清
是信息多了还是网络大了。所以缺失字段一律**填零 + 独立 mask 位**（在 `ei_command`
那边做），网络这边看到的输入维度、层数、参数量在所有条件下逐位相同。

三个编码器 + 一个共享躯干：

===================  =========================================================
几何 / region         PointNet：逐格 MLP 64→128→128，再按**指令 region 质量加权**
                     池化 + max 池化。加权池化这一路是有意的——纯 max 池化会把
                     "该碰哪"这条最重要的信息稀释进 256 个格里
时间指令              未来 H 格的低维摘要进两层 GRU，hidden 128
本体 / 反馈           proprioception + 物体状态 + 当前实测接触，两层 MLP → 128
躯干                  三路拼接 → 256 → 256 → actor 头 / critic 头
===================  =========================================================

**asymmetric actor-critic**：critic 额外吃仿真里才有的特权信息（物理参数、完整物体
状态）。这是标准做法，且只影响价值估计、不进策略，部署时不需要那些量。
"""

from __future__ import annotations

import torch
from torch import nn

from it.ei_command import SPATIAL_DIM, TEMPORAL_DIM


def _mlp(sizes: list[int], activation=nn.ELU, final_activation: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if final_activation or i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class PointNetRegion(nn.Module):
    """逐格共享 MLP + 双池化。输出 (N, 2*out)。

    两路池化各有各的用处：**max** 保留"存在某个格要求很强的作用"，
    **region 质量加权平均**保留"要求集中在哪、平均是什么样"。只用 max 会丢掉
    分布信息，只用平均会被 256 个格里大量的零稀释掉。
    """

    def __init__(self, in_dim: int = SPATIAL_DIM, hidden: int = 64, out: int = 128):
        super().__init__()
        self.net = _mlp([in_dim, hidden, out, out])
        self.out_dim = 2 * out

    def forward(self, cells: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        feature = self.net(cells)                              # (N, S, out)
        total = weight.sum(-1, keepdim=True).clamp_min(1e-6)
        mean = torch.einsum("ns,nso->no", weight / total, feature)
        return torch.cat([feature.amax(dim=1), mean], dim=-1)


class InteractionPolicy(nn.Module):
    """actor 与 critic 共享编码器结构、**不共享权重**（PPO 里共享容易互相拖）。"""

    def __init__(self, *, proprio_dim: int, action_dim: int, privileged_dim: int = 0,
                 spatial_dim: int = SPATIAL_DIM, temporal_dim: int = TEMPORAL_DIM,
                 hidden: int = 128, trunk: int = 256, init_noise_std: float = 1.0):
        super().__init__()
        self.point = PointNetRegion(spatial_dim, out=hidden)
        self.gru = nn.GRU(temporal_dim, hidden, num_layers=2, batch_first=True)
        self.proprio = _mlp([proprio_dim, hidden, hidden])
        fused = self.point.out_dim + hidden + hidden
        self.actor_trunk = _mlp([fused, trunk, trunk])
        self.actor_head = nn.Linear(trunk, action_dim)
        self.critic_point = PointNetRegion(spatial_dim, out=hidden)
        self.critic_gru = nn.GRU(temporal_dim, hidden, num_layers=2, batch_first=True)
        self.critic_proprio = _mlp([proprio_dim + privileged_dim, hidden, hidden])
        self.critic_trunk = _mlp([fused, trunk, trunk])
        self.critic_head = nn.Linear(trunk, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(torch.log(
            torch.tensor(init_noise_std)))))
        self.privileged_dim = privileged_dim

    @staticmethod
    def _encode(point: PointNetRegion, gru: nn.GRU, proprio: nn.Sequential,
                cells, weight, window, state) -> torch.Tensor:
        # GRU 只取最后一步的 hidden：窗口本身已经是"从现在起的未来"，
        # 不需要跨控制步保持隐状态——那会让同一份指令因为历史不同而被读成不同的要求。
        _, hidden = gru(window)
        return torch.cat([point(cells, weight), hidden[-1], proprio(state)], dim=-1)

    def act(self, cells, weight, window, proprio) -> torch.distributions.Normal:
        feature = self._encode(self.point, self.gru, self.proprio,
                               cells, weight, window, proprio)
        mean = self.actor_head(self.actor_trunk(feature))
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))

    def value(self, cells, weight, window, proprio, privileged=None) -> torch.Tensor:
        if self.privileged_dim:
            if privileged is None:
                raise ValueError("critic 声明了特权维度但没收到 privileged")
            proprio = torch.cat([proprio, privileged], dim=-1)
        feature = self._encode(self.critic_point, self.critic_gru, self.critic_proprio,
                               cells, weight, window, proprio)
        return self.critic_head(self.critic_trunk(feature)).squeeze(-1)


class InteractionActorCritic(nn.Module):
    """rsl_rl 要的 ActorCritic 外壳：从**扁平观测**里解出指令下标，再去指令库取特征。

    为什么观测里放的是下标而不是指令张量本身：逐格空间场是 256×34，PPO 的 rollout
    buffer 是 (horizon 32 × envs 2048 × obs)，直接放会到几个 GB。而**指令在一条
    episode 内是常量**，变的只有格号——放两个整数，用的时候现取，内存换算力。

    观测布局（由 `envs/interaction.py` 保证）::

        [ proprio | command_index | bin_index | object_quat(w,x,y,z) ]

    `object_quat` 用来把表面点与所有向量场一起转进执行器的参考系。**必须整组一起转**
    （P-53/P-54：那三次都是逐帧数值全部正常、只有结论错）。
    """

    is_recurrent = False

    def __init__(self, bank, *, action_dim: int, proprio_dim: int,
                 privileged_dim: int = 0, enabled: dict[str, bool] | None = None,
                 horizon: int | None = None, **kwargs):
        super().__init__()
        from it.ei_command import HORIZON
        self.bank = bank
        self.proprio_dim = int(proprio_dim)
        self.horizon = int(horizon or HORIZON)
        self.enabled = dict(enabled or {})
        self.net = InteractionPolicy(proprio_dim=proprio_dim, action_dim=action_dim,
                                     privileged_dim=privileged_dim, **kwargs)
        self.distribution: torch.distributions.Normal | None = None

    # -------------------------------------------------- 观测拆包

    def _unpack(self, obs: torch.Tensor):
        from it.ei_command import spatial_features, temporal_features
        proprio = obs[:, :self.proprio_dim]
        command = obs[:, self.proprio_dim].long().clamp(0, len(self.bank) - 1)
        bins = obs[:, self.proprio_dim + 1].long().clamp(0, self.bank.n_bins - 1)
        quat = obs[:, self.proprio_dim + 2:self.proprio_dim + 6]
        rotation = _quat_to_matrix(quat).transpose(1, 2)      # 世界→物体 的逆 = 物体→世界
        cells = spatial_features(self.bank, command, bins, rotation=rotation,
                                 enabled=self.enabled)
        window = temporal_features(self.bank, command, bins, horizon=self.horizon,
                                   rotation=rotation, enabled=self.enabled)
        weight = self.bank.gather_bin("region/mass/mean", command, bins)
        privileged = obs[:, self.proprio_dim + 6:]
        return cells, weight, window, proprio, privileged

    # -------------------------------------------------- rsl_rl 接口

    def act(self, obs, **_):
        cells, weight, window, proprio, _ = self._unpack(obs)
        self.distribution = self.net.act(cells, weight, window, proprio)
        return self.distribution.sample()

    def act_inference(self, obs):
        cells, weight, window, proprio, _ = self._unpack(obs)
        return self.net.act(cells, weight, window, proprio).mean

    def evaluate(self, critic_obs, **_):
        cells, weight, window, proprio, privileged = self._unpack(critic_obs)
        return self.net.value(cells, weight, window, proprio,
                              privileged if self.net.privileged_dim else None)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs):
        self.act(obs)

    def reset(self, dones=None):
        pass


def _quat_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """(N,4) (w,x,y,z) -> (N,3,3)。四元数已归一化时与 isaaclab 的实现等价，
    这里自己写是为了让本模块不依赖 Isaac——网络要能在没有 Isaac 的机器上测。"""
    q = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], dim=1)
