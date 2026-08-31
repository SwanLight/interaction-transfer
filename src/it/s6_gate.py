"""S6 scripted gate 的不可伪造元数据与源码指纹。

训练只能消费与当前 executor/object/primitive 以及当前实现逐项匹配的 smoke JSON。
这防止一份 block/padrod 的旧报告给六个不同组合放行，也防止改完 reward 后继续使用
改动前的绿灯。这里只做纯文件哈希，不依赖 torch/Isaac。
"""

from __future__ import annotations

import hashlib
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
GATE_SOURCES = (
    "src/it/ei_reward.py",
    "src/it/ei_command.py",
    "src/it/ei_policy.py",
    "src/it/transfer.py",
    "src/it/envs/interaction.py",
    "src/it/float_ctrl.py",
    "src/it/probe_scene.py",
    "tools/s6_smoke.py",
    "tools/s6_reward_probe.py",
    "tools/s6_train.py",
)


def implementation_fingerprint() -> str:
    digest = hashlib.sha256()
    for relative in GATE_SOURCES:
        path = _ROOT / relative
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def smoke_path(executor: str, object_name: str, primitive: str,
               root: str | Path = "/tmp/s6") -> Path:
    return Path(root) / f"smoke_{executor}_{object_name}_{primitive}.json"
