"""可视化辅助：相机位姿。

单独成模块，是因为录像脚本之间要共用 ``look_at_quat``，而直接
``from s0_record import look_at_quat`` 会触发 s0_record 模块级的 argparse
（它要求 ``--scene``），导入即报错。工具函数不该藏在带命令行入口的脚本里。
"""

from __future__ import annotations

import math

import numpy as np


def look_at_quat(eye, target, up=(0.0, 0.0, 1.0)):
    """USD/OpenGL 相机约定（-Z 朝前，+Y 朝上）的四元数 (w, x, y, z)。

    相机位姿必须在 ``CameraCfg.OffsetCfg`` 里静态给定——运行时的
    ``cam.set_world_poses_from_view()`` 在本环境会挂起（`log/pitfalls.md` P-24）。
    """
    eye = np.asarray(eye, float)
    target = np.asarray(target, float)
    up = np.asarray(up, float)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    z = -fwd
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, qx, qy, qz = 0.25 * s, (R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        w, qx, qy, qz = (R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s
    elif R[1,1] > R[2,2]:
        s = math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        w, qx, qy, qz = (R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s
    else:
        s = math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        w, qx, qy, qz = (R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s
    return (float(w), float(qx), float(qy), float(qz))
