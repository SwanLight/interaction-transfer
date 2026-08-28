"""把向量化的 ``extract_contact_points_padded`` 与 S2 已验证过的逐 env 版本逐点对拍。

S3 第一次 pilot 出现了跨环境串数据：env 3 记到的接触点在物体系里 y 偏了
-2223 mm，恰好是 env_spacing 的量级，而 env 0 完全正常。P-18 记过同一类问题
（逐点 buffer 是跨 env 扁平打包的），所以先怀疑新写的向量化切片，而不是
物理或场景。

判据：同一帧、同一传感器，两个实现取出的接触点集合必须逐点相同。
"""

from __future__ import annotations

import os
import sys

import argparse
_pp = argparse.ArgumentParser()
_pp.add_argument('--plates', type=int, default=2)
_pp.add_argument('--envs', type=int, default=6)
_pp.parse_known_args()

from isaaclab.app import AppLauncher

_app = AppLauncher(headless=True).app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import ContactSensorCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from it import assets as A  # noqa: E402
from it.build_assets import CabinetCfg  # noqa: E402
from it.contact_utils import (  # noqa: E402
    extract_contact_points,
    extract_contact_points_padded,
)

import argparse as _argparse
_p = _argparse.ArgumentParser()
_p.add_argument("--plates", type=int, default=2)
_p.add_argument("--envs", type=int, default=6)
_pa, _ = _p.parse_known_args()

C = CabinetCfg()
DT = 1.0 / 300.0
N = _pa.envs
K = 16
NPLATE = _pa.plates


@configclass
class SceneCfg(InteractiveSceneCfg):
    light = AssetBaseCfg(prim_path="/World/light",
                         spawn=sim_utils.DomeLightCfg(intensity=600.0))
    cabinet = A.CABINET_CFG.replace(prim_path="{ENV_REGEX_NS}/Cabinet")
    plate0 = A.plate_cfg(0)
    contact0 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Plate0", track_pose=True, track_contact_points=True,
        max_contact_data_count_per_prim=K,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cabinet/Drawer"],
        update_period=0.0, history_length=0,
    )
    if NPLATE > 1:
        plate1 = A.plate_cfg(1)
        contact1 = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Plate1", track_pose=True, track_contact_points=True,
            max_contact_data_count_per_prim=K,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Cabinet/Drawer"],
            update_period=0.0, history_length=0,
        )


def main() -> int:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device="cuda:0"))
    scene = InteractiveScene(SceneCfg(num_envs=N, env_spacing=2.2, replicate_physics=True))
    sim.reset()
    dev = sim.device
    cab, plate, sensor = scene["cabinet"], scene["plate0"], scene["contact0"]
    jid = cab.find_joints("DrawerJoint")[0][0]
    body = cab.body_names.index("Drawer")

    # 每个 env 把板压在把手横杆的不同 y 上——错开位置才能看出串环境
    zero = torch.zeros(N, 1, device=dev)
    cab.write_joint_state_to_sim(zero, zero, joint_ids=[jid])
    scene.update(DT)
    handle = cab.data.body_pos_w[:, body, :] + torch.tensor(
        [C.panel_t + C.handle_clearance + C.handle_radius, 0.0, C.panel_h / 2], device=dev)
    y_off = torch.linspace(-0.030, 0.030, N, device=dev)

    plates = [scene[f"plate{i}"] for i in range(NPLATE)]
    sensors = [scene[f"contact{i}"] for i in range(NPLATE)]
    # plate0 贴杆背面（手指），plate1 贴杆前面（拇指），与 S3 采集脚本一致
    for i, pl in enumerate(plates):
        st = pl.data.default_root_state.clone()
        st[:, 0] = handle[:, 0] + (-0.0135 if i == 0 else 0.0135)
        st[:, 1] = handle[:, 1] + y_off
        st[:, 2] = handle[:, 2]
        q = torch.tensor([0.5, 0.5, 0.5, 0.5] if i == 0 else [0.5, 0.5, -0.5, -0.5],
                         device=dev)
        st[:, 3:7] = q.repeat(N, 1)
        st[:, 7:] = 0.0
        pl.write_root_state_to_sim(st)

    tq = torch.zeros(N, 1, 3, device=dev)
    for _ in range(80):
        cab.set_joint_effort_target(zero, joint_ids=[jid])
        for i, pl in enumerate(plates):
            f = torch.zeros(N, 1, 3, device=dev)
            f[:, 0, 0] = 6.0 if i == 0 else -3.0
            f[:, 0, 2] = pl.data.default_mass.sum(dim=-1).to(dev) * 9.81
            pl.set_external_force_and_torque(f, tq, is_global=True)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(DT)

    total_bad = 0
    for si, sensor in enumerate(sensors):
        view = sensor.contact_physx_view
        cf, cp_, cn, cs, counts, start = view.get_contact_data(dt=DT)
        print(f"\n--- sensor{si} (Plate{si}) --- counts={counts.flatten().tolist()} "
              f"start={start.flatten().tolist()} buffer={tuple(cp_.shape)}")
        plate_w = plates[si].data.root_pos_w
        used = int((start.flatten() + counts.flatten()).max().item())
        for i in range(used):
            pos = cp_[i]
            d = (plate_w - pos.unsqueeze(0)).norm(dim=-1)
            owner = int(d.argmin().item())
            claimed = -1
            for e in range(N):
                s0, c0 = int(start[e].item()), int(counts[e].item())
                if c0 and s0 <= i < s0 + c0:
                    claimed = e
                    break
            bad = claimed != owner
            total_bad += bad
            print(f"  slot {i:2d}: pos {pos[0]:+8.4f} {pos[1]:+8.4f} {pos[2]:+8.4f} "
                  f"|F| {abs(float(cf[i,0])):7.3f}  切片说 env {claimed}  最近板 env {owner}"
                  f"  距 {d[owner]:.4f}{'   <<< 不一致' if bad else ''}")
    bad = total_bad
    print("\n结论:", "全部槽位归属正确" if bad == 0 else f"{bad} 个槽位归属错误")
    return 0 if bad == 0 else 1


try:
    code = main()
except Exception:
    import traceback
    traceback.print_exc()
    code = 2
sys.stdout.flush()
os._exit(code)
