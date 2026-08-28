"""S3 定位脚本：分阶段找出双板场景里 CUDA illegal memory access 的真正位置。

上一版 `s3_source_drawer.py` 在服务器上连跑十几个变体全部死在
``GpuRigidBodyView.cpp:810``。CUDA 的非法访存是**异步**报出来的，
traceback 指的是被检测到的地方，不是发生的地方，所以必须
``CUDA_LAUNCH_BLOCKING=1`` + 每阶段 ``torch.cuda.synchronize()`` 才能定位。

用法::

    CUDA_LAUNCH_BLOCKING=1 PYTHONPATH=src /isaac-sim/python.sh tools/s3_diag.py --physx 1
"""

from __future__ import annotations

import argparse
import os
import sys

_ap = argparse.ArgumentParser()
_ap.add_argument("--physx", type=int, default=1, help="1=按 S2 加大 PhysX GPU 缓冲，0=用默认值")
_ap.add_argument("--sensors", type=int, default=1, help="1=挂接触传感器")
_ap.add_argument("--envs", type=int, default=4)
_ap.add_argument("--glob", type=int, default=1, help="1=外力用 is_global=True")
_ap.add_argument("--plates", type=int, default=2, help="板的数量 1 或 2")
_ap.add_argument("--cab", type=int, default=1, help="1=场景里放抽屉柜")
_ap.add_argument("--pose", type=int, default=1, help="1=传感器开 track_pose")
_a, _ = _ap.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

_app = AppLauncher(headless=True).app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, AssetBaseCfg, RigidObject  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import ContactSensorCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from it import assets as A  # noqa: E402
from it.build_assets import CabinetCfg  # noqa: E402
from it.float_ctrl import FloatingPD  # noqa: E402

C = CabinetCfg()
DT = 1.0 / 150.0
STAGE = [0]


def ok(msg: str):
    """每个阶段后强制同步——不同步的话错误会漂到后面的调用上。"""
    torch.cuda.synchronize()
    STAGE[0] += 1
    print(f"[STAGE {STAGE[0]:02d}] OK  {msg}", flush=True)


def _sensor(idx: int):
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Plate{idx}",
        track_contact_points=True,
        **({"track_pose": True} if _a.pose else {}),
        max_contact_data_count_per_prim=16,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cabinet/Drawer"],
        update_period=0.0, history_length=0,
    )


@configclass
class SceneCfg(InteractiveSceneCfg):
    light = AssetBaseCfg(prim_path="/World/light",
                         spawn=sim_utils.DomeLightCfg(intensity=800.0))
    if _a.cab:
        cabinet = A.CABINET_CFG.replace(prim_path="{ENV_REGEX_NS}/Cabinet")
    plate0 = A.plate_cfg(0)
    if _a.plates > 1:
        plate1 = A.plate_cfg(1)
    if _a.sensors and _a.cab:
        contact0 = _sensor(0)
        if _a.plates > 1:
            contact1 = _sensor(1)


def main() -> int:
    physx = (
        sim_utils.PhysxCfg(gpu_max_rigid_contact_count=2 ** 22,
                           gpu_max_rigid_patch_count=2 ** 20)
        if _a.physx else sim_utils.PhysxCfg()
    )
    print(f"配置：physx_override={bool(_a.physx)} sensors={bool(_a.sensors)} "
          f"envs={_a.envs} is_global={bool(_a.glob)}", flush=True)

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=DT, device="cuda:0", physx=physx))
    scene = InteractiveScene(SceneCfg(num_envs=_a.envs, env_spacing=1.8,
                                      replicate_physics=True))
    ok("场景构建")

    sim.reset()
    ok("sim.reset()")

    n, dev = _a.envs, sim.device
    cab: Articulation | None = scene["cabinet"] if _a.cab else None
    plates: list[RigidObject] = [scene[f"plate{i}"] for i in range(_a.plates)]
    jid = cab.find_joints("DrawerJoint")[0][0] if cab is not None else 0

    for _ in range(5):
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(DT)
    ok("空转 5 步（无任何写入）")

    if cab is not None:
        zero = torch.zeros(n, 1, device=dev)
        cab.write_joint_state_to_sim(zero, zero, joint_ids=[jid])
        cab.set_joint_effort_target(zero, joint_ids=[jid])
        ok("cabinet 关节清零 + 力矩清零")
        body = cab.body_names.index("Drawer")
        handle = cab.data.body_pos_w[:, body, :] + torch.tensor(
            [C.panel_t + C.handle_clearance + C.handle_radius, 0.0, C.panel_h / 2], device=dev)
    else:
        handle = scene.env_origins + torch.tensor([0.2, 0.0, 0.3], device=dev)
    ok(f"读把手位姿 {handle[0].tolist()}")

    q = torch.zeros(n, 4, device=dev)
    q[:, 0] = 1.0
    for i, p in enumerate(plates):
        st = p.data.default_root_state.clone()
        st[:, :3] = handle + torch.tensor([-0.14, -0.028 + 0.056 * i, 0.015], device=dev)
        st[:, 3:7] = q
        st[:, 7:] = 0.0
        p.write_root_state_to_sim(st)
    ok("两块板 write_root_state_to_sim")

    for _ in range(3):
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(DT)
    ok("root 写入后空转 3 步")

    zw = torch.zeros(n, 1, 3, device=dev)
    for p in plates:
        p.set_external_force_and_torque(zw, zw, is_global=bool(_a.glob))
    scene.write_data_to_sim()
    sim.step(render=False)
    scene.update(DT)
    ok("零外力 + write_data_to_sim + step")

    pds = [FloatingPD(p, kp_pos=600.0, kd_pos=50.0, kp_rot=50.0, kd_rot=7.0,
                      max_force=180.0, max_torque=18.0, kd_force=35.0) for p in plates]
    tgt = [p.data.root_pos_w.clone() for p in plates]
    for step in range(20):
        for i, p in enumerate(plates):
            f, tq = pds[i].compute(tgt[i], q)
            p.set_external_force_and_torque(f, tq, is_global=bool(_a.glob))
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(DT)
        if step == 0:
            ok("第 1 步真实 PD 外力")
    ok("20 步真实 PD 外力")

    if _a.sensors and _a.cab:
        from it.contact_utils import extract_contact_points
        for name in [f"contact{i}" for i in range(_a.plates)]:
            cps = extract_contact_points(scene[name], DT)
            print(f"    {name}: 接触点数 {[c.num_contacts for c in cps]}", flush=True)
        ok("读接触传感器")

    print("\n全部阶段通过 —— 该配置没有非法访存", flush=True)
    return 0


try:
    code = main()
except Exception as exc:
    import traceback
    traceback.print_exc()
    print(f"\n>>> 死在 STAGE {STAGE[0]+1}：{type(exc).__name__}: {exc}", flush=True)
    code = 1
sys.stdout.flush()
os._exit(code)
