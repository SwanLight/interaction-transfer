"""接触传感器验证。**改动接触相关代码后必跑**（R-4 v3）。

验的是 `plan/02` 的 Contact Mode 与 Interaction Region 两个字段在当前
Isaac Lab 版本上到底能不能拿到：filter 通道有没有数据、``contact_pos_w``
有没有 NaN、stick/slide 能不能按 μmg 判出来。

filter 目标必须是 **kinematic 刚体**而不是静态碰撞体——静态碰撞体不会注册成
可过滤接触对，``net_forces_w`` 照常工作但逐点数据全失效，且不报任何错（P-17）。

结果同时写 stdout 和 ``/tmp/verify3.txt``。
"""
import os
import sys

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import torch
out = open("/tmp/verify3.txt", "w")
def P(*a):
    print(*a, file=out, flush=True)
    print(*a, flush=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass

from it.contact_utils import extract_contact_points_padded

MASS, MU, G = 0.5, 0.4, 9.81
MAT = sim_utils.RigidBodyMaterialCfg(static_friction=MU, dynamic_friction=MU, restitution=0.0)

@configclass
class SceneCfg(InteractiveSceneCfg):
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=1000.0))
    # 关键改动：地板是 kinematic 刚体，不是静态碰撞体
    floor = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Floor",
        spawn=sim_utils.CuboidCfg(size=(4.0,4.0,0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(), physics_material=MAT,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3,0.3,0.3)),
            activate_contact_sensors=True),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0,0.0,-0.1)))
    cube = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(size=(0.1,0.1,0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(), physics_material=MAT,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8,0.2,0.2)),
            activate_contact_sensors=True),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0,0.0,0.06)))
    contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Cube",
        track_pose=True, track_contact_points=True,
        max_contact_data_count_per_prim=16,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Floor"],
        update_period=0.0, history_length=0)

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/120, device="cuda:0"))
scene = InteractiveScene(SceneCfg(num_envs=2, env_spacing=6.0)); sim.reset()
cube: RigidObject = scene["cube"]; cs: ContactSensor = scene["contact"]
dt = sim.get_physics_dt()
def steps(n, f=None):
    for _ in range(n):
        if f is not None: cube.set_external_force_and_torque(f, torch.zeros_like(f))
        cube.write_data_to_sim(); scene.write_data_to_sim(); sim.step(); scene.update(dt)

P("="*70); P("R-4 v3   filter 目标 = kinematic 刚体地板"); P("="*70)
steps(300)
cd = cs.contact_physx_view.get_contact_data(dt=dt)
P(f"接触对数量 counts = {cd[4].flatten().tolist()}   <-- v2 里是 [0,0]")
P(f"force_matrix_w = {[round(x,4) for x in cs.data.force_matrix_w[0,0,0].tolist()]}")
P(f"contact_pos_w NaN 比例 = {torch.isnan(cs.data.contact_pos_w).float().mean():.3f}")
if not torch.isnan(cs.data.contact_pos_w[0,0,0]).any():
    P(f"contact_pos_w[0,0,0] = {[round(x,5) for x in cs.data.contact_pos_w[0,0,0].tolist()]}")

P(f"\n### stick/slide 判定（理论阈值 mu*m*g = {MU*MASS*G:.3f} N）###")
P(f"{'F(N)':>6} {'|v|(m/s)':>10} {'Fn(N)':>8} {'|Ff|(N)':>9} {'接触对':>7}  模式")
for F in (1.0, 1.5, 2.5, 3.0):
    scene.reset(); steps(200)
    push = torch.zeros(scene.num_envs,1,3, device=sim.device); push[:,0,0] = F
    steps(60, push)
    v = cube.data.root_lin_vel_w[0]; net = cs.data.net_forces_w[0,0]
    # **按 env 取，不能对整个 buffer 求和**：buffer 是跨 env 扁平打包的，
    # 直接 sum 得到的是 2 个 env 的合计，恰好是 2 倍，极易被当成单位问题
    # （P-18）。这里用按位置归属的提取器，它对"哪个点属于哪个 env"是可靠的
    # （P-30 说明了为什么前缀和切片不可靠）。
    cp = extract_contact_points_padded(cs, dt, body_pos_w=cube.data.root_pos_w,
                                       max_points=16, own_radius=0.15)
    n = int(cp["count"][0].item())
    tot = cp["friction_forces"][0].sum(dim=0)
    P(f"{F:6.1f} {v.norm():10.4f} {net[2]:8.3f} {tot.norm():9.4f} {n:7d}  {'SLIDE' if v[:2].norm()>0.05 else 'STICK'}")
    P(f"       摩擦力向量=[{tot[0]:+.4f},{tot[1]:+.4f},{tot[2]:+.4f}]")

P("\n### 接触点 / 法向（Interaction Region 需要）###")
scene.reset(); steps(200)
cd = cs.contact_physx_view.get_contact_data(dt=dt)
nm = ["forces","points","normals","separations","counts","start_idx"]
for i,x in enumerate(cd):
    if x.dtype.is_floating_point:
        v2 = x.view(-1, x.shape[-1])
        P(f"  {nm[i]:11s} 非零行={int((v2.norm(dim=-1)>1e-9).sum()):3d}/{len(v2):3d}  首个非零={[round(y,5) for y in v2[(v2.norm(dim=-1)>1e-9).nonzero()[0,0]].tolist()] if (v2.norm(dim=-1)>1e-9).any() else 'N/A'}")
    else:
        P(f"  {nm[i]:11s} = {x.flatten().tolist()}")
P("\n### 摩擦 buffer 的自有布局（它与接触 buffer 不是一一对应）###")
fd = cs.contact_physx_view.get_friction_data(dt=dt)
for i, nm2 in enumerate(["friction_forces", "friction_points", "counts", "start_idx"]):
    x = fd[i]
    if x.dtype.is_floating_point:
        v2 = x.view(-1, x.shape[-1])
        nz = (v2.norm(dim=-1) > 1e-9)
        P(f"  {nm2:16s} shape={tuple(x.shape)} 非零行={int(nz.sum()):3d}/{len(v2):3d}")
    else:
        P(f"  {nm2:16s} shape={tuple(x.shape)} = {x.flatten().tolist()}")
fc, fs_ = fd[2].flatten(), fd[3].flatten()
ff_all = fd[0].view(-1, 3)
for e in range(2):
    seg = ff_all[int(fs_[e]): int(fs_[e]) + int(fc[e])]
    P(f"  env{e}: 按**摩擦自己的** counts/start 切片 -> {int(fc[e])} 行，"
      f"合力 {[round(v, 4) for v in seg.sum(dim=0).tolist()]}")
cc, cs_ = cs.contact_physx_view.get_contact_data(dt=dt)[4].flatten(), \
    cs.contact_physx_view.get_contact_data(dt=dt)[5].flatten()
for e in range(2):
    seg = ff_all[int(cs_[e]): int(cs_[e]) + int(cc[e])]
    P(f"  env{e}: 按**接触的** counts/start 切片 -> {int(cc[e])} 行，"
      f"合力 {[round(v, 4) for v in seg.sum(dim=0).tolist()]}")

P("="*70)
out.close()
# P-19：SimulationApp.close() 在本环境会挂起，进程变僵尸占显存。
# 这个脚本此前就是这样"跑完了但永远不退出"的。
sys.stdout.flush()
os._exit(0)
